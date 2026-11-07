"""Every team we own, as one deduplicated training table for the loss-down retrain (2026-10-07).

Sources, each tagged so the model can tell distributions apart (metadata conditioning):
  model   - teams a diffusion model wrote inside a search loop (activesearch, entropyloop,
            gradguide, gradloop, dsame, redundancy, asked-vs-got, temperature, cem,
            reverse-score, diversity, mdts) and every non-corpus team in the matchup matrix
  real    - the 692 VGCPastes corpus teams (+ matrix 'vgcpastes' rows)
  fill    - random-fill completions of real teams (random_fill_baseline, matrix 'randomfill')
  hps     - the 100,000 uniform-legal HPS teams (teams/hps_reg_mb_100k.jsonl)

Labels (win rate vs the top-50, BC policy both sides) are pooled per team the same way
asked_vs_got.gather() does: wins summed over every measurement, divided by battles.
The matchup matrix adds each team's per-opponent row (top-50 columns), from shuffled-slot battles when they cover
40 opponents, else from listed-order battles (policy 1); one team's row never mixes the two (top50_rows).
A neutral or missing nature counts as Serious, as in the matchup database (neutral_as_serious), so a team and its
battled copy there are one row.

THE TEST SET is asked_vs_got's held-out 500 (same seed, same split as loss_diagnosis).
Any training team whose 48-field encoding equals a test team's, as written or with Serious natures, is removed.

    python lossdata.py        (run with /tmp/vgc-pilot/.venv/bin/python) -> results/lossdata.npz
"""
import glob, json, os, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np

sys.path.insert(0, "/tmp/vgc-pilot/src")
import asked_vs_got as X
import matchup_db as MDB
import heldout as HO
from corpus import parse_team_text

REPO = X.REPO
RES = X.RES
OUT = RES / "lossdata.npz"
HPS = REPO / "teams/hps_reg_mb_100k.jsonl"
SKIP = ("smoke", "random_fill_tasks", "asked_vs_got_data", "rebattle_top")
FILL_FILES = ("random_fill_baseline",)
TAGS = ["model", "real", "fill", "hps"]
FAM = ["real", "oldpool", "loops08", "temperature", "cemrev", "redund", "matrix", "fill", "hps"]
def family(name):
    if name.startswith("temperature"): return "temperature"
    if name.startswith(("cem_", "reverse_score", "diversity_baseline")): return "cemrev"
    if name.startswith(("redundancy", "asked_vs_got")): return "redund"
    if name.startswith("random_fill"): return "fill"
    return "loops08"            # activesearch(2), entropyloop, gradguide, gradloop, dsame, mdts


def neutral_as_serious(team):
    """The team with every neutral or missing nature written Serious, in place. Neutral natures change no stat and
    Showdown battles a missing one as Serious, so the matchup database stores them all as Serious
    (matchup_db.canonical_slots). Without this a team battled there gets a second row, its label on the Serious
    copy and none on the original (2,271 teams on 2026-11-06)."""
    for s in team:
        if not s.get("nature") or MDB._norm(s["nature"]) in MDB.NEUTRAL: s["nature"] = "Serious"
    return team


def test_split(V, nature_as_serious=False):
    """asked_vs_got.train()'s held-out 500, rebuilt exactly (loss_diagnosis.load_split). nature_as_serious encodes
    the same teams with neutral_as_serious, the form gather() keys its pool by."""
    rows = json.load(open(X.DATA))["rows"]
    Xs, W = [], []
    for r in rows:
        t = parse_team_text(r["paste"])
        if nature_as_serious: t = neutral_as_serious(t)
        if len(t) != 6: continue
        try: e = V.encode(t)
        except ValueError: continue                # a field outside the corpus vocabulary
        Xs.append(e); W.append(r["y"])
    Xs, W = np.stack(Xs), np.array(W, np.float32)
    idx = np.random.default_rng(X.SEED).permutation(len(Xs))
    return Xs[idx[:500]], W[idx[:500]], Xs[idx[500:]], W[idx[500:]]


def walk_pastes(o, out):  # noqa: C901
    """Collect (paste, win_rate|None, battles) from any results JSON layout seen in this repo."""
    if isinstance(o, dict):
        ps = o.get("pastes")
        if isinstance(ps, list) and ps and isinstance(ps[0], str):
            ys = o.get("y") if isinstance(o.get("y"), list) and len(o["y"]) == len(ps) else [None] * len(ps)
            bs = o.get("battles") if isinstance(o.get("battles"), list) and len(o["battles"]) == len(ps) else [24] * len(ps)
            for p, y, b in zip(ps, ys, bs): out.append((p, y, b or 24))
        sel, sc = o.get("selected"), o.get("scores")
        if isinstance(sel, list) and isinstance(sc, list) and len(sel) == len(sc) and sel and isinstance(sel[0], str):
            for p, s in zip(sel, sc):
                if isinstance(s, dict) and "wins" in s: out.append((p, s["wins"] / s["battles"], s["battles"]))
        recs = o.get("records") if isinstance(o.get("records"), list) else o.get("fills") if isinstance(o.get("fills"), list) else None
        if recs:
            for r in recs:
                if isinstance(r, dict) and isinstance(r.get("paste"), str):
                    if str(r.get("valid", "True")) == "False": continue
                    out.append((r["paste"], None, 0))
        for k, v in o.items():
            if k in ("pastes", "selected", "records", "fills"): continue
            walk_pastes(v, out)
    elif isinstance(o, list):
        for v in o: walk_pastes(v, out)


LISTED_POLICY = 1                  # behaviour-cloning policy on both sides, Pokémon in listed order
SHUFFLED_TAG = "+slots-shuffled"   # slot_order.POLICY_TAG: the same players, Pokémon shuffled every battle


def shuffled_policy(con):
    """The policy that battles LISTED_POLICY's players with slots shuffled (the default since 2026-10-28), or None."""
    listed = con.execute("SELECT row_policy, col_policy FROM policy WHERE policy_id=?", (LISTED_POLICY,)).fetchone()
    if listed is None: raise ValueError(f"policy {LISTED_POLICY} (listed slot order) is not in the matchup database")
    row, col = listed
    ids = [p for (p,) in con.execute("SELECT policy_id FROM policy WHERE row_policy=? AND col_policy=?",
                                     (row + SHUFFLED_TAG, col + SHUFFLED_TAG))]
    if len(ids) > 1: raise ValueError(f"several shuffled-slot policies {ids} match policy {LISTED_POLICY}")
    return ids[0] if ids else None


def policy_rows(con, top, policy_ids):
    """{policy_id: {team_id: (len(top), 2) array of (wins, battles) against each top-50 team}}."""
    col = {tid: j for j, tid in enumerate(top)}
    out = {p: {} for p in policy_ids}
    ph, pp = ",".join("?" * len(top)), ",".join("?" * len(policy_ids))
    cells = con.execute(f"SELECT policy_id, lo, hi, lo_wins, hi_wins, battles FROM cell WHERE policy_id IN ({pp}) "
                        f"AND (lo IN ({ph}) OR hi IN ({ph}))", [*policy_ids, *top, *top])
    for pid, lo, hi, lw, hw, n in cells:
        for me, op, w in ((lo, hi, lw), (hi, lo, hw)):
            if op in col:
                r = out[pid].setdefault(me, np.zeros((len(top), 2), np.float32))
                r[col[op], 0] += w; r[col[op], 1] += n
    return out


MIN_COVERAGE = 40          # asked_vs_got.gather()'s coverage rule: opponents battled before a team gets a label


def coverage(r, own_col):
    """Which top-50 opponents the row has battles against; a team's cell against itself is skipped, as in score()."""
    cov = r[:, 1] > 0
    if own_col is not None: cov[own_col] = False
    return cov


def matrix_label(r, own_col, cw):
    """(weighted win rate, battles) over the covered opponents, or (None, None) below MIN_COVERAGE."""
    cov = coverage(r, own_col)
    if cov.sum() < MIN_COVERAGE: return None, None
    return float((cw[cov] * r[cov, 0] / r[cov, 1]).sum() / cw[cov].sum()), int(r[cov, 1].sum())


def top50_rows(con, top):
    """Each team's row against the top-50, from one slot order, never both pooled: shuffled when those battles
    cover MIN_COVERAGE opponents, else listed. Most teams battled since 2026-10-28 have only shuffled battles."""
    sp = shuffled_policy(con)
    by_policy = policy_rows(con, top, [LISTED_POLICY] + ([sp] if sp is not None else []))
    rows = by_policy[LISTED_POLICY]
    col = {tid: j for j, tid in enumerate(top)}
    for tid, r in by_policy.get(sp, {}).items():
        if tid not in rows or coverage(r, col.get(tid)).sum() >= MIN_COVERAGE: rows[tid] = r
    return rows


def gather(observe=None):  # noqa: C901
    """Every source pooled per 48-field team: dict(pool, stats, xte, wte, top). observe(key, team), if given, is called
    with each kept appearance's encoding bytes and parsed team (Stat Points included), so a caller can see what the
    48-field key throws away."""
    corpus, V, *_ = X.setup()
    xte, wte, _, _ = test_split(V)
    test_keys = {r.tobytes() for r in xte} | {r.tobytes() for r in test_split(V, nature_as_serious=True)[0]}
    held_keys = HO.identities()             # ranks 51-100 (48 fields): never a training row, never a matrix-derived label

    pool = {}            # encoding bytes -> dict(x, wins, battles, tags:set, row:None)
    def add(x, tag, y=None, n=0, fam=None, team=None):
        k = x.tobytes()
        if k in test_keys: return "test"
        if HO.row_key(x, V) in held_keys: return "heldout"
        if observe is not None and team is not None: observe(k, team)
        e = pool.setdefault(k, dict(x=x, wins=0.0, battles=0, tags=set(), fam=0))
        e["tags"].add(tag)
        if fam: e["fam"] |= 1 << FAM.index(fam)
        if y is not None and n:
            e["wins"] += float(y) * n; e["battles"] += int(n)
        return "ok"

    cache = {}
    def enc(p):
        """(48-field encoding or None, parsed team)"""
        if p in cache: return cache[p]
        t = neutral_as_serious(parse_team_text(p))
        e = None
        if len(t) == 6:
            try: e = V.encode(t)
            except ValueError: e = None            # a field outside the corpus vocabulary
        cache[p] = (e, t)
        return cache[p]

    stats = defaultdict(lambda: defaultdict(int))
    # 1. corpus
    for t in corpus:
        t = neutral_as_serious([dict(s) for s in t])
        stats["corpus"][add(V.encode(t), "real", fam="real", team=t)] += 1
    # 2. asked_vs_got_data rows (the old training pool, labels already pooled)
    for r in json.load(open(X.DATA))["rows"]:
        e, t = enc(r["paste"])
        if e is None: stats["asked_vs_got_data"]["unencodable"] += 1; continue
        stats["asked_vs_got_data"][add(e, "model", r["y"], r.get("battles", 24), fam="oldpool", team=t)] += 1
    # 3. every results JSON
    for f in sorted(glob.glob(str(RES / "*.json"))):
        name = Path(f).stem
        if any(s in name for s in SKIP) or os.path.getsize(f) < 2000: continue
        try: d = json.load(open(f))
        except Exception: continue
        acc = []; walk_pastes(d, acc)
        if not acc: continue
        tag = "fill" if name in FILL_FILES else "model"
        seen_lab = set()
        for p, y, n in acc:
            e, t = enc(p)
            if e is None: stats[name]["unencodable"] += 1; continue
            # a labelled paste appears in several places in some files; count its label once per file
            lab_key = (p, y)
            if y is not None and lab_key in seen_lab: y = None
            if y is not None: seen_lab.add(lab_key)
            stats[name][add(e, tag, y, n if y is not None else 0, fam=family(name), team=t)] += 1
    # 4. the matchup matrix: every legal team, its top-50 row and its pooled label
    con = MDB.connect()
    top = MDB.set_members(con, "top50")
    wt = dict(con.execute("SELECT team_id, weight FROM team_set WHERE set_name='top50'").fetchall())
    col = {tid: j for j, tid in enumerate(top)}
    cw = np.array([wt[t] for t in top], np.float32)
    rows = top50_rows(con, top)
    for tid, p, origin in con.execute("SELECT team_id, paste, origin FROM team WHERE legal=1"):
        e, t = enc(p)
        if e is None: stats["matrix"]["unencodable"] += 1; continue
        tag = "real" if origin.startswith("vgcpastes") else "fill" if origin.startswith("randomfill") else "model"
        r = rows.get(tid)
        y, n = matrix_label(r, col.get(tid), cw) if r is not None else (None, None)
        s = add(e, tag, y, n or 0, fam="real" if tag == "real" else "fill" if tag == "fill" else "matrix", team=t)
        stats["matrix"][s] += 1
        if s == "ok" and r is not None:
            pr = pool[e.tobytes()].setdefault("row", np.zeros((len(top), 2), np.float32))
            pr += r
    # 5. HPS 100k
    for line in open(HPS):
        e, t = enc(json.loads(line)["team"])
        if e is None: stats["hps"]["unencodable"] += 1; continue
        stats["hps"][add(e, "hps", fam="hps", team=t)] += 1
    return dict(pool=pool, stats=stats, xte=xte, wte=wte, top=top)


def main():
    t0 = time.perf_counter()
    g = gather()
    pool, stats, xte, wte, top = g["pool"], g["stats"], g["xte"], g["wte"], g["top"]
    keys = list(pool)
    Xa = np.stack([pool[k]["x"] for k in keys])
    W = np.array([pool[k]["wins"] / pool[k]["battles"] if pool[k]["battles"] else np.nan for k in keys], np.float32)
    NB = np.array([pool[k]["battles"] for k in keys], np.int32)
    # one tag per team, by priority: a team seen in the real corpus is real even if a loop copied it
    pri = {"real": 0, "model": 1, "fill": 2, "hps": 3}
    T = np.array([TAGS.index(min(pool[k]["tags"], key=pri.get)) for k in keys], np.int8)
    M = np.stack([pool[k]["row"] if "row" in pool[k] else np.zeros((len(top), 2), np.float32) for k in keys])
    S = np.array([pool[k]["fam"] for k in keys], np.int32)
    np.savez_compressed(OUT, X=Xa, W=W, NB=NB, T=T, S=S, M=M, xte=xte, wte=wte, top=np.array(top))
    print(f"{len(keys)} distinct encoded teams -> {OUT.name}  ({time.perf_counter()-t0:.0f}s)")
    for tg in TAGS:
        sel = T == TAGS.index(tg)
        print(f"  {tg:6s} {sel.sum():7d} teams · labelled {np.isfinite(W[sel]).sum():6d} · matrix rows {(M[sel,:,1].sum(1)>0).sum():6d}")
    for j, fm in enumerate(FAM): print(f"  family {fm:12s} {((S >> j) & 1).sum():7d}")
    print("per source (ok = kept or merged, test = removed as a held-out duplicate):")
    for s, c in stats.items(): print(f"  {s[:40]:40s} {dict(c)}")
    json.dump({s: dict(c) for s, c in stats.items()}, open(RES / "lossdata_sources.json", "w"), indent=1)


if __name__ == "__main__":
    main()
