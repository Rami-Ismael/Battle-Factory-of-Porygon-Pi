"""How near the line do the training teams themselves land? (issue #72, protocol in docs/training-teams-line.md)

If a model returned training teams perfectly, this is the mean squared error it would get. For each ask from 1/24 to
17/24, draw 64 training teams whose stored label is nearest that ask and battle them fresh, in a private database so no
battle behind a label is reused. Two arms on the same 48 fields: "Stored Stat Points" (`stored`, the team's own) and
"Redrawn Stat Points" (`redrawn`, each Pokémon's copied from a same-species corpus team and its Stat Alignment repaired,
as the 48-field model's `to_paste(repair=True)` does). One battle against each of the 49 distinct top-50 meta teams,
behaviour-cloning policy on both sides, slots shuffled.

    python training_teams_line.py draw | battle | report      (TTL_SMOKE=1: 2 asks, 2 teams, smoke files)
"""
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SMOKE = bool(os.environ.get("TTL_SMOKE"))
D = Path.home() / "vgc-data/training_teams_line"
TAG = "_smoke" if SMOKE else ""
TEAMS, SCORES = D / f"teams{TAG}.json", D / f"scores{TAG}.json"
DB = D / f"battles{TAG}.sqlite"
OUT = Path.home() / f"vgc-data/results/training_teams_line{TAG}.json"
TABLE = Path.home() / "Documents/code/vgc-team-generator-pilot/results/lossdata84.npz"
TABLE_SHA = "02463cbd812499e7bd94ffb4faff505318f86e58d2fc41ba57476d21b03e11c7"

GRID = 24
ASKS = [4, 12] if SMOKE else list(range(1, 18))    # numerators of k/24
SPARSE_ASKS = list(range(18, 24))                  # label counts only: the training data has almost nothing here
COMPARE_WINDOW = (0.1, 0.7)                        # asks 0.1 to 0.7, where the 48-field model alone scored 0.0138
PER_ASK = 2 if SMOKE else 64
SEED = 20261101
ARMS = ("stored", "redrawn")
NEAR_LINE = 0.001


def nearest_ask(label):
    """The numerator k of the ask k/24 nearest a label (NaN stays unassigned, as -1)."""
    label = np.asarray(label, float)
    out = np.full(label.shape, -1, dtype=np.int64)
    ok = np.isfinite(label)
    out[ok] = np.floor(label[ok] * GRID + 0.5).astype(np.int64)
    return out


def line_error(asks, got):
    """Mean squared error between each ask and the mean got at that ask."""
    return float(np.mean((np.asarray(asks, float) - np.asarray(got, float)) ** 2))


def noise_corrected_error(asks, scores):
    """line_error minus the part sampling teams alone adds to it: E[(mean - ask)^2] = miss^2 + variance of the mean."""
    got = [np.mean(y) for y in scores]
    var_of_mean = [np.var(y, ddof=1) / len(y) for y in scores]
    return line_error(asks, got) - float(np.mean(var_of_mean))


def one_battle_noise(score, columns):
    """Variance of a team's win rate from battle luck, with one battle per opponent. The scorer's own se is zero then, because
    each cell is a single 0 or 1. This binomial variance at the team's mean is an upper bound: uneven matchups lower it."""
    return score * (1 - score) / columns


def scatter(scores, battle_vars):
    """Across-team spread of win rate: total, battle noise alone, and what remains as real differences between teams.
    With one_battle_noise, battle noise is an upper bound and the difference between teams a lower bound."""
    total = float(np.var(scores, ddof=1))
    noise = float(np.mean(battle_vars))
    return dict(total_sd=math.sqrt(total), battle_noise_sd=math.sqrt(noise), team_sd=math.sqrt(max(total - noise, 0.0)))


def bootstrap(asks, stored, redrawn, reps=4000, seed=7):
    """95% intervals on each arm's line error and on redrawn minus stored, resampling teams within each ask.
    Arms are paired: one resample picks the same teams in both. stored and redrawn are lists of per-ask score arrays."""
    rng = np.random.default_rng(seed)
    s, r = np.empty((reps, len(asks))), np.empty((reps, len(asks)))
    for j, (a, b) in enumerate(zip(stored, redrawn, strict=True)):
        pick = rng.integers(0, len(a), (reps, len(a)))
        s[:, j], r[:, j] = np.asarray(a)[pick].mean(1), np.asarray(b)[pick].mean(1)
    target = np.asarray(asks, float)
    es, er = ((s - target) ** 2).mean(1), ((r - target) ** 2).mean(1)
    ci = lambda x: [round(float(np.percentile(x, 2.5)), 5), round(float(np.percentile(x, 97.5)), 5)]
    return dict(stored=ci(es), redrawn=ci(er), redrawn_minus_stored=ci(er - es),
                got_redrawn_minus_stored=ci((r - s).mean(1)))


def save(path, obj):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=1) + "\n")
    tmp.replace(path)


def modules():
    """The frozen modules beside this script, with /tmp/vgc-pilot/src behind them for the battle plumbing."""
    sys.path.insert(0, "/tmp/vgc-pilot/src")
    sys.path.insert(0, str(HERE))
    import lossdown                                  # puts /tmp/vgc-pilot/src first again
    sys.path.insert(0, str(HERE))
    return lossdown


def draw():
    """Pick the teams and write both arms' pastes. A team enters only if both its pastes pass the Showdown validator."""
    if SCORES.exists(): raise SystemExit(f"{SCORES} holds scores for the drawn teams; move it before drawing again")
    import hashlib

    modules()
    import alignment as AL
    import asked_vs_got as X
    import hps_generate
    import matchup_db as MDB
    import team84 as T
    from hps_generate import slot_to_text
    from showdown_validator import Validator

    if hashlib.sha256(TABLE.read_bytes()).hexdigest() != TABLE_SHA: raise SystemExit("lossdata84.npz changed")
    z = np.load(TABLE)
    W, NB, val, rows = z["all_W"], z["all_NB"], z["all_val"], z["all_X"]
    V = T.regulation_vocab()
    corpus, _, _, look, spreads = X.setup()
    look.update(V.regulation["names"])
    _, _, _, modes, natures = AL.build_tables(corpus)
    g = lambda v: look.get(v, v)
    slices = {name: {r.tobytes() for r in z[f"{name}_X"]} for name in ("loop", "records")}

    def pastes(row, rng):
        stored, redrawn, fallback = [], [], 0
        for s in T.decode(V, row):
            slot = dict(species=g(s["species"]), ability=g(s["ability"]), item=g(s["item"]), nature=g(s["nature"]),
                        moves=[g(m) for m in s["moves"]], evs=s["evs"])
            stored.append(slot_to_text(slot))
            pool = spreads.get(s["species"])
            fallback += not pool
            pool = pool or [hps_generate.sample_spread(rng)]
            slot = dict(slot, evs=pool[int(rng.integers(0, len(pool)))])
            slot["nature"] = AL.repair_nature(slot, modes, natures)
            redrawn.append(slot_to_text(slot))
        return "\n\n".join(stored) + "\n", "\n\n".join(redrawn) + "\n", fallback

    train = np.isfinite(W) & ~val
    bins = nearest_ask(W)
    bins[~train] = -1
    known = battle_database_hashes()
    out = dict(table_sha256=TABLE_SHA, per_ask=PER_ASK, seed=SEED, cells={},
               sparse_label_counts={f"{k}/{GRID}": int((bins == k).sum()) for k in SPARSE_ASKS})
    v = Validator()
    try:
        for k in ASKS:
            idx = np.where(bins == k)[0]
            rng = np.random.default_rng([SEED, k])
            teams, tried, refused, fallback = [], 0, dict(stored=0, redrawn=0), 0
            for i in rng.permutation(idx):
                if len(teams) == PER_ASK: break
                tried += 1
                a, b, f = pastes(rows[i], np.random.default_rng([SEED, k, int(i)]))
                ea, eb = v(a), v(b)
                refused["stored"] += ea is not None
                refused["redrawn"] += eb is not None
                if ea is None and eb is None:
                    fallback += f
                    teams.append(dict(row=int(i), label=float(W[i]), label_battles=int(NB[i]), stored=a, redrawn=b,
                                      in_loop_teams=rows[i].tobytes() in slices["loop"],
                                      in_battle_records=rows[i].tobytes() in slices["records"],
                                      in_battle_database=MDB.canon_hash(MDB.canonical_paste(a)) in known))
            out["cells"][str(k)] = dict(ask=k / GRID, eligible=len(idx), tried=tried, refused=refused,
                                        species_without_corpus_set=fallback, teams=teams)
            print(f"ask {k:2d}/{GRID}: {len(idx):6d} eligible, {len(teams)} kept of {tried} tried, refused {refused}", flush=True)
    finally:
        v.close()
    save(TEAMS, out)
    print("wrote", TEAMS)


def battle_database_hashes():
    """Canonical hashes in the shared battle database, read only: a check that a decoded row is the labelled team."""
    import sqlite3
    con = sqlite3.connect(f"file:{Path.home() / 'vgc-data/matchup_regmb.sqlite'}?mode=ro", uri=True)
    try:
        return {h for (h,) in con.execute("SELECT canon_hash FROM team")}
    finally:
        con.close()


def private_db(MDB, T50):
    """A private database whose columns are the top-50 meta teams (as scripts/jev_rebattle.py), so every battle is new."""
    from showdown_validator import Validator
    con = MDB.connect(DB)
    pid = MDB.get_policy(con, note="training teams re-battled, slots shuffled, 2026-11-01")
    if "+slots-shuffled" not in con.execute("SELECT row_policy FROM policy WHERE policy_id=?", (pid,)).fetchone()[0]:
        raise SystemExit("these battles must shuffle slots")
    val = Validator()
    try:
        with con:
            con.execute("DELETE FROM team_set WHERE set_name='top50'")
            for t in json.load(open(T50.TOP50)):      # as matchup_db.seed: a team listed twice counts twice
                tid = MDB.add_team(con, Path(T50.resolve(t["id"])).read_text(), "vgcpastes", val)[0]
                con.execute("INSERT INTO team_set VALUES ('top50', ?, 1) ON CONFLICT (set_name, team_id) "
                            "DO UPDATE SET weight = weight + 1", (tid,))
    finally:
        val.close()
    return con, pid


def battle():
    modules()
    import matchup_db as MDB
    import top50 as T50

    if MDB.DEFAULT_DB != DB: raise SystemExit(f"run with MATCHUP_DB={DB}: the supervisor recovers stalled battles there")
    if MDB.SLOT_ORDER != "shuffled": raise SystemExit(f"VGC_SLOT_ORDER={MDB.SLOT_ORDER}: these battles must shuffle slots")
    con, pid = private_db(MDB, T50)
    teams = json.loads(TEAMS.read_text())
    done = json.loads(SCORES.read_text()) if SCORES.exists() else {}
    for k in ASKS:
        for a, arm in enumerate(ARMS):
            key = f"{k}_{arm}"
            if key in done: continue
            pastes = [t[arm] for t in teams["cells"][str(k)]["teams"]]
            sc = MDB.score_pastes(con, pastes, per_cell=1, seed=SEED + 100 * k + a, origin=f"training-teams:{key}", policy_id=pid)
            if not all(s["score"] is not None and s["coverage"] == s["columns"] and s["battles"] >= s["columns"] for s in sc):
                raise SystemExit(f"{key}: a team lacks some opponents; the next attempt fills them in")
            done[key] = sc
            save(SCORES, done)
            print(f"ask {k:2d}/{GRID} {arm:8s} got {np.mean([s['score'] for s in sc]):.3f}", flush=True)
    con.close()


def report():
    teams, scores = json.loads(TEAMS.read_text()), json.loads(SCORES.read_text())
    modules()
    from team_diversity import slots_from_paste, summary

    per_ask, stored, redrawn = [], [], []
    for k in ASKS:
        cell = teams["cells"][str(k)]
        row = dict(ask=f"{k}/{GRID}", eligible=cell["eligible"], tried=cell["tried"], refused=cell["refused"],
                   species_without_corpus_set=cell["species_without_corpus_set"],
                   label_mean=float(np.mean([t["label"] for t in cell["teams"]])),
                   label_battles_median=float(np.median([t["label_battles"] for t in cell["teams"]])),
                   in_loop_teams=sum(t["in_loop_teams"] for t in cell["teams"]),
                   in_battle_records=sum(t["in_battle_records"] for t in cell["teams"]),
                   in_battle_database=sum(t["in_battle_database"] for t in cell["teams"]),
                   same_paste_in_both_arms=sum(t["stored"] == t["redrawn"] for t in cell["teams"]),
                   diversity=summary([slots_from_paste(t["stored"]) for t in cell["teams"]]))
        for arm, keep in zip(ARMS, (stored, redrawn), strict=True):
            sc = scores[f"{k}_{arm}"]
            y = np.array([s["score"] for s in sc], float)
            keep.append(y)
            row[arm] = dict(got=float(y.mean()), miss=float(y.mean() - k / GRID), n=len(y), battles=sum(s["battles"] for s in sc),
                            **scatter(y, [one_battle_noise(s["score"], s["columns"]) for s in sc]))
        per_ask.append(row)

    asks = [k / GRID for k in ASKS]
    window = [i for i, a in enumerate(asks) if COMPARE_WINDOW[0] <= a <= COMPARE_WINDOW[1]]
    errors = {}
    for arm, ys in zip(ARMS, (stored, redrawn), strict=True):
        errors[arm] = dict(error=line_error(asks, [y.mean() for y in ys]), noise_corrected=noise_corrected_error(asks, ys),
                           near_line=line_error(asks, [y.mean() for y in ys]) < NEAR_LINE,
                           error_on_asks_0_1_to_0_7=line_error([asks[i] for i in window], [ys[i].mean() for i in window]) if window else None)
    out = dict(date="2026-11-01", protocol="docs/training-teams-line.md", near_line_bar=NEAR_LINE,
               win_rate="one battle against each of the 49 distinct top-50 meta teams, behaviour-cloning policy on both sides, slots shuffled",
               fields={"stored": "the team's own 84 fields", "redrawn": "the team's 48 fields; Stat Points copied from same-species corpus teams, Stat Alignment repaired"},
               errors=errors, intervals=bootstrap(asks, stored, redrawn), per_ask=per_ask,
               got_redrawn_minus_stored=float(np.mean([(b - a).mean() for a, b in zip(stored, redrawn, strict=True)])),
               sparse_label_counts=teams["sparse_label_counts"], asks_from_label_counts_only=f"{SPARSE_ASKS[0]}/{GRID} to {SPARSE_ASKS[-1]}/{GRID}")
    save(OUT, out)
    print(f"{'ask':>6} {'label':>6} {'stored':>7} {'redrawn':>8} {'miss':>7} {'team sd':>8} {'line-ups':>9}")
    for r in per_ask:
        print(f"{r['ask']:>6} {r['label_mean']:6.3f} {r['stored']['got']:7.3f} {r['redrawn']['got']:8.3f} "
              f"{r['stored']['miss']:+7.3f} {r['stored']['team_sd']:8.3f} {r['diversity']['distinct_lineups']:9d}")
    for arm in ARMS:
        print(f"{arm}: error {errors[arm]['error']:.5f} {out['intervals'][arm]}, noise-corrected {errors[arm]['noise_corrected']:.5f}")
    print(f"redrawn minus stored: got {out['got_redrawn_minus_stored']:+.4f} {out['intervals']['got_redrawn_minus_stored']}, "
          f"error {out['intervals']['redrawn_minus_stored']}")
    print("wrote", OUT)


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else ""
    if stage not in ("draw", "battle", "report"): raise SystemExit(__doc__)
    {"draw": draw, "battle": battle, "report": report}[stage]()
