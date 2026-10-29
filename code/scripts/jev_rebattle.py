"""Re-battle the held-out grid's Jev and Mercury teams with shuffled slots and 8 battles per opponent (2026-10-28).

Every Mask Lab grid battled its fills in listed slot order, one battle per top-50 opponent (49 per team). The
behaviour-cloning policy picks its four at team preview by slot number (src/slot_order.py), so those scores carry the
slot-order bias, and one battle per opponent leaves about ±7 points of battle noise per team. This re-battles, on the
held-out grid, every legal team from Jev's top pick, Mercury at its four reasoning settings, the random legal fill and
the five unmasked teams: 8 battles per opponent (392 per team), six Pokémon shuffled every battle, the behaviour-cloning
policy on both sides, against the Mask Lab's 49 top-50 teams paste for paste, then the same against the held-out teams
(ranks 51-100) minus the five unmasked teams, which every fill on this grid is cut from (45 teams). Battles go to a
private database.

    /tmp/vgc-pilot/.venv/bin/python scripts/jev_rebattle.py battle          # needs /tmp/vgc-pilot staging
    /tmp/vgc-pilot/.venv/bin/python scripts/jev_rebattle.py report          # reads the database

report writes ~/vgc-data/jev/rebattle-heldout.json: for each opponent set (top-50, held-out) and mask type, usable win
rate (a team that isn't legal scores 0) and, on masks both filled legally, Jev minus each Mercury setting, with 95%
intervals from resampling masks and from resampling the five unmasked teams; plus each arm's score per mask (None when
it has no legal team), which scripts/jev_numbers.py reads.
"""
import argparse
import json
import random
import sqlite3
import statistics
import sys
import time
from pathlib import Path

GRID = Path("/Users/ramiismael/Documents/yakumsi-vault/Personal Project/Using Advance method search in the whole search of "
            "possible vgc pokemon format to find the right counter to a pokemon team/team_completion_pipeline/runs/heldout-starters")
DB = Path.home() / "vgc-data/jev/rebattle_heldout.sqlite"
OUT = Path.home() / "vgc-data/jev/rebattle-heldout.json"
SRC = Path("/tmp/vgc-pilot/src")
PER_CELL, SEED, BATCH = 8, 20261028, 25
ARMS = {"jev": "jev:top:", "mercury-off": "mercury-off:", "mercury-low": "mercury-low:",
        "mercury-medium": "mercury-medium:", "mercury": "mercury:", "random": "random:"}


def candidates():
    """The grid's candidates for the re-battled arms, each with its arm and task id (None for an unmasked team)."""
    out = []
    for c in json.loads((GRID / "candidates.json").read_text())["candidates"]:
        if not c["paste"]: continue
        if c["kind"] == "original": out.append(dict(c, arm="unmasked", task_id=None)); continue
        arm = next((a for a, p in ARMS.items() if c["label"].startswith(p)), None)
        if arm: out.append(dict(c, arm=arm, task_id=c["label"].rsplit(":", 1)[1]))
    return out


def battle(_a):
    sys.path.insert(0, str(SRC))
    import matchup_db as M, top50 as T50
    from showdown_validator import Validator
    con = M.connect(DB)
    pid = M.get_policy(con, note="Jev/Mercury held-out re-battle, slots shuffled, 2026-10-28")
    assert "+slots-shuffled" in con.execute("SELECT row_policy FROM policy WHERE policy_id=?", (pid,)).fetchone()[0]
    cands, val = candidates(), Validator()
    try:
        with con:
            ids = {c["label"]: M.add_team(con, c["paste"], f"jev-rebattle:{c['arm']}", val)[0] for c in cands}
            con.execute("DELETE FROM team_set WHERE set_name='top50'")
            for t in json.load(open(T50.TOP50)):          # as matchup_db.seed: a team listed twice counts twice
                tid = M.add_team(con, Path(T50.resolve(t["id"])).read_text(), "vgcpastes", val)[0]
                con.execute("INSERT INTO team_set VALUES ('top50', ?, 1) ON CONFLICT (set_name, team_id) "
                            "DO UPDATE SET weight = weight + 1", (tid,))
        sources = {ids[c["label"]] for c in cands if c["arm"] == "unmasked"}
        hcols = [t for t in M.seed_heldout(con, val) if t not in sources]
    finally:
        val.close()
    cols = M.set_members(con, "top50")
    lab = sqlite3.connect("file:" + str(Path.home() / "vgc-data/mask_lab_matrix.sqlite") + "?mode=ro&immutable=1", uri=True)
    want = {h for (h,) in lab.execute("SELECT canon_hash FROM team JOIN team_set USING (team_id) WHERE set_name='top50'")}
    got = {h for (h,) in con.execute(f"SELECT canon_hash FROM team WHERE team_id IN ({','.join('?' * len(cols))})", cols)}
    if want != got: raise SystemExit(f"top-50 columns differ from the Mask Lab's: {len(want ^ got)} pastes")
    marks = ",".join("?" * len(ids))
    rows = sorted(t for (t,) in con.execute(f"SELECT team_id FROM team WHERE legal=1 AND team_id IN ({marks})", list(ids.values())))
    print(f"policy {pid} · {len(cands)} candidates · {len(rows)} distinct legal teams · {len(cols)} top-50 and "
          f"{len(hcols)} held-out opponents · <= {len(rows) * (len(cols) + len(hcols)) * PER_CELL} battles", flush=True)
    if _a.dry_run: con.close(); return
    t0, done = time.time(), 0
    for i in range(0, len(rows), BATCH):               # evaluation=True: the unmasked teams are held-out teams
        done += M.ensure(con, pid, rows[i:i + BATCH], cols, PER_CELL, seed=SEED + i, note="jev_rebattle heldout", evaluation=True)
        print(f"top-50 rows {min(i + BATCH, len(rows))}/{len(rows)} · {done} battles · {time.time() - t0:.0f}s", flush=True)
    for i in range(0, len(rows), BATCH):
        done += M.ensure(con, pid, rows[i:i + BATCH], hcols, PER_CELL, seed=SEED + 50000 + i, note="jev_rebattle heldout opponents", evaluation=True)
        print(f"held-out rows {min(i + BATCH, len(rows))}/{len(rows)} · {done} battles · {time.time() - t0:.0f}s", flush=True)
    con.close()


def team_scores(con, pid, cols, set_name="top50"):
    """{team_id: (score, battles)} over the given columns, set weights, own column skipped (matchup_db.score)."""
    weight = dict(con.execute("SELECT team_id, weight FROM team_set WHERE set_name=?", (set_name,)))
    out = {}
    for t in {lo for (lo,) in con.execute("SELECT lo FROM cell WHERE policy_id=?", (pid,))} | \
             {hi for (hi,) in con.execute("SELECT hi FROM cell WHERE policy_id=?", (pid,))}:
        num = den = n = 0
        for lo, hi, lw, hw, b in con.execute("SELECT lo, hi, lo_wins, hi_wins, battles FROM cell WHERE policy_id=? AND (lo=? OR hi=?)", (pid, t, t)):
            c = hi if lo == t else lo
            if c == t or c not in cols or not b: continue
            num += weight[c] * (lw if lo == t else hw) / b; den += weight[c]; n += b
        if den: out[t] = (num / den, n)
    return out


def interval(pairs, key, reps=4000, seed=7):
    """Mean of the differences and its 95% bootstrap interval, resampling masks, then resampling unmasked teams."""
    rng = random.Random(seed)
    diffs = [d for _, d in pairs]
    mean = statistics.mean(diffs)
    by_mask = sorted(statistics.mean(rng.choices(diffs, k=len(diffs))) for _ in range(reps))
    fam = {}
    for src, d in pairs: fam.setdefault(src, []).append(d)
    groups = list(fam.values())
    by_team = []
    for _ in range(reps):
        pick = [d for g in rng.choices(groups, k=len(groups)) for d in g]
        by_team.append(statistics.mean(pick))
    by_team.sort()
    q = lambda xs: [round(xs[int(0.025 * reps)], 4), round(xs[int(0.975 * reps) - 1], 4)]
    return {key: round(mean, 4), "masks": len(diffs), "ci_masks": q(by_mask), "ci_teams": q(by_team)}


def report(_a):
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    pid = con.execute("SELECT max(policy_id) FROM policy WHERE row_policy LIKE '%+slots-shuffled'").fetchone()[0]
    canon = {h: t for t, h in con.execute("SELECT team_id, canon_hash FROM team")}
    legal = {t for (t,) in con.execute("SELECT team_id FROM team WHERE legal=1")}
    sys.path.insert(0, str(SRC))
    import matchup_db as M                              # canonical form only: the team ids come from this database
    tid = lambda c: canon.get(M.canon_hash(M.canonical_paste(c["paste"])))
    cands = candidates()
    sources = {tid(c) for c in cands if c["arm"] == "unmasked"}
    opponents = {"top50": ("top50", {t for (t,) in con.execute("SELECT team_id FROM team_set WHERE set_name='top50'")}),
                 "heldout": ("heldout51_100", {t for (t,) in con.execute("SELECT team_id FROM team_set WHERE set_name='heldout51_100'")} - sources)}
    tasks = {t["id"]: t for t in json.loads((GRID / "eval2.tasks.json").read_text())["tasks"]}
    kinds = sorted({t["task"] for t in tasks.values()})
    out = {"date": "2026-10-28", "grid": "heldout", "policy_id": pid, "battles_per_opponent": PER_CELL,
           "policy": "bc_100.zip on both sides, six Pokémon shuffled every battle",
           "opponents": {"top50": "the Mask Lab's 49 top-50 teams",
                         "heldout": "the held-out teams (ranks 51-100) minus the five unmasked teams"}, "sets": {}}
    for name, (set_name, cols) in opponents.items():
        scores = team_scores(con, pid, cols, set_name)
        win, unmasked = {}, {}                          # (arm, task_id) -> score, or None when no legal team
        for c in cands:
            t = tid(c)
            sc = scores.get(t, (None,))[0] if t in legal else None
            if c["arm"] == "unmasked": unmasked[c["source"]] = sc
            else: win[(c["arm"], c["task_id"])] = sc
        res = {"opponents": len(cols), "unmasked": unmasked, "kinds": {},   # scores: per mask, for jev_numbers.py
               "scores": {arm: {i: win.get((arm, i)) for i in tasks} for arm in ARMS}}
        for kind in [*kinds, "all"]:
            ids = [i for i, t in tasks.items() if kind == "all" or t["task"] == kind]
            row = {"masks": len(ids), "usable": {}, "jev_minus": {}}
            for arm in ARMS:
                vals = [win.get((arm, i)) or 0 for i in ids]
                row["usable"][arm] = interval([(tasks[i]["source"], v) for i, v in zip(ids, vals)], "mean")
            for arm in ("mercury-off", "mercury-low", "mercury-medium", "mercury"):
                pairs = [(tasks[i]["source"], win[("jev", i)] - win[(arm, i)]) for i in ids
                         if win.get(("jev", i)) is not None and win.get((arm, i)) is not None]
                if len(pairs) >= 2: row["jev_minus"][arm] = interval(pairs, "diff")
            res["kinds"][kind] = row
        out["sets"][name] = res
        print(f"\nagainst {name} ({len(cols)} teams); unmasked teams " +
              ", ".join(f"{k} {v:.3f}" for k, v in unmasked.items() if v is not None))
        print(f"{'kind':10} {'masks':>5}  " + "  ".join(f"{a:>15}" for a in ARMS) + "   Jev − Mercury high [masks CI]")
        for kind, row in res["kinds"].items():
            jm = row["jev_minus"].get("mercury")
            print(f"{kind:10} {row['masks']:>5}  " + "  ".join(f"{row['usable'][a]['mean']:>15.3f}" for a in ARMS) +
                  (f"   {jm['diff']:+.3f} [{jm['ci_masks'][0]:+.3f}, {jm['ci_masks'][1]:+.3f}] on {jm['masks']}" if jm else ""))
    old = json.loads((Path.home() / "vgc-data/jev/heldout-report.json").read_text())
    out["old_listed_order"] = {"jev_top_mean": round(statistics.mean(m["jev_top"] for m in old["masks"]), 4)}
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print("wrote", OUT)

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("battle").add_argument("--dry-run", action="store_true", help="add the teams, count the battles, run none")
    sub.add_parser("report")
    a = p.parse_args()
    globals()[a.command](a)
