"""Slope ticket analysis for remasking on the all-teams 84-field model: read-only on the battle database.

Per-seed slopes, paired differences against no remasking with a bootstrap over whole teams within each cell and shared
opponents across arms, error against the line, accumulation, new line-ups, and win rate at equal Stat Points spend.

    python remask84_slope_analysis.py [output.json]
"""
import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path

import numpy as np

import ask_grid84  # noqa: F401  (sets up sys.path for the modules below)
import heldout as H
import matchup_db as MDB
from lineup_reference import references, species_of

RES = Path.home() / "vgc-data/results"
ASKS, SEEDS, ARMS = [4, 8, 12, 16], [1, 2, 3], ["plain", "remask_ll_10", "remask_conf_loop", "remask_even_loop"]
B = 4000
CUT = 0.531


d = json.load(open(RES / "remask84.json"))
ctl = json.load(open(RES / "stat_points_84_gen84_all_ep21.json"))["cells"]
cells = {("plain", k, s): ctl[f"gen84_all_ep21_a{k:02d}of24_s{s}"] for k in ASKS for s in SEEDS}
for arm in ARMS[1:]:
    for k in ASKS:
        for s in SEEDS:
            cells[(arm, k, s)] = d["cells"][f"{arm}_a{k:02d}of24_s{s}"]

con = sqlite3.connect(f"file:{MDB.DEFAULT_DB}?mode=ro", uri=True)
held = H.team_ids(con)
opps = sorted(c for (c,) in con.execute("SELECT team_id FROM team_set WHERE set_name='top50'") if c not in held)
out = dict(opponents=len(opps), policies={}, contract={}, by_seed={}, paired={}, line={}, novelty={}, spend={})


def matrix(c):
    rows = []
    for p in c["pastes"]:
        h = MDB.canon_hash(MDB.canonical_paste(p))
        t = con.execute("SELECT team_id FROM team WHERE format_id=? AND canon_hash=?", (MDB.FORMAT, h)).fetchone()[0]
        policy = con.execute("SELECT policy_id FROM cell WHERE lo=? OR hi=? GROUP BY policy_id ORDER BY COUNT(*) DESC LIMIT 1", (t, t)).fetchone()[0]
        c.setdefault("_policies", Counter())[policy] += 1
        r = MDB.row_cells(con, policy, t, {o: 1 for o in opps})
        rows.append([r[o][0] / r[o][1] if o in r else np.nan for o in opps])
    return np.array(rows)


M = {key: matrix(c) for key, c in cells.items()}
for arm in ARMS:
    at = [key for key in M if key[0] == arm]
    out["policies"][arm] = dict(sum((cells[key].pop("_policies") for key in at), Counter()))
    gap = max(abs(np.nanmean(M[key], 1) - np.array(cells[key]["y"])).max() for key in at)
    out["contract"][arm] = dict(cells=len(at), teams=sum(len(cells[key]["pastes"]) for key in at), battles=int(sum(np.isfinite(M[key]).sum() for key in at)),
                                missing=int(sum(np.isnan(M[key]).sum() for key in at)), largest_gap_to_saved_score=float(gap))


def fit(row, asks):
    return float(np.polyfit([a / 24 for a in asks], [row[a] for a in asks], 1)[0])


NO12 = [4, 8, 16]
for arm in ARMS:
    bys = {}
    for s in SEEDS:
        row = {k: float(np.mean(cells[(arm, k, s)]["y"])) for k in ASKS}
        bys[s] = dict(slope=fit(row, ASKS), slope_without_12=fit(row, NO12), top_ask=row[16], by_ask=row)
    out["by_seed"][arm] = bys

rng = np.random.default_rng(90)
boot = {arm: {k: np.zeros((B, len(SEEDS))) for k in ASKS} for arm in ARMS}
for b in range(B):
    ow = rng.multinomial(len(opps), np.full(len(opps), 1 / len(opps))).astype(float)
    for arm in ARMS:
        for k in ASKS:
            for j, s in enumerate(SEEDS):
                m = M[(arm, k, s)]
                tw = rng.multinomial(len(m), np.full(len(m), 1 / len(m)))
                boot[arm][k][b, j] = np.nansum(tw[:, None] * m * ow[None, :]) / (len(m) * ow.sum())


def ci(x): return [float(np.quantile(x, .025)), float(np.quantile(x, .975))]


def slopes(arm, asks):
    y = np.stack([boot[arm][k].mean(1) for k in asks], 1)
    return np.polyfit([a / 24 for a in asks], y.T, 1)[0]


point = {arm: {k: float(np.mean([np.mean(cells[(arm, k, s)]["y"]) for s in SEEDS])) for k in ASKS} for arm in ARMS}
for arm in ARMS:
    out["line"][arm] = dict(win=point[arm], squared_error_by_ask={k: (point[arm][k] - k / 24) ** 2 for k in ASKS},
                            mse_without_12=float(np.mean([(point[arm][k] - k / 24) ** 2 for k in NO12])), mse_all_four=float(np.mean([(point[arm][k] - k / 24) ** 2 for k in ASKS])))
for arm in ARMS[1:]:
    p = {}
    for name, asks in (("slope", ASKS), ("slope_without_12", NO12)):
        delta = slopes(arm, asks) - slopes("plain", asks)
        p[name] = dict(arm=fit(point[arm], asks), plain=fit(point["plain"], asks), difference=fit(point[arm], asks) - fit(point["plain"], asks), interval=ci(delta),
                       by_seed=[out["by_seed"][arm][s][name] - out["by_seed"]["plain"][s][name] for s in SEEDS])
    for k in ASKS:
        p[f"win_at_{k}"] = dict(difference=point[arm][k] - point["plain"][k], interval=ci(boot[arm][k].mean(1) - boot["plain"][k].mean(1)),
                                by_seed=[out["by_seed"][arm][s]["by_ask"][k] - out["by_seed"]["plain"][s]["by_ask"][k] for s in SEEDS])
    out["paired"][arm] = p

train_set, common = references()
out["reference_sizes"] = dict(training=len(train_set), loop_and_records=len(common))
plain_lineups = {species_of(p) for key, c in cells.items() if key[0] == "plain" for p in c["pastes"]}
for arm in ARMS:
    teams = [(species_of(p), y) for key, c in cells.items() if key[0] == arm for p, y in zip(c["pastes"], c["y"])]
    seen = {t for t, _ in teams}
    strong = {t for t, y in teams if round(y, 4) >= CUT}
    out["novelty"][arm] = dict(teams=len(teams), line_ups=len(seen), new_vs_training=len(seen - train_set), new_vs_loop_and_records=len(seen - common), new_vs_both=len(seen - train_set - common),
                               shared_with_no_remasking=len(seen & plain_lineups), teams_at_cut=sum(round(y, 4) >= CUT for _, y in teams), line_ups_at_cut_unconfirmed=len(strong),
                               line_ups_at_cut_new_vs_training=len(strong - train_set), line_ups_at_cut_new_vs_loop_and_records=len(strong - common), line_ups_at_cut_new_vs_both=len(strong - train_set - common),
                               effective_line_ups=float(1 / sum((v / len(teams)) ** 2 for v in Counter(t for t, _ in teams).values())),
                               accumulation=(d["fits"][arm]["accumulation"]))


def at_66(team):
    return sum((sum(e) if isinstance(e, (list, tuple)) else sum(e.values())) == 66 for _, e in team)


# Win rate at each count of slots spending all 66 Stat Points, and the least-likely arm reweighted to no remasking's mix of those counts.
for k in ASKS:
    rows = {}
    for arm in ARMS:
        spend = np.array([at_66(t) for s in SEEDS for t in cells[(arm, k, s)]["stat_points"]])
        y = np.array([v for s in SEEDS for v in cells[(arm, k, s)]["y"]])
        rows[arm] = {int(n): (float(y[spend == n].mean()), int((spend == n).sum())) for n in np.unique(spend)}
    base = rows["plain"]; total = sum(v[1] for v in base.values())
    row = dict(by_slots_at_66=rows)
    for arm in ARMS[1:]:
        shared = [n for n in base if n in rows[arm]]
        w = sum(base[n][1] for n in shared)
        row[f"{arm}_matched_gain"] = dict(gain=sum(base[n][1] * (rows[arm][n][0] - base[n][0]) for n in shared) / w, share_of_control_covered=w / total)
    out["spend"][k] = row

path = Path(sys.argv[1]) if len(sys.argv) > 1 else RES / "remask84_slope_analysis.json"
json.dump(out, open(path, "w"), indent=1)
show = {x: out[x] for x in ("opponents", "policies", "contract", "paired", "line", "reference_sizes")}
print(json.dumps(show))
print(json.dumps({arm: {k: v for k, v in r.items() if k != "accumulation"} for arm, r in out["novelty"].items()}))
print(json.dumps({k: {x: y for x, y in r.items() if x != "by_slots_at_66"} for k, r in out["spend"].items()}))
print(json.dumps({arm: {s: [r["slope"], r["slope_without_12"]] for s, r in v.items()} for arm, v in out["by_seed"].items()}))
print("SLOPE_ANALYSIS_DONE", path)
