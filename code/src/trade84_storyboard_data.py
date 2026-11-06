"""Numbers behind the storyboard figures for issue #100. Read-only on the results files, no new battles.

Win rate is against the top-50 meta teams, behaviour-cloning policy on both sides. Intervals resample teams
within each cell (64 teams x 3 seeds per ask), not opponents, so they are slightly narrower than curve_analysis.py's.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/tmp/vgc-pilot/src")
from team_diversity import slots_from_paste

RES = Path.home() / "vgc-data/results"
OUT = Path.home() / "vgc-data/results/least_likely_trade_storyboard.json"
ASKS, NO12, SEEDS, B = [4, 8, 12, 16], [4, 8, 16], [1, 2, 3], 2000
ctl = json.load(open(RES / "stat_points_84_gen84_all_ep21.json"))["cells"]
old = json.load(open(RES / "remask84.json"))["cells"]
new = json.load(open(RES / "least_likely_trade.json"))["cells"]
nuc = json.load(open(RES / "least_likely_trade_nucleus.json"))["cells"]
NAMES = {
    "plain": "No remasking",
    "spend66": "Stat Points raised to 66",
    "cold_sets_t06": "Temperature 0.6 split-variance",
    "cold_sets_t03": "Temperature 0.3 split-variance",
    "nucleus_sets_p09": "Nucleus 0.9 split-variance",
    "nucleus_sets_p07": "Nucleus 0.7 split-variance",
    "nucleus_sets_p05": "Nucleus 0.5 split-variance",
    "cold_all_t08": "Temperature 0.8 on all fields",
    "cold_all_t06": "Temperature 0.6 on all fields",
    "nucleus_all_p09": "Nucleus 0.9 on all fields",
    "remask_ll_05": "Least-likely 5% remasked",
    "remask_ll_10": "Least-likely 10% remasked",
    "remask_ll_20": "Least-likely 20% remasked",
    "remask_conf_loop": "Confidence-weighted remasking (paper rule)",
    "remask_even_loop": "Even-odds remasking (paper rule)",
}


def cell(a, k, s):
    if a == "plain": return ctl[f"gen84_all_ep21_a{k:02d}of24_s{s}"]
    src = old if a in ("remask_ll_10", "remask_conf_loop", "remask_even_loop") else nuc if a.startswith("nucleus") else new
    return src[f"{a}_a{k:02d}of24_s{s}"]


C = {(a, k, s): cell(a, k, s) for a in NAMES for k in ASKS for s in SEEDS}
Y = {key: np.array([v if v is not None else np.nan for v in c["y"]], float) for key, c in C.items()}


def parts(paste):
    sl = slots_from_paste(paste)
    return tuple(sorted(s["species"] for s in sl)), [json.dumps({f: s.get(f) for f in ("species", "ability", "item", "nature", "moves")}, sort_keys=True) for s in sl]


P = {key: [parts(p) for p in c["pastes"]] for key, c in C.items()}


def counts(a, asks):
    ps = [x for k in asks for s in SEEDS for x in P[(a, k, s)]]
    return dict(line_ups=len({x[0] for x in ps}), distinct_sets=len({s for x in ps for s in x[1]}), teams=len(ps))


def means(y):
    return {a: {k: float(np.mean([np.nanmean(y[(a, k, s)]) for s in SEEDS])) for k in ASKS} for a in NAMES}


def error(m, asks): return float(np.mean([(m[k] - k / 24) ** 2 for k in asks]))


def pooled(m, asks): return float(np.mean([m[k] for k in asks]))


rng = np.random.default_rng(106)
pt = means(Y)
boots = []
idx = {key: len(v) for key, v in Y.items()}
for b in range(B):
    draw = {}
    for key, v in Y.items():
        if key[0] == "spend66": continue
        draw[key] = rng.integers(0, idx[key], idx[key])
    for key in Y:
        if key[0] == "spend66": draw[key] = draw[("plain", key[1], key[2])]     # same teams as no remasking: paired
    boots.append(means({key: Y[key][draw[key]] for key in Y}))


def ci(v): return [float(np.quantile(v, .025)), float(np.quantile(v, .975))]


out = dict(names=NAMES, note=__doc__.strip(), arms={})
for a in NAMES:
    r = dict(name=NAMES[a], per_ask={}, pooled={}, error={})
    for k in ASKS:
        r["per_ask"][k] = dict(win=pt[a][k], win_ci=ci([x[a][k] for x in boots]), gain=pt[a][k] - pt["plain"][k], gain_ci=ci([x[a][k] - x["plain"][k] for x in boots]),
                               vs_cold_sets_t03=pt[a][k] - pt["cold_sets_t03"][k], vs_cold_sets_t03_ci=ci([x[a][k] - x["cold_sets_t03"][k] for x in boots]), **counts(a, [k]))
    for tag, asks in (("all", ASKS), ("without_12", NO12)):
        r["pooled"][tag] = dict(win=pooled(pt[a], asks), gain=pooled(pt[a], asks) - pooled(pt["plain"], asks), gain_ci=ci([pooled(x[a], asks) - pooled(x["plain"], asks) for x in boots]),
                                vs_cold_sets_t03=pooled(pt[a], asks) - pooled(pt["cold_sets_t03"], asks), vs_cold_sets_t03_ci=ci([pooled(x[a], asks) - pooled(x["cold_sets_t03"], asks) for x in boots]), **counts(a, asks))
        r["error"][tag] = dict(error=error(pt[a], asks), diff_vs_plain=error(pt[a], asks) - error(pt["plain"], asks), diff_vs_plain_ci=ci([error(x[a], asks) - error(x["plain"], asks) for x in boots]))
    out["arms"][a] = r
fresh = json.load(open(RES / "cold_sets_t03_fresh_report.json"))["summary"]
prev = json.load(open(RES / "least_likely_trade_final_lists.json"))["summary"]
out["rebattle"] = {"cold_sets_t03": fresh, **prev}
json.dump(out, open(OUT, "w"), indent=1)
for a, r in out["arms"].items():
    p, q = r["pooled"]["all"], r["pooled"]["without_12"]
    print(f"{r['name']:44s} win {p['win']:.3f} gain {p['gain']:+.3f} {p['gain_ci'][0]:+.3f}..{p['gain_ci'][1]:+.3f} | no12 {q['gain']:+.3f} | lu {p['line_ups']:3d}/{q['line_ups']:3d} sets {p['distinct_sets']:4d}/{q['distinct_sets']:4d} | err {r['error']['all']['error']:.4f} d {r['error']['all']['diff_vs_plain']:+.4f} {r['error']['all']['diff_vs_plain_ci'][0]:+.4f}..{r['error']['all']['diff_vs_plain_ci'][1]:+.4f} | vs t03 {p['vs_cold_sets_t03']:+.3f} {p['vs_cold_sets_t03_ci'][0]:+.3f}..{p['vs_cold_sets_t03_ci'][1]:+.3f}")
print("rebattle", json.dumps({a: {k: v[k] for k in ("candidates", "held", "candidates_without_12", "held_without_12") if k in v} for a, v in out["rebattle"].items()}))
