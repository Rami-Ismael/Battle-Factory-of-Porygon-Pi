"""Pareto curves and paired intervals for the least-likely trade. Read-only on the battle database, no new battles."""
import json
import pickle
import sqlite3
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, "/tmp/vgc-pilot/src")
sys.path.insert(0, str(Path.home() / "vgc-data/line_up_deletion_84/run"))
import heldout as H
import line_up_yield as LY
import matchup_db as MDB
from team_diversity import slots_from_paste

RES = Path.home() / "vgc-data/results"
CACHE = Path.home() / "vgc-data/least_likely_trade/matrix_cache.pkl"
ASKS, SEEDS, CUT, B = [4, 8, 12, 16], [1, 2, 3], 0.531, 2000
NEW = ["remask_ll_05", "remask_ll_20", "cold_all_t08", "cold_all_t06", "cold_sets_t06", "cold_sets_t03", "spend66"]
ARMS = ["plain", "remask_ll_10", *NEW]
NO12 = [4, 8, 16]
rng = np.random.default_rng(100)
ctl = json.load(open(RES / "stat_points_84_gen84_all_ep21.json"))["cells"]
old = json.load(open(RES / "remask84.json"))["cells"]
new = json.load(open(RES / "least_likely_trade.json"))["cells"]
cells = {}
for k in ASKS:
    for s in SEEDS:
        cells[("plain", k, s)] = ctl[f"gen84_all_ep21_a{k:02d}of24_s{s}"]
        cells[("remask_ll_10", k, s)] = old[f"remask_ll_10_a{k:02d}of24_s{s}"]
        for a in NEW: cells[(a, k, s)] = new[f"{a}_a{k:02d}of24_s{s}"]

con = sqlite3.connect(f"file:{MDB.DEFAULT_DB}?mode=ro", uri=True)
held = H.team_ids(con)
mem = {t: w for t, w in con.execute("SELECT team_id, weight FROM team_set WHERE set_name='top50'") if t not in held}
opps = sorted(mem); wt = np.array([mem[o] for o in opps], float)
pol = dict(con.execute("SELECT policy_id, row_policy FROM policy"))


def team_id(p): return con.execute("SELECT team_id FROM team WHERE format_id=? AND canon_hash=?", (MDB.FORMAT, MDB.canon_hash(MDB.canonical_paste(p)))).fetchone()[0]


if CACHE.exists():
    M, POLICY = pickle.load(open(CACHE, "rb"))
else:
    t0 = team_id(new["spend66_a04of24_s1"]["pastes"][0])
    POLICY = con.execute("SELECT policy_id FROM cell WHERE (lo=? OR hi=?) GROUP BY policy_id ORDER BY COUNT(*) DESC LIMIT 1", (t0, t0)).fetchone()[0]
    M = {}
    for key, c in cells.items():
        rows = []
        for p in c["pastes"]:
            r = MDB.row_cells(con, POLICY, team_id(p), {o: 1 for o in opps})
            rows.append([r[o][0] / r[o][1] for o in opps])
        M[key] = np.array(rows)
    pickle.dump((M, POLICY), open(CACHE, "wb"))
gap = max(np.abs(M[k] @ wt / wt.sum() - np.array(cells[k]["y"], float)).max() for k in M)
out = dict(policy=POLICY, matrix_gap=float(gap), contract={a: dict(cells=sum(k[0] == a for k in M), teams=sum(M[k].shape[0] for k in M if k[0] == a), battles=int(sum(M[k].size for k in M if k[0] == a))) for a in ARMS})
print("policy", POLICY, "gap", gap, out["contract"], flush=True)


def fit(row, asks):
    x = np.array(asks, float) / 24
    return float(np.polyfit(x, [row[a] for a in asks], 1)[0]), float(np.mean([row[a] for a in asks]))


def means(sc, tw):
    return {a: {k: float(np.mean([np.average(sc[(a, k, s)], weights=tw[(a, k, s)]) for s in SEEDS])) for k in ASKS} for a in ARMS}


pt = means({k: np.array(cells[k]["y"], float) for k in cells}, {k: np.ones(len(M[k])) for k in M})
boots = []
for b in range(B):
    ow = rng.multinomial(len(opps), np.full(len(opps), 1 / len(opps))) * wt
    sc = {k: M[k] @ ow / ow.sum() for k in M}
    tw = {k: rng.multinomial(64, np.full(64, 1 / 64)).astype(float) for k in M if k[0] != "spend66"}
    for k in M:
        if k[0] == "spend66": tw[k] = tw[("plain", k[1], k[2])]     # same teams, so the resample is paired
    boots.append(means(sc, tw))
    if b % 500 == 0: print("boot", b, flush=True)


def ci(v): return [float(np.quantile(v, .025)), float(np.quantile(v, .975))]


out["level_slope"] = {}
for a in ARMS:
    s, l = fit(pt[a], ASKS); s2, l2 = fit(pt[a], NO12)
    out["level_slope"][a] = dict(level=l, slope=s, level_no12=l2, slope_no12=s2, means=pt[a])
    for asks, tag in ((ASKS, ""), (NO12, "_no12")):
        out["level_slope"][a][f"slope{tag}_diff_vs_plain_ci"] = ci([fit(x[a], asks)[0] - fit(x["plain"], asks)[0] for x in boots])
        out["level_slope"][a][f"level{tag}_diff_vs_plain_ci"] = ci([fit(x[a], asks)[1] - fit(x["plain"], asks)[1] for x in boots])
out["gain"] = {}
for a in ARMS[1:]:
    out["gain"][a] = {}
    for k in ASKS:
        out["gain"][a][k] = dict(vs_plain=pt[a][k] - pt["plain"][k], vs_plain_ci=ci([x[a][k] - x["plain"][k] for x in boots]), vs_plain_by_seed=[float(np.mean(cells[(a, k, s)]["y"]) - np.mean(cells[("plain", k, s)]["y"])) for s in SEEDS],
                                 vs_ll10=pt[a][k] - pt["remask_ll_10"][k], vs_ll10_ci=ci([x[a][k] - x["remask_ll_10"][k] for x in boots]), vs_ll10_by_seed=[float(np.mean(cells[(a, k, s)]["y"]) - np.mean(cells[("remask_ll_10", k, s)]["y"])) for s in SEEDS])

train_set, common = LY.references("gen84_all_ep21")


def parts(paste):
    sl = slots_from_paste(paste)
    sets = [json.dumps({f: s.get(f) for f in ("species", "ability", "item", "nature", "moves")}, sort_keys=True) for s in sl]
    return tuple(sorted(s["species"] for s in sl)), sets


def hill2(c): c = np.array(list(c), float); p = c / c.sum(); return float(1 / (p ** 2).sum())


def hill1(c): c = np.array(list(c), float); p = c / c.sum(); return float(np.exp(-(p * np.log(p)).sum()))


def at66(team): return sum((sum(e) if isinstance(e, (list, tuple)) else sum(e.values())) == 66 for _, e in team)


def describe(cs):
    ys = np.array([v for c in cs for v in c["y"]], float)
    P = [parts(p) for c in cs for p in c["pastes"]]
    lus = [x[0] for x in P]; cnt = Counter(lus)
    sets = [s for x in P for s in x[1]]
    groups = {}
    for l, x in zip(lus, P): groups.setdefault(l, []).append(dict(zip(sorted(json.loads(s)["species"] for s in x[1]), [None] * 6)) and {json.loads(s)["species"]: s for s in x[1]})
    agree = []
    for v in groups.values():
        if len(v) < 2: continue
        idx = list(combinations(range(len(v)), 2))
        if len(idx) > 100: idx = [idx[i] for i in rng.choice(len(idx), 100, replace=False)]
        agree += [np.mean([v[i][k] == v[j][k] for k in v[i]]) for i, j in idx]
    strong = {l for l, y in zip(lus, ys) if y >= CUT}
    return dict(teams=len(ys), win=float(ys.mean()), by_seed=[float(np.mean(c["y"])) for c in cs], distinct_teams=len({p for c in cs for p in c["pastes"]}), line_ups=len(cnt), effective_line_ups=hill2(cnt.values()), exp_entropy_line_ups=hill1(cnt.values()),
                new_vs_training=len(set(cnt) - train_set), new_vs_loop_and_records=len(set(cnt) - common), distinct_sets=len(set(sets)), identical_set_share_same_line_up=float(np.mean(agree)) if agree else None,
                teams_at_cut=int((ys >= CUT).sum()), line_ups_at_cut=len(strong), line_ups_at_cut_new_vs_training=len(strong - train_set), slots_at_66=float(np.mean([at66(t) for c in cs for t in c["stat_points"]])))


points = {}
for k in range(1, 24):
    points[f"plain|{k}"] = describe([ctl[f"gen84_all_ep21_a{k:02d}of24_s{s}"] for s in SEEDS])
for a in ARMS[1:]:
    for k in ASKS: points[f"{a}|{k}"] = describe([cells[(a, k, s)] for s in SEEDS])
out["points"] = points
out["pooled"] = {a: describe([cells[(a, k, s)] for k in ASKS for s in SEEDS]) for a in ARMS}


def frontier(names, x="line_ups"):
    keep = []
    for n in names:
        p = points[n]
        if not any(o != n and points[o]["win"] >= p["win"] and points[o][x] >= p[x] and (points[o]["win"] > p["win"] or points[o][x] > p[x]) for o in names): keep.append(n)
    return sorted(keep, key=lambda n: points[n][x])


def scored(n): return int(n.split("|")[1]) <= 17


with12 = [n for n in points if scored(n)]; without12 = [n for n in with12 if not n.endswith("|12")]
out["frontiers"] = {f"{x}|{tag}": frontier(names, x) for x in ("line_ups", "effective_line_ups", "new_vs_training", "distinct_sets") for tag, names in (("with_12", with12), ("without_12", without12))}
# How often a point stays on the curve when its win rate is resampled over teams; line-up counts held fixed.
ys = {n: (np.concatenate([ctl[f"gen84_all_ep21_a{int(n.split('|')[1]):02d}of24_s{s}"]["y"] for s in SEEDS]) if n.startswith("plain") else np.concatenate([cells[(n.split("|")[0], int(n.split("|")[1]), s)]["y"] for s in SEEDS])).astype(float) for n in points}
for tag, names in (("with_12", with12), ("without_12", without12)):
    hits = Counter(); arr = np.array([points[n]["line_ups"] for n in names])
    for _ in range(1000):
        w = np.array([rng.choice(ys[n], len(ys[n])).mean() for n in names])
        for i, n in enumerate(names):
            if not np.any((w >= w[i]) & (arr >= arr[i]) & ((w > w[i]) | (arr > arr[i]))): hits[n] += 1
    out[f"frontier_stability_{tag}"] = {n: hits[n] / 1000 for n in names if hits[n] >= 50}

json.dump(out, open(RES / "least_likely_trade_curves.json", "w"), indent=1)
f = lambda x: round(x, 3) if isinstance(x, float) else ([f(i) for i in x] if isinstance(x, list) else ({k: f(v) for k, v in x.items()} if isinstance(x, dict) else x))
print(json.dumps(f(out["level_slope"]))); print(json.dumps(f(out["gain"])))
for a in ARMS: print(a, json.dumps(f({k: v for k, v in out["pooled"][a].items()})))
for n, v in points.items():
    if not n.startswith("plain") or int(n.split("|")[1]) in ASKS: print(n, json.dumps(f({k: v[k] for k in ("win", "line_ups", "effective_line_ups", "new_vs_training", "distinct_teams", "distinct_sets", "identical_set_share_same_line_up", "teams_at_cut", "line_ups_at_cut", "line_ups_at_cut_new_vs_training", "slots_at_66")})))
print(json.dumps(out["frontiers"])); print(json.dumps(f(out["frontier_stability_with_12"]))); print(json.dumps(f(out["frontier_stability_without_12"])))
print("CURVES_DONE")
