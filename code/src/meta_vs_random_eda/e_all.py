import json, sys, csv, sqlite3, numpy as np
sys.path.insert(0, sys.argv[1]); from mcat import parse, cat, CATS; from feats import mon_features
V = json.load(open(sys.argv[1] + "/c_moves.json"))["cost"]; val = {c: V[c]["value"][0] for c in CATS[:6]}; val["Other status"] = 0
con = sqlite3.connect("/Users/ramiismael/vgc-data/matchup_regmb.sqlite")
top = {r[0] for r in con.execute("select team_id from team_set where set_name='top50'")}
sc = {int(r[0]): (r[1], int(r[2]) / int(r[3]), int(r[3])) for r in csv.reader(open(sys.argv[1] + "/scores.csv"))}
fill_starts = {f["start_team"] for f in json.load(open('/Users/ramiismael/Documents/code/vgc-team-generator-pilot/results/random_fill_tasks.json'))["tasks"]}
X, Y, G = [], [], []
for tid, p in con.execute("select team_id, paste from team where legal=1"):
    if tid not in sc or tid in top or sc[tid][2] < 40: continue
    org = sc[tid][0]
    if org.startswith("randomfill") or tid in fill_starts: continue   # the fit set; keep this out-of-sample
    ms = parse(p); cnt = {c: 0 for c in CATS}
    for m in ms:
        for mv in m["moves"]: cnt[cat(mv, m)] += 1
    fs = [mon_features(m) for m in ms]
    X.append([sum(val[c] * cnt[c] for c in CATS)] + [cnt[c] for c in CATS] + [np.mean([f["bst"] or 0 for f in fs]), np.mean([f["ev_max"] for f in fs])])
    Y.append(sc[tid][1]); G.append(org.split(":")[0])
X = np.array(X); Y = np.array(Y); G = np.array(G)
print("teams", len(Y), "mean win rate", Y.mean().round(3))
from scipy.stats import spearmanr
names = ["move-mix score"] + ["n " + c for c in CATS] + ["mean BST", "mean max stat points"]
out = {}
for j, n in enumerate(names):
    r = spearmanr(X[:, j], Y).correlation; out[n] = r; print(f"  {n:24s} spearman {r:+.3f}")
# binned curve of win rate by number of Protect
pc = X[:, 1].astype(int); curve = {int(k): [float(Y[pc == k].mean()), int((pc == k).sum())] for k in np.unique(pc)}
print("win rate by # Protect:", {k: (round(v[0], 3), v[1]) for k, v in curve.items()})
# within-origin correlation (removes generator differences)
w = []
for g in np.unique(G):
    m = G == g
    if m.sum() >= 200: w.append((g, m.sum(), spearmanr(X[m, 0], Y[m]).correlation))
print("within-origin spearman of move-mix score (origins with >=200 teams):", np.median([x[2] for x in w]).round(3), "over", len(w), "origins; share positive", np.mean([x[2] > 0 for x in w]).round(2))
sp = X[:, 4].astype(int); scurve = {int(k): [float(Y[sp == k].mean()), int((sp == k).sum())] for k in np.unique(sp)}; print("win rate by # spread attacks:", {k: (round(v[0], 3), v[1]) for k, v in scurve.items()})
json.dump(dict(spread_curve=scurve, n=len(Y), spearman=out, protect_curve=curve, within=[[g, int(n), float(r)] for g, n, r in w]), open(sys.argv[1] + "/e_all.json", "w"), default=float)
