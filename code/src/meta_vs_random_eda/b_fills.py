import json, sys, numpy as np
sys.path.insert(0, sys.argv[1]); from feats import parse, mon_features
o = json.load(open('/Users/ramiismael/Documents/code/vgc-team-generator-pilot/results/random_fill_baseline.json'))
orig = {int(k): v for k, v in o['original'].items()}
tasks = {t["start_team"]: t["team_paste"] for t in json.load(open('/Users/ramiismael/Documents/code/vgc-team-generator-pilot/results/random_fill_tasks.json'))["tasks"]}
SUM = ["protect", "n_stab", "matched_stat", "n_dmg", "spread", "speed_ctrl", "support", "mega"]
AVG = ["bst", "spe", "off", "bulk"]
def tf(p):
    ms = [mon_features(m) for m in parse(p)]
    d = {k: sum(m[k] for m in ms) for k in SUM}; d.update({k: np.mean([m[k] for m in ms]) for k in AVG}); return d
base = {t: tf(p) for t, p in tasks.items()}
out = {}
for task, cols in [("move", SUM), ("pokemon", SUM + AVG), ("item", ["mega"])]:
    fs = [f for f in o["fills"] if f["task"] == task and f.get("score") is not None and f["start_team"] in orig]
    y = np.array([f["score"] - orig[f["start_team"]] for f in fs])
    X = np.array([[tf(f["paste"])[c] - base[f["start_team"]][c] for c in cols] + [f["k"]] for f in fs], float)
    names = [*cols, "k"]
    Z = (X - X.mean(0)) / (X.std(0) + 1e-9)
    A = np.c_[np.ones(len(Z)), Z]
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    # bootstrap CIs over start teams
    st = np.array([f["start_team"] for f in fs]); us = np.unique(st); rng = np.random.default_rng(0); B = []
    for _ in range(1000):
        pick = rng.choice(us, len(us)); idx = np.concatenate([np.where(st == u)[0] for u in pick])
        B.append(np.linalg.lstsq(A[idx], y[idx], rcond=None)[0])
    lo, hi = np.quantile(B, [.025, .975], axis=0)
    r2 = 1 - ((y - A @ beta) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    rk = np.corrcoef(X[:, -1], y)[0, 1] ** 2
    print(f"== {task}: n={len(y)} R2 all={r2:.3f}  R2 k only={rk:.3f}")
    for n, b, l, h in zip(["const", *names], beta, lo, hi): print(f"   {n:13s} {b:+.4f} [{l:+.4f},{h:+.4f}] per SD")
    print("   mean feature change per fill:", {n: round(float(X[:, i].mean()), 2) for i, n in enumerate(names)})
    out[task] = dict(n=len(y), r2=r2, r2_k=rk, coef={n: [b, l, h] for n, b, l, h in zip(["const", *names], beta, lo, hi)},
                     sd={n: float(X[:, i].std()) for i, n in enumerate(names)})
json.dump(out, open(sys.argv[1] + "/b_fills.json", "w"), default=float)
