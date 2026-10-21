import json, sys, numpy as np
sys.path.insert(0, sys.argv[1]); from mcat import parse, cat, CATS, o, orig, tasks; from feats import mon_features
V = json.load(open(sys.argv[1] + "/c_moves.json"))["cost"]; val = {c: V[c]["value"][0] for c in CATS[:6]}; val["Other status"] = 0
def movesum(mons): return sum(val[cat(mv, m)] for m in mons for mv in m["moves"])
by_k = {}; rows = []
for f in o["fills"]:
    if f["task"] != "pokemon" or f.get("score") is None or f["start_team"] not in orig: continue
    a, b = parse(tasks[f["start_team"]]), parse(f["paste"])
    pred = movesum(b) - movesum(a); d = f["score"] - orig[f["start_team"]]
    by_k.setdefault(f["k"], []).append((d, pred))
    fa = [mon_features(m) for m in a]; fb = [mon_features(m) for m in b]
    gone = [x for x in fa if x["species"] not in {y["species"] for y in fb}]; new = [x for x in fb if x["species"] not in {y["species"] for y in fa}]
    rows.append(dict(k=f["k"], resid=d - pred, d=d, **{q: sum(x[q] for x in new) - sum(x[q] for x in gone) for q in ["bst", "spe", "off", "bulk", "mega", "ev_max", "neutral_nature", "split_attacker"]}))
res = {}
for k in sorted(by_k):
    d, p = np.array(by_k[k]).T; res[k] = dict(actual=d.mean(), move_pred=p.mean()); print(f"k={k} actual {d.mean():+.3f}  predicted from move mix {p.mean():+.3f}  residual {d.mean()-p.mean():+.3f}")
# residual (non-move part) vs species/stat features, per replaced Pokémon
import itertools
y = np.array([r["resid"] for r in rows]); names = ["k", "bst", "spe", "bulk", "off", "mega", "ev_max", "neutral_nature"]
X = np.array([[r[n] for n in names] for r in rows], float); A = np.c_[np.ones(len(X)), X]
beta = np.linalg.lstsq(A, y, rcond=None)[0]
rng = np.random.default_rng(2); B = [np.linalg.lstsq(A[i], y[i], rcond=None)[0] for i in (rng.integers(0, len(y), len(y)) for _ in range(1000))]
lo, hi = np.quantile(B, [.025, .975], axis=0)
for n, b, l, h, sd in zip(["const"] + names, beta, lo, hi, [1] + list(X.std(0))): print(f"  {n:15s} {b:+.5f} [{l:+.5f},{h:+.5f}]  per unit; sd {sd:.1f}")
json.dump(dict(by_k=res, coef={n: [b, l, h] for n, b, l, h in zip(["const"] + names, beta, lo, hi)}), open(sys.argv[1] + "/d_species.json", "w"), default=float)
