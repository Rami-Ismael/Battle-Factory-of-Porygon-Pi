"""Does TWISTER's correction change the battle results? Reweights each grid team by its correction weight (tilt_weights.py)
and compares, per ask 13/24-17/24, the uniform and the reweighted mean win rate, best-of-k (k teams drawn with
replacement from the ask's 192, expected highest win rate) and pass@k at a 65% bar, with paired bootstrap intervals:
each replicate redraws the 49 opponents once (shared by every row and ask) and the 192 teams of each ask with their
weights. Also: effective sample size of the weights, Spearman correlation of log weight with win rate, and the mean
removed log mass by field type. Usage: tilt_analysis.py <tilt dir> <battles.npz> <opp_weights.npy> <out.json>"""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

tilt_dir, d, W = Path(sys.argv[1]), np.load(sys.argv[2]), np.load(sys.argv[3])
R = Path.home() / "vgc-data/results"
KS, ASKS, B, CUT = [1, 4, 16, 64], range(13, 18), 2000, 0.65
rng = np.random.default_rng(20261115)
KINDS = ["species", "ability", "item", "move", "Stat Alignment", "Stat Points"]

def npz_order(row, g):
    """team ids per ask in the row order battles.py wrote them (cells in file order)."""
    suf = "" if g == 2 else f"_g{g}"
    by = {}
    for c in json.load(open(R / f"stat_points_84_{row}{suf}.json"))["cells"].values():
        if "y" in c: by.setdefault(c["numerator"], []).extend(c["team_ids"])
    return by

def best_k(y, w, k):
    o = np.argsort(y); y, w = y[o], w[o] / w.sum(); F = np.cumsum(w); F0 = np.concatenate([[0.0], F[:-1]])
    return float((y * (np.clip(F, 0, 1) ** k - np.clip(F0, 0, 1) ** k)).sum())

def stats(y, w):
    p = float(w[y >= CUT - 1e-9].sum() / w.sum())
    return [float((w * y).sum() / w.sum())] + [best_k(y, w, k) for k in KS] + [1 - (1 - p) ** k for k in KS]
NAMES = ["mean"] + [f"best@{k}" for k in KS] + [f"pass@{k}@0.65" for k in KS]

def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b)); return float(np.corrcoef(ra, rb)[0, 1])

out = {}
for f in sorted(tilt_dir.glob("*_g*.json")):
    t = json.load(open(f)); row, g = t["model"], int(t["guidance"])
    order = npz_order(row, g)
    per_ask, mismatched, logw_all, y_all, kind_sum, fallback, n_all = {}, 0, [], [], np.zeros(len(KINDS)), 0, 0
    look_all = []
    for cell in t["cells"].values():
        if not cell["same_pastes"]: mismatched += 1; continue
        a = cell["numerator"]; V = d[f"{row}|g{g}|a{a}"]; idx = {tid: i for i, tid in enumerate(order[a])}
        for tid, r in zip(cell["team_ids"], cell["records"]):
            if r["fallback"] > 0: continue                      # weight is not the sampler ratio there (diffusion84.Refusals)
            per_ask.setdefault(a, []).append((V[idx[tid]], r["log_w"]))
            kind_sum += np.array([r["log_w_by_kind"][k] for k in KINDS]); fallback += r["fallback"] > 0; n_all += 1
            look_all.append(r["log_w_lookahead"])
    if mismatched or not per_ask: out[f"{row}|g{g}"] = dict(error=f"{mismatched} cells did not reproduce"); print(row, g, "MISMATCH", mismatched); continue
    asks = sorted(per_ask)
    Vs = {a: np.stack([v for v, _ in per_ask[a]]) for a in asks}
    LW = {a: np.array([lw for _, lw in per_ask[a]]) for a in asks}
    def point(o, ts, weighted):
        res = []
        for a in asks:
            V, lw = Vs[a][ts[a]], LW[a][ts[a]]
            y = (V[:, o] * W[o]).sum(1) / W[o].sum()
            w = np.exp(lw - lw.max()) if weighted else np.ones(len(lw))
            res.append(stats(y, w))
        return np.mean(res, 0)
    ident = {a: np.arange(len(LW[a])) for a in asks}
    reps = [(np.arange(49), ident)] + [(rng.integers(0, 49, 49), {a: rng.integers(0, len(LW[a]), len(LW[a])) for a in asks}) for _ in range(B)]
    U = np.array([point(o, ts, False) for o, ts in reps]); Wt = np.array([point(o, ts, True) for o, ts in reps])
    diff = Wt - U; lo, hi = np.percentile(diff[1:], [2.5, 97.5], 0)
    ess = {a: float(np.exp(LW[a] - LW[a].max()).sum() ** 2 / (np.exp(LW[a] - LW[a].max()) ** 2).sum() / len(LW[a])) for a in asks}
    ys = np.concatenate([(Vs[a] * W).sum(1) / W.sum() for a in asks]); lws = np.concatenate([LW[a] for a in asks])
    within = float(np.mean([spearman(LW[a], (Vs[a] * W).sum(1) / W.sum()) for a in asks]))
    # win rate by log-weight decile, deciles taken within each ask so asks mix evenly
    dec = np.concatenate([np.argsort(np.argsort(LW[a])) * 10 // len(LW[a]) for a in asks])
    bins = [dict(decile=int(b), mean_log_w=float(lws[dec == b].mean()), win=float(ys[dec == b].mean()), n=int((dec == b).sum())) for b in range(10)]
    # tempered correction: weight^beta, beta 0 = as drawn, 1 = full TWISTER target; bootstrap like above
    BETAS = [0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0]
    def tempered(beta, o, ts):
        res, ess = [], []
        for a in asks:
            V, lw = Vs[a][ts[a]], LW[a][ts[a]]
            y = (V[:, o] * W[o]).sum(1) / W[o].sum(); w = np.exp(beta * (lw - lw.max()))
            res.append(stats(y, w)); ess.append(w.sum() ** 2 / (w ** 2).sum() / len(w))
        return np.mean(res, 0), float(np.mean(ess))
    curve = []
    for beta in BETAS:
        pts = [tempered(beta, o, ts) for o, ts in reps[:501]]
        vals = np.array([p[0] for p in pts]); lo_b, hi_b = np.percentile(vals[1:], [2.5, 97.5], 0)
        curve.append(dict(beta=beta, ess_share=pts[0][1], **{n: dict(point=float(vals[0, i]), lo=float(lo_b[i]), hi=float(hi_b[i])) for i, n in enumerate(NAMES)}))
    out[f"{row}|g{g}"] = dict(row=row, spearman_within_ask=within, deciles=bins, tempered=curve, guidance=g, teams=n_all, teams_with_fallback=int(fallback), teams_used=len(lws),
        median_refused_share=float(np.median(-np.expm1(lws))), median_log_w=float(np.median(lws)),
        q10_log_w=float(np.quantile(lws, 0.1)), q90_log_w=float(np.quantile(lws, 0.9)),
        median_log_w_lookahead_only=float(np.median(look_all)),
        ess_share_by_ask=ess, spearman_logw_winrate=spearman(lws, ys),
        mean_log_w_by_kind=dict(zip(KINDS, (kind_sum / n_all).tolist())),
        log_w=lws.tolist(), winrate=ys.tolist(),
        **{n: dict(uniform=float(U[0, i]), reweighted=float(Wt[0, i]), gap=float(diff[0, i]), lo=float(lo[i]), hi=float(hi[i])) for i, n in enumerate(NAMES)})
    r = out[f"{row}|g{g}"]
    print(f"{row} g{g}: n={n_all} fallback={fallback} within-ask rho {within:+.3f} " + " ".join(f"b{c['beta']}:{c['mean']['point']:.3f}(ess {c['ess_share']:.2f})" for c in curve) + f"  median refused {r['median_refused_share']:.3f} ESS {np.mean(list(ess.values())):.2f} rho {r['spearman_logw_winrate']:+.3f} "
          + " ".join(f"{n} {r[n]['gap']:+.4f}[{r[n]['lo']:+.4f},{r[n]['hi']:+.4f}]" for n in ("mean", "best@1", "best@64", "pass@64@0.65")), flush=True)
json.dump(out, open(sys.argv[4], "w"))
