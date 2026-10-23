"""Random-play battles as a random walk: analysis (2026-10-21). Runs: src/random_battle_graph.py.

Win rate below always means the fraction of battles a team wins against the top-50 meta teams
(weighted as in the matchup database). "Behaviour-cloning win rate" = both sides played by the
behaviour-cloning policy (policy 1, at least 196 battles per team). "Random-play win rate" = both
sides play at random (392 battles per team).

1. Does the random-play win rate track the behaviour-cloning win rate across the 150 teams, and
   among strong teams beyond what range restriction alone would do?
2. Is the lead (HP left, summed over the four brought, row minus opponent) a random walk?
   Serial correlation of its per-turn steps against a within-battle shuffle; variance ratio.
3. The Pokemon-left lattice: battle = walk on the 5x5 grid of (row left, opponent left); absorption
   probability from (4, 4) by dynamic programming; does the next knockout depend on who is ahead?
4. Features read off the walk and the lattice; do they predict the behaviour-cloning win rate, and
   beyond the random-play win rate?
5. The tree. A spine is one random battle from team preview to the end; a rollout is a random battle
   played to the end from a saved copy of a spine's state, and the share of rollouts won at a turn
   is the chance to win there. Is that chance a martingale (calibration of the
   final result on each node's estimate), and where do its squared steps get resolved?
6. The play-dependence gap (behaviour-cloning win rate minus what random play predicts) against moves.
7. One search family: does random play still rank teams that share all six species?
Output: results/random_battle_graph.json, results/random_battle_graph_figure.json (the page's data).
"""
import json, os
from collections import defaultdict
from pathlib import Path
import numpy as np

DATA = Path.home() / "vgc-data" / "random-battle-graph"
OUT = Path(__file__).resolve().parent.parent / "results" / "random_battle_graph.json"
rng = np.random.default_rng(20261021)


def ranks(x):
    x = np.asarray(x, float)
    order = x.argsort(kind="mergesort")
    r = np.empty(len(x)); r[order] = np.arange(len(x))
    for v in np.unique(x):                      # average ties
        m = x == v
        if m.sum() > 1: r[m] = r[m].mean()
    return r


def spearman(x, y):
    return float(np.corrcoef(ranks(x), ranks(y))[0, 1])


def boot(stat, *cols, n=2000):
    cols = [np.asarray(c) for c in cols]
    k = len(cols[0]); vals = []
    for _ in range(n):
        i = rng.integers(0, k, k)
        vals.append(stat(*[c[i] for c in cols]))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def rank_corr(x, y):
    """Spearman correlation with its bootstrap 95% interval."""
    return {"spearman": spearman(x, y), "ci": boot(spearman, x, y)}


def partial_spearman(x, y, z):
    """Spearman of x and y after removing z linearly from both, on ranks."""
    rx, ry, rz = ranks(x), ranks(y), ranks(z)
    A = np.c_[np.ones(len(rz)), rz]
    ex = rx - A @ np.linalg.lstsq(A, rx, rcond=None)[0]
    ey = ry - A @ np.linalg.lstsq(A, ry, rcond=None)[0]
    return float(np.corrcoef(ex, ey)[0, 1])


def cv_r2(X, y, folds=10, reps=20, lam=1e-3):
    """Ridge on standardised columns, repeated k-fold; out-of-fold R^2."""
    X = np.c_[X]; n = len(y); out = []
    for _ in range(reps):
        idx = rng.permutation(n); pred = np.empty(n)
        for f in range(folds):
            te = idx[f::folds]; tr = np.setdiff1d(idx, te)
            mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-12
            A = np.c_[np.ones(len(tr)), (X[tr] - mu) / sd]
            w = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ y[tr])
            pred[te] = np.c_[np.ones(len(te)), (X[te] - mu) / sd] @ w
        out.append(1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum())
    return float(np.mean(out))


def load_walk(meta):
    w = {int(k): v for k, v in meta["top50"].items()}
    per = defaultdict(list)
    for line in open(DATA / "walk.jsonl"):
        d = json.loads(line)
        if "error" in d: continue          # counted in run_walk.json, reported in figure_data()
        d["w"] = w[int(d["col"])]
        d["rep"] = int(d["seed"].rsplit(":", 1)[1])
        per[int(d["row"])].append(d)
    return per


def wmean(vals, wts):
    vals, wts = np.asarray(vals, float), np.asarray(wts, float)
    return float((vals * wts).sum() / wts.sum()) if wts.sum() else float("nan")


# ------------------------------------------------------------------ 1. random play vs behaviour cloning
def part1(rows, per):
    bc = np.array([r["bc_win_rate"] for r in rows])
    rp = np.array([wmean([b["win"] for b in per[r["team_id"]]], [b["w"] for b in per[r["team_id"]]]) for r in rows])
    halves = []
    for r in rows:
        bs = per[r["team_id"]]
        a = [b for b in bs if b["rep"] % 2 == 0]; c = [b for b in bs if b["rep"] % 2 == 1]
        halves.append((wmean([b["win"] for b in a], [b["w"] for b in a]), wmean([b["win"] for b in c], [b["w"] for b in c])))
    h = np.array(halves)
    r_half = float(np.corrcoef(h[:, 0], h[:, 1])[0, 1])
    rel_rp = 2 * r_half / (1 + r_half)                      # Spearman-Brown, full 392 battles
    nb = np.array([r["bc_battles"] for r in rows])
    rel_bc = 1 - np.mean(bc * (1 - bc) / nb) / bc.var()      # binomial noise share of the spread
    # Keeping only teams above a behaviour-cloning cut lowers any correlation on its own (range
    # restriction). The null keeps the all-teams relationship (cubic fit plus resampled residuals),
    # applies the same cut, and asks how often that alone gets as low as the observed correlation.
    A = np.c_[np.ones(len(rp)), rp, rp ** 2, rp ** 3]
    fit = A @ np.linalg.lstsq(A, bc, rcond=None)[0]
    resid = bc - fit
    strong = {}
    for lo in (0.30, 0.40, 0.50):
        m = bc >= lo
        rh = float(np.corrcoef(h[m, 0], h[m, 1])[0, 1]); rr = 2 * rh / (1 + rh)
        rb = 1 - np.mean(bc[m] * (1 - bc[m]) / nb[m]) / bc[m].var()
        obs = spearman(rp[m], bc[m])
        sims = []
        for _ in range(4000):
            y = fit + rng.choice(resid, len(resid), replace=True)
            k = y >= lo
            if k.sum() > 8: sims.append(spearman(rp[k], y[k]))
        sims = np.array(sims)
        strong[f"bc_at_least_{lo:.2f}"] = {"teams": int(m.sum()), "spearman": obs,
                                            "ci": boot(spearman, rp[m], bc[m]),
                                            "noise_ceiling": float(np.sqrt(max(rr, 0) * max(rb, 0))),
                                            "range_restriction_only": {"median": float(np.median(sims)),
                                                                       "band": [float(np.percentile(sims, 2.5)), float(np.percentile(sims, 97.5))],
                                                                       "share_at_or_below_observed": float(np.mean(sims <= obs))}}
    top = {}
    for k in (10, 20):
        a, b = set(np.argsort(-bc)[:k]), set(np.argsort(-rp)[:k])
        top[str(k)] = {"overlap": len(a & b),
                       "bc_win_rate_of_random_play_top_k": float(bc[list(b)].mean()),
                       "bc_win_rate_of_true_top_k": float(bc[list(a)].mean())}
    return rp, {
        "strong_teams_only": strong, "top_k": top,
        "teams": len(rows),
        "random_play_win_rate": {"mean": float(rp.mean()), "sd": float(rp.std()), "min": float(rp.min()), "max": float(rp.max())},
        "behaviour_cloning_win_rate": {"mean": float(bc.mean()), "sd": float(bc.std())},
        "spearman": spearman(rp, bc), "spearman_ci": boot(spearman, rp, bc),
        "pearson": float(np.corrcoef(rp, bc)[0, 1]),
        "reliability_random_play": rel_rp, "reliability_behaviour_cloning": float(rel_bc),
        "correlation_ceiling": float(np.sqrt(rel_rp * rel_bc)),
    }


# ------------------------------------------------------------------ 2. is the lead a Brownian motion?
def part2(per):
    obs, shuf, steps_all = [], [], []
    vr = {k: {"obs": [], "shuffled": []} for k in (2, 4)}
    def ratio(s, k):
        sums = np.array([s[i:i + k].sum() for i in range(len(s) - k + 1)])
        return sums.var(), s.var() * k
    for bs in per.values():
        for b in bs:
            lead = np.array([t[2] - t[4] for t in b["traj"]])
            s = np.diff(lead)
            if len(s) < 4: continue
            steps_all.append(s)
            e = s - s.mean()
            obs.append((e[:-1] * e[1:]).sum() / (e ** 2).sum() if (e ** 2).sum() else 0)
            p = rng.permutation(e)
            shuf.append((p[:-1] * p[1:]).sum() / (p ** 2).sum() if (p ** 2).sum() else 0)
            for k in vr:
                if len(s) >= 2 * k:
                    vr[k]["obs"].append(ratio(s, k)); vr[k]["shuffled"].append(ratio(rng.permutation(s), k))
    allsteps = np.concatenate(steps_all)
    vr_out = {str(k): {w: float(np.mean([a for a, _ in v[w]]) / np.mean([b for _, b in v[w]])) for w in v}
              for k, v in vr.items()}
    return {
        "battles_used": len(obs),
        "lead_step_lag1_autocorrelation": float(np.mean(obs)),
        "same_after_within_battle_shuffle": float(np.mean(shuf)),
        "difference": float(np.mean(obs) - np.mean(shuf)),
        "difference_ci": boot(lambda a, b: a.mean() - b.mean(), np.array(obs), np.array(shuf)),
        "variance_ratio": vr_out,
        "share_of_turns_lead_unchanged": float(np.mean(np.abs(allsteps) < 1e-9)),
        "step_kurtosis": float(((allsteps - allsteps.mean()) ** 4).mean() / allsteps.var() ** 2),
    }


# ------------------------------------------------------------------ 3. the Pokemon-left lattice
def lattice_of(bs):
    """Jump chain on (row left, opponent left): counts of the next distinct state."""
    cnt = defaultdict(lambda: defaultdict(float))
    for b in bs:
        states = [(t[1], t[3]) for t in b["traj"]]
        for a, c in zip(states, states[1:]):
            if c != a: cnt[a][c] += b["w"]
    return cnt


def absorb(cnt, start=(4, 4)):
    memo = {}
    def v(s):
        i, j = s
        if j == 0 and i > 0: return 1.0
        if i == 0: return 0.0 if j > 0 else 0.5
        if s in memo: return memo[s]
        out = cnt.get(s)
        if not out:                               # unseen state: fall back to the Pokemon-left share
            memo[s] = i / (i + j); return memo[s]
        tot = sum(out.values())
        memo[s] = sum(n / tot * v(t) for t, n in out.items())
        return memo[s]
    return v(start)


def ko_sequence(b):
    """Turns with a knockout, in order: (row Pokemon lost, opponent Pokemon lost) that turn.
    A double knockout stays one event, since its order inside the turn is not recorded."""
    seq = []
    states = [(t[1], t[3]) for t in b["traj"]]
    for a, c in zip(states, states[1:]):
        if c != a: seq.append((a[0] - c[0], a[1] - c[1]))
    return seq


def lead_table(seq):
    """(lead in Pokemon before the turn, 1 if the row scored) for every single-knockout turn."""
    out, lead = [], 0
    for lost_row, lost_opp in seq:
        if lost_row + lost_opp == 1: out.append((max(-2, min(2, lead)), lost_opp))
        lead += lost_opp - lost_row
    return out


def repeats(seq):
    """For consecutive single-knockout turns: 1 if the same side scored both."""
    s = [e[1] for e in seq if e[0] + e[1] == 1]
    return [float(a == b) for a, b in zip(s, s[1:])]


def part3(rows, per, rp):
    # Snowball test: is the next knockout more likely for the side that is ahead? Compare each
    # battle with itself after shuffling its knockout order (the last knockout stays last, so the
    # shuffled battle is still a legal finish). The shuffle keeps every battle's own balance, so a
    # lopsided battle (strong leads, a bad preview) cannot fake a snowball.
    obs, null = defaultdict(list), defaultdict(list)
    per_battle, rep_obs, rep_null = [], [], []
    for bs in per.values():
        for b in bs:
            seq = ko_sequence(b)
            if len(seq) < 4: continue
            body = list(seq[:-1]); rng.shuffle(body)
            shuffled = body + seq[-1:]
            o, s = lead_table(seq), lead_table(shuffled)
            for lead, x in o: obs[lead].append(x)
            for lead, x in s: null[lead].append(x)
            per_battle.append((o, s))
            rep_obs += repeats(seq); rep_null += repeats(shuffled)
    def shuffle_gap(idx, lead):
        a = [x for i in idx for l, x in per_battle[i][0] if l == lead]
        c = [x for i in idx for l, x in per_battle[i][1] if l == lead]
        return np.mean(a) - np.mean(c)
    momentum = {}
    for lead in sorted(obs):
        cis = [shuffle_gap(rng.integers(0, len(per_battle), len(per_battle)), lead) for _ in range(200)]
        momentum[str(lead)] = {"knockouts": len(obs[lead]), "row_scores_next": float(np.mean(obs[lead])),
                               "same_after_shuffle": float(np.mean(null[lead])),
                               "gap_ci": [float(np.percentile(cis, 2.5)), float(np.percentile(cis, 97.5))]}
    # sanity: absorption probability from the team's own lattice reproduces its random-play win rate
    ab = np.array([absorb(lattice_of(per[r["team_id"]])) for r in rows])
    # the pooled lattice, oriented row vs opponent, for the figure
    pooled = lattice_of([b for bs in per.values() for b in bs])
    edges = {f"{a[0]}{a[1]}": {"n": round(sum(out.values())), "win": round(absorb(pooled, a), 4),
                               "next": {f"{c[0]}{c[1]}": round(n / sum(out.values()), 4) for c, n in out.items()}}
             for a, out in pooled.items()}
    return ab, {"snowball_by_pokemon_lead": momentum,
                "same_side_scores_twice_in_a_row": {"observed": float(np.mean(rep_obs)), "after_shuffle": float(np.mean(rep_null)),
                                                    "pairs": len(rep_obs)},
                "absorption_vs_random_play_win_rate": {"pearson": float(np.corrcoef(ab, rp)[0, 1]),
                                                       "mean_abs_gap": float(np.abs(ab - rp).mean())},
                "pooled_lattice_edges": edges}


# ------------------------------------------------------------------ 4. features vs behaviour cloning
def features(bs):
    first, conv, comeb, length, drift, vol = [], [], [], [], [], []
    for b in bs:
        f = b["faints"]
        if f:
            t0 = f[0][0]; firsts = {s for t, s, _ in f if t == t0}
            if len(firsts) == 1:
                scored = 1 in firsts
                first.append((float(scored), b["w"]))
                (conv if scored else comeb).append((b["win"], b["w"]))
        length.append((b["turns"], b["w"]))
        lead = np.array([t[2] - t[4] for t in b["traj"]])
        s = np.diff(lead)
        drift.append((s.mean(), b["w"]))
        if len(s) > 1: vol.append((s.std(), b["w"]))
    m = lambda xs: wmean([x for x, _ in xs], [w for _, w in xs])
    return {"first_knockout": m(first), "conversion": m(conv), "comeback": m(comeb),
            "length": m(length), "lead_drift": m(drift), "lead_volatility": m(vol)}


def part4(rows, per, rp):
    bc = np.array([r["bc_win_rate"] for r in rows])
    F = [features(per[r["team_id"]]) for r in rows]
    names = list(F[0])
    out = {"random_play_win_rate": rank_corr(rp, bc)}
    for k in names:
        x = np.array([f[k] for f in F])
        out[k] = {**rank_corr(x, bc),
                  "partial_given_random_play": partial_spearman(x, bc, rp),
                  "partial_ci": boot(partial_spearman, x, bc, rp),
                  "spearman_with_random_play": spearman(x, rp)}
    X_all = np.c_[rp, *[np.array([f[k] for f in F]) for k in names]]
    out["cv_r2"] = {"random_play_only": cv_r2(rp[:, None], bc), "random_play_plus_features": cv_r2(X_all, bc)}
    return F, out


# ------------------------------------------------------------------ 5. the tree: martingale resolution
def part5(rows, meta, rp_by_team):
    path = Path(os.environ.get("TREE_FILE", DATA / "tree.jsonl"))
    if not path.exists(): return None, {}
    K = meta["tree_k"]; v_hat = lambda p: p * (1 - p) / (K - 1)
    per_team = defaultdict(list); identity = []; first_step = []; cum_by_turn = defaultdict(list)
    calib = []                                   # (spine, chance to win at a node, final result)
    for line in open(path):
        d = json.loads(line)
        if "error" in d: continue
        V = [d["root"][0]] + [n[5] for n in d["nodes"]] + [d["win"]]
        noise = [v_hat(V[0])] + [v_hat(n[5]) for n in d["nodes"]] + [0.0]
        inc = [(V[i + 1] - V[i]) ** 2 - noise[i] - noise[i + 1] for i in range(len(V) - 1)]
        tot = sum(inc)
        identity.append((tot, V[0] * (1 - V[0]) * K / (K - 1)))   # unbiased for the root's variance
        calib += [(len(identity), v, d["win"]) for v in V[:-1]]
        if tot <= 0: continue
        share = np.cumsum(inc) / tot
        first_step.append(inc[0] / tot)                       # root -> turn 1: team preview and leads
        half = next(i for i, s in enumerate(share) if s >= 0.5)   # 0 = decided at preview
        per_team[int(d["row"])].append((half, half / max(1, len(inc) - 1), inc[0] / tot))
        for i in range(16): cum_by_turn[i].append(share[i] if i < len(share) else 1.0)
    bc = {r["team_id"]: r["bc_win_rate"] for r in rows}
    ids = [t for t in per_team if t in bc]
    half_turn = np.array([np.mean([h for h, _, _ in per_team[t]]) for t in ids])
    half_frac = np.array([np.mean([f for _, f, _ in per_team[t]]) for t in ids])
    prev = np.array([np.mean([p for _, _, p in per_team[t]]) for t in ids])
    y = np.array([bc[t] for t in ids])
    z = np.array([rp_by_team[t] for t in ids])
    idn = np.array(identity)
    cum = {i: float(np.mean(v)) for i, v in sorted(cum_by_turn.items())}
    tree_feats = {t: {"half_decided_step": h, "share_at_preview": p} for t, h, p in zip(ids, half_turn, prev)}
    feat = lambda x: {**rank_corr(x, y),
                      "partial_given_random_play": partial_spearman(x, y, z), "partial_ci": boot(partial_spearman, x, y, z)}
    # Martingale test: a battle that reaches a node with chance to win v goes on to win with
    # probability v, so the slope of the final result on the node's estimate is 1, attenuated
    # only by rollout noise to 1 - mean(noise variance) / var(estimate).
    C = np.array(calib)
    def slope(c):
        v, w = c[:, 1], c[:, 2]
        return float(np.cov(v, w)[0, 1] / v.var(ddof=1))
    spines = np.unique(C[:, 0]); by = {s: C[C[:, 0] == s] for s in spines}
    cis = [slope(np.concatenate([by[s] for s in rng.choice(spines, len(spines))])) for _ in range(300)]
    expected = 1 - np.mean(C[:, 1] * (1 - C[:, 1]) / (K - 1)) / C[:, 1].var(ddof=1)
    return {
        "spines": len(identity),
        "calibration": {"nodes": len(C), "slope": slope(C), "slope_ci": [float(np.percentile(cis, 2.5)), float(np.percentile(cis, 97.5))],
                        "slope_expected_from_rollout_noise": float(expected),
                        "mean_result_minus_mean_chance": float(C[:, 2].mean() - C[:, 1].mean())},
        "sum_of_squared_steps_vs_root_variance": {"mean_sum": float(idn[:, 0].mean()), "mean_v0_1_minus_v0": float(idn[:, 1].mean())},
        "share_resolved_at_preview": float(np.mean(first_step)),
        "cumulative_share_by_step": {str(i): v for i, v in cum.items()},
        "half_settled_by_end_of_turn": next(i for i, v in cum.items() if v >= 0.5),
        "half_decided_step_mean": float(half_turn.mean()),
        "features_vs_behaviour_cloning": {"half_decided_step": feat(half_turn), "half_decided_fraction_of_battle": feat(half_frac),
                                          "share_at_preview": feat(prev)},
    }, tree_feats


def pastes(rows):
    import sqlite3
    db = sqlite3.connect(f"file:{Path.home() / 'vgc-data' / 'matchup_regmb.sqlite'}?mode=ro", uri=True)
    ids = ",".join(str(r["team_id"]) for r in rows)
    return dict(db.execute(f"select team_id, paste from team where team_id in ({ids})"))


def species(p):
    return [b.strip().splitlines()[0].split(" @ ")[0].strip() for b in p.split("\n\n") if b.strip()]


def moves(p):
    return [l[2:].strip() for l in p.splitlines() if l.startswith("- ")]


SETUP = {"Swords Dance", "Dragon Dance", "Calm Mind", "Nasty Plot", "Bulk Up", "Quiver Dance", "Shell Smash",
         "Belly Drum", "Coil", "Shift Gear", "Victory Dance", "Tidy Up", "Iron Defense", "Agility", "Growth", "Curse"}
PROTECT = {"Protect", "Detect", "Spiky Shield", "King's Shield", "Baneful Bunker", "Silk Trap", "Burning Bulwark"}


def part6(rows, rp, P):
    """6. The play-dependence gap: behaviour-cloning win rate minus what random play predicts (quadratic fit).
    7. One search family: does random play still rank teams that share all six species?"""
    bc = np.array([r["bc_win_rate"] for r in rows])
    A = np.c_[np.ones(len(rp)), rp, rp ** 2]
    play_gap = bc - A @ np.linalg.lstsq(A, bc, rcond=None)[0]
    tests = {"Trick Room": lambda m: float("Trick Room" in m), "Tailwind": lambda m: float("Tailwind" in m),
             "Protect count": lambda m: float(sum(x in PROTECT for x in m)), "Fake Out": lambda m: float("Fake Out" in m),
             "setup moves": lambda m: float(sum(x in SETUP for x in m)),
             "Follow Me or Rage Powder": lambda m: float(("Follow Me" in m) or ("Rage Powder" in m))}
    out = {"gap_sd": float(play_gap.std())}
    for name, f in tests.items():
        x = np.array([f(moves(P[r["team_id"]])) for r in rows])
        out[name] = {"teams_with": int((x > 0).sum()), "spearman_with_gap": spearman(x, play_gap), "ci": boot(spearman, x, play_gap)}
    sp = [species(P[r["team_id"]]) for r in rows]
    core = {"Basculegion", "Chandelure", "Floette-Eternal", "Garchomp", "Kingambit"}
    fam = np.array([core <= set(s) for s in sp]); six = np.array([set(s) == core | {"Whimsicott"} for s in sp])
    strong_out = (bc >= 0.5) & ~fam
    fam_out = {"core_family": {"teams": int(fam.sum()), **rank_corr(rp[fam], bc[fam])},
               "same_six_species": {"teams": int(six.sum()), **rank_corr(rp[six], bc[six])},
               "strong_outside_family": {"teams": int(strong_out.sum()), "spearman": spearman(rp[strong_out], bc[strong_out])},
               "distinct_line_ups": len({tuple(sorted(s)) for s in sp}),
               "origins": dict(sorted({o: sum(r["origin"].split(":")[0] == o for r in rows)
                                       for o in {r["origin"].split(":")[0] for r in rows}}.items(), key=lambda kv: -kv[1]))}
    return out, fam_out


from random_battle_graph import FIG_ROW, FIG_COL
BC_PER_S = (88, 94)                  # fast behaviour-cloning evaluator, 6 workers: ~/vgc-data/battle-evaluations runs of 2026-10-21


def run_logs(meta):
    """Errors, speed and default-choice fallbacks per run, from the run_<mode>.json files."""
    logs = {m: json.loads((DATA / f"run_{m}.json").read_text()) for m in ("walk", "tree", "branch")
            if (DATA / f"run_{m}.json").exists()}
    tree_battles = 0
    for line in open(Path(os.environ.get("TREE_FILE", DATA / "tree.jsonl"))):
        d = json.loads(line)
        if "error" not in d: tree_battles += 1 + (len(d["nodes"]) + 1) * meta["tree_k"]
    battles = {"walk": logs["walk"]["records"], "tree": tree_battles}
    return {"errors": sum(l["errors"] for l in logs.values()),
            "random_per_s": round(logs["walk"]["records"] / logs["walk"]["seconds"]),
            "fallback_per_battle": {m: logs[m]["fallback"] / battles[m] for m in battles},
            "branch_decisions": logs.get("branch", {}).get("decisions"), "branch_fallback": logs.get("branch", {}).get("fallback")}


def figure_data(res, rows, per, meta):
    """Everything the page Blog/visuals/random-battle-tree shows, in one file (sprites are added there)."""
    P = pastes(rows + [{"team_id": FIG_ROW}, {"team_id": FIG_COL}])
    runs = run_logs(meta)
    r1, r2, r3, r4, r5 = (res[k] for k in ("1_random_vs_behaviour_cloning", "2_brownian_check", "3_lattice", "4_features", "5_tree"))
    st = r1["strong_teams_only"]
    feats = {k: {"spearman": v["spearman"], "ci": v["ci"], "partial": v.get("partial_given_random_play"), "partial_ci": v.get("partial_ci")}
             for k, v in r4.items() if k != "cv_r2"}
    for k in ("half_decided_step", "share_at_preview"):
        v = r5["features_vs_behaviour_cloning"][k]
        feats[k] = {"spearman": v["spearman"], "ci": v["ci"], "partial": v["partial_given_random_play"], "partial_ci": v["partial_ci"]}
    gap_res = res["6_play_dependence_gap"]; fam = res["7_one_family"]
    walk = sum(len(v) for v in per.values())
    turns = sorted(b["turns"] for v in per.values() for b in v)
    n_search = sum(v for k, v in fam["origins"].items() if k != "vgcpastes")
    stats = {
        "battles_label": f"{walk:,} random battles, {r5['spines']:,} trees",
        "walk_battles": walk, "median_turns": turns[len(turns) // 2],
        "row_bc": next(r["bc_win_rate"] for r in rows if r["team_id"] == FIG_ROW),
        "all": {"spearman": r1["spearman"], "ci": r1["spearman_ci"]},
        "cut": {k[-4:]: v for k, v in st.items()}, "ceiling_050": st["bc_at_least_0.50"]["noise_ceiling"],
        "family26": {"n": fam["same_six_species"]["teams"], "spearman": fam["same_six_species"]["spearman"], "ci": fam["same_six_species"]["ci"]},
        "top10": {"overlap": r1["top_k"]["10"]["overlap"], "picked": r1["top_k"]["10"]["bc_win_rate_of_random_play_top_k"],
                  "best": r1["top_k"]["10"]["bc_win_rate_of_true_top_k"]},
        "brownian": {"lag1": r2["lead_step_lag1_autocorrelation"], "lag1_shuffled": r2["same_after_within_battle_shuffle"],
                     "kurtosis": r2["step_kurtosis"], "flat": r2["share_of_turns_lead_unchanged"],
                     "variance_ratio_4": r2["variance_ratio"]["4"]},
        "walk_per_team": meta["walk_n"] * len(meta["top50"]),
        "runs": runs,
        "snowball": r3["snowball_by_pokemon_lead"],
        "lattice_gap": r3["absorption_vs_random_play_win_rate"]["mean_abs_gap"],
        "tree": {"spines": r5["spines"], "k": meta["tree_k"], "calibration": r5["calibration"],
                 "sum_sq": r5["sum_of_squared_steps_vs_root_variance"]["mean_sum"],
                 "v0var": r5["sum_of_squared_steps_vs_root_variance"]["mean_v0_1_minus_v0"],
                 "preview_share": r5["share_resolved_at_preview"], "half_turn": r5["half_settled_by_end_of_turn"]},
        "features": feats,
        "cv": {"random_play_only": r4["cv_r2"]["random_play_only"],
               "random_play_plus_features": r4["cv_r2"].get("random_play_plus_walk_and_tree_features", r4["cv_r2"]["random_play_plus_features"])},
        "gap": {"tailwind": [gap_res["Tailwind"]["spearman_with_gap"], *gap_res["Tailwind"]["ci"]],
                "protect": [gap_res["Protect count"]["spearman_with_gap"], *gap_res["Protect count"]["ci"]]},
        "speed": {"random": runs["random_per_s"], "bc": f"{BC_PER_S[0]}–{BC_PER_S[1]}",
                  "ratio": round(runs["random_per_s"] / (sum(BC_PER_S) / 2), 1)},
        "origin_note": f"{n_search} of the 150 come from earlier search runs and {fam['origins'].get('vgcpastes', 0)} from VGCPastes; "
                       f"{fam['core_family']['teams']} share one five-Pokémon core, and there are {fam['distinct_line_ups']} distinct line-ups.",
    }
    teams = [{"id": r["team_id"], "species": species(P[r["team_id"]]), "bc": round(r["bc_win_rate"], 4),
              "rp": round(x["random_play_win_rate"], 4),
              "family": {"Basculegion", "Chandelure", "Floette-Eternal", "Garchomp", "Kingambit"} <= set(species(P[r["team_id"]]))}
             for r, x in zip(rows, res["per_team"])]
    tree = json.loads(open(DATA / f"branch_{FIG_ROW}_{FIG_COL}.json").readline())
    fig = {"tree": tree, "row_team": species(P[FIG_ROW]), "opp_team": species(P[FIG_COL]), "teams": teams,
           "lattice": {k: v for k, v in r3["pooled_lattice_edges"].items() if "0" not in k}, "stats": stats}
    (OUT.parent / "random_battle_graph_figure.json").write_text(json.dumps(fig, separators=(",", ":")))


def main():
    meta = json.loads((DATA / "teams.json").read_text())
    rows = meta["rows"]
    per = load_walk(meta)
    rp, r1 = part1(rows, per)
    r2 = part2(per)
    ab, r3 = part3(rows, per, rp)
    F, r4 = part4(rows, per, rp)
    r5, tree_feats = part5(rows, meta, {r["team_id"]: x for r, x in zip(rows, rp)})
    if tree_feats:   # cross-validation with the tree features too, on the teams that have trees
        m = np.array([r["team_id"] in tree_feats for r in rows])
        sub = [r for r in rows if r["team_id"] in tree_feats]
        bc = np.array([r["bc_win_rate"] for r in sub])
        X = np.c_[rp[m], *[np.array([f[k] for f in F])[m] for k in F[0]],
                  *[np.array([tree_feats[r["team_id"]][k] for r in sub]) for k in ("half_decided_step", "share_at_preview")]]
        r4["cv_r2"]["teams_with_trees"] = int(m.sum())
        r4["cv_r2"]["random_play_only_same_teams"] = cv_r2(rp[m][:, None], bc)
        r4["cv_r2"]["random_play_plus_walk_and_tree_features"] = cv_r2(X, bc)
    r6, r7 = part6(rows, rp, pastes(rows))
    res = {"1_random_vs_behaviour_cloning": r1, "2_brownian_check": r2, "3_lattice": r3,
           "4_features": r4, "5_tree": r5, "6_play_dependence_gap": r6, "7_one_family": r7,
           "per_team": [{**r, "random_play_win_rate": float(x), "lattice_absorption": float(a), **f}
                        for r, x, a, f in zip(rows, rp, ab, F)]}
    OUT.write_text(json.dumps(res, indent=1))
    figure_data(res, rows, per, meta)
    show = {k: v for k, v in res.items() if k != "per_team"}
    show["3_lattice"] = {k: v for k, v in r3.items() if k != "pooled_lattice_edges"}
    print(json.dumps(show, indent=1))


if __name__ == "__main__":
    main()
