"""Cross-entropy loop with Diffusion-BBO's rule on the version-3 foundation (2026-10-16).
Pre-registration: docs/cem-dbbo-loop.md. QUEUED, not run: see docs/run-queue.md.

Diffusion-BBO (Wu et al., 2024) trains the score-conditioned model on EVERY scored design and asks it
for a score of (best so far) x w. This loop keeps that rule and our plumbing (512 proposals, the ridge
picks 128 to battle). Two groups, chosen by CEM_GROUP:

    dbbo      the ORIGINAL v3 members (F1 + G2) fine-tuned each generation on every scored team
    scratch   a fresh network (F1 v3 architecture, random weights) trained each generation on the
              regulation-vocabulary foundation data plus every scored team

Target = min(best scored win rate so far x w, 1.0). w is picked each generation from W_SET by the ridge's
lower confidence bound on a probe batch (CEM_W=1.2 fixes it, Diffusion-BBO's fixed-multiplier ablation).

    CEM_SMOKE=1 CEM_GROUP=dbbo python cem_dbbo.py run                       # plumbing check
    CEM_GROUP=dbbo    python battle_supervisor.py cem_dbbo.py run --then final
    CEM_GROUP=scratch python battle_supervisor.py cem_dbbo.py run --then final
    CEM_GROUP=dbbo    python cem_dbbo.py report
(run with /tmp/vgc-pilot/.venv/bin/python)
"""
import json, math, os, sys, time
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, "/tmp/vgc-pilot/src")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import foundation as FD
import lossdown as L
import lossdown_battle as LB
import asked_vs_got as X
import activesearch as A
import matchup_db as MDB
import pool
import cem_v3 as C                                   # ridge, anchors, labelled pool, legacy vocabulary
from hps_generate import Validator
from corpus import parse_team_text, norm
from encode import team_fields

GROUP = os.environ.get("CEM_GROUP", "dbbo")
assert GROUP in ("dbbo", "scratch"), "CEM_GROUP must be dbbo or scratch"
SMOKE = bool(os.environ.get("CEM_SMOKE"))
FIXED_W = float(os.environ["CEM_W"]) if os.environ.get("CEM_W") else None
TAG = f"cem_{GROUP}" + (f"_w{FIXED_W:g}" if FIXED_W is not None else "") + ("_smoke" if SMOKE else "")
OUT = L.RES / f"{TAG}.json"
RUNTIME = Path.home() / ".local/share/vgc-pilot-runtime/regmb-v3"

GENS, PROPOSE, BATTLE, CUT = 10, 512, 128, 0.531               # same budget and reference as cem_v3
GUIDE, TEMP, SEED = 2.0, 1.0, 4040
W_SET, Y_CAP, PROBE, KAPPA, NOISE_SD = (1.0, 1.1, 1.2, 1.3), 1.0, 64, 1.0, 0.07
FT_EPOCHS, FT_LR, FT_BATCH, LOOP_W = 40, 1e-4, 64, 10        # dbbo: as cem_v3's re-steer
SCRATCH_EPOCHS = int(os.environ.get("CEM_SCRATCH_EPOCHS", 10))  # scratch: lossdown's default schedule
SCRATCH_LR, SCRATCH_BATCH, SCRATCH_WARMUP = 5e-4, 256, 500
SAT_EPS, SAT_PATIENCE, COPY_MAX, SETS_FRAC = 0.01, 3, 0.10, 0.5
TABOO = os.environ.get("CEM_TABOO", "1") == "1"                # refuse proposals whose species set was battled
FINAL_N, FINAL_BATTLES = 16, 192
if SMOKE:
    GENS, PROPOSE, BATTLE, PROBE, FT_EPOCHS, SCRATCH_EPOCHS, FINAL_N, FINAL_BATTLES = 2, 16, 4, 8, 1, 1, 2, 8


def save(out): json.dump(out, open(OUT, "w"))


# ---------------------------------------------------------------- the target: best so far x w
class Target:
    """Diffusion-BBO's requested score. best = the best scored win rate so far (anchors + every battled team)."""

    def __init__(self, rows): self.best = max(r["y"] for r in rows)

    def ask(self, w): return min(w * self.best, Y_CAP)

    def pick(self, G, sur, ridge, rng):
        """Adaptive w: for each candidate, decode a probe batch at its ask and take the ridge's lower
        confidence bound (mean prediction - KAPPA x mean posterior sd). CEM_W fixes w instead."""
        if FIXED_W is not None: return FIXED_W, {"fixed": True}
        probes = {}
        for w in W_SET:
            ystar = self.ask(w)
            teams = [parse_team_text(G.to_paste(r, rng, repair=True)[0])
                     for r in G.sample(PROBE, ystar, GUIDE, TEMP, 1.0, sur, sur.lam).cpu().numpy()]
            teams = [t for t in teams if len(t) == 6]
            if not teams: probes[w] = dict(ask=ystar, mu=None, sd=None, lcb=-math.inf); continue
            mu = np.array([sur.score(t) for t in teams]); sd = ridge.sd(teams)
            probes[w] = dict(ask=ystar, mu=float(mu.mean()), sd=float(sd.mean()), lcb=float(mu.mean() - KAPPA * sd.mean()))
        w = max(probes, key=lambda k: probes[k]["lcb"])
        return w, probes


class Ridge(C.Ridge):
    """cem_v3's ridge plus the posterior sd of a prediction, for the adaptive w."""

    def fit(self, teams, ys, path):
        XtX, Xty = self.base(); XtX = XtX.copy()
        if teams:
            Xe = np.stack([self.vec(t) for t in teams]); XtX += LOOP_W * Xe.T @ Xe
        self.chol = np.linalg.cholesky(XtX + np.diag(self.lam))
        return super().fit(teams, ys, path)

    def sd(self, teams):
        if not teams: return np.zeros(0)
        Z = np.linalg.solve(self.chol, np.stack([self.vec(t) for t in teams]).T)   # L z = x  ->  x' A^-1 x = |z|^2
        return NOISE_SD * np.sqrt((Z ** 2).sum(0))


# ---------------------------------------------------------------- the models
def encoded(rows, V):
    out = []
    for r in rows:
        t = parse_team_text(r["paste"])
        if len(t) != 6: continue                                  # parse_team_text merges item-less slots (known)
        try: out.append((V.encode(t), float(r["y"])))
        except ValueError: pass                                   # outside the regulation vocabulary
    return out


def dbbo_members(V, tables, rows, g):
    """The ORIGINAL v3 members fine-tuned on every scored team (cached per generation). g = 0: as shipped."""
    nets = []
    data = encoded(rows, V) if g > 0 else None
    for name in FD.V3_MEMBERS:
        ck = L.CK / f"{TAG}_g{g}_{name}.pt"
        net, cfg, _ = L.load_net(name, V)
        if ck.exists():
            net.load_state_dict(torch.load(ck, map_location=L.DEV)["sd"])
        elif data is not None:
            torch.manual_seed(SEED + g)
            Xe = torch.as_tensor(np.stack([e[0] for e in data]), device=L.DEV)
            We = torch.as_tensor(np.array([e[1] for e in data], np.float32), device=L.DEV)
            if cfg.get("mrow"): We = torch.cat([We[:, None], torch.full((len(We), 49), float("nan"), device=L.DEV)], 1)
            sched = L.sched_tensor(cfg["sched"]).to(L.DEV); rules = tuple(cfg["rules"].split(","))
            opt = torch.optim.AdamW(net.parameters(), lr=FT_LR, weight_decay=0.01)
            net.train()
            for _ in range(FT_EPOCHS):
                perm = torch.randperm(len(Xe), device=L.DEV)
                for i in range(0, len(Xe), FT_BATCH):
                    b = perm[i:i + FT_BATCH]
                    loss = L.batch_loss(net, tables, rules, Xe[b], We[b], None, sched, cfg)
                    opt.zero_grad(set_to_none=True); loss.backward()
                    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
            net.eval()
            torch.save({"sd": net.state_dict(), "cfg": ck.name}, ck)
        nets.append(L.logprobs_fn(net, tables, tuple(cfg["rules"].split(","))))
    return L.ensemble_fn(nets)


def scratch_member(V, tables, rows, g, log):
    """A fresh F1-architecture network trained from random weights on the regulation-vocabulary foundation
    data (labelled rows + unlabelled synthetic teams, as retrain_regmb) plus every scored team at weight
    LOOP_W. Retrained every generation (cached). g = 0 trains on the foundation alone."""
    name = FD.V3_MEMBERS[0]
    ck = L.CK / f"{TAG}_g{g}.pt"
    _, cfg, _ = L.load_net(name, V)
    net = L.Net(V, d=cfg["d"], nhead=cfg["nhead"], nlayer=cfg["nlayer"], dropout=cfg["dropout"],
                time_cond=bool(cfg["time_cond"]), ntag=0, prime=cfg["prime"], mrow=0).to(L.DEV)
    rules = tuple(cfg["rules"].split(","))
    if ck.exists():
        net.load_state_dict(torch.load(ck, map_location=L.DEV)["sd"]); net.eval()
        return L.logprobs_fn(net, tables, rules)
    for f in ("remapped.npz", "synthetic.npy"):
        if not (RUNTIME / f).exists(): raise FileNotFoundError(f"{RUNTIME / f}: run scripts/retrain_regmb.py prepare --version v3 first")
    d = L.load_data(); mapped = np.load(RUNTIME / "remapped.npz"); synthetic = np.load(RUNTIME / "synthetic.npy")
    idx = np.where(mapped["valid"])[0]                            # labelled and unlabelled rows (NaN = unconditioned)
    xs, ws = [mapped["X"][idx], synthetic], [d["W"][idx].astype(np.float32), np.full(len(synthetic), np.nan, np.float32)]
    data = encoded(rows, V) if g > 0 else []
    if data:
        xs += [np.stack([e[0] for e in data])] * LOOP_W; ws += [np.array([e[1] for e in data], np.float32)] * LOOP_W
    xx = torch.as_tensor(np.concatenate(xs), device=L.DEV); ww = torch.as_tensor(np.concatenate(ws), device=L.DEV)
    if SMOKE: xx, ww = xx[:512], ww[:512]
    torch.manual_seed(SEED + g)
    sched = L.sched_tensor(cfg["sched"]).to(L.DEV)
    steps_per_ep = math.ceil(len(xx) / SCRATCH_BATCH); total = steps_per_ep * SCRATCH_EPOCHS
    opt = torch.optim.AdamW(net.parameters(), lr=SCRATCH_LR, weight_decay=0.01, betas=(0.9, 0.98))
    lr_at = lambda s: SCRATCH_LR * min(1.0, (s + 1) / SCRATCH_WARMUP) * 0.5 * (1 + math.cos(math.pi * min(s, total) / total))
    t0 = time.perf_counter(); step = 0
    print(f"[{TAG} g{g}] scratch: {len(xx)} rows ({len(idx)} foundation, {len(synthetic)} synthetic, {len(data)} scored x{LOOP_W}) "
          f"· {SCRATCH_EPOCHS} epochs · {total} steps", flush=True)
    for ep in range(SCRATCH_EPOCHS):
        net.train(); perm = torch.randperm(len(xx), device=L.DEV); run = 0.0
        for i in range(0, len(xx), SCRATCH_BATCH):
            b = perm[i:i + SCRATCH_BATCH]
            loss = L.batch_loss(net, tables, rules, xx[b], ww[b], None, sched, cfg)
            if not torch.isfinite(loss): raise RuntimeError("non-finite training loss")
            for gp in opt.param_groups: gp["lr"] = lr_at(step)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); step += 1
            run += float(loss.detach())
        log.append(dict(ep=ep + 1, train=run / steps_per_ep, minutes=(time.perf_counter() - t0) / 60))
        print(f"  ep{ep + 1:3d} train {run / steps_per_ep:.4f} ({(time.perf_counter() - t0) / 60:.1f} min)", flush=True)
    net.eval()
    torch.save({"sd": net.state_dict(), "cfg": {**cfg, "warm": "", "epochs": SCRATCH_EPOCHS}, "vocabulary": V.itos,
                "regulation": V.regulation}, ck)
    return L.logprobs_fn(net, tables, rules)


# ---------------------------------------------------------------- one run
def run():
    LB._stack()
    con = MDB.connect()
    out = json.load(open(OUT)) if OUT.exists() else {
        "config": dict(group=GROUP, gens=GENS, propose=PROPOSE, battle=BATTLE, cut=CUT, guide=GUIDE, temp=TEMP, w_set=W_SET,
                       fixed_w=FIXED_W, y_cap=Y_CAP, probe=PROBE, kappa=KAPPA, noise_sd=NOISE_SD, taboo=TABOO,
                       ft_epochs=FT_EPOCHS, ft_lr=FT_LR, scratch_epochs=SCRATCH_EPOCHS, loop_weight=LOOP_W, seed=SEED),
        "gens": {}}
    if "anchors" not in out:
        out["anchors"] = C.anchors(con); save(out)
    G = FD.Gen("v3"); V, tables = G.V, G.tables
    known = {tuple(team_fields(parse_team_text(r["paste"]))) for r in out["anchors"]}
    cV = X.setup()[1]; d = L.load_data()
    known |= {tuple(cV.decode_field(c, int(r[c])) for c in range(L.COLS)) for r in d["X"]}
    ridge = Ridge()
    best = []
    for g in range(GENS):
        c = out["gens"].setdefault(str(g), {})
        rows = C.labelled({**out, "gens": {k: v for k, v in out["gens"].items() if int(k) < g}})   # EVERY scored team
        battled_sets = {tuple(sorted(norm(s["species"]) for s in parse_team_text(p)))
                        for k, v in out["gens"].items() if int(k) < g for p in v.get("battled", [])}
        # 1. model: trained on every scored team, not the elites
        if g > 0 and "trained_on" not in c:
            c["trained_on"] = dict(n=len(rows), anchors=sum(r["src"] == "anchor" for r in rows),
                                   above_cut=sum(r["y"] >= CUT for r in rows))
        if GROUP == "dbbo": G.f = dbbo_members(V, tables, rows, g)
        else: G.f = scratch_member(V, tables, rows, g, c.setdefault("scratch_log", []))
        save(out)
        # 2. ridge
        loop_rows = [r for r in rows if r["src"] != "anchor"]
        FD.RIDGE = ridge.fit([parse_team_text(r["paste"]) for r in loop_rows], [r["y"] for r in loop_rows],
                             L.RES / f"{TAG}_ridge_g{g}.npz")
        sur = FD.Surrogate(V)
        # 3. target: best so far x w
        rng = np.random.default_rng(SEED + 100 * g); torch.manual_seed(SEED + 100 * g)
        if "ask" not in c:
            tgt = Target(rows)
            w, probes = tgt.pick(G, sur, ridge, rng)
            c.update(best_so_far=tgt.best, w=w, ask=tgt.ask(w), probes={str(k): v for k, v in probes.items()})
            save(out)
            print(f"[g{g}] best so far {tgt.best:.3f} · w {w} · ask {c['ask']:.3f}", flush=True)
        # 4. propose
        if "proposals" not in c:
            val = Validator(); keep, tried, tabooed, t0 = [], 0, 0, time.perf_counter()
            try:
                while len(keep) < PROPOSE and tried < PROPOSE * 8:
                    for r in G.sample(96, c["ask"], GUIDE, TEMP, 1.0, sur, sur.lam).cpu().numpy():
                        tried += 1
                        p, slots = G.to_paste(r, rng, repair=True)
                        if TABOO and tuple(sorted(norm(s["species"]) for s in parse_team_text(p))) in battled_sets:
                            tabooed += 1; continue
                        if val(p) is None: keep.append(p)
                        if len(keep) >= PROPOSE: break
            finally:
                val.close()
            teams = [parse_team_text(p) for p in keep]
            c.update(proposals=keep, mu=[float(sur.score(t)) for t in teams], tried=tried, tabooed=tabooed,
                     valid=len(keep) / max(tried - tabooed, 1),
                     sets=len({tuple(sorted(norm(s["species"]) for s in t)) for t in teams}),
                     copies=sum(tuple(team_fields(t)) in known for t in teams) / max(len(teams), 1),
                     new_species=sum(any(norm(s["species"]) not in C.LEGACY for s in t) for t in teams),
                     propose_s=time.perf_counter() - t0)
            save(out)
            print(f"[g{g}] proposed {len(keep)}/{tried} ({tabooed} taboo) · {c['sets']} species sets · copies {c['copies']:.1%} · "
                  f"new-species teams {c['new_species']}", flush=True)
        # 5. acquire + battle: the 128 the ridge ranks highest
        if "y" not in c:
            order = np.argsort(c["mu"])[::-1][:BATTLE]
            c["battled"] = [c["proposals"][i] for i in order]; c["battled_mu"] = [c["mu"][i] for i in order]
            v = MDB.Validator()
            try: ids = [MDB.add_team(con, p, f"{TAG}:g{g}", v)[0] for p in c["battled"]]
            finally: v.close()
            legal = {t for (t,) in con.execute(f"SELECT team_id FROM team WHERE legal=1 AND team_id IN ({','.join('?' * len(ids))})", ids)}
            MDB.ensure(con, LB.POLICY, sorted(legal), MDB.set_members(con, "top50"), LB.PER_CELL, seed=SEED + g, note=f"{TAG} g{g}")
            sc = [MDB.score(con, LB.POLICY, t) if t in legal else None for t in ids]
            c["team_id"] = ids; c["y"] = [s["score"] if s else None for s in sc]
            y = np.array([v for v in c["y"] if v is not None], float)
            bt = [parse_team_text(p) for p in c["battled"]]
            nsp = [any(norm(s["species"]) not in C.LEGACY for s in t) for t in bt]
            c["stats"] = dict(mean=float(y.mean()), se=float(y.std(ddof=1) / math.sqrt(len(y))), p90=float(np.quantile(y, .9)),
                              max=float(y.max()), above_cut=int((y >= CUT).sum()), n=len(y),
                              at_or_above_ask=int((y >= c["ask"]).sum()),
                              battled_sets=len({tuple(sorted(norm(s["species"]) for s in t)) for t in bt}),
                              new_species_battled=int(sum(nsp)),
                              new_species_mean=float(np.mean([v for v, f in zip(c["y"], nsp) if f and v is not None])) if any(nsp) else None)
            save(out)
        known |= {tuple(team_fields(parse_team_text(p))) for p in c["battled"]}
        s = c["stats"]; best.append(s["mean"])
        print(f"[g{g}] battled {s['n']}: mean {s['mean']:.3f} ± {s['se']:.3f} · p90 {s['p90']:.3f} · max {s['max']:.3f} · "
              f"≥{CUT}: {s['above_cut']} · ≥ask: {s['at_or_above_ask']} · new-species teams {s['new_species_battled']}", flush=True)
        # 6. stopping (as cem_v3)
        stop = None
        if c["copies"] > COPY_MAX: stop = f"copy rate {c['copies']:.1%} > {COPY_MAX:.0%}"
        elif g > 0 and c["sets"] < SETS_FRAC * out["gens"]["0"]["sets"]:
            stop = f"{c['sets']} species sets < half of generation 0's {out['gens']['0']['sets']}"
        elif len(best) > SAT_PATIENCE:
            run_best = [max(best[:i + 1]) for i in range(len(best))]
            if run_best[-1] - run_best[-1 - SAT_PATIENCE] < SAT_EPS: stop = f"saturated: no +{SAT_EPS} in {SAT_PATIENCE} generations"
        if stop:
            out["stopped"] = dict(generation=g, reason=stop); save(out)
            print(f"[g{g}] STOP: {stop}", flush=True); return
    out["stopped"] = dict(generation=GENS - 1, reason="max generations"); save(out)


def final():
    """The 16 best distinct battled teams by loop label, re-battled fresh: 192 battles vs the top-50 schedule."""
    out = json.load(open(OUT))
    if "final" in out: return
    rows = [r for r in C.labelled(out) if r["src"] != "anchor"]
    rows.sort(key=lambda r: -r["y"])
    seen, top = set(), []
    for r in rows:
        k = tuple(team_fields(parse_team_text(r["paste"])))
        if k in seen: continue
        seen.add(k); top.append(r)
        if len(top) >= FINAL_N: break
    pool.REPO = str(Path.home() / ".local/share/vgc-pilot-runtime/unpacked/vgc-bench-d79f9532947ac114dce1dda2456a590afcd375b2")
    d = Path(f"/tmp/vgc-pilot/{TAG}_final"); d.mkdir(parents=True, exist_ok=True)
    files = []
    for i, r in enumerate(top):
        f = d / f"{i:02d}.txt"; f.write_text(r["paste"]); files.append(str(f))
    res = pool.score([(f, A.opponents()) for f in files], battles=FINAL_BATTLES, conc=50)
    out["final"] = [dict(paste=r["paste"], label=r["y"], src=r["src"], fresh=res.get(f, {}).get("win_rate"),
                         battles=res.get(f, {}).get("battles")) for r, f in zip(top, files)]
    save(out)
    report()


def report():
    out = json.load(open(OUT))
    print(f"group {GROUP}")
    print(f"{'gen':>4s} {'best':>6s} {'w':>4s} {'ask':>6s} {'mean':>13s} {'p90':>6s} {'max':>6s} {'>=cut':>6s} {'>=ask':>6s} "
          f"{'sets/512':>9s} {'taboo':>6s} {'copies':>7s} {'new-sp':>7s} {'trained':>8s}")
    for g, c in sorted(out["gens"].items(), key=lambda kv: int(kv[0])):
        if "stats" not in c: continue
        s = c["stats"]; t = c.get("trained_on", {})
        print(f"{g:>4s} {c['best_so_far']:6.3f} {c['w']:4.1f} {c['ask']:6.3f} {s['mean']:.3f}±{s['se']:.3f} {s['p90']:6.3f} "
              f"{s['max']:6.3f} {s['above_cut']:6d} {s['at_or_above_ask']:6d} {c['sets']:9d} {c.get('tabooed', 0):6d} "
              f"{c['copies']:7.1%} {s['new_species_battled']:7d} {t.get('n', '-'):>8}")
    if out.get("stopped"): print("stopped:", out["stopped"])
    if out.get("final"):
        f = [r for r in out["final"] if r["fresh"] is not None]
        print(f"final: {len(f)} teams re-battled x {FINAL_BATTLES}: fresh mean {np.mean([r['fresh'] for r in f]):.3f} "
              f"(loop labels {np.mean([r['label'] for r in f]):.3f}) · best fresh {max(r['fresh'] for r in f):.3f} · "
              f"≥{CUT} fresh: {sum(r['fresh'] >= CUT for r in f)}")


if __name__ == "__main__":
    {"run": run, "final": final, "report": report}[sys.argv[1]]()
