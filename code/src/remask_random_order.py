"""Can remasking rescue random decoding order? (2026-10-31) Pre-registration: docs/remask-random-order.md.

Random order made legal teams on under 1% of tries (docs/decoding-order.md): a Pokémon's set drawn before its species
leaves no legal species. Remasking redraws filled fields, and a field redrawn after its species is checked against it.
Arms, all random order: remasking 10% of filled fields (least likely), 10% (random fields), 20% (random fields) and 10%
(random fields) at 96 passes. The control is the decoding-order experiment's random-order cells, reused; species-first
order from the same results is the reference. Same model, asks, seeds, sampling and battles as decoding_order.py.

    python remask_random_order.py gen | battle | report      (run with /tmp/vgc-pilot/.venv/bin/python)
    DO_SMOKE=1 python remask_random_order.py ...             # plumbing check, own names and files
"""
import json, math, os, sys, time
from collections import Counter
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, "/tmp/vgc-pilot/src")
sys.path.insert(0, str(HERE))
import decoding_order as DO                         # checks that the model module sits beside this script
import decoding_order_diversity as DD
import discrete_diffusion_model
from discrete_diffusion_model import DiscreteDiffusionModel
sys.path.insert(0, str(HERE))                       # discrete_diffusion_model's imports put /tmp/vgc-pilot/src first again
import remask_ask_gap as RA
import matchup_db as MDB
import heldout as HO
from hps_generate import Validator
from team_diversity import slots_from_paste

ARMS = [("lowest10", "lowest", 0.10, 48), ("random10", "random", 0.10, 48), ("random20", "random", 0.20, 48),
        ("random10_96", "random", 0.10, 96)]
NAMES = {"species": "species-first order (reference)", "control": "random order, no remasking",
         "lowest10": "remasking 10%, least likely fields", "random10": "remasking 10%, random fields",
         "random20": "remasking 20%, random fields", "random10_96": "remasking 10%, random fields, 96 passes"}
SMOKE, ASKS, SEEDS, KEEP, BATCH, MAX_TRIES, SEED0, GUIDANCE = (DO.SMOKE, DO.ASKS, DO.SEEDS, DO.KEEP, DO.BATCH,
                                                               DO.MAX_TRIES, DO.SEED0, DO.GUIDANCE)
OUT = DO.RES / ("remask_random_order_smoke.json" if SMOKE else "remask_random_order.json")
ORIGIN = "rrosmoke" if SMOKE else "rro"
LOWEST = discrete_diffusion_model._lowest_confidence
MATCH = 5                                           # teams per subset for size-matched diversity, as decoding_order_diversity


def random_pick(conf, fraction):
    """_lowest_confidence's count of fields, chosen at random among the filled ones (as remask_why.py)."""
    filled = conf.isfinite()
    k = torch.floor(fraction * filled.sum(1) + torch.rand(len(conf), device=conf.device)).long()
    score = torch.rand(conf.shape, device=conf.device).masked_fill(~filled, float("inf"))
    columns = torch.arange(conf.shape[1], device=conf.device).expand_as(conf)
    rank = torch.empty_like(columns).scatter_(1, score.argsort(1), columns)
    return (rank < k[:, None]) & filled


def save(out):
    tmp = OUT.with_suffix(".tmp"); json.dump(out, open(tmp, "w")); os.replace(tmp, OUT)


def draw(w, ask, rule, remask, steps, seed):
    """Sample until KEEP teams pass Showdown and are not held-out teams (cap MAX_TRIES tries), random order with
    remasking; stats collects passes, undone remasking passes and redraws per field over every try."""
    torch.manual_seed(SEED0 + seed); rng = np.random.default_rng(SEED0 + seed)
    discrete_diffusion_model._lowest_confidence = random_pick if rule == "random" else LOWEST
    val, stats = Validator(), {}
    kept, rows_kept, tried, rejected, heldout, reasons, t0 = [], [], 0, 0, 0, Counter(), time.perf_counter()
    try:
        while len(kept) < KEEP and tried < MAX_TRIES:
            for r in w.sample(BATCH, ask=ask, guidance=GUIDANCE, order="random", steps=steps, remask=remask, stats=stats).cpu().numpy():
                tried += 1
                p, _ = w.to_paste(r, rng, repair=True)
                err = val(p)
                if err is not None: rejected += 1; reasons[str(err).split("\n")[0][:120]] += 1
                elif HO.is_heldout_text(p): heldout += 1
                else: kept.append(p); rows_kept.append([int(v) for v in r])
                if len(kept) >= KEEP or tried >= MAX_TRIES: break
    finally:
        val.close(); discrete_diffusion_model._lowest_confidence = LOWEST
    return dict(kept=kept, rows=rows_kept, tried=tried, rejected=rejected, heldout=heldout, reasons=reasons.most_common(8),
                seconds=time.perf_counter() - t0, stats=stats)


def gen():
    now = dict(model_sha256=RA.sha(discrete_diffusion_model.__file__), script_sha256=RA.sha(__file__), decoding_order_sha256=RA.sha(DO.__file__))
    out = json.load(open(OUT)) if OUT.exists() else {"config": dict(
        arms=ARMS, asks=ASKS, seeds=SEEDS, seed0=SEED0, keep=KEEP, batch=BATCH, max_tries=MAX_TRIES, guidance=GUIDANCE,
        temperature=1.0, masks="on", repair=True, order="random", battles_per_opponent=1, slots="shuffled", **now), "cells": {}}
    missing = [(s, a, arm) for s in SEEDS for a in ASKS for arm in ARMS if DO.tag_of(arm[0], a, s) not in out["cells"]]
    if not missing: return
    changed = [k for k, v in now.items() if out["config"].get(k) != v]
    if out["cells"] and changed and not os.environ.get("DO_RESUME_CHANGED"):
        raise SystemExit(f"{OUT} was started with different {changed}; finishing it would mix implementations (DO_RESUME_CHANGED=1 overrides)")
    w = DiscreteDiffusionModel.load()
    for seed, ask, (arm, rule, remask, steps) in missing:
        tag = DO.tag_of(arm, ask, seed); r = draw(w, ask, rule, remask, steps, seed)
        legal = (r["tried"] - r["rejected"]) / max(r["tried"], 1)
        out["cells"][tag] = dict(arm=arm, rule=rule, remask=remask, steps=steps, ask=ask, seed=seed, tried=r["tried"],
                                 rejected=r["rejected"], heldout=r["heldout"], legal=legal, kept=len(r["kept"]),
                                 capped=len(r["kept"]) < KEEP, reasons=r["reasons"], seconds=r["seconds"], stats=r["stats"],
                                 div=RA.diversity(r["kept"]) if r["kept"] else None, pastes=r["kept"], rows=r["rows"])
        save(out)
        print(f"  {tag:26s} legal {legal:6.1%} ({r['tried'] - r['rejected']}/{r['tried']}) · kept {len(r['kept'])} · "
              f"undone {r['stats'].get('reverted', 0)} · {r['seconds']:.0f}s", flush=True)


def battle():
    RA.require_shuffled(); con = MDB.connect(); out = json.load(open(OUT))
    for tag, c in out["cells"].items():
        if "y" in c or not c["pastes"]: continue
        sc = MDB.score_pastes(con, c["pastes"], per_cell=1, seed=SEED0 + c["seed"], origin=f"{ORIGIN}:{tag}")
        if not RA.complete(sc, 1): print(f"  {tag:26s} INCOMPLETE: left for the next run", flush=True); continue
        y = np.array([s["score"] for s in sc], float)
        c["y"] = [s["score"] for s in sc]
        c["win"] = dict(mean=float(y.mean()), se=float(y.std(ddof=1) / math.sqrt(len(y))) if len(y) > 1 else 0.0, n=len(y))
        save(out); print(f"  {tag:26s} battled {len(y)}: win {y.mean():.3f}", flush=True)


def report():  # noqa: C901
    out = json.load(open(OUT)); ref = json.load(open(DO.OUT))["cells"]
    cells = list(out["cells"].values())
    cells += [dict(c, arm="control") for c in ref.values() if c["arm"] == "random1" and c["seed"] in SEEDS and c["ask"] in ASKS]
    cells += [c for c in ref.values() if c["arm"] == "species" and c["seed"] in SEEDS and c["ask"] in ASKS]
    w = DiscreteDiffusionModel.load(); full, lineups, M = DO.training_sets(w); rng = np.random.default_rng(0)
    for c in cells:
        teams = [t for t in map(slots_from_paste, c["pastes"]) if t]
        c["matched"] = DD.matched(teams, MATCH, rng) if len(teams) >= MATCH else None
        c["novelty"] = DO.novelty(c.get("rows", []), full, lineups, M)
    get = lambda arm, ask: sorted((c for c in cells if c["arm"] == arm and abs(c["ask"] - ask) < 1e-9), key=lambda c: c["seed"])
    win = lambda c: c["win"]["mean"] if "win" in c else None
    pick = lambda field, k: lambda c: (c[field] or {}).get(k)
    summ = {}
    for arm in ["species", "control"] + [a[0] for a in ARMS]:
        cs = [c for a in ASKS for c in get(arm, a)]
        if len(cs) != len(ASKS) * len(SEEDS) or not all("win" in c or not c["pastes"] for c in cs):
            print(f"{NAMES[arm]}: incomplete"); continue
        per_seed = lambda f: [DO.mean_of([f(c) for a in ASKS for c in get(arm, a) if c["seed"] == s]) for s in SEEDS]
        slope = []
        for s in SEEDS:
            pts = [(a, win(c)) for a in DO.GRID_ASKS for c in get(arm, a) if c["seed"] == s and win(c) is not None]
            slope.append(float(np.polyfit(*zip(*pts), 1)[0]) if len(pts) >= 2 else float("nan"))
        summ[arm] = dict(name=NAMES[arm], legal=DO.stat(per_seed(lambda c: c["legal"])), win=DO.stat(per_seed(win)),
                         ask_slope=DO.stat(slope), gap=DO.stat(per_seed(lambda c: None if win(c) is None else abs(c["ask"] - win(c)))),
                         effective_species=DO.stat(per_seed(pick("matched", "effective_species"))),
                         distinct_lineups=DO.stat(per_seed(pick("matched", "distinct_lineups"))),
                         top_species_share=DO.stat(per_seed(pick("matched", "top_species_share"))),
                         mean_distance=DO.stat(per_seed(pick("div", "mean_distance"))),
                         copies=DO.stat(per_seed(pick("novelty", "copies"))), new_lineups=DO.stat(per_seed(pick("novelty", "new_lineups"))),
                         lineup_distance=DO.stat(per_seed(pick("novelty", "lineup_distance"))),
                         kept=sum(c["kept"] for c in cs), tried=sum(c["tried"] for c in cs), capped=sum(c["capped"] for c in cs),
                         undone=sum((c.get("stats") or {}).get("reverted", 0) for c in cs),
                         seconds_per_team=sum(c["seconds"] for c in cs) / max(sum(c["kept"] for c in cs), 1))
    measures = ("legal", "win", "ask_slope", "gap", "effective_species", "distinct_lineups", "mean_distance", "new_lineups", "lineup_distance")
    for base in ("control", "species"):
        if base not in summ: continue
        b = summ[base]
        for arm, s in summ.items():
            if arm in ("species", base): continue
            s[f"vs_{base}"] = {m: dict(diff=s[m]["mean"] - b[m]["mean"], noise=DO.NOISE_K * max(b[m]["sd"], s[m]["sd"]),
                                       seeds=[x - y for x, y in zip(s[m]["seeds"], b[m]["seeds"])]) for m in measures}
    for arm, s in summ.items():
        v = s.get("vs_control", {}).get("legal")
        s["rescues"] = bool(v and v["diff"] > v["noise"] and s["legal"]["mean"] >= 0.5)
    out["summary"] = summ; save(out)
    for arm, s in summ.items():
        print(f"{s['name']:42s} legal {s['legal']['mean']:6.1%} · win {s['win']['mean']:.3f} · ask slope {s['ask_slope']['mean']:+.2f} · "
              f"eff species {s['effective_species']['mean']:.1f} · distance {s['mean_distance']['mean']:.1f} · new line-ups "
              f"{s['new_lineups']['mean']:.1%} · teams {s['kept']} · undone {s['undone']} · rescues {s['rescues']}")


if __name__ == "__main__":
    {"gen": gen, "battle": battle, "report": report}[sys.argv[1]]()
