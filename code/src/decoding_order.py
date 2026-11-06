"""Does the decoding order change legality, diversity, novelty and win rate? (2026-10-30)
Pre-registration: docs/decoding-order.md.

Orders (DiscreteDiffusionModel.sample(order=, steps=)): species-first (the current sampler), Pokémon-by-Pokémon, random with one
field per pass, random with two fields per pass (24 passes), and confidence order (each team's most confident masked
field first). Same model, guidance 2, temperature 1, rule masks on, to_paste(repair=True). Asks 7/24, 0.3, 8/24, 0.5,
0.531, 13/24; seeds 1-3 at base 7300; each cell draws until 64 teams pass Showdown's validator and are not held-out
teams (cap 1,280 tries). Species-first cells redraw remask_ask_gap.py's and remask_why.py's plain teams exactly, so
their battles come back from the matchup database.
Battles: one against each of the 49 distinct top-50 meta teams, behaviour-cloning policy on both sides, slots shuffled.
Novelty, against the model's own training teams (both networks' rows), on the model's drawn tokens: copies (all 48
fields equal, slot order ignored), new line-ups (six species never together in training), and distance (species away
from the nearest training line-up, 0 to 6).

    python decoding_order.py gen | battle | report        (run with /tmp/vgc-pilot/.venv/bin/python)
    DO_SMOKE=1 python decoding_order.py gen|battle|report  # plumbing check, own files
"""
import json, math, os, sys, time
from collections import Counter
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, "/tmp/vgc-pilot/src")
sys.path.insert(0, str(HERE))
import discrete_diffusion_model
from discrete_diffusion_model import DiscreteDiffusionModel, MEMBERS
sys.path.insert(0, str(HERE))                       # discrete_diffusion_model's imports put /tmp/vgc-pilot/src first again
import remask_ask_gap as RA
import matchup_db as MDB
import heldout as HO
from hps_generate import Validator
from encode import NF, NSLOT

if Path(discrete_diffusion_model.__file__).resolve().parent != HERE:
    raise SystemExit(f"loaded {discrete_diffusion_model.__file__}, not the discrete diffusion model beside this script")

GUIDANCE, KEEP, BATCH, MAX_TRIES, SEED0 = 2.0, 64, 96, 20 * 64, 7300
ASKS = [7 / 24, 0.3, 8 / 24, 0.5, 0.531, 13 / 24]
GRID_ASKS = [a for a in ASKS if abs(a * 24 - round(a * 24)) < 1e-9]
ARMS = [("species", "species", 48), ("pokemon", "pokemon", 48), ("random1", "random", 48), ("random2", "random", 24),
        ("confidence", "confidence", 48)]
NAMES = {"species": "species-first order", "pokemon": "Pokémon-by-Pokémon order", "random1": "random order, one field per pass",
         "random2": "random order, two fields per pass", "confidence": "confidence order"}
SEEDS = [1, 2, 3]
NOISE_K = 2.0
SMOKE = bool(os.environ.get("DO_SMOKE"))
if SMOKE: KEEP, BATCH, SEEDS, ASKS, MAX_TRIES = 4, 8, [1], [0.5], 64
RES = Path.home() / "vgc-data" / "results"
OUT = RES / ("decoding_order_smoke.json" if SMOKE else "decoding_order.json")
ORIGIN = "dordersmoke" if SMOKE else "dorder"


def save(out):
    tmp = OUT.with_suffix(".tmp"); json.dump(out, open(tmp, "w")); os.replace(tmp, OUT)


def tag_of(arm, ask, seed): return f"{arm}_a{ask:.4f}_s{seed}"


def draw(w, ask, order, steps, seed):
    """Sample until KEEP teams pass Showdown and are not held-out teams (cap MAX_TRIES tries). Species-first order calls
    the current sampler itself, so its teams match the earlier grids."""
    torch.manual_seed(SEED0 + seed); rng = np.random.default_rng(SEED0 + seed)
    kw = {} if order == "species" else {"order": order, "steps": steps}
    val = Validator(); kept, rows_kept, tried, rejected, heldout, reasons, t0 = [], [], 0, 0, 0, Counter(), time.perf_counter()
    try:
        while len(kept) < KEEP and tried < MAX_TRIES:
            for r in w.sample(BATCH, ask=ask, guidance=GUIDANCE, **kw).cpu().numpy():
                tried += 1
                p, _ = w.to_paste(r, rng, repair=True)
                err = val(p)
                if err is not None: rejected += 1; reasons[str(err).split("\n")[0][:120]] += 1
                elif HO.is_heldout_text(p): heldout += 1
                else: kept.append(p); rows_kept.append([int(v) for v in r])
                if len(kept) >= KEEP or tried >= MAX_TRIES: break
    finally:
        val.close()
    return dict(kept=kept, rows=rows_kept, tried=tried, rejected=rejected, heldout=heldout, reasons=reasons.most_common(8),
                seconds=time.perf_counter() - t0)


def gen():
    out = json.load(open(OUT)) if OUT.exists() else {"config": dict(
        arms=ARMS, asks=ASKS, seeds=SEEDS, seed0=SEED0, keep=KEEP, batch=BATCH, max_tries=MAX_TRIES, guidance=GUIDANCE,
        temperature=1.0, masks="on", repair=True, members=list(MEMBERS), battles_per_opponent=1, slots="shuffled",
        team_writer_sha256=RA.sha(discrete_diffusion_model.__file__), script_sha256=RA.sha(__file__)), "cells": {}}
    missing = [(s, a, arm, o, st) for s in SEEDS for a in ASKS for arm, o, st in ARMS if tag_of(arm, a, s) not in out["cells"]]
    if not missing: return
    changed = [k for k, v in dict(team_writer_sha256=RA.sha(discrete_diffusion_model.__file__), script_sha256=RA.sha(__file__)).items() if out["config"].get(k) != v]
    if out["cells"] and changed and not os.environ.get("DO_RESUME_CHANGED"):
        raise SystemExit(f"{OUT} was started with different {changed}; finishing it would mix implementations (DO_RESUME_CHANGED=1 overrides)")
    w = DiscreteDiffusionModel.load()
    for seed, ask, arm, order, steps in missing:
        tag = tag_of(arm, ask, seed); r = draw(w, ask, order, steps, seed)
        legal = (r["tried"] - r["rejected"]) / max(r["tried"], 1)
        out["cells"][tag] = dict(arm=arm, order=order, steps=steps, ask=ask, seed=seed, tried=r["tried"], rejected=r["rejected"],
                                 heldout=r["heldout"], legal=legal, kept=len(r["kept"]), capped=len(r["kept"]) < KEEP, reasons=r["reasons"],
                                 seconds=r["seconds"], div=RA.diversity(r["kept"]) if r["kept"] else None, pastes=r["kept"], rows=r["rows"])
        save(out)
        d = out["cells"][tag]["div"] or {}
        print(f"  {tag:24s} legal {legal:6.1%} ({r['tried'] - r['rejected']}/{r['tried']}) · kept {len(r['kept'])} · line-ups "
              f"{d.get('distinct_lineups')} · eff species {d.get('effective_species', 0):.1f} · {r['seconds']:.0f}s", flush=True)


def battle():
    RA.require_shuffled(); con = MDB.connect(); out = json.load(open(OUT))
    for tag, c in out["cells"].items():
        if "y" in c or not c["pastes"]: continue
        sc = MDB.score_pastes(con, c["pastes"], per_cell=1, seed=SEED0 + c["seed"], origin=f"{ORIGIN}:{tag}")
        if not RA.complete(sc, 1): print(f"  {tag:24s} INCOMPLETE: left for the next run", flush=True); continue
        y = np.array([s["score"] for s in sc], float)
        c["y"] = [s["score"] for s in sc]; c["win"] = dict(mean=float(y.mean()), se=float(y.std(ddof=1) / math.sqrt(len(y))) if len(y) > 1 else 0.0, n=len(y))
        save(out); print(f"  {tag:24s} battled {len(y)}: win {y.mean():.3f}", flush=True)


# ---------------------------------------------------------------- novelty against the training teams
def training_sets(w):
    """(canonical full teams, line-ups as sorted species tuples, species one-hot matrix of the unique line-ups) over both
    networks' training rows."""
    full, lineups = set(), set()
    for name, _, cfg in w._nets:
        xo, _ = w._original_rows(name, cfg)
        for r in xo.cpu().numpy():
            slots = [tuple(int(v) for v in r[s * NF:(s + 1) * NF]) for s in range(NSLOT)]
            if any(s[0] == 0 for s in slots): continue
            full.add(tuple(sorted(slots))); lineups.add(tuple(sorted(s[0] for s in slots)))
    nsp = w.tables.sizes["species"]; L = sorted(lineups); M = np.zeros((len(L), nsp), np.float32)
    for i, t in enumerate(L): M[i, list(t)] = 1
    return full, lineups, M


def novelty(rows, full, lineups, M):
    if not rows: return None
    copies = new = 0; dist = []
    for r in rows:
        slots = [tuple(r[s * NF:(s + 1) * NF]) for s in range(NSLOT)]
        lu = tuple(sorted(s[0] for s in slots)); copies += tuple(sorted(slots)) in full; new += lu not in lineups
        v = np.zeros(M.shape[1], np.float32); v[list(lu)] = 1; dist.append(6 - float((M @ v).max()))
    return dict(copies=copies / len(rows), new_lineups=new / len(rows), lineup_distance=float(np.mean(dist)))


def mean_of(vals):
    """Mean of the values that exist: a cell with no legal team has no win rate, one with a single team no distance."""
    v = [x for x in vals if x is not None]; return float(np.mean(v)) if v else float("nan")


def stat(vals):
    v = np.array(vals, float); ok = v[~np.isnan(v)]
    return dict(mean=float(ok.mean()) if len(ok) else float("nan"), sd=float(ok.std(ddof=1)) if len(ok) > 1 else 0.0, seeds=v.tolist())


def report():
    out = json.load(open(OUT)); cells = out["cells"]; w = DiscreteDiffusionModel.load(); full, lineups, M = training_sets(w)
    for c in cells.values(): c["novelty"] = novelty(c.get("rows", []), full, lineups, M)
    get = lambda arm, ask: sorted((c for c in cells.values() if c["arm"] == arm and abs(c["ask"] - ask) < 1e-9), key=lambda c: c["seed"])
    summ = {}
    for arm, _, _ in ARMS:
        cs_all = [c for a in ASKS for c in get(arm, a)]
        if len(cs_all) != len(ASKS) * len(SEEDS) or not all("win" in c or not c["pastes"] for c in cs_all):
            print(f"{NAMES[arm]}: incomplete"); continue
        per_seed = lambda f: [mean_of([f(c) for a in ASKS for c in get(arm, a) if c["seed"] == s]) for s in SEEDS]
        win = lambda c: c["win"]["mean"] if "win" in c else None
        div = lambda k: lambda c: (c["div"] or {}).get(k)
        nov = lambda k: lambda c: (c["novelty"] or {}).get(k)
        slope = []
        for s in SEEDS:
            pts = [(a, win(c)) for a in GRID_ASKS for c in get(arm, a) if c["seed"] == s and win(c) is not None]
            slope.append(float(np.polyfit(*zip(*pts), 1)[0]) if len(pts) >= 2 else float("nan"))
        summ[arm] = dict(name=NAMES[arm], legal=stat(per_seed(lambda c: c["legal"])), win=stat(per_seed(win)),
                         win_by_ask={f"{a:.4f}": stat([mean_of([win(c)]) for c in get(arm, a)]) for a in ASKS},
                         ask_slope=stat(slope), gap=stat(per_seed(lambda c: None if win(c) is None else abs(c["ask"] - win(c)))),
                         lineups=stat(per_seed(div("distinct_lineups"))), entropy=stat(per_seed(div("species_entropy"))),
                         distance=stat(per_seed(div("mean_distance"))), copies=stat(per_seed(nov("copies"))),
                         new_lineups=stat(per_seed(nov("new_lineups"))), lineup_distance=stat(per_seed(nov("lineup_distance"))),
                         capped=sum(c["capped"] for c in cs_all), kept=sum(c["kept"] for c in cs_all))
    base = summ.get("species")
    if base:
        for arm, s in summ.items():
            if arm == "species": continue
            s["vs_species"] = {m: dict(diff=s[m]["mean"] - base[m]["mean"], noise=NOISE_K * max(base[m]["sd"], s[m]["sd"]),
                                       seeds=[a - b for a, b in zip(s[m]["seeds"], base[m]["seeds"])])
                               for m in ("legal", "win", "ask_slope", "gap", "lineups", "entropy", "distance", "copies", "new_lineups", "lineup_distance")}
    out["summary"] = summ; save(out)
    for arm, s in summ.items():
        print(f"{s['name']:36s} legal {s['legal']['mean']:6.1%} · win {s['win']['mean']:.3f} · ask slope {s['ask_slope']['mean']:+.2f} · "
              f"line-ups {s['lineups']['mean']:5.1f} · entropy {s['entropy']['mean']:.2f} · copies {s['copies']['mean']:.1%} · "
              f"new line-ups {s['new_lineups']['mean']:.1%} · distance {s['lineup_distance']['mean']:.2f} · teams {s['kept']} · capped {s['capped']}")


if __name__ == "__main__":
    {"gen": gen, "battle": battle, "report": report}[sys.argv[1]]()
