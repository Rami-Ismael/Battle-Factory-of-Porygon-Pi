"""Does a lookahead sampler fix the decoding orders that fill sets before species? (2026-10-31)
Pre-registration: docs/lookahead-legality.md.

Random order made legal teams on under 1% of tries and confidence order on 63% (docs/decoding-order.md): a Pokémon's
set drawn before its species can rule out every species. DiscreteDiffusionModel.sample(lookahead=True) allows only
values that leave every Pokémon a legal species. Arms: random order (one and two fields per pass) and confidence order,
with the lookahead. Controls: the same orders without it, and species-first order (identical with or without it), all
reused from the decoding-order results. Same model, asks, seeds, sampling and battles as decoding_order.py.

    python lookahead_legality.py gen | battle | report      (run with /tmp/vgc-pilot/.venv/bin/python)
    DO_SMOKE=1 python lookahead_legality.py ...             # plumbing check, own names and files
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

ARMS = [("random1_la", "random", 48, "random1"), ("random2_la", "random", 24, "random2"), ("confidence_la", "confidence", 48, "confidence")]
NAMES = {"species": "species-first order (reference)", "random1": "random order, one field per pass",
         "random2": "random order, two fields per pass", "confidence": "confidence order",
         "random1_la": "random order, one field per pass, lookahead", "random2_la": "random order, two fields per pass, lookahead",
         "confidence_la": "confidence order, lookahead"}
SMOKE, ASKS, SEEDS, KEEP, BATCH, MAX_TRIES, SEED0, GUIDANCE = (DO.SMOKE, DO.ASKS, DO.SEEDS, DO.KEEP, DO.BATCH,
                                                               DO.MAX_TRIES, DO.SEED0, DO.GUIDANCE)
OUT = DO.RES / ("lookahead_legality_smoke.json" if SMOKE else "lookahead_legality.json")
ORIGIN = "lalsmoke" if SMOKE else "lal"
MATCH = 5                                           # teams per subset for size-matched diversity, as decoding_order_diversity


def save(out):
    tmp = OUT.with_suffix(".tmp"); json.dump(out, open(tmp, "w")); os.replace(tmp, OUT)


def draw(w, ask, order, steps, seed):
    """Sample until KEEP teams pass Showdown and are not held-out teams (cap MAX_TRIES tries), with the lookahead."""
    torch.manual_seed(SEED0 + seed); rng = np.random.default_rng(SEED0 + seed)
    val, kept, rows_kept, tried, rejected, heldout, reasons, t0 = Validator(), [], [], 0, 0, 0, Counter(), time.perf_counter()
    try:
        while len(kept) < KEEP and tried < MAX_TRIES:
            for r in w.sample(BATCH, ask=ask, guidance=GUIDANCE, order=order, steps=steps, lookahead=True).cpu().numpy():
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
    now = dict(model_sha256=RA.sha(discrete_diffusion_model.__file__), script_sha256=RA.sha(__file__), decoding_order_sha256=RA.sha(DO.__file__))
    out = json.load(open(OUT)) if OUT.exists() else {"config": dict(
        arms=ARMS, asks=ASKS, seeds=SEEDS, seed0=SEED0, keep=KEEP, batch=BATCH, max_tries=MAX_TRIES, guidance=GUIDANCE,
        temperature=1.0, masks="on", repair=True, lookahead=True, battles_per_opponent=1, slots="shuffled", **now), "cells": {}}
    missing = [(s, a, arm) for s in SEEDS for a in ASKS for arm in ARMS if DO.tag_of(arm[0], a, s) not in out["cells"]]
    if not missing: return
    changed = [k for k, v in now.items() if out["config"].get(k) != v]
    if out["cells"] and changed and not os.environ.get("DO_RESUME_CHANGED"):
        raise SystemExit(f"{OUT} was started with different {changed}; finishing it would mix implementations (DO_RESUME_CHANGED=1 overrides)")
    w = DiscreteDiffusionModel.load()
    for seed, ask, (arm, order, steps, _) in missing:
        tag = DO.tag_of(arm, ask, seed); r = draw(w, ask, order, steps, seed)
        legal = (r["tried"] - r["rejected"]) / max(r["tried"], 1)
        out["cells"][tag] = dict(arm=arm, order=order, steps=steps, ask=ask, seed=seed, tried=r["tried"], rejected=r["rejected"],
                                 heldout=r["heldout"], legal=legal, kept=len(r["kept"]), capped=len(r["kept"]) < KEEP,
                                 reasons=r["reasons"], seconds=r["seconds"], div=RA.diversity(r["kept"]) if r["kept"] else None,
                                 pastes=r["kept"], rows=r["rows"])
        save(out)
        print(f"  {tag:26s} legal {legal:6.1%} ({r['tried'] - r['rejected']}/{r['tried']}) · kept {len(r['kept'])} · {r['seconds']:.0f}s", flush=True)


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
    keep = {"species"} | {a[3] for a in ARMS}
    cells = list(out["cells"].values()) + [c for c in ref.values() if c["arm"] in keep and c["seed"] in SEEDS and c["ask"] in ASKS]
    w = DiscreteDiffusionModel.load(); full, lineups, M = DO.training_sets(w); rng = np.random.default_rng(0)
    for c in cells:
        teams = [t for t in map(slots_from_paste, c["pastes"]) if t]
        c["matched"] = DD.matched(teams, MATCH, rng) if len(teams) >= MATCH else None
        c["novelty"] = DO.novelty(c.get("rows", []), full, lineups, M)
    get = lambda arm, ask: sorted((c for c in cells if c["arm"] == arm and abs(c["ask"] - ask) < 1e-9), key=lambda c: c["seed"])
    win = lambda c: c["win"]["mean"] if "win" in c else None
    pick = lambda field, k: lambda c: (c[field] or {}).get(k)
    summ = {}
    for arm in ["species"] + [x for a in ARMS for x in (a[3], a[0])]:
        cs = [c for a in ASKS for c in get(arm, a)]
        if len(cs) != len(ASKS) * len(SEEDS) or not all("win" in c or not c["pastes"] for c in cs):
            print(f"{NAMES[arm]}: incomplete"); continue
        per_seed = lambda f: [DO.mean_of([f(c) for a in ASKS for c in get(arm, a) if c["seed"] == s]) for s in SEEDS]
        slope = []
        for s in SEEDS:
            pts = [(a, win(c)) for a in DO.GRID_ASKS for c in get(arm, a) if c["seed"] == s and win(c) is not None]
            slope.append(float(np.polyfit(*zip(*pts), 1)[0]) if len(pts) >= 2 else float("nan"))
        reasons = Counter()
        for c in cs:
            for r, n in c["reasons"]: reasons[r] += n
        summ[arm] = dict(name=NAMES[arm], legal=DO.stat(per_seed(lambda c: c["legal"])), win=DO.stat(per_seed(win)),
                         ask_slope=DO.stat(slope), gap=DO.stat(per_seed(lambda c: None if win(c) is None else abs(c["ask"] - win(c)))),
                         effective_species=DO.stat(per_seed(pick("matched", "effective_species"))),
                         distinct_lineups=DO.stat(per_seed(pick("matched", "distinct_lineups"))),
                         top_species_share=DO.stat(per_seed(pick("matched", "top_species_share"))),
                         mean_distance=DO.stat(per_seed(pick("div", "mean_distance"))),
                         copies=DO.stat(per_seed(pick("novelty", "copies"))), new_lineups=DO.stat(per_seed(pick("novelty", "new_lineups"))),
                         lineup_distance=DO.stat(per_seed(pick("novelty", "lineup_distance"))),
                         kept=sum(c["kept"] for c in cs), tried=sum(c["tried"] for c in cs), capped=sum(c["capped"] for c in cs),
                         reasons=reasons.most_common(5), seconds_per_team=sum(c["seconds"] for c in cs) / max(sum(c["kept"] for c in cs), 1))
    measures = ("legal", "win", "ask_slope", "gap", "effective_species", "distinct_lineups", "mean_distance", "new_lineups", "lineup_distance")
    vs = lambda s, b: {m: dict(diff=s[m]["mean"] - b[m]["mean"], noise=DO.NOISE_K * max(b[m]["sd"], s[m]["sd"]),
                               seeds=[x - y for x, y in zip(s[m]["seeds"], b[m]["seeds"])]) for m in measures}
    for arm, _, _, base in ARMS:
        if arm not in summ: continue
        s = summ[arm]
        if base in summ: s["vs_without"] = vs(s, summ[base])
        if "species" in summ: s["vs_species"] = vs(s, summ["species"])
        v, u = s.get("vs_without", {}).get("legal"), s.get("vs_species", {}).get("win")
        s["fixes_legality"] = bool(v and v["diff"] > v["noise"] and s["legal"]["mean"] >= 0.9)
        s["matches_species_first"] = bool(u and abs(u["diff"]) <= u["noise"])
    out["summary"] = summ; save(out)
    for arm, s in summ.items():
        print(f"{s['name']:44s} legal {s['legal']['mean']:6.1%} · win {s['win']['mean']:.3f} · ask slope {s['ask_slope']['mean']:+.2f} · "
              f"eff species {s['effective_species']['mean']:.1f} · distance {s['mean_distance']['mean']:.1f} · new line-ups "
              f"{s['new_lineups']['mean']:.1%} · teams {s['kept']}" + (f" · fixes legality {s['fixes_legality']} · matches species-first "
              f"{s['matches_species_first']}" if "fixes_legality" in s else ""))


if __name__ == "__main__":
    {"gen": gen, "battle": battle, "report": report}[sys.argv[1]]()
