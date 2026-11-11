"""Guidance sweep on the subset-size models (docs/subset-ceiling-tests.md, issue #284).

Every subset-size model stops near a 0.5 win rate against the training opponents whatever the ask. This draws teams at
guidance 4 and 8 for asks 12/24, 16/24, 20/24 and 23/24, with the same sampler and seeds as the stored guidance 2 grid.

    AG84_MODEL=gen65_subset_39 python src/guidance_sweep65.py gen | battle | split
GS65_SMOKE=1 runs one cell with 4 teams into results/guidance_sweep_smoke.json.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import train65 as T65  # puts /tmp/vgc-pilot/src on the path
import lossdata65 as D
import ask_grid84 as AG

SMOKE = bool(os.environ.get("GS65_SMOKE"))
GUIDANCES = [8.0] if SMOKE else [4.0, 8.0]
ASKS = [20] if SMOKE else [12, 16, 20, 23]
SEEDS = [1] if SMOKE else [1, 2, 3]
OUT = T65.RES / ("guidance_sweep_smoke.json" if SMOKE else f"guidance_sweep_{AG.MODEL}.json")


def load_out():
    return json.load(open(OUT)) if OUT.exists() else dict(model=AG.MODEL, reader=AG.MODELS[AG.MODEL]["reader"], cells={})


def save(out):
    tmp = OUT.with_suffix(".tmp")
    json.dump(out, open(tmp, "w"), indent=1)
    tmp.replace(OUT)


def gen():
    out = load_out()
    sha = AG.require_same_checkpoint(out)
    out.update(checkpoint=AG.checkpoint_path().name, checkpoint_sha256=sha, keep=AG.KEEP, max_tries=AG.MAX_TRIES)
    if SMOKE:
        AG.KEEP, AG.BATCH, AG.MAX_TRIES = 4, 8, 64
    w = AG.load_model()
    for g in GUIDANCES:
        AG.GUIDANCE = g
        for seed in SEEDS:
            for k in ASKS:
                t = f"g{g:g}_k{k}_s{seed}"
                if t in out["cells"]:
                    continue
                kept, stats, counts = AG.draw(w, k / 24, seed)
                out["cells"][t] = dict(guidance=g, numerator=k, ask=k / 24, seed=seed, **counts, pastes=kept, stat_points=stats,
                                       div=AG.subset_diversity(kept, AG.SEED0 + seed) if kept else None)
                save(out)
                print(f"  {t:14s} kept {len(kept):3d} legal {counts['legal']:6.1%}", flush=True)


def battle():
    import matchup_db as MDB

    AG.require_shuffled()
    con = MDB.connect()
    out = load_out()
    for t, c in out["cells"].items():
        if "y" in c or not c["pastes"]:
            continue
        sc = MDB.score_pastes(con, c["pastes"], per_cell=AG.BATTLES_PER_OPPONENT, seed=AG.SEED0 + c["seed"], origin=f"guidance65:{AG.MODEL}:{t}")
        if not AG.complete(sc):
            print(f"  {t} INCOMPLETE: left for the next run", flush=True)
            continue
        AG.record_scores(c, sc)
        save(out)
        print(f"  {t:14s} win {np.mean(c['y']):.3f}", flush=True)


def split():
    """Got against the training opponents per guidance and ask, with guidance 2 from the stored grid, and the share of
    teams each guidance shares with guidance 2 at the same ask and seed."""
    import sqlite3

    out = load_out()
    con = sqlite3.connect(f"file:{D.DB}?mode=ro", uri=True)
    sides, policy = T65.sides_and_policy(con)
    base = json.load(open(T65.RES / f"stat_points_84_{AG.MODEL}.json"))["cells"]
    cells = [dict(c, guidance=2.0) for c in base.values() if round(c["ask"] * 24) in ASKS] + list(out["cells"].values())
    ids2 = {(round(c["ask"] * 24), c["seed"]): set(c["team_ids"]) for c in base.values()}
    rows = {}
    for c in cells:
        k = round(c["ask"] * 24)
        r = rows.setdefault(f"{c['guidance']:g}", {}).setdefault(str(k), dict(got=[], tried=0, rejected=0, shared=[], lineups=[]))
        ids = T65.team_ids(con, c)
        r["got"] += [T65.got_by_side(con, policy, sides, t)["training"] for t in ids]
        r["tried"] += c["tried"]
        r["rejected"] += c["rejected"]
        r["shared"].append(len(set(ids) & ids2[(k, c["seed"])]) / max(len(ids), 1))
        r["lineups"].append((c.get("div") or {}).get("distinct_lineups"))
    table = {g: {k: dict(teams=len(r["got"]), got=float(np.mean(r["got"])), se=float(np.std(r["got"], ddof=1) / np.sqrt(len(r["got"]))),
                         legal=(r["tried"] - r["rejected"]) / max(r["tried"], 1), shared_with_guidance_2=float(np.mean(r["shared"])),
                         distinct_lineups=float(np.mean([x for x in r["lineups"] if x is not None])))
                 for k, r in ks.items()} for g, ks in rows.items()}
    best = {g: max(v["got"] for v in ks.values()) for g, ks in table.items()}
    rise = best.get("8", float("nan")) - best["2"]
    verdict = "raises the stop" if rise >= 0.05 else "no change" if abs(rise) <= 0.02 else "undecided"
    out["result"] = dict(table=table, best_got=best, rise_at_guidance_8=rise, verdict=verdict)
    save(out)
    print(json.dumps(dict(best_got=best, rise_at_guidance_8=rise, verdict=verdict), indent=1))


if __name__ == "__main__":
    {"gen": gen, "battle": battle, "split": split}[sys.argv[1]]()
