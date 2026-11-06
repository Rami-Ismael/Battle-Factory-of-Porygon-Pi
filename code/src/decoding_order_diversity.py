"""Size-matched diversity and novelty for the decoding-order experiment (added 2026-10-30, after sampling finished;
docs/decoding-order.md). Reads ~/vgc-data/results/decoding_order.json, writes decoding_order_diversity.json beside it.

Random-order cells kept 5 to 17 legal teams against 64 for the other orders, and distinct line-ups, distinct species,
effective species and Vendi all grow with the number of teams. So each cell's count-based measures are recomputed on
DRAWS random subsets of MATCH teams (the smallest cell's size) and averaged. Mean slot-matched distance and novelty are
averages over pairs or teams, so they use every team the cell kept. Per order: the mean over asks per seed, then mean
and seed standard deviation; against species-first order, the per-seed differences and the noise bound of the
pre-registered rule (twice the larger seed standard deviation). Reported without a rule.

    /tmp/vgc-pilot/.venv/bin/python src/decoding_order_diversity.py
"""
import json
import os

import numpy as np

import decoding_order as DO
from discrete_diffusion_model import DiscreteDiffusionModel
from remask_ask_gap import lineup, vendi
from team_diversity import slots_from_paste, summary

DRAWS = 200
COUNTED = ("distinct_lineups", "distinct_species", "effective_species", "top_species_share", "vendi")
MEASURES = (*COUNTED, "mean_distance", "copies", "new_lineups", "lineup_distance")
OUT = DO.RES / "decoding_order_diversity.json"


def matched(teams, k, rng):
    """COUNTED measures averaged over DRAWS random k-team subsets (one subset when the cell has exactly k teams)."""
    draws = 1 if len(teams) == k else DRAWS
    rows = []
    for _ in range(draws):
        sub = [teams[i] for i in rng.choice(len(teams), k, replace=False)]
        d = summary(sub, pairs_n=k)
        rows.append([d["distinct_lineups"], d["distinct_species"], d["effective_species"], d["top_species_share"],
                     vendi([lineup(t) for t in sub])])
    return dict(zip(COUNTED, np.mean(rows, 0).tolist()))


def main():
    cells = json.load(open(DO.OUT))["cells"]
    teams = {tag: [t for t in map(slots_from_paste, c["pastes"]) if t] for tag, c in cells.items()}
    k = min(len(t) for t in teams.values())
    full, lineups, M = DO.training_sets(DiscreteDiffusionModel.load())
    rng = np.random.default_rng(0); per_cell = {}
    for tag, c in cells.items():
        per_cell[tag] = dict(arm=c["arm"], ask=c["ask"], seed=c["seed"], kept=len(teams[tag]), **matched(teams[tag], k, rng),
                             mean_distance=c["div"]["mean_distance"], **DO.novelty(c["rows"], full, lineups, M))
    summ = {}
    for arm, _, _ in DO.ARMS:
        seeds = {m: [float(np.mean([v[m] for v in per_cell.values() if v["arm"] == arm and v["seed"] == s])) for s in DO.SEEDS]
                 for m in MEASURES}
        summ[arm] = {m: DO.stat(seeds[m]) for m in MEASURES}
    for arm, s in summ.items():
        if arm == "species": continue
        b = summ["species"]
        s["vs_species"] = {m: dict(diff=s[m]["mean"] - b[m]["mean"], noise=DO.NOISE_K * max(s[m]["sd"], b[m]["sd"]),
                                   seeds=[x - y for x, y in zip(s[m]["seeds"], b[m]["seeds"])]) for m in MEASURES}
    tmp = OUT.with_suffix(".tmp"); json.dump(dict(match=k, draws=DRAWS, cells=per_cell, summary=summ), open(tmp, "w"))
    os.replace(tmp, OUT)
    print(f"matched size {k} teams, {DRAWS} subsets per cell")
    for arm, s in summ.items():
        print(f"{DO.NAMES[arm]:34s} " + " · ".join(f"{m} {s[m]['mean']:.3f}±{s[m]['sd']:.3f}" for m in MEASURES))


if __name__ == "__main__":
    main()
