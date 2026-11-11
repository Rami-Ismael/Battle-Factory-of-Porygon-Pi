"""Strong training teams: do high-ask teams resemble them? (docs/subset-ceiling-tests.md, issue #285). No battles.

Strong teams are the training rows whose all-39 label is 0.7 or more. A same-size uniform draw of training rows is the
reference. For every team a subset-size grid drew at asks 18/24 to 23/24, measure the most slots it shares with any
team in each set (a slot = species, ability, item and moves) and whether its species line-up appears in each set.

    AG84_MODEL=gen65_subset_39 python src/strong_teams65.py
"""

import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import train65 as T65  # puts /tmp/vgc-pilot/src on the path
import ask_grid84 as AG
import team84 as T
from team_diversity import slots_from_paste

STRONG, SEED, BOOT = 0.7, 8500, 2000
LO, HI = map(int, os.environ.get("ST65_ASKS", "18-23").split("-"))   # ST65_ASKS=1-6: the low-ask probe
ASKS = range(LO, HI + 1)
OUT = T65.RES / (f"strong_teams_{AG.MODEL}.json" if (LO, HI) == (18, 23) else f"strong_teams_{AG.MODEL}_asks{LO}_{HI}.json")


def slot_key(s):
    return (T.norm(s["species"]), T.norm(s["ability"]), T.norm(s["item"]), tuple(sorted(T.norm(m) for m in s["moves"] if m)))


def team_keys(slots):
    return [slot_key(s) for s in slots]


def nearest(teams, ref):
    """Most slots each team shares with any team in ref, through an index from slot to ref teams."""
    index = {}
    for j, t in enumerate(ref):
        for k in set(t):
            index.setdefault(k, []).append(j)
    best = []
    for t in teams:
        hits = {}
        for k in set(t):
            for j in index.get(k, ()):
                hits[j] = hits.get(j, 0) + 1
        best.append(max(hits.values(), default=0))
    return np.array(best)


def main():
    d, tr, _, _ = T65.load()
    C = d["C"][tr]
    full = T65.full_label(C)
    X = d["X"][tr]
    V = AG.load_model().V
    strong_rows = np.flatnonzero(full >= STRONG)
    rng = np.random.default_rng(SEED)
    random_rows = rng.choice(len(full), len(strong_rows), replace=False)
    ref = {name: [team_keys(T.decode(V, X[i])) for i in rows] for name, rows in (("strong", strong_rows), ("random", random_rows))}
    lineups = {name: {tuple(sorted(k[0] for k in t)) for t in ts} for name, ts in ref.items()}
    grid = json.load(open(T65.RES / f"stat_points_84_{AG.MODEL}.json"))["cells"]
    gen = [team_keys(s) for c in grid.values() if round(c["ask"] * 24) in ASKS for s in map(slots_from_paste, c["pastes"]) if s]
    near = {name: nearest(gen, ts) for name, ts in ref.items()}
    diff = near["strong"] - near["random"]
    boot = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(BOOT)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    verdict = "resembles strong teams" if lo > 0.25 else "no nearer strong teams" if -0.25 < lo and hi < 0.25 else "undecided"
    out = dict(model=AG.MODEL, strong_threshold=STRONG, strong_teams=len(strong_rows), random_teams=len(random_rows), asks=f"{LO}/24 to {HI}/24",
               generated_teams=len(gen),
               nearest_shared_slots={n: float(v.mean()) for n, v in near.items()},
               share_exact_copy={n: float((v == 6).mean()) for n, v in near.items()},
               share_lineup_in_set={n: float(np.mean([tuple(sorted(k[0] for k in t)) in lineups[n] for t in gen])) for n in ref},
               strong_minus_random=dict(mean=float(diff.mean()), lo=float(lo), hi=float(hi)), verdict=verdict)
    json.dump(out, open(OUT, "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
