"""Do Modal battles score teams the way laptop battles do? Re-battle teams a stored ask grid already battled on the
laptop, on Modal, in a private database, and pair each team's two win rates.

Reads stat_points_84_GRID.json, takes the first N teams of seed 1 at each named ask, battles them once against each
top-50 meta team (as the grid did) through MATCHUP_BACKEND=modal, and prints the paired difference (Modal minus
laptop) with a 95% bootstrap interval over teams. Battles are not reproducible across runs on either machine
(policy seeds do not pin simulator trajectories), so the check passes when the interval holds zero and stays within
the margin; the battle-to-battle noise alone makes a per-team difference.

    MATCHUP_BACKEND=modal MATCHUP_DB=PRIVATE.sqlite python modal_battle_check.py linear_masked_filter 6,12,17 64
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, "/tmp/vgc-pilot/src")
sys.path.insert(0, str(HERE))
import matchup_db as MDB

RESULTS = Path.home() / "vgc-data/results"
MARGIN = 0.02            # a mean win-rate shift this large would move a grid's error past the 0.002 luck bar


def main(grid, asks, per_ask):
    if os.environ.get("MATCHUP_BACKEND") != "modal": raise SystemExit("set MATCHUP_BACKEND=modal")
    if MDB.DEFAULT_DB.name == "matchup_regmb.sqlite": raise SystemExit("set MATCHUP_DB to a private database")
    out = json.loads((RESULTS / f"stat_points_84_{grid}.json").read_text())
    cells = {(int(c["numerator"]), int(c["seed"])): c for c in out["cells"].values()}
    con = MDB.connect()
    if not con.execute("SELECT 1 FROM team_set WHERE set_name='top50' LIMIT 1").fetchone():
        print("seeding the private database's top-50 set", flush=True); MDB.seed(con, top50=Path("/tmp/vgc-pilot/top50_evs.json"))
    laptop, modal, started = [], [], time.perf_counter()
    for ask in asks:
        cell = cells[(ask, 1)]
        pastes, y = cell["pastes"][:per_ask], cell["y"][:per_ask]
        scores = MDB.score_pastes(con, pastes, per_cell=1, seed=int(out["seed0"]) + 1, origin=f"modal_check:{grid}:a{ask}")
        for s, old in zip(scores, y):
            if s.get("score") is None or old is None: continue
            laptop.append(old); modal.append(s["score"])
        print(f"  ask {ask}/24: {len(pastes)} teams, laptop {np.mean(y):.3f}, modal {np.mean([s['score'] for s in scores]):.3f}", flush=True)
    laptop, modal = np.array(laptop), np.array(modal)
    diff = modal - laptop
    rng = np.random.default_rng(431)
    draws = np.array([diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(4000)])
    lo, hi = np.percentile(draws, [2.5, 97.5])
    runner = MDB.modal_runner()
    report = dict(grid=grid, asks=asks, teams=len(diff), laptop=float(laptop.mean()), modal=float(modal.mean()),
                  difference=float(diff.mean()), interval=[float(lo), float(hi)], margin=MARGIN,
                  passed=bool(lo <= 0 <= hi and -MARGIN < lo and hi < MARGIN), modal_battles=runner.battles,
                  modal_seconds=round(runner.seconds, 1), wall_seconds=round(time.perf_counter() - started, 1))
    print(json.dumps(report, indent=1))
    print("MODAL_BATTLE_CHECK_DONE", "PASS" if report["passed"] else "FAIL", flush=True)
    return report


if __name__ == "__main__":
    main(sys.argv[1], [int(a) for a in sys.argv[2].split(",")], int(sys.argv[3]))
