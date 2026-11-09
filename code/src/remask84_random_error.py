"""Does remasking with random decoding bring got closer to the ask? (GitHub issue #92; docs/remask-84-fields.md)

Reads the saved battles of remask84.py's lookahead grid: random decoding with the lookahead on, three arms, asks 1/24 to 23/24,
three seeds. Each arm's ask-versus-got error (src/ask_line.py) is scored on asks 1/24 to 17/24, against no remasking under the same
decoding, with a bootstrap over whole teams within each cell and opponents shared across arms, as remask84_error_analysis.py does.
Species-first decoding without remasking (the 23-ask grid of the same checkpoint) is reported beside it as a reference.

    python remask84_random_error.py [output.json]
"""
import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path

import numpy as np

import remask84 as R
from ask_error_compare import paired_difference
from ask_line import NEAR_LINE_MSE, SCORED_ASKS
from lineup_reference import references, species_of
from remask84_error_analysis import battle_matrices, bootstrap_levels
import matchup_db as MDB

ASKS, SCORED, APART, SEEDS = R.LOOKAHEAD_ASKS, list(SCORED_ASKS), [k for k in R.LOOKAHEAD_ASKS if k not in SCORED_ASKS], R.SEEDS
CONTROL, REFERENCE = "rand_la_plain", "species_first_plain"
NAMES = {**{k: v[1] for k, v in R.LOOKAHEAD_ARMS.items()}, REFERENCE: "Species-first decoding, no remasking, no lookahead"}
CUT = 0.531


def ask_error(win_by_ask, asks):
    """Mean over asks of (mean got - ask)^2. On asks 1/24 to 17/24 it is ask_line's error; on 18/24 to 23/24 its band reported apart."""
    return float(np.mean([(win_by_ask[k] - k / 24) ** 2 for k in asks]))


def load_cells():
    la = json.load(open(R.LOOKAHEAD_OUT))["cells"]
    sf = json.load(open(R.A.RES / "stat_points_84_gen84_all_ep21.json"))["cells"]
    cells = {(arm, k, s): la[f"{arm}_a{k:02d}of24_s{s}"] for arm in R.LOOKAHEAD_ARMS for k in ASKS for s in SEEDS}
    cells.update({(REFERENCE, k, s): sf[f"gen84_all_ep21_a{k:02d}of24_s{s}"] for k in ASKS for s in SEEDS})
    missing = [key for key, c in cells.items() if not c.get("y")]
    if missing: raise SystemExit(f"{len(missing)} cells have no battles yet, first {missing[0]}")
    return cells


def battle_policy():
    """The one policy that scored this run's teams, from the battle provenance table."""
    con = sqlite3.connect(f"file:{MDB.DEFAULT_DB}?mode=ro", uri=True)
    found = [p for (p,) in con.execute("SELECT DISTINCT policy_id FROM batch WHERE note LIKE 'score_pastes:remask84_la:%'")]
    if len(found) != 1: raise SystemExit(f"expected one battle policy for the lookahead grid, found {found}")
    return found[0]


def judge(cells, boot):
    arms = {}
    by_seed = {arm: {s: {k: float(np.mean(cells[(arm, k, s)]["y"])) for k in ASKS} for s in SEEDS} for arm in NAMES}
    point = {arm: {k: float(np.mean([by_seed[arm][s][k] for s in SEEDS])) for k in ASKS} for arm in NAMES}
    for arm in NAMES:
        tries = sum(cells[(arm, k, s)]["tried"] for k in ASKS for s in SEEDS)
        rejected = sum(cells[(arm, k, s)]["rejected"] for k in ASKS for s in SEEDS)
        row = dict(name=NAMES[arm], win=point[arm], squared_gap_by_ask={k: (point[arm][k] - k / 24) ** 2 for k in ASKS},
                   tries=tries, legal_share=(tries - rejected) / tries, teams=sum(len(cells[(arm, k, s)]["pastes"]) for k in ASKS for s in SEEDS),
                   short_cells=[f"{k}/24 seed {s}" for k in ASKS for s in SEEDS if len(cells[(arm, k, s)]["pastes"]) < R.KEEP])
        for label, asks in (("scored", SCORED), ("apart", APART)):
            e = dict(error=ask_error(point[arm], asks), by_seed=[ask_error(by_seed[arm][s], asks) for s in SEEDS])
            e["near_line"] = e["error"] < NEAR_LINE_MSE
            for ref in (CONTROL, REFERENCE):
                if arm == ref: continue
                d = dict(difference=e["error"] - ask_error(point[ref], asks), **paired_difference(boot[arm], boot[ref], asks),
                         seeds_closer=sum(x < ask_error(by_seed[ref][s], asks) for x, s in zip(e["by_seed"], SEEDS)))
                e[f"minus_{ref}"] = d
            row[label] = e
        arms[arm] = row
    return arms


def novelty(cells):
    train_set, common = references()
    control = {species_of(p) for (a, _, _), c in cells.items() if a == CONTROL for p in c["pastes"]}
    out = {}
    for arm in NAMES:
        teams = [(species_of(p), y) for (a, _, _), c in cells.items() if a == arm for p, y in zip(c["pastes"], c["y"])]
        seen, strong = {t for t, _ in teams}, {t for t, y in teams if round(y, 4) >= CUT}
        out[arm] = dict(teams=len(teams), line_ups=len(seen), new_vs_training=len(seen - train_set), new_vs_both=len(seen - train_set - common),
                        shared_with_no_remasking=len(seen & control), line_ups_at_cut_unconfirmed=len(strong), line_ups_at_cut_new_vs_both=len(strong - train_set - common),
                        effective_line_ups=float(1 / sum((v / len(teams)) ** 2 for v in Counter(t for t, _ in teams).values())),
                        accumulation=R.accumulation([c for (a, _, _), c in cells.items() if a == arm]))
    return dict(reference_sizes=dict(training=len(train_set), loop_and_records=len(common)), arms=out)


def show(r, label, ref):
    e = r[label]; d = e.get(f"minus_{ref}")
    tail = f" {d['difference']:+.4f} [{d['interval'][0]:+.4f}, {d['interval'][1]:+.4f}] seeds {d['seeds_closer']}/3" if d else ""
    return f"{e['error']:.4f}{tail}"


def main(path):
    cells = load_cells()
    M, n_opps, contract = battle_matrices(cells, policy_id=battle_policy())
    arms = judge(cells, bootstrap_levels(M, n_opps, arms=list(NAMES), asks=ASKS, seeds=SEEDS))
    out = dict(opponents=n_opps, scored_asks=SCORED, asks_apart=APART, contract=contract, arms=arms, novelty=novelty(cells))
    json.dump(out, open(path, "w"), indent=1)
    for arm, r in arms.items():
        n = out["novelty"]["arms"][arm]
        print(f"{r['name']:<52} legal {r['legal_share']:6.1%} line-ups {n['line_ups']:>4} | 1/24 to 17/24 {show(r, 'scored', CONTROL)} | 18/24 to 23/24 {show(r, 'apart', CONTROL)}")
    print(json.dumps(contract))
    print("RANDOM_ERROR_DONE", path)


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else R.A.RES / "remask84_random_error.json")
