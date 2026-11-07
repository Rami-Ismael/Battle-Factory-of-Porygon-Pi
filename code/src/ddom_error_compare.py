"""Does the DDOM paper's bin weighting bring got closer to the ask than no weights? (GitHub issue #114; docs/ddom-reweighting-84-fields.md)

Reads the saved 23-ask grids of the three 84-field arms trained to epoch 21 with label noise: no weights, and the paper's equation 5
weights at tau 0.3 and tau 0.1. Each arm's ask-versus-got error (src/ask_line.py) is scored on asks 1/24 to 17/24 and reported
separately on asks 18/24 to 23/24, where training teams are scarce. The difference from no weights has a 95% interval from a
bootstrap over teams within each ask and seed; battles are not resampled. Beside the error: win rate at asks 18/24 to 23/24, and
line-ups (a team's six species), counted over all teams and over the top asks, with those new against the training teams and against
the training teams plus the loop and tournament records (src/lineup_reference.py).

    /tmp/vgc-pilot/.venv/bin/python src/ddom_error_compare.py [output.json]   (from code/)
"""
import json
import sys
from pathlib import Path

import numpy as np

from ask_error_compare import paired_difference
from ask_line import NEAR_LINE_MSE, SCORED_ASKS, line_measures

CONTROL = "gen84_noise_byepoch_ep21"
NAMES = {CONTROL: "No weights, 21 epochs", "gen84_ddom_tau03": "Tau 0.3 weights, 21 epochs", "gen84_ddom_tau01": "Tau 0.1 weights, 21 epochs"}
ASKS = list(range(1, 24))
SCORED, TOP = [k for k in ASKS if k in SCORED_ASKS], [k for k in ASKS if k not in SCORED_ASKS]
DRAWS, BOOT_SEED = 4000, 114
RES = Path.home() / "vgc-data/results"


def grid_cells(out):
    """ask numerator -> list of cells (one per sampling seed), each with its pastes and per-team got."""
    by_ask = {}
    for c in out["cells"].values():
        by_ask.setdefault(c["numerator"], []).append(c)
    return by_ask


def mean_got(by_ask):
    return {k: float(np.mean([y for c in by_ask[k] for y in c["y"]])) for k in ASKS}


def bootstrap(by_ask, rng, draws=DRAWS):
    """ask numerator -> array of mean gots, each draw resampling teams with replacement within each seed's cell."""
    out = {}
    for k in ASKS:
        sums, n = np.zeros(draws), 0
        for c in by_ask[k]:
            y = np.asarray(c["y"], float)
            sums += y[rng.integers(0, len(y), (draws, len(y)))].sum(1); n += len(y)
        out[k] = sums / n
    return out


def ask_error(win_by_ask, asks):
    """Mean squared gap between each ask and the mean got there, on any set of asks."""
    return float(np.mean([(win_by_ask[k] - k / 24) ** 2 for k in asks]))


def top_win(win_by_ask):
    return float(np.mean([win_by_ask[k] for k in TOP]))


def judge(grids, rng, names=NAMES, control=CONTROL):
    """grids: arm -> its by_ask cells. Error, top-ask win rate and slope per arm, each other arm against the control."""
    point = {a: mean_got(g) for a, g in grids.items()}
    boot = {a: bootstrap(g, rng) for a, g in grids.items()}
    arms = {}
    for a in grids:
        line = line_measures([dict(numerator=k, win=point[a][k]) for k in ASKS])
        top_boot = np.mean([boot[a][k] for k in TOP], 0)
        row = dict(name=names[a], teams=sum(len(c["y"]) for cs in grids[a].values() for c in cs), win=point[a], slope=line["slope"],
                   top_win=dict(win=top_win(point[a]), interval=[float(np.quantile(top_boot, q)) for q in (.025, .975)]))
        for label, asks in (("scored", SCORED), ("top", TOP)):
            e = dict(error=ask_error(point[a], asks))
            e["near_line"] = e["error"] < NEAR_LINE_MSE
            if a != control:
                e["minus_control"] = dict(difference=e["error"] - ask_error(point[control], asks), **paired_difference(boot[a], boot[control], asks))
            row[label] = e
        if a != control:
            d = top_boot - np.mean([boot[control][k] for k in TOP], 0)
            row["top_win"]["minus_control"] = dict(difference=row["top_win"]["win"] - top_win(point[control]),
                                                   interval=[float(np.quantile(d, q)) for q in (.025, .975)])
        arms[a] = row
    return arms


def line_ups(by_ask, species_of, training, records, asks=ASKS):
    """Distinct line-ups among an arm's teams at these asks, and how many of them neither reference holds."""
    seen = {species_of(p) for k in asks for c in by_ask[k] for p in c["pastes"]}
    return dict(line_ups=len(seen), new_vs_training=len(seen - training), new_vs_both=len(seen - training - records))


def main(path):
    from lineup_reference import references, species_of    # imports the 84-field model code; needs /tmp/vgc-pilot
    grids = {a: grid_cells(json.load(open(RES / f"stat_points_84_{a}.json"))) for a in NAMES}
    for a, g in grids.items():
        missing = [k for k in ASKS if len(g.get(k, [])) != 3]
        if missing: raise SystemExit(f"{a}: asks without three seeds: {missing}")
    arms = judge(grids, np.random.default_rng(BOOT_SEED))
    training, records = references()
    for a, g in grids.items():
        arms[a]["line_ups"] = dict(all_asks=line_ups(g, species_of, training, records), top_asks=line_ups(g, species_of, training, records, TOP))
    out = dict(win_rate="against the 49 distinct top-50 meta teams, behaviour-cloning policy on both sides, slots shuffled, one battle per opponent",
               interval="95%, bootstrap over teams within each ask and seed, battles not resampled", draws=DRAWS,
               scored_asks=SCORED, top_asks=TOP, reference_sizes=dict(training=len(training), loop_and_records=len(records)), arms=arms)
    json.dump(out, open(path, "w"), indent=1)

    def gap(d):
        return f"{d['difference']:+.4f} [{d['interval'][0]:+.4f}, {d['interval'][1]:+.4f}]" if d else ""
    for a, r in arms.items():
        lu, t = r["line_ups"]["all_asks"], r["line_ups"]["top_asks"]
        print(f"{r['name']:<28} error 1/24-17/24 {r['scored']['error']:.4f} {gap(r['scored'].get('minus_control'))} | "
              f"18/24-23/24 {r['top']['error']:.4f} {gap(r['top'].get('minus_control'))} | top-ask win {r['top_win']['win']:.3f} "
              f"{gap(r['top_win'].get('minus_control'))} | slope {r['slope']:.3f} | line-ups {lu['line_ups']} new {lu['new_vs_training']}/{lu['new_vs_both']} "
              f"| top asks {t['line_ups']} new {t['new_vs_training']}/{t['new_vs_both']}")
    print("DDOM_ERROR_DONE", path)


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else RES / "ddom_error_compare.json")
