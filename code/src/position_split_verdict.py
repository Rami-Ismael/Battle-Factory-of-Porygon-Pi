"""Should the 84-field position embedding split into slot and field-type vectors? The measures for the verdict (GitHub issues #78, #80, #84;
spec #233).

Reads the saved 23-ask grids of the first seed and the four new arms, all trained exactly 21 epochs with label noise ±1/48 on all
teams. Each arm differs from the first seed only in how a column's position vector is built (src/diffusion84.py, POSITION_OPTIONS),
or for the second No split seed only in its training seed:

    No split, first seed     gen84_all_noise_ep21   one learned vector per column (already battled)
    No split, second seed    pos_per_field_s1       the same, training seed 1
    Learned slot vectors     pos_slot_learned       learned field-type vector + learned team-position vector
    Fixed slot vectors       pos_slot_fixed         learned field-type vector + random team-position vector that never trains
    No slot vectors          pos_no_slot            learned field-type vector only

Per arm, each minus the first seed with a 95% interval: pooled win rate over the 23 asks (bootstrap over teams within each ask and
seed; battles not resampled), effective species and legal share (paired by ask and seed, normal interval over the 69 cells), and
the ask-versus-got error on asks 1/24 to 17/24 (src/ask_line.py). The second No split seed's differences are the seed gap, the size of
training luck. Each written expectation from #80 is checked. Reported, not deciding: validation loss, the share of Pokémon whose
Stat Points match a corpus set for their species, and the share with fewer than four moves.

The verdict rule itself is not here: #80's open questions Q2 to Q6 decide it. This report computes every measure they might use.

    /tmp/vgc-pilot/.venv/bin/python src/position_split_verdict.py [output.json]   (from code/)
"""
import json
import sys
from pathlib import Path

import numpy as np

from ask_error_compare import paired_difference
from ask_line import SCORED_ASKS
from ddom_error_compare import ask_error, bootstrap, grid_cells, mean_got

FIRST_SEED = "gen84_all_noise_ep21"
SECOND_SEED = "pos_per_field_s1"
NAMES = {FIRST_SEED: "No split, first seed", SECOND_SEED: "No split, second seed", "pos_slot_learned": "Learned slot vectors",
         "pos_slot_fixed": "Fixed slot vectors", "pos_no_slot": "No slot vectors"}
ASKS = list(range(1, 24))
SCORED = [k for k in ASKS if k in SCORED_ASKS]
MEASURES = ("win", "effective_species", "error", "legal")
DRAWS, BOOT_SEED = 4000, 233
RES = Path.home() / "vgc-data/results"
WIN_RATE = "against the 49 distinct top-50 meta teams, behaviour-cloning policy on both sides, slots shuffled, one battle per opponent"


def pooled(boot, by_ask):
    """Pooled mean got over every team at every ask, from per-ask means weighted by their team counts."""
    n = {k: sum(len(c["y"]) for c in by_ask[k]) for k in ASKS}
    return sum(boot[k] * n[k] for k in ASKS) / sum(n.values())


def cell_values(by_ask, measure):
    """(ask, seed) -> the cell's value for a per-cell measure."""
    get = (lambda c: c["div"]["effective_species"]) if measure == "effective_species" else (lambda c: c["legal"])
    return {(c["numerator"], c["seed"]): float(get(c)) for k in ASKS for c in by_ask[k]}


def paired_cells(arm, ref):
    """Mean of arm minus reference over the cells both have, with a normal 95% interval."""
    d = np.array([arm[key] - ref[key] for key in arm if key in ref])
    half = 1.96 * d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float("nan")
    return dict(difference=float(d.mean()), interval=[float(d.mean() - half), float(d.mean() + half)], cells=len(d))


def compare(grids, rng, first=FIRST_SEED):
    """grids: arm -> its cells by ask. Each measure per arm, and each other arm minus the first seed."""
    point = {a: mean_got(g) for a, g in grids.items()}
    boot = {a: bootstrap(g, rng, DRAWS) for a, g in grids.items()}
    win = {a: pooled(boot[a], g) for a, g in grids.items()}
    arms = {}
    for a, g in grids.items():
        row = dict(name=NAMES.get(a, a), teams=sum(len(c["y"]) for cs in g.values() for c in cs),
                   win=dict(value=float(pooled(point[a], g))),
                   error=dict(value=ask_error(point[a], SCORED)),
                   **{m: dict(value=float(np.mean(list(cell_values(g, m).values())))) for m in ("effective_species", "legal")})
        if a != first:
            d = win[a] - win[first]
            row["win"]["minus_first_seed"] = dict(difference=row["win"]["value"] - float(pooled(point[first], grids[first])),
                                                  interval=[float(np.quantile(d, q)) for q in (.025, .975)])
            row["error"]["minus_first_seed"] = dict(difference=row["error"]["value"] - ask_error(point[first], SCORED),
                                                    **paired_difference(boot[a], boot[first], SCORED))
            for m in ("effective_species", "legal"):
                row[m]["minus_first_seed"] = paired_cells(cell_values(g, m), cell_values(grids[first], m))
        arms[a] = row
    return arms


def seed_gap(arms):
    """The second No split seed minus the first on each measure: the size of training luck. Empty before it is battled."""
    if SECOND_SEED not in arms: return {}
    return {m: arms[SECOND_SEED][m]["minus_first_seed"]["difference"] for m in MEASURES}


def _diff(arms, a, m):
    return arms[a][m]["minus_first_seed"]


def _within(width):
    return lambda arms, a, m: abs(_diff(arms, a, m)["difference"]) <= width


def _no_clear_change(arms, a, m):
    lo, hi = _diff(arms, a, m)["interval"]
    return lo <= 0 <= hi


def _same_or_higher(arms, a, m):
    return _diff(arms, a, m)["interval"][1] >= 0


def _loses_at_least(width):
    return lambda arms, a, m: _diff(arms, a, m)["difference"] <= -width


def _ties_learned(arms, a, m):
    if "pos_slot_learned" not in arms: return None
    return abs(_diff(arms, a, m)["difference"] - _diff(arms, "pos_slot_learned", m)["difference"]) <= 0.01


# Written on #80 on 2026-11-01, before any run: arm -> measure -> (wording, test), and the stated chance of being right.
# Legal share is 100% for every arm under the rule masks; #80's Q2 may drop its column.
EXPECTED = {
    SECOND_SEED: (dict(win=("within 0.02", _within(0.02)), legal=("within 2 points", _within(0.02)),
                       effective_species=("no clear change", _no_clear_change)), 0.80),
    "pos_slot_learned": (dict(win=("within 0.01, a tie", _within(0.01)), legal=("within 2 points", _within(0.02)),
                              effective_species=("no clear change", _no_clear_change)), 0.75),
    "pos_slot_fixed": (dict(win=("ties learned slot vectors", _ties_learned), legal=("within 2 points", _within(0.02)),
                            effective_species=("no clear change", _no_clear_change)), 0.70),
    "pos_no_slot": (dict(win=("loses at least 0.05", _loses_at_least(0.05)), legal=("within 2 points", _within(0.02)),
                         effective_species=("same or higher", _same_or_higher)), 0.85),
}


def expectations(arms):
    """Each battled arm's written expectations, with whether each came true (None when it needs an arm not yet battled)."""
    out = {}
    for a, (by_measure, chance) in EXPECTED.items():
        if a not in arms: continue
        out[a] = dict(name=NAMES[a], chance_right=chance,
                      **{m: dict(expected=wording, right=test(arms, a, m)) for m, (wording, test) in by_measure.items()})
    return out


def reported(grid_out, training):
    """Reported, not deciding: from an arm's grid results and its training record."""
    sp = grid_out["summary"].get("stat_points", {})
    return dict(validation_loss=training.get("best_val_loss"), validation_epoch=training.get("best_pass"),
                stopped_epoch=training.get("stopped_pass"), training_seed=training.get("seed"),
                position_option=training.get("net_kw", {}).get("pos", "per_field"),
                share_matching_corpus_set=sp.get("share_matching_corpus_set"), share_under_four_moves=sp.get("share_under_four_moves"))


def _gap(d):
    return f"{d['difference']:+.4f} [{d['interval'][0]:+.4f}, {d['interval'][1]:+.4f}]" if d else "(first seed)"


def main(path):
    import asked_vs_got as X                              # training records; imports the model code, needs /tmp/vgc-pilot
    outs = {a: json.load(open(RES / f"stat_points_84_{a}.json")) for a in NAMES}
    grids = {a: grid_cells(o) for a, o in outs.items()}
    for a, g in grids.items():
        missing = [k for k in ASKS if len(g.get(k, [])) != 3]
        if missing: raise SystemExit(f"{a}: asks without three seeds: {missing}")
    arms = compare(grids, np.random.default_rng(BOOT_SEED))
    for a, o in outs.items():
        arms[a]["reported"] = reported(o, json.load(open(X.RES / f"{Path(o['checkpoint']).stem}.json")))
    out = dict(win_rate=WIN_RATE, interval="95%; win rate and error: bootstrap over teams within each ask and seed, battles not resampled; "
               "effective species and legal share: normal, over cells paired by ask and seed", draws=DRAWS, scored_asks=SCORED,
               arms=arms, seed_gap=seed_gap(arms), expectations=expectations(arms),
               verdict="not applied: the rule waits on #80's open questions Q2 to Q6")
    json.dump(out, open(path, "w"), indent=1, default=str)
    for a, r in arms.items():
        print(f"{r['name']:<24} win {r['win']['value']:.4f} {_gap(r['win'].get('minus_first_seed'))} | species "
              f"{r['effective_species']['value']:.2f} {_gap(r['effective_species'].get('minus_first_seed'))} | error "
              f"{r['error']['value']:.4f} {_gap(r['error'].get('minus_first_seed'))} | legal {r['legal']['value']:.1%}")
    print("seed gap:", out["seed_gap"])
    for a, e in out["expectations"].items():
        print(f"{e['name']:<24}", " | ".join(f"{m} {v['expected']}: {'right' if v['right'] else 'wrong' if v['right'] is False else 'not yet'}"
                                          for m, v in e.items() if isinstance(v, dict)))
    print("POSITION_SPLIT_DONE", path)


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else RES / "position_split_verdict.json")
