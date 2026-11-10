"""Should the 84-field position embedding split into slot and field-type vectors? The verdict (docs/position-embedding-split.md;
GitHub issues #78, #80, #84; spec #233).

Reads the saved 23-ask grids of the five arms, all trained exactly 21 epochs with label noise ±1/48 on the shuffled-only table.
Each arm differs from No split, first seed only in how a column's position vector is built (src/diffusion84.py,
POSITION_OPTIONS), or for the second No split seed only in its training seed:

    No split, first seed     pos_per_field_s0       one learned vector per column, training seed 0 (the control)
    No split, second seed    pos_per_field_s1       the same, training seed 1
    Learned slot vectors     pos_slot_learned       learned field-type vector + learned team-position vector
    Fixed slot vectors       pos_slot_fixed         learned field-type vector + random team-position vector that never trains
    No slot vectors          pos_no_slot            learned field-type vector only

Per arm, each minus the first seed with a 95% interval: pooled win rate over the 23 asks (bootstrap over teams within each ask and
seed; battles not resampled), effective species and legal share (paired by ask and seed, normal interval over the 69 cells), and
the ask-versus-got error on asks 1/24 to 17/24 (src/ask_line.py). Training luck is the larger of the second No split seed's gap
and the gap measured earlier without label noise. The owner's rule (2026-11-07) then gives the verdict; each written expectation
is checked. The old first seed, gen84_all_noise_ep21, trained on labels from listed-order battles, is shown for reference and
decides nothing. Reported, not deciding: validation loss, the share of Pokémon whose Stat Points match a corpus set for their
species, and the share with fewer than four moves. Legal share must read 100%.

    /tmp/vgc-pilot/.venv/bin/python src/position_split_verdict.py [output.json]   (from code/)
"""
import json
import sys
from pathlib import Path

import numpy as np

from ask_error_compare import paired_difference
from ask_line import SCORED_ASKS
from ddom_error_compare import ask_error, bootstrap, grid_cells, mean_got

FIRST_SEED = "pos_per_field_s0"
SECOND_SEED = "pos_per_field_s1"
OLD_FIRST_SEED = "gen84_all_noise_ep21"                  # listed-order labels: reference only
SPLIT_ARMS = ("pos_slot_learned", "pos_slot_fixed", "pos_no_slot")
NAMES = {FIRST_SEED: "No split, first seed", SECOND_SEED: "No split, second seed", "pos_slot_learned": "Learned slot vectors",
         "pos_slot_fixed": "Fixed slot vectors", "pos_no_slot": "No slot vectors", OLD_FIRST_SEED: "Old first seed (reference)"}
# Two 21-epoch trainings without label noise, dellu_none against gen84_all_ep21, 2026-11-07
EARLIER_GAP = dict(win=0.0035, effective_species=0.07, error=0.0019)
WIN_FLOOR = -0.01                                        # a species switch needs the win-rate interval above this
NO_GAIN = "no evidence of a gain beyond training luck"      # owner, 2026-11-09: "It not a tie"
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


def luck(arms):
    """Per measure, the larger of the new No split gap and the earlier one."""
    new = seed_gap(arms)
    return {m: max(abs(new.get(m, 0.0)), g) for m, g in EARLIER_GAP.items()}


def judge_arm(row, lk):
    """The rule's tests for one split arm against No split, first seed."""
    w, sp, e = (row[m]["minus_first_seed"] for m in ("win", "effective_species", "error"))
    on_win = w["interval"][0] > 0 and w["difference"] > lk["win"] and sp["difference"] >= -lk["effective_species"]
    on_species = (w["interval"][0] > WIN_FLOOR and sp["interval"][0] > 0 and sp["difference"] > lk["effective_species"])
    guard = e["difference"] <= lk["error"]
    matters = any((d["interval"][0] > 0 or d["interval"][1] < 0) and abs(d["difference"]) > lk[m]
                  for m, d in (("win", w), ("effective_species", sp)))
    return dict(switch_on_win_rate=bool(on_win), switch_on_species=bool(on_species), guard=bool(guard),
                passes=bool((on_win or on_species) and guard), it_matters=bool(matters))


def verdict(arms):
    """The owner's rule (2026-11-07): switch on win rate or on species, guarded by ask-versus-got error, a second seed before a
    switch counts, Fixed slot vectors on a near tie between passing arms, and otherwise no evidence of a gain beyond training luck (owner, 2026-11-09: "not a tie")."""
    lk = luck(arms)
    rows = {a: judge_arm(arms[a], lk) for a in SPLIT_ARMS if a in arms}
    passing = [a for a, r in rows.items() if r["passes"]]
    out = dict(luck=lk, arms=rows, choice=None, next="keep one learned vector per field")
    if not passing:
        return dict(out, outcome=NO_GAIN)
    if len(passing) > 1:
        close = all(abs(arms[a][m]["value"] - arms[b][m]["value"]) <= lk[m]
                    for a in passing for b in passing for m in ("win", "effective_species"))
        if not (close and "pos_slot_fixed" in passing):
            return dict(out, outcome="several pass, not within luck: the owner decides", next="the owner picks among " + ", ".join(passing))
        passing = ["pos_slot_fixed"]
    return dict(out, outcome="provisional switch", choice=passing[0],
                next=f"train {NAMES[passing[0]]} again with seed 1 and apply the rule again; the switch counts only if that second seed passes too")


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


# Written on #80 on 2026-11-01, before any run: arm -> measure -> (wording, test), and the stated chance of being right. They
# now apply against the new No split, first seed; the owner dropped the legal-share column on 2026-11-07.
EXPECTED = {
    SECOND_SEED: (dict(win=("within 0.02", _within(0.02)),
                       effective_species=("no clear change", _no_clear_change)), 0.80),
    "pos_slot_learned": (dict(win=("within 0.01, a tie", _within(0.01)),
                              effective_species=("no clear change", _no_clear_change)), 0.75),
    "pos_slot_fixed": (dict(win=("ties learned slot vectors", _ties_learned),
                            effective_species=("no clear change", _no_clear_change)), 0.70),
    "pos_no_slot": (dict(win=("loses at least 0.05", _loses_at_least(0.05)),
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
    return dict(validation_loss=training.get("best_val_loss"), validation_epoch=training.get("best_epoch", training.get("best_pass")),
                stopped_epoch=training.get("stopped_epoch", training.get("stopped_pass")), training_seed=training.get("seed"),
                position_option=training.get("net_kw", {}).get("pos", "per_field"),
                share_matching_corpus_set=sp.get("share_matching_corpus_set"), share_under_four_moves=sp.get("share_under_four_moves"))


def _gap(d):
    return f"{d['difference']:+.4f} [{d['interval'][0]:+.4f}, {d['interval'][1]:+.4f}]" if d else "(first seed)"


def main(path):
    import asked_vs_got as X                              # training records; imports the model code, needs /tmp/vgc-pilot
    outs = {a: json.load(open(RES / f"stat_points_84_{a}.json")) for a in NAMES if a != OLD_FIRST_SEED}
    grids = {a: grid_cells(o) for a, o in outs.items()}
    for a, g in grids.items():
        missing = [k for k in ASKS if len(g.get(k, [])) != 3]
        if missing: raise SystemExit(f"{a}: asks without three seeds: {missing}")
    arms = compare(grids, np.random.default_rng(BOOT_SEED))
    old_path = RES / f"stat_points_84_{OLD_FIRST_SEED}.json"
    if old_path.exists():                                 # reference only: never enters luck, the rule or the expectations
        outs[OLD_FIRST_SEED] = json.load(open(old_path))
        arms[OLD_FIRST_SEED] = compare({FIRST_SEED: grids[FIRST_SEED], OLD_FIRST_SEED: grid_cells(outs[OLD_FIRST_SEED])},
                                       np.random.default_rng(BOOT_SEED))[OLD_FIRST_SEED]
    for a, o in outs.items():
        arms[a]["reported"] = reported(o, json.load(open(X.RES / f"{Path(o['checkpoint']).stem}.json")))
    out = dict(win_rate=WIN_RATE, interval="95%; win rate and error: bootstrap over teams within each ask and seed, battles not resampled; "
               "effective species and legal share: normal, over cells paired by ask and seed", draws=DRAWS, scored_asks=SCORED,
               arms=arms, seed_gap=seed_gap(arms), expectations=expectations(arms),
               verdict=verdict(arms))
    json.dump(out, open(path, "w"), indent=1, default=str)
    for a, r in arms.items():
        print(f"{r['name']:<24} win {r['win']['value']:.4f} {_gap(r['win'].get('minus_first_seed'))} | species "
              f"{r['effective_species']['value']:.2f} {_gap(r['effective_species'].get('minus_first_seed'))} | error "
              f"{r['error']['value']:.4f} {_gap(r['error'].get('minus_first_seed'))} | legal {r['legal']['value']:.1%}")
    print("seed gap:", out["seed_gap"], "| luck:", out["verdict"]["luck"])
    not_legal = [a for a, r in arms.items() if r["legal"]["value"] < 1.0]
    if not_legal: print("CHECK FAILED: legal share under 100% for", ", ".join(not_legal))
    v = out["verdict"]
    print(f"verdict: {v['outcome']}" + (f" ({NAMES[v['choice']]})" if v["choice"] else "") + f"; next: {v['next']}")
    for a, e in out["expectations"].items():
        print(f"{e['name']:<24}", " | ".join(f"{m} {v['expected']}: {'right' if v['right'] else 'wrong' if v['right'] is False else 'not yet'}"
                                          for m, v in e.items() if isinstance(v, dict)))
    print("POSITION_SPLIT_DONE", path)


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else RES / "position_split_verdict.json")
