"""Stat Points spent (issue #264 follow-up, 2026-11-11): how many of its 66 Stat Points each Pokémon spends, in the
default model's teams against training teams, and which stat carries the gap.

Samples are those of diversity_vs_training.py and novelty_within_lineup.py, redrawn with their seeds: the default
model's 1,088 teams per sampling seed (model_s1..3, asks 1 to 17 of the 80-epoch ask grid), ask-matched training
teams (train_s1..3) and training teams over all win rates (train_all_s1..3). training_reference is every labelled
non-validation training team (170,785). Every Pokémon of every team counts, copied line-up or not.

Per sample, over its Pokémon (a Stat Point token is the value plus one, src/stat_points.py):
  share_spending_66            share whose six Stat Points sum to exactly 66, the most Showdown allows
  mean_total                   mean of the six-stat sum
  share_total_<bin>            share with a sum of 66, 60 to 65, 40 to 59 and 0 to 39
  mean_<stat>                  mean Stat Points in HP, Atk, Def, SpA, SpD, Spe
  mean_speed_room              mean of min(32, 66 minus HP..SpD): what Speed could still take when it is drawn
                               last (src/diffusion84.py ORDER draws Stat Points HP to Speed)
  share_speed_below_room       share whose Speed is below that room, so points stay unspent
  share_first_five_reach_66    share whose HP..SpD already sum to 66, leaving Speed at 0
Noise floor and verdict as in novelty_within_lineup.py: 20 pairs of disjoint ask-matched training draws, the 2.5% to
97.5% range of their differences; the model differs when all three seed differences share a sign and lie outside it.

    python scripts/stat_points_spent.py run [--out PATH]      (with /tmp/vgc-pilot/.venv/bin/python, from code/)
"""
import json
import sys
from pathlib import Path

import numpy as np

import novelty_within_lineup as NW

STATS = ("HP", "Atk", "Def", "SpA", "SpD", "Spe")
SLOT_COUNT = 6
FIRST_STAT_POINT_COLUMN = 48
STAT_POINT_TOKEN_OFFSET = 1
FULL_BUDGET = 66
MAX_PER_STAT = 32
TOTAL_BINS = (("66", 66, 66), ("60_to_65", 60, 65), ("40_to_59", 40, 59), ("0_to_39", 0, 39))
SEEDS = NW.SEEDS
OUT = NW.RESULTS / "stat_points_spent_2026-11-11.json"


def stat_point_values(rows):
    """[N, 84] rows -> [N * 6, 6] Stat Point values, one row per Pokémon, columns HP..Spe."""
    rows = np.asarray(rows, dtype=np.int64)
    return (rows[:, FIRST_STAT_POINT_COLUMN:] - STAT_POINT_TOKEN_OFFSET).reshape(-1, len(STATS))


def measures(values):
    """The module docstring's per-sample measures from [P, 6] Stat Point values."""
    values = np.asarray(values, dtype=np.int64)
    totals = values.sum(1)
    first_five = values[:, :5].sum(1)
    speed_room = np.minimum(MAX_PER_STAT, FULL_BUDGET - first_five)
    out = dict(pokemon=len(values), share_spending_66=float(np.mean(totals == FULL_BUDGET)), mean_total=float(totals.mean()))
    for name, low, high in TOTAL_BINS:
        out[f"share_total_{name}"] = float(np.mean((totals >= low) & (totals <= high)))
    out.update({f"mean_{stat}": float(values[:, k].mean()) for k, stat in enumerate(STATS)})
    out["mean_speed_room"] = float(speed_room.mean())
    out["share_speed_below_room"] = float(np.mean(values[:, 5] < speed_room))
    out["share_first_five_reach_66"] = float(np.mean(first_five >= FULL_BUDGET))
    return out


def run(out_path=OUT):
    import diversity_vs_training as D
    T, V, X, train, tgroups, vgroups, _decode_row, model, dropped = D.load()
    rng = np.random.default_rng(NW.SEED0)
    training_rows, _copied_anchor_row = NW.draw_samples(rng, train, tgroups, vgroups)
    summaries = {f"model_s{s}": measures(stat_point_values(np.stack([T.encode(V, t) for t in model[s]]))) for s in SEEDS}
    summaries.update({name: measures(stat_point_values(X[rows])) for name, rows in training_rows.items()})
    summaries["training_reference"] = measures(stat_point_values(X[train]))
    floor_differences = []
    for _ in range(NW.FLOOR_PAIRS):
        a = D.stratified(tgroups, rng); b = D.stratified(tgroups, rng, exclude=a)
        sa, sb = measures(stat_point_values(X[a])), measures(stat_point_values(X[b]))
        floor_differences.append({m: sa[m] - sb[m] for m in sa})
    model_by_seed = {s: summaries[f"model_s{s}"] for s in SEEDS}
    out = dict(source=str(D.GEN), dropped=dict(dropped), stats=STATS, summaries=summaries,
               verdict=NW.verdicts(floor_differences, model_by_seed, {s: summaries[f"train_s{s}"] for s in SEEDS}),
               verdict_all_win_rates=NW.verdicts(floor_differences, model_by_seed,
                                                 {s: summaries[f"train_all_s{s}"] for s in SEEDS}))
    Path(out_path).write_text(json.dumps(out, indent=1))
    for name, s in summaries.items():
        print(f"{name:20s} Pokémon {s['pokemon']:7d} spend 66 {s['share_spending_66']:.3f} mean total {s['mean_total']:.2f} "
              f"Speed {s['mean_Spe']:.2f} room {s['mean_speed_room']:.2f}")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args or args[0] != "run" or len(args) not in (1, 3) or (len(args) == 3 and args[1] != "--out"):
        raise SystemExit(__doc__)
    run(Path(args[2]) if len(args) == 3 else OUT)
