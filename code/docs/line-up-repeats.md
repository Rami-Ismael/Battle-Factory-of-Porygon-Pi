# Line-up repeats, trained from scratch

Pre-registration, 2026-10-31. Script: `src/line_up_repeats.py`. Queue: `~/vgc-data/line_up_repeats/lur.zsh`, log `queue.log`. Results: `~/vgc-data/results/line_up_repeats_es.json`.

The run trained from the frozen copy in `~/vgc-data/line_up_repeats/run/`, checked by its `SHA256SUMS`. Training needs that copy: the row filter and early stopping reach training through the `EPOCH_ORDER` and `EARLY` hooks, which exist only in its `lossdown.py` and `retrain_regmb.py`. `train` exits if `lossdown` has no such hooks. The `gen`, `battle` and `report` steps import from `code/src`.

## The training data holds no duplicate teams

A duplicate team is a team with the same six Pokémon as another, in any slot order (`GLOSSARY.md`). Teams that differ only in Stat Points count as duplicates, because the model does not choose Stat Points: `to_paste` copies them from a real team with the same species.

`src/lossdata.py` merges teams with the same 48 fields and pools their battles. On 2026-10-31 a count over both training stages of both deployed models found zero duplicates, in listed order and in any slot order:

| Model | Stage 1 rows | Stage 2 rows (real + synthetic) | Duplicates |
| --- | --- | --- | --- |
| Loop-teams model (`F1_final_medium_noheldout`) | 17,129 | 21,129 | 0 |
| Battle-records model (`G2_family_matrix_noprtrain_noheldout`) | 25,865 | 29,871 | 0 |

The teams the search loop kept for training also hold zero duplicates in every generation, in all three of its runs (`cem_v3`, `cem_v3_noheldout`, `cem_fixed_elite`).

Species line-ups do repeat. The battle-records model's 23,871 real stage-2 rows hold 11,118 distinct line-ups. One line-up, Basculegion, Charizard, Floette-Eternal, Garchomp, Kingambit and Whimsicott, appears 2,377 times. Its copies differ in 21 of 48 fields on average.

## Question

Does training on one team per line-up change win rate, diversity or novelty?

## Arms

Each arm is the deployed two-member model retrained from scratch with the deployed recipe: `lossdown.train_run` with the parents' configs, then `retrain_regmb train` for the Regulation M-B v3 vocabulary (150 embedding steps). Seeds, step caps and early stopping match the baseline. Only the rows each epoch draws from differ.

| Reader name | Code ID | Rows each epoch draws from |
| --- | --- | --- |
| Every row | `control` | every training row; the species quota run's early-stopped baseline, reused from `species_quota_scratch_es.json` |
| One team per line-up | `one_per_lineup` | one randomly picked team per species line-up, the same pick every epoch |
| Random rows, same count | `random_same_count` | a random subset the size of One team per line-up's, repeats kept |

The filter applies to every training row in both stages: real, battled and synthetic coverage teams. Each epoch cycles through the kept rows until it has the baseline's number of rows. In stage 2 of the battle-records model, both new arms keep 16,950 of the 29,389 rows the training run read. Random rows, same count's training rows hold 15,761 distinct line-ups, against 17,707 for the other two arms.

Random rows, same count separates the effect of fewer rows from the effect of fewer repeats.

## Early stopping

Added 2026-11-01, matching the species quota run so its baseline stays comparable. The `lossdown.py` and `retrain_regmb.py` in the run folder are byte-identical to the species quota run's frozen copies, including the `EARLY` hook. The copies in `code/src` are not.

- **Stopping split**: one seeded random 2% of the non-corpus rows (seed 20261101), 2,926 of 146,994 rows. Real-corpus rows, which hold the top-50 meta teams, never enter it; the held-out meta teams ranked 51 to 100 and the test teams are already out. Both members, both new arms and Every row hold out the same rows, before the row filter sees the training rows.
- **Stage 1**: after every epoch, the ELBO (evidence lower bound, the diffusion model's training loss) on the split, with the same mask draws every time (`lossdown.evaluate`, reps 4, seed 12345). Training stops after 3 epochs without a new lowest, or at the parent's epoch count (80 and 60), and restores the best epoch.
- **Stage 2**: 8 checks over the 2-epoch cap, patience 3 checks. The 150 embedding warm-up steps stay fixed.
- An epoch is the baseline's number of rows, so the step caps match the baseline's; a filtered arm's epoch cycles its kept rows.
- The report checks that every arm's stopping split matches Every row's, and gives each member's best checkpoint against its cap.

## Scoring

Scoring matches the species quota run, so the baseline's cells compare directly:

- Plain masking, guidance 2, temperature 1.
- 64 Showdown-legal, non-held-out teams per cell.
- Asks 7/24, 0.3, 8/24, 0.5, 0.531, 13/24, 0.6, 0.7, with seeds 1 to 3.
- Win rate: one battle against each of the 49 distinct top-50 meta teams, slots shuffled, the behaviour-cloning policy on both sides.

## Read-out

Over asks 12/24, 0.531 and 13/24, each arm against Every row:

1. Win rate change, with a 95% interval, beside each member's stopping point.
2. Diversity: distinct line-ups, effective species, and mean slot-matched distance out of 48 (`team_diversity.summary`).
3. Novelty: the share of teams at zero, one, two, and three or more species swaps from the closest training line-up. The report measures novelty against the arm's own training line-ups and against Every row's. The two differ only for Random rows, same count, which sees fewer line-ups.

There is one training seed per arm, so the result is provisional.

This pre-registration set no decision rule: it named no result that would change what the project does next.

## Result

Recorded 2026-11-01. Win rate: one battle against each of the 49 distinct top-50 meta teams, slots shuffled, the behaviour-cloning policy on both sides. Asks 12/24, 0.531 and 13/24 pooled, 576 teams per arm.

| Arm | Win rate | Change against Every row (95% interval) | Distinct line-ups per 64 | Effective species | Zero species swaps |
| --- | --- | --- | --- | --- | --- |
| Every row | 0.409 | | 31.8 | 14.7 | 75.2% |
| One team per line-up | 0.348 | −0.061 (−0.074 to −0.047) | 63.2 | 22.8 | 26.7% |
| Random rows, same count | 0.426 | +0.017 (+0.003 to +0.031) | 27.3 | 12.3 | 85.8% |

Fewer rows alone did not cost win rate. The one-per-line-up pick did, but it changed two things at once:

- **Fewer strong teams.** Stage-1 training teams at or above the elite cut (0.531) fell from 1,223 to 169 for the battle-records model and from 920 to 167 for the loop-teams model. Random rows, same count kept 562 and 442. The 1,223 strong teams sit in only 352 line-ups, so one pick per line-up drops most of them.
- **Fewer random-fill teams.** Rows from the random-fill baseline (`random_fill_baseline`, tagged `fill` in `lossdata.py`) fell from 15.1% to 2.7% of the battle-records model's training rows.

So this run can't separate the effect of repeated line-ups from the effect of losing strong teams.

**Matched diversity** (re-analysis after the results, read-only): the −0.061 holds only at the same asks. Every row at ask 0.3 has 21.0 effective species and wins 0.344. One team per line-up at ask 0.531 has 24.1 and wins 0.335, a change of −0.009. Both asks are off the 1/24 label grid. The comparison rests on one 192-team cell per arm, three sampling seeds of 64 teams. The neighbouring ask, 0.2917, gives Every row 0.361 with 16.9 species. No Every row ask reaches 63 line-ups.

Teams three or more species swaps from any training line-up stayed under 0.4% in every arm. The filter makes one-swap variants of training line-ups, not novel line-ups.

**Lesson.** With one team per line-up, which also removed most strong teams and most random-fill teams, win rate fell 0.061 at the same asks, and about 0.009 at matched species diversity, for one trained model per arm. This run can't separate repeats from strong-team count or random-fill share. Don't record "line-up repeats cost win rate", and don't rerun these arms. A next run would cross repeats with strong-team count at equal rows, and retrain Every row once more to measure training noise.

## Reproduce the duplicate count

```sh
/tmp/vgc-pilot/.venv/bin/python -I scripts/count_duplicate_teams.py "$PWD/src"
```

Run it from `code/`. The script exits 1 if any training row is a duplicate team, in listed or any slot order.
