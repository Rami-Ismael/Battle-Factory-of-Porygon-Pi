# Calibrated asks: does mislabelling explain why 1 to 39 opponent labels lost?

Pre-registration, 2026-11-11, before any calibrated team was battled. Follows [Subset size for training labels](subset-size-labels.md), whose two subset arms both moved got further from the ask than the all-39 control (issue #118, verdict on #255). Mode: **exploratory**: one training seed, and the calibration map was chosen after the original grids were seen.

## Why this run

A label drawn from a few opponents reads high more often because an ordinary team got lucky on its subset than because the team is strong. So the teams the 1 to 39 arm learned for a high label are weaker than the label says. The training data shows it without a model: among training teams whose 1 to 39 label sat at 17/24, the all-39 win rate averages 0.11 below it, against 0.01 for the control (2026-11-10 post, `label_shortfall.py`).

If that is the whole cause, asking the 1 to 39 model for the label whose teams *really* win at the target rate should bring got back to the control's. Under 1 to 39 labels no label value's teams average 17/24 or more (the highest is 0.708), so targets 17/24 and up have no calibrated label.

## Arms

| Reader name | Code ID | Difference |
| --- | --- | --- |
| 1 to 39 opponents, calibrated asks | `calibrated_1_39` | The 1 to 39 model, asked for the calibrated label of each target |
| 1 to 39 opponents per label | `subset_1_39` | The same model asked for the target itself (stored grid, no new battles) |
| All 39 opponents per label | `subset_39` | The control (stored grid, no new battles) |

## Contract

| Row | Value |
| --- | --- |
| Claim | Asking the 1 to 39 model for calibrated labels brings its ask-versus-got error on targets 3/24 to 16/24 within 0.001 of the control's. **Explains** if calibrated error is within 0.001 of the control's (0.00825 on these targets). **Partly** if it falls by 0.001 or more from the original 0.01398 but stays above. **Refuted** if it falls by less than 0.001. If it explains, the subset labels lost to mislabelling, and a calibrated label (or a label correction in training) becomes the next test before the prompt models use subsets. If refuted, suspects 2 and 3 of the post run next. Scope: win rate under the behaviour-cloning policy against the 39 training opponents. |
| Fields | All 84 generated, as in the original grid. Stat Alignment repaired after sampling (`ask_grid84.py`). |
| Vocabulary | The Regulation M-B tables in checkpoint `stat65_subset_1_39`, unchanged. |
| Training teams | None trained. The calibration reads the 265,088 training rows of `lossdata65.npz` (one-swap exclusion), with the 80 epochs of subset draws the arm trained on (seeds 6500 to 6579). The 463 rows the tables dropped in training are included; they are 0.17% of rows. |
| Calibration | For each label value on a 1/192 grid, the mean all-39 label of rows whose subset label rounds there, smoothed over ±2/192, kept where at least 200 row-epochs fall. The calibrated label for target k/24 is the smallest label whose running maximum reaches k/24. Targets 1/24 and 2/24 map to label 0, so they are not judged. 16/24 rests on 16 rows per epoch. Map stored under `calibration` in the results file. |
| Scorer | Unchanged from the original grid: `ask_grid84.draw` (guidance 2, temperature 1, 64 legal teams, at most 1,280 tries, sampling seed 7300 + seed), one battle against each of the 49 meta teams, slots shuffled, battle seed 7300 + seed. Got is split to the 39 training opponents with `train65.got_by_side`. |
| Search region | No search loop. Targets 3/24 to 16/24, seeds 1 to 3. |
| Settings | **Scientific:** the ask given to the model (target or calibrated label). **Nuisance:** none varied. **Fixed:** everything else, including the checkpoint, from the frozen copy of the original run code. |
| Plan | **Primary:** error on targets 3/24 to 16/24 against the control's and the original arm's on the same targets. **Beside:** got minus target per ask, legality, distinct line-ups. **Smallest effect worth acting on:** 0.001. **Stopping rule:** one grid, no re-runs. |

## Probes

| Probe | Result | Pass |
| --- | --- | --- |
| Calibration | Map computed; labels rise with the target (0.099 at 3/24 to 0.927 at 16/24); 17/24 and up unreachable | yes |
| Training smoke, round trip, coverage, read teams, copies, separation | Same checkpoint as the original arm, whose preflight passed (`preflight_gen65_subset_1_39.log`) | carried |
| Smoke | 4 teams at target 8/24 (label 0.333): 100% legal, battled in a private copy of the preflight database, win 0.265 against all 49 | yes |
| Split report | Run on the original arm's stored grid, the split reproduces its error on these targets exactly, 0.013978 | yes |
| Condition | The calibrated labels differ from the targets by up to +0.26, and the original grid showed got rising with the ask | yes |
| Invariance, repeat | Skipped: the scorer is unchanged and every battle shuffles slots | n/a |

## Noise floor and budget

Each target holds 192 teams; got against 39 opponents has a standard error of about 0.008 per target, which adds about 0.0001 to the error, and the error's sampling SD over 14 targets is about 0.0003. The gap to close is 0.0057, and the bar 0.001 is about three SDs, so the budget can detect it. Training luck cannot be separated: one seed.

Cost: 14 targets × 3 seeds × 64 teams = 2,688 teams, 131,712 battles, about 30 minutes of battles plus generation, on the laptop. Modal: none.

## Commands

```
python src/calibrated_asks65.py calibrate ; ... gen ; battle_supervisor.py src/calibrated_asks65.py battle ; ... split
```

Queued as `~/vgc-data/calibrated_asks/ca65.zsh`, from a frozen copy of the original run code plus this script in `~/vgc-data/calibrated_asks/run/`.
