# Subset size for training labels: does a noisier subsampled label help the no-prompt model, or break it?

Pre-registration, 2026-11-07, before any of these trainings or battles. Issue: [Training labels: which battles and subset sizes make them?](https://github.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/issues/118), on the map [Opponent prompt](https://github.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/issues/115). Mode: **exploratory**, because each arm gets one training seed.

## Why this run

Training draws each label as wins ÷ battles over a random subset of the 39 training opponents. The owner asked to test two subset-size rules, 10 to 39 and 1 to 39, for any difference in performance, since label noise has improved this model before. On the same 39 training opponents, the all-teams 84-field model scores an ask-versus-got error of 0.0201 on asks 1/24 to 17/24 without noise. With ±1/48 label noise it scores 0.0070 (`subset_size_labels_gen84_*.json`, rescored from stored battles). A subsampled label is label noise of the same kind, only larger:

| Label rule | SD of label minus the all-39 label |
| --- | --- |
| ±1/48 uniform noise (the run that helped) | 0.012 |
| 10 to 39 opponents per label | 0.063 |
| 1 to 39 opponents per label | 0.122 |

The opponent-prompt models need cross attention, which does not exist yet (#119). This run uses the no-prompt model, the map's control arm. For that model a subsampled label is pure label noise, so the run answers "does this much noise help or hurt?" before the prompt models are built.

## Arms

| Reader name | Code ID | Difference |
| --- | --- | --- |
| 10 to 39 opponents per label | `subset_10_39` | A fresh subset every epoch, size uniform in [10, 39] |
| 1 to 39 opponents per label | `subset_1_39` | A fresh subset every epoch, size uniform in [1, 39] |
| All 39 opponents per label | `subset_39` | The control: each team's label over all its training opponents, the same every epoch |
| No ask, all 39 opponents | `unguided` | The unguided baseline: the control sampled with no ask, 192 teams |

Checkpoints are `stat65_<code ID>`, and the ask grid names them `gen65_<code ID>`.

## Contract

| Row | Value |
| --- | --- |
| Claim | Under the behaviour-cloning policy against the 39 training opponents, a subsampled label of 10 to 39 opponents changes the ask-versus-got error of the no-prompt 84-field model relative to the all-39 label, and 1 to 39 changes it further. **Refuted** if both subset arms land within 0.001 of the control. **If it holds**, the better rule becomes the default for all three opponent-prompt models. If 1 to 39 is no worse than 10 to 39, the prompt models also train on prompts of 1 to 9 teams. Scope: win rate under this policy against these teams, nothing beyond. |
| Fields | All 84 generated. None copied. Stat Alignment is repaired to match the moves after sampling (`ask_grid84.py`). |
| Vocabulary | Regulation M-B tables stored in checkpoint `F1_final_medium_noheldout_regmb_v3` (`team84.regulation_vocab`), as for every 84-field model. |
| Training teams | `lossdata65.npz` (`src/lossdata65.py`, docs/opponent-prompt-training-labels.md): 276,166 teams with shuffled-slot battles. It drops the 5,672 within one species swap of a validation opponent, leaving 265,088 training and 5,406 validation rows (2%, seed 0). Teams that were never battled, and listed-order or other configurations, are not in the table. Ranks 51 to 100 are absent (0 rows). |
| Scorer | `ask_grid84.py`: behaviour-cloning policy on both sides, one battle against each of the 49 meta teams, slots shuffled, seeds 7301 to 7303. Got is split from the stored battles (`train65.py split`): against the 39 training opponents, which is what the ask means, and against the 10 validation opponents. |
| Search region | No search loop. Asks 1/24 to 23/24, 64 legal teams per ask and seed, at most 1,280 tries, guidance 2, temperature 1, species first, legal Pokémon construction filter on. |
| Settings | **Scientific:** subset-size rule. **Nuisance:** none varied; one training seed per arm. **Fixed:** training seed 0, subset seed 6500 + epoch, batch 256, learning rate 4e-4, deployed network (d 256, 6 layers, 8 heads), ask dropout 0.15, EMA 0.999, cap 80 epochs, stop after 3 epochs with no new lowest validation loss, best epoch kept. Validation rows always carry the all-39 label. No other label noise. Near-copy exclusion is one swap, **pending** the near-copy grill: if that grill settles a different rule, the table is filtered again and the arms retrain before launch. |
| Plan | **Primary:** ask-versus-got error against the training opponents on asks 1/24 to 17/24, with 18/24 to 23/24 reported apart. **Pareto question:** got against the training opponents (x) and distinct line-ups per ask (y), each arm's curve traced over the 23 asks. The strongest previous frontier is the ±1/48 label-noise model, rescored on the same opponents (error 0.0070). A subset arm is good enough if its curve moves outside that frontier. **Beside:** got against the validation opponents, effective species, top-species share, and the share of new line-ups against the fixed reference (loop and records rows of `lossdata84.npz`, 11,690 line-ups). **Smallest effect worth acting on:** an error difference of 0.001, the near-the-line bar. **Comparisons planned:** each subset arm against the control, and the two subset arms against each other. **Baselines:** the control and the unguided samples. **Stopping rule:** one grid per arm, with no re-runs to chase a result. |

## Probes

| Probe | Result | Pass |
| --- | --- | --- |
| Training smoke | `results/train65_smoke.json`. A fresh network's per-field cross-entropy with every field masked is 4.64 against ln K 4.66 (median ratio 1.00, range 0.88 to 1.08). On 64 teams, training loss falls from 2.56 to 1.25 over 400 epochs. Labels are redrawn before every epoch and differ between epochs. | yes |
| Round trip, coverage | Table: 20 of 20 teams decode to their stored paste, and 0 are dropped (docs/opponent-prompt-training-labels.md). Model: `ask_grid84.py preflight` after each training. | table yes; model after training |
| Controls | Table: training meta teams 0.500, uniform-legal teams 0.021. The scorer's control battles run in `ask_grid84.py preflight`. | table yes; scorer after training |
| Separation | Table: 0 rows from ranks 51 to 100. Validation-opponent battles never enter a label. Model: `preflight` checks ranks 51 to 100 against the arm's own training rows (`table65` branch). | table yes; model after training |
| Split report | Five table teams rescored through `split`. Training-opponent got 0.3795 equals the table's label 0.3795, and each ask of the earlier label-noise grid finds all 192 of its teams. | yes |
| Read teams, condition, copies | Need a trained checkpoint: `ask_grid84.py preflight` runs them before each grid. | after training |
| Ceiling | Skipped. There is no last stage to give a perfect input: the comparison is between training labels. | n/a |
| Invariance, repeat | Skipped. The scorer is unchanged, and every battle shuffles slots (`require_shuffled`). | n/a |

## Noise floor and budget

Each ask holds 192 teams. Got against 39 training opponents has a standard error of about 0.008 per ask, and against 10 validation opponents about 0.012. Battle noise alone therefore adds about 0.0001 to an arm's error, and the error's own sampling SD over 17 asks is about 0.0003. A difference of 0.001, the smallest worth acting on, is about three of those SDs, so the budget can detect it. Training luck cannot be separated: one seed per arm. That is why the run is exploratory, and a difference near 0.001 earns a second seed before it becomes a rule.

Cost, all on the laptop and queued behind other sessions' jobs:
- **Training:** three trainings at about 2.5 hours each (the 21-epoch all-teams training took 146 minutes).
- **Battles:** 23 asks × 192 teams × 49 battles is 216,384 per arm, about 50 minutes at the last grid's rate. Three arms plus the 9,408 unguided battles total about 658,000 battles, about 2.5 hours.
- **Modal:** none.

## Commands

```
python src/train65.py subset_39 ; python src/train65.py subset_10_39 ; python src/train65.py subset_1_39
AG84_MODEL=gen65_<arm> python src/ask_grid84.py preflight ; ... gen ; ... battle ; ... report
python src/train65.py split gen65_<arm> ; python src/train65.py unguided
```

## Figures

Storyboarded 2026-11-08, after the control's result and before either subset arm's. Code: `src/subset_size_storyboard.py` (`data`, then `figures`), from stored battles only.

| Frame | Sentence the reader takes | Expected | Surprise |
| --- | --- | --- | --- |
| 1. Error minus the control's, per label rule | A label over a random subset of opponents brings got closer to the ask than the all-39 label, or it doesn't. | 10 to 39 left of zero, near the ±1/48 frontier (−0.0026 against this control); 1 to 39 between it and zero. | 1 to 39 furthest left (more noise keeps helping), or both right of zero (this much noise hurts). |
| 2. Got against ask | The difference lives at asks below 0.5, where the control gets less than it asks for. | Subset arms lift low-ask got toward the line, as ±1/48 noise did. | They change only asks 18/24 to 23/24, which the error doesn't judge. |
| 3. Distinct line-ups at each ask | Getting closer to the ask doesn't cost variety. | Subset arms keep the control's 120 to 190 line-ups per ask. | Noisy labels collapse variety, as the no-noise model did (51 at some asks). |

A teaser waits for both subset arms.

## Launch

Queued 2026-11-07 20:46 as `~/vgc-data/subset_size_labels/ss65.zsh`, behind the masked-diffusion memory benchmark and the hindsight-relabel fine-tune. It runs from a frozen copy in `run/` (`SHA256SUMS`). `run/SNAPSHOT.txt` holds the commit, the uncommitted changes (`run/uncommitted.diff`) and the table's sha256. The vocabulary comes from the checkpoint named in Contract. Per arm, in order: train, preflight in a private database, 3-seed grid, battles, split. The unguided baseline runs last.

## Deviations

- 2026-11-07, before any result: launched while the near-copy grill was still open, with the one-swap rule it recommends. The queue waits behind other jobs for hours. If the grill settles a different rule, the queue is stopped before the first training and the table filtered again.
- Each arm's grid runs right after its training, not after all three trainings. Same work, earlier results.
- No fixed batch is decoded during training: `train_new` has no hook for it. Teams are read in each arm's preflight before its grid.
- 2026-11-08, after the control's result, before either subset arm's: the Pareto view (got against distinct line-ups, one curve per arm) became frame 3, distinct line-ups against ask. Got against variety curves loop back on themselves, because got falls at the top asks, so no frontier could be read off them. The training cap was reached: the control's best epoch was 80 of 80 (8.6 hours against the planned 2.5), and the 10 to 39 arm was still improving at epoch 64. The cap stays as pre-registered.
- 2026-11-09, after the 10 to 39 arm's training, before its battles: its grid crashed in the diversity count. A Pokémon after the first had no item, and `team_diversity.slots_from_paste` merged it into the slot before. The parser now starts a slot at a blank line; no stored paste in either grid parses differently. The queue skips arms already split and resumes the grid where it stopped. The 10 to 39 arm also hit the 80-epoch cap (best epoch 80, 8.7 hours).
- 2026-11-10, same point: the grid crashed the same way on its restart. `ask_grid84.py` was importing `team_diversity` from `/tmp/vgc-pilot/src`, which lacked the fix, and not from the frozen copy. It now imports that one module from its own folder. Eight other modules still load from `/tmp/vgc-pilot/src`, unchanged since 2026-10-22, the same ones the control used.
