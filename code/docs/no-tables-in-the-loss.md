# No tables in the loss: does the illegal share fall over training epochs?

Pre-registered 2026-11-16, before the new arm trains. Issue: [No tables in the loss: does the illegal share fall over training epochs?](https://github.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/issues/361), a sub-issue of [Log first-try legality at the end of every training epoch](https://github.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/issues/248). Run type: **confirmatory** for the arm comparison. The control was probed at three epochs before this plan was fixed, and nothing of the new arm has been seen yet.

The legal Pokémon construction tables give zero probability to every value the game's rules forbid next to the fields already on the team: a move the species cannot learn, a second copy of an item, and so on. The Stat Point budget caps each Stat Point at 32 and each Pokémon's total at 66. In the default model both act twice. In training, `diffusion84.elbo_terms` sets every forbidden value's logit to −10,000 before the cross-entropy, so no gradient ever reaches those values. In sampling, `Model84._logprobs` applies the same mask, so an illegal value cannot be drawn. The per-epoch legality check of issue #248 therefore reads 0% illegal at every epoch, by construction. This experiment takes the mask out of the loss and the check's draw, so the curve can show what each network learned.

## Arms

| Reader name | Code ID | Training |
| --- | --- | --- |
| No tables in the loss | `shuffled_linear_no_table_loss` | The default model's recipe with the tables' mask and the Stat Point budget left out of training and validation loss. Trains now, 80-epoch cap |
| Tables in the loss | `shuffled_linear` | The default model, trained 2026-11-12. Its 80 saved epoch checkpoints are reused, with no retraining |

Shared recipe (`src/train84.py`): labelled teams of the shuffled-only table, the linear masking schedule, label noise 1/48, validation loss after every epoch, patience 3 with any new lowest validation loss resetting it, a cap of 80 epochs, and one checkpoint per epoch. Training still drops the 532 rows the tables reject, so the loss mask is the only difference (owner, 2026-11-16).

## Contract

| Row | Value |
| --- | --- |
| Claim | A network trained without the tables in its loss learns to write legal teams, so its illegal share drawn without the tables falls over epochs, while the default model's stays at 1. Refuted if the new arm's illegal share stays above 0.875 at every epoch. If the claim holds, legality is something the network can learn, and a later arm can drop the tables at sampling too. The claim stops at first-try legality at the elite cut. It says nothing about win rate |
| Fields | All 84 are generated: per Pokémon its species, ability, item, four moves and Stat Alignment, plus six Stat Points. None is copied, none is fixed. Round trip: 20 training teams go through encode, paste, parse and encode and come back identical, with slot and move order ignored |
| Vocabulary | The regulation snapshot in each checkpoint (`team84.regulation_vocab`), with [MASK] at index 0: 348 species, 201 abilities, 150 items, 498 moves, 27 Stat Alignments, and Stat Points 0 to 32. Both arms share it |
| Training teams | `results/lossdata84_shuffled_only.npz`, sha256 `8aca232e…`. 214,066 labelled teams outside the seeded 2% validation split, after dropping 532 that the tables reject. Validation: 4,380 labelled teams. Labels come from shuffled-slot battles only |
| Scorer | Showdown's validator (`hps_generate.Validator`, the project's format). No battles, no policy, no opponents |
| Search region | No search. Per epoch: 64 teams at the elite cut, 0.531, guidance 2, temperature 1, species-first order, Stat Alignment repaired to match the moves, draw seed 12345 at every epoch. The draw uses neither the tables nor the Stat Point budget: any value but [MASK] can be drawn. Lookahead off |
| Settings | Scientific: the loss mask (on or off). Fixed: everything else in the shared recipe, the draw settings, and the training seed 0. No nuisance setting is varied: one training per arm |
| Plan | See below |

## Measures

- **Illegal share** (primary): teams the validator rejects on the first draw, over the 64 teams drawn. The first-try legal share is 1 minus it.
- **Top rejection reasons**: each kind of error, with names and numbers removed, counted once per team that has it.
- **Rule by rule** (added 2026-11-17 after the first result, owner: "We should add more measurement than just legality like use more the item more than onces, using the wrong mega stones as well"): for each rule, the share of the 64 drawn teams that break it: an item used more than once, a wrong Mega Stone, another item the species can't hold, a species used more than once, an ability the species can't have, a move the species can't learn, the same move twice on one Pokémon, and Stat Points over 66 (`epoch_legality.RULES`). Counted from the drawn tokens with the tables' compatibility data. Exploratory. Both networks' saved checkpoints are checked again for it (`src/checkpoint_legality.py`, which redoes each epoch saved before the counts existed, with the same seed, so the illegal share doubles as a repeat check).
- Errors per team and the copied share (drawn teams identical to a training team) are logged in the result files but are not measures of this experiment. They were proposed on 2026-11-16 after the preflight, and the owner declined them on 2026-11-17: "let see what happen". The copied share is read only as a check: an exact copy is always legal, so the result comment names it if any epoch has one.
- **Validation loss** of each arm on its own loss. The two arms' losses are not comparable: the new arm's includes the forbidden values (owner, 2026-11-16).

## Plan

A single primary measure, nothing trades against it, so the question is better, no worse or the same, with a margin.

- **Primary comparison:** the new arm's illegal share against the control's at the same epoch, over epochs 1 to the new arm's stopping epoch. The control's line is read from `results/stat84_shuffled_noise_linear_es3_cap80_legality_no_tables.json` (`src/checkpoint_legality.py`), the new arm's from the curve in `results/stat84_shuffled_noise_linear_no_table_loss_es3_cap80.json`.
- **Smallest effect worth acting on:** 0.125 in illegal share, two standard errors at 64 teams and p = 0.5.
- **Predictions, written before the new arm runs:**
  1. The control's illegal share is 1.0 at every epoch, since nothing ever taught it the forbidden values.
  2. The new arm's illegal share falls below 0.875 at some epoch. This is uncertain: a legal team needs about 36 fields right at once. Discrete diffusion models learn the data's support before its frequencies ("Support Before Frequency in Discrete Diffusion", Müller et al. 2026), so if it falls, most of the drop should come early.
- **Stopping rule:** the new arm trains to its own early stop (patience 3 on its unmasked validation loss) or the cap of 80. The control's checkpoint check covers all 80 epochs.
- **Baselines:** the untrained network drawn without the tables, at 20.5 errors per team and 100% illegal (test run 2026-11-16), and the default model with the tables, at 0% illegal (preflight).
- **Budget and cost:** no battles. The control's check takes about 5 seconds a checkpoint, so about 10 minutes for 80. The new arm's training took the control 431.5 minutes for 80 epochs on this laptop's GPU, and the check adds about 5 seconds an epoch. Run it behind other sessions' jobs.
- **Result:** one ECharts chart of illegal share by epoch, one line per arm, with the top rejection reasons beside it, posted on issue #361 (`src/no_tables_in_the_loss_figure.py`). No ask grid, no battles (owner, 2026-11-16).
- **Run:** the queue `~/vgc-data/no_tables_in_the_loss/ntl_queue.sh` runs a frozen copy of `src/` behind other sessions' jobs: the control's check over its 80 checkpoints, then the new arm's training, then the chart and the comment on #361.

## Preflight

Saved under `preflight` in `results/no_tables_in_the_loss.json` (`src/no_tables_in_the_loss_preflight.py`), 2026-11-16, run on the control's checkpoints and the shuffled-only table.

| Probe | Result | Pass |
| --- | --- | --- |
| Ceiling: 64 training teams through the same paste and validator | 0.000 illegal, against 1.000 for the control at epoch 80 without the tables | Yes |
| Round trip: encode, paste, parse and encode 20 training teams | 20 of 20 identical, slot and move order ignored | Yes |
| Coverage | Vocabulary as above. 532 training rows dropped by the tables in both arms, because the tables exclude one of their true values | Yes, explained |
| Read teams: three epoch-80 teams drawn without the tables | Each breaks rules a player would reject at a glance: three Pokémon holding Charizardite Y, Milotic with Rough Skin and Chlorophyll, Stat Point totals over 66. Each error is expected of a network the tables did all the work for | Yes, explained |
| Training smoke: per-field cross-entropy at initialization, all fields hidden, no tables | Species 5.85 against ln K 5.85; item 5.00 against 5.01; move 6.36 against 6.21; ability 5.70 against 5.30; Stat Alignment 3.58 against 3.30; Stat Points 3.75 against 3.50 | Yes, within 0.5 |
| Training smoke: overfit 32 teams without the mask, 300 steps | Loss on those teams 4.69 to 2.35. A first try, 40 steps at learning rate 4e-4, moved it only from 4.288 to 4.274: too short to show anything | Yes |
| Control: the default model drawn with the tables at epoch 80 | 0.000 illegal | Yes, the alarm of issue #248 still reads 0 |
| Control: the default model drawn without the tables | Illegal share 1.000 at epochs 1, 40 and 80; errors per team 20.64, 19.06, 18.86; copied share 0 | Matches prediction 1 so far |
| Repeat: epoch 80 checked twice with the same draw seed | Identical illegal share and errors per team | Yes |
| Invariance, condition, separation | Not reachable: the ask is fixed, nothing is battled or scored against held-out teams, and team identity ignores slot order by construction | Skipped, for those reasons |
| Copies | Copied share 0 at epochs 1, 40 and 80 for the control | Reported; logged every epoch for both arms |

## Noise floor

- **Illegal share:** standard error `sqrt(p(1-p)/64)`, at most 0.0625. One epoch against one epoch separates arms that differ by about 0.25 (`3.9 / δ²` teams per arm at 5% false positives and 80% power). Pooling the last 10 epochs of each arm, 640 teams, separates 0.078.
- **Draws:** every epoch draws from seed 12345, so the curve moves only when the model does. Epoch-to-epoch wiggles are model changes, not draw noise.
- **Training seed:** one training per arm, so a gap mixes the loss mask with the training seed's effect. A gap within 0.25 in illegal share would need a second seed before it is read as the mask's effect.

The budget can detect the smallest effect worth acting on when the last 10 epochs are pooled.

## Deviations

- 2026-11-17, before the new arm trained: errors per team and the copied share, added to the measures after the preflight, were taken out at the owner's word ("I don't agree with the table, let see what happen"). The illegal share is the only measure, as the issue planned. The second chart was dropped with them.
- 2026-11-17 04:08, epoch 33 of the new arm: the training stopped with no error logged and no reboot (its keep-awake process ended; two other sessions' queues had died the same way at 01:10). It cannot resume mid-run. Having seen the 33 epochs and the slot-binding probe below, the owner chose to report these epochs rather than retrain (2026-11-17). The stopping rule was therefore not reached, and the refutation rule is read on epochs 1 to 33 only.
- 2026-11-17, after seeing the curve: the slot-binding probe (`src/slot_binding_probe.py`) was added to explain it. It is exploratory.

## Result

Reported 2026-11-17 on issue #361. Numbers in `results/no_tables_in_the_loss.json` (`arm_epochs`, `slot_binding_epoch33`) and the control's file.

- **Illegal share:** 100% at every epoch for both arms, epochs 1 to 33 of "No tables in the loss" and 1 to 80 of "Tables in the loss". Prediction 1 held. Prediction 2 failed on the epochs trained: the new arm never fell below 0.875. Validation loss was still falling, 2.40 to 2.00, so a longer run could still differ, but the probe below suggests why it would not.
- **Why (exploratory):** at epoch 33, on 256 real validation teams with the true earlier fields shown, the new arm's probability at a Pokémon's ability splits 39% legal for its own species, 56% legal only for another team member's species, and 5% legal for none of the six. At the first move: 47%, 50%, 3%. The network learned which values belong on the team, not which Pokémon they belong to. One value on the wrong team member makes the team illegal. This matches the 2026-11-04 Stat Point swap test and the #84 position-split result. The tables do this binding in every default-model draw.
- **Rule by rule:** queued 2026-11-17 (`~/vgc-data/no_tables_in_the_loss/rules_queue.sh`); posted on #361 when it finishes.
- **What it can't show:** one training seed per arm; 33 of up to 80 epochs; first-try legality only, no battles.
