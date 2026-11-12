# Looped 84-field model: a repeated block and a carried state, each at equal compute and at equal epochs

Pre-registration, written 2026-11-12 before any loop training or grid. Map: [Loop map: does a repeated block or a carried state move the 84-field model's trade-offs?](https://github.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/issues/306). The owner chose the scope on 2026-11-12 in answer to the loop explainer's questions (`notes/Blog/visuals/looped-diffusion-teach/`): quality is judged as a trade-off, both ways of matching the control are trained, and the two loops that change training are tested. Early exit is left out as an inference-time saving. Every row uses the linear masking rate, where every field is hidden with chance t, and no row has a separate rate per field (owner, 2026-11-12).

Mode: **exploratory**. One training seed per arm cannot separate the loop from training luck, which has reached 0.0019 of error between two trainings of one recipe.

## Terms

An ask is the win rate the model is told to aim for. Got is the win rate its teams then score. A line-up is a team's six species. The ask-versus-got error is the mean squared error between each ask and the mean got at that ask, on asks 1/24 to 17/24 (`src/ask_line.py`).

Win rate means one battle against each of the 49 distinct top-50 meta teams, with the behaviour-cloning policy on both sides and slots shuffled every battle. The top-50 meta teams are the project's fixed list of real Regulation M-B teams. The behaviour-cloning policy is a network trained to imitate human battle decisions.

A pass is one run of the network while sampling. The sampler fills one field per pass, in species-first order, so a team takes 84 passes. A layer is one transformer layer. The network has 6.

A **loop** here means running some of the network's work again without new weights. The two kinds tested:

- **Repeat block.** Layers 1 and 2 (counted from 0) form a block that runs several times inside one pass, sharing its weights. Layer 0 runs before it and layers 3 to 5 after it, once each. Training draws the loop count uniformly from 1 to 4 at every step, and sampling runs it 4 times. The weights are those of the plain network. From LoopMDM: Lee et al., "Looped Diffusion Language Models", [arXiv 2605.26106](https://arxiv.org/abs/2605.26106).
- **Carried state.** Each pass hands its final hidden state to the next pass, which adds a layer-normalized copy to its field embeddings. The first pass gets a zero state. Training cannot run all 84 passes, so on 90% of steps it runs the network twice at the same masked team: first with a zero state and no gradient, then with the first run's state, and it scores the second. The other 10% score one run with a zero state. It adds one layer norm, 512 weights. From Loopholing: Jo et al., "Loopholing Discrete Diffusion: Deterministic Bypass of the Sampling Wall", ICLR 2026, [arXiv 2510.19304](https://arxiv.org/abs/2510.19304).

Training compute is counted in multiply-adds per team, with the backward pass costing twice the forward pass (`diffusion84.training_macs`). Both papers count it this way.

## What is already known

| Earlier result | What it says here | What it leaves open |
| --- | --- | --- |
| Loop explainer measurement, 2026-11-02, all-teams 21-epoch model (`src/loop_field_difficulty.py`) | The sampler fills one field per pass. If the six items were filled in one pass, 71% of teams would repeat an item. So the papers' saving in passes does not apply here, and a loop can only pay through training compute or team quality | Whether either loop changes training compute, error or line-ups |
| LoopMDM, Sudoku with a fixed fill order (Figure 4) | In one shown puzzle, errors fall from 24 to 7 to 0 at loop counts 1, 2 and 3. Our species-first order is also fixed | A team is not a puzzle with one answer |
| LoopMDM, 170 million parameters (§4.1, Table 4c) | Same test loss with 2.34× to 3.34× less training compute. A maximum loop count of 4 already gives 3.741 against the best 3.729 and the plain 3.798 | Whether a 5.5 million parameter, 6-layer network gains the same |
| Loopholing (Table 10, Appendix D.6) | At matched forward passes, test perplexity 21.90 against 22.54 for the plain model trained twice as long | Its gains are largest when many tokens are filled per pass |
| The species-blind Stat Points reading, 2026-11-04 | On the 80-epoch labelled model, swapping the Pokémon in slots 1 and 2 moves slot 1's Stat Point distribution by total variation 0.0014 | Whether more depth (repeat block) or more context (carried state) changes it |

## Rows

Each reader name gives the loop, then the training length. All rows share the control's teams, recipe and linear masking rate, so that rate is not repeated in the names.

| Loop, then training length | Code ID | Checkpoint | Training compute against the control |
| --- | --- | --- | --- |
| No loop, 80 epochs. The control | `shuffled_linear` (ask grid `sched_linear_species_first`) | `stat84_shuffled_noise_linear_es3_cap80`, training since 2026-11-12 12:20, epoch 15 of 80 at 13:45 | 1 |
| Repeat block, 53 epochs | `loop_block_equal_compute` | `stat84_shuffled_noise_loop_block_equal_compute` | 0.99 |
| Repeat block, 80 epochs | `loop_block_equal_epochs` | `stat84_shuffled_noise_loop_block_equal_epochs` | 1.49 |
| Carried state, 61 epochs | `carry_state_equal_compute` | `stat84_shuffled_noise_carry_state_equal_compute` | 0.99 |
| Carried state, 80 epochs | `carry_state_equal_epochs` | `stat84_shuffled_noise_carry_state_equal_epochs` | 1.30 |

The control is the masking-rate experiment's row Linear, species-first (`docs/masking-rate-rows.md`). It is trained there (issue #296) and its grids at guidance 2, 4 and 8 are sampled and battled there (issue #297), so this experiment adds no control training or grid. If it stops early at epoch E, its compute is E epochs, and the equal-compute caps are recomputed from E before launch and recorded under Deviations. The ask grid gets a code ID for each new row when the code ticket lands.

The 53 and 61 are the largest whole numbers of epochs whose training compute stays at or below the control's 80 epochs: 80 / 1.495 = 53.5 and 80 / 1.3 = 61.5 (`train84.equal_compute_cap`).

## Contract

| Row | Value |
| --- | --- |
| Claim | A loop moves the 84-field model's trade-offs outside the control's: a lower validation loss for the same training compute, and a lower ask-versus-got error or more line-ups at the same guidance. |
| What refutes it | At equal compute, the loop's validation loss is no lower than the control's, and its error and line-up points fall on or inside the control's curve at guidance 2, 4 and 8. |
| What changes if it holds | The winning loop becomes an option for the next default-model training, and the owner decides on a second training seed. If refuted, the network stays plain and looping is recorded as measured. |
| How far it reaches | Win rate under the behaviour-cloning policy against the 49 distinct top-50 meta teams, slots shuffled, one training seed per arm. Nothing beyond that. |
| Fields | The model generates all 84: for each of six Pokémon its species, ability, item, four moves, Stat Alignment and six Stat Points. Nothing is copied or fixed. After sampling, a repair step replaces Stat Alignment to match the moves (`src/ask_grid84.py`). |
| Vocabulary | The regulation tables in each checkpoint (`team84.regulation_vocab`): 348 species, 201 ability, 150 item, 498 move and 27 Stat Alignment tokens, Stat Points 0 to 32. Same for every row. |
| Training teams | The shuffled-only table, `lossdata84_shuffled_only.npz`, labelled teams only: 214,598 train and 4,380 validate, seeded 2% split with seed 0, the same rows as the control. Left out: the 10 never-battled teams, the 500 test teams, the held-out meta teams ranked 51 to 100, and every team labelled only by listed-order battles. |
| Scorer | `src/ask_grid84.py`: one battle against each of the 49 distinct top-50 meta teams, behaviour-cloning policy on both sides, slots shuffled, battle seed 7300 plus the sampling seed. |
| Search region | No search. Asks 1/24 to 23/24, sampling seeds 1 to 3, 64 teams for each ask and seed, at each of guidance 2, 4 and 8, drawn until 64 pass Showdown's validator and are not held-out teams. 4,416 teams and 216,384 battles per row at each guidance value. |

### Settings

| Setting | Role | Value | Source |
| --- | --- | --- | --- |
| Loop kind | Scientific | None, repeat block, or carried state | LoopMDM Eq. 4; Loopholing Eq. 5 and Algorithm 1 |
| Training length | Scientific, crossed with the loop kind | 80 epochs, or the equal-compute cap of 53 or 61 | LoopMDM §3 Eq. 6 matches total training compute by training fewer steps; LAW Appendix B.1 does the same for Loopholing (400k against 308k steps). Equal epochs is the owner's second match |
| Block position and size | Fixed | Layers 1 and 2 of 6 | LoopMDM §4.1 and Table 4a and 4b: layers 1 and 2 of 12 were best |
| Maximum loop count | Fixed | 4 in training, uniform 1 to 4 each step; 4 when sampling | LoopMDM Eq. 5 draws uniformly from 1 to the maximum; it used 12 for 12 layers and 8 for 14. **Ours: 4**, because at 6 one training step needed 13 GB of GPU memory and slowed from 1.5 s to 13 to 36 s on this laptop (probe below). Table 4c gives 3.741 at 4 against 3.729 at 12 |
| Input injection | Fixed | None | LoopMDM Appendix D: injection hurt |
| Carried state | Fixed | The network's final hidden state, layer-normalized by a new layer norm, added to the field embeddings; zero at the first pass | Loopholing Eq. 5, Algorithm 1 line 6 |
| Two-run training share | Fixed | 0.9 of steps | LAW §2.2: "p = 0.9 by default" for Loopholing |
| Stop-gradient between the two runs | Fixed | On | Loopholing §3.2, Eq. 7; LAW Appendix A.5 and Figure 9: removing it lowered accuracy |
| Carried state under guidance | Fixed | The guided and unguided passes each carry their own state | Ours. Neither paper uses guidance |
| Carried state in validation loss | Fixed | Two runs, as in training | Ours, so the logged loss scores what the second run predicts |
| Sampling seed | Nuisance | 1, 2, 3 with offset 7300 | Rows compared seed by seed |
| Guidance | Swept | 2, 4 and 8 | The project's ask-grid values |

Fixed and inherited from the control, unchanged: the linear masking rate (`1,1,1,1,1`, Stat Points 1, MD4 Appendix J.1), loss weight 1/t, t drawn evenly across the batch, sum divided by 84, time input, slot shuffling, legal Pokémon construction filter in loss and sampling, ask dropout 0.15, label noise ±1/48, network 256 wide with 8 heads and dropout 0.1, AdamW at 4e-4 with weight decay 0.01, 500 warm-up steps and cosine decay sized to the arm's cap, batch 256, gradient clip 1.0, weight average 0.999, training seed 0, validation loss each epoch on the same 4 mask draws, patience 3, early-stopping threshold 0, every epoch saved, 84 passes in species-first order, temperature 1, Stat Alignment repair.

With both options off, the code reproduces the old network exactly: the same loss, the same random-number stream and the same validation loss as `diffusion84.py` before this change (`scripts/test_looped84.py`).

### Plan

There are two Pareto questions. Neither has a single winner.

**1. Training compute against validation loss.** One point per saved epoch: the training compute spent so far, in control epochs, against the validation loss under the linear masking rate. Validation loss is measured with `masking_rate_training_report.py`, which drops the 12 validation teams the construction tables reject. The control's 80 points are the frontier. A loop moves outside it when its equal-compute endpoint has a lower loss than the control's last epoch. The papers' headline figure is also read off: the share of the control's compute at which the equal-epochs curve first reaches the control's final loss. The cosine schedule cools only at each run's own end, so a mid-run point understates the loss that run would reach if stopped there. That is why the equal-compute run exists.

The repeat block's validation loss is read at loop counts 1, 2, 4 and 8, with no battles, to show whether more loops at sampling keep helping (LoopMDM Figure 2). Only loop count 4 is battled.

**2. Error against line-ups.** As in the masking-rate rows. Guidance 2, 4 and 8 trace each row's three-point curve, and the control's curve is the frontier. A row moves outside it when one of its points has a lower error than every control point with at least as many line-ups, or more line-ups than every control point with an error at least as low.

The verdict for each row at each guidance value, paired with the control by sampling seed:

- It **wins** if its error is at least 0.001 lower and the 95% interval of the difference lies below zero.
- It **loses** if that interval lies above zero.
- It **matches** otherwise.

A row counts as winning when it wins at two or more guidance values and loses at none. A gap under 0.002 is reported as within training luck. A row that wins on error while its pooled win rate falls by more than 0.026, or its line-ups fall, is reported as a trade.

Read beside both, every row by the same code:

| Measure | How it is read |
| --- | --- |
| Win rate at the top scored asks | Mean got over asks 15/24 to 17/24 |
| Pooled win rate | Mean got over all 23 asks |
| First-try legality | Share of sampled teams the validator accepts before any redraw |
| New line-ups | Share of teams whose line-up is absent from the training teams and from earlier runs' and tournament line-ups (`src/lineup_reference.py`) |
| Effective species, top-species share | Mean over random 5-team subsets of each cell |
| Elite line-up yield | Line-ups with a team at or above the elite cut of 0.531 that reach it again in a fresh re-battle of 4 games against each opponent |
| Sampling time | Seconds per 64 teams at guidance 2, on this laptop |
| Training time | Minutes, beside the compute count |
| Stat Points swap reading | The 2026-11-04 reading on each new model: how far slot 1's Stat Point distribution moves when slots 1 and 2 swap Pokémon |
| Training-versus-validation gap | As in the masking-rate rows |

| Part | Value |
| --- | --- |
| Smallest effect worth acting on | 0.001 of error. For validation loss, 0.005, about what the per-field model gained over its last 20 epochs (2.0070 at epoch 60 to 2.0026 at 80, logged loss); checked against the linear control's own curve when it finishes. No seed-to-seed spread of validation loss has been measured for this recipe |
| Comparisons planned | Each new row against the control at the same guidance. Each equal-epochs row against its equal-compute sibling, which isolates training length |
| Baselines | The control. Each new row's unguided samples, 64 teams for each of 3 seeds. Top-50 meta teams near 0.50 and uniformly drawn legal teams near 0.02 through the same scorer |
| Stopping rule | One 3-seed grid per row at each guidance value. No re-runs to chase a result |

### Expectations

Written before any result.

| Row | Expected | Chance | Why |
| --- | --- | --- | --- |
| Repeat block, 53 epochs | Validation loss below the control's final loss; error matches | 55% for the loss, 65% for the error | LoopMDM gained at every scale it tried. The error sits mostly at asks 13/24 to 17/24, where got stops near 0.53, and depth gives no reason for that ceiling to lift |
| Repeat block, 80 epochs | Lower validation loss than its 53-epoch sibling; error matches the control | 70% and 65% | More training at more compute |
| Carried state, 61 epochs | Validation loss at or above the control's; error matches; line-ups fall | 55%, 65%, 40% | One field per pass leaves less to carry than many tokens per pass. LAW reports less diverse samples at many passes |
| Carried state, 80 epochs | As its 61-epoch sibling, slightly lower loss | 60% | |

The largest surprise would be a loop that lowers the error by 0.002 or more.

## Probes

Script: `src/loop_preflight84.py`, battle-free. Results: `~/vgc-data/results/loop_arms_preflight.json`. Tests: `scripts/test_looped84.py`, 12 tests, pass.

| Probe | Result | Pass |
| --- | --- | --- |
| Plain network unchanged | Loss, next random number and validation loss match the old code to 4 decimals. Both options start from the same weights as the plain network; the carried state adds only its layer norm | yes |
| Repeat block | Loop count 1 equals the plain network. Count 3 equals the layers unrolled by hand. Training draws every count from 1 to the maximum | yes |
| Carried state | No state equals a zero state, and a state changes the output. The two-run loss scores the second run on the first run's state, and gradient reaches the new layer norm. The sampler hands each pass the previous pass's state, separately for guided and unguided passes | yes |
| Save and load | Both options sample the same teams from a fixed seed before and after a checkpoint round trip | yes |
| Control | `stat84_shuffled_noise_linear_es3_cap80` is still training; checked again when it finishes. The first run checked the per-field model: plain network, shuffled-only table, cap 80, best epoch 80, 214,598 teams | after #296 |
| Memory at loop count 6 | One training step at batch 256: 0.6 s at count 1, rising about 0.17 s per extra count to 1.48 s at 5, then 13 to 36 s at 6, with 13.3 GB of GPU memory. Another training was running. So the maximum is 4 | count 6 **no**, count 4 yes |
| Step time | Seconds per training step at batch 256 under the linear rate, after 12 untimed steps, another training sharing the GPU: plain 0.628, repeat block 0.870 (1.39×, against 1.49× in multiply-adds), carried state 0.890 (1.42×, against 1.30×) | reported |
| Sampling time | 64 teams at guidance 2: plain 10.9 s, repeat block 17.3 s (1.59×), carried state 10.7 s (0.99×) | yes |
| Training smoke, plain network on the linear rate | Overfitting 32 teams for 300 epochs takes validation loss from 3.663 to 1.917, the same numbers the masking-rate training smoke logged, so the probe replays that run | yes |
| Training smoke, repeat block | Every field type starts within 20% of ln K. Validation loss falls from 3.646 to 1.993, a 45% fall, within 10% of the plain network (bar 2.109) | yes |
| Training smoke, carried state | Every field type starts within 20% of ln K. Validation loss falls from 3.682 to 2.090, a 43% fall, just inside the bar of 2.109 | yes, narrowly |
| Ceiling for the error | Training teams battled again miss the line by 0.0038 (`docs/training-teams-line.md`, older table). The control's own error arrives with its grid on #297. If it is under 0.0048, no row can gain the 0.001 bar on error, and the verdict rests on validation loss and line-ups | after #297 |
| Round trip, coverage, read teams, controls, separation, condition, copies | Run by `ask_grid84.py preflight` with each row's checkpoint before its grid, as in the masking-rate rows; a failure stops the row. Separation holds by construction: same table and split as the control | before each grid |
| Invariance, repeat | Skipped: the scorer is unchanged and shuffles slots every battle | not run |

**The carried-state smoke under the per-field rate.** The first run used the per-field masking rate, and there the carried state failed: 1.869 against a bar of 1.794. The new layer norm starts with weight 1, so from the first step every field embedding gets a unit-scale state added. The overfit ends at 1.855 even when scored with a zero state, so the whole network learns slower, not only the carried path. With the weight started at 0, the overfit ends at 1.670, inside the bar, but the weight is still 0.004 after 300 epochs: on 32 teams the network barely uses the state. Loopholing's from-scratch recipe uses the default start (Algorithm 1); it starts the norm at zero only when adding the path to a trained model (Appendix D.10). The bar was set for position options of the plain network, where it caught pipeline bugs. Under the linear rate the weight-1 start passes, by 0.019. The owner decides between the two starts before the carried-state arms launch ([Carried state: should its new layer norm start at weight 1 or 0?](https://github.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/issues/307)).

Other samplers that call `Model84._logprobs` directly (`remask84.py`, `nucleus84.py`, `trade84.py`) would give a carried-state model a zero state at every pass. The ask grid samples through `Model84.sample`, which carries the state.

## Noise floor and budget

Same scorer and grid as the masking-rate rows, so their measured noise applies: the difference between two rows has a standard deviation of 0.00034 to 0.00037, putting the smallest detectable gap near 0.001. Training luck is up to 0.0019 of error and 0.026 of pooled win rate, which one seed per arm cannot remove. Hence exploratory.

Cost, all on this laptop, queued behind the masking-rate queue (`~/vgc-data/masking_rate_rows/mr84.zsh`, linear training running at 2026-11-12 13:10, then the #297 grids):

| Step | Battles | Time |
| --- | --- | --- |
| Four trainings | none | about 36 hours: repeat block 6.9 h at 53 epochs and 10.4 h at 80, carried state 8.1 h at 61 epochs and 10.6 h at 80, scaling the linear control's 5.6 minutes an epoch by each option's step-time ratio |
| Grids for four rows at guidance 2, 4 and 8 | 2,596,608 | about 9.2 hours of battles, plus slower sampling for the repeat block |
| Unguided samples, 192 teams per row | 37,632 | about 8 minutes |
| Elite line-up yield re-battles | about 420,000 | about 1.5 hours |
| Validation loss per epoch, and the repeat block at counts 1, 2, 4, 8 | none | about 1 hour |

About 36 hours of training and 3.1 million battles, about 11 hours. No cloud compute.

## Deviations

None yet.
