# Masking-rate rows: per-field against linear masking rate, the t the sampler tells the model, and the decoding order

Pre-registration, written 2026-11-11 before the new training and before any of the new grids. Ticket: [Pre-register the four masking-rate rows](https://github.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/issues/294), on the map [Masking-rate map: per-field versus linear masking rate on the 84-field model, and where our code departs from MD4 and LLaDA](https://github.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/issues/291).

Mode: **exploratory** for the two linear rows, because one training seed cannot separate the masking rate from training luck. A run is confirmatory when its plan is fixed before any result and its budget can detect the effect. Per-field, corrected t compares two samplers on one checkpoint under such a plan, so it would be confirmatory for that checkpoint if its grid runs.

**One probe failed before launch.** The default model barely reads the t it is told, so the row Per-field, corrected t cannot move a result by the smallest effect worth acting on. Its battle grid is on hold for the owner. See [Per-field, corrected t is on hold](#per-field-corrected-t-is-on-hold).

## Terms

An ask is the win rate the model is told to aim for. Got is the win rate its teams then score. A line-up is a team's six species. The ask-versus-got error is the mean squared error between each ask and the mean got at that ask, on asks 1/24 to 17/24 (`src/ask_line.py`). The line is got = ask, and near the line means an error under 0.001.

Win rate means one battle against each of the 49 distinct top-50 meta teams, with the behaviour-cloning policy on both sides and slots shuffled every battle. The top-50 meta teams are the project's fixed list of 50 real teams for Regulation M-B, the game's current rule set. One team is on the list twice, which leaves 49. The behaviour-cloning policy is the battle player, a network trained to imitate human battle decisions. Slots are the order of a team's six Pokémon.

The masking rate is the chance that a field is hidden at time t during training. Time t runs from 0, nothing hidden, to 1, everything hidden. It is a dial for how much is hidden, not a step count. The network takes t as an input, so the sampler must also tell it a t at every pass. A pass is one run of the network while sampling. The decoding order is the order in which the sampler fills fields.

Showdown is the battle simulator, and its validator accepts or rejects a team. First-try legality is the share of sampled teams the validator accepts before any redraw. The legal Pokémon construction filter gives zero odds to every value the game's rules forbid next to the fields already on the team. The lookahead is a further sampler rule: a value is allowed only if every Pokémon whose species is still hidden keeps at least one species it could be. Stat Alignment is the project's name for a Pokémon's nature, such as Adamant. The remaining project terms follow `notes/GLOSSARY.md`.

The two yardstick papers, set by the owner on the map, are MD4 for training and masking rates (Shi et al., "Simplified and Generalized Masked Diffusion for Discrete Data", NeurIPS 2024, [arXiv 2406.04329](https://arxiv.org/abs/2406.04329)) and LLaDA for decoding (Nie et al., "Large Language Diffusion Models", NeurIPS 2025, [arXiv 2502.09992](https://arxiv.org/abs/2502.09992)). Every citation below was checked on 2026-11-11 against each paper's LaTeX source from arXiv, first by a subagent and then line by line by the main session. Section, appendix, algorithm and equation numbers were counted from the order of the source, so a printed number could differ by one.

## What is already known

Three earlier results bear on this run. None answers it.

| Earlier result | What it measured | What it leaves open |
| --- | --- | --- |
| The retrain that introduced the per-field masking rate, 2026-10-07, 48-field model (`docs/loss-down.md`, runs `B3`, `S7`, `N1`) | Loss only, on 500 test teams. Exact negative log-likelihood per field when decoding species-first: 1.451 with the per-field masking rate against 1.489 with the linear one. Under the standard evidence lower bound, which uses the linear masking rate, the per-field model was the worse of the two: 1.829 against 1.743. Removing the time input changed nothing: 1.743 and 1.493 against 1.743 and 1.489. | No battles. The exponents of the per-field masking rate, 16, 4, 4, 1.5 and 1, were chosen on a 0.038 gain in loss and carried into every model since. |
| Random decoding on a per-field model, 2026-11-07, all-teams 21-epoch model with the lookahead on (`docs/remask-84-fields.md`) | Ask-versus-got error 0.0534 against 0.0192 for species-first on the same checkpoint. First-try legality 83.6% against 100%. Line-ups 2,220 against 881. Without the lookahead, 1.2% of tries were legal. | Whether random order fails by itself, or because the per-field masking rate almost never shows the model a Pokémon whose moves are visible while its species is hidden. |
| Web search, 2026-11-11 | No published comparison of a masking rate set per field type against the linear one on structured records such as teams. MD4 §6 learns a rate per token value and reports likelihood only. A search summary, not read in full, says LayoutDiffusion ([arXiv 2303.11589](https://arxiv.org/abs/2303.11589)) ablates a separate rate for the type tokens of a layout. | The comparison on battle results. |

## Rows

Each name gives the masking rate used in training, then the one decoding setting that sets the row apart from its sibling. Both per-field rows decode species-first. Both linear rows tell the model t = 1 − filled/84, which under the linear masking rate is already the t of training.

| Masking rate in training, then the decoding setting | Code ID | Checkpoint | t told to the model at each pass | Decoding order |
| --- | --- | --- | --- | --- |
| Per-field, uncorrected t. The control, which is today's default model | `sched_species_first` | Epoch 80 of the labelled-teams training with label noise, `stat84_labelled_noise_byepoch_es3_cap80_ep80.pt`. Its grid exists as `gen84_noise_byepoch_ep80` | 1 − filled/84 | Species-first |
| Per-field, corrected t | `sched_species_first_t` | The same checkpoint. No training | The t whose expected hidden share in training equals the share still hidden | Species-first |
| Linear, species-first | `sched_linear_species_first` | New training, `stat84_labelled_noise_linear_es3_cap80` | 1 − filled/84 | Species-first |
| Linear, random order | `sched_linear_random` | The same new checkpoint | 1 − filled/84 | Random |

Each battled row is also sampled with no ask, 64 teams for each of the 3 sampling seeds. These unguided baselines carry the row's code ID with `_unguided` added. The linear model is the new checkpoint that both linear rows share.

Per-field masking rate with random order is not a row. The map rules it out as already measured, in the second line of the table above.

## Contract

| Row | Value |
| --- | --- |
| Claim | The owner's hypothesis on the map is that the places where our code departs from the papers cause decoding errors. This run tests three of them, one per new row: the t the sampler tells the model, the per-field masking rate, and the fixed species-first order. The claim is that the published setting gives a lower ask-versus-got error than ours. Win rate and variety are read beside it, and a gain that costs either is reported as a trade. A probe has already answered the first of the three, so that row's grid is on hold. |
| What refutes it | Every new row matches or loses to the control under the verdict rule in the Plan. |
| What changes if it holds | The winning row becomes the candidate to replace the default model, and the last ticket on the map decides the switch and any second training seed. If it is refuted, the three departures stay as ours, and the map's diagnosis records them as measured and harmless on these measures. |
| How far it reaches | Win rate under the behaviour-cloning policy against the 49 distinct top-50 meta teams, slots shuffled, for one checkpoint per masking rate and one training seed. Nothing beyond that. |
| Fields | The model generates all 84: for each of six Pokémon its species, ability, item, four moves, Stat Alignment and six Stat Points. Nothing is copied and nothing is fixed. After sampling, a repair step replaces Stat Alignment to match the moves (`src/ask_grid84.py`). |
| Vocabulary | The tables of every value Regulation M-B allows, used directly, stored in each checkpoint (`team84.regulation_vocab`): 348 species, 201 ability, 150 item, 498 move and 27 Stat Alignment tokens, and Stat Points 0 to 32. The training smoke confirmed the head sizes. Labelled training teams use 189, 126, 106, 309 and 19 of the tokens (`docs/slope-by-epoch.md`). The same for every row. |
| Training teams | The control's table, `lossdata84.npz`, sha256 `02463cbd…`. Labelled teams only: 170,785 train, of which the legal Pokémon construction tables drop 5, and 3,563 validate, a seeded 2% split with seed 0. The tables reject none of the validation teams. Left out: 130,232 unlabelled teams, the 500 test teams, and the held-out meta teams ranked 51 to 100. |
| Scorer | `src/ask_grid84.py`. One battle against each of the 49 distinct top-50 meta teams, behaviour-cloning policy on both sides, slots shuffled, battle seed 7300 plus the sampling seed. `require_shuffled` refuses any other slot order. |
| Search region | No search and no elite rule. Asks 1/24 to 23/24, sampling seeds 1 to 3, 64 teams for each ask and seed. Each cell draws until 64 teams pass Showdown's validator and are not held-out teams, at most 1,280 tries. That is 4,416 teams and 216,384 battles for each row. |

### Settings

Every setting names the paper it follows, or is marked ours. "Ours" means no source in the two yardstick papers. The last column gives the evidence this project has for the value.

Settings that differ between rows:

| Setting | Role | Value | Paper or ours | Evidence here |
| --- | --- | --- | --- | --- |
| Masking rate in training | Scientific | Per-field: species t^16, ability and item t^4, move t^1.5, Stat Alignment and Stat Points t (`SCHED = "16,4,4,1.5,1"`). Linear: every field t (`sched = "1,1,1,1,1"`) | Per-field: the form α = 1 − t^w is MD4 §6, where w is learned for each token value. Fixing w by field type is ours. Linear: MD4 §7, "a linear schedule", and Appendix J.1, "α_t = 1 − t". LLaDA §2.1 and Eq. 3 | Loss only, see What is already known |
| Loss weight on a hidden field | Follows the masking rate | Exponent / t, so 1/t under the linear rate | MD4's table of masking schedule formulas in its appendix: −w/t for the polynomial form, −1/t for the linear. LLaDA Eq. 3 | None needed, it is the masking rate's own weight |
| t told to the model while sampling | Scientific | Uncorrected: 1 − filled/84. Corrected: the t that solves mean(t^exponent) = 1 − filled/84, which is 0.650 at half filled | Ours. Modelled on MD4 Algorithm 2, which samples with the α_t of training. LLaDA gives its network no t (Appendix A.1), and MD4 §3 says the network can be made independent of t | The probe below: the default model barely reads t |
| Decoding order | Scientific | Species-first: one field for each of 84 passes, in a fixed order. Random: 84 passes, and in each pass every hidden field is filled with chance 1/(passes left), so a pass can fill none, one or several | Species-first is ours. Random is LLaDA Algorithm 4 with N = 84, where a filled token is hidden again with chance s/t, and MD4 Algorithm 2 on a uniform grid. Code: `remask84.sample_random`, arm `rand_plain` | See What is already known |
| Lookahead | Bundled with random order | On in Linear, random order. In species-first order it changes no draw, because every species is already filled | Ours | With it off, 1.2% of random-order tries were legal on a per-field model |
| Sampling seed | Nuisance | 1, 2, 3 with offset 7300, the same in every row | Ours | Rows must be compared seed by seed, see Noise floor |

Against Linear, species-first, the row Linear, random order changes three things at once: the order, how many fields a pass fills, and the lookahead. That is LLaDA Algorithm 4 as published plus the one rule random order needs to stay legal. This run cannot separate the three.

Settings whose role is fixed: the same value in every row, all inherited from the control.

| Setting | Value | Paper or ours | Evidence here |
| --- | --- | --- | --- |
| How t is drawn in training | Evenly spaced across the batch, t = (u + i/B) mod 1, then shuffled and kept at 0.001 or more | MD4 Algorithm 1. The shuffle and the floor are ours | None |
| Division by 84 | The summed loss is divided by the 84 fields | LLaDA Algorithm 1 divides by t × L. MD4 Algorithm 1 sums without dividing. The departures list, the map's closed research ticket on where our code departs from the two papers, called this ours. That entry needs correcting | None |
| Time input | The network adds a projection of t to every field | MD4 writes its network as a function of the team and t. Both papers say t is optional | Removing it changed no loss on 2026-10-07 |
| Slot shuffling | A random Pokémon order for each team in each batch | Ours | None |
| Legal Pokémon construction filter | In the loss, where a forbidden value's logit is set to −10,000, and while sampling | Ours | −0.54 loss per field on 2026-10-07 |
| Ask dropout | 0.15 of teams train with no ask | The method is classifier-free guidance, Ho and Salimans, cited by LLaDA. The value is ours | None for the value |
| Label noise | ±1/48 on training labels, fresh for each batch. Validation labels exact | Ours | Error 0.0201 to 0.0070 on the all-teams model (`docs/subset-size-labels.md`) |
| Training teams | Labelled teams only | Ours, owner 2026-11-02 | None |
| Network | 256 wide, 6 layers, 8 heads, dropout 0.1, one learned position vector per field | Ours | The position split was a tie, 2026-11-09 |
| Optimizer | AdamW, learning rate 4e-4, weight decay 0.01, 500 warm-up steps, cosine decay sized to 80 epochs, batch 256, gradient clip 1.0, exponential moving average of the weights 0.999 | Ours | None |
| Training seed | 0 | Ours | Two seeds of one recipe differed by 0.026 pooled win rate, 2026-11-09 |
| Training length | Validation loss after every epoch on the same 4 mask draws, patience 3, cap 80, every epoch saved. **Early-stopping threshold 0**, the control's rule: any new lowest validation loss resets patience | Ours | Under today's code default of 0.001 the control's own curve would have stopped at epoch 47, not 80. The new training must pass 0 |
| Passes | 84, one for each field | LLaDA Appendix B sets "the number of sampling steps equal to the generated length". MD4 §4 warns that a uniform grid "did not perform well" when a step fills several tokens, and suggests a cosine grid. With 84 passes for 84 fields that conflict is small, and absent in species-first order | None |
| Draw | Sampled at temperature 1 | MD4 Algorithm 2 draws from the distribution. LLaDA Algorithm 4 takes the most likely value. The temperature setting is ours | None on this model |
| Guidance | 2 | The mixing rule is LLaDA Appendix A.2, Eq. 16, with guidance = 1 + w. The value 2 is ours | On a sibling model scored against its 39 training opponents, guidance 2, 4 and 8 gave errors 0.0097, 0.0053 and 0.0034, while line-ups in a 5-team sample at ask 10/24 fell from 4.4 to 2.8 (`guidance_sweep_gen65_subset_39.json`, 2026-11-11). Guidance 2 is kept so the control's stored grid can be reused |
| Stat Alignment repair | After sampling | Ours | None |

Two inherited settings are the owner's to change. First, the table `lossdata84.npz` pools labels from run files that include listed-order battles, where the six Pokémon keep their listed order. The owner ruled on 2026-11-07 that only shuffled-slot battles count. The table built that day from shuffled-slot battles alone is not used here, so that the new training differs from the control in the masking rate alone. Second, guidance stays at one value, so each row is one point and not a curve.

### Plan

The primary measure is the ask-versus-got error on asks 1/24 to 17/24. The error on 18/24 to 23/24 is reported apart.

The owner asked for the rows to be read together, so these sit beside the error, every row with the same code:

| Measure | How it is read | The control's value |
| --- | --- | --- |
| Win rate at the top scored asks | Mean got over asks 15/24 to 17/24 | 0.515 |
| Pooled win rate | Mean got over all 23 asks | 0.371 |
| First-try legality | Share of sampled teams the validator accepts before any redraw | 100% |
| Line-ups | Distinct line-ups among the grid's 4,416 teams | 1,437 |
| New line-ups | Share of teams whose line-up is absent from the training teams, and from the line-ups of earlier search runs and tournament records (`src/lineup_reference.py`) | 26.6% and 73.1% |
| Effective species, top-species share | Mean over random 5-team subsets of each cell | 14.9 and 0.135 |
| Elite line-up yield | Line-ups with a team at or above the elite cut that still reach it in a fresh re-battle of 4 games against each opponent. The elite cut is 0.531, the mean win rate of a top-50 meta team against the other 49. The script is `line_up_yield.py`, which so far exists only in `~/vgc-data/line_up_deletion_84/run/` | Not yet measured. 176 line-ups to re-battle |
| Training-versus-validation gap | The same loss function on 3,563 training and 3,563 validation teams (`~/vgc-data/validation_loss_accuracy/run/validation_accuracy.py`), each model under its own masking rate | +0.0115 ± 0.0063 at epoch 80 |
| Validation loss under both masking rates | Each checkpoint scored under the per-field rate and under the linear rate. A loss is only comparable with another measured under the same rate | 1.209 under its own per-field rate, 1.404 under the linear rate |

The question is a Pareto question, with the ask-versus-got error on one axis and line-ups on the other. Guidance is fixed, so each row is one point. The strongest previous frontier among labelled-teams models with label noise is the control itself at 0.0053 and 1,437 line-ups. Epoch 21, at 0.0066 and 1,325, and the hindsight-relabeled fine-tune, at 0.0055 and 1,292, sit inside it. A row moves outside the frontier by a lower error with no fewer line-ups, or by the same error with more line-ups.

The verdict for each new row, against the control and paired by sampling seed:

- It **wins** if its error is at least 0.001 lower and the 95% interval of the difference sits below zero.
- It **loses** if that interval sits above zero.
- It **matches** otherwise.

For the two linear rows, a win or a loss smaller than 0.002 is reported as inside training luck, because two trainings of one recipe have differed by up to 0.0019 in error. A row that wins while its pooled win rate falls by more than 0.026, or its line-ups fall, is a trade and is reported as one. Which default model follows, and whether a row earns a second training seed, is the owner's call on the map's last ticket.

The linear model is read at the epoch early stopping keeps, the one with the lowest validation loss. For the control that was epoch 80. Epochs 34 and 55 are also read with sampling seed 1 and species-first decoding, beside the control's 0.0059 and 0.0061 at those epochs and 0.0060 at epoch 80 with the same seed. An earlier epoch replaces the kept one only if a 3-seed reading of it is at least 0.001 lower, with the interval of the difference clear of zero. That is the rule the glossary sets for replacing the default model. Such a swap would be written under Deviations.

The other fixed parts of the plan:

| Part | Value |
| --- | --- |
| Smallest effect worth acting on | 0.001 of error, the gap the glossary requires before a model replaces the default |
| Comparisons planned | Each new row against the control. Linear, random order against Linear, species-first |
| Baselines | The control. The unguided samples of each battled row. The scorer's two controls: top-50 meta teams score near 0.50, and teams drawn uniformly at random from the legal ones score near 0.02 |
| Stopping rule | One 3-seed grid for each row, with no re-runs to chase a result |
| Slope of got on ask | Not reported, owner 2026-11-05 |

### Expectations

Written before any result, to be scored afterwards.

| Row | Expected verdict against the control | Chance | Why |
| --- | --- | --- | --- |
| Per-field, corrected t | Matches | 90% | The probe below: the corrected t moves the model's drawing distributions by 0.005 at most |
| Linear, species-first | Matches, or differs by less than training luck | 60%, with 30% for a loss beyond luck and 10% for a win beyond it | On 2026-10-07 the linear rate cost 0.038 in species-first loss, a small gap. 91% of the control's error sits at asks 13/24 to 17/24, where got stops near 0.53, and the masking rate gives no reason for that ceiling to lift |
| Linear, random order | Loses to Linear, species-first, with an error less than 2.8 times as large. On a per-field model random order's error was 2.8 times species-first's. First-try legality above 84%, and more line-ups | 65% | Linear training shows the model every mix of hidden and visible fields, so moves drawn before their species are no longer new to it |

The largest surprise would be a linear row that wins by 0.002 or more.

## Probes

Script: `src/masking_rate_preflight.py`, battle-free. Results: `~/vgc-data/results/masking_rate_rows_preflight.json`, run 2026-11-11 from this branch.

| Probe | Result | Pass |
| --- | --- | --- |
| Ceiling for the error | Training teams battled again miss the line by 0.0038, interval 0.0033 to 0.0047 (`docs/training-teams-line.md`). The control scores 0.0053, so the best case is 0.0015 better, above the 0.001 bar. Asks 1/24 to 12/24 alone already score 0.0007, so a row can only gain where got stops rising, at asks 13/24 to 17/24. Figure: [got against ask for the default model](figures/masking_rate_rows/02_error_by_ask.html) | yes, narrowly |
| Ceiling for Per-field, corrected t | **Failed.** See the next section | **no** |
| Hidden share | With every exponent 1, the share of fields hidden at t is t on every field, within 0.007 over 40,000 draws at seven values of t. The loss weight becomes 1/t | yes |
| Corrected t values | The rule gives 0.650 at half filled and matches the corrected-t ticket's table within 0.00005. Under the linear rate it returns 1 − filled/84 exactly, so the linear rows need no correction | yes |
| Validation loss under both rates | The default model's validation loss reproduces its logged 1.2091 under the per-field rate, and reads 1.4039 under the linear rate. The linear model's loss is to be compared with 1.4039, never with 1.2091 | yes |
| Training smoke | A fresh network's cross-entropy with every field hidden sits within 20% of ln K, where K is the number of values the field can take, for every field type: species 5.83 against 5.85, ability 5.72 against 5.30, item 4.97 against 5.01, move 6.38 against 6.21, Stat Alignment 3.57 against 3.30, Stat Points 3.76 against 3.50. Overfitting 32 teams for 300 epochs takes validation loss from 3.624 to 1.865 with the linear rate and from 3.328 to 1.602 with the per-field rate, both past the 40% bar | yes |
| Recipe replay | One epoch of the control's own recipe, run with today's code and table, lands on the losses the control logged at epoch 1: training 1.7917 against 1.7918, validation 1.47690 against 1.47691. Same table by sha256 and the same 170,780 teams. The bar was 0.01. So the new training differs from the control in the masking rate alone. The epoch took 4.1 minutes | yes |
| Round trip, coverage, read teams, controls, separation, condition, copies | The control's stored preflight: 20 of 20 teams round-trip; 5 teams dropped by the tables; top-50 teams 0.50 and uniform-legal teams 0.02; 0 of 50 held-out teams among the training teams; all teams change between asks 0.1 and 0.7 at a fixed seed; 0 of 256 teams copy a training team. For each new row, `ask_grid84.py preflight`, the ask grid's own probe step, runs them with that row's decoder before its grid, and a failure stops the row | control yes; new rows before each grid |
| Invariance, repeat | Skipped. The scorer is unchanged and every battle shuffles slots | not run |

The ask grid's probe step samples species-first today. The code ticket must make it use each row's decoder, or the probes of Linear, random order would test the wrong sampler.

Before the grid of Linear, random order, a sampling-only probe also reports its first-try legality with the lookahead off: 384 tries at asks 4/24, 8/24, 12/24 and 16/24 for each of 3 seeds, as on 2026-11-07. No battles.

### Per-field, corrected t is on hold

The probe asked how far the default model's drawing distribution moves when a pass is told the corrected t and not 1 − filled/84, on the same partly filled teams. The bar, set before the run, was a mean total variation of 0.01 on at least one field type. Total variation is half the summed absolute difference between two distributions, so 0 means identical and 1 means no overlap.

Mean total variation over asks 6/24, 12/24 and 17/24, 256 teams each, guidance 2:

| Rule the pass is told | Species | Ability and item | Moves | Stat Alignment | Stat Points |
| --- | --- | --- | --- | --- | --- |
| Corrected t | 0.0007 | 0.0010 | 0.0025 | 0.0040 | 0.0030 |
| Always t = 1 | 0.0011 | 0.0019 | 0.0067 | 0.0149 | 0.0275 |
| Always t = 1/84 | 0.0369 | 0.0109 | 0.0123 | 0.0118 | 0.0066 |

Figure: [The default model barely reads the t it is told](figures/masking_rate_rows/01_t_sensitivity.html), built by `src/masking_rate_preflight_figures.py`.

The corrected t never reaches the bar. Its largest single reading is 0.0050, on Stat Alignment at ask 6/24. Even telling the model a t that is wrong at every pass moves a distribution by 0.044 at most at any single ask. With the same sampling seed, the corrected t changes 1% to 2% of fields and 1 line-up in 768 teams.

This agrees with both papers. MD4 §3 says the network can be made independent of t, because the right answer for a partly hidden team does not depend on t. LLaDA Appendix A.1 gives its network no t at all. It also agrees with this project's run of 2026-10-07 that removed the time input and lost nothing.

So the sampler's t is a departure from the papers that has no measurable effect on this model. Two ways forward, for the owner:

| Choice | Effect |
| --- | --- |
| Skip the grid of Per-field, corrected t (recommended) | The results table reports the row from this probe: same line-ups, 1% to 2% of fields changed. Saves 216,384 battles, about 46 minutes, and the row cannot show a difference anyway |
| Run it as planned | A fourth battled row, expected to match the control. Any gap it showed would be battle noise, which the Noise floor section sizes at 0.00034 |

## Noise floor and budget

All figures are measured on the control's stored grid unless a source is named.

| Quantity | Value | Source |
| --- | --- | --- |
| Error of one 3-seed row | 0.0053, standard deviation 0.00024 | Bootstrap of teams within each ask and seed, 4,000 draws |
| Difference between two rows | Standard deviation 0.00034 from that bootstrap, and 0.00037 from the seed-by-seed differences of 10 pairs of stored grids of labelled-teams models | Both put the smallest detectable gap near 0.001 at 5% false positives and 80% power |
| Sampling seed effect | Seed 1 reads the highest error on 15 of 20 stored 3-seed grids, and seed 3 the lowest on 13. One seed's error has a standard deviation of 0.0011 across seeds, against 0.00045 from the bootstrap | Rows are compared seed by seed, never one row's seed 1 against another's seed 3. The cause is not established |
| Training luck on error | 0.0006 between two seeds with label noise, and 0.0019 between two trainings without it, all at 21 epochs | `docs/position-embedding-split.md` |
| Training luck on pooled win rate | 0.026 between the same two seeds, with 0.96 effective species and 24 line-ups | Same |
| Win rate at asks 15/24 to 17/24 | 0.515, standard deviation 0.0038 | Bootstrap |
| Pooled win rate | 0.371, standard deviation 0.0015 | Bootstrap |

The budget can detect the 0.001 effect between two samplers on one checkpoint. It cannot for the two linear rows: one training seed leaves up to 0.0019 of error and 0.026 of win rate to luck, which is why those rows are exploratory.

Cost, all on the laptop, queued detached behind other sessions' jobs after `python3 ~/vgc-data/whats_running.py`:

| Step | Battles | Time |
| --- | --- | --- |
| Train the linear model, 80 epochs at about 4 minutes each | none | about 5.4 hours |
| Grid for Linear, species-first | 216,384 | about 46 minutes |
| Grid for Linear, random order | 216,384 | about 46 minutes, plus slower sampling |
| Linear model at epochs 34 and 55, sampling seed 1 | 144,256 | about 31 minutes |
| Unguided baselines, 192 teams for each of the control and the two linear rows | 28,224 | about 6 minutes |
| Elite line-up yield re-battles for the control and both linear rows | about 105,000 | about 22 minutes |
| Grid for Per-field, corrected t, only if the owner keeps it. Its unguided baseline would add 9,408 | 216,384 | about 46 minutes |

That is about 710,000 battles and 2.5 hours of battling after the training, or 936,000 and 3.3 hours with the grid of Per-field, corrected t and its unguided baseline. No cloud compute.

## What the later tickets take from this

| Ticket | What this document fixes for it |
| --- | --- |
| [Add the linear masking rate and the corrected t to the 84-field code, with tests](https://github.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/issues/295) | A `train84.py` arm that passes `sched="1,1,1,1,1"` and `early_stopping_threshold=0.0` and saves every epoch. A sampler option for the t rule, tested against `masking_rate_preflight.corrected_t`. Ask-grid entries for the four code IDs, a preflight that uses each row's decoder, and the unguided cells. Random order is `remask84.RandomDecodingArm(model, "rand_plain", lookahead=True)` |
| [Train the 84-field model with the linear masking rate](https://github.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/issues/296) | Run `masking_rate_preflight.py recipe_replay training_smoke` from the frozen copy first. Post the validation curve, the training-versus-validation gap, and the validation loss of both checkpoints under both masking rates |
| [Sample and battle the four rows and post the results table](https://github.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/issues/297) | For each row: preflight, the 3-seed grid, battles, report, then its unguided cells and its elite line-up yield. Each row is posted the moment it lands |

## Deviations

None yet.
