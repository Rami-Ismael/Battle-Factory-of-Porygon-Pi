# Masking-rate rows: per-field against linear masking rate, the t the sampler tells the model, and the decoding order

Pre-registration, written 2026-11-11 before the new training and before any of the new grids. Ticket: [Pre-register the four masking-rate rows](https://github.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/issues/294), on the map [Masking-rate map: per-field versus linear masking rate on the 84-field model, and where our code departs from MD4 and LLaDA](https://github.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/issues/291).

Mode: **exploratory** for the two linear rows, because one training seed cannot separate the masking rate from training luck. A run is confirmatory when its plan is fixed before any result and its budget can detect the effect. Per-field, corrected t compares two samplers on one checkpoint under such a plan, so it would be confirmatory for that checkpoint if its grid runs.

**Changed 2026-11-11, before any training or battle, by two owner decisions.** Both trainings use the shuffled-only table, so the control is a new per-field training and no longer today's default model. Every battled row is sampled at guidance 2, 4 and 8. See [Changes before any result](#changes-before-any-result).

**One probe failed before launch.** Today's default model barely reads the t it is told, so on it the row Per-field, corrected t cannot move a result by the smallest effect worth acting on. The same probe now decides the row: it is rerun on the new per-field model, and the row is battled only if the probe reaches its bar. See [Per-field, corrected t is on hold](#per-field-corrected-t-is-on-hold).

## Terms

An ask is the win rate the model is told to aim for. Got is the win rate its teams then score. A line-up is a team's six species. The ask-versus-got error is the mean squared error between each ask and the mean got at that ask, on asks 1/24 to 17/24 (`src/ask_line.py`). The line is got = ask, and near the line means an error under 0.001.

Win rate means one battle against each of the 49 distinct top-50 meta teams, with the behaviour-cloning policy on both sides and slots shuffled every battle. The top-50 meta teams are the project's fixed list of 50 real teams for Regulation M-B, the game's current rule set. One team is on the list twice, which leaves 49. The behaviour-cloning policy is the battle player, a network trained to imitate human battle decisions. Slots are the order of a team's six Pokémon.

The masking rate is the chance that a field is hidden at time t during training. Time t runs from 0, nothing hidden, to 1, everything hidden. It is a dial for how much is hidden, not a step count. The network takes t as an input, so the sampler must also tell it a t at every pass. A pass is one run of the network while sampling. The decoding order is the order in which the sampler fills fields.

Showdown is the battle simulator, and its validator accepts or rejects a team. First-try legality is the share of sampled teams the validator accepts before any redraw. The legal Pokémon construction filter gives zero odds to every value the game's rules forbid next to the fields already on the team. The lookahead is a further sampler rule: a value is allowed only if every Pokémon whose species is still hidden keeps at least one species it could be. Stat Alignment is the project's name for a Pokémon's nature, such as Adamant. The remaining project terms follow `notes/GLOSSARY.md`.

A probe is a quick check run before the expensive steps, here without a single battle. A probe's numbers are what that check measured. Guidance is the sampler's dial for how hard it pushes toward the ask: 1 is no push, and higher values push harder.

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
| Per-field, uncorrected t. The control | `sched_species_first` | New training on the shuffled-only table, `stat84_shuffled_noise_per_field_es3_cap80` | 1 − filled/84 | Species-first |
| Per-field, corrected t | `sched_species_first_t` | The same per-field checkpoint. No training | The t whose expected hidden share in training equals the share still hidden | Species-first |
| Linear, species-first | `sched_linear_species_first` | New training on the same table, `stat84_shuffled_noise_linear_es3_cap80` | 1 − filled/84 | Species-first |
| Linear, random order | `sched_linear_random` | The same linear checkpoint | 1 − filled/84 | Random |

Every battled row is sampled at guidance 2, 4 and 8. Each is also sampled once with no ask, 64 teams for each of the 3 sampling seeds, and these unguided baselines carry the row's code ID with `_unguided` added. The per-field model and the linear model are the two new checkpoints.

Today's default model, epoch 80 of the labelled-teams training on the older table, is a reference and not the control. Its guidance 2 grid exists as `gen84_noise_byepoch_ep80`. It differs from the control in its training table, so no verdict is read against it.

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
| Training teams | The shuffled-only table, `lossdata84_shuffled_only.npz`, sha256 `8aca232e…`, chosen by the owner on 2026-11-11. Every label comes from shuffled-slot battles alone, at least 49 for each team. Labelled teams only: 214,598 train and 4,380 validate, a seeded 2% split with seed 0. Both trainings use the same rows, and this table is the default in the training code. Left out: 10 never-battled teams, the 500 test teams, the held-out meta teams ranked 51 to 100, and every team whose labels came only from listed-order battles. Against the older table of today's default model, only 783 labelled teams are in both, the mean label falls from 0.427 to 0.284, and the teams labelled nearest asks 13/24 to 17/24 fall from 46,013 to 20,732. |
| Scorer | `src/ask_grid84.py`. One battle against each of the 49 distinct top-50 meta teams, behaviour-cloning policy on both sides, slots shuffled, battle seed 7300 plus the sampling seed. `require_shuffled` refuses any other slot order. |
| Search region | No search and no elite rule. Asks 1/24 to 23/24, sampling seeds 1 to 3, 64 teams for each ask and seed, at each of guidance 2, 4 and 8. Each cell draws until 64 teams pass Showdown's validator and are not held-out teams, at most 1,280 tries. That is 4,416 teams and 216,384 battles for each row at each guidance value. |

### Settings

Every setting names the paper it follows, or is marked ours. "Ours" means no source in the two yardstick papers. The last column gives the evidence this project has for the value.

Settings that vary:

| Setting | Role | Value | Paper or ours | Evidence here |
| --- | --- | --- | --- | --- |
| Masking rate in training | Scientific | Per-field: species t^16, ability and item t^4, move t^1.5, Stat Alignment and Stat Points t (`SCHED = "16,4,4,1.5,1"`). Linear: every field t (`sched = "1,1,1,1,1"`) | Per-field: the form α = 1 − t^w is MD4 §6, where w is learned for each token value. Fixing w by field type is ours. Linear: MD4 §7, "a linear schedule", and Appendix J.1, "α_t = 1 − t". LLaDA §2.1 and Eq. 3 | Loss only, see What is already known |
| Loss weight on a hidden field | Follows the masking rate | Exponent / t, so 1/t under the linear rate | MD4's table of masking schedule formulas in its appendix: −w/t for the polynomial form, −1/t for the linear. LLaDA Eq. 3 | None needed, it is the masking rate's own weight |
| t told to the model while sampling | Scientific | Uncorrected: 1 − filled/84. Corrected: the t that solves mean(t^exponent) = 1 − filled/84, which is 0.650 at half filled | Ours. Modelled on MD4 Algorithm 2, which samples with the α_t of training. LLaDA gives its network no t (Appendix A.1), and MD4 §3 says the network can be made independent of t | The probe below: the default model barely reads t |
| Decoding order | Scientific | Species-first: one field for each of 84 passes, in a fixed order. Random: 84 passes, and in each pass every hidden field is filled with chance 1/(passes left), so a pass can fill none, one or several | Species-first is ours. Random is LLaDA Algorithm 4 with N = 84, where a filled token is hidden again with chance s/t, and MD4 Algorithm 2 on a uniform grid. Code: `remask84.sample_random`, arm `rand_plain` | See What is already known |
| Lookahead | Bundled with random order | On in Linear, random order. In species-first order it changes no draw, because every species is already filled | Ours | With it off, 1.2% of random-order tries were legal on a per-field model |
| Sampling seed | Nuisance | 1, 2, 3 with offset 7300, the same in every row | Ours | Rows must be compared seed by seed, see Noise floor |
| Guidance | Swept in every row, to trace its curve | 2, 4 and 8, owner 2026-11-11 | The mixing rule is LLaDA Appendix A.2, Eq. 16, with guidance = 1 + w. The three values are ours. MD4 uses no guidance | On a sibling model scored against its 39 training opponents, guidance 2, 4 and 8 gave errors 0.0097, 0.0053 and 0.0034, while line-ups in a 5-team sample at ask 10/24 fell from 4.4 to 2.8 (`guidance_sweep_gen65_subset_39.json`) |

Against Linear, species-first, the row Linear, random order changes three things at once: the order, how many fields a pass fills, and the lookahead. That is LLaDA Algorithm 4 as published plus the one rule random order needs to stay legal. This run cannot separate the three.

Settings whose role is fixed: the same value in every row, all inherited from today's default model.

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
| Training length | Validation loss after every epoch on the same 4 mask draws, patience 3, cap 80, every epoch saved. **Early-stopping threshold 0**, the rule today's default model trained under: any new lowest validation loss resets patience | Ours | Under today's code default of 0.001 the default model's own curve would have stopped at epoch 47, not 80. Both new trainings must pass 0 |
| Passes | 84, one for each field | LLaDA Appendix B sets "the number of sampling steps equal to the generated length". MD4 §4 warns that a uniform grid "did not perform well" when a step fills several tokens, and suggests a cosine grid. With 84 passes for 84 fields that conflict is small, and absent in species-first order | None |
| Draw | Sampled at temperature 1 | MD4 Algorithm 2 draws from the distribution. LLaDA Algorithm 4 takes the most likely value. The temperature setting is ours | None on this model |
| Stat Alignment repair | After sampling | Ours | None |

Two settings were the owner's to choose, and both were chosen on 2026-11-11: the training table and the guidance values. See [Changes before any result](#changes-before-any-result).

### Plan

The primary measure is the ask-versus-got error on asks 1/24 to 17/24. The error on 18/24 to 23/24 is reported apart.

The owner asked for the rows to be read together, so these sit beside the error, every row with the same code:

| Measure | How it is read | Today's default model at guidance 2, for reference |
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

The question is a Pareto question, with the ask-versus-got error on one axis and line-ups on the other. Guidance 2, 4 and 8 trace a three-point curve for each row. The frontier to beat is the control's curve. A row moves outside it when one of its points has a lower error than every control point with at least as many line-ups, or more line-ups than every control point with an error at least as low. Today's default model sits on the chart as a reference point at 0.0053 and 1,437 line-ups.

The verdict for each new row is read at each guidance value, against the control at the same guidance and paired by sampling seed:

- It **wins** if its error is at least 0.001 lower and the 95% interval of the difference sits below zero.
- It **loses** if that interval sits above zero.
- It **matches** otherwise.

That gives three verdicts for each row. Three chances raise the odds of a false win, so a row counts as winning when it wins at two or more guidance values and loses at none. One win among three is reported as weak.

For the two linear rows, a win or a loss smaller than 0.002 is reported as inside training luck, because two trainings of one recipe have differed by up to 0.0019 in error. A row that wins while its pooled win rate falls by more than 0.026, or its line-ups fall, is a trade and is reported as one. Which default model follows, and whether a row earns a second training seed, is the owner's call on the map's last ticket.

Each new model is read at the epoch early stopping keeps, the one with the lowest validation loss. No other epoch is battled.

The other fixed parts of the plan:

| Part | Value |
| --- | --- |
| Smallest effect worth acting on | 0.001 of error, the gap the glossary requires before a model replaces the default |
| Comparisons planned | Each new row against the control at the same guidance. Linear, random order against Linear, species-first |
| Baselines | The control. The unguided samples of each battled row. The scorer's two controls: top-50 meta teams score near 0.50, and teams drawn uniformly at random from the legal ones score near 0.02 |
| Stopping rule | One 3-seed grid for each row at each guidance value, with no re-runs to chase a result |
| Slope of got on ask | Not reported, owner 2026-11-05 |

### Expectations

Written before any result, to be scored afterwards.

| Row | Expected verdict against the control | Chance | Why |
| --- | --- | --- | --- |
| Per-field, uncorrected t, the control | An error above the 0.0053 of today's default model at guidance 2, falling as guidance rises | 70% | The shuffled-only table holds fewer than half as many teams near asks 13/24 to 17/24. The two 21-epoch models already trained on it scored 0.0075 and 0.0081 |
| Per-field, corrected t | No battle grid: the probe stays under its bar on the new per-field model | 90% | On today's default model the corrected t moved the drawing distributions by 0.005 at most |
| Linear, species-first | Matches, or differs by less than training luck | 60%, with 30% for a loss beyond luck and 10% for a win beyond it | On 2026-10-07 the linear rate cost 0.038 in species-first loss, a small gap. 91% of the default model's error sits at asks 13/24 to 17/24, where got stops near 0.53, and the masking rate gives no reason for that ceiling to lift |
| Linear, random order | Loses to Linear, species-first, with an error less than 2.8 times as large. On a per-field model random order's error was 2.8 times species-first's. First-try legality above 84%, and more line-ups | 65% | Linear training shows the model every mix of hidden and visible fields, so moves drawn before their species are no longer new to it |

The largest surprise would be a linear row that wins by 0.002 or more.

## Probes

Script: `src/masking_rate_preflight.py`, battle-free. Results: `~/vgc-data/results/masking_rate_rows_preflight.json`, run 2026-11-11 from this branch. The first run used today's default model and its older table. After the owner chose the shuffled-only table, two probes were run again on it, marked below.

| Probe | Result | Pass |
| --- | --- | --- |
| Ceiling for the error | Training teams battled again miss the line by 0.0038, interval 0.0033 to 0.0047 (`docs/training-teams-line.md`). Today's default model scores 0.0053, so the best case is 0.0015 better, above the 0.001 bar. Asks 1/24 to 12/24 alone already score 0.0007, so a row can only gain where got stops rising, at asks 13/24 to 17/24. Figure: [got against ask for the default model](figures/masking_rate_rows/02_error_by_ask.html) | yes, narrowly, on the older table. Not measured on the shuffled-only table, whose labels are steadier: at least 49 shuffled-slot battles each, reliability 0.89 against 0.83 |
| Ceiling for Per-field, corrected t | **Failed** on today's default model, and failed again on a 21-epoch per-field model trained on the shuffled-only table, where the largest reading was 0.008. See the next section | **no** |
| Hidden share | With every exponent 1, the share of fields hidden at t is t on every field, within 0.007 over 40,000 draws at seven values of t. The loss weight becomes 1/t | yes |
| Corrected t values | The rule gives 0.650 at half filled and matches the corrected-t ticket's table within 0.00005. Under the linear rate it returns 1 − filled/84 exactly, so the linear rows need no correction | yes |
| Validation loss under both rates | The default model's validation loss reproduces its logged 1.2091 under the per-field rate, and reads 1.4039 under the linear rate. The linear model's loss is to be compared with 1.4039, never with 1.2091 | yes |
| Training smoke | A fresh network's cross-entropy with every field hidden sits within 20% of ln K, where K is the number of values the field can take, for every field type: species 5.83 against 5.85, ability 5.72 against 5.30, item 4.97 against 5.01, move 6.38 against 6.21, Stat Alignment 3.57 against 3.30, Stat Points 3.76 against 3.50. Overfitting 32 teams for 300 epochs takes validation loss from 3.624 to 1.865 with the linear rate and from 3.328 to 1.602 with the per-field rate, both past the 40% bar Run again on the shuffled-only table: 3.663 to 1.917 with the linear rate and 3.386 to 1.631 with the per-field rate, with every field type within 20% of ln K | yes |
| Recipe replay | One epoch of the default model's own recipe, run with today's code and its older table, lands on the losses that model logged at epoch 1: training 1.7917 against 1.7918, validation 1.47690 against 1.47691. Same table by sha256 and the same 170,780 teams. The bar was 0.01. So today's code trains as it did then. The two new trainings share one code path and one table, and differ in the masking rate alone. The epoch took 4.1 minutes | yes |
| Round trip, coverage, read teams, controls, separation, condition, copies | The stored probe step of today's default model: 20 of 20 teams round-trip; 5 teams dropped by the tables; top-50 teams 0.50 and uniform-legal teams 0.02; 0 of 50 held-out teams among the training teams; all teams change between asks 0.1 and 0.7 at a fixed seed; 0 of 256 teams copy a training team. For each row, the new control included, `ask_grid84.py preflight`, the ask grid's own probe step, runs them with that row's decoder before its grid, and a failure stops the row | default model yes; every row before its grids |
| Invariance, repeat | Skipped. The scorer is unchanged and every battle shuffles slots | not run |

The ask grid's probe step samples species-first today. The code ticket must make it use each row's decoder, or the probes of Linear, random order would test the wrong sampler.

Before the grid of Linear, random order, a sampling-only probe also reports its first-try legality with the lookahead off: 384 tries at asks 4/24, 8/24, 12/24 and 16/24 for each of 3 seeds, as on 2026-11-07. No battles.

### Per-field, corrected t is on hold

The probe asked how far the default model's drawing distribution moves when a pass is told the corrected t and not 1 − filled/84, on the same partly filled teams. The bar, set before the run, was 0.01 for one field type at one ask: the mean total variation over that field type's passes and the 256 teams. One field type reaching it at any of the three asks is enough. Total variation is half the summed absolute difference between two distributions, so 0 means identical and 1 means no overlap.

For reading, the table averages the three asks, 6/24, 12/24 and 17/24, with 256 teams each at guidance 2. The verdict uses each ask's own reading:

| Rule the pass is told | Species | Ability and item | Moves | Stat Alignment | Stat Points |
| --- | --- | --- | --- | --- | --- |
| Corrected t | 0.0007 | 0.0010 | 0.0025 | 0.0040 | 0.0030 |
| Always t = 1 | 0.0011 | 0.0019 | 0.0067 | 0.0149 | 0.0275 |
| Always t = 1/84 | 0.0369 | 0.0109 | 0.0123 | 0.0118 | 0.0066 |

Figure: [The default model barely reads the t it is told](figures/masking_rate_rows/01_t_sensitivity.html), built by `src/masking_rate_preflight_figures.py`.

The corrected t never reaches the bar. Its largest single reading is 0.0050, on Stat Alignment at ask 6/24. Even telling the model a t that is wrong at every pass moves a distribution by 0.044 at most at any single ask. With the same sampling seed, the corrected t changes 1% to 2% of fields and 1 line-up in 768 teams.

This agrees with both papers. MD4 §3 says the network can be made independent of t, because the right answer for a partly hidden team does not depend on t. LLaDA Appendix A.1 gives its network no t at all. It also agrees with this project's run of 2026-10-07 that removed the time input and lost nothing.

So the sampler's t is a departure from the papers that has no measurable effect on today's default model.

**How the row resolves.** The probe decides, on the model the row would use. The owner agreed on 2026-11-11. After the new per-field model trains, the same probe runs on it:

| Probe on the new per-field model | What happens to Per-field, corrected t |
| --- | --- |
| Under 0.01 on every field type at every ask | No battle grid. The results table reports the probe's numbers for the row: the largest total variation, the share of fields changed and the line-ups changed |
| 0.01 or more on any field type at any ask | The row is battled at guidance 2, 4 and 8 like the others, 649,152 battles and about 2.3 hours |

A preview of that rule exists. On the 21-epoch per-field model already trained on the shuffled-only table, `stat84_pos_per_field_s0_noise_ep21`, the corrected t moved the drawing distributions by 0.008 at most, changed 1% to 2% of fields and 3 line-ups in 768. That is under the bar, so the row is expected to need no battles.

## Noise floor and budget

All figures are measured on the stored guidance 2 grid of today's default model unless a source is named. They size the noise the new rows will carry. The new control gets its own readings.

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
| Train the per-field model on the shuffled-only table, up to 80 epochs | none | about 6.7 hours |
| Train the linear model on the same table | none | about 6.7 hours |
| Grids for the control, Linear, species-first and Linear, random order, each at guidance 2, 4 and 8 | 1,947,456 | about 6.9 hours, plus slower sampling in random order |
| Unguided baselines, 192 teams for each of the three | 28,224 | about 6 minutes |
| Elite line-up yield re-battles for the nine grids | about 315,000 | about 67 minutes |
| Grids for Per-field, corrected t at guidance 2, 4 and 8, only if its probe reaches the bar | 649,152 | about 2.3 hours |

That is about 13.4 hours of training and 2.3 million battles, about 8.1 hours, or 2.9 million and 10.4 hours if Per-field, corrected t is battled. The training time scales the default model's 320.6 minutes by the larger table. No cloud compute.

## What the later tickets take from this

| Ticket | What this document fixes for it |
| --- | --- |
| [Add the linear masking rate and the corrected t to the 84-field code, with tests](https://github.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/issues/295) | Five build items. A training arm for each masking rate on the shuffled-only table, with `early_stopping_threshold=0.0`, every epoch saved, and `sched="1,1,1,1,1"` for the linear one. A sampler option for the t rule, tested against `masking_rate_preflight.corrected_t`. Guidance as an ask-grid setting of 2, 4 and 8, where today it is the constant 2.0. Ask-grid entries for the four code IDs, a probe step that uses each row's decoder, and the unguided cells. Random order as `remask84.RandomDecodingArm(model, "rand_plain", lookahead=True)` |
| [Train the 84-field model with the linear masking rate](https://github.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/issues/296) | Two trainings on the shuffled-only table, per-field then linear. First `masking_rate_preflight.py training_smoke`, which reads the shuffled-only table. After the per-field training, `MRP_MODEL=<its checkpoint> masking_rate_preflight.py t_sensitivity`, which decides the grid of Per-field, corrected t. Post both validation curves, both training-versus-validation gaps, and the validation loss of both checkpoints under both masking rates |
| [Sample and battle the four rows and post the results table](https://github.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/issues/297) | For each row at each guidance value: the probe step, the 3-seed grid, battles, report and elite line-up yield. The unguided cells once for each row. Each grid is posted the moment it lands |

## Figures

Storyboarded 2026-11-11 with `storyboard-figures`, before any training or battle. Each frame is a sketch with no real numbers yet, except frame 5, whose first reading exists.

| Frame | Sentence the reader takes | Expected | Surprise |
| --- | --- | --- | --- |
| 1. Error against line-ups, one three-point curve for each row, guidance 2, 4 and 8 | A row is better when its curve sits lower and further right than the control's. | Linear, species-first lies on the control's curve. Linear, random order sits to the right and higher. | A linear curve wholly below the control's. |
| 2. Got against ask for the control and Linear, species-first, one panel for each guidance | The linear masking rate lifts got above 12/24, or it doesn't. | The two lines overlap, and both stop rising at the same got. | The linear line keeps rising where the control stops. |
| 3. Random order's error as a multiple of species-first's, per-field model beside linear model | Random order stops failing once training shows every mix of hidden and visible fields. | The multiple falls from the 2.8 measured on a per-field model, and stays above 1. | It stays near 2.8, so the order itself is the problem. Or it falls below 1. |
| 4. Validation loss of both new models under both masking rates, a 2-by-2 grid | Each model is best under the masking rate it trained on. The question is how much the linear model gives up on the species-first path. | The linear model is lower under the linear rate and slightly higher under the per-field rate. | The linear model is lower under both. |
| 5. How far the drawing distribution moves when a pass is told another t, on the new per-field model | The model barely reads t. | Under 0.01 on every field type, as on today's default model. | 0.01 or more, and then Per-field, corrected t gets its grids. |

Every planned run feeds a frame: the two trainings feed frame 4, the nine grids feed frames 1 to 3, and the probe feeds frame 5. The unguided baselines and the elite line-up yield feed the results table beside frame 1. The readings of the linear model at epochs 34 and 55 fed no frame and were dropped. Sketches: [storyboard](figures/masking_rate_rows/storyboard.html).

Six one-idea explainers sit beside it, built by `src/masking_rate_explainers.py`: [what a probe is](figures/masking_rate_rows/explain_1_probes.html), [what the corrected t corrects](figures/masking_rate_rows/explain_2_corrected_t.html), [the plan after the two decisions](figures/masking_rate_rows/explain_3_plan.html), [the two training tables](figures/masking_rate_rows/explain_4_tables.html), [what the code ticket builds](figures/masking_rate_rows/explain_6_code_ticket.html) and [the departures matrix](figures/masking_rate_rows/explain_7_departures.html).

## Changes before any result

Made 2026-11-11, after the first version of this document and before any training or battle of the experiment.

| Owner's words | What changed |
| --- | --- |
| "Yes each row get three guidance value" | Every battled row is sampled at guidance 2, 4 and 8, where the first version kept guidance 2. Each row is now a three-point curve, the verdict is read at each guidance, and the grids triple. |
| "I perfer shuffled only because it only represent a win rate that is not bias by the error of the rl policy" | Both trainings use the shuffled-only table. Today's default model trained on the older table, so it can no longer be the control: a table difference would mix with the masking rate. The control is a new per-field training on the shuffled-only table, about 6.7 more hours, and today's default model stays as a reference. |
| "How do I resolve the per-field corrected", then "I am fine with a prob" | The probe decides, as written under Per-field, corrected t is on hold. The owner approved the rule. |
| "delete any training example in where the win rate did go through shuffle at all. Because, shuffle give use more accurate representation of the true win rate" and "please use only shuffle example. The default bevahior should be shuffling" | The shuffled-only table is now the default training table in code. An arm that reproduces a model trained on the older table stops unless `--older-table` is passed (`src/train84.py`, tested in `scripts/test_train84_tables.py`). A new ask-grid row is compared with the shuffled-only table unless it names another. The table itself was audited: 218,978 labelled teams, each with at least 49 shuffled-slot battles and none carrying a listed-order label. The 173,544 teams whose labels came only from listed-order battles or run files were removed when the table was built. The 10 never-battled teams have no label and are not trained on. |
| "The storyboard figure are good" | The five frames under Figures stand as drawn. |

One change is Claude's. The readings of the linear model at epochs 34 and 55 were dropped, because the new control has no readings at those epochs to compare with.

## Deviations

| When | What changed | Why |
| --- | --- | --- |
| 2026-11-12, before either training started | The training-versus-validation gap is measured on about 4,380 training and 4,380 validation teams, where the Plan says 3,563 of each. The measurement takes as many training teams as there are labelled validation teams, and the shuffled-only table has 4,380 of those. The 3,563 was the older table's count | The count follows the table, and the Plan's number was carried over from today's default model |
| 2026-11-12, before either training started | The gap and the validation loss under both masking rates come from a new script, `src/masking_rate_training_report.py`. Its gap function is a port of `~/vgc-data/validation_loss_accuracy/run/validation_accuracy.py`, tested equal to it within 1e-6 on a stored model. The two trainings are the arms `shuffled_per_field` and `shuffled_linear` of `src/train84.py` | The earlier script and the probe for both rates were tied to today's default model and the older table |
| 2026-11-12, after both trainings | Each training fit 214,066 teams, not 214,598: the training code drops the 532 labelled training teams that the legal Pokémon construction tables exclude, as it did for every earlier 84-field model. The 12 such validation teams stay in the validation loss logged in training, where each forbidden value costs about 10,000 when hidden. The losses compared between the two models leave those 12 out and use 4,368 validation teams | Found when reading the run info. The Contract's count was the labelled rows before that drop |
| 2026-11-12, after both trainings | Early stopping never fired. Both trainings lowered the validation loss at every one of the 80 epochs, so both stopped at the cap and are read at epoch 80 | Threshold 0 counts any new lowest loss, and the last epochs still gained about 0.00002 each |
| 2026-11-13, before any grid | The interval of an error difference resamples the 49 opponents as well as the teams. The opponents are drawn once for both rows, and the teams within each ask and seed, separately in each row. The noise floor above resampled teams only | Battles against one opponent are correlated, so teams alone give an interval that is too narrow. Each grid saves every team's win rate against every opponent the moment it ends (`src/masking_rate_rows.py opponents`) |
| 2026-11-13, before any grid | The Pareto reading counts a point as outside the control's curve only when the intervals clear. Error uses its 95% bootstrap interval. Line-ups use the range over the three sampling seeds. The reading by the measured values alone is reported beside it | The Plan gave the rule without intervals. A bootstrap of teams undercounts distinct line-ups, so the seeds give the spread |
| 2026-11-13, before any grid | New line-ups are counted against the labelled training teams of the shuffled-only table. The Plan pointed at `src/lineup_reference.py`, which reads the older table. The line-ups of earlier search runs and tournament records still come from the older table, the only one that stores them | Both new models trained on the shuffled-only table |
| 2026-11-13, before any grid | The elite line-up yield script is now `src/line_up_yield.py`, ported from `~/vgc-data/line_up_deletion_84/run/` with the reference table as a setting. It scores three sets of asks. The Plan's count of 176 line-ups for today's default model is the set of asks 1/24 to 17/24 | The Plan named a script outside the repository |
| 2026-11-13, before any grid | The grids run guidance by guidance, the control first at each: 2, then 4, then 8. Each grid runs the ask grid's probe step with its own decoder and guidance first. The unguided baseline of a row follows its guidance 2 grid | With the control first, every other grid has its verdict the moment it lands |
| 2026-11-13, before any grid | The code ticket's remaining items were built in this ticket: guidance as a setting of the ask grid (`AG84_GUIDANCE`, with guidance 2 keeping the stored file names), the three code IDs with their decoders, the unguided cells (`AG84_UNGUIDED`) and a probe step that samples with the row's decoder. Guidance 2 redrew a stored cell of today's default model exactly, 64 of 64 teams. The corrected t sampler option was not built, because its row is not battled | [Add the linear masking rate and the corrected t to the 84-field code, with tests](https://github.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/issues/295) was still open when this ticket started |
| 2026-11-13, during the grids | In the control's grid at guidance 4, one battle worker timed out. The battle supervisor restarted the battle step and it finished. No battle was lost or counted twice: the grid holds 69 of 69 cells and 216,384 battles, and each team's win rate against each opponent adds up to the stored scores exactly. The grid took 2 hours 19 minutes in battles, where the others took about 35 minutes | A `TimeoutError` in one worker of the battle runner. The cause inside the worker was not looked for |
| 2026-11-13, during the grids | The post for the control at guidance 8 failed on a network reset at 20:42 and was posted by hand at 22:30, after the post for Linear, species-first at guidance 8. Its numbers and figure are the ones the queue wrote at 20:27 | The queue keeps a failed post's text and goes on |
| 2026-11-13, after the grids | The elite line-up yield re-battles came to about 546,000 battles on 2,786 teams, where the budget said about 315,000. The count is from the private yield database: 273,028 stored pairings of 4 battles, each battle stored from both sides. It includes the re-battle of today's default model | The script re-battles the elite teams of all 23 asks. The budget counted the asks 1/24 to 17/24 only |
| 2026-11-13, after the grids | The queue ran 11 hours 58 minutes, from 11:18 to 23:16, where the budget said about 8.1 hours | The restart above, the yield re-battles above, and sampling in random order: 36 minutes for the guidance 8 grid against 9 for a species-first grid |
| 2026-11-13, after the results were posted | The summary gained two measures for each grid: the distinct line-ups among each ask's teams, and the mean number of asks that share one line-up. Every other number in the summary file is unchanged | The Plan counted line-ups over the whole grid only, and that total rises with guidance for a reason that is not diversity. See Line-ups within one ask |
| 2026-11-13, after the grids | The Pareto rule's first clause is true for every point of Linear, random order only because no control point has as many line-ups, so the clause compares it with nothing. The reading under Results rests on the second clause alone and says so | Found when the rule called a row with 4 to 8 times the control's error "outside" on both clauses |

## Results of the two trainings

Measured 2026-11-12 for [Train the 84-field model with the linear masking rate](https://github.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/issues/296). The table gives each new model at epoch 80. A validation loss is compared only down a column, under one masking rate.

| Model, by the masking rate it trained with | Validation loss under the per-field masking rate | Validation loss under the linear masking rate | Training-versus-validation gap, 95% interval | Logged validation loss, 12 excluded teams included |
| --- | ---: | ---: | --- | ---: |
| Per-field masking rate, `stat84_shuffled_noise_per_field_es3_cap80` | 1.4541 | 1.6477 | +0.0121, −0.0059 to +0.0301 | 2.0026 |
| Linear masking rate, `stat84_shuffled_noise_linear_es3_cap80` | 1.4735 | 1.5987 | +0.0053, −0.0126 to +0.0233 | 2.0142 |

Frame 4 came out as expected: the linear model is lower under the linear masking rate, by 0.0491, and slightly higher under the per-field rate, by 0.0194. Frame 5 came out as expected too. On the new per-field model the corrected t moves the drawing distribution by 0.0078 at most, on Stat Alignment at ask 6/24, under the 0.01 bar, so Per-field, corrected t gets no battle grid. The corrected t changes at most 3.3% of fields and 0.8% of line-ups at one ask. Charts: [validation loss by epoch](figures/masking_rate_rows/04a_validation_loss_by_epoch.html), [loss under both masking rates](figures/masking_rate_rows/04b_validation_loss_two_by_two.html), [gap](figures/masking_rate_rows/04c_training_versus_validation_gap.html). Script: `src/masking_rate_training_report.py`.

## Results of the nine grids

Measured 2026-11-13 for [Sample and battle the four rows and post the results table](https://github.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/issues/297). Win rate is one battle against each of the 49 distinct top-50 meta teams, with the behaviour-cloning policy on both sides and slots shuffled. Script: `src/masking_rate_rows.py summary`, which wrote `~/vgc-data/results/masking_rate_rows_summary.json`. A second run of the script from this branch gave the same file byte for byte.

**The linear masking rate matches the per-field one when decoding species-first. Random order loses.**

### Planned against actual counts

| Step | Planned | Actual |
| --- | --- | --- |
| Grids | 9, each of 4,416 teams and 216,384 battles | 9, each of 4,416 teams and 216,384 battles. No cell short of 64 teams. 1,947,456 battles in all |
| Probe step before each grid | 9, each with its row's decoder | 9 of 9 passed |
| Unguided baselines | 3, each of 192 teams | 3, each of 192 teams and 9,408 battles |
| Lookahead-off legality probe | 4,608 tries, no battles | 4,608 tries |
| Elite line-up yield | 9 grids, about 315,000 battles | 9 grids and today's default model, about 546,000 battles. See Deviations |
| Grids of Per-field, corrected t | None unless its probe reaches 0.01 | None. The probe read 0.0078 |

### Every measure

| Row | Guidance | Error on asks 1/24 to 17/24, with 95% interval | Error on asks 18/24 to 23/24 | Win rate at asks 15/24 to 17/24 | Pooled win rate | First-try legality | Line-ups | New line-ups against training teams | Effective species | Top-species share | Elite line-ups, then those that held |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Per-field, uncorrected t, the control | 2 | 0.0084, 0.0045 to 0.0134 | 0.1352 | 0.497 | 0.354 | 100.0% | 1,513 | 17.5% | 15.0 | 0.136 | 186, then 81 |
| Per-field, uncorrected t, the control | 4 | 0.0049, 0.0022 to 0.0091 | 0.1079 | 0.532 | 0.375 | 100.0% | 1,918 | 24.4% | 14.2 | 0.143 | 254, then 132 |
| Per-field, uncorrected t, the control | 8 | 0.0033, 0.0014 to 0.0068 | 0.0909 | 0.556 | 0.390 | 100.0% | 2,126 | 31.9% | 13.2 | 0.147 | 314, then 162 |
| Linear, species-first | 2 | 0.0075, 0.0039 to 0.0126 | 0.1521 | 0.503 | 0.353 | 100.0% | 1,578 | 18.3% | 14.9 | 0.138 | 172, then 64 |
| Linear, species-first | 4 | 0.0049, 0.0022 to 0.0092 | 0.1210 | 0.527 | 0.373 | 100.0% | 1,971 | 26.4% | 14.1 | 0.143 | 231, then 95 |
| Linear, species-first | 8 | 0.0038, 0.0017 to 0.0074 | 0.1087 | 0.539 | 0.385 | 100.0% | 2,060 | 35.6% | 13.1 | 0.145 | 230, then 127 |
| Linear, random order | 2 | 0.0358, 0.0276 to 0.0447 | 0.2879 | 0.340 | 0.247 | 90.4% | 2,646 | 87.8% | 20.0 | 0.110 | 31, then 7 |
| Linear, random order | 4 | 0.0293, 0.0220 to 0.0377 | 0.2529 | 0.364 | 0.266 | 90.5% | 3,336 | 88.3% | 19.5 | 0.114 | 79, then 17 |
| Linear, random order | 8 | 0.0269, 0.0198 to 0.0352 | 0.2492 | 0.368 | 0.272 | 89.2% | 3,820 | 90.2% | 19.0 | 0.118 | 91, then 28 |
| Today's default model, for reference | 2 | 0.0053 | not read here | 0.515 | 0.371 | 100.0% | 1,437 | not comparable | 14.9 | 0.135 | 176, then 81 |

The last column counts line-ups with a team at or above the elite cut of 0.531 on asks 1/24 to 17/24, then how many still reach the cut in a fresh re-battle of 4 games against each opponent. New line-ups of today's default model were counted against a different training table, so the cell is left out. The single-grid intervals are wide because they resample the 49 opponents. The differences below draw the same opponents for both rows and are narrower.

### Verdicts against the control at the same guidance

| Row | Guidance | Error minus the control's | 95% interval of the difference | Sampling seeds with a lower error | Verdict |
| --- | ---: | ---: | --- | ---: | --- |
| Linear, species-first | 2 | -0.0008 | -0.0022 to +0.0005 | 2 of 3 | matches |
| Linear, species-first | 4 | +0.0000 | -0.0009 to +0.0010 | 1 of 3 | matches |
| Linear, species-first | 8 | +0.0005 | -0.0002 to +0.0013 | 0 of 3 | matches |
| Linear, random order | 2 | +0.0274 | +0.0228 to +0.0319 | 0 of 3 | loses |
| Linear, random order | 4 | +0.0245 | +0.0196 to +0.0290 | 0 of 3 | loses |
| Linear, random order | 8 | +0.0236 | +0.0182 to +0.0287 | 0 of 3 | loses |

Linear, species-first matches at all three guidance values, so the row matches. Its gap moves from 0.0008 lower at guidance 2 to 0.0005 higher at guidance 8, and the count of sampling seeds with a lower error falls from 2 to 1 to 0. No gap reaches the 0.002 of training luck. Its pooled win rate differs from the control's by 0.001 to 0.005, inside the 0.026 of training luck. At guidance 2 the interval still admits a gain as large as 0.0022, so that grid does not rule out the 0.001 effect.

Linear, random order loses at all three, on every sampling seed, so the row loses.

One reading outside the verdict rule: fewer of Linear, species-first's elite line-ups held in the re-battle than the control's at every guidance, 64 against 81, 95 against 132 and 127 against 162. The yield has no interval, and the three counts share one checkpoint.

### Random order against species-first on the linear model

| Guidance | Random order's error minus species-first's | 95% interval | Error multiple, with 95% interval | Pooled win rate change | Line-ups change |
| ---: | ---: | --- | --- | ---: | ---: |
| 2 | +0.0283 | +0.0233 to +0.0329 | 4.8, 3.5 to 7.1 | -0.106 | +1,068 |
| 4 | +0.0244 | +0.0195 to +0.0290 | 6.0, 4.0 to 10.0 | -0.107 | +1,365 |
| 8 | +0.0232 | +0.0179 to +0.0281 | 7.2, 4.6 to 11.8 | -0.113 | +1,760 |

Random order's first-try legality is 89.2% to 90.5% with the lookahead on. With the lookahead off it is 1.9%: 86 legal teams in 4,608 tries. The per-field model read 1.2% on 2026-11-07. So training with the linear masking rate did not teach the model which moves a species can learn when the moves are drawn first. The lookahead does that work.

### Unguided baselines

| Row | Unguided win rate | By sampling seed | Line-ups among 192 teams | First-try legality |
| --- | ---: | --- | ---: | ---: |
| Per-field, uncorrected t, the control | 0.219 | 0.206, 0.218, 0.233 | 167 | 100.0% |
| Linear, species-first | 0.220 | 0.223, 0.210, 0.226 | 169 | 100.0% |
| Linear, random order | 0.165 | 0.177, 0.161, 0.156 | 192 | 91.0% |

### Per-field, corrected t

Not battled. On the control's checkpoint the corrected t moves the drawing distribution by 0.0078 at most, against the bar of 0.01. It changes at most 3.3% of fields and 0.8% of line-ups at one ask.

The largest move is Stat Alignment at ask 6/24, the same field type and ask as the default model's largest, 0.0050. Figure: [the new per-field model barely reads the t it is told](figures/masking_rate_rows/11_t_sensitivity_new_per_field_model.html), built by `src/masking_rate_preflight_figures.py`. It is the storyboard's frame 5.

### Expectations, scored

| Row | Expected | Result |
| --- | --- | --- |
| Per-field, uncorrected t, the control | An error above 0.0053 at guidance 2, falling as guidance rises. 70% | Came true: 0.0084, then 0.0049, then 0.0033 |
| Per-field, corrected t | No battle grid. 90% | Came true |
| Linear, species-first | Matches, or differs by less than training luck. 60% | Came true |
| Linear, random order | Loses to Linear, species-first with an error less than 2.8 times as large, first-try legality above 84%, and more line-ups. 65% | Missed on the multiple: 4.8, 6.0 and 7.2, and the low end of every interval is 3.5 or more. The other three parts came true |

The largest surprise named in advance, a linear row that wins by 0.002 or more, did not happen.

### Why the multiple missed

The expectation said linear training would make random order safe, because the model would have seen moves visible while their species is hidden. Three suspects, each with the test that would settle it:

| Suspect | What points to it | Test |
| --- | --- | --- |
| The 2.8 came from a weaker model, so its species-first error was large and the ratio small | That model was the all-teams model after 21 epochs: 0.0534 against 0.0192. Random order's own error fell to 0.0358 here. Species-first's fell further, to 0.0075 | Random order on the control's checkpoint, the same three grids. A multiple near 4.8 there says the masking rate changed nothing |
| Random order draws weaker teams whatever the ask | Got stays between 0.32 and 0.39 from ask 13/24 to 21/24, where species-first on the same model reads 0.46 to 0.55. The unguided win rate is 0.165 against 0.220 | Decode the six species first and the other 78 fields in random order. If got returns to 0.50, the loss comes from drawing species late |
| The lookahead bends the draw | Without it 98% of the teams drawn are illegal, so it overrides the model's own draw often. 88% to 90% of the teams have a line-up no training team has | Compare the species the model wanted with the species the lookahead left, at the passes where a species is drawn after its moves |

None of these was run. The stopping rule allows no re-run to chase a result, and a new row is the owner's call.

### Pareto reading

Linear, species-first is inside the control's curve at every guidance, by the intervals and by the measured values alone. Its three points lie on the control's curve.

Linear, random order has more line-ups than every control point: 2,646 to 3,820 against 2,126 at most. By the rule that puts it outside the control's curve. The cost is an error 4 to 8 times the control's, a pooled win rate lower by 0.11 to 0.12, and 7 to 28 elite line-ups that held against the control's 81 to 162. The extra line-ups are weak ones.

### Line-ups within one ask

Added 2026-11-13 after the owner asked whether the line-up count was wrong, because the grid total rises with guidance while the papers report that guidance lowers diversity. The count is right. A recount from the stored teams with a separate parser gave the same nine totals.

The grid total pools 23 asks. The papers measure diversity among samples that share one condition, and the mean count among the 192 teams of one ask falls with guidance on both species-first rows. It does not fall at every ask: from guidance 2 to 8 the control has fewer line-ups at 16 of the 23 asks, the same at 1 and more at 6, and Linear, species-first has fewer at 16, the same at 2 and more at 5:

| Row | Guidance | Distinct line-ups over the grid's 4,416 teams | Mean distinct line-ups among one ask's 192 teams | Mean at asks 15/24 to 17/24 | Asks that share one line-up, mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| Per-field, uncorrected t, the control | 2 | 1,513 | 167.0 | 172 | 2.54 |
| Per-field, uncorrected t, the control | 4 | 1,918 | 156.3 | 159 | 1.87 |
| Per-field, uncorrected t, the control | 8 | 2,126 | 134.9 | 112 | 1.46 |
| Linear, species-first | 2 | 1,578 | 162.9 | 161 | 2.37 |
| Linear, species-first | 4 | 1,971 | 155.3 | 145 | 1.81 |
| Linear, species-first | 8 | 2,060 | 132.4 | 96 | 1.48 |
| Linear, random order | 2 | 2,646 | 191.7 | 192 | 1.67 |
| Linear, random order | 4 | 3,336 | 191.6 | 192 | 1.32 |
| Linear, random order | 8 | 3,820 | 191.2 | 192 | 1.15 |

Stronger guidance makes each ask's teams more specific to that ask. A line-up of the control is drawn at 2.54 asks on average at guidance 2 and at 1.46 at guidance 8, so the asks overlap less and their union grows while each ask repeats itself more. At asks 15/24 to 17/24 the control goes from 172 line-ups to 112.

The grid total is the count of each ask, added over the 23 asks, minus the repeats between asks. For the control, with all three rows in the figure:

| | Guidance 2 | Guidance 4 | Guidance 8 |
| --- | ---: | ---: | ---: |
| Distinct line-ups in each ask, added over the 23 asks | 3,842 | 3,594 | 3,102 |
| Repeats of a line-up already counted at another ask | 2,329 | 1,676 | 976 |
| Distinct over the grid, the first row minus the second | 1,513 | 1,918 | 2,126 |

The repeats fall faster than the counts do. Figure: [each row's line-ups counted ask by ask, split into the grid total and the repeats](figures/masking_rate_rows/09_line_ups_shared_between_asks.html).

Two earlier runs agree. At the single ask 0.5 on 48-field models, 2026-10-28, the line-ups among 192 teams were 120, 54, 23 and 8 at guidance 1, 2, 4 and 8 (`guidance_legality.json`). On the 84-field model with all 39 opponents for each label, 2026-11-11, the grid total rose from 1,427 to 1,774 to 1,891 while the mean for one ask fell from 167.5 to 153.2 to 128.0 (`guidance_sweep_gen65_subset_39.json`).

So the line-ups axis of the Pareto chart rewards guidance for separating the asks. The verdicts compare rows at one guidance value and are not affected. Read across guidance values, the grid total is not a diversity measure, and the count for one ask is. Linear, random order is at 191 or 192 of 192 at every ask and guidance, so its rising total comes only from less overlap between asks. Figure: [the control's line-ups at each ask](figures/masking_rate_rows/08_line_ups_by_ask.html).

### Why the totals differ, row by row

Figure: [each grid's line-ups split into training line-ups and new line-ups](figures/masking_rate_rows/10_training_and_new_line_ups.html). In all three rows the line-ups that guidance 8 adds over guidance 2 are new ones: they rise by 734, 677 and 1,110, while training line-ups move by −121, −195 and +64. Linear, random order is 90% to 92% new line-ups at every guidance.

Measured 2026-11-14 from the stored teams. One model analysed each row, and a fourth recounted 19 of their claims with its own parser: 18 matched, and one had two numbers off by 5 and 1, corrected below. A training line-up is a line-up of one of the labelled training teams. An ask slot of a line-up is one ask at which it is drawn.

| Row | Why its total differs | Evidence |
| --- | --- | --- |
| Per-field, uncorrected t, the control | At guidance 2 the asks blur together: neighbouring asks draw the same training line-ups, so one line-up is counted at many asks. Stronger guidance pins each line-up to a few asks. The low asks then fill with new line-ups that are drawn at one ask only, and the high asks narrow onto a few training line-ups | 2,216 of the 2,329 repeats at guidance 2 are a line-up drawn at the next ask too, and 45 of the 50 most widespread line-ups are training line-ups. The 20 most widespread take 239 ask slots at guidance 2, 96 at guidance 4 and 34 at guidance 8. Line-ups not in training go from 502 to 834 to 1,236, and 1,130 of the 1,236 are drawn at one ask. Training line-ups go from 1,011 to 890. At ask 21 the count falls from 151 to 58 and the most common line-up goes from 6 teams to 35 |
| Linear, species-first | The same thing happens, on many of the same line-ups. Its gap to the control is small and mostly inside the spread between sampling seeds | The two rows share 776 line-ups at guidance 2. The gap of +65 at guidance 2 is +48, +6 and +25 by sampling seed. The gap of −66 at guidance 8 is −2, +2 and +20 by seed, so it is noise. One difference stands out: at asks 15/24 to 23/24 under guidance 8 the control holds 226 line-ups, 93% of its teams on training line-ups, and this row holds 345, with 77% |
| Linear, random order | It is the same model as Linear, species-first, but it rarely puts a training team's six species together, so almost every team in an ask has its own line-up. Its total rises with guidance only because the few repeats between asks disappear | 12% of its teams have a training line-up at guidance 2, against 82% for species-first. Its 10 most used species take 43% of species slots, against 64%. It draws 58 species that appear nowhere in the training table, in 24% of its teams, and species-first draws none. Repeats fall from 1,762 to 578, and most of the lost repeats are line-ups not in training: 1,486 to 479. Teams with a new line-up win 0.236, against 0.323 for teams with a training line-up |

Two readings are inferences, not measurements. The first is why the low asks leave the training line-ups under strong guidance: few training teams carry low win-rate labels, or guidance extrapolates past the data. Counting, for each training line-up, its labels at or below 8/24 would separate them. The second is where random order's 58 unseen species come from. The likely source is the lookahead: once a Pokémon's moves are drawn, only species that learn them stay possible, whatever the model would have chosen. Logging the fill order of each team, and counting the fields drawn before each species, would test it. That is the third suspect under Why the multiple missed, now with a number behind it.

### What this result cannot show

- Each masking rate has one training seed. Two trainings of one recipe have differed by 0.0019 in error and 0.026 in pooled win rate, so "matches" cannot separate the two masking rates more finely than that.
- The three guidance values share one checkpoint for each masking rate. Three matches are three readings of one training, not three tests of the masking rate.
- One opponent set, the 49 distinct top-50 meta teams, and one battle policy, behaviour cloning.
- Both models are read at epoch 80, where early stopping had not fired.
- The elite line-up yield has no interval.

Figures: [error against line-ups](figures/masking_rate_rows/05_error_against_line_ups.html), [got against ask](figures/masking_rate_rows/06_got_against_ask.html), [random order's error multiple](figures/masking_rate_rows/07_random_order_error_multiple.html), built by `src/masking_rate_rows_figures.py`. Frame 1 came out as expected. Frame 2 came out as expected: the two lines overlap and stop rising at the same got. Frame 3 came out as its surprise: the multiple rose above 2.8.

The rejection reasons stored with the random-order grids repeat a word, as in "can'tcan't learn Heat Wave". The fault is in how `ask_grid84.py` shortens a validator message. The counts are right.
