# Should the 84-field model's position embedding split into slot and field-type vectors?

Pre-registered 2026-11-07, before any training, sampling or battle of these arms. Map: [issue #78](https://github.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/issues/78). Plan ticket: [issue #80](https://github.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/issues/80).

**Mode: confirmatory**, once the training length is written in from the training-length decision ([issue #108](https://github.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/issues/108)) before the first training. Nothing below may change after that.

**Blocked on the epoch count**, which comes from #108 (still open). The owner chose on 2026-11-07 to wait for #108 rather than train at 21 epochs. The training table's code ([PR #232](https://github.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/pull/232)) merged on 2026-11-07.

Prior work: no project doc or `experiments-index.md` entry tests this split. A web search for a slot-plus-field-type position embedding on a set of entities found no direct match.

## Terms

The model writes six Pokémon as 84 fields: 14 per Pokémon (species, ability, item, four moves, Stat Alignment, six Stat Points). Today each of the 84 fields has its own learned position vector (`Net84.pos`, `src/diffusion84.py`). A **field-type vector** marks what a field is, such as Speed or Move 1, and there are 14. A **slot vector** marks which fields belong to the same Pokémon. A slot vector carries no order, because training shuffles slots every batch.

## Arms

| Reader name | Code ID | Position of a field |
| --- | --- | --- |
| No split, first seed | `pos_per_field_s0` | one of 84 learned vectors, training seed 0 |
| No split, second seed | `pos_per_field_s1` | the same, training seed 1 |
| Learned slot vectors | `pos_slot_learned` | learned field-type vector + learned slot vector |
| Fixed slot vectors | `pos_slot_fixed` | learned field-type vector + random slot vector that never trains |
| No slot vectors | `pos_no_slot` | learned field-type vector only |

Every new vector starts at standard deviation 0.02, like the current table. The three split arms use training seed 0, so each differs from No split, first seed in the position embedding alone.

**Changed 2026-11-07 (owner):** No split, first seed is a new training on the shuffled-only table. The earlier first seed, `stat84_all_teams_noise_ep21` (pooled win rate 0.384), trained on `lossdata84.npz`, which holds labels from listed-order battles. It is reported for reference and decides nothing.

## Contract

| Row | Value |
| --- | --- |
| Claim | Replacing the 84 per-field position vectors with field-type vectors plus slot vectors changes pooled win rate or effective species by more than the gap between two training seeds. |
| What refutes it | Every split arm ties No split, first seed under the verdict rule below. |
| What changes if it holds | The default 84-field network takes the split arm that passes the rule. On a tie it keeps one learned vector per field, and slot vectors stay as they are. |
| How far it reaches | Win rate under the behaviour-cloning policy against the 49 distinct top-50 meta teams, and team measures on the 23-ask grid. Nothing beyond them. |
| Fields | Generated: all 84. Copied: none. A repair step replaces Stat Alignment to match the moves after sampling (`src/ask_grid84.py`). |
| Vocabulary | The Regulation M-B tables directly, stored in each checkpoint: 348 species, 201 ability, 150 item, 498 move and 27 Stat Alignment tokens, and Stat Points 0 to 32. The same for every arm. |
| Training teams | `lossdata84_shuffled_only.npz` (sha256 `8aca232e…`, built 2026-11-07 by `src/lossdata84.py --shuffled-only`): 218,988 teams, 218,978 labelled from shuffled-slot battles only (at least 49 battles each) and 10 never-battled teams with no label. A seeded 2% (4,380 teams, seed 0) is validation, leaving 214,608 to train on. Excluded: the test 500 and the held-out meta teams ranked 51 to 100; 173,544 teams whose every label came from listed-order battles or run files. |
| Scorer | Behaviour-cloning policy on both sides, one battle against each of the 49 distinct top-50 meta teams, slots shuffled (`require_shuffled` refuses any other order). Win rate means this throughout. |
| Search region | No search and no elite rule. Asks 1/24 to 23/24, sampling seeds 1 to 3 with offset 7300, 64 teams per ask and seed, at most 1,280 tries per cell, guidance 2, temperature 1, species first, rule masks on. |

### Settings

| Setting | Role | Value |
| --- | --- | --- |
| Position embedding | Scientific | The five arms above |
| Training seed | Nuisance | 0 for every arm; 1 for No split, second seed, which measures training luck |
| Sampling seed | Nuisance | 1, 2, 3 |
| Training length | Fixed | From #108, the same for every arm. Patience off; the last epoch's weights kept |
| Learning-rate schedule | Fixed | AdamW, warm-up and cosine decay sized to a cap of 80 epochs, learning rate 4e-4, batch 256, EMA 0.999 |
| Label noise | Fixed | ±1/48 on training labels, a separate random generator; validation labels exact |
| Masking | Fixed | Exponents 16, 4, 4, 1.5, 1 for species, ability, item, move, Stat Alignment; 1 for Stat Points; ask dropout 0.15 |
| Network | Fixed | 256 wide, 6 layers, 8 heads, dropout 0.1 |
| Training table | Fixed | `lossdata84_shuffled_only.npz` |

## Plan

**Primary measures:** pooled win rate over the 23 asks, and effective species. Effective species is the exponential of the species entropy in a random 5-team subset, averaged over 200 subsets per cell, then over cells.

**Guard:** ask-versus-got error, the mean squared error between each ask and the mean win rate got there, on asks 1/24 to 17/24 (`src/ask_line.py`).

The Pareto question does not apply: no arm trades one measure against cost, and every arm gets the same training and battle budget. The rule below asks whether an arm wins, ties or loses against No split, first seed.

**Training luck** is the largest gap measured between two training seeds of the same recipe: the gap between the two new No split seeds, or the earlier one measured without label noise, whichever is larger. That earlier gap (`dellu_none` against `gen84_all_ep21`, 2026-11-07) is +0.0035 pooled win rate, +0.07 effective species and -0.0019 ask-versus-got error.

**Verdict rule** (owner, 2026-11-07, answers Q2 to Q5). Every interval is 95% and computed over teams, each team's 49 battles forming one cluster.

1. **Switch on win rate.** The arm's pooled win-rate gain has an interval above zero and is larger than luck, and its effective species is not lower by more than luck.
2. **Switch on species.** The arm's win-rate interval stays above -0.01, and its effective-species gain has an interval above zero and is larger than luck.
3. **Both switches** also need ask-versus-got error no higher than No split's by more than luck.
4. **Second seed.** A switch counts only after the passing arm trains again with seed 1 and passes the same rule, about 3 hours of extra training and battles.
5. **Tie-break.** If two arms pass within luck of each other, take Fixed slot vectors.
6. **Tie.** Anything else keeps one learned vector per field.
7. **It matters** (reported beside the verdict): an arm's interval against No split, first seed excludes zero on win rate or effective species, and the gap is larger than luck.

Legal share leaves the rule (owner, 2026-11-07): rule masks keep every sampled team legal, so it reads 100% for every arm. It is still printed as a check that must read 100%.

**Reported, not deciding:** validation loss; win rate at asks 18/24 to 23/24; distinct line-ups, new line-ups and elite line-up yield; the share of Pokémon whose Stat Points match a corpus set for their species; the share with fewer than four moves (15.7% on the old first seed, 2.6% in training teams).

**Baselines:** No split, first seed is the control. No random-search baseline, because there is no search. The two controls below fix the scorer's range.

### Expectations

Written before any run on 2026-11-01 against the old first seed, and kept as written except the legal-share column, which the owner dropped on 2026-11-07. They now apply to the new No split, first seed.

| Arm | Win rate | Effective species | Chance right |
| --- | --- | --- | --- |
| No split, second seed | within 0.02 | no clear change | 80% |
| Learned slot vectors | within 0.01, a tie | no clear change | 75% |
| Fixed slot vectors | ties learned slot vectors | no clear change | 70% |
| No slot vectors | loses at least 0.05 | same or higher | 85% |

Likely verdict: a tie, with about a 10% chance the split wins. The share of Stat Points matching a corpus set should drop sharply with no slot vectors. The biggest surprise would be no slot vectors holding their win rate.

## Probes

`src/ask_grid84.py preflight` already runs most of these. Each run saves them under `preflight` in its grid JSON. The probes run on the network options from [issue #81](https://github.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/issues/81), which do not exist yet, so none has a result today. A failed probe stops the launch.

| Probe | What this experiment checks | Where |
| --- | --- | --- |
| Ceiling | Pooled win rate can rise only where an ask is requested above what it gets. The old first seed gets 0.509 at most (ask 18/24) against the top-50 meta teams' 0.499 among themselves, so a gain must come at the low and middle asks. The guard catches an arm that gains by ignoring low asks. | This doc; no run needed |
| Round trip | Every one of the 84 fields reaches the paste from the model, for 20 teams per arm | `preflight` |
| Coverage | Same Regulation M-B tables in every checkpoint; teams the rule tables drop counted | `preflight` |
| Read teams | 20 decoded teams per arm read by hand; generated and training shares of species, items, abilities and moves | `preflight` |
| Training smoke | Per-field cross-entropy at the start near `ln K`; a few teams overfit; with no slot vectors, swapping two species changes the other fields only by rounding; fixed slot vectors get no gradient; No split matches today's parameter names, starting values and loss at a fixed seed | #81 tests |
| Controls | Top-50 meta teams near 0.50; uniform-legal teams near 0.015 | `preflight` |
| Invariance | Skipped: every battle shuffles slots and the scorer is unchanged | written reason |
| Repeat | Skipped: the scorer and seed path are unchanged since the line-up repeats run | written reason |
| Separation | Training teams against the test 500 and ranks 51 to 100: empty. `preflight` reads `lossdata84.npz` today and must read the shuffled-only table | `preflight`, after the table change |
| Condition | A fixed sampling seed at asks 0.1 and 0.7 changes more than half the teams | `preflight` |
| Copies | Share of generated teams identical to a training team, and shared slots with the nearest | `preflight` |

## Noise floor and budget

Measured on the old first seed's grid (4,416 teams, 216,384 battles):

| Measure | Standard error, one arm | 95% half-width of a gap between two arms | Luck measured |
| --- | --- | --- | --- |
| Pooled win rate (stratified by ask) | 0.0015 | 0.0042 | 0.0035 |
| Effective species (stratified by ask) | 0.057 | 0.157 | 0.07 |
| Ask-versus-got error (bootstrap over teams) | 0.00027 | 0.00075 | 0.0019 |

Battle and team noise on win rate is about the size of one measured training-seed gap. With one seed per arm, the rule can detect a win-rate gap of about 0.008 (half-width plus luck). The expected loss of No slot vectors, 0.05, sits far above that. A true tie within 0.004 cannot be told apart from training luck, which is why a tie keeps the current model. Luck rests on one pair of seeds, plus the new No split pair; a larger true seed spread would let a lucky arm pass, and the second-seed rule guards against that.

**Budget, same for every arm:**

| Item | Count | Time |
| --- | --- | --- |
| Trainings | 5 (two No split seeds, three split arms), plus 1 if an arm passes | about 1 h 45 min each at 21 epochs, scaled from the old first seed's 146 minutes by training teams (214,608 against 298,488); not measured. Scales with the #108 length |
| Battles | 216,384 per run, 1,081,920 for five, plus 216,384 if an arm passes | about 42 minutes per run |

Trainings run on the laptop at the end of the queue, detached, two at a time. No Modal spend. Each run records its wall-clock minutes.

## Deviations

None yet.
