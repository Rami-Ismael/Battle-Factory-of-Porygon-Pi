# Hindsight-relabeled fine-tune against the ask-versus-got error

Pre-registration, written 2026-11-07 before the fine-tune or any of its battles. Issue: GitHub #106, on the training-length map #102. Code ID: `ft_hindsight_relabel`. Status: **exploratory**, because the fine-tune has one training seed.

In these terms, an ask is the win rate the model is told to aim for, got is the win rate its teams then score, and a line-up is a team's six species. The ask-versus-got error is the mean squared error between each ask and the mean got at that ask, on asks 1/24 to 17/24 (`src/ask_line.py`). Near the line means under 0.001.

Hindsight relabeling trains on generated samples labelled with what they achieved, not what was asked. Hindsight Experience Replay introduced it for goals; Online Decision Transformer calls it "Hindsight Return Relabeling" for returns. Here a team the model drew at ask 12/24 that scored 0.31 enters training labelled 0.31.

## Contract

| Row | Value |
| --- | --- |
| Claim | Fine-tuning the epoch-21 model on its own battled teams, each labelled with its measured win rate, lowers the ask-versus-got error on asks 1/24 to 17/24 below the same model without the fine-tune. |
| What refutes it | The fine-tune's error is not at least 0.001 lower than the control's, with the 95% interval of the difference below zero. |
| What changes if it holds | Hindsight relabeling becomes a candidate step after training, if top-ask win rate is no worse and line-ups do not collapse. The training-length decision (#108) then weighs it against training to epoch 80. |
| How far it reaches | Win rate under the behaviour-cloning policy against the top-50 meta teams, no further. |
| Fields | The model generates all 84 fields. A repair step replaces Stat Alignment to match the moves (`src/ask_grid84.py`). Nothing is copied. The relabelled teams enter as the repaired pastes that were battled. |
| Vocabulary | The Regulation M-B tables directly, unchanged from the base checkpoint: 348 species, 201 ability, 150 item, 498 move and 27 Stat Alignment tokens. |
| Base model | Epoch 21 of the labelled-teams training with label noise ±1/48 (`stat84_labelled_noise_byepoch_es3_cap80_ep21.pt`). The script checks its sha256 against the one the control's grid recorded. |
| Training teams | Every team in the control's 23-ask grid (`stat_points_84_gen84_noise_byepoch_ep21.json`): 4,416 teams, 64 per ask and sampling seed, 3 seeds, each labelled with its mean over 49 battles. No elite filter, which is what separates this from the search loop. A seeded 20% (883 teams, split seed 0) is held out to stop on; 3,533 train. Held-out teams ranked 51 to 100 never entered the grid, and the script refuses any that would. The original training teams stay out. |
| Scorer | One battle against each of the 49 distinct top-50 meta teams, behaviour-cloning policy on both sides, slots shuffled. Win rate means this throughout. |
| Search region | No search and no elite rule. Asks 1/24 to 23/24, guidance 2, temperature 1, rule masks on, species first, sampling seeds 1 to 3 with the control's seed offset 7300. Each cell draws until 64 teams pass Showdown and are not held-out teams, at most 1,280 tries. |

### Where the relabels sit

Relabels nearest each ask, out of 4,416. Training teams thin out above 15/24 and stop at 19/24, so the fine-tune can learn little about asks 18/24 to 23/24.

| Nearest ask | 0 to 8/24 | 9/24 to 11/24 | 12/24 | 13/24 to 15/24 | 16/24 | 17/24 | 18/24 and 19/24 | 20/24 and above |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Teams | 1,809 | 1,149 | 683 | 688 | 57 | 22 | 8 | 0 |

### Settings

| Setting | Role | Value |
| --- | --- | --- |
| Hindsight-relabeled fine-tune | Scientific | On or off |
| Sampling seed | Nuisance | 1, 2, 3, the control's seeds |
| Learning rate, batch | Fixed | 1e-4 and 64, the search loop's fine-tune (`src/cem_dbbo.py`) |
| Schedule | Fixed | AdamW, warm-up and cosine decay sized to the cap, EMA 0.999, ask dropout 0.15, as in the base training |
| Label noise | Fixed | ±1/48 on training labels, as in the base training; validation labels stay exact |
| Stopping rule | Fixed | Patience 3 on validation loss over the held-out relabelled teams, cap 40 epochs. Epoch 0 is the starting weights, so a fine-tune that never beats them returns the control |
| Fine-tune seed | Fixed | 0 |
| Network | Fixed | The base checkpoint's: 256 wide, 6 layers, 8 heads |

The original labelled validation teams (3,563) are scored after every epoch as a record of what the fine-tune forgets. They never stop training.

### Plan

- **Primary measure:** the ask-versus-got error on asks 1/24 to 17/24.
- **Reported apart:** the error on asks 18/24 to 23/24, where training teams are scarce.
- **Reported beside the error:** response to the ask, the risk the issue names. It is the mean got at asks 15/24 to 17/24 minus the mean got at 1/24 to 3/24: 14/24 (0.583) on the line, 0.425 for the control. Online Decision Transformer's Figure 4.2 warns that fine-tuning can flatten this, so a fine-tune that lowers the error by moving every got to the middle shows up here. Also beside it: win rate at asks 18/24 to 23/24; win-rate scatter across teams within an ask; line-ups and new line-ups against the fixed references (`src/lineup_reference.py`); the share of Pokémon spending all 66 Stat Points; legality; and how many fine-tuned teams repeat a relabelled team or its line-up.
- **Question:** a Pareto question on two axes, the error on asks 1/24 to 17/24 against line-ups over all asks. Two arms trace no curve, so the fine-tune is one point. The strongest previous frontier among labelled-teams trainings with label noise, 3 seeds each: epoch 21 with no fine-tune (0.0066, 1,325 line-ups) and epoch 80 with no fine-tune (0.0053, 1,437 line-ups); the DDOM weights (0.0085 and 0.0117) sit inside it. The fine-tune moves outside it only by beating epoch 80 on error without fewer line-ups, or by matching epoch 80's error with more line-ups. Against its own control, the verdict is: **wins** if the error is at least 0.001 lower with the interval below zero, **loses** if the interval sits above zero, **matches** otherwise.
- **Smallest effect worth acting on:** 0.001 of error, the near-the-line bar, Claude's choice on map #102, open to the owner's change.
- **Decision rule:** the fine-tune counts as an improvement only if it wins against its control, its top-ask win rate is no worse, and line-ups do not collapse. Whether a drop in line-ups counts as collapse is the owner's call.
- **Stopping rule:** none for battles. The grid is fixed at 3 seeds.
- **Random-search baseline:** none, because the error needs an ask and random search takes none.
- **Slope:** not reported. The owner dropped it on 2026-11-05 and asked on 2026-11-06 that it not appear even as a secondary measure. The issue's "Slope is a secondary diagnostic" predates that correction, and the response to the ask above covers the sensitivity risk the issue names.

## Arms

Each arm starts from epoch 21 of the 84-field training on labelled teams with label noise ±1/48 and differs in what follows it. Each grid has 23 asks, 3 seeds, 64 teams per seed and 49 battles per team.

| What follows epoch 21 | Role | Code ID |
| --- | --- | --- |
| Hindsight-relabeled fine-tune | Arm | `ft_hindsight_relabel` |
| No fine-tune | Control | `gen84_noise_byepoch_ep21` (existing grid) |
| 59 more training epochs | Reference point on the frontier | `gen84_noise_byepoch_ep80` (existing grid) |

## Noise floor and budget

Noise is a bootstrap over teams within each ask and seed on the control's grid, 4,416 teams. The error of one 3-seed arm has a standard deviation of 0.00025, and the difference between two arms 0.00035. A gap of 0.00097 is detectable at 5% false positives and 80% power, at the 0.001 bar. The response to the ask has a standard deviation of 0.0046 per arm and 0.0063 for a difference. Training-seed noise is not included, and one fine-tune cannot measure it.

Budget: 216,384 battles for the fine-tune's grid, about 46 minutes of battles plus generation, the same as the control. The fine-tune trains on 3,533 teams, 56 batches an epoch, at most 40 epochs. The smoke run took 0.1 minutes for one epoch on 79 teams.

## Probes

| Probe | Result |
| --- | --- |
| Ceiling | Passed 2026-11-07, battle-free. If the fine-tune reproduced its relabels exactly, teams drawn at ask k/24 would be teams that scored k/24. Re-battled, such a team regresses toward its cell's mean by the grid's reliability, 0.875 (true variance 0.0283 against battle noise 0.0040 at 49 battles). The predicted error on asks 1/24 to 17/24 is 0.00014, against the control's 0.0066: well past the 0.001 bar. Asks 20/24 and above have no relabels, so no ceiling there. |
| Training smoke | Passed 2026-11-07 on 96 relabelled teams: epoch 0 validation loss 2.059, epoch 1 2.027; no team dropped by the rule tables. `scripts/test_fine_tuned84.py` checks that a fine-tune that learns nothing returns its starting weights and that the base model is left unchanged. The refactor of the training loop into `Model84._fit` reproduced a tiny training's losses to six decimals. |
| Ask-grid smoke | Passed 2026-11-07 after the training-loop refactor: the `verify-ask-grid84` smoke on the control checkpoint drew 4 legal teams at ask 12/24 and recorded 196 battles in a private database. A one-off check loaded the fine-tune's smoke checkpoint and drew 8 teams at ask 12/24, all legal. |
| Round trip, coverage, read teams, controls, separation, condition, copies | Run by the queue on the fine-tuned checkpoint before any battle (`ask_grid84.py preflight`). A failure stops the queue. Copies there compare with the original labelled training teams; repeats of relabelled teams are counted by `src/hindsight_error_compare.py`. |
| Invariance, repeat | Skipped: the scorer is unchanged, and every battle shuffles slots. |

## Deviations

None yet.
