# Ask-versus-got slope by training epoch

Pre-registration, written 2026-11-02 before any run. Map: GitHub issue #102. Code ID: `slope_by_epoch`. Status: **exploratory**, because one training seed cannot separate an epoch effect from training noise.

In these terms, an ask is the win rate the model is told to aim for, got is the win rate its teams then score, and a line-up is a team's six species.

## Contract

| Row | Value |
| --- | --- |
| Claim | Owner's hypothesis: the more epochs, the closer the ask-versus-got slope gets to 1. |
| What refutes it | The best epoch's slope is within 0.05 of an earlier epoch's slope, or falls below it, in the 3-seed readings. |
| What changes if it holds | Training length stays long enough to reach the highest slope. If the slope peaks before the best epoch, and the decision rule below passes, the default moves to that epoch. |
| How far it reaches | Win rate under the behaviour-cloning policy against the top-50 meta teams, no further. |
| Fields | The model generates all 84 fields. A repair step replaces Stat Alignment to match the moves (`ask_grid84.py` docstring). Nothing is copied. |
| Vocabulary | The Regulation M-B tables directly: 348 species, 201 ability, 150 item, 498 move and 27 Stat Alignment tokens. Labelled training teams use 189, 126, 106, 309 and 19 of them. |
| Training teams | Only teams with a known win rate against the top-50 meta teams (owner, 2026-11-02): 170,785 labelled teams from the all-teams table in `lossdata84.npz`, before the rule tables drop any. Validation holds the 3,563 labelled teams in its existing 2% split. The 130,232 unlabelled teams, the 500 test teams and the held-out teams ranked 51 to 100 stay out. |
| Scorer | One battle against each of the 49 distinct top-50 meta teams, behaviour-cloning policy on both sides, slots shuffled. Win rate means this throughout. |
| Search region | No search and no elite rule. Asks 1/24 to 23/24, guidance 2, temperature 1, rule masks on, species first. Each cell draws until 64 teams pass Showdown and are not held-out teams, at most 1,280 tries. |

### Settings

| Setting | Role | Value |
| --- | --- | --- |
| Training epoch | Scientific | 1, 2, 3, 5, 8, 13, 21, 34, 55, and the best epoch |
| Sampling seed | Nuisance | 1 at each epoch, 3 in the rechecks |
| Training teams | Fixed | Labelled teams only |
| Label noise | Fixed | ±1/48, fresh for each batch |
| Training seed | Fixed | 0 |
| Learning-rate schedule | Fixed | Sized to the cap of 80, so an early checkpoint is the same training partway through, not a short training |
| Batch, learning rate, network | Fixed | 256, 4e-4, 256 wide with 6 layers and 8 heads |
| Stopping rule | Fixed | Patience 3 on validation loss, cap 80 |

### Plan

- **Primary measure:** the slope of mean got on ask, over all 23 asks.
- **Second line:** the slope over asks 1/24 to 17/24. Only this line compares with the training teams' 0.715, because asks 22/24 and 23/24 have no training teams and 21/24 has 10.
- **Reported beside each slope:** win rate per ask, line-ups, new line-ups against the fixed reference, and the share of Pokemon spending all 66 Stat Points.
- **Question:** better, with the interval above zero.
- **Smallest effect worth acting on:** a slope change of 0.05.
- **Baselines:** the earlier label-noise training at 21 epochs, slope 0.425 (`gen84_all_noise_ep21`), and the model without label noise, slope 0.214 (`gen84_all_ep21`). Both trained with unlabelled teams, so epoch 21 here is not expected to reproduce either; its gap from 0.425 measures what removing them does. The training teams' line, 0.715, is the ceiling.
- **Random-search baseline:** none, because a slope needs an ask and random search takes none.

### Decision rule

Move the default epoch only if all three hold in the 3-seed readings:

1. Its slope beats the best epoch's by at least 0.05, with the interval above zero.
2. Win rate at the top asks is no worse.
3. Line-ups do not collapse.

Otherwise, validation loss keeps setting the training length.

## Arms

All new arms use the 84-field model on labelled teams with label noise ±1/48: 23 asks, 64 teams per seed, 49 battles per team. A 3-seed reading reuses the arm's seed 1 grid, so its results stay in one file.

| Reader name | Code ID |
| --- | --- |
| Epoch 1, 1 seed (one arm per spaced epoch) | `gen84_noise_byepoch_ep01` |
| Epoch 21, 3 seeds | `gen84_noise_byepoch_ep21` |
| Best epoch, 3 seeds | `gen84_noise_byepoch_ep<best>` |
| Turning-point epoch, 3 seeds | `gen84_noise_byepoch_ep<turn>`, run by hand if the curve bends |
| Unlabelled teams included, 21 epochs, 3 seeds | `gen84_all_noise_ep21` (existing) |

## Noise floor and budget

Slope noise is a bootstrap over teams on the 21-epoch model without label noise, one training seed. The grid sizes come from that run: 216,384 battles and generation took 46 minutes.

| Grid | Battles per epoch | Minutes | Slope noise, 23 asks | Slope noise, 1/24 to 17/24 |
| --- | --- | --- | --- | --- |
| 1 seed | 72,128 | 15 | 0.009 | 0.015 |
| 3 seeds | 216,384 | 46 | 0.005 | 0.009 |

The 1-seed grid can detect a gap of about 0.035 between two epochs on 23 asks, and the 3-seed grid about 0.02, at 5% false positives and 80% power. Both are below the 0.05 bar. Neither figure includes training-seed noise, which this run cannot measure.

Budget: 721,280 battles for the ten spaced epochs, about 2.5 hours, plus 144,256 for each 3-seed reading on top of its seed 1 grid, about 31 minutes. Epochs in between run only where the curve bends. Training takes about 4.2 minutes per epoch, scaled from 7.3 minutes on 298,488 teams, so at most about 5.6 hours.

## Probes

| Probe | Result |
| --- | --- |
| Training smoke | Passed 2026-11-02 on 512 labelled teams: one epoch, validation loss 2.827, per-epoch checkpoint written. The queue repeats it before training. |
| Ceiling | Passed on asks 1/24 to 17/24: the training teams' 0.715 beats 0.644 by 0.071. Not measured on all 23 asks. |
| Round trip, coverage, read teams, controls, separation, condition, copies | Run by the queue on the best checkpoint before any battle. A failure stops the queue. |
| Invariance, repeat | Skipped: the scorer is unchanged, and every battle shuffles slots. |

## Deviations

None yet.

The owner confirmed the label-noise recipe on 2026-11-02, after training was queued and before it started. The owner then limited training to labelled teams, on the same day. I stopped the waiting queue before any training and rebuilt it, so no result existed when either choice was made.

## Hindsight-relabeled fine-tune

The map also tests this fine-tune, issue #106, with its own contract before its run. It does not change the plan above.

## Correction, 2026-11-05: the ask-versus-got error judges this, not slope

On 2026-11-05 the owner replaced this plan's primary measure, because it answers the wrong question. A slope says how steep the fitted line is, and a steeper line can sit further from got = ask. The measure the owner settled on 2026-11-01 for [the ask-got map](https://github.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/issues/70) is mean squared error between each ask and the mean got at that ask, on asks 1/24 to 17/24, with near the line under 0.001. The glossary calls it the ask-versus-got error, and `src/ask_line.py` reports it first. The contract above stays as written, because it ran under it.

Read from each arm's `summary.line`, with win rate against the 49 distinct top-50 meta teams and the behaviour-cloning policy on both sides:

| Arm | Mean squared error, asks 1/24 to 17/24 | Asks 18/24 to 23/24 | Slope, all 23 asks |
| --- | --- | --- | --- |
| Epoch 13 | 0.0077 | 0.1645 | 0.488 |
| Epoch 21 | 0.0066 | 0.1504 | 0.489 |
| Epoch 34 | 0.0059 | 0.1344 | 0.509 |
| Epoch 55 | 0.0061 | 0.1350 | 0.505 |
| Epoch 80 | 0.0053 | 0.1383 | 0.502 |

These are the stored file summaries: epochs 21 and 80 pool three sampling seeds, while epochs 13, 34 and 55 use one. They do not form a matched one-seed epoch comparison. The [checkpoint report](../reports/training-checkpoints-2026-11-05/index.html) compares sampling seed 1 at all ten epochs, records the pooled summaries separately, and includes the underlying observations and analysis scripts. In that matched comparison, error falls from 0.01750 at epoch 1 to 0.00591 at epoch 34, then stays near 0.006 through epoch 80. The difference between epochs 34 and 80 is uncertain under a bootstrap of whole teams and shared opponent columns. The report also records line-up diversity, novelty against a fixed labelled-row reference, win-rate scatter, and Stat Points use.

No arm is near the line.

By this error epoch 80 beats epoch 21 by 0.0013 (0.0006 to 0.0020). Intervals resample teams only, so they leave out training-seed noise. The smallest effect worth acting on is now 0.001, chosen by Claude and open to the owner's change.
