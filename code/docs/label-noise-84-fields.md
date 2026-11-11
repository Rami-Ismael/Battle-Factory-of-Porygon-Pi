# Does label noise stop the 84-field model reading a team's source from its label?

Pre-registered 2026-11-01, before this training, sampling or any of its battles. Mode: confirmatory on the questions below. Results go to `~/vgc-data/results/stat_points_84_gen84_all_noise_ep21.json`.

## Question

The all-teams 84-field model behaves differently at asks that differ by 0.002. At ask 11/24, 0.8% of Pokémon have fewer than four moves. At 11/24 + 0.002, 14.4% do. At ask 12/24, win rate jumps to 0.446 against 0.371 at 11/24 and 0.395 at 13/24, and distinct line-ups jump from 58 to 183. In training labels from 0.25 to 0.7, the share of Pokémon with fewer than four moves is 0.04% on multiples of 1/24 only and 3.71% on multiples of 1/50 only. The explanation under test is that the model reads which kind of team it is writing from the label's exact value.

## Arms

| Reader name | Code ID | Difference |
|---|---|---|
| Label noise ±1/48, all teams, 21 epochs | `gen84_all_noise_ep21` | Each batch's labels get fresh uniform noise in [-1/48, +1/48], clipped to [0, 1] |
| No label noise, all teams, 21 epochs | `gen84_all_ep21` | The run the owner stopped at epoch 21, reused |

## Contract

| Row | Value |
|---|---|
| Claim | Win rate and team measures under this policy against these teams. Nothing beyond them. |
| Fields | Generated: all 84. Copied: none. Stat Alignment is repaired to match the moves after sampling. |
| Vocabulary | Regulation M-B tables stored in the checkpoint, as the arm without noise. |
| Training teams | The all-teams set in `lossdata84.npz`: 298,488 teams before the legal Pokémon construction tables, 6,092 in validation, test 500 and ranks 51 to 100 excluded. |
| Scorer | Behaviour-cloning policy on both sides, 49 distinct top-50 meta teams, one battle per opponent, slots shuffled. Seeds 7301 to 7303. |
| Search region | No search loop. 64 valid teams per ask and seed, at most 1,280 tries, guidance 2, temperature 1, species first, legal Pokémon construction filter on. |
| Scientific setting | Label noise width. |
| Fixed settings | Seed 0, batch 256, learning rate 4e-4, masking exponents, network size. The learning-rate schedule stays sized to a cap of 80 and training stops at epoch 21. The noise has its own random generator, so the other random draws match the arm without noise. Validation labels stay exact. |
| Stopping rule | Epoch 21, or earlier if 3 epochs pass with no new lowest validation loss. This deliberately differs from early stopping alone, because the comparison needs matching epochs. |
| Baselines | No random-search baseline: there is no search here. The arm without noise is the control. |

## Budget and noise

The 23-ask grid costs 216,384 battles. Asks 0.3, 0.531, 0.6 and 0.7 cost 37,632 battles per model. Each ask holds 192 teams, so its standard error is about 0.007. A difference of 0.03 between arms at one ask is detectable, and one seed of training per arm cannot separate training luck from the effect.

## Measures

Primary, with 256 sampled teams and no battles: the share of Pokémon with fewer than four moves at asks 6, 8, 10, 11, 12, 13, 14 and 16 out of 24, on the ask and at +0.002, -0.002 and +1/48.

Secondary, with battles: win rate per ask with a 95% interval, mean squared error against the line on asks 1/24 to 17/24, and win rate at asks 0.3, 0.531, 0.6 and 0.7.

Beside win rate: distinct line-ups, effective species, top species share, mean slot-matched distance, the share of line-ups absent from the 48-field models' training teams, and the share of Stat Points and movesets that match the training teams.

## Expected result

The explanation holds if all three happen:

1. The largest gap between an ask and its nudged neighbours in short-moveset share falls below 0.03. Without noise it is 0.206, at 6/24 against 6/24 + 1/48.
2. Ask 12/24 no longer stands out: its win rate is within 0.03 of the mean of 11/24 and 13/24, and its distinct line-ups within 30 of theirs.
3. Asks that are not multiples of 1/24 score within 0.03 of their neighbours on the grid.

I do not expect the flat stretch to go away. Fine-tuned 48-field models with the same noise still peak near 0.44.

## Deviations

**The report written by the queue lacks some measures.** The frozen report left out win rate at asks 0.3, 0.531, 0.6 and 0.7, mean slot-matched distance, and the share of line-ups absent from the 48-field models' training teams. The teams and scores it needs are saved. `report` now adds the first two; the share of new line-ups still has to be computed separately.
