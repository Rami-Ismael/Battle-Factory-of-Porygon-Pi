# How near the line is the model with no guidance?

Pre-registration, written 2026-11-15 before any guidance 1 team was drawn. Ticket: [How near the line is the model with no guidance?](https://github.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/issues/73), on the map [Why the ask does not follow the line, and whether the cross-entropy method helps](https://github.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/issues/70).

Mode: **confirmatory** for this checkpoint. The plan, the rule and the prediction are fixed here, before the first guidance 1 battle. One training seed, so the result describes this checkpoint, not every training of it.

## Terms

An ask is the win rate the model is told to aim for, k/24 for k = 1 to 23. Got is the win rate its teams then score. The line is got = ask. The ask-versus-got error is the mean squared error between each ask and the mean got at that ask, on asks 1/24 to 17/24 (`src/ask_line.py`); asks 18/24 to 23/24 are reported apart because the training data holds almost nothing there. Near the line means an error under 0.001.

Win rate means one battle against each of the 49 distinct top-50 meta teams, behaviour-cloning policy on both sides, slots shuffled every battle. A line-up is a team's six species.

Guidance is the classifier-free weight w in `src/diffusion84.py` (`sample`, line 591): the sampler mixes the logits as w × conditional + (1 − w) × unconditional. At guidance 1 the unconditional pass drops out, so the model is the inverse model that maps an ask to teams and nothing else. Guidance 2, 4 and 8 push past it toward the ask.

## Why this model

The ticket was written for the all-teams 84-field model of issue #60. On 2026-11-15 the owner closed #60 and named its model as the existing linear masking-schedule training `stat84_shuffled_noise_linear_es3_cap80` (issue #296): shuffled-only table, label noise ±1/48, early stopping on validation loss with patience 3 and cap 80, read at epoch 80. That checkpoint is used here.

It is decoded species-first, the row Linear, species-first (`sched_linear_species_first`) of `docs/masking-rate-rows.md`, so its guidance 2, 4 and 8 grids already exist and only guidance 1 is new. Species-first is `ask_grid84.py`'s default decoding order. This choice is the agent's, not yet the owner's.

## Arms

All four arms use the same checkpoint and decoding order: Linear, species-first, temperature 1, legal Pokémon construction filter on, Stat Alignment repaired, 64 Showdown-legal teams per cell, asks 1/24 to 23/24, sampling seeds 1, 2 and 3.

| Arm | Code ID | Status |
| --- | --- | --- |
| Guidance 1, no push | `sched_linear_species_first_g1` | new: 69 cells, 4,416 teams, 216,384 battles |
| Guidance 2 | `sched_linear_species_first` | stored, issue #297 |
| Guidance 4 | `sched_linear_species_first_g4` | stored, issue #297 |
| Guidance 8 | `sched_linear_species_first_g8` | stored, issue #297 |

The ticket estimated 159,936 battles on asks 1/24 to 17/24 alone. The grid runs all 23 asks, so asks 18/24 to 23/24 can be reported apart as the map requires and the grid matches the stored ones cell for cell. Cost: 216,384 battles, about 40 minutes of battles after about 30 minutes of sampling on the laptop, judging by the guidance 4 grid of issue #329.

## Measures

For each arm:

1. Ask-versus-got error on asks 1/24 to 17/24, with its 95% interval, and the error on asks 18/24 to 23/24.
2. Got at each ask.
3. Distinct line-ups per 64 teams: the number of different line-ups in one cell (one ask, one sampling seed), averaged over the cells of asks 1/24 to 17/24.
4. Paired difference of each guided arm's error minus guidance 1's, averaged over sampling seeds.

The interval is a bootstrap with 4,000 draws: teams resampled within each ask and sampling seed, separately in each arm, and the 49 opponents resampled once for all four arms (`masking_rate_rows.py`, `species_late.py read_row`). The guided arms keep the team-resampling streams of `masking_rate_rows.py`; guidance 1 gets its own.

## Rule

- **Near the line** if guidance 1's error is under 0.001.
- For each guided arm against guidance 1: **nearer** if the paired difference is at most −0.001 and its interval lies below zero; **farther** if the interval lies above zero; otherwise **no measurable change**. 0.001 is the map's bar.

## Prediction

Guidance 1 lands farther from the line than guidance 2, so guidance 2 reads **nearer**, with an error at guidance 1 between 0.008 and 0.03. Got at guidance 1 should rise more slowly with the ask than at guidance 2: on the older 48-field model, guidance 1 at ask 0.5 got 0.362 against 0.127 with no ask, and the October guidance 1 grid of an earlier model (`results/asked_vs_got.json`) stayed between 0.12 and 0.22 across asks 0.0 to 0.9. Guidance 1 should write at least as many line-ups per 64 teams as guidance 2.

## What it decides for the map

The map splits the 84-field model's miss into three parts. The data limit is 0.0038, the error of re-battled training teams (issue #72). The fit is guidance 1's error above that limit. The guidance share is guidance 1's error minus guidance 2's. The ticket What limits the line, and what should change first? (#74) reads the three side by side.

The training-teams limit came from the all-teams set on the older table, while this checkpoint trained on the shuffled-only table, so the split is approximate.

## Run

`~/vgc-data/no_guidance_line/ngl.zsh`, a frozen copy of this commit in `run/`, queued behind other sessions' jobs. It runs the probes, sampling and battles of `ask_grid84.py` with `AG84_MODEL=sched_linear_species_first AG84_GUIDANCE=1`, saves the per-opponent win rates, then `python src/no_guidance_line.py read`, the figure and the comment, which it posts to #73.

## Result, 2026-11-15

Posted to #73: https://github.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/issues/73#issuecomment-6903180083

| Arm | Error, asks 1/24 to 17/24 | Against guidance 1 | Error, asks 18/24 to 23/24 | Line-ups per 64 teams |
|---|---|---|---|---:|
| Guidance 1, no push | 0.0162 [0.0112, 0.0224] | baseline | 0.2093 | 58.1 |
| Guidance 2 | 0.0075 [0.0039, 0.0126] | nearer, −0.0086 [−0.0108, −0.0067] | 0.1521 | 57.5 |
| Guidance 4 | 0.0049 [0.0022, 0.0092] | nearer, −0.0112 [−0.0139, −0.0087] | 0.1210 | 57.2 |
| Guidance 8 | 0.0038 [0.0017, 0.0074] | nearer, −0.0124 [−0.0156, −0.0094] | 0.1087 | 53.1 |

With no guidance the model is not near the line. Every guided arm is nearer. The prediction held on all three parts. For the map: data limit 0.0038, fit above it 0.0124, guidance 2 takes away 0.0086.
