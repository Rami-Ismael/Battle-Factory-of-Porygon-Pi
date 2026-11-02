# Remasking on the all-teams 84-field model

Exploratory. The owner asked on 2026-11-01 what ReMDM does and whether it works on the 84-field model, and set the bar: the ask-versus-got slope should move closer to 1; a gain at the top asks is a bonus. Paper: Wang, Schiff, Sahoo and Kuleshov, [Remasking Discrete Diffusion Models with Inference-Time Scaling](https://www.alphaxiv.org/abs/2503.00307), NeurIPS 2025; its code: [kuleshov-group/remdm](https://github.com/kuleshov-group/remdm). Script: `src/remask84.py`. The 2026-10-30 run did not test the paper's selection rule, so its failure belongs to least-likely selection only.

## Contract

| Row | Value |
| --- | --- |
| Claim | Does remasking move ask-versus-got closer to the line, against the top-50 meta teams under the behaviour-cloning policy? The claim stops there. |
| Fields | All 84 generated, except that the repair replaces Stat Alignment, as in `ask_grid84.py`. Nothing copied. |
| Vocabulary | Unchanged from the checkpoint `stat84_all_teams_ep21.pt`; its coverage probe is in the grid's `preflight`. |
| Training teams | Unchanged: `lossdata84.npz` all-teams rows, 304,580 kept of 308,336, 2% validation. No training here. |
| Scorer | One battle against each of the 49 distinct top-50 meta teams, behaviour-cloning policy on both sides, slots shuffled, seed 7300 + sampling seed. |
| Search region | No loop. Species-first: each cell draws until 64 legal, non-held-out teams or 1,280 tries. Random decoding: 384 tries per cell, no stop at a number of legal teams, no battles yet. |
| Settings | Scientific: the remasking rule and decoding order. Nuisance: asks 4/24, 8/24, 12/24, 16/24, seeds 1 to 3. Fixed: guidance 2, temperature 1, rule masks. |
| Plan | Primary: ask-versus-got slope against no remasking, per seed, with and without 12/24. Bonus: win rate at the top ask. Reported beside them: species accumulation curves of line-ups and of line-ups at or above 0.531, and line-ups new against the fixed reference of 11,690. Species-first budget: 3 arms x 4 asks x 3 seeds x 64 teams x 49 battles = 112,896 battles, control reused. Random decoding budget: set after the probe. |

Ask 12/24 is read apart: it selects a different team population ([label noise protocol](label-noise-84-fields.md)).

## Arms, species-first decoding

Shared by every arm: all-teams 84-field model, 21 epochs, species first. Amounts are per pass. Queue: `~/vgc-data/remask84/rm84.zsh`.

| Reader name | Code ID | Rule |
| --- | --- | --- |
| No remasking, 84 passes | `plain` | Existing grid, `stat_points_84_gen84_all_ep21.json` |
| Least-likely 10% remasked, 84 passes | `remask_ll_10` | The 2026-10-30 rule: always the lowest-probability filled fields, redrawn in the same pass |
| Confidence-weighted 0.3 fields remasked, 210 passes | `remask_conf_loop` | The paper's confidence odds, 0.3 x exp(-p) / sum(exp(-p)), inside a 126-pass loop once 63 fields are filled; redrawn in the same pass |
| Even-odds 0.5 fields remasked, 210 passes | `remask_even_loop` | The paper's even odds, 0.008 per filled field, in the same loop; redrawn in the same pass |

## Random decoding legality probe

Sampling only, to set the try cap and battle budget. Shared by every arm: each masked field unmasks independently, as in the paper; a remasked field waits until the unmasking odds pick it. Queue: `~/vgc-data/remask84/rd84.zsh`, 3 arms x 4 asks x 3 seeds x 384 tries = 13,824 tries. Results: `remask84_random_decoding.json`, summarised under Result. A pass here is one step of the schedule; the saved counts, 84 and 210 per batch, match the arm names.

| Reader name | Code ID |
| --- | --- |
| No remasking, 84 passes | `rand_plain` |
| Confidence-weighted 0.3 fields remasked, 210 passes | `rand_remask_conf` |
| Even-odds 0.5 fields remasked, 210 passes | `rand_remask_even` |

## Where the arms depart from the paper

| Setting | Paper and its code | Here |
| --- | --- | --- |
| Filled token | Kept unless remasked: the code's substitution step gives it probability 1 | Same |
| Unmasking order | Independent per token | Random-decoding arms match; species-first arms keep the fixed order |
| Draws in one pass | Independent, from one set of logits | One set of logits, drawn in fixed order so legality sees earlier draws |
| Remasked token | Back to the pool | Random-decoding arms match; species-first arms redraw in the same pass |
| Confidence in the loop | Text combines them; the code's text loop uses even odds only, and gives masked tokens zero odds under confidence, so they would stay masked while alpha is held | Masked fields unmask at 0.3 / masked count, which keeps the count level. My choice |
| Loop | t_on 0.25 with molecule guidance; T up to 4L | 63 of 84 fields filled; 63 + 126 + 21 = 210 passes |
| Even-odds value | 0.008, the best of three MAUVE settings on text | 0.008; no value published for this data |
| Legality | None | Rule masks at each draw, with the fallback to all values when none is allowed |

## Noise floor

One arm at one ask has 9,408 battles, so battle noise is about 0.005. Across four asks, 37,632 battles. A 0.03 gap needs about 4,400 battles per arm, so the budget can detect it. Three sampling seeds are the only repeat of generation, so sign agreement across seeds is reported beside each difference.

## Probes

Inherited from the grid, same checkpoint and scorer: coverage, training smoke (no training here), controls, invariance, repeat, separation. To run per arm: round trip, read teams, condition (asks change win rate), copies. `remask84.py check` proves the no-remasking path is bit-identical to `Model84.sample`.

## Result

### Species-first decoding

None of the three arms moves the slope closer to 1. Win rate is against the 49 distinct top-50 meta teams, behaviour-cloning policy on both sides, slots shuffled. Each arm has 768 teams and 37,632 battles. Differences are against no remasking, with a bootstrap interval over teams and opponents.

| Arm | Slope (difference, interval) | Slope difference by seed | Win at 16/24 (difference, interval) | Line-ups of 768 |
| --- | --- | --- | --- | --- |
| No remasking, 84 passes | 0.338 | | 0.404 | 438 |
| Least-likely 10% remasked, 84 passes | 0.239 (-0.099, -0.171 to -0.025) | -0.140, -0.045, -0.112 | 0.448 (+0.044, 0.015 to 0.076) | 228 |
| Confidence-weighted 0.3 fields remasked, 210 passes | 0.323 (-0.015, -0.091 to 0.059) | -0.056, +0.024, -0.014 | 0.416 (+0.012, -0.017 to 0.042) | 440 |
| Even-odds 0.5 fields remasked, 210 passes | 0.303 (-0.036, -0.110 to 0.036) | -0.084, -0.002, -0.021 | 0.391 (-0.014, -0.042 to 0.016) | 437 |

Without 12/24 the slopes are 0.291, 0.196, 0.287 and 0.251 in the same order. Least-likely selection misses the slope bar, gains win rate at every ask and loses line-ups, so it is a trade to weigh, not a plain failure. The paper rules keep 92% and 92% of the no-remasking line-ups (401 and 404 of 438). With species set first they have little to change, so random decoding is their real test.

Line-ups new against the 11,690 fixed reference: 335, 147, 333 and 333 in the same order.

Per-seed slopes, paired intervals, novelty and accumulation are in `remask84_slope_analysis.json`, written by `src/remask84_slope_analysis.py`. `remask84.py report` writes only the pooled slopes and curves, and it now leaves the fit empty until all four asks have battled.

### Random decoding

Legal share of 1,152 tries per ask, three seeds. All cells finished and the round trip passes on 20 of 20. These are few legal teams: 122 across 13,824 tries, a count that excludes held-out teams. Try cap and battle budget are still to be set in issue #91, so there is no slope yet.

| Arm | 4/24 | 8/24 | 12/24 | 16/24 | Mean |
| --- | --- | --- | --- | --- | --- |
| No remasking, 84 passes | 1.56% | 1.48% | 1.22% | 0.69% | 1.24% |
| Confidence-weighted 0.3 fields remasked, 210 passes | 0.95% | 0.87% | 1.04% | 0.52% | 0.85% |
| Even-odds 0.5 fields remasked, 210 passes | 0.43% | 0.61% | 0.87% | 0.35% | 0.56% |

### Provenance

The results came from the frozen copy in `~/vgc-data/remask84/run/`. The script here differs from that copy in the report's guard on partial grids and a lint refactor of `sample`; on a stand-in network, both copies draw identical teams in every species-first arm. `ask_grid84.py` and `diffusion84.py` here are newer than the frozen copies; their changes add label noise, off-grid asks and a checkpoint guard, and leave `Model84.sample` as it was.

### Pilot before the battles

Sampling-only pilot, seed 1, 64 legal teams per cell, kept under its earlier code IDs in `remdm84_pilot.json` (`remask10`, `remdm_conf`, `remdm_cap`). After seeing it, the owner set the slope as the bar, and I renamed the arms and added random decoding. Line-ups out of 64 at asks 4/24, 8/24, 12/24, 16/24:

| Arm | Line-ups | Exactly 66 Stat Points |
| --- | --- | --- |
| No remasking, 84 passes | 59, 38, 61, 26 | 27%, 27%, 28%, 25% |
| Least-likely 10% remasked, 84 passes | 34, 10, 42, 9 | 62%, 67%, 45%, 57% |
| Confidence-weighted 0.3 fields remasked, 210 passes | 59, 38, 61, 26 | 27%, 29%, 27%, 24% |
| Even-odds 0.5 fields remasked, 210 passes | 59, 38, 61, 26 | 26%, 27%, 25%, 27% |

Probes on 96 teams at ask 8/24 pass: round trip 20 of 20, no team identical to a training team, at most 2 shared slots with the nearest one.
