# The DDOM paper's bin weights on the 84-field model

Pre-registration, written 2026-11-02 before any run. Map: GitHub issue #109. Status: **exploratory**, because each arm has one training seed.

In these terms, an ask is the win rate the model is told to aim for, got is the win rate its teams then score, and a line-up is a team's six species.

DDOM is the paper's method, Denoising Diffusion Optimization Models. Its equation 5 gives each win-rate bin a weight: `|B| / (|B| + K) * exp(-(best label - bin middle) / tau)`. Tau sets how fast weight falls away from the best label: a smaller tau favours high labels more strongly.

## Contract

| Row | Value |
| --- | --- |
| Claim | Weighting training teams by equation 5 of "Diffusion Models for Black-Box Optimization" (DDOM) brings the ask-versus-got slope closer to 1 than no weights. |
| What refutes it | Neither weighted arm beats No weights by 0.05 with the interval above zero. |
| What changes if it holds | The better arm's weights join the training recipe, if win rate at the top asks is no worse and line-ups do not collapse. |
| How far it reaches | Win rate under the behaviour-cloning policy against the top-50 meta teams, no further. |
| Fields | The model generates all 84 fields. A repair step replaces Stat Alignment to match the moves. Nothing is copied. |
| Vocabulary | The Regulation M-B tables directly: 348 species, 201 ability, 150 item, 498 move and 27 Stat Alignment tokens. Labelled training teams use 189, 126, 106, 309 and 19 of them. |
| Training teams | Only teams with a known win rate against the top-50 meta teams: 170,785 labelled teams from `lossdata84.npz`, before the rule tables drop any. Validation holds the 3,563 labelled teams in the existing 2% split. The 130,232 unlabelled teams, the 500 test teams and the held-out teams ranked 51 to 100 stay out. |
| Scorer | One battle against each of the 49 distinct top-50 meta teams, behaviour-cloning policy on both sides, slots shuffled. Win rate means this throughout. |
| Search region | No search and no elite rule. Asks 1/24 to 23/24, guidance 2, temperature 1, rule masks on, species first. Each cell draws until 64 teams pass Showdown and are not held-out teams, at most 1,280 tries. |

### Settings

| Setting | Role | Value |
| --- | --- | --- |
| Tau | Scientific | No weights, 0.1 or 0.3 |
| Sampling seed | Nuisance | 1, 2, 3 |
| Weight formula | Fixed | Equation 5, K = 1% of the teams, 20 equal-width bins over the true labels, weights averaging 1 |
| Label noise | Fixed | ±1/48, on the condition only. Bins use the true label |
| Epochs | Fixed | 21, with the learning-rate schedule sized to a cap of 80 |
| Training seed, batch, learning rate, network | Fixed | 0, 256, 4e-4, 256 wide with 6 layers and 8 heads |

Equation 5 sets no bin count, so 20 comes from the code the paper links. That code uses a different formula, which this run does not test. Validation loss stays unweighted, so the three arms share one scale.

### Plan

- **Primary measure:** the slope of mean got on ask, over all 23 asks.
- **Second line:** the slope over asks 1/24 to 17/24, the only one comparable with the training teams' 0.715.
- **Reported beside each slope:** win rate per ask, line-ups, new line-ups against the fixed reference, and the share of Pokemon spending all 66 Stat Points.
- **Question:** better, with the interval above zero.
- **Smallest effect worth acting on:** a slope change of 0.05.
- **Random-search baseline:** none, because a slope needs an ask and random search takes none.
- **Second comparison:** No weights against Unlabelled teams included. It shows what removing unlabelled teams does. The research note found no paper that tests this directly.

## Arms

All arms use label noise ±1/48, seed 0, 23 asks, 3 seeds, 64 teams per seed, 49 battles per team.

| Reader name | Code ID |
| --- | --- |
| Tau 0.1 weights, 21 epochs | `gen84_ddom_tau01` |
| Tau 0.3 weights, 21 epochs | `gen84_ddom_tau03` |
| No weights, 21 epochs | `gen84_noise_byepoch_ep21` |
| Unlabelled teams included, 21 epochs | `gen84_all_noise_ep21` (existing) |

No weights is epoch 21 of the slope-by-epoch training, so it needs no training of its own. All three follow one recipe to epoch 21, because each schedule is sized to the same cap. The weighted arms use no patience, so each saves exactly epoch 21 and never rolls back to an earlier best. If the slope-by-epoch training stops before epoch 21, the control is missing, its queue stops, and this run does not start.

## What the weights do

Share of training weight on the 170,785 labelled training teams, before the rule tables drop any. `code/scripts/test_ddom_weights.py` covers the formula.

| Label band | Share of teams | Tau 0.1 | Tau 0.3 |
| --- | --- | --- | --- |
| Under 1/4 | 13.02% | 0.35% | 3.86% |
| 1/4 to 1/2 | 48.20% | 18.00% | 39.95% |
| 1/2 to 17/24 | 37.94% | 78.66% | 55.38% |
| 17/24 and above | 0.85% | 2.99% | 0.81% |

Risk: at tau 0.1, low labels nearly vanish, and an earlier run on an older model found that removing low-scoring teams cost 5 to 11 points, because the condition needs a contrast.

## Noise floor and budget

Slope noise is a bootstrap over teams on the 21-epoch model without label noise: 0.005 on 23 asks with 3 seeds. A gap of about 0.02 between two arms is detectable at 5% false positives and 80% power, below the 0.05 bar. Training-seed noise is not included, and one seed per arm cannot measure it.

Budget: 216,384 battles per weighted arm, about 46 minutes. Training takes about 1.5 hours per arm, estimated from 4.2 minutes per epoch. The two arms take about 4.5 hours in total.

## Probes

| Probe | Result |
| --- | --- |
| Training smoke | Passed 2026-11-02 on 512 labelled teams at tau 0.1: validation loss 2.825. The queue repeats it before each training. |
| Weight formula | Passed 2026-11-02: reproduces the eight shares measured separately on all 174,348 labelled teams. |
| Ceiling | Passed on asks 1/24 to 17/24: the training teams' 0.715 sits 0.071 above the 21-epoch label-noise model's 0.644. Not measured on all 23 asks. |
| Round trip, coverage, read teams, controls, separation, condition, copies | Run by the queue on each checkpoint before its battles. A failure stops the queue. |
| Invariance, repeat | Skipped: the scorer is unchanged, and every battle shuffles slots. |

## Deviations

Before any training, code review found that patience could stop or roll back a weighted arm before epoch 21. I removed patience from both weighted arms on 2026-11-02, with no result in hand.
