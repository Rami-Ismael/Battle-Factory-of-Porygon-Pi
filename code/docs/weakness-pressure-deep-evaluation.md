# Weakness pressure and deep evaluation: pre-registration (2026-11-15)

Status: part A **done 2026-11-18**: 16 battles per meta team becomes the re-battle depth at guidance 2, 4 and 8 (flip share 0.250, 0.243, 0.213; every 95% interval above 0.10; results on #345); part B **draft**. Written before any battle of either arm. Source idea: Li et al. 2026, *Beyond Static Evaluation*, [arXiv 2606.10389](https://arxiv.org/abs/2606.10389), §4.5 (weakness pressure) and §4.6 (deep evaluation). Glossary: **Weakness pressure**, **Deep evaluation** (added 2026-11-15).

Owner decisions, 2026-11-15: put weakness pressure in a new 84-field elite loop, not `cem_v3.py`; judge it as a Pareto question, fixed-weight win rate against worst-matchup win rate; test deep evaluation at 4 against 16 battles per meta team.

Prior work searched: `experiments-index.md` (no run reweights opponents; `novel-elite-rebattle-2026-10-31` and `line_up_yield.py` set today's 4-battle re-battle), `code/docs/` (no match for opponent reweighting). Web, 2026-11-17, for re-evaluation depth and verdict flips after selection: no direct match; the nearest framing is best-arm identification under a trial budget (re-battling only the line-ups near the cut), noted under Noise and budget.

Map: #344. Tickets: part A #345; part B settings #346 (grilling); part B run #347, blocked by both.

Two parts, run in this order because part B uses part A's depth.

## Part A: deep evaluation of elite line-ups

| Row | Value |
| --- | --- |
| Claim | Re-battling elite line-ups at 4 battles per meta team flips the held / not-held verdict for a share of line-ups large enough to change counts reported on #297 and #335. Refuted if fewer than 10% of verdicts flip at 16 battles. If it holds, every elite yield since 2026-11-12 is re-read at 16 and new grids re-battle at 16. Scope: win rate under the behaviour-cloning policy against the top-50 meta. |
| Fields | No generation. The teams are the stored best team per elite line-up of the control grid, `line_up_yield_sched_species_first*.json`. |
| Vocabulary | Not applicable: no model runs. |
| Training teams | Not applicable. |
| Scorer | `matchup_db.score_pastes`; behaviour-cloning policy on both sides; slots shuffled; the 49 distinct top-50 meta teams, the duplicated one weighted twice (weights sum to 50). The 4-battle re-battle (4 battles per meta team): the stored `fresh` results in `line_up_yield_sched_species_first*.json`, seed 7520, database `~/vgc-data/masking_rate_battles/fresh.sqlite`. The 16-battle re-battle (16 battles per meta team): `LY_PER_CELL=16`, seed 1,007,520 (worker seeds are seed + job id, so the two re-battles' seed ranges never meet), its own database `~/vgc-data/deep_evaluation/fresh_d16.sqlite`, results `line_up_yield_<grid>_d16.json`. A database records its depth and refuses another (`claim_depth`): in a shared database a cell asked for 16 would be topped up from the stored 4, and the two re-battles would share those battles. The re-battle also refuses a team holding more than 16 per meta team. |
| Search region | The elite line-ups of the control grid, the per-field masking schedule decoded species first (checkpoint `stat84_shuffled_noise_per_field_es3_cap80`), re-battled on 2026-11-13: 272 at guidance 2, 334 at guidance 4, 352 at guidance 8 (`rebattled` in the `all_23_asks` view). The 16-battle re-battle battles only these teams, not the other two views' best teams that the 4-battle re-battle also covered (309, 372, 393 teams). |
| Settings | Scientific: battles per meta team (4, 16). Nuisance: the battle seed (new for the 16-battle re-battle). Fixed: elite cut 0.531, policy, opponents, the team per line-up. |
| Plan | Primary measure (`line_up_yield.py flips`): the share of line-ups whose verdict at 16 differs from the verdict at 4, with a 95% bootstrap interval over line-ups (2,000 resamples). Second: held counts at 4 and 16 per guidance. A single measure with nothing traded against it except cost, so the question is whether the flip share exceeds 10%. Budget: 958 line-ups × 49 × 16 = 751,072 battles; guidance 2 alone is 213,248. Stop after guidance 2 if its flip share's interval lies below 10%. Baseline: the 4-battle re-battle is today's rule. |

### Arms

- **4 battles per meta team** (today's re-battle) `rebattle_4`
- **16 battles per meta team** `rebattle_16`

### Probes

Generation probes (round trip, coverage, read teams, training smoke, condition, copies) do not apply: no model runs. Run before launch: **controls** (5 top-50 teams and 5 uniform-legal teams from `~/vgc-data/teams/hps_reg_mb_100k.jsonl` through the same `score_pastes` call at 16, near 0.50 and near 0.015), **repeat** (one line-up twice at one seed: same schedule, scores within battle noise), **separation** (no held-out team among the 958). Save each in the result JSON under `preflight`.

Results, 2026-11-17, saved under `preflight` in `~/vgc-data/results/line_up_yield_sched_species_first_d16.json`; all pass:

| Probe | Result |
| --- | --- |
| Controls, 16 per meta team | 5 top-50 teams 0.550; 5 uniform-legal teams 0.018 |
| Repeat, one line-up twice at one seed | identical opponent schedule and slot orders; 0.440 and 0.464 over 784 battles each, gap 0.024 inside the 99% bound 0.061 |
| Separation | 0 held-out teams among the 272, 334 and 352 teams |
| The 4-battle arm | every line-up present at all three guidances, 196 battles each, seed 7520 |
| Independence | `fresh_d16.sqlite` holds no cells and records depth 16 |
| Prediction | flip share 0.24, 0.24, 0.20 at guidance 2, 4, 8 (see Noise and budget) |
| Smoke | 2 teams at 1 per meta team through `rebattle` into a scratch database: 49 battles each, depth recorded; a smoke now defaults to `fresh_d16_smoke.sqlite` |

Invariance is skipped: every battle shuffles slots and the re-battle refuses any other order.

### Figures (storyboard)

Storyboarded 2026-11-17 after guidance 2, before guidance 4 and 8 (`storyboard-figures`). `deep_evaluation_post.py figure` draws all three with ECharts; the queue posts them on #345 after each guidance, in this order. Colors keep one meaning in all three: blue held at both depths, orange held at 4 only, green held at 16 only, grey held at neither, violet the flip share, amber the prediction.

| Frame | The reader should take | Expected | Would surprise |
| --- | --- | --- | --- |
| **Scores** (teaser): each elite line-up's best team, 4-battle score across, 16-battle score up, the cut on both axes, one panel per guidance | Verdicts flip for line-ups whose 4-battle score sat near the cut; those far from it keep their verdict | Flipped line-ups within 0.07 (two standard errors at 196 battles) of the cut; the cloud on the diagonal, no shift | Flips far from the cut (pairing or scorer fault), or 16-battle scores shifted below the diagonal (a bias in the stored 4-battle scores) |
| **Flip share**: share of verdicts that flip per guidance, its 95% interval, the prediction from battle noise alone, the 0.10 rule | About a quarter of verdicts flip at every guidance, well above the rule and where battle noise predicts | 0.20 to 0.25, each interval above 0.10 and around its prediction | A share far below its prediction (the re-battles share battles, or scores are less noisy than binomial) or far above it (noise beyond binomial: the policy, Showdown's random numbers) |
| **Held**: line-ups held at each depth, split into held at both and held at one only | The held counts reported on #297 and #335 barely move, but a quarter of the line-ups they count are different ones | Counts within 10% of each other; a quarter to a third of held line-ups lost | A large net move in one direction, which would mean the 4-battle counts were biased, not only noisy |

Each frame needs both re-battles and nothing else; no planned measurement lacks a frame. Titles state the figure's sentence from the data (`scores_title`, `held_title`), so a result that misses the sketch changes the title instead of contradicting it.

Guidance 2 (2026-11-17): 96% of the 68 flipped line-ups sat within 0.07 of the cut; flip share 0.250 against a prediction of 0.244; held 114 at 4 and 118 at 16, with 32 lost and 36 gained, so 28% of the line-ups held at 4 are not held at 16. All three frames came out as sketched.

### Commands

```
cd code/src
LY_PER_CELL=16 LY_MODEL=sched_species_first python line_up_yield.py preflight   # probes, private databases
LY_PER_CELL=16 LY_MODEL=sched_species_first python line_up_yield.py rebattle    # 213,248 battles
LY_PER_CELL=16 LY_MODEL=sched_species_first python line_up_yield.py report      # all-23-asks view at 16
LY_PER_CELL=16 LY_MODEL=sched_species_first python line_up_yield.py flips       # flip share, verdict by the rule
```

Then guidance 4 and guidance 8 (`LY_MODEL=sched_species_first_g4`, then `sched_species_first_g8`) unless guidance 2 stops the run. `LY_PER_CELL` takes 4 or 16 only. Tests: `scripts/test_deep_evaluation.py`.

### Noise and budget

One line-up's win rate has standard error 0.036 at 196 battles (4 per meta team) and 0.018 at 784 (16 per meta team), at p = 0.5. On 2026-11-13, 124 of 272 guidance-2 line-ups sat near the cut, so a flip share well above 10% is plausible. The bootstrap over 272 line-ups resolves a flip share to about ±0.05. **Confirmatory.**

Predicted before any 16-battle result (the `prediction` probe, 2026-11-17): battle noise alone, with each team's true win rate drawn from an empirical-Bayes posterior fitted to the spread of the 4-battle scores, gives a flip share of 0.24 (90% of draws 0.20 to 0.28) at guidance 2, 0.24 (0.20 to 0.27) at 4 and 0.20 (0.17 to 0.23) at 8. A flat prior gives 0.19, 0.19 and 0.17. Corrected after the code review (2026-11-17) to count the duplicated meta team's double weight (48.08 effective columns, not 49): 0.244 (0.20 to 0.29), 0.239 (0.20 to 0.28) and 0.200 (0.17 to 0.23). Every version lies above 10%, so the rule's verdict, 16, is close to settled before the run; the run's value is the held counts at 16, which re-read #297 and #335, and the check that the noise model holds. A measured share far below the prediction would mean the 4-battle scores are less noisy than binomial, or the two re-battles share something they should not.

Time: the 4-battle re-battle ran 60,564 battles in about 9 minutes on 2026-11-13, so guidance 2 at 16 takes about 35 minutes and all three about 2 hours on the laptop. A cheaper design, re-battling only line-ups within 0.036 of the cut (124 of 272 at guidance 2), would answer the held counts but not the flip share the rule reads, so the run keeps every line-up.

## Part B: weakness pressure in an 84-field elite loop

| Row | Value |
| --- | --- |
| Claim | An elite loop that picks elites under weakness pressure finds teams whose worst matchups are stronger, without losing fixed-weight win rate. As a Pareto question: on (fixed-weight win rate, worst-matchup win rate) the weakness-pressure curve moves outside the fixed-weight curve. Refuted if its curve lies inside or on the fixed-weight curve at every generation. If it holds, the loop uses weakness pressure to pick elites. Scope: win rate under the behaviour-cloning policy against the top-50 meta; it says nothing about the ladder. |
| Fields | All 84 fields generated by the model; none copied. |
| Vocabulary | The default model's token lists against the regulation tables, read by the coverage probe. |
| Training teams | Start: the default model `gen84_noise_byepoch_ep80` (80-epoch labelled). Each generation fine-tunes it (`Model84.fine_tuned`) on that arm's elites. Held-out teams excluded (`heldout.py`). |
| Scorer | Labelling: one battle against each of the 49 distinct meta teams, slots shuffled, `matchup_db`. Final reading of each run's best line-ups: deep evaluation at the depth part A picks. |
| Search region | Per generation: 256 teams sampled at ask 17/24, guidance 2, temperature 1, then battled. Elites: the cumulative elite rule (cut 0.531, floor 64), ranked by the arm's score. 5 generations. These settings are proposed and need the owner's approval. |
| Settings | Scientific: the opponent weights that rank elites (fixed, or weakness pressure). Nuisance: training seed and sampling seed, 3 each, paired across arms. Fixed: model, ask, guidance, generation count, elite rule, battle budget. |
| Plan | Primary measure: each run's top 8 line-ups by fixed-weight win rate, deep-evaluated, giving fixed-weight win rate and worst-matchup win rate. Worst-matchup win rate is the mean over the team's 5 hardest meta teams, chosen from the labelling battles and measured on the deep-evaluation battles, so the choice and the measurement do not share noise. Pareto: plot both arms' points per generation; the strongest previous frontier is the no-loop baseline's best draws. Smallest effect worth acting on: +0.05 worst-matchup win rate at no more than −0.01 fixed-weight win rate. Budget per run: 5 × 256 × 49 = 62,720 labelling battles, plus 8 × 784 = 6,272 for deep evaluation; 6 loop runs and 3 baseline runs at the same budget, 620,928 battles in all. Report diversity (distinct line-ups, effective species) beside win rate. |

### Arms

- **Weakness pressure, elite loop** `loop84_weakness`: after each generation, the meta team the generation's best team beats least often gets double weight and the rest shrink to keep the sum (Li et al. Eq. 3–4); every labelled team is re-scored from its stored per-opponent results, so no new battles.
- **Fixed weights, elite loop** (the control) `loop84_fixed`: the duplicated meta team weighted twice, the rest once, every generation.
- **No loop, default model's draws** (baseline) `draws84_noloop`: 1,280 teams sampled once at the same ask and guidance, battled at the same budget; its best line-ups deep-evaluated.

### Probes

Not run yet; each runs once the loop is built: round trip, coverage, read teams, training smoke (fine-tuning on 64 elites lowers their loss), controls, invariance (slot order), repeat, separation, condition (ask 6/24 against 17/24 at one seed), copies. **Ceiling** first: on stored grids, re-rank the default model's guidance-2 teams by weakness-pressure weights and check that the top line-ups differ from the fixed-weight top line-ups; if they are the same, the loop cannot move and the run stops.

### Noise and budget

Labelling scores have standard error 0.071 per team. Worst-matchup win rate over 5 meta teams at 16 battles each is 80 battles, standard error 0.056 per team; over 8 line-ups per run and 3 seeds, about 0.011 from battles alone, before seed variance. Three training seeds may not separate +0.05 if seeds vary as much as on #297; then the run is **exploratory**.

## Build

Part A needs a depth argument on `line_up_yield.py rebattle`. Part B needs a new loop on the 84-field model: the sampling, battling and fine-tuning steps exist (`ask_grid84`, `matchup_db`, `Model84.fine_tuned`); the weight update and the re-scoring are new.
