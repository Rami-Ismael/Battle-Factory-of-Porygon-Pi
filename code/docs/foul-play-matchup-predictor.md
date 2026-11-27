# Can a matchup predictor trained on 500 Foul Play-labelled teams label the rest?

Pre-registration, 2026-11-27, before running cross-validation on the 500 pilot teams. Issue: [Test whether a matchup predictor trained on 500 Foul Play-labelled teams can label the rest](https://github.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/issues/419), on the map [Offline Rust search labeling for the surrogate and inverse-diffusion dataset](https://github.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/issues/416). Mode: **confirmatory** for the three pre-registered arms under the ridge matchup predictor; **exploratory** for the follow-up sensitivity and nonlinear checks.

## Why this run

Battling all ~439,600 eligible training teams against the 49 meta teams with **Foul Play deterministic argmax** (`rust_foulplay_det`, 30,000 iterations, 1 sample) would take roughly 62 laptop days. Issue #447 battled 500 random training teams (`pilot_teams.json`, seed 447, `policy_id = 10`, 2 battles per meta cell) and confirmed that Foul Play deterministic argmax gives the team pairing almost three times as much of a battle's outcome (Matchup ICC 0.176 vs behaviour cloning's 0.063) with similar split-half label reliability (*r* = 0.857 vs 0.869), while correlating only 0.783 with behaviour cloning (`policy_id = 8`).

This experiment tests whether a matchup predictor trained on those 500 teams can predict each team's Foul Play deterministic argmax win rate accurately enough to label the remaining ~439,600 teams—specifically, whether it beats simply reusing each team's existing behaviour-cloning win rate as the guess.

## Arms

Shared by every arm: 5-fold cross-validation on the 500 pilot teams (`pilot_teams.json`, seed 447), using the exact same five disjoint 100-team folds (`seed = 419`), scored against each team's Foul Play deterministic argmax win rate over the 39 training opponents (`policy_id = 10`, 2 battles per cell = 78 battles per team).

| Reader name | Code ID | Inputs |
| --- | --- | --- |
| Team + behaviour-cloning win rate | `predictor_team_bc` | The team's 84 fields + its stored behaviour-cloning win rate (`policy_id = 8`, 39 training opponents) |
| Team only, no behaviour-cloning win rate | `predictor_team_only` | The team's 84 fields only |
| Behaviour-cloning win rate as the guess | `guess_bc_label` | No training; the stored behaviour-cloning win rate (`policy_id = 8`, 39 training opponents) is the prediction |

### Secondary diagnostic controls (not replacing the three arms)

- **Linear-calibrated behaviour-cloning win rate** (`calibrate_bc_only`): ordinary least squares `a + b * bc_wr` fit on each fold's 400 training teams, with no team features. Foul Play averages 0.047 lower than behaviour cloning (0.262 vs 0.309) with a smaller SD (0.152 vs 0.178), so this control separates fixing the global mean/scale shift from learning team-specific discrepancies between the two labelers.
- **Training-fold mean as the guess** (`guess_train_mean`): constant mean of each fold's 400 training teams, giving the zero-information error ceiling.

### Secondary sensitivity checks (exploratory, reported apart)

1. **Deployment-noise check (1 battle per cell for behaviour cloning):** ~98% of the ~439,600 training teams have 1 battle per opponent under `policy_id = 8`, whereas #447 topped the 500 pilot teams up to ≥ 2 battles per opponent. We re-score `predictor_team_bc`, `calibrate_bc_only`, and `guess_bc_label` using a single drawn battle per cell (seed 419) for the behaviour-cloning input.
2. **Unseen-opponent proxy check (10 validation opponents):** Evaluate how well the out-of-fold predictions (trained on the 39 training opponents) predict each pilot team's Foul Play deterministic argmax win rate against the 10 validation opponents (kept strictly out of training labels and inputs).
3. **Lineup-grouped 5-fold check:** Group the 500 teams by their 6-species lineup (436 distinct lineups, max multiplicity 28) so no lineup spans train and test, verifying that team-feature gains do not come only from repeated lineups.
4. **Nonlinear model check (PyTorch residual MLP):** Same five folds, same team features (with and without behaviour-cloning win rate), early-stopped on an inner 20% validation split, to check whether nonlinear interactions improve over ridge regression on 400 training rows.

## Contract

| Row | Value |
| --- | --- |
| Claim | Under 5-fold cross-validation on the 500 pilot teams from #447, a ridge matchup predictor trained on the team and its existing behaviour-cloning win rate (`predictor_team_bc`), or on the team alone (`predictor_team_only`), achieves lower out-of-fold RMSE against the Foul Play deterministic argmax win rate than using the behaviour-cloning win rate directly as the guess (`guess_bc_label`). **Refuted** if the 95% paired bootstrap confidence interval of `arm RMSE − guess_bc_label RMSE` does not lie strictly below zero for either predictor arm. **If it holds**, the winning predictor design (and whether team features improve over linear BC calibration alone) informs how to label the ~439,600 training teams under map #416. **Scope:** stops at predicting the Foul Play deterministic argmax win rate against the 39 training opponents (plus the 10 validation opponents proxy check); does not measure downstream diffusion-model conditioning gain. |
| Fields | Input: all 84 team fields decoded from each team's canonical database paste (6 Pokémon × species, ability, item, 4 moves, Stat Alignment, 6 Stat Points) plus, for `predictor_team_bc` only, the scalar behaviour-cloning win rate over the 39 training opponents. None generated; none copied. |
| Vocabulary | Built strictly inside each training fold (400 teams) from the training teams' normalized species, abilities, items, moves, Stat Alignments, and species pairs appearing ≥ 5 times in that training fold, plus 10 permutation-invariant Stat Point summaries (6 per-stat team means, mean total Stat Points spent, mean max-stat focus, share with Spe ≥ 28, share with HP ≥ 28) standardized using training-fold mean and SD. Zero test-fold vocabulary or scaling leakage. |
| Training teams | The 500 pilot teams in `~/vgc-data/foul_play_label_pilot_447/pilot_teams.json` (seed 447, drawn from 439,554 eligible generated training teams with full `policy_id = 8` coverage; excludes meta teams, held-out ranks 51–100, and VGCPastes pool teams). Each fold trains on 400 teams and tests on 100. |
| Scorer | Stored battles in `~/vgc-data/matchup_regmb.sqlite`: target is `policy_id = 10` (`rust_foulplay_det`, 30,000 iterations, 1 sample, both sides, slots shuffled), 2 battles per opponent over the 39 training opponents (`training_opponents` in `pilot_teams.json`). Auxiliary input is `policy_id = 8` (`bc_100.zip`, both sides, slots shuffled) over the same 39 training opponents. No new battles are run. |
| Search region | No search loop. Inner regularization grid for ridge regression: single-field penalty `lam_s` ∈ {1, 3, 10, 30, 100, 300, 1000} and species-pair penalty `lam_p = 3 * lam_s`, selected per fold by exact closed-form leave-one-out CV (LOOCV) inside the 400 training teams. Intercept and (for `predictor_team_bc`) the scalar behaviour-cloning win-rate slope are unpenalized (`lam = 0`), implementing Kennedy–O'Hagan multi-fidelity residual shrinkage. Predictions are clipped to `[0, 1]`. |
| Settings | **Scientific:** predictor input set (`predictor_team_bc` vs `predictor_team_only` vs `guess_bc_label`). **Nuisance:** ridge regularization strength `lam_s` tuned independently per arm and fold via inner LOOCV so neither arm is handicapped by a mis-scaled penalty. **Fixed:** 5-fold split (`seed = 419`), `min_pair = 5`, Stat Point summary definitions, bootstrap resamples = 10,000 (`seed = 419`). |
| Plan | **Primary measure:** out-of-fold RMSE against the Foul Play deterministic argmax win rate over the 500 teams (monotone with MSE, directly comparable to the 0.0423 battle-noise RMSE floor and 0.1524 label SD). **Single-axis better question:** no accuracy–diversity trade-off exists for a fixed-team labeler; a predictor arm is "good enough" if its paired 95% bootstrap CI for `RMSE(arm) − RMSE(guess_bc_label)` lies strictly below 0 (smallest effect worth acting on: 0.005 RMSE). **Also reported:** MSE, MAE, $R^2$, Pearson *r*, Spearman ρ, per-fold RMSE and win/loss signs across the 5 folds, and the paired gap `RMSE(predictor_team_bc) − RMSE(calibrate_bc_only)` to test whether team features add signal beyond global linear rescaling of behaviour cloning. **Stopping rule:** one run of the pre-registered 5-fold CV; no post-hoc fold or feature cherry-picking. |

## Probes

Each probe runs inside `src/foul_play_matchup_predictor.py preflight` and saves its pass/fail verdict and numbers under `preflight` in the results JSON before the CV results are judged.

| Probe | Check | Pass criterion |
| --- | --- | --- |
| **Ceiling / battle-noise floor** | Irreducible RMSE floor from the two halves of `policy_id = 10` (`report_pilot.json`: game 0 vs game 1 per cell), `sqrt(mean((h0 - h1)^2) / 4)`. | Floor (`0.0423`) is far below `guess_bc_label` RMSE (`0.1206`), leaving ~0.078 RMSE of headroom above battle noise. |
| **Round trip** | Parse 20 pilot teams from SQLite pastes, encode via `team84`, decode, and verify all 84 fields match the raw paste; verify feature extraction reads every parsed slot. | 0 mismatches across 20 teams. |
| **Coverage** | Verify all 500 pilot teams exist in `matchup_regmb.sqlite`, have 2 battles in all 49 `policy_id = 10` cells and ≥ 2 battles in all 39 `policy_id = 8` training cells, and encode cleanly in `team84`. | 500/500 teams complete and encodable, 0 dropped. |
| **Training smoke** | Fit `predictor_team_bc` and `predictor_team_only` on synthetic linear targets over the pilot feature matrix and verify closed-form LOOCV matches brute-force leave-one-out refits to `1e-8`, and training loss drops near zero on low regularization. | Max LOOCV discrepancy `< 1e-8`; synthetic recovery $R^2 > 0.99$. |
| **Controls** | Verify `guess_train_mean` out-of-fold $R^2 \approx 0$ and `guess_bc_label` reproduces the #447 inter-labeler Pearson *r* (`0.783`) and means (`0.262` FP vs `0.309` BC). | Means and *r* match #447 to 3 decimals. |
| **Invariance** | Randomly permute the 6 Pokémon slots of 50 pilot teams and compare extracted feature vectors. | Max absolute feature difference = `0.0`. |
| **Repeat** | Run the full 5-fold CV twice with `seed = 419` and compare predictions bit-for-bit. | Max prediction difference = `0.0`. |
| **Separation** | Check 0 held-out teams (ranks 51–100) in the 500 pilot teams, 0 team overlap across the 5 test folds, 0 validation-opponent cells read into training inputs or labels, and verify training-fold vocabulary excludes test-only tokens. | 0 held-out teams, 0 fold overlap, 0 validation-opponent leakage. |
| **Read teams / condition / copies** | Skipped with reason: a matchup predictor outputs scalar win-rate predictions, not generated team pastes, and takes no generative conditioning ask. | n/a (written reason saved in `preflight`). |

## Noise floor and budget

- **Irreducible battle-noise floor:** From #447's split halves on the 500 pilot teams, the 2-battle Foul Play deterministic argmax label (78 battles per team across 39 training opponents) has battle-noise variance `0.001792`, i.e., an irreducible **RMSE floor of 0.0423** (`0.1524` total label SD, `0.1464` true team SD, Spearman–Brown reliability `0.923`).
- **Paired comparison precision:** With $N = 500$ out-of-fold teams, a paired RMSE difference of `0.005` between two correlated predictions (e.g., `predictor_team_bc` vs `guess_bc_label` or vs `calibrate_bc_only`) has a paired bootstrap SE of roughly `0.0015–0.003`, giving > 80% power to detect effects ≥ `0.005` RMSE.
- **Compute cost:** Zero new battles. All ridge CV fits, exact LOOCV tuning, bootstrap intervals, and exploratory MLP fits run on the laptop CPU in under 1 minute. **Modal:** none.

## Figures (storyboarded before CV results)

| Frame | Sentence the reader takes | Expected | Surprise |
| --- | --- | --- | --- |
| 1. Out-of-fold RMSE across arms (with 95% CI, linear BC calibration control, and battle-noise floor) | Which predictor beats reusing the behaviour-cloning win rate, and how much of that gain comes from team features vs simple linear rescaling? | `predictor_team_bc` lowest RMSE (around 0.08–0.09), beating `guess_bc_label` (~0.121) and beating `calibrate_bc_only` (~0.095); `predictor_team_only` near or slightly worse than `guess_bc_label`. | `predictor_team_bc` fails to beat `calibrate_bc_only` (team features add nothing on 400 teams), or `predictor_team_only` beats `guess_bc_label` without seeing BC at all. |
| 2. Predicted vs measured Foul Play win rate (side-by-side panels: `guess_bc_label`, `predictor_team_only`, `predictor_team_bc`) | How closely each arm tracks the diagonal across the full win-rate range, and where the old label systematically over-rates teams. | `guess_bc_label` sits systematically right of the diagonal (over-predicting by ~0.047); `predictor_team_bc` centers on the diagonal with tighter scatter; `predictor_team_only` shrinks more toward the mean. | Strong nonlinear curvature or heteroscedasticity (e.g., predictor works at low win rates but fails on strong teams > 0.45). |
| 3. Top team features driving the Foul Play minus behaviour-cloning gap (or sensitivity checks: 1-battle deployment noise, unseen 10 validation opponents, lineup-grouped folds) | Whether the gain survives real deployment conditions (1-battle BC input, unseen validation opponents, and unseen 6-species lineups). | Gain of `predictor_team_bc` over `guess_bc_label` holds under 1-battle BC input, on the 10 unseen validation opponents, and under lineup-grouped folds. | Gain disappears under 1-battle BC input (the pilot's 2-battle BC top-up flattered `predictor_team_bc`) or collapses under lineup-grouped folds (memorizing repeated lineups). |

A teaser combining the arm RMSE comparison (with battle-noise floor and gap to `guess_bc_label`) and the deployment-readiness checks will be drawn once the results are in.

## Commands

```bash
PY=/tmp/vgc-pilot/.venv/bin/python
$PY code/scripts/test_foul_play_matchup_predictor.py
$PY code/src/foul_play_matchup_predictor.py run
$PY code/src/foul_play_matchup_predictor.py figures
```

## Result (2026-11-27)

**Verdict:** Both pre-registered predictor arms beat reusing the behaviour-cloning win rate as the guess (`guess_bc_label`) in 5/5 folds. **Team + behaviour-cloning win rate** (`predictor_team_bc`) achieves 5-fold out-of-fold RMSE **0.0818** [0.0765, 0.0872] ($R^2 = 0.711$, Pearson *r* = 0.844, Spearman ρ = 0.813) against `guess_bc_label`'s **0.1206** [0.1121, 0.1291] ($R^2 = 0.372$, *r* = 0.783)—a paired RMSE reduction of **−0.0388** [−0.0464, −0.0311] (5/5 folds). **Team only, no behaviour-cloning win rate** (`predictor_team_only`) achieves RMSE **0.0913** [0.0851, 0.0976] ($R^2 = 0.641$, *r* = 0.802), beating `guess_bc_label` by **−0.0293** [−0.0396, −0.0192] (5/5 folds) and even beating linear BC calibration alone (`0.0947`).

Crucially, `predictor_team_bc` beats **Linear-calibrated behaviour-cloning win rate** (`calibrate_bc_only`, RMSE **0.0947** [0.0890, 0.1005]) by **−0.0129** [−0.0165, −0.0093] (5/5 folds), showing that the team features learn genuine team-specific discrepancies between Foul Play and behaviour cloning rather than merely correcting the −0.047 global mean shift. Subtracting the irreducible 2-battle noise variance (`0.0423² = 0.001792`), `predictor_team_bc` explains **77.1%** of the true infinite-battle Foul Play variance (noise-corrected RMSE ≈ `0.0700`, noise-corrected *r* ≈ `0.878`).

All 9 preflight probes passed (`all_passed: true`, including `loocv_max_abs_diff = 9.2e-17`, slot-order invariance `0.0`, repeat diff `0.0`, `0` held-out teams). Saved results and folds: `~/vgc-data/foul_play_matchup_predictor_419/results.json`.

### Primary 5-fold cross-validation (500 pilot teams, 39 training opponents)

| Reader name | Code ID | RMSE [95% CI] | Gap vs `guess_bc_label` [95% CI] | MAE | R² | Pearson *r* | Spearman ρ |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Team + behaviour-cloning win rate | `predictor_team_bc` | **0.0818** [0.0765, 0.0872] | **−0.0388** [−0.0464, −0.0311] | 0.0639 | 0.711 | 0.844 | 0.813 |
| Team only, no behaviour-cloning win rate | `predictor_team_only` | **0.0913** [0.0851, 0.0976] | **−0.0293** [−0.0396, −0.0192] | 0.0715 | 0.641 | 0.802 | 0.770 |
| Behaviour-cloning win rate as the guess | `guess_bc_label` | **0.1206** [0.1121, 0.1291] | — | 0.0902 | 0.372 | 0.783 | 0.761 |
| Linear-calibrated behaviour-cloning win rate | `calibrate_bc_only` | **0.0947** [0.0890, 0.1005] | −0.0259 [−0.0325, −0.0192] | 0.0778 | 0.613 | 0.783 | 0.760 |
| Training-fold mean as the guess | `guess_train_mean` | **0.1525** [0.1453, 0.1595] | +0.0319 | 0.1265 | −0.003 | −0.073 | −0.064 |

### Secondary sensitivity & exploratory checks

| Check | `predictor_team_bc` RMSE | `predictor_team_only` RMSE | `calibrate_bc_only` RMSE | `guess_bc_label` RMSE | `team+bc` vs `guess_bc` [95% CI] | `team+bc` vs `calibrate_bc` [95% CI] |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Primary (2-battle BC, 39 train opps) | **0.0818** | 0.0913 | 0.0947 | 0.1206 | −0.0388 [−0.0464, −0.0311] | −0.0129 [−0.0165, −0.0093] |
| 1-battle BC input (deployment noise) | **0.0841** | 0.0913 | 0.0979 | 0.1248 | −0.0407 [−0.0485, −0.0328] | −0.0138 [−0.0175, −0.0101] |
| Lineup-grouped 5-fold CV (unseen lineups) | **0.0819** | 0.0912 | 0.0949 | 0.1206 | −0.0387 [−0.0465, −0.0307] | −0.0130 [−0.0168, −0.0091] |
| 10 unseen validation opponents (20 battles/team) | **0.1335** | 0.1377 | 0.1490 | 0.1564 | −0.0230 [−0.0312, −0.0147] | −0.0156 [−0.0192, −0.0117] |
| Exploratory PyTorch residual MLP (same 5 folds) | **0.0845** | 0.0928 | — | 0.1206 | −0.0361 (ridge wins by 0.0027) | — |

### What Foul Play rates differently from behaviour cloning (observational)

Fitting `predictor_team_bc` on all 500 teams (`lam_s = 100.0`, `intercept = 0.0206`, `bc_slope = 0.5658`), the largest positive residual weights (Foul Play rates higher than behaviour cloning) are `sp_mean_total` (`+0.0125`, mean total Stat Points spent), `item:charizarditey` (`+0.0124`), `move:flareblitz` (`+0.0103`), `move:stompingtantrum` (`+0.0102`), `item:floettite` (`+0.0096`), `move:rockslide` (`+0.0085`), `move:electroshot` (`+0.0074`), and `sp_share_spe28` (`+0.0073`). The largest negative residual weights (Foul Play rates lower) are `item:choicescarf` (`−0.0147`), `move:substitute` (`−0.0106`), `nature:bold` (`−0.0068`), `species:kommoo` (`−0.0068`), `item:fairyfeather` (`−0.0068`), and `move:lightscreen` (`−0.0061`). The Stat Points residual weight is observational only; causal Stat-Point sensitivity belongs to #448.

## Deviations

None. All pre-registered arms, folds, regularization grid, probes, and sensitivity checks ran exactly as written.
