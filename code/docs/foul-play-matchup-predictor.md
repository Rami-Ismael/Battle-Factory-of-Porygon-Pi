# Can a matchup predictor trained on 500 Foul Play-labelled teams label the rest?

Pre-registration, 2026-11-27, before running cross-validation on the 500 pilot teams. Issue: [Test whether a matchup predictor trained on 500 Foul Play-labelled teams can label the rest](https://github.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/issues/419), on the map [Offline Rust search labeling for the surrogate and inverse-diffusion dataset](https://github.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/issues/416). Mode: **confirmatory** for the three pre-registered arms under the ridge matchup predictor; **exploratory** for the follow-up sensitivity and nonlinear checks.

## Why this run

Battling all ~439,600 eligible training teams against the 49 meta teams with **Foul Play deterministic argmax** (`rust_foulplay_det`, 30,000 iterations, 1 sample) would take roughly 62 laptop days. Issue #447 battled 500 random training teams (`pilot_teams.json`, seed 447, `policy_id = 10`, 2 battles per meta cell) and confirmed that Foul Play deterministic argmax gives the team pairing almost three times as much of a battle's outcome (Matchup Intraclass Correlation (ICC) 0.176 vs behaviour cloning's 0.063) with similar split-half label reliability (*r* = 0.857 vs 0.869), while correlating only 0.783 with behaviour cloning (`policy_id = 8`).

This experiment tests whether a matchup predictor trained on those 500 teams can predict each team's Foul Play deterministic argmax win rate accurately enough to label the remaining ~439,600 teams—specifically, whether it beats simply reusing each team's existing behaviour-cloning win rate as the guess.

## Arms

Shared by every arm: 5-fold cross-validation on the 500 pilot teams (`pilot_teams.json`, seed 447), using the exact same five disjoint 100-team folds (`seed = 419`), scored against each team's Foul Play deterministic argmax win rate over the 39 training opponents (`policy_id = 10`, 2 battles per cell = 78 battles per team).

| Reader name | Code ID | Inputs |
| --- | --- | --- |
| Team and behaviour-cloning win rate | `predictor_team_bc` | The team's 84 fields plus its stored behaviour-cloning win rate (`policy_id = 8`, 39 training opponents) |
| Team only, no behaviour-cloning win rate | `predictor_team_only` | The team's 84 fields only |
| No training, behaviour-cloning win rate as the guess | `guess_bc_label` | No training; the stored behaviour-cloning win rate (`policy_id = 8`, 39 training opponents) is the prediction |

### Secondary diagnostic controls (not replacing the three arms)

- **No team features, linear behaviour-cloning calibration** (`calibrate_bc_only`): ordinary least squares linear fit on each fold's 400 training teams, using only the stored behaviour-cloning win rate. Foul Play averages 0.047 lower than behaviour cloning (0.262 vs 0.309) with a smaller standard deviation (SD: 0.152 vs 0.178), so this control separates fixing the global mean and scale shift from learning team-specific discrepancies between the two labelers.
- **No features, training-fold mean as the guess** (`guess_train_mean`): constant mean of each fold's 400 training teams, giving the zero-information error ceiling.

### Secondary sensitivity checks (exploratory, reported apart)

1. **Deployment-noise checks (single battle per cell for behaviour cloning):** ~98% of the ~439,600 training teams have 1 battle per opponent under `policy_id = 8`, whereas #447 topped the 500 pilot teams up to ≥ 2 battles per opponent. We test both (a) training and testing on 1 drawn battle per cell (`seed = 419`) and (b) training on the 2-battle pilot labels and testing on 1-battle inputs (the exact deployment mismatch).
2. **Unseen-opponent proxy check (10 validation opponents):** Evaluate how well the out-of-fold predictions (trained on the 39 training opponents) predict each pilot team's Foul Play deterministic argmax win rate against the 10 validation opponents (kept strictly out of training labels and inputs).
3. **Lineup-grouped 5-fold check:** Group the 500 teams by their 6-species lineup (436 distinct lineups, max multiplicity 28) so no lineup spans train and test, verifying that team-feature gains do not come only from repeated lineups.
4. **Nonlinear model check (PyTorch residual multilayer perceptron, MLP):** Same five folds, same team features (with and without behaviour-cloning win rate), early-stopped on an inner 20% validation split, to check whether nonlinear interactions improve over ridge regression on 400 training rows.

## Contract

| Row | Value |
| --- | --- |
| Claim | Under 5-fold cross-validation on the 500 pilot teams from #447, a ridge matchup predictor trained on **Team and behaviour-cloning win rate**, or on **Team only, no behaviour-cloning win rate**, achieves lower out-of-fold root mean squared error (RMSE) against the Foul Play deterministic argmax win rate than **No training, behaviour-cloning win rate as the guess**. **Refuted** if the 95% paired bootstrap confidence interval (CI) of the predictor RMSE minus the no-training guess RMSE does not lie strictly below zero for either predictor arm. **If it holds**, the winning predictor design (and whether team features improve over linear behaviour-cloning calibration alone) informs how to label the ~439,600 training teams under map #416. **Scope:** stops at predicting the Foul Play deterministic argmax win rate against the 39 training opponents (plus the 10 validation opponents proxy check); does not measure downstream diffusion-model conditioning gain. |
| Fields | Input: all 84 team fields decoded from each team's canonical database paste (6 Pokémon × species, ability, item, 4 moves, Stat Alignment, 6 Stat Points) plus, for the combined arm only, the scalar behaviour-cloning win rate over the 39 training opponents. None generated; none copied. |
| Vocabulary | Built strictly inside each training fold (400 teams) from the training teams' normalized species, abilities, items, moves, Stat Alignments, and species pairs appearing ≥ 5 times in that training fold, plus 10 permutation-invariant Stat Point summaries (6 per-stat team means, mean total Stat Points spent, mean max-stat focus, share of Pokémon with Speed Stat Points ≥ 28, share with HP Stat Points ≥ 28) standardized using training-fold mean and SD. Zero test-fold vocabulary or scaling leakage. |
| Training teams | The 500 pilot teams in `~/vgc-data/foul_play_label_pilot_447/pilot_teams.json` (seed 447, drawn from 439,554 eligible generated training teams with full `policy_id = 8` coverage; excludes meta teams, held-out ranks 51–100, and VGCPastes pool teams). Each fold trains on 400 teams and tests on 100. |
| Scorer | Stored battles in `~/vgc-data/matchup_regmb.sqlite`: target is `policy_id = 10` (`rust_foulplay_det`, 30,000 iterations, 1 sample, both sides, slots shuffled), 2 battles per opponent over the 39 training opponents (`training_opponents` in `pilot_teams.json`). Auxiliary input is `policy_id = 8` (`bc_100.zip`, both sides, slots shuffled) over the same 39 training opponents. No new battles are run. |
| Search region | No search loop. Inner regularization grid for ridge regression: single-field penalty in {1, 3, 10, 30, 100, 300, 1000} and species-pair penalty equal to 3× the single-field penalty, selected per fold by exact closed-form leave-one-out cross-validation (LOOCV) inside the 400 training teams. The intercept and (for the combined arm) the scalar behaviour-cloning win-rate slope are unpenalized, implementing Kennedy–O'Hagan multi-fidelity residual shrinkage. Predictions are clipped to `[0, 1]`. |
| Settings | **Scientific:** predictor input set (combined team and behaviour-cloning win rate vs team only vs no-training guess). **Nuisance:** ridge regularization penalty tuned independently per arm and fold via inner LOOCV so neither arm is handicapped by a mis-scaled penalty. **Fixed:** 5-fold split (`seed = 419`), minimum species-pair count 5, Stat Point summary definitions, 10,000 bootstrap resamples (`seed = 419`). |
| Plan | **Primary measure:** out-of-fold RMSE against the Foul Play deterministic argmax win rate over the 500 teams (monotone with mean squared error (MSE), directly comparable to the 0.0423 battle-noise RMSE floor and 0.1524 label SD). **Single-axis better question:** no accuracy–diversity trade-off exists for a fixed-team labeler; a predictor arm is good enough if its paired 95% bootstrap CI for predictor RMSE minus no-training guess RMSE lies strictly below 0 (smallest effect worth acting on: 0.005 RMSE). **Also reported:** MSE, mean absolute error (MAE), R², Pearson *r*, Spearman ρ, per-fold RMSE and win/loss signs across the 5 folds, and the paired gap between Team and behaviour-cloning win rate and linear behaviour-cloning calibration alone. **Stopping rule:** one run of the pre-registered 5-fold cross-validation; no post-hoc fold or feature cherry-picking. |

## Probes

Each probe runs inside `src/foul_play_matchup_predictor.py preflight` and saves its pass/fail verdict and numbers under `preflight` in the results JSON before the cross-validation results are judged.

| Probe | Check | Pass criterion |
| --- | --- | --- |
| **Ceiling / battle-noise floor** | Irreducible RMSE floor from the two halves of `policy_id = 10` (`report_pilot.json`: game 0 vs game 1 per cell), `sqrt(mean((h0 - h1)^2) / 4)`. | Floor (`0.0423`) is far below the no-training guess RMSE (`0.1206`), leaving ~0.078 RMSE of headroom above battle noise. |
| **Round trip** | Parse 20 pilot teams from SQLite pastes, encode via `team84`, decode, and verify all 84 fields match the raw paste; verify feature extraction reads every parsed slot. | 0 mismatches across 20 teams. |
| **Coverage** | Verify all 500 pilot teams exist in `matchup_regmb.sqlite`, have 2 battles in all 49 `policy_id = 10` cells and ≥ 2 battles in all 39 `policy_id = 8` training cells, and encode cleanly in `team84`. | 500/500 teams complete and encodable, 0 dropped. |
| **Training smoke** | Fit both predictor arms on synthetic linear targets over the pilot feature matrix and verify closed-form LOOCV matches brute-force leave-one-out refits to `1e-8`, and training loss drops near zero on low regularization. | Max LOOCV discrepancy `< 1e-8`; synthetic recovery R² > 0.99. |
| **Controls** | Verify the training-fold mean out-of-fold R² ≈ 0 and the no-training guess reproduces the #447 inter-labeler Pearson *r* (`0.783`) and means (`0.262` Foul Play vs `0.309` behaviour cloning). | Means and *r* match #447 to 3 decimals. |
| **Invariance** | Randomly permute the 6 Pokémon slots of 50 pilot teams and compare extracted feature vectors. | Max absolute feature difference = `0.0`. |
| **Repeat** | Run the full 5-fold cross-validation twice with `seed = 419` and compare predictions bit-for-bit. | Max prediction difference = `0.0`. |
| **Separation** | Check 0 held-out teams (ranks 51–100) in the 500 pilot teams, 0 team overlap across the 5 test folds, 0 validation-opponent cells read into training inputs or labels, and verify training-fold vocabulary excludes test-only tokens. | 0 held-out teams, 0 fold overlap, 0 validation-opponent leakage. |
| **Read teams / condition / copies** | Skipped with reason: a matchup predictor outputs scalar win-rate predictions, not generated team pastes, and takes no generative conditioning ask. | n/a (written reason saved in `preflight`). |

## Noise floor and budget

- **Irreducible battle-noise floor:** From #447's split halves on the 500 pilot teams, the 2-battle Foul Play deterministic argmax label (78 battles per team across 39 training opponents) has battle-noise variance `0.001792`, i.e., an irreducible **RMSE floor of 0.0423** (`0.1524` total label SD, `0.1464` true team SD, Spearman–Brown reliability `0.923`).
- **Paired comparison precision:** With 500 out-of-fold teams, a paired RMSE difference of `0.005` between two correlated predictions has a paired bootstrap standard error (SE) of roughly `0.0015–0.003`, giving > 80% power to detect effects ≥ `0.005` RMSE.
- **Compute cost:** Zero new battles. All ridge cross-validation fits, exact LOOCV tuning, bootstrap intervals, and exploratory MLP fits run on the laptop CPU in under 1 minute. **Modal:** none.

## Figures (storyboarded before cross-validation results)

| Frame | Sentence the reader takes | Expected | Surprise |
| --- | --- | --- | --- |
| 1. Out-of-fold RMSE across arms (with 95% bootstrap CI, linear calibration control, and battle-noise floor) | Which predictor beats reusing the behaviour-cloning win rate, and how much of that gain comes from team features vs simple linear rescaling? | Team and behaviour-cloning win rate lowest RMSE (~0.08–0.09), beating both the no-training guess (~0.121) and linear calibration (~0.095); Team only near or slightly worse than the no-training guess. | Adding team features fails to beat linear calibration alone, or Team only beats the no-training guess without seeing behaviour cloning at all. |
| 2. Predicted vs measured Foul Play win rate (side-by-side panels for the three arms) | How closely each arm tracks the diagonal across the full win-rate range, and where the old label systematically over-rates teams. | The no-training guess sits systematically right of the diagonal (over-predicting by ~0.047); Team and behaviour-cloning win rate centers on the diagonal with tighter scatter; Team only shrinks more toward the mean. | Strong nonlinear curvature or heteroscedasticity (e.g., predictor works at low win rates but fails on strong teams > 0.45). |
| 3. Deployment-readiness checks (deployment noise mismatch, unseen 6-species lineups, 10 unseen validation opponents) | Whether the gain survives real deployment conditions (single-battle corpus inputs, unseen lineups, and unseen opponents). | Gain of Team and behaviour-cloning win rate holds under single-battle inputs, on unseen validation opponents, and under lineup-grouped folds. | Gain disappears under single-battle inputs (the pilot's 2-battle top-up flattered the combined arm) or collapses under lineup-grouped folds. |

A teaser combining the arm RMSE comparison (with battle-noise floor and paired error gaps) was drawn once the results were in.

## Commands

```bash
PY=/tmp/vgc-pilot/.venv/bin/python
$PY code/scripts/test_foul_play_matchup_predictor.py
$PY code/src/foul_play_matchup_predictor.py run
$PY code/src/foul_play_matchup_predictor.py figures
$PY code/src/foul_play_matchup_predictor.py label-corpus
```

## Result (2026-11-27)

**Verdict:** Both pre-registered predictor arms beat **No training, behaviour-cloning win rate as the guess** (`guess_bc_label`) in 5/5 folds. **Team and behaviour-cloning win rate** (`predictor_team_bc`) achieves 5-fold out-of-fold RMSE **0.0818** [0.0765, 0.0872] (R² = 0.711, Pearson *r* = 0.844, Spearman ρ = 0.813) against the no-training guess's **0.1206** [0.1121, 0.1291] (R² = 0.372, *r* = 0.783)—a paired RMSE reduction of **−0.0388** [−0.0464, −0.0311] (5/5 folds). **Team only, no behaviour-cloning win rate** (`predictor_team_only`) achieves RMSE **0.0913** [0.0851, 0.0976] (R² = 0.641, *r* = 0.802), beating the no-training guess by **−0.0293** [−0.0396, −0.0192] (5/5 folds) and even beating linear calibration alone (`0.0947`).

Crucially, Team and behaviour-cloning win rate beats **No team features, linear behaviour-cloning calibration** (`calibrate_bc_only`, RMSE **0.0947** [0.0890, 0.1005]) by **−0.0129** [−0.0165, −0.0093] (5/5 folds), showing that the team features learn genuine team-specific discrepancies between Foul Play and behaviour cloning rather than merely correcting the −0.047 global mean shift. Subtracting the irreducible two-battle noise variance (`0.0423² = 0.001792`), the combined predictor explains **77.1%** of the true infinite-battle Foul Play variance (noise-corrected RMSE ≈ `0.0700`, noise-corrected *r* ≈ `0.878`).

All 9 preflight probes passed (`all_passed: true`, including LOOCV discrepancy `9.2e-17`, slot-order invariance `0.0`, repeat discrepancy `0.0`, `0` held-out teams). Saved results and folds: `~/vgc-data/foul_play_matchup_predictor_419/results.json`.

### Primary 5-fold cross-validation (500 pilot teams, 39 training opponents)

| Reader name | Code ID | RMSE [95% CI] | Gap vs no-training guess [95% CI] | MAE | R² | Pearson *r* | Spearman ρ |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Team and behaviour-cloning win rate | `predictor_team_bc` | **0.0818** [0.0765, 0.0872] | **−0.0388** [−0.0464, −0.0311] | 0.0639 | 0.711 | 0.844 | 0.813 |
| Team only, no behaviour-cloning win rate | `predictor_team_only` | **0.0913** [0.0851, 0.0976] | **−0.0293** [−0.0396, −0.0192] | 0.0715 | 0.641 | 0.802 | 0.770 |
| No training, behaviour-cloning win rate as the guess | `guess_bc_label` | **0.1206** [0.1121, 0.1291] | — | 0.0902 | 0.372 | 0.783 | 0.761 |
| No team features, linear behaviour-cloning calibration | `calibrate_bc_only` | **0.0947** [0.0890, 0.1005] | −0.0259 [−0.0325, −0.0192] | 0.0778 | 0.613 | 0.783 | 0.760 |
| No features, training-fold mean as the guess | `guess_train_mean` | **0.1525** [0.1453, 0.1595] | +0.0319 | 0.1265 | −0.003 | −0.073 | −0.064 |

### Secondary sensitivity & exploratory checks

| Check | Team and behaviour-cloning RMSE | Team only RMSE | Linear calibration RMSE | No-training guess RMSE | Combined vs no-training guess [95% CI] | Combined vs linear calibration [95% CI] |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Primary (2-battle input, 39 training opponents) | **0.0818** | 0.0913 | 0.0947 | 0.1206 | −0.0388 [−0.0464, −0.0311] | −0.0129 [−0.0165, −0.0093] |
| Deployment mismatch (2-battle train, 1-battle test input) | **0.0846** | 0.0913 | 0.0979 | 0.1248 | −0.0402 [−0.0479, −0.0326] | −0.0134 [−0.0169, −0.0098] |
| Single-battle input (1-battle train and test) | **0.0841** | 0.0913 | 0.0979 | 0.1248 | −0.0407 [−0.0485, −0.0328] | −0.0138 [−0.0175, −0.0101] |
| Lineup-grouped 5-fold cross-validation (unseen 6-species lineups) | **0.0819** | 0.0912 | 0.0949 | 0.1206 | −0.0387 [−0.0465, −0.0307] | −0.0130 [−0.0168, −0.0091] |
| 10 unseen validation opponents (20 battles per team) | **0.1335** | 0.1377 | 0.1490 | 0.1564 | −0.0230 [−0.0312, −0.0147] | −0.0156 [−0.0192, −0.0117] |
| Exploratory PyTorch residual MLP (same 5 folds) | **0.0845** | 0.0928 | — | 0.1206 | −0.0361 (ridge wins by 0.0027) | — |

### What Foul Play rates differently from behaviour cloning (observational)

Fitting Team and behaviour-cloning win rate on all 500 teams (selected single-field penalty 100.0, intercept 0.0206, slope on behaviour-cloning win rate 0.5658), the largest positive residual weights (features Foul Play rates higher than behaviour cloning) are mean total Stat Points spent (`+0.0125`), Charizardite Y (`+0.0124`), Flare Blitz (`+0.0103`), Stomping Tantrum (`+0.0102`), Floettite (`+0.0096`), Rock Slide (`+0.0085`), Electro Shot (`+0.0074`), and share of Pokémon with Speed Stat Points ≥ 28 (`+0.0073`). The largest negative residual weights (features Foul Play rates lower) are Choice Scarf (`−0.0147`), Substitute (`−0.0106`), Bold Stat Alignment (`−0.0068`), Kommo-o (`−0.0068`), Fairy Feather (`−0.0068`), and Light Screen (`−0.0061`). The Stat Points residual weight is observational only; causal Stat-Point sensitivity belongs to #448.

## Deviations

- Added one post-review deployment check (**Deployment mismatch: 2-battle train, 1-battle test input**) alongside the pre-registered single-battle check, to match the exact mismatch between the topped-up 500 pilot teams and the single-battle ~439,600 corpus. All pre-registered primary arms, folds, regularization grid, probes, and other checks ran unchanged.

## Full-corpus labeling with predictive standard deviation (2026-11-28)

Fitted **Team and behaviour-cloning win rate** (`predictor_team_bc`) on all 500 pilot teams (`policy_id = 10` over the 39 training opponents; `lam_single = 100.0`, `lam_pair = 300.0`, intercept `0.02056`, `bc_slope = 0.56578`, 961 features, effective degrees of freedom `78.30`, residual SD `0.07589`, true infinite-battle residual SD `0.06299`) and wrote predictions plus predictive standard deviations for all **459,681** eligible legal teams with full `policy_id = 8` coverage against the 39 training opponents (0 held-out teams, 500 pilot teams marked `in_pilot_train = 1`) into dedicated predictor tables in `~/vgc-data/matchup_regmb.sqlite`:

- **`team_score_predictor`**: registry row `predictor_id = 1`, `predictor_name = 'foul_play_det_ridge_team_bc_v1'`, `kind = 'predicted_score'` (explicitly distinct from `policy` / `cell` battle outcomes), storing provenance, hyperparameters, variance components, and 5-fold calibration checks in `metadata_json`.
- **`team_predicted_score`** (and joined view **`team_predicted_score_summary`**): per-team `pred_win_rate`, `pred_std`, `epistemic_std`, `bc_win_rate`, `bc_opponents`, `bc_battles`, and `in_pilot_train`.
  - `epistemic_std`: Bayesian ridge parameter SD $\hat{\sigma}_{\text{resid}} \sqrt{x^\top (X^\top X + P)^{-1} x}$ (corpus mean `0.0321`, range `0.0144`–`0.0686`).
  - `pred_std`: total predictive SD for the team's true (infinite-battle) Foul Play win rate $\sqrt{\hat{\sigma}_{\text{true,resid}}^2 + \text{epistemic\_std}^2 + \Delta v_{\text{bc}}}$, where $\Delta v_{\text{bc}}$ adds per-team binomial input noise when a team has fewer than 2 behaviour-cloning battles per opponent (corpus mean `0.0760`, range `0.0648`–`0.0934`).
  - 5-fold out-of-fold z-score calibration against the 2-battle target ($\sqrt{\text{pred\_std}^2 + \sigma_{\text{fp,2b}}^2}$): 2-battle input `z_std = 0.990`, `|z| <= 1: 70.2%`, `|z| <= 2: 95.6%`; 1-battle deployment input `z_std = 0.962`, `|z| <= 1: 71.4%`, `|z| <= 2: 95.4%`.
- Corpus distribution (`~/vgc-data/foul_play_matchup_predictor_419/corpus_labeling_summary.json`): `pred_win_rate` mean **0.2666** (SD `0.1298`, median `0.2837`, range `0.0000`–`0.6333`) vs `bc_win_rate` mean **0.3114** (SD `0.1872`, correlation `r = 0.9345`).
