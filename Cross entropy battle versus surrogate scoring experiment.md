# Cross-entropy search: battle scores versus surrogate predictions

## Question

Can a surrogate replace battle evaluations during cross-entropy search while preserving the quality of the selected teams and reducing the number of real battles needed?

This changes the scorer used to select elites. It does not replace the independent final battle evaluation.

## Experiment arms

| Arm | Candidate score during search | Final evaluation |
| --- | --- | --- |
| CEM–Battle | Measured win rate from 24 battles against the fixed meta distribution | 192 fresh battles per reported team |
| CEM–Surrogate | Predicted win rate from a frozen surrogate trained before search | The same 192-battle evaluation protocol |
| CEM–Hybrid, optional follow-up | Surrogate screens candidates; battles score a predefined shortlist before elite selection | The same 192-battle evaluation protocol |

Start with the first two arms. The hybrid is a separate follow-up experiment rather than a hidden modification of the surrogate arm.

## Keep fixed

- The initial sampling distribution or generator checkpoint.
- Starting teams, tasks, masks, and fixed team fields when running completion tasks.
- Candidate-attempt count, legal-candidate handling, elite fraction, iterations, and update/training settings.
- The meta distribution, opponent weights, battle policy, simulator, and format.
- Paired random seeds and the final opponent/seed schedule.

Use a shared initial candidate batch for each paired run. After the first elite update, the candidate populations may diverge: that is the intended consequence of changing the scorer. Do not force later populations to remain identical.

No Tera Types are used for the chosen regulation.

## Surrogate preparation

Train the surrogate only on independently collected team-score examples. Use a team-level held-out validation split; repeated records of an identical team must not cross that split. Do not train or tune on the final 192-battle results of teams selected by this experiment.

Predict the same objective used by CEM–Battle: mean win rate against the fixed meta distribution under the same battle policy. A surrogate trained on a different opponent pool or policy is a separate transfer experiment.

Freeze the trained surrogate before the main comparison. Record its data snapshot, features, training settings, and checkpoint. Include all generated features that can vary in the experiment. For stat-spread completion, this requires stat-spread inputs; a model that ignores them cannot rank those candidates.

Use a separate held-out candidate set to measure prediction error, ranking agreement, and whether surrogate-selected elites have high measured battle performance. Ranking quality matters directly to CEM's elite selection. Validation estimates based on 24-battle labels remain noisy; confirm the most important ranking conclusions with fresh battles.

## Select and evaluate the reported teams

1. Run both arms for the predefined number of iterations.
2. Select each arm's best team and top 16 using that arm's search scores. Record the selection before fresh evaluation. If fewer than 16 distinct legal teams are available, report the actual count.
3. Evaluate every reported team using 192 fresh battles across the fixed meta pool, with the same schedule across arms. Here 192 means total battles per team, not 192 per matchup.
4. Report the independently evaluated best-selected-team win rate and top-16 mean. Do not reselect the best team using the fresh results.

## Performance and cost metrics

- Fresh best-selected-team win rate, top-16 mean, and uncertainty across paired search runs.
- Improvement over the same real-team reference, including the proposed 0.47 mean when that reference is measured under these exact evaluation conditions.
- Battles spent before finding a reported candidate that reaches the reference on independent confirmation. Include the initial surrogate-label collection cost in the surrogate arm's end-to-end battle count.
- Search-only real battles, surrogate-training-label battles, and final confirmation battles, reported separately.
- Candidate count, surrogate-inference time, model-training time, and total elapsed time.
- Legal-generation rate, distinct-team rate, and diversity of generated completions within the same mask/task/k.
- Search-score minus fresh win-rate difference for the selected teams, to reveal overestimation.

Report both cost views: an end-to-end comparison charging the surrogate for its training labels, and a reuse comparison treating an existing trained surrogate as an already available asset. Do not claim zero total battle cost just because inference replaces battles during search.

First compare equal candidate and iteration budgets to isolate the scorer change. Then run a separate equal-wall-time or equal-total-battle-budget experiment to test practical efficiency. State whether final confirmation battles are excluded from the search budget; always report their cost.

## Interpreting the result

Similar fresh performance with substantially fewer total battles supports surrogate scoring as a useful replacement. Worse performance despite optimistic predictions suggests ranking errors or search exploiting inaccuracies in the surrogate. A hybrid can then test whether occasional battle verification recovers performance.

A surrogate score cannot establish whether a team beats the meta. Only the held-out battle evaluation supplies that performance result.

## Current implementation status

The pilot repository contains battle-scored CEM in `scripts/cem_score_direction.py` and a separate ridge-surrogate experiment in `src/hps_surrogate.py`. The existing surrogate uses categorical field indicators and species-pair features with 24-battle labels. It omits stat-spread features and is not yet connected as the CEM elite-selection scorer.

This document specifies the comparison; it does not report a completed battle-versus-surrogate run. Wiring the scorer, training/freezing a compatible surrogate, and conducting fresh evaluation are still required before reporting a performance difference.
