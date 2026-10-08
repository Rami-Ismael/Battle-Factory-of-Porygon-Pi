# DIMES citation audit: project-fit rubric

Prepared 2026-09-08 from the owner's current notes. This is a screening aid, not a new reading queue. Paper-specific empirical claims must be checked against primary papers separately.

## Authoritative local sources

- [Active reading list](../../../Papers/Reading%20List.md): current topic-organized queue, with separate deferred and rejected sections. A title occurring anywhere in this file does **not** imply it is active. Do not duplicate entries or revive rejected papers merely because they cite DIMES.
- [Decision history](../../../Papers/Reading%20List%20%E2%80%94%20notes%20and%20decision%20history.md): use current dated decisions above archived rankings. Read HIGH, then reimplement, then MED. Reward integration remains deferred even for HIGH entries.
- [Current proposer experiment plan](../../../Diffusion%20as%20candidate%20proposer%20in%20black-box%20optimization%20over%20structured%20inputs.md): primary statement of current design choices and measurements.
- [Prior DIFUSCO reference assessment](../../../Papers/DIFUSCO%20references%20-%20relevance%20to%20counter-team%20search.md): existing treatment of DIMES, EAS, NLNS, and black-box search; useful background, subject to the owner's subsequent skepticism in this conversation.

## Actual project and useful transfer

The current design unit is a complete legal six-Pokémon team with categorical species/form, moves, items, abilities, nature, and stat spreads. A generator proposes teams, battle evaluation measures performance, and elite selection/fine-tuning updates the proposal distribution. The goal includes legality, strength, diversity, and novelty. Do not recast it as selecting a subset from a fixed candidate set, a tour, or an MIS without explicit justification.

The current plan varies training-data source and size, distribution of measured training win rates, model scale, classifier-free guidance, field-value sampling temperature, regeneration of 1/3/6 whole Pokémon sets, CEM elite fraction, and original-data replay. Its metrics are legality before/after repair, unique legal teams per 1,000 draws, composition diversity/concentration, copying/novelty, and fresh battle win rate with uncertainty. Generation frequencies matter: evaluating only cherry-picked winners does not measure improvement in the proposer distribution.

A paper earns consideration when it changes a specific decision in that loop. Strong candidates teach a transferable distribution update, candidate selection/ranking procedure, surrogate reliability test, reuse of evaluated teams, diversity-preserving mechanism, constrained categorical generation mechanism, or rigorous evaluation comparison. Tree search and policy methods need an explicit connection to current work rather than a generic claim that both involve search.

## Screening decisions

| Decision | Required case |
|---|---|
| Add now | A new, nonredundant method or experiment addresses a live axis; name the component changed, matched baseline, read sections, adaptation, and limitation. |
| Existing | Already in the active queue; report grade/status without adding a duplicate. |
| Defer / selective background | Useful only if a stated direction or missing prerequisite becomes active; no immediate queue inflation. |
| Skip | Domain-specific solver engineering, survey, redundancy, retired direction, or no concrete transferable experiment. |
| Unverified | Citation count, citation relationship, identity, or relevant paper contents cannot be verified. Do not turn missing evidence into a confident verdict. |

HIGH should require direct relevance to a current decision, more than shared diffusion/optimization terminology. MED may be justified for a focused section or experiment despite substantial adaptation. Citation count greater than one is an inclusion filter, not evidence of utility or quality.

For every proposed addition answer: (1) What would we change? (2) What existing baseline does it improve or challenge? (3) What must be built first? (4) Which paper result actually supports the proposal? (5) Which assumptions do not carry over? A generic “replace graph nodes with Pokémon” analogy fails.

## Critical distinctions and constraints

- Cheap deterministic tour length/MIS size is different from noisy battle win rates. Neither runtime speedups nor more parallel samples establish fewer battle queries or statistically reliable selection. This is a transfer caveat, not automatic rejection of all deterministic CO methods.
- A differentiable closed-form energy cannot be computed from battle simulation. A compatible surrogate or score-function estimator must be specified; neither is free or already validated.
- Optimal-solution supervision or expert repaired children require a data-production route. Strong labeled solutions available in routing do not automatically exist for teams.
- Reward on partial states is not directly available: an incomplete team cannot be battled. Completion rollouts or a surrogate introduce extra assumptions.
- Learned crossover/repair with frozen weights is different from CEM retraining. A search population is different from the model's training distribution.
- Full-team objective optimization differs from battle-policy learning and fixed-subset maximum coverage. Only connect those if a current experiment needs them.
- Matched training updates and battle counts are experimental controls. The current plan requests cost as a reported metric only for model-scale experiments; do not introduce an obligatory cost leaderboard for every comparison.
- Owner does not want surveys (2026-09-02 decision). Molecular-optimization family, online bandit-regret family, QUBO/annealing, and older subset/submodular approximations have explicit rejected or parked decisions. Preserve those unless a new concrete reason and the owner's authorization support a reassessment.

## Existing relevant entries to deduplicate

Active queue already includes DIFUSCO, DDEA, COExpander (explicit background), NLNS, A Diffusion Model Framework for Unsupervised Neural Combinatorial Optimization, Tackling Prevalent Conditions in Unsupervised Combinatorial Optimization, Efficient Active Search, Optimistic Tree Searches for Combinatorial Black-Box Optimization, Unify ML4TSP, Variational Search Distributions, Design by Adaptive Sampling, Generative Bayesian Optimization, Diffusion Models for Black-Box Optimization, guided diffusion methods, constrained discrete diffusion, and MAP-Elites deck-building work. Search by canonical title, DOI, arXiv identifier and aliases before proposing additions.

*Update 2026-09-10:* Efficient Active Search and Variational Search Distributions were rejected by the owner and now sit under Rejected readings in the Reading List. Do not treat them as queued.

DIMES is discussed in the prior DIFUSCO reference note but is not a direct titled entry in the active Reading List at this inspection. Distinguish “already discussed” from “already queued.”

The owner's current conversation rejects COExpander as useful current project work and challenges NLNS. The existing nuanced assessment is: NLNS can motivate a specific frozen-completion-versus-donor-replacement test in partial regeneration; it is peripheral to the whole-team CEM loop. Do not present either as a new recommendation from this sweep. DDEA and DIFUSCO have already received diagrams and explanation, so a citation match does not warrant adding them again.

Trust-Region Noise Search was rejected earlier but explicitly reinstated for selective MED reading on 2026-09-07. The historical claim that expensive rewards automatically invalidate it was corrected. This illustrates why dated current decisions must override old rejected rows.

Only two SPIGM additions are accepted: A Tale of Two Temperatures and Re-evaluating Confidence Remasking, both MED. Archived initial recommendations are not authorization to restore the others.
