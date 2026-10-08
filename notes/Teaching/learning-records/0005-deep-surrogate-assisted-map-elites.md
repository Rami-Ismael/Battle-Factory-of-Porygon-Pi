---
created_at: 2026-08-19
type: learning-record
lesson: 0004-deep-surrogate-assisted-map-elites.html
---

# 0005 — Deep Surrogate Assisted MAP-Elites (Zhang et al., GECCO 2022)

Taught 2026-08-19, on the $f$ strand; retrieval quiz not yet confirmed taken.

Key insights the lesson carries:

- **The two-loop architecture is the transferable unit:** train an estimator of the objective online, run the whole search on the estimator for free, spend every real simulated evaluation grounding the candidates the estimator is most optimistic about. Applies to any search family (their MAP-Elites, or the owner's annealing / Bayesian-optimisation comparison).
- **Same budget, not fewer simulations:** at an identical 10,000-evaluation budget DSA-ME reaches ~2.5× the QD-score of plain MAP-Elites (338.52 vs 136.89). The honest claim is quality-per-simulation.
- **Estimator training data must come from the search itself:** the Offline DSA-ME ablation (surrogate trained on random decks) collapses to 194.9 — directly relevant to fitting any estimator of $f$.
- **The surrogate predicts a smoothed proxy (health difference) + the archive measures, not win rate** — a variance-reduction trick with a plausible VGC analogue.
- **The unpublished step is $f$'s opponent structure:** their six-deck suite is weighted evenly; surrogate search against a frequency-weighted meta-team list exists in no published paper (two sweeps + adversarial deep-read, 2026-08-19).

Open question left with the owner (lesson's closing decision): does a surrogate join simulated annealing and Bayesian optimisation in the set of methods compared against $f$?
