---
created_at: 2026-09-07
updated_at: 2026-09-07
type: research-report
---

# PyMC for VGC team search

**Best application: estimating which promising teams deserve more battles when their measured win rates are noisy.** PyMC provides Bayesian statistical modeling and inference. Its posterior samples can support uncertainty-aware decisions inside the existing search loop; they do not enumerate legal Pokémon teams or establish a globally optimal counter. [PyMC overview](https://www.pymc.io/welcome.html)

This assessment uses the recorded results in [[List of Experiments]]; underlying battle files were not inspected or reanalyzed. The log reports 24 battles per team in the elite-filter sweep, a useful ridge surrogate, and an unresolved playstyle comparison after clustering errors by team. These make uncertainty estimation a concrete project need.

## Most useful applications

1. **Rank a shortlist while accounting for limited evidence.** A 21/24 record is an estimate, not a known 87.5% win probability. Model wins as binomial observations of an unknown probability. With a fixed Beta prior, the posterior is available analytically: `Beta(alpha + wins, beta + losses)`. This simple baseline does not require PyMC. PyMC becomes more useful when a hierarchical model learns how much related candidates should share information. Its baseball example implements precisely the analogous problem: estimating individual success probabilities from unequal, sometimes tiny, trial counts. [Hierarchical partial pooling](https://www.pymc.io/projects/examples/en/latest/case_studies/hierarchical_partial_pooling.html)

2. **Allocate further simulation to unresolved contenders.** Proposed project use: estimate the posterior probability that each candidate beats the incumbent by a chosen practical margin, then spend additional battles where that decision remains uncertain. This allocation rule would be our implementation, not an automatic PyMC feature. Pool within a defensible candidate population; mixing random legal teams with strong donor-based teams under one homogeneous prior could distort estimates. Hierarchical sharing reduces extreme estimates caused by sparse data, but inappropriate sharing introduces bias. [Partial-pooling model](https://www.pymc.io/projects/examples/en/latest/case_studies/hierarchical_partial_pooling.html)

3. **Add uncertainty to the existing surrogate.** A later experiment could use Bayesian binomial regression with species/pair features and shrinkage priors, predicting win probabilities plus parameter uncertainty. This could help choose between a high predicted score and an uncertain candidate worth testing. PyMC demonstrates regression with a binomial likelihood and logistic link; transferring it to team features is a proposal, not an established improvement over the recorded ridge model. [Binomial regression](https://www.pymc.io/projects/examples/en/latest/generalized_linear_models/GLM-binomial-regression.html)

## First experiment

Freeze a shortlist from the current copy-paste proposer, the battle policies, target opponent distribution, and total battle budget. Compare three selection methods: raw win-rate ranking, independent fixed-prior Beta updates, and hierarchical partial pooling. Give each candidate the same initial opponent schedule. Initially isolate ranking quality; test adaptive allocation separately afterward.

Select finalists using only those initial outcomes, then evaluate finalists on fresh battles excluded from fitting and selection. Repeat across independent shortlist/battle runs. Measure fresh finalist win rate, predictive calibration, and total simulation plus fitting time. PyMC should earn its overhead by improving selection or saving battles relative to the conjugate baseline. Use prior/posterior predictive checks and sampler diagnostics before trusting results. [Predictive checks](https://www.pymc.io/projects/docs/en/stable/learn/core_notebooks/posterior_predictive.html)

## Limits that affect the conclusion

- Win probability is conditional on the piloting policies and opponent distribution. A better pilot or changed metagame changes the target.
- Preserve opponent, candidate, run, and pairing identifiers. Repeated battles sharing teams or experimental blocks do not create independent evidence about population-level improvements. Use an appropriate hierarchical/paired model; a beta-binomial likelihood alone does not capture arbitrary dependence. [Beta-binomial definition](https://www.pymc.io/projects/docs/en/stable/api/distributions/generated/pymc.BetaBinomial.html)
- Searching many candidates favors lucky scores. Fresh validation remains necessary; posterior intervals do not automatically remove optimizer selection effects.
- Model uncertainty can still be overconfident on unfamiliar teams. Validate on held-out team families, not only extra battles of familiar teams.

**Recommendation:** pilot PyMC as an offline shortlist evaluator first. Retain the existing legality checks, proposer, simulator, and simple statistical baseline.
