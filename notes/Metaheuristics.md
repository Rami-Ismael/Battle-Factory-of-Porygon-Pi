---
created_at: 2026-09-01
updated_at: 2026-09-01
type: concept
title: Metaheuristics
one_line: the parent class of the search templates already in this vault — a problem-independent loop laid over a problem-specific representation and sampler; it supplies the loop, never the moves
discriminating_axis: not trajectory-vs-population and not nature-inspired-vs-not — what separates the families here is whether the template's per-evaluation decision is a single pairwise comparison of f(x') against f(x); at ±0.10 on a win rate that comparison is close to a coin flip, so pairwise templates spend the battle budget on noise
family_pairwise: one accept/reject per evaluation, noise-fragile — simulated annealing, tabu search, iterated local search, variable neighbourhood search
family_set_level: one elite cut per generation, noise averaged over the batch — the cross-entropy method and estimation-of-distribution algorithms generally
family_quality_diversity: returns an archive of distinct high performers instead of one optimum
family_surrogate_assisted: the template optimises the surrogate and spends real battles only on the top-ranked candidates — the only branch that survives an expensive noisy objective
running_here: the elite-refit loop — sample a batch from the generator, battle, keep the elite fraction ρ, refit on the elites, resample — is the cross-entropy method, i.e. an estimation-of-distribution algorithm; ranking 512 draws by the ridge before battling 128 makes it surrogate-assisted
renames: "selection ratio (128 of 512 vs top 20 of 2,176) is selection pressure, and the ρ 0.20→0.01 elite cut is the same knob; the validity 0.477→0.535 concentration alarm is premature convergence, whose remedies are variance rescaling and restarts"
explains: margin scoring null at 0.95 correlation, and 73% of label variance binomial — both are per-comparison signal-to-noise, not modelling failures
naming_collision: "[[Cross Entropy]] currently covers the cross-entropy loss, an information-theory quantity; the cross-entropy method is the estimation-of-distribution algorithm above — same name, unrelated object, and no note holds the method yet"
not_this: hyper-heuristics searches over heuristics rather than over teams — its own checkbox, and the StreamLLM pruning-rule thread is already an instance of it
retired_verdict: "the metaheuristics-not-needed line in [[VGC AI Competition (IEEE CoG)]] rested on the 593,775 enumeration count — retired 2026-09-01, marked in place there; the k=3 and one-rival-team objections in that row still stand"
related:
  - "[[Simulating Annealing]]"
  - "[[Evolutionary Algorithms]]"
  - "[[Evolutionary Strategies]]"
  - "[[Quality Diversity]]"
  - "[[MAP Elite]]"
  - "[[Cross Entropy]]"
  - "[[Diffusion as candidate proposer in black-box optimization over structured inputs]]"
tags:
  - metaheuristics
  - combinatorial-search
  - cross-entropy-method
  - estimation-of-distribution
  - selection-pressure
  - noisy-objective
---
- [ ] Given what list of meta family of meta heurisitc I want to give me the list of each meta heurisitci sinside each blaxk box optimizaton we are given if there are more in the family please share with me
	- [ ] Local search with escape mechanisms
		- [ ] Hill climbing with random restarts
	- [ ] Population and evolutionary
	- [ ] [[Estimation-of-distribution algorithms]]
	- [ ] [[Quality Diversity]]
	- [ ] Model-based (from today's benchmark sweep)
	- [ ] Noise handling, which wraps any of the above
	- [ ] Global partitioning and random-search baselines
- [ ] Look through before active flow mapping artifact

# What current list of them us have used

1. [[Cross Entropy Method]]
2. Diffusion-based proposal generation



# Definition 

- A **metaheuristic is a general strategy for guiding a search toward good solutions**, which you adapt to a particular problem