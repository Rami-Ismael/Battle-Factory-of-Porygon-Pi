# Black-box optimization families for Pokémon team search

This is a broad checklist of established methods and notable variants, not an exhaustive catalog of every named metaheuristic. The categories overlap: a method can be population-based and distribution-based at once. Black-box optimization is broader than metaheuristics; it also includes deterministic direct search and Bayesian optimization.

**Current blog focus:** the cross-entropy method (CEM), using one scalar battle-performance objective. See [the CEM blog plan](cross-entropy-method-blog-plan.md). The remaining families are background for future work.

The seven requested groups appear first. Additional families and useful cross-cutting strategies follow. “#1” and “today’s benchmark sweep” were not identified in the conversation, so the EDA and model-based lists are general rather than a reconstruction of that sweep.

## 1. Local search with escape mechanisms

These methods modify a current solution and use memory, worsening moves, perturbations, or neighborhood changes to avoid getting stuck.

- [ ] **Simulated annealing (SA)** — accepts some worsening changes, with acceptance controlled by temperature.
  - [ ] Adaptive simulated annealing.
  - [ ] Reheating/reannealing variants.
  - [ ] Parallel tempering / replica exchange — exchanges candidates between searches at different temperatures; applicable to optimization as well as sampling.
- [ ] **Tabu search (TS)** — uses memory to restrict recently used moves or solutions.
  - [ ] Reactive tabu search — adapts restrictions during search.
  - [ ] Robust tabu search — a notable tabu-search variant, especially in assignment problems.
- [ ] **Iterated local search (ILS)** — alternates local improvement with perturbation.
- [ ] **Variable neighborhood search (VNS)** — changes neighborhoods to escape local optima.
  - [ ] Reduced VNS.
  - [ ] General VNS.
  - [ ] Skewed VNS.
  - Variable neighborhood descent (VND) is a related local-improvement component; by itself it need not provide an escape mechanism.
- [ ] **Guided local search (GLS)** — penalizes features associated with local optima.
- [ ] **Greedy randomized adaptive search procedure (GRASP)** — repeatedly constructs randomized greedy solutions and locally improves them.
- [ ] **Iterated greedy (IG)** — repeatedly destroys and reconstructs part of a solution.
- [ ] **Large neighborhood search (LNS)** — searches neighborhoods created by substantial partial changes.
  - [ ] Adaptive LNS (ALNS) — learns which destroy/repair operators to use.
- [ ] **Threshold accepting** — accepts worsening moves within a threshold.
- [ ] **Great deluge** — uses a changing acceptance boundary.
- [ ] **Record-to-record travel** — accepts solutions within a tolerance of the best record.
- [ ] **Late acceptance hill climbing (LAHC)** — compares candidates with older objective values.
- [ ] **Breakout local search** — uses adaptive perturbations to escape stagnation.

**Pokémon interpretation:** change one move or item for a small neighborhood; replace several team members for a larger neighborhood. Destroy/repair procedures must preserve or restore legality.

For the relationship between local improvement, escape, and multistart methods, see the authors’ [Iterated Local Search: Framework and Applications](https://iridia.ulb.ac.be/~stuetzle/publications/LouMarStu10.pdf).

## 2. Hill climbing with random restarts

This is a subfamily of local search, retained separately to match your checklist. A restart strategy can wrap different local optimizers.

- [ ] **Random-restart hill climbing / multistart local search**.
- [ ] **First-improvement hill climbing with restarts** — take the first improving neighbor.
- [ ] **Best-improvement / steepest-ascent hill climbing with restarts** — evaluate a neighborhood and choose its best improvement.
- [ ] **Stochastic hill climbing with restarts** — randomize the choice of improving moves.
- [ ] **Random-mutation hill climbing (RMHC) with restarts** — propose random mutations and retain improvements.
- [ ] **Coordinate-wise local search with restarts** — optimize one decision variable or block at a time.
- [ ] **Adaptive-restart local search** — trigger restarts using stagnation or progress measures.

Plain hill climbing is a local heuristic; adding restarts supplies a broader search mechanism. Uniform restarts, biased restarts, and restart schedules are design choices, not necessarily separate algorithm families.

**Pokémon interpretation:** generate a legal team, improve it until progress stops, then start from a new team. Keep the best across runs.

## 3. Population and evolutionary methods

### Evolutionary algorithms

- [ ] **Genetic algorithms (GA)**.
  - [ ] Generational GA.
  - [ ] Steady-state GA.
  - [ ] Island / distributed GA.
  - [ ] Cellular GA.
- [ ] **Evolution strategies (ES)**.
  - [ ] (1+1)-ES.
  - [ ] (μ,λ)-ES and (μ+λ)-ES.
  - [ ] Covariance matrix adaptation ES (**CMA-ES**).
  - [ ] Separable CMA-ES.
  - [ ] Limited-memory CMA-ES.
  - [ ] IPOP-CMA-ES and BIPOP-CMA-ES — restart variants.
- [ ] **Differential evolution (DE)**.
  - [ ] DE/rand/1 and DE/best/1 — mutation schemes.
  - [ ] jDE, JADE, SHADE, L-SHADE — adaptation variants.
- [ ] **Evolutionary programming (EP)**.
- [ ] **Genetic programming (GP)** — evolves programs or expressions; more directly useful for battle policies or team-construction rules than fixed team vectors.
- [ ] **Memetic algorithms** — combine population evolution with individual local improvement.
- [ ] **Cooperative coevolution** — evolves interacting components separately.
- [ ] **Competitive coevolution** — evolves candidates against evolving opponents.

### Other population methods

- [ ] **Particle swarm optimization (PSO)** — including discrete/binary adaptations.
- [ ] **Ant colony optimization (ACO)**.
  - [ ] Ant System.
  - [ ] Ant Colony System (ACS).
  - [ ] MAX–MIN Ant System (MMAS).
- [ ] **Artificial bee colony (ABC)**.
- [ ] **Scatter search**.
- [ ] **Population-based incremental learning (PBIL)** — also an EDA; listed in section 4.

**Encoding matters:** standard CMA-ES and DE operate on continuous vectors. Species, moves, and items require suitable discrete/mixed adaptations or a decoder. GA, ACO, and discrete local search can operate directly on categorical decisions.

See the official [pymoo algorithm catalog](https://pymoo.org/algorithms/) for implemented evolutionary and swarm methods, and [IRIDIA’s stochastic local search overview](https://iridia.ulb.ac.be/sls2007/index.php) for their relationship to other metaheuristics.

## 4. Estimation-of-distribution algorithms (EDAs)

EDAs learn distributions over promising solutions and sample new candidates from them. “Other EDAs” needs an identified reference algorithm; the following is the broader family.

### Independent-variable models

- [ ] **UMDA** — univariate marginal distribution algorithm.
- [ ] **PBIL** — population-based incremental learning.
- [ ] **Compact genetic algorithm (cGA)**.
- [ ] **Cross-entropy method (CEM/CE)** — fits a distribution to selected samples; often grouped with EDAs.

### Dependency-learning models

- [ ] **MIMIC** — mutual-information-maximizing input clustering; uses a chain-structured probabilistic model.
- [ ] **COMIT** — combining optimizers with mutual information trees.
- [ ] **ECGA** — extended compact genetic algorithm; learns groups of dependent variables.
- [ ] **BOA** — Bayesian optimization algorithm; learns a Bayesian network over promising solutions.
- [ ] **hBOA** — hierarchical Bayesian optimization algorithm.
- [ ] **EBNA** — estimation of Bayesian network algorithm.

### Continuous distribution models

- [ ] **EMNA** — estimation of multivariate normal algorithm.
- [ ] **EGNA** — estimation of Gaussian network algorithm.
- [ ] **AMaLGaM** — adapted maximum-likelihood Gaussian model.
- [ ] **GOMEA** — gene-pool optimal mixing evolutionary algorithm; a related model-building evolutionary family.
  - [ ] Discrete GOMEA.
  - [ ] Real-valued GOMEA (RV-GOMEA).

### Closely related distribution-search methods

- [ ] **Natural evolution strategies (NES)**.
  - [ ] xNES.
  - [ ] Separable NES (SNES).
- [ ] **CMA-ES** — overlaps through adaptive Gaussian distribution search; usually classified as an evolution strategy.
- [ ] **Information-geometric optimization (IGO)** — a unifying framework rather than one fixed implementation.

**Important naming distinction:** BOA is an EDA. It is not the same as Gaussian-process Bayesian optimization in section 6.

**Pokémon interpretation:** independent models learn individual choice frequencies; dependency models can learn combinations such as species–item or teammate relationships. Sampling must handle conditional choices and legality. A team is an unordered set, so arbitrary slot ordering can distort learned dependencies.

See [Theory of Estimation-of-Distribution Algorithms](https://arxiv.org/abs/1806.05392) and [Information-Geometric Optimization](https://www.jmlr.org/beta/papers/v18/14-467.html).

## 5. Quality diversity (QD)

QD seeks a collection of high-performing, behaviorally different solutions rather than only one winner.

- [ ] **MAP-Elites** — retains elites in descriptor bins.
- [ ] **CVT-MAP-Elites** — uses centroidal Voronoi tessellation for the archive.
- [ ] **CMA-ME** — covariance matrix adaptation MAP-Elites.
- [ ] **CMA-MAE** — covariance matrix adaptation MAP-Annealing.
- [ ] **CMA-MEGA** — uses gradients of objective and descriptors; it needs gradient information or estimates and is not a straightforward gradient-free baseline.
- [ ] **Novelty search with local competition (NSLC)**.
- [ ] **AURORA** — learns behavioral descriptors for QD.
- [ ] **Surrogate-assisted illumination (SAIL)** — combines QD and surrogate modeling.

Plain novelty search is adjacent: it rewards novelty, but does not necessarily optimize quality. Archive choice, emitter choice, and scheduling can create additional QD variants without defining entirely new families.

**Pokémon interpretation:** maintain strong teams across descriptors such as speed-control strategy, weather usage, or offensive/defensive style. Those descriptors are design proposals, not standardized Pokémon measures.

The official [pyribs documentation](https://docs.pyribs.org/en/latest/index.html) describes MAP-Elites-related algorithms and separates archives, emitters, and schedulers.

## 6. Model-based / surrogate-assisted optimization

Here “model-based” means learning from evaluations to guide future evaluations. Some methods predict the objective; TPE instead models parameter densities conditioned on performance. EDAs are also model-based in a broader sense, but model promising solutions rather than necessarily predicting their scores.

- [ ] **Gaussian-process Bayesian optimization (GP-BO)**.
  - [ ] Expected improvement (EI), probability of improvement (PI), and upper/lower confidence bounds — acquisition rules, not independent optimizer families.
  - [ ] Thompson sampling — posterior-sampling decision rule.
- [ ] **SMAC** — sequential model-based algorithm configuration; commonly uses random-forest surrogates.
- [ ] **TPE** — tree-structured Parzen estimator.
  - [ ] Multivariate TPE.
  - [ ] Grouped/conditional TPE variants.
- [ ] **TuRBO** — trust-region Bayesian optimization.
- [ ] **BOCS** — Bayesian optimization of combinatorial structures.
- [ ] **COMBO** — combinatorial Bayesian optimization using graph representations.
- [ ] **BOHB** — Bayesian optimization combined with Hyperband; also multi-fidelity.
- [ ] **RBF surrogate optimization**, including **DYCORS** — dynamically perturbed candidate generation with a surrogate.
- [ ] **Surrogate-assisted evolutionary algorithms (SAEAs)** — use predictive models to guide evolutionary search.
- [ ] **SMAC-style surrogate-assisted local search** — a hybrid architecture rather than one universal named algorithm.

**Pokémon interpretation:** useful when battles are expensive and evaluations are limited. Conditional search spaces are important: legal moves and abilities depend on the species. A good categorical encoding is essential for continuous-model approaches.

See the official [SMAC3 repository](https://github.com/automl/SMAC3), [Optuna sampler documentation](https://optuna.readthedocs.io/en/stable/reference/samplers/index.html), and [RoBO documentation](https://www.automl.org/automl/robo/) for surrogate and acquisition distinctions.

## 7. Noise handling — a wrapper or component

These are evaluation and selection strategies, not a competing metaheuristic family. Most can accompany the methods above, although some are specific to a model or optimizer.

- [ ] **Fixed resampling** — use multiple battles per candidate.
- [ ] **Adaptive resampling** — spend additional evaluations where uncertainty affects a decision.
- [ ] **Incumbent reevaluation** — periodically retest the current best.
- [ ] **Common random numbers / paired evaluation** — compare teams under matched opponent and scenario seeds; useful when pairing reduces variance.
- [ ] **Racing** — eliminate candidates as evidence accumulates.
  - [ ] F-Race.
  - [ ] Iterated racing / irace — a configuration method that combines racing with sampling.
- [ ] **Sequential statistical comparisons** — gather evidence until a comparison meets an appropriate stopping rule.
- [ ] **Bandit-based evaluation allocation** — focus battles on competitive or uncertain candidates.
- [ ] **Robust aggregation** — trimmed means, medians, or median-of-means when justified by the noise distribution.
- [ ] **Uncertainty-aware ranking** — incorporate confidence or posterior uncertainty in selection.
- [ ] **Noise-aware Bayesian optimization** — model observation noise.
  - [ ] Noisy expected improvement / qNEI.
  - [ ] qLogNEI — numerically improved log formulation.
- [ ] **Independent finalist validation** — reevaluate selected teams on fresh seeds and held-out opponents to reduce selection bias.

Robust aggregation changes the estimated quantity in some settings: a median battle score is not an expected win rate. Use an estimator that matches the objective. Robustness against different opponents is also distinct from repeated-evaluation noise.

See [Noisy Optimization: Convergence with a Fixed Number of Resamplings](https://arxiv.org/abs/1404.2553) and the official [BoTorch acquisition documentation](https://botorch.readthedocs.io/en/stable/acquisition.html).

## Additional families and strategies missing from the list

### 8. Direct-search / derivative-free numerical optimization

Black-box methods, though not all are metaheuristics. Mainly relevant to numerical subproblems rather than direct species selection.

- [ ] Nelder–Mead simplex.
- [ ] Powell’s direction-set method.
- [ ] Hooke–Jeeves pattern search.
- [ ] Generalized pattern search (GPS).
- [ ] Mesh adaptive direct search (MADS).
- [ ] COBYLA.
- [ ] BOBYQA — builds local quadratic models, so also model-based.

Numerical stat allocations may still have integer constraints and discontinuities; these methods need appropriate handling. See [NLopt’s official algorithm guide](https://nlopt.readthedocs.io/en/stable/NLopt_Algorithms/).

### 9. Global partitioning and random-search baselines

- [ ] Uniform random search.
- [ ] Stratified / Latin-hypercube sampling.
- [ ] Quasi-random / low-discrepancy sampling, such as Sobol sequences.
- [ ] DIRECT and DIRECT-L — partition continuous search regions.
- [ ] Controlled random search (CRS).
- [ ] Multilevel single-linkage (MLSL) — multistart with clustering.

Sampling designs are baselines or initialization tools, not necessarily metaheuristics. See [NLopt](https://nlopt.readthedocs.io/en/stable/NLopt_Algorithms/) and [Optuna’s sampler catalog](https://optuna.readthedocs.io/en/stable/reference/samplers/index.html).

### 10. Multi-fidelity / budget allocation

These can wrap or combine with optimizers; they do not define how teams are mutated.

- [ ] Successive halving.
- [ ] Hyperband.
- [ ] BOHB.
- [ ] DEHB — differential evolution with Hyperband.
- [ ] Multi-fidelity Bayesian optimization.

Pokémon example: screen teams with a small battle budget, then give promising teams more battles. A small sample is a noisy estimate, and promotion can discard strong teams by chance. See the [AutoML project overview](https://automl.org/).

### 11. Hybrids and hyper-heuristics

- [ ] Memetic search — GA or another population method plus local refinement.
- [ ] Surrogate-assisted GA, DE, or local search.
- [ ] EDA plus local search.
- [ ] QD plus a surrogate model.
- [ ] Adaptive operator selection.
- [ ] Selection hyper-heuristics — select search operators during a run.
- [ ] Algorithm portfolios — allocate work across different optimizers.

These combinations overlap earlier groups rather than extending a clean, mutually exclusive hierarchy.

## A practical interpretation for your project

Separate three decisions when comparing approaches:

1. **Search mechanism:** local search, evolution, distribution learning, or surrogate-guided search.
2. **Desired result:** one strong team or a diverse repertoire of strong teams.
3. **Evaluation strategy:** battle budget, noise handling, opponent distribution, and independent validation.

For the current blog, focus on categorical CEM and a random-search baseline using the same candidate space and battle budget. The broader algorithm comparison can be future work. This is a proposed benchmark design, not a claim that these algorithms have already won your benchmark.

Use a consistent battle-playing agent, legal candidate representation, opponent suite, and total battle budget. Compare several optimizer seeds and validate finalists independently. A more elaborate algorithm cannot compensate for a scoring function that rewards exploitable or unrealistic battle behavior.
