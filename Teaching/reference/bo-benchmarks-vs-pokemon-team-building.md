*Retired 2026-09-24 — wrong game (Regulation H, EVs/IVs/Tera); superseded by [[bo-benchmarks-vs-vgc-team-building]]*

# Academic Bayesian optimization benchmarks versus Pokémon team building

Research collected 2026-09-13; extended with explicit format assumptions and additional benchmark families on 2026-09-16. BO means **Bayesian optimization**. This is a representative collection of established benchmark families, not an exhaustive list or a measured ranking of how frequently papers use them. A benchmark is a task or suite; BOCS and COMBO are optimizers whose experimental tasks have become useful reference problems.

## Format and evaluation assumptions

This comparison uses **Pokémon Scarlet/Violet, historical Regulation H**, with a frozen simulator and legality database. It is not a statement of today's tournament format. Regulation H supplies a concrete eligible-species list, level-50 battles, and no duplicate held items. [Official Regulation H specification](https://sv-news.pokemon.co.jp/en/page/245.html)

Assume six registered members, doubles, four selected for each game, and an open-team-sheet information regime. The pilot makes preview, lead, move, switch, and Terastallization decisions. These mechanics are described in [VGC-Bench, §2](https://arxiv.org/html/2506.10326v2). For the baseline experiment, score independent single games under that information regime; this is a controlled research protocol, not a faithful reproduction of best-of-three tournament performance. The latter needs a series-level evaluator and a policy able to use between-game information. [Official discussion of Regulation H competition formats](https://www.pokemon.com/uk/strategy/how-to-build-a-team-and-battle-in-pokemon-scarlet-and-pokemon-violet-regulation-set-h)

Optimize expected win probability against a frozen distribution of opposing teams **and pilots**, using a fixed team-conditioned pilot for our teams. Count a win as 1 and any other outcome as 0; alternatively predeclare a draw score and call the resulting quantity expected score. No human-level pilot, universal team ranking, or known global optimum is assumed.

## Species selection is not complete team optimization

| Search task | Decisions actually optimized | What remains fixed or missing |
|---|---|---|
| Select six species | A size-six subset, subject to format eligibility and species restrictions. With N distinct eligible species and no further restrictions, there are binomial(N,6) subsets. Forms and additional clauses complicate this count. | A species subset cannot be battled until someone supplies its sets. If a fixed template supplies each set, the score belongs to those templates. |
| Select six prebuilt sets | A constrained subset of a library of fully configured members. | Moves, items, abilities, EVs, and Tera types are only optimized to the extent represented in that library; species/item conflicts still need checking. |
| Optimize complete teams | Species/form, up to four legal moves, held item or none, legal ability, EV allocation, and Tera type for each of six members; also nature and IVs unless explicitly fixed. | Battle decisions remain the pilot's responsibility. Selecting four from six at preview is a separate decision inside evaluation. |

The configuration fields are documented in [Showdown's team representation](https://github.com/smogon/pokemon-showdown/blob/master/sim/TEAMS.md). Its [validator](https://github.com/smogon/pokemon-showdown/blob/master/sim/team-validator.ts) checks species-dependent learnsets and legal attributes, including standard Scarlet/Violet EV limits of 252 per stat and 510 total. [Species and Item Clause definitions](https://github.com/smogon/pokemon-showdown/blob/master/config/CUSTOM-RULES.md#clauses) establish team-wide restrictions.

**Analysis:** The full domain is a finite, constrained, conditional space of categorical and integer choices. EVs are integers; treating them as continuous inputs is an optimization relaxation, not the native game. Species IDs have no meaningful numeric distance. Species changes can invalidate moves and abilities; item uniqueness couples different members; integer stat calculations create plateaus and thresholds. Large cardinality alone neither measures optimization difficulty nor proves that one search algorithm is appropriate. Roster-order equivalence must be checked against mechanics and the pilot before canonicalizing encodings.

## Representative academic benchmarks

| Benchmark / family | Domain and scale | Objective, noise, and constraints | What the experiment actually tests |
| --- | --- | --- | --- |
| **Branin** | Two continuous variables, conventionally `[-5,10] × [0,15]`. | Analytic scalar minimization with three global minima; deterministic baseline and box bounds. Added-noise variants must be specified separately. | Small multimodal search with a known optimum; a cheap diagnostic for exploration, rather than a realistic expensive evaluator. [Official implementation](https://botorch.readthedocs.io/en/v0.15.0/test_functions.html#botorch.test_functions.synthetic.Branin) |
| **Hartmann-3 / Hartmann-6** | Three or six continuous variables; Hartmann-6 conventionally `[0,1]^6`. | Analytic scalar minimization; Hartmann-6 has multiple local minima. Noise is optional in implementations. | Modest-dimensional nonlinear search with a known optimum. [BoTorch implementation documentation](https://botorch.readthedocs.io/en/stable/test_functions.html#botorch.test_functions.synthetic.Hartmann) |
| **Ackley** | Dimension `d` is configurable; common mathematical domain `[-32.768,32.768]^d`, sometimes narrowed. | Deterministic analytic objective with many local minima and a central optimum; box bounds. | Multimodality and dimensional scaling. Dimension and chosen bounds matter; “Ackley” alone does not fully specify an experiment. [Official implementation](https://botorch.readthedocs.io/en/v0.15.0/test_functions.html#botorch.test_functions.synthetic.Ackley) |
| **Rosenbrock** | Configurable continuous dimension `d`, often `[-5,10]^d`. | Analytic objective with a narrow curved valley leading to the known global optimum; optional injected noise in BO implementations. | Conditioning and local refinement across interacting coordinates. [BoTorch documentation](https://botorch.readthedocs.io/en/stable/test_functions.html#botorch.test_functions.synthetic.Rosenbrock) |
| **Bayesmark / NeurIPS BBO Challenge 2020** | ML hyperparameter spaces containing real, integer, categorical, and Boolean variables; dimension varies by model. | Validation loss from training/evaluating an ML model on a fixed dataset; noise properties depend on the task. | Real HPO, batch suggestion, optimizer overhead, and generalization across held-out optimization tasks. Challenge budget: 16 batches of 8 evaluations, with 640 seconds total optimizer suggestion time. [Challenge paper](https://proceedings.mlr.press/v133/turner21a/turner21a.pdf), [Bayesmark repository](https://github.com/uber/bayesmark) |
| **HPOBench** | Multiple raw, tabular, and surrogate families. Examples: SVM and logistic regression: 2 hyperparameters; random forest and XGBoost: 4; MLP: 5. Includes categorical NAS families. | ML validation performance with configurable training fidelity; raw runs, stored evaluations, and learned emulators have different noise/cost semantics. New tabular families include five seeds per grid configuration. | Mixed parameter types, multi-fidelity allocation, reproducibility, and transfer. A stored table lookup reproduces recorded training outcomes; it does not rerun the expensive training process. [HPOBench paper, Table 1 and §4](https://arxiv.org/html/2109.06716) |
| **HPO-B** | OpenML-derived historical HPO evaluations; original collection: 176 search spaces, 196 datasets, 6.4 million evaluations. Standard v2/v3 subsets use 16 frequent spaces. | Accuracy maximization from recorded evaluations, with optional learned surrogates for continuous queries; dimensionality varies by space. | Reproducible black-box HPO and learning across tasks. A finite table and a fitted continuous surrogate define different accessible search domains. [Paper](https://arxiv.org/abs/2106.06257), [authors’ repository and version definitions](https://github.com/machinelearningnuremberg/HPO-B) |
| **Binary quadratic / Ising sparsification tasks in BOCS** | Paper examples: 10 binary choices for quadratic programming; 24 binary edge-retention choices for Ising sparsification. | Scalar objectives over interacting discrete decisions; sparsification trades approximation quality against retained structure. | Combinatorial interactions and evaluation efficiency. The benchmark task is distinct from the BOCS model’s chosen approximation to it. [BOCS paper, §4](https://proceedings.mlr.press/v80/baptista18a/baptista18a.pdf) |
| **Contamination control in BOCS** | 25 binary prevention decisions: `2^25` schedules. | Simulated contamination dynamics; prevention costs, penalties for contamination-limit violations, and a sparsity term. Paper evaluates each candidate using 100 simulations. | Costly stochastic simulation, coupled sequential decisions, and risk penalties. Constraints are incorporated into the reported scalar objective by relaxation. [BOCS paper, §4.3](https://proceedings.mlr.press/v80/baptista18a/baptista18a.pdf) |
| **Pest control in COMBO** | Published NeurIPS 2019 experiment: 21 stages × 5 categorical choices, or `5^21 ≈ 4.77×10^14` configurations. | Modified contamination-control objective with more complex, higher-order interactions. | Large categorical search; paper uses 320 evaluations including 20 initial random points. Later code/benchmark variants may use different stage counts: report the exact version, rather than silently equating them. [COMBO paper, §4.2](https://papers.neurips.cc/paper/8557-combinatorial-bayesian-optimization-using-the-graph-cartesian-product.pdf) |
| **Weighted MaxSAT tasks in COMBO** | Binary assignment problems; experiments named wMaxSAT28, wMaxSAT43, and wMaxSAT60. | Deterministic weighted clause-satisfaction objective, negated for minimization; binary assignments are the search domain. | Rugged discrete interactions under limited evaluations; paper budgets 270 evaluations including 20 initial points. This evaluates black-box search, rather than superiority over specialized MaxSAT solvers given full formula access. [COMBO paper, §4.3](https://papers.neurips.cc/paper/8557-combinatorial-bayesian-optimization-using-the-graph-cartesian-product.pdf) |
| **NAS-Bench-201** | Neural-network cell architectures: four nodes, six edge-operation choices with five alternatives each, giving 15,625 encodings. | Recorded training/evaluation results on CIFAR-10, CIFAR-100, and ImageNet16-120; cheap lookup replaces new training. | Structured composition and interacting categorical choices. Useful adjacent benchmark for team composition, although architecture evaluation has a fixed training protocol rather than an opponent. [Authors’ paper](https://arxiv.org/abs/2001.00326), [official repository](https://github.com/D-X-Y/NAS-Bench-201) |
| **Welded-beam design, constrained single-objective variant** | Four continuous design variables. | Analytic engineering-design objective with six black-box constraints in the documented BoTorch variant; distinguish its modified multiobjective variant. | Feasible-region discovery and constraint handling; differs from a team legality checker that can reject a composition before simulating a battle. [BoTorch specification](https://botorch.readthedocs.io/en/stable/test_functions.html) |
| **Robot pushing / rover / lunar lander in TuRBO** | Continuous: 14D robot control, 60D trajectory parameters, 12D lunar controller. | Robot pushing is noisy. Lunar landing averages reward over a fixed set of 50 scenarios; stochastic environments need not imply fresh noise on every query. | Simulation-based evaluation, rollout aggregation, and high-dimensional search. These cover some evaluation difficulties of Pokémon without a strategic opponent. [TuRBO paper, §3](https://papers.nips.cc/paper/8788-scalable-global-optimization-via-local-bayesian-optimization.pdf) |
| **Molecular/reaction optimization in GAUCHE** | Structured molecular representations including graphs, strings, and fingerprints; domain depends on the dataset. | Property or reaction-yield targets; benchmark datasets and experimental measurements must be distinguished from live physical experiments. | An adjacent family for learning similarity between structured candidates. GAUCHE supplies GP tools and demonstrations rather than one universal optimization task. [GAUCHE paper](https://proceedings.neurips.cc/paper_files/paper/2023/file/f2b1b2e974fa5ea622dd87f22815f423-Paper-Conference.pdf), [code](https://github.com/leojklarner/gauche) |

The 25-stage, five-choice pest-control variant is also used in [CASMOPOLITAN, §4.1](https://proceedings.mlr.press/v139/wan21b/wan21b.pdf). Its search space has roughly 3 × 10^17 configurations. Both it and COMBO’s 21-stage task are legitimate named variants.

## Interpretation guardrails

### Additional standard families and their relevance

| Benchmark | Documented task and evaluation | What it tests | Fit to complete VGC teams — our analysis |
|---|---|---|---|
| **COCO/BBOB** | Base suite: 24 noiseless scalar continuous functions across dimensions and instances. Separate noisy, mixed-integer, constrained, and biobjective suites exist. Formula evaluations are cheap. [Official base suite](https://numbbo.github.io/coco/testsuites/bbob), [suite inventory](https://coco-platform.org/data-archive/index.html) | Separability, conditioning, multimodality, and dimensional scaling. This is general black-box optimization infrastructure, not exclusively BO. | Good controlled stress tests; even mixed-integer versions do not supply species-dependent legal sets or competitive opponents. |
| **YAHPO Gym** | Surrogate HPO suite with mixed and hierarchical scenarios, fidelity controls, and multiple outputs including performance, runtime, and memory. The documented rbv2_super scenario has 38 parameters. Queries cheaply emulate training. [Pfisterer et al., 2022](https://proceedings.mlr.press/v188/pfisterer22a.html), [official scenarios](https://slds-lmu.github.io/yahpo_gym/scenarios.html) | Conditional configuration, allocation of training resources, and Pareto trade-offs. Select an actual hierarchical scenario rather than assuming every task is hierarchical. | One of the closest analogues for dependent set choices. Still lacks roster symmetry, species/item clauses, and opponent-conditioned payoff; surrogate errors differ from fresh battle noise. |
| **CoCaBO experimental tasks** | Func-2C combines two categorical and two continuous coordinates; mixed Ackley variants discretize some coordinates. Synthetic tasks are cheap scalar functions. [Ru et al., 2020](https://proceedings.mlr.press/v119/ru20a/ru20a.pdf) | Statistical sharing and interactions between numeric and categorical choices. | Useful for testing mixed encodings, but numeric coordinates are shared across categories in this formulation. That is weaker than species-dependent legal move/ability domains, and EVs remain integer-valued. |
| **Branin–Currin, DTLZ, ZDT** | Branin–Currin: two inputs/two outputs. DTLZ varies dimension, objective count, and front geometry. ZDT1–3 are continuous two-objective tests; the whole ZDT family is not exclusively continuous. Analytic evaluations are cheap and ordinarily deterministic; noise must be added explicitly. [Official BoTorch implementations](https://botorch.readthedocs.io/en/v0.15.0/test_functions.html#module-botorch.test_functions.multi_objective), [Deb et al. DTLZ report](https://www.research-collection.ethz.ch/server/api/core/bitstreams/4d08c69c-9f97-424d-bbf6-7a647bcb3d13/content), [authors' DTLZ implementation](https://sop.tik.ee.ethz.ch/pisa/variators/dtlz/index.html) | Convergence to and coverage of a Pareto front; these families originated in broader multiobjective optimization. | Appropriate only when the team problem has explicit competing objectives, such as average and worst-matchup performance. Known analytic fronts omit legality and strategic adaptation. |

All these baseline instances have fixed task definitions. None inherently evaluates candidates against an opponent that learns in response to the candidates. The same applies to the standard analytic, HPO, NAS, and control tasks above.

An enormous number of possible configurations does not itself make a benchmark Pokémon-like. Independent categorical coordinates, legal team composition, and a collection of interchangeable team slots imply different geometry and constraints. Likewise, real-valued inputs can have complicated interactions: continuous benchmarks are not necessarily additive or easy.

Separate **intrinsic evaluation cost** from **benchmark execution cost**. An analytic function is cheap but can be assigned a strict evaluation budget; a tabular HPO benchmark cheaply replays measurements from expensive training. A live battle simulator spends real compute on each trial. Compare both objective-call efficiency and total compute when the latter matters.

Separate **stochasticity of the underlying process** from **repeatability of the provided evaluator**. Fixed seeds or stored samples can make a stochastic-process benchmark repeatable. Conversely, injected Gaussian noise on an analytic function is a specific noise model, not evidence that it reproduces battle-outcome uncertainty.

## How these benchmarks compare with Pokémon team building

The comparisons below are methodological deductions for this project, not findings that the cited BO papers tested on Pokémon. “BO” means Bayesian optimization. The relevant Pokémon task is building a legal VGC roster; exact attributes and legality must be tied to a named game and regulation, rather than assuming that every generation uses the same mechanics.

### Define what a team’s score means

For a controlled experiment, define

\[
f(T;D,\pi)=\mathbb{E}_{(O,\rho)\sim D,\,\omega}
\big[R(T,O,\pi,\rho;\omega)\big].
\]

Here, \(T\) is our team, \(O\) the opponent team, \(\pi\) our team-conditioned policy including preview, \(\rho\) the opponent policy, and \(\omega\) battle/policy randomness. Define \(R\) as 1 for a win and 0 otherwise if the target is win probability; explicitly choose a different score if draws count partially.

- **Counterteam:** concentrate \(D\) on one opponent team and a specified opponent policy or policy mixture.
- **Anti-meta team:** use a fixed, frequency-weighted distribution of opponent teams and policies.
- **Robust team:** optimize a worst-case or other robustness criterion over a declared opponent set. This is a different objective from average win rate.
- **Team plus trained pilot:** evaluate a specified training procedure for each team. This adds training cost, training randomness, and dependence on the training budget.

With \(D\), policies, rules, and simulator held fixed, \(f\) is a stationary function even though individual battle results are random. Updating the surrogate does not change \(f\). Updating the pilot or opponent population generally does. BO is applicable to the fixed version; adaptive versions need their changing context handled explicitly.

This distinction follows the team-conditioned game formulation in [VGC-Bench, §2](https://arxiv.org/html/2506.10326v2), but the outer team-search objective above is our proposed formulation.

### The differences that matter

| Property | What existing BO benchmarks already cover | What Pokémon team building requires |
|---|---|---|
| Input structure | Continuous functions use vectors; combinatorial tasks use binary/categorical assignments; NAS and molecular tasks have richer structures. | A roster of configured members: species and species-dependent moves/abilities, items, and regulation-specific stat/mechanic choices. A small number of team slots is not a low-dimensional search space. |
| Feasibility | Box bounds, constrained analytic surfaces, conditional HPO spaces, and valid graph restrictions all exist. | Both member-level legality and cross-team restrictions. Check legality before spending battle evaluations; a known rule violation should not require learning from losses. |
| Meaning of “nearby” | Euclidean distance is natural for many continuous tasks; categorical/graph kernels use other similarities. | Changing one species ID is not a small numerical change. Similarity should reflect legal edits or relevant features; even a one-move edit can change the strategic role. |
| Interactions | Rosenbrock, Ising, pest control, NAS, and chemistry already test coupled choices. | The usefulness of a member depends on teammates, opponents, and available battle decisions. A strong individual component need not improve the assembled team. |
| Symmetry | Graph benchmarks can have equivalent encodings. Many vector benchmarks have meaningful coordinate positions. | Roster serialization can produce duplicate representations of the same composition. Canonicalize only after checking order-sensitive policy behavior and relevant game mechanics; preview lead selection remains meaningful. |
| Evaluation | Analytic tests return a cheap exact value; HPO and simulation can involve costly, noisy runs. | One battle returns an outcome. A useful team estimate usually requires repeated battles across the declared matchups. “100 evaluated teams” does not specify the actual cost. |
| Noise | Noisy BO and stochastic simulators already exist. | Win-rate uncertainty depends on the underlying probability, number of games, matchup sampling, and policy randomness. More games reduce sampling error, but cannot correct a poorly chosen evaluator. |
| Multiple objectives | DTLZ/Branin–Currin explicitly test Pareto trade-offs; YAHPO exposes performance/resource outputs. | Average win probability is a single objective even if many opponents are sampled. Separate matchup win rates, average versus worst-case performance, or performance versus pilot cost can define multiple objectives. Speed, bulk, and type coverage are not automatically objectives; they may only predict wins. |
| Pilot dependence | HPO scores depend on the learning procedure; controller benchmarks evaluate a policy directly. | Team value depends on how well the pilot uses that particular team. A frozen weak pilot may systematically undervalue unfamiliar strategies. |
| Opponent dependence | A fixed dataset/environment is typical for a benchmark instance; contextual and robust variants also exist. | Rankings can change with the opponent mixture. A team can be excellent against a target and poor against the wider meta. |
| Ground truth | Many analytic functions and finite tables expose an optimum to the evaluator; realistic problems often do not. | The global best legal team is generally unknown. Report performance against a declared reference and evaluation population rather than pretending to measure exact global regret. |

Pokémon legality examples are grounded in [Showdown’s rule documentation](https://github.com/smogon/pokemon-showdown/blob/master/config/CUSTOM-RULES.md?plain=1) and [team validator](https://github.com/smogon/pokemon-showdown/blob/master/sim/team-validator.ts). These distinguish obtainable moves/abilities and team-wide clauses. Use the chosen format’s actual rules; this note does not prescribe the current tournament regulation.

The evaluator concern has empirical support: [VGC-Bench](https://arxiv.org/html/2506.10326v2) reports difficulty generalizing battle policies across diverse teams. Its benchmark primarily evaluates policies over supplied team pools; it is useful infrastructure, not an already complete benchmark of unrestricted BO team generation.

### Two concrete consequences

**Precision costs battles.** Under independent identically distributed win/loss trials with win probability \(p\), a mean over \(n\) battles has variance \(p(1-p)/n\). At \(p=0.5\), the standard error is about 5 percentage points for 100 games and 2.5 points for 400. These are standard errors, not 95% confidence intervals. Fixed per-matchup allocations require the corresponding weighted variance calculation. Selecting the largest noisy result among many teams also introduces selection bias, so confirm finalists with fresh evaluation games.

**Pilot error is different from randomness.** Suppose a candidate requires a setup sequence that the evaluation policy rarely uses. More games will estimate that policy’s performance more accurately; they will not establish how well a capable player would perform with the team. Freezing a pilot gives a reproducible objective, but the claim remains “best for this evaluation setup.”

Neither issue makes Pokémon impossible for BO. They determine what must be modeled and what an experimental result actually establishes.

## The most informative benchmark collection for this project

This is a proposed evaluation progression, not a claim that these tasks predict Pokémon performance.

1. **Branin and Hartmann-6:** verify that the basic BO loop behaves sensibly on familiar controlled problems.
2. **Pest control and contamination control:** test sample efficiency with interacting discrete decisions.
3. **One conditional HPO or valid NAS task:** test dependencies between choices and a structured search space. Check that the selected instance actually includes the desired conditional/validity restrictions.
4. **One stochastic control task:** test handling of rollout noise and evaluation cost.
5. **A small Pokémon candidate pool:** freeze regulation, policies, and opponent weights; evaluate all candidates extensively to obtain a high-precision reference within that pool. It remains an estimated reference, not an exact win-probability oracle.
6. **Generation of new legal teams:** keep the evaluator fixed and search outside the pool. Report that the global optimum is unknown.
7. **Separate transfer test:** evaluate selected teams against held-out pilots or a later opponent distribution. This measures robustness beyond the search setup.

For fair comparisons, use the same feasible search domain and starting data, charge repeated battles and any team-specific training to the budget, and report both total battles and elapsed time. Compare with legal random search, local mutation search, and an evolutionary baseline as well as an appropriate discrete/conditional BO method. Validate finalists on fresh games and repeat the complete search under multiple seeds. Any hand-designed candidate restrictions should apply equally across methods.

Use **YAHPO's hierarchical scenarios** when the specific research claim is handling conditional configurations; **pest control** for interacting categorical search; and **NAS** for compositional representation. Use **DTLZ or Branin–Currin** only for a declared multiobjective method. These are complementary tests, not a validated ranking of predictors of VGC success.

More battles with the same opponent distribution and pilot primarily improve the precision of the same mean. A weaker pilot, shortened game, or smaller biased opponent pool can change the target itself. Such shortcuts are not automatically trustworthy low-fidelity evaluations. If each team gets a separately trained pilot, specify the training procedure and budget as part of the objective; if the pilot population changes during search, treat the evaluation context as changing.

An opponent-indexed payoff vector can have cyclic preferences: a team can beat one style and lose to another, so no universal total ordering is guaranteed. With a fixed weighted opponent population, the scalar expected score is nevertheless well-defined and stationary. These are deductions from the declared objective, not claims that the cited BO papers demonstrated a Pokémon advantage.

## What this supports as an academic claim

The defensible motivation is **sample-efficient search over legal, structured teams when feedback comes from policy-dependent competitive simulations**.

Large search spaces, discrete choices, interactions, expensive evaluations, and observation noise each already occur in BO benchmarks. Pokémon combines them with opponent-dependent utility and the possibility that the evaluator cannot competently use the generated design. The benchmark contribution would be to isolate and measure that combination.

Success on Branin alone would provide little evidence of effective team building. Success on combinatorial/HPO/control tasks would establish relevant capabilities, while controlled Pokémon experiments would still be needed to establish the project’s main claim.
