# GECCO methods and reading priorities

Checked October 20, 2026. The goal remains higher independently evaluated battle win rate. Rankings assess fit to the current team-search question, not venue prestige or demonstrated VGC performance.

The strongest directions are analysing existing search trajectories, testing bounded local refinement inside the current search, and learning which candidates merit expensive battle evaluation. Five MED references were added to the reading list. Tutorials and interesting continuous-only workshop papers remain separate from that active shortlist.

## Five additions, with a reading order

This order applies to the new topic-specific shortlist. It does not replace the current experiment queue. Citation counts were not refreshed.

| Priority | Addition | Read for | Boundary |
| --- | --- | --- | --- |
| 1 | [Search trajectory networks: A tool for analysing and visualising the behaviour of metaheuristics](https://doi.org/10.1016/j.asoc.2021.107492), Ochoa, Malan and Blum, 2021, MED | Analyse transitions, revisits and visited regions from actual search logs. Sections 3 and 5 include fixed-cardinality p-median and compare different search methods. [Open manuscript](https://www.storre.stir.ac.uk/bitstream/1893/32613/1/stns_asoc_2021.pdf). | A trajectory graph describes where an algorithm searched. It does not reveal every basin or improve search by itself. |
| 2 | [Cross entropy-based memetic algorithms: An application study over the tool switching problem](https://doi.org/10.1080/18756891.2013.792542), Amaya, Cotta and Fernández-Leiva, 2013, MED | Combine cross-entropy probability models with local improvement. Section 3 exposes refinement probability and an evaluation cap. [Publisher full text](https://link.springer.com/content/pdf/10.1080/18756891.2013.792542.pdf). | Deterministic permutation scheduling differs from noisy, expensive battle scores. A bounded refinement test still needs matched costs and fresh validation. |
| 3 | [Local Optima Networks: A New Model of Combinatorial Fitness Landscapes](https://arxiv.org/abs/1402.2959), Ochoa, Verel, Daolio and Tomassini, 2014, MED | Learn local optima, basins and possible transitions between them. This is a chapter with quadratic-assignment and NK examples. | Metadata and abstract verified here. Identifying VGC basins could require substantial extra evaluation. |
| 4 | [Greedy Randomized Adaptive Search Procedures](https://doi.org/10.1007/BF01096763), Feo and Resende, 1995, MED | Sections 1 and 2 explain incremental randomized construction followed by local improvement. [Author full text](https://mauricio.resende.info/doc/gtut.pdf). | A design foundation with combinatorial examples, not evidence that a partial-team heuristic predicts final team strength. |
| 5 | [Exploratory Landscape Analysis for Mixed-Variable Problems](https://arxiv.org/abs/2402.16467), Prager and Trautmann, online 2024, final journal issue 2025, MED | Understand categorical encoding, landscape features and algorithm selection on mixed-variable hyperparameter tasks. [Publisher DOI](https://doi.org/10.1109/TEVC.2024.3399560). | Later methodology. VGC needs its own representation, feature-cost and held-out algorithm-selection validation. |

These references include real discrete or mixed-variable settings. Their MED grades express selective reading value. They do not establish a ready-made solver for this project.

The existing Hearthstone surrogate paper remains in the reading list. The existing GECCO 2026 diffusion-EDA entry remains MED with a new access caveat: its acceptance and ECOM track are verified, while its full text and performance were not verified in this screening. [Official 2026 accepted papers](https://gecco-2026.sigevo.org/Accepted%2BPapers).

## Rank the tracks by the contribution

| Rank | Main track | Why it fits |
| --- | --- | --- |
| 1 | [ECOM](https://gecco-2026.sigevo.org/Track?itemId=52), Evolutionary Combinatorial Optimization and Metaheuristics | The broadest direct fit for legal discrete team search. Its scope includes representations, neighborhoods, constraints and landscape analysis used to design stronger optimizers. |
| 2 | [L4EC](https://gecco-2026.sigevo.org/Track?itemId=2671), Learning for Evolutionary Computation | Best when the contribution is a learned generator, scorer, operator controller or landscape model that improves evolutionary search. |
| 3 | [GECH](https://gecco-2026.sigevo.org/Track?itemId=57), General Evolutionary Computation and Hybrids | Best when the contribution is how search components cooperate, such as evolution with local search, constructive heuristics or different evaluation fidelities. This is the source of the hybrid list in your question. |

Move L4EC to first for a learned-search contribution, or GECH to first for a hybrid-coordination contribution. A game's presence in the experiments does not determine the track by itself.

### Excluded from the current shortlist

On 20 October 2026, the project owner excluded the former fourth and fifth choices, BBSR and GA. This is a contribution-fit decision for the current project, not a claim of formal ineligibility.

**BBSR:** The current contribution aims to improve independently evaluated battle win rate through stronger search, learned proposals and efficient acquisition. No standalone benchmark, diagnostic methodology, software contribution or replication study is currently proposed. Evaluation, reproducibility and landscape diagnostics support that search contribution. BBSR explicitly includes landscape analysis and statistical evaluation, so retain those practices while directing the method contribution toward ECOM or L4EC. [Official BBSR scope](https://gecco-2026.sigevo.org/Track?itemId=2672).

**GA:** The current claim concerns neural CEM proposal learning and efficient evaluation under noisy battle outcomes, without a declared GA-specific advance in representation, selection, recombination, mutation or population design. L4EC is the clearer fit for learned search; GECH fits a substantive contribution in how search components cooperate. GA includes model-based genetic algorithms, surrogates and landscape analysis. Its exclusion therefore follows the current contribution and framing, not an assumption that CEM or these methods are outside its scope. [Official GA scope](https://gecco-2026.sigevo.org/Track?itemId=56).

**RWA remains conditional and unranked:** a genuinely new application formulation would be needed. The call excludes merely applying an algorithm to an existing public benchmark. Removing BBSR and GA does not promote RWA into fourth place. [Official RWA scope](https://gecco-2026.sigevo.org/Track?itemId=60).

The remaining tracks are conditional. GP fits evolved programs or explicit heuristics; EML and NE fit evolution of learning systems or policies; EMO fits a genuinely multiobjective goal; SI fits a swarm-specific contribution; THEORY fits formal results; CS fits complex-systems research; ENUM centres on numerical optimization. Using machine learning to improve search usually points toward L4EC. [Official track inventory](https://gecco-2026.sigevo.org/Tracks).

## Rank the actual 2026 workshops

The official list contains 17 workshops. This review screened their scopes and selected paper leads. It did not fully read every accepted contribution. [Official workshop list](https://gecco-2026.sigevo.org/Workshops).

| Rank | Workshop | Reason and limit |
| --- | --- | --- |
| 1 | [SAEOpt](https://gecco-2026.sigevo.org/Workshop?itemId=8255), Surrogate-Assisted Evolutionary Optimisation | Closest to expensive candidate scoring, predicted rankings, model management and multiple fidelities. Its scope acknowledges limited combinatorial work. |
| 2 | [AABOH](https://gecco-2026.sigevo.org/Workshop?itemId=8238), Analysing algorithmic behaviour of optimisation heuristics | Closest to stagnation, structural bias, exploration and information collected during search. Diagnostics still need a tested algorithmic response. |
| 3 | [ECADA](https://gecco-2026.sigevo.org/Workshop?itemId=8243), Evolutionary Computation for the Automated Design of Algorithms | Useful for evolving constructive heuristics, mutation rules or operator selectors. An outer search over algorithms adds tuning cost. |
| 4 | [BENCH](https://gecco-2026.sigevo.org/Workshop?itemId=8239), Good Benchmarking Practices for Evolutionary Computation | Its 2026 focus is dynamic optimization, useful for future changes in opponents or formats. It explicitly accepts no paper submissions in 2026. |
| 5 | [Decomposition Techniques](https://gecco-2026.sigevo.org/Workshop?itemId=8241) | Useful for grouping interacting variables or dividing expensive search. It does not establish that Pokémon slots are independent. |
| 6 | [LLMfwEC](https://gecco-2026.sigevo.org/Workshop?itemId=8250), Large Language Models for and with Evolutionary Computation | Conditional on LLM-generated search operators or analysis of LLM variation. It is not automatically a priority for the current sampler. |
| 7 | [EvoOSS](https://gecco-2026.sigevo.org/Workshop?itemId=8245), Open Source Software for Evolutionary Computation | A possible route for reusable search or evaluation software. Its 2026 call requires already released software and a public repository. |

Landscape-Aware Heuristic Search appears in the 2025 companion programme, not the 2026 workshop list. Neither that workshop nor Evolutionary Computation in Games should be assumed to recur. The direct 2026 learning resource is [Landscape Analysis of Optimization Problems and Algorithms](https://gecco-2026.sigevo.org/Tutorial?itemId=8226), by Gabriela Ochoa and Katherine Malan. Other useful tutorials cover [ML for combinatorial optimization](https://gecco-2026.sigevo.org/Tutorial?itemId=8207), [model-based evolutionary algorithms](https://gecco-2026.sigevo.org/Tutorial?itemId=8215) and [automatic algorithm configuration](https://gecco-2026.sigevo.org/Tutorial?itemId=8228). These are tutorials, not research-paper results.

The workshop screen found interesting continuous-only papers on flexible EDAs, lagged behavioral diagnostics and pairwise surrogate classification. They remain LOW under the reading list's existing rule. The noisy LON/STN workshop paper also remains LOW because its verified abstract studies synthetic objectives with additive Gaussian noise. Title-only scheduling and GA-configuration leads remain ungraded holds. Their metadata, access limits and reasons are in the [workshop memo](../research_notes/GECCO%20methods%20and%20reading%2020261020/workshops.md) and [landscape memo](../research_notes/GECCO%20methods%20and%20reading%2020261020/landscape.md).

## Constructive heuristics build a candidate incrementally

A constructive heuristic starts with an empty or partial solution and repeatedly chooses a component to add. Its choices may be deterministic or randomized. GRASP constructs using a restricted list of promising choices, then applies a separate local-improvement phase. [Original GRASP paper](https://mauricio.resende.info/doc/gtut.pdf).

For VGC, a hypothetical constructor could choose a core pair, add a teammate to address an uncovered matchup, then fill settings while preserving legality. The difficulty is scoring partial teams: a species or partial team has no unique full-team win rate without a completion or aggregation rule.

Construction fills a partial candidate. Repair restores legality. Local search changes an already complete candidate. A legality checker is not a constructive strategy for finding strong teams.

## Multi-fidelity and multi-resolution are different choices

Multi-fidelity optimization uses cheaper approximations alongside an expensive reference evaluator. The target evaluator remains involved to establish accuracy. A surrogate prediction is one possible approximation. [Peherstorfer, Willcox and Gunzburger's survey](https://arxiv.org/abs/1806.10761).

For this project, define f(T) as expected win rate for legal team T under the fixed opponent distribution, battle policy, simulator and rules. Then distinguish these cases:

| Change | Meaning for f(T) |
| --- | --- |
| Fewer battles under the same sampling protocol | A less precise estimate of the same expectation. This changes statistical effort, often broadly called fidelity. |
| Surrogate prediction | A cheap approximation whose errors must be checked on unseen teams and simulation-tested finalists. |
| Cheaper battle policy, altered rules or truncated battles | Usually a biased proxy for a different evaluator. More repeats do not automatically remove that bias. |
| Fixed smaller opponent subset | Usually a different opponent objective. Sampling and weights must preserve the target expectation to estimate the same f(T). |
| Species or archetype choices before full sets | Coarser representation resolution. A completion rule is needed before assigning a full-team score. |

Multi-resolution concerns granularity. It can mean coarse-to-fine candidate representation or coarse-to-fine evaluation models. The GECH scope names it without defining a universal implementation. Fewer repeated battles do not create a finer battle model. Multiple approximations of one win-rate target also do not create a multiobjective problem. [GECH scope](https://gecco-2026.sigevo.org/Track?itemId=57).

## What the hybrid combinations do

EA means evolutionary algorithm. The useful question is what each component contributes and how it earns its extra evaluation cost. The VGC examples below are proposals. [GECH scope](https://gecco-2026.sigevo.org/Track?itemId=57), [mechanism sources and detailed definitions](../research_notes/GECCO%20methods%20and%20reading%2020261020/concepts.md).

| Hybrid | Mechanism | VGC interpretation and current priority |
| --- | --- | --- |
| EA + LS, memetic | Population search plus local improvement of selected candidates. | First comparison to consider: bounded legal item, move or whole-set refinement, versus spending that effort on more ordinary proposals. |
| EA + EA, including meta-EA | Evolution can tune another evolutionary process, or multiple processes can exchange candidates. | Second, conditional on a tuning need. An outer trial may require an entire team-search run. Separate tuning from final test tasks. |
| EA + ACO | Ant colony optimization builds candidates using learned pheromone preferences and heuristic information. | Third. It needs a justified construction graph, legality treatment and partial-team scoring. |
| EA + PSO | Swarm movement uses personal and shared best positions. | Lower priority. Ordinary continuous movement does not directly specify a legal species, move or item replacement. |
| EA + fuzzy methods | Graded membership and rules control or model algorithm behavior. | Lower priority. For example, graded stagnation rules could adjust mutation, but membership rules add design choices and are not battle-event probabilities. |

This ranks the clarity of a useful comparison and the additional machinery, not measured win rate. The cross-entropy memetic paper supplies a concrete discrete example for the first comparison. It does not justify unlimited local search or exempt refinement from the battle budget.

## Landscape analysis can inform the search

A fitness landscape requires candidates, an objective and a neighborhood. A team can be locally optimal under item changes while improvable through a whole-Pokémon replacement. Changing the policy or opponent distribution changes the objective and therefore the landscape.

Start with the information already produced by completed search runs:

1. Define legal item, move and whole-set neighborhoods separately. Record additional changes caused by legality repair.
2. Build search trajectory networks from logged candidates and transitions, if available. Retain run, operator, iteration, scores, uncertainty and evaluation provenance. Never infer an unlogged transition.
3. Check revisits, repeated regions and score changes. These describe visited states, not the complete search space.
4. Use controlled local comparisons when extra evaluations are justified. Paired opponents and randomness help compare edits. Similar sampled scores mean indistinguishable at current precision, not certified neutrality.
5. Measure dependencies where useful. For edits A and B, the contrast f(AB) - f(A) - f(B) + f(base) describes their interaction in that context. It does not establish every higher-order interaction.
6. Test a response only after a diagnostic is repeatable. Revisits might motivate restarts; interactions might motivate joint edits. Compare against the same applicable total costs and independently evaluate final win rate.

Autocorrelation along legal random walks can describe how scores change across successive moves, but battle noise can falsely suggest ruggedness. Distances also require care: species identifiers are labels, and team permutations should be merged only when the actual battle policy is invariant to them. A best team found so far is not a known global optimum for fitness-distance correlation.

Search trajectory networks are the first read because they can use existing traces. Local-optima networks and feature-based algorithm selection are later steps with additional assumptions and possible sampling costs. The proposed VGC analysis is an inference from those tools, not a result reported by their authors. [Landscape evidence and analysis design](../research_notes/GECCO%20methods%20and%20reading%2020261020/landscape.md).

No new experiment was run and no new evaluation budget was chosen. The ranked scopes describe the 2026 programme. The partially updated 2027 call still needs verification before relying on submission dates, formats or workshop availability.
