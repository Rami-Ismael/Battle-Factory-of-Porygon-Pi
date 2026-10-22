# GECCO concepts for counter-team search

Checked October 20, 2026. Definitions below use primary papers and official programme material. VGC designs and rankings are proposals, not measured improvements. The objective remains independently evaluated battle win rate under the established opponent distribution and battle policy.

## Constructive heuristics build a solution

A constructive heuristic starts with an empty or partial solution and repeatedly chooses a component to add until it has a complete solution. The choices use a rule intended to produce useful solutions. The rule may be deterministic or randomized. In GRASP, the constructor chooses from a restricted list of promising components, updating the greedy scores as the partial solution changes. A separate local-search phase then improves the completed solution. [Feo and Resende, original GRASP paper, sections 1 and 2](https://mauricio.resende.info/doc/gtut.pdf).

A possible VGC constructor could choose a core pair, add a Pokémon that addresses an uncovered matchup, then fill moves and items while preserving legality. This is a proposed design. A Pokémon's value depends on its teammates, so a cheap partial-team score needs validation. A partial team does not have a unique six-Pokémon win rate until a completion rule is specified.

Construction, repair, and local improvement are different operations. Construction fills a partial solution. Repair restores a constraint after an invalid proposal. Local improvement changes a complete candidate to try to improve its objective. A legality checker alone does not choose good components. Generating a whole team in one step is not an incremental constructive heuristic merely because the result is legal.

## Multi-fidelity concerns the cost and accuracy of evaluation

Multi-fidelity methods combine cheaper approximations with an expensive reference evaluator of the same quantity. The reference remains involved to establish accuracy. A learned surrogate can be one low-fidelity evaluator. Model management decides when to use or correct an approximation and when to pay for the reference evaluation. [Peherstorfer, Willcox and Gunzburger, 2018 survey](https://arxiv.org/abs/1806.10761).

For this project, let f(T) denote expected battle win rate of complete legal team T under the fixed opponent distribution, policy, simulator, and rules. Distinguish the following designs.

| Change | What it does | What to validate |
| --- | --- | --- |
| Run fewer battles with the same sampling protocol | Produces a less precise Monte Carlo estimate of the same f(T). This is evaluation-effort allocation; calling it fidelity is a broader convention. | Selection can favour lucky estimates. Use independently evaluated finalists and account for repeated battles. |
| Predict f(T) with a surrogate | Substitutes a cheap model for direct simulation during search. | Test predictions on unseen candidate teams and confirm selected candidates in the reference simulator. |
| Replace the battle policy with a cheaper policy | Changes the evaluator and usually the expected win-rate function. | Establish whether it preserves useful candidate rankings, then confirm using the target policy. |
| Evaluate a fixed smaller opponent subset | Usually changes the objective's opponent distribution. | Sampling and weighting must preserve the intended expectation if this is to estimate the same target. Otherwise label it a proxy objective. |
| Simplify battle rules or truncate a battle | Creates a proxy simulator or a proxy outcome. | Show that the proxy is useful for the full-rule win-rate target. A remaining-health score is not a battle win by definition. |

These are project-specific implications of the fixed target definition, not performance results from the survey. The crucial distinction is reduced random error versus systematic approximation error. More repetitions can reduce the first; they do not automatically remove bias from a weaker policy or altered rules.

## Multi-resolution needs an explicit definition

The GECH scope lists multi-resolution objective evaluation but supplies no single technical definition. Do not assume it is a synonym for multi-fidelity. Resolution concerns the granularity at which a model or candidate is represented or evaluated. A coarser model can also be lower fidelity, but that relationship needs to be demonstrated. [Official GECH scope](https://gecco-2026.sigevo.org/Track?itemId=57).

Two hypothetical VGC interpretations are worth separating:

- Search first over archetypes or species, then expand promising choices into full moves, items, abilities, and EVs. This changes representation resolution. A coarse species choice needs a specified completion or aggregation rule before it has a score.
- Evaluate a coarse battle abstraction, then evaluate a finer simulator. This changes evaluation resolution and may introduce bias. Pokémon's discrete mechanics can make small details decisive, so a coarser simulator needs evidence that its ranking is useful.

Fewer battle repetitions is statistical precision, not a finer description of a battle. A coarse team representation is not automatically a cheaper objective evaluation. The research question should name exactly what changes, what stays fixed, and how a coarse result maps to the target f(T).

## What the hybrid abbreviations mean

EA means evolutionary algorithm. A hybrid combines algorithmic components with distinct jobs. GECH explicitly includes all the combinations below, while rejecting hybrids justified only by superficial metaphors. [Official GECH scope](https://gecco-2026.sigevo.org/Track?itemId=57).

| Hybrid | Meaning | Hypothetical use in this project | Main cost or mismatch |
| --- | --- | --- | --- |
| EA + EA, including a meta-EA | One evolutionary process can optimise the settings or structure of another. Other combinations can exchange candidates between evolutionary processes. | Outer search chooses mutation rates or operator mixtures; inner runs search for teams. | A meta-level evaluation can require an entire team-search run. Tuning and testing need separate tasks and random seeds. |
| EA + PSO | Particle swarm optimisation adds movement influenced by a particle's previous best and shared best positions. | Use swarm search for a small continuous proposal-parameter vector, while the team generator still handles legal discrete teams. | Ordinary continuous movement does not directly define a legal item, move, or species change. Discrete adaptations need their own justification. |
| EA + ACO | Ant colony optimisation constructs candidates incrementally using learned pheromone preferences and heuristic information. | An ACO constructor assembles teams; an evolutionary population recombines or refines them. | The construction graph, partial-team scores, and ordering conventions must represent team interactions and legality. |
| EA + LS, often called memetic | Local search improves selected complete candidates inside population search. | After proposing a team, try bounded legal item, move, or Pokémon replacements. Feed improved candidates back into the population or distribution update. | Every proposed improvement consumes scoring effort. Search can overfit noisy local differences. |
| EA + fuzzy methods | Fuzzy sets represent graded membership in concepts, enabling rule-based control or modelling. | A controller uses graded notions of low diversity or stagnation to adjust mutation strength. | The membership functions and rules add design choices. Fuzzy membership is not the probability of a random battle event. |

Sources for the mechanisms: [Grefenstette, meta-level GA parameter optimisation](https://ieeexplore.ieee.org/document/4075583); [Kennedy and Eberhart, PSO publisher record](https://doi.org/10.1109/ICNN.1995.488968); [Dorigo's author-written ACO description](https://www.scholarpedia.org/article/Ant_colony_optimization); [Amaya, Cotta and Fernández-Leiva, cross-entropy memetic algorithm](https://doi.org/10.1080/18756891.2013.792542); [Zadeh's original fuzzy-sets paper](https://doi.org/10.1016/S0019-9958(65)90241-X).

Within this list, my current project priority is EA + LS first, EA + EA second, EA + ACO third, then EA + PSO and EA + fuzzy methods. This ranks the clarity of the next test and the amount of extra machinery required. It is not an empirical ranking of VGC win rate. Constructive generation is a separate useful comparison when building teams incrementally addresses a concrete legality or proposal-quality problem.

The smallest informative memetic comparison would add a bounded local-refinement step to the existing search, with all scoring charged to the applicable total budget. Compare it with spending that effort on more ordinary proposals. Confirm finalists independently. A fixed local-refinement budget does not prove that a candidate is a local optimum across every legal neighbour.

## Reading-list recommendations

The following titles and authors were not found in the current reading list by title and author searches. Citation counts were not assessed. Grades follow the project's rule that continuous-only or toy results do not merit a direct-reuse MED grade.

### Add first: cross-entropy plus local improvement

**Cross entropy-based memetic algorithms: An application study over the tool switching problem.** Jhon Edgar Amaya, Carlos Cotta, and Antonio J. Fernández-Leiva. 2013. International Journal of Computational Intelligence Systems, 6(3), 559 to 584. DOI [10.1080/18756891.2013.792542](https://doi.org/10.1080/18756891.2013.792542). [Open publisher PDF](https://link.springer.com/content/pdf/10.1080/18756891.2013.792542.pdf).

**MED, method donor.** It combines CE probability models with local search on a discrete permutation problem. Section 3 exposes a probability of local refinement and an evaluation cap, useful controls for a bounded ablation. Its tool-switching objective differs from noisy, expensive battle win rate, so its performance results do not transfer directly. Read this before a broad memetic-algorithms survey if the decision is whether to refine CEM-generated teams.

### Add as a short foundation: construction followed by improvement

**Greedy Randomized Adaptive Search Procedures.** Thomas A. Feo and Mauricio G. C. Resende. 1995. Journal of Global Optimization, volume 6, commonly indexed as pages 109 to 133; the author's PDF header prints 109 to 134. DOI [10.1007/BF01096763](https://doi.org/10.1007/BF01096763). [Author-hosted full text](https://mauricio.resende.info/doc/gtut.pdf).

**MED, foundational reference.** Sections 1 and 2 distinguish incremental construction, a restricted candidate list, and improvement in a chosen neighbourhood. Worked examples concern set covering and maximum independent set. A reusable design pattern, with no VGC result or proof that partial-team heuristics are informative. Treat this as a targeted foundation, not a reason to pause the current experiment queue.

### Keep as concept sources, not extra active-queue entries

- **Survey of multifidelity methods in uncertainty propagation, inference, and optimization.** Benjamin Peherstorfer, Karen Willcox, and Max Gunzburger. 2018. SIAM Review 60(3), 550 to 591. DOI [10.1137/16M1082469](https://doi.org/10.1137/16M1082469). [Author arXiv preprint](https://arxiv.org/abs/1806.10761). LOW for direct reuse under the reading-list rule, useful for terminology and model-management principles. Numerical-science setting; no categorical team-search result.
- **An Accelerated Introduction to Memetic Algorithms.** Pablo Moscato and Carlos Cotta. 2019 book edition, first online September 2018. Handbook of Metaheuristics, pages 275 to 309. DOI [10.1007/978-3-319-91086-4_9](https://doi.org/10.1007/978-3-319-91086-4_9). Metadata and abstract verified from the publisher and author bibliography. The author PDF link was indexed but failed to retrieve in this review, and the publisher full chapter requires access. Redundant as an immediate addition when the concrete CE-plus-local-search paper is available.

No new battle experiment was run, no universal evaluation budget was chosen, and no result here establishes that any proposed hybrid improves this project's held-out battle win rate.
