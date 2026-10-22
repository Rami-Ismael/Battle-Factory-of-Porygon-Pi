# Landscape analysis for VGC counter search

Checked October 20, 2026. This note recommends reading and research design, not a new experiment or a claim of improved Pokémon win rate. No existing entry for these five papers was found in the current reading list. Grades below follow its current restriction on continuous-only and synthetic toy evidence. Topic interest and transfer grade are separate judgments.

## Recommended additions, in reading order

### 1. Search trajectory networks

**Title:** Search trajectory networks: A tool for analysing and visualising the behaviour of metaheuristics.

**Authors and publication:** Gabriela Ochoa, Katherine M. Malan, Christian Blum. Applied Soft Computing 109, article 107492, 2021. DOI [10.1016/j.asoc.2021.107492](https://doi.org/10.1016/j.asoc.2021.107492).

**Access:** Open [accepted manuscript](https://www.storre.stir.ac.uk/bitstream/1893/32613/1/stns_asoc_2021.pdf), [institutional record](https://www.storre.stir.ac.uk/handle/1893/32613), and [author code](https://github.com/gabro8a/STNs). Manuscript sections 3 and 5 inspected.

**Proposed grade:** MED, first landscape read. It passes the continuous-only exclusion because its combinatorial case selects fixed-size facility subsets for p-median. It compares ant colony optimisation, a biased random-key genetic algorithm, and iterated local search. STNs use logged search states and transitions instead of requiring a separate landscape-sampling algorithm.

**Transfer idea:** Build a graph from actual candidate trajectories to see revisits, separate regions, and differences between generators. This can begin with existing logs if those logs retain candidate identities, scores, iteration, run, and transitions. A graph of the algorithm's visited states is not a map of the entire space. P-median costs are deterministic and cheaply available; battle-score uncertainty remains an additional problem. Logging and graph construction do not themselves improve search or establish win-rate gains.

### 2. Local optima networks, foundational introduction

**Title:** Local Optima Networks: A New Model of Combinatorial Fitness Landscapes.

**Authors and publication:** Gabriela Ochoa, Sébastien Verel, Fabio Daolio, Marco Tomassini. Chapter in Recent Advances in the Theory and Application of Fitness Landscapes, 2014, pages 233 to 262. DOI [10.1007/978-3-642-41888-4_9](https://doi.org/10.1007/978-3-642-41888-4_9). [Author manuscript, arXiv:1402.2959](https://arxiv.org/abs/1402.2959).

**Access:** Open author manuscript. Metadata and abstract verified. This is a chapter, not a GECCO paper.

**Proposed grade:** MED, foundation. It introduces networks whose nodes are local optima and whose edges model transitions between basins. Examples include NK landscapes and the quadratic assignment problem.

**Transfer idea:** Understand what a basin, local optimum, and escape transition mean before drawing a team-space network. VGC can support only sampled, uncertainty-qualified claims unless every legal neighbor has been assessed with adequate precision. Local optimisation and basin sampling can consume many battles, so begin with STNs from existing search traces. Do not equate a team that won the latest sample with a certified local optimum.

### 3. Exploratory landscape analysis for mixed variables

**Title:** Exploratory Landscape Analysis for Mixed-Variable Problems.

**Authors and publication:** Raphael Patrick Prager, Heike Trautmann. First published online May 10, 2024. Final issue is IEEE Transactions on Evolutionary Computation 29, issue 4, August 2025, pages 936 to 947. DOI [10.1109/TEVC.2024.3399560](https://doi.org/10.1109/TEVC.2024.3399560). [IEEE record](https://ieeexplore.ieee.org/document/10529111/), [open manuscript, arXiv:2402.16467](https://arxiv.org/html/2402.16467v1).

**Access:** Open author manuscript, including categorical encoding and experimental-method sections. Use 2024 for the first-publication year and explain the 2025 issue in details to avoid duplicate entries.

**Proposed grade:** MED, later than trace analysis. The paper adapts ELA features for continuous, integer, and categorical variables, then studies algorithm selection on mixed-variable hyperparameter tasks. Its benchmarks use YAHPO Gym. This is not a GECCO paper.

**Transfer idea:** A route toward choosing an algorithm using measured problem features. It requires multiple problem instances, complementary algorithms, and independent algorithm-selection validation. High-cardinality categories, conditional legality, and arbitrary species numbering make direct numerical encoding unsafe. Its preprocessing discussion is useful, but the VGC representation, sampling cost, and performance prediction would need validation. Do not copy its reported HPO improvement as a VGC expectation.

### 4. Subfunction structure in local optima networks

**Title:** Subfunction Structure Matters: A New Perspective on Local Optima Networks.

**Authors and publication:** Sarah L. Thomson, Michal W. Przewozniczek. GECCO 2025 main proceedings, pages 85 to 93. DOI [10.1145/3712256.3726426](https://doi.org/10.1145/3712256.3726426). [Open manuscript, arXiv:2504.17799](https://arxiv.org/html/2504.17799v1), [institutional record](https://napier-repository.worktribe.com/output/4237989/subfunction-structure-matters-a-new-perspective-on-local-optima-networks).

**Access:** Open author manuscript inspected. Repository fetch was blocked, but arXiv full text was available.

**Proposed grade:** LOW under the current toy-evidence rule. Keep it as a focused background reference for the user's new landscape interest, not a high-priority implementation donor. Its study uses artificial deceptive functions, NK landscapes, and artificial MAX3SAT instances.

**Transfer idea:** Compare black-box perturbation against changes informed by known or learned variable interactions. The paper makes that relationship concrete. It does not establish that a stochastic VGC win-rate objective decomposes into bounded subfunctions. Existing project epistasis measurements could motivate a linkage hypothesis, but would not prove the required structure.

### 5. Noisy combinatorial landscape networks

**Title:** Local Optima Networks (LONs) and Search Trajectory Networks (STNs) for Noisy Combinatorial Problems.

**Authors and publication:** John Payne, Aishwaryaprajna, David Walker, Edward Keedwell. GECCO 2025 Companion, pages 2010 to 2018. DOI [10.1145/3712255.3734318](https://doi.org/10.1145/3712255.3734318).

**Access:** Title, authors, pages, DOI, and official proceedings abstract verified. Full text was not retrieved. [Official proceedings, search the exact title](https://www.sigevo.org/gecco-2025/toc-companion.html), [DOI registry metadata](https://api.crossref.org/works/10.1145/3712255.3734318).

**Proposed grade:** LOW under the current toy and Gaussian-evidence restriction, with a useful noise-related topic match. The official abstract studies OneMax and Knapsack with additive Gaussian noise using three optimisation algorithms.

**Transfer idea:** Learn how observation noise can change the network one draws from search data. VGC sampled wins have a different observation model and opponent structure. This is an abstract-screened workshop reference, not a verified implementation recommendation or a demonstrated battle-search improvement.

## GECCO tutorial to use as orientation

[Landscape Analysis of Optimization Problems and Algorithms](https://gecco-2026.sigevo.org/Tutorial?itemId=8226), GECCO 2026, Gabriela Ochoa and Katherine Malan. The specific item page identifies these organisers even though the overall tutorial list repeats the title with another co-organiser.

Its scope includes algorithm trajectories, search trajectory networks, Pareto-local-optima networks, and discrete and continuous case studies. For this project's single win-rate objective, STNs and ordinary local optima networks are the immediate ideas. Pareto-local-optima networks become relevant only if a genuinely multiobjective research question is adopted. No recording was played or inspected.

## What a VGC landscape is

A landscape needs all three parts: the legal candidate teams, the legal edits that define neighbors, and the objective. For this project, define the objective as expected win rate under the fixed battle policy and opponent distribution. A finite battle sample estimates that expectation and introduces observation noise. Changing the opponent distribution or battle policy changes the objective and therefore the landscape.

The project already has notes on local sensitivity and higher-order interactions. Those notes describe measured runs, but they do not prove the complete team space is globally rugged or that every edit type behaves alike. A useful next design extends the existing measurements rather than declaring ruggedness from a picture.

## A concrete analysis design

This is a proposal. It assigns no new battle budget and does not run an experiment.

1. **Choose explicit neighborhoods.** Separate one legal move replacement, one item replacement, and one whole-Pokémon-set replacement. Whole-set replacement changes species, moves, ability, item, and stats together, so report it separately. If a legality repair changes additional fields, log those changes instead of calling the result a single-field edit.
2. **Use a representation with justified distances.** Species IDs are labels, not coordinates. Avoid measuring species 100 as twice as far from species 98 as from species 99. Team permutations and move-order permutations need an explicit invariance decision. Do not merge them if the deployed policy's behaviour actually depends on order.
3. **Reuse search traces first.** Build STNs from completed runs, recording canonical candidate identity where justified, score, uncertainty, parent or transition, operator, iteration, policy, opponent schedule, and seed. Report revisits and coverage of visited regions. Such traces are biased toward where the algorithms searched.
4. **Probe local changes with paired controls.** Re-evaluate promising comparisons using the applicable run's fixed opponents, weights, policy, format, and paired randomness. Compare the distribution of score changes by edit type. Equal observed scores mean indistinguishable at the current precision, not proven neutrality.
5. **Measure serial structure and interactions.** Legal random walks support lagged score autocorrelation. Noise can weaken that correlation and falsely suggest ruggedness. For two edits A and B, compare all four teams and compute the interaction f(AB) - f(A) - f(B) + f(base). This measures a context-specific pair interaction, not every higher-order interaction in team building.
6. **Test an algorithmic response.** Repeated visits might motivate a restart or different mutation. Strong interactions might motivate joint edits. These are hypotheses. Compare the response against the fixed baseline under the same applicable battle and elapsed-time budgets, and retain final independent win-rate evaluation. A visually attractive network alone cannot justify an adaptive rule.

The papers above support the analysis tools and their benchmark settings. The VGC applications in this section are proposed transfers.

## Foundation to consult without adding another urgent read

Terry Jones and Stephanie Forrest's 1995 [Fitness Distance Correlation as a Measure of Problem Difficulty for Genetic Algorithms](https://sfi-edu.s3.amazonaws.com/sfi-edu/production/uploads/sfi-com/dev/uploads/filer/bf/eb/bfeb9cb0-100f-44d9-a35f-e95243dba350/95-02-022.pdf) is an open Santa Fe Institute author working paper. It relates fitness to distance from known optima. The global best VGC team is unknown, so distance to the best team found so far is only a reference-based diagnostic. It is not the original measure with a certified global optimum. Do not infer general search difficulty from one correlation coefficient.

## Suggested reading-list action

Add STNs 2021, the LON introduction, and mixed-variable ELA as MED landscape methods, in that reading order. Keep the two GECCO 2025 papers as LOW references in this memo, outside the active additions. The first practical action is trace analysis; online algorithm selection and learned interaction-aware mutations follow only after useful, repeatable landscape features have been established. No citations counts were checked, so do not fill unknown counts with zero.
