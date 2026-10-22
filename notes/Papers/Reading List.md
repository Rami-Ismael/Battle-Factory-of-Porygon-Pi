---
created_at: 2026-08-24
updated_at: '2026-10-21'
type: reading-list
tags:
- reading-list
- vgc
- combinatorial-search
- reinforcement-learning
ordering: By project topic and method subtopic; current-stage relevance takes precedence over grades
sequencing_rule: Current search-loop readings first; HIGH before MED only within the same stage; fine-tuning and reward integration deferred
downgrade_rule: 'Continuous-only or toy experiments (synthetic test functions, robot or control weights, images, Gaussian-setting theory) grade LOW: no categorical, team-like search space. Inferred from the SPARROW and HADES rejections, 2026-10-16.'
spigm_accepted: 2
citations_source: 'Semantic Scholar citation counts, pulled 2026-10-17'
history: '[[Reading List — notes and decision history]]'
---

# Reading List

Browse by topic and subtopic. **Current-stage relevance comes before HIGH/MED grades:** prioritize search-loop design, baselines and evaluation before fine-tuning-specific methods or diagnostics. Fine-tuning has not started (owner, 2026-10-13). Within the same stage, read **HIGH** before **MED**, then implement what answers the current project question. Papers without a grade remain ungraded. Fine-tuning and reward-integration readings are deferred until those experiments are planned; their grades do not move them ahead of current search readings.

**SPIGM: only A Tale of Two Temperatures and Re-evaluating Confidence Remasking are accepted, both MED.**

The owner's GECCO follow-up on 2026-10-20 adds five MED references for landscape analysis, constructive heuristics, and cross-entropy search with local improvement. Start with search trajectory networks for diagnostics and the cross-entropy memetic paper for a bounded hybrid comparison. The other three are foundations or later methodology. This topic-specific order does not replace the current experiment queue. See [the ranked GECCO methods guide](../reports/GECCO%20methods%20and%20reading%2020261020.md).

[[#Textbooks and foundations|Textbooks]] · [[#Pokémon teams and benchmarks|Pokémon]] · [[#Team building in other games|Other games]] · [[#Search and black-box optimization|Search]] · [[#Diffusion models and sampling|Diffusion]] · [[#Retraining, diversity and model collapse|Retraining]] · [[#Surrogates and active learning|Surrogates]] · [[#Legality and constraint programming|Constraints]] · [[#Automated heuristic design|Heuristic design]] · [[#Battle policies and self-play|Battle policies]] · [[#Training dynamics and engineering|Training]] · [[#Looped transformers|Looped transformers]]

## Textbooks and foundations

Selected by the owner on 2026-10-21 to learn search vocabulary. Start with Sean Luke, then use Eiben and Smith for more detail. These are foundation readings alongside the current search papers.

| Book | Authors and edition | Access | Read first | Reading guide |
| --- | --- | --- | --- | --- |
| [Essentials of Metaheuristics](https://people.cs.gmu.edu/~sean/book/metaheuristics/Essentials.pdf) | Sean Luke. Second print edition, 2013; online version 2.4 dated January 2026. | Free author copy. | First sitting: sections 0.1 and 2.1. Then chapter 3, section 3.2.1, and section 3.3.4 for memetic algorithms / local search. | [[Essentials of Metaheuristics - reading guide]] |
| [Introduction to Evolutionary Computing](https://link.springer.com/book/10.1007/978-3-662-44874-8) | A. E. Eiben and J. E. Smith. Second edition, 2015. | Purchase or institutional library access. | After Luke: chapter 3, then chapters 4 and 5. Keep chapters 10, 13, and 11 for local search, legality, and noisy evaluation when needed. | [[Introduction to Evolutionary Computing - reading guide]] |

## Pokémon teams and benchmarks

### Team construction and metagame analysis

| Paper | Year | Citations | Grade | Venue and details | Notes |
| --- | --- | --- | --- | --- | --- |
| 🔒 [An Adversarial Approach for Automated Pokémon Team Building and Metagame Balance](https://doi.org/10.1109/tg.2023.3273157) | 2023 | 8 | MED |  |  |
| [Pokémon GO Team Optimization](https://journals-sol.sbc.org.br/index.php/jis/article/view/6773) | 2026 | 0 | LOW |  |  |
| [A Framework for Predicting the Impact of Game Balance Changes Through Meta Discovery](https://arxiv.org/abs/2409.07340) | 2024 | 0 | LOW |  |  |
| [From Rules to Nash Equilibria](https://arxiv.org/abs/2607.08692) | 2026 | 0 | LOW |  |  |

### Battle agents

| Paper | Year | Citations | Grade | Venue and details | Notes |
| --- | --- | --- | --- | --- | --- |
| [[Human-Level Competitive Pokémon via Scalable Offline Reinforcement Learning with Transformers]] | 2025 | 6 | MED |  |  |
| [PokéChamp: An Expert-level Minimax Language Agent](https://arxiv.org/abs/2503.04094) | 2025 | 17 | LOW |  |  |
| [PokéLLMon: A Human-Parity Agent for Pokémon Battles with Large Language Models](https://arxiv.org/abs/2402.01118) | 2024 | 26 | LOW |  |  |

### Benchmarks and competitions

| Paper | Year | Citations | Grade | Venue and details | Notes |
| --- | --- | --- | --- | --- | --- |
| [[VGC-Bench Towards Mastering Diverse Team Strategies in Competitive Pokémon]] | 2025 | 3 | MED |  |  |
| [The PokeAgent Challenge: Competitive and Long-Context Learning at Scale](https://www.semanticscholar.org/paper/80c6fcde8b5e8190a2e970db128eaf095887a46b) | 2026 | 13 | MED |  |  |

## Team building in other games

### Quality diversity and deck-space exploration

| Paper | Year | Citations | Grade | Venue and details | Notes |
| --- | --- | --- | --- | --- | --- |
| [Deep Surrogate Assisted MAP-Elites for Automated Hearthstone Deckbuilding](https://www.semanticscholar.org/paper/e46811df6731df51fb5c209dea7b835d0cd734da) | 2021 | 43 | HIGH | direction paper — the loop itself: generate decks → battle vs fixed opponents → refit → repeat · [[Reading List — notes and decision history\|Reading notes]] |  |
| [Mapping Hearthstone Deck Spaces through MAP-Elites with Sliding Boundaries](https://arxiv.org/abs/1904.10656) | 2019 | 84 | HIGH | 🟢 GECCO 2019 · [Alternate source](https://www.semanticscholar.org/paper/9eb4d5e751b83a2afb240b93c0c2b1ea025ac14f) · [[Reading List — notes and decision history\|Reading notes]] |  |
| [CabbageCat's Blogs — MAP-Elites, CMA-ME, Differentiable QD step-by-step](https://szhaovas.github.io/2022-09-15-me/) | 2022 |  |  | 🟢 grad-researcher QD tutorial series (MAP-Elites → CMA-ME → Differentiable QD); read before/with the MAP-Elites Hearthstone papers |  |
| 🔒 [Exploring the Hearthstone Deck Space](https://dl.acm.org/doi/10.1145/3235765.3235791) | 2018 | 30 | MED | 🟢 [open copy: Internet Archive capture of the ACM PDF](https://web.archive.org/web/20200602172549/https://dl.acm.org/doi/pdf/10.1145/3235765.3235791?download=true) |  |

### Evolutionary deck building and playtesting

| Paper | Year | Citations | Grade | Venue and details | Notes |
| --- | --- | --- | --- | --- | --- |
| 🔒 [Automated playtesting in collectible card games using evolutionary algorithms](https://doi.org/10.1016/j.knosys.2018.04.030) | 2018 | 46 | MED |  |  |
| [Evolving the Hearthstone Meta](https://arxiv.org/abs/1907.01623) | 2019 | 35 | LOW |  |  |
| [Evolutionary Arena Deckbuilding with Active Genes](https://arxiv.org/abs/2001.01326) | 2020 | 16 | LOW |  |  |

### Drafting and deck recommendation

| Paper | Year | Citations | Grade | Venue and details | Notes |
| --- | --- | --- | --- | --- | --- |
| [Q-DeckRec: A Fast Deck Recommendation System for Collectible Card Games](https://www.semanticscholar.org/paper/ca1e69d0d5a74a05ee390ea6aa0129965a83950b) | 2018 | 42 | MED |  |  |
| [Drafting in Collectible Card Games via Reinforcement Learning](https://www.sbgames.org/proceedings2020/ComputacaoFull/209690.pdf) | 2020 | 9 | LOW |  |  |

### Game prototyping and competition reports

| Paper | Year | Citations | Grade | Venue and details | Notes |
| --- | --- | --- | --- | --- | --- |
| [Cardiverse: Harnessing LLMs for Novel Card Game Prototyping](https://www.semanticscholar.org/paper/6590ba146e330de09ce7798164cc1ddeecfc0b17) | 2025 | 8 | LOW |  |  |
| [Summarizing Strategy Card Game AI Competition](https://www.semanticscholar.org/paper/551a4a4d231e24bacbd7a1285e8f44dd553e5b3f) | 2023 | 16 | LOW | · [[Reading List — notes and decision history\|Reading notes]] |  |

## Search and black-box optimization

### Classical search and set functions

| Paper                                                                                                                                                                                     | Year | Citations | Grade    | Venue and details                                                                                                                                                                                                                                    | Notes |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---- | --------- | -------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----- |
| [Optimistic Tree Searches for Combinatorial Black-Box Optimization](https://proceedings.neurips.cc/paper_files/paper/2022/hash/d6099a36f6c1720438de00c366aa1737-Abstract-Conference.html) | 2022 | 3         | ungraded | NeurIPS 2022 — first read among the DIFUSCO references for search framing (§§2–3); discrete black-box search, with additional work needed for noisy battle evaluations. · [[DIFUSCO references - relevance to counter-team search\|Relevance notes]] |       |
| [[Simulating Annealing]]                                                                                                                                                                  |      |           |          |                                                                                                                                                                                                                                                      |       |
| [[Submodular Function]]                                                                                                                                                                   |      |           |          |                                                                                                                                                                                                                                                      |       |
| [[Local Search]]                                                                                                                                                                          |      |           |          | or Direct search                                                                                                                                                                                                                                     |       |

### Neural combinatorial optimization

| Paper                                                                                                                                                                                              | Year | Citations | Grade | Venue and details                        | Notes |
| -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---- | --------- | ----- | ---------------------------------------- | ----- |
| [A Denoising Diffusion-Based Evolutionary Algorithm Framework: Application to the Maximum Independent Set Problem](https://www.semanticscholar.org/paper/9b21ee2ac4fde7868f485774eda250973e808694) | 2025 | 2         | MED   | 🟢 arXiv preprint, no venue listed       |       |
| [COExpander: Adaptive Solution Expansion for Combinatorial Optimization](https://proceedings.mlr.press/v267/ma25r.html)                                                                            | 2025 | 16        | LOW   | ( consider this background information ) |       |
| [Neural Large Neighborhood Search for the Capacitated Vehicle Routing Problem](https://arxiv.org/abs/1911.09539)                                                                                   | 2019 | 214       | LOW   |                                          |       |
| [DIFUSCO: Graph-based Diffusion Solvers for Combinatorial Optimization](https://www.semanticscholar.org/paper/5f90d43e6ece5c6ee6e8186e4b57d46c85377713)                                            | 2023 | 363       | LOW   |                                          |       |
| [A Diffusion Model Framework for Unsupervised Neural Combinatorial Optimization](https://www.semanticscholar.org/paper/4a5ba3075d61f6258059dc311f761eb85848b22d)                                   | 2024 | 80        | LOW   |                                          |       |
| [Tackling Prevalent Conditions in Unsupervised Combinatorial Optimization](https://www.semanticscholar.org/paper/f7daa82747a13314c42a1df0c8eb2d4958ac27d6)                                         | 2024 | 13        | LOW   |                                          |       |
| [From Distribution Learning in Training to Gradient Search in Testing for Combinatorial Optimization](https://www.semanticscholar.org/paper/b933fb2ab8cc4704953a6e97576a921eee56dd76)              | 2023 | 146       | LOW   | 🟢 NeurIPS 2023                          |       |

### Adaptive search distributions and the cross-entropy method (Reading this now)

| Paper | Year | Citations | Grade | Venue and details | Notes |
| --- | --- | --- | --- | --- | --- |
| [Design by adaptive sampling](https://arxiv.org/abs/1810.03714) | 2018 | 79 | MED | It think adaptive sampling can be used for something else, also take about the reiventing the cross entropy method, this should be higher than on the list compare to active search method |  |
| [Addressing Combinatorial Optimization with Estimation of Distribution Algorithms Based on Diffusion Models](https://doi.org/10.1145/3795095.3805121) | 2026 | 0 | MED | 🟢 GECCO 2026 | Acceptance, authors and ECOM track verified on 2026-10-20; full text was not retrieved. Method and performance remain unverified in this screening. |

### Generative optimization and learned representations

| Paper                                                                                                                                                               | Year | Citations | Grade         | Venue and details                                                                                                                                                                                                                                  | Notes                                                                                                                                                                                                                                                                          |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---- | --- | ------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| [Trust-Region Noise Search for Black-Box Alignment of Diffusion and Flow Models](https://arxiv.org/abs/2603.14504)                                                  | 2026 | 0 | MED           |                                                                                                                                                                                                                                                    | Paper it reference<br>↳ Scalable Global Optimization via Local Bayesian Optimization<br>I liked the SMC approach that might be a useful direction<br>↳ KF sterring<br>↳ DAS, because it require reward i should not use then instead if I plan to add reward then KF won ahead |
| [Regret Analysis of Guided Diffusion for Black-Box Optimization over Structured Inputs](Ahttps://arxiv.org/abs/2605.10385)                                          | 2026 | 1 | MED           |                                                                                                                                                                                                                                                    | I wished it has good diagram it does not any good on<br>I don't know what is an acquistion score is<br>I need to find some youtube vide about regrest anaylsis                                                                                                                 |
| [Diffusion Large Language Models for Black-Box Optimization](https://arxiv.org/html/2601.14446v1)                                                                   | 2026 | 2 | ungraded      | 2026, arXiv preprint — combines a prompted diffusion LLM with a Gaussian-process predictor and masked diffusion tree search over partially masked candidates; compares against an autoregressive LLM optimizer and conventional diffusion methods. | I feel like the section in the related work section about llm for bbo will be great for why I choose an llm                                                                                                                                                                    |
| [Large Language Models as Optimizers](https://arxiv.org/abs/2309.03409)                                                                                             | 2023 | 1,215 | MED           | 🟢 ICLR 2024 ( It just propose and just create meta prompt afterward has a lot of citation ) it should be like the baseline                                                                                                                        |                                                                                                                                                                                                                                                                                |
| [Language Model Crossover: Variation through Few-Shot Prompting](https://arxiv.org/abs/2302.12170)                                                                  | 2023 | 186 | MED           | 🟢 ACM TELO 2024                                                                                                                                                                                                                                   | Read this next Combinatorial Optimization for All: Using LLMs to Aid Non-Experts in Improving Optimization Algorithms<br>↳ Read this next pyCombinatorial - A library to solve TSP (Travelling Salesman Problem) using Exact Algorithms, Heuristics and Metaheuristics         |
| Steering Generative Models with Experimental Data for Protein Fitness Optimization                                                                                  | 2025 | 13 | MED           |                                                                                                                                                                                                                                                    | For through this method of guidance Decoupled Annealing Posterior Sampling (DAPS)                                                                                                                                                                                              |
| [Generative Bayesian Optimization: Generative Models as Acquisition Functions](https://arxiv.org/abs/2510.25240)                                                    | 2025 | 4 | HIGH          | 🟢 ICLR 2026                                                                                                                                                                                                                                       | go through this paper                                                                                                                                                                                                                                                          |
| [From Generation to Discovery: Diffusion Mutation Kernels for Circuit and Physical Design](https://arxiv.org/abs/2608.27649)                                        | 2026 | 0 | HIGH          | 🟢 arXiv preprint, no venue listed                                                                                                                                                                                                                 |                                                                                                                                                                                                                                                                                |
| [[Active Flow Matching]]                                                                                                                                            | 2026 | 0 | MED           | 🟢 arXiv preprint, no venue listed                                                                                                                                                                                                                 |                                                                                                                                                                                                                                                                                |
| [Posterior Inference with Diffusion Models for High-dimensional Black-box Optimization](https://arxiv.org/abs/2502.16824)                                           | 2025 | 14 | MED           | 🟢 ICML 2025 Readint this right now                                                                                                                                                                                                                | I think the major problem with my method i doing a cross netopy nethod want to do forward prediction but i struggle with diverse set of dataset at allzz                                                                                                                       |
| [Diffusion Models for Black-Box Optimization](https://proceedings.mlr.press/v202/krishnamoorthy23a.html)                                                            | 2023 | 112 | MED           | 🟢 ICML 2023                                                                                                                                                                                                                                       |                                                                                                                                                                                                                                                                                |
| [Deep Learning-Based Operators for Evolutionary Algorithms](https://arxiv.org/abs/2407.10477)                                                                       | 2024 | 4 | MED           | 🟢 GPTP XXI 2024                                                                                                                                                                                                                                   |                                                                                                                                                                                                                                                                                |
| [Evolutionary Discovery of Bivariate Bicycle Codes with LLM-Guided Search](https://arxiv.org/abs/2606.02418)                                                        | 2026 | 11 | MED           | 🟢 arXiv preprint, no venue listed                                                                                                                                                                                                                 |                                                                                                                                                                                                                                                                                |
| [Illuminating Mario Scenes in the Latent Space of a Generative Adversarial Network](https://www.semanticscholar.org/paper/aa79cd3983e6cfd877a830eb3e55a8ded425fe28) | 2020 | 105 | LOW           | 🟢 AAAI 2021                                                                                                                                                                                                                                       |                                                                                                                                                                                                                                                                                |
| [Diffusion Models are Evolutionary Algorithms](https://www.semanticscholar.org/paper/cfa4ef02c7ca657506768fb53075caecc1b7161c)                                      | 2024 | 34 | LOW (was MED) | 🟢 ICLR 2025                                                                                                                                                                                                                                       |                                                                                                                                                                                                                                                                                |
| [Discovering Representations for Black-box Optimization](https://arxiv.org/abs/2003.04389)                                                                          | 2020 | 7 | LOW (was MED) | 🟢 GECCO 2020                                                                                                                                                                                                                                      | Interesting example that use map elit                                                                                                                                                                                                                                          |
| [Evolution through Large Models](https://arxiv.org/abs/2206.08896)                                                                                                  | 2022 | 183 | LOW           | 🟢 arXiv preprint, no venue listed                                                                                                                                                                                                                 |                                                                                                                                                                                                                                                                                |

- [ ] [[Diffusion-BBO Diffusion-Based Inverse Modeling for Online Black-Box Optimization]]

### Noisy evaluations and Bayesian optimization

| Paper | Year | Citations | Grade | Venue and details | Notes |
| --- | --- | --- | --- | --- | --- |
| [SMAC3: A Versatile Bayesian Optimization Package for Hyperparameter Optimization](https://arxiv.org/abs/2109.09831) | 2021 | 580 | ungraded | JMLR 2022 — practical model-based search baseline to adapt to categorical team choices and noisy battle evaluations; configure repeated evaluations and encode team legality. · [Software documentation](https://automl.github.io/SMAC3/latest/3_getting_started/) |  |
| [Bayesian Optimization of Combinatorial Structures](https://proceedings.mlr.press/v80/baptista18a.html) | 2018 | 172 | ungraded | Ricardo Baptista and Matthias Poloczek, ICML 2018 — owner-selected (2026-09-28): vocabulary and methodology for expensive combinatorial black-box search; study how the model learns interactions between discrete choices from limited evaluations. · [PDF](https://proceedings.mlr.press/v80/baptista18a/baptista18a.pdf) |  |
| [Think Global and Act Local: Bayesian Optimisation over High-Dimensional Categorical and Mixed Search Spaces](https://proceedings.mlr.press/v139/wan21b.html) | 2021 | 94 | ungraded | Xingchen Wan et al., ICML 2021 — owner-selected (2026-09-28): local optimisation and tailored kernels for categorical and mixed search spaces; a reference for representing team-building choices and defining useful search neighborhoods. · [PDF](https://proceedings.mlr.press/v139/wan21b/wan21b.pdf) |  |
| [Using Common Random Numbers for Simulation-based Planning with Rollouts](https://rlj.cs.umass.edu/2026/papers/Paper52.pdf) | 2026 | 5 | MED |  |  |
| 🔒 [Toward Reliable Uncertainty Quantification in Surrogate-Assisted Evolutionary Algorithms via Temporal Conformal Prediction](https://doi.org/10.1007/978-3-032-23604-3_24) | 2026 | 0 | LOW |  |  |
| [Scaling Gaussian Process Optimization by Evaluating a Few Unique Candidates Multiple Times](https://proceedings.mlr.press/v162/calandriello22a/calandriello22a.pdf) | 2022 | 20 | LOW | 🟢 ICML 2022 |  |
| [Bayesian Optimization for Intrinsically Noisy Response Surfaces](https://arxiv.org/abs/2503.00327) | 2025 | 0 | LOW | 🟢 arXiv preprint, no venue listed |  |
| [Efficient Exploration in Binary and Preferential Bayesian Optimization](https://arxiv.org/abs/2110.09361) | 2021 | 9 | LOW (was MED) | 🟢 arXiv preprint, no venue listed |  |

### Optimization benchmarks and validation

| Paper | Year | Citations | Grade | Venue and details | Notes |
| --- | --- | --- | --- | --- | --- |
| [Design-Bench: Benchmarks for Data-Driven Offline Model-Based Optimization](https://www.semanticscholar.org/paper/f23222ad51b9b1f4c1addcf848906db7698ccbff) | 2022 | 141 | ungraded | ICML 2022 — owner-selected focus (2026-09-13): test whether stronger search improves measured battle performance or exploits surrogate errors. Offline benchmark reference; Pokémon experiments would need fresh battle validation. · [Published paper](https://proceedings.mlr.press/v162/trabucco22a.html) |  |
| [What's Wrong with Deep Learning in Tree Search for Combinatorial Optimization](https://arxiv.org/abs/2201.10494) | 2022 | not checked | MED | 🟢 ICLR 2022. First selective read from DDEA's references: §§3.1–3.2 test learned versus random guidance while separating reductions and local search. Adapt the ablation to fixed battle budgets; deterministic MIS scores do not address battle noise. |  |
| [Machine Learning for Combinatorial Optimization: A Methodological Tour d'Horizon](https://arxiv.org/abs/1811.06128) | 2021 | not checked | MED | 🟢 EJOR 290, 405–421. Selective reading: §§3.2, 4.2, 4.3 and 4.6 distinguish learned-operator fit, actual search performance, held-out generalization and offline cost. DOI 10.1016/j.ejor.2020.07.063. |  |
| [Unify ML4TSP: Drawing Methodological Principles for TSP and Beyond from Streamlined Design Space of Learning and Search](https://proceedings.iclr.cc/paper_files/paper/2025/file/a3dc6e903082902d8e916bb9fccbfcbc-Paper-Conference.pdf) | 2025 | 14 | LOW |  |  |
| [What Ails Generative Structure-based Drug Design: Expressivity is Too Little or Too Much?](https://arxiv.org/abs/2408.06050) | 2024 | 0 | LOW | 🟢 AISTATS 2025 |  |
| [Exploring validation metrics for offline model-based optimisation with diffusion models](https://arxiv.org/abs/2211.10747) | 2022 | 1 | LOW (was MED) | 🟢 Transactions on Machine Learning Research 2024 |  |

### Constructive and memetic search

| Paper | Year | Citations | Grade | Venue and details | Notes |
| --- | --- | --- | --- | --- | --- |
| [Cross entropy-based memetic algorithms: An application study over the tool switching problem](https://doi.org/10.1080/18756891.2013.792542) | 2013 | not checked | MED | Jhon Edgar Amaya, Carlos Cotta and Antonio J. Fernández-Leiva. International Journal of Computational Intelligence Systems 6(3). [Publisher full text](https://link.springer.com/content/pdf/10.1080/18756891.2013.792542.pdf). | First hybrid read from the 2026-10-20 GECCO follow-up. Section 3 combines cross-entropy probability models with bounded local refinement. Deterministic permutation scheduling differs from noisy battle win rate; charge refinement to the same comparison budget. |
| [Greedy Randomized Adaptive Search Procedures](https://doi.org/10.1007/BF01096763) | 1995 | not checked | MED | Thomas A. Feo and Mauricio G. C. Resende. Journal of Global Optimization 6. [Author full text](https://mauricio.resende.info/doc/gtut.pdf). | Foundation, sections 1 and 2: incremental randomized construction followed by improvement in a chosen neighborhood. Set-covering and maximum-independent-set examples. A partial-team scoring heuristic still needs validation. |

### Landscape analysis

| Paper | Year | Citations | Grade | Venue and details | Notes |
| --- | --- | --- | --- | --- | --- |
| [Search trajectory networks: A tool for analysing and visualising the behaviour of metaheuristics](https://doi.org/10.1016/j.asoc.2021.107492) | 2021 | not checked | MED | Gabriela Ochoa, Katherine M. Malan and Christian Blum. Applied Soft Computing 109, 107492. [Accepted manuscript](https://www.storre.stir.ac.uk/bitstream/1893/32613/1/stns_asoc_2021.pdf). [Author code](https://github.com/gabro8a/STNs). | First landscape read. Sections 3 and 5 use actual search traces and a fixed-cardinality p-median case. Reuse candidate identities and transitions where logged; visited trajectories do not map the entire landscape or prove a win-rate improvement. |
| [Local Optima Networks: A New Model of Combinatorial Fitness Landscapes](https://arxiv.org/abs/1402.2959) | 2014 | not checked | MED | Gabriela Ochoa, Sébastien Verel, Fabio Daolio and Marco Tomassini. Chapter in Recent Advances in the Theory and Application of Fitness Landscapes. [Publisher DOI](https://doi.org/10.1007/978-3-642-41888-4_9). | Foundation for optima, basins and escape transitions, with quadratic-assignment and NK examples. Metadata and abstract verified. VGC needs an explicit legal neighborhood and uncertainty-qualified local-optimum claims; basin sampling can consume many battles. |
| [Exploratory Landscape Analysis for Mixed-Variable Problems](https://arxiv.org/abs/2402.16467) | 2024 | not checked | MED | Raphael Patrick Prager and Heike Trautmann. IEEE Transactions on Evolutionary Computation, first online 2024, final issue 29(4) in 2025. [Publisher DOI](https://doi.org/10.1109/TEVC.2024.3399560). | Later methodology for mixed continuous, integer and categorical features and algorithm selection on hyperparameter tasks. Species IDs are not meaningful numeric distances. Requires VGC feature, sampling-cost and held-out selector validation. |

## Diffusion models and sampling

### Discrete diffusion foundations

| Paper | Year | Citations | Grade | Venue and details | Notes |
| --- | --- | --- | --- | --- | --- |
| [Simplified and Generalized Masked Diffusion for Discrete Data](https://www.alphaxiv.org/abs/2406.04329) | 2024 | 715 | HIGH | 🟢 NeurIPS 2024 |  |
| [Structured Denoising Diffusion Models in Discrete State-Spaces](https://www.semanticscholar.org/paper/91b32fc0a23f0af53229fceaae9cce43a0406d2e) | 2021 | 2,330 | MED |  |  |
| [Simple and Effective Masked Diffusion Language Models](https://www.semanticscholar.org/paper/f8d357d38bbcdd93889fe71762eb57842b2ab063) | 2024 | 993 | MED |  |  |
| [Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution](https://www.semanticscholar.org/paper/ce806f8d32f6fb1eaa821248a1bc4fa2cd949fbb) | 2023 | 834 | LOW |  |  |
| [Argmax Flows and Multinomial Diffusion: Learning Categorical Distributions](https://www.semanticscholar.org/paper/1913d3edcc00d0aba097a9df190dd16f6fdfbf0c) | 2021 | 799 | LOW |  |  |
| [BERT has a Mouth, and It Must Speak: BERT as a Markov Random Field Language Model](https://www.semanticscholar.org/paper/d79ac7a7bafdc9a782fb8c53285ca11c7f2e3f18) | 2019 | 401 | LOW |  |  |
| [Your Absorbing Discrete Diffusion Secretly Models the Conditional Distributions of Clean Data](https://www.alphaxiv.org/abs/2406.03736) | 2024 | 423 | LOW | 🟢 ICLR 2025 |  |
| [Beyond Masked and Unmasked: Discrete Diffusion Models via Partial Masking](https://www.alphaxiv.org/abs/2505.18495) | 2025 | 21 | LOW | 🟢 NeurIPS 2025 |  |
| [Scaling Beyond Masked Diffusion Language Models](https://www.alphaxiv.org/abs/2602.15014) | 2026 | 29 | LOW | 🟢 ICML 2026 |  |
| [Bringing Stability to Diffusion: Decomposing and Reducing Variance of Training Masked Diffusion Models](https://www.alphaxiv.org/abs/2511.18159) | 2025 | 2 | LOW | 🟢 ICLR 2026 |  |

### Sampling order, remasking and diversity

| Paper | Year | Citations | Grade | Venue and details | Notes |
| --- | --- | --- | --- | --- | --- |
| [Masked Diffusion Models are Secretly Learned-Order Autoregressive Models](https://www.alphaxiv.org/abs/2511.19152) | 2025 | 4 | HIGH | 🟢 EurIPS 2025 PriGM workshop |  |
| [Remasking Discrete Diffusion Models with Inference-Time Scaling](https://www.semanticscholar.org/paper/19c3a5d9d32c57cd1482c8376f208cc2b2333334) | 2025 | 210 | MED |  |  |
| [[Low-density sampling from diffusion models]] | 2026 |  | MED |  |  |
| [A Tale of Two Temperatures: Simple, Efficient, and Diverse Sampling from Diffusion Language Models](https://arxiv.org/abs/2604.09921) | 2026 | 6 | MED | SPIGM 2026 · Compare token temperature with randomness in the order fields are filled. |  |
| [Re-evaluating Confidence Remasking in Masked Diffusion Language Models](https://arxiv.org/abs/2606.12232) | 2026 | 1 | MED | SPIGM 2026 · Evaluate whether revisiting generated fields justifies its cost; compare tuned baselines and diversity. |  |
| [Free Lunch for Pass@k? Low Cost Diverse Sampling for Diffusion Language Models](https://www.alphaxiv.org/abs/2603.04893) | 2026 |  | MED | 🟢 arXiv preprint, no venue listed |  |
| [D5P4: Partition Determinantal Point Process for Diversity in Parallel Discrete Diffusion Decoding](https://www.alphaxiv.org/abs/2603.19146) | 2026 |  | MED | 🟢 arXiv preprint, no venue listed |  |
| [Particle Guidance: non-I.I.D. Diverse Sampling with Diffusion Models](https://www.alphaxiv.org/abs/2310.13102) | 2023 |  | LOW | 🟢 ICLR 2024 |  |

### Structured generation and application examples

| Paper | Year | Citations | Grade | Venue and details | Notes |
| --- | --- | --- | --- | --- | --- |
| [Prompt-to-Slate: Diffusion Models for Prompt-Conditioned Slate Generation](https://www.semanticscholar.org/paper/da1f9ea84ef798c82a876d37fdf95e4568eafcad) | 2024 | 9 | MED |  |  |
| [Protein Design with Guided Discrete Diffusion](https://arxiv.org/abs/2305.20009) | 2023 | 248 | HIGH | 🟢 NeurIPS 2023 |  |
| [SDEdit: Guided Image Synthesis and Editing with Stochastic Differential Equations](https://www.semanticscholar.org/paper/f671a09e3e5922e6d38cb77dda8d76d5ceac2a27) | 2021 | 2,489 | LOW |  |  |
| [Compositional Visual Generation with Composable Diffusion Models](https://www.semanticscholar.org/paper/3ff7153fd6bd47d08084c7f50f8fd70026c126e7) | 2022 | 784 | LOW |  |  |
| [TabDDPM: Modelling Tabular Data with Diffusion Models](https://www.semanticscholar.org/paper/25d3a4e048d0020ba9cffc6442ebd4e7bb548a55) | 2022 | 733 | LOW |  |  |
| [Diffusion-LM Improves Controllable Text Generation](https://www.semanticscholar.org/paper/1386b8a11929cf02da291c56aca353e33bbc22ed) | 2022 | 1,576 | LOW |  |  |
| [TabDiff: a Mixed-type Diffusion Model for Tabular Data Generation](https://www.alphaxiv.org/abs/2410.20626) | 2024 | 99 | LOW | 🟢 ICLR 2025 |  |

### Guidance methods

| Paper | Year | Citations | Grade | Venue and details | Notes |
| --- | --- | --- | --- | --- | --- |
| [Diffusion Models as Plug-and-Play Priors](https://arxiv.org/abs/2206.09012) | 2022 | 325 | ungraded | NeurIPS 2022 — conditional reading for steering a diffusion prior with a compatible differentiable matchup surrogate; battle simulation supplies no direct gradient. · [[DIFUSCO references - relevance to counter-team search\|Relevance notes]] |  |
| [Gradient Guidance for Diffusion Models: An Optimization Perspective](https://www.semanticscholar.org/paper/5aabe3a210270711c72bddaa0290c4edd3db7720) | 2024 | 80 | MED | 🟢 NeurIPS 2024 |  |
| [Robust Guided Diffusion for Offline Black-Box Optimization](https://www.semanticscholar.org/paper/45957487086436f6fcbdd88c6a6dd5f268d9310f) | 2024 | 11 | MED |  |  |
| 🟢 [Prof. Mengdi Wang \| Guiding Diffusion Models Towards Generative Optimization by INI Satellite Events](https://www.youtube.com/watch?v=8W9qHXN0weg) | 2024 |  | MED | talk, Isaac Newton Institute DMLW01 workshop (2024-07-18) |  |
| [Derivative-Free Guidance in Continuous and Discrete Diffusion Models with Soft Value-Based Decoding](https://www.semanticscholar.org/paper/a673ef39237e227381fccf5b1d154d96de428f1d) | 2024 | 151 | MED |  |  |
| [Practical and Asymptotically Exact Conditional Sampling in Diffusion Models](https://www.semanticscholar.org/paper/79531b47bb27cb18022891eb2ab1fcb41745fca6) | 2023 | 283 | MED | 🟢 NeurIPS 2023 |  |
| [Simple Guidance Mechanisms for Discrete Diffusion Models](https://www.semanticscholar.org/paper/05f9997d61460fea9b586f98722c4be32f4a8b22) | 2024 | 168 | MED |  |  |
| [Unlocking Guidance for Discrete State-Space Diffusion and Flow Models](https://www.semanticscholar.org/paper/c8bfa3abf6a8cbf0a5b3a093661510f34cfa0098) | 2024 | 146 | MED |  |  |
| [Classifier-Free Diffusion Guidance](https://www.semanticscholar.org/paper/af9f365ed86614c800f082bd8eb14be76072ad16) | 2022 | 7,470 | MED |  |  |
| [Diffusion Tree Sampling: Scalable inference-time alignment of diffusion models](https://www.semanticscholar.org/paper/aabd199f5094c726021f9a4175491549f6feb593) | 2025 | 44 | MED | 🟢 NeurIPS 2025 |  |
| [UnMaskFork: Test-Time Scaling for Masked Diffusion via Deterministic Action Branching](https://www.semanticscholar.org/paper/eac84de6e4ce26fbbd7fb2013a923afceda4d797) | 2026 | 1 | MED | 🟢 ICML 2026 |  |
| [Inference-Time Alignment in Diffusion Models with Reward-Guided Generation: Tutorial and Review](https://www.semanticscholar.org/paper/78f37226009495fc11f5b64c5462c9470d2726e8) | 2025 | 97 | MED | 🟢 arXiv preprint, no venue listed |  |
| [Plug-and-Play Guidance for Discrete Diffusion Models via Gradient-Informed Logit Correction](https://www.semanticscholar.org/paper/30aff605819e4a05c309040dc980ff82a767d828) | 2026 | 3 | LOW |  |  |
| [Guess & Guide: Gradient-Free Zero-Shot Diffusion Guidance](https://arxiv.org/abs/2603.07860v1) | 2026 | 1 | LOW | 🟢 ReALM-GEN (ICLR 2026 workshop) |  |
| [Dynamic Search for Inference-Time Alignment in Diffusion Models](https://www.semanticscholar.org/paper/3546ed4907a006c8b4c8379b6059648c912f6a65) | 2025 | 41 | LOW | 🟢 arXiv preprint, no venue listed |  |
| [Inference-Time Alignment of Diffusion Models via Evolutionary Algorithms](https://www.semanticscholar.org/paper/281ea51be4268dbace01236879a56b575436474c) | 2025 | 1 | LOW | 🟢 arXiv preprint, no venue listed |  |
| [Diffusion Crossover: Defining Evolutionary Recombination in Diffusion Models via Noise Sequence Interpolation](https://www.semanticscholar.org/paper/4e598ea023dc54074a89c9b869effefa1b61e8b2) | 2026 | 0 | LOW | 🟢 arXiv preprint, no venue listed |  |
| [Inference-Time Scaling for Diffusion Models beyond Scaling Denoising Steps](https://www.semanticscholar.org/paper/9c8f548b85af8d3c861c9d97b6172c4ee44673d6) | 2025 | 266 | LOW | 🟢 arXiv preprint, no venue listed |  |

### Guidance analysis and correction

| Paper | Year | Citations | Grade | Venue and details | Notes |
| --- | --- | --- | --- | --- | --- |
| [What Exactly Does Guidance Do in Masked Discrete Diffusion Models](https://api.semanticscholar.org/arXiv:2506.10971) | 2025 | 8 | HIGH | 🟢 arXiv preprint, no venue listed |  |
| [Debiasing Guidance for Discrete Diffusion with Sequential Monte Carlo](https://arxiv.org/abs/2502.06079) | 2025 | 32 | MED | 🟢 Frontiers in Probabilistic Inference workshop at ICLR 2025 |  |
| [Improving Classifier-Free Guidance in Masked Diffusion: Low-Dim Theoretical Insights with High-Dim Impact](https://www.alphaxiv.org/abs/2507.08965) | 2025 |  | MED | 🟢 ICLR 2026 |  |
| [What does guidance do? A fine-grained analysis in a simple setting](https://proceedings.neurips.cc/paper_files/paper/2024/file/9a3942c235daa9c5f62a8598ae81a946-Paper-Conference.pdf) | 2024 | 56 | LOW (was MED) | 🟢 NeurIPS 2024 |  |
| [Classifier-Free Guidance is a Predictor-Corrector](https://arxiv.org/abs/2408.09000) | 2024 | 91 | LOW (was MED) | 🟢 M3L Workshop at NeurIPS 2024 |  |
| [Generative Uncertainty in Diffusion Models](https://www.semanticscholar.org/paper/54b4c9a67d0c55b5d430d45ead5d5c7246283232) | 2025 | 32 | LOW (was MED) | 🟢 UAI 2025 |  |
| [CADS: Unleashing the Diversity of Diffusion Models through Condition-Annealed Sampling](https://www.alphaxiv.org/abs/2310.17347) | 2023 |  | LOW | 🟢 ICLR 2024 |  |
| [Applying Guidance in a Limited Interval Improves Sample and Distribution Quality in Diffusion Models](https://www.alphaxiv.org/abs/2404.07724) | 2024 |  | LOW | 🟢 NeurIPS 2024 |  |
| [Guiding a Diffusion Model with a Bad Version of Itself](https://www.alphaxiv.org/abs/2406.02507) | 2024 |  | LOW | 🟢 NeurIPS 2024 |  |
| [Correction Training for Diffusion Models](https://www.alphaxiv.org/abs/2610.14136) | 2026 |  | LOW | 🟢 arXiv preprint, no venue listed |  |

## Retraining, diversity and model collapse

Read selectively when choosing training data or evaluating diversity. These diagnostics follow the current search-loop readings; a HIGH grade here does not make a future retraining failure more urgent than establishing the baseline. Fine-tuning-specific collapse is deferred below.

### Memorization and data replication

| Paper | Year | Citations | Grade | Venue and details | Notes |
| --- | --- | --- | --- | --- | --- |
| [Detecting, Explaining, and Mitigating Memorization in Diffusion Models](https://api.semanticscholar.org/arXiv:2407.21720) | 2024 | 133 | HIGH | 🟢 ICLR 2024 |  |
| [On Memorization in Diffusion Models](https://www.semanticscholar.org/paper/122a7e217fe70d5a1a44a6e2b67e859d1fc8e28d) | 2023 | 143 | LOW |  |  |
| [Diffusion Art or Digital Forgery? Investigating Data Replication in Diffusion Models](https://arxiv.org/abs/2212.03860) | 2022 | 562 | LOW |  |  |

### Recursive training and collapse dynamics

| Paper | Year | Citations | Grade | Venue and details | Notes |
| --- | --- | --- | --- | --- | --- |
| [Self-Consuming Generative Models Go MAD](https://api.semanticscholar.org/arXiv:2307.01850) | 2023 | 377 | HIGH | 🟢 ICLR 2024 |  |
| [On the Stability of Iterative Retraining of Generative Models on their own Data](https://api.semanticscholar.org/arXiv:2310.00429) | 2023 | 125 | HIGH | 🟢 ICLR 2024 spotlight |  |
| [AI models collapse when trained on recursively generated data](https://www.nature.com/articles/s41586-024-07566-y) | 2024 | 1,120 | MED | 🟢 Nature 2024 |  |
| [Sharp Bounds for Genetic Drift in Estimation of Distribution Algorithms](https://api.semanticscholar.org/arXiv:1910.14389) | 2019 | 39 | HIGH | 🟢 IEEE Transactions on Evolutionary Computation 2020 |  |
| [A Closer Look at Model Collapse: From a Generalization-to-Memorization Perspective](https://api.semanticscholar.org/arXiv:2509.16499) | 2025 | 22 | HIGH | 🟢 NeurIPS 2025 spotlight |  |

### Verification, correction and data mixing

| Paper | Year | Citations | Grade | Venue and details | Notes |
| --- | --- | --- | --- | --- | --- |
| [Beyond Model Collapse: Scaling Up with Synthesized Data Requires Verification](https://api.semanticscholar.org/arXiv:2406.07515) | 2024 | 52 | MED | 🟢 ICLR 2025 |  |
| [Self-Correcting Self-Consuming Loops for Generative Model Training](https://api.semanticscholar.org/arXiv:2402.07087) | 2024 | 42 | MED | 🟢 ICML 2024 |  |
| [Is Model Collapse Inevitable? Breaking the Curse of Recursion by Accumulating Real and Synthetic Data](https://api.semanticscholar.org/arXiv:2404.01413) | 2024 | 186 | MED | 🟢 COLM 2024 |  |

### Diversity measurement and diversity-preserving fine-tuning

| Paper | Year | Citations | Grade | Venue and details | Notes |
| --- | --- | --- | --- | --- | --- |
| [The Vendi Score: A Diversity Evaluation Metric for Machine Learning](https://www.alphaxiv.org/abs/2210.02410) | 2022 |  | HIGH | 🟢 TMLR 2023 |  |
| [Diversity-Rewarded CFG Distillation](https://www.alphaxiv.org/abs/2410.06084) | 2024 |  | MED | 🟢 ICLR 2025 |  |
| [Quality-Weighted Vendi Scores And Their Application To Diverse Experimental Design](https://www.alphaxiv.org/abs/2405.02449) | 2024 |  | MED | 🟢 ICML 2024 |  |
| [Exposing Diversity Bias in Deep Generative Models: Statistical Origins and Correction of Diversity Error](https://www.alphaxiv.org/abs/2602.14682) | 2026 |  | LOW | 🟢 ICLR 2026 |  |

## Surrogates and active learning

### Learning objectives for team selection

| Paper | Year | Citations | Grade | Venue and details | Notes |
| --- | --- | --- | --- | --- | --- |
| [Decision-Focused Learning: Through the Lens of Learning to Rank](https://proceedings.mlr.press/v162/mandi22a.html) | 2021 | 93 | ungraded | ICML 2022 — owner-selected focus (2026-09-13): compare surrogate objectives by the strength of discovered teams, rather than prediction error alone. Connection to Pokémon is a proposed experiment, not a result of this paper. |  |

### Structured surrogate models

| Paper | Year | Citations | Grade | Venue and details | Notes |
| --- | --- | --- | --- | --- | --- |
| [Constrained Discrete Black-Box Optimization using Mixed-Integer Programming](https://proceedings.mlr.press/v162/papalexopoulos22a.html) | 2021 | 27 | ungraded | ICML 2022 — NN+MILP combines a learned score predictor with constrained optimization; candidate approach for a team-recombination teacher. Assumes noiseless evaluations, so battle noise needs additional handling; optimality applies to the learned acquisition model, not actual team strength. Read §§3.1–3.3. |  |
| [Distributional Energy-Based Models for Uncertainty-Aware Structured LLM Reasoning](https://arxiv.org/abs/2605.18871) | 2026 | 2 | MED | 🟢 EIML (ICML 2026 workshop) — targeted reading: §4.2 and Appendix F.8; test whether a team scorer adds value beyond choosing the strongest proposal source. VGC transfer is an audit design, not a validated scoring method. |  |

### Active learning and uncertainty sampling

| Paper | Year | Citations | Grade | Venue and details | Notes |
| --- | --- | --- | --- | --- | --- |
| [Active Learning Literature Survey](https://burrsettles.com/pub/settles.activelearning.pdf) | 2009 | 6,858 | MED | 🟢 Computer Sciences Technical Report 1648, University of Wisconsin-Madison, 2009 |  |
| [On the Relationship between Data Efficiency and Error for Uncertainty Sampling](https://proceedings.mlr.press/v80/mussmann18a.html) | 2018 | 38 | HIGH | 🟢 ICML 2018 |  |
| [Active Learning for Affinity Prediction of Antibodies](https://www.alphaxiv.org/abs/2406.07263) | 2024 | 10 | LOW | 🟢 NeurIPS 2024 BDU workshop |  |
| [Rethinking Aleatoric and Epistemic Uncertainty](https://www.alphaxiv.org/abs/2412.20892) | 2024 | 51 | LOW | 🟢 NeurIPS 2024 BDU workshop |  |

### Membership queries and concept learning

| Paper | Year | Citations | Grade | Venue and details | Notes |
| --- | --- | --- | --- | --- | --- |
| [Textual Membership Queries](https://www.ijcai.org/proceedings/2020/0369.pdf) | 2020 | 5 | HIGH | 🟢 IJCAI 2020 |  |
| [Queries and Concept Learning](http://machinelearning202.pbworks.com/f/AngluinQueriesConceptLearningfulltext.pdf) | 1988 | 1,480 | MED | 🟢 Machine Learning 2:319-342, 1988 |  |
| [Active Query Synthesis for Preference Learning](https://arxiv.org/abs/2605.26072) | 2026 | 0 | LOW |  |  |

## Legality and constraint programming

### Constrained diffusion and decoding

| Paper | Year | Citations | Grade | Venue and details | Notes |
| --- | --- | --- | --- | --- | --- |
| [Constrained Discrete Diffusion](https://arxiv.org/abs/2503.09790) | 2025 | 24 | MED | 🟢 NeurIPS 2025 |  |
| [Constrained Code Generation with Discrete Diffusion](https://arxiv.org/abs/2605.16829) | 2026 | 1 | MED | 🟢 arXiv preprint, no venue listed |  |
| [Constrained Decoding of Diffusion LLMs with Context-Free Grammars](https://arxiv.org/abs/2508.10111) | 2025 | 21 | MED | 🟢 arXiv preprint, no venue listed |  |

### Streamliners

| Paper | Year | Citations | Grade | Venue and details | Notes |
| --- | --- | --- | --- | --- | --- |
| [StreamLLM: Enhancing Constraint Programming with Large Language Model-Generated Streamliners](https://jair.org/index.php/jair/article/view/18965) | 2025 | 8 | MED |  |  |
| [Streamlined Constraint Reasoning](https://www.cs.cornell.edu/gomes/papers/gomes-sellman-cp04.pdf) | 2004 |  | MED |  |  |

### Learning constraints from examples and queries

| Paper | Year | Citations | Grade | Venue and details | Notes |
| --- | --- | --- | --- | --- | --- |
| [A Model Seeker: Extracting Global Constraint Models from Positive Examples](https://www.semanticscholar.org/paper/d7a118530bc40fe3c2cf3d81fde56b1fc3cb1f30) | 2012 | 134 | LOW |  |  |
| [Learning Constraints through Partial Queries](https://www.lirmm.fr/~bessiere/Site/stock/aij23.pdf) | 2023 | 10 | LOW |  |  |

### Symmetry breaking

| Paper | Year | Citations | Grade | Venue and details | Notes |
| --- | --- | --- | --- | --- | --- |
| [Symmetry in Constraint Programming](https://sites.cs.st-andrews.ac.uk/people/ipg1/papers/GentPetriePugetFinalDraft.pdf) | 2006 | 146 | MED | 🟢 author draft (🔒 Elsevier), Handbook of Constraint Programming ch. 10, 2006 |  |
| [Breaking symmetries in all different problems](https://www.ijcai.org/Proceedings/05/Papers/1244.pdf) | 2005 | 75 | MED | 🟢 IJCAI 2005 |  |
| [Symmetry-Breaking Predicates for Search Problems](https://ix.cs.uoregon.edu/~luks/symmetrybreaking.pdf) | 1996 | 566 | LOW | 🟢 KR 1996 |  |
| [Breaking Row and Column Symmetries in Matrix Models](https://pierre-flener.github.io/research/pub/CP02.pdf) | 2002 | 240 | LOW | 🟢 author preprint, CP 2002 |  |

### Dominance breaking

| Paper | Year | Citations | Grade | Venue and details | Notes |
| --- | --- | --- | --- | --- | --- |
| [Dominance breaking constraints](https://people.eng.unimelb.edu.au/pstuckey/papers/domjournal.pdf) | 2015 | 26 | MED | 🟢 author manuscript (🔒 Springer), Constraints 20(2), 2015 |  |
| [Automatic Generation of Dominance Breaking Nogoods for a Class of Constraint Optimization Problems](http://www.cse.cuhk.edu.hk/~jlee/publ/23/aij23AutoDomBreak.pdf) | 2023 | 2 | LOW | 🟢 author preprint, Artificial Intelligence 323, 2023 |  |

## Automated heuristic design

### Heuristic generation and strategy evolution

| Paper | Year | Citations | Grade | Venue and details | Notes |
| --- | --- | --- | --- | --- | --- |
| [G-LNS: Generative Large Neighborhood Search for LLM-Based Automatic Heuristic Design](https://www.semanticscholar.org/paper/deefec531e1ccc8798e29f9c38125fdbc34a9a74) | 2026 | 10 | MED | 🟢 arXiv 2602.08253 (2026-02), no venue listed |  |
| [Beyond Static Evaluation: Co-Evolutionary Mechanisms for LLM-Driven Strategy Evolution in Adversarial Games](https://arxiv.org/abs/2606.10389) | 2026 | 3 | MED | 🟢 AAMAS 2026 MCTF Competition |  |
| [RuleSmith: Multi-Agent LLMs for Automated Game Balancing](https://arxiv.org/abs/2602.06232) | 2026 | 2 | LOW | 🟢 arXiv preprint, no venue listed |  |

### Hyper-heuristic taxonomy and selection

| Paper | Year | Citations | Grade | Venue and details | Notes |
| --- | --- | --- | --- | --- | --- |
| [A Classification of Hyper-Heuristic Approaches: Revisited](https://people.cs.nott.ac.uk/pszeo/docs/publications/HHClassChapterRevisited.pdf) | 2019 | 188 | MED | 🟢 author PDF (🔒 Springer), Handbook of Metaheuristics 3rd ed. ch. 14, 2019 |  |
| [Recent advances in selection hyper-heuristics](https://www.sciencedirect.com/science/article/pii/S0377221719306526) | 2020 | 347 | LOW | 🟢 CC-BY open access, European Journal of Operational Research 285(2), 2020 |  |

## Battle policies and self-play

### Self-play and game-theoretic foundations

| Paper | Year | Citations | Grade | Venue and details | Notes |
| --- | --- | --- | --- | --- | --- |
| [[A Survey on Self-play Methods in Reinforcement Learning]] | 2024 | 37 | MED |  |  |
| [A Unified Game-Theoretic Approach to Multiagent Reinforcement Learning](https://arxiv.org/abs/1711.00832) | 2017 | 780 | MED |  |  |
| [Planning in the Presence of Cost Functions Controlled by an Adversary](https://www.aaai.org/Papers/ICML/2003/ICML03-071.pdf) | 2003 | 392 | MED |  |  |
| [Game-Theoretic Multiagent Reinforcement Learning](https://www.semanticscholar.org/paper/c3662e9176a7ad90020bdd025c179c5925d0b5b0) | 2020 | 217 | LOW |  |  |

### Regret minimization

| Paper | Year | Citations | Grade | Venue and details | Notes |
| --- | --- | --- | --- | --- | --- |
| [An Introduction to Counterfactual Regret Minimization](http://modelai.gettysburg.edu/2013/cfr/cfr.pdf) | 2013 | 36 | LOW |  |  |
| [Efficient Monte Carlo Counterfactual Regret Minimization in Games with Many Player Actions](https://papers.nips.cc/paper/2012/file/3df1d4b96d8976ff5986393e8767f5b2-Paper.pdf) | 2012 | 39 | LOW |  |  |

### Policy learning and game environments

| Paper | Year | Citations | Grade | Venue and details | Notes |
| --- | --- | --- | --- | --- | --- |
| [[Grandmaster level in StarCraft II using multi-agent reinforcement learning]] | 2019 | 4,567 | MED |  |  |
| [Overcoming Valid Action Suppression in Unmasked Policy Gradient Algorithms](https://rlj.cs.umass.edu/2026/papers/Paper1.pdf) | 2026 | 2 | LOW |  |  |
| [FootsiesGym: A Fighting Game Benchmark for Two-Player Zero-Sum Imperfect-Information Games](https://arxiv.org/abs/2607.06514) | 2026 | 0 | LOW |  |  |
| [Superhuman AI for Generals.io Using Self-Play Reinforcement Learning](https://arxiv.org/abs/2606.23348) | 2026 | 1 | LOW |  |  |
| [Emergent Complexity via Multi-Agent Competition](https://arxiv.org/abs/1710.03748) | 2017 | 444 | LOW |  |  |
| [Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model](https://www.semanticscholar.org/paper/c39fb7a46335c23f7529dd6f9f980462fd38653a) | 2019 | 2,803 | LOW |  |  |

### Ranking and evaluation in non-transitive games

| Paper | Year | Citations | Grade | Venue and details | Notes |
| --- | --- | --- | --- | --- | --- |
| [[Alpha-Rank Multi-Agent Evaluation by Evolution]] | 2019 | 146 | MED |  |  |
| [Real World Games Look Like Spinning Tops](https://arxiv.org/abs/2004.09468) | 2020 | 124 | MED |  |  |
| [Re-evaluating Evaluation](https://arxiv.org/abs/1806.02643) | 2018 | 123 | MED |  |  |

### Competitive coevolution

| Paper | Year | Citations | Grade | Venue and details | Notes |
| --- | --- | --- | --- | --- | --- |
| [Coevolutionary Principles](https://www.cs.tufts.edu/comp/150GA/handouts/nchb-main.pdf) | 2012 |  | LOW |  |  |
| [New Methods for Competitive Coevolution](https://cseweb.ucsd.edu/~crosin/newmethods.ps) | 1997 | 667 | LOW |  |  |
| [A comparison of evaluation methods in coevolution](https://arxiv.org/abs/1905.08723) | 2019 | 4 | LOW |  |  |
| 🔒 [Tripping Over the Past: Measuring Deceptive Progress in Competitive Coevolutionary Algorithms Employing Hall of Fame](https://doi.org/10.1007/978-3-032-23607-4_32) | 2026 | 0 | LOW |  |  |

## Training dynamics and engineering

### Plasticity loss and recovery

| Paper | Year | Citations | Grade | Venue and details | Notes |
| --- | --- | --- | --- | --- | --- |
| [On Warm-Starting Neural Network Training](https://api.semanticscholar.org/arXiv:1910.08475) | 2019 | 24 | MED |  |  |
| [Plasticity Loss in Deep Reinforcement Learning: A Survey](https://api.semanticscholar.org/arXiv:2411.04832) | 2024 | 30 | LOW |  |  |
| [A Study of Plasticity Loss in On-Policy Deep Reinforcement Learning](https://api.semanticscholar.org/arXiv:2405.19153) | 2024 | 37 | LOW |  |  |
| [The Primacy Bias in Deep Reinforcement Learning](https://api.semanticscholar.org/arXiv:2205.07802) | 2022 | 343 | LOW |  |  |
| [The Dormant Neuron Phenomenon in Deep Reinforcement Learning](https://api.semanticscholar.org/arXiv:2302.12902) | 2023 | 210 | LOW |  |  |
| [Loss of Plasticity in Continual Deep Reinforcement Learning](https://api.semanticscholar.org/arXiv:2303.07507) | 2023 | 158 | LOW |  |  |
| [Disentangling the Causes of Plasticity Loss in Neural Networks](https://api.semanticscholar.org/arXiv:2402.18762) | 2024 | 103 | LOW |  |  |
| [Deep Reinforcement Learning with Plasticity Injection](https://api.semanticscholar.org/arXiv:2305.15555) | 2023 | 99 | LOW |  |  |
| [Maintaining Plasticity in Continual Learning via Regenerative Regularization](https://api.semanticscholar.org/arXiv:2308.11958) | 2023 | 78 | LOW |  |  |

### Batch size and parallel training

| Paper | Year | Citations | Grade | Venue and details | Notes |
| --- | --- | --- | --- | --- | --- |
| [Batch size-invariance for policy optimization](https://api.semanticscholar.org/arXiv:2110.00641) | 2021 | 42 | LOW |  |  |
| [An Empirical Model of Large-Batch Training](https://api.semanticscholar.org/arXiv:1812.06162) | 2018 | 457 | LOW |  |  |
| [SAPG: Split and Aggregate Policy Gradients](https://api.semanticscholar.org/arXiv:2407.20230) | 2024 | 27 | LOW |  |  |
| [Preventing Learning Stagnation in PPO by Scaling to 1 Million Parallel Environments](https://rlj.cs.umass.edu/2026/papers/Paper19.pdf) | 2026 | 4 | LOW |  |  |

### Exploration and sample efficiency

| Paper | Year | Citations | Grade | Venue and details | Notes |
| --- | --- | --- | --- | --- | --- |
| [Improving Sample Efficiency in Multi-Agent Reinforcement Learning for Simulated Football Games via Exploration](https://doi.org/10.1145/3815598.3815627) | 2025 | 1 | LOW |  |  |
| [[Human-Like Goalkeeping in a Realistic Football Simulation a Sample-Efficient Reinforcement Learning Approach]] | 2025 | 2 | LOW |  |  |

### Adaptive and multi-task agents

| Paper | Year | Citations | Grade | Venue and details | Notes |
| --- | --- | --- | --- | --- | --- |
| [AMAGO: Scalable In-Context Reinforcement Learning for Adaptive Agents](https://arxiv.org/abs/2310.09971) | 2023 |  | LOW |  |  |
| [AMAGO-2: Breaking the Multi-Task Barrier in Meta-Reinforcement Learning with Transformers](https://arxiv.org/abs/2411.11188) | 2024 | 18 | LOW |  |  |
| [Multi-Task Reinforcement Learning Enables Parameter Scaling](https://api.semanticscholar.org/arXiv:2503.05126) | 2025 | 12 | LOW |  |  |

### Offline data, imitation and reward assignment

| Paper | Year | Citations | Grade | Venue and details | Notes |
| --- | --- | --- | --- | --- | --- |
| [Accelerating Online Reinforcement Learning with Offline Datasets](https://arxiv.org/abs/2006.09359) | 2020 | 936 | LOW |  |  |
| [Critic Regularized Regression](https://arxiv.org/abs/2006.15134) | 2020 | 414 | LOW |  |  |
| [Discovering High Quality Chess Puzzles with Offline Reinforcement Learning](https://rlj.cs.umass.edu/2026/papers/Paper147.pdf) | 2026 | 0 | LOW |  |  |
| [Minimal Ingredients for Reward Assignment from Expert Demonstrations](https://rlj.cs.umass.edu/2026/papers/Paper86.pdf) | 2025 | 0 | LOW |  |  |
| [Vanquish Your Past: Shifted Imitation Learning in Hades](https://doi.org/10.1145/3815598.3815650) | 2026 | 0 | LOW |  |  |

### Simulation infrastructure and benchmarks

| Paper | Year | Citations | Grade | Venue and details | Notes |
| --- | --- | --- | --- | --- | --- |
| [Assistax: A Multi-Agent Hardware-Accelerated RL Benchmark for Assistive Robotics](https://rlj.cs.umass.edu/2026/papers/Paper110.pdf) | 2025 | 1 | LOW |  |  |
| [Ludax: A GPU-Accelerated Description Language for Board Games](https://rlj.cs.umass.edu/2026/papers/Paper46.pdf) | 2025 | 3 | LOW |  |  |

### Statistical evaluation and non-stationarity

| Paper | Year | Citations | Grade | Venue and details | Notes |
| --- | --- | --- | --- | --- | --- |
| [Space-sampled Value Decay: Forgetting Mechanisms for Non-stationary Deep Reinforcement Learning](https://arxiv.org/abs/2606.11797) | 2026 | 0 | MED | 🟢 EIML (ICML 2026 workshop) — stale value estimates under changing opponents; DQN/SAC evidence, VGC state sampling unresolved. Read §§3–6 after HIGH papers. |  |
| [Confidence Intervals for the Interquartile Mean](https://rlj.cs.umass.edu/2026/papers/Paper28.pdf) | 2026 | 0 | LOW |  |  |

### Data mixtures and metadata

| Paper | Year | Citations | Grade | Venue and details | Notes |
| --- | --- | --- | --- | --- | --- |
| [Metadata Conditioning Accelerates Language Model Pre-training](https://www.alphaxiv.org/abs/2501.01956) | 2025 | 22 | LOW | 🟢 ICML 2025 |  |

## Looped transformers

### Looping and depth growth

| Paper | Year | Citations | Grade | Venue and details | Notes |
| --- | --- | --- | --- | --- | --- |
| [From Growing to Looping: A Unified View of Iterative Computation in LLMs](https://www.alphaxiv.org/abs/2602.16490) | 2026 | 7 | MED | 🟢 arXiv preprint, no venue listed |  |

## Deferred readings

These are outside the active queue.

### Fine-tuning methods and diagnostics

Revisit when planning the first fine-tuning experiment or repeated fine-tuning rounds. Read Diffusion-BBO and the current search baselines first, regardless of the grades below.

| Paper | Year | Citations | Grade | Venue and details | Notes |
| --- | --- | --- | --- | --- | --- |
| [Training Diffusion Language Models for Black-Box Optimization (DiBO)](https://arxiv.org/html/2603.17919v1) | 2026 | 4 | ungraded | 2026, arXiv preprint — adds domain adaptation, supervised fine-tuning, and reinforcement learning based on label improvements; examines steering generation through training beyond supplying examples in a prompt. | 2.3 LLMs for Black-Box Optimization<br>Inverse Methods. |
| [Model Collapse in the Self-Consuming Chain of Diffusion Finetuning: A Novel Perspective from Quantitative Trait Modeling](https://api.semanticscholar.org/arXiv:2407.17493) | 2024 | 12 | HIGH | 🟢 ICLR 2025 |  |

### Reward integration

Revisit when planning reward objectives, fine-tuning or preference alignment; HIGH grades here remain deferred.

| Paper | Year | Citations | Grade | Venue and details | Notes |
| --- | --- | --- | --- | --- | --- |
| [Feedback Efficient Online Fine-Tuning of Diffusion Models](https://www.semanticscholar.org/paper/6a225bb2454f66d1b106bbae7e5129d6569aa260) | 2024 | 58 | HIGH | 🟢 ICML 2024 |  |
| [Sample-Efficient Optimization in the Latent Space of Deep Generative Models via Weighted Retraining](https://arxiv.org/abs/2006.09191) | 2020 | 190 | HIGH | 🟢 NeurIPS 2020 |  |
| [Fine-Tuning Discrete Diffusion Models via Reward Optimization with Applications to DNA and Protein Design](https://proceedings.iclr.cc/paper_files/paper/2025/hash/771e09dd204ea339da0d8114c48afd21-Abstract-Conference.html) | 2024 | 93 | MED | 🟢 ICLR 2025 · [Alternate source](https://www.semanticscholar.org/paper/d1461167c9fef8fe3ba129c514acfd14bbe7a51e) |  |
| [Self-Consuming Generative Models with Curated Data Provably Optimize Human Preferences](https://api.semanticscholar.org/arXiv:2407.09499) | 2024 | 43 | HIGH | 🟢 NeurIPS 2024 spotlight |  |
| [CADO: From Imitation to Cost Minimization for Heatmap-based Solvers in Combinatorial Optimization](https://arxiv.org/abs/2602.08210) | 2026 | 2 | MED | 🟢 TMLR 2026 |  |
| [Scalable Discrete Diffusion Samplers: Combinatorial Optimization and Statistical Physics](https://www.semanticscholar.org/paper/ec3d27dbfdd9431720c8a848b0ac33145dac58be) | 2025 | 33 | MED | 🟢 ICLR 2025 |  |
| [DPOK: Reinforcement Learning for Fine-tuning Text-to-Image Diffusion Models](https://www.semanticscholar.org/paper/a553bf27d801d09f667fe121c0ba9632257f364b) | 2023 | 543 | LOW | 🟢 NeurIPS 2023 |  |
| [Iterated Denoising Energy Matching for Sampling from Boltzmann Densities](https://www.semanticscholar.org/paper/9f0f134bc53f9aea130f1a561acb5fa6b08ae4ce) | 2024 | 134 | LOW | 🟢 ICML 2024 |  |
| [Improved sampling via learned diffusions](https://www.semanticscholar.org/paper/d174e9d35a9d6d899acbc661e05a937a659ffc42) | 2023 | 120 | LOW | 🟢 ICLR 2024 |  |
| [Reward-Directed Conditional Diffusion: Provable Distribution Estimation and Reward Improvement](https://arxiv.org/abs/2307.07055) | 2023 | 59 | LOW (was MED) | 🟢 NeurIPS 2023 |  |
| [Training Diffusion Models with Reinforcement Learning](https://www.semanticscholar.org/paper/d8c78221e4366d6a72a6b3e41e35b706cc45c01d) | 2023 | 1,151 | LOW (was MED) | 🟢 ICLR 2024 |  |
| [The tractability landscape of diffusion alignment: regularization, rewards, and computational primitives](https://arxiv.org/abs/2605.11361) | 2026 | 3 | LOW (was MED) | 🟢 arXiv preprint, no venue listed |  |

### Other readings kept for later

| Paper | Year | Citations | Grade | Venue and details | Notes |
| --- | --- | --- | --- | --- | --- |
| [Identifying and Clustering Counter Relationships of Team Compositions in PvP Games for Efficient Balance Analysis](https://arxiv.org/abs/2408.17180) | 2024 | 0 | MED | TMLR 2024. Counter-matrix between team compositions; take the measure, not the balance-analysis pipeline. |  |
| [Team Composition in PES2018 Using Submodular Function Optimization](https://doi.org/10.1109/ACCESS.2019.2919447) | 2019 | 6 | LOW | IEEE Access 2019. Team selection as a submodular coverage function with a greedy solver; the closest published cousin of the weighted maximum coverage objective. |  |

### Parked directions

Revisit if the project direction changes.

| Paper | Year | Citations | Grade | Venue and details | Notes |
| --- | --- | --- | --- | --- | --- |
| [A Tutorial on Bayesian Optimization](https://api.semanticscholar.org/arXiv:1807.02811) | 2018 | 2,548 | MED |  |  |
| [Evaluating the Stability of the Smogon Tier List for Competitive Pokémon Battling](https://ceur-ws.org/Vol-3926/paper2.pdf) | 2024 | 0 | MED | 🟢 EXAG 2024 |  |
| [A QUBO Framework for Team Formation](https://arxiv.org/abs/2503.23209) | 2025 | 3 | LOW |  |  |
| [A Tutorial on Formulating and Using QUBO Models](https://api.semanticscholar.org/arXiv:1811.11538) | 2018 | 919 | LOW |  |  |
| [Designing Metamaterials with Quantum Annealing and Factorization Machines](https://arxiv.org/abs/1902.06573) | 2019 | 161 | LOW | kept only for the pairwise-interaction surrogate idea, not annealing |  |
| 🟢 [Calculating Pokémon Teams v1.0 by John (JaybeeVGC)](https://docs.google.com/document/d/1IIXfm49kbrij4FouD3x9RgYdaM2BetE1/edit) |  |  | LOW | PuLP linear programming over type coverage; retired framing |  |
| [Pokémon team optimization by Nicolas Chagnet](https://nchagnet.eu/blog/pokemon-team-optimization/) | 2025 |  | LOW | 🟢 blog + PuLP repo, 2025 |  |
| [Which Heroes to Pick? Learning to Draft in MOBA Games with Neural Networks and Tree Search](https://arxiv.org/abs/2012.10171) | 2020 | 36 | LOW | tree search is the right tool class, but the sequential draft model does not fit simultaneous bring-six selection. |  |
| [Introducing Tales of Tribute AI Competition](https://www.semanticscholar.org/paper/2cd0f87d70a0b6afd79324f959c68f04ef6c0945) | 2023 | 5 | LOW | about playing agents for a card game, not team construction. |  |

## SPIGM papers not accepted

The other 16 initial additions are retained for reference. Earlier assessments are in [[SPIGM 2026 — full-paper reassessment]] and [[Reading List — notes and decision history]].

| Paper | Year | Citations | Grade | Venue and details | Notes |
| --- | --- | --- | --- | --- | --- |
| [Learn from Your Mistakes: Self-Correcting Masked Diffusion Models](https://arxiv.org/abs/2602.11590) | 2026 | 20 | DEFER | SPIGM 2026 · Learned correction is legitimate, but no established benefit beyond current repair or for the active data comparison. |  |
| [Breaking the Factorization Barrier in Diffusion Language Models](https://arxiv.org/abs/2603.00045) | 2026 | 11 | DEFER | SPIGM 2026 · A joint head chiefly addresses simultaneous field generation, which the current sampler does not do. |  |
| [Internal Data Repetition Destroys Language Models](https://arxiv.org/abs/2606.24998) | 2026 | 4 | DROP | SPIGM 2026 · Exact duplicate removal is already available. The paper studies concentrated exact repetition; extending it to distinct mutations did not justify this reading. |  |
| [Finetuning Generative Models to Match Feature Distributions](https://arxiv.org/abs/2606.19496) | 2026 | 0 | NOT ACCEPTED | SPIGM 2026 · Outside the two-paper owner-confirmed shortlist. |  |
| [The Confidence Shortcut: A Reasoning Failure Mode of Masked Diffusion Models](https://arxiv.org/abs/2605.29123) | 2026 | 1 | DEFER | SPIGM 2026 · Relevant if confidence-aligned training becomes active; the current model does not use it. |  |
| [Uniform Diffusion Models Revisited: Leave-One-Out Denoiser and Absorbing State Reformulation](https://arxiv.org/abs/2605.22765) | 2026 | 12 | DEFER | SPIGM 2026 · A uniform-corruption objective issue does not establish a defect in the current mask-only model. |  |
| [Tensor-Train Joint Modeling for Few-Step Discrete Diffusion](https://arxiv.org/abs/2607.03788) | 2026 | 1 | DEFER | SPIGM 2026 · Low-rank joint sampling is a future parallel-decoding option, with ordering and rank assumptions. |  |
| [Latent-Augmented Discrete Diffusion Models](https://arxiv.org/abs/2510.18114) | 2025 | 0 | DEFER | SPIGM 2026 · A latent channel adds substantial modeling requirements before a demonstrated current need. |  |
| [Learned Relay Representations for Forward-Thinking Discrete Diffusion Models](https://arxiv.org/abs/2605.22967) | 2026 | 1 | DEFER | SPIGM 2026 · Compatible with sequential generation, but a persistent-state benefit has not been established here. |  |
| [Recursive Scaling in Masked Diffusion Models](https://arxiv.org/abs/2606.18022) | 2026 | 2 | DEFER | SPIGM 2026 · A future capacity-versus-compute study, not evidence about the current data or CEM questions. |  |
| [Understanding and Accelerating the Training of Masked Diffusion Language Models](https://arxiv.org/abs/2605.13026) | 2026 | 1 | DROP | SPIGM 2026 · The owner confirms diffusion training is not a problem. Mask-context diagnostics do not supply a sufficiently relevant current reading task. |  |
| [Time-Annealed Perturbation Sampling: Diverse Generation for Diffusion Language Models](https://arxiv.org/abs/2601.22629) | 2026 | 4 | DROP | SPIGM 2026 · Weak categorical-conditioning transfer, plus filtering and reporting issues undermine the proposed payoff. |  |
| [Hacking Generative Perplexity: Why Unconditional Text Evaluation Needs Distributional Metrics](https://arxiv.org/abs/2606.08417) | 2026 | 8 | DROP | SPIGM 2026 · The project does not use generative perplexity and already plans full-team/composition diversity measurements. The earlier team-template stress-test connection was an analogy, insufficient to justify reading this paper. |  |
| [DUEL: Exact Likelihood for Masked Diffusion via Deterministic Unmasking](https://arxiv.org/abs/2603.01367) | 2026 | 12 | DROP | SPIGM 2026 · Exact likelihood for a specified reveal schedule is not a current project requirement. |  |
| [TUBE: Tangent Upper Bound on Evidence for Discrete Diffusion Language Models](https://arxiv.org/abs/2605.24292) | 2026 | 2 | DROP | SPIGM 2026 · Sophisticated likelihood bounds do not answer the current team-quality or diversity decisions. |  |
| [Contrastive Distribution Matching for Amortized Sequential Monte Carlo in Discrete Diffusion](https://arxiv.org/abs/2605.23346) | 2026 | 5 | DEFER | SPIGM 2026 · Keep as a future reward-integration reference, requiring a suitable reward and trajectory model. |  |

## Rejected readings

Retained with the recorded reasons so they are not suggested again.

| Paper | Year | Citations | Grade | Venue and details | Notes |
| --- | --- | --- | --- | --- | --- |
| [Variational Search Distributions](https://arxiv.org/abs/2409.06142) | 2024 | 9 | Rejected | 🟢 ICLR 2025, was HIGH. Rejected at the owner's request, 2026-09-10: 9 citations, and the papers citing it are themselves low-cited. |  |
| [Closed-Loop Generative Selection: Convergence, Memory, and Noisy Oracles](https://arxiv.org/abs/2607.22211) | 2026 | 0 | Rejected | 🟢 arXiv preprint, no venue listed, was MED. Owner's verdict, 2026-09-10: does not belong in the reading list. |  |
| [Convergence properties of the cross-entropy method for discrete optimization](https://people.smp.uq.edu.au/DirkKroese/ps/CEconv.pdf) | 2007 | 143 | Rejected | 🟢 Operations Research Letters 2007, was HIGH. Rejected at the owner's request, 2026-09-10: somewhat relevant, but he dislikes the writing. |  |
| [Efficient Active Search for Combinatorial Optimization Problems](https://arxiv.org/abs/2106.05126) | 2021 | 142 | Rejected | ICLR 2022, I just learn what was active search method which is about training the whole model using only insteance to increase probability more likely to generate the examples but the thing only one example not what I am looking for — Rejected 2026-09-10: per-instance fine-tuning is already the CEM loop; saves compute, not battles. |  |
| [Batch Bayesian Optimization for Replicable Experimental Design](https://arxiv.org/abs/2311.01195) | 2023 | 11 | Rejected | 🟢 NeurIPS 2023, previously MED — Owner's note: not useful for this project. |  |
| [Verifier-Constrained Flow Expansion for Discovery Beyond the Data](https://arxiv.org/abs/2602.15984) | 2026 | 7 | Rejected | ICLR 2026, was MED. Rejected 2026-09-06: no demonstrated categorical discrete-diffusion update; Appendix G.5 updates only continuous atom positions in the mixed molecular model. Outside the owner's current discrete team-generation scope. |  |
| [Replication or exploration? Sequential design for stochastic simulation experiments](https://arxiv.org/abs/1710.03206) | 2017 | 136 | Rejected | Technometrics 61(1):7-23, 2019, was MED. Owner's verdict, 2026-09-06: not relevant to the project. |  |
| [Bayesian Optimal Active Search and Surveying](https://arxiv.org/abs/1206.6406) | 2012 | 132 | Rejected | ICML 2012, was MED. Owner's verdict, 2026-09-06: not relevant to the project. |  |
| [The Offline-Frontier Shift: Diagnosing Distributional Limits in Generative Multi-Objective Optimization](https://arxiv.org/abs/2602.11126) | 2026 | 0 | Rejected | was MED. Rejected 2026-09-06: offline multi-objective / Pareto-front evaluation is outside the current project scope. |  |
| [Generative Refinement for Low-Budget Black-Box Optimization](https://arxiv.org/abs/2607.00691) | 2026 | 0 | Rejected | was MED. Rejected at the owner's request, 2026-09-06; not useful for the current scope. |  |
| [QUBO, I Choose You: Building an Optimal Pokémon Team with a Quantum Computer](https://www.dwavequantum.com/events-section/events/webinar-qubo-i-choose-you-building-an-optimal-pokemon-team-with-a-quantum-computer/) | 2019 |  | Rejected | D-Wave webinar 2019; species/type-level QUBO, formulation only in the recording, no published code; the pairwise-surrogate idea is already carried by the metamaterials entry. |  |
| [A Threshold of ln n for Approximating Set Cover](https://courses.cs.duke.edu/spring07/cps296.2/papers/p634-feige.pdf) | 1996 | 867 | Rejected | Feige 1996, hardness-of-approximation proof; requires approximation-algorithms background I do not have. The usable result (greedy gets within 1−1/e of optimal max coverage) is already covered by the submodular survey entries. |  |
| [Heuristically Adaptive Diffusion-Model Evolutionary Strategy](https://arxiv.org/abs/2411.13420) | 2024 | 10 | Rejected | HADES: a diffusion model as the offspring sampler inside a continuous-parameter evolutionary strategy (CMA-ES family). Real-valued genomes, toy fitness landscapes (double-peak, Rastrigin) and cart-pole only. Loop shape matches our direction but the object optimized does not; the idea is already carried by Diffusion Models are Evolutionary Algorithms and Design by adaptive sampling. 🟢 Published as Advanced Science 13(20), 2026 (CC BY, PMC13067789): adds MountainCar and Lunar Lander, and says in print “we have applied our algorithm only to continuous parameter spaces.” Owner's verdict, 2026-10-16: rejected. |  |
| [Evolutionary Optimization in Uncertain Environments — A Survey](https://www.honda-ri.de/pubs/pdf/1660.pdf) | 2005 | 1,686 | Rejected | Jin & Branke, IEEE Transactions on Evolutionary Computation 2005; superseded by the same group's newer surveys on evolutionary optimization in uncertain and dynamic environments. |  |
| [DraftRec: Personalized Draft Recommendation for Winning in Multi-Player Online Battle Arena Games](https://arxiv.org/abs/2204.12750) | 2022 | 19 | Rejected | recommender for each human player's champion preferences; no preference-draft phase exists in this project. |  |
| [Modeling Game Avatar Synergy and Opposition through Embedding in Multiplayer Online Battle Arena Games](https://arxiv.org/abs/1803.10402) | 2018 | 9 | Rejected | its own arXiv page notes it was rejected by AIIDE 2017; superseded idea. |  |
| [A Recommender System for Hero Line-Ups in MOBA Games](https://ojs.aaai.org/index.php/AIIDE/article/view/12938) | 2017 | 33 | Rejected | association-rule recommender, not an optimizer. |  |
| [Draft-Analysis of the Ancients: Predicting Draft Picks in DotA 2 Using Machine Learning](https://ojs.aaai.org/index.php/AIIDE/article/view/12899) | 2016 | 33 | Rejected | predicts what humans will pick; the project searches for counters instead. |  |
| [Finding a Team of Experts in Social Networks](https://cs-people.bu.edu/evimaria/papers/fp525-lappas.pdf) | 2009 | 764 | Rejected | objective is communication cost on a social graph, unrelated to coverage plus clauses. |  |
| [NBA Chemistry: Positive and Negative Synergies in Basketball](https://philipmaymin.com/papers/Maymin%20Maymin%20and%20Shen%20-%20NBA%20Chemistry%20-%20IJCSS.pdf) | 2013 | 28 | Rejected | heuristic synergy extraction, no transferable search method. |  |
| [OptMatch: Optimized Matchmaking via Modeling the High-Order Interactions on the Arena](https://linxiagong.github.io/OptMatch/KDD2020_OptMatch.pdf) | 2020 | 40 | Rejected | matchmaking pairs players; it does not build teams. |  |
| [Picking Winners in Daily Fantasy Sports Using Integer Programming](https://arxiv.org/abs/1604.01455) | 2016 | 7 | Rejected | the source list titled this as a football-team paper, but the arXiv record is the fantasy-sports integer-programming paper; either way the framing is retired here. |  |
| [A Comparison of Coevolution, Fixed, and Hybrid Training for Evolving Agents that Play Tales of Tribute Videogame](https://doi.org/10.1007/978-3-032-23607-4_33) | 2026 | 0 | Rejected | coevolution for playing agents, not team selection. |  |
| [Large Language Models As Evolution Strategies](https://arxiv.org/abs/2402.18381) | 2024 | 104 | Rejected | a large language model used as an in-context recombination operator for evolution strategies; continuous black-box functions, no constraints, no team structure. |  |
| [SOPL: A Sequential Optimal Learning Approach to Automated Prompt Engineering in Large Language Models](https://aclanthology.org/2025.findings-emnlp.1155/) | 2025 | 7 | Rejected | optimizes natural-language prompt templates; wrong domain, and it economizes on expensive model queries rather than simulator evaluations. |  |
| 🟢 [A Tight Combinatorial Algorithm for Submodular Maximization Subject to a Matroid Constraint](https://arxiv.org/abs/1204.4526) | 2012 | 127 | Rejected | Filmus & Ward, arXiv 2012 (rev. 2013), no journal venue stated. Gets 1−1/e under a matroid constraint; there is no matroid here. Its restricted-curvature extension (1−e^(−c))/c still assumes the function is submodular, and the non-submodular version of that result is already carried by Guarantees for Greedy Maximization of Non-submodular Functions. |  |
| 🟢 [Chamberlin–Courant Rule with Approval Ballots: Approximating the MaxCover Problem with Bounded Frequencies in FPT Time](https://jair.org/index.php/jair/article/view/11095) | 2017 | 36 | Rejected | Skowron & Faliszewski, Journal of Artificial Intelligence Research 60 (2017). Fixed-parameter-tractable approximation for multiwinner voting; the MaxCover shape matches but the machinery is exact committee election over known ballots, not noisy black-box search. |  |
| 🟢 [An Online Algorithm for Maximizing Submodular Functions](https://papers.nips.cc/paper/3569-an-online-algorithm-for-maximizing-submodular-functions) | 2008 | 299 | Rejected | Streeter & Golovin, NIPS 2008, pp. 1577–1584. Online no-regret setting, adjacent to the family already ruled out; here f is queried offline in batches, not as a sequential regret game. |  |
| 🟢 [Subset Selection by Pareto Optimization](https://papers.nips.cc/paper_files/paper/2015/hash/b4d168b48157c623fbd095b4a565b5bb-Abstract.html) | 2015 | 189 | Rejected | Qian, Yu & Zhou, NIPS 2015. Bi-objective evolutionary algorithm over k-subsets of a ground set, assuming an exact value oracle; the search unit here is a full team, and f is noisy. |  |
| 🟢 [Approximation Guarantees of Local Search Algorithms via Localizability of Set Functions](https://arxiv.org/abs/2006.01400) | 2020 | 1 | Rejected | Kaito Fujii, ICML 2020, PMLR v119 pp. 3327–3336. Localizability has to be derived from restricted strong concavity and smoothness of a differentiable relaxation, which a win rate measured by battles does not have; exact value oracle. |  |
| 🟢 [The Unreasonable Effectiveness of Greedy Algorithms in Multi-Armed Bandit with Many Arms](https://arxiv.org/abs/2002.10121) | 2020 | 40 | Rejected | Bayati, Hamidi, Johari & Khosravi, NeurIPS 2020. Bandit family ruled out 2026-08-20; the many-armed regret setting is not the search here. |  |
| 🟢 [Sample Efficiency Matters: A Benchmark for Practical Molecular Optimization](https://www.semanticscholar.org/paper/e318c18e6f3faa5c514e6b85270b2211603fc419) | 2022 | 234 | Rejected | owner's verdict 2026-09-02: does not contribute to this project. |  |
| 🟢 [Diffusion-based Evolutionary Optimization for 3D Multi-Objective Molecular Generation](https://www.semanticscholar.org/paper/e01f94b7bcf583241e5afe9d4a22bba4504778c9) | 2025 | 2 | Rejected | 2025 preprint, no venue, 1 citation (checked 2026-09-02). Rejected on uptake. |  |
| 🟢 [Reward-Guided Discrete Diffusion via Clean-Sample Markov Chain for Molecule and Biological Sequence Design](https://arxiv.org/abs/2602.09424) | 2026 | 7 | Rejected | ReALM-GEN (ICLR 2026 workshop). Rejected 2026-09-02 with the molecular-optimization family. |  |
| 🟢 [On the Design Space of Discrete Diffusion Online Adaptation for Molecular Optimization](https://arxiv.org/abs/2607.02834) | 2026 | 0 | Rejected | arXiv preprint, no venue listed. Rejected 2026-09-02 with the molecular-optimization family. |  |
| 🟢 [On failure modes in molecule generation and optimization](https://epub.jku.at/obvulioa/download/pdf/5687408?originalFilename=true) | 2020 | 145 | Rejected | Drug Discovery Today: Technologies 2020, was HIGH. Rejected 2026-09-02 with the molecular-optimization family. |  |
| 🟢 [Genetic algorithms are strong baselines for molecule generation](https://arxiv.org/abs/2310.09267) | 2023 | 45 | Rejected | arXiv preprint, no venue listed, was HIGH. Rejected 2026-09-02 with the molecular-optimization family. |  |
| 🟢 [HardFlow: Hard-Constrained Sampling for Flow-Matching Models via Trajectory Optimization](https://arxiv.org/abs/2511.08425) | 2025 | 13 | Rejected | ReALM-GEN (ICLR 2026 workshop), was LOW. Continuous optimal control; team legality is categorical, and Constrained Discrete Diffusion carries this job. |  |
| 🟢 [Trust-Region Noise Search for Black-Box Alignment of Diffusion and Flow Models](https://arxiv.org/abs/2603.14504) | 2026 | 0 | Rejected | ReALM-GEN (ICLR 2026 workshop), was LOW. Searches a continuous source-noise vector against many reward calls; battles are too costly and too noisy for that. | **Superseded 2026-09-07: accepted MED for selective reading**, with an active entry above. The full paper evaluates expensive rewards; applicability to categorical team generation and noisy battle ranking remains unvalidated. See [[Trust-Region Noise Search for Black-Box Alignment of Diffusion and Flow Models]] for the reassessment. |
| 🟢 [Discrete Diffusion Inference-Time Control with Nested Sequential Monte Carlo](https://arxiv.org/abs/2608.20123) | 2026 | 2 | Rejected | ReALM-GEN (ICLR 2026 workshop), was LOW. Feynman-Kac particle steering needs a reward on partial sequences; a half-built team cannot be battled. |  |
| 🔒 DDNO: Discrete Diffusion Noise Optimization (Eyring et al.) | 2026 |  | Rejected | ReALM-GEN (ICLR 2026 workshop), no open copy, ungraded. Discrete twin of Trust-Region Noise Search; noise-space search needs many reward calls and battles are too costly. |  |
| 🟢 [Diffusion Active Learning: Towards Data-Driven Experimental Design in Computed Tomography](https://arxiv.org/abs/2504.03491) | 2025 | 5 | Rejected | 2025 preprint, no venue, was MED. Inverse problem: acquisition picks the next measurement of a fixed object, informativeness-denominated. |  |
| 🟢 [Active Diffusion-Based Inference for Ill-Posed Inverse Problems under Incomplete Priors](https://arxiv.org/abs/2608.27080) | 2026 | 1 | Rejected | IJCAI-ECAI 2026 (venue was missing), was LOW. Inverse problem: recovers hidden true parameters, acquisition driven by posterior uncertainty. |  |
| 🟢 [Offline Model-Based Optimization: Comprehensive Review](https://arxiv.org/abs/2503.17286) | 2025 | 27 | Rejected | TMLR 2026 (Survey Certification), was HIGH. Offline MBO assumes no oracle access; Showdown gives one on demand. Owner's call 2026-09-02: no surveys. |  |
| 🟢 [When does return-conditioned supervised learning work for offline reinforcement learning?](https://arxiv.org/abs/2206.01079) | 2022 | 108 | Rejected | NeurIPS 2022, was HIGH. Owner's verdict 2026-09-02: an offline-RL policy result, no design generator; not useful. |  |
| [Real-time eSports Match Result Prediction](https://arxiv.org/abs/1701.03162) | 2017 | 50 | Rejected | Dota 2 win prediction from in-match features; predicts outcomes, does not construct teams. |  |
| [Win Prediction in Esports: Mixed-Rank Match Prediction in Multi-player Online Battle Arena Games](https://arxiv.org/abs/1711.06498) | 2017 | 37 | Rejected | same win-prediction family. |  |
| [League of Legends: Real-Time Result Prediction](https://arxiv.org/abs/2309.02449) | 2023 | 6 | Rejected | in-match LSTM prediction; no team construction. |  |
| [Using Machine Learning to Predict Game Outcomes Based on Player-Champion Experience in League of Legends](https://arxiv.org/abs/2108.02799) | 2021 | 33 | Rejected | predicts outcomes from player skill, explicitly "regardless of team composition". |  |
| [MobaQA: MOBA Games Prediction Based on Large Language Model Fine-Tuning](https://www.semanticscholar.org/paper/MobaQA%3A-MOBA-Games-Prediction-Based-on-Large-Nie-Wang/) | 2026 | 0 | Rejected | LLM fine-tuning to predict real-time win rate and match outcome; prediction, not search (IEEE Transactions on Games 2026). |  |

## Sources and reading notes

| Paper | Year | Citations | Grade | Venue and details | Notes |
| --- | --- | --- | --- | --- | --- |
| [[Reading List — notes and decision history]] |  |  |  | personal notes, original properties, prior decisions and the complete pre-reorganization list. |  |
| [[SPIGM 2026 — paper-by-paper relevance screening]] |  |  |  | 2 accepted MED readings; remaining screening decisions are historical. |  |
| [[EIML 2026 — paper-by-paper relevance screening]] |  |  |  | 2 MED additions; see the screening note for the assessment. |  |
| [[_Workshops index]] |  |  |  |  |  |
| [[VGC AI Competition (IEEE CoG)]] |  |  |  |  |  |
| [[PokeAgent Challenge (NeurIPS 2025)]] |  |  |  |  |  |
| [[Finding the Frame — papers relevant to the counter-team search]] |  |  |  |  |  |
| [[Reading path — neural CO for team selection]] |  |  |  |  |  |
| [[Teaching/RESOURCES\|Teaching resources]] |  |  |  | analogue-sweep grades and verdicts. |  |
| [[Citation]] |  |  |  |  |  |
