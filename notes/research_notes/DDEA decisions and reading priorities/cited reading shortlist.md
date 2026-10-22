# DDEA citations and reading priorities for Pokémon counter-search

Checked on 2026-10-20. This is a screened bibliography and a selective reading order. Presence columns record the list before integration. The final integration added only Böther et al. and Bengio et al. at MED, as recorded in the dated reading-history note. No measured benefits for Pokémon are claimed.

## Which cited works answer the project's current questions?

### Takeaway

The strongest immediate addition is Böther et al.'s evaluation paper. Bengio, Lodi and Prouvost provide useful selective methodology. ReEvo is relevant to learned heuristic operators, but this project already has a lesson on it. No candidate below warrants an automatic HIGH grade, and later reward-training readings should not displace current search and evaluation work. These priorities follow the current [reading-list rules](<../../Papers/Reading List.md>), its existing entries, and the [project's win-rate objective](<../../What counts as a good counter pokemon team.md>).

### Cited findings

The original DDEA v1 HTML contains **29 bibliography entries**, counted by extracting its `ltx_bibitem` elements. The inventory below includes every extracted entry once, in three screening groups. A source link on each row identifies the original citation. This is a reference-level screen, not a claim to have fully read all 29 papers. The stronger candidates received additional primary-source checks below. [DDEA bibliography](https://arxiv.org/html/2510.08627v1#bib)

The local list prioritizes search-loop design, baselines and evaluation before fine-tuning or reward integration. Its current downgrade rule gives LOW priority to continuous-only or toy studies without a categorical, team-like search setting. The current list includes DDEA as MED, while DIFUSCO, DiffUCO and T2T are LOW. Preserve those decisions unless the owner changes them. [Current reading list](<../../Papers/Reading List.md>)

#### Higher relative fit, with selective reading at MED rather than HIGH

The fit judgments in the last column are project inferences, not classifications made by DDEA's authors.

| Ref. | Cited work | Present at screening start? | Project screening judgment |
| --- | --- | --- | --- |
| 1 | Bengio, Lodi and Prouvost, 2021. *Machine learning for combinatorial optimization: A methodological tour d'horizon*. [Citation](https://arxiv.org/html/2510.08627v1#bib.bib1) | Absent | MED selective methodology for separating learned proposals, optimizer performance, training cost and generalization. |
| 13 | Ye et al., 2024. *ReEvo: Large language models as hyper-heuristics with reflective evolution*. [Citation](https://arxiv.org/html/2510.08627v1#bib.bib13) | Absent from list; already covered in Teaching lesson 0005 | MED when the project resumes generating or evolving heuristic code. Not a newly discovered reading. |
| 27 | Böther et al., 2022. *What's wrong with deep learning in tree search for combinatorial optimization*. [Citation](https://arxiv.org/html/2510.08627v1#bib.bib27) | Absent | MED and first immediate read. Tests whether learned guidance adds anything beyond search and problem-specific improvement. |

#### Context or conditional reading

| Ref. | Cited work | Present at screening start? | Project screening judgment |
| --- | --- | --- | --- |
| 3 | Sun and Yang, 2023. *DIFUSCO: Graph-based diffusion solvers for combinatorial optimization*. [Citation](https://arxiv.org/html/2510.08627v1#bib.bib3) | Present, LOW | Keep as DDEA backbone context. Do not duplicate or upgrade merely because DDEA builds on it. |
| 4 | Yu et al., 2024. *DISCO: Efficient diffusion solver for large-scale combinatorial optimization problems*. [Citation](https://arxiv.org/html/2510.08627v1#bib.bib4) | Absent | Conditional inference-engineering context if proposal inference becomes the measured bottleneck. Not a solution to expensive noisy battles by itself. |
| 5 | Li et al., 2023. *From distribution learning in training to gradient search in testing for combinatorial optimization*. [Citation](https://arxiv.org/html/2510.08627v1#bib.bib5) | Present, LOW | Keep existing grade. Test-time gradients need an accessible differentiable objective or surrogate; battle win rate supplies neither directly. |
| 9 | Sanokowski, Hochreiter and Lehner, 2024. *A diffusion model framework for unsupervised neural combinatorial optimization*. [Citation](https://arxiv.org/html/2510.08627v1#bib.bib9) | Present, LOW | Keep existing grade. Unsupervised white-box graph objectives are a different setting from noisy simulator scoring. |
| 11 | Zhang et al., 2023. *Let the flows tell: Solving graph combinatorial optimization problems with GFlowNets*. [Citation](https://arxiv.org/html/2510.08627v1#bib.bib11) | Absent | MED for a later reward-integration question about sampling several valuable solutions. Training, reward cost and team representation still need new work. |
| 15 | Georgiev et al., 2023. *Neural algorithmic reasoning for combinatorial optimisation*. [Citation](https://arxiv.org/html/2510.08627v1#bib.bib15) | Absent | Architecture context if learning a graph algorithm becomes an active design task. Not a current team-search baseline. |
| 19 | Bäck, Fogel and Michalewicz, 1997. *Handbook of Evolutionary Computation*. [Citation](https://arxiv.org/html/2510.08627v1#bib.bib19) | Absent | Reference lookup for established EA machinery. A broad handbook is not a focused addition to the immediate queue. |
| 22 | Liang et al., 2024 online. *A survey of surrogate-assisted evolutionary algorithms for expensive optimization*. [Citation](https://arxiv.org/html/2510.08627v1#bib.bib22) | Absent | Potential surrogate/model-management context. Full text was not accessible in this screening. Correct the citation metadata before adding; do not claim a full-paper grade. |
| 24 | Michalski, 2000. *Learnable evolution model: Evolutionary processes guided by machine learning*. [Citation](https://arxiv.org/html/2510.08627v1#bib.bib24) | Absent | Historical learned-proposal context. Do not promote on evolutionary-step speedups alone; the current list's benchmark-fit rule still applies. |
| 25 | Fischetti and Lodi, 2003. *Local branching*. [Citation](https://arxiv.org/html/2510.08627v1#bib.bib25) | Absent | Useful origin of the teacher's Hamming-neighborhood idea. Exact MIP neighborhoods do not turn simulator win rate into a linear objective. |
| 29 | Berthold and Csizmadia, 2021. *The confined primal integral: A measure to benchmark heuristic MINLP solvers against global MINLP solvers*. [Citation](https://arxiv.org/html/2510.08627v1#bib.bib29) | Absent | Conditional metric reference for speed versus final quality. Does not resolve battle-score uncertainty or validate DDEA's inconsistent PI reporting. |

#### Lower fit for the current reading queue

| Ref. | Cited work | Present? | Why it does not enter the immediate queue |
| --- | --- | --- | --- |
| 2 | Jin et al., 2024. *A unified framework for combinatorial optimization based on graph neural networks*. [Citation](https://arxiv.org/html/2510.08627v1#bib.bib2) | Absent | General graph representation and decomposition, without a current identified team-search representation decision. |
| 6 | Khalil et al., 2016. *Learning to branch in mixed integer programming*. [Citation](https://arxiv.org/html/2510.08627v1#bib.bib6) | Absent | Learns decisions inside a MIP solver with explicit optimization structure, rather than proposing teams to a simulator. |
| 7 | Kool, van Hoof and Welling, 2019. *Attention, learn to solve routing problems!*. [Citation](https://arxiv.org/html/2510.08627v1#bib.bib7) | Absent | Constructive routing policy with task-specific cheap objective. No current need to replace the team's proposal architecture with this routing model. |
| 8 | Karalias and Loukas, 2020. *Erdös goes neural: An unsupervised learning framework for combinatorial optimization on graphs*. [Citation](https://arxiv.org/html/2510.08627v1#bib.bib8) | Absent | Uses accessible graph-objective structure. That structure is unavailable for direct battle scoring. |
| 10 | Vinyals, Fortunato and Jaitly, 2015. *Pointer networks*. [Citation](https://arxiv.org/html/2510.08627v1#bib.bib10) | Absent | Foundational architecture rather than a focused answer about proposal benefit, noisy evaluations or search budget. |
| 12 | Kemminer et al., 2024. *Configuring mixed-integer programming solvers for large-scale instances*. [Citation](https://arxiv.org/html/2510.08627v1#bib.bib12) | Absent | MIP-configuration setting. Current SMAC3 and categorical optimization readings already address a closer model-based baseline question. |
| 14 | Scavuzzo et al., 2024. *Machine learning augmented branch and bound for mixed integer linear programming*. [Citation](https://arxiv.org/html/2510.08627v1#bib.bib14) | Absent | Useful solver internals, but team win rate has no supplied MILP formulation or branch-and-bound bounds. |
| 16 | Khalil et al., 2017. *Learning combinatorial optimization algorithms over graphs*. [Citation](https://arxiv.org/html/2510.08627v1#bib.bib16) | Absent | General graph-policy background. Deferred unless a graph policy is a concrete comparison against the current generator. |
| 17 | Ho, Jain and Abbeel, 2020. *Denoising diffusion probabilistic models*. [Citation](https://arxiv.org/html/2510.08627v1#bib.bib17) | Absent | Foundational continuous image-diffusion paper. Do not add simply because diffusion occurs in DDEA's title. |
| 18 | Huang, Sun and Yang, 2023. *Accelerating diffusion-based combinatorial optimization solvers by progressive distillation*. [Citation](https://arxiv.org/html/2510.08627v1#bib.bib18) | Absent | Inference acceleration matters only after measuring it as the project bottleneck. Not a noisy-fitness allocation method. |
| 20 | Bäck and Khuri, 1994. *An evolutionary heuristic for the maximum independent set problem*. [Citation](https://arxiv.org/html/2510.08627v1#bib.bib20) | Absent | MIS-specific permutation and decoder precedent. Its graph objective and representation need a separate transfer argument. |
| 21 | Liu, Sakamoto and Shimamoto, 1996. *A genetic algorithm for maximum independent set problems*. [Citation](https://arxiv.org/html/2510.08627v1#bib.bib21) | Absent | MIS-specific binary encoding and constraint-handling precedent. No automatic reason to read before closer team/deck evolution papers. |
| 23 | Bai, Cheng and Jin, 2023. *Evolutionary reinforcement learning: A survey*. [Citation](https://arxiv.org/html/2510.08627v1#bib.bib23) | Absent | Broader policy-learning strand. The current team-search comparison uses a fixed battle policy rather than jointly evolving it. |
| 26 | Vaswani et al., 2017. *Attention is all you need*. [Citation](https://arxiv.org/html/2510.08627v1#bib.bib26) | Absent | Embedding and architecture background, not an immediate experiment about better counter teams. |
| 28 | Lamm et al., 2017. *Finding near-optimal independent sets at scale*. [Citation](https://arxiv.org/html/2510.08627v1#bib.bib28) | Absent | The actual KaMIS reference. Its benchmark win does not establish that its graph reductions transfer to Pokémon. The separate KaMIS audit covers the mechanism. |

### Inferences

- The group sizes are 3 higher-fit, 11 contextual and 15 lower-fit entries, totaling the 29 extracted citations. These are screening categories rather than new HIGH/MED/LOW decisions for every cited work. At screening start, the only existing cited-paper rows found in the list were DIFUSCO, DiffUCO and T2T, all LOW. [DDEA bibliography](https://arxiv.org/html/2510.08627v1#bib), [reading list](<../../Papers/Reading List.md>)
- Current priority favors a controlled test of the learned proposal's contribution. That makes an audit of random guidance, repair and search more useful now than another neural architecture. This recommendation follows the evaluation mechanism in Böther et al., not a claim that their MIS finding already holds for Pokémon. [Böther et al., §3.1–3.2](https://arxiv.org/pdf/2201.10494)
- The list already has the owner-selected *Constrained Discrete Black-Box Optimization using Mixed-Integer Programming* as a candidate source for a team-recombination teacher. It also has categorical BO, SMAC3, Design-Bench and decision-focused ranking references. New MIP or broad surrogate rows should add a specific question rather than repeat those jobs. [Current list, structured surrogates and search sections](<../../Papers/Reading List.md>)

### Gaps

- No citation counts were refreshed. Existing counts belong to the list's recorded snapshot and are not evidence of project fit. Every proposed new row below uses `not checked` for that field. [Current list metadata](<../../Papers/Reading List.md>)
- The 29-entry inventory uses DDEA's cited metadata for works not shortlisted. It does not independently validate every old venue, author initial or page range. In particular, the 1996 MIS GA entry is retained as cited rather than expanded with unverified publication details. [Original entry](https://arxiv.org/html/2510.08627v1#bib.bib21)
- Full-text access to the Liang survey was unavailable. Its publisher supplies the abstract, author list and publication metadata, but the body is subscription content. [Publisher record](https://link.springer.com/article/10.1007/s41965-024-00165-w)

## Which additions are worth selective reading, and what can transfer?

### Takeaway

Recommend two new current-stage readings: Böther et al. first, then selected Bengio et al. sections. ReEvo can be indexed as previously covered work, while GFlowNets and the surrogate survey remain conditional context. This is a five-item shortlist with different reading stages, not a recommendation to add five immediate tasks. [Local sequencing rule](<../../Papers/Reading List.md>)

### Cited findings

#### 1. Böther et al. for auditing the learned proposal

**Metadata.** Maximilian Böther, Otto Kißig, Martin Taraz, Sarel Cohen, Karen Seidel and Tobias Friedrich. *What's Wrong with Deep Learning in Tree Search for Combinatorial Optimization*. ICLR 2022. arXiv:2201.10494. The primary arXiv record states ICLR acceptance, and the paper itself identifies the conference. [Primary record](https://arxiv.org/abs/2201.10494), [paper](https://arxiv.org/pdf/2201.10494)

**Question it answers.** Does learned guidance improve the actual search, or do reduction, local search and other algorithmic components explain the result? In the tree-search configurations studied, replacing GCN outputs with random values gives similar performance once reduction and local search are active. The same paper finds the GNN useful in the separate Learning What to Defer solver. Its lesson is an ablation design, not a universal anti-learning result. Read §3.1, especially the random-output comparison, and §3.2. [Paper, pp. 5–9](https://arxiv.org/pdf/2201.10494)

**Proposed transfer and limit.** Compare the current proposal against random legal proposals while keeping decoder/repair, selection, incumbent handling and battle budget controlled. Then vary those other components separately. Pokémon still requires repeated noisy battle evaluation and fresh validation; deterministic MIS sizes do not solve that issue. This experiment is an inference from the paper's design. [Source experiment](https://arxiv.org/pdf/2201.10494), [project objective](<../../What counts as a good counter pokemon team.md>)

**Reading decision.** First immediate addition, MED selective reading. It was absent at screening start and is now added. Do not claim a measured win-rate gain or override the user's existing search priorities. [Current list](<../../Papers/Reading List.md>)

#### 2. Bengio, Lodi and Prouvost for choosing the right success metric

**Metadata.** Yoshua Bengio, Andrea Lodi and Antoine Prouvost. *Machine Learning for Combinatorial Optimization: A Methodological Tour d'Horizon*. European Journal of Operational Research 290, issue 2, pp. 405–421, 2021. DOI 10.1016/j.ejor.2020.07.063. Open preprint arXiv:1811.06128, revised 2020. [Primary preprint](https://arxiv.org/abs/1811.06128), [author publication record](https://www.gerad.ca/fr/people/andrea-lodi/publications)

**Question it answers.** Where should learning enter an optimizer, what distribution should it generalize to, and why does improving a learning loss not automatically improve optimization performance? Section 3.2 separates end-to-end generation, algorithm configuration and learning alongside an optimizer. Section 4.2 explicitly distinguishes surrogate learning objectives from the algorithm's actual performance metric. Sections 4.3 and 4.6 discuss held-out generalization and offline learning costs. [Paper, §§3.2, 4.2, 4.3 and 4.6](https://arxiv.org/pdf/1811.06128)

**Proposed transfer and limit.** Report the strength of teams discovered by a generator/search combination, rather than treating likelihood, teacher imitation or predictor accuracy as the final outcome. Specify the training and evaluation distributions and retain offline cost. The survey does not supply a ready-made method for stochastic battles or a categorical team decoder. [Paper](https://arxiv.org/pdf/1811.06128), [project objective](<../../What counts as a good counter pokemon team.md>)

**Reading decision.** Second, MED selective methodology. It was absent at screening start and is now added. Read the named sections rather than adding the entire survey as an urgent task. [Current list](<../../Papers/Reading List.md>)

#### 3. ReEvo for the existing heuristic-generation strand

**Metadata.** Haoran Ye, Jiarui Wang, Zhiguang Cao, Federico Berto, Chuanbo Hua, Haeyeon Kim, Jinkyoo Park and Guojie Song. *ReEvo: Large Language Models as Hyper-Heuristics with Reflective Evolution*. NeurIPS 2024. arXiv:2402.01145. Proceedings DOI 10.52202/079017-1381. [Official proceedings](https://papers.nips.cc/paper_files/paper/2024/hash/4ced59d480e07d290b6f29fc8798f195-Abstract-Conference.html), [paper](https://papers.nips.cc/paper_files/paper/2024/file/4ced59d480e07d290b6f29fc8798f195-Paper-Conference.pdf)

**Question it answers.** Can an LLM evolve reusable heuristic code using comparative evaluation and short- and long-term reflections? ReEvo's individuals are code snippets satisfying a function signature. It evolves heuristics and genetic operators, rather than merely sampling finished solutions. Its experiments include combinatorial problems and black-box prompting. In that prompting condition, problem descriptions and variable names are concealed; this is not the same claim as solving a noisy simulator objective. The paper argues for counting heuristic evaluations because evaluations can dominate model-query costs. [Paper, §3, black-box experiments and evaluation discussion](https://arxiv.org/pdf/2402.01145)

**Proposed transfer and limit.** A possible use is evolving a team-editing or recombination function and judging its discovered teams under a controlled battle budget. That adds an outer search over code, its own validation problem and extra evaluation cost. It is different from asking whether the current diffusion generator beats a baseline. [Paper](https://arxiv.org/pdf/2402.01145)

**Reading decision.** MED when the heuristic-code strand resumes. The current reading list lacks the row, but Teaching/NOTES.md records lesson 0005 on ReEvo and an existing reference explanation. Index that prior work rather than present the paper as a new discovery. [Teaching notes](<../../Teaching/NOTES.md>), [current list](<../../Papers/Reading List.md>)

#### 4. Let the Flows Tell for a later quality/diversity distribution question

**Metadata.** Dinghuai Zhang, Hanjun Dai, Esmeralda S. Whitammer, Aaron Courville, Yoshua Bengio and Ling Pan, using the current primary arXiv author record. *Let the Flows Tell: Solving Graph Combinatorial Optimization Problems with GFlowNets*. NeurIPS 2023. arXiv:2305.17010. The proceedings abstract uses the shorter title *Solving Graph Combinatorial Problems with GFlowNets*. The stable arXiv ID and proceedings paper identify the work despite the title variation. [Current primary record](https://arxiv.org/abs/2305.17010), [official proceedings](https://proceedings.neurips.cc/paper_files/paper/2023/hash/27571b74d6cd650b8eb6cf1837953ae8-Abstract.html)

**Question it answers.** How can a constructive policy sample multiple high-valued terminal solutions, rather than optimize one deterministic answer or confuse trajectory diversity with solution diversity? The paper sets a terminal reward distribution proportional to exp(−E(x)/T), constructs graph-specific MDPs, and studies training efficiency. Its MIS construction generates maximal independent sets and uses set size in the reward. [Paper, §§2–3](https://arxiv.org/pdf/2305.17010)

**Proposed transfer and limit.** It offers an alternative way to train a proposal distribution over completed teams. It requires a suitable team construction process, a reward transformation and an evaluation/training budget. It does not demonstrate cheap or calibrated handling of noisy battle rewards. This belongs to the deferred reward-integration stage, not ahead of current search comparisons. [Paper](https://arxiv.org/pdf/2305.17010), [local sequencing rule](<../../Papers/Reading List.md>)

**Reading decision.** Optional MED for later reward integration, currently absent. Do not treat the paper's diversity motivation as an already demonstrated improvement in this project's team diversity or win rate. [Current list](<../../Papers/Reading List.md>)

#### 5. Liang et al. as access-limited surrogate context

**Metadata.** Jing Liang, Yahang Lou, Mingyuan Yu, Ying Bi and Kunjie Yu. *A survey of surrogate-assisted evolutionary algorithms for expensive optimization*. Published online 13 August 2024, Journal of Membrane Computing 7, pp. 108–127, 2025 issue. The matching publisher DOI is **10.1007/s41965-024-00165-w**. DDEA instead prints 10.1007/s41965-024-00155-x. Use the matching publisher record rather than copy that conflicting identifier. [Publisher](https://link.springer.com/article/10.1007/s41965-024-00165-w), [DDEA citation](https://arxiv.org/html/2510.08627v1#bib.bib22)

**Question it may answer.** The publisher abstract identifies surrogate construction and model management among its topics, which are relevant when deciding which candidate receives an expensive true evaluation. That is more directly related to battle-allocation cost than DDEA's cheap exact cardinality evaluations. This is a scope inference from the accessible abstract, not a verified claim about its detailed recommendations. [Publisher abstract](https://link.springer.com/article/10.1007/s41965-024-00165-w)

**Transfer limit and reading decision.** Full text was not accessible. Keep it as context awaiting access and assessment against the list's continuous-only downgrade rule. Do not assign HIGH or claim that the survey establishes a categorical noisy-team method. The current list already has closer categorical BO and structured surrogate references, so this should not displace them. [Publisher access notice](https://link.springer.com/article/10.1007/s41965-024-00165-w), [current list](<../../Papers/Reading List.md>)

### Inferences

- LEM is useful historical context for learning from better and worse populations, but its origin alone does not make it a more useful current reading than the three candidates above. The original author-hosted paper supports the learned-proposal mechanism. It does not establish battle-win-rate or team-like categorical results. [Michalski, original author PDF](https://www.mli.gmu.edu/papers/96-2000/00-2.pdf)
- Local branching explains the source of a Hamming-neighborhood restriction around a known solution. The original method explores neighborhoods using a MIP solver with a known optimization model. A team's legality can be encoded separately, but actual simulator win rate is not thereby a known MIP objective. The list's existing NN+MILP reading is a closer starting point for a learned acquisition teacher. [Local branching publisher abstract](https://link.springer.com/article/10.1007/s10107-003-0395-5), [existing structured-surrogate row](<../../Papers/Reading List.md>)
- The confined primal integral paper is a useful original source for the time/quality tradeoff: it uses exponential decay and provides a formulation that need not assume a fixed reference solution. That does not repair DDEA's conflicting formula and values, and a noisy battle incumbent would require independent uncertainty treatment. [Author preprint](https://optimization-online.org/wp-content/uploads/2020/07/7910.pdf)

### Gaps

- No shortlisted paper is evidence that the proposed transfer improves Pokémon battle win rate. Proposed uses remain hypotheses requiring controlled battles against the established opponents. [Project objective](<../../What counts as a good counter pokemon team.md>)
- No claim is made that ReEvo's black-box prompting validates robustness to noisy function values. Hiding task semantics from an LLM and observing a stochastic objective are separate conditions. [ReEvo paper](https://arxiv.org/pdf/2402.01145)
- No new grade is assigned to the access-limited Liang survey from its abstract alone. [Publisher access restriction](https://link.springer.com/article/10.1007/s41965-024-00165-w)

## What should the reading-list integrator add or preserve?

### Takeaway

Add a small set, keep existing rows and grades, and preserve the distinction between previously covered work and genuinely new reading. The rows below match the six-column list format. Only the first two rows were integrated. The optional rows remain recommendations rather than queued tasks. [Current format and rules](<../../Papers/Reading List.md>), [integration decision](<../../Papers/Reading List — notes and decision history.md>)

### Cited findings

Suggested current-stage rows, subject to the integrator preserving the latest file bytes:

| Paper | Year | Citations | Grade | Venue and details | Notes |
| --- | --- | --- | --- | --- | --- |
| [What's Wrong with Deep Learning in Tree Search for Combinatorial Optimization](https://arxiv.org/abs/2201.10494) | 2022 | not checked | MED | 🟢 ICLR 2022. First selective read from DDEA's references: §3.1–3.2 tests learned versus random guidance while separating reductions and local search. Adapt the ablation to fixed battle budgets; deterministic MIS scores do not address battle noise. | |
| [Machine Learning for Combinatorial Optimization: A Methodological Tour d'Horizon](https://arxiv.org/abs/1811.06128) | 2021 | not checked | MED | 🟢 EJOR 290, 405–421. Selective reading: §§3.2, 4.2, 4.3 and 4.6 distinguish learned-operator fit, actual search performance, held-out generalization and offline cost. DOI 10.1016/j.ejor.2020.07.063. | |

Optional indexing of existing coverage, placed in heuristic generation rather than ahead of current baseline work:

| Paper | Year | Citations | Grade | Venue and details | Notes |
| --- | --- | --- | --- | --- | --- |
| [ReEvo: Large Language Models as Hyper-Heuristics with Reflective Evolution](https://papers.nips.cc/paper_files/paper/2024/hash/4ced59d480e07d290b6f29fc8798f195-Abstract-Conference.html) | 2024 | not checked | MED | 🟢 NeurIPS 2024. Previously covered in Teaching lesson 0005. Evolve reusable heuristic or genetic-operator code with evaluated reflections; read when the heuristic-code strand resumes. Black-box prompting does not establish noisy battle-evaluation robustness. | |

Optional later-stage row, only if the integrator wants a deferred reward-distribution reference:

| Paper | Year | Citations | Grade | Venue and details | Notes |
| --- | --- | --- | --- | --- | --- |
| [Let the Flows Tell: Solving Graph Combinatorial Optimization Problems with GFlowNets](https://arxiv.org/abs/2305.17010) | 2023 | not checked | MED | 🟢 NeurIPS 2023. Deferred reward-integration reading: sample diverse high-valued graph solutions from a learned terminal-reward distribution. A team construction process, noisy battle reward treatment and evaluation budget remain untested adaptations. | |

The Liang survey should remain a note awaiting full-text access rather than a confident new MED row. If the owner specifically wants it indexed anyway, an honest row is:

| Paper | Year | Citations | Grade | Venue and details | Notes |
| --- | --- | --- | --- | --- | --- |
| [A survey of surrogate-assisted evolutionary algorithms for expensive optimization](https://link.springer.com/article/10.1007/s41965-024-00165-w) | 2024 online; 2025 issue | not checked | ungraded | 🔒 Journal of Membrane Computing 7, 108–127. Publisher abstract and metadata verified; full text unavailable in this screening. Potential model-management context, with categorical and noisy-evaluation fit not yet assessed. Correct DOI ends 00165-w. | |

Preserve existing DIFUSCO, DiffUCO and T2T rows as LOW. DDEA is itself already listed as MED. None needs a duplicate row. The existing ReEvo lesson means an added row records prior coverage rather than a new reading commitment. [Reading list](<../../Papers/Reading List.md>), [Teaching notes](<../../Teaching/NOTES.md>)

### Inferences

- The most useful current question is whether a better proposal improves validated win rate per stated budget after controlling the rest of search. The first two rows support the design and interpretation of that test. ReEvo and GFlowNets answer different later questions. [Böther et al.](https://arxiv.org/pdf/2201.10494), [Bengio et al.](https://arxiv.org/pdf/1811.06128), [current sequencing rule](<../../Papers/Reading List.md>)
- KaMIS's result should motivate checking strong baselines and decomposition of the neural contribution. It does not justify adding MIS-specific engineering at HIGH. The original DDEA citations to classical evolutionary MIS are exactly Bäck and Khuri 1994 and Liu et al. 1996; the KaMIS reference is Lamm et al. 2017. [DDEA refs. 20–21](https://arxiv.org/html/2510.08627v1#bib.bib20), [DDEA ref. 28](https://arxiv.org/html/2510.08627v1#bib.bib28)

### Gaps

- The active reading list is shared mutable state. Its integrator must reread its current contents immediately before applying approved additions and preserve unrelated edits. This note supplies proposed rows without taking that shared write. [Project shared-state instructions](<../../AGENTS.md>)
- No newly researched citation counts, full-text access claims, Pokémon performance gains or consensus relevance rankings are supplied. The recommendation is a bounded project judgment with the sources and limits stated above.
