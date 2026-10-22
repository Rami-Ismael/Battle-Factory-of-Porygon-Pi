# GECCO for counter search

Checked October 20, 2026. This is a targeted review for the Pokémon counter-search project, not an exhaustive proceedings survey. Paper priorities and venue-fit judgments are assessments.

GECCO is worth following for surrogate-assisted search, discrete candidate generation, and evaluation under noise. The most useful next step is a focused methods check tied to the current search loop. This review found no evidence that a GECCO method already improves this project's battle win rate.

## What GECCO covers

GECCO is the Genetic and Evolutionary Computation Conference, sponsored by ACM SIGEVO. Its scope includes evolutionary algorithms, combinatorial optimisation, machine learning, hybrid search, benchmarking, and reproducibility. It is both a source of methods and a possible publication venue for this project. Publication fit depends on the research contribution and evidence. [Official conference site](https://gecco-2027.sigevo.org/HomePage), [SIGEVO conference portal](https://sig.sigevo.org/GECCOs).

The existing project notes already identify useful GECCO papers. The reading list rates the 2022 surrogate-assisted Hearthstone paper and the 2019 sliding-boundaries MAP-Elites paper HIGH. It lists the 2026 diffusion-EDA paper as MED. The workshop index's LOW-MED assessment concerns the 2026 workshop selection, particularly the absence of a dedicated quality-diversity workshop. That assessment does not establish the value of every main-conference paper. [Existing reading list](</Users/ramiismael/Documents/yakumsi-vault/Personal Project/Using Advance method search in the whole search of possible vgc pokemon format to find the right counter to a pokemon team/Papers/Reading List.md:58>), [workshop assessment](</Users/ramiismael/Documents/yakumsi-vault/Personal Project/Using Advance method search in the whole search of possible vgc pokemon format to find the right counter to a pokemon team/Workshops/_Workshops index.md:93>).

## The material worth checking

| Material | What it contributes | Fit and limitation |
| --- | --- | --- |
| [Deep Surrogate Assisted MAP-Elites for Automated Hearthstone Deckbuilding, GECCO 2022](https://arxiv.org/html/2112.03534v4) | Trains a surrogate, searches with its predictions, and checks proposed candidates in simulation. | HIGH as an existing method donor. Its search objective is average final health difference. QD-score gains cannot be read as battle win-rate gains. |
| [Extract-QD Framework, GECCO 2025](https://arxiv.org/html/2502.06585v1) | Re-evaluates archive members, accumulates observations, and retains multiple candidates per archive cell under noisy evaluation. | LOW for direct reuse under the reading list's continuous-only and toy-evidence rule, corrected in the follow-up on October 20. Conditional background for a QD branch; its evidence comes from robotics and control tasks with corrected QD-score as the main measure. |
| [Addressing Combinatorial Optimization with Estimation of Distribution Algorithms Based on Diffusion Models, GECCO 2026](https://doi.org/10.1145/3795095.3805121) | A directly relevant reading lead for learned candidate distributions in combinatorial search. | Already MED in the project list. Acceptance, authors, and ECOM track are verified; algorithm and result claims remain unverified in this review. |
| [Evolutionary Computation meets Machine Learning for Combinatorial Optimisation, GECCO 2026 tutorial](https://gecco-2026.sigevo.org/Tutorial?itemId=8207) | Explains ways to use learning in combinatorial search, including learned heuristics and decision policies. | Useful orientation when choosing what to learn. The public description supplies no VGC performance evidence. |

The Hearthstone paper remains the closest verified architecture in this shortlist. It evaluates decks using 200 games across six opponent decks and also reports win percentage. Differences in maximum win percentage among its surrogate variants were not statistically significant. Borrow the loop of prediction followed by simulation checks while retaining the project's battle win-rate target. [Paper, sections 3 through 5](https://arxiv.org/html/2112.03534v4).

Extract-QD offers a relevant question for a QD branch: whether repeated evaluations prevent lucky teams from occupying the archive. It remains background rather than an active reading-list addition under the current evidence rule. Any test must charge those repeat battles to the same total allowance. The paper does not establish the best repeat-evaluation allocation for VGC, and it does not justify introducing an archive when the goal is one strong counter. [Extract-QD paper](https://arxiv.org/html/2502.06585v1), [author implementation](https://github.com/adaptive-intelligent-robotics/Uncertain_Quality_Diversity).

The diffusion-EDA paper is by Roberto Santana and José Antonio Lozano. The official accepted-paper list confirms it in Evolutionary Combinatorial Optimization and Metaheuristics. ACM access failed during this review, and no accessible author copy was found. Obtain the full text before deciding whether its selection rule, representation, constraint handling, or training cost merits implementation. Its title alone cannot establish those details. [Official accepted-paper list](https://gecco-2026.sigevo.org/Accepted%2BPapers).

## SAEOpt is a particularly relevant workshop

The 2026 programme includes the Workshop on Surrogate-Assisted Evolutionary Optimisation, SAEOpt. Its scope explicitly covers expensive objectives, predicted candidate rankings, model management, multiple evaluation fidelities, feasible-region prediction, and performance assessment. It also acknowledges limited work on combinatorial problems. That makes it a useful community to inspect for the project's candidate-scoring question. Its listed application successes do not demonstrate transfer to Pokémon. [Official SAEOpt description](https://gecco-2026.sigevo.org/Workshop?itemId=8255).

For a possible paper, choose a track by the contribution. The listed scopes suggest these routes:

| Contribution | Track to investigate |
| --- | --- |
| A better search method for legal, discrete teams | Evolutionary Combinatorial Optimization and Metaheuristics, ECOM |
| A learned scorer or generator that improves evolutionary search with expensive evaluations | Learning for Evolutionary Computation, L4EC |
| A hybrid of evolutionary search with tree search, coevolution, or different evaluation fidelities | General Evolutionary Computation and Hybrids, GECH |
| A reusable benchmark, evaluation method, or software contribution | Benchmarking, Benchmarks, Software, and Reproducibility, BBSR |

These are fit judgments based on published scopes, not acceptance predictions. [2026 tracks](https://gecco-2026.sigevo.org/Tracks), [L4EC scope](https://gecco-2027.sigevo.org/Track?itemId=2671), [GECH scope](https://gecco-2027.sigevo.org/Track?itemId=57), [BBSR scope](https://gecco-2027.sigevo.org/Track?itemId=2672). The partly updated 2027 site requires the date caveat below.

## What would count as useful project evidence

The project's meaning of better remains higher battle win rate. QD coverage and diversity can be diagnostic measurements, but they do not replace that objective. The notes leave the MAP-Elites comparison and archive descriptors unresolved, so Extract-QD belongs to a secondary investigation at present. [MAP-Elites comparison](</Users/ramiismael/Documents/yakumsi-vault/Personal Project/Using Advance method search in the whole search of possible vgc pokemon format to find the right counter to a pokemon team/MAP Elite.md:6>), [archive descriptors](</Users/ramiismael/Documents/yakumsi-vault/Personal Project/Using Advance method search in the whole search of possible vgc pokemon format to find the right counter to a pokemon team/Quality Diversity.md:18>).

The existing scorer-comparison plan already specifies the controls a borrowed method needs. Preserve the applicable run's starting distribution, task fields, legality rules, opponent distribution and weights, battle policy, simulator, format, and seed schedule. Keep candidate-generation settings fixed for a scorer-only comparison. For a generator comparison, change the generator deliberately and account for both battle evaluations and elapsed time. Select finalists before fresh evaluation and do not select again using the final scores. [Project comparison plan](</Users/ramiismael/Documents/yakumsi-vault/Personal Project/Using Advance method search in the whole search of possible vgc pokemon format to find the right counter to a pokemon team/Cross entropy battle versus surrogate scoring experiment.md:21>).

Keep three checks distinct. A team-level validation split tests a surrogate on unseen candidate teams. Fresh battles against the original opponent pool test a selected team's result with new randomness. A held-out opponent pool tests opponent generalisation. Reusing one of these as evidence for another would overstate the result. This review assigns no new comparison budget. [Validation requirements](</Users/ramiismael/Documents/yakumsi-vault/Personal Project/Using Advance method search in the whole search of possible vgc pokemon format to find the right counter to a pokemon team/Cross entropy battle versus surrogate scoring experiment.md:33>), [recorded opponent-generalisation context](</Users/ramiismael/Documents/yakumsi-vault/Personal Project/Using Advance method search in the whole search of possible vgc pokemon format to find the right counter to a pokemon team/research_notes/Next paper for counter search/project_bottleneck.md:12>).

My recommendation is to retain the Hearthstone surrogate loop as the reference architecture, obtain the 2026 diffusion-EDA paper before promoting it, and read Extract-QD when an archive experiment becomes concrete. SAEOpt and L4EC are the most direct communities to follow for the expensive-scoring question.

## Confirmed conference details

| Edition | Dates | Location and format |
| --- | --- | --- |
| Most recent, GECCO 2026 | July 13 to 17, 2026 | San José, Costa Rica, hybrid |
| Next, GECCO 2027 | July 12 to 16, 2027 | Kraków, Poland, in person only |

Sources: [2026 homepage](https://gecco-2026.sigevo.org/HomePage), [2027 homepage](https://gecco-2027.sigevo.org/HomePage).

The 2027 homepage confirms its event details, but the linked Call for Papers still describes the 2026 location and gives 2026 deadlines. Reliable 2027 submission deadlines were not established from the official pages checked. Treat the old dates, page limits, fees, and submission formats as historical until the relevant 2027 information is confirmed. [Partly updated 2027 Call for Papers](https://gecco-2027.sigevo.org/Call%2Bfor%2BPapers).

For further targeted reading, SIGEVO maintains links to main and companion proceedings. This review checked a small methods shortlist and relevant programme descriptions; it did not screen every 2025 or 2026 paper, inspect tutorial recordings, or run a new experiment. [Official proceedings portal](https://sig.sigevo.org/SIGEVO-Proceedings-in-the-Digital-Library).
