# KaMIS evidence audit

## What do the reported comparisons actually establish?

### Takeaway

The user's reading is supported. KaMIS returns the largest reported mean independent sets on the 300–400, 700–800, and 1,300–1,400-node families with a 60-second budget. The paper does not establish DDEA as the strongest practical MIS solver among its baselines. KaMIS does not win every runtime comparison, and the smallest family favors Gurobi. [DDEA Tables 4 and 5, PDF p. 9](https://arxiv.org/pdf/2510.08627v1#page=9)

### Cited Findings

The following numbers reproduce Table 4. Size is the reported average independent-set cardinality, so higher is better. T is the paper's runtime column in seconds. KaMIS is explicitly assigned a 60-second limit. The source supplies no uncertainty intervals for these rows. [Table 4 and Section 4.4](https://arxiv.org/pdf/2510.08627v1#page=9)

| ER node range | Method | Mean set size | T, seconds | E, evaluations | D, diffusion calls |
| --- | --- | ---: | ---: | ---: | ---: |
| 50 to 100 | DIFUSCO ×16 | 20.32 | 0.7 | 16 | 1 |
| 50 to 100 | DIFUSCO-timed | 20.73 | 13.9 | 640 | 20 |
| 50 to 100 | DDEA-long | 20.75 | 13.3 | 540 | 16 |
| 50 to 100 | KaMIS, 60-second limit | 20.75 | 60.0 | Not reported | Not applicable |
| 50 to 100 | Gurobi | 20.75 | 0.1 | Not reported | Not applicable |
| 300 to 400 | DIFUSCO ×16 | 35.53 | 8.7 | 16 | 1 |
| 300 to 400 | DIFUSCO-timed | 36.76 | 207.1 | 960 | 30 |
| 300 to 400 | DDEA-long | 36.85 | 183.3 | 720 | 16 |
| 300 to 400 | KaMIS, 60-second limit | 37.05 | 60.0 | Not reported | Not applicable |
| 300 to 400 | Gurobi | 35.47 | 171.2 | Not reported | Not applicable |
| 700 to 800 | DIFUSCO ×16 | 42.12 | 43.6 | 16 | 1 |
| 700 to 800 | DIFUSCO-timed | 43.99 | 851.3 | 960 | 30 |
| 700 to 800 | DDEA-long | 44.13 | 803.1 | 720 | 11 |
| 700 to 800 | KaMIS, 60-second limit | 44.85 | 60.0 | Not reported | Not applicable |
| 700 to 800 | Gurobi | 41.06 | 803.6 | Not reported | Not applicable |

The following numbers reproduce Table 5. This is the 42-graph test at 1,300–1,400 nodes. DDEA and DIFUSCO use diffusion models trained on ER-700-800. The task remains MIS and the test remains in the stated ER graph setup. This establishes a node-count shift, not generalization to every graph family or optimization problem. [Sections 4.1 and 4.5](https://arxiv.org/pdf/2510.08627v1#page=6), [Table 5](https://arxiv.org/pdf/2510.08627v1#page=9)

| Method | Mean set size | T, seconds | E, evaluations | D, diffusion calls |
| --- | ---: | ---: | ---: | ---: |
| DIFUSCO ×16 | 37.07 | 108.1 | 16 | 1 |
| DIFUSCO-timed | 38.69 | 947.1 | 384 | 12 |
| DDEA, P = 16, G = 10 | 43.17 | 909.5 | 240 | 21 |
| KaMIS, 60-second limit | 49.91 | 60.0 | Not reported | Not applicable |

- At 50–100 nodes, KaMIS ties DDEA-long and Gurobi in mean size at 20.75. Gurobi's listed runtime is 0.1 seconds, DDEA's is 13.3, and KaMIS's budget is 60. This does not support KaMIS as universally fastest. [Table 4](https://arxiv.org/pdf/2510.08627v1#page=9)
- At 300–400 and 700–800 nodes, KaMIS has larger mean sets and less listed time than DDEA-long, DIFUSCO-timed, and Gurobi. The one-shot DIFUSCO ×16 rows are faster than 60 seconds but return smaller sets. They represent a time/quality tradeoff, rather than a method that KaMIS beats in both displayed quantities. [Table 4](https://arxiv.org/pdf/2510.08627v1#page=9)
- In the OOD table, KaMIS has both a larger mean set and less listed time than every other displayed row. This is the clearest practical result favoring KaMIS in the paper. It remains a comparison of reported summaries, not a per-instance guarantee. [Table 5](https://arxiv.org/pdf/2510.08627v1#page=9)
- Each in-distribution family has 128 test graphs. The OOD family has 42 graphs. These graph counts do not tell us the number of repeated random-seed runs. Tables 4 and 5 give no spread or confidence intervals, and the experimental description does not state a clear repeated-seed protocol. [Dataset description](https://arxiv.org/pdf/2510.08627v1#page=6), [Tables 4 and 5](https://arxiv.org/pdf/2510.08627v1#page=9)

### Inferences

These are arithmetic derived from the displayed, rounded means. Let K be KaMIS mean size and D be DDEA mean size. KaMIS's relative size advantage is 100 × [K − D] / D. DDEA's shortfall relative to KaMIS is 100 × [K − D] / K. The different denominators explain why the two percentages differ. Neither quantity is the mean of per-graph percentage gaps. [Input values, Tables 4 and 5](https://arxiv.org/pdf/2510.08627v1#page=9)

| Node range | KaMIS minus DDEA mean size | KaMIS size advantage over DDEA | DDEA shortfall relative to KaMIS | DDEA listed T divided by 60-second KaMIS budget |
| --- | ---: | ---: | ---: | ---: |
| 300 to 400 | 0.20 | 0.543% | 0.540% | 3.055 |
| 700 to 800 | 0.72 | 1.632% | 1.605% | 13.385 |
| 1,300 to 1,400 | 6.74 | 15.613% | 13.504% | 15.158 |

- The duration ratios above compare reported run durations or budgets. They are not equal-quality speedup measurements or measurements of time to first discover the final set. A KaMIS run may find its final best set before its limit, but these tables do not say when. [Budget definition](https://arxiv.org/pdf/2510.08627v1#page=8), [Tables 4 and 5](https://arxiv.org/pdf/2510.08627v1#page=9)
- A size/time nondominance check of the displayed rows leaves only Gurobi on the smallest family; DIFUSCO ×16 and KaMIS on each larger in-distribution family; and only KaMIS on the OOD family. This is an arithmetic comparison of the table entries, with no claim about unreported time budgets or uncertainty. [Tables 4 and 5](https://arxiv.org/pdf/2510.08627v1#page=9)
- The paper's 3.9% and 7.5% headline comparisons use Gurobi, not KaMIS or timed DIFUSCO, as the denominator. From the rounded table cells, the larger in-distribution gains over timed DIFUSCO are only about 0.245% and 0.318%. The OOD 11.6% claim is about DDEA versus timed DIFUSCO: 100 × [43.17 − 38.69] / 38.69 = 11.579%. None establishes superiority over KaMIS. [Section 4.4 and Table 5](https://arxiv.org/pdf/2510.08627v1#page=9)

### Gaps

- There are no per-instance quality/time records in the displayed tables, so they cannot establish that KaMIS wins on every graph. They also cannot establish a paired-test significance level for the 0.20 or 0.72 mean differences. [Reported summaries](https://arxiv.org/pdf/2510.08627v1#page=9)
- The paper does not provide KaMIS anytime curves in Figure 2. Therefore, the 60-second KaMIS endpoint does not identify its quality at 0.1, 1, 10, or 30 seconds. [Figure 2](https://arxiv.org/pdf/2510.08627v1#page=8)
- The paper reports averaged cost and plots average runtime at generation or sampling-call boundaries. Table 4 labels T as runtime without fully documenting its averaging and preprocessing boundary. The safest course wording is "reported runtime T" or "60-second KaMIS budget," rather than asserting a more detailed aggregation convention or time-to-best statistic. [Figure 2 description](https://arxiv.org/pdf/2510.08627v1#page=8), [Table 4 caption](https://arxiv.org/pdf/2510.08627v1#page=9)

## What is KaMIS, and how precisely can we identify the tested variant?

### Takeaway

KaMIS is Karlsruhe Maximum Independent Sets, a collection of specialized graph algorithms. Its ReduMIS solver is itself evolutionary search combined with graph reduction. The DDEA paper names KaMIS but does not pin a source revision or executable. The author's code provides more specific implementation evidence, which must remain separate from a paper-revision reproduction claim. [Official KaMIS README](https://github.com/KarlsruheMIS/KaMIS/blob/791334ef6aebb2ccf005316258c043cca58923a0/README.md), [DDEA setup](https://arxiv.org/pdf/2510.08627v1#page=6), [author wrapper](https://github.com/jsalvasoler/difusco_ddea/blob/ff1d9a8e58f6a7fcf92f9bd426f2e1b4853d7393/src/mis_benchmark_framework/solvers/kamis.py)

### Cited Findings

- The KaMIS README at revision `791334ef6aebb2ccf005316258c043cca58923a0` identifies that release as v2.0 and lists ReduMIS as an evolutionary algorithm on a reduced graph. It also lists a separate local-search solver and separate weighted solvers. "KaMIS" is consequently a software-family name, not a synonym for one generic exact algorithm. [Pinned KaMIS README](https://github.com/KarlsruheMIS/KaMIS/blob/791334ef6aebb2ccf005316258c043cca58923a0/README.md)
- The cited KaMIS research develops a heuristic evolutionary algorithm combined with kernelization. It uses graph reductions and selected likely solution vertices to expose further reductions. The original abstract does not claim every output has a global optimality certificate. [Lamm et al., primary preprint](https://arxiv.org/abs/1509.00764), [journal publication record](https://doi.org/10.1007/s10732-017-9337-x)
- At the author's repository revision returned by code search, `ff1d9a8e58f6a7fcf92f9bd426f2e1b4853d7393`, the unweighted wrapper launches `KaMIS/deploy/redumis`. The weighted branch instead launches `weighted_branch_reduce`. The same wrapper checks out KaMIS revision `791334ef6aebb2ccf005316258c043cca58923a0` when installing its dependency. [Pinned author wrapper](https://github.com/jsalvasoler/difusco_ddea/blob/ff1d9a8e58f6a7fcf92f9bd426f2e1b4853d7393/src/mis_benchmark_framework/solvers/kamis.py)
- The wrapper passes a per-graph time-limit argument and records both total solver duration and, if parsed from output, a solution discovery time. This confirms that those timing concepts are distinct in the available code. It does not establish which exact timing summary produced each paper cell. [Pinned author wrapper](https://github.com/jsalvasoler/difusco_ddea/blob/ff1d9a8e58f6a7fcf92f9bd426f2e1b4853d7393/src/mis_benchmark_framework/solvers/kamis.py)
- DDEA uses KaMIS with a 60-second limit to generate the initial solution labels. Its recombination teacher is a different component, a parent-restricted Gurobi ILP with a 15-second limit. Do not call KaMIS the ILP recombination teacher. [Dataset and implementation description](https://arxiv.org/pdf/2510.08627v1#page=6), [expert construction](https://arxiv.org/pdf/2510.08627v1#page=5)

### Inferences

- The relevant contrast is a learned initialization/recombination framework versus a highly engineered MIS solver that already combines evolution with graph reductions. It is not a comparison of evolutionary search against a wholly non-evolutionary baseline. This distinction follows from the DDEA operator definitions and ReduMIS description. [DDEA framework](https://arxiv.org/pdf/2510.08627v1#page=3), [ReduMIS description](https://github.com/KarlsruheMIS/KaMIS/blob/791334ef6aebb2ccf005316258c043cca58923a0/README.md)
- The code supports describing the available unweighted implementation as ReduMIS. For a claim specifically about the published experiment, use "KaMIS, with the available author wrapper selecting ReduMIS" until the source revision and result provenance are connected. [Paper's KaMIS label](https://arxiv.org/pdf/2510.08627v1#page=9), [pinned author wrapper](https://github.com/jsalvasoler/difusco_ddea/blob/ff1d9a8e58f6a7fcf92f9bd426f2e1b4853d7393/src/mis_benchmark_framework/solvers/kamis.py)
- Safe reductions can preserve an optimum through reconstruction, while a heuristic evolutionary search over the remaining choices can still lack a global certificate. Calling the combined ReduMIS procedure "exact" because it uses reductions would confuse those two facts. [KaMIS algorithm paper](https://arxiv.org/abs/1509.00764)

### Gaps

- The DDEA paper's data/code statement links a repository but does not identify a commit. The audit did not establish that the accessed author revision is exactly the revision used to produce Tables 4 and 5. Do not present the code-level executable identification as a completed reproduction of those tables. [Data and code statement](https://arxiv.org/pdf/2510.08627v1#page=10), [accessed author revision](https://github.com/jsalvasoler/difusco_ddea/tree/ff1d9a8e58f6a7fcf92f9bd426f2e1b4853d7393)
- Single-threading is specified for the ILP expert, not comprehensively for every benchmark method. A wrapper that schedules one graph process does not prove that every underlying native dependency uses only one CPU thread. [DDEA implementation](https://arxiv.org/pdf/2510.08627v1#page=6), [author wrapper](https://github.com/jsalvasoler/difusco_ddea/blob/ff1d9a8e58f6a7fcf92f9bd426f2e1b4853d7393/src/mis_benchmark_framework/solvers/kamis.py)

## What should the course preserve, correct, or add?

### Takeaway

The numeric tables and existing KaMIS cautions are accurate. The course should make KaMIS's practical advantage more prominent, explain that ReduMIS is also evolutionary, and narrow one unsupported runtime-aggregation sentence. A general framework can be an interesting research contribution without being the best solver for its demonstration problem. [Course evidence](../../sites/ddea-socratic-20261020-b8e4/dist/course-evidence.js), [course foundations](../../sites/ddea-socratic-20261020-b8e4/dist/course-foundations.js), [course assessment](../../sites/ddea-socratic-20261020-b8e4/dist/course-assessment.js), [DDEA framework and results](https://arxiv.org/pdf/2510.08627v1#page=3)

### Cited Findings

- The course transcribes all Table 1–5 cells correctly, including the unusual P = 50 in the smallest-family RG/CDX row, the Table 2 DI/Diffusion values 0.90 and 2.95, and the OOD D counts 21 versus 12. The assessed local data has 5 operator rows, 15 component rows, 5 PI rows, 15 benchmark rows, and 4 OOD rows. [Course table arrays](../../sites/ddea-socratic-20261020-b8e4/dist/course-evidence.js), [Tables 1 and 2](https://arxiv.org/pdf/2510.08627v1#page=7), [Table 3](https://arxiv.org/pdf/2510.08627v1#page=8), [Tables 4 and 5](https://arxiv.org/pdf/2510.08627v1#page=9)
- The foundations chapter already limits "state of the art" to DIFUSCO's neural/diffusion context and acknowledges stronger KaMIS solution sizes. Assessment a37 gives the correct KaMIS comparison, and a38 retains the larger-ER scope of OOD. Their claims are supported, but KaMIS is presently a caveat following the DDEA results rather than the first practical takeaway. [Foundations](../../sites/ddea-socratic-20261020-b8e4/dist/course-foundations.js), [assessment a37 and a38](../../sites/ddea-socratic-20261020-b8e4/dist/course-assessment.js), [primary benchmark](https://arxiv.org/pdf/2510.08627v1#page=9)
- The evidence setup currently says Tables 4 and 5 report mean size and runtime across instances and are not totals for the collection. The paper is clear about mean cost and plotted mean runtimes, but less explicit about the exact T aggregation in the tables. Prefer "reported solver runtime T; KaMIS has a 60-second budget" and identify the precise aggregation as insufficiently documented. [Current course setup](../../sites/ddea-socratic-20261020-b8e4/dist/course-evidence.js), [Figure 2 aggregation description](https://arxiv.org/pdf/2510.08627v1#page=8), [Table 4 caption](https://arxiv.org/pdf/2510.08627v1#page=9)
- GPU training and inference use an A100 80 GB, while other experiments use the reported Xeon cluster. Equal elapsed time is not equal hardware cost, energy, or resource consumption. The paper's offline teacher-data generation and neural training are outside these per-solve comparison tables, and total offline wall time is not supplied. [Implementation](https://arxiv.org/pdf/2510.08627v1#page=6), [limitations](https://arxiv.org/pdf/2510.08627v1#page=10)
- Section 4.3's prose quotes about 5 versus 88 seconds per generation, 0.94 versus 1.39 for initialization, and 2.81 versus 5.75 for the larger ablation. Table 2's corresponding DI/Diffusion figures are 9.4 seconds, 0.90%, and 2.95%. Calling all displayed rows time-matched is also too strong given their differing runtimes. The course correctly retains these discrepancies. [Section 4.3 and Table 2](https://arxiv.org/pdf/2510.08627v1#page=7)
- The printed PI formula normalizes an incumbent-size area, which increases with better solutions. Table 3's preferred values behave like a gap area instead. Its prose also calls P = 16 best on the smallest family despite P = 24 having the lower listed value. The OOD PI passage cites 2.27 and 1.21, which appear in different in-distribution columns, while Table 5 has no PI column. A consistent OOD PI comparison cannot be recovered from these passages alone. [Printed formula](https://arxiv.org/pdf/2510.08627v1#page=6), [Table 3 and discussion](https://arxiv.org/pdf/2510.08627v1#page=8), [OOD discussion](https://arxiv.org/pdf/2510.08627v1#page=9)
- The OOD prose claims fewer diffusion calls, but Table 5 lists DDEA at 21 and timed DIFUSCO at 12. Fewer candidate evaluations, E = 240 versus 384, is supported. Fewer calls is not. Call counts also have different batch costs, so they are not an elapsed-time substitute. [Table 5 and surrounding text](https://arxiv.org/pdf/2510.08627v1#page=9)

### Inferences

- Suggested new course opening: "Your reading of the larger-graph benchmarks is right. KaMIS returns larger mean independent sets than DDEA with a 60-second budget, while DDEA's listed runs take roughly 183, 803, and 910 seconds. On the smallest graphs, Gurobi ties the best mean size in 0.1 seconds. DDEA's supported contribution is improving this diffusion-based solver and demonstrating learned recombination inside an EA. These experiments do not establish it as the best MIS solver." This wording is derived from the table comparisons and contribution description. [Tables 4 and 5](https://arxiv.org/pdf/2510.08627v1#page=9), [DDEA framework](https://arxiv.org/pdf/2510.08627v1#page=3)
- Suggested question: "Why can a paper be worth reading even when KaMIS wins the practical benchmark?" The supported answer distinguishes a tested learned operator and reusable research design from a demonstrated best solver. Portability to other problems is an aim of the framework, not an empirical win on those other problems. [Framework scope](https://arxiv.org/pdf/2510.08627v1#page=3), [future work](https://arxiv.org/pdf/2510.08627v1#page=10)
- Suggested check: ask the learner to find a row where KaMIS is larger but slower, a row where it is larger and faster, and the smallest-family quality tie. That tests whether "better" has been tied to the actual objective and budget instead of a blanket ranking. [Tables 4 and 5](https://arxiv.org/pdf/2510.08627v1#page=9)
- For a practical MIS task resembling these larger ER tests, KaMIS is the strongest starting baseline in the supplied evidence. For a new application, one can study learned recombination as a hypothesis to test against strong specialized and generic baselines. Neither transfer success nor the cost of adapting KaMIS to that new problem is established by this paper. [Reported benchmarks](https://arxiv.org/pdf/2510.08627v1#page=9), [future work](https://arxiv.org/pdf/2510.08627v1#page=10)

### Gaps

- The paper does not establish all-in energy, monetary cost, memory use, or offline-training amortization. These cannot be inferred from T, E, or D alone. [Available metrics](https://arxiv.org/pdf/2510.08627v1#page=6), [reported result columns](https://arxiv.org/pdf/2510.08627v1#page=9)
- The tables do not establish how either method performs on the user's Pokémon counter-search objective. MIS cardinality and battle win rate are different objectives with different constraints and evaluation costs. Transfer to another combinatorial problem is listed as future work, not supplied evidence. [Problem definition](https://arxiv.org/pdf/2510.08627v1#page=4), [future work](https://arxiv.org/pdf/2510.08627v1#page=10)
- No new benchmark runs or reproductions were performed for this audit. It verifies the paper transcription, derives arithmetic from reported entries, checks a pinned author-code snapshot, and audits the present course text. [Primary paper](https://arxiv.org/abs/2510.08627v1), [accessed author source](https://github.com/jsalvasoler/difusco_ddea/tree/ff1d9a8e58f6a7fcf92f9bd426f2e1b4853d7393)
