---
created_at: 2026-09-07
updated_at: 2026-09-07
type: reading-list-history
tags:
  - reading-list
  - archive
---

# Reading List — notes and decision history

Historical snapshot before the 2026-09-07 reorganization. Personal notes, old grades, numbering and decisions are preserved below. Use [[Reading List]] for the current queue; this snapshot does not define current acceptance.

## Generative Bayesian Optimization — reassessed 2026-09-07

Owner requested the same presence-and-relevance check for *Generative Bayesian Optimization: Generative Models as Acquisition Functions*. Already present in the active list under Generative optimization and learned representations, graded **HIGH**. Retain HIGH; no duplicate added. Direct generator learning from evaluated candidates closely matches the project's batch generate → evaluate → refit loop and offers a concrete comparison with elite-only CEM. See [[Generative Bayesian Optimization — relevance to counter-team search]] for the primary-source assessment and transfer caveats. [Paper, revised v3](https://arxiv.org/abs/2510.25240).

## Regret Analysis of Guided Diffusion — direct entry added 2026-09-07

The owner requested a presence check, relevance assessment, and addition if relevant. The paper was already recorded as **MED** in [[Diffusion Model as Black Box Opimizer]], itself linked from the active list, and mentioned in [[Active Learning narrow focus in Membership Query Synthesis]]. Added its direct linked title under Search and black-box optimization → Generative optimization and learned representations; preserved the existing notes and pointer.

**Retain MED for selective reading.** Its mass-lift viewpoint is useful for assessing whether the guided team proposer puts more probability on strong candidates, alongside the existing guidance-strength and candidate-selection experiments. It does not establish a convergence guarantee for the project's current sampler, legality repair or noisy battle-ranking procedure. Read for search diagnostics, with implementation decisions separate from this reading recommendation. [Primary paper](https://arxiv.org/abs/2605.10385).

Assessment and suggested reading sections: [[Regret Analysis of Guided Diffusion — relevance to counter-team search]].

## Original document

### Later reassessment: Trust-Region Noise Search — 2026-09-07

Owner requested a presence check, reassessment and addition if relevant. Already present in the rejected section, previously LOW. **Accepted MED for selective reading** and added as a title-only entry under Search and black-box optimization → Generative optimization and learned representations. The rejected entry remains as explicitly superseded history.

The previous claim that reward calls make the method unsuitable was too categorical: the authors explicitly evaluate expensive black-box rewards. Adaptive local perturbations and restarts are useful search-design references. Transfer of continuous source-noise neighborhoods to the project's categorical masked generator, and reliable decisions from noisy battle scores, remain unvalidated; no immediate implementation is recommended. See [[Trust-Region Noise Search for Black-Box Alignment of Diffusion and Flow Models]] and the [primary full paper](https://arxiv.org/html/2603.14504v2).

---
created_at: 2026-08-24
updated_at: 2026-09-07
tags:
  - reading-list
  - vgc
  - combinatorial-search
  - reinforcement-learning
sources: Todo Section.md · Teaching/RESOURCES.md · Gradient-guided diffusion as team search · When learning stops · papers-closest-to-what-i-hoped-reevo-was · Papers/PokeAgent Challenge · Clause · Online Courses
ordering: "single ranking, 1 to 150, most relevant first (renumbered 2026-08-29; stored numbers had drifted from deletions); 2026-09-01 Gradient Guidance for Diffusion Models (NeurIPS 2024) added at vacant slot 41 — the paper behind the Mengdi Wang talk at 48; 2026-09-01 three submodular papers moved from Tier 3 to rejected (Filmus & Ward matroid, Skowron & Faliszewski Chamberlin–Courant, Streeter & Golovin online) — each assumes clean submodularity or an online setting, which the near-modular objective is not; 41 and 42 left vacant, not renumbered; 2026-08-29 HADES (2411.13420) demoted from direction_papers to rejected (continuous genome, toy tasks); 2026-08-29 win-prediction family sweep added 5 papers to rejected (Real-time eSports 1701.03162, Hodge 1711.06498, LoL 2309.02449, Do & Wang 2108.02799, MobaQA); 2026-08-29 Feige set-cover paper moved from Tier 3 to rejected (background gap; 1−1/e greedy result covered by submodular surveys); 2026-08-29 Jin & Branke 2005 uncertain-environments survey moved from Tier 3 to rejected (superseded); 2026-08-29 rejected-papers section appended (15-paper MOBA-draft sweep + EvoLLM + SOPL verdicts); 2026-08-29 Mengdi Wang INI talk (video) added to the Tier 3 diffusion block; 2026-08-27 ReALM-GEN (ICLR 2026) sweep added 6 to the Tier 3 diffusion block; 2026-08-26 diffusion-generator sweep added 21 papers to the Tier 3 diffusion block and BERTeam to Tier 2; re-ranked 2026-08-25 — Pokémon core → deck/draft analogues → outer search loop → battle-policy RL → training dynamics → CP → peripheral (QUBO/annealing demoted, project does not use them); 2026-08-29 self-consuming-loop/model-collapse block (6 papers, adversarially verified — 2 HIGH, 4 MED) appended to the end of Tier 3 for the dataset-expansion loop; 2026-08-29 CP/MiniZinc goal sweep: symmetry + dominance canon (6 papers) added to Tier 6, Chagnet type-coverage MILP to Tier 7, D-Wave QUBO webinar to rejected; 2026-08-29 uncertainty-guided diffusion sweep added 8 papers to the Tier 3 diffusion block; 2026-08-29 classifier-free-guidance vs cross-entropy-method elite-retraining sweep added 8 papers to the end of Tier 3 (all HIGH, adversarially verified) — the elite filter, not the guidance, is what the theorems concentrate; 2026-08-29 diffusion-as-candidate-proposer sweep added 30 papers (174-203) to the end of Tier 3 — offline/online model-based optimization vocabulary, active search, replication-vs-exploration, constrained discrete decoding, guidance theory; 2026-08-29 membership-query-synthesis sweep added 3 papers to the end of Tier 3 (Settles, LaMBO-2 and Info-Synth were already listed) — the scenario is defined by de novo provenance, not by informativeness, and the Lang & Baum objection is scoped to human oracles, so it does not apply to a Showdown oracle; 2026-09-01 hyper-heuristics sweep added 2 to Tier 7 (Burke taxonomy chapter MED, Drake selection survey LOW); 2026-09-01 LLM-as-hyper-heuristic follow-up added 2 (Beyond Static Evaluation to Tier 4 MED, RuleSmith to Tier 7 LOW) — first two systems putting a noisy game under an LLM heuristic-evolution loop; 2026-09-01 Subset Selection by Pareto Optimization moved from Tier 3 to rejected (exact value oracle, k-subset representation retired); 39 left vacant, not renumbered; 2026-09-01 submodular block closed — Fujii localizability (ICML 2020) and Bayati greedy-many-armed (NeurIPS 2020) moved to rejected; 37, 39, 40 left vacant; 38 [[Submodular Function]] is the only survivor, tied to weighted maximum coverage; 2026-09-02 Illuminating Mario Scenes (AAAI 2021) graded LOW and moved from 43 to the end of Tier 3 — latent-space quality-diversity search was closed on win rate by the 2026-08-26 pilot; 43 left vacant, not renumbered; 2026-09-02 HADES (2411.13420) row removed from the ranking at 72 — it was already in the rejected section; 72 left vacant, not renumbered; 2026-09-02 two molecular-optimization rows moved to rejected — Sample Efficiency Matters (73, owner: does not contribute) and Diffusion-based Evolutionary Optimization for 3D Multi-Objective Molecular Generation (74, 1 citation); 73 and 74 left vacant, not renumbered; 2026-09-02 molecular-optimization family closed — Reward-Guided Discrete Diffusion (50) and Design Space of Discrete Diffusion Online Adaptation (93) also moved to rejected; 50 and 93 left vacant, not renumbered; 2026-09-02 last two molecular entries rejected — On failure modes in molecule generation (106) and Genetic algorithms are strong baselines (112), both were HIGH; 106 and 112 left vacant, not renumbered; the molecular-optimization block is now empty; 2026-09-02 HardFlow (2511.08425) moved from Tier 3 to rejected — continuous trajectory optimization, and constrained decoding is already covered in discrete space; 79 left vacant, not renumbered; 2026-09-02 two more ReALM-GEN rows rejected — Trust-Region Noise Search (78, noise-space search against a costly noisy reward) and Nested Sequential Monte Carlo (80, no reward on partial teams); 78 and 80 left vacant, not renumbered; 2026-09-02 new '## Automated heuristic design' section added after Tier 3, collecting the LLM-writes-the-operator family — G-LNS 84→162, Beyond Static Evaluation 160→163, Burke taxonomy 158→164, RuleSmith 161→165, Drake survey 159→166; 84, 158, 159, 160, 161 left vacant; StreamLLM and Streamlined Constraint Reasoning stay in Tier 6 (CP streamliners first); 2026-09-02 DDNO (82) and Diffusion Active Learning in CT (95) moved to rejected — noise-space search with a costly reward, and an inverse-problem acquisition over measurements; 82 and 95 left vacant; 83 formatted from a bare heading (Gaier, Asteroth & Mouret, Discovering Representations for Black-box Optimization, GECCO 2020, MED); 2026-09-02 Active Diffusion-Based Inference (96) moved to rejected — second inverse-problem paper out after the CT one; venue corrected to IJCAI-ECAI 2026 (the entry said no venue listed); 96 left vacant; 2026-09-02 Offline Model-Based Optimization survey (105) moved to rejected — owner does not read surveys, and offline MBO assumes no oracle access while Showdown provides one; 105 left vacant; 2026-09-02 return-conditioned supervised learning paper (108) moved to rejected — owner's call, an offline-RL policy result with no design generator; 108 left vacant"
sequencing_rule: read every HIGH-graded paper first, then reimplement, then return to MED; exception — the reward subsection is deferred until reward integration is planned, including its HIGH entries (owner, 2026-09-06)
direction: closing the loop (2026-08-27) — the cross-entropy method with a generator as the density, run against a copy-paste baseline; direction papers also listed in frontmatter `direction_papers` below
direction_papers:
  - Deep Surrogate Assisted MAP-Elites for Automated Hearthstone Deckbuilding
  - A Tutorial on the Cross-Entropy Method
  - Design by adaptive sampling
  - Diffusion Models are Evolutionary Algorithms
  - Diffusion Models for Black-Box Optimization
---

> [!important] Current owner decision — confirmed 2026-09-07
> **Only two SPIGM papers are accepted for the current reading list, both MED:** #169 **A Tale of Two Temperatures: Simple, Efficient, and Diverse Sampling from Diffusion Language Models** and #170 **Re-evaluating Confidence Remasking in Masked Diffusion Language Models** (screening entries #15 and #83).
> This records the owner's 2026-09-06 confirmation and 2026-09-07 correction. The other 16 initial additions are not in the active queue, including #174 feature-distribution matching. Earlier grades, counts and recommendations below are historical and are superseded by this decision.

> [!important] Latest relevance correction — 2026-09-06
> **Current SPIGM recommendations: 3 KEEP MED, 9 DEFER, 6 DROP; none HIGH.** #183, Hacking Generative Perplexity, no longer earns a current reading recommendation. Its established result concerns frozen-language-model perplexity and token entropy. The proposed Pokémon diversity connection was an analogy, not a tested team metric or algorithm. The existing Metrics for Diversity note already calls for full-team uniqueness, novelty, unordered six-species/form composition counts and structural generalization. A general warning about marginal metrics adds insufficient value to justify this paper.
> Remaining MED readings: #174 feature-distribution matching, #169 Two Temperatures, and #170 Re-evaluating Confidence Remasking. This supersedes the four-paper shortlist below; earlier assessments remain as history.



> [!important] Latest owner correction — 2026-09-06
> **Current SPIGM recommendations: 4 KEEP MED, 9 DEFER, 5 DROP; none HIGH.** This supersedes the earlier six-paper shortlist. The owner clarified that diffusion training is not a current problem and exact duplicate teams can already be removed. Consequently, #181 (training acceleration) and #173 (exact data repetition) no longer earn a current reading recommendation. The repetition paper does not establish that distinct mutations have the same effect as exact duplicates.
> The four remaining MED readings are #183 Hacking Generative Perplexity, #174 feature-distribution matching, #169 Two Temperatures, and #170 Re-evaluating Confidence Remasking. Earlier judgments are retained below as history.



> [!important] SPIGM full-paper reassessment — 2026-09-06
> The 18 SPIGM additions have now received full-paper reads. **Current recommendations: 6 KEEP MED, 9 DEFER, 3 DROP; none HIGH.** The per-entry full-read annotations and [[SPIGM 2026 — full-paper reassessment]] supersede their original screening grades. Original entries remain as history; DROP means no longer recommended for the current queue. Other workshop and reading-list grades are unaffected.



## Tier 1 — Pokémon itself (team building + battling + benchmarks)

1. [[VGC-Bench Towards Mastering Diverse Team Strategies in Competitive Pokémon]]
2. [[Human-Level Competitive Pokémon via Scalable Offline Reinforcement Learning with Transformers]]
3. 🔒 [An Adversarial Approach for Automated Pokémon Team Building and Metagame Balance](https://doi.org/10.1109/tg.2023.3273157)
4. [PokéChamp: An Expert-level Minimax Language Agent](https://arxiv.org/abs/2503.04094)
5. [PokéLLMon: A Human-Parity Agent for Pokémon Battles with Large Language Models](https://arxiv.org/abs/2402.01118)
6. [The PokeAgent Challenge: Competitive and Long-Context Learning at Scale](https://www.semanticscholar.org/paper/80c6fcde8b5e8190a2e970db128eaf095887a46b)
7. [Pokémon GO Team Optimization](https://journals-sol.sbc.org.br/index.php/jis/article/view/6773)
8. [A Framework for Predicting the Impact of Game Balance Changes Through Meta Discovery](https://arxiv.org/abs/2409.07340)
9. [From Rules to Nash Equilibria](https://arxiv.org/abs/2607.08692)

## Tier 2 — Deck / draft / team building in other games (closest analogues)

10. [Deep Surrogate Assisted MAP-Elites for Automated Hearthstone Deckbuilding](https://www.semanticscholar.org/paper/e46811df6731df51fb5c209dea7b835d0cd734da) — direction paper — the loop itself: generate decks → battle vs fixed opponents → refit → repeat
	1. I am learning the first time about quality diversity because of the paper [Deep Surrogate Assisted MAP-Elites for Automated Hearthstone Deckbuilding](https://www.semanticscholar.org/paper/e46811df6731df51fb5c209dea7b835d0cd734da). The point of quality diversity for example Pokemon. We have metagame where we can measure the behavior of a pokemon base on numerical feature. One axis for for the beavhior of the team is static offense another axis could is static offense/bulk ratio. Another list of possible axis could be speed, number of turn and etc. The ideas we have a N by N matrix one let how long the battle takes and the other metric you define by metric you can quantify. We have a battle policy beat the current meta. That will be the objective function. Where it place performances it place on the grid. I was talking about in a good game meta there will be variety of meta stratigies that have counter each other.
	2. We can focus on later what will be the second axis for VGC, right now one axis quantify measure how long does it take win a battle.
	3. "A healthy game ecosystem requires a variety of playstyles—such as "aggro" (fast-paced, aggressive) or "control" (slow-paced, defensive)—to keep gameplay engaging and balanced."
11. [Mapping Hearthstone Deck Spaces through MAP-Elites with Sliding Boundaries](https://www.semanticscholar.org/paper/9eb4d5e751b83a2afb240b93c0c2b1ea025ac14f)
	1. On big issues how to use evolution appraoch is that do i modified for pokemon. Mutation Steps for pokemon. If i choose a mutation that could not be legal at all.
	2. feasibility-preserving operators its's possible then use smogon to verify it a legal team
12. [CabbageCat's Blogs — MAP-Elites, CMA-ME, Differentiable QD step-by-step](https://szhaovas.github.io/2022-09-15-me/) — 🟢 grad-researcher QD tutorial series (MAP-Elites → CMA-ME → Differentiable QD); read before/with the MAP-Elites Hearthstone papers
13. 🔒 [Exploring the Hearthstone Deck Space](https://dl.acm.org/doi/10.1145/3235765.3235791) — 🟢 [open copy: Internet Archive capture of the ACM PDF](https://web.archive.org/web/20200602172549/https://dl.acm.org/doi/pdf/10.1145/3235765.3235791?download=true)
14. [Evolving the Hearthstone Meta](https://arxiv.org/abs/1907.01623)
15. 🔒 [Automated playtesting in collectible card games using evolutionary algorithms](https://doi.org/10.1016/j.knosys.2018.04.030)
16. [Drafting in Collectible Card Games via Reinforcement Learning](https://www.sbgames.org/proceedings2020/ComputacaoFull/209690.pdf)
17. [Q-DeckRec: A Fast Deck Recommendation System for Collectible Card Games](https://www.semanticscholar.org/paper/ca1e69d0d5a74a05ee390ea6aa0129965a83950b)
18. [Evolutionary Arena Deckbuilding with Active Genes](https://arxiv.org/abs/2001.01326)
19. [Cardiverse: Harnessing LLMs for Novel Card Game Prototyping](https://www.semanticscholar.org/paper/6590ba146e330de09ce7798164cc1ddeecfc0b17)
20. [Summarizing Strategy Card Game AI Competition](https://www.semanticscholar.org/paper/551a4a4d231e24bacbd7a1285e8f44dd553e5b3f)
	1. My main problem Expected More
	2. [[Evolutionary Algorithms]]
	3. [[Evolutionary Strategies]]
	4. [[MAP-Elites]


## Tier 3 — The outer loop: search methods you would actually run on teams


34. [[Simulating Annealing]]
35. [Using Common Random Numbers for Simulation-based Planning with Rollouts](https://rlj.cs.umass.edu/2026/papers/Paper52.pdf)
36. 🔒 [Toward Reliable Uncertainty Quantification in Surrogate-Assisted Evolutionary Algorithms via Temporal Conformal Prediction](https://doi.org/10.1007/978-3-032-23604-3_24)
38. [[Submodular Function]]
44. [DIFUSCO: Graph-based Diffusion Solvers for Combinatorial Optimization](https://www.semanticscholar.org/paper/5f90d43e6ece5c6ee6e8186e4b57d46c85377713) — 🟢 NeurIPS 2023 spotlight, LOW
45. [A Diffusion Model Framework for Unsupervised Neural Combinatorial Optimization](https://www.semanticscholar.org/paper/4a5ba3075d61f6258059dc311f761eb85848b22d) — 🟢 ICML 2024, MED
45b. [Scalable Discrete Diffusion Samplers: Combinatorial Optimization and Statistical Physics](https://arxiv.org/abs/2502.08696) — 🟢 ICLR 2025, ungraded
46. [Tackling Prevalent Conditions in Unsupervised Combinatorial Optimization](https://www.semanticscholar.org/paper/f7daa82747a13314c42a1df0c8eb2d4958ac27d6)
50. 
52. [Structured Denoising Diffusion Models in Discrete State-Spaces](https://www.semanticscholar.org/paper/91b32fc0a23f0af53229fceaae9cce43a0406d2e)
53. [Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution](https://www.semanticscholar.org/paper/ce806f8d32f6fb1eaa821248a1bc4fa2cd949fbb)
54. [Argmax Flows and Multinomial Diffusion: Learning Categorical Distributions](https://www.semanticscholar.org/paper/1913d3edcc00d0aba097a9df190dd16f6fdfbf0c)
55. [Prompt-to-Slate: Diffusion Models for Prompt-Conditioned Slate Generation](https://www.semanticscholar.org/paper/da1f9ea84ef798c82a876d37fdf95e4568eafcad)
56. [Learning the Travelling Salesperson problem Requires Rethinking Generalization](https://arxiv.org/abs/2006.07054)
57. [SDEdit: Guided Image Synthesis and Editing with Stochastic Differential Equations](https://www.semanticscholar.org/paper/f671a09e3e5922e6d38cb77dda8d76d5ceac2a27)
58. [Remasking Discrete Diffusion Models with Inference-Time Scaling](https://www.semanticscholar.org/paper/19c3a5d9d32c57cd1482c8376f208cc2b2333334)
59. [Simple and Effective Masked Diffusion Language Models](https://www.semanticscholar.org/paper/f8d357d38bbcdd93889fe71762eb57842b2ab063)
60. [BERT has a Mouth, and It Must Speak: BERT as a Markov Random Field Language Model](https://www.semanticscholar.org/paper/d79ac7a7bafdc9a782fb8c53285ca11c7f2e3f18)
65. [Compositional Visual Generation with Composable Diffusion Models](https://www.semanticscholar.org/paper/3ff7153fd6bd47d08084c7f50f8fd70026c126e7)
66. [TabDDPM: Modelling Tabular Data with Diffusion Models](https://www.semanticscholar.org/paper/25d3a4e048d0020ba9cffc6442ebd4e7bb548a55)
67. [Design-Bench: Benchmarks for Data-Driven Offline Model-Based Optimization](https://www.semanticscholar.org/paper/f23222ad51b9b1f4c1addcf848906db7698ccbff)
68. [On Memorization in Diffusion Models](https://www.semanticscholar.org/paper/122a7e217fe70d5a1a44a6e2b67e859d1fc8e28d)
69. [Diffusion Art or Digital Forgery? Investigating Data Replication in Diffusion Models](https://arxiv.org/abs/2212.03860)
70. [Design by adaptive sampling](https://arxiv.org/abs/1810.03714) — direction paper — says in print that refit-a-generator = the cross-entropy method
71. 
72. 
73. 
74. 
75. [A Denoising Diffusion-Based Evolutionary Algorithm Framework: Application to the Maximum Independent Set Problem](https://www.semanticscholar.org/paper/9b21ee2ac4fde7868f485774eda250973e808694)
78. 
79. 
80. 
82. 
83. [Discovering Representations for Black-box Optimization](https://arxiv.org/abs/2003.04389) — 🟢 GECCO 2020, MED
84. 
85. [Self-Consuming Generative Models Go MAD](https://api.semanticscholar.org/arXiv:2307.01850) — 🟢 ICLR 2024, HIGH
86. [On the Stability of Iterative Retraining of Generative Models on their own Data](https://api.semanticscholar.org/arXiv:2310.00429) — 🟢 ICLR 2024 spotlight, HIGH
87. [AI models collapse when trained on recursively generated data](https://www.nature.com/articles/s41586-024-07566-y) — 🟢 Nature 2024, MED
88. [Beyond Model Collapse: Scaling Up with Synthesized Data Requires Verification](https://api.semanticscholar.org/arXiv:2406.07515) — 🟢 ICLR 2025, MED
89. [Self-Correcting Self-Consuming Loops for Generative Model Training](https://api.semanticscholar.org/arXiv:2402.07087) — 🟢 ICML 2024, MED
90. [Is Model Collapse Inevitable? Breaking the Curse of Recursion by Accumulating Real and Synthetic Data](https://api.semanticscholar.org/arXiv:2404.01413) — 🟢 COLM 2024, MED
91. [Generative Bayesian Optimization: Generative Models as Acquisition Functions](https://arxiv.org/abs/2510.25240) — 🟢 ICLR 2026, HIGH
92. [Active Flow Matching](https://arxiv.org/abs/2603.00877) — 🟢 arXiv preprint, no venue listed, MED
93. [[Low-density sampling from diffusion models]]
94. [[Diffusion Model as Black Box Opimizer]]
95. 
96. 
98. [Convergence properties of the cross-entropy method for discrete optimization](https://people.smp.uq.edu.au/DirkKroese/ps/CEconv.pdf) — 🟢 Operations Research Letters 2007, HIGH
99. [Sharp Bounds for Genetic Drift in Estimation of Distribution Algorithms](https://api.semanticscholar.org/arXiv:1910.14389) — 🟢 IEEE Transactions on Evolutionary Computation 2020, HIGH
100. [A Closer Look at Model Collapse: From a Generalization-to-Memorization Perspective](https://api.semanticscholar.org/arXiv:2509.16499) — 🟢 NeurIPS 2025 spotlight, HIGH
101. [Model Collapse in the Self-Consuming Chain of Diffusion Finetuning: A Novel Perspective from Quantitative Trait Modeling](https://api.semanticscholar.org/arXiv:2407.17493) — 🟢 ICLR 2025, HIGH
103. [Detecting, Explaining, and Mitigating Memorization in Diffusion Models](https://api.semanticscholar.org/arXiv:2407.21720) — 🟢 ICLR 2024, HIGH
104. 

105. 
106. [Variational Search Distributions](https://arxiv.org/abs/2409.06142) — 🟢 ICLR 2025, HIGH
108. 
109. [Sample-Efficient Optimization in the Latent Space of Deep Generative Models via Weighted Retraining](https://arxiv.org/abs/2006.09191) — 🟢 NeurIPS 2020, HIGH
110. 
111. [Mapping Hearthstone Deck Spaces through MAP-Elites with Sliding Boundaries](https://arxiv.org/abs/1904.10656) — 🟢 GECCO 2019, HIGH
112. [Batch Bayesian Optimization for Replicable Experimental Design](https://arxiv.org/abs/2311.01195) — 🟢 NeurIPS 2023, MED(This is not usefull at all )
113. [Efficient Exploration in Binary and Preferential Bayesian Optimization](https://arxiv.org/abs/2110.09361) — 🟢 arXiv preprint, no venue listed, MED
115. [Posterior Inference with Diffusion Models for High-dimensional Black-box Optimization](https://arxiv.org/abs/2502.16824) — 🟢 ICML 2025, MED
116. [Diffusion Models for Black-Box Optimization](https://proceedings.mlr.press/v202/krishnamoorthy23a.html) — 🟢 ICML 2023, MED
117. [Exploring validation metrics for offline model-based optimisation with diffusion models](https://arxiv.org/abs/2211.10747) — 🟢 Transactions on Machine Learning Research 2024, MED
125. [Closed-Loop Generative Selection: Convergence, Memory, and Noisy Oracles](https://arxiv.org/abs/2607.22211) — 🟢 arXiv preprint, no venue listed, MED
132. [What Ails Generative Structure-based Drug Design: Expressivity is Too Little or Too Much?](https://arxiv.org/abs/2408.06050) — 🟢 AISTATS 2025, LOW
133. [Scaling Gaussian Process Optimization by Evaluating a Few Unique Candidates Multiple Times](https://proceedings.mlr.press/v162/calandriello22a/calandriello22a.pdf) — 🟢 ICML 2022, LOW
134. [Bayesian Optimization for Intrinsically Noisy Response Surfaces](https://arxiv.org/abs/2503.00327) — 🟢 arXiv preprint, no venue listed, LOW
138. [Illuminating Mario Scenes in the Latent Space of a Generative Adversarial Network](https://www.semanticscholar.org/paper/aa79cd3983e6cfd877a830eb3e55a8ded425fe28) — 🟢 AAAI 2021, LOW

### Discrete generation — sampling, dependence and training

Accepted from [[SPIGM 2026 — paper-by-paper relevance screening]]: only the two MED readings below. Read after the existing HIGH/reimplementation sequence. SPIGM is an ICML workshop.

- [A Tale of Two Temperatures: Simple, Efficient, and Diverse Sampling from Diffusion Language Models](https://arxiv.org/abs/2604.09921) — 🟢 SPIGM 2026, **MED — owner-confirmed 2026-09-06** (initial screening: HIGH) — Two temperatures separate token choice from commitment order; read §3, §4.1 and Appendix B for the diversity experiment. — **Full-read decision (2026-09-06): KEEP MED.** Separate randomness in field order from token temperature; useful optional sampling comparison. See [[SPIGM 2026 — full-paper reassessment]].
- [Re-evaluating Confidence Remasking in Masked Diffusion Language Models](https://arxiv.org/abs/2606.12232) — 🟢 SPIGM 2026, **MED — owner-confirmed 2026-09-06** (initial screening: HIGH) — Remasking can improve one-sample success while reducing diversity; compare tuned baselines and actual throughput. — **Full-read decision (2026-09-06): KEEP MED.** A useful audit of remasking gains, strong baselines, actual revisions and runtime. See [[SPIGM 2026 — full-paper reassessment]].

### Generator evaluation


### Surrogate validation

168. [Distributional Energy-Based Models for Uncertainty-Aware Structured LLM Reasoning](https://arxiv.org/abs/2605.18871) — 🟢 EIML (ICML 2026 workshop), MED — targeted reading: §4.2 and Appendix F.8; test whether a team scorer adds value beyond choosing the strongest proposal source. VGC transfer is an audit design, not a validated scoring method.

### Active learning

Selecting or synthesizing queries to learn from their labels: active-learning foundations, uncertainty sampling, and membership queries. See also [[Membership Query Synthesis]].

131. [Active Learning Literature Survey](https://burrsettles.com/pub/settles.activelearning.pdf) — 🟢 Computer Sciences Technical Report 1648, University of Wisconsin-Madison, 2009, MED
135. [On the Relationship between Data Efficiency and Error for Uncertainty Sampling](https://proceedings.mlr.press/v80/mussmann18a.html) — 🟢 ICML 2018, HIGH
136. [Textual Membership Queries](https://www.ijcai.org/proceedings/2020/0369.pdf) — 🟢 IJCAI 2020, HIGH
137. [Queries and Concept Learning](http://machinelearning202.pbworks.com/f/AngluinQueriesConceptLearningfulltext.pdf) — 🟢 Machine Learning 2:319-342, 1988, MED

### Diffusion guidance — methods and theory

Methods for steering diffusion sampling and analyses of how guidance changes generated distributions.

41. [Gradient Guidance for Diffusion Models: An Optimization Perspective](https://www.semanticscholar.org/paper/5aabe3a210270711c72bddaa0290c4edd3db7720) — 🟢 NeurIPS 2024, MED
47. [Robust Guided Diffusion for Offline Black-Box Optimization](https://www.semanticscholar.org/paper/45957487086436f6fcbdd88c6a6dd5f268d9310f)
48. 🟢 [Prof. Mengdi Wang | Guiding Diffusion Models Towards Generative Optimization by INI Satellite Events](https://www.youtube.com/watch?v=8W9qHXN0weg) — talk, Isaac Newton Institute DMLW01 workshop (2024-07-18), MED
49. [Derivative-Free Guidance in Continuous and Discrete Diffusion Models with Soft Value-Based Decoding](https://www.semanticscholar.org/paper/a673ef39237e227381fccf5b1d154d96de428f1d)
51. [Practical and Asymptotically Exact Conditional Sampling in Diffusion Models](https://www.semanticscholar.org/paper/79531b47bb27cb18022891eb2ab1fcb41745fca6) — 🟢 NeurIPS 2023, MED
61. [Simple Guidance Mechanisms for Discrete Diffusion Models](https://www.semanticscholar.org/paper/05f9997d61460fea9b586f98722c4be32f4a8b22)
62. [Unlocking Guidance for Discrete State-Space Diffusion and Flow Models](https://www.semanticscholar.org/paper/c8bfa3abf6a8cbf0a5b3a093661510f34cfa0098)
63. [Classifier-Free Diffusion Guidance](https://www.semanticscholar.org/paper/af9f365ed86614c800f082bd8eb14be76072ad16)
64. [Plug-and-Play Guidance for Discrete Diffusion Models via Gradient-Informed Logit Correction](https://www.semanticscholar.org/paper/30aff605819e4a05c309040dc980ff82a767d828)
77. [Diffusion-LM Improves Controllable Text Generation](https://www.semanticscholar.org/paper/1386b8a11929cf02da291c56aca353e33bbc22ed)
81. [Guess & Guide: Gradient-Free Zero-Shot Diffusion Guidance](https://arxiv.org/abs/2603.07860v1) — 🟢 ReALM-GEN (ICLR 2026 workshop), LOW
102. [What Exactly Does Guidance Do in Masked Discrete Diffusion Models](https://api.semanticscholar.org/arXiv:2506.10971) — 🟢 arXiv preprint, no venue listed, HIGH
107. [Protein Design with Guided Discrete Diffusion](https://arxiv.org/abs/2305.20009) — 🟢 NeurIPS 2023, HIGH
118. [What does guidance do? A fine-grained analysis in a simple setting](https://proceedings.neurips.cc/paper_files/paper/2024/file/9a3942c235daa9c5f62a8598ae81a946-Paper-Conference.pdf) — 🟢 NeurIPS 2024, MED
120. [Classifier-Free Guidance is a Predictor-Corrector](https://arxiv.org/abs/2408.09000) — 🟢 M3L Workshop at NeurIPS 2024, MED
121. [Debiasing Guidance for Discrete Diffusion with Sequential Monte Carlo](https://arxiv.org/abs/2502.06079) — 🟢 Frontiers in Probabilistic Inference workshop at ICLR 2025, MED

### Constrained generation — validity and decoding

Generating candidates that satisfy constraints, including verifier-based expansion and constrained decoding for diffusion models and diffusion LLMs. For learning constraints themselves or generating CP streamliners, see Tier 6.

115. [Constrained Discrete Diffusion](https://arxiv.org/abs/2503.09790) — 🟢 NeurIPS 2025, MED
116. [Constrained Code Generation with Discrete Diffusion](https://arxiv.org/abs/2605.16829) — 🟢 arXiv preprint, no venue listed, MED
117. [Constrained Decoding of Diffusion LLMs with Context-Free Grammars](https://arxiv.org/abs/2508.10111) — 🟢 arXiv preprint, no venue listed, MED

### Reward optimization and alignment — read later

Reward integration is planned for a later stage, not the current implementation (owner, 2026-09-06). Revisit these papers when planning reward objectives, reward-based fine-tuning, or preference alignment. Existing grades describe relevance within that stage; even HIGH entries here are deferred.

76. [Fine-Tuning Discrete Diffusion Models via Reward Optimization with Applications to DNA and Protein Design](https://www.semanticscholar.org/paper/d1461167c9fef8fe3ba129c514acfd14bbe7a51e)
97. [Self-Consuming Generative Models with Curated Data Provably Optimize Human Preferences](https://api.semanticscholar.org/arXiv:2407.09499) — 🟢 NeurIPS 2024 spotlight, HIGH
119. [Reward-Directed Conditional Diffusion: Provable Distribution Estimation and Reward Improvement](https://arxiv.org/abs/2307.07055) — 🟢 NeurIPS 2023, MED
128. [The tractability landscape of diffusion alignment: regularization, rewards, and computational primitives](https://arxiv.org/abs/2605.11361) — 🟢 arXiv preprint, no venue listed, MED



## Automated heuristic design — the LLM writes the search operator

Moved here 2026-09-02 from Tiers 3, 4 and 7; the family was split across three tiers. See [[Hyper Heuristic]] and [[Metaheuristics]].

162. [G-LNS: Generative Large Neighborhood Search for LLM-Based Automatic Heuristic Design](https://www.semanticscholar.org/paper/deefec531e1ccc8798e29f9c38125fdbc34a9a74) — 🟢 arXiv 2602.08253 (2026-02), no venue listed, MED
163. [Beyond Static Evaluation: Co-Evolutionary Mechanisms for LLM-Driven Strategy Evolution in Adversarial Games](https://arxiv.org/abs/2606.10389) — 🟢 AAMAS 2026 MCTF Competition, MED
164. [A Classification of Hyper-Heuristic Approaches: Revisited](https://people.cs.nott.ac.uk/pszeo/docs/publications/HHClassChapterRevisited.pdf) — 🟢 author PDF (🔒 Springer), Handbook of Metaheuristics 3rd ed. ch. 14, 2019, MED
165. [RuleSmith: Multi-Agent LLMs for Automated Game Balancing](https://arxiv.org/abs/2602.06232) — 🟢 arXiv preprint, no venue listed, LOW
166. [Recent advances in selection hyper-heuristics](https://www.sciencedirect.com/science/article/pii/S0377221719306526) — 🟢 CC-BY open access, European Journal of Operational Research 285(2), 2020, LOW

## Tier 4 — The inner loop: battle policy, self-play, game theory, coevolution, evaluation

96. [[A Survey on Self-play Methods in Reinforcement Learning]]
97. [A Unified Game-Theoretic Approach to Multiagent Reinforcement Learning](https://arxiv.org/abs/1711.00832)
98. [[Alpha-Rank Multi-Agent Evaluation by Evolution]]
99. [Game-Theoretic Multiagent Reinforcement Learning](https://www.semanticscholar.org/paper/c3662e9176a7ad90020bdd025c179c5925d0b5b0)
100. [Planning in the Presence of Cost Functions Controlled by an Adversary](https://www.aaai.org/Papers/ICML/2003/ICML03-071.pdf)
101. [Overcoming Valid Action Suppression in Unmasked Policy Gradient Algorithms](https://rlj.cs.umass.edu/2026/papers/Paper1.pdf)
102. [FootsiesGym: A Fighting Game Benchmark for Two-Player Zero-Sum Imperfect-Information Games](https://arxiv.org/abs/2607.06514)
103. [[Grandmaster level in StarCraft II using multi-agent reinforcement learning]]
104. [Superhuman AI for Generals.io Using Self-Play Reinforcement Learning](https://arxiv.org/abs/2606.23348)
105. [Emergent Complexity via Multi-Agent Competition](https://arxiv.org/abs/1710.03748)
106. [Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model](https://www.semanticscholar.org/paper/c39fb7a46335c23f7529dd6f9f980462fd38653a)
107. [An Introduction to Counterfactual Regret Minimization](http://modelai.gettysburg.edu/2013/cfr/cfr.pdf)
108. [Efficient Monte Carlo Counterfactual Regret Minimization in Games with Many Player Actions](https://papers.nips.cc/paper/2012/file/3df1d4b96d8976ff5986393e8767f5b2-Paper.pdf)
109. [Real World Games Look Like Spinning Tops](https://arxiv.org/abs/2004.09468)
110. [Re-evaluating Evaluation](https://arxiv.org/abs/1806.02643)
111. [Coevolutionary Principles](https://www.cs.tufts.edu/comp/150GA/handouts/nchb-main.pdf)
112. [New Methods for Competitive Coevolution](https://cseweb.ucsd.edu/~crosin/newmethods.ps)
113. [A comparison of evaluation methods in coevolution](https://arxiv.org/abs/1905.08723)
114. 🔒 [Tripping Over the Past: Measuring Deceptive Progress in Competitive Coevolutionary Algorithms Employing Hall of Fame](https://doi.org/10.1007/978-3-032-23607-4_32)
160. 

## Tier 5 — Training dynamics and engineering (plasticity, scaling, offline-to-online)

115. [Plasticity Loss in Deep Reinforcement Learning: A Survey](https://api.semanticscholar.org/arXiv:2411.04832)
116. [A Study of Plasticity Loss in On-Policy Deep Reinforcement Learning](https://api.semanticscholar.org/arXiv:2405.19153)
117. [The Primacy Bias in Deep Reinforcement Learning](https://api.semanticscholar.org/arXiv:2205.07802)
118. [The Dormant Neuron Phenomenon in Deep Reinforcement Learning](https://api.semanticscholar.org/arXiv:2302.12902)
119. [Loss of Plasticity in Continual Deep Reinforcement Learning](https://api.semanticscholar.org/arXiv:2303.07507)
120. [Disentangling the Causes of Plasticity Loss in Neural Networks](https://api.semanticscholar.org/arXiv:2402.18762)
121. [Deep Reinforcement Learning with Plasticity Injection](https://api.semanticscholar.org/arXiv:2305.15555)
122. [Maintaining Plasticity in Continual Learning via Regenerative Regularization](https://api.semanticscholar.org/arXiv:2308.11958)
123. [On Warm-Starting Neural Network Training](https://api.semanticscholar.org/arXiv:1910.08475)
124. [Batch size-invariance for policy optimization](https://api.semanticscholar.org/arXiv:2110.00641)
125. [An Empirical Model of Large-Batch Training](https://api.semanticscholar.org/arXiv:1812.06162)
126. [SAPG: Split and Aggregate Policy Gradients](https://api.semanticscholar.org/arXiv:2407.20230)
127. [Preventing Learning Stagnation in PPO by Scaling to 1 Million Parallel Environments](https://rlj.cs.umass.edu/2026/papers/Paper19.pdf)
128. [Improving Sample Efficiency in Multi-Agent Reinforcement Learning for Simulated Football Games via Exploration](https://doi.org/10.1145/3815598.3815627)
129. [Assistax: A Multi-Agent Hardware-Accelerated RL Benchmark for Assistive Robotics](https://rlj.cs.umass.edu/2026/papers/Paper110.pdf)
130. [[Human-Like Goalkeeping in a Realistic Football Simulation a Sample-Efficient Reinforcement Learning Approach]]
131. [AMAGO: Scalable In-Context Reinforcement Learning for Adaptive Agents](https://arxiv.org/abs/2310.09971)
132. [AMAGO-2: Breaking the Multi-Task Barrier in Meta-Reinforcement Learning with Transformers](https://arxiv.org/abs/2411.11188)
133. [Accelerating Online Reinforcement Learning with Offline Datasets](https://arxiv.org/abs/2006.09359)
134. [Critic Regularized Regression](https://arxiv.org/abs/2006.15134)
135. [Multi-Task Reinforcement Learning Enables Parameter Scaling](https://api.semanticscholar.org/arXiv:2503.05126)
136. [Discovering High Quality Chess Puzzles with Offline Reinforcement Learning](https://rlj.cs.umass.edu/2026/papers/Paper147.pdf)
137. [Minimal Ingredients for Reward Assignment from Expert Demonstrations](https://rlj.cs.umass.edu/2026/papers/Paper86.pdf)
138. [Vanquish Your Past: Shifted Imitation Learning in Hades](https://doi.org/10.1145/3815598.3815650)
139. [Ludax: A GPU-Accelerated Description Language for Board Games](https://rlj.cs.umass.edu/2026/papers/Paper46.pdf)
140. [Confidence Intervals for the Interquartile Mean](https://rlj.cs.umass.edu/2026/papers/Paper28.pdf)
167. [Space-sampled Value Decay: Forgetting Mechanisms for Non-stationary Deep Reinforcement Learning](https://arxiv.org/abs/2606.11797) — 🟢 EIML (ICML 2026 workshop), MED — stale value estimates under changing opponents; DQN/SAC evidence, VGC state sampling unresolved. Read §§3–6 after HIGH papers.

## Tier 6 — Constraint learning / CP streamliners

141. [StreamLLM: Enhancing Constraint Programming with Large Language Model-Generated Streamliners](https://jair.org/index.php/jair/article/view/18965)
142. [Streamlined Constraint Reasoning](https://www.cs.cornell.edu/gomes/papers/gomes-sellman-cp04.pdf)
143. [A Model Seeker: Extracting Global Constraint Models from Positive Examples](https://www.semanticscholar.org/paper/d7a118530bc40fe3c2cf3d81fde56b1fc3cb1f30)
144. [Learning Constraints through Partial Queries](https://www.lirmm.fr/~bessiere/Site/stock/aij23.pdf)
151. [Symmetry in Constraint Programming](https://sites.cs.st-andrews.ac.uk/people/ipg1/papers/GentPetriePugetFinalDraft.pdf) — 🟢 author draft (🔒 Elsevier), Handbook of Constraint Programming ch. 10, 2006, MED
152. [Breaking symmetries in all different problems](https://www.ijcai.org/Proceedings/05/Papers/1244.pdf) — 🟢 IJCAI 2005, MED
153. [Dominance breaking constraints](https://people.eng.unimelb.edu.au/pstuckey/papers/domjournal.pdf) — 🟢 author manuscript (🔒 Springer), Constraints 20(2), 2015, MED
154. [Symmetry-Breaking Predicates for Search Problems](https://ix.cs.uoregon.edu/~luks/symmetrybreaking.pdf) — 🟢 KR 1996, LOW
155. [Breaking Row and Column Symmetries in Matrix Models](https://pierre-flener.github.io/research/pub/CP02.pdf) — 🟢 author preprint, CP 2002, LOW
156. [Automatic Generation of Dominance Breaking Nogoods for a Class of Constraint Optimization Problems](http://www.cse.cuhk.edu.hk/~jlee/publ/23/aij23AutoDomBreak.pdf) — 🟢 author preprint, Artificial Intelligence 323, 2023, LOW

## Tier 7 — Peripheral / retired framings (QUBO, annealing, prompt opt, OR applications)

145. [A QUBO Framework for Team Formation](https://arxiv.org/abs/2503.23209)
146. [A Tutorial on Formulating and Using QUBO Models](https://api.semanticscholar.org/arXiv:1811.11538)
147. [Designing Metamaterials with Quantum Annealing and Factorization Machines](https://arxiv.org/abs/1902.06573) — kept only for the pairwise-interaction surrogate idea, not annealing
148. [A Tutorial on Bayesian Optimization](https://api.semanticscholar.org/arXiv:1807.02811)
149. [Evaluating the Stability of the Smogon Tier List for Competitive Pokémon Battling](https://ceur-ws.org/Vol-3926/paper2.pdf) — 🟢 EXAG 2024, MED
158. 
150. 🟢 [Calculating Pokémon Teams v1.0 by John (JaybeeVGC)](https://docs.google.com/document/d/1IIXfm49kbrij4FouD3x9RgYdaM2BetE1/edit) — PuLP linear programming over type coverage; retired framing, LOW
157. [Pokémon team optimization by Nicolas Chagnet](https://nchagnet.eu/blog/pokemon-team-optimization/) — 🟢 blog + PuLP repo, 2025, LOW
159. 
161. 

## Rejected papers — verdicts 2026-08-29

Sorted so they don't get re-suggested. Kept out of the tier ranking above.

Kept (read later):
- [Identifying and Clustering Counter Relationships of Team Compositions in PvP Games for Efficient Balance Analysis](https://arxiv.org/abs/2408.17180) — TMLR 2024. Counter-matrix between team compositions; take the measure, not the balance-analysis pipeline.
- [Team Composition in PES2018 Using Submodular Function Optimization](https://doi.org/10.1109/ACCESS.2019.2919447) — IEEE Access 2019. Team selection as a submodular coverage function with a greedy solver; the closest published cousin of the weighted maximum coverage objective.

Marginal (park unless direction changes):
- [Which Heroes to Pick? Learning to Draft in MOBA Games with Neural Networks and Tree Search](https://arxiv.org/abs/2012.10171) — tree search is the right tool class, but the sequential draft model does not fit simultaneous bring-six selection.
- [Introducing Tales of Tribute AI Competition](https://www.semanticscholar.org/paper/2cd0f87d70a0b6afd79324f959c68f04ef6c0945) — about playing agents for a card game, not team construction.

Rejected:
- [Verifier-Constrained Flow Expansion for Discovery Beyond the Data](https://arxiv.org/abs/2602.15984) — ICLR 2026, was MED. Rejected 2026-09-06: no demonstrated categorical discrete-diffusion update; Appendix G.5 updates only continuous atom positions in the mixed molecular model. Outside the owner's current discrete team-generation scope.
- [Replication or exploration? Sequential design for stochastic simulation experiments](https://arxiv.org/abs/1710.03206) — Technometrics 61(1):7-23, 2019, was MED. Owner's verdict, 2026-09-06: not relevant to the project.
- [Bayesian Optimal Active Search and Surveying](https://arxiv.org/abs/1206.6406) — ICML 2012, was MED. Owner's verdict, 2026-09-06: not relevant to the project.
- [The Offline-Frontier Shift: Diagnosing Distributional Limits in Generative Multi-Objective Optimization](https://arxiv.org/abs/2602.11126) — was MED. Rejected 2026-09-06: offline multi-objective / Pareto-front evaluation is outside the current project scope.
- [Generative Refinement for Low-Budget Black-Box Optimization](https://arxiv.org/abs/2607.00691) — was MED. Rejected at the owner's request, 2026-09-06; not useful for the current scope.
- [QUBO, I Choose You: Building an Optimal Pokémon Team with a Quantum Computer](https://www.dwavequantum.com/events-section/events/webinar-qubo-i-choose-you-building-an-optimal-pokemon-team-with-a-quantum-computer/) — D-Wave webinar 2019; species/type-level QUBO, formulation only in the recording, no published code; the pairwise-surrogate idea is already carried by the metamaterials entry.
- [A Threshold of ln n for Approximating Set Cover](https://courses.cs.duke.edu/spring07/cps296.2/papers/p634-feige.pdf) — Feige 1996, hardness-of-approximation proof; requires approximation-algorithms background I do not have. The usable result (greedy gets within 1−1/e of optimal max coverage) is already covered by the submodular survey entries.
- [Heuristically Adaptive Diffusion-Model Evolutionary Strategy](https://arxiv.org/abs/2411.13420) — HADES: a diffusion model as the offspring sampler inside a continuous-parameter evolutionary strategy (CMA-ES family). Real-valued genomes, toy fitness landscapes (double-peak, Rastrigin) and cart-pole only. Loop shape matches our direction but the object optimized does not; the idea is already carried by Diffusion Models are Evolutionary Algorithms and Design by adaptive sampling. 🟢 Published as Advanced Science 13(20), 2026 (CC BY, PMC13067789): adds MountainCar and Lunar Lander, and says in print “we have applied our algorithm only to continuous parameter spaces.”
- [Evolutionary Optimization in Uncertain Environments — A Survey](https://www.honda-ri.de/pubs/pdf/1660.pdf) — Jin & Branke, IEEE Transactions on Evolutionary Computation 2005; superseded by the same group's newer surveys on evolutionary optimization in uncertain and dynamic environments.
- [DraftRec: Personalized Draft Recommendation for Winning in Multi-Player Online Battle Arena Games](https://arxiv.org/abs/2204.12750) — recommender for each human player's champion preferences; no preference-draft phase exists in this project.
- [Modeling Game Avatar Synergy and Opposition through Embedding in Multiplayer Online Battle Arena Games](https://arxiv.org/abs/1803.10402) — its own arXiv page notes it was rejected by AIIDE 2017; superseded idea.
- [A Recommender System for Hero Line-Ups in MOBA Games](https://ojs.aaai.org/index.php/AIIDE/article/view/12938) — association-rule recommender, not an optimizer.
- [Draft-Analysis of the Ancients: Predicting Draft Picks in DotA 2 Using Machine Learning](https://ojs.aaai.org/index.php/AIIDE/article/view/12899) — predicts what humans will pick; the project searches for counters instead.
- [Finding a Team of Experts in Social Networks](https://cs-people.bu.edu/evimaria/papers/fp525-lappas.pdf) — objective is communication cost on a social graph, unrelated to coverage plus clauses.
- [NBA Chemistry: Positive and Negative Synergies in Basketball](https://philipmaymin.com/papers/Maymin%20Maymin%20and%20Shen%20-%20NBA%20Chemistry%20-%20IJCSS.pdf) — heuristic synergy extraction, no transferable search method.
- [OptMatch: Optimized Matchmaking via Modeling the High-Order Interactions on the Arena](https://linxiagong.github.io/OptMatch/KDD2020_OptMatch.pdf) — matchmaking pairs players; it does not build teams.
- [Picking Winners in Daily Fantasy Sports Using Integer Programming](https://arxiv.org/abs/1604.01455) — the source list titled this as a football-team paper, but the arXiv record is the fantasy-sports integer-programming paper; either way the framing is retired here.
- [A Comparison of Coevolution, Fixed, and Hybrid Training for Evolving Agents that Play Tales of Tribute Videogame](https://doi.org/10.1007/978-3-032-23607-4_33) — coevolution for playing agents, not team selection.
- [Large Language Models As Evolution Strategies](https://arxiv.org/abs/2402.18381) — a large language model used as an in-context recombination operator for evolution strategies; continuous black-box functions, no constraints, no team structure.
- [SOPL: A Sequential Optimal Learning Approach to Automated Prompt Engineering in Large Language Models](https://aclanthology.org/2025.findings-emnlp.1155/) — optimizes natural-language prompt templates; wrong domain, and it economizes on expensive model queries rather than simulator evaluations.
- 🟢 [A Tight Combinatorial Algorithm for Submodular Maximization Subject to a Matroid Constraint](https://arxiv.org/abs/1204.4526) — Filmus & Ward, arXiv 2012 (rev. 2013), no journal venue stated. Gets 1−1/e under a matroid constraint; there is no matroid here. Its restricted-curvature extension (1−e^(−c))/c still assumes the function is submodular, and the non-submodular version of that result is already carried by Guarantees for Greedy Maximization of Non-submodular Functions.
- 🟢 [Chamberlin–Courant Rule with Approval Ballots: Approximating the MaxCover Problem with Bounded Frequencies in FPT Time](https://jair.org/index.php/jair/article/view/11095) — Skowron & Faliszewski, Journal of Artificial Intelligence Research 60 (2017). Fixed-parameter-tractable approximation for multiwinner voting; the MaxCover shape matches but the machinery is exact committee election over known ballots, not noisy black-box search.
- 🟢 [An Online Algorithm for Maximizing Submodular Functions](https://papers.nips.cc/paper/3569-an-online-algorithm-for-maximizing-submodular-functions) — Streeter & Golovin, NIPS 2008, pp. 1577–1584. Online no-regret setting, adjacent to the family already ruled out; here f is queried offline in batches, not as a sequential regret game.
- 🟢 [Subset Selection by Pareto Optimization](https://papers.nips.cc/paper_files/paper/2015/hash/b4d168b48157c623fbd095b4a565b5bb-Abstract.html) — Qian, Yu & Zhou, NIPS 2015. Bi-objective evolutionary algorithm over k-subsets of a ground set, assuming an exact value oracle; the search unit here is a full team, and f is noisy.
- 🟢 [Approximation Guarantees of Local Search Algorithms via Localizability of Set Functions](https://arxiv.org/abs/2006.01400) — Kaito Fujii, ICML 2020, PMLR v119 pp. 3327–3336. Localizability has to be derived from restricted strong concavity and smoothness of a differentiable relaxation, which a win rate measured by battles does not have; exact value oracle.
- 🟢 [The Unreasonable Effectiveness of Greedy Algorithms in Multi-Armed Bandit with Many Arms](https://arxiv.org/abs/2002.10121) — Bayati, Hamidi, Johari & Khosravi, NeurIPS 2020. Bandit family ruled out 2026-08-20; the many-armed regret setting is not the search here.
- 🟢 [Sample Efficiency Matters: A Benchmark for Practical Molecular Optimization](https://www.semanticscholar.org/paper/e318c18e6f3faa5c514e6b85270b2211603fc419) — owner's verdict 2026-09-02: does not contribute to this project.
- 🟢 [Diffusion-based Evolutionary Optimization for 3D Multi-Objective Molecular Generation](https://www.semanticscholar.org/paper/e01f94b7bcf583241e5afe9d4a22bba4504778c9) — 2025 preprint, no venue, 1 citation (checked 2026-09-02). Rejected on uptake.
- 🟢 [Reward-Guided Discrete Diffusion via Clean-Sample Markov Chain for Molecule and Biological Sequence Design](https://arxiv.org/abs/2602.09424) — ReALM-GEN (ICLR 2026 workshop). Rejected 2026-09-02 with the molecular-optimization family.
- 🟢 [On the Design Space of Discrete Diffusion Online Adaptation for Molecular Optimization](https://arxiv.org/abs/2607.02834) — arXiv preprint, no venue listed. Rejected 2026-09-02 with the molecular-optimization family.
- 🟢 [On failure modes in molecule generation and optimization](https://epub.jku.at/obvulioa/download/pdf/5687408?originalFilename=true) — Drug Discovery Today: Technologies 2020, was HIGH. Rejected 2026-09-02 with the molecular-optimization family.
- 🟢 [Genetic algorithms are strong baselines for molecule generation](https://arxiv.org/abs/2310.09267) — arXiv preprint, no venue listed, was HIGH. Rejected 2026-09-02 with the molecular-optimization family.
- 🟢 [HardFlow: Hard-Constrained Sampling for Flow-Matching Models via Trajectory Optimization](https://arxiv.org/abs/2511.08425) — ReALM-GEN (ICLR 2026 workshop), was LOW. Continuous optimal control; team legality is categorical, and Constrained Discrete Diffusion carries this job.
- 🟢 [Trust-Region Noise Search for Black-Box Alignment of Diffusion and Flow Models](https://arxiv.org/abs/2603.14504) — ReALM-GEN (ICLR 2026 workshop), was LOW. Searches a continuous source-noise vector against many reward calls; battles are too costly and too noisy for that.
- 🟢 [Discrete Diffusion Inference-Time Control with Nested Sequential Monte Carlo](https://arxiv.org/abs/2608.20123) — ReALM-GEN (ICLR 2026 workshop), was LOW. Feynman-Kac particle steering needs a reward on partial sequences; a half-built team cannot be battled.
- 🔒 DDNO: Discrete Diffusion Noise Optimization (Eyring et al.) — ReALM-GEN (ICLR 2026 workshop), no open copy, ungraded. Discrete twin of Trust-Region Noise Search; noise-space search needs many reward calls and battles are too costly.
- 🟢 [Diffusion Active Learning: Towards Data-Driven Experimental Design in Computed Tomography](https://arxiv.org/abs/2504.03491) — 2025 preprint, no venue, was MED. Inverse problem: acquisition picks the next measurement of a fixed object, informativeness-denominated.
- 🟢 [Active Diffusion-Based Inference for Ill-Posed Inverse Problems under Incomplete Priors](https://arxiv.org/abs/2608.27080) — IJCAI-ECAI 2026 (venue was missing), was LOW. Inverse problem: recovers hidden true parameters, acquisition driven by posterior uncertainty.
- 🟢 [Offline Model-Based Optimization: Comprehensive Review](https://arxiv.org/abs/2503.17286) — TMLR 2026 (Survey Certification), was HIGH. Offline MBO assumes no oracle access; Showdown gives one on demand. Owner's call 2026-09-02: no surveys.
- 🟢 [When does return-conditioned supervised learning work for offline reinforcement learning?](https://arxiv.org/abs/2206.01079) — NeurIPS 2022, was HIGH. Owner's verdict 2026-09-02: an offline-RL policy result, no design generator; not useful.

Sweep-added same family (2026-08-29) — all prediction or recommendation, none run a search:
- [Real-time eSports Match Result Prediction](https://arxiv.org/abs/1701.03162) — Dota 2 win prediction from in-match features; predicts outcomes, does not construct teams.
- [Win Prediction in Esports: Mixed-Rank Match Prediction in Multi-player Online Battle Arena Games](https://arxiv.org/abs/1711.06498) — same win-prediction family.
- [League of Legends: Real-Time Result Prediction](https://arxiv.org/abs/2309.02449) — in-match LSTM prediction; no team construction.
- [Using Machine Learning to Predict Game Outcomes Based on Player-Champion Experience in League of Legends](https://arxiv.org/abs/2108.02799) — predicts outcomes from player skill, explicitly "regardless of team composition".
- [MobaQA: MOBA Games Prediction Based on Large Language Model Fine-Tuning](https://www.semanticscholar.org/paper/MobaQA%3A-MOBA-Games-Prediction-Based-on-Large-Nie-Wang/) — LLM fine-tuning to predict real-time win rate and match outcome; prediction, not search (IEEE Transactions on Games 2026).

## SPIGM screening history — not in the active reading queue

The following 16 entries retain their original grades and assessments for reference. None is currently accepted; the only accepted SPIGM readings are #169 and #170 above.

- [Learn from Your Mistakes: Self-Correcting Masked Diffusion Models](https://arxiv.org/abs/2602.11590) — 🟢 SPIGM 2026, HIGH — Learn to correct model-generated errors; distinguish corrupted inputs from mutations treated as positive training targets. — **Full-read decision (2026-09-06): DEFER.** Learned correction is legitimate, but no established benefit beyond current repair or for the active data comparison. See [[SPIGM 2026 — full-paper reassessment]].
- [Breaking the Factorization Barrier in Diffusion Language Models](https://arxiv.org/abs/2603.00045) — 🟢 SPIGM 2026, HIGH — Use a tractable joint head for dependencies between simultaneously generated fields; extra cost and diversity trade-offs remain. — **Full-read decision (2026-09-06): DEFER.** A joint head chiefly addresses simultaneous field generation, which the current sampler does not do. See [[SPIGM 2026 — full-paper reassessment]].
- [Internal Data Repetition Destroys Language Models](https://arxiv.org/abs/2606.24998) — 🟢 SPIGM 2026, MED — Separate duplicate concentration from total training exposure when comparing mutations, elites and original-data replay. — **Latest owner-based decision (2026-09-06): DROP.** Exact duplicate removal is already available. The paper studies concentrated exact repetition; extending it to distinct mutations did not justify this reading. — **Full-read decision (2026-09-06): KEEP MED.** Useful controls for repeated exposure and concentrated team lineages in training-data comparisons. See [[SPIGM 2026 — full-paper reassessment]].
- [Finetuning Generative Models to Match Feature Distributions](https://arxiv.org/abs/2606.19496) — 🟢 SPIGM 2026, MED — Match chosen feature distributions with a pretrained-model KL anchor; conceptual guidance for preserving team structure. — **Full-read decision (2026-09-06): KEEP MED.** Feature-distribution matching gives a concrete possible anchor, with important feature and diversity caveats. See [[SPIGM 2026 — full-paper reassessment]].
- [The Confidence Shortcut: A Reasoning Failure Mode of Masked Diffusion Models](https://arxiv.org/abs/2605.29123) — 🟢 SPIGM 2026, MED — Check confidence order against dependency difficulty; include the paper’s Sudoku success case, not only failures. — **Full-read decision (2026-09-06): DEFER.** Relevant if confidence-aligned training becomes active; the current model does not use it. See [[SPIGM 2026 — full-paper reassessment]].
- [Uniform Diffusion Models Revisited: Leave-One-Out Denoiser and Absorbing State Reformulation](https://arxiv.org/abs/2605.22765) — 🟢 SPIGM 2026, MED — Understand denoiser/objective mismatch in uniform diffusion; not an established bug in the current masked model. — **Full-read decision (2026-09-06): DEFER.** A uniform-corruption objective issue does not establish a defect in the current mask-only model. See [[SPIGM 2026 — full-paper reassessment]].
- [Tensor-Train Joint Modeling for Few-Step Discrete Diffusion](https://arxiv.org/abs/2607.03788) — 🟢 SPIGM 2026, MED — Low-rank joint categorical heads; assess token-order bias and head cost before applying to exchangeable teammates. — **Full-read decision (2026-09-06): DEFER.** Low-rank joint sampling is a future parallel-decoding option, with ordering and rank assumptions. See [[SPIGM 2026 — full-paper reassessment]].
- [Latent-Augmented Discrete Diffusion Models](https://arxiv.org/abs/2510.18114) — 🟢 SPIGM 2026, MED — Auxiliary latents offer an alternative to explicit joint heads for cross-field dependencies; extra encoder/compute required. — **Full-read decision (2026-09-06): DEFER.** A latent channel adds substantial modeling requirements before a demonstrated current need. See [[SPIGM 2026 — full-paper reassessment]].
- [Learned Relay Representations for Forward-Thinking Discrete Diffusion Models](https://arxiv.org/abs/2605.22967) — 🟢 SPIGM 2026, MED — Carry learned hidden state between denoising passes; structured-task ablations, with added training and memory cost. — **Full-read decision (2026-09-06): DEFER.** Compatible with sequential generation, but a persistent-state benefit has not been established here. See [[SPIGM 2026 — full-paper reassessment]].
- [Recursive Scaling in Masked Diffusion Models](https://arxiv.org/abs/2606.18022) — 🟢 SPIGM 2026, MED — Shared-block recursion as learned constraint propagation; read the parameter- and compute-controlled comparisons. — **Full-read decision (2026-09-06): DEFER.** A future capacity-versus-compute study, not evidence about the current data or CEM questions. See [[SPIGM 2026 — full-paper reassessment]].
- [Understanding and Accelerating the Training of Masked Diffusion Language Models](https://arxiv.org/abs/2605.13026) — 🟢 SPIGM 2026, MED — Training mask regimes and loss weighting affect data comparisons; text-locality assumptions need checking on teams. — **Latest owner-based decision (2026-09-06): DROP.** The owner confirms diffusion training is not a problem. Mask-context diagnostics do not supply a sufficiently relevant current reading task. — **Full-read decision (2026-09-06): KEEP MED.** Mask-context allocation and irreducible loss matter when interpreting training-data comparisons. See [[SPIGM 2026 — full-paper reassessment]].
- [Time-Annealed Perturbation Sampling: Diverse Generation for Diffusion Language Models](https://arxiv.org/abs/2601.22629) — 🟢 SPIGM 2026, MED — Early conditioning perturbation followed by late refinement; a diversity idea whose categorical conditioning transfer is untested. — **Full-read decision (2026-09-06): DROP.** Weak categorical-conditioning transfer, plus filtering and reporting issues undermine the proposed payoff. See [[SPIGM 2026 — full-paper reassessment]].
- [Hacking Generative Perplexity: Why Unconditional Text Evaluation Needs Distributional Metrics](https://arxiv.org/abs/2606.08417) — 🟢 SPIGM 2026, MED — A good frozen-scorer metric plus token entropy can hide poor joint samples; reference fidelity is a diagnostic, not the search objective. — **Latest relevance decision (2026-09-06): DROP.** The project does not use generative perplexity and already plans full-team/composition diversity measurements. The earlier team-template stress-test connection was an analogy, insufficient to justify reading this paper. — **Full-read decision (2026-09-06): KEEP MED.** Directly useful when choosing diversity metrics: healthy marginals can conceal poor joint samples. See [[SPIGM 2026 — full-paper reassessment]].
- [DUEL: Exact Likelihood for Masked Diffusion via Deterministic Unmasking](https://arxiv.org/abs/2603.01367) — 🟢 SPIGM 2026, MED — Exact likelihood for specified deterministic unmasking; does not cover arbitrary random/remasking/truncated samplers. — **Full-read decision (2026-09-06): DROP.** Exact likelihood for a specified reveal schedule is not a current project requirement. See [[SPIGM 2026 — full-paper reassessment]].
- [TUBE: Tangent Upper Bound on Evidence for Discrete Diffusion Language Models](https://arxiv.org/abs/2605.24292) — 🟢 SPIGM 2026, MED — Upper/lower likelihood bounds expose rankings that ELBO alone cannot establish; likelihood still does not measure team strength. — **Full-read decision (2026-09-06): DROP.** Sophisticated likelihood bounds do not answer the current team-quality or diversity decisions. See [[SPIGM 2026 — full-paper reassessment]].
- [Contrastive Distribution Matching for Amortized Sequential Monte Carlo in Discrete Diffusion](https://arxiv.org/abs/2605.23346) — 🟢 SPIGM 2026, MED — Deferred: learn partial-state twist values from complete reward-labelled samples; positive collection costs and noisy binary rewards remain concerns. — **Full-read decision (2026-09-06): DEFER.** Keep as a future reward-integration reference, requiring a suitable reward and trajectory model. See [[SPIGM 2026 — full-paper reassessment]].

## Where more papers live

- [[SPIGM 2026 — paper-by-paper relevance screening]] — 210 workshop-site papers plus one ICML-only entry screened; **2 accepted MED readings, none HIGH**. Initial screening history: 18 additions (4 HIGH, 14 MED; one reward-deferred), 7 held, 186 skipped; the other 16 additions are outside the active queue.
- [[EIML 2026 — paper-by-paper relevance screening]] — reassessed 2026-09-06: 2 MED additions, 57 skipped, 20 insufficiently verified; no HIGH. Initial result: 59 skipped and no additions; the implementation-focused reading threshold was too narrow.
- [[_Workshops index]]
- [[VGC AI Competition (IEEE CoG)]]
- [[PokeAgent Challenge (NeurIPS 2025)]]
- [[Finding the Frame — papers relevant to the counter-team search]]
- [[Reading path — neural CO for team selection]]
- [Teaching/RESOURCES.md](Teaching/RESOURCES.md) — grades and one-line verdicts for the analogue sweep, now merged above
- [[Citation]]

# Papers

150. /goal I think [Todo Section.md](Todo%20Section.md) there is huge problem that text too long and will not follow the instruction properpely instead the right direction to strip all the paper into a new markdown called reading list we will organize the paper reading list by the order relevant to the projects, make sure to only keep the title of the paper in this format [Title of the paper](link to the paper) or it have markdown file just have [[Title of the paper]] Make sure don't miss any paper.I just the the paper be organize the relevance of the paper not the orgnize base on the section in where they comes from I want a pure list number 1 to n base on relevance to my project. There is no link to deleted files


## Neural combinatorial optimization — grading 2026-09-07

The six current entries were assessed against legal team generation, generator evaluation, and model-based counter-team search, with reward integration deferred. Neural Large Neighborhood Search: **MED**; DIFUSCO: **LOW** (unchanged); DiffUCO: **LOW** (previously MED); Tackling Prevalent Conditions: **LOW**; Learning TSP Requires Rethinking Generalization: **MED**; Denoising Diffusion-Based Evolutionary Algorithm: **MED**. No HIGH entries. MED means selective reading for a specific design or evaluation decision, not a proven Pokémon implementation. Earlier grades above remain historical. See [[Neural combinatorial optimization - relevance grades]] for primary sources and rationale. No entries were removed.

During final verification, Learning TSP was absent from the active list following a concurrent edit. That removal was preserved. The active subsection therefore ends with **2 MED and 3 LOW**; the six-paper assessment records what was reviewed, not acceptance of the removed paper.


## COExpander and Unify ML4TSP — accepted 2026-09-07

Owner requested both forward-citation candidates be added and graded. Both receive **MED for selective reading**, with implementation deferred pending a concrete need; neither is HIGH or a demonstrated solution to noisy VGC search. The grades express limited design/evaluation usefulness and do not override the caveats in [[Variational Annealing - forward citation evidence]].

- **COExpander — MED:** Search and black-box optimization → Neural combinatorial optimization. Read for adaptive completion of partial candidates; expert supervision and task-specific determination operators limit direct reuse. Placed first in that subsection as the newly selected generator-design reference.
- **Unify ML4TSP — MED:** Search and black-box optimization → Optimization benchmarks and validation. Read for separating learned proposal quality from downstream construction and improvement search. Placed first in that subsection because controlled generator/search comparisons fit the online project better than the existing offline-only benchmarks.

Both active entries use only linked paper titles, following the owner's formatting preference. Grades and reasons are recorded here rather than appended to the entries.
