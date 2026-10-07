---
created_at: 2026-08-24
updated_at: '2026-09-28'
type: reading-list
tags:
- reading-list
- vgc
- combinatorial-search
- reinforcement-learning
ordering: By project topic and method subtopic; prior order within each subtopic
sequencing_rule: HIGH, then reimplement, then MED; reward papers remain deferred
spigm_accepted: 2
history: '[[Reading List — notes and decision history]]'
---

# Reading List

Browse by topic and subtopic. Grades stay beside each paper: read **HIGH** first, then reimplement, then **MED**. Papers without a grade remain ungraded. Reward readings are deferred until reward integration is planned.

**SPIGM: only A Tale of Two Temperatures and Re-evaluating Confidence Remasking are accepted, both MED.**

[[#Pokémon teams and benchmarks|Pokémon]] · [[#Team building in other games|Other games]] · [[#Search and black-box optimization|Search]] · [[#Diffusion models and sampling|Diffusion]] · [[#Retraining, diversity and model collapse|Retraining]] · [[#Surrogates and active learning|Surrogates]] · [[#Legality and constraint programming|Constraints]] · [[#Automated heuristic design|Heuristic design]] · [[#Battle policies and self-play|Battle policies]] · [[#Training dynamics and engineering|Training]]

## Pokémon teams and benchmarks

### Team construction and metagame analysis

- 🔒 [An Adversarial Approach for Automated Pokémon Team Building and Metagame Balance](https://doi.org/10.1109/tg.2023.3273157)
- [Pokémon GO Team Optimization](https://journals-sol.sbc.org.br/index.php/jis/article/view/6773)
- [A Framework for Predicting the Impact of Game Balance Changes Through Meta Discovery](https://arxiv.org/abs/2409.07340)
- [From Rules to Nash Equilibria](https://arxiv.org/abs/2607.08692)

### Battle agents

- [[Human-Level Competitive Pokémon via Scalable Offline Reinforcement Learning with Transformers]]
- [PokéChamp: An Expert-level Minimax Language Agent](https://arxiv.org/abs/2503.04094)
- [PokéLLMon: A Human-Parity Agent for Pokémon Battles with Large Language Models](https://arxiv.org/abs/2402.01118)

### Benchmarks and competitions

- [[VGC-Bench Towards Mastering Diverse Team Strategies in Competitive Pokémon]]
- [The PokeAgent Challenge: Competitive and Long-Context Learning at Scale](https://www.semanticscholar.org/paper/80c6fcde8b5e8190a2e970db128eaf095887a46b)

## Team building in other games

### Quality diversity and deck-space exploration

- [Deep Surrogate Assisted MAP-Elites for Automated Hearthstone Deckbuilding](https://www.semanticscholar.org/paper/e46811df6731df51fb5c209dea7b835d0cd734da) — direction paper — the loop itself: generate decks → battle vs fixed opponents → refit → repeat · [[Reading List — notes and decision history|Reading notes]]
- [Mapping Hearthstone Deck Spaces through MAP-Elites with Sliding Boundaries](https://arxiv.org/abs/1904.10656) — 🟢 GECCO 2019, HIGH · [Alternate source](https://www.semanticscholar.org/paper/9eb4d5e751b83a2afb240b93c0c2b1ea025ac14f) · [[Reading List — notes and decision history|Reading notes]]
- [CabbageCat's Blogs — MAP-Elites, CMA-ME, Differentiable QD step-by-step](https://szhaovas.github.io/2022-09-15-me/) — 🟢 grad-researcher QD tutorial series (MAP-Elites → CMA-ME → Differentiable QD); read before/with the MAP-Elites Hearthstone papers
- 🔒 [Exploring the Hearthstone Deck Space](https://dl.acm.org/doi/10.1145/3235765.3235791) — 🟢 [open copy: Internet Archive capture of the ACM PDF](https://web.archive.org/web/20200602172549/https://dl.acm.org/doi/pdf/10.1145/3235765.3235791?download=true)

### Evolutionary deck building and playtesting

- [Evolving the Hearthstone Meta](https://arxiv.org/abs/1907.01623)
- 🔒 [Automated playtesting in collectible card games using evolutionary algorithms](https://doi.org/10.1016/j.knosys.2018.04.030)
- [Evolutionary Arena Deckbuilding with Active Genes](https://arxiv.org/abs/2001.01326)

### Drafting and deck recommendation

- [Drafting in Collectible Card Games via Reinforcement Learning](https://www.sbgames.org/proceedings2020/ComputacaoFull/209690.pdf)
- [Q-DeckRec: A Fast Deck Recommendation System for Collectible Card Games](https://www.semanticscholar.org/paper/ca1e69d0d5a74a05ee390ea6aa0129965a83950b)

### Game prototyping and competition reports

- [Cardiverse: Harnessing LLMs for Novel Card Game Prototyping](https://www.semanticscholar.org/paper/6590ba146e330de09ce7798164cc1ddeecfc0b17)
- [Summarizing Strategy Card Game AI Competition](https://www.semanticscholar.org/paper/551a4a4d231e24bacbd7a1285e8f44dd553e5b3f) · [[Reading List — notes and decision history|Reading notes]]

## Search and black-box optimization

### Classical search and set functions

- [Optimistic Tree Searches for Combinatorial Black-Box Optimization](https://proceedings.neurips.cc/paper_files/paper/2022/hash/d6099a36f6c1720438de00c366aa1737-Abstract-Conference.html) — NeurIPS 2022, ungraded — first read among the DIFUSCO references for search framing (§§2–3); discrete black-box search, with additional work needed for noisy battle evaluations. · [[DIFUSCO references - relevance to counter-team search|Relevance notes]]
- [[Simulating Annealing]]
- [[Submodular Function]]
- [[Local Search]] or Direct search 

### Neural combinatorial optimization

- [COExpander: Adaptive Solution Expansion for Combinatorial Optimization](https://proceedings.mlr.press/v267/ma25r.html) ( consider this background information )
- [Neural Large Neighborhood Search for the Capacitated Vehicle Routing Problem](https://arxiv.org/abs/1911.09539)
- [A Denoising Diffusion-Based Evolutionary Algorithm Framework: Application to the Maximum Independent Set Problem](https://www.semanticscholar.org/paper/9b21ee2ac4fde7868f485774eda250973e808694)
- [DIFUSCO: Graph-based Diffusion Solvers for Combinatorial Optimization](https://www.semanticscholar.org/paper/5f90d43e6ece5c6ee6e8186e4b57d46c85377713)
- [A Diffusion Model Framework for Unsupervised Neural Combinatorial Optimization](https://www.semanticscholar.org/paper/4a5ba3075d61f6258059dc311f761eb85848b22d)
- [Tackling Prevalent Conditions in Unsupervised Combinatorial Optimization](https://www.semanticscholar.org/paper/f7daa82747a13314c42a1df0c8eb2d4958ac27d6)
- [From Distribution Learning in Training to Gradient Search in Testing for Combinatorial Optimization](https://www.semanticscholar.org/paper/b933fb2ab8cc4704953a6e97576a921eee56dd76) — 🟢 NeurIPS 2023, LOW

### Adaptive search distributions and the cross-entropy method (Reading this now)

- [Design by adaptive sampling](https://arxiv.org/abs/1810.03714) — It think adaptive sampling can be used for something else, also take about the reiventing the cross entropy method, this should be higher than on the list compare to active search method

### Generative optimization and learned representations

- [Trust-Region Noise Search for Black-Box Alignment of Diffusion and Flow Models](https://arxiv.org/abs/2603.14504)
	- Paper it reference
		- Scalable Global Optimization via Local Bayesian Optimization
	- I liked the SMC approach that might be a useful direction
		- KF sterring
		- DAS, because it require reward i should not use then instead if I plan to add reward then KF won ahead
- [Regret Analysis of Guided Diffusion for Black-Box Optimization over Structured Inputs](Ahttps://arxiv.org/abs/2605.10385)
	- I wished it has good diagram it does not any good on
	- I don't know what is an acquistion score is
	- I need to find some youtube vide about regrest anaylsis
- [Diffusion Large Language Models for Black-Box Optimization](https://arxiv.org/html/2601.14446v1) — 2026, arXiv preprint, ungraded — combines a prompted diffusion LLM with a Gaussian-process predictor and masked diffusion tree search over partially masked candidates; compares against an autoregressive LLM optimizer and conventional diffusion methods.
	- I feel like the section in the related work section about llm for bbo will be great for why I choose an llm 
- [Training Diffusion Language Models for Black-Box Optimization (DiBO)](https://arxiv.org/html/2603.17919v1) — 2026, arXiv preprint, ungraded — adds domain adaptation, supervised fine-tuning, and reinforcement learning based on label improvements; examines steering generation through training beyond supplying examples in a prompt.
	- 2.3 LLMs for Black-Box Optimization
	- Inverse Methods.
- [Large Language Models as Optimizers](https://arxiv.org/abs/2309.03409) — 🟢 ICLR 2024, MED ( It just propose and just create meta prompt afterward has a lot of citation ) it should be like the baseline
- [Language Model Crossover: Variation through Few-Shot Prompting](https://arxiv.org/abs/2302.12170) — 🟢 ACM TELO 2024, MED 
	- Read this next Combinatorial Optimization for All: Using LLMs to Aid Non-Experts in Improving Optimization Algorithms
		- Read this next pyCombinatorial - A library to solve TSP (Travelling Salesman Problem) using Exact Algorithms, Heuristics and Metaheuristics
- Steering Generative Models with Experimental Data for Protein Fitness Optimization
	- For through this method of guidance Decoupled Annealing Posterior Sampling (DAPS)
- [Discovering Representations for Black-box Optimization](https://arxiv.org/abs/2003.04389) — 🟢 GECCO 2020, MED 
	- Interesting example that use map elit
- [Generative Bayesian Optimization: Generative Models as Acquisition Functions](https://arxiv.org/abs/2510.25240) — 🟢 ICLR 2026, HIGH
	- go through this paper
- [[Active Flow Matching]] 🟢 arXiv preprint, no venue listed, MED
- [Posterior Inference with Diffusion Models for High-dimensional Black-box Optimization](https://arxiv.org/abs/2502.16824) — 🟢 ICML 2025, MED
- [Diffusion Models for Black-Box Optimization](https://proceedings.mlr.press/v202/krishnamoorthy23a.html) — 🟢 ICML 2023, MED
- [Feedback Efficient Online Fine-Tuning of Diffusion Models](https://www.semanticscholar.org/paper/6a225bb2454f66d1b106bbae7e5129d6569aa260) — 🟢 ICML 2024, HIGH
- [Diffusion-BBO: Diffusion-Based Inverse Modeling for Online Black-Box Optimization](https://www.semanticscholar.org/paper/d3e0e933576e485274e66e6df452dd0a0eb78765) — 🟢 NeurIPS 2024 BDU workshop, MED
- [Generative Refinement for Low-Budget Black-Box Optimization](https://arxiv.org/abs/2607.00691) — 🟢 arXiv preprint, no venue listed, MED
- [Heuristically Adaptive Diffusion-Model Evolutionary Strategy](https://www.semanticscholar.org/paper/2b7f4ccb8e001398883b756aa1cbeda7275b2bde) — 🟢 Advanced Science 2026, MED
- [Diffusion Models are Evolutionary Algorithms](https://www.semanticscholar.org/paper/cfa4ef02c7ca657506768fb53075caecc1b7161c) — 🟢 ICLR 2025, MED
- [Fine-Tuning Discrete Diffusion Models via Reward Optimization with Applications to DNA and Protein Design](https://proceedings.iclr.cc/paper_files/paper/2025/hash/771e09dd204ea339da0d8114c48afd21-Abstract-Conference.html) — 🟢 ICLR 2025, MED
- [Large Language Models to Enhance Bayesian Optimization](https://arxiv.org/abs/2402.03921) — 🟢 ICLR 2024, LOW
- [Illuminating Mario Scenes in the Latent Space of a Generative Adversarial Network](https://www.semanticscholar.org/paper/aa79cd3983e6cfd877a830eb3e55a8ded425fe28) — 🟢 AAAI 2021, LOW

### Noisy evaluations and Bayesian optimization

- [SMAC3: A Versatile Bayesian Optimization Package for Hyperparameter Optimization](https://arxiv.org/abs/2109.09831) — JMLR 2022, ungraded — practical model-based search baseline to adapt to categorical team choices and noisy battle evaluations; configure repeated evaluations and encode team legality. · [Software documentation](https://automl.github.io/SMAC3/latest/3_getting_started/)
- [Bayesian Optimization of Combinatorial Structures](https://proceedings.mlr.press/v80/baptista18a.html) — Ricardo Baptista and Matthias Poloczek, ICML 2018, ungraded — owner-selected (2026-09-28): vocabulary and methodology for expensive combinatorial black-box search; study how the model learns interactions between discrete choices from limited evaluations. · [PDF](https://proceedings.mlr.press/v80/baptista18a/baptista18a.pdf)
- [Think Global and Act Local: Bayesian Optimisation over High-Dimensional Categorical and Mixed Search Spaces](https://proceedings.mlr.press/v139/wan21b.html) — Xingchen Wan et al., ICML 2021, ungraded — owner-selected (2026-09-28): local optimisation and tailored kernels for categorical and mixed search spaces; a reference for representing team-building choices and defining useful search neighborhoods. · [PDF](https://proceedings.mlr.press/v139/wan21b/wan21b.pdf)
- [Using Common Random Numbers for Simulation-based Planning with Rollouts](https://rlj.cs.umass.edu/2026/papers/Paper52.pdf)
- 🔒 [Toward Reliable Uncertainty Quantification in Surrogate-Assisted Evolutionary Algorithms via Temporal Conformal Prediction](https://doi.org/10.1007/978-3-032-23604-3_24)
- [Efficient Exploration in Binary and Preferential Bayesian Optimization](https://arxiv.org/abs/2110.09361) — 🟢 arXiv preprint, no venue listed, MED
- [Scaling Gaussian Process Optimization by Evaluating a Few Unique Candidates Multiple Times](https://proceedings.mlr.press/v162/calandriello22a/calandriello22a.pdf) — 🟢 ICML 2022, LOW
- [Bayesian Optimization for Intrinsically Noisy Response Surfaces](https://arxiv.org/abs/2503.00327) — 🟢 arXiv preprint, no venue listed, LOW

### Optimization benchmarks and validation

- [Unify ML4TSP: Drawing Methodological Principles for TSP and Beyond from Streamlined Design Space of Learning and Search](https://proceedings.iclr.cc/paper_files/paper/2025/file/a3dc6e903082902d8e916bb9fccbfcbc-Paper-Conference.pdf)
- [Design-Bench: Benchmarks for Data-Driven Offline Model-Based Optimization](https://www.semanticscholar.org/paper/f23222ad51b9b1f4c1addcf848906db7698ccbff) — ICML 2022, ungraded — owner-selected focus (2026-09-13): test whether stronger search improves measured battle performance or exploits surrogate errors. Offline benchmark reference; Pokémon experiments would need fresh battle validation. · [Published paper](https://proceedings.mlr.press/v162/trabucco22a.html)
- [Exploring validation metrics for offline model-based optimisation with diffusion models](https://arxiv.org/abs/2211.10747) — 🟢 Transactions on Machine Learning Research 2024, MED
- [What Ails Generative Structure-based Drug Design: Expressivity is Too Little or Too Much?](https://arxiv.org/abs/2408.06050) — 🟢 AISTATS 2025, LOW

## Diffusion models and sampling

### Discrete diffusion foundations

- [Structured Denoising Diffusion Models in Discrete State-Spaces](https://www.semanticscholar.org/paper/91b32fc0a23f0af53229fceaae9cce43a0406d2e)
- [Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution](https://www.semanticscholar.org/paper/ce806f8d32f6fb1eaa821248a1bc4fa2cd949fbb)
- [Argmax Flows and Multinomial Diffusion: Learning Categorical Distributions](https://www.semanticscholar.org/paper/1913d3edcc00d0aba097a9df190dd16f6fdfbf0c)
- [Simple and Effective Masked Diffusion Language Models](https://www.semanticscholar.org/paper/f8d357d38bbcdd93889fe71762eb57842b2ab063)
- [BERT has a Mouth, and It Must Speak: BERT as a Markov Random Field Language Model](https://www.semanticscholar.org/paper/d79ac7a7bafdc9a782fb8c53285ca11c7f2e3f18)

### Sampling order, remasking and diversity

- [Remasking Discrete Diffusion Models with Inference-Time Scaling](https://www.semanticscholar.org/paper/19c3a5d9d32c57cd1482c8376f208cc2b2333334)
- [[Low-density sampling from diffusion models]]
- [A Tale of Two Temperatures: Simple, Efficient, and Diverse Sampling from Diffusion Language Models](https://arxiv.org/abs/2604.09921) — **MED** · SPIGM 2026 · Compare token temperature with randomness in the order fields are filled.
- [Re-evaluating Confidence Remasking in Masked Diffusion Language Models](https://arxiv.org/abs/2606.12232) — **MED** · SPIGM 2026 · Evaluate whether revisiting generated fields justifies its cost; compare tuned baselines and diversity.

### Structured generation and application examples

- [Prompt-to-Slate: Diffusion Models for Prompt-Conditioned Slate Generation](https://www.semanticscholar.org/paper/da1f9ea84ef798c82a876d37fdf95e4568eafcad)
- [SDEdit: Guided Image Synthesis and Editing with Stochastic Differential Equations](https://www.semanticscholar.org/paper/f671a09e3e5922e6d38cb77dda8d76d5ceac2a27)
- [Compositional Visual Generation with Composable Diffusion Models](https://www.semanticscholar.org/paper/3ff7153fd6bd47d08084c7f50f8fd70026c126e7)
- [TabDDPM: Modelling Tabular Data with Diffusion Models](https://www.semanticscholar.org/paper/25d3a4e048d0020ba9cffc6442ebd4e7bb548a55)
- [Diffusion-LM Improves Controllable Text Generation](https://www.semanticscholar.org/paper/1386b8a11929cf02da291c56aca353e33bbc22ed)
- [Protein Design with Guided Discrete Diffusion](https://arxiv.org/abs/2305.20009) — 🟢 NeurIPS 2023, HIGH

### Guidance methods

- [Diffusion Models as Plug-and-Play Priors](https://arxiv.org/abs/2206.09012) — NeurIPS 2022, ungraded — conditional reading for steering a diffusion prior with a compatible differentiable matchup surrogate; battle simulation supplies no direct gradient. · [[DIFUSCO references - relevance to counter-team search|Relevance notes]]
- [Gradient Guidance for Diffusion Models: An Optimization Perspective](https://www.semanticscholar.org/paper/5aabe3a210270711c72bddaa0290c4edd3db7720) — 🟢 NeurIPS 2024, MED
- [Robust Guided Diffusion for Offline Black-Box Optimization](https://www.semanticscholar.org/paper/45957487086436f6fcbdd88c6a6dd5f268d9310f)
- 🟢 [Prof. Mengdi Wang | Guiding Diffusion Models Towards Generative Optimization by INI Satellite Events](https://www.youtube.com/watch?v=8W9qHXN0weg) — talk, Isaac Newton Institute DMLW01 workshop (2024-07-18), MED
- [Derivative-Free Guidance in Continuous and Discrete Diffusion Models with Soft Value-Based Decoding](https://www.semanticscholar.org/paper/a673ef39237e227381fccf5b1d154d96de428f1d)
- [Practical and Asymptotically Exact Conditional Sampling in Diffusion Models](https://www.semanticscholar.org/paper/79531b47bb27cb18022891eb2ab1fcb41745fca6) — 🟢 NeurIPS 2023, MED
- [Simple Guidance Mechanisms for Discrete Diffusion Models](https://www.semanticscholar.org/paper/05f9997d61460fea9b586f98722c4be32f4a8b22)
- [Unlocking Guidance for Discrete State-Space Diffusion and Flow Models](https://www.semanticscholar.org/paper/c8bfa3abf6a8cbf0a5b3a093661510f34cfa0098)
- [Classifier-Free Diffusion Guidance](https://www.semanticscholar.org/paper/af9f365ed86614c800f082bd8eb14be76072ad16)
- [Plug-and-Play Guidance for Discrete Diffusion Models via Gradient-Informed Logit Correction](https://www.semanticscholar.org/paper/30aff605819e4a05c309040dc980ff82a767d828)
- [Diffusion Tree Sampling: Scalable inference-time alignment of diffusion models](https://www.semanticscholar.org/paper/aabd199f5094c726021f9a4175491549f6feb593) — 🟢 NeurIPS 2025, MED
- [UnMaskFork: Test-Time Scaling for Masked Diffusion via Deterministic Action Branching](https://www.semanticscholar.org/paper/eac84de6e4ce26fbbd7fb2013a923afceda4d797) — 🟢 ICML 2026, MED
- [Inference-Time Alignment in Diffusion Models with Reward-Guided Generation: Tutorial and Review](https://www.semanticscholar.org/paper/78f37226009495fc11f5b64c5462c9470d2726e8) — 🟢 arXiv preprint, no venue listed, MED
- [Guess & Guide: Gradient-Free Zero-Shot Diffusion Guidance](https://arxiv.org/abs/2603.07860v1) — 🟢 ReALM-GEN (ICLR 2026 workshop), LOW
- [Dynamic Search for Inference-Time Alignment in Diffusion Models](https://www.semanticscholar.org/paper/3546ed4907a006c8b4c8379b6059648c912f6a65) — 🟢 arXiv preprint, no venue listed, LOW
- [Inference-Time Alignment of Diffusion Models via Evolutionary Algorithms](https://www.semanticscholar.org/paper/281ea51be4268dbace01236879a56b575436474c) — 🟢 arXiv preprint, no venue listed, LOW
- [Diffusion Crossover: Defining Evolutionary Recombination in Diffusion Models via Noise Sequence Interpolation](https://www.semanticscholar.org/paper/4e598ea023dc54074a89c9b869effefa1b61e8b2) — 🟢 arXiv preprint, no venue listed, LOW
- [Inference-Time Scaling for Diffusion Models beyond Scaling Denoising Steps](https://www.semanticscholar.org/paper/9c8f548b85af8d3c861c9d97b6172c4ee44673d6) — 🟢 arXiv preprint, no venue listed, LOW

### Guidance analysis and correction

- [What Exactly Does Guidance Do in Masked Discrete Diffusion Models](https://api.semanticscholar.org/arXiv:2506.10971) — 🟢 arXiv preprint, no venue listed, HIGH
- [What does guidance do? A fine-grained analysis in a simple setting](https://proceedings.neurips.cc/paper_files/paper/2024/file/9a3942c235daa9c5f62a8598ae81a946-Paper-Conference.pdf) — 🟢 NeurIPS 2024, MED
- [Classifier-Free Guidance is a Predictor-Corrector](https://arxiv.org/abs/2408.09000) — 🟢 M3L Workshop at NeurIPS 2024, MED
- [Debiasing Guidance for Discrete Diffusion with Sequential Monte Carlo](https://arxiv.org/abs/2502.06079) — 🟢 Frontiers in Probabilistic Inference workshop at ICLR 2025, MED
- [Generative Uncertainty in Diffusion Models](https://www.semanticscholar.org/paper/54b4c9a67d0c55b5d430d45ead5d5c7246283232) — 🟢 UAI 2025, MED

## Retraining, diversity and model collapse

### Memorization and data replication

- [On Memorization in Diffusion Models](https://www.semanticscholar.org/paper/122a7e217fe70d5a1a44a6e2b67e859d1fc8e28d)
- [Diffusion Art or Digital Forgery? Investigating Data Replication in Diffusion Models](https://arxiv.org/abs/2212.03860)
- [Detecting, Explaining, and Mitigating Memorization in Diffusion Models](https://api.semanticscholar.org/arXiv:2407.21720) — 🟢 ICLR 2024, HIGH

### Recursive training and collapse dynamics

- [Self-Consuming Generative Models Go MAD](https://api.semanticscholar.org/arXiv:2307.01850) — 🟢 ICLR 2024, HIGH
- [On the Stability of Iterative Retraining of Generative Models on their own Data](https://api.semanticscholar.org/arXiv:2310.00429) — 🟢 ICLR 2024 spotlight, HIGH
- [AI models collapse when trained on recursively generated data](https://www.nature.com/articles/s41586-024-07566-y) — 🟢 Nature 2024, MED
- [Sharp Bounds for Genetic Drift in Estimation of Distribution Algorithms](https://api.semanticscholar.org/arXiv:1910.14389) — 🟢 IEEE Transactions on Evolutionary Computation 2020, HIGH
- [A Closer Look at Model Collapse: From a Generalization-to-Memorization Perspective](https://api.semanticscholar.org/arXiv:2509.16499) — 🟢 NeurIPS 2025 spotlight, HIGH
- [Model Collapse in the Self-Consuming Chain of Diffusion Finetuning: A Novel Perspective from Quantitative Trait Modeling](https://api.semanticscholar.org/arXiv:2407.17493) — 🟢 ICLR 2025, HIGH

### Verification, correction and data mixing

- [Beyond Model Collapse: Scaling Up with Synthesized Data Requires Verification](https://api.semanticscholar.org/arXiv:2406.07515) — 🟢 ICLR 2025, MED
- [Self-Correcting Self-Consuming Loops for Generative Model Training](https://api.semanticscholar.org/arXiv:2402.07087) — 🟢 ICML 2024, MED
- [Is Model Collapse Inevitable? Breaking the Curse of Recursion by Accumulating Real and Synthetic Data](https://api.semanticscholar.org/arXiv:2404.01413) — 🟢 COLM 2024, MED

### Reward-weighted retraining

- [Sample-Efficient Optimization in the Latent Space of Deep Generative Models via Weighted Retraining](https://arxiv.org/abs/2006.09191) — 🟢 NeurIPS 2020, HIGH

## Surrogates and active learning

### Learning objectives for team selection

- [Decision-Focused Learning: Through the Lens of Learning to Rank](https://proceedings.mlr.press/v162/mandi22a.html) — ICML 2022, ungraded — owner-selected focus (2026-09-13): compare surrogate objectives by the strength of discovered teams, rather than prediction error alone. Connection to Pokémon is a proposed experiment, not a result of this paper.

### Structured surrogate models

- [Constrained Discrete Black-Box Optimization using Mixed-Integer Programming](https://proceedings.mlr.press/v162/papalexopoulos22a.html) — ICML 2022, ungraded — NN+MILP combines a learned score predictor with constrained optimization; candidate approach for a team-recombination teacher. Assumes noiseless evaluations, so battle noise needs additional handling; optimality applies to the learned acquisition model, not actual team strength. Read §§3.1–3.3.
- [Distributional Energy-Based Models for Uncertainty-Aware Structured LLM Reasoning](https://arxiv.org/abs/2605.18871) — 🟢 EIML (ICML 2026 workshop), MED — targeted reading: §4.2 and Appendix F.8; test whether a team scorer adds value beyond choosing the strongest proposal source. VGC transfer is an audit design, not a validated scoring method.

### Active learning and uncertainty sampling

- [Active Learning Literature Survey](https://burrsettles.com/pub/settles.activelearning.pdf) — 🟢 Computer Sciences Technical Report 1648, University of Wisconsin-Madison, 2009, MED
- [On the Relationship between Data Efficiency and Error for Uncertainty Sampling](https://proceedings.mlr.press/v80/mussmann18a.html) — 🟢 ICML 2018, HIGH

### Membership queries and concept learning

- [Active Query Synthesis for Preference Learning](https://arxiv.org/abs/2605.26072)
- [Textual Membership Queries](https://www.ijcai.org/proceedings/2020/0369.pdf) — 🟢 IJCAI 2020, HIGH
- [Queries and Concept Learning](http://machinelearning202.pbworks.com/f/AngluinQueriesConceptLearningfulltext.pdf) — 🟢 Machine Learning 2:319-342, 1988, MED

## Legality and constraint programming

### Constrained diffusion and decoding

- [Constrained Discrete Diffusion](https://arxiv.org/abs/2503.09790) — 🟢 NeurIPS 2025, MED
- [Constrained Code Generation with Discrete Diffusion](https://arxiv.org/abs/2605.16829) — 🟢 arXiv preprint, no venue listed, MED
- [Constrained Decoding of Diffusion LLMs with Context-Free Grammars](https://arxiv.org/abs/2508.10111) — 🟢 arXiv preprint, no venue listed, MED

### Streamliners

- [StreamLLM: Enhancing Constraint Programming with Large Language Model-Generated Streamliners](https://jair.org/index.php/jair/article/view/18965)
- [Streamlined Constraint Reasoning](https://www.cs.cornell.edu/gomes/papers/gomes-sellman-cp04.pdf)

### Learning constraints from examples and queries

- [A Model Seeker: Extracting Global Constraint Models from Positive Examples](https://www.semanticscholar.org/paper/d7a118530bc40fe3c2cf3d81fde56b1fc3cb1f30)
- [Learning Constraints through Partial Queries](https://www.lirmm.fr/~bessiere/Site/stock/aij23.pdf)

### Symmetry breaking

- [Symmetry in Constraint Programming](https://sites.cs.st-andrews.ac.uk/people/ipg1/papers/GentPetriePugetFinalDraft.pdf) — 🟢 author draft (🔒 Elsevier), Handbook of Constraint Programming ch. 10, 2006, MED
- [Breaking symmetries in all different problems](https://www.ijcai.org/Proceedings/05/Papers/1244.pdf) — 🟢 IJCAI 2005, MED
- [Symmetry-Breaking Predicates for Search Problems](https://ix.cs.uoregon.edu/~luks/symmetrybreaking.pdf) — 🟢 KR 1996, LOW
- [Breaking Row and Column Symmetries in Matrix Models](https://pierre-flener.github.io/research/pub/CP02.pdf) — 🟢 author preprint, CP 2002, LOW

### Dominance breaking

- [Dominance breaking constraints](https://people.eng.unimelb.edu.au/pstuckey/papers/domjournal.pdf) — 🟢 author manuscript (🔒 Springer), Constraints 20(2), 2015, MED
- [Automatic Generation of Dominance Breaking Nogoods for a Class of Constraint Optimization Problems](http://www.cse.cuhk.edu.hk/~jlee/publ/23/aij23AutoDomBreak.pdf) — 🟢 author preprint, Artificial Intelligence 323, 2023, LOW

## Automated heuristic design

### Heuristic generation and strategy evolution

- [G-LNS: Generative Large Neighborhood Search for LLM-Based Automatic Heuristic Design](https://www.semanticscholar.org/paper/deefec531e1ccc8798e29f9c38125fdbc34a9a74) — 🟢 arXiv 2602.08253 (2026-02), no venue listed, MED
- [Beyond Static Evaluation: Co-Evolutionary Mechanisms for LLM-Driven Strategy Evolution in Adversarial Games](https://arxiv.org/abs/2606.10389) — 🟢 AAMAS 2026 MCTF Competition, MED
- [RuleSmith: Multi-Agent LLMs for Automated Game Balancing](https://arxiv.org/abs/2602.06232) — 🟢 arXiv preprint, no venue listed, LOW

### Hyper-heuristic taxonomy and selection

- [A Classification of Hyper-Heuristic Approaches: Revisited](https://people.cs.nott.ac.uk/pszeo/docs/publications/HHClassChapterRevisited.pdf) — 🟢 author PDF (🔒 Springer), Handbook of Metaheuristics 3rd ed. ch. 14, 2019, MED
- [Recent advances in selection hyper-heuristics](https://www.sciencedirect.com/science/article/pii/S0377221719306526) — 🟢 CC-BY open access, European Journal of Operational Research 285(2), 2020, LOW

## Battle policies and self-play

### Self-play and game-theoretic foundations

- [[A Survey on Self-play Methods in Reinforcement Learning]]
- [A Unified Game-Theoretic Approach to Multiagent Reinforcement Learning](https://arxiv.org/abs/1711.00832)
- [Game-Theoretic Multiagent Reinforcement Learning](https://www.semanticscholar.org/paper/c3662e9176a7ad90020bdd025c179c5925d0b5b0)
- [Planning in the Presence of Cost Functions Controlled by an Adversary](https://www.aaai.org/Papers/ICML/2003/ICML03-071.pdf)

### Regret minimization

- [An Introduction to Counterfactual Regret Minimization](http://modelai.gettysburg.edu/2013/cfr/cfr.pdf)
- [Efficient Monte Carlo Counterfactual Regret Minimization in Games with Many Player Actions](https://papers.nips.cc/paper/2012/file/3df1d4b96d8976ff5986393e8767f5b2-Paper.pdf)

### Policy learning and game environments

- [Overcoming Valid Action Suppression in Unmasked Policy Gradient Algorithms](https://rlj.cs.umass.edu/2026/papers/Paper1.pdf)
- [FootsiesGym: A Fighting Game Benchmark for Two-Player Zero-Sum Imperfect-Information Games](https://arxiv.org/abs/2607.06514)
- [[Grandmaster level in StarCraft II using multi-agent reinforcement learning]]
- [Superhuman AI for Generals.io Using Self-Play Reinforcement Learning](https://arxiv.org/abs/2606.23348)
- [Emergent Complexity via Multi-Agent Competition](https://arxiv.org/abs/1710.03748)
- [Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model](https://www.semanticscholar.org/paper/c39fb7a46335c23f7529dd6f9f980462fd38653a)

### Ranking and evaluation in non-transitive games

- [[Alpha-Rank Multi-Agent Evaluation by Evolution]]
- [Real World Games Look Like Spinning Tops](https://arxiv.org/abs/2004.09468)
- [Re-evaluating Evaluation](https://arxiv.org/abs/1806.02643)

### Competitive coevolution

- [Coevolutionary Principles](https://www.cs.tufts.edu/comp/150GA/handouts/nchb-main.pdf)
- [New Methods for Competitive Coevolution](https://cseweb.ucsd.edu/~crosin/newmethods.ps)
- [A comparison of evaluation methods in coevolution](https://arxiv.org/abs/1905.08723)
- 🔒 [Tripping Over the Past: Measuring Deceptive Progress in Competitive Coevolutionary Algorithms Employing Hall of Fame](https://doi.org/10.1007/978-3-032-23607-4_32)

## Training dynamics and engineering

### Plasticity loss and recovery

- [Plasticity Loss in Deep Reinforcement Learning: A Survey](https://api.semanticscholar.org/arXiv:2411.04832)
- [A Study of Plasticity Loss in On-Policy Deep Reinforcement Learning](https://api.semanticscholar.org/arXiv:2405.19153)
- [The Primacy Bias in Deep Reinforcement Learning](https://api.semanticscholar.org/arXiv:2205.07802)
- [The Dormant Neuron Phenomenon in Deep Reinforcement Learning](https://api.semanticscholar.org/arXiv:2302.12902)
- [Loss of Plasticity in Continual Deep Reinforcement Learning](https://api.semanticscholar.org/arXiv:2303.07507)
- [Disentangling the Causes of Plasticity Loss in Neural Networks](https://api.semanticscholar.org/arXiv:2402.18762)
- [Deep Reinforcement Learning with Plasticity Injection](https://api.semanticscholar.org/arXiv:2305.15555)
- [Maintaining Plasticity in Continual Learning via Regenerative Regularization](https://api.semanticscholar.org/arXiv:2308.11958)
- [On Warm-Starting Neural Network Training](https://api.semanticscholar.org/arXiv:1910.08475)

### Batch size and parallel training

- [Batch size-invariance for policy optimization](https://api.semanticscholar.org/arXiv:2110.00641)
- [An Empirical Model of Large-Batch Training](https://api.semanticscholar.org/arXiv:1812.06162)
- [SAPG: Split and Aggregate Policy Gradients](https://api.semanticscholar.org/arXiv:2407.20230)
- [Preventing Learning Stagnation in PPO by Scaling to 1 Million Parallel Environments](https://rlj.cs.umass.edu/2026/papers/Paper19.pdf)

### Exploration and sample efficiency

- [Improving Sample Efficiency in Multi-Agent Reinforcement Learning for Simulated Football Games via Exploration](https://doi.org/10.1145/3815598.3815627)
- [[Human-Like Goalkeeping in a Realistic Football Simulation a Sample-Efficient Reinforcement Learning Approach]]

### Adaptive and multi-task agents

- [AMAGO: Scalable In-Context Reinforcement Learning for Adaptive Agents](https://arxiv.org/abs/2310.09971)
- [AMAGO-2: Breaking the Multi-Task Barrier in Meta-Reinforcement Learning with Transformers](https://arxiv.org/abs/2411.11188)
- [Multi-Task Reinforcement Learning Enables Parameter Scaling](https://api.semanticscholar.org/arXiv:2503.05126)

### Offline data, imitation and reward assignment

- [Accelerating Online Reinforcement Learning with Offline Datasets](https://arxiv.org/abs/2006.09359)
- [Critic Regularized Regression](https://arxiv.org/abs/2006.15134)
- [Discovering High Quality Chess Puzzles with Offline Reinforcement Learning](https://rlj.cs.umass.edu/2026/papers/Paper147.pdf)
- [Minimal Ingredients for Reward Assignment from Expert Demonstrations](https://rlj.cs.umass.edu/2026/papers/Paper86.pdf)
- [Vanquish Your Past: Shifted Imitation Learning in Hades](https://doi.org/10.1145/3815598.3815650)

### Simulation infrastructure and benchmarks

- [Assistax: A Multi-Agent Hardware-Accelerated RL Benchmark for Assistive Robotics](https://rlj.cs.umass.edu/2026/papers/Paper110.pdf)
- [Ludax: A GPU-Accelerated Description Language for Board Games](https://rlj.cs.umass.edu/2026/papers/Paper46.pdf)

### Statistical evaluation and non-stationarity

- [Confidence Intervals for the Interquartile Mean](https://rlj.cs.umass.edu/2026/papers/Paper28.pdf)
- [Space-sampled Value Decay: Forgetting Mechanisms for Non-stationary Deep Reinforcement Learning](https://arxiv.org/abs/2606.11797) — 🟢 EIML (ICML 2026 workshop), MED — stale value estimates under changing opponents; DQN/SAC evidence, VGC state sampling unresolved. Read §§3–6 after HIGH papers.

## Deferred readings

These are outside the active queue.

### Reward integration

Revisit when planning reward objectives, fine-tuning or preference alignment; HIGH grades here remain deferred.

- [Fine-Tuning Discrete Diffusion Models via Reward Optimization with Applications to DNA and Protein Design](https://www.semanticscholar.org/paper/d1461167c9fef8fe3ba129c514acfd14bbe7a51e)
- [Self-Consuming Generative Models with Curated Data Provably Optimize Human Preferences](https://api.semanticscholar.org/arXiv:2407.09499) — 🟢 NeurIPS 2024 spotlight, HIGH
- [Reward-Directed Conditional Diffusion: Provable Distribution Estimation and Reward Improvement](https://arxiv.org/abs/2307.07055) — 🟢 NeurIPS 2023, MED
- [The tractability landscape of diffusion alignment: regularization, rewards, and computational primitives](https://arxiv.org/abs/2605.11361) — 🟢 arXiv preprint, no venue listed, MED
- [CADO: From Imitation to Cost Minimization for Heatmap-based Solvers in Combinatorial Optimization](https://arxiv.org/abs/2602.08210) — 🟢 TMLR 2026, MED
- [Training Diffusion Models with Reinforcement Learning](https://www.semanticscholar.org/paper/d8c78221e4366d6a72a6b3e41e35b706cc45c01d) — 🟢 ICLR 2024, MED
- [Scalable Discrete Diffusion Samplers: Combinatorial Optimization and Statistical Physics](https://www.semanticscholar.org/paper/ec3d27dbfdd9431720c8a848b0ac33145dac58be) — 🟢 ICLR 2025, MED
- [DPOK: Reinforcement Learning for Fine-tuning Text-to-Image Diffusion Models](https://www.semanticscholar.org/paper/a553bf27d801d09f667fe121c0ba9632257f364b) — 🟢 NeurIPS 2023, LOW
- [Iterated Denoising Energy Matching for Sampling from Boltzmann Densities](https://www.semanticscholar.org/paper/9f0f134bc53f9aea130f1a561acb5fa6b08ae4ce) — 🟢 ICML 2024, LOW
- [Improved sampling via learned diffusions](https://www.semanticscholar.org/paper/d174e9d35a9d6d899acbc661e05a937a659ffc42) — 🟢 ICLR 2024, LOW

### Other readings kept for later

- [Identifying and Clustering Counter Relationships of Team Compositions in PvP Games for Efficient Balance Analysis](https://arxiv.org/abs/2408.17180) — TMLR 2024. Counter-matrix between team compositions; take the measure, not the balance-analysis pipeline.
- [Team Composition in PES2018 Using Submodular Function Optimization](https://doi.org/10.1109/ACCESS.2019.2919447) — IEEE Access 2019. Team selection as a submodular coverage function with a greedy solver; the closest published cousin of the weighted maximum coverage objective.

### Parked directions

Revisit if the project direction changes.

- [A QUBO Framework for Team Formation](https://arxiv.org/abs/2503.23209)
- [A Tutorial on Formulating and Using QUBO Models](https://api.semanticscholar.org/arXiv:1811.11538)
- [Designing Metamaterials with Quantum Annealing and Factorization Machines](https://arxiv.org/abs/1902.06573) — kept only for the pairwise-interaction surrogate idea, not annealing
- [A Tutorial on Bayesian Optimization](https://api.semanticscholar.org/arXiv:1807.02811)
- [Evaluating the Stability of the Smogon Tier List for Competitive Pokémon Battling](https://ceur-ws.org/Vol-3926/paper2.pdf) — 🟢 EXAG 2024, MED
- 🟢 [Calculating Pokémon Teams v1.0 by John (JaybeeVGC)](https://docs.google.com/document/d/1IIXfm49kbrij4FouD3x9RgYdaM2BetE1/edit) — PuLP linear programming over type coverage; retired framing, LOW
- [Pokémon team optimization by Nicolas Chagnet](https://nchagnet.eu/blog/pokemon-team-optimization/) — 🟢 blog + PuLP repo, 2025, LOW
- [Which Heroes to Pick? Learning to Draft in MOBA Games with Neural Networks and Tree Search](https://arxiv.org/abs/2012.10171) — tree search is the right tool class, but the sequential draft model does not fit simultaneous bring-six selection.
- [Introducing Tales of Tribute AI Competition](https://www.semanticscholar.org/paper/2cd0f87d70a0b6afd79324f959c68f04ef6c0945) — about playing agents for a card game, not team construction.

## SPIGM papers not accepted

The other 16 initial additions are retained for reference. Earlier assessments are in [[SPIGM 2026 — full-paper reassessment]] and [[Reading List — notes and decision history]].

- [Learn from Your Mistakes: Self-Correcting Masked Diffusion Models](https://arxiv.org/abs/2602.11590) — **DEFER** · SPIGM 2026 · Learned correction is legitimate, but no established benefit beyond current repair or for the active data comparison.
- [Breaking the Factorization Barrier in Diffusion Language Models](https://arxiv.org/abs/2603.00045) — **DEFER** · SPIGM 2026 · A joint head chiefly addresses simultaneous field generation, which the current sampler does not do.
- [Internal Data Repetition Destroys Language Models](https://arxiv.org/abs/2606.24998) — **DROP** · SPIGM 2026 · Exact duplicate removal is already available. The paper studies concentrated exact repetition; extending it to distinct mutations did not justify this reading.
- [Finetuning Generative Models to Match Feature Distributions](https://arxiv.org/abs/2606.19496) — **NOT ACCEPTED** · SPIGM 2026 · Outside the two-paper owner-confirmed shortlist.
- [The Confidence Shortcut: A Reasoning Failure Mode of Masked Diffusion Models](https://arxiv.org/abs/2605.29123) — **DEFER** · SPIGM 2026 · Relevant if confidence-aligned training becomes active; the current model does not use it.
- [Uniform Diffusion Models Revisited: Leave-One-Out Denoiser and Absorbing State Reformulation](https://arxiv.org/abs/2605.22765) — **DEFER** · SPIGM 2026 · A uniform-corruption objective issue does not establish a defect in the current mask-only model.
- [Tensor-Train Joint Modeling for Few-Step Discrete Diffusion](https://arxiv.org/abs/2607.03788) — **DEFER** · SPIGM 2026 · Low-rank joint sampling is a future parallel-decoding option, with ordering and rank assumptions.
- [Latent-Augmented Discrete Diffusion Models](https://arxiv.org/abs/2510.18114) — **DEFER** · SPIGM 2026 · A latent channel adds substantial modeling requirements before a demonstrated current need.
- [Learned Relay Representations for Forward-Thinking Discrete Diffusion Models](https://arxiv.org/abs/2605.22967) — **DEFER** · SPIGM 2026 · Compatible with sequential generation, but a persistent-state benefit has not been established here.
- [Recursive Scaling in Masked Diffusion Models](https://arxiv.org/abs/2606.18022) — **DEFER** · SPIGM 2026 · A future capacity-versus-compute study, not evidence about the current data or CEM questions.
- [Understanding and Accelerating the Training of Masked Diffusion Language Models](https://arxiv.org/abs/2605.13026) — **DROP** · SPIGM 2026 · The owner confirms diffusion training is not a problem. Mask-context diagnostics do not supply a sufficiently relevant current reading task.
- [Time-Annealed Perturbation Sampling: Diverse Generation for Diffusion Language Models](https://arxiv.org/abs/2601.22629) — **DROP** · SPIGM 2026 · Weak categorical-conditioning transfer, plus filtering and reporting issues undermine the proposed payoff.
- [Hacking Generative Perplexity: Why Unconditional Text Evaluation Needs Distributional Metrics](https://arxiv.org/abs/2606.08417) — **DROP** · SPIGM 2026 · The project does not use generative perplexity and already plans full-team/composition diversity measurements. The earlier team-template stress-test connection was an analogy, insufficient to justify reading this paper.
- [DUEL: Exact Likelihood for Masked Diffusion via Deterministic Unmasking](https://arxiv.org/abs/2603.01367) — **DROP** · SPIGM 2026 · Exact likelihood for a specified reveal schedule is not a current project requirement.
- [TUBE: Tangent Upper Bound on Evidence for Discrete Diffusion Language Models](https://arxiv.org/abs/2605.24292) — **DROP** · SPIGM 2026 · Sophisticated likelihood bounds do not answer the current team-quality or diversity decisions.
- [Contrastive Distribution Matching for Amortized Sequential Monte Carlo in Discrete Diffusion](https://arxiv.org/abs/2605.23346) — **DEFER** · SPIGM 2026 · Keep as a future reward-integration reference, requiring a suitable reward and trajectory model.

## Rejected readings

Retained with the recorded reasons so they are not suggested again.

- [Variational Search Distributions](https://arxiv.org/abs/2409.06142) — 🟢 ICLR 2025, was HIGH. Rejected at the owner's request, 2026-09-10: 9 citations, and the papers citing it are themselves low-cited.
- [Closed-Loop Generative Selection: Convergence, Memory, and Noisy Oracles](https://arxiv.org/abs/2607.22211) — 🟢 arXiv preprint, no venue listed, was MED. Owner's verdict, 2026-09-10: does not belong in the reading list.
- [Convergence properties of the cross-entropy method for discrete optimization](https://people.smp.uq.edu.au/DirkKroese/ps/CEconv.pdf) — 🟢 Operations Research Letters 2007, was HIGH. Rejected at the owner's request, 2026-09-10: somewhat relevant, but he dislikes the writing.
- [Efficient Active Search for Combinatorial Optimization Problems](https://arxiv.org/abs/2106.05126) — ICLR 2022, I just learn what was active search method which is about training the whole model using only insteance to increase probability more likely to generate the examples but the thing only one example not what I am looking for — Rejected 2026-09-10: per-instance fine-tuning is already the CEM loop; saves compute, not battles.
- [Batch Bayesian Optimization for Replicable Experimental Design](https://arxiv.org/abs/2311.01195) — 🟢 NeurIPS 2023, previously MED — Owner's note: not useful for this project.
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
  - **Superseded 2026-09-07: accepted MED for selective reading**, with an active entry above. The full paper evaluates expensive rewards; applicability to categorical team generation and noisy battle ranking remains unvalidated. See [[Trust-Region Noise Search for Black-Box Alignment of Diffusion and Flow Models]] for the reassessment.
- 🟢 [Discrete Diffusion Inference-Time Control with Nested Sequential Monte Carlo](https://arxiv.org/abs/2608.20123) — ReALM-GEN (ICLR 2026 workshop), was LOW. Feynman-Kac particle steering needs a reward on partial sequences; a half-built team cannot be battled.
- 🔒 DDNO: Discrete Diffusion Noise Optimization (Eyring et al.) — ReALM-GEN (ICLR 2026 workshop), no open copy, ungraded. Discrete twin of Trust-Region Noise Search; noise-space search needs many reward calls and battles are too costly.
- 🟢 [Diffusion Active Learning: Towards Data-Driven Experimental Design in Computed Tomography](https://arxiv.org/abs/2504.03491) — 2025 preprint, no venue, was MED. Inverse problem: acquisition picks the next measurement of a fixed object, informativeness-denominated.
- 🟢 [Active Diffusion-Based Inference for Ill-Posed Inverse Problems under Incomplete Priors](https://arxiv.org/abs/2608.27080) — IJCAI-ECAI 2026 (venue was missing), was LOW. Inverse problem: recovers hidden true parameters, acquisition driven by posterior uncertainty.
- 🟢 [Offline Model-Based Optimization: Comprehensive Review](https://arxiv.org/abs/2503.17286) — TMLR 2026 (Survey Certification), was HIGH. Offline MBO assumes no oracle access; Showdown gives one on demand. Owner's call 2026-09-02: no surveys.
- 🟢 [When does return-conditioned supervised learning work for offline reinforcement learning?](https://arxiv.org/abs/2206.01079) — NeurIPS 2022, was HIGH. Owner's verdict 2026-09-02: an offline-RL policy result, no design generator; not useful.
- [Real-time eSports Match Result Prediction](https://arxiv.org/abs/1701.03162) — Dota 2 win prediction from in-match features; predicts outcomes, does not construct teams.
- [Win Prediction in Esports: Mixed-Rank Match Prediction in Multi-player Online Battle Arena Games](https://arxiv.org/abs/1711.06498) — same win-prediction family.
- [League of Legends: Real-Time Result Prediction](https://arxiv.org/abs/2309.02449) — in-match LSTM prediction; no team construction.
- [Using Machine Learning to Predict Game Outcomes Based on Player-Champion Experience in League of Legends](https://arxiv.org/abs/2108.02799) — predicts outcomes from player skill, explicitly "regardless of team composition".
- [MobaQA: MOBA Games Prediction Based on Large Language Model Fine-Tuning](https://www.semanticscholar.org/paper/MobaQA%3A-MOBA-Games-Prediction-Based-on-Large-Nie-Wang/) — LLM fine-tuning to predict real-time win rate and match outcome; prediction, not search (IEEE Transactions on Games 2026).

## Sources and reading notes

- [[Reading List — notes and decision history]] — personal notes, original properties, prior decisions and the complete pre-reorganization list.
- [[SPIGM 2026 — paper-by-paper relevance screening]] — 2 accepted MED readings; remaining screening decisions are historical.
- [[EIML 2026 — paper-by-paper relevance screening]] — 2 MED additions; see the screening note for the assessment.
- [[_Workshops index]]
- [[VGC AI Competition (IEEE CoG)]]
- [[PokeAgent Challenge (NeurIPS 2025)]]
- [[Finding the Frame — papers relevant to the counter-team search]]
- [[Reading path — neural CO for team selection]]
- [[Teaching/RESOURCES|Teaching resources]] — analogue-sweep grades and verdicts.
- [[Citation]]
