# LLMs as black-box optimizers — evidence and mechanisms

Research date: 2026-09-06. Focus: generating structured candidates from stored evaluations, with Pokémon counter-team search as the intended application.

## What the LLM papers actually establish

### OPRO — Large Language Models as Optimizers

Yang et al., ICLR 2024. [Paper, full text](https://arxiv.org/html/2309.03409v3). [Official code](https://github.com/google-deepmind/opro).

OPRO repeatedly supplies previous solutions and numerical scores to an LLM, evaluates its new proposals externally, and updates the history. In its two-variable linear-regression experiment, it supplies the best 20 historical pairs, sorted by objective value, and generates up to eight proposals per iteration. Its demonstrations include discrete traveling-salesman tours, although prompt optimization is the main application.

The authors report sensitivity to low-quality history and use multiple proposals for stability. Sampling temperature controls exploration. Their TSP results deteriorate with problem size: at 50 nodes, GPT-4 has an 11.0% mean optimality gap versus 9.8% for farthest insertion. The authors explicitly do not position OPRO as outperforming specialized mathematical solvers. They identify context limits and difficult objective landscapes as limitations. See §§2.3, 3.1–3.2 and Appendix A.

**Project interpretation:** this is a concrete baseline for a propose–battle–store loop. Its history selection is programmed; how the LLM turns that history into a proposal remains learned and prompt-dependent. These experiments do not establish Pokémon search performance.

### LLAMBO — Large Language Models to Enhance Bayesian Optimization

Liu et al., ICLR 2024. [Paper, full text](https://arxiv.org/html/2402.03921v2). [Official code](https://github.com/tennisonliu/LLAMBO).

LLAMBO uses problem descriptions and serialized configuration–score observations for in-context candidate generation and surrogate prediction. It conditions generation on a numerical target score, then uses an acquisition function to select an actual evaluation. This is a stronger control mechanism than asking vaguely for improvement, but generation still depends on prompting.

Crucially, §5.1 reports sensitivity to example ordering. Shuffling examples across repeated predictions improves robustness and calibration. Appendix C.2 reports candidate acceptance of 91.60%±0.45% with standard instructions versus 69.26%±0.79% when non-formatting instructions are removed. Prompt design therefore has measurable operational consequences.

The paper reports strong hyperparameter-tuning results, especially with few observations. Its end-to-end Bayesmark experiments use five shared initial points and 25 trials per run. §8 identifies higher inference cost and relatively low-dimensional tasks as limitations; extending to complex, high-dimensional spaces remains future work.

**Project interpretation:** this directly motivates testing order sensitivity and validity rates, but does not prove attention cannot use team history effectively. Nor does it establish that target-score conditioning guarantees target performance.

### LMX — Language Model Crossover: Variation through Few-Shot Prompting

Meyerson et al., ACM Transactions on Evolutionary Learning and Optimization, 2024; first preprint 2023. [Paper](https://arxiv.org/abs/2302.12170). [Publisher](https://doi.org/10.1145/3694791). [Author publication page](https://www.kmjn.org/publications/LanguageModelCrossover_TELO24-abstract.html). [Official code](https://github.com/jal278/lmx).

LMX concatenates parent solutions represented as text, asks a pretrained language model to continue, and parses generated offspring. It serves as a variation operator within evolutionary search. Demonstrations span bit strings, equations, sentences, image prompts, and Python code.

**Project interpretation:** this is especially relevant when we want the search algorithm to select parents explicitly and let the LLM propose offspring. It avoids requiring every historical team to appear in one prompt. It does not eliminate learned generation behavior or establish constraint guarantees. A Pokémon adaptation would need an external legality checker and measured battle fitness.

Access note: the primary publisher search extract and author abstract/code were readable; full-text fetches of the publisher and author PDF failed. Treat detailed LMX implementation choices beyond the mechanism above as unverified here.

## What to test in this project

The following are proposed experiments, not findings from these papers:

- Keep every candidate and battle result in a database. Feed a bounded, explicitly selected subset to the generator.
- Compare direct OPRO-style proposal, LMX-style parent variation, the project's diffusion approach, copy-and-mutate, and random legal generation.
- Hold opponents, team representation, legality enforcement, battle policy, evaluation seeds, and initial data constant where possible.
- Count both simulator evaluations and total computational cost, including model training, inference, rejected proposals, and retries.
- Measure best independently re-evaluated win rate, progress per battle budget, validity, uniqueness, and diversity among competitive teams.
- Test the concern about history directly: reorder the same examples, shuffle score labels as a negative control, omit feedback, vary history size, and compare top-only versus mixed/diverse history. Use repeated runs.
- If testing preservation of a specific Pokémon or fixed slots, enforce those restrictions outside either generator and report raw violations before repair.

Attention is a mechanism, not itself an empirical verdict. The useful question is whether the generator changes its proposals appropriately when the relevant battle evidence changes, under matched evaluation budgets.

## Diffusion comparisons

### dLLM — Diffusion Large Language Models for Black-Box Optimization

Yuan et al., January 2026 preprint. [Full text, version inspected](https://arxiv.org/html/2601.14446v1).

A frozen LLaDA-8B model receives task text and design–score examples. A Gaussian process fits the offline observations; expected improvement guides Monte Carlo tree search over partially masked designs. Thus, prompting and explicit search coexist.

The experiments use 10 offline examples and evaluate 128 final candidates on four Design-Bench tasks. They compare against an OPRO-style LLaMA3-8B baseline and DDOM. Removing tree search reduces TF Bind 8's best normalized score from 0.876 to 0.798 (Tables 1–2). The paper inconsistently labels OPRO as ORPO in its table/discussion.

**Interpretation:** this is a direct method comparison, but model pretraining and search differ. It does not isolate attention, architecture, or Pokémon performance. The setting is offline, unlike repeated propose–battle–update rounds.

### DiBO — Training Diffusion Language Models for Black-Box Optimization

**2026-09-16 update:** The historical paragraph below describes v1. The [May 29 v3, Table 2](https://arxiv.org/html/2603.17919v3#S4.T2) adds a matched post-trained autoregressive baseline and revises results; see [[LLM versus diffusion for Bayesian search - evidence and verdict]]. Use that newer comparison for claims about which model family performs better.

Sun et al., March 2026. [Full text, v1 inspected](https://arxiv.org/html/2603.17919v1). [Official released code](https://github.com/zpointS/DiBO).

DiBO adapts LLaDA using explicit design/label delimiters, supervised fine-tuning, and reward-based training. Version 1 uses 500 offline observations, seven examples per prompt, and evaluates 128 candidates. Table 1 compares with OPRO and DDOM, reporting best-of-batch normalized scores averaged over eight seeds. For TF Bind 8: OPRO 0.758, DDOM 0.739, DiBO 0.965.

**Interpretation:** this compares whole pipelines; trained DiBO versus prompted OPRO does not isolate the generation architecture. It remains an offline benchmark. Review depth here is arXiv v1; the later conference PDF was not successfully retrieved, so these figures and qualifications refer specifically to v1.

### LaMBO-2 — Protein Design with Guided Discrete Diffusion

Gruver et al., NeurIPS 2023. [Official proceedings abstract](https://proceedings.neurips.cc/paper_files/paper/2023/hash/29591f355702c3f4436991335784b503-Abstract.html). [Author code](https://github.com/ngruver/NOS).

NOS guides discrete denoising through gradients in the network's continuous hidden states. LaMBO-2 incorporates this into Bayesian optimization with multiple objectives and edit constraints. This supplies a concrete precedent for steering a structured generator with a learned fitness signal. It does not supply gradients through a black-box experiment, and the cited evidence does not compare it against a general-purpose prompted LLM for Pokémon teams. Review depth for this entry: official abstract and author repository.

## Reading decision

Read OPRO to understand the history-driven proposal loop, LLAMBO for numerical conditioning and order-sensitivity evidence, then dLLM for an actual comparison incorporating explicit search. DiBO examines training as an additional control mechanism; LaMBO-2 connects to the project's original gradient-guidance motivation.

For this project, distinguish three choices: autoregressive versus diffusion generation, frozen versus adapted weights, and prompt-only versus explicit search/guidance. The papers above do not establish that diffusion avoids attention or that attention fails to learn team interactions. Our proposed benchmark must test that concern behaviorally rather than assume it.
