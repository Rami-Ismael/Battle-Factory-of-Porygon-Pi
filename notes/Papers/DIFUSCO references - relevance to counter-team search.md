# DIFUSCO references relevant to counter-team search

Checked 2026-09-07. Scope: papers **cited by DIFUSCO**, not later papers that cite it. Reference numbers below were verified in [DIFUSCO's reference list](https://arxiv.org/pdf/2302.08224).

The project already uses a diffusion prior over teams, candidate ranking with a ridge model, elite updates, and warm starts. The useful question is how to improve this proposal-and-evaluation loop against a particular opponent. The Pokémon applications below are proposed adaptations, not findings demonstrated by these papers.

## 1. Optimistic Tree Searches for Combinatorial Black-Box Optimization — DIFUSCO [80]

[Paper, NeurIPS 2022](https://proceedings.neurips.cc/paper_files/paper/2022/file/d6099a36f6c1720438de00c366aa1737-Paper-Conference.pdf)

**Closest problem framing.** It considers sequentially querying a black-box objective over Boolean vectors and searches partitions of that combinatorial space optimistically. OLTS assumes a known Lipschitz constant; OCTS addresses the unknown case.

**Project application:** study how to allocate exploration across regions of candidate space rather than only drawing candidates from the prior. It gives a concrete contrast to proposal ranking and elite updates.

**Limitation:** Section 2's oracle returns the objective directly. Repeated stochastic battle outcomes do not satisfy that setup without additional treatment. The introduction targets moderate or cheap evaluation, describing the methods as complementary to Bayesian optimization for expensive evaluations with few queries. Team encoding, legal subsets, and meaningful distances also need work. Do not read this as a proven solution to expensive noisy battles.

## 2. Efficient Active Search for Combinatorial Optimization Problems — DIFUSCO [48]

[Paper, ICLR 2022](https://arxiv.org/abs/2106.05126) · [Full text](https://arxiv.org/html/2106.05126v3)

**Read first for improving the existing generator.** It adapts a learned solution constructor to one test instance by updating only a subset of its parameters. Its variants update embeddings, added layers, or a table. Neural variants also use an imitation term to reinforce the best solution found.

**Project application:** treat the target opponent as the instance. Use battle results to adapt a small portion of the proposer, retaining the broad team prior. This directly informs the choice between frozen proposal sampling and online elite-based updates.

**Limitation:** its construction models and losses are not directly interchangeable with a diffusion sampler. Its benchmarks have inexpensive deterministic objective evaluation; lower optimization overhead does not establish fewer required battle simulations. Noisy winner selection needs separate handling.

**Read:** introduction, Section 3, and Section 4.5's imitation-loss ablation.

## 3. DIMES: A Differentiable Meta Solver for Combinatorial Optimization Problems — DIFUSCO [92]

[Paper, NeurIPS 2022](https://arxiv.org/abs/2210.04123) · [Authors' code](https://github.com/DIMESTeam/DIMES)

DIMES represents a distribution over candidate solutions through continuous parameters, updates it using REINFORCE and parallel sampling, and meta-learns an initialization for instance-specific fine-tuning.

**Project application:** optimize the probabilities of proposing different teams instead of differentiating through a battle simulator. A learned initialization could eventually transfer experience from previous opponents to a new opponent. This is closer to the project's distribution-update problem than DIFUSCO's graph encoding.

**Limitation:** its graph representation, sampler, and decoding are task-specific. Parallel sampling can still demand many costly evaluations. It does not establish query efficiency for noisy Pokémon battles, and meta-learning would require a collection of prior optimization tasks.

## 4. Neural Large Neighborhood Search for the Capacitated Vehicle Routing Problem — DIFUSCO [47]

[Paper, ECAI 2020](https://arxiv.org/abs/1911.09539) · [Full text](https://arxiv.org/html/1911.09539v2)

It repeatedly removes parts of a solution and uses a learned repair policy to complete it, then evaluates the resulting candidate within a search procedure.

**Project application:** preserve a promising team's core and regenerate selected Pokémon or set choices. Compare this with drawing whole teams from scratch. This offers a concrete way to exploit warm starts while exploring coordinated changes.

**Limitation:** the routing repair policy cannot be reused as-is; conditional team completion, legality, and acceptance under noisy battle outcomes need new work. Neighborhood search alone also cannot show that distant regions of team space are unpromising.

## 5. Diffusion Models as Plug-and-Play Priors — DIFUSCO [32]

[NeurIPS 2022 paper](https://proceedings.neurips.cc/paper_files/paper/2022/file/5e6cec2a9520708381fe520246018e8b-Paper-Conference.pdf) · [Authors' code](https://github.com/AlexGraikos/diffusion_priors)

It combines a separately trained diffusion prior with a differentiable constraint/objective at inference, differentiating through the fixed denoiser.

**Project application:** this is the most relevant diffusion-specific reference if the goal becomes steering generation using a differentiable matchup surrogate. It motivates combining a team prior with opponent-specific preferences without retraining the whole prior.

**Limitation:** scoring discrete completed teams with ridge regression is not sufficient to supply gradients through the entire generation-and-decoding path. A compatible relaxation or representation is needed. Optimizing a surrogate can exploit prediction errors; the paper does not establish efficient noisy battle optimization.

## Optional foundation: Neural Combinatorial Optimization with Reinforcement Learning — DIFUSCO [6]

[Paper](https://arxiv.org/abs/1611.09940)

Read this if EAS's active-search idea is unfamiliar: sample complete solutions, evaluate their objective, and update the construction policy using reinforcement learning. Its objective-feedback training avoids requiring a labeled optimal solution for every training instance. The TSP/knapsack results do not establish practicality when rewards require many stochastic battles.

## What this shortlist does not answer

These references supply ideas for adaptive proposals and structured local search. They do not resolve the project's central evaluation-budget question: which teams deserve additional battles, how uncertainty should affect selection, and whether improvement survives independent evaluation. Treat their algorithms as hypotheses to test against the current proposer under the same battle budget, not as evidence that a more complex model will help.
