# Learning the Travelling Salesperson Problem Requires Rethinking Generalization — relevance assessment

Assessed: 2026-09-07. **Decision: remove from the active reading queue; low immediate relevance.** This is a project-fit judgment, not a judgment of paper quality.

## What the paper actually tests

The study tests whether neural Euclidean TSP solvers trained on small graphs generalize to larger graphs. Its controlled experiments compare graph construction, GNN aggregation and normalization, autoregressive versus one-shot decoding, and supervised versus reinforcement learning. The objective is explicit tour length; supervised labels come from Concorde. [Sections 3–5](https://arxiv.org/html/2006.07054#S3)

The closest transferable finding is in Section 5.5: the ranking of supervised and RL models changes between greedy decoding and beam search/sampling. This supports evaluating a learned proposer together with its search procedure. It does **not** establish a general superiority of supervised learning for arbitrary proposal models. [Section 5.5](https://arxiv.org/html/2006.07054#S5.SS5)

## Fit to the counter-team project

The current task is to allocate expensive, noisy battle evaluations among legal counter-team candidates, using a masked-diffusion proposal prior, ridge ranking, elite updates, and warm starts. Battle policy is a separate component.

- Growing TSP graph size is a different generalization problem from changing opponents, Pokémon sets, or metagames while retaining six-member teams.
- Tour length is directly computable. The study does not supply an evaluation-allocation method for uncertain matchup estimates.
- Its autoregressive advantage concerns sequential tour construction compared with independent edge prediction. This is not evidence against iterative masked diffusion for team generation.
- The training-versus-search lesson is useful background, but it provides no immediate replacement or justified modification for ridge ranking, elite updates, or battle-budget allocation.

These are project-specific transfer judgments based on the method and experimental scope, rather than claims tested by the authors. [Problem definition and experiments](https://arxiv.org/html/2006.07054#S3.SS1)

## One lesson to retain

Compare complete search pipelines under the same battle budget, including a simple baseline. Test on held-out opponents. These are proposed evaluation practices for this project; the paper does not validate them on Pokémon battles. A full read is unnecessary for the current implementation.

Primary source: [Joshi et al., arXiv:2006.07054v6](https://arxiv.org/abs/2006.07054v6), accepted to CP 2021 and Constraints (2022).
