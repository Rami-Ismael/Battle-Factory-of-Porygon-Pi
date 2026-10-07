---
created_at: 2026-08-18
updated_at: 2026-08-18
type: paper-conversion
source_paper: Discovering High Quality Chess Puzzles with Offline Reinforcement Learning
authors: Allen Nie, Anirudhan Badrinath, Nicholas Tomlin, Timothy Dai, Carissa Yip, Rose E Wang, Emma Brunskill, Chris Piech
affiliations: Stanford, UC Berkeley
venue: RLC 2026 / RLJ 2026
award: Outstanding Paper — Applications of RL
pdf: https://rlj.cs.umass.edu/2026/papers/Paper147.pdf
access: 🟢 open
method: advantage-weighted actor-critic (AWAC/IQL-style) with KL-to-behaviour, causal transformer policy, action embeddings + in-batch softmax
dataset: 1.5B Chess.com puzzle plays, 3.1M users, 441K puzzles, 2021-03 to 2022-03
evaluation: one-step importance sampling OPE, expert rubric (8 titled players), LLM-scaled annotation
relevance_to_search_stage: LOW
relevance_to_battle_policy_stage: MED
tags:
  - offline-rl
  - rlc-2026
  - battle-policy
  - paper-conversion
  - chess
---

| Piece | Chess paper |
|---|---|
| Dataset | 1.5B puzzle plays, 3.1M Chess.com users, 441K puzzles, one year. Behaviour policy = Chess.com's *bucketed uniform* (puzzle Elo within ±200 of user, easing after misses) — **propensities are known** |
| State | transformer over the user's history: user features (Elo, correctness) + puzzle features (CNN on the board + first move), 32-step context |
| Action | which puzzle to serve next — 441K discrete |
| Reward | $r = \frac{c}{N}\exp(\alpha(\text{Elo}_\text{puzzle}-\text{Elo}_\text{user}))$, $\alpha=0.002$: fraction of correct moves, weighted up when the puzzle is harder than you |
| Algorithm | advantage-weighted MLE (AWAC / IQL): $L_\pi = -\log\pi_\theta(a\|s)\exp(\tfrac1\beta(Q-V))$, $V$ by expectile regression, KL-to-behaviour. Policy prob = softmax of dot product between predicted embedding and item embedding, normalised over in-batch negatives — this is the trick that makes 441K actions tractable |
| Evaluation | no RCT possible → one-step importance sampling with weights clipped to [0.1, 10], broken down by Elo band and growth/stagnant users; then 8 titled players (2 GMs) score 30 puzzles on a rubric; LLM used to scale the rubric |

- [[What are the step before reading  the paper]]