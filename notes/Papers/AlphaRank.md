---
created_at: 2026-08-18
updated_at: 2026-08-18
type: concept
title: AlphaRank
aliases:
  - α-Rank
  - Alpha-Rank
one_line: cross-play win matrix → invasion Markov chain (large α) → stationary distribution; rank agents by long-run survival mass; handles non-transitive pools where Elo cannot
paper: "[[Alpha-Rank Multi-Agent Evaluation by Evolution]]"
arxiv: https://arxiv.org/abs/1903.01373
access: 🟢 open
used_in: "[[VGC-Bench Towards Mastering Diverse Team Strategies in Competitive Pokémon]] §4.3, Table 2 — ordering only, picks the pool champion for §5.2–5.3"
implementation: open_spiel.egt.alpharank.compute(payoff_tables, alpha=…)
where_it_lives: battle-policy experiment (rank BC / SP / FP / DO from the cross-play matrix) — not ladder validation, not the objective function
related:
  - "[[Why Self-Play is consider Multi-Agent]]"
  - Czarnecki et al. 2020, Real World Games Look Like Spinning Tops — non-transitivity background
relevance: medium
tags:
  - multi-agent-evaluation
  - cross-play
  - non-transitivity
  - battle-policy
  - evolutionary-game-theory
---

- Alpha-Rank is already implemented in OpenSpiel — 🟢 [open_spiel/python/egt/alpharank.py on GitHub](https://github.com/google-deepmind/open_spiel/blob/master/open_spiel/python/egt/alpharank.py) (`compute(payoff_tables, m=50, alpha=100)`) — call it on the cross-play matrix; Claude does not vibe-code it.
