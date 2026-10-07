---
created_at: 2026-08-20
updated_at: 2026-08-20
tags: [human-ai-interaction, offline-rl, chess, rlc-2026]
relevance: LOW for both — the two chess-from-logs papers at RLC 2026; skim the human-play-model and data-pipeline sections only
value_aware_interventions: "Improving Human Performance with Value-Aware Interventions: A Case Study in Chess (Narayanan, Panjwani, Sen, Ho) · RLC 2026 · 🟢 https://rlj.cs.umass.edu/2026/papers/Paper123.pdf"
chess_puzzles_offline_rl: "Discovering High Quality Chess Puzzles with Offline Reinforcement Learning (Nie, Badrinath, Tomlin, Dai, Yip, Wang, Brunskill, Piech; Applications award) · RLC 2026 · 🟢 https://rlj.cs.umass.edu/2026/papers/Paper147.pdf · 🟢 https://openreview.net/forum?id=MiQ10BUfRX"
related: "[[Convert Chess puzzles via offline RL (Applications) to pokemon vgc]]"
---
# Goal

**Goal:** Build the VGC equivalent of chess.com's Game Review — an AI coach that reviews and rates your decisions, so you improve by understanding your own mistakes.
# ToDo
- [ ] *Improving Human Performance with Value-Aware Interventions: A Case Study in Chess* (Narayanan, Panjwani, Sen, Ho) → 🟢 [RLJ pdf](https://rlj.cs.umass.edu/2026/papers/Paper123.pdf) — learns human-play models from large game datasets; the second chess-from-logs paper at RLC 2026, same skim as the puzzles paper · combined note:
- [ ] What is the web app beavhior
- [ ] Ship before the value function exists? , When we don't have a value function yet instead use a llm for now
- [ ] User identity, for now is just 
- [ ] Scoring Rule

# My 2026-08-21 current understanding of scoring rule


1. For example let say there is 100 point that you can give out, the probability of the value function is taking it will make the choice because the choice give probability distribution i was thinking about at the moment 70 percent .7*100 equal 70 it get 70 points while the .3for the other choice is the distribution is 70/30 x

# How does it handle user identiy

1. It start out using static website then use clerk i learn from theo modern full stack course

# Problem Statment for this 