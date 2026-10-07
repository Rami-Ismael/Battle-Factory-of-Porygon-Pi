---
created_at: 2026-08-19
updated_at: 2026-08-23
authors: Grigsby, Xie, Sasek, Zheng, Zhu (UT Austin RPL)
venue: RLC / Reinforcement Learning Journal 2025
arxiv: https://arxiv.org/abs/2504.04395
access: 🟢
relevance: HIGH battle policy · LOW f
tags:
  - offline-rl
  - pokemon-singles
---
- [x] use the teach me from amttpolockc — done: Teaching/lessons/0003-metamon-offline-rl-from-human-replays.html + Teaching/reference/metamon-cheat-sheet.html
- [x] What do you the word use in context RL does mean it have other pokemon battle in their context — done: no, one battle only (fn. 1 Bo3 = unbuilt); Teaching/reference/metamon-in-context-window.html
- [ ] Determine how you to reimplement this work for best of three
- [x] Determine why they use glicko instead of cross matrix instead
- [ ] **Why from scratch?**: SynRL-V1 and SynRL-V2 were trained _from scratch_ on the enlarged dataset rather than fine-tuning the previous model. From what you know of offline RL here, give one reason why retraining from scratch is the safer choice when the dataset changes this much. Please use evidence from the paper like figure or experiment they implement
- [ ] What reward shapring did they do 
- [ ] [[What are the step before reading  the paper]]
- [ ] The hardest part: turning spectator replays into first-person training data
- [ ] Understand the sequence policy



1.  **Glicko: nothing to build.** It's computed server-side by the official Showdown ladder; you couldn't opt out if you wanted to. The account your agent plays from gets its Glicko-1 ± RD updated after every ranked game, and you just read the number off. Your local Showdown instance is only for training and cross-play battles, where no rating is involved at all.