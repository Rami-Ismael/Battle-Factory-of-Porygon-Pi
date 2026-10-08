---
created_at: 2026-08-19
updated_at: 2026-08-24
authors: Sestini, Bergdahl, Barrette-LaPierre, Fuchs, Chen, Zinno, Jones, Gisslén (SEED / EA)
venue: RLC 2026 main track (confirmed 2026-08-20 — the earlier "slot unconfirmed" reading is retracted) · GDC 2026 talk
rlj: https://rlj.cs.umass.edu/2026/papers/Paper148.pdf
gdc_talk: https://schedule.gdconf.com/session/reinforcement-learning-in-fc26-shipping-human-like-goalkeepers-with-a-designer-first-approach/914241
arxiv: https://arxiv.org/abs/2510.23216
semantic_scholar: https://www.semanticscholar.org/paper/9982dbfdb91160c557bb7b020d794bce4b2e9299
access: 🟢
relevance: MED battle policy · LOW search
tags:
  - sample-efficient-rl
  - game-ai
  - human-like-agents
---
- [x] Can you teach me about the increase network plasticity — done 2026-08-19: hard resets of policy + Q-nets every 10⁵ steps, then an offline phase on the buffer at replay ratio 6.4×10³ before resuming online at 1; LayerNorm keeps it stable. Detail in chat
- [ ] Read the reset lineage this paper builds on: [Bigger, Better, Faster — human-level Atari with human-level efficiency](https://api.semanticscholar.org/arXiv:2305.19452) and [Breaking the Replay Ratio Barrier](https://openreview.net/forum?id=OpC-9aBBVJe).
- [ ] Decide which discrete off-policy implementation to start from — CleanRL discrete SAC, Tianshou discrete SAC, or the BBF code — since this paper ships no code and its SAC is EA-internal.
- [ ] Decide whether the FP / DO arms' best-response inner loop uses an off-policy learner — legal there because the opponent pool is frozen; [Neural Fictitious Self-Play](https://api.semanticscholar.org/arXiv:1603.01121) is the canonical example. The SP arm stays on-policy.
- [ ] Read the PPO-shaped translations of this paper's tricks: [Reincarnating Reinforcement Learning](https://api.semanticscholar.org/arXiv:2206.01626) for reset-then-distill and [Kickstarting Deep Reinforcement Learning](https://api.semanticscholar.org/arXiv:1803.03835) for annealed teacher distillation — the paper coins the term (Schmitt et al. 2018, arXiv preprint only); distinct from a permanent KL anchor, and [[Grandmaster level in StarCraft II using multi-agent reinforcement learning|AlphaStar]] never cites it. Detail in chat.
- [ ] Assess raising PPO's epochs-per-rollout with a KL early stop — the on-policy version of this paper's replay-ratio lever — as part of whichever PPO arm runs.
- [ ] Consider apply off policy algorithm to solve pokemon showdown if possible instead of ppo policy see the compare performances
- [ ] What are the best implementation of soft actor critic i think there was some from berekly unviersity do /research on this topic /goal that is your goal from now to find high recommend academic implemenation of Soft Actor Critic in PyTorch that can on my mac 
- [ ] - **PPO translations** — read Reincarnating RL (reset-then-distill) + Kickstarting (annealed teacher distillation).
- [ ] Start research alphrastar get the alpha star paper — paper note made 2026-08-23: [[Grandmaster level in StarCraft II using multi-agent reinforcement learning|AlphaStar]]

# BFF

| Component | Why it matters in the lineage |
| --- | --- |
| Larger Impala ResNet | Uses the plasticity afforded by resets to make model capacity beneficial rather than harmful. |
| Replay ratio $8$ | Reuses each scarce Atari interaction far more than older agents; BBF chooses $8$ rather than SR-SPR's demonstrated $16$ to control the cost of its larger model. |
| Stronger reset perturbation | Counteracts faster fitting and overfitting from capacity plus repeated updates. |
| Receding $n$-step horizon and increasing $\gamma$ | Starts with faster propagation of reward information, then shifts toward lower-asymptotic-error backups. |
| AdamW weight decay | Explicitly targets statistical overfitting, with larger gains at higher replay ratios. |
| SPR auxiliary objective and EMA target network | Stabilize representation/value learning; BBF finds the target network becomes critical after scaling. |

