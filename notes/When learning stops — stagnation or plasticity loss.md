---
created_at: 2026-08-20
updated_at: 2026-08-20
verdict: two failure modes with disjoint fixes — run the learning-rate test before picking a remedy
stagnation_paper: https://api.semanticscholar.org/arXiv:2603.06009
plasticity_paper: https://api.semanticscholar.org/arXiv:2405.19153
tags:
  - ppo
  - plasticity
  - training-diagnosis
---

## Diagnosis — the learning-rate test

When the battle policy stops improving, rerun with the learning rate lowered ~5×. If the plateau moves later or disappears, the failure was **stagnation** (sample noise — the loss estimate stopped tracking the true objective). If the plateau stays put, suspect **plasticity loss**. Evidence for the test: a plateaued agent resumes learning immediately when the outer step size is reduced mid-training (Figure 3(a) of the stagnation paper) — a network that had lost plasticity could not do that. Background: [[PPO learning stagnation is not plasticity loss]].

## If stagnation — fixes from [Preventing Learning Stagnation in PPO by Scaling to 1 Million Parallel Environments](https://api.semanticscholar.org/arXiv:2603.06009) 🟢

- Collect more samples per update — more parallel environments; keep minibatch size and learning rate fixed and let the number of minibatches grow ("increase the number of minibatches while keeping the learning rate and minibatch size fixed", §5).
- Or lower the learning rate — escapes the plateau but too slow in wall-clock to be practical.
- No architecture or algorithm change — standard PPO throughout.
- Caveat for this project: Showdown + poke-env is CPU-bound, so the massive-parallelism version of the fix is capped; the per-update sample count is the knob that remains.

## If plasticity loss — fixes from [A Study of Plasticity Loss in On-Policy Deep Reinforcement Learning](https://api.semanticscholar.org/arXiv:2405.19153) 🟢

- Best: **soft shrink+perturb combined with LayerNorm** ("performs the best across our evaluated settings", Conclusion).
- Also works (the regenerative class — pull weights toward their initial distribution): shrink+perturb, regenerative regularization, L2 regularization.
- Avoid: plasticity injection and final-layer resets (underperform doing nothing on-policy); CReLU works in only some shift conditions; LayerNorm alone fixes training but is inconsistent on generalization.
- Self-play is a continuously shifting opponent distribution, so this is the likelier mode for the battle-policy experiment.

## Reading list

	

- [ ] Stagnation / noise-scale side:
	- [ ] 1. Read "[Batch size-invariance for policy optimization](https://api.semanticscholar.org/arXiv:2110.00641)" 🟢 (Hilton et al., NeurIPS 2022) — it owns PPO-EWMA, the tool the stagnation paper uses to vary the outer step size.
	- [ ] 2. Read "[An Empirical Model of Large-Batch Training](https://api.semanticscholar.org/arXiv:1812.06162)" 🟢 (McCandlish et al., arXiv preprint 2018) — it owns the gradient noise scale, which predicts when more samples per update stop helping.
	- [ ] 3. Read "[SAPG: Split and Aggregate Policy Gradients](https://api.semanticscholar.org/arXiv:2407.20230)" 🟢 (Singla et al., ICML 2024 Oral) — it is the large-scale-parallelism baseline the stagnation paper compares against.
	- [ ] 4. Read "[Multi-Task Reinforcement Learning Enables Parameter Scaling](https://api.semanticscholar.org/arXiv:2503.05126)" 🟢 (McLean et al., RLJ / RLC 2025) — it bridges the two branches: raising the task count mitigates plasticity loss.
	- [ ] 5. Decide whether the per-update sample count in Showdown + poke-env can be raised far enough to make the noise fix testable at all.
- [ ] Plasticity-mitigation side:
	- [ ] 1. Read "[Maintaining Plasticity in Continual Learning via Regenerative Regularization](https://api.semanticscholar.org/arXiv:2308.11958)" 🟢 (Kumar et al., CoLLAs 2025) — it owns L2 Init, regularization toward the initial parameters.
	- [ ] 2. Read "[On Warm-Starting Neural Network Training](https://api.semanticscholar.org/arXiv:1910.08475)" 🟢 (Ash & Adams, NeurIPS 2020) — it owns shrink-and-perturb.
	- [ ] 3. Read "[The Primacy Bias in Deep Reinforcement Learning](https://api.semanticscholar.org/arXiv:2205.07802)" 🟢 (Nikishin et al., ICML 2022) — it owns periodic partial resets.
	- [ ] 4. Read "[The Dormant Neuron Phenomenon in Deep Reinforcement Learning](https://api.semanticscholar.org/arXiv:2302.12902)" 🟢 (Sokar et al., ICML 2023 Oral) — it owns ReDo, dormant-neuron recycling.
	- [ ] 5. Read "[Deep Reinforcement Learning with Plasticity Injection](https://api.semanticscholar.org/arXiv:2305.15555)" 🟢 (Nikishin et al., NeurIPS 2023) — it owns plasticity injection, which failed on-policy but doubles as a diagnostic.
	- [ ] 6. Read "[Loss of Plasticity in Continual Deep Reinforcement Learning](https://api.semanticscholar.org/arXiv:2303.07507)" 🟢 (Abbas et al., CoLLAs 2023) — it owns CReLU and the Atari evidence that plasticity loss is real.
	- [ ] 7. Read "[Loss of plasticity in deep continual learning](https://www.nature.com/articles/s41586-024-07711-7)" 🟢 (Dohare et al., Nature 2024, open access) — it owns continual backpropagation.
	- [ ] 8. Read "[Disentangling the Causes of Plasticity Loss in Neural Networks](https://api.semanticscholar.org/arXiv:2402.18762)" 🟢 (Lyle et al., CoLLAs 2025) — it owns the LayerNorm-plus-weight-decay prescription and separates the mechanisms of plasticity loss.
	- [ ] 9. Read "[Plasticity Loss in Deep Reinforcement Learning: A Survey](https://api.semanticscholar.org/arXiv:2411.04832)" 🟢 (Klein et al., arXiv preprint 2024) — the taxonomy of 50+ mitigations; it finds generic regularization often beats RL-specific fixes.
	- [ ] 10. Decide which regenerative method goes into the battle-policy training loop first (soft shrink+perturb plus LayerNorm is the paper-backed default).
- [ ] Determine if possible to run multiple env parrel with pufferlib to run self play or that requires soft actor critic 
