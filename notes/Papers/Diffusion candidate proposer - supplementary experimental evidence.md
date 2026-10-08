# Diffusion candidate proposers: verified offline experiments

Checked against primary full papers on 2026-09-08. These are experiments the authors actually ran, rather than proposed experiments for this project.

## DDOM — Diffusion Models for Black-Box Optimization (ICML 2023)

**Role:** offline inverse model; generates candidates conditioned on a requested objective value. Discrete inputs are relaxed to smoothed logits.

- **Benchmark:** TFBind8/TFBind10 DNA, ChEMBL molecules, Ant/D’Kitty morphology, Superconductor. Six tasks; NAS and Hopper are excluded from the main comparison.
- **Protocol:** fixed offline data; 256 final candidates; five seeds. Report normalized candidate scores, average rank, and variability. Bayesian optimization baselines operate on a learned surrogate, not new true-oracle feedback.
- **Comparators:** dataset best, CbAS, GP-qEI, CMA-ES, gradient ascent, REINFORCE, MINs, COMs.
- **Mechanism tests:** Branin denoising trajectories and inverse contours; remove the top 10% of training examples to test improvement beyond observed data. Ablate loss reweighting, conditioning value, classifier-free guidance. Appendix tests candidate budget, reweighting parameters, bin count, and training-data size.
- **Important distinction:** this is a final-batch offline experiment, not best-so-far performance over an online evaluation loop. The relevant structured tasks are DNA and molecules; success there does not establish hard combinatorial-constraint handling.

Evidence: [full paper, §§4.1–4.3, Tables 1–2, Appendices A–B](https://proceedings.mlr.press/v202/krishnamoorthy23a/krishnamoorthy23a.pdf).

## Diffusion Model for Data-Driven Black-Box Optimization (arXiv:2403.13219)

This theory-oriented paper tests how stronger reward guidance interacts with distribution shift.

- **Synthetic design:** 64-dimensional vectors on a 16-dimensional linear subspace. Train the reward predictor on 8,192 examples and diffusion on 65,536; evaluate 2,048 generated samples, averaged over five runs. Sweep target reward; measure actual reward, distribution shift, distance outside the subspace, and reward distributions.
- **Images:** Stable Diffusion v1.5 guided by a learned reward predictor. CIFAR-10 labels come from a synthetic ResNet-18-based reward function. Sweep guidance strength {25,50,100,200,400} and target {1,2,4,8,16}; generate 100 images per combination. Compare predicted versus ground-truth reward and unguided generation. Qualitative comparisons fix randomness.
- **Trajectories:** reproduce Decision Diffuser on Hopper medium-expert; vary requested return; report actual-return mean/standard deviation over ten episodes, with Trajectory Transformer/MoReL comparisons.
- **Finding:** requesting ever-higher reward can lower actual reward and worsen fidelity.

Evidence: [full paper, §7 and Appendix F](https://arxiv.org/html/2403.13219). This is useful evidence for a guidance/fidelity diagnostic; it is not a discrete candidate-proposer benchmark or a measured biological experiment.

## dLLM — Diffusion Large Language Models for Black-Box Optimization (arXiv:2601.14446v1, 2026)

**Role:** pretrained masked diffusion LLM proposes denoised designs; tree search uses GP expected improvement to select promising continuations.

- **Benchmark/protocol:** Ant, D’Kitty, TFBind8, TFBind10; uniformly sample ten offline examples. Evaluate 128 final designs. Report maximum and median normalized oracle scores and mean/median ranks.
- **Comparators:** gradient methods, COMs, ICT, MATCH-OPT, UniSO-T, ExPT, MIN, BONET, ORPO, GTG, DDOM, CMA-ES, MCTS-transfer.
- **Ablations:** replace pretrained diffusion LLM with continuous diffusion trained on the small dataset; remove tree search while retaining GP selection; remove task description. Also test classifier-free guidance.
- **Sensitivity:** Ant/TFBind8 tree depth {1,2,4,6,8}, branching {1,3,5,7,9}, offline examples {2,5,10,20}; alternative backbone and prompt wording.
- **Cost:** reported search episode roughly 30 minutes on Ant, 40 on TFBind8, using one A100 80GB.
- **Caveat:** this is offline optimization despite internal tree search. Pretraining is an additional resource, so its advantage over freshly trained DDOM does not isolate diffusion architecture alone. The inspected experimental prose does not specify the number of independent seeds behind reported ± values.

Evidence: [full paper, §§4.1–4.6 and Appendix A](https://arxiv.org/html/2601.14446v1).
