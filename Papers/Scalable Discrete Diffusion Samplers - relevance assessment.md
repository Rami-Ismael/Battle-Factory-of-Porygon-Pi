# Scalable Discrete Diffusion Samplers — relevance assessment

Checked 2026-09-07. **Recommendation: DEFER for the current counter-team search phase.** There is a real connection to reward-based generator training, but it does not justify prioritizing this paper for the current evaluation bottleneck.

## What the paper actually supplies

SDDS trains discrete diffusion samplers against an accessible energy function, without optimal-solution labels. Its two training approaches reduce memory requirements by batching diffusion steps: reverse KL with PPO and forward KL with importance weighting. The PPO formulation uses the completed sample's energy as a terminal reward; its gradient differentiates policy log-probabilities, not the energy function. An analytic energy gradient is therefore unnecessary. [Sections 1–3.1 and Appendix A.2.5](https://arxiv.org/html/2502.08696v3)

Experiments cover graph optimization with explicit objectives and Ising systems. Conditional graph models are trained across instances and evaluated on held-out instances. Optional conditional-expectation decoding exploits the chosen energy formulations. These experiments do not establish efficiency for costly, noisy battle evaluations. [Sections 2.2, 5 and Appendix A.3.2](https://arxiv.org/html/2502.08696v3)

Its sampling corrections concern estimating expectations under a specified target distribution using diffusion-path probabilities. They are not a treatment of uncertainty in estimated candidate quality. [Section 3.2](https://arxiv.org/html/2502.08696v3)

## Project judgment

The project searches for strong teams using a diffusion prior, candidate ranking, elite updates, and warm starts. Its immediate question is which candidates deserve scarce battle evaluations and how to trust improvements under battle noise. Saving diffusion-training memory is not the same as reducing the number of battles needed to discover a better team.

**The useful future adaptation:** treat denoising steps as policy actions and use a completed team's battle performance as the terminal reward. This is a proposed Pokémon adaptation, not a demonstrated result. It would require compatible team encoding and legality handling, opponent conditioning, and a controlled comparison against the existing update procedure under equal battle budgets. A differentiable battle simulator would not be a prerequisite.

For the importance-weighted route, simply replacing energy with a noisy score does not automatically preserve the target or sampling guarantees. As a mathematical observation, even when an estimate of reward is unbiased, exponentiating that estimate generally does not produce an unbiased estimate of the exponentiated expected reward. A fixed surrogate would define a different target whose quality must still be checked with battles.

**Revisit when:** direct reward-based training of the discrete diffusion generator becomes an active implementation goal, especially if training memory limits the denoising horizon. Then read Section 3.1 and Appendix A.3.6 first. Until then, keep it as an optional training reference and skip it in the active reading queue.
