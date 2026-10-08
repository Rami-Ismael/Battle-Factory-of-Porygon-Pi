---
created_at: 2026-09-01
updated_at: 2026-09-30
tags:
  - active-learning
  - membership-query-synthesis
  - discrete-diffusion
  - proposal-distribution
---

- Use a diffusion model to generate new, informative inputs, then feed those synthesized inputs back into the diffusion model—or into an evaluator—to probe its behavior.

- 2026-09-30: membership query synthesis means the learner builds the query instead of drawing it from a pool. Choosing for information or for win rate is the acquisition, a separate decision.

- [ ] > How can sampling-time guidance steer a discrete diffusion model toward regions of high epistemic uncertainty, when the guidance signal is by construction strongest where the model is already confident?

# Paper

- [ ] [Active Query Synthesis for Preference Learning](https://arxiv.org/abs/2605.26072) — 🟢 arXiv preprint, no venue listed, HIGH