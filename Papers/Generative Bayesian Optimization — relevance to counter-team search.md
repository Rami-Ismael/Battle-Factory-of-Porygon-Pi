---
created_at: 2026-09-07
updated_at: 2026-09-07
grade: HIGH
tags:
  - black-box-optimization
  - proposal-distribution
---

# Generative Bayesian Optimization: Generative Models as Acquisition Functions

Already in [[Reading List]] as **HIGH**. Retain that grade: it directly addresses the learned proposal → expensive batch evaluation → generator update loop in [[Diffusion as candidate proposer in black-box optimization over structured inputs]].

Oliveira, Steinberg and Bonilla; **ICLR 2026**. Read arXiv v3, which includes a small Section 3 revision after the proceedings version. [Publication record](https://arxiv.org/abs/2510.25240).

GenBO trains a generator from observed utilities, samples the next evaluation batch, and accumulates observations. Its acquisition-driven distribution can replace a separate reward surrogate and acquisition optimizer. Utilities include probability of improvement and expected improvement. [Section 3, Algorithm 1](https://arxiv.org/html/2510.25240v3#S3).

Forward KL uses utility-weighted log likelihood; its importance-corrected form accounts for earlier sampling distributions. Experiments generally omit those importance corrections. Balanced forward KL adds a finite-sample penalty at low-utility observations. [Sections 3.2–4](https://arxiv.org/html/2510.25240v3#S3.SS2).

For our CEM work, the useful comparison is replacing binary elite inclusion with improvement-magnitude weights and testing the additional penalty. The overlapping fit–sample–evaluate loop alone does not establish an advantage over CEM.

Experiments cover discrete text and protein sequences; GenBO uses categorical models and causal transformers. Diffusion extensions are discussed as future work. Robust preference training assumes a preference-flip probability below one half. [Sections 3.1–3.3 and 6](https://arxiv.org/html/2510.25240v3#S3.SS1).

Our adaptation should preserve legal generation and account for how closure changes sampling probabilities. Battle estimates need a consistent evaluation budget; a single preference-flip rate may poorly describe teams with different uncertainty. These are implementation questions to test, not reasons to discard the idea.

First experiment: compare current CEM with improvement-weighted generator updates, holding initial teams, evaluation protocol and battle budget fixed. Measure independently evaluated best-team win rate and canonicalized diversity, matching [[Todo Section]].

Read **Section 3, Algorithm 1, Section 6.2 and Appendix C.3–C.5** first.
