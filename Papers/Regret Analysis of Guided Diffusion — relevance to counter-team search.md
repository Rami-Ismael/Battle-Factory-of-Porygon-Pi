# Regret Analysis of Guided Diffusion — relevance to counter-team search

Checked 2026-09-07. **Relevant; retain MED priority.** Already recorded in [[Diffusion Model as Black Box Opimizer]], which [[Reading List]] links indirectly. Making its full title visible in the main list improves discoverability.

Masaki Adachi, Anita Yang, Yakun Wang, and Song Liu. *Regret Analysis of Guided Diffusion for Black-Box Optimization over Structured Inputs*. arXiv:2605.10385v1, 11 May 2026; the arXiv record lists no publication venue. [Paper record](https://arxiv.org/abs/2605.10385).

## What it contributes

The paper analyzes a generator → acquisition ranking → expensive evaluation loop. Its central diagnostic is **mass lift**: increasing the chance of proposing near-optimal candidates. Regret separates search failure from score-learning error and sampler approximation. Experiments concern molecules and crystals. [Sections 2–5](https://arxiv.org/html/2605.10385v1).

## Why it fits this project

[[Diffusion as candidate proposer in black-box optimization over structured inputs]] already describes diffusion proposals, ridge ranking, elite selection, and rare successful draws. My assessment: this paper directly helps interpret whether guidance increases useful candidate yield, or whether ranking is the bottleneck.

**Suggested experiment, not a paper result:** compare frozen-prior draws, guided draws, and the selected battle batch against the same fixed win-rate threshold. Use fresh battle seeds to estimate success fractions and uncertainty; record legality, duplicates, candidate count, and battle budget. Keep the opponent distribution and battle policies fixed during comparison. A chosen threshold measures useful-team yield; it does not identify the unknown global optimum.

## What does not transfer automatically

The formal setup uses a fixed finite prior support and independent additive observation noise. Generic guarantees require calibrated scores, smoothness/learning assumptions, labeled independent prior-refresh samples, and terminal sampler certificates. Ordinary diffusion losses alone do not establish those certificates; the proved resampling wrapper can be costly. Global score bounds can have huge constants. [Sections 2–4; Appendices G.4 and K](https://arxiv.org/html/2605.10385v1).

For VGC, Bernoulli battle outcomes have team-dependent variance; the noise assumptions need checking. Masking, repairs, changing policies, and ridge-based adaptive training also need separate justification. This is useful theory and diagnostics, not a guarantee that the current implementation converges.

Read Sections 2–3 and Appendix I first; consult Appendix K before interpreting guarantees. Keep this alongside the existing diffusion and active-search notes, behind immediate experiment implementation.
