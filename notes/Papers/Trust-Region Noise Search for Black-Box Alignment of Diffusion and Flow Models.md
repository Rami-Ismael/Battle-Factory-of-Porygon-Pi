---
created_at: 2026-09-02
updated_at: 2026-09-02
authors: Niklas Schweiger, Daniel Cremers, Keerthan Ram
venue: ReALM-GEN workshop, ICLR 2026 (short); accepted to ECCV 2026 (full, per arXiv v2)
url: https://arxiv.org/abs/2603.14504
code: https://github.com/niklasschweiger/trust-region-noise-search
access: 🟢
port_writeup: vgc-team-generator-pilot/docs/trs-port.md
tags:
  - trust-region-search
  - source-noise-optimization
  - diffusion-models
  - black-box-optimization
---

## Reassessment — 2026-09-07

**MED for reading; an exploratory search reference, not a ready-made replacement for the team generator.** The main [[Reading List]] already contained this paper under rejected readings, formerly LOW. The current [[Todo Section]] asks how discrete diffusion guidance changes team win rate and diversity. My assessment is that a way to improve generated candidates without retraining deserves a targeted read alongside that work.

TRS searches continuous source-noise vectors, keeping several promising centers and adapting local perturbation sizes using completed-sample rewards. It needs neither reward gradients nor a fitted surrogate. This is relevant to black-box team evaluation, but its coordinate mask is a noise-perturbation mask, not discrete token masking. [Sections 3.1–3.2](https://arxiv.org/html/2603.14504v2#S3)

The old cost rationale is too categorical: expensive rewards are an explicit motivation. Table 3 uses 400 reward calls for both TRS and random search, and Section 4.3 evaluates expensive protein-design rewards. Appendix E tests stochastic generation, with smaller protein gains when sampling noise weakens locality. These are not demonstrations of reliable ranking under binomial battle-result noise. [Experiments and Appendix E](https://arxiv.org/html/2603.14504v2)

**Project-specific inference:** categorical teams do not make adaptation impossible, but our masked generator needs an explicit, reproducible noise-to-team mapping before “nearby noise” has useful meaning. A future feasibility check should first measure duplicate rates, legality, and team changes under local perturbations. Any performance comparison must match total battles, not merely candidate counts: one reward estimate can require many battles. Compare against random generation and existing mutation/CEM search; account for lucky estimates when selecting finalists. This paper does not yet establish that such an adaptation earns its cost here.

Read **Section 3, Table 3, and Appendix E** first. Metadata correction: the third author is **Karnik Ram**, not Keerthan Ram; the original frontmatter is retained above for provenance. The full paper states ECCV 2026 acceptance. [Primary paper](https://arxiv.org/html/2603.14504v2)
