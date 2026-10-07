---
created_at: 2026-09-18
updated_at: 2026-09-25
tags:
  - related-work
  - diffusion-models
  - black-box-optimization
---

| Item                                         | Source                                                                                                                                                  | Category     | What changed / what's being asked                                                                                                                 |
| -------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| Masked diffusion objective                   | [Simple and Effective Masked Diffusion Language Models](https://www.semanticscholar.org/paper/f8d357d38bbcdd93889fe71762eb57842b2ab063) 🟢 NeurIPS 2024 | Adapted      | Tokens become team fields; pinned slots carry over unchanged. Without constrained decoding, 0 of 80,000 samples were legal                        |
| Discrete classifier-free guidance            | [Simple Guidance Mechanisms for Discrete Diffusion Models](https://www.semanticscholar.org/paper/05f9997d61460fea9b586f98722c4be32f4a8b22) 🟢 ICLR 2025 | Adapted      | Conditions on playstyle, a wins bin and a legality token (the legality token is new). Playstyle steering works: sand appears at 27× its base rate |
| Guidance strength                            | [Classifier-Free Diffusion Guidance](https://www.semanticscholar.org/paper/af9f365ed86614c800f082bd8eb14be76072ad16) 🟢 NeurIPS 2021 workshop           | Adapted      | Does stronger guidance buy win rate? Strength 4 gives 0.193, the fidelity–diversity cost the paper warns of                                       |
| Condition on a high score                    | [Diffusion Models for Black-Box Optimization](https://proceedings.mlr.press/v202/krishnamoorthy23a.html) 🟢 ICML 2023                                   | Adapted      | Does asking for the top wins bin help? Null result: 0.008 vs 0.015 unconditioned. Its bin reweighting was not run                                 |
| Partial noise, then denoise                  | [SDEdit](https://www.semanticscholar.org/paper/f671a09e3e5922e6d38cb77dda8d76d5ceac2a27) 🟢 ICLR 2022                                                   | Adapted      | Mask k fields of a real team and regenerate them. Over 12,288 battles, every edit lost to leaving the team alone                                  |
| Look-ahead gradient guidance                 | [Guiding Diffusion Models Towards Generative Optimization, Mengdi Wang](https://www.youtube.com/watch?v=8W9qHXN0weg) 🟢 INI talk, 2024                  | Adapted      | Ported to a constrained discrete decoder. The loop reaches 0.411 by generation 10; best team 0.589 ± 0.036 at 192 battles                         |
| Trust-region noise search                    | [[Trust-Region Noise Search for Black-Box Alignment of Diffusion and Flow Models\|Trust-Region Noise Search]] 🟢 ICLR 2026 workshop                     | Adapted      | +0.057 over budget-matched random search (CI excludes 0), but capped at 0.109 by the frozen prior                                                 |
| Greedy expected-value acquisition            | [Bayesian Optimal Active Search and Surveying](https://arxiv.org/abs/1206.6406) 🟢 ICML 2012                                                            | Adapted      | Battle the proposals a surrogate ranks highest: +0.055, the first acquisition that beat random. The gain decays to nothing by generation 4        |
| Training on your own samples                 | [[Self-Consuming Generative Models Go MAD - reading guide\|Self-Consuming Generative Models Go MAD]] 🟢 ICLR 2024                                       | Investigated | Does a self-training loop collapse? It stopped at generation 2 on a diversity alarm, before win rate moved                                        |
| Cross-entropy loop with a diffusion proposer | Mine. Compared against [A Tutorial on the Cross-Entropy Method](https://people.smp.uq.edu.au/DirkKroese/ps/aortut.pdf) 🟢 author copy                   | Independent  | Elite fraction 0.20 → 0.01 raises copy-paste from 0.479 to 0.540, past a real team's 0.470. My sweep landed on the tutorial's default fraction    |

*Win rate against the top-50 meta teams, with the behaviour-cloning policy on both sides, 24 battles per team unless marked 192. Real teams average 0.47–0.49, depending on the run.*

Nothing in this project is a reproduction. The simulator, validator and battle policy are used as released. Every diffusion idea in the table was built for images, text or continuous design spaces, and here each one runs on a six-slot team that must pass Showdown's validator and is scored by noisy battles. Each row records what changed on the way over.

What the experiments add are measurements, not methods: which of these ideas survive that move. Most don't. Score conditioning, local edits and self-training are null or worse. Three carry over: gradient guidance at decode time, trust-region search over the noise, and ranking proposals by predicted value before battling them. The cross-entropy loop is my own, and the tutorial is cited only for comparison.


- [ ] Can you look at releted work section to the list of papers in related work section there will be list of heading that talked about the grouping can you list into a markdwon table and determine which be relevant for a releted work section
	- [ ] grill me
- [ ] Read the table
- [ ] How to write a related work section
