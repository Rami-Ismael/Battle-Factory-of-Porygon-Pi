# Variational Annealing: selected forward citation evidence

> Current decision (2026-09-07): The owner accepted **COExpander** and **Unify ML4TSP** for the reading list. Both are graded **MED for selective reading**: COExpander for adaptive partial generation, Unify ML4TSP for generator/search evaluation. Their implementation limitations remain. The initial recommendation and non-acceptance statements below describe the earlier screening decision and are superseded by this acceptance. See [[Reading List — notes and decision history]] for placements and grade rationale.

Research evidence only; this is not an accepted reading queue. Checked 2026-09-07. This focused search covered adaptation, partial generation, and black-box diversity. It is not an exhaustive citation census.

Target: [Variational Annealing on Graphs for Combinatorial Optimization](https://arxiv.org/abs/2311.14156).

The relevance judgments below are project-specific inferences: the project generates legal VGC teams and ultimately compares counter-teams through costly, noisy battles. A method needing a known differentiable graph objective or expert optimal solutions does not transfer directly.

## COExpander: Adaptive Solution Expansion for Combinatorial Optimization

[Paper](https://proceedings.mlr.press/v267/ma25r.html), ICML 2025. **Conditional generator-design reading, otherwise LOW.**

Citation verified in the [published PDF](https://raw.githubusercontent.com/mlresearch/v267/main/assets/ma25r/ma25r.pdf), printed page 11: references include the full VAG-CO title, authors, NeurIPS volume 36, pages 63907–63930. Section 5 also attributes objective-guided MaxCut fine-tuning to Sanokowski et al. (2023; 2024).

The method progressively fixes decisions, with a consistency model conditioned on partial solutions; Section 4.2 describes task-specific determination operators. This could inform a team generator that fixes confident choices and completes remaining fields. That is an inferred adaptation, and matters only if the current generator's decoding or partial completion is a demonstrated problem. It does not resolve noisy battle comparisons. Sections 4.1 and H.1 state reliance on supervised solutions; H.2 reports limitations with more complex constraints, and H.3 leaves the proposed PPO extension as future work. These limitations make it inappropriate to present as a ready VGC search algorithm. [Method and limitations](https://raw.githubusercontent.com/mlresearch/v267/main/assets/ma25r/ma25r.pdf).

## Test-Time Adaptation for Unsupervised Combinatorial Optimization

[Paper](https://arxiv.org/abs/2601.21048). **LOW now; conditional future fine-tuning reading.**

Citation verified in the [v2 references](https://arxiv.org/html/2601.21048v2): full VAG-CO title, four authors and NeurIPS 36 pages 63907–63930 appear in the references (HTML line 394 when checked).

TACO initializes adaptation by shrinking trained weights and adding random perturbations, then runs unsupervised gradient updates. The paper compares against directly fine-tuning pretrained models and training from random initializations; pretrained weights can be an inferior starting point in its graph experiments. [Sections 2–3](https://arxiv.org/html/2601.21048v2#S3).

Potential VGC inference: if one later fine-tunes a pretrained team generator for a specific opponent, compare initialization strategies. Its tested losses are explicit graph objectives, however; it neither supplies battle-reward integration nor establishes savings in stochastic simulator calls. This is not a priority for the present training-pool comparisons. [Problem formulation and experiments](https://arxiv.org/html/2601.21048v2).

## Natural Variational Annealing for Multimodal Optimization

[Paper](https://arxiv.org/abs/2501.04667). **LOW; reject as a current recommendation.**

Citation verified in the [v3 references](https://arxiv.org/html/2501.04667v3): VAG-CO appears under Sanokowski et al. (2023), cited from the introduction (HTML line 718 when checked).

It fits a mixture search distribution while annealing exploration toward multiple modes, and includes derivative-free fitness-shaped variants. This provides a real black-box/diversity connection, but the formulation is continuous, smooth optimization and its concrete algorithm uses Gaussian mixtures. Turning it into discrete legal-team search needs new machinery. Crucially, Section 6.1 and Appendix S acknowledge larger evaluation budgets and reduced competitiveness on harder black-box benchmarks; the framework principally targets access to derivative information. This weakens its fit to expensive battles. [Sections 1, 6.1, Appendix S](https://arxiv.org/html/2501.04667v3).

## Result of this focused screen

No strong immediate recommendation among these three. COExpander is the closest architectural connection if adaptive partial generation is an active design question. TACO is a later adaptation experiment; Natural Variational Annealing has an actual black-box connection but unfavorable domain and evaluation-budget assumptions. Citation status alone should not earn any of them a place in the active queue.


## Additional primary-source checks

The combined review verified **nine direct citing papers** (the three above and six below). This is a focused screen, not the paper's total citation count. Semantic Scholar's API returned HTTP 429; OpenAlex returned duplicate records with incomplete citation coverage. Discovery therefore used indexed references and direct primary-text checks. Two OpenReview-only search hits could not be screened because their pages required browser verification; no recommendation is based on them.

| Direct citing paper | Citation evidence and current-project judgment |
| --- | --- |
| [Unify ML4TSP: Drawing Methodological Principles for TSP and Beyond from Streamlined Design Space of Learning and Search](https://proceedings.iclr.cc/paper_files/paper/2025/file/a3dc6e903082902d8e916bb9fccbfcbc-Paper-Conference.pdf) | VAG-CO appears in the references on printed page 14 and is discussed in §4.3.1. **Optional evaluation background.** Separates prediction, construction and improvement search; §4.2 examines when better learned predictions translate into better final solutions. Potential project use: hold downstream search fixed while comparing team proposers, then vary search separately. This is a proposed evaluation adaptation; the TSP results do not show battle-query efficiency. The general lesson alone does not justify a full read now. |
| [Learning to Explore and Exploit with GNNs for Unsupervised Combinatorial Optimization](https://www.cs.cornell.edu/gomes/pdf/2025_acikalin_iclr_x2gnn.pdf) | Full VAG-CO reference on printed page 12. **LOW now.** §4 constructs several coupled solutions, with communication between them, stochastic refinement and a diversity loss. This is more specific than merely generating many candidates, but its graph representation and differentiable objective/constraint/diversity losses require substantial redesign for teams. No demonstrated current need for a jointly trained population model. |
| [Clique Number Estimation via Differentiable Functions of Adjacency Matrix Permutations](https://proceedings.iclr.cc/paper_files/paper/2025/file/79f2e8ef5ca0b45ff549475a46b69c1f-Paper-Conference.pdf) | Full VAG-CO reference on printed page 14. **LOW.** Estimates a graph's clique number; neither the objective nor the representation supplies counter-team generation or stochastic battle evaluation. |
| [A Diffusion Model Framework for Unsupervised Neural Combinatorial Optimization](https://arxiv.org/html/2406.01661v3) | Full VAG-CO reference under Sanokowski et al. (2023). **LOW**, consistent with the earlier review: known-energy objective-driven training is deferred. See [[Neural combinatorial optimization - relevance grades]]. |
| [Scalable Discrete Diffusion Samplers: Combinatorial Optimization and Statistical Physics](https://arxiv.org/html/2502.08696v3) | Full VAG-CO reference under Sanokowski et al. (2023). **Skip current queue**, consistent with [[Scalable Discrete Diffusion Samplers - relevance assessment]]; diffusion-training memory and sampling correction are not the current project bottlenecks. |
| [Discrete Adjoint Schrödinger Bridge Sampler](https://arxiv.org/html/2602.08243) | Full VAG-CO reference under Sanokowski et al. (2023). **LOW.** A method for training discrete samplers using stochastic optimal control and bridge formulations, rather than evidence for legal-team proposal quality or noisy battle-budget allocation. |

## Related papers excluded from the direct-citation shortlist

These were inspected because they sounded relevant, but the checked versions reference later descendants rather than the requested VAG-CO paper. They are not counted among the nine direct citations:

- [Blocked Gibbs meets Diffusion Transformers](https://arxiv.org/html/2605.25129v1): references DiffUCO and Scalable Discrete Diffusion Samplers; no VAG-CO reference found.
- [StruDiCO](https://papers.neurips.cc/paper_files/paper/2025/file/6728fcf94660c59c938319a6833a6073-Paper-Conference.pdf): no VAG-CO reference found in the published reference list.
- [Discrete Diffusion Samplers and Bridges: Off-Policy Algorithms and Applications in Latent Spaces](https://arxiv.org/html/2602.05961): no VAG-CO reference found in the checked v3 reference list.

## Overall recommendation

**No strong new current-project reading recommendation emerged.** If retaining optional references, the nearest connections are listed below in project-relevance order. Neither is an accepted addition to the active queue:

1. [COExpander: Adaptive Solution Expansion for Combinatorial Optimization](https://proceedings.mlr.press/v267/ma25r.html)
2. [Unify ML4TSP: Drawing Methodological Principles for TSP and Beyond from Streamlined Design Space of Learning and Search](https://proceedings.iclr.cc/paper_files/paper/2025/file/a3dc6e903082902d8e916bb9fccbfcbc-Paper-Conference.pdf)

COExpander concerns how to complete partial candidates; Unify ML4TSP concerns how to isolate the contribution of learning within a search pipeline. These are limited architectural/methodological connections, not demonstrated VGC solutions. The main [[Reading List]] was not changed.
