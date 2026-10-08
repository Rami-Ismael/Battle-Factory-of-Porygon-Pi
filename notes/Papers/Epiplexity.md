1. https://www.youtube.com/watch?v=lKoJJxjUfdw

## Reassessment — 2026-09-07

**Recommendation withdrawn for the current project.** The proposed experiment below does not establish that an epiplexity score helps us choose training data. A selected mixture beating real-team-only training could also be found through ordinary mixture comparisons. Demonstrating added value would require better decisions on new mixtures at a lower total selection cost, including probe training and battle validation. We have no such evidence.

The paper's strongest practical case is inexpensive dataset diagnostics from training curves already available (§4.3). It also explicitly says epiplexity alone does not measure generalization (§8). Neither establishes that our proposed masked-loss adaptation predicts VGC win rate. [Paper](https://arxiv.org/html/2601.03220v2#S8)

The existing [[Todo Section]] experiment already compares real-team-only training with real teams plus mutations using win rate and diversity. Prioritize that direct comparison. Epiplexity remains relevant background for thinking about synthetic data; implementing and validating a new proxy has no demonstrated advantage here. The earlier proposal is retained below as a record, not a current recommendation.

## Research note — 2026-09-07

**What it means.** Epiplexity measures the structure a computationally bounded observer can extract. Formally, it is the description length of the probabilistic program minimizing **model bits + expected data-given-model bits** under a runtime limit. The second term is time-bounded entropy. Noise can have high entropy and little epiplexity; simple repetition can have little of either. The measure depends on the observer and compute budget. [Finzi et al., current v2, revised 2026-03-16, §3](https://arxiv.org/html/2601.03220v2#S3).

**Measurement and evidence.** The prequential heuristic sums each example's loss before training on it, minus its loss under the final model. Requential coding instead accumulates teacher-to-student KL on teacher-generated data. Estimating epiplexity also requires seeking the smallest two-part code within the compute budget. Prequential estimation is not rigorous; architecture, optimization and finite sweeps introduce error. The paper's cheap final-loss approximation assumes fresh, one-pass data and a small generalization gap. Its chess intervention matched puzzle accuracy and improved centipawn prediction, but higher epiplexity does not guarantee any particular downstream benefit. [§§4, 6.1 and Appendix B](https://arxiv.org/html/2601.03220v2#S4).

The [official implementation](https://github.com/shikaiqiu/epiplexity/blob/main/soph/train.py) logs `K_auc` as training-loss area above current loss, with processed tokens on the horizontal axis; `K_req` accumulates teacher–student KL. Its [README](https://github.com/shikaiqiu/epiplexity#estimating-epiplexity) distinguishes these model-code quantities from epiplexity, which requires a sweep over model size and training duration.

**Proposed use here — a hypothesis to test.** Use this idea to screen generator training corpora for learnable team structure. The [pilot's recorded HPS result](/Users/ramiismael/Documents/code/vgc-team-generator-pilot/README.md:127) already shows why legality and dataset size are insufficient: sampled legal teams averaged 0.015 win rate versus 0.491 for real teams. Those figures describe that recorded evaluation, not current universal performance.

Compare equal-size mixtures of real teams, whole-Pokémon set mutations, and HPS teams, including a real-team-only baseline. For a cheap first-pass score, use a shared vocabulary, encoding, model and training budget; freeze each example's corruption mask and slot ordering. Process each example once and record its masked prediction loss **before** its update. Re-evaluate those identical inputs, masks and targets with the final checkpoint, then sum the loss reductions over the same predicted fields. Repeat across seeds; inspect held-out loss and the train/held-out gap so memorization or unstable optimization does not masquerade as useful learning.

Call this an **epiplexity-inspired masked-loss proxy**: the [generator objective](/Users/ramiismael/Documents/code/vgc-team-generator-pilot/src/hpsdiffusion.py:107) scores masked fields, rather than a normalized joint team likelihood, and this pilot does not search the full compute-constrained code frontier. Neither final loss nor surrogate uncertainty measures epiplexity.

Test whether the proxy's ordering predicts generated-team win rate under matched training, proposal and battle budgets, using fresh battles against opponents held out from mixture selection. Track legality, diversity and copying alongside win rate; retain useful losing examples for conditioning. Adopt the screen only if its ranking predicts battle improvement beyond the baseline. Otherwise record a null and keep measured battle performance as the selection criterion.
