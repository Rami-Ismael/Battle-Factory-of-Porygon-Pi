Verdict: the 2026-11-12 answer stands. The post should say the measured benchmarks are smooth and name them, not that benchmarks are mostly smooth, and not that they are smooth because methods need it.
New this pass: papers do give a reason why benchmarks are simple, synthetic functions: they are cheap, easy to interpret and built to isolate one difficulty. None of them says designers pick smooth functions, and COCO puts rugged functions in on purpose.

# Why are academic optimisation benchmarks mostly smooth?

Checked 2026-11-13 for the third LinkedIn post.

## 1. The premise in Rami's own data

Where the numbers live. The smoothness note named in the brief, `notes/Landscape properties comparison with other black box function by smoothness.md`, holds his prose and a todo list but no benchmark numbers. The ten-task table is in `notes/Teaching/reference/bo-benchmark-derived-vgc-research-comparison.md`, section "Smoothness is inside the benchmark range, but it differs by field". Its verdict line, "Overall roughness does not set VGC apart", is in the paragraph after that table.

Neighbour correlation ρ = 1 − share²/2. Each cell gives ρ at random starting points, then at hill-climbed starting points.

| Task | ρ random · hill-climbed | Smooth near good points? |
| --- | --- | --- |
| Contamination control, 25 binary | 0.96 · 0.95 | yes |
| Ising sparsification, 24 binary | 0.93 · 0.92 | yes |
| Pest Control, both codes | 0.99 · 0.99 | yes |
| Weighted MaxSAT-60 | 0.995 · 0.999 | yes |
| Ackley-53 | 0.998 · 0.98 | yes |
| LABS-50 | 0.98 · 0.67 | no, turns rough near good points |
| Therapeutics Data Commons DRD2 | 0.99 · 0.08 | no, turns rough near good points |
| GuacaMol Zaleplon | 0.86 · 0.53 | no, middling at random points, rough near good ones |
| GuacaMol Perindopril | 0.43 · 0.03 | no, rough at both |
| Penalised logP | −4.2 · −5.8 | no, rough at both |

"Five of ten" holds if Pest Control counts once and the two GuacaMol tasks count separately. One correction: only LABS-50 and DRD2 really "turn rough near good points". Penalised logP and Perindopril are rough everywhere, and Zaleplon is already middling at random points. Better wording: "five stay smooth, five are rough, at least near good points".

The Pokemon row of that table, 0.96 · 0.92, comes from the September battles and is retired by FACTS.md. The benchmark rows involve no battles, so they still stand.

The edit-kind rerun measured two benchmarks again. `code/docs/ruggedness2.md`, "What is added", gives the design: Pest Control with one station changed, and a gridded sphere with one coordinate moved one grid step. The numbers are in `~/vgc-data/edit_kind_rerun/run/results/ruggedness2_shuffled50.json`, key `benchmarks`:

| Benchmark | ρ random | ρ hill-climbed |
| --- | --- | --- |
| Pest Control | 0.991 | 0.990 |
| Gridded sphere, one grid step | 0.994 | 0.997 |

The sanity check `sanity_sphere_step_rho_ge_0.95` is true. I did not open the Hartmann-6 numbers this pass. They are in `code/docs/smoothness-experiment.md`, not in either file named above.

## 2. The three sources already on record, quotes checked word for word

| Source | Section | Exact quote | What it supports |
| --- | --- | --- | --- |
| Oh et al. 2019, COMBO, https://arxiv.org/abs/1902.00448 | 1, Introduction | "As GPs rely on the smoothness defined by a kernel to model uncertainty [37], they are originally proposed for, and mostly used in, continuous input spaces." | Gaussian process methods assume smoothness. Says nothing about why benchmarks are chosen. Pest Control: Section 4.2, "a modified version of the contamination control with more complex, higher-order variable interactions". |
| Hansen et al., COCO, https://arxiv.org/abs/1603.08785 | 2, Why COCO? | Benchmark functions are "(b) designed to be comprehensible, to allow a meaningful interpretation of performance results" and "(f) models for 'real-world' problems; the currently available test suites (see Section 4) model in particular well-known problem difficulties like ill-conditioning, multimodality and ruggedness." | Confirmed. A major suite puts ruggedness in on purpose, so "benchmarks are smooth" is not a general truth. |
| Papenmeier, Poloczek, Nardi 2025, https://arxiv.org/abs/2502.09198 | 4, Discussion, "Notes on Popular HDBO Benchmarks" | "In short, these benchmarks have a simple structure that benefits models with long length scales, posing the risk of algorithms being 'overfitted' to these benchmarks." | Confirmed. Appendix D is titled "Popular Benchmarks Seem Simpler Than Expected" and covers Lasso-DNA and Mopta08. Simple here means few variables matter, not smooth. |

Two more passages in COCO that the 2026-11-12 pass did not cite:
- Section 1.1: "The majority of the mentioned test suites contain a large proportion of simple-to-solve and non-scalable problems without instance variations."
- Appendix A, How We Chose Test Functions: "We also wanted the functions to be comprehensible, in order to facilitate interpretation, and we required them to be scalable with the dimension."

## 3. Sources that give a reason, each quote verified in the paper

| Source | Section | Exact quote | The reason it gives |
| --- | --- | --- | --- |
| Kononova et al. 2025, Benchmarking that Matters, https://arxiv.org/abs/2511.12264 | 1, Introduction | "Most widely used academic suites are synthetic and deliberately constructed to isolate algorithmic phenomena: functions are chosen to highlight separability, ill-conditioning, multimodality, or plateaus and not to resemble real-world optimization landscapes" | Designers build synthetic functions to isolate one property at a time. Section 2.1 adds a cost reason: "To enable extensive experimentation, the evaluations are also usually demanded to be rather computationally inexpensive." |
| Stanton et al. 2024, Ehrlich functions, https://arxiv.org/abs/2407.00236 | 2.2, Existing Sequence Optimization Benchmarks | "closed-form functions have many appealing characteristics, including low cost, arbitrarily large search spaces, and amenability to analysis, however existing test functions for sequence optimization are so easy to solve that they are mostly used to catch major bugs." | Closed-form functions are cheap and easy to analyse, and too easy. Appendix A on the Rough Mount Fuji model, the closest thing to a smoothness critique: "those models assume mutational effects are either independent or additive, which disagrees with the correlated non-additive structure we observe empirically" and "We implemented the Rough Mt. Fuji model as an additional test function and verified that a genetic algorithm can easily optimize it." |
| Papenmeier, Nardi, Poloczek 2023, Bounce, https://arxiv.org/abs/2307.00618 | 1, Introduction, contribution 1 | "The analysis reveals that their performances often degrade considerably when the optimum of the optimization problem does not exhibit a particular structure common for synthetic test problems." | Synthetic combinatorial benchmarks have a built-in structure, such as MaxSAT-60's optimum at all zeros "by construction", Appendix B.1.3. Section 4.6 shows that COMBO on Pest Control drops to random-search level once the category labels are shuffled. That is about artificial structure, not smoothness. |

## 4. What the post can state, and what stays Rami's inference

The post can say: "The benchmarks we measured, Pest Control, weighted MaxSAT, Ackley-53, contamination control and Ising, are smooth under a one-variable step. Benchmark designers often use simple synthetic functions because they are cheap and isolate one difficulty at a time, and some designers warn that such functions are too easy." Sources: the table in section 1, Kononova Section 1, Stanton Section 2.2.

Rami's inference, not to be written as fact: that benchmarks are mostly smooth, since COCO adds ruggedness on purpose and five of his ten measured tasks are rough, and that benchmarks are smooth because methods need smoothness.

## What I could not verify

- No paper found says benchmark designers choose smooth functions, or that benchmarks are smooth because Bayesian optimisation assumes smoothness. Searched with alphaxiv discovery and the papers above.
- Hartmann-6 step results in `code/docs/smoothness-experiment.md` were not reopened this pass.
- The ten-task ρ values were read from the reference note, not recomputed from their results files.
