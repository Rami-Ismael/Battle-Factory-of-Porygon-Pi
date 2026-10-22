# Primary-source paper verification

## Which paper is the strongest immediate choice?

### Takeaway
Recommend *Generative Bayesian Optimization: Generative Models as Acquisition Functions* for a focused reading session. This recommendation combines the verified mechanism below with the other researchers' current-project evidence. It is a reading recommendation, not evidence that implementing GenBO would improve battle win rate.

### Cited findings
- Rafael Oliveira, Daniel M. Steinberg and Edwin V. Bonilla wrote the paper, published at ICLR 2026. First submitted 29 October 2025, its latest listed revision is v3, 14 May 2026. [Canonical paper](https://arxiv.org/abs/2510.25240), [PDF](https://arxiv.org/pdf/2510.25240), [v3 text](https://arxiv.org/html/2510.25240v3).
- GenBO fits proposal distributions directly from observed utilities, samples batches, evaluates them and accumulates observations. It supports preference and divergence losses; the latter can retain utility magnitude. [§3 and Algorithm 1](https://arxiv.org/html/2510.25240v3#S3).
- The paper tests discrete sequences, including protein design with categorical and causal-transformer proposals. Diffusion and flow-matching extensions are left for future work. [§6.2 and §3.3](https://arxiv.org/html/2510.25240v3).
- Reading targets, verified with high confidence: §3, “A general recipe for generative Bayesian optimization”; Algorithm 1; §3.2, “Divergence-based learning”; §6.2, “Protein design”; C.3, “GenBO settings”; C.4, “Results summary”; C.5, “Ablations.” [v3](https://arxiv.org/html/2510.25240v3).

### Inferences
- The useful project question is whether improvement-weighted proposal fitting would be a worthwhile controlled alternative to elite-only CEM. Read to define that comparison before planning a new implementation.
- Suggested manageable tasks: 20 minutes mapping Algorithm 1 onto the existing loop; 30 minutes comparing elite selection with the divergence loss; 20 minutes extracting the relevant experimental controls. Each task should produce a short note, not a new experimental budget.

### Gaps
- The inspected paper does not establish higher VGC win rate, held-out-opponent generalization, or correction for selecting winners from many noisy battle estimates. It is not a substitute for the project's evaluation protocol.
- Do not describe its diffusion extension as an experimentally validated implementation for the current masked text sampler.

## Why do the strongest alternatives come later?

### Takeaway
*From Generation to Discovery* is the strongest alternative if the next concrete decision concerns learned mutation, novelty and independent verification rather than proposal fitting.

### Cited findings
- *From Generation to Discovery: Diffusion Mutation Kernels for Circuit and Physical Design* is by Dinithi Jayasuriya, Aravind Saravanan, Nilesh Ahuja, Amanda Rios and Amit Trivedi, dated 27 August 2026. It uses partial re-noising and denoising to mutate existing artifacts, with external tools judging quality and validity. Its domains include discrete circuit topology and continuous placement. [Primary paper](https://arxiv.org/html/2608.27649v1).
- Its Table 4 separates the learned prior from the learned mutation operator under matched candidate budgets. Read §3, “Diffusion as a Learned Search Operator,” and Table 4 selectively. [Primary paper](https://arxiv.org/html/2608.27649v1).
- *Design by adaptive sampling*, David H. Brookes and Jennifer Listgarten, initially 2018, connects probability-of-exceeding-threshold weighted fitting with CEM. Read §2, Algorithm 1 and “Connections to other methods” for that foundation. [Primary paper](https://arxiv.org/html/1810.03714v4).

### Inferences
- Prefer GenBO now because it gives the more direct comparison for changing the current proposal-update rule. Prefer *From Generation to Discovery* next if mutation structure becomes the active problem.

### Gaps
- Neither paper establishes that its benchmark gains transfer to Pokémon. DbAS being under a “Reading this now” heading does not prove completion.

## Why should Diffusion-BBO not automatically win?

### Takeaway
The October 13 priority correction placed Diffusion-BBO ahead of deferred fine-tuning papers. It did not put it ahead of every active paper, and the coordinator found a later project experiment using it.

### Cited findings
- *Diffusion-BBO: Diffusion-Based Inverse Modeling for Online Black-Box Optimization* is by Dongxia Wu, Nikki Lijing Kuang, Ruijia Niu, Yi-An Ma and Rose Yu, first submitted in 2024 and revised in February 2025. It trains a conditional diffusion model and chooses conditioning scores through uncertainty-aware exploration. [Primary paper](https://arxiv.org/html/2407.00610v3).
- Its §6.4 ablation shows that requesting higher conditioning scores does not reliably improve optimization. The most useful rereading targets are §4.1, Algorithm 1, §5 and §6.4. [Primary paper](https://arxiv.org/html/2407.00610v3).

### Inferences
- Consult these sections while diagnosing the existing experiment, rather than treating Diffusion-BBO as an unread next method by default.

### Gaps
- Its uncertainty estimates are not automatically the uncertainty of a Pokémon team's finite battle win-rate estimate.
