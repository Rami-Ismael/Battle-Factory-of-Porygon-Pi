# Original-paper verification for the next reading decision

## Which paper fits the current questions?

### Takeaway

Recommend continuing **Diffusion-BBO** now, with a focused pass through its search loop and target selection. Recommend **Generative Bayesian Optimization** as the strongest next new paper. This is a judgment about the user's current unanswered questions, not a ranking of research quality.

### Cited findings

- **Diffusion-BBO: Diffusion-Based Inverse Modeling for Online Black-Box Optimization**, Dongxia Wu, Nikki Lijing Kuang, Ruijia Niu, Yi-An Ma, Rose Yu. First posted 2024; latest arXiv revision v3, February 2, 2025. [Author manuscript and metadata](https://arxiv.org/abs/2407.00610), [free PDF](https://arxiv.org/pdf/2407.00610v3), [full HTML](https://arxiv.org/html/2407.00610v3).
- **Generative Bayesian Optimization: Generative Models as Acquisition Functions**, Rafael Oliveira, Daniel M. Steinberg, Edwin V. Bonilla. ICLR 2026; revised arXiv v3 includes a Section 3 correction. It learns proposal distributions directly from observed utilities, avoiding a separate regression or classification surrogate. [Author manuscript and metadata](https://arxiv.org/abs/2510.25240), [free PDF](https://arxiv.org/pdf/2510.25240v3), [full HTML](https://arxiv.org/html/2510.25240v3).
- **Posterior Inference with Diffusion Models for High-dimensional Black-box Optimization**, Taeyoung Yun, Kiyoung Om, Jaewoo Lee, Sujin Yun, Jinkyoo Park, 2025. This DiBO uses a diffusion prior, deep-ensemble forward predictions, and diffusion fine-tuning for posterior sampling. It is different from both Diffusion-BBO and the 2026 language-model DiBO paper. [Author manuscript and metadata](https://arxiv.org/abs/2502.16824), [free PDF](https://arxiv.org/pdf/2502.16824v2), [full HTML](https://arxiv.org/html/2502.16824v2).

### Inferences

- The coordinator recovered the newer current queue and paper note: the owner is already reading Diffusion-BBO and asks how inverse modeling actually optimizes, how targets are selected, and how this differs from CEM. Finish those questions before moving to another framework.
- GenBO is especially useful next for comparing elite-only retraining with utility-informed generation. Its lack of a separate forward surrogate means it is less direct as a response to the owner's wish to understand forward prediction.
- The 2025 posterior-inference DiBO is the strongest alternative if understanding a forward predictor becomes the immediate priority. Its Sections 3.1, 3.2 and 3.3 separate model training, candidate sampling, and evaluation/data updates. [Paper](https://arxiv.org/html/2502.16824v2).

### Gaps

- No verified evidence establishes which method improves this project's battle win rate. Reading relevance does not establish that an implementation will outperform CEM.

## What does Diffusion-BBO actually choose, and what transfers?

### Takeaway

It selects a conditioning score, samples designs, obtains new oracle labels, and updates the dataset. Asking for score y does not establish that generated designs achieve y.

### Cited findings

- Section 4.1, Figure 1 and Algorithm 1 describe learning p(x|y,D), forming candidate y values as a fixed weight times the maximum observed objective value, selecting one y by acquisition score, generating a batch, querying the oracle, and appending all evaluated pairs. The training labels are measured objective values; the proposed y is a request derived from observed values. [Section 4.1](https://arxiv.org/html/2407.00610v3#S4.SS1).
- Section 5.3 defines UaE as alpha(y,D) = y - Delta_epistemic(y,D). Section 5.2 separates model uncertainty from variability in generated designs. Figure 3 compares adaptive selection with fixed conditioning weights. [Sections 5.2 and 5.3](https://arxiv.org/html/2407.00610v3#S5.SS2).
- Discrete TFBind8/10 experiments use smoothed one-hot/logit representations. The theory assumes conditions including Lipschitz objectives and sub-Gaussian samples; it does not prove guarantees for arbitrary constrained categorical generators. [Sections 5.4, 6.1 and Appendix D.2](https://arxiv.org/html/2407.00610v3).

### Inferences

- This is relevant to the project's choice of what quality to request from its generator and how evaluated teams become later training examples.
- Generated-team variability, full-team uniqueness and battle-estimation noise are different quantities. An inverse-model uncertainty term is not automatically a measure of uncertainty in estimated battle win rate.
- Pokémon legality repair, unordered-team representation, repeated battles, and the chosen battle policy need separate validation. Oracle labels in a project adaptation should represent its established fixed-target battle objective. Broad metagame robustness is a different evaluation question.
- The paper does not supply a direct answer for ridge feature dimension 512 versus 128 or a guarantee against repeated teams. Those choices require project evidence.

### Gaps

- I did not verify an exact count of 40 fine-tuning rounds. That should not become a recommendation claim.
- No verified project transfer of the paper's assumptions, uncertainty calibration, or categorical embedding was established.

## What should the owner read and produce in one sitting?

### Takeaway

Use a 60-minute first pass to explain one complete iteration. This is a reading exercise, with no new training or experiment required.

### Cited findings

The relevant paper headings are Section 4.1, **The Diffusion-BBO Framework**; Section 4.2, **Conditional Diffusion Model Training**; Section 5.1, **Uncertainty Quantification on Conditional Diffusion Model**; Section 5.2, **Uncertainty Decomposition**; Section 5.3, **Uncertainty-aware Exploration**; Section 5.4, **Sub-optimality of UaE**; and Section 6.4, **Ablation study**. Figures 1 and 3 show the loop and target-selection comparison. [Full paper](https://arxiv.org/html/2407.00610v3).

### Inferences

Suggested task split:

1. **15 minutes.** Read Figure 1, Algorithm 1 and Section 4.1. Draw the iteration and mark which values come from past oracle calls.
2. **20 minutes.** Read Sections 5.1 through 5.3. Explain what y requests, what acquisition chooses, and why generated x still requires evaluation. Distinguish within-model sampling variability from disagreement across models.
3. **10 minutes.** Read Figure 3 and Section 6.4. Identify what the fixed-weight comparison tests. Skim Section 5.4 for assumptions, leaving proofs for later.
4. **15 minutes.** Write a one-page worked iteration. Start from hypothetical evaluated teams, construct candidate requests, assign clearly hypothetical acquisition values, choose one request, sample a batch, and append its hypothetical measured outcomes. Allow outcomes to fall below the requested score. Compare this data flow with the project's CEM loop and record one unresolved transfer question.

The concrete output is that worked iteration plus a short comparison of target selection, training-data selection and evaluation. If the owner can already produce it accurately, move directly to GenBO Sections 2 and 3 instead of rereading.

### Gaps

- The time allocation is an estimate for a conceptual pass. Mathematical proofs or implementing uncertainty estimation will take longer.
