# DiffUCO — selection temperature experiment

Checked 2026-09-07 against the requested **arXiv v1**. **Project decision: test annealed selection as a controlled retraining ablation.** This note specifies the adaptation and interpretation; it reports no Pokémon experiment result.

## What the paper establishes

DiffUCO targets a Boltzmann distribution, $p_{\mathcal T}(x)\propto\exp[-H(x)/\mathcal T]$. The associated free-energy objective is expected energy minus temperature times model entropy. [Sections 2–2.1](https://arxiv.org/html/2406.01661v1#S2).

For diffusion models, it minimizes a reverse **joint** KL that upper-bounds the intractable marginal reverse KL. Equation 6 combines expected energy, conditional-entropy regularization, and forward/reverse diffusion-path coupling. The partition function contributes an additive constant independent of model parameters and is unnecessary for the gradient. [Sections 3–3.1](https://arxiv.org/html/2406.01661v1#S3).

Section 4 explicitly specifies a **linear decrease from the starting temperature to zero**. Appendix C.5 lists task-specific starting temperatures and annealing durations. This training schedule is separate from the diffusion-time noise schedule in Section 3.2. The implementation uses tractable conditional expected energies for its combinatorial problems. [Sections 3.2–4](https://arxiv.org/html/2406.01661v1#S4), [Appendix C.5](https://arxiv.org/html/2406.01661v1#A3.SS5).

## Our adaptation and its limits

The proposed project experiment resamples a finite evaluated pool before ordinary denoising retraining. For measured win rate $\widehat w_i\in[0,1]$, define

$$
a_i(\tau)=\frac{\exp[(\widehat w_i-\max_j\widehat w_j)/\tau]}{\sum_j\exp[(\widehat w_j-\max_k\widehat w_k)/\tau]}.
$$

Subtracting the maximum changes no probabilities and prevents overflow. The normalization is over the available pool and is directly computable. This does not establish sampling from the Boltzmann distribution over all legal teams, nor implement the paper's training objective. It is a project adaptation inspired by its temperature curriculum.

The intended comparison is linear selection temperature $0.30\rightarrow0.10$, fixed $0.10$, $0.20$, and $0.30$, and adaptive selection temperature. With $R>1$ rounds, an endpoint-inclusive linear schedule is $\tau_r=0.30+(0.10-0.30)r/(R-1)$ for $r=0,\ldots,R-1$. Report the single-round convention explicitly. These temperatures assume fractional win rates; percentage scores would require temperatures scaled by 100.

Selection temperature, decoding temperature, and an added loss-entropy coefficient are different controls. Hold decoding temperature fixed and the separate loss-entropy coefficient at zero. Keep the selection floor positive; literal zero requires an explicit limiting rule and would put weight only on tied maximum measured scores.

## Proposed evaluation rules

These are experimental-design recommendations and mathematical deductions for this project, not findings reported by DiffUCO.

- **Match the comparison.** Reuse the same starting checkpoint and initial pool within each seed; match architecture, optimizer, updates, batch sizes, generation count, legal-team handling, battle policy, opponent mixture, and battle budgets. Across rounds, each arm may develop its own pool; that is part of the feedback effect being tested. Track actual completed battles and failed evaluations.
- **Define adaptation before running.** For normalized weights, use effective sample size $\mathrm{ESS}=1/\sum_i a_i^2$ and log $\mathrm{ESS}/n$, maximum weight, and selected-team counts. Choose a target fraction and temperature bounds in advance. Increase temperature when weights concentrate too much. If the target cannot be reached within the bounds, record that condition. Equal scores produce uniform weights at every positive temperature; tied maxima can limit attainable concentration.
- **Handle the pool's base distribution.** Repeated copies of a team acquire extra aggregate probability if each copy is an entry. Canonicalize team identity and document whether repeated observations are merged before weighting. Pool ESS measures weight concentration, not generated-team diversity.
- **Measure diversity on equal-size raw draws.** Record exact-team duplicate rate, unique species-composition rate, species usage entropy, and a team-composition distance such as pairwise Jaccard distance. Compute diversity before selecting finalists or discarding duplicates, with canonical identities invariant to roster order. Report legality separately and identify whether distance concerns species or full sets. Preserve a fixed sample size so a larger candidate batch does not appear more diverse merely by offering more draws.
- **Treat battle estimates as noisy.** Under independent Bernoulli outcomes, $\operatorname{Var}(\widehat w_i)=w_i(1-w_i)/n_i$. Low temperatures magnify score errors because the log weight ratio is $(\widehat w_i-\widehat w_j)/\tau$. Also, $\mathbb E[\exp(\widehat w_i/\tau)]\geq\exp(\mathbb E[\widehat w_i]/\tau)$: exponentiating an unbiased score does not yield unbiased exponentiated quality. Match evaluation effort per candidate; any shrinkage or uncertainty correction belongs in a separately controlled ablation.
- **Separate generation quality from finalist quality.** Evaluate a fixed random batch of final-generator samples, and separately verify the same predeclared number of finalists per arm using fresh battle seeds excluded from selection and retraining. A winning finalist alone does not establish higher average generator win rate. Keep the confirmatory opponent distribution fixed; an additional opponent holdout answers generalization separately.
- **Make the claim across seeds.** Pair arms within seeds and report seed-level win-rate differences, diversity differences, and battle counts. Predeclare a practically useful win-rate gain and allowable diversity loss. Summarize battle uncertainty and between-training-seed variation separately. Do not treat many battles from one trained model as many independent training replicates. If selecting the best fixed temperature from these runs, acknowledge that selection or use a separate confirmatory comparison.

An improvement claim requires fresh-battle evidence of a win-rate gain and diversity within the predeclared tolerance. An underpowered or mixed pilot should remain inconclusive and identify the additional seed or battle budget needed.
