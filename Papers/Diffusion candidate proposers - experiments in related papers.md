---
created_at: 2026-09-08
tags:
  - diffusion
  - black-box-optimization
  - experimental-design
---

# Diffusion candidate proposers: experiments in related papers

Research question from [[Diffusion as candidate proposer in black-box optimization over structured inputs]]: **What experiments do other papers actually run when diffusion generates optimization candidates?**

Two settings need separate comparisons. **Offline optimization** trains on a fixed labelled dataset and proposes a final batch. **Online optimization** repeatedly proposes candidates, obtains new objective evaluations, and updates its models. Your proposed CEM loop belongs to the latter. Generation-0 measurements can separately test the proposer before adaptation.

This is a focused comparison of primary papers, not an exhaustive survey. Paper observations below are separated from proposed VGC adaptations. “Oracle” means the benchmark's evaluator; it can be a lookup table, simulator, or learned predictor, so it does not necessarily mean a fresh physical experiment.

## Online experiments

### Diffusion-BBO — Wu et al., first posted 2024

**Role:** a score-conditioned diffusion model proposes designs; uncertainty-aware exploration selects the requested score.

**Tasks:** TFBind8/10 DNA strings, Ant/D’Kitty morphology, superconductors, and molecular discovery. The molecular task searches a continuous embedding, not molecular graphs directly.

**Protocol:** initialize from the 25th–50th score percentiles; query 100 candidates per round for 16 rounds, giving 1,600 new evaluations beyond initialization. Repeat three runs. Compare optimization curves against random sampling, evolution, CbAS, GP acquisition methods, TuRBO, LOL-BO and LFBO.

**Ablations:** adaptive requested-score selection versus fixed multipliers; batch size at equal total evaluations; training/acquisition runtime. A larger requested score does not automatically improve achieved performance.

**Boundary:** categorical tasks use relaxed one-hot/logit representations. Evaluators include measured DNA tables, robot simulation and learned material/molecular predictors. This is evidence for an online proposal loop, not specifically masked categorical diffusion.

Source: [full paper, §§6, D–E](https://arxiv.org/html/2407.00610).

### DiBO — Yun et al., ICML 2025

**Role:** diffusion supplies a learned prior, then is fine-tuned to sample a distribution favouring surrogate reward and uncertainty; local search and filtering select evaluations.

**Tasks/protocol:** Ackley, Rastrigin, Levy and Rosenbrock at 200/400 dimensions: 200 initial observations, batches of 100, maximum 10,000 evaluations, four seeds. HalfCheetah-102D, Rover-100D and LassoBench DNA-180D: 100 initial observations, batches of 50, maximum 2,000 evaluations, ten seeds. DNA-180D is a continuous hyperparameter problem, not DNA sequence generation.

**Comparisons:** optimization progress against CMA-ES, TuRBO, LA-MCTS, MCMC-BO, CMA-BO, CbAS, MINs, DDOM and Diff-BBO. One 400D baseline is capped at 6,000 evaluations for memory reasons.

**Ablations:** remove reweighting, local search, filtering or posterior fine-tuning; vary inverse temperature, buffer size, initial data, batch size, training effort and uncertainty estimation.

**Boundary:** useful for isolating which search components help; its continuous search machinery does not directly establish effectiveness for discrete legal teams.

Sources: [paper, §§4–5, appendices A/D](https://arxiv.org/html/2502.16824v2); [venue record](https://proceedings.mlr.press/v267/yun25a.html).

### SGPO — Yang et al., NeurIPS 2025

*Steering Generative Models with Experimental Data for Protein Fitness Optimization.*

**Role/tasks:** guide protein-sequence diffusion using small labelled datasets for TrpB, CreiLOV and GB1.

**Experiments:**

- Compare generative priors using 1,000 draws, retaining repeats; measure sequence likelihood, mean fitness and positional entropy.
- Compare continuous, D3PM and masked diffusion guidance against autoregressive DPO. Sweep guidance strength; plot fitness–diversity trade-offs. Use 200 labelled sequences, 200 generated samples and ten labelled-set repeats; also inspect uniqueness/novelty.
- Adaptive optimization acquires 100 unique novel sequences per round, with five random initializations. Compare mean/maximum fitness with unconditional generation, autoregressive DPO and APEXGo. Compare one guiding predictor with a ten-predictor ensemble.

**Boundary:** fitness comes from supervised oracles trained on experimental data, not newly assayed candidates. During adaptive guidance the generative prior stays fixed; labelled data update the guiding predictors. This is not diffusion CEM fine-tuning.

Source: [paper, §§4.1–4.3, A.2/A.5, Figures 4–6](https://arxiv.org/html/2505.15093v2).

## Offline experiments and guidance diagnostics

The companion note [[Diffusion candidate proposer - supplementary experimental evidence]] records the detailed tasks, baselines, budgets and ablations for these three papers.

| Paper | Experiment they actually run | Why it matters here |
|---|---|---|
| **DDOM, ICML 2023** | Six Design-Bench tasks; evaluate 256 final candidates across five seeds. Test reweighting, guidance, requested score, data size and candidate budget; a Branin test withholds the top 10% of data. | A template for your generation-0 benchmark and training-data experiments. [Paper, §4 and appendices](https://proceedings.mlr.press/v202/krishnamoorthy23a/krishnamoorthy23a.pdf). |
| **Diffusion Model for Data-Driven Black-Box Optimization, 2024 preprint** | Synthetic subspace, image and Hopper trajectory tests; vary requested reward/guidance and compare achieved reward with predicted reward and distribution shift. | A diagnostic for requesting implausibly high win rates. It is not a discrete-team benchmark. [Paper, §7/Appendix F](https://arxiv.org/html/2403.13219). |
| **dLLM, 2026 preprint** | Ant, D’Kitty and TFBind8/10; ten offline examples and 128 final designs. Report maximum/median normalized scores; ablate tree search, pretraining and prompts, and vary data size, tree depth and branching. | A generation-versus-selection comparison. Internal GP tree search does not make this an online oracle-feedback experiment. [Paper, §4/Appendix A](https://arxiv.org/html/2601.14446v1). |

## What I would transfer to the VGC project

These are proposed experiments, not experiments already performed in those papers. Keep your existing legality, diversity, copying and random-draw win-rate measurements.

| Experiment | Concrete comparison | Main result to inspect |
|---|---|---|
| Does diffusion help as the proposer? | Legal hierarchical sampling, real-team mutations, frozen diffusion and diffusion with CEM; same initial labelled data and battle budget | Freshly verified best-team win rate and fraction of proposals above a fixed target |
| Does improvement require adaptation? | Frozen diffusion versus CEM updates, starting from the same checkpoint | Win rate and diversity after each round |
| Does ranking explain the improvement? | A 2×2 comparison: mutation/diffusion proposer × random/surrogate-ranked selection, with equal candidate-pool sizes and evaluated batch sizes | How much comes from generation and how much from selection |
| Is stronger guidance useful? | Sweep requested win-rate condition separately from guidance strength | Achieved battle win rate versus diversity, legality and novelty |
| Does the initial data quality matter? | Equal-sized datasets with different score distributions, alongside your existing source-mixture comparison | Generation-0 performance and subsequent improvement |
| Does the loop need exploration? | If using a surrogate, compare mean-only ranking with an uncertainty-aware rule | Verified progress and distinct successful teams at equal battle budgets |

For each run, use the same regulation, battle policies, opponent weights and battle allocation. Record the number of proposals, evaluated teams and actual battles separately. A candidate evaluated through 100 battles is not equivalent in evaluation budget to one evaluated through 1,000 battles.

Keep **proposal quality** and **search success** as separate results. The former uses unranked random generated draws, preserving frequencies and duplicates as in your existing note. The latter can evaluate selected candidates, but finalists need fresh battles to avoid reporting a lucky observed maximum. Show the initial-data best alongside the generated best.

The most useful additions to your existing experiment axes are therefore **external proposer baselines, frozen-versus-updated diffusion, and generation-versus-ranking controls**. Parameter sweeps alone cannot establish that diffusion is contributing to search.
