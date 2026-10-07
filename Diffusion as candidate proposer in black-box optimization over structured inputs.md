---
created_at: 2026-08-29
updated_at: 2026-09-01
verdict: right setting, wrong name — the term of art is a learned proposal distribution for batch online black-box optimization over a discrete design space, inverse (conditional-generation) variant; utility is active search, not membership query synthesis
tags:
  - black-box-optimization
  - proposal-distribution
  - discrete-diffusion
  - active-search
  - mass-lift
---

# Axis


- [ ] The initial traiining dataset of the model which could be copy paste method, [[hierarchical product sampling]], and etc determine what type of properties it produce for example what is the number valid pokemon increaseing we could do some abalation studies or something
- [ ] Use nested subsets containing 25%, 50%, and 100% of the same pool, with similar source proportions. to Does more data improve win rate, legality, and diversity?
- [ ] This is like [[Ablation Experiments]]

# Todo

## Proposed experiment axes and measurements — 2026-09-07

An experiment axis is one choice we deliberately change between runs. A metric is a number we record to measure what that change does. The values below are proposed comparison settings, not established best values.

| Experiment axis                                                                  | Exact comparison                                                                                                                                                                                                                                                                                                                                                                                   | Question it answers                                                                                                                         |
| -------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| **Initial training-data source — Rami's proposed axis**                          | Train on real teams only; copy-and-paste mutations only; hierarchical product sampling only; 50% real teams + 50% copy-and-paste mutations; or 50% real teams + 50% hierarchical product sampling. Use the same number of distinct legal training teams and training updates in each run.                                                                                                          | Which source or mixture produces more legal, distinct, strong teams after training?                                                         |
| **Number of distinct training teams**                                            | Train on nested 25%, 50%, and 100% subsets of one dataset, preserving source proportions. Keep model size, batch size, and training-update count fixed for the first comparison.                                                                                                                                                                                                                   | Does access to more distinct examples improve generation at the same training budget?                                                       |
| **Distribution of measured training-team win rates**                             | From one labelled pool, construct equal-sized training sets following its observed win-rate distribution, balancing predefined win-rate brackets, or allocating a predefined larger share to high-win-rate brackets. Keep source proportions fixed where the pool permits; record the exact bracket boundaries and shares.                                                                         | Which distribution of weak and strong examples leads to better generated teams?                                                             |
| **Number of model parameters**                                                   | Compare small, medium, and large versions of the same architecture using the same dataset. Record exact parameter counts and compare at a matched training-compute budget.                                                                                                                                                                                                                         | Does additional capacity improve generation enough to justify its cost?                                                                     |
| **Classifier-free guidance coefficient during generation**                       | On the same checkpoint and requested win-rate condition, compare conditional sampling without amplification with two stronger coefficients. For the explicitly defined logits rule `guided = unconditional + s * (conditional - unconditional)`, example coefficients are `s = 1, 2, 4`; map these to the implementation's convention before running. Hold field-value sampling temperature fixed. | Does amplifying the requested condition improve verified battle win rate, and how does it affect legality and diversity?                    |
| **Temperature used to sample a value for a masked team field during generation** | On the same checkpoint, compare `T = 0.7, 1.0, 1.3` in `P(value j) = exp(logit_j / T) / sum_k exp(logit_k / T)`, summing over values allowed by the fixed decoding rules. Hold guidance, masking schedule, and legality handling fixed.                                                                                                                                                            | Does concentrating or spreading the probabilities used to choose species, moves, items, and other field values improve the generated teams? |
| **Number of Pokemon sets regenerated from a starting team**                      | Use the same real starting teams and regenerate 1, 3, or all 6 complete Pokemon sets. Keep the remaining sets fixed; specify and reuse how edited slots are selected. A complete set includes the species/form, moves, item, ability, nature, and stat spread represented by the model.                                                                                                            | How much of a known team can be replaced while preserving strength and producing new teams?                                                 |
| **Fraction of evaluated teams retained for CEM fine-tuning**                     | Retain the top 5%, 20%, or 50% by measured win rate from equal-sized evaluated batches. Keep batch size and fine-tuning-update count fixed.                                                                                                                                                                                                                                                        | How does stronger selection affect next-generation win rate and diversity?                                                                  |
| **Fraction of fine-tuning examples replayed from the original training dataset** | Allocate 0%, 25%, or 50% of sampled training examples to the fixed original dataset; allocate the remainder to current elites. Keep the elite rule and total training-update count fixed.                                                                                                                                                                                                          | If diversity declines across CEM generations, does replay preserve it while allowing win rate to improve?                                   |

The field-value sampling temperature controls how a team is generated. Lower positive values concentrate probability on higher-logit choices; higher values spread it more evenly. The separate temperature in [[New Todo Section with No Long Verbose Opus text geneation]] controls how evaluated teams are selected as training examples using win-rate weights. Record and vary these as separate parameters.

### Metrics recorded for every experiment

“Measurement panel” meant this shared list of recorded results; it is not an additional method or software component. Definitions and implementation belong in [[Metrics for Diversity]].

- **Whole-team legality:** legal six-Pokemon teams divided by all generation attempts. Record legality before repair and after repair separately, with the same attempt denominator.
- **Unique legal teams per 1,000 attempts:** count distinct legal full teams in a fixed batch of 1,000 draws. Keep duplicate draws in the batch. Use the same canonical team representation for every comparison, with the treatment of stat spreads explicitly defined.
- **Species/form composition diversity:** count distinct unordered six-species/form compositions and record their frequencies among legal generated teams. Also record the fraction occupied by the most common composition. Report the number of legal draws, since unequal legal sample sizes affect distinct-composition counts.
- **Copying and novelty:** report the fraction of legal draws that exactly match the model's own training set, and separately the fraction absent from a fixed real-team reference corpus shared by every run. Keep that reference fixed across CEM generations too.
- **Battle win rate:** evaluate an unranked random sample of legal generated draws against the same opponent pool using the same policies, opponent weights, and battles per draw. Preserve generation frequencies rather than evaluating only unique teams or preselected winners. Report sample size and uncertainty; record finalist results separately using fresh battles.

### Optional cost metric for model-scaling experiments

- **Cost:** record training compute/time, generation time for the fixed number of attempts, and the number of battle simulations.

For the current experiments on the local computer, report cost only when comparing model scales. For other comparisons, keep training updates and battle counts as experimental controls; cost is not a required outcome metric.

### Shared experimental controls

Fix the regulation and verifier across comparisons. Repeat across independent seeds. Measure before search fine-tuning (generation 0) and after each CEM update when evaluating the loop. For the data-source comparison, choose a dataset size that every source can supply as distinct legal teams; record source win-rate distributions to help explain any differences.

### Other research tasks

- [x] Look at other paper in the field that use diffusion model as candidate proposer what type of experiment they they do.
  - 2026-09-08: Compared six papers in [[Papers/Diffusion candidate proposers - experiments in related papers]], with offline details in [[Papers/Diffusion candidate proposer - supplementary experimental evidence]]. Includes actual tasks, evaluation budgets, baselines, ablations, and proposed VGC adaptations.


## Initial Candidate Pool

1. The current meta pokemon team add random mutation step until it reach a valid pokemon team 
2. The current problem for today 2026-09-01 that is when doing [[hierarchical product sampling]] the number of pokemon team that can beat the meta is like .15 percent we have to produce bias long tailed therefore it better to start from the meta pokemon add random mutation until it reach a valid pokemon team as a candidiate pool
3. Measured 2026-09-01 — a diffusion model trained on the 692 real teams is a third pool: 1.6% of its de novo draws reach the real median, and ranking 512 draws by a ridge lifts the battled batch to 9.4%. Repo README + `results/activesearch.json`. instead we should warm start should be base on the current meta pokemon

## Elite rule, and what the label is scored on

Measured 2026-09-01, 2×2 — a fixed elite cut beats a fixed 0.25 fraction by +0.012 [+0.002, +0.023]; scoring the margin instead of the win is null (+0.004) because margin correlates 0.95 with win rate. Live axis is now the selection ratio: 128 of 512 vs top 20 of 2,176.
