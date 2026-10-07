# DIMES direct-citer transfer assessment

> Preliminary mechanism notes, superseded for queue decisions by `project-fit-rubric.md` and `screen-038-074.md`. After consulting current owner decisions, preference/reward integration remains deferred; NGS is MAYBE for a selective frozen-search comparison, not ADD. Molecular results alone do not justify a recommendation. CADO is a discovery lead outside the supplied 111-paper qualified inventory, so it is not an eligible recommendation in this sweep without a separately verified >1 citation count. SRT was subsequently examined and deferred because this project already trains with slot shuffling; generic augmentation would be redundant.

Checked 2026-09-08. This is a focused primary-source assessment, not the exhaustive citation inventory. Citation counts must be joined from the inventory before applying the user's **more than one citation** rule. No reading list was changed.

Project anchor: [[Diffusion as candidate proposer in black-box optimization over structured inputs]] defines masked categorical team generation, legality/diversity measurements, partial regeneration, and CEM elite/replay updates. Battle evaluation is noisy and the intended outcome includes the quality of random generated draws, not just a selected winner. That last distinction changes the value of best-of-N optimization papers.

All six entries below explicitly reference Qiu, Sun, and Yang's DIMES in the primary fulltext. Recommendations and proposed experiments are our transfer judgments, not claims that the papers validate Pokémon optimization.

## 1. Neural Genetic Search in Discrete Spaces — useful search experiment

[Paper, §§3 and 5.3 and references](https://arxiv.org/html/2502.10433v2) · [Official code](https://github.com/hyeonahkimm/ngs)

**Evidence:** A pretrained generator produces children by restricting available tokens to those found in two parents; occasional or constraint-triggered mutation removes the restriction. Population selection adapts search. This is test-time search with a pretrained policy, not CEM retraining. Beyond routing, its molecular experiment explicitly limits objective evaluations to 10,000. DIMES appears in the references and search comparisons.

**What to read:** §3, Algorithm 1, §5.3. **Decision:** add as a conditional search-method reading if the inventory verifies >1 citations.

**Experiment it would change:** From identical starting teams, compare ordinary frozen-model sampling, one-parent partial regeneration, and two-parent-restricted generation. Match evaluated-team and battle counts. Use typed field restrictions, not a union of every token: species, item, and move vocabularies have different meanings. Score fresh finalist battles separately from random-draw generator quality.

**Missing assumption:** The paper assumes sequential factorized generation. Applying its restrictions to masked diffusion is an adaptation; legality across fields and battle uncertainty remain unresolved. It supports testing crossover, not assuming it wins.

## 2. CADO: From Imitation to Cost Minimization for Heatmap-based Solvers in Combinatorial Optimization — strongest diffusion-training connection

[Paper, §§3–4, 6.5 and references](https://arxiv.org/html/2602.08210v1)

**Evidence:** CADO treats denoising as an MDP and uses the final decoded solution's cost for RL fine-tuning. It separates imitating labels from optimizing outcomes after decoding. Its standard-reward variant needs no labels during RL; label-centered reward instead uses an action-independent instance baseline. Hybrid adaptation combines LoRA with full updates to final layers. DIMES is explicitly cited and benchmarked.

**What to read:** §§3–4 and reward/parameter-update ablations. **Decision:** add if the citation threshold is met and direct reward-based fine-tuning is an intended alternative to CEM.

**Experiment it would change:** Starting from the same team checkpoint, compare CEM elite imitation with an RL update using measured reward after fixed legality handling. Match battle evaluations and preserve the current random-draw quality/diversity measurements. Initially use a small model, not the paper's graph architecture.

**Missing assumption:** CADO's evaluations have exact objective costs and deterministic decoders. Its low-quality labels are suboptimal solutions, not noisy repeated battle estimates. It does not establish sample efficiency or stability with stochastic battle rewards. Likelihood access for diffusion transitions is required.

## 3. Preference Optimization for Combinatorial Optimization Problems — conditional training alternative

[Paper, §3 and Appendix F.1](https://arxiv.org/html/2505.08735v1)

**Evidence:** Learns from relative preferences between generated solutions instead of depending directly on reward magnitudes; integrates local search into fine-tuning. Appendix F.1 actually applies preference training to DIMES and compares it with REINFORCE across decoding methods. The preference formulation uses trajectory policy probabilities.

**What to read:** §3.3–3.4 and Appendix F.1. **Decision:** secondary reading if evaluating alternatives to elite-only updates; not a required diffusion prerequisite.

**Experiment it would change:** Use one fixed evaluated batch to compare elite imitation with pairwise preference fine-tuning. Keep battle counts equal and count how many comparisons are supported by sufficiently separated estimates. Evaluate on fresh battles. This tests whether retaining loser-versus-winner information helps.

**Missing assumption:** Routing preferences are derived from known costs. They do not resolve noisy rankings. Comparing two teams' performance against the same fixed opponent distribution can define a scalar preference; direct head-to-head Pokémon outcomes may be nontransitive and do not automatically fit a scalar ranking model. Diffusion likelihood handling also needs adaptation.

## 4. Winner Takes It All: Training Performant RL Populations for Combinatorial Optimization (Poppy) — useful distinction, limited present fit

[Paper, §3.2 and references](https://papers.nips.cc/paper_files/paper/2023/file/97b983c974551153d20ddfabb62a5203-Paper-Conference.pdf)

**Evidence:** Trains complementary policies for expected best performance across a population, updating the winning policy for each instance. Shared encoders reduce overhead. The method does not guarantee diversity. DIMES is in its references.

**What to read:** Figure 1, §3.2. **Decision:** skim/conditional; not a direct remedy for diversity loss in the existing CEM distribution.

**Experiment it would change:** Compare one generator with a small collection of specialized heads at matched total candidate and battle counts, especially if conditioning on multiple target opponents. Measure per-head specialization and random-draw quality as well as the best candidate.

**Missing assumption:** Best-of-population performance is a different objective from producing a high fraction of strong random teams. Specialization across many problem instances may be less helpful for one fixed opponent pool. A lucky noisy winner can receive the update. It is a population of policies, not just a population of candidate teams.

## 5. Memory-Enhanced Neural Solvers for Routing Problems (MEMENTO) — defer until search memory matters

[Final NeurIPS 2025 paper, §3 and references](https://papers.nips.cc/paper_files/paper/2025/file/2f888eaccece0de24cc29d3ed02fd54b-Paper-Conference.pdf)

**Evidence:** Uses memory of previous attempts to adjust action distributions during inference. It learns a memory-processing/update mechanism rather than retraining model weights on each new observation. It targets the best solution found within a budget and is evaluated on TSP/CVRP. The final publication title differs from earlier “Efficient Adaptation in Combinatorial Optimization” versions. DIMES is referenced.

**What to read:** §3 and budget comparisons. **Decision:** defer: useful if the current search repeatedly ignores evaluated candidates, but a more involved mechanism than replay or deduplication.

**Experiment it would change:** Compare a generator conditioned on an archive of evaluated candidates with an otherwise identical generator that cannot see the archive; measure whether subsequent batches improve under equal battle budgets.

**Missing assumption:** This is a proposed adaptation, not a ready-made implementation for diffusion. It needs an appropriate representation of previous team choices, their interactions, and uncertain rewards. A batch of six-set teams does not inherit routing's transition-memory structure.

## 6. Regularized Langevin Dynamics for Combinatorial Optimization — promising words, weak current evidence

[Paper, §4 and references](https://arxiv.org/html/2502.00277v3) · [Official repository, RLNN section](https://github.com/Shengyu-Feng/RLD4CO)

**Evidence:** Studies discrete search using exact or neural approximate gradients. DIMES is referenced and benchmarked. The official repository distinguishes its published unsupervised neural loss, which still uses a differentiable objective, from a REINFORCE black-box option. It explicitly says the latter results were not reported in the paper.

**What to read:** Only the objective/gradient requirements before investing further. **Decision:** defer; do not sell it as validated expensive black-box optimization.

**Experiment it could change:** With a trustworthy learned win-rate surrogate, compare local guided proposals against unguided proposals using the same battle budget. That prerequisite is substantial; first measure surrogate ranking on newly generated teams.

**Missing assumption:** The team objective lacks an analytic gradient and may be noisy and misleading outside the observed data. Published graph benchmarks do not establish robustness to that situation. Optional code is not published experimental evidence of sample-efficient battle optimization.

## Citation-filter exclusions and caution

- [Genetic-guided GFlowNets for Sample Efficient Molecular Optimization](https://arxiv.org/html/2402.05961v3): conceptually strong combination of genetic refinement and off-policy replay, but neither v3 nor [v1](https://arxiv.org/html/2402.05961v1) contains DIMES, Qiu, or the DIMES title in the inspected reference list. Do not include as a verified direct citer without contrary bibliographic evidence. A GFlowNet is a different generator/training framework and should not silently become a prerequisite for the user's diffusion plan.
- [Self-Improvement for Neural Combinatorial Optimization: Sample Without Replacement, but Improvement](https://arxiv.org/html/2403.15180v2): its sampled-pseudo-expert loop is relevant, but the inspected v2 has no DIMES reference. Do not expand the user's citation filter simply because it is a good conceptual match.
- No paper assessed here removes the need for the project's existing legality checks, opponent-policy controls, and fresh-battle verification. This is a transfer limitation, not an added requirement to implement more methods.

## Suggested order if counts qualify

For **frozen-generator search**: NGS first. For **replacing CEM with direct reward fine-tuning**: CADO first, preference optimization next. For **multiple target instances and policy specialization**: skim Poppy. Defer MEMENTO and regularized Langevin until the corresponding bottleneck or prerequisite is demonstrated. Citation count is an eligibility gate, not evidence of project usefulness.
