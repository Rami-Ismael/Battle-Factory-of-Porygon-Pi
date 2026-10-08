---
created_at: 2026-09-06
updated_at: 2026-09-07
type: full-paper-relevance-reassessment
workshop: 4th Structured Probabilistic Inference & Generative Modeling
papers_fully_read: 18
initial_keep_med: 6
initial_defer: 9
initial_not_recommended: 3
initial_high: 0
supersedes: SPIGM 2026 — paper-by-paper relevance screening (accepted-paper recommendations only)
tags:
  - reading-list
  - workshop
  - discrete-diffusion
accepted_readings: 2
accepted_med: 2
accepted_high: 0
keep_med: 2
---

> [!important] Current owner decision — confirmed 2026-09-07
> **Only two SPIGM papers are accepted for the current reading list, both MED:** #169 **A Tale of Two Temperatures: Simple, Efficient, and Diverse Sampling from Diffusion Language Models** and #170 **Re-evaluating Confidence Remasking in Masked Diffusion Language Models** (screening entries #15 and #83).
> This records the owner's 2026-09-06 confirmation and 2026-09-07 correction. The other 16 initial additions are not in the active queue, including #174 feature-distribution matching. Earlier grades, counts and recommendations below are historical and are superseded by this decision.

## Historical assessments — superseded by the owner decision above


> [!important] Latest relevance correction — 2026-09-06
> **Current SPIGM recommendations: 3 KEEP MED, 9 DEFER, 6 DROP; none HIGH.** #183, Hacking Generative Perplexity, no longer earns a current reading recommendation. Its established result concerns frozen-language-model perplexity and token entropy. The proposed Pokémon diversity connection was an analogy, not a tested team metric or algorithm. The existing Metrics for Diversity note already calls for full-team uniqueness, novelty, unordered six-species/form composition counts and structural generalization. A general warning about marginal metrics adds insufficient value to justify this paper.
> Remaining MED readings: #174 feature-distribution matching, #169 Two Temperatures, and #170 Re-evaluating Confidence Remasking. This supersedes the four-paper shortlist below; earlier assessments remain as history.



> [!important] Latest owner correction — 2026-09-06
> **Current SPIGM recommendations: 4 KEEP MED, 9 DEFER, 5 DROP; none HIGH.** This supersedes the earlier six-paper shortlist. The owner clarified that diffusion training is not a current problem and exact duplicate teams can already be removed. Consequently, #181 (training acceleration) and #173 (exact data repetition) no longer earn a current reading recommendation. The repetition paper does not establish that distinct mutations have the same effect as exact duplicates.
> The four remaining MED readings are #183 Hacking Generative Perplexity, #174 feature-distribution matching, #169 Two Temperatures, and #170 Re-evaluating Confidence Remasking. Earlier judgments are retained below as history.



# SPIGM 2026 — full-paper reassessment

**Of the 18 papers previously accepted, six remain worthwhile MED readings, nine are deferred, and three no longer earn a place in the current recommended queue. None warrants HIGH priority for the present project.** These decisions supersede the original four HIGH/fourteen MED screening recommendations.

All 18 main texts and all substantive appendices were read, including methods, mathematical arguments, experimental protocols, ablations and limitations. Substantive figures received a separate visual check. The source-version and section receipt below makes the scope explicit. This was critical reading, not reproduction of experiments or formal proof certification; cited bibliographies were not themselves read in full. Later author manuscripts are identified, rather than assumed identical to workshop submissions.

KEEP MED means there is a distinct useful lesson for the active project, after the existing HIGH/reimplementation sequence. DEFER means a specific prerequisite or later project phase is missing. DROP means the reading cost is not justified in the present queue; it is not a judgment that the paper should not be published. Original reading-list entries are retained with superseding annotations; this report does not delete vault content.

## What changed after checking the actual project

The current questions are real-only versus real-plus-mutation training, CFG versus verified legality/win rate/diversity, and diversity across CEM generations with equal-sized samples and duplicate draws retained. Original-data replay or prior regularization is a possible response to measured diversity loss. Reward integration remains deferred. See [[Todo Section]] and [[Metrics for Diversity]].

The inspected pilot code changes several reading decisions:

- The generator has 48 categorical fields and fills **one field per fresh transformer call in fixed dependency order**, with feasible-value masking. It is neither confidence-first nor simultaneously sampling multiple fields from one pass. Later fields already condition on earlier sampled choices. See [sampler](</Users/ramiismael/Documents/code/vgc-team-generator-pilot/src/hpsdiffusion.py:181>) and [field order](</Users/ramiismael/Documents/code/vgc-team-generator-pilot/src/diffusion.py:310>).
- Training uses random masks, a 1/t weight and masked-field normalization; team members are permuted. Text-locality and objective claims need a careful transfer. See [training loss](</Users/ramiismael/Documents/code/vgc-team-generator-pilot/src/hpsdiffusion.py:90>) and [team permutation](</Users/ramiismael/Documents/code/vgc-team-generator-pilot/src/activesearch.py:136>).
- Each inspected CEM refit reloads the original p0, rather than starting from the previous fitted model. This is different from both data replay and a KL anchor. Generated-data concentration can still change. See [refit](</Users/ramiismael/Documents/code/vgc-team-generator-pilot/src/activesearch.py:183>).
- Proposals receive copied spreads and final validation; duplicates remain. Raw grid probability is not automatically accepted, canonical-team probability. See [proposal path](</Users/ramiismael/Documents/code/vgc-team-generator-pilot/src/activesearch.py:197>). Actual evaluation targets battle performance and structural diversity, with no identified likelihood-ranking requirement.

Historical results also need care. Confidence versus dependency decoding was approximately tied in win rate, with much greater generation time for confidence. Other orders differed and had different feasibility/fallback behavior. The best temperature arm changed both temperature and top-p. Exact uniqueness saturated at 1.0 in these small samples, so it did not establish equal structural diversity. These are observations from [the recorded sampler sweep](</Users/ramiismael/Documents/code/vgc-team-generator-pilot/results/schemes_results.json:1>), not new experiments.

[[List of Experiments]] records nature/move contradictions and a small replicated rule-repair benefit. That is a real historical connection for ProSeCo, but does not establish a remaining post-repair failure or superiority of learned correction. The old 0.540 number is a copy-paste proposer win rate, not a diversity-collapse threshold.

## Revised decisions for every accepted paper

List IDs remain stable. Source links point to the exact manuscript versions read. The original workshop title for #174 differs from its author manuscript, *Calibrating Generative Models to Feature Distributions with MMD Finetuning*.

| List ID / workshop # | Paper | Earlier | Current | Concrete reason |
| --- | --- | --- | --- | --- |
| 169 / 15 | [A Tale of Two Temperatures: Simple, Efficient, and Diverse Sampling from Diffusion Language Models](https://arxiv.org/abs/2604.09921v2) | HIGH | **KEEP MED** | Separate randomness in field order from token temperature; useful optional sampling comparison. |
| 170 / 83 | [Re-evaluating Confidence Remasking in Masked Diffusion Language Models](https://arxiv.org/abs/2606.12232v1) | HIGH | **KEEP MED** | A useful audit of remasking gains, strong baselines, actual revisions and runtime. |
| 171 / 130 | [Learn from Your Mistakes: Self-Correcting Masked Diffusion Models](https://arxiv.org/abs/2602.11590v3) | HIGH | **DEFER** | Learned correction is legitimate, but no established benefit beyond current repair or for the active data comparison. |
| 172 / 132 | [Breaking the Factorization Barrier in Diffusion Language Models](https://arxiv.org/abs/2603.00045v3) | HIGH | **DEFER** | A joint head chiefly addresses simultaneous field generation, which the current sampler does not do. |
| 173 / 175 | [Internal Data Repetition Destroys Language Models](https://arxiv.org/abs/2606.24998v1) | MED | **KEEP MED** | Useful controls for repeated exposure and concentrated team lineages in training-data comparisons. |
| 174 / 26 | [Finetuning Generative Models to Match Feature Distributions](https://arxiv.org/abs/2606.19496v1) | MED | **KEEP MED** | Feature-distribution matching gives a concrete possible anchor, with important feature and diversity caveats. |
| 175 / 161 | [The Confidence Shortcut: A Reasoning Failure Mode of Masked Diffusion Models](https://arxiv.org/abs/2605.29123v1) | MED | **DEFER** | Relevant if confidence-aligned training becomes active; the current model does not use it. |
| 176 / 84 | [Uniform Diffusion Models Revisited: Leave-One-Out Denoiser and Absorbing State Reformulation](https://arxiv.org/abs/2605.22765v1) | MED | **DEFER** | A uniform-corruption objective issue does not establish a defect in the current mask-only model. |
| 177 / 188 | [Tensor-Train Joint Modeling for Few-Step Discrete Diffusion](https://arxiv.org/abs/2607.03788v2) | MED | **DEFER** | Low-rank joint sampling is a future parallel-decoding option, with ordering and rank assumptions. |
| 178 / 55 | [Latent-Augmented Discrete Diffusion Models](https://arxiv.org/abs/2510.18114v3) | MED | **DEFER** | A latent channel adds substantial modeling requirements before a demonstrated current need. |
| 179 / 158 | [Learned Relay Representations for Forward-Thinking Discrete Diffusion Models](https://arxiv.org/abs/2605.22967v3) | MED | **DEFER** | Compatible with sequential generation, but a persistent-state benefit has not been established here. |
| 180 / 31 | [Recursive Scaling in Masked Diffusion Models](https://arxiv.org/abs/2606.18022v1) | MED | **DEFER** | A future capacity-versus-compute study, not evidence about the current data or CEM questions. |
| 181 / 170 | [Understanding and Accelerating the Training of Masked Diffusion Language Models](https://arxiv.org/abs/2605.13026v2) | MED | **KEEP MED** | Mask-context allocation and irreducible loss matter when interpreting training-data comparisons. |
| 182 / 64 | [Time-Annealed Perturbation Sampling: Diverse Generation for Diffusion Language Models](https://arxiv.org/abs/2601.22629v2) | MED | **DROP** | Weak categorical-conditioning transfer, plus filtering and reporting issues undermine the proposed payoff. |
| 183 / 124 | [Hacking Generative Perplexity: Why Unconditional Text Evaluation Needs Distributional Metrics](https://arxiv.org/abs/2606.08417v2) | MED | **KEEP MED** | Directly useful when choosing diversity metrics: healthy marginals can conceal poor joint samples. |
| 184 / 137 | [DUEL: Exact Likelihood for Masked Diffusion via Deterministic Unmasking](https://arxiv.org/abs/2603.01367v2) | MED | **DROP** | Exact likelihood for a specified reveal schedule is not a current project requirement. |
| 185 / 183 | [TUBE: Tangent Upper Bound on Evidence for Discrete Diffusion Language Models](https://arxiv.org/abs/2605.24292v1) | MED | **DROP** | Sophisticated likelihood bounds do not answer the current team-quality or diversity decisions. |
| 186 / 190 | [Contrastive Distribution Matching for Amortized Sequential Monte Carlo in Discrete Diffusion](https://arxiv.org/abs/2605.23346v1) | MED | **DEFER** | Keep as a future reward-integration reference, requiring a suitable reward and trajectory model. |

## Useful reading order

1. **Understanding and Accelerating the Training of Masked Diffusion Language Models (#181)** and **Internal Data Repetition Destroys Language Models (#173)**: mask-context effects and exposure/concentration controls for the data experiment.
2. **Hacking Generative Perplexity (#183)**: a short metric-design reading. A practical inference is to test a proposed diversity score against a pool cycling a few team templates while maintaining high marginal species/item entropy.
3. **Feature-distribution matching / kCGM (#174)**: a possible selective anchor after identifying what diversity is actually being lost. Its regularizer is on trajectories; final rejection/canonicalization needs separate accounting.
4. **Two Temperatures (#169)** and **Re-evaluating Confidence Remasking (#170)**: optional sampling-comparison readings. These do not justify replacing the current decoder before measuring its quality/diversity trade-offs.

None directly settles whether mutations should be added as positive training examples, picks the best CFG strength, or proves that an anchor improves this CEM loop. Those remain empirical project questions.

## Individual full-paper assessments

### 169 — A Tale of Two Temperatures: Simple, Efficient, and Diverse Sampling from Diffusion Language Models

**KEEP MED** · [Full manuscript read](https://arxiv.org/abs/2604.09921v2)

TLC randomizes which masked position is committed; TCT softens confidence thresholds. These are separate controls from token-value temperature, although their effects interact. The semantic-fork theory needs specific confidence-gap assumptions; it is not a general guarantee of diversity. Experiments expose a real trade-off: on LLaDA HumanEval, TLC improves pass@64 while reducing pass@1 substantially (§§3–5; Appendices B–E).

Your fixed dependency order does not have the paper’s confidence-order mechanism. The reading still supplies a distinct comparison if structural diversity proves inadequate: field-order randomness versus token temperature, assessed alongside win rate and actual runtime. Feasibility dependencies restrict allowable orders. It does not answer CFG strength, CEM diversity or the mutation-data question. **HIGH was overstated; keep as an optional MED sampling reading.**

### 170 — Re-evaluating Confidence Remasking in Masked Diffusion Language Models

**KEEP MED** · [Full manuscript read](https://arxiv.org/abs/2606.12232v1)

WINO approximates leave-one-out confidence using shadow tokens, then revisits low-confidence commitments. The reassessment compares tuned unmaskers, block sizes, stochastic sampling, network calls and throughput. Most revisited tokens return unchanged; stronger small-block baselines erase many apparent gains. Learned unmasking plus WINO nevertheless provides a positive counterexample to ‘remasking never helps’ (§§2–4; Appendices A–B).

For your historical remasking experiments, the useful lesson is to measure actual changed-field fraction, accepted-team throughput and quality/diversity against a strong comparator. Pass@k is only indirect diversity evidence and does not measure team structure. Your current fixed-order sampler has no shadow-block mechanism. This earns **MED diagnostic reading**, not a recommendation to add remasking or evidence that it will preserve CEM diversity.

### 171 — Learn from Your Mistakes: Self-Correcting Masked Diffusion Models

**DEFER** · [Full manuscript read](https://arxiv.org/abs/2602.11590v3)

ProSeCo trains recovery from the denoiser’s own predictions, including editing committed values. Clean training targets do not inherently forbid novelty: ordinary denoising uses clean targets too, and this paper evaluates novel generation. Its ideal stationary-kernel argument does not guarantee that practical argmax corruption, tied networks and finite correction steps preserve diversity (§§3–5; Appendices A–D).

Your historical experiments measured nature/move contradictions and a small benefit from rule repair. That is a plausible connection, but does not establish a remaining post-repair problem or that learned recovery would solve it. Reconstruction supervision does not identify strategically weak teams and does not decide whether mutations should be positive examples. **Defer until an error category beyond existing repair warrants learned correction.** A fair comparison would separate extra training, correction at sampling and existing rules. My earlier HIGH grade was unjustified; my earlier explanation that restoring clean examples necessarily suppresses novelty was also wrong.

### 172 — Breaking the Factorization Barrier in Diffusion Language Models

**DEFER** · [Full manuscript read](https://arxiv.org/abs/2603.00045v3)

CoDD combines neural potentials with a tractable probabilistic-circuit prior. The tested static prior uses an HMM; its full-diffusion joint branch applies when multiple positions are selected. Gains at small step budgets come with overhead, and the static prior can hurt at high noise. Open-ended results also show lower reported diversity and more repetition despite improved reference-model perplexity (§§4–5; Appendices C–E).

Your transformer runs again after each field is sampled, so later choices already condition on earlier ones. Separate categorical heads do not make that whole-team sampling distribution independent. ‘Teams have dependencies’ therefore did not justify HIGH. **Reopen if generating several fields per pass becomes a deliberate speed objective** and measured joint-sampling errors justify extra machinery. Heterogeneous vocabularies and feasibility constraints need their own design.

### 173 — Internal Data Repetition Destroys Language Models

**KEEP MED** · [Full manuscript read](https://arxiv.org/abs/2606.24998v1)

The experiments keep compute and token budget fixed while allocating 10% of tokens to an exactly repeated document pool. They vary its size/repetition, finding a nonmonotone degradation pattern. The headline compute-equivalent loss comes from a fitted scaling curve; each experimental cell has one training seed. The linear-model appendix illustrates a mechanism rather than proving general CEM collapse (§§2–4; Appendices A–J).

This directly sharpens your real-only versus real-plus-mutations comparison: record distinct canonical teams, mutation lineages and repeated exposure, rather than counting rows alone. Elite multiplicity can be intentional; silently deduplicating changes its weight. Mutations are not necessarily exact duplicates, and replaying original data is not automatically harmful. Your current refit restarts from p0, so this does not diagnose recursive weight degradation. **Keep MED for experiment design**, without importing text-specific thresholds or claiming it selects an anchor fraction.

### 174 — Finetuning Generative Models to Match Feature Distributions

**KEEP MED** · [Full manuscript read](https://arxiv.org/abs/2606.19496v1)

The author manuscript formulates kernel-based feature-distribution matching with a KL penalty on sampling trajectories relative to the pretrained generator, using score-function estimators for nondifferentiable/discrete generation. Coupled MMD terms compare whole feature distributions rather than just a target mean. The discrete experiments provide a more concrete transfer than a generic continuous-optimization analogy (method and experimental sections; Appendices A–E).

For your possible response to measured CEM diversity loss, it distinguishes matching selected strategic-feature distributions from replaying original teams or merely restarting at p0. Feature choice determines what is preserved: marginal species frequencies need not preserve combinations, and preserving corpus features need not maximize battle strength. Appendix A.5 shows that reducing self-repulsion below its distribution-matching setting can sharpen categorical targets and eliminate rare categories. **Keep MED as an anchoring concept**, not a proven diversity safeguard or a reason to add regularization before measuring collapse.

### 175 — The Confidence Shortcut: A Reasoning Failure Mode of Masked Diffusion Models

**DEFER** · [Full manuscript read](https://arxiv.org/abs/2605.29123v1)

The paper compares random-mask training with confidence-weighted/aligned objectives on synthetic reasoning tasks. Rare dependency patterns reveal confident failures, but Sudoku gives an important counterexample: confidence-aligned training improves cell accuracy. Those are not whole-puzzle solved rates. Task-informed oracle orders and supplied partial answers are diagnostic interventions, not generally available decoding procedures (§§3–5; Appendices A–E).

Your current training does not use these confidence-aligned objectives, and the decoder follows an explicit dependency order. The historical confidence/dependency win-rate tie does not establish this failure mechanism. **Defer until such training or a reasoning-policy model is under consideration.** Then inspect rare dependency cases rather than only average accuracy. The earlier MED rationale transferred ‘dependency difficulty’ without identifying a current decision.

### 176 — Uniform Diffusion Models Revisited: Leave-One-Out Denoiser and Absorbing State Reformulation

**DEFER** · [Full manuscript read](https://arxiv.org/abs/2605.22765v1)

With its particular nonlinear uniform-diffusion bridge, the ELBO targets a leave-one-out posterior while ordinary clean cross-entropy targets the denoising posterior. An affine bridge changes that conclusion: it is not a universal defect of categorical diffusion. The paper derives posterior conversions and absorbing-state lifts, with language and Sudoku experiments (§§3–6; Appendices A–I).

Your model uses mask-only corruption; at masked positions the relevant posterior distinction disappears. No uniform-replacement path or matching objective bug was found. **Defer until uniform corruption or revisable visible tokens becomes a real design candidate.** Its ideal lifting identities need exact-model assumptions, and practical parallel corrections are heuristic. The mask-context training paper addresses the nearer training question. This is a technically substantial paper whose particular contribution is outside your present setup.

### 177 — Tensor-Train Joint Modeling for Few-Step Discrete Diffusion

**DEFER** · [Full manuscript read](https://arxiv.org/abs/2607.03788v2)

Normalized tensor-train heads represent dependencies among multiple draws made from one transformer output. Their contractions can sample conditionally without rerunning that transformer. The low-rank expressivity example relies on favorable dependency structure; arbitrary cross-team dependencies may require larger ranks. The smallest advertised overhead is one setting, not a universal cost. Improved generative perplexity sometimes accompanies reduced entropy (§§4–5; Appendices A–E).

Your current sampler pays for a fresh contextual pass for each field, avoiding the simultaneous-independent-draw issue. Team member permutation and heterogeneous fields also complicate a fixed tensor chain. **Defer until fewer transformer passes becomes an explicit goal**, then compare rank/order choices, actual runtime, legality and multivariate diversity. This overlaps CoDD and LADD; none is required just to establish that teammate interactions exist.

### 178 — Latent-Augmented Discrete Diffusion Models

**DEFER** · [Full manuscript read](https://arxiv.org/abs/2510.18114v3)

LADD generates an auxiliary latent channel intended to make remaining token predictions more independent. The latent must itself be modeled. Its synthetic experiment usefully shows why lower token reconstruction loss need not mean an easier full generative problem. Text experiments add a pretrained encoder and extra training cost; validation token likelihood excludes the latent-channel term, and marginal token entropy does not establish equal joint diversity (§§2–5; Appendices B–E).

Your sequential sampler lacks the main simultaneous-factorization bottleneck, and there is no independently validated latent team representation. Latent dropout is not evidence about your proposed CFG sweep. **Defer until parallel decoding or a compact team representation has an independent justification.** This would add an encoder, latent generator and schedules before resolving the active data/CFG/diversity questions. Retired latent-space search does not provide a new reason to prioritize it.

### 179 — Learned Relay Representations for Forward-Thinking Discrete Diffusion Models

**DEFER** · [Full manuscript read](https://arxiv.org/abs/2605.22967v3)

RELAY carries hidden state between denoising calls and trains short rollouts with truncated backpropagation. It can apply to sequential decoding. Sudoku ablations distinguish rollout exposure, persistent state and gradient flow, but rollout-only controls remain competitive on some language tasks. The headline reduction is in network evaluations; it should not be repeated as an established equal wall-clock latency saving (§§3–6; Appendices A–B).

A fresh forward pass is not by itself evidence that your model loses necessary reasoning state: the updated partial team remains visible. **Defer until measurements identify a computation/state limitation** beyond data mixture and ordinary capacity. Then compare rollout-only and detached-state controls with true runtime and team diversity. Its earlier MED explanation described a mechanism without establishing why your project needed it.

### 180 — Recursive Scaling in Masked Diffusion Models

**DEFER** · [Full manuscript read](https://arxiv.org/abs/2606.18022v1)

Shared transformer blocks recur several times within a denoising call. Dense supervision and training schedules matter; extra test-time recursion is not uniformly helpful. Strong structured-puzzle results coexist with weaker Text8 outcomes. Parameter matching is not compute matching, and large Sudoku constructions use transformations of a common base grid rather than demonstrating unrestricted combinatorial generalization (main experiments; Appendices B–F).

This recursion is neither the CEM refit loop nor editing already generated fields. Historical width scaling makes capacity a legitimate future question, but does not establish that recursive depth is preferable. Each of your 48 field decisions would pay for inner iterations unless decoding also changes. **Defer until a fixed memory/parameter budget motivates the trade-off**, using ordinary depth/width and actual training/inference cost as controls.

### 181 — Understanding and Accelerating the Training of Masked Diffusion Language Models

**KEEP MED** · [Full manuscript read](https://arxiv.org/abs/2605.13026v2)

The paper separates irreducible conditional entropy from remaining learnable error. Its faster-training recipe changes mask-probability sampling, removes 1/t loss weighting and uses stratified sampling. Merely changing the mask schedule in your existing weighted loss is not that recipe. The unchanged-optimum argument assumes infinite capacity and a time-agnostic parameterization; language locality and mask-run proxies do not establish team-learning speedups (§§3–5; Appendices A–C).

Your model has 48 heterogeneous fields and shuffled teammate records. The immediate lesson is to inspect learning by observed-context count and field type before attributing loss differences to real versus mutated data. High loss with little context may contain irreducible uncertainty. A later ablation must separate sampling, weighting, normalization and compute. **Keep MED for training interpretation**, without assuming the paper’s preferred mask distribution or text NLL speedups transfer to win rate.

### 182 — Time-Annealed Perturbation Sampling: Diverse Generation for Diffusion Language Models

**DROP** · [Full manuscript read](https://arxiv.org/abs/2601.22629v2)

TAPS adds annealed Gaussian noise to prompt embeddings while preserving selected embedding statistics. This is a particular conditioning perturbation, not token-temperature control. The evaluation keeps longest valid outputs and excludes prompts with too few valid samples, complicating raw quality/diversity interpretation. EAD is described as an embedding metric although its formula measures n-gram occupancy; the reported preference confidence interval includes parity despite stronger main-text wording (§3; Appendices A–D).

A meaningful perturbation geometry has not been established for your categorical team conditioning, and changing the condition can change the target rather than produce useful alternatives. The printed algorithm also leaves details requiring author-code clarification. **Do not retain as a current reading recommendation.** These issues do not prove the method ineffective, but its speculative transfer and uncertain quality-preservation evidence do not justify the earlier MED slot.

### 183 — Hacking Generative Perplexity: Why Unconditional Text Evaluation Needs Distributional Metrics

**KEEP MED** · [Full manuscript read](https://arxiv.org/abs/2606.08417v2)

Deliberately incoherent generators can obtain favorable frozen-model generative perplexity while retaining substantial token entropy. The paper constructs counterexamples and compares distributional evaluations, making a specific failure visible rather than merely asserting that metrics can be gamed (main counterexamples and experiments; substantive appendices).

For your CEM diversity plot, species/item frequencies or exact uniqueness alone can miss loss of meaningful combinations. Pair structural diversity and copy/concentration measurements with actual battle outcomes on comparable samples. Preserve duplicate draws as your plan specifies. Corpus distributional similarity is a diagnostic, not the optimization objective: a useful counter-team may deliberately depart from real-team frequencies. Nor should text MAUVE be copied without a justified team representation. **Keep MED for metric design**, not as a claim that your current scorer has been hacked or that diversity substitutes for strength.

### 184 — DUEL: Exact Likelihood for Masked Diffusion via Deterministic Unmasking

**DROP** · [Full manuscript read](https://arxiv.org/abs/2603.01367v2)

DUEL evaluates exact likelihood under a specified deterministic unmasking procedure. Such a procedure defines its own normalized model; likelihood does not transfer unchanged across alternative reveal schedules. A reported oracle permutation uses the full target block and cannot be treated as an executable target-independent sampler beating an autoregressive model (main likelihood construction; Appendix C and experimental controls).

Your fixed-order one-field sampler already permits a chain-rule calculation for its raw field draws if ever needed. Final feasibility masks, random copied spreads, validation rejection and canonical team probabilities need additional accounting. No likelihood-ranking requirement was found in the active project; win rate and diversity are the target measurements. **Drop from the current queue for relevance and redundancy**, not because its basic deterministic-likelihood result is invalid. Reopen only for an explicit likelihood-evaluation question.

### 185 — TUBE: Tangent Upper Bound on Evidence for Discrete Diffusion Language Models

**DROP** · [Full manuscript read](https://arxiv.org/abs/2605.24292v1)

TUBE introduces a variational upper bound to complement likelihood lower bounds, using a suitable surrogate and Monte Carlo over reveal orders. An unbiased estimator of that upper bound is unbiased in expectation; a finite realization is not automatically a certified upper bound or confidence interval. Tightness depends on the surrogate, and exact-control experiments use small blocks (main derivation and experiments; Appendices A–D).

This matters when comparing likelihoods of diffusion models with latent schedules. Your current fixed deterministic decoder has no demonstrated need for that machinery, and likelihood is not team strength or useful diversity. **Drop from the current queue.** A concrete future likelihood-estimation study could reopen it, but generic evaluation rigor was too weak a reason for the earlier MED recommendation.

### 186 — Contrastive Distribution Matching for Amortized Sequential Monte Carlo in Discrete Diffusion

**DEFER** · [Full manuscript read](https://arxiv.org/abs/2605.23346v1)

CDM learns distribution-matching/twist components for amortized sequential Monte Carlo using positive completed samples collected near the reward-tilted target by SMC and negative samples from the current proposal. This offers a way to learn partial-state preferences without directly evaluating reward on every partial object. Positive-sample collection still costs work, and sparse/noisy battle outcomes do not automatically fit the tested setting (method, experiments and Appendix A).

The ideal forward-buffer identity additionally needs consistency between the model-induced reverse joint and the claimed forward corruption; that cannot be assumed for your fixed dependency-order sampler versus random-mask training. **Remain deferred with the reward subsection**, as originally intended. Revisit when the reward, positive-data budget, noise treatment and trajectory process are specified. It does not resolve the immediate real-data/mutations comparison or establish a free replacement for battle evaluation.

## Full-reading coverage receipt

PDF counts include reference pages; those bibliographies were not followed citation by citation. HTML sources are recorded by their complete section coverage instead of an invented page count. All substantive appendices include their proof arguments, tables, experimental details and qualitative examples. Figures were visually checked; values were not independently digitized or measurements reproduced.

| List ID | Version read | PDF pages / format | Complete text coverage |
| --- | --- | --- | --- |
| 169 | [2604.09921v2](https://arxiv.org/abs/2604.09921v2) | HTML | §§1–6; A–E, including all algorithms, C.1–C.5 proofs and D–E experiments |
| 170 | [2606.12232v1](https://arxiv.org/abs/2606.12232v1) | HTML + PDF figures | §§1–5 and limitations; A–B, including both remasking ablations |
| 171 | [2602.11590v3](https://arxiv.org/abs/2602.11590v3) | 30 | §§1–7; A–H, all proofs, sampling algorithms, experiments and examples |
| 172 | [2603.00045v3](https://arxiv.org/abs/2603.00045v3) | 17 | Entire main text; A–F, including normalization proof, sampling and examples |
| 173 | [2606.24998v1](https://arxiv.org/abs/2606.24998v1) | HTML | §§1–5; A–J, including linear-model proof, scaling calibration and all experiments |
| 174 | [2606.19496v1](https://arxiv.org/abs/2606.19496v1) | 31 | §§1–4; A.1–A.5, B, C.1–C.5, D.1–D.2, E.1–E.2 |
| 175 | [2605.29123v1](https://arxiv.org/abs/2605.29123v1) | HTML | §§1–6; A–E, all task construction, solver details and supplementary analyses |
| 176 | [2605.22765v1](https://arxiv.org/abs/2605.22765v1) | 47 | §§1–7; A–I, all posterior/lifting proofs, samplers, architecture and experiments |
| 177 | [2607.03788v2](https://arxiv.org/abs/2607.03788v2) | 17 | §§1–6; A–E, all normalization/rank/RoPE proofs and results |
| 178 | [2510.18114v3](https://arxiv.org/abs/2510.18114v3) | 35 | §§1–6; A–E, including all bounds, latent training details and generated examples |
| 179 | [2605.22967v3](https://arxiv.org/abs/2605.22967v3) | 18 | §§1–7; A–B, all filtering, adaptation and memory-profile details |
| 180 | [2606.18022v1](https://arxiv.org/abs/2606.18022v1) | 30 | Entire main text; A–F, all curricula, ablations, extended tables and examples |
| 181 | [2605.13026v2](https://arxiv.org/abs/2605.13026v2) | HTML | §§1–6; A–F, including B.1–B.3 proofs and C.1–C.5 experimental controls |
| 182 | [2601.22629v2](https://arxiv.org/abs/2601.22629v2) | HTML | §§1–5; A–F, implementation, evaluation, algorithm, ablations and both case studies |
| 183 | [2606.08417v2](https://arxiv.org/abs/2606.08417v2) | 15 | §§1–5; entire A, Tables 3–6 and every generated/reference example |
| 184 | [2603.01367v2](https://arxiv.org/abs/2603.01367v2) | 22 | §§1–7; A–D, including all C.1–C.4 proofs and evaluation details |
| 185 | [2605.24292v1](https://arxiv.org/abs/2605.24292v1) | 20 | §§1–6; A–D, all estimator proofs, alternative bounds and runtime details |
| 186 | [2605.23346v1](https://arxiv.org/abs/2605.23346v1) | 33 | §§1–7; impacts/limitations; A–F, all proofs, algorithms, ablations and examples |

## Boundaries of the conclusions

These are project-specific reading judgments based on primary manuscripts and the inspected code/notes. The technical results were critically read but not independently reproduced, and none was tested on Pokémon teams during this review. Scientific caveats matter where they affect the proposed transfer; they should not be inflated into blanket rejection of a paper.

The initial 211-paper screening remains available at [[SPIGM 2026 — paper-by-paper relevance screening]]. This follow-up revisits exactly its 18 accepted papers; it does not claim full reads of the other 193 entries.
