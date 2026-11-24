# Uniform discrete diffusion comparison: does uniform discrete diffusion make guidance cost less novelty and diversity than the default model?

Pre-registration, written 2026-11-24 before any uniform discrete diffusion training, sampling, or battle starts. Ticket: [Pre-register the uniform discrete diffusion comparison, with its predicted outcome and port-equality test](https://github.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/issues/274), on the map [Uniform discrete diffusion map: does uniform discrete diffusion make guidance cost less novelty and diversity than the default model?](https://github.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/issues/270).

This experiment is a clean **two-arm comparison** between the project's default absorbing masked discrete diffusion model (**Masked, full filter**) and a faithful port of **UDLM** (Schiff et al., ICLR 2025, *Simple Guidance Mechanisms for Discrete Diffusion Models*, [arXiv:2412.10193v3](https://arxiv.org/pdf/2412.10193v3), upstream `kuleshov-group/discrete-diffusion-guidance@edb0f8c28b7caeb4ea7a06a2fee8d74ab6da1661`): **Uniform, no filter**. Only **one** new model trains (UDLM), using the paper's default baseline schedule `α(t) = 1 − t` (corruption probability `1 − α(t) = t` on all 84 fields, matching the masked control's linear schedule) and unconstrained sampling with Showdown's validator checking legality afterward—matching the paper's QM9 molecule evaluation with no custom heuristics added.

Mode: **exploratory**, because one training seed per checkpoint cannot separate the noise process from training luck, which has reached `0.0019` in ask-versus-got error and `0.026` in pooled win rate between two trainings of one recipe (`docs/position-embedding-split.md`). Within-checkpoint comparisons on the UDLM checkpoint—pass count (#277) and the guidance curve (#279)—are **confirmatory** for that checkpoint, because their plan, metrics, usability gate, and bend-point rules are fixed here before any run.

## Terms

An **ask** is the win rate the model is told to aim for (`k/24` for `k = 1` to `23`). **Got** is the win rate its teams then score. A **line-up** is a team's unordered set of six Pokémon species. The **ask-versus-got error** is the mean squared error between each ask and the mean got at that ask, scored on asks `1/24` to `17/24` (`src/ask_line.py`); asks `18/24` to `23/24` are reported apart because the training table holds almost nothing there. Near the line means an error under `0.001`.

**Win rate** means one battle against each of the 49 distinct top-50 meta teams, with the behaviour-cloning policy on both sides and slots shuffled every battle. The **top-50 meta teams** are the project's fixed list of real Regulation M-B teams (one duplicate leaves 49 distinct opponents). The **behaviour-cloning policy** is the battle player trained to imitate human battle decisions.

A **masked diffusion model** corrupts each field toward an absorbing `[MASK]` token; its reverse sampler fills one masked field per pass in species-first order (84 passes) and never changes a filled field. A **uniform discrete diffusion model** (UDLM, Schiff et al. 2024) has no `[MASK]` token: forward corruption replaces each field with a random token drawn uniformly from that field's real vocabulary. Sampling starts from a fully random team, runs a fixed number of reverse **passes** (one full network evaluation of all 84 fields), and may change any field on any pass.

The **masking schedule** is the probability `1 − α(t)` that the forward process has corrupted a field by time `t ∈ [0, 1]` during training: replacing it with `[MASK]` in the masked model, or with a random real token in the uniform model. Schiff et al. call `σ(t) = −log(1 − t)` **log-linear** noise, which gives clean survival `α(t) = exp(−σ(t)) = 1 − t` and corruption probability `1 − α(t) = t` on every field—the exact same linear masking schedule used by the masked control.

**Stat Alignment** is a Pokémon's nature (such as Adamant). **Stat Points** are the six stat allocations per Pokémon (0 to 32 per stat, at most 66 in total). The **legal Pokémon construction filter** zeroes the probability of values that Regulation M-B rules forbid next to the team's other fields (`lossdown.Tables` plus `stat_points.allowed_stat_points`). **First-try legality** is the share of sampled teams that Pokémon Showdown's Regulation M-B validator accepts before any redraw. **Species replacement distance** between two six-species line-ups is `6 − shared species` (forms distinct, order and sets ignored). **Novelty** is the share of generated teams whose six-species line-up appears in none of the labelled training teams. **Guidance** is the classifier-free guidance weight: `1` is conditional with no extra push, `0` is unguided (no ask), and higher weights push harder toward the ask.

## What is already known

Searches across `code/docs/`, `code/docs/experiment-names.md`, `notes/List of Experiments.md`, the closed research tickets of map #270 (#271–#273), and the web (2026-11-24) found no existing uniform discrete diffusion experiment in this repository and no published comparison of uniform against masked discrete diffusion on structured game-team generation or equal-error novelty/diversity frontiers.

### Schedules used in published uniform discrete diffusion papers

Every published uniform discrete diffusion paper uses a single schedule shared across all positions (none uses per-field schedules):

| Paper | Schedule | Formula | Shared across positions? |
| --- | --- | --- | --- |
| Multinomial Diffusion (Hoogeboom et al., NeurIPS 2021, [arXiv:2102.05379](https://arxiv.org/abs/2102.05379)) | Cosine | `ᾱ_t = f(t)/f(0)`, `f(t) = cos(((t/T + s)/(1 + s)) · π/2)²` | Yes |
| D3PM-Uniform (Austin et al., NeurIPS 2021, [arXiv:2107.03006](https://arxiv.org/abs/2107.03006)) | Cosine & mutual-information | Cosine `f(t)` (images); `I(x_t; x_0) ≈ (1 − t/T) H(x_0)` (text) | Yes |
| SEDD-Uniform (Lou et al., ICML 2024, [arXiv:2310.16834](https://arxiv.org/abs/2310.16834), `noise_lib.py:41–54`) | Geometric | `σ(t) = σ_min^(1−t) · σ_max^t` (`σ_min = 1e-3, σ_max = 1`) | Yes |
| **UDLM** (Schiff et al., ICLR 2025, [arXiv:2412.10193v3](https://arxiv.org/pdf/2412.10193v3), `diffusion.py:563–614`) | **Log-linear** (default baseline used here) | `σ(t) = −log(1 − (1 − ε)t)` → `α(t) = 1 − t`, `1 − α(t) = t` | Yes |
| Duo (Sahoo et al., ICML 2025, [arXiv:2506.10892](https://arxiv.org/abs/2506.10892), `algo.py:375–405`) | Gaussian-duality integral | `α(t) = Γ(γ(t))` mapped from Gaussian log-SNR (plus log-linear baseline) | Yes |
| GIDD (von Rütte et al., ICML 2025, [arXiv:2503.04482](https://arxiv.org/abs/2503.04482), `diffusion_process.py:54–105`) | Hybrid linear-mask + uniform mixing | `α_t = (1 − t)/(1 + c_t)`, `c_t = B · t^(γ/2)(1 − t)^(γ/2)` | Yes |

### Prior project results that bear on this run

| Earlier result | What it established | What it leaves open |
| --- | --- | --- |
| Upstream UDLM reading (#271, `research/uniform-noise-udlm-sampling`) | Continuous-time uniform loss (paper Eq. 19, `diffusion.py:563–614`) is built for UDLM's baseline log-linear schedule (`α(t) = 1 − t`). Classifier-free guidance mixes reverse posteriors (`diffusion.py:1314–1334`), not clean logits. Sampling runs unconstrained for a fixed pass count on `linspace(1, 1e-5, K+1)` with no early stop and no clean-up pass; validity is checked afterward. | How paper-faithful UDLM behaves on 84-field heterogeneous-vocabulary teams compared with the masked default model. |
| Revision-count and repeat-revision research (#272, #273, `research/uniform-diffusion-token-revisions`) | No standard revision count exists in the literature. Uniform reverse transitions cause repeated state changes even with a fixed, certain predictor: `4.404443` expected changes beyond the first per field at 128 passes, `N = 189`, ideal `α(t) = 1 − t`. | How often a trained UDLM actually changes its top prediction (**changes of mind**) and how that tracks the guidance trade-off. |
| Masking-schedule rows and no-guidance line (#297, #298, #73, `docs/masking-rate-rows.md`, `docs/no-guidance-line.md`) | On the shuffled-only table (`stat84_shuffled_noise_linear_es3_cap80`, species-first), ask-versus-got error falls from `0.0162` at guidance `1` to `0.0075`, `0.0049`, and `0.0038` at guidance `2, 4, 8`. Pooled distinct line-ups across 23 asks *rise* with guidance as asks stop overlapping, while **within-ask** line-ups (64 teams per ask × seed) and within-ask species replacement distance *fall* (`57.6 → 53.9` line-ups per 64 teams; `4.200 → 3.524` species replacements). | Whether uniform discrete diffusion's repeated editing and posterior-level guidance flatten that within-ask diversity/novelty drop at equal error. |
| No tables in the loss (#361, #367, closed 2026-11-24) | Training the 6-layer, 256-wide masked network for 80 epochs on the filtered shuffled-only table (`214,055` train / `4,368` val, `407.7` min) **without** the legal Pokémon construction filter in the loss learned slot-to-Pokémon binding between epochs 45 and 48: illegal abilities fell `82% → 0%`, illegal moves `58% → 0%`, and first-try legality **without any sampling filter** rose from `0%` to `67%` (33% illegal). | Whether UDLM (which also trains and samples without the tables, exactly as in Schiff et al. 2024) learns enough legality by epoch 80 for post-sampling validator redraws to work cleanly across guidance strengths. |

## Arms

Named with `.claude/skills/name-experiments/SKILL.md` (difference first, shared settings moved to the caption).

**Shared by both arms:** 84-field transformer (256 wide, 6 layers, 8 heads, dropout `0.1`), filtered shuffled-only labelled training table (`214,055` train / `4,368` validation), corruption probability `1 − α(t) = t` on all 84 fields (linear masking schedule for masked; UDLM's baseline log-linear schedule for uniform), ask dropout `0.15`, label noise `±1/48`, AdamW (`lr 4e-4`, weight decay `0.01`, 500 warm-up steps, cosine decay, batch `256`, gradient clip `1.0`, EMA `0.999`), training seed `0`, early stopping on validation loss (patience `3`, threshold `0`, cap `80` epochs, every epoch saved), Stat Alignment repair after sampling, and the same shuffled-slot battle scorer.

| Noise process, then sampling filter | Code ID | Checkpoint | Legal Pokémon construction filter |
| --- | --- | --- | --- |
| Masked, full filter. The control | `linear_masked_filter` | Existing masked linear default model (clean retrain in #385; provisional reference `stat84_shuffled_noise_linear_es3_cap80` until #385 lands; no masked retraining inside this map) | Full bidirectional tables + Stat Point budget in loss and sampling, species-first, 84 passes |
| Uniform, no filter | `udlm_loglinear_nofilter` | `stat84_udlm_loglinear_es3_cap80` (the **only** new training in this map, `udlm_loglinear`) | None in loss or sampling; Showdown validator checks complete teams after pass `K` and redraws rejects (exact UDLM paper setup) |

Only **one** model trains inside this map (`stat84_udlm_loglinear_es3_cap80`). Note that the two arms differ in both the noise process (absorbing `[MASK]` vs uniform random real token) and the presence of the legal Pokémon construction filter (which the masked default model already carries and which UDLM leaves out as published); keeping the arm names explicit preserves that distinction.

## Contract

| Row | Value |
| --- | --- |
| Claim | Paper-faithful uniform discrete diffusion (**Uniform, no filter**, trained with UDLM's baseline schedule `α(t) = 1 − t` and posterior-level CFG, checked by Showdown's validator after sampling) reduces the novelty and diversity cost of guidance relative to **Masked, full filter** at equal ask-versus-got error, yielding a larger dominated area on the novelty and within-ask diversity frontiers and/or a higher usable guidance weight. |
| What refutes it | **Uniform, no filter** fails to achieve a larger dominated area than **Masked, full filter** on novelty, within-ask line-up uniqueness, or within-ask mean species replacement distance with all 3 sampling seeds agreeing in sign and the 95% bootstrap interval excluding zero; or it fails the usability gate (e.g., first-try legality drops below 5%) before matching the control's error. |
| What changes if it holds | **Uniform, no filter** becomes a candidate for a second training seed and for the project's next guided team-generation model (and motivates testing remasking on the masked default model, currently held under map #270's Not yet specified). If refuted, masked diffusion remains the default architecture and the map's verdict records how much paper-faithful UDLM cost on each frontier and why (from first-try legality and the revision numbers). |
| How far it reaches | Ask-versus-got error, novelty, diversity, legality, and win rate under the behaviour-cloning policy against the 49 distinct top-50 meta teams, slots shuffled, on one training seed per checkpoint. It does not claim generalization to human opponents, other battle policies, or held-out meta ranks 51–100 without a separate held-out evaluation. |
| Fields | All 84 fields are generated by the model: for each of six Pokémon, its species, ability, item, four moves, Stat Alignment, and six Stat Points. Nothing is copied or fixed. After sampling, `Model84.to_paste(..., repair=True)` replaces Stat Alignment to match the Pokémon's moves, identically in both arms (`src/ask_grid84.py`). |
| Vocabulary | Source: `team84.regulation_vocab()` (Regulation M-B snapshot stored in checkpoints). In the masked encoding, index `0` is `[MASK]`: totals are 348 species, 201 abilities, 150 items, 498 moves, 27 Stat Alignments, and 34 Stat Point tokens (0 = `[MASK]`, 1–33 = values 0–32). Uniform discrete diffusion has no `[MASK]` token, so its real vocabulary sizes `N_k` exclude index `0`: **347 species, 200 abilities, 149 items, 497 moves, 26 Stat Alignments, and 33 Stat Points** (values 0–32, mapped to `0..N_k − 1` inside the uniform loss/posterior and shifted by `+1` to/from the 84-field table encoding). Labelled training teams in the filtered shuffled-only table use `189, 127, 111, 316, 17, and 33` of these real values respectively (`src/uniform_preflight84.py`). |
| Training teams | Filtered shuffled-only table `lossdata84_shuffled_only.npz` (`sha256: e58af7689e62a873ce19361237d7ff8e77cb791fc4c196cc705410755c5a03e2`, after issue #368's wrong-stone drop). Labelled teams only: **214,055 training teams and 4,368 validation teams** (218,423 labelled total; seeded 2% validation split with seed 0 preserved across the wrong-stone filter; minimum 49 shuffled-slot battles per team). Excluded: 10 unlabelled teams, 173,544 teams whose labels came only from non-shuffled battles or run files, 555 wrong-Mega-Stone teams, the 500 test teams, and the 50 held-out meta teams (ranks 51–100). Verified 0 Showdown validator failures and 0 `Tables.rejects` failures on both train and val sides. |
| Scorer | `src/ask_grid84.py` (`matchup_db.score_pastes`): 1 battle against each of the 49 distinct top-50 meta teams per generated team, behaviour-cloning policy on both sides, slots shuffled (`require_shuffled` enforces `VGC_SLOT_ORDER=shuffled`). Each new grid gets its own name-hashed sampling seed base via `src/grid_seeds.py` (`100,000 + ...`, never legacy `7300`, preventing seed replay of training teams per #353), with cell sampling seeds `s ∈ {1, 2, 3}` and battle seed `seed0 + s`. |
| Search region | No search and no elite selection rule. Each battled ask grid covers asks `1/24` to `23/24` (23 asks; error scored on `1/24` to `17/24`, `18/24` to `23/24` reported apart), 3 sampling seeds (`1, 2, 3`), 64 validator-accepted non-held-out teams per cell, try cap `MAX_TRIES = 1,280` per cell (23 × 3 × 64 = 4,416 teams, 216,384 battles per grid). |

### Settings

Every setting names its role and either cites Schiff et al. (2024, `arXiv:2412.10193v3`, upstream `edb0f8c2`) or is marked **Ours**.

| Setting | Role | Value | Paper or ours |
| --- | --- | --- | --- |
| Diffusion noise process | Scientific | Absorbing `[MASK]` (**Masked, full filter**) vs uniform random real token (`q(z_t \| x_0) = Cat(α_t x_0 + (1 − α_t) u)`, no `[MASK]`) | Schiff et al. §2.1 Eq. 1, §4.1; `diffusion.py:473–476` |
| Legal Pokémon construction filter | Bundled with model family | Full bidirectional in loss and sampling (**Masked, full filter**, existing default) vs **none** in loss or sampling (**Uniform, no filter**, Showdown checks after sampling) | No filter (post-sampling check): Schiff et al. QM9. Full filter on control: **Ours** (audited in #292) |
| Guidance strength (`w` or `γ`) | Swept | Coarse grid `{1, 2, 4, 8}` per arm + unguided baseline (`γ = 0` / no ask), plus up to 2 bend/extension points per arm | Posterior CFG mixing: Schiff et al. §3.1 Eqs. 6–8 (`diffusion.py:1314–1334`). Grid & bend rule: **Ours** |
| Reverse pass count `K` | Nuisance (fixed before battles by #277) | Control: 84 (one field per pass). **Uniform, no filter**: smallest `K ∈ {84, 128, 256}` within seed noise of 256 at `γ = 1` on `linspace(1, 1e-5, K+1)` | Grid & no clean-up pass: Schiff et al. `configs/config.yaml:35–41`, `diffusion.py:1115–1209`. Selection rule: **Ours** |
| Sampling seed | Nuisance | 3 seeds (`1, 2, 3`) per arm and guidance strength, with per-grid `grid_seeds.seed0` base | **Ours** (#353 anti-replay rule) |
| Masking schedule (`1 − α(t)`) | Fixed (shared across both arms) | UDLM's baseline schedule `α(t) = 1 − t` (`1 − α(t) = t`, `α′(t) = −1`) on all 84 fields | Schiff et al. §4.2, App. D.3 (`diffusion.py:572–573`); identical corruption curve to the control's linear schedule (MD4 App. J.1; LLaDA Eq. 3) |
| Continuous-time loss | Follows noise process | Control: masked weighted cross-entropy (`diffusion84.elbo_terms`). **Uniform, no filter**: exact continuous-time Eq. 19 (`T = 0`, `zero_recon_loss = True`, `use_simple_ce_loss = False`), averaged over 84 fields | Schiff et al. §4.2 Eqs. 17–19 (`diffusion.py:563–614`) |
| Time sampling in training | Fixed | Stratified antithetic `t = (u + i/B) mod 1`, shuffled, clamped to `[1e-3, 1 − 1e-3]` so `α(t) > 0` and `1 − α(t) > 0` everywhere | `training.sampling_eps = 1e-3`: Schiff et al. `configs/config.yaml:43–47`, `diffusion.py:449–458`; upper clamp `1 − 1e-3` avoids `1 / α_t` singularity at `t = 1` |
| Time conditioning & caching | Fixed | Time input `t` fed to network (`tproj`); prediction cache disabled (`use_cache = False`) | Schiff et al. `scripts/train_qm9_guidance.sh:51–66` (`time_conditioning = True`, `sampling.use_cache = False`) |
| Network, data, optimizer, dropout, label noise, EMA, stopping | Fixed | `Net84` (`d = 256, nlayer = 6, nhead = 8, dropout = 0.1`, per-field learned pos), real-token embeddings/heads (no `[MASK]` output in uniform), ask dropout `0.15` (paper uses `0.1`), label noise `±1/48` (exact on val), slot shuffling every batch, AdamW `lr = 4e-4`, `wd = 0.01`, `betas = (0.9, 0.98)`, 500 warm-up + cosine decay to 80 epochs, batch `256`, grad clip `1.0`, EMA `0.999`, training seed `0`, validation loss on 4 fixed stratified CRN draws, patience `3`, threshold `0`, cap `80`, every epoch saved | Inherited from the default model (`src/train84.py`, `src/diffusion84.py`); ask dropout `0.15` fixed to match control (map #270 Notes) |

## Plan

### 1. Pass-count selection (#277, before any battles)

Sample **Uniform, no filter** (`udlm_loglinear_nofilter`) at `K ∈ {84, 128, 256}` reverse passes, guidance `γ = 1` (conditional, no extra push), 3 sampling seeds (`64` teams per cell across the 17 scored asks `1/24` to `17/24`, i.e., `1,088` teams per seed; no battles). Tabulate first-try legality (Showdown validator) and every diversity metric below (within-ask and pooled).

- **Selection rule:** Choose the smallest `K* ∈ {84, 128, 256}` whose 3-seed mean is **within seed noise of `K = 256`** on BOTH first-try legality (within 2 percentage points of `K = 256`, or 3 seeds do not agree in sign against `256`) and within-ask line-up uniqueness and species replacement distance (difference smaller than 1 seed standard error, or 3 seeds do not agree in sign against `256`).
- All battled UDLM runs (#278 and #279) use that chosen `K*`.
- **Legality floor short-circuit rule:** If **Uniform, no filter** has first-try legality at `γ = 1` below **5.0%** (`< 64 / 1,280`) across all three pass counts `84, 128, 256`, it cannot complete cells within `MAX_TRIES = 1,280`; record its pass-count and unguided legality measurements, mark `g_max = none`, and report the result without spending battle grids in #279.

### 2. Unguided and conditional baseline comparison (#278)

Compare **Masked, full filter** and **Uniform, no filter** at guidance `γ = 1` (conditional, no extra guidance push) at the chosen pass count `K*` with 3 sampling seeds, plus the unguided baseline (`γ = 0`, no ask, 64 teams × 3 seeds = 192 teams per arm):
- Validation loss bound (each model's own ELBO on the 4,368 validation teams; both absorbing ELBO and uniform continuous-time ELBO are upper bounds on `-log p(x)` in nats per field);
- First-try legality from Showdown's validator;
- Ask-versus-got error on asks `1/24` to `17/24` (asks `18/24` to `23/24` apart).

The `γ = 1` battled grid from #278 is **reused directly** as the `γ = 1` point of #279 (zero duplicate battles).

### 3. Measures and Pareto frontiers (#279 and #280)

At every guidance strength, run the `verify-ask-grid84` smoke test before launching the grid, battle the full 23-ask grid (`4,416` teams, `216,384` battles), and measure every metric below on the 17 scored asks (`1/24` to `17/24`, `1,088` teams per seed; `18/24` to `23/24` reported apart). Because #297 showed that pooling 23 asks makes distinct line-ups *rise* with guidance as asks separate while within-ask diversity *falls*, every diversity metric is measured **within each ask** (64 teams per ask × seed, averaged over asks `1/24` to `17/24`) as the primary diversity axis, with the **pooled** reading (1,088 teams per seed) reported beside it.

| Frontier / measure | Exact definition and source | Direction |
| --- | --- | --- |
| **Ask-versus-got error** (x-axis of every frontier) | Mean squared error between `k/24` and mean got over `k = 1..17` (`src/ask_line.py`). Error on `18/24..23/24` reported apart. | Lower is better |
| **Novelty** | `composition.novel_draw_fraction` against the labelled training line-ups of `lossdata84_shuffled_only.npz` (`src/diversity_metrics.py`, `src/grid_novelty.py`), with `novel_unique_fraction` and swap counts (0, 1, 2, 3+) beside it. | Higher is better |
| **Every metric in `code/src/diversity_metrics.py`** | For all three representations (`categorical_48`, `with_spreads`, `composition`): `unique_fraction`, `repeat_fraction`, `novel_draw_fraction`, `novel_unique_fraction`, plus `composition.largest_share`. Primary frontier: within-ask mean over asks `1/24..17/24`; pooled over `1/24..17/24` reported beside it. | Higher is better (for `repeat_fraction` and `largest_share`, frontier uses `1 − metric`) |
| **Mean species replacement distance** | `6 − shared_pairs / pairs` over unordered distinct team pairs (`scripts/analyze_dibo_diversity.py:30–44`, `src/uniform_preflight84.py`). Primary: within-ask mean over asks `1/24..17/24`; pooled beside it. | Higher is better |
| **First-try legality** | Share of tries accepted by Showdown's validator before redraws, over asks `1/24..17/24`. | Higher is better |
| **Win rate** | Mean got against the 49 distinct top-50 meta teams in shuffled-slot battles: top scored asks (`15/24..17/24`) and pooled (`1/24..23/24`). | Higher is better |
| **Combined Pareto front** | 5-axis front over `(−error, novelty, within-ask composition unique_fraction, within-ask mean species replacement distance, top-ask win rate)`: point A dominates point B iff A is at least as good on every axis on each of the 3 sampling seeds and strictly better on at least one, seed for seed (`src/diversity_novelty_frontier.py` rule). | Non-dominated set |

### 4. Trade-off summary: dominated area and highest usable guidance weight

Fixed here before any run (`src/uniform_preflight84.py`):

1. **Dominated area (2D hypervolume)** `HV_A^{(s)}` for arm `A` and sampling seed `s ∈ {1, 2, 3}` on each `(error, metric)` frontier, using only **usable** guidance strengths of arm `A`:
   - Reference box `(error_ref = 0.025, metric_ref = 0.0)` (encloses every usable guidance point from `γ = 1` downward; for `repeat_fraction` and `largest_share` plotted as `1 − metric`, `metric_ref = 0.0`).
   - Computed as the exact 2D area dominated by the arm's upper Pareto envelope (`uniform_preflight84.dominated_area`).
   - Reported as: 3-seed mean, all 3 individual seed values, min–max seed range, and paired difference `HV_U − HV_D` with sign agreement across all 3 seeds plus a 4,000-draw within-cell team and opponent bootstrap 95% interval.
2. **Equal-error mean advantage** `Δm_eq^{(s)}` (`uniform_preflight84.equal_error_advantage`):
   - Over the shared error overlap `[e_lo, e_hi]` where both **Uniform, no filter** and **Masked, full filter** have usable Pareto envelopes (reported whenever `e_hi − e_lo ≥ 0.002`, else `no overlap`), compute `(1 / (e_hi − e_lo)) ∫_{e_lo}^{e_hi} [f_U^{(s)}(e) − f_D^{(s)}(e)] de` via linear interpolation along each arm's upper Pareto envelope. This gives the direct metric gain or loss at equal ask-versus-got error in the metric's own units.
3. **Highest usable guidance weight (`g_max`)**:
   - A guidance strength `g` is **usable** (`uniform_preflight84.is_usable_strength`) iff ALL THREE hold on asks `1/24` to `17/24` across all 3 sampling seeds:
     1. **Cell completion:** all 51 scored cells (`17 asks × 3 seeds`) fill 64 validator-accepted non-held-out teams within `MAX_TRIES = 1,280` (effective acceptance `≥ 5.0%` in every scored cell).
     2. **Non-collapsed diversity:** within-ask mean `composition.unique_fraction ≥ 0.20` (at least `12.8` distinct line-ups per 64 teams) AND within-ask mean `composition.largest_share ≤ 0.50`.
     3. **Non-degraded error:** ask-versus-got error does not rise by more than `+0.005` above the arm's lowest error observed at lower guidance weights.
   - `g_max` is the largest tested guidance strength passing all three conditions.
4. **Verdict rule per frontier (#280):**
   - **Uniform, no filter** **beats** **Masked, full filter** on a frontier iff its dominated area `HV_U > HV_D` on all 3 sampling seeds, the 95% bootstrap interval of `HV_U − HV_D` lies above zero, AND on the equal-error overlap (when `e_hi − e_lo ≥ 0.002`) `Δm_eq > 0` on all 3 seeds.
   - It **loses** iff `HV_U < HV_D` on all 3 seeds with the 95% interval below zero.
   - Otherwise it **matches / is inconclusive**.
   - Any cross-model gap smaller than training luck (`0.0019` error shift, corresponding to `~0.0009` on novelty dominated area and `~0.0017` on within-ask uniqueness dominated area) is explicitly flagged as within single-seed training luck.

### 5. Revision measure (mechanism, step 7)

Logged inside the reverse sampling loop for all 84 fields across `z^{(0)}, z^{(1)}, ..., z^{(K*)}` (before validator redraws and before Stat Alignment repair), reported at every guidance strength, averaged per field type with **species reported separately** (`species`, `ability`, `item`, `move`, `Stat Alignment`, `Stat Points`):

1. **Value changes beyond the first:** `R_c = max(0, sum_{i=1}^{K*} 1[z^{(i)}_c ≠ z^{(i-1)}_c] − 1)`. **Masked, full filter** scores `0`.
2. **Fixed-predictor noise floor:** Exact deterministic expectation `E[max(0, C_c − 1)]` from Schiff et al. App. A Eq. 25 for a fixed one-hot predictor starting from uniform initialization over real vocabulary size `N_k`, on UDLM's baseline schedule (`α(t) = 1 − t`, `eps = 1e-5`) and pass count `K*` (`uniform_preflight84.revision_noise_floor`):

| Field type | Real vocab `N_k` | 84 passes | 128 passes | 256 passes |
| --- | ---: | ---: | ---: | ---: |
| Species | 347 | 4.000 | 4.418 | 5.107 |
| Ability | 200 | 3.989 | 4.406 | 5.094 |
| Item | 149 | 3.980 | 4.397 | 5.083 |
| Move | 497 | 4.004 | 4.422 | 5.112 |
| Stat Alignment | 26 | 3.822 | 4.225 | 4.889 |
| Stat Points | 33 | 3.862 | 4.269 | 4.939 |

3. **Changes of mind (headline):** `M_c = sum_{i=1}^{K*-1} 1[top^{(i)}_c ≠ top^{(i-1)}_c]`, where `top^{(i)}_c = argmax_{v} log x_θ(z^{(i)}, t_i, ask)[c, v]` (the conditional clean-field top prediction for `γ > 0`, and the unconditional clean-field top prediction for `γ = 0`). **Masked, full filter** scores `0`.

### 6. Coarse guidance grid and bend-point rule

Because **Masked, full filter** mixes clean-field log-probabilities while **Uniform, no filter** mixes reverse posteriors, weight scales are not comparable and every frontier is plotted against ask-versus-got error.

- **Coarse grid (both arms):** `{1, 2, 4, 8}`, plus the unguided baseline (`γ = 0` / no ask, 192 teams × 49 = 9,408 battles).
  - For **Masked, full filter**, reuse the battled grids from the existing default checkpoint (#385 / stored grids at `w ∈ {1, 2, 4, 8}`, plus any matching grids from diversity map #265).
  - For **Uniform, no filter**, `γ = 1` comes from #278; #279 adds `γ ∈ {2, 4, 8}` if #277 passes the 5% legality floor.
- **Bend-point and extension rule (`uniform_preflight84.bend_points_to_add`):** After the coarse grid lands, add up to **2 extra guidance weights per arm** (strict cap, so at most 6 battled guidance strengths per arm):
  1. **Interior bend / usability bracket:** Between adjacent coarse weights `g_a < g_b`, insert geometric midpoint `round(sqrt(g_a × g_b), 1)` (i.e., `1.4`, `2.8`, or `5.7`) if: (a) `g_a` is usable and `g_b` is not usable (highest priority, bracketing `g_max`), OR (b) `|error(g_b) − error(g_a)| > 0.003`, OR (c) `|within_ask_unique_fraction(g_b) − within_ask_unique_fraction(g_a)| > 0.10`.
  2. **Upper extension:** If `g = 8` is usable AND `error(4) − error(8) ≥ 0.001`, add `g = 16.0`.
  3. If more than 2 candidates qualify, take the top 2 by priority (usability bracket first, then largest normalized error + diversity jump).

### 7. Other fixed parts of the plan

| Part | Value |
| --- | --- |
| Smallest effect worth acting on | `0.001` in ask-versus-got error (glossary rule for replacing the default model), or at equal error: `≥ 0.05` higher novelty, `≥ 0.05` higher within-ask line-up uniqueness (`+3.2` line-ups per 64 teams), or `≥ 0.25` higher within-ask mean species replacement distance. |
| Comparisons planned | **Uniform, no filter** against **Masked, full filter** at equal ask-versus-got error across every frontier and on the 5-axis combined Pareto front. |
| Baselines | **Masked, full filter** (`w ∈ {1, 2, 4, 8}` + bend points); each arm's unguided (`γ = 0`, 192 teams) baseline; scorer controls (top-50 meta teams near `0.50`, uniformly drawn legal teams near `0.02`). |
| Stopping rule | Pass-count table (#277), unguided/conditional comparison (#278), coarse grid + up to 2 rule-selected bend points per arm (#279). No extra runs to chase a result. |
| Slope of got on ask | Not reported (owner, 2026-11-05). |

## Port-equality test

Required gate before training starts (#275 blocks #276): on one identical deterministic batch spanning all six field types (`species: 347, ability: 200, item: 149, move: 497, Stat Alignment: 26, Stat Points: 33`), at UDLM's baseline log-linear schedule (`α(t) = 1 − t`, `α′(t) = −1`, `zero_recon_loss = True`), our uniform loss implementation in #275 must equal the verbatim upstream loss of `kuleshov-group/discrete-diffusion-guidance@edb0f8c28b7caeb4ea7a06a2fee8d74ab6da1661` (`diffusion.py:563–614`), verified by `uniform_preflight84.port_equality_check`:

- **Forward value equality:** every `(batch, column)` loss value matches upstream `diffusion.py:572–597` to `atol = 1e-12, rtol = 1e-10` in `float64` and `atol = 1e-3, rtol = 1e-3` in `float32` across `t ∈ [1e-3, 1 − 1e-3]` (the `float32` margin reflects upstream's `1 − (1 − t)` mantissa rounding near `t = 1e-3` and `t = 1 − 1e-3`).
- **Gradient equality:** `∂loss / ∂logits` matches upstream to `atol = 1e-10, rtol = 1e-8` in `float64` and `atol = 1e-4, rtol = 1e-3` in `float32`.
- **Reverse posterior and CFG equality:** our reverse posterior matches upstream `_compute_posterior` (`diffusion.py:356–381`) and our uniform CFG distribution at `γ ∈ {0, 1, 2, 4}` matches `_cfg_denoise` (`diffusion.py:1298–1328`) to `1e-6` (`float32`) / `1e-12` (`float64`).
- **Numerical hygiene (`mps` `float32`):** compute `move_chance = t` directly (never `1 − (1 − t)`), `α(t) = (1 − t).clamp(min=1e-6)`, and in the reverse posterior `Δα = t − s` directly, keeping every loss, posterior, and gradient finite on Apple Silicon `mps`.

## Predicted outcome

Written 2026-11-24 before any uniform training or sampling begins, to be scored in #280:

| Stage / arm | Predicted outcome | Chance | Why |
| --- | --- | ---: | --- |
| Pass count (#277) | `K* = 128` or `256` passes is needed for **Uniform, no filter** (`128`: 50%; `256`: 40%; `84`: 10%). | 50% for `128`, 40% for `256` | Schiff et al.'s default is `128`; without an in-loop legality filter, finer `dt` gives the model more passes to resolve cross-field consistency after late token updates. |
| **Uniform, no filter** (`udlm_loglinear_nofilter`) | Non-trivial first-try legality at `γ = 1` (`15%–55%`, well above `0%` because #361 showed the 6-layer transformer learns slot binding by epoch 80 without tables), ask-versus-got error `~0.020–0.040` at `γ = 1`, with legality falling as `γ` rises (`g_max ≤ 2–4`). Against **Masked, full filter**: **loses** on overall dominated area (80% lose; 15% match; 5% win), though accepted UDLM teams may hold equal or slightly higher novelty/diversity in any shared mid-error overlap (`0.008–0.016`). | 80% lose overall | #361 still left 33% illegal teams at epoch 80 under sequential masked decoding; simultaneous uniform updates with `eps = 1e-5` noise, no Stat Point / Species / Item Clause mask, and a harder continuous-time uniform objective (Schiff et al. Table 1) handicap legality and low-error accuracy relative to **Masked, full filter**. |
| Revision mechanism (step 7) | Raw value changes beyond the first sit near or modestly above the fixed-predictor noise floor (`~4.4–5.5` on species at 128 passes). **Changes of mind** (headline) are nonzero (`~2–6` flips on species, more on moves/items) and **fall as guidance `γ` rises** on species because sharper guided posteriors lock earlier. | 75% | Posterior-level CFG sharpens transition mass toward high-ask species early, reducing top-prediction flips as `γ` increases. |

**Largest surprise:** **Uniform, no filter** reaching `≥ 50%` first-try legality AND `≤ 0.005` ask-versus-got error while holding higher novelty and within-ask diversity than **Masked, full filter**—which would show that pure, filter-free uniform discrete diffusion beats our heavily engineered masked sampler.

## Probes

Battle-free pre-training probes run 2026-11-24 by `code/src/uniform_preflight84.py`; saved to `~/vgc-data/results/uniform_discrete_diffusion_preflight.json`. Unit tests: `code/scripts/test_uniform_preflight84.py` (7 tests, all pass; ruff clean).

| Probe | Result | Pass |
| --- | --- | --- |
| **Training table & wrong-stone check** | `lossdata84_shuffled_only.npz` (`sha256: e58af768…`): `218,433` rows (`218,423` labelled, `10` unlabelled), `214,055` labelled train, `4,368` labelled val, min `49` shuffled battles per team, `0` wrong-stone rejects on train and val. | yes |
| **Separation** | All 50 held-out meta teams (ranks 51–100) encode; `0` overlap with training teams. Test 500 excluded at table build (`lossdata84.py`). | yes |
| **Coverage (vocabulary)** | Regulation vocab total (`[MASK]` at 0): `348, 201, 150, 498, 27, 34`. Uniform real vocab (no `[MASK]`): `347, 200, 149, 497, 26, 33`. Labelled training uses `189, 127, 111, 316, 17, 33`. | yes |
| **Port-equality oracle (`float64` & `float32`)** | Uniform Eq. 19 loss at `α(t) = 1 − t` matches verbatim upstream `diffusion.py:572–597` on all 6 field types: `float64` max abs loss diff `1.1e-13` to `1.9e-13`, grad diff `2.8e-17`; `float32` max abs loss diff `5.7e-5` to `1.2e-4`, grad diff `1.9e-8` to `5.2e-8`. | yes |
| **Finite-difference KL identity** | Uniform Eq. 19 loss matches `KL(q(z_{t−dt} \| z_t, x_0) \|\| p_θ(z_{t−dt} \| z_t)) / dt` (`dt = 1e-7`, `float64`) within `6.4e-7` relative error on `α(t) = 1 − t`. | yes |
| **Revision noise floor golden** | Reproduces #273 golden `4.404443` (`N = 189, K = 128, α(t) = 1 − t`); full 6-field × 3-pass table computed. | yes |
| **Ceiling** | Provisional shuffled-only linear reference (`sched_linear_species_first`, guidance `1, 2, 4, 8`) has error `0.0162 → 0.0038`, novelty `0.448 → 0.383 → 0.420`, within-ask line-up uniqueness `0.900 → 0.842`, within-ask species replacement distance `4.200 → 3.524` (a `−0.676` drop across guidance, far above the `0.25` smallest effect worth acting on). Data-limit floor on older table is `0.0038` (`docs/training-teams-line.md`). | yes |
| **Training smoke (init loss & 32-team overfit)** | Runs in #275 once uniform training code exists: per-column cross-entropy at init near `ln N_k`, continuous-time uniform loss finite and falling, 32-team batch overfits `≥ 40%`. | gated in #275 |
| **Round trip, read teams, controls, condition, copies** | Run by `ask_grid84.py preflight` (and `verify-ask-grid84` smoke test in a private DB) with the UDLM checkpoint and sampler before #278/#279 grids; any failure stops the launch. | gated before #278/#279 |
| **Invariance, repeat** | Skipped now: the battle scorer (`ask_grid84.py` / `matchup_db.py`) is unchanged and enforces shuffled slots on every battle (`require_shuffled`). | not run (unchanged scorer) |

## Noise floor and budget

Measured on this laptop from the provisional shuffled-only reference grids (`sched_linear_species_first` battled at `w ∈ {1, 2, 4, 8}`, diversity/novelty read from the non-replayed redraws on `grid_seeds` bases; `src/uniform_preflight84.py`) and from prior shuffled-only runs (`docs/masking-rate-rows.md`):

| Quantity | Measured value | Seed spread / SD | Source |
| --- | --- | --- | --- |
| Ask-versus-got error (`w = 1, 2, 4, 8`) | `0.0162, 0.0075, 0.0049, 0.0038` | Paired row difference SD `0.00034–0.00037` (detectable gap `~0.001` at 5% / 80% power); single-seed training luck up to `0.0019` | `docs/masking-rate-rows.md`, `uniform_preflight84.py` |
| Novelty (`novel_share`, asks `1/24..17/24`) | `0.448, 0.383, 0.397, 0.420` | Dominated area `0.00926`, 3-seed range `[0.00886, 0.00955]` (range `0.00069`, ~7.5% of mean) | `uniform_preflight84.py` |
| Within-ask line-up uniqueness (`1/24..17/24`) | `0.900, 0.894, 0.898, 0.842` (`57.6 → 53.9` per 64) | Dominated area `0.01911`, 3-seed range `[0.01899, 0.01930]` (range `0.00031`, ~1.6% of mean) | `uniform_preflight84.py` |
| Within-ask mean species replacement distance | `4.200, 3.907, 3.808, 3.524` | Dominated area `0.08476`, 3-seed range `[0.08387, 0.08586]` (range `0.00199`, ~2.3% of mean) | `uniform_preflight84.py` |
| Training luck on pooled win rate | — | Up to `0.026` between two seeds of one recipe (`21` epochs) | `docs/position-embedding-split.md` |

The 3-seed ask-grid budget easily detects the smallest effects worth acting on within a checkpoint (`0.001` error, `0.05` novelty/uniqueness, `0.25` species replacements; seed-to-seed dominated-area spread is only `1.6%–7.5%`). Because UDLM trains on one seed (`seed = 0`), cross-checkpoint effects smaller than training luck (`0.0019` error) remain exploratory.

**Compute budget (laptop only, queued detached behind any running jobs after `python3 ~/vgc-data/whats_running.py`; no Modal):**

| Step | Ticket | Grids / teams | Battles | Estimated laptop time |
| --- | --- | --- | ---: | --- |
| Train `udlm_loglinear` (cap 80, patience 3, scaling #361's 407.7 min on the same 214,055-team table) | #276 | 1 training | 0 | ~6.8 hours |
| Pass-count table: **Uniform, no filter** × 3 counts (`84, 128, 256`) at `γ = 1`, 3 seeds, asks `1/24..17/24` | #277 | 3 sampling sweeps | 0 | ~25–40 minutes |
| Unguided baseline (`γ = 0`, 192 teams × 49) for **Uniform, no filter** (+ control if new) | #278 | up to 2 unguided sets | 18,816 | ~4 minutes |
| Conditional `γ = 1` ask grid (**Uniform, no filter** + control `w = 1` if not already battled) | #278 | up to 2 grids | 432,768 | ~1.3 hours (~40 min/grid) |
| Coarse guidance sweep (`γ ∈ {2, 4, 8}`, reusing `γ = 1` from #278 and existing control grids at `w ∈ {2, 4, 8}`) | #279 | up to 3 uniform grids | 649,152 | ~2.0 hours battles (+ sampling; 0 if short-circuited) |
| Bend-point / extension grids (at most 2 added weights per arm) | #279 | up to 4 grids | 865,536 | ~2.7 hours battles (upper bound) |
| **Total (worst case, UDLM usable + max 2 bend points per arm)** | | **up to 9 new grids** | **~1.97M battles** | **~6.8 h training + ~6.5 h sampling & battles** |

## What the later tickets take from this

| Ticket | What this pre-registration fixes for it |
| --- | --- |
| [#275 Write the uniform discrete diffusion training and sampler, and pass the port-equality test](https://github.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/issues/275) | Implement `udlm_loglinear` (UDLM's baseline schedule `α(t) = 1 − t`) on real vocab sizes `(347, 200, 149, 497, 26, 33)` with no loss legality mask, `t ∈ [1e-3, 1 − 1e-3]`, pass `uniform_preflight84.port_equality_check` (`float64` & `float32`) + training smoke (init CE near `ln N_k`, 32-team overfit `≥ 40%`), implement Eq. 15 reverse posterior + Eq. 6–8 posterior-mixing CFG + unconstrained sampler logging the 3 revision numbers per pass (no in-loop filter; Showdown validator checks after sampling), and wire `ask_grid84.py` with `grid_seeds.seed0` (never legacy `7300`). |
| [#276 Train the log-linear uniform discrete diffusion model on the laptop](https://github.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/issues/276) | Queue detached on laptop on `lossdata84_shuffled_only.npz` (`sha256: e58af768…`), patience 3, threshold `0`, cap 80, every epoch saved; checkpoint `stat84_udlm_loglinear_es3_cap80`. Post chosen epoch and validation loss when it lands. |
| [#277 Tabulate 84, 128 and 256 passes at guidance 1 for the uniform discrete diffusion arm](https://github.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/issues/277) | Run 3 seeds × 17 scored asks (`1,088` teams/seed) at `γ = 1` for **Uniform, no filter**, no battles; apply the seed-noise rule to pick `K*`, and apply the `< 5%` legality short-circuit check. |
| [#278 Compare the arms unguided: validation loss bound, first-try legality and ask-versus-got error](https://github.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/issues/278) | Report validation loss bound, `γ = 0` unguided baselines, and `γ = 1` ask grids at `K*` (reused directly by #279). Run `ask_grid84.py preflight` and `verify-ask-grid84` smoke test first. |
| [#279 Sweep guidance on the uniform discrete diffusion arm](https://github.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/issues/279) | Run coarse grid `γ ∈ {2, 4, 8}` (reusing `γ = 1` and existing control grids), evaluate `is_usable_strength` and `bend_points_to_add` (max 2 added weights per arm), log novelty, every `diversity_metrics.py` metric (within-ask & pooled), mean species replacement distance (within-ask & pooled), and the 3 revision numbers per field type. |
| [#280 Uniform discrete diffusion trade-off: smaller or bigger than the default model's, frontier by frontier?](https://github.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/issues/280) | Score every frontier with `dominated_area`, `equal_error_advantage`, `g_max`, and the 5-axis combined Pareto front; explain results with the 3 revision numbers; score the pre-registered predictions above. |

## Scope note

Two custom additions originally sketched on map #270 were dropped by the owner on 2026-11-24 before any training or battle started so that UDLM stays faithful to published research: (1) the per-field uniform masking schedule (no published uniform discrete diffusion paper uses per-field schedules, and masking-schedule differences have barely any effect here), and (2) the one-direction in-loop legal Pokémon construction filter (invented without a paper citation). Only one new model trains: **Uniform, no filter** (`stat84_udlm_loglinear_es3_cap80`), using UDLM's baseline `α(t) = 1 − t` schedule and post-sampling validation as in Schiff et al. (2024).

## Deviations

None yet.
