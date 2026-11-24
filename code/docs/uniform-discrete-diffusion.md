# Uniform discrete diffusion comparison: does uniform discrete diffusion make guidance cost less novelty and diversity than the default model?

Pre-registration, written 2026-11-24 before any uniform discrete diffusion training, sampling, or battle starts. Ticket: [Pre-register the uniform discrete diffusion comparison, with its predicted outcome and port-equality test](https://github.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/issues/274), on the map [Uniform discrete diffusion map: does uniform discrete diffusion make guidance cost less novelty and diversity than the default model?](https://github.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/issues/270).

**Changed 2026-11-24, before any training or battle, by two owner decisions:**
1. The control arm is locked as **Linear masked, full filter** (`linear_masked_filter`), using the linear masking schedule (`1 − α(t) = t` on all 84 fields) confirmed for [Retrain the default checkpoint on the shuffled-only table without wrong-stone teams](https://github.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/issues/385).
2. The per-field uniform masking schedule is **dropped** (owner: *"We are not apply per field uniform masking schedule because there has been no paper i the field of uniform diffusion model that use per field masking"*). Only the paper's log-linear schedule (`α(t) = 1 − t`, i.e., linear corruption probability `1 − α(t) = t` on all 84 fields) trains—matching the control's masking schedule so the comparison isolates the noise process (uniform vs `[MASK]`) and the sampling filter (one-direction vs none).

Mode: **exploratory** for cross-model comparisons (uniform discrete diffusion arms against **Linear masked, full filter**), because one training seed per checkpoint cannot separate the noise process from training luck, which has reached 0.0019 in ask-versus-got error and 0.026 in pooled win rate between two trainings of one recipe (`docs/position-embedding-split.md`). Comparisons on the single uniform checkpoint—pass count (#277), one-direction filter against no filter, and guidance curves within one checkpoint—are **confirmatory** for that checkpoint, because their plan, metrics, usability gate, and bend-point rules are fixed here before any run.

## Terms

An **ask** is the win rate the model is told to aim for (`k/24` for `k = 1` to `23`). **Got** is the win rate its teams then score. A **line-up** is a team's unordered set of six Pokémon species. The **ask-versus-got error** is the mean squared error between each ask and the mean got at that ask, scored on asks `1/24` to `17/24` (`src/ask_line.py`); asks `18/24` to `23/24` are reported apart because the training table holds almost nothing there. Near the line means an error under `0.001`.

**Win rate** means one battle against each of the 49 distinct top-50 meta teams, with the behaviour-cloning policy on both sides and slots shuffled every battle. The **top-50 meta teams** are the project's fixed list of real Regulation M-B teams (one duplicate leaves 49 distinct opponents). The **behaviour-cloning policy** is the battle player trained to imitate human battle decisions.

A **masked diffusion model** corrupts each field toward an absorbing `[MASK]` token; its reverse sampler fills one masked field per pass in species-first order (84 passes) and never changes a filled field. A **uniform discrete diffusion model** (UDLM, Schiff et al. 2024, *Simple Guidance Mechanisms for Discrete Diffusion Models*, [arXiv:2412.10193v3](https://arxiv.org/pdf/2412.10193v3), upstream code pinned at `kuleshov-group/discrete-diffusion-guidance@edb0f8c28b7caeb4ea7a06a2fee8d74ab6da1661`) has no `[MASK]` token: forward corruption replaces each field with a random token drawn uniformly from that field's real vocabulary. Sampling starts from a fully random team, runs a fixed number of reverse **passes** (one full network evaluation of all 84 fields), and may change any field on any pass.

The **masking schedule** is the probability that the forward process has corrupted a field by time `t ∈ [0, 1]` during training (`1 − α(t)` in paper notation): replacing it with `[MASK]` in a masked diffusion model, or with a random real value in a uniform discrete diffusion model. Schiff et al. call `σ(t) = −log(1 − t)` **log-linear** noise, which gives `α(t) = exp(−σ(t)) = 1 − t` and therefore corruption probability `1 − α(t) = t` on every field—the exact same linear masking schedule used by **Linear masked, full filter**. **Stat Alignment** is a Pokémon's nature (such as Adamant). **Stat Points** are the six stat allocations per Pokémon (0 to 32 per stat, at most 66 in total).

The **legal Pokémon construction filter** zeroes the probability of values that Regulation M-B rules forbid next to the team's other fields (`lossdown.Tables` plus `stat_points.allowed_stat_points`). **First-try legality** is the share of sampled teams that Pokémon Showdown's Regulation M-B validator accepts before any redraw. **Species replacement distance** between two six-species line-ups is `6 − shared species` (forms distinct, order and sets ignored). **Novelty** is the share of generated teams whose six-species line-up appears in none of the labelled training teams. **Guidance** is the classifier-free guidance weight: `1` is conditional with no extra push, `0` is unguided (no ask), and higher weights push harder toward the ask.

## What is already known

Searches across `code/docs/`, `code/docs/experiment-names.md`, `notes/List of Experiments.md`, the closed research tickets of map #270 (#271–#273), and the web (2026-11-24) found no existing uniform discrete diffusion experiment in this repository and no published comparison of uniform against masked discrete diffusion on structured game-team generation or equal-error novelty/diversity frontiers. Across the published uniform discrete diffusion literature (Multinomial Diffusion, D3PM-Uniform, SEDD-Uniform, UDLM, Duo, GIDD), every paper uses a single position-shared schedule `α(t)` across the sequence rather than a per-field schedule.

| Earlier result | What it established | What it leaves open |
| --- | --- | --- |
| Upstream UDLM reading (#271, `research/uniform-noise-udlm-sampling`) | Continuous-time uniform loss (paper Eq. 19, `diffusion.py:563–614`) is implemented for log-linear noise (`α(t) = 1 − t`). Classifier-free guidance mixes reverse posteriors (`diffusion.py:1314–1334`), not clean logits. Sampling runs a fixed pass count on `linspace(1, 1e-5, K+1)` with no early stop and no clean-up pass. | How UDLM behaves on 84-field heterogeneous-vocabulary teams compared with linear masked diffusion. |
| Revision-count and repeat-revision research (#272, #273, `research/uniform-diffusion-token-revisions`) | No standard revision count exists in the discrete diffusion literature. Uniform reverse transitions cause repeated state changes even with a completely fixed, certain predictor: `4.404443` expected changes beyond the first per field at 128 passes, `N = 189`, ideal `α(t) = 1 − t`. | How often trained uniform models actually change their top prediction (**changes of mind**) and how that tracks the guidance trade-off. |
| Masking-schedule rows and no-guidance line (#297, #298, #73, `docs/masking-rate-rows.md`, `docs/no-guidance-line.md`) | On the shuffled-only table (`stat84_shuffled_noise_linear_es3_cap80`, species-first), ask-versus-got error falls from `0.0162` at guidance `1` to `0.0075`, `0.0049`, and `0.0038` at guidance `2, 4, 8`. Pooled distinct line-ups across 23 asks *rise* with guidance as asks stop overlapping, while **within-ask** line-ups (64 teams per ask × seed) and within-ask species replacement distance *fall* (`57.6 → 53.9` line-ups per 64 teams; `4.200 → 3.524` species replacements). | Whether uniform discrete diffusion's repeated editing and posterior-level guidance flatten that within-ask diversity/novelty drop at equal error. |
| No tables in the loss (#361, #367, closed 2026-11-24) | Training the 6-layer, 256-wide masked network for 80 epochs on the filtered shuffled-only table (`214,055` train / `4,368` val, `407.7` min) **without** the legal Pokémon construction filter in the loss learned slot-to-Pokémon binding between epochs 45 and 48: illegal abilities fell `82% → 0%`, illegal moves `58% → 0%`, and first-try legality without the sampling filter rose from `0%` to `67%` (33% illegal). | Whether a uniform discrete diffusion model (which cannot apply the tables in its loss because `z_t` holds noisy species) also learns slot binding within 80 epochs, and what first-try legality it reaches with and without a one-direction sampling filter. |
| Web literature (Hoogeboom 2021; Austin 2021; Lou 2024; Schiff 2024; Sahoo 2025; von Rütte 2025) | Published uniform discrete diffusion models all use position-shared schedules (cosine, mutual-information, geometric, log-linear `α(t) = 1 − t`, Gaussian-duality integral, or hybrid mask/uniform). UDLM is competitive with absorbing MDLM on conditional molecular (QM9) and genomic guidance; high CFG still trades fidelity against mode coverage/novelty in both families. | Direct equal-error Pareto measurement of novelty and diversity on our 84-field Pokémon domain. |

## Arms

Named with `.claude/skills/name-experiments/SKILL.md`. All three arms share the 84-field transformer architecture (256 wide, 6 layers, 8 heads, dropout 0.1), the filtered shuffled-only labelled training table (`214,055` train / `4,368` validation), the same linear corruption probability `1 − α(t) = t` on all 84 fields, ask dropout `0.15`, label noise `±1/48`, AdamW (`lr 4e-4`, weight decay `0.01`, 500 warm-up steps, cosine decay, batch 256, gradient clip `1.0`, EMA `0.999`), training seed `0`, early stopping on validation loss (patience 3, threshold `0`, cap 80 epochs, every epoch saved), Stat Alignment repair after sampling, and the same shuffled-slot battle scorer.

Each arm name leads with its distinguishing model and schedule name, followed by its legal Pokémon construction filter while sampling:

| Model and masking schedule, then sampling filter | Code ID | Checkpoint | Masking schedule (`1 − α(t)`) | Legal Pokémon construction filter while sampling |
| --- | --- | --- | --- | --- |
| Linear masked, full filter. The control | `linear_masked_filter` | Retrained default from #385 with the linear masking schedule (confirmed by owner 2026-11-24; provisional reference `sched_linear_species_first` until #385 lands) | Absorbing `[MASK]`: linear `t` on all 84 fields (`α(t) = 1 − t`) | Full filter (bidirectional tables + Stat Point budget), species-first, 84 passes |
| Log-linear uniform, one-direction filter | `udlm_loglinear_filter` | `stat84_udlm_loglinear_es3_cap80` (one new uniform training `udlm_loglinear`) | Uniform random real token: linear `t` on all 84 fields (Schiff et al. `LogLinearNoise`, `α(t) = 1 − t`) | One-direction: ability, item, and moves fit current species in `z_t`; species limited only by Species Clause; Stat Point budget ≤ 66 |
| Log-linear uniform, no filter | `udlm_loglinear_nofilter` | Same checkpoint `stat84_udlm_loglinear_es3_cap80` | Same log-linear uniform schedule (`1 − α(t) = t` on all 84 fields) | None; Showdown validator redraw does the work |

The two uniform discrete diffusion arms share **one** trained checkpoint (`stat84_udlm_loglinear_es3_cap80`) and differ only in the sampler's legal Pokémon construction filter. Only the noise process and continuous-time loss differ from the control's training recipe; the uniform loss necessarily runs without the legal Pokémon construction mask in training (see Settings).

## Contract

| Row | Value |
| --- | --- |
| Claim | Uniform discrete diffusion (with log-linear masking schedule `α(t) = 1 − t`, with or without the one-direction legal Pokémon construction filter) reduces the novelty and diversity cost of classifier-free guidance relative to **Linear masked, full filter** at equal ask-versus-got error, yielding a larger dominated area on the novelty and within-ask diversity frontiers and/or a higher usable guidance weight. |
| What refutes it | Neither uniform discrete diffusion arm achieves a larger dominated area than **Linear masked, full filter** on novelty, within-ask line-up uniqueness, or within-ask mean species replacement distance with all 3 sampling seeds agreeing in sign and the 95% bootstrap interval excluding zero; or every uniform arm that matches the control's lowest error does so with lower diversity/novelty or fails the usability gate. |
| What changes if it holds | The winning uniform discrete diffusion arm becomes a candidate for a second training seed and for the project's next guided team-generation model (and motivates testing remasking on the default masked model, currently held under map #270's Not yet specified). If refuted, absorbing masked diffusion remains the default architecture and the map's verdict records how much uniform discrete diffusion cost on each frontier and why (from the revision numbers). |
| How far it reaches | Ask-versus-got error, novelty, diversity, legality, and win rate under the behaviour-cloning policy against the 49 distinct top-50 meta teams, slots shuffled, on one training seed per checkpoint. It does not claim generalization to human opponents, other battle policies, or held-out meta ranks 51–100 without a separate held-out evaluation. |
| Fields | All 84 fields are generated by the model: for each of six Pokémon, its species, ability, item, four moves, Stat Alignment, and six Stat Points. Nothing is copied or fixed. After sampling, `Model84.to_paste(..., repair=True)` replaces Stat Alignment to match the Pokémon's moves, identically in all three arms (`src/ask_grid84.py`). |
| Vocabulary | Source: `team84.regulation_vocab()` (Regulation M-B snapshot stored in checkpoints). In the masked encoding, index `0` is `[MASK]`: totals are 348 species, 201 abilities, 150 items, 498 moves, 27 Stat Alignments, and 34 Stat Point tokens (0 = `[MASK]`, 1–33 = values 0–32). Uniform discrete diffusion has no `[MASK]` token, so its real vocabulary sizes `N_k` exclude index `0`: **347 species, 200 abilities, 149 items, 497 moves, 26 Stat Alignments, and 33 Stat Points** (values 0–32, mapped cleanly to `0..N_k − 1` inside the uniform loss/posterior and shifted by `+1` to/from the 84-field table encoding). Labelled training teams in the filtered shuffled-only table use `189, 127, 111, 316, 17, and 33` of these real values respectively (`src/uniform_preflight84.py`). |
| Training teams | Filtered shuffled-only table `lossdata84_shuffled_only.npz` (`sha256: e58af7689e62a873ce19361237d7ff8e77cb791fc4c196cc705410755c5a03e2`, after issue #368's wrong-stone drop). Labelled teams only: **214,055 training teams and 4,368 validation teams** (218,423 labelled total; seeded 2% validation split with seed 0 preserved across the wrong-stone filter; minimum 49 shuffled-slot battles per team). Excluded: 10 unlabelled teams, 173,544 teams whose labels came only from non-shuffled battles or run files, 555 wrong-Mega-Stone teams, the 500 test teams, and the 50 held-out meta teams (ranks 51–100). Verified 0 Showdown validator failures and 0 `Tables.rejects` failures on both train and val sides. |
| Scorer | `src/ask_grid84.py` (`matchup_db.score_pastes`): 1 battle against each of the 49 distinct top-50 meta teams per generated team, behaviour-cloning policy on both sides, slots shuffled (`require_shuffled` enforces `VGC_SLOT_ORDER=shuffled`). Each new grid gets its own name-hashed sampling seed base via `src/grid_seeds.py` (`100,000 + ...`, never legacy `7300`, preventing seed replay of training teams per #353), with cell sampling seeds `s ∈ {1, 2, 3}` and battle seed `seed0 + s`. |
| Search region | No search and no elite selection rule. Each battled ask grid covers asks `1/24` to `23/24` (23 asks; error scored on `1/24` to `17/24`, `18/24` to `23/24` reported apart), 3 sampling seeds (`1, 2, 3`), 64 validator-accepted non-held-out teams per cell, try cap `MAX_TRIES = 1,280` per cell (23 × 3 × 64 = 4,416 teams, 216,384 battles per grid). |

### Settings

Every setting names its role and either cites Schiff et al. (2024, `arXiv:2412.10193v3`, upstream `edb0f8c2`) or is marked **Ours**.

| Setting | Role | Value | Paper or ours |
| --- | --- | --- | --- |
| Diffusion noise process | Scientific | Absorbing `[MASK]` (control) vs uniform random real token (`q(z_t \| x_0) = Cat(α_t x_0 + (1 − α_t) u)`, no `[MASK]`) | Schiff et al. §2.1 Eq. 1, §4.1; `diffusion.py:473–476` |
| Legal Pokémon construction filter while sampling | Scientific | Full bidirectional (control) vs one-direction (`udlm_loglinear_filter`) vs none (`udlm_loglinear_nofilter`) | No filter (post-sampling check): Schiff et al. QM9. One-direction filter: **Ours** |
| Guidance strength (`w` or `γ`) | Swept | Coarse grid `{1, 2, 4, 8}` per arm + unguided baseline (`γ = 0` / no ask), plus up to 2 bend/extension points per arm | Posterior CFG mixing: Schiff et al. §3.1 Eqs. 6–8 (`diffusion.py:1314–1334`). Grid & bend rule: **Ours** |
| Reverse pass count `K` | Nuisance (fixed before battles by #277) | Control: 84 (one field per pass). Uniform arms: smallest `K ∈ {84, 128, 256}` within seed noise of 256 at `γ = 1` on `linspace(1, 1e-5, K+1)` | Grid & no clean-up pass: Schiff et al. `configs/config.yaml:35–41`, `diffusion.py:1115–1209`. Selection rule: **Ours** |
| Sampling seed | Nuisance | 3 seeds (`1, 2, 3`) per arm and guidance strength, with per-grid `grid_seeds.seed0` base | **Ours** (#353 anti-replay rule) |
| Masking schedule (`1 − α(t)`) | Fixed (shared across all 3 arms) | `1 − α(t) = t` (`α(t) = 1 − t`, `α′(t) = −1`) on all 84 fields | Schiff et al. §4.2, App. D.3 (`diffusion.py:572–573`); matches the control's linear masking schedule (MD4 App. J.1; LLaDA Eq. 3) |
| Continuous-time loss | Follows noise process | Control: masked weighted cross-entropy (`diffusion84.elbo_terms`). Uniform: exact continuous-time Eq. 19 (`T = 0`, `zero_recon_loss = True`, `use_simple_ce_loss = False`), averaged over 84 fields | Schiff et al. §4.2 Eqs. 17–19 (`diffusion.py:563–614`) |
| Legal Pokémon construction filter in training loss | Fixed by noise process | Control: on (`table_loss = True`). Uniform model: **off**—in uniform noise `z_t` contains corrupted species that can contradict `x_0`'s true ability/item/moves, so conditioning a loss mask on `z_t` would forbid the true clean token | **Ours** (required by uniform corruption; supported by #361) |
| Time sampling in training | Fixed | Stratified antithetic `t = (u + i/B) mod 1`, shuffled, clamped to `[1e-3, 1 − 1e-3]` so `α(t) > 0` and `1 − α(t) > 0` everywhere | `training.sampling_eps = 1e-3`: Schiff et al. `configs/config.yaml:43–47`, `diffusion.py:449–458`; upper clamp `1 − 1e-3` avoids `1 / α_t` singularity at `t = 1` |
| Time conditioning & caching | Fixed | Time input `t` fed to network (`tproj`); prediction cache disabled (`use_cache = False`) | Schiff et al. `scripts/train_qm9_guidance.sh:51–66` (`time_conditioning = True`, `sampling.use_cache = False`) |
| Network, data, optimizer, dropout, label noise, EMA, stopping | Fixed | `Net84` (`d = 256, nlayer = 6, nhead = 8, dropout = 0.1`, per-field learned pos), real-token embeddings/heads (no `[MASK]` output in uniform), ask dropout `0.15` (paper uses `0.1`), label noise `±1/48` (exact on val), slot shuffling every batch, AdamW `lr = 4e-4`, `wd = 0.01`, `betas = (0.9, 0.98)`, 500 warm-up + cosine decay to 80 epochs, batch `256`, grad clip `1.0`, EMA `0.999`, training seed `0`, validation loss on 4 fixed stratified CRN draws, patience `3`, threshold `0`, cap `80`, every epoch saved | Inherited from the default model (`src/train84.py`, `src/diffusion84.py`); ask dropout `0.15` fixed to match control (map #270 Notes) |
| One-direction filter details (`udlm_loglinear_filter`) | Fixed | Applied to `q_guided` before categorical draw `z_s ~ Cat(q_guided)`: species masked only by Species Clause (`tables.BASE`) against the other 5 slots in `z_t`; ability, item, and moves masked to fit the slot's current species `z_t[slot, species]` (`AB, IT, MV`) plus Item Clause against the other 5 slots' current non-empty items, no duplicate non-empty moves, and non-empty first move; Stat Alignment unmasked (all 26 real values, repaired post-sampling); Stat Points masked by `budget = clamp(66 − others, min=0, max=32)` so an over-budget random start contracts to `≤ 66`; fallback to all real tokens if an intersection is empty | **Ours** (map #270 Notes) |

## Plan

### 1. Pass-count selection (#277, before any battles)

Sample both uniform discrete diffusion arms (`udlm_loglinear_filter`, `udlm_loglinear_nofilter`) at `K ∈ {84, 128, 256}` reverse passes, guidance `γ = 1` (conditional, no extra push), 3 sampling seeds (`64` teams per cell across the 17 scored asks `1/24` to `17/24`, i.e., `1,088` teams per seed; no battles). Tabulate first-try legality (Showdown validator) and every diversity metric below (within-ask and pooled).

- **Selection rule:** Choose the smallest `K* ∈ {84, 128, 256}` whose 3-seed mean is **within seed noise of `K = 256`** on BOTH first-try legality (within 2 percentage points of `K = 256`, or 3 seeds do not agree in sign against `256`) and within-ask line-up uniqueness and species replacement distance (difference smaller than 1 seed standard error, or 3 seeds do not agree in sign against `256`).
- All battled uniform runs (#278 and #279) use that chosen `K*`. If the two arms need different `K`, pick the larger `K*` required among arms that pass the 5% legality floor at `K = 256` so every battled uniform arm shares one pass count.
- **No-filter short-circuit rule:** If **Log-linear uniform, no filter** has first-try legality at `γ = 1` below **5.0%** (`< 64 / 1,280`) across all three pass counts `84, 128, 256`, it cannot complete cells within `MAX_TRIES = 1,280`; record its pass-count and unguided legality measurements and mark its highest usable guidance weight `g_max = none` without spending battle grids in #279.

### 2. Unguided and conditional baseline comparison (#278)

Compare the three arms at guidance `γ = 1` (conditional, no extra guidance push) at the chosen pass count `K*` with 3 sampling seeds, plus the unguided baseline (`γ = 0`, no ask, 64 teams × 3 seeds = 192 teams per arm):
- Validation loss bound (each model's own ELBO on the 4,368 validation teams; both absorbing ELBO and uniform continuous-time ELBO are upper bounds on `-log p(x)` in nats per field);
- First-try legality from Showdown's validator;
- Ask-versus-got error on asks `1/24` to `17/24` (asks `18/24` to `23/24` apart).

The `γ = 1` battled grids from #278 are **reused directly** as the `γ = 1` points of #279 (zero duplicate battles).

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
   - Over the shared error overlap `[e_lo, e_hi]` where both arm `U` and control `D` have usable Pareto envelopes (reported whenever `e_hi − e_lo ≥ 0.002`, else `no overlap`), compute `(1 / (e_hi − e_lo)) ∫_{e_lo}^{e_hi} [f_U^{(s)}(e) − f_D^{(s)}(e)] de` via linear interpolation along each arm's upper Pareto envelope. This gives the direct metric gain or loss at equal ask-versus-got error in the metric's own units.
3. **Highest usable guidance weight (`g_max`)**:
   - A guidance strength `g` is **usable** (`uniform_preflight84.is_usable_strength`) iff ALL THREE hold on asks `1/24` to `17/24` across all 3 sampling seeds:
     1. **Cell completion:** all 51 scored cells (`17 asks × 3 seeds`) fill 64 validator-accepted non-held-out teams within `MAX_TRIES = 1,280` (effective acceptance ≥ 5.0% in every scored cell).
     2. **Non-collapsed diversity:** within-ask mean `composition.unique_fraction ≥ 0.20` (at least `12.8` distinct line-ups per 64 teams) AND within-ask mean `composition.largest_share ≤ 0.50`.
     3. **Non-degraded error:** ask-versus-got error does not rise by more than `+0.005` above the arm's lowest error observed at lower guidance weights.
   - `g_max` is the largest tested guidance strength passing all three conditions.
4. **Verdict rule per frontier (#280):**
   - A uniform arm **beats** **Linear masked, full filter** on a frontier iff its dominated area `HV_U > HV_D` on all 3 sampling seeds, the 95% bootstrap interval of `HV_U − HV_D` lies above zero, AND on the equal-error overlap (when `e_hi − e_lo ≥ 0.002`) `Δm_eq > 0` on all 3 seeds.
   - It **loses** iff `HV_U < HV_D` on all 3 seeds with the 95% interval below zero.
   - Otherwise it **matches / is inconclusive**.
   - Any cross-model gap smaller than training luck (`0.0019` error shift, corresponding to `~0.0009` on novelty dominated area and `~0.0017` on within-ask uniqueness dominated area) is explicitly flagged as within single-seed training luck.

### 5. Revision measure (mechanism, step 7)

Logged inside the reverse sampling loop for all 84 fields across `z^{(0)}, z^{(1)}, ..., z^{(K*)}` (before validator redraws and before Stat Alignment repair), reported at every guidance strength, averaged per field type with **species reported separately** (`species`, `ability`, `item`, `move`, `Stat Alignment`, `Stat Points`):

1. **Value changes beyond the first:** `R_c = max(0, sum_{i=1}^{K*} 1[z^{(i)}_c ≠ z^{(i-1)}_c] − 1)`. **Linear masked, full filter** scores `0`.
2. **Fixed-predictor noise floor:** Exact deterministic expectation `E[max(0, C_c − 1)]` from Schiff et al. App. A Eq. 25 for a fixed one-hot predictor starting from uniform initialization over real vocabulary size `N_k`, on the log-linear schedule (`α(t) = 1 − t`, `eps = 1e-5`) and pass count `K*` (`uniform_preflight84.revision_noise_floor`):

| Field type | Real vocab `N_k` | Log-linear 84 passes | Log-linear 128 passes | Log-linear 256 passes |
| --- | ---: | ---: | ---: | ---: |
| Species | 347 | 4.000 | 4.418 | 5.107 |
| Ability | 200 | 3.989 | 4.406 | 5.094 |
| Item | 149 | 3.980 | 4.397 | 5.083 |
| Move | 497 | 4.004 | 4.422 | 5.112 |
| Stat Alignment | 26 | 3.822 | 4.225 | 4.889 |
| Stat Points | 33 | 3.862 | 4.269 | 4.939 |

3. **Changes of mind (headline):** `M_c = sum_{i=1}^{K*-1} 1[top^{(i)}_c ≠ top^{(i-1)}_c]`, where `top^{(i)}_c = argmax_{v} log x_θ(z^{(i)}, t_i, ask)[c, v]` (the conditional clean-field top prediction for `γ > 0`, and the unconditional clean-field top prediction for `γ = 0`). **Linear masked, full filter** scores `0`.

### 6. Coarse guidance grid and bend-point rule

Because **Linear masked, full filter** mixes clean-field log-probabilities while uniform discrete diffusion mixes reverse posteriors, weight scales are not comparable and every frontier is plotted against ask-versus-got error.

- **Coarse grid (all three arms):** `{1, 2, 4, 8}`, plus the unguided baseline (`γ = 0` / no ask, 192 teams × 49 = 9,408 battles).
  - For **Linear masked, full filter**, reuse the battled grids from #385 at `w ∈ {2, 4, 8}` (and any matching grids from diversity map #265); add `w = 1` if not yet battled on #385's checkpoint.
  - For the two uniform arms, `γ = 1` comes from #278; #279 adds `γ ∈ {2, 4, 8}` for each arm that passed #277's 5% legality floor.
- **Bend-point and extension rule (`uniform_preflight84.bend_points_to_add`):** After the coarse grid lands, add up to **2 extra guidance weights per arm** (strict cap, so at most 6 battled guidance strengths per arm):
  1. **Interior bend / usability bracket:** Between adjacent coarse weights `g_a < g_b`, insert geometric midpoint `round(sqrt(g_a × g_b), 1)` (i.e., `1.4`, `2.8`, or `5.7`) if: (a) `g_a` is usable and `g_b` is not usable (highest priority, bracketing `g_max`), OR (b) `|error(g_b) − error(g_a)| > 0.003`, OR (c) `|within_ask_unique_fraction(g_b) − within_ask_unique_fraction(g_a)| > 0.10`.
  2. **Upper extension:** If `g = 8` is usable AND `error(4) − error(8) ≥ 0.001`, add `g = 16.0`.
  3. If more than 2 candidates qualify, take the top 2 by priority (usability bracket first, then largest normalized error + diversity jump).

### 7. Other fixed parts of the plan

| Part | Value |
| --- | --- |
| Smallest effect worth acting on | `0.001` in ask-versus-got error (glossary rule for replacing the default model), or at equal error: `≥ 0.05` higher novelty, `≥ 0.05` higher within-ask line-up uniqueness (`+3.2` line-ups per 64 teams), or `≥ 0.25` higher within-ask mean species replacement distance. |
| Comparisons planned | Each uniform arm against **Linear masked, full filter**; **Log-linear uniform, one-direction filter** against **Log-linear uniform, no filter** on the same checkpoint. |
| Baselines | **Linear masked, full filter** (`w ∈ {1, 2, 4, 8}` + bend points); each arm's unguided (`γ = 0`, 192 teams) baseline; scorer controls (top-50 meta teams near `0.50`, uniformly drawn legal teams near `0.02`). |
| Stopping rule | Pass-count table (#277), unguided/conditional comparison (#278), coarse grid + up to 2 rule-selected bend points per arm (#279). No extra runs to chase a result. |
| Slope of got on ask | Not reported (owner, 2026-11-05). |

## Port-equality test

Required gate before training starts (#275 blocks #276): on one identical deterministic batch spanning all six field types (`species: 347, ability: 200, item: 149, move: 497, Stat Alignment: 26, Stat Points: 33`), at log-linear masking schedule (`α(t) = 1 − t`, `α′(t) = −1`, `zero_recon_loss = True`), our uniform loss implementation in #275 must equal the verbatim upstream loss of `kuleshov-group/discrete-diffusion-guidance@edb0f8c28b7caeb4ea7a06a2fee8d74ab6da1661` (`diffusion.py:563–614`), verified by `uniform_preflight84.port_equality_check`:

- **Forward value equality:** every `(batch, column)` loss value matches upstream `diffusion.py:572–597` to `atol = 1e-12, rtol = 1e-10` in `float64` and `atol = 1e-3, rtol = 1e-3` in `float32` across `t ∈ [1e-3, 1 − 1e-3]` (the `float32` margin reflects upstream's `1 − (1 − t)` mantissa rounding near `t = 1e-3` and `t = 1 − 1e-3`).
- **Gradient equality:** `∂loss / ∂logits` matches upstream to `atol = 1e-10, rtol = 1e-8` in `float64` and `atol = 1e-4, rtol = 1e-3` in `float32`.
- **Reverse posterior and CFG equality:** our reverse posterior matches upstream `_compute_posterior` (`diffusion.py:356–381`) and our uniform CFG distribution at `γ ∈ {0, 1, 2, 4}` matches `_cfg_denoise` (`diffusion.py:1298–1328`) to `1e-6` (`float32`) / `1e-12` (`float64`).
- **Numerical hygiene (`mps` `float32`):** compute `move_chance = t` directly (never `1 − (1 − t)`), `α(t) = (1 − t).clamp(min=1e-6)`, and in the reverse posterior `Δα = t − s` directly, keeping every loss, posterior, and gradient finite on Apple Silicon `mps`.

## Predicted outcome

Written 2026-11-24 before any uniform training or sampling begins, to be scored in #280:

| Stage / arm | Predicted outcome | Chance | Why |
| --- | --- | ---: | --- |
| Pass count (#277) | `K* = 128` or `256` passes is needed for **Log-linear uniform, one-direction filter** (`128` most likely: 55%; `256`: 35%; `84`: 10%). | 55% for `128`, 35% for `256` | Schiff et al.'s default is `128`; with `α(t) = 1 − t` on all fields, species keep updating late so smaller `dt` gives ability/item/moves more passes to settle after a late species edit. |
| **Log-linear uniform, one-direction filter** (`udlm_loglinear_filter`) | First-try legality `75%–92%` at `γ = 1`, ask-versus-got error `~0.020–0.035` at `γ = 1` falling to `~0.006–0.011` at `γ = 4–8`. Against **Linear masked, full filter**: **loses or matches** on overall dominated area (70% lose/match; 25% win on equal-error diversity in the mid-error overlap `0.006–0.016`), because uniform diffusion is harder to fit at 5.5M parameters / 80 epochs and late species flips under `α(t) = 1 − t` occasionally leave last-pass mismatches with set fields. | 70% lose/match overall; 25% win on equal-error diversity in overlap | Schiff et al. Table 1 shows UDLM trails MDLM in unconditional likelihood on small budgets, and the noise floor for log-linear species at 128 passes is `4.42` changes beyond the first. |
| **Log-linear uniform, no filter** (`udlm_loglinear_nofilter`) | Non-trivial first-try legality at `γ = 1` (`15%–50%`, well above `0%` thanks to #361 showing the 6-layer transformer learns slot binding by epoch 80), but far below the one-direction filter arm; legality drops sharply as `γ` rises, giving a lower `g_max` (`≤ 2–4`) and smaller dominated area than both **Linear masked, full filter** and **Log-linear uniform, one-direction filter**. | 85% lose to filtered sibling and control | #361 still left 33% illegal teams at epoch 80 under sequential masked decoding; simultaneous uniform updates with `eps = 1e-5` noise and no Stat Point / Item Clause / Species Clause mask add further violations. |
| Revision mechanism (step 7) | Raw value changes beyond the first sit near or modestly above the fixed-predictor noise floor (`~4.4–5.5` on species at 128 passes). **Changes of mind** (headline) are nonzero (`~2–6` flips on species, more on moves/items) and **fall as guidance `γ` rises** on species because sharper guided posteriors lock earlier. | 75% | Posterior-level CFG sharpens transition mass toward high-ask species early, reducing top-prediction flips as `γ` increases. |

**Largest surprise:** **Log-linear uniform, no filter** reaching `≥ 50%` first-try legality AND `≤ 0.005` ask-versus-got error while holding higher novelty/diversity than **Linear masked, full filter**; OR **Log-linear uniform, one-direction filter** dominating **Linear masked, full filter** across the entire error band `[0.004, 0.016]` on both novelty and within-ask species replacement distance.

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
| **Round trip, read teams, controls, condition, copies** | Run by `ask_grid84.py preflight` (and `verify-ask-grid84` smoke test in a private DB) with the uniform checkpoint and both samplers before #278/#279 grids; any failure stops the launch. | gated before #278/#279 |
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

The 3-seed ask-grid budget easily detects the smallest effects worth acting on within a checkpoint (`0.001` error, `0.05` novelty/uniqueness, `0.25` species replacements; seed-to-seed dominated-area spread is only `1.6%–7.5%`). Because the uniform model trains on one seed (`seed = 0`), cross-checkpoint effects smaller than training luck (`0.0019` error) remain exploratory.

**Compute budget (laptop only, queued detached behind any running jobs after `python3 ~/vgc-data/whats_running.py`; no Modal):**

| Step | Ticket | Grids / teams | Battles | Estimated laptop time |
| --- | --- | --- | ---: | --- |
| Train `udlm_loglinear` (cap 80, patience 3, scaling #361's 407.7 min on the same 214,055-team table) | #276 | 1 training | 0 | ~6.8 hours |
| Pass-count table: 2 uniform arms × 3 counts (`84, 128, 256`) at `γ = 1`, 3 seeds, asks `1/24..17/24` | #277 | 6 sampling sweeps | 0 | ~45–75 minutes |
| Unguided baselines (`γ = 0`, 192 teams × 49) for 2 uniform arms + control (if new) | #278 | 3 unguided sets | 28,224 | ~6 minutes |
| Conditional `γ = 1` ask grids (2 uniform arms + control `w = 1` if not in #385) | #278 | up to 3 grids | 649,152 | ~2.0 hours (~40 min/grid) |
| Coarse guidance sweep (`γ ∈ {2, 4, 8}`, reusing `γ = 1` from #278 and #385's control grids at `w ∈ {2, 4, 8}`) | #279 | up to 6 uniform grids | 1,298,304 | ~4.0 hours battles (+ sampling; 3 grids if no-filter short-circuits) |
| Bend-point / extension grids (at most 2 added weights per arm) | #279 | up to 6 grids | 1,298,304 | ~4.0 hours battles (upper bound) |
| **Total (worst case, both uniform arms usable + max 2 bend points each)** | | **up to 15 new grids** | **~3.3M battles** | **~6.8 h training + ~11 h sampling & battles** |

## What the later tickets take from this

| Ticket | What this pre-registration fixes for it |
| --- | --- |
| [#275 Write the uniform discrete diffusion training and sampler, and pass the port-equality test](https://github.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/issues/275) | Implement `udlm_loglinear` (`α(t) = 1 − t`) on real vocab sizes `(347, 200, 149, 497, 26, 33)` with no loss legality mask, `t ∈ [1e-3, 1 − 1e-3]`, pass `uniform_preflight84.port_equality_check` (`float64` & `float32`) + training smoke (init CE near `ln N_k`, 32-team overfit `≥ 40%`), implement Eq. 15 reverse posterior + Eq. 6–8 posterior-mixing CFG + one-direction filter + no-filter sampler logging the 3 revision numbers per pass, and wire `ask_grid84.py` with `grid_seeds.seed0` (never legacy `7300`). |
| [#276 Train the log-linear uniform discrete diffusion model on the laptop](https://github.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/issues/276) | Queue detached on laptop on `lossdata84_shuffled_only.npz` (`sha256: e58af768…`), patience 3, threshold `0`, cap 80, every epoch saved; checkpoint `stat84_udlm_loglinear_es3_cap80`. Post chosen epoch and validation loss when it lands. |
| [#277 Tabulate 84, 128 and 256 passes at guidance 1 for the two uniform discrete diffusion arms](https://github.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/issues/277) | Run 3 seeds × 17 scored asks (`1,088` teams/seed) at `γ = 1`, no battles; apply the seed-noise rule to pick `K*`, and apply the `< 5%` legality short-circuit check to **Log-linear uniform, no filter**. |
| [#278 Compare the arms unguided: validation loss bound, first-try legality and ask-versus-got error](https://github.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/issues/278) | Report validation loss bound, `γ = 0` unguided baselines, and `γ = 1` ask grids at `K*` (reused directly by #279). Run `ask_grid84.py preflight` and `verify-ask-grid84` smoke test first. |
| [#279 Sweep guidance on the uniform discrete diffusion arms](https://github.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/issues/279) | Run coarse grid `γ ∈ {2, 4, 8}` (reusing `γ = 1` and #385 control grids), evaluate `is_usable_strength` and `bend_points_to_add` (max 2 added weights per arm), log novelty, every `diversity_metrics.py` metric (within-ask & pooled), mean species replacement distance (within-ask & pooled), and the 3 revision numbers per field type. |
| [#280 Uniform discrete diffusion trade-off: smaller or bigger than the default model's, frontier by frontier?](https://github.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/issues/280) | Score every frontier with `dominated_area`, `equal_error_advantage`, `g_max`, and the 5-axis combined Pareto front; explain results with the 3 revision numbers; score the pre-registered predictions above. |

## Deviations

- **2026-11-24 (before any training or battle):** Dropped the per-field uniform masking schedule (`udlm_per_field_filter`, `udlm_per_field_nofilter`) per owner decision because no published uniform discrete diffusion paper uses per-field masking; locked **Linear masked, full filter** (`linear_masked_filter`) as the control arm.
