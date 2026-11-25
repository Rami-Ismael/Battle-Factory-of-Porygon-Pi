# Retrain the default checkpoint on the shuffled-only table without wrong-stone teams

Pre-registration, written 2026-11-24 before training, sampling, or battles start. Ticket: [Retrain the default checkpoint on the shuffled-only table without wrong-stone teams](https://github.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/issues/385), on the map [Masking-schedule map: per-field versus linear masking schedule on the 84-field model, and where our code departs from MD4 and LLaDA](https://github.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/issues/291). Hands its checkpoint to [Make the retrained shuffled-only checkpoint the default and drop the 48-field model](https://github.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/issues/64) and serves as the control arm of [Uniform discrete diffusion map](https://github.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/issues/270).

Mode: **exploratory** with respect to the 11 dropped wrong-stone training teams, because one training seed (`seed = 0`) cannot separate an 11-row data change from training luck (`0.0019` in ask-versus-got error and `0.026` in pooled win rate between two trainings of one recipe, `docs/position-embedding-split.md`). The run's primary purpose is principled data hygiene (owner, 2026-11-17: "I want the retrained to happen for the principle"): producing a clean default checkpoint trained on zero wrong-stone teams, with every trade-off measure measured on seeds of its own across guidance 2, 4, and 8.

## Terms

An **ask** is the win rate the model is told to aim for (`k/24` for `k = 1` to `23`). **Got** is the win rate its teams then score. The **ask-versus-got error** is the mean squared error between each ask and the mean got at that ask over asks `1/24` to `17/24` (`src/ask_line.py`); asks `18/24` to `23/24` are reported apart. Near the line means an error under `0.001`.

**Win rate** means one battle against each of the 49 distinct top-50 meta teams, with the behaviour-cloning policy on both sides and slots shuffled every battle. A **line-up** is a team's unordered set of six Pokémon species. **Novel share** is the share of generated teams whose six-species line-up appears in none of the labelled training teams of `lossdata84_shuffled_only.npz`.

The **linear masking schedule** hides every one of the 84 fields with probability `1 − α(t) = t` at time `t ∈ [0, 1]` during training (MD4 Appendix J.1; LLaDA Eq. 3), confirmed by the owner on 2026-11-24 on [Choose the default model from shuffled-slot checkpoints only](https://github.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/issues/382). The **legal Pokémon construction filter** zeroes the probability of values Regulation M-B rules forbid next to the team's visible fields (`lossdown.Tables` plus `stat_points.allowed_stat_points`) in both the loss and sampling. **Stat Alignment** is a Pokémon's nature; **Stat Points** are its six stat allocations (`0..32` each, sum at most `66`).

An **elite line-up** has a team scoring at least `0.531` (the top-50 meta average) in the grid; an **elite line-up that held** still reaches `0.531` when its best team is re-battled in a fresh private database with 4 games against each of the 49 distinct top-50 meta teams (`src/line_up_yield.py`, `all_23_asks` view). **Pass@k** (`k = 1..48`, asks `13/24` to `17/24`) is the unbiased chance that at least one of `k` drawn teams reaches `0.531` and belongs to an elite line-up that held (`src/pass_at_k.py`).

## What is already known

Searches across `code/docs/`, `code/docs/experiment-names.md`, and the issue tracker (2026-11-24) show the exact baseline this retrain replaces:

| Earlier result | What it measured | What it leaves open |
| --- | --- | --- |
| Masking-schedule rows (`#296`, `#297`, `#298`, `docs/masking-rate-rows.md`) | Trained `stat84_shuffled_noise_linear_es3_cap80` on the pre-#368 shuffled-only table (`214,598` labelled train before `Tables.check`, `4,380` val). `Tables.check` dropped 532 rows under random masks, so 11 wrong-stone teams reached training (`n_train = 214,066`) and 12 sat in validation. Reached cap 80 with `best_val_loss = 2.0142`. Battled at guidance 2, 4, 8 on legacy seed base `7300`: ask-versus-got error `0.0075`, `0.0049`, `0.0038`; pooled win rate `0.353`, `0.373`, `0.385`; elite line-ups held `105`, `163`, `198` (novel held `18`, `29`, `54`); Pass@8 `0.808`, `0.886`, `0.981`; Pass@48 `1.000`, `1.000`, `1.000`. | Trained on 11 wrong-stone teams (bad data per #368) and battled on legacy seed base `7300`, which replayed training teams (#353; unbiased non-replayed redraws gave novel share `33%` vs `18.3%` at guidance 2). |
| Wrong-stone table filter (`#368`, PR `#372`, 2026-11-17) | Dropped all 555 wrong-Mega-Stone teams from `lossdata84_shuffled_only.npz` without re-drawing the validation split (`sha256: e58af7689e62a873ce19361237d7ff8e77cb791fc4c196cc705410755c5a03e2`): `214,055` labelled training teams, `4,368` labelled validation teams, `0 / 0` `Tables.rejects` failures with every field visible. | No linear masked checkpoint with the full filter has yet been trained on this clean table. |

## Arms

Named with `.claude/skills/name-experiments/SKILL.md` (difference first, shared settings in the caption).

**Shared by both checkpoints:** 84-field transformer (`d = 256`, 6 layers, 8 heads, dropout `0.1`, per-field learned position vectors), linear masking schedule (`1 − α(t) = t` on all 84 fields), full legal Pokémon construction filter in loss and sampling, ask dropout `0.15`, label noise `±1/48` (exact on validation), AdamW (`lr = 4e-4`, weight decay `0.01`, `betas = (0.9, 0.98)`, 500 warm-up steps, cosine decay to 80 epochs, batch `256`, gradient clip `1.0`, EMA `0.999`), float32 training (`VGC_AMP=0`), training seed `0`, early stopping on validation loss (`patience = 3`, `threshold = 0`, `cap = 80`, every epoch saved), species-first decoding in 84 passes with Stat Alignment repair, and the same shuffled-slot battle scorer across guidance `2, 4, 8`.

| Training table, then checkpoint label | Code ID | Checkpoint | Wrong-stone teams in table (train / val) |
| --- | --- | --- | --- |
| Unfiltered table (Linear, species-first). The reference | `sched_linear_species_first` | `stat84_shuffled_noise_linear_es3_cap80` | 11 reached training (532 dropped by random-mask check) / 12 in validation |
| Filtered table (Linear masked, full filter) | `linear_masked_filter` | `stat84_shuffled_noise_linear_filtered_es3_cap80` | 0 / 0, verified with every field visible (`Tables.rejects`) before training starts |

## Contract

| Row | Value |
| --- | --- |
| Claim | Retraining the linear masked 84-field model (`Linear masked, full filter`) on the wrong-stone-filtered shuffled-only table produces a clean default checkpoint free of bad data while matching the unfiltered checkpoint (`stat84_shuffled_noise_linear_es3_cap80`) within training luck (`0.002` in ask-versus-got error, `0.026` in pooled win rate) across guidance `2, 4, 8`. |
| What refutes it | Any measure differing from the unfiltered checkpoint beyond training luck (`\|Δ error\| ≥ 0.002`, `\|Δ pooled win\| ≥ 0.026`, or a paired difference outside training luck on novelty, distinct line-ups, elite line-ups held, or Pass@k) is reported and investigated rather than assumed away. |
| What changes if it holds | `stat84_shuffled_noise_linear_filtered_es3_cap80` (`linear_masked_filter`) is handed to #64 to become the project's default checkpoint and to #270 as the confirmed **Linear masked, full filter** control arm. |
| How far it reaches | Ask-versus-got error, win rate, novelty, distinct line-ups, elite line-ups held, novel elite line-ups held, and Pass@k under the behaviour-cloning policy against the 49 distinct top-50 meta teams, slots shuffled, on one training seed. Does not generalize to other policies, human play, or held-out ranks 51–100 without a separate evaluation. |
| Fields | All 84 fields generated by the model: for each of six Pokémon, species, ability, item, four moves, Stat Alignment, and six Stat Points. Nothing copied or fixed. After sampling, `Model84.to_paste(..., repair=True)` replaces Stat Alignment to match the moves (`src/ask_grid84.py`). |
| Vocabulary | `team84.regulation_vocab()` (Regulation M-B snapshot stored in the checkpoint): 348 species, 201 abilities, 150 items, 498 moves, 27 Stat Alignments, and 34 Stat Point tokens (0 = `[MASK]`, 1–33 = values 0–32). |
| Training teams | Filtered shuffled-only table `lossdata84_shuffled_only.npz` (`sha256: e58af7689e62a873ce19361237d7ff8e77cb791fc4c196cc705410755c5a03e2`). Labelled teams only: **214,055 training teams and 4,368 validation teams** (seeded 2% split with seed `0`, minimum 49 shuffled-slot battles per team). Excluded: 10 unlabelled teams, 173,544 non-shuffled-only teams, all 555 wrong-Mega-Stone teams, the 500 test teams, and the 50 held-out meta teams (ranks 51–100). Verified `0 / 0` `Tables.rejects` failures with every field visible before training starts. |
| Scorer | `src/ask_grid84.py` (`matchup_db.score_pastes`): 1 battle against each of the 49 distinct top-50 meta teams per generated team, behaviour-cloning policy on both sides, slots shuffled (`require_shuffled`). Each guidance grid (`linear_masked_filter`, `linear_masked_filter_g4`, `linear_masked_filter_g8`) uses its own name-hashed sampling seed base via `src/grid_seeds.py` (`≥ 100,000`, never legacy `7300`), cell sampling seeds `s ∈ {1, 2, 3}`, battle seed `seed0 + s`. |
| Search region | No search loop. Asks `1/24` to `23/24`, sampling seeds `1, 2, 3`, `64` Showdown-legal non-held-out teams per cell (`MAX_TRIES = 1,280`), at each of guidance `2, 4, 8`: `23 × 3 × 64 = 4,416` teams and `216,384` battles per guidance (`649,152` grid battles total), plus fresh 4-game-per-opponent elite line-up re-battles (`src/line_up_yield.py`) in a private database and Pass@k (`src/pass_at_k.py`). |

### Settings

| Setting | Role | Value | Source |
| --- | --- | --- | --- |
| Training table wrong-stone filter | Scientific (only changed setting) | Filtered (`0 / 0` `Tables.rejects`, `214,055` train / `4,368` val) vs unfiltered (`11` train / `12` val wrong-stone rows) | Issue #368, PR #372 |
| Guidance | Swept to trace trade-off curve | `2, 4, 8` | Issue #385, matching #297 |
| Sampling seed | Nuisance | `1, 2, 3` on each grid's own SHA-256 `grid_seeds.seed0` base (never legacy `7300`) | Issue #353 anti-replay rule |
| Masking schedule | Fixed | Linear (`1 − α(t) = t` on all 84 fields, `sched = "1,1,1,1,1"`) | Owner 2026-11-24 (#382); MD4 App. J.1; LLaDA Eq. 3 |
| Network, optimizer, precision, stopping | Fixed | `Net84` (`d=256, nlayer=6, nhead=8, dropout=0.1, pos="per_field"`), AdamW (`lr=4e-4, wd=0.01, betas=(0.9, 0.98)`), 500 warm-up + cosine decay to 80 epochs, batch `256`, grad clip `1.0`, EMA `0.999`, float32 (`VGC_AMP=0`), ask dropout `0.15`, label noise `±1/48`, seed `0`, early stopping `patience=3, threshold=0.0, cap=80`, every epoch saved | Exact match to `stat84_shuffled_noise_linear_es3_cap80` (`src/train84.py`) |
| Decoding & legality | Fixed | Species-first, 84 passes, full legal Pokémon construction filter in loss and sampling, temperature `1`, Stat Alignment repair | Exact match to `sched_linear_species_first` |

### Plan, noise floor, and budget

- **Primary comparison:** trade-off table comparing **Filtered table (`linear_masked_filter`)** against **Unfiltered table (`sched_linear_species_first`)** at each of guidance `2, 4, 8` on:
  1. Ask-versus-got error (`1/24` to `17/24`, with `18/24` to `23/24` apart), paired opponent bootstrap (`4,000` draws, seed `2971`), per-seed signs, glossary bar `0.001`, training-luck bar `0.002`.
  2. Pooled win rate (all 23 asks) and top-ask win rate (`15/24` to `17/24`), training-luck bar `0.026`.
  3. Novel share and distinct line-ups (pooled over 23 asks and within-ask mean), compared primarily against the unfiltered checkpoint's unbiased non-replayed seed redraws (`seed940135810`, `seed630821980`, `seed111619920`) so legacy seed-7300 replay does not create a false novelty jump, with the battled 7300 grid numbers noted beside them.
  4. Elite line-ups that held and novel elite line-ups that held (`src/line_up_yield.py`, `all_23_asks`, 4 fresh games per meta team).
  5. Pass@1, Pass@8, Pass@48, and Potential@48 (`src/pass_at_k.py`, asks `13/24` to `17/24`, held definition).
  6. First-try legality and effective species (5-team subsets).
- **Noise floor and budget:** One 3-seed grid (`4,416` teams, `216,384` battles) resolves ask-versus-got error to roughly `±0.0005` within one checkpoint, well below the `0.001` glossary bar and `0.002` training-luck bar. Because only one training seed runs (`seed = 0`), differences inside training luck (`< 0.002` error, `< 0.026` pooled win) cannot be attributed to the 11 dropped rows; any difference exceeding training luck is flagged and reported explicitly.
- **Stopping rule:** Train to early stopping (`patience = 3`, `threshold = 0`, `cap = 80`), keep `best_epoch`, verify via `ask_grid84` smoke check and preflight probes, then run exactly one 3-seed grid + 4-game elite re-battle + Pass@k read at each of guidance `2, 4, 8` with zero cell shortfalls (`69/69` cells with `64` teams each).
- **Compute budget (laptop):** ~7.2 h training (80 epochs × ~5.4 min/epoch in float32 on MPS) + ~2.7 h for 3 grids and 3 re-battles (~10 h total, detached queue in `~/vgc-data/retrain_default_linear_filtered/`).

### Expectations

Written before any run:

- **Training & validation curves:** Reach the 80-epoch cap (or stop very near it) with `dropped_by_tables = 0` and validation loss slightly below `2.0142` (primarily because the 12 dropped wrong-stone validation rows had high loss, plus minor seed/batch-order shift from dropping 11 training rows).
- **Trade-off measures at guidance 2, 4, 8:** Ask-versus-got error, win rate, unbiased novel share, distinct line-ups, novel elite line-ups held, and Pass@k all match the unfiltered linear checkpoint within training luck (`|Δ error| < 0.002`, `|Δ pooled win| < 0.026`). Total elite line-ups held may read slightly lower than the unfiltered checkpoint's legacy 7300 battled grids (`105, 163, 198`) because legacy seed 7300 replayed strong training line-ups whereas `linear_masked_filter` draws on fresh seeds of its own; novel elite line-ups held (`18, 29, 54`) removes that replay bias.
