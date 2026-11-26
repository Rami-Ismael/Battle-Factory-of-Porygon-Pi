# Cosine learning rate, no tables in the loss: does it beat both parents on held-out nats?

Pre-registered 2026-11-25, before the new model trains. Issue: [Train Cosine learning rate, no tables in the loss: does it beat both parents on held-out nats?](https://github.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/issues/451). Run type: **confirmatory** for the nats comparison against the two parents, **exploratory** for the rule-by-rule curves and the interaction read. Nothing of the new model has been seen; a 1-epoch smoke on 512 teams ran 2026-11-25 19:51 (below).

On 2026-11-25 the owner made no tables in the loss the default for new 84-field trainings (commit `e2e66109`). The two settings it combines have each been trained, never together. With the fourth corner, the four models make a 2 × 2 on the same table and seed.

## Arms

Two words name each model. **Table-free** means the legal Pokémon construction tables are left out of the training loss (#361). **Table-masked** means the loss masks every forbidden value, the old default. The **5e-4 recipe** is peak 5e-4, 1,000 warm-up steps, cosine to a 5e-5 floor (#431, the GIDD paper's). The **4e-4 recipe** is 4e-4, 500 warm-up steps, cosine to 0. Sampling is the same for all four. "Tables on" and "tables off" below always mean the sampler's mode in the nats check, never the loss.

| Reader name | Code ID | Status |
| --- | --- | --- |
| Table-free 5e-4 recipe (the issue's "Cosine learning rate, no tables in the loss") | `shuffled_linear_filtered_gidd_lr_no_table_loss` | trains now, on Modal |
| Table-masked 5e-4 recipe (parent) | `shuffled_linear_filtered_gidd_lr` | trained 2026-11-25 (#431), 80 of 80 epochs |
| Table-free 4e-4 recipe (parent) | `shuffled_linear_no_table_loss` | trained 2026-11-24 (#361 rerun), 80 of 80 epochs |
| Table-masked 4e-4 recipe (fourth corner) | `shuffled_linear_filtered` | trained 2026-11-25 (#385), 80 of 80 epochs |

Shared recipe (`src/train84.py`): the labelled teams of the filtered shuffled-only table, the linear masking schedule, network 256 wide with 6 layers and 8 heads, batch 256, label noise 1/48, seed 0, patience 3 on validation loss with any new lowest validation loss resetting it, a cap of 80 epochs, every epoch saved. Training drops no row: the filtered table holds no team the tables reject.

## Contract

| Row | Value |
| --- | --- |
| Claim | The table-free 5e-4 recipe gives held-out teams fewer nats than each parent, tables on. Refuted against a parent if the paired 95% interval of new minus parent lies wholly above −1 nat per team (no gain worth acting on). It says nothing about win rate, novelty or diversity: no ask grid, no battles |
| Fields | All 84 generated: per Pokémon its species, ability, item, four moves and Stat Alignment, plus six Stat Points. None copied, none fixed. Round trip: #361's preflight, 20 of 20 training teams identical through encode, paste, parse and encode (same vocabulary and table family); rerun on the final checkpoint |
| Vocabulary | The regulation snapshot in each checkpoint (`team84.regulation_vocab`), [MASK] at index 0: 348 species, 201 abilities, 150 items, 498 moves, 27 Stat Alignments, Stat Points 0 to 32. Shared by all four models |
| Training teams | `results/lossdata84_shuffled_only.npz` after the wrong-stone filter, sha256 `e58af768…`: 214,055 labelled training teams, 2% validation split with seed 0. Labels from shuffled-slot battles only. Same file and count in all four run JSONs |
| Scorer | The nats check (`src/nats_check.py`, `docs/nats-check.md`): exact held-out log-likelihood under the species-first sampler on 1,024 held-out teams (seed 20261114), tables on and off, plus 256 teams drawn with the tables and the Stat Point budget off for first-try legality under Showdown's validator. `score` refuses a model that trained on any scored team |
| Search region | No search. Per epoch during training: 64 teams at the elite cut, 0.531, guidance 2, temperature 1, species-first, draw seed 12345, drawn without the tables or the Stat Point budget |
| Settings | Scientific: the loss (against the table-masked 5e-4 recipe) and the learning-rate recipe (against the table-free 4e-4 recipe). Fixed: everything in the shared recipe. Not varied: the training seed, one training per model |
| Plan | See below |

## Measures

- **Nats per team, tables on** (primary): mean over the 1,024 held-out teams, tables-rejected teams left out.
- **Nats per team, tables off**, by field type in both modes, the 16-order gap and moves-first gap (secondary).
- **First-try legality, tables off**, 256 teams (secondary). The table-free 4e-4 recipe's 60.9% is the number to keep.
- **Rule by rule during training** (owner, 2026-11-25, at launch: "Make sure we track the rule table breaking as we trained the model"): after every epoch, the share of the 64 drawn teams that break each rule in `epoch_legality.RULES` (an item used more than once, a wrong Mega Stone, another item the species can't hold, a species used more than once, an ability the species can't have, a move the species can't learn, the same move twice on one Pokémon, Stat Points over 66), with the illegal share, errors per team and copied share. Logged in the run JSON's `curves`, the training log and Weights & Biases (`rule_shares/<rule>`). Compared by epoch with the table-free 4e-4 recipe's own training curves. Exploratory.
- **Validation loss** per epoch on the model's own loss. Not comparable with the table-masked models', whose loss masks the forbidden values.

## Plan

- **Primary comparison:** new minus parent, nats per team tables on, paired bootstrap over the 1,024 teams (2,000 resamples), once against each parent. "Better than both" needs both intervals wholly below −1. An interval inside ±1 reads "matches". Anything else is undecided.
- **Smallest effect worth acting on:** 1 nat per team, a third of the 3.0-nat gap between the two parents tables on. Chosen by this session, not the owner.
- **Predictions, written before training:**
  1. Tables on: at or below the table-masked 5e-4 recipe's 98.51. Each change cut nats on its own from the fourth corner (106.59 → 98.51 and 106.59 → 101.52), so an additive 2 × 2 predicts 93.4.
  2. Tables off: near the table-free 4e-4 recipe's 103.17, not the additive 65.1. Tables-off nats cannot sit far below tables-on nats for legal held-out teams, so the two changes must interact here.
  3. First-try legality at or above 60.9%, and rule shares falling over epochs as the table-free 4e-4 recipe's did, with "Stat Points over 66" the last to fall (0.19 at its epoch 80).
  4. 16-order gap tables on near the table-free 4e-4 recipe's +3.30.
- **Stopping rule:** early stop (patience 3 on validation loss) or the cap of 80, whichever comes first. Every other model reached the cap with its best epoch at 80, so expect 80. The best epoch's weights are scored.
- **Baselines:** the other three models, already scored on the same 1,024 teams (`docs/nats-check.md`, results 2026-11-25).
- **Budget and cost:** training on one Modal L4 or A10G (the launcher's list). The table-masked 5e-4 recipe took 72.4 minutes; #431's two trainings cost $3.74 together, so about $2. The nats check runs on the laptop, about 8 minutes, no battles.
- **Result:** the three-model table with paired intervals, the 2 × 2 of nats, and a per-epoch rule-by-rule chart, posted on #451.
- **Run:** training launched 2026-11-25 20:07 on a Modal A10 (`scripts/train84_modal.py`, commit `e2b8f67b`). The queue `~/vgc-data/cosine_no_tables/nats_after_training.zsh` starts the nats check only after training ends (owner, 2026-11-25): it waits for the run JSON on the Modal volume, fetches the checkpoint, waits for an idle laptop, runs `nats_check.py score` and `report` from a frozen copy of `src/`, and draws the figures (`src/cosine_no_tables_report.py`). It posts nothing; the session judges and posts.

## Storyboard

| Frame | Reader takes away | Expected | Would surprise |
| --- | --- | --- | --- |
| 1. Nats tables on, 2 × 2 of recipe by loss, colored cells with intervals | Which corner fits held-out teams best, and whether the two changes add | Table-free 5e-4 the lowest, below 98.5 | It sits above either parent |
| 2. New minus each parent, tables on and off, intervals against the ±1 band | Does it beat both, by how much | Both tables-on intervals below −1 | An interval crossing −1 |
| 3. Rule shares by epoch, one panel per rule, table-free 5e-4 against table-free 4e-4 | How fast each rule is learned, and whether the 5e-4 recipe learns it sooner | The 5e-4 curves at or below the 4e-4 curves from the warm-up on | A rule that rises late in training |

## Preflight

| Probe | Result | Pass |
| --- | --- | --- |
| Training smoke: 1 epoch on 512 teams, 2026-11-25 19:51 (`results/stat84smoke_…_no_table_loss_es3_cap80.json`) | `table_loss` False, `legality_tables` False, `lr_schedule` gidd_cosine, peak 5e-4, floor 5e-5, warm-up 1,000; train loss 4.95, validation 4.14; the per-epoch check logged illegal share 1.0, errors per team 19.1 and all 8 rule shares | Yes |
| Round trip | Same vocabulary and table family as #361's preflight, 20 of 20 identical; rerun on the final checkpoint | Yes, inherited |
| Coverage | 214,055 training rows, 0 dropped by the tables, same sha256 as the other three models | Yes |
| Scorer checks | The nats check's six checks (`docs/nats-check.md`) run inside `score` and `report`; `score` writes nothing if checks 3 or 4 fail | Yes, built in |
| Controls | The other three models are scored with the same scorer and teams; the 2026-11-23 numbers reproduce to 0.05 nats | Yes |
| Invariance, condition, separation | Not reachable: no ask is varied, no battles | Skipped, for those reasons |

## Noise floor

- **Held-out teams:** paired bootstrap intervals in the 2026-11-25 results run about ±0.5 to ±1 nat per team tables on (e.g. −25.54 [−26.56, −24.51]). A 1-nat difference is detectable on 1,024 teams.
- **Training seed:** one training per model, so a gap mixes the change with the seed's effect, which the interval does not cover. The owner decided on 2026-11-17 not to spend compute on second seeds for matched recipes; a gap within 1 nat reads as "matches", not "better".
- **Rule shares:** standard error `sqrt(p(1-p)/64)`, at most 0.0625 per epoch. Fixed draw seed, so epoch-to-epoch moves are model changes. Pool 10 epochs (640 teams) to compare the two table-free models.

## Deviations

- 2026-11-25, before any result: the child-minus-parent figure kept only the tables-on rows. The tables-off gap against the table-masked 5e-4 recipe (about −60 nats) flattened the axis; the tables-off numbers stay in the 2 × 2 and the comment's table.
- 2026-11-25, after the result: the three figure titles were rewritten to state the measured finding. No number, interval or verdict rule changed.

## Result

Reported 2026-11-25 on [#451](https://github.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/issues/451#issuecomment-7181823954); #451 closed, line added to map #291. Figures: `docs/figures/cosine_no_tables/` (`src/cosine_no_tables_report.py`). Numbers: `~/vgc-data/results/nats_check/report.json` and the run JSON.

| Model | Nats, tables on | Nats, tables off | First-try legality, tables off |
| --- | --- | --- | --- |
| Table-masked 5e-4 recipe | **98.51** | 162.73 | 0.0% |
| Table-free 4e-4 recipe | 101.52 | 103.17 | 60.9% |
| Table-free 5e-4 recipe | 99.27 | **100.62** | **68.8%** |

- **Primary, tables on, new minus parent:** against the table-free 4e-4 recipe −2.26 [−2.46, −2.04], better beyond the 1-nat margin. Against the table-masked 5e-4 recipe +0.75 [+0.44, +1.08], worse; the interval sits mostly inside the margin and its top end leaves it. The claim is refuted against the table-masked 5e-4 recipe: the new model does not beat both parents.
- **Predictions:** 1 failed (99.27, not at or below 98.51; the additive 93.4 missed by 5.9). 2 held (100.62 with the tables off, near 103.17). 3 held in direction: legality 68.8% against 60.9%, +7.9 points [−0.3, +16.1] at 256 teams each, undecided; rule shares fell, and Stat Points over 66 fell last. 4 held (+3.25 [+2.84, +3.67] against +3.30).
- **Where the 0.75 nats go:** moves +0.75, items +0.21, species +0.15; Stat Points −0.22 and Stat Alignment −0.17.
- **Rules during training (exploratory):** both table-free models bind fields to their own Pokémon at the same time (ability rule below 50% at epoch 47 against 46; illegal share below 0.875 at epoch 48 for both). Over epochs 71–80: 26% illegal against 37%; species used more than once 3% against 9%; Stat Points over 66 17% against 20%. Every epoch draws from the same seed, so these 640 draws per model are correlated.
- **Run checks:** 214,055 training teams, sha256 `e58af768…`, seed 0, 80 of 80 epochs, best epoch 80, validation loss 1.2159; training loss fell every epoch; round trip 20 of 20 on the final checkpoint; 64 training teams through the validator 0 illegal; scorer checks 1.8e-6 and 4.5e-6 nats (limit 1e-3); no scored team is a training row. Modal A10, 69 minutes, $2.24.
- **What it can't show:** one training seed per model, so training luck is outside the intervals; nats rank fit to held-out teams, not win rate (no ask grid, no battles).
