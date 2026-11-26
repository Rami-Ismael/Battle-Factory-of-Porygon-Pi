# Nats check: does a new masked model find real held-out teams likely?

`src/nats_check.py` scores an 84-field masked model by the exact log-likelihood it gives held-out teams, in nats per team. A nat is the unit of −log p with the natural logarithm; a team that costs 124 nats has probability e^−124 under the model. Lower is better. The check needs no battles and takes about 8 minutes per checkpoint on the laptop.

```
python src/nats_check.py selftest                      # the scorer on a toy distribution
python src/nats_check.py score linear_filtered         # a code ID in MODELS, or a checkpoint name
python src/nats_check.py report                        # tables, intervals, references, rank agreement, chart
```

`score` writes `<code ID>.json` and `<code ID>.npz` (per-team, per-field nats) to `~/vgc-data/results/nats_check/`. `report` reads every scored file there and writes `report.json`, `report.md` and `nats-by-field-type.html`.

## What it measures

A masked model that fills one field per pass in a fixed order gives every team an exact probability: the product of the probabilities its sampler gave each true value when that field's turn came. The check shows the model a held-out team with every field hidden, reads the probability the sampler would give the first field's true value, reveals that field, and repeats for all 84 fields. Each field's probabilities are renormalised over the values the sampler may draw, as `sampler84.sample_species_first` does. The sum of −log p is the team's nats. Nothing is bounded or sampled, which is the difference from the validation loss, a bound that averages over random masks.

| Part | Definition |
| --- | --- |
| Held-out teams | 1,024 validation rows of `lossdata84_shuffled_only.npz` that are not training rows, drawn with seed 20261114. The 2026-11-15 and 2026-11-23 probes drew the same teams, so their numbers stay comparable. `score` finds the table each checkpoint trained on by its SHA-256 and refuses to score if any of the teams is one of its training rows. |
| Fixed fill order | Species-first, `diffusion84.ORDER`: six species, then abilities and items, moves, Stat Alignment, Stat Points. Every ask grid samples in this order. |
| Tables on | The sampler with the legal Pokémon construction tables: the tables' values, without `[MASK]` and without an empty first move. |
| Tables off | The sampler without the tables. It also drops the Stat Point budget, and can draw every value but `[MASK]`. |
| Reference convention | The 2026-11-15 and 2026-11-23 probes read the model's probabilities before the sampler renormalises them, and kept the Stat Point budget with the tables off. Each score stores this beside the sampler's number, only to reproduce the old numbers. |
| Per field | Nats per team for each field type (species, ability, item, move, Stat Alignment, Stat Points) and per column. |
| Per team | Each team's total, stored in the `.npz`. |
| Tables-rejected team | A team holding a value the tables-on sampler cannot draw when its turn comes, so its probability is 0. These teams are listed apart and left out of the fixed-order means of both modes. A team only another order rejects is named and left out of that order's gap alone. |
| Moves-first gap | Nats with every move filled first, minus nats in the fixed order. The 2026-11-15 reference measure. |
| 16-order gap | The first 256 teams scored in 16 random fill orders (seed 20261125). Gap = mean nats over the 16 orders minus nats in the fixed order, per team. A model that is one joint distribution has gap 0 in every order. |
| Interval | Paired bootstrap: 2,000 resamples of the per-team difference, 2.5% and 97.5%. |
| First-try legality | 256 teams drawn with the tables and the Stat Point budget off (elite cut 0.531, guidance 2, species-first); the share Showdown accepts on the first try. |

A uniform discrete diffusion model starts from random values and has no hidden fields to fill in an order, so it has no fixed-order likelihood. `score` refuses it by name.

## Checks on the scorer

1. **Known entropy.** `selftest` builds a toy joint distribution of four correlated columns (72 outcomes) with exact conditionals. The probability-weighted mean nats must equal the entropy, in the fixed order and in 16 random orders, to 1e-9.
2. **A broken scorer fails.** A scorer that ignores the revealed fields must pay the sum of the columns' own entropies, which is higher. This proves the entropy check can fail.
3. **Per-field terms sum to the team total.** On the toy, each outcome's terms must sum to −log of its joint probability. On a real model, 8 teams per table mode are totalled a second way (one batch holds a single team's 84 partly filled states) and must match the per-field sum to 1e-3 nats.
4. **The scorer matches the sampler.** The sampler draws 8 teams per table mode and reports the probability it gave each value it drew. The scorer must give those teams the same nats, field by field, to 1e-3. This is the check that the masks and the token lookup are the sampler's own. `score` writes nothing when check 3 or 4 fails.
5. **Old numbers reproduce.** The three checkpoints the 2026-11-23 probe scored are scored again; under the reference convention each must land within 0.05 nats per team of the stored number.
6. **Same teams in every file.** `report` refuses to pair files whose team hash or order settings differ.

`scripts/test_nats_check.py` runs checks 1 to 3 on the toy, and tests the order gap, the bootstrap interval, the rank agreement, the held-out check and the report's refusals. It needs no data and runs on every pull request. Checks 3 to 5 on a real model run inside `score` and `report`, on the laptop.

The 48-field module `diffusion.py`, which fixes the first 48 columns of the fill order, still loads from `/tmp/vgc-pilot/src` through older imports, and that copy differs from this checkout's file. Each score records the path it loaded. The modules the score computes with (`diffusion84`, `sampler84`, `train84`, `lossdown`) must load from the checkout, or `score` stops.

## Reading the result

Nats rank models by fit to held-out teams. They do not measure win rate. On 2026-10-07 the loss-down retrain cut nats by 40% and its win rate had to be battled separately. The report's last table sets the nats ranking against first-try legality, win rate and novel share for the models that have each number, and says where the rankings agree.

## Results, 2026-11-25

Seven checkpoints, scored on the laptop; the Table-free 5e-4 recipe (issue #451, `docs/cosine-lr-no-tables-in-the-loss.md`) was added the same evening. Chart: `docs/figures/nats_check/nats-by-field-type.html`. Data: `~/vgc-data/results/nats_check/`.

Exact held-out log-likelihood under the species-first sampler, in nats per team. Lower is better. Teams: 1,024 held-out teams; means leave out tables-rejected teams.

### Is the scorer right?

- Toy distribution, 72 outcomes: entropy 3.689052 nats, scorer 3.689052; worst of 16 random orders off by 4.4e-16; order gap 3.6e-15. Passes.
- A scorer that ignores revealed fields pays 4.1779 nats, the sum of the columns' own entropies (4.1779), so the check can fail.
- Per-field terms against team totals computed by a second route (8 teams per model and table mode): largest gap 2.9e-06 nats (limit 1e-03).
- Scorer against the sampler's own probability for each value it drew (8 sampled teams per model and table mode): largest gap 7.6e-06 nats per field (limit 1e-03).
- Did any model train on a scored team? Per-field masking schedule: held out; Linear masking schedule: held out; No tables in the loss, epoch 33: held out; Wrong-stone teams filtered: held out; Cosine learning rate, wrong-stone teams filtered: held out; No tables in the loss, 80 epochs: held out; Table-free 5e-4 recipe: held out.

### Nats per team, tables on and off

| Model | Tables on | Tables off | Rejected teams | First-try legality, tables off |
|---|---|---|---|---|
| Per-field masking schedule | 124.05 | 222.73 | 0 | 0.0% |
| Linear masking schedule | 127.15 | 213.62 | 0 | 0.0% |
| No tables in the loss, epoch 33 | 135.21 | 167.83 | 0 | 0.0% |
| Wrong-stone teams filtered | 106.59 | 200.79 | 0 | 0.0% |
| Cosine learning rate, wrong-stone teams filtered | 98.51 | 162.73 | 0 | 0.0% |
| No tables in the loss, 80 epochs | 101.52 | 103.17 | 0 | 60.9% |
| Table-free 5e-4 recipe | 99.27 | 100.62 | 0 | 68.8% |

### Nats per team by field type, tables on

| Model | species | ability | item | move | Stat Alignment | Stat Points | Total |
|---|---|---|---|---|---|---|---|
| Per-field masking schedule | 19.98 | 1.35 | 11.74 | 27.04 | 8.39 | 55.56 | 124.05 |
| Linear masking schedule | 20.11 | 1.42 | 11.99 | 28.15 | 8.52 | 56.96 | 127.15 |
| No tables in the loss, epoch 33 | 20.37 | 1.57 | 13.08 | 32.69 | 8.76 | 58.75 | 135.21 |
| Wrong-stone teams filtered | 20.39 | 1.49 | 12.24 | 28.17 | 6.91 | 37.38 | 106.59 |
| Cosine learning rate, wrong-stone teams filtered | 20.20 | 1.43 | 12.01 | 27.01 | 5.86 | 32.01 | 98.51 |
| No tables in the loss, 80 epochs | 20.41 | 1.48 | 12.25 | 28.24 | 5.89 | 33.25 | 101.52 |
| Table-free 5e-4 recipe | 20.35 | 1.46 | 12.22 | 27.76 | 5.69 | 31.79 | 99.27 |

### Nats per team by field type, tables off

| Model | species | ability | item | move | Stat Alignment | Stat Points | Total |
|---|---|---|---|---|---|---|---|
| Per-field masking schedule | 20.83 | 25.96 | 21.79 | 73.69 | 8.39 | 72.07 | 222.73 |
| Linear masking schedule | 20.47 | 27.50 | 19.79 | 63.54 | 8.52 | 73.81 | 213.62 |
| No tables in the loss, epoch 33 | 20.45 | 8.91 | 14.20 | 52.42 | 8.76 | 63.10 | 167.83 |
| Wrong-stone teams filtered | 20.87 | 26.25 | 17.26 | 51.43 | 6.91 | 78.07 | 200.79 |
| Cosine learning rate, wrong-stone teams filtered | 20.55 | 13.54 | 15.30 | 44.88 | 5.86 | 62.61 | 162.73 |
| No tables in the loss, 80 epochs | 20.55 | 1.49 | 12.33 | 29.26 | 5.89 | 33.64 | 103.17 |
| Table-free 5e-4 recipe | 20.48 | 1.47 | 12.30 | 28.52 | 5.69 | 32.16 | 100.62 |

### Order gaps

Nats in the other order minus nats in the species-first order, per team. Positive = species-first scores the team better. A single joint distribution has gap 0. Random orders: 16, on the first 256 teams.

| Model | 16-order gap, tables on | 16-order gap, tables off | Moves-first gap, tables on | Moves-first gap, tables off |
|---|---|---|---|---|
| Per-field masking schedule | +15.57 [+14.54, +16.69] | +4.90 [+2.55, +7.03] | +18.15 [+17.59, +18.78] | -7.28 [-7.97, -6.60] |
| Linear masking schedule | +8.56 [+7.76, +9.32] | +8.63 [+6.55, +10.55] | +12.46 [+12.08, +12.85] | -3.55 [-4.01, -3.06] |
| No tables in the loss, epoch 33 | +9.95 [+9.35, +10.56] | +0.96 [+0.59, +1.31] | +8.74 [+8.41, +9.08] | +1.71 [+1.52, +1.92] |
| Wrong-stone teams filtered | +5.36 [+4.78, +5.93] | +1.42 [-0.83, +3.80] | +5.12 [+4.79, +5.43] | -8.85 [-9.32, -8.38] |
| Cosine learning rate, wrong-stone teams filtered | +4.91 [+4.47, +5.37] | +3.81 [+2.11, +5.66] | +3.53 [+3.27, +3.82] | -3.73 [-4.08, -3.35] |
| No tables in the loss, 80 epochs | +3.30 [+2.89, +3.73] | +3.10 [+2.72, +3.51] | +2.70 [+2.47, +2.93] | +3.11 [+2.86, +3.35] |
| Table-free 5e-4 recipe | +3.25 [+2.84, +3.67] | +3.03 [+2.65, +3.43] | +2.23 [+2.01, +2.46] | +2.53 [+2.31, +2.76] |

### Tables-rejected teams

None in the fixed order, for any model.

### Each new model against the rescored references

New minus reference, nats per team, paired bootstrap 95%. Negative = the new model is better.

| New model | Reference | Tables on | Tables off |
|---|---|---|---|
| Wrong-stone teams filtered | Per-field masking schedule | -17.46 [-18.45, -16.48] | -21.93 [-24.03, -19.89] |
| Wrong-stone teams filtered | Linear masking schedule | -20.56 [-21.41, -19.70] | -12.83 [-14.49, -11.18] |
| Wrong-stone teams filtered | No tables in the loss, epoch 33 | -28.62 [-29.37, -27.81] | +32.96 [+31.17, +34.67] |
| Cosine learning rate, wrong-stone teams filtered | Per-field masking schedule | -25.54 [-26.56, -24.51] | -59.99 [-62.91, -57.24] |
| Cosine learning rate, wrong-stone teams filtered | Linear masking schedule | -28.64 [-29.59, -27.71] | -50.89 [-53.37, -48.49] |
| Cosine learning rate, wrong-stone teams filtered | No tables in the loss, epoch 33 | -36.70 [-37.59, -35.80] | -5.10 [-7.38, -2.81] |
| No tables in the loss, 80 epochs | Per-field masking schedule | -22.53 [-23.77, -21.28] | -119.56 [-121.59, -117.54] |
| No tables in the loss, 80 epochs | Linear masking schedule | -25.63 [-26.75, -24.49] | -110.45 [-112.06, -108.75] |
| No tables in the loss, 80 epochs | No tables in the loss, epoch 33 | -33.69 [-34.68, -32.66] | -64.66 [-65.66, -63.64] |
| Table-free 5e-4 recipe | Per-field masking schedule | -24.79 [-25.96, -23.59] | -122.11 [-124.12, -120.07] |
| Table-free 5e-4 recipe | Linear masking schedule | -27.89 [-28.97, -26.77] | -113.00 [-114.60, -111.33] |
| Table-free 5e-4 recipe | No tables in the loss, epoch 33 | -35.95 [-36.93, -34.93] | -67.21 [-68.18, -66.19] |

### Each model against its parents

A model that combines two trained changes, against each model it combines. Child minus parent, nats per team, paired bootstrap 95%. Negative = the child is better.

| Model | Parent | Tables on | Tables off |
|---|---|---|---|
| Table-free 5e-4 recipe | Cosine learning rate, wrong-stone teams filtered | +0.75 [+0.44, +1.08] | -62.12 [-64.32, -59.94] |
| Table-free 5e-4 recipe | No tables in the loss, 80 epochs | -2.26 [-2.46, -2.04] | -2.55 [-2.77, -2.33] |

### Against the stored reference numbers

Each row is one stored number, the same measure rescored now on the checkpoint it came from, and that measure on each new model. Rows from 2026-11-15 and 2026-11-23 use those probes' convention: the model's probabilities before the sampler renormalises them.

| Date | Stored number | Measured on | Stored | Rescored now | Wrong-stone teams filtered | Cosine learning rate, wrong-stone teams filtered | No tables in the loss, 80 epochs | Table-free 5e-4 recipe |
|---|---|---|---|---|---|---|---|---|
| 2026-10-04 | Masked cross-entropy at random masks, 48-field model (loss diagnosis) | the 48-field model | 2.570 nats per field | n/a | 2.557 | 2.086 | 1.448 | 1.426 |
| 2026-10-07 | Fixed-order nats, 48-field model before the retrain, no tables | the 48-field model | 2.362 nats per field | n/a | 2.557 | 2.086 | 1.448 | 1.426 |
| 2026-10-07 | Fixed-order nats, 48-field mixture after the retrain, tables on | the 48-field model | 1.418 nats per field | n/a | 1.442 | 1.385 | 1.422 | 1.406 |
| 2026-11-15 | Species-first minus moves-first gap, per-field schedule | Per-field masking schedule | 18.220 nats per team | 18.150 | 5.120 | 3.534 | 2.703 | 2.230 |
| 2026-11-15 | Species-first minus moves-first gap, linear schedule | Linear masking schedule | 12.330 nats per team | 12.461 | 5.120 | 3.534 | 2.703 | 2.230 |
| 2026-11-23 | Fixed-order nats, tables on | Per-field masking schedule | 124.055 nats per team | 124.055 (reproduced) | 106.592 | 98.515 | 101.523 | 99.265 |
| 2026-11-23 | Species-first minus moves-first gap, tables on | Per-field masking schedule | 18.150 nats per team | 18.150 (reproduced) | 5.120 | 3.534 | 2.703 | 2.230 |
| 2026-11-23 | Fixed-order nats, tables off | Per-field masking schedule | 207.955 nats per team | 207.955 (reproduced) | 162.752 | 135.484 | 102.778 | 100.243 |
| 2026-11-23 | Species-first minus moves-first gap, tables off | Per-field masking schedule | -7.150 nats per team | -7.150 (reproduced) | -8.620 | -3.612 | 3.105 | 2.534 |
| 2026-11-23 | Fixed-order nats, tables on | Linear masking schedule | 127.152 nats per team | 127.152 (reproduced) | 106.592 | 98.515 | 101.523 | 99.265 |
| 2026-11-23 | Species-first minus moves-first gap, tables on | Linear masking schedule | 12.461 nats per team | 12.461 (reproduced) | 5.120 | 3.534 | 2.703 | 2.230 |
| 2026-11-23 | Fixed-order nats, tables off | Linear masking schedule | 199.752 nats per team | 199.752 (reproduced) | 162.752 | 135.484 | 102.778 | 100.243 |
| 2026-11-23 | Species-first minus moves-first gap, tables off | Linear masking schedule | -3.461 nats per team | -3.461 (reproduced) | -8.620 | -3.612 | 3.105 | 2.534 |
| 2026-11-23 | Fixed-order nats, tables on | No tables in the loss, epoch 33 | 135.212 nats per team | 135.212 (reproduced) | 106.592 | 98.515 | 101.523 | 99.265 |
| 2026-11-23 | Species-first minus moves-first gap, tables on | No tables in the loss, epoch 33 | 8.736 nats per team | 8.736 (reproduced) | 5.120 | 3.534 | 2.703 | 2.230 |
| 2026-11-23 | Fixed-order nats, tables off | No tables in the loss, epoch 33 | 163.478 nats per team | 163.478 (reproduced) | 162.752 | 135.484 | 102.778 | 100.243 |
| 2026-11-23 | Species-first minus moves-first gap, tables off | No tables in the loss, epoch 33 | 1.713 nats per team | 1.713 (reproduced) | -8.620 | -3.612 | 3.105 | 2.534 |

Not the same quantity as ours, so read these as neighbours: 2026-10-04: random masks instead of a fill order, a 48-field model, 500 other test teams; 2026-10-07: 48-field model, 500 other test teams, 4 slot shuffles; 2026-11-15: same seed, but the table still held wrong-stone teams (2 fields at the floor).

### Does the nats ranking agree with the other measures?

| Model | Nats, tables on | Nats, tables off | First-try legality, tables off | Win rate | Novel share |
|---|---|---|---|---|---|
| Per-field masking schedule | 124.05 | 222.73 | 0.0% | 0.354 | 17.5% |
| Linear masking schedule | 127.15 | 213.62 | 0.0% | 0.353 | 18.3% |
| No tables in the loss, epoch 33 | 135.21 | 167.83 | 0.0% | no grid | no grid |
| Wrong-stone teams filtered | 106.59 | 200.79 | 0.0% | 0.389 | 25.8% |
| Cosine learning rate, wrong-stone teams filtered | 98.51 | 162.73 | 0.0% | 0.400 | 20.7% |
| No tables in the loss, 80 epochs | 101.52 | 103.17 | 60.9% | no grid | no grid |
| Table-free 5e-4 recipe | 99.27 | 100.62 | 68.8% | no grid | no grid |

| Measure | Set against | Models | Pairs agreeing | Pairs disagreeing | Pairs tied | Spearman | Verdict |
|---|---|---|---|---|---|---|---|
| first-try legality, tables off | nats, tables off | 7 | 11 | 0 | 10 | +0.80 | agrees |
| win rate | nats, tables on | 4 | 6 | 0 | 0 | +1.00 | agrees |
| novel share | nats, tables on | 4 | 4 | 2 | 0 | +0.60 | partly agrees |

Win rate: mean over the 23 asks of a stored species-first ask grid at guidance 2, against the 49 distinct top-50 meta teams, behaviour-cloning policy on both sides, slots shuffled, one battle per opponent. Novel share: teams whose six species match no labelled training team. First-try legality: teams drawn with the tables and the Stat Point budget off that Showdown accepts, after the Stat Alignment is repaired to match the moves.
