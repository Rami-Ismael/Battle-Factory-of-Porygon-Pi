# Do uniform grids keep more novel line-ups at equal ask-versus-got error?

The comparison holds the learning-rate recipe fixed and compares two models by the error they achieve. The recipe warms up for 1,000 steps, peaks at 0.0005, and decays on a cosine schedule to 0.00005. Both checkpoints reached the 80-epoch cap. [Training results and recipe](https://github.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/issues/431)

| Model | Ask-grid row | Checkpoint |
| --- | --- | --- |
| Uniform, no filter | `udlm_loglinear_gidd_lr` | `stat84_udlm_loglinear_lr5e4_min5e5_es3_cap80` |
| Masked, full filter | `shuffled_linear_filtered_gidd_lr` | `stat84_shuffled_noise_linear_filtered_lr5e4_min5e5_es3_cap80` |

Uniform, no filter starts from random tokens and uses Showdown to validate complete teams. Masked, full filter starts from masked fields and uses the legal Pokémon construction filter during sampling. Guidance mixes reverse posteriors in the uniform model and clean-field log probabilities in the masked model. Equal guidance weights therefore need not give equal error. [Experiment issue](https://github.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/issues/438) · [Uniform discrete diffusion map](https://github.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/issues/270)

## Pass count and battle grids

At guidance 1, sample asks 1/24 to 17/24 with three seeds for each of 84, 128, and 256 passes. For line-up uniqueness, use the smallest kept count across the three pass counts at each ask and seed. Draw that many teams without replacement at each pass count using a fixed random seed. Drop a cell from the uniqueness comparison when any pass count kept zero teams there. Report how many cells reached 64 teams at each pass count.

The selection rule otherwise follows the earlier sweep. Choose the smallest pass count within seed noise of 256 on first-try legality, within-ask line-up uniqueness, and within-ask species replacement distance. If no cells support matched uniqueness, neither shorter pass count qualifies. Stop before battling if all three counts fall below 5% first-try legality. [Earlier pass-count sweep](https://github.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/issues/277)

After selecting the pass count, smoke-test the fixed-pass row in a private database. Run the uniform grids on the laptop at guidance 1, 2, 4, and 8. Each grid has 23 asks and three sampling seeds. The pass-count sweep and the battle grids both target 64 validator-accepted teams per cell with a cap of 20,000 tries. Log cells below target before battling. Battle every kept team and report the shortfall with the grid results. A full grid records 216,384 battle outcomes across the 49 distinct top-50 meta opponents, with shuffled slots. Skip Pass@48 bands where resampling can yield fewer than 192 teams at an ask. The masked arm reuses the shared-recipe grids at guidance 2, 4, and 8. The older recipe's stored grids are inputs to the report dry run. [Masked grids](https://github.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/issues/431) · [Older grids](https://github.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/issues/385)

Ask-versus-got error is the mean squared difference between the requested win rate and the measured mean win rate on asks 1/24 to 17/24. Error requires at least one battled team at each scored ask. Novel share is the fraction of sampled teams whose unordered six-species line-up is absent from the labelled training split. Novel share, distinct line-up count, and pooled win rate use the teams kept across all 23 asks.

Use 4,000 bootstrap draws within each ask-by-seed cell. Error and win-rate draws also resample the 49 opponents, using the same opponent draws across grids. Center the distinct-count bootstrap variation around the observed count because resampling itself duplicates teams. The 95% intervals describe these checkpoints and the three observed sampling seeds. Each checkpoint comes from one training seed.

Join each model's measured points in order of error. The equal-error verdict uses novel share and applies where the error ranges overlap. A point clears the other curve when its novel-share interval clears the interpolated interval band across its full error interval. Report the point-estimate difference at matched error. Distinct line-up counts and their intervals describe the observed sample sizes, which can differ by grid. Pooled win rates describe the observed grids.

## Figure storyboard

### Choosing the number of passes

- **Sentence.** The smallest qualifying pass count keeps first-try legality and both within-ask diversity measures within seed noise of 256 passes.
- **Expected.** The 128-pass sampler is close enough to 256 on all three measures.
- **Surprising.** All three pass counts fall below the 5% first-try legality floor, ending the run before battles.

### How fields change during sampling

- **Sentence.** Simultaneous updates, repeated species, and flips in the model's top prediction show how the pass count changes the sampling trajectory.
- **Expected.** Longer runs revise more fields after each field first receives a value.
- **Surprising.** More passes increase the share of finished teams with duplicate species.

### Requested and measured win rates

- **Sentence.** The gap between the measured curve and the diagonal shows where the uniform model misses its requested win rate.
- **Expected.** Guidance reduces error through the scored asks, with larger misses at the highest asks.
- **Surprising.** Higher guidance flattens the response across the scored asks.

### Novel line-up share at matched error

- **Sentence.** At the same error, the higher curve yields more teams with line-ups absent from labelled training.
- **Expected.** Uniform, no filter retains more novel line-ups at matched error.
- **Surprising.** The masked model's interval clears the uniform curve in the shared error range.

### Distinct line-ups at matched error

- **Sentence.** Each point counts distinct six-species line-ups among the teams that grid kept.
- **Expected.** Uniform grids contain more distinct line-ups among grids with similar team counts and error.
- **Surprising.** Uniform, no filter produces more novel teams but repeats the same novel line-ups often enough to lose on distinct count.

### Pooled win rate at matched error

- **Sentence.** At the same error on asks 1/24 to 17/24, pooled win rate shows how the two models fare overall across the full dial.
- **Expected.** Pooled win rates are similar through the shared error range.
- **Surprising.** Pooled win rates differ at similar levels of calibration error.
