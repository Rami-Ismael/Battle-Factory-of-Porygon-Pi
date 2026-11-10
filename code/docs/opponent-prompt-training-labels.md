# Opponent-prompt training labels: which battles and subset sizes make them

Pre-registration for [Training labels: which battles and subset sizes make them?](https://github.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/issues/118), on the map [Opponent prompt: does telling the diffusion model its opponents help it beat them?](https://github.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/issues/115). It builds the training table that all three of that map's models read: the UniXcoder opponent prompt, the field-embedding opponent prompt and the no-prompt control.

Status: **exploratory**. I measured label reliability on the stored battles before writing this plan, so the plan was not fixed blind.

Code: `src/opponent_split.py` draws the frozen split into `docs/opponent-split.json`. `src/lossdata65.py` builds `~/vgc-data/opponent_prompt_labels/lossdata65.npz` and writes `lossdata65_counts.json`, with the probes under `preflight`.

## Contract

| Row | Value |
| --- | --- |
| Claim | A label computed as wins ÷ battles over a random subset of the 39 training opponents, from shuffled-slot battles only, measures the team's strength well enough to train on: reliability at least 0.5 at the smallest subset trained. It carries no validation-opponent outcome and no team ranked 51 to 100. **Refuted by** reliability under 0.5 at 10 opponents, or any validation battle or ranks 51 to 100 team in the table. **If it holds**, the three opponent-prompt models train on this table. **Scope:** win rate under the behaviour-cloning policy on both sides, Pokémon shuffled every battle, against the 39 training opponents. |
| Fields | All 84 fields come from the stored team. The table copies none and fixes none: it is training input, and every field is a reconstruction target. |
| Vocabulary | The Regulation M-B vocabulary stored in checkpoint `F1_final_medium_noheldout_regmb_v3` (`team84.regulation_vocab`), with the same token ids as the deployed models. Stat Points use `stat_points` tokens: each stat 0 to 32, at most 66 per Pokémon (`lossdata84.legal_stat_points`). A team with a field outside the vocabulary is dropped and counted. |
| Training teams | Every team in `~/vgc-data/matchup_regmb.sqlite` with shuffled-slot battles against the training opponents. Battles from any other configuration supply no row and no label: listed order (`policy 1`), the Ling-baseline Showdown builds (`policy 2–5`), the optimized runner (`policy 6`) and the untagged runner (`policy 7`). The owner ruled on 2026-11-07 that listed-order battles are biased. Excluded: ranks 51 to 100 by their 48 fields (`heldout.py`), and unencodable teams. Identical 84-field teams are pooled. Excluding near copies of validation opponents is **pending**, being decided in a separate grill. The table stores each row's species distance to the nearest validation opponent, plus a same-48-fields flag, so the trainer can apply the rule once it is settled. Teams with no battles, such as unbattled corpus or uniform-legal teams, are not in this table. |
| Scorer | No new battles. Stored shuffled-slot battles, configuration `bc_100.zip:57f5edcab415cf6c+slots-shuffled` (`policy 8`), behaviour-cloning policy on both sides. Its simulator commit is recorded as `showdown:unknown`. Most rows hold one battle per opponent (see Battle depth). |
| Opponents | The frozen split `docs/opponent-split.json`, seed 0, drawn 2026-11-07 by the rule in #117. Validation opponents: MB129, MB383, MB517, MB573, MB581, MB583, MB689, MB712, MB715, MB756, all one-team families. The other 39 are training opponents. Validation-opponent battles are skipped while reading and never stored. |
| Search region | None: this table feeds training. The label sampler `subset_labels` draws a fresh subset for each row each time it is called. Its size k is uniform between 10 and 39, capped at the opponents the row has battled. A training meta team never battles itself, so it has 38. |
| Settings | Subset size range 10 to 39: **fixed**, proposed and awaiting the owner (#118 Q4). Fresh subset each epoch: **fixed**, proposed and awaiting the owner (#118 Q5). Subset draw seed: **nuisance**, varied across training seeds. Split seed 0 and validation-row seed 0 (2% of rows, for early stopping): **fixed**. Near-copy exclusion: **fixed per run**, pending the separate grill. |
| Plan | Primary measure: label reliability at each subset size, 1 − mean battle-noise variance ÷ variance of labels across rows. This is a single measure that nothing trades against, so the bar is reliability ≥ 0.5 at 10 opponents. Baseline: the full 39-opponent label. Battle budget: zero. Smallest effect worth acting on: reliability under 0.5 at the smallest size, which would raise the minimum subset size. Stopping rule: one build; it is rebuilt only when the database or the split changes. |

## Probes

Built 2026-11-07 in 1.8 minutes, table sha256 `8d8b595c…`, split sha256 `46f74c96…`. The table holds 276,166 rows: every team with shuffled-slot battles against the training opponents, including all 49 meta teams. No team was dropped: none is a ranks 51 to 100 team, and every one encodes. No two pooled into one row. 3,068,620 validation-opponent battles were skipped. 5,523 rows (2%) are validation rows.

| Probe | Result | Pass |
| --- | --- | --- |
| Ceiling | Skipped: the table has no last stage to feed a perfect input. The ceiling belongs to the model runs (#119, #121). | n/a |
| Round trip | 20 random rows decoded. Species, ability, item, moves, Stat Alignment and Stat Points equal the database paste on every slot. | yes |
| Coverage | 0 teams dropped, so none needs an explanation. Vocabulary coverage is inherited from `team84.regulation_vocab`. | yes |
| Read teams | I read 4 decoded rows. The search-written rows hold teams a player would reject, such as Garchomp @ Mystic Water and a Modest Charizard beside an Adamant Glimmora running special moves, and their labels price them at 0.23 to 0.31. The uniform-legal row scores 0.026. The table stores teams as battled, so this is expected. | yes |
| Training smoke | Skipped: nothing trains here. It belongs to #119. | n/a |
| Controls | The 39 training meta teams average **0.500** against the other 38 training opponents. The 12,112 uniform-legal teams average **0.021**. | yes |
| Invariance | Skipped: every battle is shuffled-slot, so slot order is already randomized. | n/a |
| Repeat | Skipped: no battles run. | n/a |
| Separation | 39 columns, matching the split. Validation battles are skipped while reading. 0 rows are ranks 51 to 100, re-derived from the stored tokens. Distance to the nearest validation opponent: 824 rows share a line-up (0 swaps), and 4,848 are one swap away. 11 rows have a validation opponent's 48 fields: the 10 validation teams themselves and one copy. | yes |
| Condition, Copies | Skipped: no model is sampled. | n/a |

`subset_labels` with the proposed sizes drew k from 10 to 39. The mean label is 0.292.

## Noise floor

There is one battle per opponent for 272,314 rows. 3,852 have at least 2 per opponent, 2,959 at least 8 and 159 at least 16.

| Subset size | Label spread across rows (SD) | Battle noise (SD) | Reliability |
| --- | --- | --- | --- |
| 5 | 0.250 | 0.168 | 0.55 |
| 10 | 0.211 | 0.126 | **0.64** |
| 20 | 0.190 | 0.092 | 0.77 |
| 30 | 0.182 | 0.076 | 0.83 |
| 39 | 0.178 | 0.067 | 0.86 |

The bar is met: reliability at 10 opponents is 0.64, above 0.5. At one battle per cell, nearly all of a row's label change between subsets is battle noise. The matchup between this team and this opponent explains about 3.6% of one battle's variance (#115). That limits what the prompt can learn from these labels, and it is the open item "can 39 opponents teach reading a team" on #115, not a defect of this table.
