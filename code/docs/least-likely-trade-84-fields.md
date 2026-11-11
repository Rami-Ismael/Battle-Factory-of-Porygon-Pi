# The least-likely trade on the all-teams 84-field model

Exploratory: the plan was written after seeing the species-first remasking results. The owner asked on 2026-11-02 for the fresh re-battle, the controls and shares, the ranks 51 to 100 check, and a Pareto curve of win rate against line-ups. The owner, not this run, chooses how much line-up loss is acceptable. Script: `src/trade84.py` as a frozen copy in `~/vgc-data/least_likely_trade/run`.

## Contract

| Row | Value |
| --- | --- |
| Claim | Least-likely 10% remasked misses the slope bar, raises win rate at every ask, and pays with fewer line-ups, and a control can or cannot reach the same win rate at the same line-up count. It stops at win rate against the top-50 meta teams under the behaviour-cloning policy; the ranks 51 to 100 check tests the final team lists only. A control on the curve above least-likely refutes the need for remasking. |
| Fields | All 84 generated, and Stat Alignment replaced by the repair, except the Stat Points control: its Stat Points are the no-remasking team's, raised by a hand rule to 66 per Pokemon. Nothing else is copied. |
| Vocabulary | Unchanged from the checkpoint `stat84_all_teams_ep21.pt`. |
| Training teams | Unchanged, no training here. |
| Scorer | One battle against each of the 49 distinct top-50 meta teams, one listed twice and weighted twice, behaviour-cloning policy on both sides, slots shuffled, seed 7300 + sampling seed. Fresh re-battle: 4 games each, seed 7520, private database. Ranks 51 to 100: 4 games each, seed 7530, same private database. |
| Search region | No loop. Each cell draws until 64 legal, non-held-out teams or 1,280 tries. Asks 4/24, 8/24, 12/24, 16/24. Elite cut 0.531. |
| Settings | Scientific: remasking share, temperature, Stat Points. Nuisance: ask, seeds 1 to 3. Fixed: guidance 2, species-first order, legal Pokémon construction filter. |
| Plan | Primary: mean win rate against distinct line-ups at 192 teams per point, drawn as a Pareto curve with no remasking at every ask. Beside it: effective line-ups, new line-ups against training, fresh line-ups holding 0.531. Budget: 7 arms x 4 asks x 3 seeds x 64 teams x 49 battles = 263,424, about 40 minutes. Re-battle: 95 line-ups x 196 battles, 8,820 new after reuse. Ranks 51 to 100: 200 battles per team on the frozen lists. Smallest effect worth acting on: 0.03 win rate. |

## Arms

Shared by every arm: all-teams 84-field model, 21 epochs, species first, guidance 2.

| Reader name | Code ID | What differs |
| --- | --- | --- |
| No remasking, 84 passes | `plain` | Existing grid at 23 asks, `stat_points_84_gen84_all_ep21.json` |
| Least-likely 10% remasked, 84 passes | `remask_ll_10` | Existing, `remask84.json` |
| Least-likely 5% remasked, 84 passes | `remask_ll_05` | Share |
| Least-likely 20% remasked, 84 passes | `remask_ll_20` | Share |
| Temperature 0.8 on all fields, no remasking | `cold_all_t08` | Temperature |
| Temperature 0.6 on all fields, no remasking | `cold_all_t06` | Temperature |
| Temperature 0.6 on sets only, no remasking | `cold_sets_t06` | Species keep their diversity; sets go colder |
| Temperature 0.3 on sets only, no remasking | `cold_sets_t03` | Same, colder |
| Stat Points raised to all 66, same no-remasking teams | `spend66` | Same teams, paired, only the Stat Points line changes |

## Order and freezing

1. Fresh re-battle of the best team of each line-up at the cut, both rules, ties to the lowest ask, seed and position: `remask84_rebattle_plan.json`.
2. A line-up holds if its team's fresh win rate is at least 0.531. That freezes one final team list per rule: `least_likely_trade_final_lists.json`.
3. Ranks 51 to 100, once per list, to report and never to choose: `least_likely_trade_heldout.json`.
4. Controls and shares: `least_likely_trade.json`.

## Noise floor

One point has 9,408 battles, so battle noise on a mean is about 0.005. Three sampling seeds are the only repeat of generation. A 0.03 gap between arms needs about 4,400 battles per arm, so the budget can detect it. A re-battled team has 196 games, standard error about 0.036, so a team within that of the cut could fall either way.

## Probes

`check` proves the temperature sampler at temperature 1 is bit-identical to `Model84.sample`. `probes` runs round trip and copies per sampled arm; a failed round trip stops the queue. The Stat Points control: every Pokemon reaches 66, and all 64 teams of one test cell pass Showdown. Inherited from the grid, same checkpoint and scorer: coverage, controls, invariance, repeat, separation. Skipped: training smoke, no training. Ceiling, because the question is a curve, not a fix to a later stage.

## Deviations

The controls were chosen after seeing the least-likely results. The first queue crash-looped for an hour at ranks 51 to 100, because those teams had no legality flag in the private database; no such battle was played, and a resumed queue, `ltq2.zsh`, added the flag and a three-crash stop. The queue runs in front of the line-up deletion queue because it takes under an hour and that queue takes days.

## Nucleus sampling and the re-battle of temperature 0.3 on sets only

Exploratory, written after seeing the results below. The owner asked on 2026-11-02 for the code to be reviewed by GPT-6 and for both to run behind the line-up deletion queue. Script: `src/nucleus84.py`, frozen in `~/vgc-data/nucleus_84/run`. Queue: `~/vgc-data/nucleus_84/nq84.zsh`.

| Row | Value |
| --- | --- |
| Claim | Removing the unlikely options from the set fields buys some of least-likely's win gain at a smaller cost in Pokemon sets than temperature 0.3 on sets only. It stops at win rate against the top-50 meta teams under the behaviour-cloning policy. If a nucleus arm matches the win rate of temperature 0.3 on sets only with clearly more distinct sets, it replaces that arm on the curve; if not, temperature stays. |
| Fields | All 84 generated, Stat Alignment replaced by the repair. Nucleus filtering runs after guidance and the legal Pokémon construction filter, and "set fields" means every field except species, as the temperature arms. |
| Vocabulary | Unchanged from `stat84_all_teams_ep21.pt`. |
| Training teams | Unchanged, no training. |
| Scorer | As above. One battle against each of the 49 distinct top-50 teams, shuffled slots, seed 7300 + sampling seed. Re-battle: 4 games each, seed 7540, new private database copied from the earlier fresh one. |
| Search region | No loop. 64 legal, non-held-out teams or 1,280 tries per cell. Asks 4/24, 8/24, 12/24, 16/24. Elite cut 0.531. |
| Settings | Scientific: nucleus share and which fields it covers. Nuisance: ask, seeds 1 to 3. Fixed: temperature 1, guidance 2, species-first order, legal Pokémon construction filter. The shares are picks, not derived. |
| Plan | Primary: win rate at each ask against distinct line-ups and distinct sets, added to the existing curves. Budget: 4 arms x 4 asks x 3 seeds x 64 teams x 49 battles = 150,528. Re-battle: best team of each of the 68 line-ups of temperature 0.3 on sets only at the cut, ties to lowest ask, seed and position: 13,328 battles at most. Smallest effect worth acting on: 0.03 win rate. Ranks 51 to 100 are not spent here; the re-battle freezes no list. |

| Reader name | Code ID | What differs |
| --- | --- | --- |
| Nucleus 0.9 on sets only | `nucleus_sets_p09` | Keeps the likeliest set options up to 90% of the odds |
| Nucleus 0.7 on sets only | `nucleus_sets_p07` | Up to 70% |
| Nucleus 0.5 on sets only | `nucleus_sets_p05` | Up to 50% |
| Nucleus 0.9 on all fields | `nucleus_all_p09` | Species filtered too, so line-ups can fall |

Noise floor and inherited probes as above. `check` proves p = 1 matches `Model84.sample` bit for bit, that a sets-only filter leaves species draws alone at one seed, and the filter's choices on a known table. `probes` runs round trip and copies per arm; a failed probe stops the queue. `check` also runs the sampler on a stand-in model with known odds and checks which options each arm can draw, with and without a legal Pokémon construction filter. The earlier temperature 0.7 with top-p 0.9 result (`src/schemes.py`, 2026-08-27) came from an older model with copied Stat Points, and combined the two.

GPT-6 (`gpt-6-sol`) reviewed the code before the freeze and found three defects, all fixed before the queue started. Stalls that recovered nothing could loop forever: three now stop the queue, while a stall that recovers battles still retries. The report left out distinct Pokemon sets, which the claim needs. The checks could pass with a sampler that ignored the filter; a test that disables the filter now fails them. Training copies stay reported with no pass mark, as `run-experiment` asks. Of the 68 line-ups at the cut, 52 come from ask 12/24, so the re-battle reports with and without it.

## Result

Reported on the issue "Is the least-likely trade real at equal win rate?" (#100), closed 2026-11-06. Win rate is against the top-50 meta teams, behaviour-cloning policy on both sides. Every number here is exploratory.

- Temperature 0.3 on sets only gains +0.054 win rate (+0.043 to +0.063) and keeps all 438 line-ups. Least-likely 10% gains +0.063 and keeps 228. Both keep about half the Pokemon sets: 54% and 48%.
- Nucleus 0.5 on sets only ties temperature 0.3 on sets only at -0.001 (-0.010 to +0.009), with 1,223 sets against 1,252. By the claim's rule, temperature stays. Nucleus 0.7 sits on the temperature line, with and without ask 12/24.
- The nucleus queue ran on 2026-11-03 from 02:29 to 02:58 with no restart. Its checks and probes passed, and none of the 384 probe teams was identical to a training team.
- Fresh re-battle, 196 games per team: 18 of 52 line-ups hold for no remasking, 18 of 43 for least-likely 10%, 27 of 68 for temperature 0.3 on sets only. Without ask 12/24: 3 of 14, 1 of 8, 2 of 16.
- Ranks 51 to 100, once per frozen list: 0.530 and 0.524.
- Ask-versus-got error, rechecked after the owner replaced slope on 2026-11-05: on asks 4/24 to 16/24 no arm comes closer to got = ask than no remasking's 0.0197. Least-likely 20% moves further away, +0.0045 (+0.0016 to +0.0074). This covers four asks, not the full 1/24 to 17/24.
- The owner asked for the point on the curve to be picked for him on 2026-11-06: temperature 0.3 on sets only.

Intervals resample teams within each cell, not opponents. Numbers: `~/vgc-data/results/least_likely_trade_storyboard.json`, from `src/trade84_storyboard_data.py`. Pareto curves and paired intervals over opponents: `least_likely_trade_curves.json`, from `src/trade84_curves.py`.

### Storyboard

The figures are cut to one idea each (`storyboard-figures`), in `src/trade84_storyboard_figures.py`. They were drawn after the results, so each line records a reading, not a prediction.

| Figure | The sentence it carries | What would have changed it |
| --- | --- | --- |
| Teaser | Set-only sharpening buys most of least-likely's gain and keeps every line-up. | A set-only sharpening arrow that bent left, losing line-ups. |
| Moves per ask | Least-likely pays in line-ups and sets; set-only sharpening pays only in sets. | A set-only sharpening arrow clearly shorter than least-likely's outside 12/24. |
| Nucleus against temperature | Cutting unlikely set options costs the same sets per unit of win rate as cooling them. | A nucleus point above the temperature line, which would have replaced temperature. |
| The menu | Set-only sharpening alone gains over 0.05 without losing a line-up. | Another option over 0.05 that keeps every line-up. |
| Re-battle | The strong line-ups of temperature 0.3 on sets only hold on fresh games as often as the others'. | Far fewer than about a third holding, which would mean its single-battle highs were luck. |
| Ask-versus-got error | No remasking rule or control brings got closer to the ask. | Any interval wholly left of zero. |

Colours: least-likely remasking orange, temperature aqua, nucleus blue, no remasking ink. These are validated all-pairs on the light surface.
