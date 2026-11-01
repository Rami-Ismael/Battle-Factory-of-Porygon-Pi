# What happens when the model generates Stat Points instead of copying them?

Pre-registered 2026-10-31, before any 84-field training, sampling or battle. Plan settled in the owner's grill of 2026-10-31. Tickets: #56 (this doc) to #65 on Battle-Factory-of-Porygon-Pi. Results will go to `~/vgc-data/results/stat_points_84.json`.

## Question

Every from-scratch discrete diffusion model since 2026-08-26 generated 48 fields per team: six Pokémon, each with species, ability, item, four moves and Stat Alignment. It never generated Stat Points. When the model turned a sampled team into a Showdown paste, it copied each Pokémon's Stat Points from a random corpus team with the same species. The owner wanted every field generated and counts the copying as a failure of the approach.

This experiment builds models that generate all 84 fields: the 48 fields plus six Stat Points per Pokémon. It answers two questions, each with one difference between the models compared:

1. **What does generating Stat Points do?** Same teams, same training settings; only the Stat Points differ.
2. **What does more data do?** Both models generate Stat Points; one trains on the deployed model's teams, the other on every team that has Stat Points.

Neither answer gates anything. The owner decided that the 84-field model trained on all teams becomes the default whatever the battles show (ticket #64).

## Models

Each name gives where the Stat Points come from, then which teams the model trains on.

| Reader name | Code ID | Fields | Teams | Training |
| --- | --- | --- | --- | --- |
| **Copied, same teams** | `copy48` | 48, Stat Points copied | same teams | the deployed mixture, unchanged |
| **Generated, same teams** | `gen84_same` | 84 | same teams | the deployed recipe, retrained from scratch |
| **Generated, all teams** | `gen84_all` | 84 | all teams | one model, early stopping |

**Same teams** means the rows the deployed 48-field mixture trains on: the loop-teams model's rows and the battle-records model's rows. **All teams** means every team in the rebuilt training table, combined into one dataset.

"Copied, same teams" is today's deployed mixture: `F1_final_medium_noheldout_regmb_v3` (the loop-teams model) and `G2_family_matrix_noprtrain_noheldout_regmb_v3` (the battle-records model, which also reads each team's results against the top-50 meta teams). It is not retrained.

## Encoding

A team is six Pokémon of 14 fields each, 84 fields in all:

- the 8 existing fields: species, ability, item, four moves, Stat Alignment;
- 6 Stat Points: HP, Atk, Def, SpA, SpD and Spe, each a value from 0 to 32.

Sampling decodes species first, as today, and decodes Stat Points after Stat Alignment. When it draws one Stat Point, it allows only values that keep that stat at 32 or less and the Pokémon's drawn Stat Points at 66 or less in total. Those are the only Stat Point rules. Nothing forces a Pokémon to spend all 66, because the search learns from battles, not from rules a person adds.

Training masks the new fields with the same schedule as Stat Alignment. The deployed schedule exponents are `16,4,4,1.5,1` for species, ability, item, move and Stat Alignment. Stat Points get exponent 1.

The paste writes the generated Stat Points. Nothing is copied from the corpus.

## Data

The 84-field training table rebuilds every training team from its stored paste, from the same sources as today's 48-field table (`lossdata.py`).

- **Keep a team only if every Pokémon has exactly one recorded set of Stat Points.** A team stored as one 48-field row was sometimes battled several times with different copied Stat Points. As 84-field teams those are different teams with different results, so the row is dropped rather than split. Teams whose paste can no longer be found are dropped too.
- **Each label is the win rate of that exact 84-field team**, with the same pooling rule as today.
- The 500 test teams and the held-out meta teams ranked 51 to 100 never enter training.

Counts on 2026-10-31, from a one-off count over today's sources (ticket #58 rebuilds them and prints the final numbers):

| Teams | Rows today | Kept | Dropped: several sets | Dropped: no paste found |
| --- | --- | --- | --- | --- |
| Loop-teams model's rows | 17,179 | 16,534 | 402 | 243 |
| Battle-records model's rows | 25,914 | 25,152 | 408 | 354 |
| All teams | 304,786 | 301,126 | 3,660 | 0 |

The 304,786 teams include about 100,000 uniform-legal teams. The count grows as other sessions' results arrive.

## Training

**Generated, same teams.** The deployed recipe with 84 fields:

- loop-teams model: 80 passes over its rows;
- battle-records model: 60 passes over its rows, with its results input;
- both: width 256, 6 layers, 8 heads, batch 256, classifier-free-guidance dropout 0.15, rules `clause,compat,order`.

Both train directly on the Regulation M-B vocabulary. This replaces the deployed route of training first and then running 2 passes of continued training on Regulation M-B.

**Generated, all teams.** One model, with no mixture and no results input, on all teams combined:

- a seeded random 2% of the teams (about 6,000) is held out for validation;
- validation loss is measured after every pass;
- training stops after 3 passes with no new lowest validation loss, and keeps the checkpoint with the lowest;
- at most 80 passes;
- the same width, depth and vocabulary as above.

## Battles

The same protocol for all three models:

- **Asks:** 1/24, 2/24, …, 23/24 (23 asks).
- **Sampling:** classifier-free guidance weight 2, temperature 1, rule masks on, species first, Stat Alignment repaired to match the moves (`to_paste(repair=True)`). Seeds 1, 2 and 3.
- **Per ask and seed:** draw until 64 teams pass Showdown's validator and are not held-out teams, at most 1,280 tries.
- **Battles:** each team once against each of the 49 distinct top-50 meta teams; the team listed twice counts twice. Behaviour-cloning policy on both sides, slots shuffled.

That is 216,384 battles per model and 649,152 in all. "Copied, same teams" can run as soon as the protocol exists (ticket #61).

## Measures

- **Win rate** against the top-50 meta teams, per ask, with intervals clustered by team.
- **Asked against measured:** the slope of measured win rate on ask, and mean |ask − measured|.
- **Legality:** Showdown-accepted tries over all tries, and the most common rejection reasons.
- **Diversity:** effective species, distinct line-ups and top-species share on random 5-team subsets (200 per cell), as in the decoding-order experiment.
- **Stat Points:** the share of Pokémon that spend exactly 66; distinct sets of Stat Points per species; and the share of Pokémon whose Stat Points match a corpus set for that species exactly.

## Comparisons

Two gaps, reported per ask and pooled over asks, each with a team-bootstrap interval:

- **Gap 1, what generating Stat Points does:** "Generated, same teams" minus "Copied, same teams".
- **Gap 2, what more data does:** "Generated, all teams" minus "Generated, same teams".

No pass rule. The gaps describe the change; they do not decide the default.

## Expectation, written before the run

Gap 1 should be small, between −0.03 and +0.01. In August an older model generated Stat Points and lost 2.1 points against copied ones, and corpus Stat Points matched tuned ones within noise. The random-fill baseline also showed the behaviour-cloning policy barely notices Stat Points until all six Pokémon change. Copied corpus sets are already reasonable, so generating them has little room to gain.

Gap 2 could go either way. In August win rate grew with data. In October, balancing training across all data families lost 0.047. All teams adds 100,000 uniform-legal teams that win almost nothing. The win-rate label should keep the model away from them at high asks. The unconditional model that guidance pushes away from learns from them too, which changes what guidance does.

Generated Stat Points should spend exactly 66 on most Pokémon (the corpus does on 99%), and most should match a corpus set for their species, because most training teams carry copied corpus sets.
