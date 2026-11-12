---
aliases: []
subset: []
Created: "[[2026-11-09]]"
---

Team diversity at four levels: line-up, sets, Stat Points, whole team. Code inventory, plan and the 2026-11-09 result: [page](reports/diversity-measurement-2026-11-09/index.html). 2026-11-11 builds, Novelty, nearest training distance, archetypes: [page](reports/diversity-measurement-2026-11-11/index.html). Reading: [[Diversity — collected project notes]].

## What already exists

![[Diversity — what already exists.excalidraw]]

| Source | Measure | What it counts | Level | All 84 fields? |
|---|---|---|---|---|
| `team_diversity.py` | distinct_lineups | Distinct sorted six-species sets | Line-up | No |
| `team_diversity.py` | distinct_species | Species used anywhere | Line-up | No |
| `team_diversity.py` | effective_species | exp(entropy) of species frequency over all slots | Line-up | No |
| `team_diversity.py` | top_species_share | Share of *teams* carrying the commonest species | Line-up | No |
| `team_diversity.py` | mean_distance | Mean pairwise distance in fields out of 48, slots matched, on 128 teams | Sets | No, Stat Points left out |
| `diversity_metrics.py` | unique / repeat fraction | Exact repeats at three keys: composition, categorical_48, with_spreads | Line-up, sets, team | Only as exact equality |
| `diversity_metrics.py` | novel draw / novel unique fraction | Share absent from a fixed reference, same three keys | Line-up, sets, team | Only as exact equality |
| `diversity_metrics.py` | composition frequencies, largest share | Count of every line-up | Line-up | No |
| `diversity_loops.py` | diversity() | Per generation on 512 proposals: distinct and effective compositions, largest share, top three, composition novelty, plus the fractions above | Line-up, sets, team | Only as exact equality |
| `diversity_loops.py` | DPP selection kernel | Species overlap between teams; picks teams, reports no score | Line-up | No |
| `diversity_loops.py` | Taboo list | At most two copies of a line-up per generation; a limit, not a measure | Line-up | No |
| `novel_elite_rebattle.py` | novel_elites() | Keeps elite teams whose line-up is absent from training, then re-battles them; a novelty filter with no diversity score | Line-up | No |
| `ask_grid84.py` | subset_diversity() | Means over 5-team subsets: effective species, distinct line-ups (at most 5), top_species_share as a share of *slots* | Line-up | No |
| [[Diversity — collected project notes]] | Index of 73 notes | Keeps four questions apart: complete teams different, compositions different, absent from a corpus, play differently | All | Not a measure |
| [[Metrics for Diversity]] (repository `notes/` copy only) | Measurement contract | Exact uniqueness, repeats, novelty against a fixed reference, composition entropy as 2^H, surprisal −log p (model-dependent, not a distance) | Line-up, sets, team | Only as exact equality |
| [[Quality Diversity]] | Problem class | QD score, MAP-Elites, CVT, CMA-MEGA, novelty search; descriptor still open | None chosen | Nothing measured |
| [[Rami Hand written notes with no claude interacting at all about diversity]] | Open checkboxes | Same-team definition, unique by species and by every field, repeat fraction, U/N, novelty and its degree, information-theory distance | Line-up, team | Not built |


- [ ] Determine a way to draw the html table the same way in osbdian markdown table
- [ ] Read this https://amitness.com/posts/diversity-evals/

--- 
#topic
