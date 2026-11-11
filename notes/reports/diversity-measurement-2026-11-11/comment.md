**Result of [Measure diversity of the default model's teams](https://github.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/issues/264), 2026-11-11.** The default model writes about 5 fewer builds per species than its training teams, and its teams sit about 12 fields further from the closest training team. Both gaps are far outside the training-split noise floor.

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/465a1cc00d7ef316fa96a466c50537ff61053902/notes/reports/diversity-measurement-2026-11-11/figures/teaser-dark.svg"><img width="800" alt="The default model writes 5.1 fewer builds per species than matched training, 20.6 against 25.7, yet its teams sit 11.5 fields further from the closest training team, 36.0 against 24.5 of 84 fields. Both training rows sit between one team repeated and random legal teams." src="https://raw.githubusercontent.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/465a1cc00d7ef316fa96a466c50537ff61053902/notes/reports/diversity-measurement-2026-11-11/figures/teaser-light.svg"></picture>

The model writes fewer kinds of sets for each species, yet it combines them into teams further from any training team than training teams sit from each other. Each panel runs from one team repeated to random legal teams, and training sits between the two.

The default model is the 80-epoch labelled 84-field model at guidance 2. An ask is the win-rate label the model is asked to write a team for. Each row is 1,088 teams, 64 per ask on asks 1/24 to 17/24. The baseline is training teams matched to the asks, drawn the same way, 64 training teams per ask. A second training row draws training teams over all win rates. The noise floor is the range of the difference between two training draws; a gap counts when all three sampling seeds land outside it with the same sign.

### Builds per species

A build is a set without its Stat Points. Builds per species counts the distinct builds among 30 uses of one species, averaged over species.

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/465a1cc00d7ef316fa96a466c50537ff61053902/notes/reports/diversity-measurement-2026-11-11/figures/builds-dark.svg"><img width="800" alt="All three seeds of the default model sit at 20.5 to 20.6 builds per species, 5.1 below matched training at 25.6 to 25.8 and far outside the noise floor; training drawn over all win rates sits between at 23.8 to 24.0." src="https://raw.githubusercontent.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/465a1cc00d7ef316fa96a466c50537ff61053902/notes/reports/diversity-measurement-2026-11-11/figures/builds-light.svg"></picture>

All three seeds sit 5.1 builds below matched training, against a noise floor of -0.68 to +0.58 builds. Restricted to the species both sides use 30 times, the gap is 4.84 to 5.00 builds. Training drawn over all win rates shows fewer builds per species than the matched row, a mean of 23.9, so part of the matched row's variety comes from matching the label mix.

### Novelty

A line-up is a team's six species. Novelty counts the species swaps from a team's line-up to the closest training line-up: 0 swaps means the line-up is copied from training, 3 or more means a new line-up.

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/465a1cc00d7ef316fa96a466c50537ff61053902/notes/reports/diversity-measurement-2026-11-11/figures/novelty-dark.svg"><img width="800" alt="The default model copies a training line-up in 59.1% of teams against 71.1% for matched training; the difference goes to 1 swap, 27.5% against 23.0%, and 2 swaps, 11.0% against 5.4%, while 3 or more swaps stays at 2.3% against 0.6%." src="https://raw.githubusercontent.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/465a1cc00d7ef316fa96a466c50537ff61053902/notes/reports/diversity-measurement-2026-11-11/figures/novelty-light.svg"></picture>

The model copies a training line-up less often, 59% against 71% for matched training, and the difference goes to one and two swaps. Line-ups three or more swaps away stay under 4% on every seed and do not clear the floor against matched training.

### Nearest training distance

Nearest training distance counts the fields, of a team's 84, that differ from the closest training team. It is the right panel of the first figure. No model team is a copy at 0 fields. The median is 34 fields against 24 for a training team's closest other training team.

### Archetypes

An archetype is sun, rain, sand, snow, one of four terrains, Trick Room or Tailwind. A team is in one when any set carries the setting move, ability or mega stone; whether the team sets it in battle is not measured.

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/465a1cc00d7ef316fa96a466c50537ff61053902/notes/reports/diversity-measurement-2026-11-11/figures/archetypes-dark.svg"><img width="800" alt="Trick Room teams are 31.2% of the default model's teams against 25.2% of matched training, and sun 33.8% against 28.7%, beyond the noise floor on every seed; Tailwind, 46.0% against 49.1%, clears the floor on 1 of 3 seeds and rain, 12.4% against 8.9%, on 2 of 3." src="https://raw.githubusercontent.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/465a1cc00d7ef316fa96a466c50537ff61053902/notes/reports/diversity-measurement-2026-11-11/figures/archetypes-light.svg"></picture>

Trick Room and sun are above matched training beyond the floor on every seed. Tailwind and rain are not. Grassy, misty and psychic terrain are near 0% everywhere; their setting abilities are not in the Regulation M-B vocabulary.

<details><summary>Full numbers: builds, Novelty and nearest training distance</summary>

Ranges are the three sampling seeds or the three training draws.

| Teams | Builds per species at 30 uses | Line-up copied, 0 swaps | 1 swap | 2 swaps | 3 or more swaps | Nearest training distance, mean fields of 84 |
|---|---|---|---|---|---|---|
| Default model | 20.50 to 20.62 | 58.7% to 59.4% | 26.4% to 28.8% | 9.5% to 11.9% | 0.9% to 3.7% | 35.9 to 36.0 |
| Training teams matched to the asks | 25.56 to 25.76 | 70.6% to 71.9% | 22.3% to 23.3% | 5.2% to 5.5% | 0.6% | 24.3 to 24.7 |
| Training teams drawn over all win rates | 23.75 to 23.97 | 73.8% to 77.1% | 19.9% to 23.3% | 2.6% to 2.9% | 0% to 0.3% | 22.9 to 23.4 |
| Validation teams | 25.74 | 72.4% | 22.2% | 4.7% | 0.7% | 24.6 |
| Random legal teams | 30.00 | 0% | 0.1% | 3.5% | 96.4% | 68.5 |
| Noise floor, difference of two training draws | -0.68 to +0.58 | -3.0 to +1.5 points | -2.4 to +2.6 points | -1.2 to +1.1 points | -0.5 to +0.5 points | -0.23 to +0.63 |

The figures show means of the three seeds or draws. One team repeated, the copy anchor, has 1 build per species and a nearest training distance of 17 fields, its distance to the next closest training team.

</details>

<details><summary>Full numbers: archetype shares</summary>

Setters only, with mega stones counted as setters.

| Teams | Tailwind | Trick Room | Sun | Rain |
|---|---|---|---|---|
| Default model | 42.1% to 48.2% | 27.8% to 37.0% | 32.0% to 35.2% | 10.8% to 14.2% |
| Training teams matched to the asks | 48.3% to 49.9% | 24.6% to 26.0% | 27.4% to 29.7% | 7.9% to 9.7% |
| Noise floor | -3.9 to +2.4 points | -2.7 to +2.3 points | -2.9 to +4.1 points | -2.5 to +1.9 points |

</details>

<details><summary>Method notes and caveats</summary>

- The unchanged script reproduced the 2026-11-09 result exactly, 351 of 351 values, before any measure was added.
- A set stores the base ability, so a Charizardite Y holder was missed as sun in 196 of 197 cases in model seed 1 until the five setter mega stones were added to the setter list.
- A row drawn from the training teams leaves each team out of its own reference once.
- The model stores Snow Warning on 30 of 67 Froslassite holders, against 3 of 53 in training. Whether that is legal is unchecked.
- The comparison against training teams drawn over all win rates reuses the noise floor from the ask-matched pairs.
- Setters only: whether an archetype is used in battle is not measured.
- Figures are ECharts 6.1.0 rendered to SVG by `build_figures.py` and `render_svg.mjs`; `storyboard.md` gives each figure's sentence.

</details>

Code, tests and the page are on branch `t3/queue-experiments-from-issues`, commit `102fc80d`: `code/scripts/diversity_vs_training.py`, `code/scripts/nearest_training_distance.py`, page `notes/reports/diversity-measurement-2026-11-11/index.html`. Figures are in commit `465a1cc0`. Result file: `~/vgc-data/results/diversity_vs_training_2026-11-11.json`.
