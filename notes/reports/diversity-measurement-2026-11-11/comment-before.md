**Result of [Measure diversity of the default model's teams](https://github.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/issues/264), 2026-11-11.** The default model writes about 5 fewer builds per species than its training teams, and its teams sit about 12 fields further from the closest training team. Both gaps are far outside the training-split noise floor.

The default model is the 80-epoch labelled 84-field model at guidance 2. Each row is 1,088 teams, 64 per ask on asks 1/24 to 17/24. The baseline is training teams matched to the asks, drawn the same way. Ranges are the three sampling seeds or the three training draws.

| Teams | Builds per species at 30 uses | Line-up copied, 0 swaps | 1 swap | 2 swaps | 3 or more swaps | Nearest training distance, mean fields of 84 |
|---|---|---|---|---|---|---|
| Default model | 20.50 to 20.62 | 58.7% to 59.4% | 26.4% to 28.8% | 9.5% to 11.9% | 0.9% to 3.7% | 35.9 to 36.0 |
| Training teams matched to the asks | 25.56 to 25.76 | 70.6% to 71.9% | 22.3% to 23.3% | 5.2% to 5.5% | 0.6% | 24.3 to 24.7 |
| Training teams drawn over all win rates | 23.75 to 23.97 | 73.8% to 77.1% | 19.9% to 23.3% | 2.6% to 2.9% | 0% to 0.3% | 22.9 to 23.4 |
| Validation teams | 25.74 | 72.4% | 22.2% | 4.7% | 0.7% | 24.6 |
| Random legal teams | 30.00 | 0% | 0.1% | 3.5% | 96.4% | 68.5 |
| Noise floor, difference of two training draws | -0.68 to +0.58 | -3.0 to +1.5 points | -2.4 to +2.6 points | -1.2 to +1.1 points | -0.5 to +0.5 points | -0.23 to +0.63 |

What to read in the table:

- **Builds.** A build is a set without its Stat Points. At 30 uses of a species the model shows 20.6 builds against 25.7 in matched training. Restricted to the species both sides use 30 times, the gap is 4.84 to 5.00 builds.
- **Novelty.** The model copies a training line-up less often, 59% against 71%, and the difference goes to one and two swaps. Line-ups three or more swaps away stay under 4% and do not clear the floor against matched training.
- **Nearest training distance.** No model team is a copy at 0 fields. The median is 34 fields against 24 for a training team's closest other training team.
- **Second training row.** Training teams drawn over all win rates show fewer builds per species, 23.8, than the ask-matched row, so part of the matched row's variety comes from matching the label mix.

Archetype shares, setters only, with mega stones counted as setters:

| Teams | Tailwind | Trick Room | Sun | Rain |
|---|---|---|---|---|
| Default model | 42.1% to 48.2% | 27.8% to 37.0% | 32.0% to 35.2% | 10.8% to 14.2% |
| Training teams matched to the asks | 48.3% to 49.9% | 24.6% to 26.0% | 27.4% to 29.7% | 7.9% to 9.7% |
| Noise floor | -3.9 to +2.4 points | -2.7 to +2.3 points | -2.9 to +4.1 points | -2.5 to +1.9 points |

Trick Room and sun are above matched training beyond the floor. Tailwind and rain are not. Grassy, misty and psychic terrain are near 0% everywhere; their setting abilities are not in the Regulation M-B vocabulary.

Method notes and caveats:

- The unchanged script reproduced the 2026-11-09 result exactly, 351 of 351 values, before any measure was added.
- A set stores the base ability, so a Charizardite Y holder was missed as sun in 196 of 197 cases in model seed 1 until the five setter mega stones were added to the setter list.
- A row drawn from the training teams leaves each team out of its own reference once.
- The model stores Snow Warning on 30 of 67 Froslassite holders, against 3 of 53 in training. Whether that is legal is unchecked.
- The comparison against training teams drawn over all win rates reuses the noise floor from the ask-matched pairs.
- Setters only: whether an archetype is used in battle is not measured.

Code, tests and the page are in commit `102fc80d` on the local branch `t3/queue-experiments-from-issues`, not yet pushed: `code/scripts/diversity_vs_training.py`, `code/scripts/nearest_training_distance.py`, page `notes/reports/diversity-measurement-2026-11-11/index.html`. Result file: `~/vgc-data/results/diversity_vs_training_2026-11-11.json`.

