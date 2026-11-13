**Storyboard for [Rerun the one-edit study per edit kind with shuffled slots against the full meta](https://github.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/issues/313), 2026-11-13.**

A team is six Pokemon, listed in six slots. The meta is the 50 highest-placing team files, one team listed twice. We change one thing on a team, battle the old and the new team against the meta, and see how far the win rate moves. September's battles used rules the project no longer counts, so the study ran again.

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/COMMIT/notes/reports/edit-kind-rerun-2026-11-13/issue-313/figures/storyboard-dark.svg"><img width="800" alt="Eight frames. The rerun battled every team against all 50 meta files with the six Pokemon shuffled before each battle. An edit changes one thing on one Pokemon, in one of six kinds, and each kind is tried 3 times on a team. Replays of the unchanged team show battle noise, about 3.40 points. With noise removed, one edit moves win rate by 5.90 points for a whole Pokemon swap, 4.84 for a move, 4.35 for an item, 3.66 for an ability, 1.95 for stat alignment and 1.60 for Stat Points. No size cutoff is used. Four kinds have a 95% interval above zero from tournament-ready teams, and the two Stat kinds do not. As a share of the 9.04-point spread between teams, the same four kinds are larger than one step on the Pest Control benchmark, which is 0.14 of its own spread. From search-found teams only the whole Pokemon swap and the move edit hold." src="https://raw.githubusercontent.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/COMMIT/notes/reports/edit-kind-rerun-2026-11-13/issue-313/figures/storyboard-light.svg"></picture>

### How to read this

| Term | What it means here |
|---|---|
| Edit kind | What one rule-abiding edit changes on one Pokemon. Four kinds pick a new legal value at random: one of its four moves, its ability, the item it holds, its stat alignment |
| Copied edits | The other two kinds copy from real teams. Stat Points: all six numbers are replaced by those of another real set of the same species. Whole Pokemon: the slot is replaced by a real set of a species not on the team, with its moves, item, ability, stat alignment and Stat Points |
| Stat alignment, Stat Points | Two settings of a Pokemon's numbers: which stat is raised and which lowered, and how its training points are divided among its stats |
| Shuffled slots | The six Pokemon get a new random order before every battle. The battle policy picks by position, so a fixed order biases scores |
| Replay | The same team battled 384 more times with nothing changed. Every starting team gets 3 replays |
| Battle noise | How far a win rate moves on a replay, by luck alone. Here a replay typically moves a score by 3.40 points |
| Change, in points | The size of the real move one edit causes, in win-rate percentage points, in either direction. Computed as the square root of the mean squared move after an edit minus the mean squared move of a replay, because independent parts add as squares |
| 95% interval | The range the true size plausibly lies in. Found by redrawing the starting teams 2,000 times and recomputing the size each time |
| Above zero | Reading 1. The whole interval of an edit kind's size sits above zero, so the edit does something a replay does not. It says the edit is real, not how large |
| Spread between teams | How far the starting teams' true win rates lie from one another, as a standard deviation with battle noise removed: 9.04 points |
| Share of the spread | An edit's size divided by the spread between teams. A share of 0.65 means one edit moves a team about two thirds as far as teams differ from each other |
| One Pest Control step | Reading 2. Pest Control is a standard optimisation benchmark where one step barely changes the score: 0.14 of that benchmark's own spread. An edit kind is larger when the interval of its share minus the step's share sits above zero, and undecided when that interval includes zero |
| The dropped 5-point line | The plan of 2026-09-28 also required a change of at least 5 points and gave no reason for the number. The owner removed it on 2026-11-13, after the result. Under it only the whole Pokemon swap passed |
| Tournament-ready teams, search-found teams | The two groups of starting teams: 24 real teams players built for tournaments, drawn from outside the top 50, and 16 teams an earlier search produced |

The frames and their sentences are in `notes/reports/edit-kind-rerun-2026-11-13/issue-313/`. The plan is `code/docs/ruggedness2.md`.
