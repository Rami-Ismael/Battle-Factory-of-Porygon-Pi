**Storyboard for [Rerun the one-edit study per edit kind with shuffled slots against the full meta](https://github.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/issues/313), 2026-11-13.**

A team is six Pokemon, listed in six slots. The meta is the 50 highest-placing team files, one team listed twice. We change one thing on a team, battle the old and the new team against the meta, and see how far the win rate moves. September's battles used rules the project no longer counts, so the study ran again.

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/COMMIT/notes/reports/edit-kind-rerun-2026-11-13/issue-313/figures/storyboard-dark.svg"><img width="800" alt="Eight frames. The rerun battled every team against all 50 meta files with the six Pokemon shuffled before each battle. An edit changes one thing on one Pokemon, in one of six kinds, and each kind is tried 3 times on a team. Replays of the unchanged team show battle noise, about 3.40 points. With noise removed, one edit moves win rate by 5.90 points for a whole Pokemon swap, 4.84 for a move, 4.35 for an item, 3.66 for an ability, 1.95 for stat alignment and 1.60 for Stat Points. Only the whole Pokemon swap clears the 5-point cutoff set in the plan on 2026-09-28, and only it passes both tests, from both groups of starting teams." src="https://raw.githubusercontent.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/COMMIT/notes/reports/edit-kind-rerun-2026-11-13/issue-313/figures/storyboard-light.svg"></picture>

### How to read this

| Term | What it means here |
|---|---|
| Edit kind | What one rule-abiding edit changes on one Pokemon: one of its four moves, its ability, the item it holds, its stat alignment, its Stat Points, or the whole Pokemon. Frame 2 shows one real edit of each kind |
| Stat alignment, Stat Points | Two settings of a Pokemon's numbers: which stat is raised and which lowered, and how its training points are divided among its stats |
| Shuffled slots | The six Pokemon get a new random order before every battle. The battle policy picks by position, so a fixed order biases scores |
| Replay | The same team battled 384 more times with nothing changed. Every starting team gets 3 replays |
| Battle noise | How far a win rate moves on a replay, by luck alone. Here a replay typically moves a score by 3.40 points |
| Change, in points | The typical size of the win-rate move from one edit, in percentage points, in either direction, with the typical replay move taken out |
| 5-point cutoff | How large the change must be for an edit kind to count as beyond battle noise. Written into the plan on 2026-09-28, before any battle. The plan gives no reason for the number 5 |
| Beyond battle noise | Test 1. A larger share of edits than of replays moves the win rate past what chance allows, and the change is at least 5 points |
| Rougher than Pest Control | Test 2. Pest Control is a standard optimisation benchmark where one step barely changes the score. Passing means a team's score predicts its edited neighbour's score less well than there, by correlation |
| Meta teams, search-found teams | The two groups of starting teams: 24 real teams from outside the top 50, and 16 teams an earlier search produced |

The frames and their sentences are in `notes/reports/edit-kind-rerun-2026-11-13/issue-313/`. The plan is `code/docs/ruggedness2.md`.
