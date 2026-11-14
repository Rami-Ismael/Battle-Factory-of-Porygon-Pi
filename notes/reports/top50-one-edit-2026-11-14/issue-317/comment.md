**Storyboard for [One edit on the top-50 teams themselves, and edit size against a team's rank](https://github.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/issues/317), 2026-11-14.**

A team is six Pokemon. The top 50 are the 50 highest-placing team files, one team listed twice, so 49 distinct teams. We change one thing on a top-50 team, battle the old and the new team against all 50 files, and see how far the win rate moves. Every score is 384 battles with the six Pokemon shuffled and the same behaviour-cloning policy playing both sides.

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/COMMIT/notes/reports/top50-one-edit-2026-11-14/issue-317/figures/storyboard-dark.svg"><img width="800" alt="Nine frames. The 49 distinct top-50 teams each got 18 edits, 3 of each of six kinds, and 3 replays with nothing changed. A replay moves a score by 3.73 points of battle luck. With luck removed, one edit moves a top-50 team&#x27;s win rate by 4.91 points for a whole Pokemon swap, 3.04 for a move and 3.12 for an item, all with 95% intervals above zero. Ability edits read 2.11 with an interval that reaches zero, and stat alignment and Stat Points read 0.00. Using all four unchanged scores as the comparison gives 5.28, 3.87, 3.62 and 2.76, and the ability edit then sits above zero. As a share of the 7.90-point spread between teams, only the whole Pokemon swap, 0.62, is larger than one Pest Control step, 0.32 to 0.37. Random edits lower win rate on average for four kinds. No difference between the three groups of teams is shown, and no trend with a team&#x27;s rank." src="https://raw.githubusercontent.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/COMMIT/notes/reports/top50-one-edit-2026-11-14/issue-317/figures/storyboard-light.svg"></picture>

### How to read this

| Term | What it means here |
|---|---|
| Top-50 teams | The 50 highest-placing team files in the project's pool, 49 distinct teams. In this run they are both the teams being edited and the opponents |
| Edit kind | What one rule-abiding edit changes on one Pokemon. Four kinds pick a new legal value at random: one of its four moves, its ability, the item it holds, its stat alignment |
| Copied edits | The other two kinds copy from real teams. Stat Points: all six numbers are replaced by those of another real set of the same species. Whole Pokemon: the slot is replaced by a real set of a species not on the team |
| Stat alignment, Stat Points | Two settings of a Pokemon's numbers: which stat is raised and which lowered, and how its training points are divided among its stats |
| Replay | The same team battled 384 more times with nothing changed. Every starting team gets 3 replays |
| Battle luck | How far a win rate moves on a replay, with nothing changed. For top-50 teams a replay moves a score by 3.73 points |
| Change, in points | The size of the real move one edit causes, in win-rate percentage points, in either direction. Computed as the square root of the mean squared move after an edit minus the mean squared move of a replay |
| 95% interval | The range the true size plausibly lies in. Found by redrawing the starting teams 2,000 times and recomputing the size each time |
| Above zero | Reading 1. The whole interval of an edit kind's size sits above zero, so the edit does something a replay does not |
| First score, four unchanged scores | Each team has four scores with nothing changed: the first score and three replays. The plan measures every edit against the first score. The check in frame 5 uses each of the four in turn and averages |
| Spread between teams | How far the starting teams' true win rates lie from one another, as a standard deviation with battle luck removed: 7.90 points over the three groups together |
| One Pest Control step | Reading 2. Pest Control is a standard optimisation benchmark. One step on it is 0.37 of the spread between its random starting points and 0.32 between its hill-climbed ones. An edit kind is larger when the interval of its share minus the step's share sits above zero for both, smaller when it sits below zero for both, and undecided otherwise |
| Average change | The mean of the edits' moves with their sign kept, so a kind that usually lowers win rate reads negative |
| Tournament-ready teams, search-found teams | The two groups from the earlier run: 24 real teams players built for tournaments, drawn from outside the top 50, and 16 teams an earlier search produced |
| Rank correlation | Whether teams that win more also change more when edited. Minus 1 means the strongest teams change most, 0 means no link |

The frames and their sentences are in `notes/reports/top50-one-edit-2026-11-14/issue-317/`. The plan is `code/docs/top50-one-edit.md`.
