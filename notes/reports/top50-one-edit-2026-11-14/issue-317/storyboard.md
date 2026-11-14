# Storyboard: One edit on the top-50 teams themselves, and edit size against a team's rank

Figure sentence: On the top-50 teams, swapping a whole Pokemon, a move or an item changes win rate by more than battle luck.

## 1. What is new: the top 50 are the teams being edited

- Sentence: The earlier run edited strong teams from outside the top 50. This run edits the top-50 teams themselves, so the result speaks about the teams a search has to beat.
- What we measured: The run matched its plan: 49 teams, 882 edited teams and 147 replays, which is 1,078 scores of 384 battles and 413,952 battles.

## 2. The six kinds of edit, on one real top-50 team

- Sentence: Four kinds pick a new legal value at random, so many edits are not what a player would choose. Two kinds copy from real teams.
- What we measured: This team scored 52.3%, and 56.3%, 50.5% and 51.0% on its 3 replays. The six edited teams scored 49.5%, 45.1%, 41.7%, 49.5%, 52.1% and 46.9%, in the order shown.

## 3. From scores to a change in points

- Sentence: What we measure after an edit is its real move plus battle luck. The two are independent, so they add as squares, like the short sides of a right triangle.
- What we measured: For the whole Pokemon swap on top-50 teams: 6.16 squared minus 3.73 squared is 4.91 squared.

## 4. Is the edit real? Change in points with its 95% interval, top-50 teams

- Sentence: An edit kind is real when its whole interval, the thin capped line, sits above zero. On top-50 teams three kinds are: the whole Pokemon swap, the move edit and the item edit.
- What we measured: Ability edits read 2.11 with an interval that reaches zero. Stat alignment and Stat Points read 0.00, with intervals up to 1.81 and 1.01. No effect is shown for those three, which is not proof they do nothing.

## 5. A check made after the result: does the comparison score matter?

- Sentence: Each team has four scores with nothing changed. The plan measured every edit against the first one only. Using all four gives the same three real kinds and adds the ability edit.
- What we measured: The first scores of the top-50 teams came out 0.78 points below their replays. Edits mostly lower win rate, so part of each drop was hidden and the planned sizes read low. The right column was computed after seeing the result, so it is a check, not the registered result.

## 6. Is the edit large? Its share of the spread between teams, against one step on a benchmark

- Sentence: Each size is divided by how far teams' scores spread, 7.90 points. One step on the Pest Control benchmark is 0.37 of the spread between its random starting points, and 0.32 between its hill-climbed ones.
- What we measured: The whole Pokemon swap is 0.62 of the spread and reads larger than the step, narrowly: the interval of the difference starts at 0.01. Move, item and ability edits are undecided. Stat alignment and Stat Points read smaller than a step.

## 7. Which way does a random edit push a top-50 team?

- Sentence: A random legal edit to a top-50 team usually costs win rate. Four kinds lower it on average. Stat alignment and Stat Points show no direction.
- What we measured: The edits are random legal values, not choices a player would make: 38 of the 147 item edits take away a Mega Stone, and 27 of the 147 move edits take away Protect.

## 8. Top-50 teams beside the other two groups, change in points

- Sentence: Green means the interval sits above zero. The whole Pokemon swap and the move edit are real in all three groups.
- What we measured: The top-50 sizes look smaller than the tournament-ready ones, 3.04 against 4.84 for a move. Against all four unchanged scores they are 3.87 and 4.52 with overlapping intervals, so no difference between the groups is shown.

## 9. Do teams that win more change more?

- Sentence: The planned reading leans slightly towards stronger teams changing more. It vanishes when the comparison score changes, so no trend with rank is shown.
- What we measured: Both readings are for the 49 top-50 teams. Over all 89 teams the planned correlation is −0.17, interval −0.37 to 0.02. Each team's edit size rests on one noisy comparison score, which is why the answer moves.
## Cut

- The 5-point cutoff: the owner removed it from the plan before any battle, so no frame draws it.
- The scatter of all 89 teams: it lives in `figures/edit-size-by-rank.html`. Frame 9 carries its finding as two readings side by side, because the trend depends on the comparison score.
- Shares of the spread for the other two groups: the spread pooled over three groups is 7.90 points against 9.04 in the earlier run, so their shares moved without any new battle. The earlier run's storyboard on issue 313 keeps them.
- The same-seed pairing and the leave-one-seed-out checks: one check, the four unchanged scores, is enough for frame 5. The rest is in the plan's Deviations section.
- Limits such as one battle policy, one opponent pool and many readings with no correction: in the comment below the figure.
