# How near the line do the training teams themselves land?

Issue #72 on the map in issue #70. The line is got = ask, where the ask is the win rate requested from the model and got is the win rate its teams measure in fresh battles. Win rate here means one battle against each of the 49 distinct top-50 meta teams, behaviour-cloning policy on both sides, with slots shuffled every battle.

If a model returned training teams perfectly, its error could not fall below what those teams score when battled again. This experiment measures that error. A label comes from battles that are already over, some with listed slot order, and a team is filed under its label partly by luck. Neither survives a fresh battle, so the training data alone may miss the line.

## Arms

Both arms use the same 1,088 teams and the same 48 fields, so they differ in where each team's Stat Points come from.

| Reader name | Code ID | Generated or stored | Copied |
|---|---|---|---|
| Stored Stat Points | `stored` | All 84 fields stored with the team | Nothing |
| Redrawn Stat Points | `redrawn` | 48 fields stored with the team, Stat Alignment repaired | Stat Points, per Pokémon, from a random same-species corpus team |

"Redrawn Stat Points" repeats what the 48-field model does when it writes a team, `to_paste(repair=True)`. This arm repairs Stat Alignment because the 48-field model repairs its own. The gap between the arms therefore measures the cost of hiding Stat Points and of repairing Stat Alignment together, as the 48-field model pays them.

## Teams

The teams come from the all-teams set in `lossdata84.npz` (SHA-256 `02463cbd…`): labelled, not in its 2% validation split, and holding exactly one set of Stat Points. The test 500 and the held-out meta teams ranked 51-100 are not in that set.

A team belongs to the ask k/24 nearest its label. Each of the asks 1/24 to 17/24 draws 64 teams at random from its group, with a seeded generator, and keeps a team only if both of its pastes pass the Showdown validator. Picking the 64 labels closest to the ask would favour teams with few battles, because only those land exactly on simple fractions.

All 1,088 teams passed on the first try. No team has the same paste in both arms. The groups hold 2,044 to 28,216 teams.

Asks 18/24 to 23/24 get no battles. The ticket covers asks up to 17/24, and the training data holds only 610, 119, 30, 10, 0, and 0 teams there. The report lists those counts.

The all-teams set is not the set the 48-field model trained on. Of the 64 teams at each ask from 13/24 up, at most 3 are in that model's loop-teams or battle-records slices. Read these numbers as the limit for the all-teams 84-field model. The report gives the overlap per ask.

## Battles

Each team battles once against each top-50 meta team, 49 battles per team and 106,624 in all. The battles go to a private database, `~/vgc-data/training_teams_line/battles.sqlite`, so no battle behind a label is reused. The queue `~/vgc-data/training_teams_line/ttl.zsh` runs a frozen copy of the code from `run/` after the line-up repeats and 84-field training queues end, and after two idle checks 5 minutes apart. A rerun battles only the cells still missing.

## Measures

The main number is the mean squared error between each ask and the mean got at that ask, on asks 1/24 to 17/24. The map's bar for near the line is under 0.001, a typical miss of about 0.03.

The report also gives:

- A 95% interval on each arm's error and on redrawn minus stored, resampling teams within each ask. A resample picks the same teams in both arms.
- The error with sampling noise removed, because the mean of 64 teams adds its own variance to a squared miss.
- The error on asks 0.1 to 0.7, the range where the 48-field model alone scored 0.0138.
- Per ask: mean label, median label battles, signed miss, win-rate scatter across teams split into battle noise and real team differences, distinct line-ups, effective species, and top species share.

## Expected result

Got should sit above the ask at low asks and below it at high asks, because a label holds some luck and fresh battles remove it. A re-battle of 12 elite teams fell from a label mean of 0.568 to 0.490, and shuffling slots took the top 300 search teams from 0.672 to 0.603. Those teams were picked for high scores, so this random draw should fall by less.

The stored arm should miss the 0.001 bar, with an error between 0.002 and 0.006. The two arms should differ by less than 0.01 in mean got, because the behaviour-cloning policy has barely noticed Stat Points in earlier runs.

## How the result feeds the decision

Issue #74 uses this number as the limit from the data. If the stored arm's error is close to the error of the 84-field model in issue #71, the model cannot do much better on these labels and the first change belongs in the data. If the stored arm lands well below the model, the gap belongs to the fit or to guidance, which issue #73 separates.

## Result

Mean squared error on asks 1/24 to 17/24, against the bar of 0.001:

| Reader name | Error | 95% interval |
|---|---:|---:|
| Stored Stat Points | 0.0038 | 0.0033 to 0.0047 |
| Redrawn Stat Points | 0.0050 | 0.0044 to 0.0060 |
| Generated, all teams, 21 epochs (`gen84_all_ep21`, from its own grid) | 0.0192 | not computed |

Both arms miss the bar, so even a perfect copy of the training teams would miss it. The stored arm lands well below the model, so under the rule above the gap belongs to the fit or to guidance. Hiding Stat Points adds 0.0012 to the error (0.0005 to 0.0020).

Labels exaggerate at both ends: a mean label of 0.046 re-battles at 0.117, and 0.705 at 0.580. This mixes battle luck with the change from listed to shuffled slots.

Across teams at one ask, win rate spreads by a standard deviation of about 0.094 on average in the stored arm. Battle luck accounts for at most 0.064 of that and real team differences for at least 0.067. One battle per opponent cannot separate them further, so these two are bounds.

Against the expected result: the stored error fell inside 0.002 to 0.006. The arms did not stay within 0.01 in mean got: redrawn teams got 0.012 more (0.007 to 0.018).

## Reproduce

Run the stages from `~/vgc-data/training_teams_line/run` with `/tmp/vgc-pilot/.venv/bin/python`. The results go to `~/vgc-data/results/training_teams_line.json`.

```
python training_teams_line.py draw
MATCHUP_DB=~/vgc-data/training_teams_line/battles.sqlite python training_teams_line.py battle
python training_teams_line.py report
```

Set `TTL_SMOKE=1` for 2 asks and 2 teams with separate files. Run the arithmetic tests from `code/` with `.venv/bin/python scripts/test_training_teams_line.py`.
