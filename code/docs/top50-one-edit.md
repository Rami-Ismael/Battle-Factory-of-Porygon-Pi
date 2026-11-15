# One edit on the top-50 meta teams themselves: pre-registration (2026-11-13)

Written before any battle of this study. Script: `scripts/top50_one_edit.py`. Tests: `scripts/test_top50_one_edit.py`.
Output folder: `~/vgc-data/top50_one_edit/`. Issue 317. Owner request, 2026-11-13: "test the actually top 50 pokemom one edit
difference in performance", and a figure "where x is ranking of the pokemon team base on the win then y is the how much
does one edit change the performance".

**Confirmatory** for the two readings of the six edit kinds and for the rank correlation. Everything else is exploratory.

## Why

The edit-kind study and its rerun (`docs/ruggedness2.md`, issue 313) edited two groups of starting teams: 24 real teams
from outside the top 50, and 16 search-found teams. The top 50 were only the opponents. No study under the current
battle rules has edited the top-50 teams. The October move-edit study (`docs/smoothness-experiment.md`) did edit them,
with move edits only and in listed slot order, so its battles no longer count.

Earlier work searched: `experiments-index.md` (one match, the September edit-kind study), `code/docs/` (the two studies
above), and the web for single-edit sensitivity of competitive Pokemon teams (no match, 2026-11-13).

## Contract

| Row | Value |
| --- | --- |
| Claim | On the top-50 meta teams, one legal edit of some kind has an effect above zero and is larger than one step on the Pest Control benchmark. Refuted if no kind's size has an interval above zero. If it holds for a kind, a search that proposes edits of that kind near strong teams cannot assume the next team scores like the last. The claim stops at win rate under the behaviour-cloning policy against these 50 files. |
| Fields | No model. Every field of an edited team is copied from its starting team except the one thing the edit replaces. Whole-Pokemon edits copy a real set from the team corpus. |
| Vocabulary | Moves: the validator-probed learnset of the species, 189 species. Abilities: the species' legal abilities. Items: the conditional item domain of `mutation_closure_test.py`. Stat alignment: 21 values. Stat Points and whole sets: copied from the 636-team corpus. |
| Training teams | None. |
| Scorer | Behaviour-cloning policy on both sides. Opponents: all 50 files of `top50.files()`, 49 distinct teams, a starting team's own file included. 384 battles per score. Slot order shuffled every battle. Seeds: 101 for the first score, 202 for edits, 202, 303 and 404 for the three replays. |
| Search region | No search. Starting teams: the 49 distinct top-50 teams, rewritten by `team_to_text`. Edits: 3 distinct edits of each of 6 kinds for each team, 18 edited teams, drawn by the seeded operators A to F with seed 1. Every team passes Showdown's validator. |
| Settings | Scientific: the edit kind, and the group of starting teams when the three groups are compared. Nuisance: which Pokemon and which replacement an edit draws, the slot shuffles. Fixed: opponents, policy, battles per score, seeds. No size cutoff is used. |
| Plan | Below. |

### What an edit of each kind does

One Pokemon of the six is picked at random, then:

- **One move replaced.** One of its moves becomes another move its species can learn.
- **Ability replaced.** Its ability becomes another ability legal for its species.
- **Held item replaced.** Its item becomes another item from the conditional item domain, never one a team-mate holds.
- **Stat alignment replaced.** Its stat alignment becomes one of the other 20.
- **Stat Points copied from another set.** Its Stat Points are replaced, all six numbers at once, by the Stat Points of a
  real set of the same species from the 636-team corpus. Nothing else on the Pokemon changes. It is not a nudge of a
  few points: `Garchomp: HP 32, Atk 20, SpD 5, Spe 9` became `Atk 32, Def 1, SpD 1, Spe 32`.
- **Whole Pokemon replaced.** The slot is replaced by a real corpus set of a species not on the team, with its item,
  ability, moves, stat alignment and Stat Points. A clashing item is redrawn.

Four kinds draw a new value. Two kinds copy from real teams: the Stat Points edit and the whole-Pokemon edit.

Edits are uniform over what is legal, not over what a player would choose. Read from the plan: `Grimmsnarl: Prankster ->
Pickpocket`, `Dragapult: Protect -> Phantom Force`, `Kingambit: Focus Sash -> Wacan Berry`. Most edits to a tuned team
should lower its win rate, so the analysis also reports the signed mean change.

## Plan

- **Primary measure.** The noise-corrected size of one edit, for each edit kind, in win-rate percentage points:
  `sqrt(mean((edit - first score)^2) - mean((replay - first score)^2))`, with a 95% interval from 2,000 bootstrap
  resamples of starting teams. The rerun saved no interval for this number; this study does.
  Why squares: a measured move is the edit's real move plus battle noise, and the two are independent, so their
  variances add. A replay is the same measurement with a real move of zero, so its mean square is the noise part alone,
  including the noise of the shared first score. Subtracting it leaves the real part. This assumes an edited team is
  about as noisy as its starting team, which holds while both win rates are near 0.5; `test_top50_one_edit.py` checks
  the formula recovers a planted size. The September plan used the same formula without stating the reason.
- **Question.** Two readings of each kind, and no chosen number in either. A trade-off is not involved.
  - **Above zero.** The 95% interval of the size excludes zero. The edit then does something battle noise does not.
    This reading says an edit is real and nothing about how large it is.
  - **Against one Pest Control step.** The size is divided by the spread of true scores between starting teams, the
    standard deviation pooled over all three groups, so the groups share one scale; top-50 teams alone are
    range-restricted. One step on Pest Control is divided by the spread between benchmark starting points of its own
    kind: 0.37 from random points, whose scores sit 0.88 from their average, and 0.32 from hill-climbed points, whose
    scores sit 1.05 from theirs. The two kinds are never pooled, because their mean scores are 18.5 and 14.2 and that
    gap would enter the spread and shrink the step to 0.14 (second amendment below). A kind is **larger** when the
    bootstrap interval of the difference in shares lies above zero against both, **smaller** when it lies below zero
    against both, and **undecided** otherwise. This is no longer the September Pest Control test in other units,
    since that test pooled the two kinds. The edit's neighbour correlation, `1 - share^2 / 2`, is still reported.
    Assumption stated: the Pokemon spread stays pooled over the three groups. If their mean scores differ, the pooled
    spread grows and every Pokemon share shrinks, which works against "larger" and not for it.
  - The share of edits beyond 1.96 standard errors, and the replays' share, are reported and decide nothing.
- **Secondary, the rank figure.** One dot for each starting team of all three groups, 89 teams. x: the team's rank by
  win rate, rank 1 wins most. y: the typical move of its 18 edits, in points, noise included, with the replay move as
  a reference line. Statistic: Spearman correlation between rank and edit size, with a bootstrap interval, over all 89
  teams and over the 49 top-50 teams alone. Rank uses the mean of a team's three replays, not the first score its edits
  are measured against, so a lucky first score cannot raise a team's rank and enlarge its edits at once.
- **Budget.** 49 teams × (1 first score + 3 replays + 18 edits) = 1,078 scores × 384 battles = 413,952 battles, about
  2.7 hours at the rerun's pace of 43 battles a second. Laptop only.
- **Stopping rule.** The battle stage ends when every score has 384 battles. It stops early if a score is short, or if
  the top-50 teams' mean first score is outside 0.47 to 0.53.
- **Baselines.** The replays are the noise baseline. The rerun's two groups are the comparison groups, measured with
  the same opponents, policy, battles, seeds and edit operators.
- **Diversity.** Not applicable: nothing is generated or selected.

## Noise floor

- One score: standard error 2.55 points at 384 battles and a win rate of 0.5. A replay moved a score away from
  the first score by 3.40 points for the tournament-ready teams and 3.73 for the search-found teams in the rerun.
- Size of one edit: in the rerun, with 72 edits of a kind, the 95% interval was about ±1.1 to ±1.8 points. The move
  edit read 4.84 with interval 3.35 to 6.23, and the whole-Pokemon swap 5.90 with interval 4.78 to 7.01. With 147
  edits of a kind from 49 teams, expect about ±0.8 to ±1.3 points.
- **What the budget can and cannot detect.** With an interval of about ±0.8 to ±1.3 points, a kind whose true size is
  3 points or more should read above zero. Against the benchmark step the limit is on the benchmark's side: the step's
  share has a 95% interval of 0.28 to 0.54 from random points and 0.27 to 0.45 from hill-climbed points, from 24 and 16
  starting points, and no number of Pokemon battles narrows it. On the rerun's scale of 9.04 points the step is about
  3.3 and 2.9 points. A kind must sit near 0.6 of the spread or above to read larger, as the whole-Pokemon swap did
  from tournament-ready teams at 0.65. A kind between about 0.2 and 0.6 will read undecided, as the move, item and
  ability edits do in the rerun. Doubling to 6 edits of each kind would cost 752,640 battles, about 4.9 hours, and
  would narrow the interval only by the part that comes from the draw of edits, not the part that comes from the draw
  of teams.
- Rank correlation over 89 teams: the preview on the rerun's 40 teams read −0.16 with interval −0.44 to 0.15, so a
  correlation weaker than about ±0.2 will stay undecided at 89 teams.

## Probes

Saved in `~/vgc-data/top50_one_edit/preflight.json`.

| Probe | Result |
| --- | --- |
| Round trip | Pass. All 49 starting teams come back from `parse_team` and `team_to_text` with every line intact except gender marks (127 dropped across the 50 files) and shiny lines (11). Battles then draw gender at random, which touches only gender-dependent effects. |
| Coverage | Pass. 882 edited teams, 3 of every kind for every team, from 949 draws. |
| Read teams | Done on 24 edits, 4 of each kind. Finding: edits are legal but often not what a player would pick. Reported above; the signed mean change is added to the analysis. |
| Copies | 0 edited teams equal a starting team, a corpus team or another edited team. |
| Separation | Pass. `heldout.assert_clear_files` on all 931 files; no model is trained. |
| Repeat prep | Pass. A second `prep` wrote a byte-identical `plan.json`. |
| Controls, top-50 near 0.50 | Runs inside the battle stage: it stops unless the 49 first scores average 0.47 to 0.53. |
| Controls, uniform-legal near 0 | `smoke` stage, before the battles: 5 teams of `hps_reg_mb_100k.jsonl`, 100 battles each, mean under 0.10. Not run yet: other sessions' jobs hold the laptop. |
| Repeat | `smoke` stage: one team twice on seed 101, same battle count and win rates within battle noise. Not run yet, same reason. |
| Invariance | Covered by the design: every battle shuffles slots. |
| Ceiling, Training smoke, Condition | Skipped: no model, no sampler, no ask. |

## Figures planned

1. **Size of one edit by kind, top-50 teams.** Sentence: on the strongest teams, one edit of this kind moves win rate
   by this much. Expected: the whole-Pokemon swap is largest and larger than a Pest Control step; move, item and ability
   edits are above zero and undecided against the step; Stat Points and stat alignment are smallest. Surprise: a move or
   item edit is as large as the whole-Pokemon swap.
2. **The same sizes for the three groups side by side.** Sentence: edits matter more, or the same, on stronger teams.
   Expected: top-50 sizes are at least as large as the other two groups', as the October move study suggested.
   Surprise: top-50 teams change less than ordinary teams.
3. **Edit size against rank.** Sentence: the more a team wins, the more one edit moves it. Expected: a negative rank
   correlation over the 89 teams. Surprise: flat, as in the 40-team preview.
4. **Direction.** Sentence: a random legal edit to a top-50 team usually costs win rate. Expected: a negative signed
   mean for every kind. Surprise: a kind whose random edits help on average.

## Amendments before any battle

- **2026-11-13, the 5-point size cutoff is removed.** Owner: "you cannot make a 5 point cutoff justifcation you should
  eliminated or replacing with something". The first version of this plan carried the September noise test, which
  needed a size of at least 5 points. No plan ever gave a reason for the number 5. The owner chose two replacements
  from three shown: the interval above zero, and the size against one Pest Control step. The third, the replay move as
  the line, was dropped because it moves with the battles per score. The queue was stopped while it waited, the
  script, tests and report were changed, and the frozen copy was rebuilt. No battle of this study had run.
  `SIZE_CUTOFF` is gone from the script and a test checks that.
- **2026-11-13, the benchmark step is divided by the spread of its own kind of starting point.** An audit of the
  first amendment found that the step's share, 0.14, came from pooling Pest Control's random and hill-climbed
  starting points. Their mean scores are 18.5 and 14.2, so the pooled variance is 5.44 against 0.77 and 1.10 inside
  each kind, and most of the pooled spread is the gap between the two kinds. The Pokemon side has no such gap in the
  rerun: 9.04 points pooled, 9.18 and 7.66 alone. The owner was shown both choices with their effect on the rerun and
  answered "b", each kind on its own spread. `step_share` in the script now does this and a test checks that pooling
  two kinds with different means would shrink the step. On the rerun's saved scores the reading becomes: the
  whole-Pokemon swap is larger from tournament-ready teams and undecided from search-found teams, whose interval of
  the difference starts at −0.01; move, item and ability edits are undecided; stat alignment from search-found teams
  is smaller. The September plan pooled the same way, so its Pest Control test carried the same weakness. The queue
  was stopped while it waited, and the frozen copy was rebuilt. No battle of this study had run.

## Result, checked 2026-11-14

Ran 2026-11-13 23:26 to 2026-11-14 01:30. Counts as planned: 49 teams, 882 edited teams, 147 replays, 1,078 scores of
384 battles, 413,952 battles. Checks passed: uniform-legal teams 1.6%, one team twice on seed 101 at 51.3% and 52.1%,
top-50 first scores average 0.494. The sizes were recomputed from the raw battle counts with separate code and agree.
Comment and storyboard: issue #317, 2026-11-14. Storyboard folder: `notes/reports/top50-one-edit-2026-11-14/`.

| Edit, top-50 teams | Size as planned, points | Above zero | Share of the spread | Against one Pest Control step | Average change, points |
| --- | --- | --- | ---: | --- | --- |
| Whole Pokemon copied from a real set | 4.91, 3.74 to 5.83 | yes | 0.62 | larger | −2.51, −3.53 to −1.48 |
| One move replaced | 3.04, 1.05 to 4.10 | yes | 0.39 | undecided | −2.08, −2.93 to −1.23 |
| Held item replaced | 3.12, 1.21 to 4.32 | yes | 0.40 | undecided | −1.52, −2.55 to −0.57 |
| Ability replaced | 2.11, 0.00 to 3.31 | not shown | 0.27 | undecided | −1.25, −2.10 to −0.43 |
| Stat alignment replaced | 0.00, 0.00 to 1.81 | not shown | 0.00 | smaller | −0.04, −0.82 to 0.78 |
| Stat Points copied from another set | 0.00, 0.00 to 1.01 | not shown | 0.00 | smaller | 0.52, −0.24 to 1.30 |

The claim holds as registered: the whole-Pokemon swap is above zero and larger than a Pest Control step, the second by
a margin of 0.01. Rank correlation: −0.28, interval −0.53 to −0.00, over the 49 top-50 teams, and −0.17, interval −0.37
to 0.02, over all 89. Against the figures planned: figure 1 as expected except that the ability edit is not shown above
zero; figure 2 looked like the named surprise, top-50 sizes below the tournament-ready ones, and the check below does
not support it; figure 3 undecided; figure 4 as expected for four kinds, with no direction for the two Stat kinds.

## Deviations

- **The run.** None. Seeds, counts, battles per score and the stopping rule are as planned.
- **Checks added after seeing the result, exploratory.**
  - *The comparison score.* The plan measures each edit against the team's first score, seed 101. For the top-50
    teams the first scores average 0.78 points below the three replays, and 1.25 points below the replay on the edits'
    own seed, 202. Random edits mostly lower win rate, so that gap hides part of each drop and the planned sizes read
    low. A test for luck shared between an edit and the replay on its seed found none: correlations of −0.19 and 0.13
    with a null standard error of 0.14. So the four unchanged scores are equally fair comparisons. Using each in turn,
    with the other three as the noise, and averaging gives: whole Pokemon 5.28, 4.54 to 5.98; move 3.87, 2.96 to 4.69;
    item 3.62, 2.58 to 4.46; ability 2.76, 1.81 to 3.50; stat alignment 1.06, 0.00 to 2.02; Stat Points 0.00, 0.00 to
    0.58. One reading changes: the ability edit is above zero. The same check on the earlier groups gives, for
    tournament-ready teams, whole Pokemon 5.14, move 4.52, item 4.16, ability 3.57, and for search-found teams 6.00,
    5.86, 2.03, 1.89. No difference between the three groups is shown.
  - *The rank.* With rank taken from the replays on seeds 303 and 404, which share no seed with the edits, and edit
    size measured against the replay on seed 202, the rank correlation over the top-50 teams is −0.01, interval −0.28 to
    0.26. No trend with rank is shown.
- **A wrong sentence in this plan.** The Question section says that pooling the Pokemon spread over the three groups
  can only shrink a Pokemon share. The pooled spread fell from 9.04 points in the rerun to 7.90, because the top-50
  teams sit closer together, 6.89 points on their own. Every share rose, and the rerun's whole-Pokemon swap from
  search-found teams moved from undecided to larger with no new battle. The rerun's shares should be quoted from
  `docs/ruggedness2.md`.
- **For the next study of this kind.** Score the unchanged team on the same seeds as its edits, or measure every edit
  against the mean of several unchanged scores, so one first score cannot shift every edit of a team.
