Verdict: keep "the effect of one move depends on the other five Pokemon" as a hypothesis. Do not cite the interaction result.
Why: the only interaction measurement swapped whole Pokemon, not moves, and ran in retired battles. No saved data scores the same move edit on the same set on two different teams under the current setup, except one near-match that is a single post-hoc pair.

Checked 2026-11-13 from saved files only. No battles were run.

## 1. The interaction result you mention

Source: `code/docs/interaction-order.md`, pre-registered 2026-09-30. Result: `~/vgc-data/results/interaction_order.json`.

What it measured. Around each of 16 starting teams, four Pokemon slots were each replaced by a real corpus set of another species, and all 16 on/off combinations were scored. Each combination was scored twice at 384 battles. The variance of win rate over the 16 combinations was split into the part each swap explains alone, the part pairs of swaps explain together, and the part three or more explain together.

| Share of the variance, primary cubes | Estimate | 95% interval |
|---|---|---|
| Each swap alone | 0.871 | 0.816 to 0.916 |
| Pairs of swaps together | 0.116 | 0.078 to 0.168 |
| Three or more together | 0.013 | -0.001 to 0.026 |

Registered verdict, field `verdict`: "no measurable interaction beyond pairs under four species edits". The pair share excludes zero, so in that study the effect of one whole-Pokemon swap did depend on which other swaps were made.

Why it cannot be cited now.
- It battled 36 opponents, not the top 50. `code/docs/top50-36-opponents.md`, table "Produced against 36", row for `results/interaction_order.json`: "every cube battle; the 16 primary anchors were also drawn from the 36, not the 50".
- It ran before 2026-11-07 in listed slot order. Listed-order battles never count. Both rules retire it, as FACTS.md "Numbers that no longer count" states.
- It edits whole Pokemon, not moves. Even rerun, it answers "does a swap depend on other swaps", not "does a move edit depend on the teammates".

## 2. The edit-kind rerun's saved scores

Files: `~/vgc-data/edit_kind_rerun/teams/ruggedness/` and `teams/ruggedness2/`, scores from `run/results/ruggedness_shuffled50.json` and `run/results/ruggedness2_shuffled50_parts/vgc_battles.json`. I diffed all 720 edited teams against their starting team and keyed each edit by the changed set, the field, the old value and the new value.

| Match rule | Edits that occur on two starting teams | Of them, move edits |
|---|---|---|
| Identical set, same field, same old and new value | 2 | 0 |
| Same species and same four moves, same move swap, other fields may differ | 1 | 1 |
| Same species, same field change, rest of the set may differ | 35 | 3 |

The two identical edits are not on different teams. Tournament-ready teams `a02` (MB655) and `a14` (MB607) are byte-identical files, one team listed twice in the corpus. So they serve as a luck check instead: the same edit on the same team, scored twice.

| Edit, same team twice | Effect on `a02` | Effect on `a14` | Difference |
|---|---|---|---|
| Milotic ability Competitive to Cute Charm | -6.2 | -7.8 | 1.6 |
| Incineroar ability Intimidate to Blaze | +0.6 | -1.6 | 2.1 |

Effect = edited score minus the mean of the team's four unchanged scores, in win-rate points. The eight unchanged scores of that one team have a standard deviation of 2.34 points, close to the 2.55 points of one score's noise in FACTS.md.

The three move near-matches:

| Move edit | Team 1, effect | Team 2, effect | Difference | Sets otherwise |
|---|---|---|---|---|
| Sylveon Detect to Giga Impact | `a04` +5.8 | `a23` -5.1 | 10.9 | same item, ability, alignment and four moves; Stat Points differ |
| Basculegion Aqua Jet to Surf | `a08` -1.8 | `s00` -0.2 | 1.6 | item, Stat Points and one other move differ |
| Whimsicott Tailwind to Beat Up | `s04` -6.6 | `s10` -9.2 | 2.5 | item, alignment and Stat Points differ |

Noise of a difference between two such effects: each effect has about 2.85 points of noise, one edited score at 2.55 against a four-score mean at 1.28, so a difference has about 4.0 points. The Sylveon pair differs by about 2.7 times that. It is the only same-set move pair in 720 edits, it was picked after looking, each edited score is a single 384-battle score, and the two Sylveon sets do not have identical Stat Points. It is a lead for the measurement below, not a result.

## 3. The older move experiment

Source: `code/docs/smoothness-experiment.md`, "Frozen design", and `code/docs/smoothness-database.md`. Results: `~/Documents/code/vgc-team-generator-pilot/results/smoothness_top50_50x50/` and `smoothness_member_coverage/`.

Battle setup: 2026-10-01, 50 random-sampler teams and the 50 top-placement entries, 8 random move replacements per team, 4 panels of 100 battles per team, the 50 top-placement entries as opponents, behaviour cloning on both sides. The design "preserves roster and move order because those may affect the fixed policy", so slot order was listed, not shuffled. It is retired under the 2026-11-07 rule.

Same move edit on two teams, from `edits.csv` and the team files:

| Match rule | Count | Edits |
|---|---|---|
| Identical set | 0 | |
| Same species and four moves | 1 | Sinistcha Matcha Gotcha to Memento: -2.5 on `tournament-001`, -12.0 on `tournament-037` |
| Same species only | 3 | adds Incineroar Flare Blitz to Low Sweep, -0.5 and -9.0, and Vileplume Sleep Powder to Seed Bomb, -1.75 and -2.25 |

The member-coverage extension adds 34 edits on 8 teams and repeats none. The Sinistcha gap of 9.5 points is about 1.9 times its noise. It does not count: listed slot order, one pair, found after the fact.

## 4. A measurement that would answer it

Recurring sets, same species and same four moves, counted with `corpus.load_corpus()` and `top50.ids()` minus the MB494 duplicate:

| Pool | Teams | Distinct sets | Sets on two or more teams | Also same item, ability and alignment |
|---|---|---|---|---|
| 49 distinct top-50 teams | 49 | 168 | 49 | 46 |
| 636-team training corpus | 636 | 1,124 | 410 | 501 |

In the top 50, the most shared set is Charizard Heat Wave, Weather Ball, Solar Beam, Protect on 12 teams. For 23 of the 49 recurring sets, two of the carrying teams share none of their other five species. Only one set recurs solely on teams with the same other five species.

Smallest design.
- Take 48 recurring top-50 sets. For each, draw one random legal move edit and apply the identical edit to that set on two top-50 teams with different teammates.
- Score each edited team twice, 384 battles, 50 opponent files, shuffled slots.
- Starting-team scores: four per team. The queued top-50 study, issue #317, already scores each of the 49 teams once plus three replays under the same evaluator, so they can be reused if it runs first.
- Test, following the noise correction of `interaction-order.md`: for each set, take the difference between the two teams' effects in replicate one times the same difference in replicate two, with replicate one using two of the four starting scores and replicate two the other two. Its mean is unbiased for the true squared difference. The hypothesis holds if the 2,000-resample bootstrap interval over sets lies above zero.

Cost:

| Part | Scores | Battles | Time at 43 battles a second |
|---|---|---|---|
| Edited teams, 48 sets x 2 teams x 2 replicates | 192 | 73,728 | about 29 minutes |
| Starting teams, if #317 has not run, 49 x 4 | 196 | 75,264 | about 29 minutes |
| Total, worst case | 388 | 148,992 | about 58 minutes |

Power, from a simulation in my scratchpad at the stated noise. Let tau be the standard deviation of a team-specific part of one edit's effect. With 48 sets: tau 2 points detected 58% of the time, tau 3 points 96%, tau 4 points nearly always, false alarms 4%. With 24 sets, half the edited cost: 38%, 73% and 95%. Tau 3 means two teams' effects of the same edit differ by about 4.2 points on average, close to the 4.84-point typical move edit.

## 5. The name for the hypothesis

| Source | Exact quote | Where |
|---|---|---|
| Szendro, Schenk, Franke, Krug, de Visser 2013, "Quantitative analyses of empirical fitness landscapes", J. Stat. Mech. P01005, https://arxiv.org/abs/1202.4378 | "Magnitude epistasis is present when the fitness effect of a mutation at a given locus has a definite sign (beneficial or deleterious) irrespective of the alleles at other loci, while the magnitude depends on the genetic background." | Section 1, Introduction |
| Same | "sign epistasis occurs when mutations are beneficial in some genetic backgrounds, but not in others." | Section 1, Introduction |
| Same | "Sign epistasis causes pathways to become inaccessible by natural selection and thus introduces ruggedness into the landscapes" | Section 1, Introduction |

Mapping for the post: a Pokemon set is a locus, the other five Pokemon are the genetic background, and "the move's effect depends on the other five" is epistasis. If the sign flips, as in the Sylveon near-match, it is sign epistasis, the term the project already uses.

## What I could not verify

- No game-specific source stating the idea. I searched only the fitness-landscape literature.
- Whether the two Sylveon sets' Stat Points difference alone explains the 10.9-point gap. Only the design above can separate it from teammates.
- Battle pace: 43 battles a second is the rerun's pace quoted in `code/docs/top50-one-edit.md`, "Budget". It was not remeasured.
