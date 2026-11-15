# Ruggedness per edit type, from meta and search-found teams — pre-registration (2026-09-28)

Extends docs/ruggedness.md (run 1: 24 meta anchors, edits A–E, Pest Control, Hartmann-6).
Written before any run-2 battle. Script: scripts/ruggedness2.py; output results/ruggedness2.json.

## What is added
- Edit type F (species edit): one Pokémon replaced by a real corpus set of a species not on the
  team (candidate-copy, Species + Item Clause repaired; operator F of mutation_closure_test.py).
  Repair rule is fixed: the whole set is copied, nothing of the old slot is kept.
- Search-found anchors: the 16 teams of results/rebattle_top.json (combined gradient-guidance loop).
  Edits A–F × 3 each; anchor measured once (seed 101) + 3 replicates (seeds 202/303/404).
- Meta anchors: the 24 of run 1, reused as measured; new F edits × 3 each.
- VGC f = win rate vs the top-50 meta, behaviour-cloning policy both sides, 384 battles,
  one fixed ordered opponent schedule (common random numbers), edits at seed 202 as in run 1.
  Every child passes Showdown's TeamValidator.
- Benchmarks, same 24 random + 16 search-found anchors, 15 edits, 3 replicates:
  - Pest Control (COMBO, 25 stages × 5 choices), edit = one stage set to another value.
  - Gridded sphere, 25 coordinates × 5 levels {−1, −½, 0, ½, 1}, f = Σx²; primary edit = one
    coordinate moved one grid step (inward at the edge); secondary edit = one coordinate set to
    any other level (the Pest Control neighbourhood).
  - Search-found benchmark anchors = first-improvement hill climb from a random point, 100
    single-edit proposals, minimising f (noisy f for Pest Control, as the optimiser would see it).

## Statistics (per landscape × start type × edit type)
- Noise variance σ²ₙ from replicates: E[r²] = mean (replicate − anchor)².
- Noise-corrected RMS edit size Δ* = √(E[d²] − E[r²]), d = edit − anchor (win-rate units for VGC).
- var_f = variance of anchor values pooled over BOTH start types of that landscape, minus mean σ²ₙ
  (pooled so search-found anchors, whose spread is range-restricted, share one scale).
- s = Δ*/√var_f (edit size in units of f's spread); ρ = 1 − s²/2 (noise-corrected neighbour
  correlation). 95% CIs by bootstrap over anchors (2,000 resamples).
- VGC only: fraction of edits with |Δ| > 1.96·SE(Δ), against the same fraction for replicates.

## Threshold, fixed now
T1 — beyond battle noise. An edit type passes if (a) its fraction beyond 1.96·SE exceeds the
replicate fraction with the bootstrap 95% CI of the difference above 0, and (b) Δ* ≥ 0.05 win rate.
T2 — beyond the benchmark. VGC is more rugged than Pest Control under an edit type and start type
if the bootstrap 95% CI of ρ_VGC − ρ_Pest lies entirely below 0 (meta vs random anchors;
search-found vs search-found).
Sanity: the gridded sphere (primary edit) must give ρ ≥ 0.95, else the comparison is broken.

Wording the subsection is allowed to use:
- T1 and T2 hold for some edit type → "win rate is rugged under a single legal edit, beyond
  battle noise and beyond Pest Control" (naming the edit types).
- T1 holds, T2 fails everywhere → "one edit moves win rate beyond battle noise, but win rate is
  not more rugged than Pest Control".
- T1 fails everywhere → the non-smoothness claim is dropped.
Reported whatever the outcome.

## Rerun with shuffled slots against the full meta — pre-registered 2026-11-12

Owner request, 2026-11-12: "rerun the experiment with 49 pokemon with a shuffle battle policy on".
Written before any rerun battle. Queue: `~/vgc-data/edit_kind_rerun/ek.zsh`; frozen scripts in its `run/`.

Why: the 2026-09-28 battles faced 36 of the 50 meta files and used listed slot order. Listed-order
battles never count (owner, 2026-11-07), so no number above is a measurement any more.

Naming, 2026-11-13 (owner): the first group of starting teams is called "tournament-ready teams" from here on: real teams
players built for tournaments, drawn here from outside the top 50. The script and the September text call them "meta anchors",
but in this project the meta is the top 50, which are only the opponents in this study.

What changes, and only this:
- Opponents: all 50 files of `top50.files()`, which hold 49 distinct teams. MB493 and MB494 are the
  same team and both stay, so that team has weight 2, as everywhere else in the project.
- Slot order: every battle shuffles the six Pokemon of both teams (`shard.py`, `ShuffledTeamBuilder`).
- The 24 tournament-ready teams, their A–E edits and their replicates are battled again under these two rules.
  The first run's scores are not reused.

What stays: the anchors (24 tournament-ready teams, 16 search-found teams of `results/rebattle_top.json`), six edit
types × 3 edits, 384 battles per score, seeds 101 (anchor), 202 (edits), 202/303/404 (replicates), the
benchmarks, every statistic and both tests T1 and T2 with their cutoffs. Budget: 880 scores × 384 =
337,920 battles, about 3 hours at the October pace.

Known differences from the first run that the rerun cannot remove:
- The edit files of 2026-09-28 were staged in /tmp and are gone. Edits are drawn again by the same
  seeded code. The 24 anchors come out the same (a00 is MB373 again); the edits were not compared.
  The rerun saves every team file under `~/vgc-data/edit_kind_rerun/teams/`.
- The fixed opponent schedule still gives every team the same opponent order. Slot shuffles are seeded
  per score, so an edit and its anchor do not share shuffles. The replicates measure that noise.

Outputs: `run/results/ruggedness_shuffled50.json` (first stage) and `run/results/ruggedness2_shuffled50.json`.
The wording rules of "Threshold, fixed now" apply to the rerun's result unchanged.

### Deviations of the rerun (written 2026-11-13, after the result)

- The 24 tournament-ready teams are not the September ones: 2 of 24 are shared. The line above saying "a00 is MB373 again"
  is wrong; that check read the September result file, not the dry run. The seeded draw permutes the team corpus,
  and the corpus has grown since September. The rerun is therefore a fresh sample of 24 corpus teams outside the
  top 50, measured as pre-registered. September's numbers are not a paired baseline for it.
- Nothing else deviated: 880 scores × 384 battles, 50 opponent files, shuffled slots, same seeds and tests.

Result (issue #313): T1 and T2 both hold for whole-Pokemon replacement only, in both groups (tournament-ready 5.90 points,
ρ 0.787 [0.576, 0.881]; search-found 5.85 points, ρ 0.791 [0.580, 0.906]). Move replacement passes T2 in both
groups and misses T1 on the 5-point size cutoff alone (4.84 and 4.92 points).

### Reading without the 5-point cutoff (2026-11-13, after the result, post hoc)

Owner, 2026-11-13: "you cannot make a 5 point cutoff justifcation you should eliminated or replacing with something".
The plan of 2026-09-28 gives no reason for the number 5. The pre-registered verdict above stays on the record. This
section adds a second reading that uses no chosen number. It was picked after the result was known, from three
options shown to the owner, so it is exploratory for this rerun. It is pre-registered for the top-50 study
(`docs/top50-one-edit.md`).

- **Above zero.** The 95% bootstrap interval of the noise-corrected size excludes zero.
- **Against one Pest Control step.** The size divided by the spread of true scores between starting teams, 9.04
  points pooled over both groups, against one Pest Control step divided by the spread between benchmark starting
  points of its own kind: 0.37 from random points, 0.32 from hill-climbed points. Larger when the interval of the
  difference in shares lies above zero against both.
- **Corrected the same day.** The first version of this section divided the step by a spread pooled over random and
  hill-climbed points, which gave 0.14 and read the move, item and ability edits as larger. The two kinds of point
  have mean scores of 18.5 and 14.2, so the pooled variance, 5.44, is mostly the gap between them, against 0.77 and
  1.10 inside each kind. The owner chose each kind's own spread ("b", 2026-11-13). T2 above pooled the same way, so
  this reading no longer matches T2's verdicts.

Computed by `analyse_groups` of `scripts/top50_one_edit.py` on this rerun's saved scores, 2,000 resamples of
starting teams.

| Edit | Tournament-ready: points, interval | Share | Above zero | Against the step | Search-found: points, interval | Share | Above zero | Against the step |
| --- | --- | ---: | --- | --- | --- | ---: | --- | --- |
| Whole Pokemon copied from a real set | 5.90, 4.78 to 7.01 | 0.65 | yes | larger | 5.85, 3.83 to 7.47 | 0.65 | yes | undecided |
| One move replaced | 4.84, 3.35 to 6.23 | 0.54 | yes | undecided | 4.92, 2.05 to 7.24 | 0.54 | yes | undecided |
| Held item replaced | 4.35, 2.33 to 5.89 | 0.48 | yes | undecided | 1.44, 0.00 to 3.33 | 0.16 | not shown | undecided |
| Ability replaced | 3.66, 2.22 to 4.71 | 0.40 | yes | undecided | 1.35, 0.00 to 2.96 | 0.15 | not shown | undecided |
| Stat alignment replaced | 1.95, 0.00 to 2.92 | 0.22 | not shown | undecided | 0.00, 0.00 to 2.53 | 0.00 | not shown | smaller |
| Stat Points copied from another set | 1.60, 0.00 to 2.63 | 0.18 | not shown | undecided | 1.72, 0.00 to 2.69 | 0.19 | not shown | undecided |

What changes against the pre-registered verdict: the move edit now reads real in both groups, where the 5-point line
had failed it by 0.16 and 0.08 points. Item and ability edits read real from tournament-ready teams only. Against the
benchmark step only the whole-Pokemon swap from tournament-ready teams reads larger. From search-found teams its
interval of the difference starts at −0.01 against random points, so it reads undecided. Stat alignment from
search-found teams reads smaller than a step.
