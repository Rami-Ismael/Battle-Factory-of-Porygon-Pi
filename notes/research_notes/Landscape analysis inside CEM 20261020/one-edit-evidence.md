# Existing one-edit landscape evidence

Audited October 20, 2026. This is a read-only source audit. No new battles, experimental settings, or source files were changed. The authoritative measured files are in `/Users/ramiismael/Documents/code/vgc-team-generator-pilot`. There are two different studies, plus illustrative teaching pages. Their results must remain separate.

## What is measured and what is invented

- `Teaching/reference/smoothness-interactive.html:64` explicitly labels all scores invented. Its example numbers must not enter a measured chart.
- `Teaching/reference/smoothness-design.html:29,31` uses the measured 4,800-battle pilot only in figure 3. Its first two figures remain illustrative. It is not the full 400,000-battle result.
- `Teaching/reference/smoothness-experiment-implementation.md:48–83` records the completed October 1 full study and points to the actual report, analysis, manifest, labels and verification.
- The original smoothness Markdown at lines 34–40 contains writing-stage claims such as moves barely matter and a cross-benchmark ruggedness ranking. Those passages are not an adequate statement of the later completed evidence.

## September 28 study: different legal edit operators

Sources: [preregistration](</Users/ramiismael/Documents/code/vgc-team-generator-pilot/docs/ruggedness2.md>), [saved statistics](</Users/ramiismael/Documents/code/vgc-team-generator-pilot/results/ruggedness2.json>), [analysis code](</Users/ramiismael/Documents/code/vgc-team-generator-pilot/scripts/ruggedness2.py:163>).

The study uses 24 meta anchors and 16 previously search-found anchors, three edits per operator per anchor, and 384 battles per score. Operators A through F are move replacement, ability replacement, item replacement, alignment replacement, spread copying, and whole-Pokémon-set copying. F replaces a slot with a real corpus set of a species absent from the team, including its item, ability, moves and spread, with legality repair. This is a block replacement, not a small change in a numerical species ID. See preregistration lines 7–15.

The historical experiment used the behavior-cloning evaluator. `results/ruggedness.json.vgc.opponents` is **36**, despite early descriptions calling it top-50. The current code was repaired later: `scripts/ruggedness.py:216–218` explicitly records the old 36-opponent bug. The workspace comparison note `Teaching/reference/bo-benchmark-derived-vgc-research-comparison.md:27` also scopes the September runs to 36 actual opponents. The `results/ruggedness2_opp50_parts` folder contains only a plan and benchmark file, with no battle-results file; it does not establish that a corrected full-50 rerun completed. Do not infer historical execution from today's repaired `top50.files()` call.

### The reported rho is a normalized increment statistic

In `scripts/ruggedness2.py:163–181`, for one cohort and operator:

```text
d = score(edited team) - score(original)
r = score(unchanged replicate) - score(original)
signal_squared = max(mean(d²) - mean(r²), 0)
pooled_variance = sample_variance(anchor scores across both cohorts)
                  - mean(replicate differences² across both cohorts) / 2
delta_star = sqrt(signal_squared)
rho = 1 - signal_squared / (2 * pooled_variance)
```

The common saved denominator is `var_f = 0.010991975939102564` in squared win-probability units. It pools the two anchor cohorts to avoid a different scale caused by range restriction. `rho` is therefore not a directly computed Pearson correlation between parent and child scores. Nor is it multi-step random-walk autocorrelation. Its interpretation depends on this normalization and sampling design. The study clips negative corrected squared increments before the square root; the later study deliberately does not.

### Chartable saved results

Every value below is directly recorded in `ruggedness2.json`. Keys are `vgc.meta.<operator>` and `vgc.search.<operator>`, with fields `rho`, `rho_ci95`, `delta_star`, `edits`, `T1` and `T2`. Multiply `delta_star` by 100 for percentage points. Meta has 72 edited neighbors per operator; search has 48. Intervals are the saved 95% anchor-bootstrap intervals for rho, using 2,000 resamples.

| Operator | Meta delta-star, pp | Meta rho and 95% interval | Search delta-star, pp | Search rho and 95% interval |
| --- | ---: | --- | ---: | --- |
| `A_move_swap` | 3.4625 | 0.94546 [0.85063, 0.98633] | 4.1616 | 0.92122 [0.80033, 1.00000] |
| `B_ability_swap` | 0.2810 | 0.99964 [0.96876, 1.00000] | 3.4047 | 0.94727 [0.87520, 0.98718] |
| `C_item_swap` | 3.1247 | 0.95559 [0.89730, 0.98281] | 4.9700 | 0.88764 [0.73449, 0.97423] |
| `D_alignment_swap` | 0.6694 | 0.99796 [0.96899, 1.00000] | 1.1899 | 0.99356 [0.94511, 1.00000] |
| `E_spread_copy` | 0.4527 | 0.99907 [0.97550, 1.00000] | 1.4527 | 0.99040 [0.93399, 1.00000] |
| `F_candidate_copy` | 5.5352 | 0.86064 [0.74349, 0.92715] | 7.3636 | 0.75336 [0.48190, 0.88920] |

The saved Pest Control reference is `benchmarks.pest_control.random.rho = 0.9905205965`, interval [0.9868905523, 0.9929395856], and `.search.rho = 0.9899418907`, interval [0.9864823809, 0.9923907621]. This is descriptive for the selected edits and benchmark steps, not a universal problem-hardness ordering.

Only F passes both preregistered criteria in both cohorts. T1 requires the excess fraction of edits beyond the noise threshold to have a positive interval **and** delta-star at least 5 pp. T2 requires rho below the corresponding Pest Control reference with a difference interval entirely below zero. Failing T1 does not prove an operator has no effect. The 5 pp cutoff is part of the criterion, and uncertainty differs by operator and cohort.

The aggregate statistic uses squared changes. It does not establish that large edits tend to improve teams. A large effect can be harmful. It also does not certify local optima, basins, global mathematical nonsmoothness, or a profitable proposal mix for the current CEM.

I did not reconcile an older prose claim of 177,408 battles with the saved design and reused/new raw collections. Avoid an overall historical battle-total claim here. The per-score design and anchor/edit counts above are explicit in the saved artifact.

## October 1 study: single legal move replacements

Sources: [full measured report](</Users/ramiismael/Documents/code/vgc-team-generator-pilot/results/smoothness_top50_50x50/report.md>), [manifest](</Users/ramiismael/Documents/code/vgc-team-generator-pilot/results/smoothness_top50_50x50/manifest.json>), [analysis](</Users/ramiismael/Documents/code/vgc-team-generator-pilot/results/smoothness_top50_50x50/analysis.json>), [protocol](</Users/ramiismael/Documents/code/vgc-team-generator-pilot/docs/smoothness-experiment.md:88>).

This study is not whole-Pokémon replacement. It tests eight distinct legal move replacements for each of 50 HPS-corpus originals and 50 frozen top-placement entries. Every edited team preserves the rest of the text, including roster and move order. Proposal sampling chooses an eligible move position and then a legal alternative, rejecting illegal, no-op and duplicate proposals. It is not uniform over every legal neighbor.

The format is `gen9championsvgc2026regmb`. Both sides use the frozen BC checkpoint. The objective is win probability with ties counted as non-wins. Every candidate faces the same 50 opponent entries, each twice per 100-battle panel. Those entries include 49 distinct identities, because MB493 and MB494 are duplicates whose original weight is retained. Self matchups are included consistently.

The owner explicitly retained the snapshot's six ranked-season and four Showdown-ladder entries. Call the cohort **frozen top placements**, not strictly tournament-only. The random group is sampled from the frozen HPS corpus, not uniformly from the full legal team space. Team and move ordering remain part of identity because the policy may depend on them.

### What 4,000 panels and 900 effects mean

There are 100 anchors. Each has its original, one unchanged-team control and eight move edits, giving 1,000 candidate records. Each candidate has four panels of 100 battles, giving 4,000 panels and 400,000 battles, split equally across the two groups. There are 800 move effects plus 100 unchanged-team control effects, giving 900 edge records. The `analysis.edges[].kind` values are `edit-000` through `edit-007` or `null`, not the literal string `edit`.

### Noise correction and chartable group values

For an edited team, let `d[r]` be its win-rate estimate minus its original's estimate in panel r. The statistic in [protocol lines 124–154](</Users/ramiismael/Documents/code/vgc-team-generator-pilot/docs/smoothness-experiment.md:124>) is:

```text
corrected squared change = mean(d)² - sample_variance(d) / 4
                         = mean of d[r] * d[s] over the six distinct panel pairs
```

Under independent panels this estimates the squared difference in expected win rate. Negative estimates are retained. Do not clip them or take a square root and call it an unbiased average edit size. The observed signed-effect distribution remains noisy.

All chart values below come from `analysis.json.summary`. Means and intervals use squared win-probability units internally; multiply by 10,000 to obtain pp². `tournament` is the historical internal key for the frozen placement cohort.

| Quantity | Exact source key | Mean, pp² | Saved 95% cluster interval, pp² |
| --- | --- | ---: | --- |
| Random move sensitivity | `summary.random.mean`, `.ci95` | 0.2916667 | [0.0008229, 0.6904375] |
| Frozen-placement move sensitivity | `summary.tournament.mean`, `.ci95` | 18.8704167 | [12.9241483, 25.5107124] |
| Placement minus random | `summary.tournament_minus_random.mean`, `.ci95` | 18.5787500 | [12.6980550, 25.2616745] |
| Random unchanged-team control | `summary.random.null_control.mean`, `.ci95` | 0.4033333 | [-0.1734167, 1.1133333] |
| Placement unchanged-team control | `summary.tournament.null_control.mean`, `.ci95` | -1.6733333 | [-5.4804150, 2.6943824] |

Original mean win rates are `summary.random.mean_original_win_rate = 0.01845`, or 1.845%, and `summary.tournament.mean_original_win_rate = 0.4995`, or 49.950%. The random group's outcome floor partly confounds a quality interpretation. Both control intervals contain zero. The tiny random-edit effect is small relative to control uncertainty.

The intervals resample original-identity clusters, retaining the placement duplicate weights, with 50 random and 49 placement identity clusters. They condition on the particular cohorts, edits, policy and opponent pool. They do not account for new opponents or a changed policy, and do not establish a universal smoothness ranking.

Independent candidate/panel seeds initialize action RNGs, but the protocol explicitly disclaims exact Showdown RNG replay and paired common battle RNG. See protocol lines 118–122. A shared opponent schedule does not establish paired stochastic battle outcomes.

### Three concrete observed move edits

These are the first three non-null edges from the first placement anchor, `tournament-000`, selected for traceability rather than representativeness. The original has 212 wins out of 400, or 53.0%. Intervals are the saved approximate pointwise Newcombe/Wilson intervals, not multiplicity-adjusted discoveries.

| Edge ID | Move edit | Child win rate | Observed delta and saved 95% interval, pp |
| --- | --- | ---: | --- |
| `tournament-000-edit-000` | Charizard, Heat Wave to Rest | 40.75% | -12.25 [-18.99965, -5.33622] |
| `tournament-000-edit-001` | Farigiraf, Trick Room to Shadow Ball | 46.50% | -6.50 [-13.33823, 0.42567] |
| `tournament-000-edit-002` | Incineroar, Darkest Lariat to Cross Chop | 44.25% | -8.75 [-15.55891, -1.82348] |

Exact source fields are `analysis.edges[]` matching `candidate`, with `.edit`, `.delta`, `.ci95`, `.replicate_deltas` and `.corrected_mse`. The same edits occur in `manifest.candidates[]`. The observed examples are not predictions for the present CEM distribution or optimal play.

I freshly checked the four base panels and four panels for each of those three children, 16 raw JSON files total. Base wins were [57, 53, 49, 53]; children were [50, 41, 35, 37], [43, 45, 50, 48], and [41, 43, 46, 47], each out of 100. Independently calculated deltas and cross-panel squared effects match the saved analysis within 9e-18 win-probability-squared units.

The existing `verification.json` records the historical full audit of all 4,000 complete panels and all 900 independently recomputed effects, with matching manifest fingerprints. I inspected that record and the retained manifest/analysis. A fresh attempt to scan every raw panel was stopped after unexpectedly slow execution and produced no result; do not describe this turn as a fresh full 4,000-panel audit.

## Optional interaction context

`results/interaction_order.json` is another separate experiment, using 16 primary anchors, four whole-set swaps and all 16 on/off combinations, two 384-battle repetitions, with the historical 36-opponent limitation recorded in the workspace comparison note. Its primary noise-corrected variance shares are `vgc.primary.o1.est = 0.8713407355`, `o2.est = 0.1160281019`, and `o3plus.est = 0.0126311626`. Saved `o3plus.ci95 = [-0.0014554160, 0.0263779117]` includes zero. This supports a limited pair-interaction question on those sampled four-edit cubes, not proof that team quality is separable globally.

## Consequences for the CEM explanation

The older study supports treating whole-set replacements as a meaningfully different neighborhood from small field edits. The newer study prevents the blanket claim that moves have no measurable effect. Both are fixed-evaluator, sampled-region statements. Neither determines the best current mutation probabilities, number of edits to make, or elite-update strength.

Use the findings to motivate an operator-conditioned diagnostic or bounded proposal comparison. Record signed gain and gain uncertainty, not only absolute sensitivity. A sensitive operator can mostly damage strong candidates. Any online adaptation must use current CEM candidates, preserve the current objective, account for its evaluation cost and retain independent final validation. Do not merge the two historical studies or carry their old pool/policy statistics into the current loop as measured current effects.
