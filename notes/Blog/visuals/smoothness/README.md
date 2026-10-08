# One move, a different team

Updated 2026-10-01: every one of the **48 displayed team members** now has a measured edit. The visual has **50 move-edit comparisons and 9,900 saved battles**. The original 4,800-battle pilot and 5,100-battle coverage extension are stored in `/Users/ramiismael/vgc-data/matchup_regmb.sqlite`. The group and benchmark views still use the original two randomly sampled edits per team; added member coverage is clearly labeled and does not change those statistics.

The current local server reads results and available sprite assets directly from SQLite. Rebuilding or refreshing this visual starts no battles. `data.js` is a database export for offline use. See [database reuse instructions](/Users/ramiismael/Documents/code/vgc-team-generator-pilot/docs/smoothness-database.md) for SQL queries, the read-only JSON API, exports, incremental imports and backups.

An interactive explanation of local Pokémon team sensitivity, implementing the approved Excalidraw flow. Open `index.html` directly for the exported snapshot or use `smoothness_server.py` to read the database live at `/smoothness/`. The figure uses native browser controls, CSS and JavaScript; it has no package installation or external network dependency. Offline Pokémon sprites are reused from the sibling `search-loop/sprites` directory; the database server serves their archived bytes. Dragalge-Mega has a letter placeholder because no matching local sprite was available.

Held-item icons and names follow the exact saved battle-start loadout in slot order. All 48 members have an item in this snapshot, covering 33 distinct items. Itemless members show “No held item” without an icon. This is not a turn-by-turn inventory: a berry may be consumed during a battle. Artwork comes from the [Pokémon Showdown sprite repository](https://github.com/smogon/sprites/tree/master/src/minisprites/items); individual source URLs and hashes are in `items/sources.json` and archived in the matchup database alongside the PNGs. Offline copies are in `items/`.

## Learning question

Can one legal move replacement change expected win rate beyond the uncertainty in battle evaluation—and does that sensitivity depend on the starting region?

The figure shows measured pilot observations. It does not establish mathematical non-smoothness or claim the full experiment is complete.

## Three connected views

- **Try one edit:** choose a cohort, original team, Pokémon and recorded replacement. A fixed zero line and fixed −30 to +30 percentage-point axis preserve context. The dot and interval show the saved edited-minus-original estimate. Undo restores the recorded original without inventing a zero-width confidence interval. The unchanged-team repeat exposes evaluation noise. Unmeasured Pokémon show a missing-result state.
- **Compare the teams:** every selectable dot is one original team. The default statistic is the mean noise-corrected squared change across its two sampled edits. A second view shows observed absolute changes, explicitly retaining battle noise. Selecting a dot opens that team's edits. Group means and recorded cluster intervals are displayed separately.
- **Benchmarks:** hold the Pokémon reference fixed while selecting a measured Pest Control or Hartmann-6 step. Both rows use the same fixed symmetric logarithmic axis, including negative corrected estimates. Exact numeric means and intervals accompany the plot. Normalization uses the recorded fixed objective bounds; it does not equate the neighborhoods.

## Motion and accessibility

The animation's purpose is state indication and spatial consistency. Pointer-triggered edit changes retarget CSS `transform` transitions over 240 ms with `cubic-bezier(.77, 0, .175, 1)`; interval visibility uses a 140 ms opacity transition. The original and zero reference stay still. Team switches snap to their new context. Frequent tab navigation is immediate. Button press feedback uses a small 140 ms transform.

Keyboard-triggered actions and reduced-motion mode are immediate. Tabs support left/right arrows, Home and End. Native selects retain platform behavior. Every chart has text equivalents, every team dot is a labeled button, and changed edit results are announced through a polite status region. There is no automatic loop, autoplay or random regeneration. Missing measurements are not treated as zero.

## Data provenance and limits

The original pilot was imported into the matchup database from:

`/Users/ramiismael/Documents/code/vgc-team-generator-pilot/results/smoothness_pilot/`

The source files are `analysis.json`, `benchmarks.json`, `manifest.json`, original team text files, and the completed `labels/*.json` battle panels. The manifest SHA-256 is `47ca1e96fe27f322c8ad69d53095b44e0c5a45e3e5ca66021af7c152e1f047dd`. The snapshot was created on 2026-10-01. Source experiment files were not changed.

This pilot contains **4 random originals + 4 top-placement originals, 2 legal move replacements per original, an unchanged-team control, and 3 panels of 50 battles per candidate: 4,800 battles**. All candidates face the same frozen 50-entry opponent pool under the same behavior-cloned policy. Ties count as non-wins. The internal group key `tournament` is displayed as **Frozen top placements** to match the experiment's actual cohort definition.

The random cohort is sampled from the frozen hierarchical product sampling corpus, not uniformly from every legal team. The placement pilot uses the first four frozen snapshot entries. The full snapshot includes ladder and ranked-season entries and contains a duplicated identity. The pilot is an implementation check; its four placement originals are not a random sample of competitive play.

Per-edit intervals are the saved pointwise Newcombe/Wilson difference intervals. They are approximate for fixed opponent strata and not corrected for multiple testing. The corrected squared-change estimator is `mean(d)² − sample_variance(d) / 3`, averaged over edits within each original. Negative estimates are preserved. Group intervals resample original-team identity clusters, so shared baselines are not treated as independent teams. Four clusters per group provide only exploratory evidence.

Benchmark summaries are already divided by squared fixed objective bounds: win probability 1, Pest Control 50, Hartmann-6 8.4. The chart applies the display transform `sign(x) × log10(1 + abs(x) / 1e-9)` with fixed bounds −0.001 to 0.01. The transform is approximately linear near zero; it is not a new statistical estimator. Hartmann-6 is deterministic; its cluster interval reflects variation across reference points. Pest Control has independent stochastic evaluations.

The separate **50 + 50** study is running and has a partial archive in the same database. This visual intentionally retains its eight original teams. To migrate its group comparison to a larger study, use completed results, update the visible counts and study wording, and check chart bounds and dot layout against the complete dataset. Do not simply relabel the current eight points as 100 teams. The current comparison dot placement and summary divisor are intentionally pilot-specific.

## Design basis

Applies the local `animate`, `frontend-design`, `pick-ui-library` and `interactive-explanations` skills. The control needs are small enough for native selects and buttons; no custom popup or UI dependency is required. The approved layout is retained: white surface, restrained blue selection, a six-member roster and one result at a time.

The explanation follows [Communicating with Interactive Articles](https://distill.pub/2020/communicating-with-interactive-articles/): each interaction tests a specific comparison, defaults provide a readable result, changing one variable preserves the baseline, and interpretation sits next to the visual evidence. A useful comprehension check is whether a reader can distinguish a measured edit, a noisy unchanged repeat, and an unmeasured edit after exploring a new team.

## Verification

Verified on 2026-10-01: all checks below passed in headless Google Chrome, including the database-backed server and every team member. Seven database tests also passed, including export with source-file reads blocked, duplicate import protection, rollback on conflicting counts, and reused baseline panels. Desktop and mobile previews were visually inspected. Opening the figure directly from a local file also worked, and expanded methods fit at 320 pixels.

`check.cjs` uses Playwright with installed Google Chrome. It checks all 50 recorded edits and all 48 team members against their battle counts, the saved estimator, noise controls, undo, missing-result protection, keyboard tab navigation, linked team selection, benchmark reference stability, chart bounds, reduced motion, rapid retargeting, console errors and horizontal overflow at 320, 390, 768 and 1280 pixels. It also produces desktop and mobile previews.

```sh
python3 -m http.server 8769 --bind 127.0.0.1 --directory Blog/visuals
```

In another terminal, from the project root with Playwright on Node's module search path:

```sh
node Blog/visuals/smoothness/check.cjs
```

Set `SMOOTHNESS_URL` to test another local URL or an absolute `file://` URL. Screenshots are generated in `previews/`.
