# Does the Foul Play deterministic argmax win rate respond to Stat Points?

Issue #448 on the map in issue #416 (*Offline Rust search labeling for the surrogate and inverse-diffusion dataset*). Settled in the 2026-11-27 grill (human-constructed top-50 meta teams only, no generated training teams).

Under behaviour cloning (`policy_id = 8`), redrawing Stat Points and repairing Stat Alignment moved training teams' mean win rate by only +0.012 ([training-teams-line.md](training-teams-line.md)), because the behaviour-cloning policy barely responds to Stat Points. Before training a matchup predictor on **Foul Play deterministic argmax** labels (#419), we measure whether the **Foul Play deterministic argmax win rate** actually responds to Stat Points—and by how much more than behaviour cloning—on real, tournament-tuned human teams.

## Arms

All four Stat-Point arms share the exact same 48 non-Stat-Point fields (species, item, ability, four moves, and **Stat Alignment**) on every Pokémon. Only the 36 Stat-Point fields change.

| Stat-Point arm | Code ID | What changes |
|---|---|---|
| Stored | `stored` | Nothing — the team's unchanged human tournament Stat Points (control) |
| Redrawn same-species | `redrawn` | Each Pokémon's Stat Points redrawn from real same-species VGCPastes spreads in `code/data/foul-play-doubles-spreads.json` (occurrence-weighted, preferring a spread different from Stored), holding Stat Alignment fixed |
| Random legal | `random_legal` | Each Pokémon's 66 Stat Points allocated uniformly at random across the 6 stats (0 to 32 per stat, total 66) |
| Zeroed | `zeroed` | 0 Stat Points in all six stats on all six Pokémon (ceiling check removing all 396 Stat Points) |

## Base teams

20 human-constructed top-50 meta teams drawn without replacement (seed `448`) from the 48 distinct canonical top-50 meta teams whose six species all appear in `code/data/foul-play-doubles-spreads.json` (VGCPastes Regulation M-B team ID `MB292` is excluded because Alcremie is absent from the spreads file). Zero generated training teams enter the experiment, and zero held-out teams (ranks 51–100) enter as base teams, opponents, or spread sources (`foul-play-doubles-spreads.json` already excludes top-50 meta teams and held-out ranks 51–100).

Every variant paste (80 pastes = 20 base teams × 4 arms) is checked with Showdown's validator before battling, verified to keep an identical 48-field `grid_hash`, and verified to differ from Stored in Stat Points (`canon_hash`) on every edited arm.

## Opponents and battles

Each arm of base team $i$ battles the **48 other distinct top-50 meta teams** (excluding base team $i$ itself so all four arms face the exact same opponent set), weighted by `team_set = 'top50'` (weight 2 for the duplicated top-50 meta team when it is among the 48 opponents, 1 for all others).

- **Replicates:** 4 battles per opponent per arm per labeler (192 battles per variant; 15,360 directed variant-opponent battles per labeler, or 14,600 unique battles after sharing undirected Stored × Stored pairs between drawn base teams).
- **Labelers (both sides, Pokémon order shuffled every battle):**
  - **Foul Play deterministic argmax**: `rust_foulplay_det`, 30,000 iterations, 1 sample.
  - **Behaviour cloning**: `bc_100.zip`, shuffled slots.
- **Isolation:** Battles write to a private SQLite database (`~/vgc-data/foul_play_stat_points_448/battles.sqlite`) so synthetic random and zeroed Stat-Point teams never pollute `~/vgc-data/matchup_regmb.sqlite`.

## Measures

1. **Meta-weighted win rate** $W_{i,a}$ for each base team $i$ and arm $a$, plus exact unbiased within-cell battle-noise SE computed from the 4 games per opponent cell.
2. **Paired arm effect** $\Delta_{i,a} = W_{i,a} - W_{i,\text{stored}}$ for Redrawn same-species, Random legal, and Zeroed, reported as the 20-team mean with a paired bootstrap 95% confidence interval (2,000 resamples over base teams).
3. **Labeler difference-in-differences** (Foul Play deterministic argmax paired change minus behaviour-cloning paired change) with a paired bootstrap 95% CI (same base-team resamples across both labelers): directly tests whether Foul Play deterministic argmax responds more strongly to Stat Points than behaviour cloning.
4. **Team-level consistency:** how many of the 20 human teams lose win rate (and how many drop by more than 2× battle-noise SE) under each arm, alongside each variant's L1 Stat-Point distance from Stored.

## Pre-registered storyboard

Written before any battle runs (pre-run figure storyboard). Each frame tests one claim; final figures will be cut to these frames once results arrive.

### Frame 1 — Stat-Point severity ladder (`1_arm_ladder.html`)
- **Sentence:** Removing or scrambling Stat Points drops the Foul Play deterministic argmax win rate sharply, while behaviour cloning moves much less.
- **Expected:** Under Foul Play deterministic argmax, mean win rate falls monotonically from Stored → Redrawn same-species → Random legal → Zeroed (Zeroed dropping > 0.10, Random legal > 0.05, Redrawn same-species by ~0.01–0.04); behaviour cloning shows a much smaller drop at every rung.
- **Surprise:** Either Foul Play barely drops even on Random legal or Zeroed (search does not exploit Stat Points), or behaviour cloning drops nearly as hard as Foul Play on broken spreads (meaning behaviour cloning *does* notice Stat Points and only looked flat earlier because redrawn spreads replaced noisy generated ones).

### Frame 2 — Labeler sensitivity gap (`2_labeler_gap.html`)
- **Sentence:** On the exact same Stat-Point edits, Foul Play deterministic argmax shifts win rate more than behaviour cloning, and the gap widens with edit severity.
- **Expected:** Paired deltas (arm minus Stored) with 95% intervals put Foul Play cleanly below behaviour cloning (more negative) on Redrawn same-species, Random legal, and Zeroed.
- **Surprise:** The two labelers' 95% intervals overlap on Redrawn same-species, meaning realistic human-to-human spread differences sit below the 4-battle noise floor even under Foul Play.

### Frame 3 — Team-by-team response (`3_team_drops.html`)
- **Sentence:** The Stat-Point drop holds across individual human meta teams, not just one or two outlier line-ups.
- **Expected:** Almost all 20 base teams lose win rate under Zeroed and Random legal in Foul Play, while on Redrawn same-species a subset of spread-sensitive teams drops and generic-spread teams stay near zero.
- **Surprise:** Only a couple of teams move while most stay flat (archetype-specific sensitivity), or several meta teams *gain* win rate under Redrawn same-species.

### Teaser (`0_teaser.html`, drawn after results exist)
- Combined view: the four-arm win-rate ladder for both labelers beside the paired drops vs Stored, stating the main finding in one glance.

## Result (2026-11-27)

**Yes — Foul Play deterministic argmax responds strongly to broken Stat Points, and so does behaviour cloning; neither labeler separates two realistic same-species human spreads beyond battle noise.**

All 14,600 unique battles per labeler (`29,200` total) finished in the private database `~/vgc-data/foul_play_stat_points_448/battles.sqlite` with `0` errors, `0` fallbacks, `0` unattributed battles, `0` incomplete cells, and `0` held-out (ranks 51–100) cells. `~/vgc-data/matchup_regmb.sqlite` was not touched.

| Stat-Point arm | Mean L1 Stat Points moved | Foul Play deterministic argmax win rate (Δ vs Stored [95% CI]) | Behaviour-cloning win rate (Δ vs Stored [95% CI]) | Labeler gap (Foul Play − behaviour cloning) [95% CI] |
|---|---:|---|---|---|
| Stored | 0 | **0.503** (control) | 0.503 (control) | — |
| Redrawn same-species | 219 | **0.522** (+0.018 [−0.001, +0.037]) | 0.490 (−0.013 [−0.031, +0.005]) | **+0.031** [+0.005, +0.058] |
| Random legal | 444 | **0.436** (−0.068 [−0.095, −0.040]) | 0.417 (−0.086 [−0.110, −0.063]) | **+0.019** [−0.011, +0.046] |
| Zeroed | 396 | **0.243** (−0.260 [−0.285, −0.233]) | 0.282 (−0.221 [−0.242, −0.199]) | **−0.039** [−0.068, −0.008] |

### Against the pre-registered expectations and surprises

1. **Zeroed Stat Points** crushes every single team under both labelers (20/20 teams drop by > 2× battle-noise SE in both). Foul Play deterministic argmax penalizes zeroed spreads **0.039 more** (`1.17×`, 95% CI `[−0.068, −0.008]`, 14/20 teams dropped more).
2. **Random legal Stat Points** (mean L1 distance 444) costs Foul Play deterministic argmax **−0.068** (17/20 teams dropped, 9/20 by > 2× SE) and behaviour cloning **−0.086** (19/20 dropped, 7/20 by > 2× SE) — matching the Frame 1 surprise: behaviour cloning *does* respond strongly to broken Stat Points once tested on human teams.
3. **Redrawn same-species Stat Points** (mean L1 distance 219) stays inside battle noise under both labelers — matching the Frame 2 surprise. Observed across-team SD of the Redrawn same-species change equals pure battle-noise contrast SE in Foul Play deterministic argmax (`0.0464` vs `0.0466`, `0/20` teams beyond −2× SE) and in behaviour cloning (`0.0426` vs `0.0503`, `0/20` beyond −2× SE), with cross-labeler team correlation `r = −0.096`. The three largest apparent Foul Play gainers (VGCPastes teams `MB580`, `MB581`, `MB722`) had 4-battle Stored draws 0.056–0.071 below their 8-battle baselines in *Battle 500 training teams and the meta × meta baseline with Foul Play deterministic argmax* (#447).

### Two-narrative check (storyboard vs prose)

- **Figures alone (`0_teaser.html` → `3_team_drops.html`):** Both labelers track closely from Stored through Redrawn same-species and Random legal, then plunge on Zeroed; Foul Play deterministic argmax drops slightly harder on Zeroed (−0.039) while matching behaviour cloning on Random legal; all 20 human teams fall far below the ±2× battle-noise band on Zeroed, whereas Redrawn same-species stays inside the noise band.
- **Words alone:** Both labelers strongly punish missing or scrambled Stat Points (Foul Play deterministic argmax 1.17× steeper on Zeroed), while swapping one realistic human same-species spread for another stays below the 4-battle noise floor under both. Both readings reach the same conclusion.

## Reproduce

Run from `code/` with `/tmp/vgc-pilot/.venv/bin/python`:

```sh
python src/foul_play_stat_points.py sample
python src/foul_play_stat_points.py smoke
python src/foul_play_stat_points.py battle-bc
python src/foul_play_stat_points.py battle-fp
python src/foul_play_stat_points.py report
```

CI unit tests (no `/tmp/vgc-pilot` or `~/vgc-data` required):

```sh
python scripts/test_foul_play_stat_points.py
```
