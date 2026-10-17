# Random noise to Pokémon team archetypes

*Retired 2026-10-17 — superseded by [Noise to Team Archetypes](../noise-to-archetypes/README.md), the 2026-09-25 slot-reel version the owner prefers. Kept for its method notes.*

Updated and audited 2026-10-11. Open `archetype-diffusion.html` directly in a browser: a standalone offline page with embedded data, CSS, JavaScript and Pokémon sprites. Press Play, scrub the 48 steps, or select any of 300 teams. Reset stops playback. The page starts paused, including for reduced-motion users. A mean-only toggle removes the Gaussian offsets; a path toggle preserves the selected team's observed history.

The trace contains teams sampled directly from `temperature_p0.pt` with `hpsdiffusion.sample_constrained`, seed 20260928, unconditional, temperature 1 and guidance 1. The checkpoint is a masked-diffusion-trained Transformer; its constrained decoder fills categorical fields in dependency order. It starts with MASK tokens, not Gaussian noise. The browser replays recorded inference, not live inference.

## How a partly masked team gets a 2D position

For each team and each decode step *s* (0–48), the state is the team with the first *s* fields in dependency order revealed and the rest masked. `record-archetypes.py` resumes the same constrained sampler loop from that state and draws 24 completions (512 for the fully masked state, which every team shares). Each completion is labelled and projected:

- **Position:** `μ + Lz`, with completion mean μ, covariance `Σ = LLᵀ`, and one seeded standard Gaussian vector z per team, held fixed across all steps. Mean-only mode shows μ; every fully masked team then occupies the same point.
- **Spread:** Gaussian offsets illustrate the estimated completion covariance. They are not posterior samples or calibrated confidence regions, can conceal multiple modes, and need not shrink monotonically. Twenty-four completions give a noisy Monte Carlo estimate. Final spread is zero.
- **Colour:** the share of completions in each archetype. Grey means undecided.

Projection: shrinkage linear discriminant analysis (λ = 0.05, all three axes) on unordered species/ability/item/move multi-hot vectors of 1,500 separately seeded model samples, labelled by the archetype rule. The recorder stores each state's 3D mean and covariance. `build.py` views them through one fixed linear 2D map, fitted toward left (Trick Room), top (Weather), right (Tailwind) and bottom (Other) anchors. Means and covariances transform linearly; tiny negative residuals from trace rounding are clamped before factorization. The first two discriminant axes alone put Tailwind and Other on top of each other. Projection, labels and display bounds stay fixed throughout playback.

**This is supervised separation, not discovery of natural clusters.** A projected mean need not correspond to a valid team; map distances do not measure strategic similarity. Smooth movement interpolates means and covariance factors between recorded integer states. Fields and completion shares only update at integer steps. The Gaussian cloud and interpolation are illustrative, not the model's noise process. Natures are excluded from the projection, so steps 42–48 reveal them without moving points.

Archetype rule, first match wins: Trick Room move; weather setter (Drizzle, Drought, Sand Stream, Snow Warning, or Mega Charizard Y / Tyranitar / Froslass / Abomasnow and their stones, checked against the bundled Showdown dex); Tailwind move; Other. Final counts: 98 Trick Room, 93 Weather, 44 Tailwind, 65 Other. These are heuristic buckets. Hybrid teams can contain all three cues; the inspector shows every revealed cue and the priority rule. Move presence does not establish strategic coherence or effectiveness. Other means no qualifying cue.

All 140 species IDs have embedded artwork. Eelektross-Mega, Pyroar-Mega, Raichu-Mega-X and Staraptor-Mega use base-form sprites, explicitly disclosed in the inspector. Unknown species have placeholders; every unrevealed field shows `[MASK]`.

Followed teams: the team nearest each archetype centroid, plus two random teams (seeded).

Team identity uses a fixed descriptive name drawn from two actual members, rather than arbitrary numbers or a single sprite. The builder prefers less common pairs and chooses another real pair to avoid duplicate names; all 300 names are unique. The map, example buttons, roster and inspector use the same name. The dropdown also previews the final heuristic archetype. These names intentionally preview the finished lineup, as disclosed above the plot; masked slots still reflect only revealed model fields. They do not imply a strategic core or competitive success.

The selected marker is drawn on top with a white ring and its name; other dots stay unlabeled to avoid overlapping long names. Its six-member roster remains beneath the map as it moves. Every slot keeps its position and shows a placeholder until its species is revealed. Different samples can share Pokémon; abilities, items and moves also distinguish teams. Cluster headings show text only. The selected name badge and all dots are clickable.

No record in this trace has tournament, placing, player or event metadata: these teams were sampled from the model. Labels such as “Regional champion” would be unsupported, even if generated species overlapped a real winning team. The inspector explicitly says “Model-generated sample · no tournament result attached.” Numeric indices remain internal references only.

## Files

### Current VGC strategy vocabulary, applied 2026-10-11

[TrickRoom’s M-C archetype catalog](https://www.trickroomvgc.com/archetypes) currently lists Rain, Sun, Sand, Snow, Trick Room, Tailwind and Balance. Its weather pages define [Rain](https://www.trickroomvgc.com/archetypes/rain), [Sun](https://www.trickroomvgc.com/archetypes/sun), [Sand](https://www.trickroomvgc.com/archetypes/sand) and [Snow](https://www.trickroomvgc.com/archetypes/snow). This is a vocabulary reference, not a claim that the older M-B checkpoint is current M-C or that source-site usage rates describe these samples.

[Wolfe Glick’s Speed Control](https://www.vgcguide.com/speed-control) and [Cores and Modes](https://www.vgcguide.com/cores-and-modes) motivate overlapping speed-control and weather options. [Aaron Traylor’s Archetypes](https://www.vgcguide.com/archetypes) explains why a shared game plan matters and not every team fits an established archetype. We infer only inspectable cues, not proven strategy or success.

`strategies.py` applies all matching modes to every team: weather-setting abilities/manual moves, compatible weather Mega form/stone, Trick Room and Tailwind moves. The map label and dropdown preview all final cues; the inspector links each revealed cue to its definition and shows the exact member/field. Evidence stays hidden until every required field is revealed. A Tyranitar with Unnerve does not establish Sand. Trick Room can be a counter-option rather than the intended primary plan. No match is Unclassified, never automatically Balance or hyper offense. Missing stat spreads and battle validation preclude stronger claims.

The original four-color LDA map is explicitly retained as a coarse recorded projection. New overlapping tags do not refit it or relabel its recorded completion probabilities. Thus a legacy Weather-group sample can show Rain + Tailwind, and manual-weather tags can appear on samples in another legacy group. Run `python test-strategies.py` to check hybrid labels, species-only false positives, Mega compatibility, manual weather, and cited evidence for all 300 samples.

- `record-archetypes.py`: records `trace.json`, about 12 min on 11 spawned CPU processes (a forked pool deadlocks in PyTorch). Sizes can be overridden with `ARCH_SIZES=N,K,K0,REF`.
- `build.py`: embeds `trace.json` and the needed sprites (from `../search-loop/sprites`, Megas without a sprite fall back to the base forme) into `archetype-diffusion.html`.
- `template.html`: the page source. Edit this, not the built file.

```sh
/Users/ramiismael/.local/share/vgc-pilot-runtime/venv/bin/python record-archetypes.py
python3 build.py
/Users/ramiismael/.local/share/vgc-pilot-runtime/venv/bin/python verify.py
/Users/ramiismael/.local/share/vgc-pilot-runtime/venv/bin/python verify-build.py
```

The older GIF is a historical preview and does not show the updated controls. No battles or full-team legality validation were run for this figure. Stat Points are not sampled.

## Verification on 2026-10-11

- Checkpoint and 692 corpus hashes match the training manifest. Three recorded source hashes match; `encode.py` has a reviewed additive regulation-vocabulary API. Its legacy constructor still reproduces the saved trace.
- All 300 teams and heuristic labels exactly reproduced from the original sampler. Resumed-sampler parity passed at steps 0, 6, 18, 30 and 41.
- Reproduced the independent 1,500-team LDA fit, centroids and all 300 endpoint positions.
- Reproduced all 41 intermediate conditional summaries for team #1 and the shared 512-completion prior. Other teams' conditional samples were structurally checked, not fully resampled.
- Verified finite values, probabilities and covariances for all 14,700 states. Independently checked the embedded 2D means/covariance factors and zero final spread.
- All 140 embedded PNGs decode and have visible pixels; the page has no external script/style dependencies.
- Browser checks cover play/pause, completion/replay, reset during playback, step controls, scrubbing, keyboard input, team selection, comparison toggles, loaded sprites, exactly `48 − step` masked fields at all 49 states, and a 390-pixel mobile layout without horizontal overflow.
- The earlier numeric-identity revision checked stable identity and six roster slots across all 49 states, correct masked species, shared Pokémon across different samples, badge selection, and mobile layout. The current descriptive-name revision additionally verifies that every name is unique and consists of actual members; names stay fixed while slot masking progresses. Historical numbered references in the sampling audit are internal trace indices.
