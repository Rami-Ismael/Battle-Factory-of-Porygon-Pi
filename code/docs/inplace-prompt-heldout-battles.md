# Today's in-place prompt on the held-out grid: redraws and battles

Pre-registration, 2026-11-20. Ticket: [Check whether a candidate writes legal teams from the prompt alone](https://github.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/issues/171) (#171), under the masked text diffusion map (#160). It follows [the first-try measurement](inplace-prompt-heldout-baseline.md) (#237). Mode: **confirmatory** for the primary measure and the prediction below, both fixed before any redraw or battle. The per-kind and per-draw breakdowns are **exploratory**.

Scripts: `code/scripts/inplace_prompt_grid.py redraw` and `teams` (the validator loop), then `code/scripts/inplace_prompt_battle.py` (battles and report). Tests: `code/scripts/test_inplace_prompt_grid.py`. Results: `~/vgc-data/inplace_prompt_grid/`.

## Why this runs

#169's bar: a candidate with at least 3 of 270 masks legal on the first try goes to battles. All four small masked models passed in #237, but 53 of their 54 legal masks hide only Stat Points. The owner decided on 2026-11-20 that the pass counts and the four go to battles anyway. So this run measures what #169 asked for: how much win rate the legal teams keep, and how that compares with the reference rows on the same 270 masks.

## What came before

Searched `experiments-index.md` and `code/docs/` for battles of a masked text model's held-out grid fills: none. The reference rows on these masks are in `code/docs/jev-measurements.md`, re-battled with the design used here (2026-10-29), against the top-50 teams:

| Arm | Legal | Usable win rate | On Stat Points masks |
| --- | --- | --- | --- |
| Jev, top pick | 270/270 | 47.6% | 47.2% |
| Random legal fill | 267/270 | 32.2% | 45.3% |
| Mercury, no reasoning | 93/270 | 16.3% | 28.9% |

## Contract

| Row | Value |
| --- | --- |
| Claim | The small masked models, prompted in place and given #169's redraws, write teams that keep less win rate than any reference row: usable win rate below Mercury with no reasoning (16.3%). Refuted if any model reaches 16.3%. If it holds, the prompt-only route gives nothing to battle, and the map's fine-tunes (#172) are the only way these models write teams. It stops at "win rate under the behaviour-cloning policy against the top-50 and held-out teams". |
| Fields | Generated: every value the mask hides (one of item, ability, Stat Alignment, Stat Points, whole Pokémon, moves). Copied: every visible value, which step 2 of the judge checks is unchanged. Fixed: level 50, the Regulation M-B format. |
| Vocabulary | Each model's own tokenizer; no team vocabulary. Every filled name must be a Reg M-B value by Showdown ID (`code/data/regmb-vocabulary.json`), as in #237. |
| Training teams | None. Base checkpoints at #162's pinned revisions; no fine-tune. |
| Scorer | `bc_100.zip` behaviour-cloning policy on both sides, six Pokémon shuffled every battle. Opponents: the Mask Lab's 49 top-50 teams, then the held-out teams (ranks 51 to 100) minus the five unmasked teams, 45. 8 battles per opponent for fills (392 against the top-50 teams), 32 for the five unmasked teams. Seed 20261120 plus an offset per stage. Private database `~/vgc-data/inplace_prompt_grid/battles.sqlite`. |
| Search region | The held-out grid's 270 masks, ask `unknown`. Draw 0 is #237's first try. A failing mask is drawn again up to 4 more times, each draw with its own seed (sha256 of `2026:<mask id>#redraw<d>`), stopping at the first legal draw. A mask's team is its first legal draw. No filter beyond the four-step judge, no elite rule. |
| Settings | Scientific: the model (four arms). Fixed: prompt `inplace_header`, the #170 sampler (128 steps, temperature 1.0, slots scope, native precision, batch 1), 5 draws at most, the battle design above. Nuisance: none varied. |
| Plan | Primary: usable win rate against the top-50 teams over all 270 masks (a mask with no legal team scores 0), with a 95% interval from resampling the five unmasked teams. A single measure, so each model is read against fixed lines: Mercury with no reasoning (16.3%), the random legal fill (32.2%), Jev (47.6%). Secondary, on the same masks: the model minus the unmasked team it was cut from, minus the random legal fill and minus Jev. Also the share of filled Pokémon spending all 66 Stat Points, legal masks by draw and kind, and the held-out opponents. Budget below. Stopping rule: none; the run is one fixed pass, and a failed probe stops it. |

### Arms

| Arm | What it is |
| --- | --- |
| DiffuGPT small | `diffugpt_s`, GPT-2 small adapted to masked diffusion |
| DiffuGPT medium | `diffugpt_m`, GPT-2 medium adapted to masked diffusion |
| Masked Qwen2.5 Coder 0.5B | `qwen25c`, Qwen2.5 Coder 0.5B adapted to masked diffusion |
| Masked Qwen3 0.6B | `qwen3`, Qwen3 0.6B adapted to masked diffusion |
| Unmasked teams (control) | the five teams every mask is cut from, battled in this database |
| Uniform-legal teams (control) | five teams drawn with seed 20261120 from the first 10,000 rows of `hps_reg_mb_100k.jsonl` |

The random legal fill and Jev rows come from the 2026-10-29 re-battle (`~/vgc-data/jev/rebattle-heldout.json`), the same design. They are not battled again. The unmasked teams are battled again here, so the two databases can be checked against each other.

## Prediction

Written before any redraw:

- **Redraws.** Legal masks rise almost only on Stat Points masks. DiffuGPT small and medium reach 27 to 30 of the 30 Stat Points masks; both masked Qwen models reach 8 to 18. Fewer than 5 legal masks of any other kind per model.
- **Usable win rate.** Every model below 6% against the top-50 teams, all far below Mercury with no reasoning (16.3%). The claim holds.
- **Same masks.** The legal teams spend fewer Stat Points than the unmasked teams, so they lose 1 to 5 points against the unmasked team on the same masks, about as much as the random legal fill or more (the random fill costs 1.9 points on Stat Points masks).

## Noise floor and budget

One fill team plays 392 battles against the top-50 teams: standard error 0.025 at p = 0.5. One unmasked team plays 1,568: 0.013. On n same masks across five unmasked teams, the model-minus-unmasked difference has a standard error of about sqrt(0.025²/n + 0.013²/5): 0.0073 at n = 30, 0.0095 at n = 12. So a 2-point gap shows at 30 masks and is borderline at 12. Smallest effect worth acting on: 3 points on the same masks, the cost of half the Stat Points. The primary measure needs no such power: the claim's line, 16.3%, sits more than 10 points above every predicted value.

Budget, worst case:
- **Redraws.** At most 4 × 1,026 = 4,104 draws. At #237's seconds per draw (1.9, 5.1, 6.5, 9.4) that is about 6.6 hours on the laptop GPU.
- **Battles.** At most 1,080 legal teams × 94 opponents × 8, plus the controls. Expected about 120 teams, so about 105,000 battles. At 0.462 s per 49 battles (#169) that is about 17 minutes.

## Probes

| Probe | Result |
| --- | --- |
| Ceiling | The unmasked team on its own masks: 47.6% (Jev re-battle). A perfect fill at 270/270 keeps 47.6%, above every reference line. Pass. |
| Round trip | #237's CPU preflight passed for all four tokenizers. Today, each of the 54 legal first tries was judged again from its saved values with the tokenizer alone: 54 of 54 legal, the same pastes. Pass. |
| Coverage | #237's preflight: no unknown token in any slot and no dropped mask. Pass. |
| Read teams | #237's audit read every legal fill: unmasked teams with new Stat Points, Qwen models writing 10 in every stat. The redraws are read before the result is posted. |
| Training smoke | Skipped: no training. |
| Controls | In the queue, before any fill is battled (`check-controls`): the five unmasked teams within 3 points of their Jev re-battle mean, and the uniform-legal teams below 10%. A miss stops the queue. |
| Invariance | Skipped as a separate probe: the policy shuffles the six Pokémon every battle, so slot order is randomized, not fixed. |
| Repeat | The unmasked teams in a fresh database and seed against the Jev re-battle: the same check as the controls row. For the models, #237's probe passed (same seed, 100% of slot tokens the same). |
| Separation | The base models never saw our teams. Every fill is cut from a held-out team, which is battled as the Jev rows were (`evaluation=True`), and no fill or unmasked team is an opponent. Pass by construction. |
| Condition | Skipped: the ask is fixed at `unknown`. |
| Copies | Counted in the report: masks whose fill gives back the unmasked team. 0 of 54 on the first try. |
| Redraws differ (new) | In the queue, per model before its redraws: the first redraw's seed must change at least one slot token on three masks, or the validator loop would repeat the first try. |
| Battle setup | Dry run on a scratch database, 2026-11-20: the top-50 columns match the Mask Lab's paste for paste, 54 legal fills (53 distinct), 49 top-50 and 45 held-out opponents. Pass. |
