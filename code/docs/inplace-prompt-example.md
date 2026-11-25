# One fixed example team in the in-place prompt: the change on the held-out grid

Pre-registration, 2026-11-25. Ticket: [Add one fixed example team to the prompt and measure the change](https://github.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/issues/239) (#239), under the masked text diffusion map (#160). It reopens #167's decision to leave example teams out of the prompt. Starting points: [today's in-place prompt on the held-out grid](inplace-prompt-heldout-baseline.md) (#237) and [the task statement](inplace-prompt-instructions.md) (#238). Mode: **confirmatory** for the two paired changes in first-try legal masks named in the plan, fixed before any fill. The interaction, the per-field and per-reason breakdowns and the copy counts are **exploratory**.

Runner: `code/scripts/inplace_prompt_fill_modal.py` (fills on a Modal L4, judged on the laptop by `inplace_prompt_grid.judge`). Prompt builder: `masked_diffusion_text.in_place_text(..., example=example_text())`, registered as `inplace_header_example` and `inplace_example` in `code/scripts/inplace_prompt_grid.py`. Example team: `code/data/inplace-example-team.json`. Tests: `test_inplace_prompt_grid.py`, `test_inplace_prompt_fill_modal.py` (both in `ci_tests.txt`). Results: `~/vgc-data/inplace_prompt_grid/modal-l4/<prompt>/<model>/fills.jsonl`.

## What came before

Searched `experiments-index.md` and `code/docs/` for an example team in a masked diffusion fill prompt: none. #167 left example teams out without testing them. #238's task statement raised no model's legal count and halved DiffuGPT small's (20 → 10, p = 0.006), with fills like `[` and `Sp` in Stat Point slots. Every legal mask in #237 and #238 hides Stat Points or one Stat Alignment, and the base models write almost no real species (0 of 105 for three of the four).

## The example team

Meta team MB756, the top-50 meta team (`top50_evs.json`) that is not held out, shares no species with the held-out grid's five unmasked teams (MB401, MB198, MB574, MB514, MB680), and costs the fewest tokens of the four such teams:

> Maushold-Four, Azumarill, Mawile, Farigiraf, Gallade, Camerupt

It is written once, in the same compact JSON as the masked team, on its own line before `Team: `:

```
Example team: [{"species":"Maushold-Four","item":"Wide Lens","ability":"Technician","nature":"Jolly","moves":[...],"statPoints":{...}}, ...]
```

Showdown accepts it, every name is Reg M-B, and it is neither a grid team nor a held-out team (`preflight/cpu-inplace_example.json`, `example_team`). Gender marks in the paste (`Azumarill (M)`) are dropped, as the grid's teams drop them.

| Model | Example tokens | Texts with the example, header only | Texts with the example and the statement |
| --- | --- | --- | --- |
| DiffuGPT small | 418 | 885–1,012 | 956–1,024 |
| DiffuGPT medium | 418 | 885–1,012 | 956–1,024 |
| Masked Qwen2.5 Coder 0.5B | 425 | 901–1,022 | 969–1,024 |
| Masked Qwen3 0.6B | 425 | 901–1,022 | 969–1,024 |

## The 1,024-token limit

The example team is longer than any earlier addition. DiffuGPT has 1,024 learned positions, and the grid's limit is 1,024 for every model. With the example, some masks no longer fit:

| Prompt | Masks that fit all four models | Dropped |
| --- | --- | --- |
| Example team, header only | 257 of 270 | 13: whole Pokémon ×4 (3 of 5), ×5 and ×6 |
| Example team, task statement | 189 of 270 | 81: moves ×13 to ×24 (all or most), whole Pokémon ×2 to ×6 |

No mask legal in #238's fills is among the dropped ones. Each comparison pairs only the masks both its prompts fill (`inplace_prompt_grid.shared_ids`); the list is frozen in `run239/data/fitting-ids.json`, and the merge refuses a run that fills any other set.

## Contract

| Row | Value |
| --- | --- |
| Claim | Adding one fixed legal example team changes how many held-out masks a small model fills legally on the first try. **Refuted** for a model and base if its paired change has McNemar exact p ≥ 0.025 (two bases per model, Bonferroni). If it helps, the example stays in the prompt for #240 and later fine-tunes start from it; a gain that comes with many copies of the example's values is a gain in copying, not in team writing, and would cost novelty. Scope: first-try legality of filled text on these masks, one sampler setting, one seed per mask. It says nothing about win rate. |
| Fields | As #237. **Generated:** every value the mask hides. **Copied:** nothing; visible values are fixed text the judge checks, and the example team is text the model reads, never a value written into a slot. **Fixed:** IVs 31 and level 50. |
| Vocabulary | As #237: each candidate's own tokenizer, free text judged against `code/data/regmb-vocabulary.json` (347 species, 148 items plus blank, 200 abilities, 496 moves plus blank, 25 Stat Alignments plus blank). |
| Training teams | None; base models. The example team is one top-50 meta team, outside the held-out set. |
| Scorer | As #237, no battles. Four steps in order: parses as a six-Pokémon team; visible values unchanged; every filled name Reg M-B; Showdown (`gen9championsvgc2026regmb`) accepts. |
| Search region | The held-out grid (grid file SHA-256 `596b3adc19ab`): 270 masks for the two prompts without the example, 257 and 189 for the two with it. One first-try fill per mask, no redraws. |
| Settings | **Scientific:** the example team (absent or present). **Nuisance, crossed with it:** the base prompt (header only or task statement), so the example is measured on both bases. **Fixed:** the four candidates, the masks, each candidate's sampler setting (#392's shared default: 128 steps, temperature 1.0, most confident first, dLLM's linear schedule), run seed 2026 with each mask's seed from its id, ask `unknown`, batch size 1, the device (one Modal L4 container per model fills all four prompts). |
| Plan | **Primary, per model:** (1) example team on the header-only prompt against the header-only prompt, on 257 masks; (2) example team on the task-statement prompt against the task-statement prompt, on 189 masks (the ticket's own comparison). Each pairs masks and mask seeds; McNemar's exact test on masks gained against masks lost, significant below p = 0.025. Legality trades against prompt tokens and seconds per team, so each result is also a point on that plane: the example moves a model outside the earlier prompts' frontier only if it gains legal masks for its ~420 extra tokens. **Copies (ticket):** per field, filled values equal to the example's value in the same place (same Pokémon number, move number or stat) and, for names, equal to any of the example's values of that field; the same counts on the prompt without the example are the chance rate. **Also reported, before and after:** JSON parse share, real values per field, real species, seconds per team, peak memory, legal masks per task, the validator's most common reasons. **Exploratory:** all four prompts on the 189 masks they share (the example × statement interaction); how many header-only and task-statement fills repeat #238's L4 fills token for token. **Budget:** 270 + 270 + 257 + 189 = 986 fills per model. **Stopping rule:** none. |

## Arms

All four run on the Modal L4 at the fixed setting above, on DiffuGPT small, DiffuGPT medium, Masked Qwen2.5 Coder 0.5B and Masked Qwen3 0.6B, in one container per model. They differ only in the prompt.

| Reader name | Code ID | Masks |
| --- | --- | --- |
| No example team, header only | `inplace_header` | 270 |
| No example team, task statement | `inplace_instructions` | 270 |
| Example team, header only | `inplace_header_example` | 257 |
| Example team, task statement | `inplace_example` | 189 |

## Prediction

Written before any fill. The example shows 6 real species, 6 items, 6 abilities, 24 moves and 36 Stat Point numbers in the exact JSON the model fills. Masked Qwen3 already copies visible species into its fills (7 to 10 real species of 105, nearly all copies). So:

- **Copies:** both Qwen models copy example values well above the chance rate: at least 10% of their filled items and moves equal an example value of that field, against under 1% without it. The DiffuGPT models copy far less.
- **Real values:** the share of filled items and moves that are real Reg M-B names rises for both Qwen models; real species stay near 0 for the DiffuGPT models.
- **Legal count:** on the header-only base, no model changes by more than 5 masks and no test reaches p < 0.025. Copied names break species rules (an item or move the species can't have), so real names do not turn into legal teams. Legal masks stay Stat Points masks.
- **DiffuGPT small on the task-statement base** stays near its 10 legal masks: the statement already cost it the Stat Points masks the example could not restore.
- **Cost:** seconds per team up 20% to 60% (about 420 more tokens on texts of about 515 to 600); peak memory stays far under the 17.76 GiB ceiling (#238 peaked at 1.51 GiB).

## Departures from the ticket

- **Both bases, not only the task statement.** Owner, 2026-11-25: fill the example on the header-only prompt as well as on the task-statement prompt, since #238's statement helped no model. The ticket's comparison (`inplace_example` against `inplace_instructions`) is primary comparison (2).
- **Fewer than 270 masks for the example prompts.** The 1,024-token limit drops 13 and 81 masks (table above). The ticket's "same 270 masks" holds for the prompts without the example; each comparison uses the masks both its prompts fill.
- **Modal L4, not the laptop GPU** (owner, 2026-11-25, as in #238). The container refills both prompts without the example, so every pair differs in the prompt alone. #238's L4 fills stay a reference row: the token-for-token match count shows whether a second L4 container reproduces them.
- **Bonferroni across the two bases**, p < 0.025 per comparison, because each model now gets two primary tests where #238 had one.
- **Probes on the L4.** Sees later tokens and same seed twice run in each container before its fills, on its first fill (a header-only mask on a fresh run). No reverse-order probe on the GPU, as #238: each mask's seed comes from its id and its tensors are built fresh.

## Preflight

| Probe | Result | Pass |
| --- | --- | --- |
| Ceiling, round trip, coverage, floor | Both example prompts, every tokenizer, on every mask that fits: ceiling 257/257 and 189/189 legal; round trip all legal and the same paste; no unknown tokens; uniform random tokens 0 legal (Showdown alone 18 to 23). `preflight/cpu-inplace_header_example.json`, `preflight/cpu-inplace_example.json` | pass |
| Example team | Showdown accepts it; every name Reg M-B; not a grid team, not held out; no species shared with the grid teams | pass |
| Only difference | `test_the_only_difference_from_the_base_is_the_example_team_before_the_team`: each example prompt is its base with the example line inserted before `Team: `; every slot keeps its field and length and moves by the example's length | pass |
| Controls | As #237: the checker is unchanged; rerun with each preflight, 5/5 unmasked legal, 20/20 uniform legal, 5/5 corrupted rejected | pass |
| Model probes | Sees later tokens and same seed twice, on the L4, before each model's fills (reverse order: CPU test only) | with the run |
| Separation, training smoke, condition | Skipped, as #237: nothing is trained and the ask is fixed | n/a |
| Copies | `example_copies`, reported with the result, with the chance rate from the prompts without the example | with the result |

## Noise floor

From #237's plan: pairing on masks and mask seeds, McNemar on the discordant masks. #238's pairs differed on 2 to 12 masks of 270. At 257 masks, 80% power at p < 0.025 detects about a 6-point change in the legal share (about 15 masks); at 189 masks, about 7 points (about 13 masks). A smaller real change can pass as no change; one seed per mask, as the ticket fixes.
