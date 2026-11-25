# A task statement in the in-place prompt: the change on the held-out grid

Pre-registration, 2026-11-25. Ticket: [Add task instructions to the in-place prompt and measure the change](https://github.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/issues/238) (#238), under the masked text diffusion map (#160). Starting point: [today's in-place prompt on the held-out grid](inplace-prompt-heldout-baseline.md) (#237). Mode: **confirmatory** for the paired change in first-try legal masks, fixed before any fill; the per-reason and per-task breakdowns are **exploratory**.

Runner: `code/scripts/inplace_prompt_fill_modal.py` (fills on a Modal L4, judged on the laptop by `inplace_prompt_grid.judge`). Prompt builder: `masked_diffusion_text.in_place_text(..., statement=TASK_STATEMENT)`, registered as `inplace_instructions` in `code/scripts/inplace_prompt_grid.py`. Tests: `test_inplace_prompt_grid.py`, `test_inplace_prompt_fill_modal.py` (both in `ci_tests.txt`). Results: `~/vgc-data/inplace_prompt_grid/modal-l4/<prompt>/<model>/fills.jsonl`.

## What came before

Searched `experiments-index.md` and `code/docs/` for a task statement in a masked diffusion fill prompt: none. #237's fills write JSON keys and quotes into the values (`"species":"1","item":"1"`) and copy stat keys (`SpA`); nothing in that prompt says what belongs inside a value. #237's result: 21, 26, 4 and 3 of 270 legal, 53 of the 54 legal masks on Stat Points masks.

## The statement

Worded once, used unchanged for all four models. It names no model's mask token, so it reads the same to each:

> Task: each run of mask tokens in the team below is one hidden value of its field. Fill it with a legal Reg M-B value: a species, item, ability, move, Stat Alignment (nature) or Stat Point number. Write names as Showdown spells them. Leave the JSON keys, quotes and commas as they are.

It sits on its own line between the ask line and `Team: `. "(nature)" ties Stat Alignment to the JSON key the models see.

| Model | Statement tokens | Tokens added to each mask's text |
| --- | --- | --- |
| DiffuGPT small | 71 | 71 on all 270 |
| DiffuGPT medium | 71 | 71 on all 270 |
| Masked Qwen2.5 Coder 0.5B | 68 | 68 on all 270 |
| Masked Qwen3 0.6B | 68 | 68 on all 270 |

Texts grow from 467–662 tokens to 538–731, under the 1,024 limit. Source: `~/vgc-data/inplace_prompt_grid/preflight/cpu-inplace_instructions.json`.

## Contract

| Row | Value |
| --- | --- |
| Claim | Adding the task statement changes how many of the 270 held-out masks a small model fills legally on the first try. **Refuted** for a model if its paired change has McNemar exact p ≥ 0.05. If the statement helps, it stays in the prompt for #239 and #240 to build on; if it does nothing, those prompts are measured against `inplace_header` and the format has to come from fine-tuning. Scope: first-try legality of filled text on these 270 masks, one sampler setting, one seed per mask. It says nothing about win rate. |
| Fields | As #237. **Generated:** every value the mask hides. **Copied:** nothing; visible values are fixed text the judge checks. **Fixed:** IVs 31 and level 50. |
| Vocabulary | As #237: each candidate's own tokenizer, free text judged against `code/data/regmb-vocabulary.json` (347 species, 148 items plus blank, 200 abilities, 496 moves plus blank, 25 Stat Alignments plus blank). |
| Training teams | None; base models. |
| Scorer | As #237, no battles. Four steps in order: parses as a six-Pokémon team; visible values unchanged; every filled name Reg M-B; Showdown (`gen9championsvgc2026regmb`) accepts. |
| Search region | The held-out grid, 270 masks (grid file SHA-256 `596b3adc19ab`), one first-try fill per mask, no redraws. |
| Settings | **Scientific:** the prompt (`inplace_header` against `inplace_instructions`). **Fixed:** the four candidates, masks, each candidate's sampler setting (#392's shared default: 128 steps, temperature 1.0, most confident first, dLLM's linear schedule), run seed 2026 with each mask's seed from its id, ask `unknown`, batch size 1, the device (one Modal L4 container per model fills both prompts). |
| Plan | **Primary:** per model, first-try legal masks with the statement against without it, on the same device, masks and mask seeds; paired by mask, McNemar's exact test on masks gained against masks lost. Legality trades against prompt tokens and seconds per team, so the comparison is a point on that plane: the statement moves a model outside #237's frontier only if it gains legal masks at its 68–71 extra tokens. **Also reported per model, before and after:** JSON parse share, real-species share, seconds per team, legal masks per task, the validator's most common reasons. **Reference rows:** #237's laptop fills of `inplace_header`, and how many of the header-only fills on the L4 match them token for token. **Budget:** 2 × 270 fills per model. **Stopping rule:** none. |

## Arms

Both arms run on the Modal L4 at the fixed setting above, on DiffuGPT small, DiffuGPT medium, Masked Qwen2.5 Coder 0.5B and Masked Qwen3 0.6B. They differ only in the prompt.

| Reader name | Code ID |
| --- | --- |
| Header only (no task statement) | `inplace_header` |
| Task statement added | `inplace_instructions` |

## Prediction

Written before any fill. The statement says what goes in a value, but these base models never learned to follow instructions inside JSON, and no sentence can teach a species name the model does not know. So:

- **Legal count:** no model changes by more than 5 masks, and no McNemar test reaches p < 0.05. Any legal mask is still a Stat Points mask.
- **Parses as JSON:** Masked Qwen2.5 Coder 0.5B, built from an instruction-tuned coder, rises by 10 or more masks; the two DiffuGPT models change by fewer than 10.
- **Real species:** stays 0 for DiffuGPT and Masked Qwen2.5 Coder; Masked Qwen3 keeps copying the visible species.
- **Cost:** seconds per team up 10% to 15%, in line with 68–71 more tokens on texts of about 515.

## Departures from the ticket

- **Modal L4, not the laptop GPU.** Owner, 2026-11-25: run on Modal today, to keep the laptop's fan down. A CUDA generator draws other noise than an MPS one for the same seed, so the container refills the header-only prompt too and the pair differs in the prompt alone. #237's laptop row stays the reference; the token-for-token match count shows how much the device moves it.
- **Probes on the L4.** Sees later tokens and same seed twice run in each container before its fills; the run-order probe is covered by the CPU test that a mask's seed comes from its id.

## Preflight

| Probe | Result | Pass |
| --- | --- | --- |
| Ceiling, round trip, coverage, floor | With the statement, every tokenizer: ceiling 270/270 legal; round trip 270/270 legal and the same paste; all 270 masks build, no unknown tokens; uniform random tokens 0/270 legal (Showdown alone 22 to 24) | pass |
| Only difference | `test_the_only_difference_is_the_task_statement_before_the_team`: the text is today's with the statement inserted before `Team: `; every slot keeps its field and length and moves by the statement's length | pass |
| Controls | As #237: the checker is unchanged | pass (#237) |
| Model probes | Sees later tokens and same seed twice, on the L4, before each model's fills | with the run |
| Separation, training smoke, condition | Skipped, as #237: nothing is trained and the ask is fixed | n/a |
| Copies | Reported with the result | with the result |

## Noise floor

From #237's plan: pairing on masks and mask seeds, McNemar on the discordant masks. With 270 masks and about 7% of masks differing, 80% power at 5% false positives detects a 5-point change in the legal share (about 14 masks). A smaller real change can pass as no change; one seed per mask, as the ticket fixes.
