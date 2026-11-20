# Today's in-place prompt on the held-out grid: the starting point

Pre-registration, 2026-11-18. Ticket: [Measure today's in-place prompt on the held-out grid: the starting point](https://github.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/issues/237) (#237), under the masked text diffusion map (#160). Mode: **confirmatory** against the bar #169 set (at least 3 of 270 legal on the first try), fixed before any fill. The per-task and per-reason breakdowns are **exploratory**.

Runner: `code/scripts/inplace_prompt_grid.py`. Tests: `code/scripts/test_inplace_prompt_grid.py` (in `ci_tests.txt`). Results: `~/vgc-data/inplace_prompt_grid/<prompt>/<model>/fills.jsonl`, one line per mask.

## What came before

Searched `experiments-index.md`, `code/docs/` and the web for "masked diffusion language model fills a JSON team in place, first try, no training": no earlier measurement of this prompt on this grid. Closest:

- #170's memory benchmark filled the longest mask only (all six Pokémon hidden). No model wrote a Pokémon name (#171's early warning, 2026-11-08).
- #236 ruled the sampler out: ours matches dLLM's token for token, and every model copies a visible value in the setting it was trained with. The bad text belongs to the models.
- Reference rows on these 270 masks (`code/docs/jev-measurements.md`, legality by `jev_fill.check_legal`): random legal fill 267/270, Mercury with no reasoning 93/270, Jev's top pick 270/270.

## Result, 2026-11-18

**The claim holds by the bar, and only through the Stat Points masks.** All four models write at least 3 legal masks of 270 on the first try. The prediction (all four miss, 0 to 2 each) was wrong on the count. Its exception was right: 53 of the 54 legal masks hide only Stat Points, and the one other hides one Stat Alignment (DiffuGPT medium). No model wrote a legal team on any item, ability, move or whole-Pokémon mask.

| Model | Legal, first try | 95% interval | Legal Stat Points masks | Showdown alone accepts | Parses as a team | Real species written |
| --- | --- | --- | --- | --- | --- | --- |
| DiffuGPT small | 21/270 | 4.9% to 11.6% | 21/30 | 27/270 | 75/270 | 0/105 |
| DiffuGPT medium | 26/270 | 6.4% to 13.8% | 25/30 | 31/270 | 111/270 | 0/105 |
| Masked Qwen2.5 Coder 0.5B | 4/270 | 0.4% to 3.8% | 4/30 | 10/270 | 82/270 | 0/105 |
| Masked Qwen3 0.6B | 3/270 | 0.2% to 3.2% | 3/30 | 5/270 | 113/270 | 21/105 |

Each count is 270 first-try fills, one per mask, exact 95% interval. Seconds per team: 1.9, 5.1, 6.5 and 9.4.

What the legal teams are. A legal Stat Points fill is the unmasked team with new small numbers: 16 of 194 filled Pokémon spend all 66 points (training teams: 99.3%). Both masked Qwen models write 10 in every stat (60 in all). DiffuGPT writes 32 then mostly zeros. These pass Showdown because any numbers up to 32 each and 66 in all are legal.

Checks on the result:
- **Run matched the contract.** 270 fills per model, 270 distinct mask ids, setting and grid file hash in each `run.json`. Model probes passed for all four: attention sees later tokens, a repeated seed and the reverse run order gave 100% of slot tokens the same.
- **A second scorer agrees.** `jev_fill.check_legal`, the Mask Lab checker behind the Jev and Mercury rows, accepts all 54 legal fills and rejects every fill Showdown alone accepted (each for an invalid Stat Alignment). So these counts use the same definition of legal as the reference rows. The gap is in `showdown_validator.Validator`, the batch script this map uses: it passes Stat Alignments the Mask Lab pipeline's validator rejects.
- **Copies.** No fill gave back the unmasked team. Masked Qwen3 0.6B's 21 real species are only 3 distinct species, and 152 of its filled names repeat a name shown in the same text (DiffuGPT small 4, DiffuGPT medium 18, Masked Qwen2.5 Coder 48): it copies from the visible team rather than writing new names.

What it can't show: whether any of these models writes a usable team. The bar was meant to admit a candidate to battles (#171); every legal team here is a held-out team with its Stat Points rewritten, so its win rate would measure the unmasked team, not the model. Results posted on #237, one comment per model and one comparing all four.

## Contract

| Row | Value |
| --- | --- |
| Claim | With today's in-place prompt and no training, at least one of the small four candidates writes a legal team on the first try for at least 3 of the held-out grid's 270 masks. **Refuted** if all four stay under 3. If it holds, that candidate passes #169's bar, and the prompt-only test (#171) battles its legal teams. If it is refuted, the prompt changes (#238 to #240) are measured against this row, and fine-tuning has to teach the format. Scope: first-try legality of filled text for these 270 masks, under one sampler setting. It says nothing about win rate. |
| Fields | **Generated:** every value the mask hides, written token by token into its slot: species, item, ability, Stat Alignment, moves, and each Stat Point number. **Copied:** nothing. Every value the mask does not hide is fixed text the sampler never writes; the judge checks it reads back unchanged. **Fixed:** IVs 31 and level 50, which the JSON leaves out and `mask_paste.team_paste` adds. |
| Vocabulary | The candidate's own tokenizer at #162's pinned revision; values are free text, not a pick from options. A filled name counts as Reg M-B when its Showdown ID is in `code/data/regmb-vocabulary.json`: 347 species, 148 items plus blank, 200 abilities, 496 moves plus blank, 25 Stat Alignments plus blank. Stat Point numbers are whatever digits the slot holds, judged by Showdown (at most 32 each, 66 in all). |
| Training teams | None. The four base models are untrained on our teams. |
| Scorer | No battles. A filled mask is legal when it passes four steps, in order: (1) the team part parses as JSON with six Pokémon, the six keys in order and the right value types; (2) every visible value reads back unchanged; (3) every filled name is a Reg M-B value of its field; (4) Showdown's validator (`showdown_validator.Validator`, checkout `validator-913da36`, format `gen9championsvgc2026regmb`) accepts the paste. The report also counts teams Showdown alone accepts (steps 1, 2 and 4). |
| Search region | The held-out grid: 270 Mask Lab masks, 5 unmasked teams (MB198, MB401, MB514, MB574, MB680; ranks 51 to 100) × 54 cells: items, abilities, Stat Alignments, Stat Points and whole Pokémon at 1 to 6 hidden, moves at 1 to 24. One fill per mask (first try, no validator redraws). No filters, no elite rule. |
| Settings | **Scientific:** the candidate (four arms). **Fixed for every prompt and model** (owner, 2026-11-07, #237): the prompt `inplace_header` (#168's text), the 270 masks, first try only, the #170 sampler (one window, dLLM's linear schedule, most confident masks first, float32 Gumbel-max), 128 steps, temperature 1.0, mask positions scored, native precision, run seed 2026 with each mask's seed derived from the run seed and its id, the ask line `unknown`. **Batch size 1**, not #170's largest batch (see "Departures"). |
| Plan | **Primary:** first-try legal masks out of 270 per model, with its exact 95% interval, against the bar of 3. A single measure nothing trades against here, so the question is pass or miss. Later prompts (#238 to #240) trade legality against prompt tokens and seconds per team; for them this row is the previous frontier, and each later prompt is paired with it on the same masks and mask seeds. **Also reported per model:** teams Showdown alone accepts; share that parse as JSON and as a team; share of filled values that are real Reg M-B values, per field (species is the ticket's row); distinct real species written; legal masks per task; the validator's and parser's reasons, grouped, and the first step each mask failed; prompt tokens; seconds per team; peak memory. **Budget:** 270 fills per model, about 15,000 forward passes each, about 2.3 h of laptop GPU for the four. **Baselines:** the ceiling (true values in every slot) and the floor (uniform random tokens in every slot), both measured below with each tokenizer; random legal fill, Mercury and Jev from `jev-measurements.md`. **Stopping rule:** none; all four run. |

## Arms

All four arms fill the held-out grid with the header-only in-place prompt (`inplace_header`), first try, at the fixed setting above. They differ only in the model.

| Reader name | Code ID | Seconds per team on the longest mask, batch 1 (#170) | Estimated hours for 270 masks |
| --- | --- | --- | --- |
| DiffuGPT small | `inplace_header` / `diffugpt_s` | 6.1 | 0.20 |
| DiffuGPT medium | `inplace_header` / `diffugpt_m` | 15.0 | 0.50 |
| Masked Qwen2.5 Coder 0.5B | `inplace_header` / `qwen25c` | 20.8 | 0.68 |
| Masked Qwen3 0.6B | `inplace_header` / `qwen3` | 28.8 | 0.94 |

Hours count each mask's forward passes (one per step that commits a token: 15,355 for DiffuGPT, 15,025 for the masked Qwen models) at #170's time per pass.

## Prediction

Written before any fill. All four miss the bar, 0 to 2 legal each. The #170 and #236 texts hold no real species name and break their quotes on 26% to 88% of values. If any mask is legal, it is a single Stat Alignment or Stat Points mask: one short slot, next to visible examples of the same field.

## Figures

Storyboarded before any fill; `code/scripts/inplace_prompt_post.py` draws both, and the queue posts each one with its result.

| Frame | The reader should take | Expected | Would surprise |
| --- | --- | --- | --- |
| One model's 270 masks as cells: 5 unmasked teams × 54 cells, each colored by how far its fill got (does not parse, overwrote a shown value, wrote a name outside Reg M-B, Showdown rejects, legal) | Where this model's fills break, and which kinds of mask get furthest | Most cells fail before they parse; the single Stat Alignment and Stat Points cells get furthest | Legal cells, or many cells reaching Showdown, in the move or whole-Pokémon columns, which hide the most |
| All four models × the five outcomes, masks counted in each cell | Whether any small model clears the bar of 3, and which check stops each one | Every model's mass sits in the first or third column, none in the last | One model writes valid JSON almost always and fails only on names: the format is learnable from the prompt, the names are not |

Colors: one blue ramp, lighter for a fill that stopped earlier; validated as an ordinal ramp on both surfaces with the `dataviz` validator.

## Deviations

- 2026-11-18, before any fill: the owner launched with `run-experiment` after the build reply named the three choices (batch size 1, ask `unknown`, #170's sampler setting) and did not object, so they stand as written.

## Departures from the ticket

- **Batch size 1.** #237 asks for "the batch size #170 found fits for that model". The 270 masks differ in length and in slot count, so each has its own step schedule; batching them would need padding the models never saw. Every mask fits at batch 1 (#170), the run takes about 2.3 h anyway, and seconds per team stay clean of batching.
- **A Reg M-B step before Showdown.** Showdown's validator accepts junk Stat Alignments. The floor probe below filled every slot with random tokens: Showdown alone accepted 22 to 24 of 270 masks, all of them Stat Alignment masks, which would pass #169's bar on noise. A Stat Alignment of `00` also passes. The bar is judged on all four steps; the Showdown-only count is reported beside it.
- **The ask is `unknown`.** #237 does not name one. No base model was trained on the ask, and the fine-tune row uses `unknown` for unlabelled teams (#175).

## Preflight

CPU probes, 2026-11-18, every mask with every small-four tokenizer: `~/vgc-data/inplace_prompt_grid/preflight/cpu.json`. Model probes run as the first step of the queue, one model at a time; a failure stops the queue before that model's fill.

| Probe | Result | Pass |
| --- | --- | --- |
| Ceiling: the true values written into every slot | 270/270 legal with every tokenizer | pass |
| Round trip: true values padded as the fine-tune pads, encoded, decoded from the slot tokens, read back, pasted | 270/270 legal and 270/270 give the unmasked team's paste exactly, every tokenizer | pass |
| Coverage | All 270 masks build with every tokenizer, none dropped, no unknown tokens; texts 467 to 662 tokens, under 1,024; every Reg M-B value fits its slot (slot lengths: species 10 and 9, item 6, ability 5, Stat Alignment 3, move 7 and 6, Stat Point 1 and 2, DiffuGPT and Qwen) | pass |
| Read teams | The #170 and #236 fills of this prompt by these models were read and counted: no real species name from any of the four; values inside their quotes 12% to 75% (#236 check 3). Held-out fills are read from the first 20 masks of each model's run | pass, for the scorer |
| Controls | The 5 unmasked teams legal 5/5; 20 uniform-legal teams from `hps_reg_mb_100k.jsonl` legal 20/20; 5 corrupted copies rejected 5/5 (Showdown alone accepted 1, the `00` Stat Alignment) | pass |
| Floor: uniform random tokens in every slot | 0/270 legal with every tokenizer; Showdown alone 22 to 24/270 | pass |
| Invariance: run order | Each mask's seed comes from its id, so a mask fills the same whatever ran before it (fake-model test). On the GPU: three masks run forward and in reverse | model probe |
| Repeat | Three masks filled twice with one seed; pass at 90% of slot tokens the same or more | model probe |
| Sees later tokens | Changing the text's last token moves the first slot's logits, so attention is bidirectional | model probe |
| Separation | Skipped: no checkpoint of ours, so there is no training set to intersect | n/a |
| Training smoke | Skipped: nothing is trained | n/a |
| Condition | Skipped: the ask is fixed at `unknown` and no base model learned it, so there is no intended direction to test | n/a |
| Copies | Reported with the result: fills identical to the unmasked team, and filled names identical to a visible value in the same text | with the result |

## Noise floor

The primary measure is a count of 270 independent first-try fills, one seed per mask.

- **One model against the bar.** 0 of 270 has an exact 95% interval of 0% to 1.36%; 3 of 270 starts at 0.23% (#169). At a first-try rate near 50%, the standard error is 3.0 points.
- **Two prompts on one model** (#238 to #240). Each pair shares masks and mask seeds, so compare them with a paired test on the masks where they differ (McNemar). With 270 masks and about 7% of masks differing, 80% power at 5% false positives detects a 5-point change in the legal share; a 3-point change needs about 430 masks. A smaller gain needs a second mask seed, at the same 270 fills per seed.
- **Seed noise** is not measured here: one seed per mask, as the ticket fixes. A legal count near 0 makes it moot; a count near the bar would call for a second seed before #171 battles.

The budget detects the smallest effect worth acting on, the bar of 3 of 270.
