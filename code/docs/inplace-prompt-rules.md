# Three team-building rules spelled out in the in-place prompt: the change on the held-out grid

Pre-registration, 2026-11-27. Ticket: [Spell out three team-building rules in the prompt and measure the change](https://github.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/issues/445) (#445), under the masked text diffusion map (#160). Follows [today's in-place prompt on the held-out grid](inplace-prompt-heldout-baseline.md) (#237), [the task statement](inplace-prompt-instructions.md) (#238), and [one fixed example team](inplace-prompt-example.md) (#239). Mode: **confirmatory** for the two paired changes in first-try legal masks per model named in the plan (rules spelled out against the neutral length-control line, and rules spelled out against header only), fixed before any fill. The three named rule-break shares, the full 17-rule audit matrix, the per-task breakdown, and the neutral-line against header-only shift are **exploratory**.

Runner: `code/scripts/inplace_prompt_fill_modal.py` (fills on a Modal L40S, judged on the laptop by `inplace_prompt_grid.judge`). Prompt builder: `masked_diffusion_text.in_place_text(..., rules=RULES_LINE)` and `in_place_text(..., filler=FILLER_LINE)`, registered as `inplace_rules` and `inplace_filler` in `code/scripts/inplace_prompt_grid.py`. Tests: `test_inplace_prompt_grid.py` (in `ci_tests.txt`) and `test_inplace_rules_tokenizers.py` (local real-tokenizer parity test). Figures and comments: `code/scripts/inplace_rules_post.py`. Rule audit: `code/scripts/inplace_rule_audit.py`. Results: `~/vgc-data/inplace_prompt_grid/modal-l40s-445/<prompt>/<model>/fills.jsonl`.

## What came before

Searched `code/docs/` and the web for spelling out team-building rules with a token-matched neutral length control in a masked text diffusion JSON fill prompt: no prior experiment exists. Neither #238's task statement nor #239's example team raised any small model's legal count beyond noise, and both lowered DiffuGPT small's legal count because extra prefix tokens pushed its Stat Points masks past its 512-token training window (Gong et al. 2024, Table 4). On #239's 270 header-only fills (`modal-l4-239/inplace_header`) and #443's batched fills (`modal-l4-443/inplace_header`), the three team-building rules tested here are the most common non-name rule breaks:

- **Item used more than once:** 0% (DiffuGPT small), 4.6% (DiffuGPT medium), 17.1–20.0% (Masked Qwen2.5 Coder 0.5B), 15.1–16.3% (Masked Qwen3 0.6B) of parsed fills.
- **Stat Points over 66:** 3.9% (DiffuGPT small), 2.8% (DiffuGPT medium), 5.3–6.6% (Masked Qwen2.5 Coder 0.5B), 0–1.9% (Masked Qwen3 0.6B) of parsed fills (and rose to 19% for DiffuGPT medium with #239's example team).
- **A Stat Point over 32:** 1.3% (DiffuGPT small), 1.8–2.8% (DiffuGPT medium), 0% (both masked Qwen models) of parsed fills.

The header today states only `Item Clause` and says nothing about the 66-point total or 32-point single-stat cap. Adding a neutral line of the exact same token count separates what the rules say (content) from where the team tokens sit (position/length).

## Result, 2026-11-27

**Refuted for all four models.** Spelling out the three team-building rules moves no model's first-try legal fills beyond noise, neither against the neutral line of the same length (content effect, all McNemar p ≥ 0.50) nor against header only (total effect, all McNemar p ≥ 0.50). Nor does it reduce violations of the three named rules on parsed fills. Every model filled all three arms in one Modal L40S container at batch size 8 on all 270 held-out masks (3,240 fills total, 21.4 min combined L40S time, $0.60 actual Modal credit used).

| Model | No rules (`inplace_header`) | Neutral line (`inplace_filler`) | Rules spelled out (`inplace_rules`) | Content (rules vs filler, p) | Total (rules vs header, p) |
| --- | --- | --- | --- | --- | --- |
| DiffuGPT small (flagged: 512 window) | 18/270 (6.7%) | 20/270 (7.4%) | 20/270 (7.4%) | +1 / −1 (p = 1.00) | +4 / −2 (p = 0.69) |
| DiffuGPT medium | 23/270 (8.5%) | 23/270 (8.5%) | 21/270 (7.8%) | +3 / −5 (p = 0.73) | +3 / −5 (p = 0.73) |
| Masked Qwen2.5 Coder 0.5B | 3/270 (1.1%) | 1/270 (0.4%) | 2/270 (0.7%) | +1 / −0 (p = 1.00) | +0 / −1 (p = 1.00) |
| Masked Qwen3 0.6B | 3/270 (1.1%) | 3/270 (1.1%) | 1/270 (0.4%) | +0 / −2 (p = 0.50) | +0 / −2 (p = 0.50) |

**Three named rules broken (% of parsed fills, `none / filler / rules`):**

| Model | Item used more than once | Stat Points over 66 | A Stat Point over 32 |
| --- | --- | --- | --- |
| DiffuGPT small (flagged) | 1.3% / 1.4% / 1.5% | 5.2% / 2.9% / 4.5% | 5.2% / 5.8% / 3.0% |
| DiffuGPT medium | 4.5% / 3.7% / 2.8% | 5.5% / 6.5% / 6.5% | 2.7% / 0.0% / 2.8% |
| Masked Qwen2.5 Coder 0.5B | 20.3% / 16.4% / 16.2% | 7.6% / 6.0% / 14.7% | 0.0% / 0.0% / 0.0% |
| Masked Qwen3 0.6B | 15.7% / 12.0% / 17.6% | 2.6% / 1.7% / 2.5% | 0.9% / 0.9% / 0.0% |

- **No evidence the models read the rules:** Masked Qwen2.5 Coder's drop in `item used more than once` (20.3% → 16.2%) appears equally under the neutral line (16.4%), and its `Stat Points over 66` rises from 6.0% (4/67) under the neutral line to 14.7% (10/68) under the rules line (writing two-digit numbers such as `32`, `20`, `13`, `10` that each stay ≤ 32 but sum past 66). Masked Qwen3's `item used more than once` ticks up under `inplace_rules` (17.6% vs 12.0% under `inplace_filler`).
- **DiffuGPT small did not drop at +39 tokens:** unlike #238's +71 tokens (which dropped it 20 → 10), +39 tokens leaves 24 of its 30 Stat Points masks at or below 512 tokens and leaves legal count at 20 on both `inplace_filler` and `inplace_rules`.
- **Prediction check:** held on content effect (all four within 2 masks, all p ≥ 0.50), held on total effect for DiffuGPT medium and both Qwen models, held on cost ($0.60 actual vs $0.55–$0.85 estimated, peak 0.59–1.92 GiB). Missed on DiffuGPT small's predicted drop (−3 to −10 expected; actually +2, within noise) and on Masked Qwen2.5 Coder's `Stat Points over 66` uptick (+8.7 percentage points vs filler).
- **Rule audit (`inplace_rule_audit.py`):** all 3,240 fills re-judged from saved values. Zero rejudge mismatches, zero legal-but-broken fills, zero illegal-but-clean fills across all four models. Two `step_without_rule` notes appeared on `qwen3` (mask `b847f9f0bd0716f1`, `pokemon k=4`, in `inplace_header` and `inplace_filler`): the model wrote `"---------"` for all four hidden species, which has Showdown ID `""`; Showdown reports Species Clause on the four empty species IDs while `_own_check` skips empty IDs. Every team rule and all three named rules match cleanly.

## Deviations

- **Padding probe updated before Masked Qwen3 0.6B (`5a8dea2c`, `run445/CHANGES`).** `qwen3`'s initial L40S container stopped before filling any mask because its shortest mask has only 3 slots (`team[4].nature`) and slot 1 has a 1-ULP bfloat16 near-tie (12.000 vs 11.938), causing a single-slot tie flip between batch 1 and batch 2 (`top1_same = 2/3`, `max_logit_difference = 0.266`, smaller than #443's 0.289). A local MPS test confirmed `0.0` logit change when varying padded token IDs. `padding_probe` was updated to test padding-token invariance directly (`pad_id` 0 vs 1, requiring `pad_leak_max_diff <= 1e-3`) alongside small batch-shape rounding (`< 1.0`). On the relaunched L40S container, `qwen3` measured `pad_leak_max_diff = 0.0` and passed. No fill, sampler, prompt, or batching logic changed.

## The rules line and the length-control line

Both lines are fixed strings placed on their own line between the ask line and `Team: `, using the exact same wording for all four models and naming no model's mask token.

**Rules spelled out (`inplace_rules`, `D.RULES_LINE`):**

> Team rules: each item appears at most once on the team. Each Pokémon's six Stat Points add up to 66 or less. Each Stat Point is a whole number from 0 to 32.

**Neutral line of the same length (`inplace_filler`, `D.FILLER_LINE`):**

> Note: morning clouds drift across the quiet farmland toward the old lighthouse. A flickering lantern sits beside a worn satchel under the tall sycamore. Warm bread rests on the wooden shelf.

The filler line contains no digits and no Pokémon, game, rule, or JSON vocabulary, so it cannot prime item uniqueness or Stat Point numbers while matching `RULES_LINE`'s exact token count in every small-four tokenizer:

| Model | Rules tokens added (all 270 masks) | Filler tokens added (all 270 masks) | Prompt token range | Over 1,024 | Over 512 (Stat Points masks) |
| --- | --- | --- | --- | --- | --- |
| DiffuGPT small (flagged) | 39 | 39 | 506–699 | 0 | 221/270 (6/30) |
| DiffuGPT medium | 39 | 39 | 506–699 | 0 | 221/270 (6/30) |
| Masked Qwen2.5 Coder 0.5B | 43 | 43 | 519–705 | 0 | 270/270 (30/30) |
| Masked Qwen3 0.6B | 43 | 43 | 519–705 | 0 | 270/270 (30/30) |

Every mask of the 270-mask held-out grid fits under 1,024 tokens for all three arms and all four models, with identical token positions for every slot in `inplace_rules` and `inplace_filler` (`test_inplace_rules_tokenizers.py`, `preflight/cpu-inplace_rules.json`, `preflight/cpu-inplace_filler.json`).

**DiffuGPT small flag.** DiffuGPT small was trained on 512-token texts (Gong et al. 2024, Table 4). Its 30 Stat Points masks are 467–477 tokens on `inplace_header` and 506–516 tokens on both `inplace_rules` and `inplace_filler` (6 of 30 Stat Points masks and 221 of 270 masks overall cross 512). DiffuGPT small is reported and flagged; because `inplace_rules` and `inplace_filler` share the exact same 39-token shift, their difference still isolates content from position.

## Contract

| Row | Value |
| --- | --- |
| Claim | Spelling out the three team-building rules in plain words and numbers changes how many of the 270 held-out masks a small model fills legally on the first try, beyond a neutral line of the same token length (content effect) and against the header-only prompt (total effect). **Refuted** for a model if neither paired change reaches McNemar exact p < 0.025 (Bonferroni across the two primary comparisons per model). Even if legal count does not move, a drop in any of the three named rule-break shares on parsed fills counts as exploratory evidence that the model reads the rules. If the rules line helps beyond the filler line, it is kept as a candidate prompt component before fine-tuning; if not, rules enforcement must come from fine-tuning or sampling. Scope: first-try legality and per-rule violation rates on these 270 masks, one sampler setting, one seed per mask. Says nothing about battle win rate. |
| Fields | As #237. **Generated:** every value the mask hides. **Copied:** nothing; visible values are fixed text the judge checks. **Fixed:** IVs 31 and level 50. |
| Vocabulary | As #237: each candidate's own tokenizer, free text judged against `code/data/regmb-vocabulary.json` (347 species, 148 items plus blank, 200 abilities, 496 moves plus blank, 25 Stat Alignments plus blank). |
| Training teams | None; base models at #162's pinned revisions. |
| Scorer | As #237, no battles. Four steps in order: parses as a six-Pokémon team; visible values unchanged; every filled name Reg M-B; Showdown (`gen9championsvgc2026regmb`) accepts. Every fill also records all 17 rule breaks in `inplace_prompt_grid.RULES` and is verified by `inplace_rule_audit.py`. |
| Search region | The held-out grid, all 270 masks (grid file SHA-256 `596b3adc19ab`), one first-try fill per mask, no redraws. |
| Settings | **Scientific:** the added line (`inplace_header` none, `inplace_filler` neutral length-control line, `inplace_rules` three team-building rules spelled out). **Fixed:** the four candidates (`diffugpt_s`, `diffugpt_m`, `qwen25c`, `qwen3`), all 270 masks, each candidate's shared sampler setting (#392 default: 128 steps, temperature 1.0, most confident first, dLLM's exact linear schedule), run seed 2026 with per-mask `mask_seed`, ask `unknown`, batched fills in identical `mask_groups` ordered by `inplace_header` up to the L40S speed batch cap of 8 (PR #458), and device (one Modal L40S container per model fills all three arms). |
| Plan | **Primary, per model (confirmatory):** (1) `inplace_rules` against `inplace_filler` (content effect) on all 270 masks; (2) `inplace_rules` against `inplace_header` (total effect) on all 270 masks. Paired by mask and mask seed; McNemar exact two-sided test on masks gained against lost, significant below α = 0.025. Legality trades against prompt length (+39 or +43 tokens) and seconds per team, so `inplace_rules` improves on the header-only frontier only if it gains legal masks beyond noise. **Three named rules (exploratory):** share of parsed fills breaking `item used more than once`, `Stat Points over 66`, and `a Stat Point over 32` per arm. **Also reported:** JSON parse share, reads-as-team share, full 17-rule matrix via `inplace_rule_audit.py`, tokens added per tokenizer, seconds per team, peak memory, and `inplace_filler` against `inplace_header` (position/length effect alone). **Budget:** 3 × 270 = 810 fills per model (3,240 fills total across four models). **Stopping rule:** none (complete all 810 fills per model). |

## Arms

All three arms fill in one Modal L40S container per model (at batch cap 8, identical `mask_groups`), on DiffuGPT small, DiffuGPT medium, Masked Qwen2.5 Coder 0.5B, and Masked Qwen3 0.6B. They differ only in the added line before `Team: `.

| Reader name | Code ID | Differs from the header only by | Masks |
| --- | --- | --- | --- |
| No rules spelled out | `inplace_header` | — | 270 |
| Neutral line of the same length | `inplace_filler` | `D.FILLER_LINE` (39 / 43 tokens) | 270 |
| Rules spelled out | `inplace_rules` | `D.RULES_LINE` (39 / 43 tokens) | 270 |

## Prediction

Written before any fill. The base models were not fine-tuned to follow prose constraints inside compact JSON, and #238's task statement changed no model's legal count beyond position shift. However, putting `0`, `32`, and `66` right before the team could gently shift Stat Point numbers even without raising overall legality.

- **Content effect on legality (`inplace_rules` vs `inplace_filler`):** no model changes by more than 5 legal masks, and no McNemar test reaches p < 0.025.
- **Total effect on legality (`inplace_rules` vs `inplace_header`):** DiffuGPT medium, Masked Qwen2.5 Coder 0.5B, and Masked Qwen3 0.6B stay within 5 masks (p ≥ 0.025). DiffuGPT small drops by 3–10 legal masks on *both* `inplace_rules` and `inplace_filler` relative to `inplace_header` because +39 tokens moves 6 of its 30 Stat Points masks past 512 (smaller than #238's −10 at +71 tokens).
- **Three named rules on parsed fills:**
  - `item used more than once` stays near baseline for both masked Qwen models (within 5 percentage points of ~15–20%); base models copy visible items regardless of the rule sentence.
  - `Stat Points over 66` and `a Stat Point over 32` stay near baseline (0–7% of parsed fills) for all four models; any drop is under 5 percentage points.
- **Cost on Modal L40S (batch 8):** ~16–25 minutes total across all four containers (~$0.55–$0.85 at $1.95/h), with peak memory under 2 GiB (far below 44.4 GiB).

## Figures

Storyboarded before any fill. Built by `code/scripts/inplace_rules_post.py`.

| Figure | The sentence it should carry | Expected | Would surprise |
| --- | --- | --- | --- |
| One model, two panels: first-try legal fills (left) beside the three named rule-break shares (right) across all three arms | Whether spelling out the three rules moves this model's legality or named rule breaks beyond the neutral line | Rules and filler bars sit close on both panels; DiffuGPT small drops on both relative to header only | Rules drops item duplicates or Stat Points > 66 by ≥10 percentage points relative to filler, or gains ≥10 legal masks |
| All four models × three arms legality matrix | No model gains legal fills from spelling out the rules beyond its length control | Rules column matches filler column for every model | Any model gaining ≥8 legal masks over filler |
| Three named rules × (4 models × 3 arms) matrix | Whether the rules line reduces the three named rule breaks even where overall legality does not change | Each model's three columns stay within a few points of each other | A consistent drop across all four models in the `rules` column for one of the three rules |
| Full 17-rule matrix across 4 models × 3 arms | Complete audit of every rule broken across arms | No large shift in unmentioned rules | A surprise parse-rate drop or spike in another rule under `inplace_rules` or `inplace_filler` |

## Departures from the ticket

- **Modal L40S for all four models** (owner decision 2026-11-27, PR #458, replacing #445's 2026-11-25 split of DiffuGPT medium on A10 and the other three on L4). Ticket #445's body was updated and commented on 2026-11-27. All three arms still run in one container per model with identical batching (L40S speed batch cap 8).
- **Bonferroni α = 0.025** per model because each model has two primary confirmatory tests (`inplace_rules` vs `inplace_filler` and `inplace_rules` vs `inplace_header`).
- **One fixed neutral filler line for all four tokenizers.** `FILLER_LINE` achieves exact token parity (39 in DiffuGPT small/medium, 43 in Masked Qwen2.5 Coder/Qwen3) across all 270 masks with a single fixed text containing no digits or domain terms.

## Preflight

| Probe | Result | Pass |
| --- | --- | --- |
| Ceiling, round trip, coverage, floor (`inplace_rules`) | All four small-four tokenizers, all 270 masks: ceiling 270/270 legal; round trip 270/270 legal and identical paste; 0 dropped, 0 unknown tokens; floor uniform random tokens 0/270 legal (Showdown alone 22–24); rules line adds 39 tokens (DiffuGPT) and 43 tokens (Qwen) on all 270 masks (`modal-l40s-445/preflight/cpu-inplace_rules.json`) | pass |
| Ceiling, round trip, coverage, floor (`inplace_filler`) | All four small-four tokenizers, all 270 masks: ceiling 270/270 legal; round trip 270/270 legal and identical paste; 0 dropped, 0 unknown tokens; floor 0/270 legal (Showdown alone 22–24); filler line adds 39 tokens (DiffuGPT) and 43 tokens (Qwen) on all 270 masks, `matches_rules_tokens: true` for all four (`modal-l40s-445/preflight/cpu-inplace_filler.json`) | pass |
| Only difference & tokenizer parity | `test_inplace_prompt_grid.py` (`RulesAndFillerPromptTest`, 5 tests) and `test_inplace_rules_tokenizers.py` (all 270 masks × 4 real tokenizers: exact length and slot-position equality between `inplace_rules` and `inplace_filler`) | pass |
| Controls | Unchanged checker; rerun inside both preflight runs: 5/5 unmasked legal, 20/20 uniform legal, 5/5 corrupted rejected | pass |
| Model probes (L40S) | `sees_later_tokens`, `repeat_same_tokens`, `batch1_same_tokens`, and `padding_probe` run inside each L40S container before its fills | with the run |
| Separation, training smoke, condition, copies | Skipped as #237/#238: no training, fixed ask `unknown`, no example team | n/a |

## Noise floor

Paired on all 270 masks and mask seeds, McNemar exact on discordant pairs. At 270 masks and two primary tests per model (α = 0.025), 80% power detects a ~5.5-percentage-point change in legal share (about 15 masks) when discordant pairs are ~7%. A smaller real change can pass as within noise; one seed per mask as fixed by the ticket.
