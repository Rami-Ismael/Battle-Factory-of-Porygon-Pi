# Free-text answer instead of in-place mask slots

**Ticket:** [#240](https://github.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/issues/240) (under [#160](https://github.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/issues/160); reopens [#167](https://github.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/issues/167)) · **Date:** 2026-11-27 · **Mode:** Confirmatory on first-try legality

> **Question:** When part of a Pokémon team is missing, does a model build more legal teams if it **writes the missing pieces at the end** instead of **filling blanks inside the team**?

```text
In-place (fill fixed-size blanks inside the team):
  "item": "<MASK><MASK><MASK><MASK><MASK><MASK>"   ← always 6 tokens wide

Free-text (write the missing answers at the end):
  "item": null
  Answer: ["Choice Scarf"]                          ← model writes the list and stops
```

---

## Why we ran this

In earlier tests ([#237](inplace-prompt-heldout-baseline.md)–[#465](https://github.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/issues/465)), filling blanks inside the team caused two recurring problems:

- **Extra blank tokens broke the JSON.** Every blank had to be wide enough for the longest legal name (9 tokens for a species, 6 for an item or move, 2 per Stat Point). When the real answer was shorter, models filled the leftover blanks with stray quotes or punctuation (1,809 values with quotes on iLLaDA 8B Instruct in #465).
- **Prompt edits did not fix it on small models.** Only adding one example team helped iLLaDA 8B Instruct (`12/270 → 28/257`, p = 0.0009).

Letting the model write its answer at the end removes those fixed-size blanks — at the risk of syntax errors or models failing to stop.

---

## Experiment design

All 5 models run on the same **270 held-out masks** (one first-try fill per mask, Modal L40S, batch cap 8, shared sampler: 128 steps, temperature 1.0, confidence order, exact linear schedule, seed 2026).

### Arms tested

| Reader name | Code ID | Output layout | Base prompt | Models |
| --- | --- | --- | --- | --- |
| In-place slots, header only | `inplace_header` | Fixed mask slots in `Team:` | Header only | All 5 models |
| Free-text answer, header only | `freetext_answer` | Free-text JSON list after `Team:` | Header only | All 5 models |
| In-place slots, example team on header only | `inplace_header_example` | Fixed mask slots in `Team:` | Header + example team | iLLaDA 8B Instruct |
| Free-text answer, example team on header only | `freetext_header_example` | Free-text JSON list after `Team:` | Header + example team | iLLaDA 8B Instruct |

### How each model is prompted

Inside `Team:`, hidden values are written as JSON `null` (zero mask tokens inside the team). All mask tokens sit in one answer region at the end:

- **DiffuGPT small & medium** (base text-completion models): plain text ending in `Answer:\n` + masks. *(DiffuGPT small is flagged: its 512-token training window is crossed by almost every free-text prompt.)*
- **Masked Qwen2.5 Coder 0.5B & Masked Qwen3 0.6B** (chat-tuned): wrapped in each model's native chat template (`enable_thinking=False` on Qwen3), with end-of-sequence/pad confidence gating on free-text fills (LLaDA Appendix B.4) so JSON content commits before trailing padding.
- **iLLaDA 8B Instruct** (chat-tuned): native chat template with a fixed **`[`** prefix at the start of the free-text assistant turn so it writes JSON directly instead of spending the short budget on `<think>` prose. *(Testing explicit `<think>...</think>` reasoning space is tracked separately in [#469](https://github.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/issues/469).)*

### How answers are bounded and judged

- **Answer length (`L`):** fixed per `(task, k)` cell to `worst_compact_tokens + 8` (1 end token + 7 formatting tokens), so length never leaks the hidden answer. There is **no** extra reasoning/thinking budget inside `L`.
- **Trimming:** cut at the first `eos_token_id` (`<|endoftext|>`, `<|im_end|>`, or `<[EOS]>`). If no end token appears, `hit_length_limit = True`. One outer ` ```json ... ``` ` fence is unwrapped if present; surrounding prose is never stripped.
- **Judged in 4 steps:** parses as the expected JSON list → visible fields unchanged → all names legal in Regulation M-B → accepted by Showdown (`gen9championsvgc2026regmb`). Every fill is also checked against all 17 team-building rules (`inplace_rule_audit.py`).

### Decision rule (confirmatory)

- **Four small models:** paired McNemar exact test on `inplace_header` → `freetext_answer`, significant if **p < 0.05**.
- **iLLaDA 8B Instruct:** paired McNemar exact test on both bases, significant if **p < 0.025** (Bonferroni across 2 tests).
- **Refuted** for a model if free text does not gain legal fills past its threshold (80% power detects ~13–15 masks out of 270). If free text wins, it becomes a candidate prompt layout for #160 fine-tuning; otherwise #167's in-place layout stands.

---

## Pre-registered prediction (before any fill)

- **Four small models:** legal fills and JSON parse share drop sharply (legal near 0/270, parse < 15%, length-limit hits > 70%).
- **iLLaDA 8B Instruct:** stays competitive or gains slightly (`freetext_answer` ~15–25/270 vs 12; `freetext_header_example` ~25–40/270 vs 28), with 50–85% JSON parses, < 35% length-limit hits, and 0/30 whole-Pokémon masks.

---

## What changed mid-run (v1 → Option A v2)

In our first run (v1), all four small models got plain text (`Answer:\n`), and iLLaDA 8B Instruct got an empty assistant turn. Checking the raw tokens and each model's paper (`ZHZisZZ/dllm` `ca176752`, iLLaDA Appendices A & B.4) uncovered three setup bugs:

1. **Both Masked Qwen models were missing their chat templates.** They were trained only on chat conversations, so plain `Answer:\n` text caused **0 of 28,950** `<|im_end|>` stop tokens (`hit_length_limit` 270/270, 0/270 legal).
2. **Masked Qwen free text collapsed to `[]` or echoed `"null"`.** Confidence-first decoding committed `<|im_end|>`/`<|endoftext|>` before writing any answer, and `"Write the k null ..."` caused literal `["null"]` outputs.
3. **iLLaDA 8B Instruct spent its short budget on `<think>` prose.** All 56 v1 free-text rows opened with `"<think>\nGot it, let's tackle this problem..."` and ran out of tokens on sentence 1. A 48-fill L40S smoke probe ($0.13) showed `<think>\n\n</think>\n` prefill still wrote prose (0/12 JSON), while starting the turn with **`[`** stopped `<think>` prose immediately (7/12 JSON, 1/12 legal).

### What Option A v2 fixed

- **DiffuGPT small & medium:** kept v1 runs as-is (base text-completion models have no chat template).
- **Masked Qwen2.5 Coder 0.5B & Masked Qwen3 0.6B:** added each model's native chat template (`enable_thinking=False` on Qwen3) on **both** arms, reworded the free-text prompt (`"Replace each null ... (never output null or a Pokémon name)"`), trimmed strictly at `eos_token_id` (`<|im_end|>`), and zeroed confidence on proposed EOS/pad tokens during free-text filling (LLaDA Appendix B.4).
- **iLLaDA 8B Instruct:** reused its 270 completed `inplace_header` fills and ran its other three arms with the fixed **`[`** assistant prefix and v2 prompt wording.
- **Standing rule added:** updated `AGENTS.md` (`# Running external models`) to require primary-source template/sampler checks and a 2–3 mask smoke decode read by eye before every future launch.

### Positive impact of Option A v2 (measured v1 → v2)

| Model & arm | Legal (of 270) | Parses as JSON | Hit length limit | What improved |
| --- | --- | --- | --- | --- |
| **Masked Qwen2.5 Coder 0.5B** — in-place | **3 → 10** (+3.3×) | **144 → 217** (53% → 80%) | — | Native chat template tripled legal Stat Points fills |
| **Masked Qwen2.5 Coder 0.5B** — free-text | 0 → 0 | **1 → 25** (0% → 9%) | **270 → 0** (100% → 0%) | Stopped cleanly with `<|im_end|>` on every mask |
| **Masked Qwen3 0.6B** — in-place | **3 → 6** (2×) | **129 → 244** (48% → 90%) | — | Native chat template doubled legal fills & hit 90% JSON |
| **Masked Qwen3 0.6B** — free-text | **0 → 9** | **0 → 119** (0% → 44%) | **270 → 1** (100% → 0.4%) | Solved 9/30 Stat Alignments (outscoring its own in-place arm) |
| **iLLaDA 8B Instruct** — free-text (header) | **0/56 → 6/270** | **0/56 → 95/270** (35%) | 56/56 → 178/270 (66%) | Eliminated `<think>` prose (0/270) and unlocked **20/105** real species (vs 0/105 in-place) |

> **Takeaway for future work:** Just matching the checkpoint's native chat template raised Masked Qwen2.5 Coder 0.5B (`3 → 10`) and Masked Qwen3 0.6B (`3 → 6`) on the existing in-place prompt, and turned Masked Qwen3 0.6B free-text from a 0/270 failure into 9/270 legal fills with 99.6% clean EOS stopping. Earlier un-templated Qwen baselines (#237–#445) likely understate what those two checkpoints can do.

---

## Preflight summary

All checks passed before launching v2 on Modal L40S:

- **Tokenizer & chat-template parity:** exact byte-for-byte match against Hugging Face `apply_chat_template` on Masked Qwen2.5 Coder 0.5B, Masked Qwen3 0.6B (`enable_thinking=False`), and iLLaDA 8B Instruct; zero BPE boundary merges.
- **CPU grid preflight (all 270 masks, all 5 real tokenizers):** 270/270 built, 0 dropped, 0 unknown tokens, 270/270 ceiling legal, 270/270 round-trip identical paste, 0/270 round-trip limit hits, 0/270 random-token floor.
- **Pre-launch smoke decodes:** 3-mask local MPS decode on both Qwen models (3/3 clean `<|im_end|>` stops, zero `"null"` echoes, zero `[]` collapse); 48-fill L40S smoke probe on iLLaDA (`smoke-illada.json`).
- **Unit tests & GPU probes:** `test_inplace_prompt_grid.py`, `test_inplace_prompt_fill_modal.py`, and `test_batched_fills.py` (31 tests) all pass; every L40S container passed `sees_later_tokens`, `repeat_same_tokens = 1.0`, and `pad_leak_max_diff = 0.0`.

---

## Results (2026-11-28)

**Refuted for all five models.** Writing hidden values as a free-text JSON list at the end raises **zero** models above filling fixed-length slots in place. **#167's in-place layout stands.**

| Model & base | Legal, first try (of 270) | Gained / lost (McNemar p) | Parses as JSON | Reads as team | Real species | Hit length limit | Peak GPU | Seconds / team |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **DiffuGPT small** *(flagged: 512 window)*, header | 18 → **0** | +0 / −18 (`p = 7.6e-6`) | 80 → 0 | 77 → 0 | 0 → 0 / 105 | 270/270 (100%) | 0.83 GiB | 0.10 → 0.16 |
| **DiffuGPT medium**, header | 23 → **0** | +0 / −23 (`p = 2.4e-7`) | 236 → 0 | 110 → 0 | 0 → 0 / 105 | 270/270 (100%) | 1.41 GiB | 0.26 → 0.43 |
| **Masked Qwen2.5 Coder 0.5B** *(chat + EOS gate)*, header | 10 → **0** | +0 / −10 (`p = 0.0020`) | 217 → 25 | 95 → 4 | 0 → 0 / 105 | **0/270 (0%)** | 2.69 GiB | 0.32 → 0.55 |
| **Masked Qwen3 0.6B** *(chat + EOS gate)*, header | 6 → **9** | +9 / −6 (`p = 0.61`) | 244 → 119 | 117 → 48 | 0 → 0 / 105 | **1/270 (0.4%)** | 2.88 GiB | 0.44 → 0.74 |
| **iLLaDA 8B Instruct** *(chat + `[`)*, header | 12 → **6** | +5 / −11 (`p = 0.21`) | 267 → 95 | 140 → 71 | **0 → 20 / 105** | 178/270 (66%) | 15.93 GiB | 2.56 → 4.27 |
| **iLLaDA 8B Instruct** *(chat + `[`)*, example team | 28 → **3** | +1 / −26 (`p = 4.2e-7`) | 269 → 64 | 161 → 26 | **0 → 31 / 105** | 200/270 (74%) | 15.93 GiB | 5.09 → 7.53 |

### Why free text lost — and what it revealed

- **JSON syntax broke far more often than in-place slots.** Removing the surrounding JSON skeleton pushed `does not parse as a team` up on every model (`48% → 74%` on iLLaDA header, `40% → 90%` on iLLaDA example, `57% → 82%` on Qwen3, `65% → 99%` on Qwen Coder, `59–72% → 100%` on DiffuGPT).
- **When free text *did* parse, names and stat sums were much cleaner.** On parsed fills, `not a Reg M-B species` fell from 100% to 40% (iLLaDA header) and 9% (iLLaDA example, 18/32 copied from the example team — producing 1 legal whole Pokémon, Camerupt), `not a Reg M-B Stat Alignment` fell to 12% / 0%, and `Stat Points over 66` fell from 21–27% to **0–3%**.
- **Masked Qwen3 0.6B solved completely different masks under the two layouts:** in-place solved Stat Points (`5/30`) + 1 Stat Alignment, while free text solved **Stat Alignments (`9/30`)** and lost all Stat Points (`0/30`), matching within noise (`p = 0.61`).
- **iLLaDA 8B Instruct rewrote the entire team JSON instead of only the missing fields — especially with the example team:** starting the assistant turn with `[` primed iLLaDA to copy the `[{"species": ...}]` array format from `Team:` and `Example team:`, writing `"species"` on **72/240 (30%)** of non-Pokémon masks under header only and **161/240 (67%)** under the example team (dropping `natures` from 25/30 in-place to 2/30 free text). On top of that, multiline pretty-printing (178/270 header) and missing `<[EOS]>` after `]` (43/270 header, 92/270 example) caused 66–74% length-limit hits. Giving instruct models explicit `<think>...</think>` reasoning space is tracked in [#469](https://github.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/issues/469).
- **Prediction check:** held on DiffuGPT small/medium and Masked Qwen2.5 Coder 0.5B dropping to 0; missed on Masked Qwen3 0.6B (reached 9/270 legal and 44% JSON parses with 0.4% limit hits once given its chat template and EOS gate) and missed on iLLaDA 8B Instruct (dropped 28 → 3 on example team and hit the length limit on 66–74% instead of <35%, while unexpectedly jumping from 0 to 20–31/105 real species).
- **Audit, memory, and cost:** `inplace_rule_audit.py` checked all **3,240 fills** with **0 rejudge mismatches, 0 legal-but-broken, and 0 illegal-but-clean** (2 expected Showdown Species-Clause notes on fabricated species names in iLLaDA). Every model stayed under both the 17.76 GiB laptop ceiling and L40S capacity (peak 0.83–15.93 GiB). Actual Modal credit spent on Option A v2: **$3.11** (**$2.63** November credit left of $30.00, $0.00 billed).

---

<details>
<summary>Details: exact answer-region token budgets (L), prompt templates, storyboard, and file paths</summary>

### Answer-region length (`L` tokens) per task type

| Task type | `k` range | DiffuGPT small & medium | Masked Qwen2.5 Coder & Qwen3 | iLLaDA 8B Instruct (incl. `[`) |
| --- | --- | --- | --- | --- |
| Items (`items`) | 1–6 | 16–51 (`9 + 7k`) | 16–51 (`9 + 7k`) | 15–45 (`9 + 6k`) |
| Abilities (`abilities`) | 1–6 | 15–45 (`9 + 6k`) | 15–45 (`9 + 6k`) | 15–45 (`9 + 6k`) |
| Stat Alignments (`natures`) | 1–6 | 13–33 (`9 + 4k`) | 13–33 (`9 + 4k`) | 13–33 (`9 + 4k`) |
| Stat Points (`stats`) | 1–6 | 38–173 (`11 + 27k`) | 43–208 (`10 + 33k`) | 42–207 (`9 + 33k`) |
| Moves (`moves`) | 1–24 | 17–201 (`9 + 8k`) | 16–177 (`9 + 7k`) | 15–153 (`9 + 6k`) |
| Whole Pokémon (`pokemon`) | 1–6 | 112–622 (`10 + 102k`) | 112–622 (`10 + 102k`) | 108–603 (`9 + 99k`) |

**Total prompt + answer tokens (270 masks, v2 preflight):**
- `freetext_answer`: 523–736 (DiffuGPT), 553–757 (Masked Qwen2.5 Coder 0.5B), 536–740 (Masked Qwen3 0.6B), 551–721 (iLLaDA 8B Instruct) — 0/270 over 1,024.
- `inplace_header` (chat-templated Qwen): 507–693 (Masked Qwen2.5 Coder 0.5B), 490–676 (Masked Qwen3 0.6B).
- `freetext_header_example` (iLLaDA 8B Instruct): 993–1,163 of 8,192 — 0/270 dropped.

### Exact v2 instruction lines (`freetext-answer-v2-no-meta-teams`)

- **Scalar / moves:** `Replace {the/each} null {singular} in Team with a legal Champions VGC Reg M-B {singular} (never output null or a Pokémon name). Output only a compact JSON list of {1 string / k strings in order}.`
- **Stat Points:** `Replace {the/each} null statPoints in Team with a JSON object with keys HP, Atk, Def, SpA, SpD, Spe (never output null). Output only a compact JSON list of {1 object / k objects in order}.`
- **Whole Pokémon:** `Replace {the/each} null Pokémon in Team with a legal Champions VGC Reg M-B Pokémon object with keys species, item, ability, nature, moves, statPoints (never output null). Output only a compact JSON list of {1 object / k objects in order}.`

### Planned storyboard figures (`code/scripts/freetext_answer_post.py`)

1. **Five-model comparison:** first-try legal fills beside JSON parse share, in-place vs free-text.
2. **iLLaDA 8B Instruct 2 × 2 matrix:** layout (in-place vs free-text) × base (header vs example team) with gained/lost and McNemar p.
3. **Length-limit hit share (`hit_length_limit`):** by model and task type.
4. **Full 17-rule audit matrix:** every rule broken across all five models and arms.

### Files & reproducibility

- **Prompt builder:** `code/scripts/masked_diffusion_text.py` (`freetext_text`)
- **Grid runner & Modal launcher:** `code/scripts/inplace_prompt_grid.py`, `code/scripts/inplace_prompt_fill_modal.py`
- **Rule audit & figures:** `code/scripts/inplace_rule_audit.py`, `code/scripts/freetext_answer_post.py`
- **Grid file:** `eval2.tasks.json` (SHA-256 `596b3adc19ab`), vocabulary `code/data/regmb-vocabulary.json`
- **Frozen code & results:** `~/vgc-data/inplace_prompt_grid/run240v2`, `~/vgc-data/inplace_prompt_grid/modal-l40s-240-v2/` (v1 diagnostic runs in `modal-l40s-240/`)
- **Sampler temperature note:** Official dLLM (`MDLMSampler`) and iLLaDA evaluations use `temperature = 0` (no nucleus/`top_p`). #240 keeps the repo's shared `temperature = 1.0` across paired arms so layout is not confounded with temperature.

</details>
