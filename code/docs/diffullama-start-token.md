# Why DiffuLLaMA 7B cannot copy a visible answer: the missing start token

Pre-registration, 2026-11-17. Part of [Rule out the sampler as the cause of the masked models' bad text](https://github.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/issues/236) (#236), under the masked text diffusion map (#160). The owner chose, 2026-11-17, to keep #236 open and probe DiffuLLaMA before closing it. Mode: **confirmatory** for the start-token arm (bar fixed below before any run), **exploratory** for the eager-attention arm and the redraw.

## What came before

#236's check 2, the copy test (11-16), wrote one team twice and hid one value of the second copy. Every other shifted model copied the value above it in the setting it was trained with: DiffuGPT small 93%, DiffuGPT medium 91%, Dream 7B 99%. DiffuLLaMA 7B copied 23 of 168 values (14%) with the shift and 7 (4%) without. Its saved fills (`~/vgc-data/sampler_checks_236/copy/diffullama.json`) show why: 136 of 168 are only newlines, in both settings. A newline holds no quote, colon or comma, so DiffuLLaMA's check 3 share (43.8% of values inside their quotes) counts those fills as clean values and does not measure the model either.

Ruled out on 2026-11-17, from code and files:

| Suspect | Finding | Source |
| --- | --- | --- |
| Wrong mask token | Token 811 (`ти`) is the tokenizer's own `mask_token` at the pinned revision | `special_tokens_map.json` of `diffusionfamily/diffullama` at `31b22850` |
| Causal attention | The 11-16 redraw recorded `sees_later_tokens: true`; transformers 4.57.3 passes a 4D mask through unchanged (`masking_utils._preprocess_mask_arguments`); a random two-layer LLaMA confirms it under sdpa and eager | `rerun/diffullama/slots-batch004.json`; `scripts/test_sampler_checks.py::LlamaAttentionTest` |
| Shift direction | DiffuLLaMA's `generate_samples` writes position i from position i − 1's logits, as our `shift_right` does | `model.py` in HKUNLP/DiffuLLaMA at `c17e897f` |
| Precision | bfloat16, the stored type, as DiffuLLaMA's own script loads it | `config.json`; `inf_diffullama.py` |

What still differs from DiffuLLaMA's own code: every text here is encoded with `add_special_tokens=False`, so the model never sees the start token `<s>`. DiffuLLaMA's `inf_diffullama.py` puts `[tokenizer.bos_token_id]` first, and its Lambada evaluation encodes with the start token. Its infilling evaluations (ROCStories, HumanEval infilling) encode without it, so the start token is a suspect, not a known requirement. The second difference is the attention code: its scripts default to `eager`, ours runs `sdpa`.

## Contract

| Row | Value |
| --- | --- |
| Claim | DiffuLLaMA 7B fails the copy test because our texts lack the start token `<s>`. **Refuted** if, with `<s>` first, it copies fewer than 135 of 168 values whole with the shift (80%, near DiffuGPT's 91 to 93%). If it holds, every DiffuLLaMA text gets `<s>` first, DiffuLLaMA's #236 and #170 numbers are replaced by the start-token runs, and only then can its bad text be blamed on the model. Scope: copying a value visible in the same text, for this one model. It says nothing about prompt quality or win rate. |
| Fields | Nothing is trained. The copy test hides one value of the second team copy (species, item, ability, Stat Alignment, four moves, six Stat Point numbers); the model writes that value's tokens and copies nothing else. The redraw hides all 84 values of six Pokémon, as #170. |
| Vocabulary | DiffuLLaMA's LLaMA-2 tokenizer, 32,000 tokens; values are free text, not regulation tables. The 188 species in `~/vgc-data/teams/hps_reg_mb_100k.jsonl` define a real species name for the redraw count. |
| Teams | The copy test: halves (Pokémon 1 to 3 and 4 to 6) of the first two distinct meta-grid teams, 168 values, as on 11-16. The redraw: the #170 benchmark text (`~/vgc-data/masked_memory_bench/canvas/diffullama.json`) with `<s>` before it. |
| Scorer | No battles. Copy test: a value counts when every answer token is filled exactly, at temperature 0, one token per step, most confident mask first. Redraw: share of values with no `"`, `:` or `,`, blank values, and real species names. |
| Search region | None: no proposals, asks or filters. |
| Settings | **Scientific:** start token (`<s>` first or not); attention code (sdpa or eager), exploratory. **Fixed:** checkpoint and revision, bfloat16, laptop GPU, mask token 811, the 168 values, shift on and off both reported, temperature 0 for the copy test; temperature 1.0, 128 steps, seed 2026, batch 4 and mask positions scored for the redraw. |
| Plan | **Primary:** values copied whole with the shift, start token against the 11-16 run, paired per value. One measure with nothing traded against it, so the question is pass or fail at the bar. **Stopping rule:** the eager arm runs only if the start-token arm stays under 135. **Baselines:** the 11-16 run (no start token, sdpa); DiffuGPT and Dream on the same test. |

## Arms

Copy-test arms use sdpa attention and no start token unless their name says otherwise. The control is the 11-16 run, **No start token, sdpa attention** (`benchmarked`).

| Reader name | Code ID | Values | Runs when |
| --- | --- | --- | --- |
| Repeated control, first 20 values | `repeat` | 20 | Always, first |
| Start token first, 168 values | `start_token` | 168 | Always |
| Eager attention, 168 values | `eager` | 168 | Start token first under 135 of 168 |
| Redrawn #170 text, start token first | `redraw` | 4 texts, 336 values | Always |

If neither the start token nor eager attention reaches the bar, the sampler is not ruled out for DiffuLLaMA. The next step would run DiffuLLaMA's own `generate_samples` with its `attention_patch.py`, which targets an older transformers and needs its own environment; that is the owner's call.

## Probes

| Probe | Result | Pass |
| --- | --- | --- |
| Ceiling | A model that copies perfectly scores 168 of 168; DiffuGPT and Dream reach 91 to 99% on the same test, so the bar can be met | yes |
| Round trip | `copy_cases` with `start_id`: ids gain `<s>` first, every masked position moves up by one, the answer is unchanged (`StartTokenTest`). On the real tokenizer: 42 values per team half, 168 in all, text starting `<s> Team: [{"species":"Blaziken-M` | yes |
| Coverage | The mask token is the tokenizer's own; the start token is id 1 | yes |
| Read teams | 11-16 fills read: 136 of 168 are newlines only. The redraw post prints the first Pokémon's 14 values from each text | yes |
| Training smoke | No training | skipped: nothing trains |
| Controls | DiffuGPT small and medium, Dream and both masked Qwen models already pass the same copy test (11-16) | carried |
| Invariance | Shift on and off are both reported for every arm | yes |
| Repeat | Repeated control refills 20 values as on 11-16 and posts how many fills match | runs first |
| Separation, condition, copies | No training teams, ask or search | skipped: not reachable |
| Benchmark canvas | `start-canvas` output: `<s>` first, every slot still holds token 811 and no other position does (`StartTokenTest`, and checked on the real file) | yes |
| Posts | Every post rendered from stand-in files before launch | yes |

## Noise and budget

At temperature 0 the fill is the argmax at every step, so the only noise is GPU arithmetic; the repeated control measures it. Read as an estimate of copy ability, 168 values give a binomial standard error of about 0.031 at 80%. The gap from 23 to 135 is 112 values; a paired sign test on the values that change detects it far below the 5% level, so the budget can detect the smallest effect worth acting on. Cost: one 13.48 GB download (34 GB free, 4 GB kept free by `fetch`), about 36 minutes per 168-value arm (11-16 timing), about 20 minutes for the redraw; 1.5 to 2 hours in all, laptop GPU, one model process at a time, behind other sessions' jobs.

## Running it

Frozen copy and queue: `~/vgc-data/diffullama_start_token/run/` (`dst.sh`, `post.py`, `src/`, `SHA256SUMS`). Each step posts on #236. Code: `scripts/sampler_checks.py copy --start-token`, `copy --attention eager`, `copy --limit`, `start-canvas`.
