# Small Qwen masked checkpoints

## What do the released files prove?

### Takeaway
Both named masked candidates resolve to public checkpoints. Both use full bidirectional attention by default. Their serialized tensor counts exceed the parameter counts implied by their tied-embedding configs. This mismatch must remain explicit when quoting a model size.

### Cited Findings

Inspected 2026-11-05. Immutable checkpoint revisions:

| Reader name | Exact repository | Revision |
|---|---|---|
| Coder-derived masked Qwen, nominal 0.5B | `dllm-hub/Qwen2.5-Coder-0.5B-Instruct-diffusion-mdlm-v0.1` | `a284e895a6248256baf2f60502e54aba61b24c1a` |
| Qwen3-derived masked Qwen, nominal 0.6B | `dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1` | `c8d24a3f4adaeef46881b450e1bf7d1005203bd7` |

The Hub API response, config, tokenizer JSON/config, model Python and safetensors JSON header were saved under `qwen/`. Every network file has a separate newly created directory, `data` and provenance JSON. Safetensors reads were strictly bytes 0–7 and 8 through 7+header length, using HTTP Range, never payload. Python ran with `-I`; none of the model Python was executed. The API inventory is discovery evidence, not architectural evidence. [Coder API](https://huggingface.co/api/models/dllm-hub/Qwen2.5-Coder-0.5B-Instruct-diffusion-mdlm-v0.1/revision/a284e895a6248256baf2f60502e54aba61b24c1a?blobs=true), [Qwen3 API](https://huggingface.co/api/models/dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1/revision/c8d24a3f4adaeef46881b450e1bf7d1005203bd7?blobs=true).

| File-derived property | Coder-derived masked Qwen | Qwen3-derived masked Qwen |
|---|---:|---:|
| Stored tensor scalar count, all tensors | 630,167,424 | 751,632,384 |
| `model.embed_tokens.weight` | 151,936 × 896 = 136,134,656 | 151,936 × 1,024 = 155,582,464 |
| `lm_head.weight`, also present separately | 136,134,656 | 155,582,464 |
| Transformer scalars, excluding both token matrices | 357,898,112 | 440,467,456 |
| Unique parameter count implied by config tying input/output | 494,032,768 | 596,049,920 |
| Stored dtype | BF16, every tensor | BF16, every tensor |
| Safetensors payload bytes | 1,260,334,848 | 1,503,264,768 |
| Entire safetensors file bytes | 1,260,367,448 | 1,503,300,328 |
| Layers / hidden / MLP width | 24 / 896 / 4,864 | 28 / 1,024 / 3,072 |
| Query heads / KV heads / head dimension | 14 / 2 / 64 | 16 / 8 / 128 |
| Maximum configured positions | 32,768 | 40,960 |
| Tokenizer declared maximum | 32,768 | 131,072 |
| Mask token | `<\|mask\|>`, 151665 | `<\|mask\|>`, 151669 |

Counts are integer products and sums of header shapes, not counts from the model card. File byte sizes are Hub blob metadata, independently consistent with BF16 payload plus header bytes. [Coder header range target](https://huggingface.co/dllm-hub/Qwen2.5-Coder-0.5B-Instruct-diffusion-mdlm-v0.1/resolve/a284e895a6248256baf2f60502e54aba61b24c1a/model.safetensors), [Qwen3 header range target](https://huggingface.co/dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1/resolve/c8d24a3f4adaeef46881b450e1bf7d1005203bd7/model.safetensors), [Coder config](https://huggingface.co/dllm-hub/Qwen2.5-Coder-0.5B-Instruct-diffusion-mdlm-v0.1/blob/a284e895a6248256baf2f60502e54aba61b24c1a/config.json), [Qwen3 config](https://huggingface.co/dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1/blob/c8d24a3f4adaeef46881b450e1bf7d1005203bd7/config.json).

Both configs set `tie_word_embeddings: true`. Their remote head constructors call `post_init()` and inherit Transformers' tying machinery. At Transformers 4.57.0, immutable commit `8ac2b916b042b1f78b75c9eb941c0f5d2cdd8e10`, `tie_embeddings_and_encoder_decoder` ties output and input when that flag is true; `from_pretrained` also calls `tie_weights`. Therefore 751.6M stored scalars explains a rounded “0.8B with embeddings,” while the config-implied tied architecture is 596.0M unique parameters. Do not state that 751.6M is necessarily the in-memory trainable parameter count. The analogous discrepancy is 630.2M stored versus 494.0M unique for the coder model. The inspected current MDLM trainer does not explicitly untie weights. [Coder head code](https://huggingface.co/dllm-hub/Qwen2.5-Coder-0.5B-Instruct-diffusion-mdlm-v0.1/blob/a284e895a6248256baf2f60502e54aba61b24c1a/modeling_qwen2.py), [Qwen3 head code](https://huggingface.co/dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1/blob/c8d24a3f4adaeef46881b450e1bf7d1005203bd7/modeling_qwen3.py), [Transformers tying implementation](https://github.com/huggingface/transformers/blob/8ac2b916b042b1f78b75c9eb941c0f5d2cdd8e10/src/transformers/modeling_utils.py), [MDLM trainer](https://github.com/ZHZisZZ/dllm/blob/ca176752fbceec49c6b4777a2c18ae88e4eb10ed/dllm/core/trainers/mdlm.py).

Both remote forwards replace causal mask creation with a padding-only 4D mask for every attention layer. Config `layer_types` is entirely `full_attention`, `sliding_window` null and `use_sliding_window` false. There is no architectural block mask imposed in the normal forward. A caller may deliberately supply a custom 4D/dict/BlockMask, so this is the released default, not an inability to run with blocks. The inherited eager attention uses only the supplied additive mask; SDPA derives `is_causal=False` when that mask is present. [Coder forward](https://huggingface.co/dllm-hub/Qwen2.5-Coder-0.5B-Instruct-diffusion-mdlm-v0.1/blob/a284e895a6248256baf2f60502e54aba61b24c1a/modeling_qwen2.py), [Qwen3 forward](https://huggingface.co/dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1/blob/c8d24a3f4adaeef46881b450e1bf7d1005203bd7/modeling_qwen3.py), [inherited Qwen3 attention](https://github.com/huggingface/transformers/blob/8ac2b916b042b1f78b75c9eb941c0f5d2cdd8e10/src/transformers/models/qwen3/modeling_qwen3.py), [SDPA dispatch](https://github.com/huggingface/transformers/blob/8ac2b916b042b1f78b75c9eb941c0f5d2cdd8e10/src/transformers/integrations/sdpa_attention.py).

Both use Qwen2Tokenizer byte-level BPE, with explicit added mask tokens in `tokenizer.json` and `tokenizer_config.json`. Qwen3's tokenizer maximum of 131,072 does not override its model config of 40,960. A prompt already counted as 24,644 tokens fits both configured position budgets, leaving 8,124 and 16,316 token positions respectively for completion and any not-yet-counted template tokens. Actual Jev prompt counts per tokenizer have not been measured here. [Coder tokenizer](https://huggingface.co/dllm-hub/Qwen2.5-Coder-0.5B-Instruct-diffusion-mdlm-v0.1/blob/a284e895a6248256baf2f60502e54aba61b24c1a/tokenizer.json), [Coder tokenizer settings](https://huggingface.co/dllm-hub/Qwen2.5-Coder-0.5B-Instruct-diffusion-mdlm-v0.1/blob/a284e895a6248256baf2f60502e54aba61b24c1a/tokenizer_config.json), [Qwen3 tokenizer](https://huggingface.co/dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1/blob/c8d24a3f4adaeef46881b450e1bf7d1005203bd7/tokenizer.json), [Qwen3 tokenizer settings](https://huggingface.co/dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1/blob/c8d24a3f4adaeef46881b450e1bf7d1005203bd7/tokenizer_config.json).

### Inferences
- Both are genuine checkpoint candidates. Their small resident weights make them more plausible than 7–8B models, but long-prompt logits and attention dominate memory. Parameter count alone does not establish 24 GB fit.
- The parameter arithmetic above defines “without embeddings” as removing both the input token matrix and output vocabulary matrix; removing a single shared matrix from the config-implied tied total gives the same transformer count.

### Gaps
- Headers cannot prove whether the two serialized token matrices contain identical values. No weights were read. If they differ, tied loading may discard one distinct trained matrix. Investigate the intended loading convention before benchmarking or fine-tuning.
- Training history and whether the originally trained models had untied matrices are not proved by the current snapshots or current trainer.

## Can the laptop execute these and at what memory cost?

### Takeaway
The model forwards offer generic PyTorch eager and SDPA paths and do not require CUDA attention. Neither a working MPS run nor a peak-memory measurement is established. Rule out unmodified long-prompt eager attention, while retaining the checkpoints for a memory-aware port or shorter prompt.

### Cited Findings
The configs require custom AutoConfig and AutoModelForMaskedLM code through `auto_map` and nonstandard `a2d-qwen2` / `a2d-qwen3` model types. A normal Auto loader therefore needs trusted remote code or an audited local equivalent. The forward itself uses generic PyTorch and inherited Transformers attention. The `torch.set_default_device('cuda')` in each file is inside its `__main__` demo, not the imported forward. Flex attention is optional and a fallback annotation type exists; eager/SDPA need no FlashAttention package. [Coder code](https://huggingface.co/dllm-hub/Qwen2.5-Coder-0.5B-Instruct-diffusion-mdlm-v0.1/blob/a284e895a6248256baf2f60502e54aba61b24c1a/modeling_qwen2.py), [Qwen3 code](https://huggingface.co/dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1/blob/c8d24a3f4adaeef46881b450e1bf7d1005203bd7/modeling_qwen3.py).

The library MDLM sampler sends the complete prompt-plus-completion canvas on each denoising pass, without passing a reusable `past_key_values`. The forward only initializes a DynamicCache if its explicit `use_cache` argument is truthy, so the config's `use_cache:true` does not by itself cache this custom forward. The sampler calls full-length logits, then low-confidence remasking computes full-vocabulary softmax. Its configurable `block_size` schedules which output masks are revealed, not who may attend to whom; even with multiple reveal windows every forward still sees the entire canvas. [Sampler, revision ca176752fbceec49c6b4777a2c18ae88e4eb10ed](https://github.com/ZHZisZZ/dllm/blob/ca176752fbceec49c6b4777a2c18ae88e4eb10ed/dllm/core/samplers/mdlm.py).

The sampler helper casts token-transfer counts to float64 even at zero noise temperature; nonzero-temperature Gumbel noise also uses float64. The checkpoint README demo explicitly chooses CUDA or CPU, not MPS. These paths need inspection/porting for MPS rather than claiming an unmodified laptop-ready release. [Sampler helper](https://github.com/ZHZisZZ/dllm/blob/ca176752fbceec49c6b4777a2c18ae88e4eb10ed/dllm/core/samplers/utils.py), [Qwen3 README executable example only](https://huggingface.co/dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1/blob/c8d24a3f4adaeef46881b450e1bf7d1005203bd7/README.md).

### Inferences
All following numbers are arithmetic estimates, not measurements. Batch one, no classifier-free guidance, BF16 weights and activations, no training gradients, N=24,644 before adding generated tokens, V=151,936, GiB=2^30 bytes. Weight allocation is 0.920/1.110 GiB if tied, or 1.174/1.400 GiB if both matrices remain resident. One N×V BF16 logits tensor costs 6.974 GiB for either model. FP32 costs 13.949 GiB. Low-confidence softmax adds another logits-sized allocation, so retained BF16 logits plus BF16 probabilities already cost 13.949 GiB. Adding tied weights and the explicitly expanded BF16 N×N padding mask gives component sums of 16.000/16.190 GiB, before attention workspaces, layer intermediates, allocator overhead, generated tokens, operating system and battles. This is a component sum, not a predicted exact peak. FP32 retained logits plus probabilities alone cost 27.897 GiB and rule out that implementation for a 24 GB machine. The inherited model head does not explicitly force FP32 inference logits. [Head implementation](https://github.com/huggingface/transformers/blob/8ac2b916b042b1f78b75c9eb941c0f5d2cdd8e10/src/transformers/models/qwen3/modeling_qwen3.py).

Unmodified eager attention is worse: one BF16 H×N×N attention matrix costs 15.837 GiB for the coder model and 18.100 GiB for Qwen3; eager softmax explicitly computes an FP32 matrix of 31.675/36.199 GiB before converting back. The FP32 softmax matrix alone exceeds physical memory. Rule out that long-prompt eager path, even without battles. Generic SDPA is available, but its MPS allocation pattern was not measured. A memory-aware sampler should project logits only for masked/output positions, retain fewer vocabulary-sized arrays, and use efficient full attention; these are proposed changes, not released verified properties. [Eager attention](https://github.com/huggingface/transformers/blob/8ac2b916b042b1f78b75c9eb941c0f5d2cdd8e10/src/transformers/models/qwen3/modeling_qwen3.py), [mask expansion utility](https://github.com/huggingface/transformers/blob/8ac2b916b042b1f78b75c9eb941c0f5d2cdd8e10/src/transformers/modeling_attn_mask_utils.py).

For a short N=2,048 canvas, full BF16 logits and probabilities together are about 1.159 GiB; weights plus those tensors plus small activations leave substantially more space than at 24,644. This makes both reasonable candidates for a later laptop test. No claim of battle-concurrent fit or speed is measured.

A fixed prompt's higher-layer representations change when generated tokens change because attention is bidirectional. Standard autoregressive prefix KV caching is therefore not exact across denoising steps. Token IDs, embeddings of fixed tokens, and position encodings can be retained in a custom implementation; full per-layer prompt KV reuse changes the computation unless using an approximate cache or changing the attention scheme. Each step's dense attention scales approximately with (prompt+generation)^2, while projections, MLP and vocabulary scoring scale linearly in that total length. This follows from the inspected forward and sampler, not a benchmark.

### Gaps
- Exact peak memory, MPS kernel support at the laptop's installed Torch/macOS versions, BF16 support there, sustained generation speed and battle-concurrent headroom remain unverified.
- A generic eager/SDPA code path is evidence of portability potential, not proof that every operation runs on MPS unchanged.

## Licensing and the closest block swap

### Takeaway
The closest block counterparts exist with exactly matching configs and tokenizer settings. The weight snapshots have no LICENSE file; the declared Apache-2.0 license is corroborated by actual upstream license files, but the adapted checkpoints' grant itself remains metadata-level evidence.

### Cited Findings
The closest coder counterpart is `dllm-hub/Qwen2.5-Coder-0.5B-Instruct-diffusion-bd3lm-v0.1`, revision `4bcf3a00ba6ec28d9cf9c3ebc4a5a5dc01c6023c`. The closest Qwen3 counterpart is `dllm-hub/Qwen3-0.6B-diffusion-bd3lm-v0.1`, revision `60f77bba67ba94231c16f5fb29bc66c500b038f1`. Their configs match their respective masked counterparts, including the custom full-attention-capable architecture; block behavior is supplied by the block recipe/mask rather than a model-size difference. These are closer architectural controls than the map's existing Fast-dLLM v2 1.5B and Qwen2.5 3B fine-tunes. Config equality alone does not prove equal training corpora/steps or training compute. [Coder block config](https://huggingface.co/dllm-hub/Qwen2.5-Coder-0.5B-Instruct-diffusion-bd3lm-v0.1/blob/4bcf3a00ba6ec28d9cf9c3ebc4a5a5dc01c6023c/config.json), [Qwen3 block config](https://huggingface.co/dllm-hub/Qwen3-0.6B-diffusion-bd3lm-v0.1/blob/60f77bba67ba94231c16f5fb29bc66c500b038f1/config.json).

Checkpoint repository inventories contain no LICENSE or NOTICE, and direct LICENSE requests returned 404. Their README YAML declares Apache-2.0. Actual upstream `Qwen/Qwen2.5-Coder-0.5B-Instruct` revision `ea3f2471cf1b1f0db85067f1ef93848e38e88c25` and `Qwen/Qwen3-0.6B` revision `c1899de289a04d12100db370d81485cdf75e47ca` contain full Apache License 2.0 texts. The dLLM code repository has an Apache-2.0 LICENSE at `ca176752fbceec49c6b4777a2c18ae88e4eb10ed`. [Coder upstream license](https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B-Instruct/blob/ea3f2471cf1b1f0db85067f1ef93848e38e88c25/LICENSE), [Qwen3 upstream license](https://huggingface.co/Qwen/Qwen3-0.6B/blob/c1899de289a04d12100db370d81485cdf75e47ca/LICENSE), [dLLM code license](https://github.com/ZHZisZZ/dllm/blob/ca176752fbceec49c6b4777a2c18ae88e4eb10ed/LICENSE).

Apache section 4 requires a distributed derivative to carry the license, prominent modification notices, retained applicable copyright/patent/trademark/attribution notices and any applicable upstream NOTICE attribution. Neither inspected checkpoint nor inspected upstream listed a NOTICE file. A public fine-tune should ship the actual Apache text, provenance identifying exact parent checkpoint/revision and notices describing the weight/tokenizer/code modifications; preserve any applicable notices received with included code. The provenance and model-card recommendations go beyond the literal section-4 minimum. No Apache requirement makes all original training data public or requires a renamed checkpoint. [Apache section 4 in upstream file](https://huggingface.co/Qwen/Qwen3-0.6B/blob/c1899de289a04d12100db370d81485cdf75e47ca/LICENSE).

### Inferences
The adapted weights are plausibly publishable under their declared Apache-2.0 terms, but do not claim a standalone license document exists in those snapshots. Treat the absence as a release-record gap to resolve with maintainers if strict documentary evidence is required.

### Gaps
- No standalone adapted-weight LICENSE beyond model-card YAML was found. Upstream/code Apache licenses do not independently prove the adapted-weight owner's grant.
- Fine-tuning dataset redistribution and Pokémon-specific material rights are outside the inspected checkpoint license evidence.
