# DiffuGPT and DiffuLLaMA checkpoint evidence

## What are the real checkpoints and their architecture?

### Takeaway
All three names resolve to real checkpoints. Their released inference wrapper uses fully bidirectional attention and absorbing mask diffusion. The checkpoint architecture names alone do not prove this: the external wrapper and attention patches are necessary. All three are ruled out for an unchanged 24,644-token prompt by their configured position limits. Short-prompt memory is a separate question.

### Cited Findings

Immutable Hugging Face revisions inspected on 2026-11-05:

| Reader name and checkpoint | Revision | Config and header facts |
| --- | --- | --- |
| Small GPT diffusion, `diffusionfamily/diffugpt-s` | `391d9850148c0beda355a8874bea958a40164c74` | GPT-2, 12 layers, hidden 768, 12 heads, learned positions 1024. Config `vocab_size=50257`. [Config](https://huggingface.co/diffusionfamily/diffugpt-s/resolve/391d9850148c0beda355a8874bea958a40164c74/config.json), [weight header source](https://huggingface.co/diffusionfamily/diffugpt-s/resolve/391d9850148c0beda355a8874bea958a40164c74/model.safetensors) |
| Medium GPT diffusion, `diffusionfamily/diffugpt-m` | `ce62f89724018390198ded702c4926791cf17c21` | GPT-2, 24 layers, hidden 1024, 16 heads, learned positions 1024. Config says vocabulary 50257 but stored embedding has 50258 rows: use header and tokenizer, not config alone. [Config](https://huggingface.co/diffusionfamily/diffugpt-m/resolve/ce62f89724018390198ded702c4926791cf17c21/config.json), [weight header source](https://huggingface.co/diffusionfamily/diffugpt-m/resolve/ce62f89724018390198ded702c4926791cf17c21/model.safetensors) |
| Llama 2 diffusion, `diffusionfamily/diffullama` | `31b22850c7b397d6948ea04076099b9ff7110adc` | Llama, 32 layers, hidden 4096, 32 query and KV heads, vocabulary 32000, max positions 4096, `rope_scaling=null`, untied input/output embeddings. [Config](https://huggingface.co/diffusionfamily/diffullama/resolve/31b22850c7b397d6948ea04076099b9ff7110adc/config.json) |

- External code pinned at HKUNLP/DiffuLLaMA `c17e897f6476c174b4623da594e4c65554f1613d`: GPT wrapper fills every causal bias with True; sampler calls `get_anneal_attn_mask(...attn_mask_ratio=1.0)` which constructs an all-zero additive attention mask. Llama patch passes that 4D mask through instead of constructing the causal mask. Flash path explicitly sets `is_causal=False`. No block boundaries or block mask occur in this inference route. [Wrapper and sampler](https://github.com/HKUNLP/DiffuLLaMA/blob/c17e897f6476c174b4623da594e4c65554f1613d/model.py), [attention patch](https://github.com/HKUNLP/DiffuLLaMA/blob/c17e897f6476c174b4623da594e4c65554f1613d/attention_patch.py).
- At every denoising step `model(xt, attention_mask=attention_mask)` embeds and forwards the whole `xt`, including the fixed prefix. The prefix is protected against replacement by `src_mask`, not excluded from attention or forward computation. No `past_key_values` is supplied or reused across steps even though configs say `use_cache=true`. The same attention mask is constructed once and reused; sampling state and fixed token IDs persist. [Sampler](https://github.com/HKUNLP/DiffuLLaMA/blob/c17e897f6476c174b4623da594e4c65554f1613d/model.py).
- These HF repositories have no `auto_map` or packaged modelling Python in their file lists. GPT config architecture is `DiscreteDiffusionModel`, Llama is `LlamaForCausalLM`. Correct diffusion inference requires the external custom wrapper and patch. Loading stock AutoModel alone is not enough. [Small API](https://huggingface.co/api/models/diffusionfamily/diffugpt-s/revision/391d9850148c0beda355a8874bea958a40164c74), [medium API](https://huggingface.co/api/models/diffusionfamily/diffugpt-m/revision/ce62f89724018390198ded702c4926791cf17c21), [Llama API](https://huggingface.co/api/models/diffusionfamily/diffullama/revision/31b22850c7b397d6948ea04076099b9ff7110adc).
- Released CLI scripts hardcode `device='cuda'` and `.to('cuda')`. Llama permits `eager`, `sdpa`, or `flash_attention_2`; the eager path uses the regular patched transformer operations and does not require the flash kernel. [GPT CLI](https://github.com/HKUNLP/DiffuLLaMA/blob/c17e897f6476c174b4623da594e4c65554f1613d/inf_diffugpt.py), [Llama CLI](https://github.com/HKUNLP/DiffuLLaMA/blob/c17e897f6476c174b4623da594e4c65554f1613d/inf_diffullama.py), [patch](https://github.com/HKUNLP/DiffuLLaMA/blob/c17e897f6476c174b4623da594e4c65554f1613d/attention_patch.py).

### Inferences
- A device-port plus eager attention is a plausible Apple-silicon path. The released CLIs do not run unchanged on MPS. No Apple execution was tested. FlashAttention 2 is not the path to plan for on this laptop.
- Exact reusable prompt KV caching is invalid in the general bidirectional network: prompt hidden states at deeper layers depend on changing generated tokens. Fixed embeddings and masks can be reused; an approximate-cache method would change the algorithm and needs separate verification.
- None is an architecture-matched swap with the existing Qwen2.5-based Fast-dLLM v2 1.5B or Qwen2.5 3B block fine-tunes. GPT small/medium would require a GPT-2 small/medium block counterpart; Llama requires a Llama-2-7B block counterpart. No exact pretrained block counterpart was verified here. Among the project's existing block baselines neither is a controlled architecture match. Qwen2.5 3B is closer in scale to DiffuLLaMA than 1.5B, but remains half its size and a different architecture. Do not call that a matched swap.

### Gaps
- Runtime Apple support, throughput and exact peak allocation remain unverified by design. Static eager availability does not prove MPS correctness.
- No exact block checkpoint for these GPT/Llama architectures was established. There is no evidence here for extending learned GPT positions or Llama's 4096 configured context to 24,644.

## What do headers and tokenizer files prove about size and context?

### Takeaway
The stored unique tensor totals are 124.440M, 354.824M, and 6.738B. Small and medium GPT are comfortably small for short-prompt planning; DiffuLLaMA's short-prompt fit remains conditional on battle/OS memory and implementation. Full Jev context is excluded independently of memory.

### Cited Findings
All counts below are sums of products of tensor shapes in HTTP-range-read safetensors headers. File bytes include headers. No full weights were downloaded. See `gpt_llama/manifest.json`, per-request `data` files, and `gpt_llama/results.txt` for raw evidence and computation.

| Candidate | Stored parameter elements, including embeddings | Excluding input and output token embeddings | Excluding all embeddings, including learned position embeddings | File bytes and header dtype |
| --- | ---: | ---: | ---: | --- |
| Small GPT diffusion | 124,439,808 | 85,842,432 | 85,056,000 | 248,894,800; F16 |
| Medium GPT diffusion | 354,824,192 | 303,360,000 | 302,311,424 | 709,679,000; F16 |
| Llama 2 diffusion | 6,738,415,616 | 6,476,271,616 | 6,476,271,616 | 13,476,865,048 over three shards; BF16 |

Sources: [small header](https://huggingface.co/diffusionfamily/diffugpt-s/resolve/391d9850148c0beda355a8874bea958a40164c74/model.safetensors), [medium header](https://huggingface.co/diffusionfamily/diffugpt-m/resolve/ce62f89724018390198ded702c4926791cf17c21/model.safetensors), [Llama shard 1](https://huggingface.co/diffusionfamily/diffullama/resolve/31b22850c7b397d6948ea04076099b9ff7110adc/model-00001-of-00003.safetensors), [Llama shard 2](https://huggingface.co/diffusionfamily/diffullama/resolve/31b22850c7b397d6948ea04076099b9ff7110adc/model-00002-of-00003.safetensors), [Llama shard 3](https://huggingface.co/diffusionfamily/diffullama/resolve/31b22850c7b397d6948ea04076099b9ff7110adc/model-00003-of-00003.safetensors).

GPT headers contain a single shared token embedding tensor; wrapper uses the GPT LM head from the same base model. Do not double-count a tied head. Small token embedding is `[50257,768]`, position embedding `[1024,768]`. Medium token embedding is `[50258,1024]`, position embedding `[1024,1024]`. Llama input embedding and LM head are separate `[32000,4096]` tensors. Header sum is distinct from the paper's rounded family size. [Wrapper](https://github.com/HKUNLP/DiffuLLaMA/blob/c17e897f6476c174b4623da594e4c65554f1613d/model.py), header links above.

| Candidate | Tokenizer and mask | Configured positions | 24,644-token prompt |
| --- | --- | ---: | --- |
| Small GPT diffusion | GPT2Tokenizer, byte-level BPE; `------------` token ID 10541, reused vocabulary token | 1024 | No, before adding any output |
| Medium GPT diffusion | GPT2Tokenizer, byte-level BPE; `[¨M¨]` added token ID 50257 | 1024 | No, before adding any output |
| Llama 2 diffusion | LlamaTokenizer, SentencePiece files; reused `ти` vocabulary token ID 811 | 4096 | No configured support, before adding output |

Sources: [small tokenizer config](https://huggingface.co/diffusionfamily/diffugpt-s/resolve/391d9850148c0beda355a8874bea958a40164c74/tokenizer_config.json), [small tokenizer vocabulary](https://huggingface.co/diffusionfamily/diffugpt-s/resolve/391d9850148c0beda355a8874bea958a40164c74/tokenizer.json), [medium tokenizer config](https://huggingface.co/diffusionfamily/diffugpt-m/resolve/ce62f89724018390198ded702c4926791cf17c21/tokenizer_config.json), [medium added token](https://huggingface.co/diffusionfamily/diffugpt-m/resolve/ce62f89724018390198ded702c4926791cf17c21/added_tokens.json), [Llama tokenizer config](https://huggingface.co/diffusionfamily/diffullama/resolve/31b22850c7b397d6948ea04076099b9ff7110adc/tokenizer_config.json), [Llama vocabulary](https://huggingface.co/diffusionfamily/diffullama/resolve/31b22850c7b397d6948ea04076099b9ff7110adc/tokenizer.json), model configs above. Llama tokenizer's enormous `model_max_length` is a sentinel, not evidence for context exceeding the model config.

### Inferences
Memory arithmetic, not a benchmark or a measured peak:

- Resident 16-bit weight payloads are 0.232 GiB, 0.661 GiB, and 12.551 GiB. Do not confuse decimal GB file sizes with GiB unified memory.
- Batch one at configured maximum context, one FP32 attention-score tensor is `4*heads*L²` bytes: 0.046875 GiB small GPT, 0.0625 GiB medium GPT, 2.0 GiB Llama. One FP32 full-vocabulary logits tensor is `4*L*vocab`: 0.192 GiB, 0.192 GiB, 0.488 GiB respectively. Temporary tensors from nucleus sorting, probabilities, int64 sorted indices and scores can coexist in the released sampler. Single-tensor arithmetic is a lower bound, not peak.
- A reproducible planning scenario for a port using global inference mode, batch one, 16-bit weights and eager attention is `weight payload + 2*(FP32 score tensor) + 6*(FP32 logits tensor) + 1 GiB other workspace`. At maximum native context this estimates 2.48 GiB small GPT, 2.94 GiB medium GPT and 20.48 GiB Llama, before OS/battles. The multipliers are conservative planning assumptions, not verified liveness bounds. Adding an assumed 4 GiB OS/battle reserve gives 6.48/6.94/24.48 GiB. Thus small/medium are plausible short-context candidates; Llama at maximum context should be excluded under this scenario, while shorter context can fit and must not be categorically ruled out.
- For an unsupported 24,644-token hypothetical Llama sequence, one FP32 attention-score tensor alone would be about 72.40 GiB, before weights. Optimized tiled attention could change this allocation, but not establish supported positions. GPT context exclusion already settles that prompt without extrapolating memory.
- The released sampler's initial forward is outside the later loop's `torch.no_grad()` block, so these inference-mode planning estimates do not assert the published script's actual peak. A device port should also address this first-forward graph retention.

### Gaps
No measured peak, reserve for the actual battle workload, tokenizer count of the actual Jev text, extrapolated long-context quality, or MPS kernel-memory behavior was verified. `24,644` is the ticket's token-count assumption, not a fresh measurement with these tokenizers.

## What license evidence supports a public fine tune?

### Takeaway
All three HF metadata entries label the checkpoint Apache-2.0, but none has an actual LICENSE file in the inspected snapshot. For DiffuLLaMA, a Llama-2-derived model must not be treated as Apache-only based on a card label. The base-model redistribution obligations remain relevant.

### Cited Findings
- Pinned snapshot file lists contain no LICENSE, and metadata declares `apache-2.0` for all three. This is a declaration, not a verified weight provenance chain. [Small snapshot](https://huggingface.co/api/models/diffusionfamily/diffugpt-s/revision/391d9850148c0beda355a8874bea958a40164c74), [medium snapshot](https://huggingface.co/api/models/diffusionfamily/diffugpt-m/revision/ce62f89724018390198ded702c4926791cf17c21), [Llama snapshot](https://huggingface.co/api/models/diffusionfamily/diffullama/revision/31b22850c7b397d6948ea04076099b9ff7110adc).
- GPT-2's upstream Modified MIT license requires preserving copyright and permission notices in copies or substantial portions. A public fine-tune should preserve this base-model notice as well as the diffusion authors' applicable notices. [OpenAI GPT-2 license, revision `9b63575ef42771a015060c964af2c3da4cf7c8ab`](https://github.com/openai/gpt-2/blob/9b63575ef42771a015060c964af2c3da4cf7c8ab/LICENSE).
- Llama 2's actual license requires supplying the agreement with derivative distributions, retaining the Llama 2 attribution in a Notice file, compliance with its incorporated acceptable-use policy, and bars using its materials or results to improve other LLMs outside Llama 2 derivatives. Section 2 imposes an additional commercial permission condition for certain organizations above 700M monthly active users. [Meta Llama 2 license, revision `7565eb6fee2175b2d4fe2cfb45067a61b35d7f5e`](https://github.com/meta-llama/llama/blob/7565eb6fee2175b2d4fe2cfb45067a61b35d7f5e/LICENSE).

### Inferences
For GPT public derivatives, retain the base MIT notice and the declared Apache license obligations: provide the license, preserve attribution/NOTICE where applicable and mark modifications. For DiffuLLaMA, carry the Llama 2 agreement and Notice and respect its restrictions; do not advertise unrestricted Apache-only redistribution. Confirm the release's conflicting Apache metadata with its authors before publication. Exact licensed lineage cannot be reconstructed from tensor dimensions alone.

### Gaps
The checkpoint repositories contain no weight-specific license text resolving their declarations. DiffuLLaMA's upstream Llama 2 provenance is asserted by the authors and consistent with its config/header, not cryptographically proven from weights. The incorporated AUP was not independently archived here. Apache text obligations should be linked to the report's independently verified Apache license source. No legal ruling on sublicensing or dataset rights is claimed.
