# Quantizing masked diffusion for laptop team generation

Research date: 2026-11-05. Scope: inference on the Apple M4 Pro with 24 GB unified memory. Fine-tuning stays in bfloat16, abbreviated bf16, on Modal.

```text
Measure the exact checkpoint and sampler in bf16
    fits the agreed memory ceiling -> keep bf16
    exceeds the ceiling            -> frozen 8-bit weights, bf16 activations
        fits                       -> evaluate this deployed model
        exceeds the ceiling        -> 4-bit weights, bf16 activations
Every stage: check legality, seconds per legal team, and peak memory
```

## Decision and evidence limits

Use **Optimum Quanto with frozen 8-bit weights and bf16 activations** for the first integration check across the large candidates. It preserves the existing PyTorch forward and denoising sampler. Treat this as a compatibility-first recommendation, conditional on a small device test, not a measured speed winner. Current bitsandbytes also supports MPS, correcting the older NVIDIA-only assumption, but its eight-bit path changes activation numerics and does not establish faster integer compute on this GPU. For Dream and LLaDA, upstream llama.cpp's Metal diffusion runner is a concrete alternative with GGUF quantization. DiffuLLaMA needs separate support verification.

This follows the [owner decision on benchmark issue #170](https://github.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/issues/170#issuecomment-6674805677). Do not preemptively quantize DiffuGPT small/medium or the masked Qwen 0.5B/0.6B candidates. Do not infer that 13–16 GB of bf16 weights automatically exceed 24 GB of unified memory. Peak allocations, the runtime's device working-set limit, and the agreed ceiling decide that. [Issue #169](https://github.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/issues/169) still contains a question rather than a numeric ceiling at inspection.

The evidence supports a compute bottleneck at approximately 1,000 tokens. It does **not** establish unchanged speed after quantization. Decompression, kernel quality, attention, vocabulary projection, CPU sampling, and memory pressure can change wall time. No source located reports real peak memory or seconds per legal Pokémon team for these exact models, sequence lengths, and laptop. All laptop performance and peak-memory estimates below are explicitly unverified.

No models were downloaded, packages installed, or GPU probes run for this report. The requested `code/scripts/whats_running.py` was absent from this checkout and the checked project directory; no applicable GPU queue-status command was located. Consequently, no compute ran without the requested prior check. Research used source files, issue discussions, documentation, and full-PDF methods/results/appendices, rather than abstracts alone. Reproduction commands below are proposed work, not completed tests.

## Which backends actually provide a path?

Metal Performance Shaders, abbreviated MPS, is PyTorch's Apple GPU backend. Metal is also the GPU interface used by MLX and llama.cpp. A low-bit file format alone does not prove low-bit computation or low runtime memory.

| Path | Device and bit widths | What changes for these models? | Maintenance and qualification |
| --- | --- | --- | --- |
| Standard torchao inference configurations | Official stable workflows list 8-bit weight-only, 4-bit weight-only, and dynamic activation configurations. The Transformers hardware table lists CUDA, Intel GPU, and CPU, not a general MPS guarantee. | Do not assume `Int8WeightOnlyConfig`, `Int4WeightOnlyConfig`, or dynamic 8-bit activation/weight configurations run on MPS just because PyTorch does. Unsupported operators may error; CPU fallback requires separate verification. | Torchao now says it is minimizing new feature development and moving work into PyTorch and consuming repositories. See [workflows](https://docs.pytorch.org/ao/stable/workflows/inference.html), [integration matrix](https://huggingface.co/docs/transformers/en/quantization/torchao), and [project notice](https://github.com/pytorch/ao). |
| Experimental torchao MPS kernels | Real GPU weight-only path. Current source accepts **1–7 bits**, including 4-bit, with floating-point activations and groups of 32, 64, 128, or 256. This is not an 8-bit MPS path. | Replaces linear modules; no diffusion-specific rewrite should be necessary for ordinary `nn.Linear` layers. Needs experimental compiled operators. Exact PyTorch 2.14.1 compatibility and model execution are unverified. | [Current quantization API](https://github.com/pytorch/ao/blob/main/torchao/experimental/quant_api.py) and [build controls](https://github.com/pytorch/ao/pull/1826) demonstrate support. A stock `pip install torchao` is insufficient evidence that these operators exist locally. |
| Optimum Quanto | Advertises MPS placement, 2/4/8-bit integer weights, float8 weights, and optional 8-bit integer/float activations. The 8-bit default implementation rescales weights and uses ordinary `torch.matmul`; it does not move them to CPU. Four-bit MPS uses generic tensor operations rather than the listed CUDA/CPU optimized subclasses. | Quantize and freeze supported linear modules, keep the existing remote-code model and sampler. Inspect custom projections which may bypass `nn.Linear`. Disable implicit CPU fallback for the acceptance test. | Explicit **maintenance mode**. [README](https://github.com/huggingface/optimum-quanto), [8-bit matmul dispatch](https://github.com/huggingface/optimum-quanto/blob/main/optimum/quanto/library/qbytes_mm.py), [4-bit tensor dispatch](https://github.com/huggingface/optimum-quanto/blob/main/optimum/quanto/tensor/weights/qbits.py). Exact model/PyTorch combination remains untested. |
| MLX and MLX LM | Metal GPU, with 4-bit and 8-bit weight quantization among supported configurations. This is a different model implementation and sampler. | Dream-style inference exists in [MacPaw/Fast-dLLM-mlx](https://github.com/MacPaw/Fast-dLLM-mlx), demonstrated with 8-bit DiffuCoder, not proof of parity for every Dream checkpoint. LLaDA ports exist in forks, including [akkikiki/mlx-dlm](https://github.com/akkikiki/mlx-dlm). Upstream requests [#14](https://github.com/ml-explore/mlx-lm/pull/14) and [#804](https://github.com/ml-explore/mlx-lm/pull/804) were closed without merge. | MLX development is active; diffusion support is fragmented. MacPaw's runner uses caching and threshold decoding options that change the benchmark unless held off. No verified ready-made port of the two exact dllm-hub masked Qwen checkpoints was found. Ordinary causal Qwen support does not establish masked diffusion support. |
| llama.cpp and GGUF | Metal GPU offload, CPU execution also available. Supports 8-bit `Q8_0` and 4-bit families such as `Q4_0` and mixed `Q4_K_M`. GPU offload must be checked in logs. | Upstream has **Dream and LLaDA architecture converters and a diffusion CLI**. The C++ implementation replaces Python remote code; conversion may still load custom tokenizer code. Base versus instruct weights need architecture/tokenizer/sampler checks. LLaDA-1.5 and DreamOn are not individually certified here. No verified DiffuLLaMA converter/runner or masked-Qwen diffusion recipe was found. | Maintained upstream, not merely an old fork. See [diffusion examples](https://github.com/ggml-org/llama.cpp/tree/master/examples/diffusion), [Dream converter](https://github.com/ggml-org/llama.cpp/blob/master/conversion/dream.py), [LLaDA converter](https://github.com/ggml-org/llama.cpp/blob/master/conversion/llada.py), and [formats](https://github.com/ggml-org/llama.cpp/blob/master/src/llama-model-loader.cpp). |
| bitsandbytes 0.50.x | MPS GPU supports LLM.int8 and NormalFloat4/FP4. Long-sequence four-bit computation reconstructs floating weights. | Replaces supported linear layers within PyTorch; preserve remote code and the diffusion sampler. LLM.int8 also quantizes activations. | Maintained, with macOS/PyTorch requirements and dispatch qualifications explained below. |
| Core ML | Apple documents 4/8-bit weights and 8-bit activations. Placement may use CPU, GPU, or Neural Engine depending on the converted graph and supported operations. | Export the full bidirectional denoiser; implement the iterative sampler around it. Python remote code cannot run as a Core ML graph. Trace/export compatibility, dynamic shapes, attention, and deployment placement need work. No verified ready-made path for the listed checkpoints was found. | Apple maintains [Core ML Tools](https://apple.github.io/coremltools/docs-guides/source/opt-quantization-overview.html). Its M4 8-bit compute claim specifically concerns the **Neural Engine**, not PyTorch MPS or generic Metal GPU execution. |

**Correction to #170's environment assumption:** bitsandbytes no longer requires NVIDIA. The [0.50.0 release](https://github.com/bitsandbytes-foundation/bitsandbytes/releases/tag/0.50.0), released July 25, describes working MPS four-bit and LLM.int8 configurations with PyTorch 2.9 or newer. It recommends the `kernels` package on macOS 26 or newer for optimized Metal kernels. Older systems use naive implementations. This is actively maintained and worth a separate device check; exact checkpoint integration is unverified.

The [0.50.2 MPS source](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/0.50.2/bitsandbytes/backends/mps/ops.py) registers four-bit GPU quantization, dequantization, and multiplication. Its optimized matrix-vector path handles one row. For multiple rows, including our long-sequence forward, `gemm_4bit` reconstructs weights and calls floating-point `linear`. This fallback stays on the tensors' device; “fallback” does not automatically mean CPU.

The [eight-bit dispatch](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/0.50.2/bitsandbytes/autograd/_functions.py) uses `MatMul8bitLt` on MPS, including activation quantization and outlier handling. The [default integer multiply implementation](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/0.50.2/bitsandbytes/backends/default/ops.py) actually multiplies float32 casts before returning integer accumulators. The inspected MPS file does not replace that operation with accelerated integer matrix multiplication. Thus working LLM.int8 inference is not evidence of a higher M4 integer compute ceiling. Quanto remains the initial recommendation to keep activation quantization out of the first comparison; bitsandbytes is a maintained alternative, especially for NormalFloat4 testing if four bits become necessary.

## Speed: checking the compute argument

The following is a roofline estimate, calculated here, not a laptop measurement. It uses the owner's approximately 8 billion parameters, 1,000 tokens, 273 GB/s bandwidth assumption, and the repository's [measured 5.7 trillion floating-point operations/s](../textdiff.md). A multiply-add counts as two operations.

```text
Dense weight projections: 2 × 8e9 × 1,000 = 16e12 operations per forward
Compute time floor:       16 / 5.7       = 2.81 seconds
One bf16 weight read:     16 GB / 273    = 0.0586 seconds
Ratio:                    2.81 / 0.0586  = 47.9
Arithmetic intensity:     1,000 operations/byte
Machine balance:          5.7e12 / 273e9 = 20.9 operations/byte
```

That approximately 48-fold ratio validates the *direction* of the claim under its assumptions. It is not a predicted speedup or an exact runtime. Parameter counts include embeddings, which are gathered rather than fully multiplied; output projection depends on which positions need logits. Attention adds work. Weight tiles can be reread, and the assumed bandwidth is an upper bound rather than measured sustained bandwidth.

For LLaDA's 32 layers and width 4,096, the two attention matrix products add approximately `4 × layers × length² × width = 0.524 TFLOP` at 1,000 tokens. Repeatedly evaluating a full sequence therefore resembles repeated prompt processing, not single-token autoregressive decoding. If 1,000 is prompt length, use **prompt plus generated canvas**, not 1,000, in the calculation. Classifier-free guidance may require two forwards.

Weight-only quantization reduces stored weights while activations remain floating point. Quanto's generic MPS 8-bit path reconstructs floating weights before multiplication, so it cannot claim an integer compute ceiling. A complete float copy cached for every layer would erase much of the memory benefit; inspect peak memory across repeated forwards.

There is direct M-series evidence against assuming constant or faster throughput. An [upstream MLX issue](https://github.com/ml-explore/mlx/issues/4621) reports the following on an M2 Max, MLX 0.32.3, with a 12,288 × 4,096 projection and group size 64:

| Input rows | 4-bit quantized matmul | Dequantize then bf16 matmul | 8-bit quantized matmul | Dequantize then bf16 matmul |
| --- | --- | --- | --- | --- |
| 512 | 7.63 ms | 6.00 ms | 7.46 ms | 6.52 ms |
| 2,048 | 29.81 ms | 21.82 ms | 29.17 ms | 22.13 ms |
| 8,192 | 118.03 ms | 83.64 ms | 115.44 ms | 84.54 ms |

A November 3 comment reports reproduction on an M4 Max. This is a contributor's reproducible benchmark, not an independently repeated M4 Pro measurement. It compares quantized kernels with reconstruction plus float multiplication, not two different-quality checkpoints. It suggests possible slowdown at our sequence lengths, but supplies no exact 1,000-token team-generation penalty.

A newer [MLX integer matmul proposal](https://github.com/ml-explore/mlx/pull/4722) reports approximately 1.35–1.53-fold prefill gains for tested shapes using staged 8-bit computation on **M5/M6 matrix hardware**. Its dispatch description excludes older non-matrix-unit GPUs, and activation quantization introduces an additional quality change. Do not apply those gains to the M4 Pro. The [associated issue](https://github.com/ml-explore/mlx/issues/4627) explains the hardware restriction.

Apple's [Core ML documentation](https://apple.github.io/coremltools/docs-guides/source/opt-quantization-overview.html) does identify an M4 compute-bound acceleration path: quantize weights **and activations** to 8-bit for optimized Neural Engine compute. This requires a successful Core ML port and measured placement. No source found establishes a generally available M4 Metal GPU 8-bit path that increases the compute ceiling for these denoisers.

**Planning assumption:** buy memory with 8-bit weights, budget for equal or slower forward passes, and measure actual elapsed time. Avoid swapping, which can make quantization faster indirectly. Neither that effect nor a different backend's speed proves low-bit arithmetic is faster.

## Quality: diffusion-specific evidence

Post-training quantization means compressing an already trained model. Below, `W4A16` means 4-bit weights and 16-bit activations; `W8A8` means both use 8 bits. Those settings must not be pooled into one “4-bit” or “8-bit” quality claim.

### Weight-only versus activation quantization

The full [Quantization Meets dLLMs paper, revision 3](https://arxiv.org/pdf/2508.14896), especially Tables 1–5 and Appendix A, evaluates GPTQ, activation-aware weight quantization called AWQ, SmoothQuant, and rotation methods on LLaDA and Dream. Examples:

- LLaDA-Instruct commonsense mean: 65.7% unquantized, 66.0% with 4-bit GPTQ.
- Dream mean over four reported tasks: 71.2% unquantized, 70.6% with 4-bit GPTQ, 68.2% with 4-bit AWQ.
- LLaDA-Instruct under SmoothQuant with 8-bit weights and activations: 67.1% to 67.0% on a different five-task mean. With both at 4 bits: 47.7%.

These are task accuracies, not Showdown legality. The paper finds greater sensitivity on mathematics/code and worse results at three bits. Its evaluation/calibration settings differ by model and task; Appendix A does not justify claiming every calibration sample follows an actual masked trajectory. Its Section 3.1 also describes Gaussian masking, which does not correctly define absorbing-mask text diffusion. Use its measured tables cautiously, not that description.

### Error across denoising steps

The full [DLLMQuant paper](https://arxiv.org/pdf/2508.14090), Table 1, evaluates **4-bit weights and activations**. LLaDA's nine-task mean falls from 59.87 to 48.09 with its AWQ baseline; the proposed rotation-based variant reaches 54.29. LLaDA-1.5 and Dream are also evaluated.

The method samples calibration states across denoising steps and mask ratios, weights reconstruction by token certainty/mask state, and changes attention activation quantization. Figures 1–2 and the analysis connect changing activation distributions and repeated prediction to error propagation. This supports trajectory-aware calibration. It does not show that a four-bit weight-only GGUF will lose the same amount, nor that every diffusion model compounds error more than an architecture-matched autoregressive model. Both generation families feed predictions back into later inputs.

More steps can change token confidence and correct some outputs; they are not a guaranteed cure. Preserve the exact sampler and number of forwards when comparing precision.

### Runtime-aware results and the smaller masked-diffusion family

The full [2026 masked-diffusion storage/runtime study](https://aclanthology.org/2026.emnlp-industry.241.pdf), Tables 1, 15–17, 41, 46, and 48, provides a useful warning. Its LLaDA AWQ-4 quality baseline reconstructs float weights and still peaks around 15.6 GiB. The integrated NormalFloat4 bitsandbytes path peaks around 6 GiB. Those are NVIDIA results, not Apple memory estimates.

Its Dream transfer loses 7.2 percentage points on MMLU and 9 points on GSM8K with AWQ-4, whereas its LLaDA changes are smaller. AWQ here uses a scaling fallback rather than the deployment kernel from AutoAWQ, so the method label alone does not make results interchangeable with the earlier study. For the 170-million-parameter OpenWebText masked-diffusion model, AWQ-4 changes masked-token negative log likelihood from 2.236 to 2.248; uniform four-bit gives 2.492. That single-mask-ratio result does not measure full-generation quality. The paper's smaller diagnostic slices and inconsistent trajectory-summary wording warrant checking logs before using a diagnostic as a universal rule.

### What each method buys this project

| Method | Diffusion-specific evidence located | Consequence |
| --- | --- | --- |
| GPTQ, a reconstruction method using approximate curvature | Direct LLaDA/Dream measurements in Quantization Meets dLLMs | Credible four-bit quality candidate, but its calibration and packed format need a compatible deployment backend. |
| AWQ, activation-aware weight quantization | Both studies above, with materially different recipes/results | Calibrate on the fine-tuned domain and verify the implementation. A name alone predicts neither quality nor live memory. |
| SmoothQuant, which moves activation scale variation into weights | Direct 8-bit and 4-bit weight/activation tests | Eight-bit results are encouraging. This does not create an M4 GPU integer kernel. |
| HQQ, Half-Quadratic Quantization | General [implementation](https://github.com/dropbox/hqq); torchao's experimental MPS API exposes an HQQ option | No controlled diffusion-specific HQQ accuracy study was verified in this search. Treat its quality benefit here as unverified. |
| Simple 8-bit round-and-scale | Supported by Quanto | Lowest integration burden. No exact published legal-team loss or direct eight-bit weight-only result for our fine-tuned checkpoints was located. |
| Simple four-bit or GGUF mixed four-bit | Supported formats, distinct from GPTQ/AWQ | Cannot borrow GPTQ's accuracy numbers. Needs its own domain evaluation. |

This is a scoped evidence review, not a claim to exhaust every October 2026 preprint.

Expect eight-bit weight-only quantization to be less risky than aggressive weight-and-activation quantization, as an engineering inference. There is no defensible fixed percentage loss for legality, diversity, or battle strength before testing the actual fine-tuned checkpoint.

For a maintained alternative smoke test in another isolated environment, install `bitsandbytes==0.50.2` against the existing PyTorch pins, then run `pip check`. On macOS 26 or newer, install `kernels` and verify whether its optional Hub kernel actually loads. Use `Linear8bitLt(..., has_fp16_weights=False)` for a tiny eight-bit device test before using the corresponding Transformers integration. If four bits become necessary, test `Linear4bit(..., compute_dtype=torch.bfloat16, quant_type="nf4")`. Check serialization and the supported model loader separately. These are unrun integration checks, not claims that a generic causal-language-model loading example handles every remote-code checkpoint.

## Memory at approximately 1,000 tokens

All numbers in this subsection are **calculated storage or allocation estimates**, except the explicitly cited NVIDIA peaks above. GB means decimal gigabytes; GiB means 1,073,741,824 bytes. None is a measured M4 Pro peak.

Raw parameter storage, using the candidate sizes from the task:

| Candidate | bf16, 2 bytes/parameter | Raw 8-bit | Raw 4-bit |
| --- | --- | --- | --- |
| DiffuLLaMA, 6.7 billion | 13.40 GB | 6.70 GB | 3.35 GB |
| Dream Base, 7.6 billion | 15.20 GB | 7.60 GB | 3.80 GB |
| LLaDA Base, 8 billion | 16.00 GB | 8.00 GB | 4.00 GB |

Actual storage includes scales, zero points, unquantized modules, and alignment. Keeping one billion parameters in bf16 while quantizing the rest gives a LLaDA illustration of **9.0 GB at eight bits** or **5.5 GB at four bits**, before quantization metadata. For groups of 64 with two bf16 metadata values per group, add 0.0625 bytes per quantized parameter. Different formats use different metadata. GGUF `Q8_0` uses 32 values plus a two-byte scale, or 1.0625 bytes per value, before other tensors. Inspect the actual converted file; do not label mixed `Q4_K_M` as exactly four bits/parameter.

The [LLaDA configuration](https://huggingface.co/GSAI-ML/LLaDA-8B-Base/blob/main/config.json) has width 4,096, 32 attention heads, 32 layers, vocabulary 126,464, and untied embeddings. The [Dream configuration](https://huggingface.co/Dream-org/Dream-v0-Base-7B/blob/main/config.json) has width 3,584, 28 attention heads/layers, vocabulary 152,064, and untied embeddings.

For batch one and total sequence length 1,000, the allocation arithmetic is:

| Allocation | LLaDA | Dream | Precision assumption |
| --- | --- | --- | --- |
| Full-sequence logits | 0.253 GB | 0.304 GB | bf16 |
| Full-sequence logits or probabilities | 0.506 GB | 0.608 GB | float32 |
| One dense attention score tensor | 0.064 GB | 0.056 GB | bf16 |
| One dense attention score tensor | 0.128 GB | 0.112 GB | float32 |
| One sequence hidden state | 0.0082 GB | 0.0072 GB | bf16 |

These use `batch × length × vocabulary × bytes` for logits and `batch × heads × length² × bytes` for scores. Multiple logits/probability/noise/sort buffers can coexist. Eager attention may hold scores and probabilities together. Fused scaled-dot-product attention can avoid materializing the full score matrix. In inference mode, temporary tensors normally get reused layer by layer; multiplying every activation by all layers would overestimate peak unless outputs or caches are retained.

The [#170 preparation comment](https://github.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/issues/170#issuecomment-6668208646) calls out Qwen's `logits_to_keep`. Project only the output positions the diffusion sampler actually needs. `logits_to_keep=1` is generally wrong for simultaneous masked-token prediction. A suffix selection can remove prompt logits; interior masks may require indexing hidden states before the output head. Verify shifted-logit alignment. The Qwen option is not proof that LLaDA/Dream expose the same argument.

At 1,000 tokens, logits are substantial but not tens of gigabytes. If the prompt is 1,000 and the output canvas adds 512, multiply logits by 1.512 and dense attention by approximately 2.286. Classifier-free guidance batched as two examples doubles these terms. Ordinary autoregressive key/value caching cannot simply be assumed exact under fully bidirectional repeated updates.

**Capacity expectation, unverified:** frozen eight-bit inference is plausible on 24 GB if it retains only quantized weights and per-layer reconstruction temporaries. Four-bit should leave more room. There is no honest single “real peak” to report before loading the exact checkpoint and sampler. Check loading/conversion peak separately; loading a full bf16 checkpoint and then quantizing it can fail even when quantized inference would fit. Prepare artifacts on Modal or load/replace tensors incrementally if necessary.

## Concrete integration path

Use a separate environment so the current runtime remains reproducible. These are proposed install and smoke-test steps, not changes performed by this research task.

1. Locate and run the current job-status script before GPU work. The requested script was absent here. Do not substitute an empty process search for a working queue/lease check.
2. Record the current runtime's package versions. Create an environment with the same PyTorch and Transformers versions, then install Quanto without allowing dependency changes.
3. Pin the exact successful Quanto version in the benchmark record. Test a small linear layer before loading a candidate.

```bash
RUNTIME="$HOME/.local/share/vgc-pilot-runtime"
"$RUNTIME/venv-dlm/bin/python" -m pip freeze > dlm-before.txt
python3.11 -m venv "$RUNTIME/venv-dlm-quanto"
"$RUNTIME/venv-dlm-quanto/bin/python" -m pip install -r dlm-before.txt
"$RUNTIME/venv-dlm-quanto/bin/python" -m pip install --no-deps optimum-quanto
"$RUNTIME/venv-dlm-quanto/bin/python" -m pip check
```

If `pip check` reports missing dependencies, resolve them against the existing pins before continuing. Run the following with `PYTORCH_ENABLE_MPS_FALLBACK=0` set **before importing PyTorch**, after the status check:

```python
import torch
from optimum.quanto import freeze, qint8, quantize

assert torch.backends.mps.is_available()
model = torch.nn.Sequential(torch.nn.Linear(256, 256)).eval()
model = model.to(dtype=torch.bfloat16)  # CPU conversion avoids a full GPU staging copy.
quantize(model, weights=qint8, activations=None)
freeze(model)
model = model.to("mps")
x = torch.randn(1, 64, 256, dtype=torch.bfloat16, device="mps")
with torch.inference_mode():
    y = model(x)
torch.mps.synchronize()
assert y.device.type == "mps"
assert torch.isfinite(y).all().item()
print(y.shape, torch.mps.current_allocated_memory(),
      torch.mps.driver_allocated_memory())
```

For a candidate, use its existing loader with pinned remote-code revision; do not replace the diffusion model with `AutoModelForCausalLM.generate`. Merge the fine-tuned adapter into bf16 weights on Modal, then quantize the merged model. Replace only supported modules, call `freeze`, retain the existing sampler, and inspect coverage by parameter count. Keep activations, normalization, and attention in the validated floating-point type. Whether to keep the output head in bf16 is an explicit quality/memory trade-off; log the exclusion.

For four-bit fallback, rerun the same process from the merged bf16 checkpoint with `qint4`. Do not quantize an already eight-bit artifact. Save both the state and Quanto's `quantization_map`; the [official save/reload recipe](https://github.com/huggingface/optimum-quanto#quantize-a-model) uses `requantize` for reconstruction. Round-trip the artifact and check output agreement before timing it. Local package/model compatibility is still unverified.

### Alternative for Dream and LLaDA

Use upstream llama.cpp's diffusion runner if Quanto is too slow, fails, or retains excess memory. Build rather than assume a packaged binary includes the example:

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
git rev-parse HEAD
cmake -B build -DGGML_METAL=ON -DLLAMA_BUILD_EXAMPLES=ON
cmake --build build --config Release --target llama-diffusion-cli llama-quantize -j 4
python3 -m venv .venv-convert
.venv-convert/bin/python -m pip install -r requirements.txt
.venv-convert/bin/python convert_hf_to_gguf.py /path/to/merged-checkpoint \
    --outfile model-bf16.gguf --outtype bf16
build/bin/llama-quantize model-bf16.gguf model-q8.gguf Q8_0
```

Conversion of the large checkpoint belongs on the machine already holding the merged bf16 weights; do not download it as part of a smoke test. Build/install guidance comes from [upstream build documentation](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md) and [converter arguments](https://github.com/ggml-org/llama.cpp/blob/master/convert_hf_to_gguf.py). Pin the source commit in the benchmark.

The [diffusion CLI source](https://github.com/ggml-org/llama.cpp/blob/master/examples/diffusion/diffusion-cli.cpp) uses `-ub` as total canvas length, including prompt, and reads a shifted-logit metadata flag. Use `-ngl 99` and verify Metal layer offload in logs. Set `-ub`, `-b`, and `-c` to cover the whole sequence. Dream's documented time schedule uses `--diffusion-eps`; LLaDA's block schedule uses `--diffusion-block-length`. They are mutually exclusive in the source. A four-bit fallback can use `Q4_K_M`, with its mixed-bit composition recorded.

This CLI is a working starting point, not a drop-in substitute for arbitrary interior infilling or classifier-free guidance. Port and compare the project's prompt, masks, guidance, remasking, stopping, and sampling rules before using it for #170. Dream and LLaDA support does not establish DiffuLLaMA support.

## What benchmark issue #170 should measure

Before running, fix the [known sampler setup](https://github.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/issues/170#issuecomment-6668208646): float32 time schedules, explicit temperature and lengths, end-token trimming, package versions, and the supported masked-Qwen samplers. Keep generation and battles sequential as decided.

Use the same prompt/task set and fine-tuned weights across precision comparisons. Suggested reader names and code identifiers:

| Reader name | Code identifier | Purpose |
| --- | --- | --- |
| Bf16 weights | `weights_bf16` | Reference when it fits; otherwise generate on Modal |
| Eight-bit weights | `weights_int8` | First memory fallback |
| Four-bit weights | `weights_int4` | Fallback only after eight-bit exceeds the ceiling |

All three keep the same floating-point activations and sampler settings. These are proposed definitions, not existing results.

For a small compatibility check, run one short forward and one complete denoising sample. For timing, warm up separately, synchronize before and after the timed interval, then use several prompts and repetitions at the actual total length. Report:

- Forward seconds at fixed batch and sequence length, denoiser evaluations per sample, end-to-end seconds per attempt, and median/tail latency.
- **Seconds per legal team = total elapsed generation/validation time divided by legal teams.** Count failed and rejected attempts. Zero legal teams means no finite estimate, not zero cost.
- Legal share with a binomial confidence interval, parser failure, six-slot completion, clause violations, and truncated outputs.
- Distinct species sets, duplicate teams, copied training teams, and later battle strength with the existing evaluator. A more legal generator can still collapse diversity.
- Load peak, warmed inference peak, process resident memory, GPU driver allocation, and system swap change. PyTorch exposes [current](https://docs.pytorch.org/docs/stable/generated/torch.mps.current_allocated_memory.html) and [driver](https://docs.pytorch.org/docs/stable/generated/torch.mps.driver_allocated_memory.html) allocations, not a substitute for a sampled peak. Synchronize and sample each denoising step; include sampling temporaries. Do not add resident memory to unified GPU allocation as though they were disjoint.
- Checkpoint, adapter, tokenizer, remote-code, backend, and package revisions; total prompt/output length; quantization coverage, group size, excluded modules, effective stored bytes; fallback/offload logs.
- Estimated grid hours = `4,416 × seconds_per_legal_team / 3,600`. Separate cold load/conversion from amortized generation. Do not extrapolate from one successful team.

Test the roofline claim directly with matched projection shapes and 1,000 total tokens, then the real forward. Compare dense bf16, eight-bit weight-only, and four-bit only when required. If full bf16 cannot fit, a representative single-layer comparison isolates kernel cost but does not establish whole-model speed. Do not interpret a backend/sampler switch as the causal effect of quantization.

## Estimating loss without laptop bf16

The strongest comparison is **generation only on Modal**, with the same merged checkpoint in bf16. This is a proposed follow-up, not a cloud job started by this report.

First separate quantization error from backend error. On a small shared set of saved masked states at early, middle, and late denoising steps, compare the original and quantized-reconstructed weights in the same framework on Modal. Then compare that quantized reference with the laptop backend. Save mask states, timestep, tokenizer ids, logits at needed positions, confidence-ranked fill decisions, and all sampler settings. This catches wrong masks or shifted logits that might otherwise look like quantization loss.

Measure masked-token cross-entropy, top-token agreement, probability divergence, and which positions are filled. Then generate from a fixed prompt/seed list with both deployed precisions. Equal seeds across frameworks do not imply identical random draws; use paired prompts and enough independent draws. Report legality differences, diversity, and battle strength separately. Save every attempt before rejection.

A reasonable first screen is 100–200 attempts per precision. This detects large failures; it cannot prove a one-percentage-point legality loss is absent. For two independent legal-share estimates near 80%, about 500 samples per model still give an approximate 95% difference interval with five-percentage-point half-width. Calculate power around the owner's eventual practical loss margin before promising equivalence.

If cloud bf16 generation is unavailable, report absolute legality and speed for the deployed model and compare eight-bit with four-bit if both exist. The small bf16 masked-Qwen models can check the harness but cannot bound an eight-billion-parameter model's quantization loss. Teacher-forced masked-token diagnostics help locate damage; they do not replace free-generation legality or battle quality. Label the bf16-relative quality cost **unknown**, not “negligible.”
