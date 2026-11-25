---
title: Inference batch size in masked text diffusion papers
date: 2026-11-25
tags: [research, masked-diffusion]
---

# Inference batch size in masked text diffusion papers

Question (owner, 2026-11-25): when papers compare methods for masked text diffusion models, what batch size do they use at inference, and does the batch change the results? It decides how [#443](https://github.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/issues/443) batches our in-place fills.

Read from primary sources: papers and the authors' code. Each claim names its file or section. "Not stated" means the pages read do not say. Claims marked *(summary)* come from a fetch tool's summary of a page's later part, not from the text itself, and need a check before anyone cites them.

## Answer

- **Benchmarks of quality do not batch different prompts.** The unconditional generators (SEDD, MDLM, ReMDM, MD4) sample fixed 1,024-token texts, so every row of a batch has the same shape and needs no padding. LLaDA's own evaluation generates one question at a time.
- **Batching different prompts is newer, added for speed.** LLaDA added padded batch inference in October 2025. Dream masks left padding out of attention. Fast-dLLM reports throughput up to batch 32 *(summary)*.
- **No source seeds each sample.** Every code base read uses one global seed, or none, and draws noise for the whole batch from one shared stream. None says batch size changes accuracy, and none tests it.
- **So batching is standard, but a fill that depends on its batch neighbours is not something the papers guard against.** They report averages over thousands of samples. Our paired tests compare one fill per mask across prompts, so #443 needs per-row seeds and per-row unmasking. That is our requirement, not the papers' practice.

## By source

| Source | Batch at inference | Different-length prompts in one batch | Seeding | Batch changes results? |
| --- | --- | --- | --- | --- |
| [SEDD](https://github.com/louaaron/Score-Entropy-Discrete-Diffusion) | `run_sample.py`: `--batch_size`, default 1; shape `(batch_size, 1024)` | No: unconditional, fixed 1,024 tokens | None set in `run_sample.py` | Not stated |
| [MDLM](https://github.com/kuleshov-group/mdlm) | `configs/config.yaml`: samples = GPUs × `loader.eval_batch_size` × `num_sample_batches` (2) | No: unconditional | `seed: 1`, one global seed | Not stated |
| [ReMDM](https://arxiv.org/abs/2503.00307) | Sec. 5.1.1: 5,000 samples per model and sampler for MAUVE; batch size not in the text read | No: wrapped to L = 1,024 | Not stated for these tables | Not stated |
| [MD4](https://arxiv.org/abs/2406.04329) | Text batch size not stated; text samples of 1,024 tokens (appendix) | No | Not stated | Not stated |
| [LLaDA](https://github.com/ML-GSAI/LLaDA) | `eval_llada.py`: generation one request at a time (`unsqueeze(0)`), temperature 0; likelihood `batch_size=32` copies of one sequence, `mc_num=128` | `generate.py` takes `attention_mask` and left padding, added in "batch inference support" (README news, 2025-10-27); iLLaDA asserts batch 1 | `set_seed(1234)` once; Gumbel noise from the global RNG with `torch.rand_like`, one draw for the whole batch | Not stated |
| [Dream](https://github.com/HKUNLP/Dream) | README gives no batch size; eval scripts not read | Yes: `generation_utils.py` turns the 2D mask into a 4D mask that hides padding as query and key, and counts position ids over real tokens only | Global torch RNG, no `generator` argument | Not stated. From the code (inference): confidence modes set tokens per step from the batch's average mask count, `mask_index.sum() / mask_index.shape[0]`, so one prompt's fill can depend on its batch |
| [DiffuGPT / DiffuLLaMA](https://arxiv.org/abs/2410.17891) | Fig. 4 measures decoding latency at batch size 1; README gives no eval batch size | Not stated | Not stated | Not stated |
| [Fast-dLLM](https://arxiv.org/abs/2505.22618) | README eval command `--batch_size 1`; Appendix C.5 batches 1 to 32 for throughput *(summary)* | Not stated | Not stated | Not stated; accuracy tables give no batch size |
| [BD3-LM](https://arxiv.org/abs/2503.09573) | Generation batch not stated in the part read | Training wraps texts to 1,024 with no padding (App. C.1) | Not stated | Not stated |
| [dllm](https://github.com/ZHZisZZ/dllm) | Eval batch not stated | Not stated | Not stated | Not stated |

## Training lengths that bound our prompts

Our prompts with the example team run to 1,024 tokens, so each model's training length matters as much as its batch.

| Model | Trained on texts of | Source |
| --- | --- | --- |
| DiffuGPT small | 512 tokens | [Gong et al. 2024](https://arxiv.org/abs/2410.17891), Table 4 |
| DiffuGPT medium | 1,024 tokens | same table |
| Masked Qwen3 0.6B (`Qwen3-0.6B-diffusion-mdlm-v0.1`) | 1,024 tokens (`--max_length 1024`) | [dllm `examples/a2d/README.md`](https://github.com/ZHZisZZ/dllm/blob/main/examples/a2d/README.md) |
| Masked Qwen2.5 Coder 0.5B | Not stated | same page lists it only in its results table |

On #239 a probe moved the same DiffuGPT small text later by 418 positions, and its legal fills fell from 21 to 0. See the [audit comment](https://github.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/issues/239).

## What it means for #443

- Batch as large as the GPU fits: this matches every speed measurement above.
- Keep a mask's fill independent of its batch, which no source does by default. Give each row its own noise from its mask seed, its own slots and its own unmasking count. Hide padding from attention as Dream does, and count position ids over real tokens.
- Test it the way the papers never do: a batch of N masks against N batch-1 fills, token for token.

## Not read

ReMDM Appendix D.1, the end of BD3-LM, Fast-dLLM Appendix C.5 verbatim, LLaDA `EVAL.md`, Dream's eval scripts, DiffuLLaMA's inference script.
