# Each masked model's own sampler, and what later papers ran on it (2026-11-18)

A **sampler** is the loop that picks which masked positions to fill at each step and with which tokens. Every masked model in this project runs one shared sampler (`fill` in `code/scripts/masked_diffusion_memory_bench.py`). It fills the most confident masks first, keeps exactly a linear count of masks after each step, draws tokens at temperature 1.0 with no top-p, fills the whole text at once, and uses 128 steps. On the #170 benchmark text that is about 2.7 tokens per step, because each team has about 350 hidden tokens. The right shift and DiffuLLaMA's start token follow how each model was trained (#236). This note asks what each model's own authors sample with, and what later papers that ran the same released checkpoint sampled with. It covers the three fine-tune picks in depth: DiffuGPT small, DiffuGPT medium and Masked Qwen3 0.6B.

**Figure:** [each model's own sampler against ours, setting by setting](figures/samplers-by-model-2026-11-18/sampler-settings.html), built by `code/scripts/sampler_by_model_figures.py`.

**What this means for the tickets.** The owner's decision (2026-11-20): each model gets the sampler its own authors evaluate it with. For DiffuGPT that is the evaluation sampler below, not the standalone script; for the masked Qwen models it is the dLLM evaluation script's setting, block length 256.

- **DiffuGPT, small and medium.** The authors' evaluation sampler already matches ours in fill order, temperature, top-p and shift. It differs in two ways. It runs one step per masked token. And it can leave fewer masks than the linear count after a step, because already-filled tokens also compete for the stay-masked slots. Their standalone script is the real alternative: random order, temperature 0.95, top-p 0.9, 64 steps.
- **Masked Qwen3 0.6B.** The authors' sampler matches ours in order, schedule and shift. It differs in temperature (0, so every draw is the same text), in block length (64 on the model card, 256 in the evaluation script) and in one step per token.
- **Gains from other samplers.** No later paper shows a sampler gain on any of these three checkpoints that it also checked against diversity.

## How the evidence was gathered

GPT-6.1 Sol at high effort wrote a first pass over seven models ([note at 9a13cdd7](https://github.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/blob/9a13cdd7f4a072cf385e389673e55d1878f04ae1/notes/research_notes/masked-text-diffusion-decoding-by-model-2026-11-18.md)). A fresh Opus 5.5 agent then tried to refute it from the full sources. It checked 55 claims: 45 held, 5 were overstated, 3 were wrong and 2 could not be checked. It also found five papers the first pass had missed ([refutation at 6eb21c05](https://github.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/blob/6eb21c051e59e8fa24d60b1c6b5ec8a3482bdd00/notes/research_notes/masked-text-diffusion-decoding-by-model-2026-11-18-refutation.md)). This note keeps only what survived. Opus re-read the DiffuGPT confidence sampler (`trainer.py` lines 357–465 and `workflow.py` lines 85–103 at `c17e897f`) and UNIFUSION's step-sweep table myself. Both pinned notes stay on branch `research/masked-diffusion-decoding-by-model` and hold every line-level citation and the Dream, LLaDA and DiffuLLaMA detail this note leaves out.

## DiffuGPT small and medium

### The authors' samplers

The DiffuGPT paper describes two samplers. Algorithm 2 is random ancestral sampling for unconditional text: each step reveals masks at random. Section 3.4 adds that "for conditional generation tasks, we improve the sampling procedure to ensure that only tokens with high probabilities from neural networks are denoised", so its conditional results use confidence-first filling. Algorithm 1 is the adaptation training procedure, not a sampler; the 2026-11-16 note said otherwise. [Paper v3, Section 3](https://arxiv.org/html/2410.17891v3#S3)

The official repository has three code paths at [`c17e897f`](https://github.com/HKUNLP/DiffuLLaMA/tree/c17e897f6476c174b4623da594e4c65554f1613d).

1. **Evaluation sampler** (`generate_samples` in `LLaMA-Factory/src/llamafactory/train/ddm/trainer.py`, lines 357–465). Each step draws a token for every masked position at the set temperature, scores each draw by its log probability, and keeps the lowest-scoring positions masked. No top-p is applied (the call is commented out), there are no blocks, and the logits are right-shifted. The ranking includes positions that are already committed, which are never remasked. When a committed token ranks low it takes up a "stay masked" slot, so *at most* the linear count of masks remains after each step, and often fewer. Ours keeps exactly the linear count. With `gsm=True` it takes the most likely token instead of drawing one. [trainer.py L357–465](https://github.com/HKUNLP/DiffuLLaMA/blob/c17e897f6476c174b4623da594e4c65554f1613d/LLaMA-Factory/src/llamafactory/train/ddm/trainer.py#L357-L465)
2. **Alternative random sampler** (`generate_samples_v2`, lines 467–566). It reveals each mask with probability 1/(t+1) and applies top-p 0.9, but it can't run: line 530 assigns `maskable_mask` inside the function, so line 496 raises `UnboundLocalError` on the first call. The training pipeline sends every dataset except GSM to this function, so that route is broken at this commit. [trainer.py L467–566](https://github.com/HKUNLP/DiffuLLaMA/blob/c17e897f6476c174b4623da594e4c65554f1613d/LLaMA-Factory/src/llamafactory/train/ddm/trainer.py#L467-L566), [workflow.py L85–103](https://github.com/HKUNLP/DiffuLLaMA/blob/c17e897f6476c174b4623da594e4c65554f1613d/LLaMA-Factory/src/llamafactory/train/ddm/workflow.py#L85-L103)
3. **Standalone script** (`inf_diffugpt.py` with `model.py`), the one the released checkpoints are meant to be run with. It reveals each mask with probability 1/(t+1), applies temperature 0.95 and then top-p 0.9, right-shifts, and never remasks. It runs 64 steps over 512 tokens for small and 1,024 for medium, which is 8 or 16 tokens per step. The `--shift` flag parses any value as true, so the shift can't be turned off from the command line. [inf_diffugpt.py L9–60](https://github.com/HKUNLP/DiffuLLaMA/blob/c17e897f6476c174b4623da594e4c65554f1613d/inf_diffugpt.py#L9-L60), [model.py L80–158](https://github.com/HKUNLP/DiffuLLaMA/blob/c17e897f6476c174b4623da594e4c65554f1613d/model.py#L80-L158)

The paper's own settings (Appendix B.3) set the step count to the generation length for generation tasks, and 64 steps for GSM. Unconditional samples use top-p 0.9 with temperature 1.0 (small) or 0.98 (medium). [Appendix B.3](https://arxiv.org/html/2410.17891v3#A2.SS3) The evaluation script can't confirm that step rule for the infilling tasks. `eval_Lambada` runs first and overwrites the shared `args.diffusion_steps` and `args.logits_temp = 1.0`. ROCStories and HumanEval then run with the last LAMBADA example's step count at temperature 1.0, not the command-line defaults of 32 steps at 0.95. [eval-diffugpt.py L46–73, L249–391](https://github.com/HKUNLP/DiffuLLaMA/blob/c17e897f6476c174b4623da594e4c65554f1613d/evaluation/eval-diffugpt.py#L46-L391) The repository's inference configuration files load local training checkpoints, not the released `diffugpt-m` or `diffullama`, so they aren't recipes for the released models. The model cards give no sampler settings.

### What later papers ran on these checkpoints

**Figure:** [more steps make DiffuGPT text more fluent and less varied](figures/samplers-by-model-2026-11-18/diffugpt-step-sweep.html), values from UNIFUSION's Appendix C.3 table.

| Paper | Checkpoint | Sampler | Result | What it shows for us |
|---|---|---|---|---|
| [UNIFUSION](https://arxiv.org/html/2607.24507v1) (Appendix C.3) | Released small and medium | The DiffuGPT paper's own reverse process at temperature 1.0, 16 to 256 steps, 1,024 unconditional samples per seed, five seeds | Perplexity falls from 494.5 to 117.3 (small) and from 536.0 to 90.6 (medium). Unigram entropy also falls, from 5.511 to 5.279 and from 5.923 to 5.647 nats | Steps trade fluency for variety on both checkpoints. This is the only fresh run of the released medium checkpoint, and the only source that measures fluency and variety together on either size |
| [Learnable Sampler Distillation, LSD+](https://arxiv.org/html/2509.19962) (Appendix E.3, Table 14) | Small | A learned step schedule and solver, distilled from the model's own many-step samples, at 16 to 128 model calls | Perplexity 117.32 → 53.95 at 16 calls, 37.16 → 22.25 at 128 | Lower perplexity at the same number of model calls, but diversity isn't reported and the baseline sampler isn't named. UNIFUSION shows that lower perplexity can come with less variety, so this isn't yet a sampler gain |
| [Any-subset speculative decoding, ASSD](https://arxiv.org/html/2504.20456) (Table 2) | Small, as a baseline | 32 or 64 steps; order and temperature not stated | ROCStories ROUGE 14.0/1.5/13.0 and 16.4/2.0/14.2 | A baseline only; the new method runs on a different model |
| [Path planning, P2](https://arxiv.org/html/2502.03540v5) (Table 3) | Small and medium, as quoted numbers | Not run on them | Quotes LAMBADA 45.0 and 60.5 | These are the DiffuGPT paper's own v1/v2 numbers, revised to 21.6 and 30.3 in v3 ([v1 Table 1](https://arxiv.org/html/2410.17891v1)). P2's medium GSM8K 52.6 is the WinoGrande cell copied into the wrong column; the paper's GSM8K is 61.8 in every version. SPMDM repeats the same 52.6 |
| [Token attribution over time](https://arxiv.org/html/2610.01177), [knowledge editing](https://arxiv.org/html/2606.03924) | Medium | 64 steps; not a sampler study | Interpretability and editing | No sampler comparison |

## Masked Qwen3 0.6B

### The authors' sampler

The dLLM paper converts Qwen3 0.6B into a masked diffusion model and drops the right shift, because "in our experiments this leads to performance degradation" (Section 4.2.2). It gives no per-task sampler settings. [dLLM paper §4.2.2](https://arxiv.org/html/2602.22661v1#S4.SS2.SSS2) Every official source fills the most confident masks first, keeps the linear mask count, has no top-p and never remasks a committed token. They disagree on block length and step count:

| Source | Tokens | Steps | Block length | Temperature |
|---|---|---|---|---|
| Model card example and command ([card L156–198](https://huggingface.co/dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1/blob/c8d24a3f4adaeef46881b450e1bf7d1005203bd7/README.md#L156-L198)) | 256 | 256 | 64 | 0 |
| Library default, `MDLMSamplerConfig` ([mdlm.py L16–31](https://github.com/ZHZisZZ/dllm/blob/ca176752fbceec49c6b4777a2c18ae88e4eb10ed/dllm/core/samplers/mdlm.py#L16-L31)) | 128 | 128 | 128 | 0 |
| Chat example ([chat.py L32–39](https://github.com/ZHZisZZ/dllm/blob/ca176752fbceec49c6b4777a2c18ae88e4eb10ed/examples/a2d/mdlm/chat.py#L32-L39)) | 128 | 128 | 32 | 0 |
| Evaluation script, generation tasks ([eval.sh L61–93](https://github.com/ZHZisZZ/dllm/blob/ca176752fbceec49c6b4777a2c18ae88e4eb10ed/examples/a2d/mdlm/eval.sh#L61-L93)) | 256 | 256 | 256 | 0 |

The evaluation entry point's docstring runs GSM8K with block length 32 as well. Each source fills one token per step. The paper's Table 3 scores (GSM8K 29.3, HumanEval 30.5) differ slightly from the card's (29.8, 30.5), and no source says which settings produced them.

### What later papers ran on this checkpoint

| Paper | Sampler | Result | What it shows for us |
|---|---|---|---|
| [Value-guided backtracking, VGB](https://arxiv.org/html/2606.28301v1) (§5, Table 1) | A verifier guides which masks to reveal; VGB can also remask committed tokens. Length 32, temperature 1, 16 rollouts' worth of compute | Avoiding a banned letter: base sampler 0.0%, best of 16 0.1%, verifier-guided reveal 100.0%, VGB 100.0% | The gain comes from the verifier guiding the reveal. Guided reveal without backtracking also reaches 100%, and backtracking adds only +0.14 on the quality score (5.86 against 5.72). All guided methods score below best-of-16 on quality (7.00). A Showdown validator could play the verifier's role |
| [Remasking under equal compute, Ca*re*](https://arxiv.org/html/2607.24763v1) (Table 21) | No remasking against high-entropy remasking, temperature 0.25, 256 steps | Perplexity 36.8 → 32.7, but MAUVE (closeness to human text) 0.691 → 0.476 | Remasking lowers perplexity *and* closeness to human text. Temperature explains most of the MAUVE differences (η² = 0.91). The paper says only "Tiny-A2D (Qwen-0.6B)", so it may be the block-diffusion model, not this one |
| [Risk-sensitive GRPO](https://arxiv.org/html/2610.00661) (Appendix C.1) | Block 32, 16 steps per block, temperature 0.8 in training and 1.0 in evaluation | Reinforcement learning raises GSM8K pass@1 from 40.70 to 44.57 | A training result, not a sampler test, but it is the one source that runs this checkpoint at a positive temperature |
| [Critique and revision feedback](https://aclanthology.org/2026.acl-srw.61.pdf) (Appendix B, Table 2) | Confidence order, 256 steps, block 16, temperature 0, plus the same model's critique and revision | Global constraint score 0.640 → 0.880; some checks get worse (bullet compliance 0.08 → 0.06) | A prompting loop, not a sampler. It loads `dllm-collection/…`, which may not be the same weights as `dllm-hub/…` |
| [TrajDLM](https://arxiv.org/html/2605.10020v2), [Hidden Messages](https://arxiv.org/html/2610.28905v1) | Left-to-right constrained filling; a single masked answer position | Training and task results | No sampler comparison |

## The other four models in brief

The Masked Qwen2.5 Coder 0.5B card matches the Masked Qwen3 card (256 tokens, 256 steps, block 64, temperature 0), and the only later paper that used it ([FINDERS](https://arxiv.org/html/2610.22400v1)) doesn't generate text. DiffuLLaMA's standalone script uses 64 steps, temperature 0.9, top-p 0.9 and the start token. P2's gain on DiffuLLaMA is measured against its own rerun baseline, which scores below the authors' numbers, and P2's Table 6 shows plain greedy decoding reaches most of the same ROCStories gain. Dream's code reveals masks in random order by default, but its evaluation script uses entropy order. LLaDA fills by confidence in blocks at temperature 0. The cache and speed-up papers on Dream and LLaDA report speed gains, sometimes with small accuracy changes in either direction. Pinned detail for all four is in the two branch notes linked above.

## Open items

- ~~Which block length the Masked Qwen3 sampler ticket should copy.~~ Settled 2026-11-20: 256, the evaluation script's, because each model gets its authors' evaluation sampler.
- At temperature 0 the authors' Qwen3 sampler writes the same team every time from a given mask, so it can't be compared with ours on diversity without a positive temperature. Risk-sensitive GRPO's 1.0 at evaluation is the only published positive temperature for this checkpoint.
- LSD+ publishes no adapter or configuration for DiffuGPT, so its sampler can't be reproduced as published.
- Whether Ca*re*'s "Tiny-A2D (Qwen-0.6B)" is the masked Qwen3 checkpoint.
- Every result above is on free text or benchmarks, not teams.

## Sources

- https://arxiv.org/html/2410.17891v3
- https://arxiv.org/html/2410.17891v1
- https://github.com/HKUNLP/DiffuLLaMA/tree/c17e897f6476c174b4623da594e4c65554f1613d
- https://arxiv.org/html/2602.22661v1
- https://huggingface.co/dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1/blob/c8d24a3f4adaeef46881b450e1bf7d1005203bd7/README.md
- https://github.com/ZHZisZZ/dllm/tree/ca176752fbceec49c6b4777a2c18ae88e4eb10ed
- https://arxiv.org/html/2607.24507v1
- https://arxiv.org/html/2509.19962
- https://arxiv.org/html/2504.20456
- https://arxiv.org/html/2502.03540v5
- https://arxiv.org/html/2610.01177
- https://arxiv.org/html/2606.03924
- https://arxiv.org/html/2606.28301v1
- https://arxiv.org/html/2607.24763v1
- https://arxiv.org/html/2610.00661
- https://aclanthology.org/2026.acl-srw.61.pdf
- https://arxiv.org/html/2605.10020v2
- https://arxiv.org/html/2610.28905v1
- https://arxiv.org/html/2610.22400v1
- https://github.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/blob/6eb21c051e59e8fa24d60b1c6b5ec8a3482bdd00/notes/research_notes/masked-text-diffusion-decoding-by-model-2026-11-18-refutation.md
