# Which sampler each masked text diffusion model used, and how it scored (2026-11-16)

Each model group's own paper and official code generate text with a sampler that departs from the default assumption in at least one way. LLaDA uses low-confidence remasking, a uniform (linear) step schedule, temperature 0, and fills the text block by block (`block_length`). DiffuGPT and DiffuLLaMA shift logits right by one position, a change inherited from autoregressive initialisation. Dream keeps the right shift, and its own code's default reveals masks in random order, not by confidence. The dLLM group's default MDLM sampler is closest to LLaDA, but its Tiny-A2D chatbot explicitly turns the right shift off. The table gives the sampler and the paper's reported result; cells marked "not extracted" were not reached in this pass.

## Comparison

| Model group | Sampler in the paper | Official code | Logit right shift | Temperature | Reported result (paper's own numbers) |
|---|---|---|---|---|---|
| LLaDA (8B base and instruct) | Low-confidence remasking, uniform timestep schedule, semi-autoregressive blocks | Yes, `generate.py` | No | 0 by default | Base-model ablation (Table 7) compares remasking choices; exact values not extracted |
| DiffuGPT and DiffuLLaMA (AR-initialised) | Algorithm 1, with right shift; steps set to the number of masked tokens for evaluation | Evaluation scripts only; sampler body in external `llamafactory` package (unconfirmed) | Yes | 1.0 on Lambada | Not extracted |
| Dream 7B | Paper keeps the AR shift (Section 4.1); unmasking order set by `alg`; the code's default `origin` reveals each mask at random with probability 1 − s/t, not by confidence | `generation_utils.py` in the model repo (checked at revision 6572adb5) | Yes, `torch.cat([logits[:,:1], logits[:,:-1]], 1)` | 0.0 by default; top_p off | Section 5.2 compares with LLaDA; values not verified, see Open items |
| dLLM MDLM sampler (library default) | Block-wise low-confidence remasking, linear schedule, one block per window | Yes, `dllm/core/samplers/mdlm.py` | No (default False) | 0.0 | Not a paper result; a library default |
| Masked Qwen3-0.6B (Tiny-A2D, MDLM and BD3LM) | Not found in indexed text | Checkpoints on Hugging Face | Explicitly not used | Not found | Not extracted |

## LLaDA

**Sampler.** The paper's Algorithm 5 is low-confidence remasking; Algorithm 4 is random remasking (Appendix A.3). Section 2.4 says "We employ uniformly distributed timesteps by default." The official `generate.py` uses `get_num_transfer_tokens`, which splits the masked count evenly across steps and adds the remainder to the earliest steps. Generation runs block by block (`block_length`), and committed tokens are never re-masked. At temperature 0 the Gumbel noise is off; when temperature is positive, noise is computed in float64.

**Performance.** Table 7 (base-model ablation over BBH, GSM8K, Math, HumanEval and MBPP) is where the remasking comparison belongs. The numeric comparison of random against low-confidence remasking on GSM8K was not located in the indexed text, so this note does not state a gap between them. Table 8 (instruct ablation) was truncated in the search output and is not reproduced here.

Sources: [LLaDA paper](https://arxiv.org/html/2502.09992v3), [official code, generate.py](https://raw.githubusercontent.com/ML-GSAI/LLaDA/main/generate.py), [repository](https://github.com/ML-GSAI/LLaDA).

## DiffuGPT and DiffuLLaMA

**Correction (2026-11-18):** Algorithm 1 is the adaptation training procedure, not the sampler, and the sampler body is in the repository after all (`LLaMA-Factory/src/llamafactory/train/ddm/trainer.py`). See `masked-text-diffusion-samplers-by-model-2026-11-18.md`.

**Sampler.** Algorithm 1 in Section 3.3 includes "Right shift logits by one position," so the model predicts the masked token from the position before it, as the autoregressive model did. Evaluation (`eval_diffugpt.py`, `eval_diffullama.py`) sets `diffusion_steps` to the masked-token count and `logits_temp` to 1.0 on Lambada. A commented-out alternative applies `top_p_logits(logits/0.8, p=0.8)` and categorical sampling. The sampler body is imported from external `llamafactory.train.ddm.trainer`, which is not in the repository, so its default shift and step handling are unconfirmed.

**Start token.** DiffuLLaMA's `inf_diffullama.py` puts the LLaMA-2 start token `<s>` before the text. Without it, DiffuLLaMA 7B filled nearly every hidden value of #236's copy test with newlines and copied 23 of 168; with it, it copied 168 of 168 (2026-11-17, `code/docs/diffullama-start-token.md`). Our DiffuLLaMA texts now start with `<s>`.

**Performance.** Appendix B.3 gives the decoding settings (LoRA rank 8; reasoning uses T=64; top-k 0.98 and top-p 0.9 for medium; top-k 1.0 and top-p 0.9 for small). Appendix C.1 (Figure 5) and Appendix C.3 (Table 6) hold the results; not extracted in this pass. Table 8 (HumanEval infilling) is also not extracted.

Sources: [DiffuGPT and DiffuLLaMA paper](https://arxiv.org/html/2410.17891v3), [repository](https://github.com/HKUNLP/DiffuLLaMA).

## Dream 7B

**Sampler.** Section 4.1 keeps the autoregressive shift. The decoding code ships in the model repo as `generation_utils.py`; this pass read it from the local Hugging Face module cache at revision `6572adb5535263e4d1a337b56942ba48b6dee2a9`, the revision the project pins:

- Defaults: `temperature=0.0`, `top_p=None`, `alg='origin'` (lines 101–109).
- Timesteps are `torch.linspace(1, eps, steps + 1)` (line 405), and every step shifts the logits right by one (line 412).
- `alg='origin'` (lines 421–426) reveals each still-masked position independently at random with probability `1 - s/t`; it does not rank positions by confidence.
- `maskgit_plus` (top probability), `topk_margin` (top-two margin) and `entropy` (negative entropy) rank positions by confidence and reveal `int(masks * (1 - s/t))` of them, rounded down, and every remaining mask on the last step (lines 428–436). `alg_temp > 0` samples the positions from a softmax over confidence instead of taking the top k.
- Revealed tokens are never remasked.

The paper's explicit decoding settings for its reported results (temperature, top-p, `alg`) were not found in the text searched.

**Performance.** Section 5.2 compares Dream with LLaDA on ARC-C, GSM8K, Countdown and Sudoku. The pass did not verify which number belongs to which model, so none is quoted here.

Sources: [Dream 7B paper](https://arxiv.org/html/2508.15487v1), [official repository](https://github.com/DreamLM/Dream), [model repo](https://huggingface.co/Dream-org/Dream-v0-Base-7B).

## dLLM library: MDLM sampler and Tiny-A2D models

**Sampler.** `mdlm.py` defaults are `block_size=128`, `steps=128`, `temperature=0.0`, `remasking="low_confidence"`, `stochastic_transfer=False`, `right_shift_logits=False`. The loop splits the generation into blocks, computes transfer counts with `get_num_transfer_tokens`, optionally right-shifts logits, takes the argmax after Gumbel noise, and commits the top-k positions in the current block. The library's Dream sampler defaults to `right_shift_logits=True`.

**Tiny-A2D (masked Qwen3-0.6B).** Section 4.2.2 trains Qwen3-0.6B under MDLM and BD3LM objectives with SFT only, and states "we do not apply the logits right shifting tricks as in prior work … because in our experiments this leads to performance degradation." The paper's inference settings (steps, block size, temperature, remasking) were not found in the indexed text. Training: 10 epochs, learning rate 1e-4, global batch 2048, cosine schedule, 64 A100 GPUs. Section 3.3 notes that diffusion LMs are sensitive to inference hyperparameters (Figure 5 shows this for LLaDA-Instruct and Dream-Instruct).

Sources: [dLLM paper](https://arxiv.org/html/2602.22661), [code repository](https://github.com/ZHZisZZ/dllm), [MDLM sampler source](https://raw.githubusercontent.com/ZHZisZZ/dllm/main/dllm/core/samplers/mdlm.py), [MDLM checkpoint](https://huggingface.co/dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1), [BD3LM checkpoint](https://huggingface.co/dllm-hub/Qwen3-0.6B-diffusion-bd3lm-v0.1).

## Open items

- LLaDA: the random versus low-confidence GSM8K comparison and Table 8 values.
- DiffuGPT and DiffuLLaMA: Appendix C.1 and C.3 results, Table 8, and the `llamafactory` defaults.
- Dream: the paper's settings for its reported results, and the Section 5.2 values.
- Tiny-A2D: inference settings from the dLLM paper.
- Hugging Face model cards were not checked.

## Sources

- https://arxiv.org/html/2502.09992v3
- https://raw.githubusercontent.com/ML-GSAI/LLaDA/main/generate.py
- https://github.com/ML-GSAI/LLaDA
- https://arxiv.org/html/2410.17891v3
- https://github.com/HKUNLP/DiffuLLaMA
- https://arxiv.org/html/2508.15487v1
- https://github.com/DreamLM/Dream
- https://arxiv.org/html/2602.22661
- https://github.com/ZHZisZZ/dllm
- https://raw.githubusercontent.com/ZHZisZZ/dllm/main/dllm/core/samplers/mdlm.py
- https://huggingface.co/dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1
- https://huggingface.co/dllm-hub/Qwen3-0.6B-diffusion-bd3lm-v0.1
