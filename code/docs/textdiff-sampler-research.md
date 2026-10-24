# Which sampler for the block-diffusion team writer (research, 2026-10-23)

Question: which decoding rule gives block-diffusion language models like Fast-dLLM v2 (blocks of 32 tokens, sub-blocks of
8, KV cache across blocks) their best output, and what does each rule do to the diversity of many samples of one prompt?
Our writer needs both: a legal team (about 50% now) and many different teams from one prompt.

Sources are primary (arXiv papers and the authors' repositories). Claims marked *(PDF)* were checked against the paper's
PDF text; the rest come from the arXiv HTML pages as read on 2026-10-23.

## What each paper decodes with

**Fast-dLLM v2** [[1]](#ref1). Within a block, every masked token whose confidence (top softmax probability) exceeds a
threshold is unmasked in parallel; the rest stay masked. When none passes, the most confident one is unmasked (stated
for v1 [[2]](#ref2); the released v2 `generate` does the same). Defaults: block 32, sub-block 8 *(PDF)*; the GSM8K
speed comparison uses threshold 0.9, and threshold 1.0 "recovers the standard non-parallel decoding process". All
evaluations use greedy decoding (argmax) *(PDF)*. Sub-block size against accuracy *(PDF, Table 3)*:

| sub-block | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|
| GSM8K | 62.8 | 61.8 | 62.0 | 61.3 | 60.2 |
| HumanEval | 42.7 | 43.3 | 43.9 | 39.6 | 38.4 |

The sub-block cache gives "negligible gains when the batch size is small" and a substantial speed-up in the compute-bound
regime, with no observable effect on accuracy. Diversity is not discussed.

**Fast-dLLM v1** [[2]](#ref2) introduced the threshold rule (confidence = maximum softmax probability), default 0.9 with
block size 32. Its repository's table shows cache + parallel decoding at about the same GSM8K accuracy as plain LLaDA
(78.5 vs 79.3 at length 256) at 8.1× the tokens per second.

**LLaDA** [[3]](#ref3). Two remasking rules: random, and low-confidence (remask the least confident predictions; the
predictions themselves are argmax). Low-confidence is the default and wins everywhere *(PDF, Table 9)*: BBH 45.0 vs 32.1,
GSM8K 70.0 vs 21.3, Math 30.3 vs 9.2, HumanEval 32.9 vs 11.6, MBPP 40.2 vs 21.0. The authors motivate it by the
"annealing tricks" of LLM sampling and do not measure diversity.

**Dream 7B** [[4]](#ref4). The repository offers four unmasking orders: `origin` (random, the default, which "may
degrade performance in some tasks"), `maskgit_plus` (top-1 confidence), `topk_margin` (top-1 minus top-2 probability) and
`entropy`. Its example call uses `alg="entropy"`, temperature 0.2, top-p 0.95. No quality comparison between the orders is
published.

**Margin ordering, Kim et al.** [[5]](#ref5). Choosing which positions to unmask by the margin between the two most
likely values beats random order on constraint puzzles *(PDF, Table 2)*: Sudoku 89.49% vs 6.88% (top probability alone
18.51%). On text they note that margin or top-probability ordering "often leads to greedy sampling, which harms the
diversity (entropy)", and add a noise term to the margin *(PDF, Appendix D.1.2)*.

**Entropy-bounded unmasking (EB-sampler)** [[6]](#ref6). Picks how many tokens to unmask per step: the largest set whose
summed entropy minus its largest entropy stays below γ. About 2-3× fewer model calls at equal accuracy on LLaDA and Dream
coding and math benchmarks, with zero-temperature sampling. It is a speed rule; it reports nothing on diversity and does
not use a KV cache.

**Remasking samplers (ReMDM)** [[7]](#ref7), NeurIPS 2025. Lets already-decoded tokens be remasked so later steps can fix
them; quality (MAUVE) rises with more steps. It needs many more steps than tokens to pay off, and the paper states nothing
about block decoding or a KV cache.

**Block diffusion (BD3-LM)** [[8]](#ref8), ICLR 2025 Oral. Samples with nucleus sampling, p = 0.9 *(PDF, Appendix)*, and
caches keys and values across blocks.

**Sampling precision (Zheng et al.)** [[9]](#ref9), ICLR 2025. Gumbel-max categorical sampling in 32-bit floats truncates
the noise, which "lowers the effective temperature": on their text benchmark 64-bit sampling raised token entropy from
5.17 to 5.66. ReMDM samples in 64-bit for this reason. Our writer samples with `torch.multinomial`, not Gumbel noise, but the
released Fast-dLLM v2 `generate` takes its softmax over bf16 logits, a coarser version of the same distortion;
`generate_batch` upcasts to fp32 unless `exact=True`.

## What this means for the team writer

1. **The standard dLLM decoder is greedy with confidence-threshold unmasking** (Fast-dLLM v1/v2, LLaDA, EB-sampler). It is
   what the papers' quality numbers come from. For one fixed prompt it is deterministic: it writes the same team every
   time, so it cannot fill a pool of different teams. It is worth one sample per prompt as a quality reference only.
2. **For many samples, the common choice is temperature with nucleus (top-p) sampling** under the same confidence order:
   p = 0.9 in BD3-LM, temperature 0.2 / top-p 0.95 in Dream's example. Our writer uses temperature 1 and top-p 1. That is
   the most diverse setting, with no pruning of the tail where invented formes like `Alcremie-Mega` live.
3. **Unmasking order:** we already use confidence order. Margin order is the strongest result on constraint puzzles, and a
   legal team is a constraint problem, but it reduces diversity unless noise is added. It is an experiment, not a
   default.
4. **Keep sampling arithmetic in fp32** even when the model runs in bf16 or a lower type (Zheng et al.).
5. **Not for us now:** ReMDM (needs far more steps than tokens, no block/KV statement) and EB-sampler (a speed rule; in a
   batch, steps already wait for the least confident row, measured 2026-10-23).

Arms to measure (legality, distinct species sets, then win rate): greedy (one team, the reference); temperature 1 / top-p 1
(current); temperature 1 / top-p 0.9 (BD3-LM); temperature 0.7 / top-p 0.9 (the scratch model's best decoding,
2026-08-27).

## References

<a id="ref1"></a>1. 🟢 Wu, Zhang, Xue et al. Fast-dLLM v2: Efficient Block-Diffusion LLM. arXiv 2509.26328 (no venue on arXiv;
the repository's news says ICLR 2026). https://arxiv.org/abs/2509.26328 · code: https://github.com/NVlabs/Fast-dLLM

<a id="ref2"></a>2. 🟢 Wu, Zhang, Xue et al. Fast-dLLM: Training-free Acceleration of Diffusion LLM by Enabling KV Cache and
Parallel Decoding. arXiv 2505.22618 (no venue on arXiv; the repository says ICLR 2026). https://arxiv.org/abs/2505.22618

<a id="ref3"></a>3. 🟢 Nie, Zhu, You et al. Large Language Diffusion Models. arXiv 2502.09992 (no venue stated on arXiv).
https://arxiv.org/abs/2502.09992

<a id="ref4"></a>4. 🟢 Ye, Xie, Zheng et al. Dream 7B: Diffusion Large Language Models. arXiv 2508.15487 (no venue stated).
https://arxiv.org/abs/2508.15487 · code: https://github.com/HKUNLP/Dream

<a id="ref5"></a>5. 🟢 Kim, Shah, Kontonis et al. Train for the Worst, Plan for the Best: Understanding Token Ordering in Masked
Diffusions. arXiv 2502.06768 (no venue stated on arXiv). https://arxiv.org/abs/2502.06768

<a id="ref6"></a>6. 🟢 Ben-Hamu, Gat, Severo et al. Accelerated Sampling from Masked Diffusion Models via Entropy Bounded
Unmasking. arXiv 2505.24857 (no venue stated). https://arxiv.org/abs/2505.24857

<a id="ref7"></a>7. 🟢 Wang, Schiff, Sahoo, Kuleshov. Remasking Discrete Diffusion Models with Inference-Time Scaling. NeurIPS
2025 (arXiv comments). https://arxiv.org/abs/2503.00307

<a id="ref8"></a>8. 🟢 Arriola, Gokaslan, Chiu et al. Block Diffusion: Interpolating Between Autoregressive and Diffusion
Language Models. ICLR 2025 Oral (arXiv comments). https://arxiv.org/abs/2503.09573

<a id="ref9"></a>9. 🟢 Zheng, Chen, Mao et al. Masked Diffusion Models are Secretly Time-Agnostic Masked Models and Exploit
Inaccurate Categorical Sampling. ICLR 2025 (arXiv comments). https://arxiv.org/abs/2409.02908
