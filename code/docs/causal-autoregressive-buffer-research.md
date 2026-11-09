# Causal autoregressive buffer (Hassan, Loka et al.): what the paper shows, and what it means for the position-embedding split

Read 2026-11-07 from the paper itself: "Efficient Autoregressive Inference for Transformer Probabilistic Models", Conor Hassan, Nasrulloh Loka, Cen-You Li, Daolang Huang, Paul E. Chang, Yang Yang, Francesco Silvestrin, Samuel Kaski, Luigi Acerbi. arXiv 2510.09477v2, 20 April 2026, CC BY 4.0. Source: the arXiv HTML version, https://arxiv.org/html/2510.09477. Section numbers below are the paper's. Labels: "Stated" means the paper says it. "Inference" means I derived it; the paper does not say it.

I cited this paper in the plan for the position-embedding split (`code/docs/position-embedding-split.md`, map #78) as the closest match to a web search. It is a weak match. Section 5 explains why.

## 1. The problem it solves

Stated (Sec. 1, 2). Neural processes, prior-fitted networks and tabular foundation models such as TabPFN and TabICL read a **context set** of observed input-output pairs. They predict each new target in one forward pass. That prediction is marginal: each target is predicted independently of the others. A joint distribution over several targets comes from running the model autoregressively: predict one target, add it to the context, re-encode everything, and repeat. Re-encoding at each of K steps costs O(K(N+K)²) for N context points.

## 2. The method

Stated (Sec. 3). The model encodes the context once and caches its keys and values. Generated targets go into a small **causal buffer**. Each buffer token attends to the cached context and to earlier buffer tokens. Each target attends to the context and the visible buffer prefix. The attention cost drops to O(N² + NK + K²).

Four attention rules enforce this, all applied through one attention mask in a single training forward pass:

| Rule | What it says |
| --- | --- |
| R1 | The context is encoded once and cached as read-only |
| R2 | Buffer token j attends only to the context and to buffer tokens before j |
| R3 | Information flows out of the context, never back into it |
| R4 | Each target attends to the context and the visible buffer prefix; targets never attend to each other |

Context, buffer and target tokens carry distinct **role embeddings**. Buffer tokens also carry a learned **positional embedding** that marks their autoregressive order (Sec. 3, App. H.2). The context set has no positional embedding, so the model treats it as unordered.

Training conditions each target on a ground-truth buffer prefix of random length, so all targets get a loss in one pass (Sec. 3).

## 3. Results

Stated. Speed (Sec. 5, Fig. 1; one NVIDIA L40S GPU; buffer size K = 16, M = 16 targets, batch 256). Joint sampling is 3 to 20× faster than the fully autoregressive baselines TNP-A and TNP-D-AR. Density evaluation is as fast as TNP-A and K× faster than TNP-D-AR. Training-step time is close to the fastest baselines. The abstract reports up to 7× lower memory. The authors also sped up every baseline 3 to 10× over its public code with KV caching, FlashAttention-2 and compilation.

Stated. Quality, average joint predictive log-density (higher is better), Table 1, M = 16 targets, mean (standard error):

| Task | Re-encode at every step (TNP-D-AR) | Fully autoregressive (TNP-A) | Buffer, K = 16 (fast) | Buffer, K = 1 (slow) |
| --- | --- | --- | --- | --- |
| Gaussian-process functions | 2.57 (0.020) | 2.24 (0.018) | 2.51 (0.019) | 2.56 (0.019) |
| Sawtooth functions | 1.05 (0.004) | 0.98 (0.004) | 1.00 (0.005) | 1.09 (0.004) |
| EEG interpolation | 0.51 (0.013) | 0.58 (0.014) | 0.52 (0.013) | 0.54 (0.014) |
| EEG forecasting | 1.07 (0.004) | 1.23 (0.003) | 0.85 (0.004) | 1.21 (0.003) |

The fast buffer gives up a little quality on most tasks. It gives up more on EEG forecasting: 0.85 against 1.07 for re-encoding.

Stated (Sec. 5). On a neuroscience model-comparison task (an audio-visual localization model, 15 participants, 30 datasets of 400 trials), the buffer's log marginal likelihood lands close to the ground truth. It matches TNP-D-AR on model comparison. Built into TabICL and pre-trained from scratch on 10.24 million synthetic datasets with K = 32, both autoregressive modes beat independent prediction in low-context settings and in high-context forecasting.

Stated (Sec. 6, limitations). Cost still grows with K through the O(K²) buffer term, and the buffer positional embeddings form a fixed learned set. Quality can drift from exact re-encoding on long buffers.

## 4. The two ablations that touch our map

**Positional embeddings in the buffer** (App. H.2, Table A7, Gaussian-process task, M = K = 16). Stated: training without buffer positional embeddings gave no statistically significant difference in predictive performance.

| Model | Joint log-density, mean (standard error) |
| --- | --- |
| Buffer, with positional embeddings | 2.51 (0.02) |
| Buffer, without positional embeddings | 2.51 (0.02) |
| Re-encode at every step (TNP-D-AR) | 2.57 (0.02) |

The authors connect this to work showing that causal transformers infer position without explicit encodings (Haviv et al. 2022; Irie 2024; Zuo et al. 2025). They suggest RoPE or ALiBi for scaling to larger buffers.

**Permutation invariance of the context** (App. H.1, Fig. A14). Stated: a plain decoder transformer that reads the context as a fixed sequence does substantially worse on the Gaussian-process task at every context size. The authors conclude that keeping the context, or at least part of it, permutation invariant "is critical for performance".

## 5. What this means for the position-embedding split (map #78)

Inference throughout.

- **Why the positional-embedding ablation transfers poorly.** Our closest arm is No slot vectors. Field-type vectors play the part of the paper's role embeddings, and the dropped slot vector plays the part of its order embedding. In the paper, the causal mask still gives each buffer token its order after the embedding is gone. Nothing in our model replaces what the slot vector carries. Attention runs in both directions with no mask between Pokémon, so without slot vectors the network can't tell which fields belong to the same Pokémon. The #78 measurement found that, with field vectors only, swapping two species changes the other fields by rounding only (1.1e-7, on untrained networks). Our expectation that No slot vectors loses at least 0.05 win rate does not conflict with Table A7.
- **The permutation result points the same way as our design.** Our slots are shuffled every batch, so slot order means nothing, much like the paper's unordered context. The paper's H.1 result supports treating the six Pokémon as a set. It says nothing about whether a slot vector helps fields find their own Pokémon, which is what Learned and Fixed slot vectors test.
- **RoPE and ALiBi.** The paper suggests them for longer buffers, a sequence-length problem. The owner dropped both for this map on 2026-11-01 because a team has no sequence distance. This paper gives no reason to revisit that.
- **The buffer itself** solves a cost problem our model doesn't have. Our sampler unmasks fields of one fixed 84-field team. It never re-encodes a growing context of other teams.

Net: the paper doesn't change the plan or its expectations. Its main use here is as a citation for "learned position vectors on an ordered buffer can be dropped when a causal mask carries the order", which is not our setting.
