# Newer measurements of discrete-diffusion token revisions and official sampler code

Scope: primary-source search through **9 November 2026**, following the supplied earlier note. Seven previously covered papers are not re-reviewed here. Their code appears only where the requested repository audit adds a missing detail. This is a source review, not a sampling experiment. Search coverage is broad but does **not** establish an exhaustive bibliography through November. Full PDFs, including appendices, were downloaded and text-searched for changes, flips, revisions, remasking, counts, convergence, and trajectories; relevant passages were read in context. Figure 6 of the interpolating-model paper was also inspected visually. Negative findings below mean no matching measurement located in that inspected source.

## 1. Which newer papers define and report revision activity?

### Takeaway

| Measurement | Strongest new matches | What it actually measures |
| --- | --- | --- |
| Realized token-ID changes | Correct-token retention; controllable resampling | Adjacent greedy states, or accumulated categorical state transitions |
| Revision operations and effective revisions | Context-preserving verification | Remasks/replacements, separating operations that change lexical identity |
| Remasking frequency | Self-reflective remasking | Remasking activity per generated block |
| Prediction flips | Longest stable prefixes; mutual-stability sampling | Top-1 predictions across model calls, potentially before a token is set |

These are different quantities and should not share one numerical column without the definitions below. Sources: [retention](https://arxiv.org/pdf/2610.01275v1), [resampling](https://arxiv.org/pdf/2604.17310v1), [verification](https://arxiv.org/pdf/2602.06161v2), [remasking](https://arxiv.org/pdf/2509.23653v1), [prefixes](https://arxiv.org/pdf/2603.05454v1), [stability](https://arxiv.org/pdf/2606.16908v1).

### Cited Findings

#### Direct uniform-state measurement: correct-token retention

Nafez and Henderson, *Know When to Hold 'em: Correct-Token Retention in Uniform-State Diffusion Language Models*, 1 October 2026, §1 and Fig. 2, PDF pp. 2–3. The statistic is the mean number of positions out of 512 changed **at each greedy-tail step**. The experiment uses 500 ancestral steps followed by 20 deterministic forward-pass/argmax updates, on 512 samples. DUO, UDLM, and uniform-noise score entropy discrete diffusion, all 169M parameters trained on 20B OpenWebText tokens, still change **173–270 positions at step 20** without the added retention loss. With correct-token retention regularization, the count plateaus at **3–11 within about ten steps**. The ranges span models; no exact per-model table accompanies the figure. [Paper](https://arxiv.org/pdf/2610.01275v1)

Definition consequences: the measured tail excludes departure from the original random prior and the preceding 500 ancestral updates. Its first comparison includes the first tail update relative to its input. **Inference:** a position that changes on two tail updates contributes on both updates; this is an adjacent-step count rather than endpoint distance. The publication gives no cumulative per-position histogram or last-change-time distribution. [Fig. 2 and sampler description](https://arxiv.org/pdf/2610.01275v1)

#### Cumulative categorical transitions: controllable resampling

Kollovieh, Ayadi, and Günnemann, *Interpolating Discrete Diffusion Models with Controllable Resampling*, 19 April 2026, §4.4, Fig. 6, PDF p. 7. The authors count state transitions **`x_t != x_s` over 50 sampling steps**, averaged for nodes and edges of QM9 molecular graphs. The count rises approximately linearly with the resampling coefficient. At coefficient zero it is below one, partly because initialization uses the imbalanced categorical marginal. Figure 6, visually read, gives roughly **15 node / 12.5 edge changes** at coefficient one with time-grid exponent one, versus roughly **6.5 / 5.5** with exponent four. These are approximate graph readings, not table entries. [Paper](https://arxiv.org/pdf/2604.17310v1)

**Inference:** this definition counts a departure from the sampled prior, and counts repeated transitions at one coordinate repeatedly; there is no subtraction for a first setting. It is unusually close to the requested multiplicity measure, but the reported count is for molecular categorical states, not language tokens. Exact plotted data and normalization details beyond the paper's wording were not recovered. [§4.4 and Fig. 6](https://arxiv.org/pdf/2604.17310v1)

#### Explicit revision-operation counts: context-preserving verification

Xiang et al., *Stop the Flip-Flop: Context-Preserving Verification for Fast Revocable Diffusion Decoding*, inspected v2, 28 May 2026. §4 defines a flip-flop when two consecutive unmaskings of one position yield the same token. First unmasking is excluded; repeated cycles count repeatedly. §6.1 defines total revisions, effective revisions that change the assigned token, and their ratio. Saber/WINO revisions are remasks; COVER revisions include remasks and direct replacements. These are operation counts, not all adjacent token-ID transitions. [§§4, 6.1 and footnote 2](https://arxiv.org/pdf/2602.06161v2)

Exact deterministic counts from Appendix Table 6, PDF p. 14, shown as **effective / total**:

| Model | Dataset | Saber | WINO | COVER |
| --- | --- | ---: | ---: | ---: |
| LLaDA-Instruct 8B | HumanEval | 144 / 16,081 | 246 / 3,093 | 99 / 138 |
| LLaDA-Instruct 8B | MATH500 | 95 / 26,721 | 1,044 / 8,655 | 554 / 702 |
| Dream-Instruct 7B | HumanEval | 459 / 30,744 | 142 / 6,660 | 24 / 34 |
| Dream-Instruct 7B | MATH500 | 79 / 23,221 | 420 / 10,650 | 95 / 122 |

These are dataset totals, not changes per sample. Table 7 also reports temperatures 0.1, 0.2, 0.3; it uses the same definitions. [Tables 6–7](https://arxiv.org/pdf/2602.06161v2)

#### Remasking-return frequency: confidence-remasking re-evaluation

Frković et al., *Re-evaluating Confidence Remasking in Masked Diffusion Language Models*, inspected v2, 26 October 2026, §3, Fig. 3 and Appendix B.3. Flip-flop rate is the proportion of remask events later restored to the same token. For WINO, it is approximately **75–90% on LLaDA-8B-Instruct** and **80–95% on Dream-v0-Instruct-7B**, across GSM8K, MATH-500, HumanEval, MBPP and block lengths 32/128. This differs slightly from the ranges in an OpenReview version; use the pinned arXiv v2 consistently. [Paper](https://arxiv.org/pdf/2606.12232v2)

Appendix Table 2 gives exact LLaDA rates for correct versus incorrect generations at block length 32, thresholds 0.6/0.8: GSM8K **74.6/77.9%**, MATH-500 **76.3/77.6%**, HumanEval **85.6/82.4%**, MBPP **81.9/81.9%**. First setting is outside this event denominator; repeat remasks remain separate events. A same-token restoration counts as an ineffective operation, not a lexical edit. [Appendix B.3](https://arxiv.org/pdf/2606.12232v2)

#### Remasking frequency per block: self-reflective remasking

Huang et al., *Don't Settle Too Early: Self-Reflective Remasking for Diffusion Language Models*, inspected v1, 28 September 2025, §5.2, Fig. 3 and Table 3. RemeDi adapts LLaDA-8B-Instruct. For 32-token blocks, remasking frequency mean ± standard deviation is **11.81 ± 10.23** on MATH-500, **28.52 ± 12.04** on HumanEval, and **2.78 ± 5.33** on AlpacaEval. MATH-500 difficulty levels 1–5 give **9.13 ± 9.54; 8.91 ± 7.29; 10.13 ± 8.64; 13.91 ± 11.44; 13.95 ± 11.12**. [PDF pp. 8–9](https://arxiv.org/pdf/2509.23653v1)

The paper calls these remasking frequencies and discusses tokens remasked per block. It does not give a formal summation or the metric-generating script. First unmasking is conceptually outside remasking. Whether the plotted statistic deduplicates repeated remasks at a position is **not established** by the inspected release; do not silently interpret it as either distinct positions or cumulative lexical changes. The released sampler supplies no frequency accumulator. [Paper](https://arxiv.org/pdf/2509.23653v1); [official sampler, lines 70–92](https://github.com/maple-research-lab/RemeDi/blob/2e1b19d34db09e345898f6c5b6ed60418659821a/inference.py#L70-L92)

#### Top-1 prediction flips: longest stable prefixes

Li et al., *Beyond Scattered Acceptance: Fast and Coherent Inference for DLMs via Longest Stable Prefixes*, inspected v1, 5 March 2026, §4.3.4 and Fig. 2. Flip rate is the percentage of tokens in the active suffix whose top-1 prediction changes between consecutive diffusion steps. During 25–75% completion it is **14.2% for scattered acceptance versus 4.3% for prefix commitment**. The paper evaluates LLaDA-8B and Dream-7B, but this paragraph/caption does not identify the particular model and benchmark or pooling used for these two values. [PDF p. 9](https://arxiv.org/pdf/2603.05454v1)

This is a prediction statistic. It does not explicitly count sampled-state replacements, exclude a first prediction as a commitment event, or report total edits per position. **Inference:** repeated prediction flips can contribute at successive comparisons, including before a position is unmasked. [Definition](https://arxiv.org/pdf/2603.05454v1)

#### Top-1 prediction flips over time: mutual-stability sampling

Mohamed et al., *LESS Is More: Mutual-Stability Sampling for Diffusion Language Models*, v1, 15 June 2026, §6.2, Fig. 2, Appendix E.6. Dream-7B's per-step top-1 flip rate over generation positions is averaged over **50 samples per benchmark**, with maximum step budget **256**, on GSM8K, MATH, HumanEval, MBPP. It is highest early and falls to **at most 5% during the final quarter of executed trajectories**. Full plotted values are curves, not a numeric table. The statistic concerns predictive top-1 tokens, not a cumulative count of already-unmasked token replacements. No explicit initial-random/mask replacement convention is given. [Paper](https://arxiv.org/pdf/2606.16908v1)

### Inferences

- **Inference:** The categorical-transition count and greedy-tail count are the closest new matches to realized editing activity. The verification paper is the clearest precedent for explicitly excluding first setting and counting repeated revision operations. Prediction-flip statistics require a separate label. Sources are the definitions linked above.
- **Inference:** A high remask-return rate can coexist with many raw mask/token transitions and few lexical edits. The same-token restoration statistics cannot be interpreted as successful self-correction rates. [Verification definition](https://arxiv.org/pdf/2602.06161v2); [re-evaluation](https://arxiv.org/pdf/2606.12232v2).

### Gaps

The following full PDFs were screened, including appendices; no additional reported cumulative revision-count or settling-time statistic was located. These are checked candidates, **not positive measurement findings**:

| Candidate and pinned PDF | What makes it a near miss |
| --- | --- |
| [Path Planning for Masked Diffusion Model Sampling, v5](https://arxiv.org/pdf/2502.03540v5) | Planner and remasking controls; no located realized re-edit-count result. |
| [Fine-Tuning Masked Diffusion for Provable Self-Correction, PRISM, v4](https://arxiv.org/pdf/2510.01384v4) | Quality head, fixed/adaptive remasking budgets, task quality; no located realized multiplicity result. |
| [Prism: Efficient Test-Time Scaling via Hierarchical Search and Self-Verification, v3](https://arxiv.org/pdf/2602.01842v3) | A different Prism: trajectory search and partial remasking; no located token revision count. |
| [Learn from Your Mistakes: Self-Correcting Masked Diffusion Models, ProSeCo, v4](https://arxiv.org/pdf/2602.11590v4) | Corrector steps and top-k correction controls; no located realized edit statistic. |
| [The Diffusion Duality, Chapter II: Ψ-Samplers, v2](https://arxiv.org/pdf/2602.21185v2) | Uniform-state predictor/corrector sampling; no located token-count result. |
| [Generalized Discrete Diffusion with Self-Correction, v2](https://arxiv.org/pdf/2603.02230v2) | Hybrid/self-correcting method and quality results; no located revision-count result. |
| [BackPlay: Head-Only Look-Back Self-Correction, v3](https://arxiv.org/pdf/2601.06428v3) | Per-round correction budget; no located realized multiplicity result. |
| [From Truncation to Commitment: Persistent Context in Uniform Discrete Diffusion, v2](https://arxiv.org/pdf/2609.01043v2) | Persistent-context intervention; 30.2 ± 0.8% top-rank change in a shared-state intervention audit is **not** a temporal revision rate. |
| [Think While You Generate: Discrete Diffusion with Planned Denoising, v2](https://arxiv.org/pdf/2410.06264v2) | Useful earlier planner comparator; adaptive denoising moves do not supply an empirical per-position revision distribution. First submitted before Schiff. |
| [Principled Self-Correction in Discrete Diffusion: A UCB-Guided Framework for Text Generation](https://aclanthology.org/2026.eacl-long.314.pdf) | Remasking selection mechanism; no located realized token revision statistic. |

*Denoising is not the End: Discrete Diffusion Language Models with Self-Correction* was discovered, but OpenReview's verification challenge prevented full-PDF retrieval. Its indexed abstract is insufficient to rule a metric in or out. [Primary paper location](https://openreview.net/forum?id=uelrt017SR). Additional late-2026 work may be missing; no exhaustive citation-index crawl was completed.

## 2. Do the five requested official repositories retain trajectories or count revisions?

### Takeaway

The default-branch heads resolved to the same commits as the supplied earlier note. The added findings concern apparent intermediate-state logging. No new per-position cumulative counter was found in the inspected generation paths. GitHub files were read using `gh api repos/<owner>/<repo>/contents/<path>?ref=<full-SHA>` and base64 decoding.

### Cited Findings

| Repository and pinned commit | Inspected location and result |
| --- | --- |
| `kuleshov-group/discrete-diffusion-guidance`, `edb0f8c28b7caeb4ea7a06a2fee8d74ab6da1661` | [`diffusion.py:1115–1209`](https://github.com/kuleshov-group/discrete-diffusion-guidance/blob/edb0f8c28b7caeb4ea7a06a2fee8d74ab6da1661/diffusion.py#L1115-L1209): final state only; equality controls cache invalidation. This confirms the background finding; no additional temporal log found here. |
| `s-sahoo/duo`, `7c9b498f5b717de064d6fad7e2509c866e6cb620` | **Addition:** [`trainer_base.py:799–837`](https://github.com/s-sahoo/duo/blob/7c9b498f5b717de064d6fad7e2509c866e6cb620/trainer_base.py#L799-L837) retains `intermediate_tokens` and returns `intermediate_text_samples`, but appends a completed stride only after its inner denoising loop. These are growing semi-autoregressive text prefixes, not successive states of fixed positions. `sampling_steps` counts uncached evaluations, not token changes. Ordinary generation was already covered in the earlier note. |
| `dvruette/gidd`, `f67acc961cc858cc69e8e6d91e7eb67a0a79151c` | **Addition:** [`gidd/sampling.py:79–88`](https://github.com/dvruette/gidd/blob/f67acc961cc858cc69e8e6d91e7eb67a0a79151c/gidd/sampling.py#L79-L88) overwrites `z_t` each step and returns the endpoint. [`gidd/eval/generate_samples.py:28–38`](https://github.com/dvruette/gidd/blob/f67acc961cc858cc69e8e6d91e7eb67a0a79151c/gidd/eval/generate_samples.py#L28-L38) saves batches of completed samples, not time trajectories. The previously documented [`self_correction.py:80–96`](https://github.com/dvruette/gidd/blob/f67acc961cc858cc69e8e6d91e7eb67a0a79151c/gidd/eval/self_correction.py#L80-L96) saves endpoint `num_changes`, not edit multiplicity. |
| `louaaron/Score-Entropy-Discrete-Diffusion`, `0605786da5ccb5747545e26d66fdf477187598b6` | **Addition:** [`run_sample.py:23–31`](https://github.com/louaaron/Score-Entropy-Discrete-Diffusion/blob/0605786da5ccb5747545e26d66fdf477187598b6/run_sample.py#L23-L31) decodes/prints completed samples. [`sampling.py:59–148`](https://github.com/louaaron/Score-Entropy-Discrete-Diffusion/blob/0605786da5ccb5747545e26d66fdf477187598b6/sampling.py#L59-L148) supplies no full-state history or per-position change accumulator. |
| `kuleshov-group/remdm`, `f830d659fb98d9b2c79ff71bfa12d578312b688a` | **Addition:** [`diffusion.py:1152–1192`](https://github.com/kuleshov-group/remdm/blob/f830d659fb98d9b2c79ff71bfa12d578312b688a/diffusion.py#L1152-L1192) has the same potentially misleading `intermediate_tokens` / `intermediate_text_samples` pattern as DUO. Each retained chunk follows a whole inner denoising loop; no full temporal trace is returned. Remask/unmask indicators in [`diffusion.py:740–788`](https://github.com/kuleshov-group/remdm/blob/f830d659fb98d9b2c79ff71bfa12d578312b688a/diffusion.py#L740-L788) update confidence, not a cumulative count. |

For the positive remasking-frequency paper, the official RemeDi sampler is additionally pinned to `2e1b19d34db09e345898f6c5b6ed60418659821a`. [`inference.py:70–92`](https://github.com/maple-research-lab/RemeDi/blob/2e1b19d34db09e345898f6c5b6ed60418659821a/inference.py#L70-L92) keeps current unmasked IDs, chooses retained positions, remasks their complement, and saves a block after completion. `remask_index` includes positions already masked, so summing it directly would overcount actual remask events. No published-frequency accumulator appears in that function.

### Inferences

- **Inference:** Instrumentation should compare fixed-position states inside each sampler's inner loop. Prefix snapshots and final saved sample batches cannot recover revision multiplicity. Include the actual prior and identify final argmax cleanup separately if total changes are desired. [DUO](https://github.com/s-sahoo/duo/blob/7c9b498f5b717de064d6fad7e2509c866e6cb620/trainer_base.py#L799-L837); [GIDD](https://github.com/dvruette/gidd/blob/f67acc961cc858cc69e8e6d91e7eb67a0a79151c/gidd/sampling.py#L79-L88).

### Gaps

This audit covers sampler/generation/correction Python paths, not every historical branch, experiment artifact, notebook, or unpublished logging setup. No model was run. The exact code generating the new papers' figures was not located and validated; paper definitions take precedence over inferred implementations.

## 3. Do uniform samplers settle early, keep editing late, or change edit rate under guidance?

### Takeaway

There is direct evidence of continued late editing for standard uniform-state models under **greedy-tail refinement**, and evidence that retention training sharply reduces it. Early stabilization results in masked models concern **top-1 predictions** under different samplers. They do not establish that uniform-noise trajectories usually settle early. [Uniform-tail experiment](https://arxiv.org/pdf/2610.01275v1); [masked prediction audit](https://arxiv.org/pdf/2606.16908v1).

### Cited Findings

- The uniform-model experiment still finds 173–270 changed positions at greedy-tail step 20 after 500 ancestral steps, versus a 3–11 plateau after retention regularization. A small positive plateau is **not** full settlement of every token. [Fig. 2](https://arxiv.org/pdf/2610.01275v1).
- Dream-7B's prediction-flip rate falls to at most 5% in the final quarter under mutual-stability sampling. This measures temporal prediction stability for a masked model, not revisions after an irreversible reveal. [Fig. 2, Appendix E.6](https://arxiv.org/pdf/2606.16908v1).
- Changing resampling strength and the time grid changes cumulative categorical transition activity on QM9. This is a sampler-control result, not classifier guidance. [Fig. 6](https://arxiv.org/pdf/2604.17310v1).
- Persistently inserting selected token hypotheses changes the top-ranked prediction at 30.2 ± 0.8% of coordinates in a shared-state audit. The reference is a paired **intervention at the same mutable state**, not consecutive sampling steps. [Persistent-context paper, §5.2](https://arxiv.org/pdf/2609.01043v2).

### Inferences

- **Inference:** None of the positive findings establishes a general uniform-noise settling-time law. Counts of changed positions at a timestep, cumulative transition counts, and the retrospective last-change time answer different questions. [Retention definition](https://arxiv.org/pdf/2610.01275v1); [transition definition](https://arxiv.org/pdf/2604.17310v1).

### Gaps

No inspected primary source reports a controlled comparison of **classifier-based or classifier-free guidance strength versus actual token edit rate or last-change time** for uniform-noise generation. Planner choice, remasking strength, confidence thresholds, persistent context, and retention training should not be relabeled as that finding. The earlier note already checks Schiff's guidance sampler. No new paper here reports the distribution of changes per language-token position over the entire random-prior-to-final trajectory, with an explicit first-setting subtraction and a last-change-time statistic together.
