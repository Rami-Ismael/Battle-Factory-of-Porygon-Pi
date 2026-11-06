# Guidance for masked text diffusion

**Inference, recommendation for #191:** choose **(c), no classifier-free guidance (CFG), for the first masked-model experiment**, with `MDLMSampler.cfg_scale=0`; retain the requested win rate in the conditional prompt. This is a budget and evidence decision, not a requirement of masked diffusion. **Stated:** the masked-model scaling paper explicitly demonstrates guidance after conditional-only fine-tuning, without prompt dropout, and beats its 10% prompt-dropout comparison, so #191's assertion that the model has *never* learned masked-prompt inputs is too strong [SMDM §4, §6, Appendix B.4, Table 3](https://arxiv.org/abs/2410.18514v3). **Inference:** neither that result nor the earlier block experiment establishes improved ask-versus-got calibration for Pokémon Video Game Championships (VGC) teams. Guidance roughly doubles denoiser work, while the project's earlier guidance increment was only +0.009 win rate, 0.5 standard errors, at 64 teams [project results](textdiff.md). If guidance later earns that cost, compare a matched masked-prompt branch, option (a), against the existing unknown-win-rate branch, option (b); do not declare dropout mandatory or assume the two remove the same information. **Stated:** the pinned fast cached sampler ignores `cfg_scale` entirely [implementation, line 165](https://github.com/ZHZisZZ/dllm/blob/ca176752fbceec49c6b4777a2c18ae88e4eb10ed/dllm/pipelines/fastdllm/llada/sampler.py#L165).

## Scope and evidence convention

**Stated, project scope:** [issue #191](https://github.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/issues/191) and its [parent #160](https://github.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/issues/160) were read first. This report extends [the existing scratch-model guidance research](classifier-free-guidance-unlabelled-data-research.md); its discussion of Ho and Salimans, discrete guidance, and the unlabelled-source mismatch is incorporated by reference rather than repeated. No training, generation, paid compute, or battles were run.

**Stated, research method:** papers below were retrieved as complete extracted text through alphaXiv `get_paper_content(fullText=true)`, including appendices, rather than its generated reports. Findings cite the authors' methods and experiments; background and related-work descriptions are not used as evidence about other methods. Paper versions are pinned in the source table. Source-code findings use GitHub files at explicit commits, inspected on 2026-11-05. “Not reported” records a gap in the inspected source, not a claim that an implementation cannot exist elsewhere. **Inference** labels deductions and recommendations. Table evidence cells apply their Stated/Inference label to the whole cell.

## 1. What published models actually do

### Put every guidance scale on the same convention

**Inference, algebra from [LLaDA Appendix A.2 Eq. 16](https://arxiv.org/abs/2502.09992v3) and the code in §5:** use an *extra guidance weight* `w` throughout this report:

```text
conditional input:   [prompt tokens, partially revealed response]
masked-prompt input: [MASK × prompt length, same partially revealed response]

guided_logits = conditional_logits + w × (conditional_logits - null_logits)
guided_probs  = softmax(guided_logits)
              ∝ conditional_probs^(1+w) / null_probs^w

w = 0: ordinary conditional generation, one row
w > 0: extrapolate away from the comparison distribution, two rows
```

**Inference:** this is a normalized geometric combination, not an arithmetic average of probabilities. Combining raw vocabulary logits this way and then applying softmax equals combining their normalized log probabilities and renormalizing, since the extra log-partition terms are constants across vocabulary entries. It does **not** establish that the terminal sequence distribution equals the same powered combination. It also does not make denoised-token guidance identical to guidance on complete reverse-transition probabilities, which include the chance of remaining masked.

**Stated:** Schiff et al. use `γ`, with `γ=1` ordinary conditional and `γ=0` unconditional, in §3.1 Eqs. 6–8 [discrete guidance](https://arxiv.org/abs/2412.10193v3). **Inference:** their `γ=1+w`; our scratch/block parameter `guidance=1+w` also has this offset [existing project write-up](classifier-free-guidance-unlabelled-data-research.md). Thus old `guidance=2` corresponds to new `cfg_scale=1`, and old `guidance=1` to new `cfg_scale=0`. Setting new `cfg_scale=2` would strengthen guidance beyond the old comparison.

### Masked-prompt guidance learned during pretraining

| Published model or method | Unconditional input and training | Formula and reported weights | Measured effect and limits |
| --- | --- | --- | --- |
| Scaling masked models with unsupervised guidance, SMDM | **Stated:** a mask sequence of the same length as the prompt, keeping the noisy response unchanged. Unsupervised text pretraining learns both terms. Conditional fine-tuning leaves prompts intact. Its separate standard-guidance comparison masks the whole prompt on 10% of paired rows. §4 Eqs. 7–8, Appendix B.4, Table 9. | **Stated:** per-token denoised probabilities raised to `1+w` and `−w`, Eq. 7. Conditional-generation search `w ∈ {0.4,0.6,0.8,1}`; mathematics evaluation uses `w=0.1`, Appendix B.3. | **Stated:** MT-Bench conversation score, Table 3: unguided 1.32; standard dropout guidance 1.53; conditional-only fine-tuning with unsupervised guidance 1.60. Table 1 zero-shot examples: OpenBookQA 27.00→34.20 and LAMBADA 36.00→40.99. These are language tasks, not reward calibration. [Paper](https://arxiv.org/abs/2410.18514v3). |
| Large language diffusion, LLaDA 8B Base | **Stated:** same-length masked prompt, Eq. 16. Pretraining independently masks all text tokens at a random masking rate, §2.2. No separate prompt-dropout stage is reported for Base. | **Stated:** Eq. 16 has the same denoised-probability ratio. Appendix B.3 searches `w ∈ {0.5,1,1.5,2}` separately for each task. Main-text results omit guidance. | **Stated:** Table 6, ordinary→best guided accuracy: ARC Challenge 45.9→47.9; HellaSwag 70.5→72.5; TruthfulQA 46.1→46.4; WinoGrande 74.8→74.8; Graduate-Level Google-Proof Question Answering, GPQA, 25.2→26.1; Physical Interaction Question Answering, PIQA, 73.6→74.4. “Consistently improves” includes a tie and best-of-grid selection. [Paper §3.3, Appendix B.3](https://arxiv.org/abs/2502.09992v3). |
| Instruction-tuned masked model, LLaDA-8B-Instruct | **Stated:** supervised fine-tuning, SFT, keeps the prompt unchanged and masks only the response, §2.3 Eq. 5, Fig. 2b, Algorithm 2. The official sampler nevertheless supports masked-prompt guidance. **Inference:** there is no dedicated condition-dropout coin flip in this specified SFT recipe. | **Stated:** official `generate.py` defaults to `cfg_scale=0`; a positive value mixes raw vocabulary logits as above. | **Stated:** the original paper's isolated guidance ablation is explicitly on **Base**, not Instruct. The six accuracy changes above therefore do not establish an Instruct guidance gain. Later studies below do evaluate Instruct. [Paper](https://arxiv.org/abs/2502.09992v3), [pinned sampler](https://github.com/ML-GSAI/LLaDA/blob/9182493720ed723ef8031210d85959364e51cbe0/generate.py#L148). |
| Preference-aligned masked model, LLaDA 1.5 | **Stated:** starts from LLaDA Instruct and performs variance-reduced preference optimization on 350,000 preference pairs, Appendix D.3. No CFG null representation, condition-dropout rate, or guidance-scale sweep is specified in this paper. | **Stated:** Appendix D.4 and Table 6 specify sampling, answer lengths, block lengths, and end-token handling, but no CFG weight. Consequently a paper-specific CFG formula or used weight is **not reported**. | **Stated:** Table 4 compares random diffusion, low-confidence reveal, and sequential block sampling; those are not guidance ablations. Its gains must not be credited to CFG. **Inference:** inheriting a backbone or a compatible generator is insufficient to assert that the reported 1.5 evaluation used positive guidance. [Paper](https://arxiv.org/abs/2505.19223v2). |
| Autoregressive-initialized masked model, Dream 7B | **Stated:** initialized from Qwen2.5-7B, continued masked training on 580 billion tokens, then SFT with intact prompts and response-only noise, §4.1–4.3. The paper does not specify a CFG null input, dropout rate, guidance formula, or used guidance weight. | **Stated:** no isolated CFG implementation or weight is reported in the Dream paper. “Token-specific guidance” in §4.2 refers to training loss/noise reweighting, not CFG. | **Stated:** Dream's paper supplies no guided-versus-unguided CFG ablation. A later adaptive-masking paper does test Dream, below. **Inference:** do not fill this gap with LLaDA settings just because both are masked models. [Paper](https://arxiv.org/abs/2508.15487v1). |

### Class-conditioned masked diffusion is a different training setup

**Stated:** the original **masked diffusion language model, MDLM**, introduces a weighted masked-language loss and an efficient parameterization, §3; its language-generation experiments are unconditional, §4.2 and Appendix E.6 [MDLM paper](https://arxiv.org/abs/2406.07524v2). It does not report prompt-dropout CFG, a prompt null representation, a CFG weight, or a measured CFG gain. **Inference:** the class named `MDLMSampler` in `dllm` is not evidence that the original MDLM paper used that library's LLaDA-derived guidance rule.

**Stated:** Schiff et al. add discrete CFG to MDLM and to uniform-noise models. They use a **class [MASK] token in the conditioning channel**, fused through adaptive layer normalization, and replace the class with it on **10%** of training rows, Appendix D.4.2. This is not deletion of a natural-language prompt or masking the response. Their §3.1 Eqs. 6–8 combine **reverse-transition probabilities**, using normalized `p_cond^γ p_null^(1−γ)` [paper](https://arxiv.org/abs/2412.10193v3).

**Stated:** for species-conditioned genome generation they sweep `γ ∈ {1,2,3}`, equivalent to this report's `w ∈ {0,1,2}`, and 128/256/512 sampling steps. Appendix E.1 Table 15, at 128 steps, masked diffusion's species-classification F1 goes 0.88±0.01→0.90±0.011→0.78±0.009; its 3-mer Jensen–Shannon divergence worsens 0.02→0.06→0.12. Higher guidance therefore does not monotonically improve both quality and control. The uniform-noise model tolerates larger guidance better. Their image grid uses `γ ∈ {1,2,3,4,5}`. **Inference:** these are evidence for trained null-class guidance and a quality/control trade-off, not a rate recommendation for prose prompts or an optimum for VGC teams [Appendix E.1](https://arxiv.org/abs/2412.10193v3).

### Relevant 2025–2026 extensions

**Stated:** **uncertainty-focused comparison inputs**, called Adaptive CFG in Li et al., temporarily mask the lowest-confidence fraction of currently non-mask tokens, including eligible prompt and revealed response positions. Algorithm 1 and §3.2 use the same `L_cond + w(L_cond−L_comparison)` formula. It is an inference-time modification of existing LLaDA and Dream models, with no new condition-dropout training reported. The masking fraction is 0.7. §4.1.3 searches standard `w ∈ {0.5,1,1.5,2}` and says adaptive `w` is similarly tuned, but does not supply the final model-specific values. Its wording is internally unclear about task-specific versus model-wide fixed scales [paper §3–4](https://arxiv.org/abs/2505.20199v1).

**Stated:** Table 1 reports LLaDA GPQA standard→adaptive guidance 29.4→33.3, and Sudoku 34.0→42.0. Dream ordinary→adaptive Sudoku is 72.0→80.0, but WinoGrande worsens 73.3→72.5. The text inconsistently calls the standard comparison a masked *target* and elsewhere a masked *prompt*, so do not use that table as an unambiguous specification of the static baseline. **Inference:** a dynamically degraded input is not necessarily an unconditional marginal. This method still needs a second model evaluation, sequentially after confidence assessment, rather than removing CFG's cost [Algorithm 1, Table 1](https://arxiv.org/abs/2505.20199v1).

**Stated:** **normalized jump guidance**, Rojas et al., separates the jump rate from the guided token distribution, §3.2 Eqs. 11–12, to avoid guidance inadvertently speeding unmasking. It uses `softmax(γ log p + (1−γ) log q)` for token choice; their symbol `w` is this report's `γ=1+w`. No additional condition-dropout training is introduced for the LLaDA-Instruct test. Figure 8 evaluates MATH-500 with a simple Euler sampler, 32-token blocks, 256 total steps, and scales 1–3 in their convention; normalization wins across tested scales. These are not the official confidence-based sampler's benchmark numbers [paper §4.1](https://arxiv.org/abs/2507.08965v2).

**Stated:** the same paper analyzes guidance schedules in §3.4 and tests them in §4.2, finding increasing guidance better than decreasing guidance on masked **image** generation. Its schedule experiment is not a VGC or text-calibration result. **Inference:** the library's fixed reveal quotas already separate token choice from mask-count scheduling; do not mechanically transplant a correction for guidance-altered jump rates into that sampler.

**Stated:** **prompt-specific switch-off**, Zhou et al. 2026, uses conditional logits plus `w` times conditional-minus-unconditional logits, §3.2. Its center configuration is LLaDA-Instruct, `w=2`, 64 tokens, 64 steps, and 13 constraint subtasks of 200 prompts each. No new prompt-dropout training is reported; the exact unconditional token construction is not specified in the inspected paper. §5.2 Table 1 reports switching off at an estimated prompt-specific horizon noninferior to full CFG on all 13 subtasks at a prespecified 0.03 margin. Horizons use repeated continuation rollouts and cross-fitting, not a free online oracle [paper](https://arxiv.org/abs/2608.08082v1).

**Inference:** neither “late guidance is best” nor “early guidance is sufficient” is universal. The normalized-jump paper analyzes low-dimensional distributions and tests masked-image schedules; the switch-off paper tests prompt constraints on an instruction model. They study different samplers, comparisons, and success criteria. Neither reports win-rate calibration, Showdown validity, or team diversity.

## 2. Does a masked prompt work without prompt dropout?

**Stated, direct answer: yes in published settings.** SMDM §4 Eq. 8 explicitly argues that unsupervised masked pretraining learns the joint text distribution and its conditionals, including predictions with the prompt replaced by same-length masks. §6 Table 3 then tests conditional-only SFT plus guidance against 10% masked-prompt dropout plus guidance, obtaining MT-Bench scores 1.60 versus 1.53, compared with 1.32 without guidance. Appendix B.4 explicitly says it only enhances the conditional distribution during unsupervised-guidance fine-tuning. This is stronger evidence than merely noting that a sampler exposes a guidance argument [SMDM](https://arxiv.org/abs/2410.18514v3).

**Stated:** LLaDA pretraining masks all positions, §2.2; its SFT leaves prompts intact, §2.3. Its Base guidance ablation and official generator corroborate masked-prompt guidance as an intended use, but the original paper does not isolate its effect on the instruction-tuned model [LLaDA §3.3, Appendix A.2 and B.3](https://arxiv.org/abs/2502.09992v3).

**Inference, what pretraining actually guarantees:** with `P` prompt positions, independently masking the prompt has probability `t^P` at masking rate `t`. With uniform `t`, the marginal chance is `1/(P+1)`. But the probability of masking all `P` prompt positions while retaining a particular set of `K` response positions contains `t^P(1−t)^K`, which can be very small. Thus the input has support under the objective, but a useful particular prompt/response configuration is not guaranteed frequent, and its denoising accuracy is not guaranteed. This deduction uses [LLaDA Eq. 3 and §2.2](https://arxiv.org/abs/2502.09992v3), not an empirical count of training inputs.

**Inference, limitation for this project:** Qwen autoregressive pretraining alone does not teach arbitrary bidirectional masked contexts. The subsequent masked diffusion training does. After a narrow domain fine-tune, a retained pretraining capability may weaken or supply a general-text comparison rather than a Pokémon-team marginal. No inspected paper establishes that this fine-tune preserves that branch, or that a highly structured, long prompt matches the pretraining text distribution assumed in SMDM §4. The correct issue wording is therefore “the current SFT recipe does not explicitly train the sampler's null input,” not “the model never sees that input in training.”

**Stated, implementation boundary:** `dllm`'s generic trainer sets `maskable_mask = labels != -100`; its stochastic diffusion mask intersects that mask. If prompt labels are `-100`, prompts stay clean. It has no independent prompt-dropout option in `MDLMConfig` [trainer lines 26–30 and 147–166](https://github.com/ZHZisZZ/dllm/blob/ca176752fbceec49c6b4777a2c18ae88e4eb10ed/dllm/core/trainers/mdlm.py#L147). **Inference:** if a caller instead labels the whole concatenated sequence, the same trainer can mask prompt positions. The claim is about this response-only training setup, not every possible use of the class.

## 3. Which issue option best fits this experiment?

| Choice | Evidence and consequence | Judgment |
| --- | --- | --- |
| No guidance, option (c) | **Stated:** LLaDA's main results omit CFG; the project's earlier extra guidance gave +0.009 win rate, 0.5 standard errors [textdiff results](textdiff.md). **Inference:** the conditional model still receives the requested win rate. Neither result shows guidance cannot help calibration. | **Inference:** recommended initial configuration because the added cost has no demonstrated payoff on the chosen metric. This is a provisional project decision, not a claim of literature-wide superiority. |
| Masked-prompt training, option (a) | **Stated:** SMDM tests 10% whole-prompt masking during paired SFT, but its no-dropout guided recipe scores higher, Table 3 and Appendix B.4. **Inference:** explicit dropout matches the sampler and maintains a domain-trained comparison; it also removes format, regulation, opponent, and other prompt information if present. | **Inference:** defensible if a guided domain-trained branch is required. Start with 10% only as a published reference setting, not an established optimum. It is not a mandatory fix justified by lack of SFT dropout. |
| Unknown-win-rate training, option (b) | **Stated:** the earlier block recipe uses `unknown` on 15% of labelled rows and all unlabelled rows [textdiff](textdiff.md). **Inference:** retaining all other prompt content removes specifically the ask, closer to the intended win-rate contrast. It requires a custom sampler input rather than the existing whole-prompt mask. | **Inference:** the cleaner semantic contrast if the goal becomes specifically amplifying the win-rate label. However, the null-source mismatch already documented for the scratch model remains; no inspected masked-text paper directly compares this option with (a) on reward calibration. |

**Inference, omitted fourth possibility:** conditional-only SFT plus masked-prompt guidance is supported by SMDM but missing from #191's three-way menu. It should be acknowledged even if the first run uses (c). Otherwise the decision would falsely present prompt dropout as a prerequisite for using the existing sampler.

**Inference, calibration:** the ask is a desired output win rate, not simply a class to maximize. Amplifying a likelihood ratio for “ask 0.50” can sharpen typical examples of that condition, shift the achieved mean, or reduce diversity; none guarantees lower mean squared error against the ask-versus-got diagonal. The +0.009 observation is an aggregate win-rate difference at one ask. It cannot answer whether the slope, intercept, or extremes of that diagonal improve. Any later guidance comparison must retain the same teams/asks, sampling temperature, reveal schedule, and evaluation protocol, and examine calibration alongside validity and diversity [metric definitions in #160](https://github.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/issues/160).

### Keep unlabelled teams separate from the dropout decision

**Stated:** #160 records 304,580 teams, of which 130,232 have no win-rate label. **Inference, arithmetic:** that leaves 174,348 labelled teams. At 15% label dropout, the expected null total is 156,384.2 rows per epoch, 51.34% of all rows; 83.28% of null-path rows are genuinely unlabelled. At 10% dropout the corresponding figures are 147,666.8, 48.48%, and 88.19%. These fractional values are expectations, not exact sampled counts. The earlier [scratch write-up](classifier-free-guidance-unlabelled-data-research.md) already explains why a different unlabelled source can make the guidance ratio partly a labelled-source detector.

**Inference:** choosing (c) does not require throwing away 130,232 teams. A proposed two-stage recipe can learn team syntax from all teams without a numeric ask, then learn numeric conditioning from the labelled subset. An alternative is to retain unknown-labelled rows in the conditional training mixture. Which distribution and weights to use is a training-design decision; this report does not claim either already exists or is empirically best. If (a) is chosen, do not automatically put all unlabelled rows into its masked-prompt branch and call that 10% dropout. If (b) is chosen, distinguish dropped labelled rows from genuinely unlabelled rows in data accounting. Neither label the missing win rates as zero nor invent numeric pseudo-labels.

## 4. Cost and ways to reduce it

**Stated:** the generic guided sampler concatenates conditional and masked-prompt rows and calls the model once on a double batch [sampler lines 165–173](https://github.com/ZHZisZZ/dllm/blob/ca176752fbceec49c6b4777a2c18ae88e4eb10ed/dllm/core/samplers/mdlm.py#L165). **Inference:** this is two row-equivalent evaluations per denoising step, not necessarily two Python calls or precisely twice the wall-clock time. Activation/logit storage grows; weights are shared. On a memory-limited laptop a doubled batch can matter more than a simple factor-of-two estimate.

**Stated:** [prompt ticket #168](https://github.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/issues/168) remains open, contains no approved concrete prompt or token count, and had no comments when inspected. **Inference:** an approved prompt length cannot honestly be supplied here. The old ~230-token prompt-plus-team figure in [textdiff](textdiff.md) is not the new prompt size. Let `P` be the eventual tokenizer-specific prompt length, `R` response length, `B` batch size and `S` denoising steps. Ordinary generation processes roughly `BS(P+R)` token positions; full CFG doubles it. Full bidirectional attention has a component scaling as `BS(P+R)^2`, also doubled; actual cost also includes projections, feed-forward layers and vocabulary logits. Masking prompt tokens does not shorten the sequence or remove attention to them. These are symbolic estimates, not measured laptop runtimes.

| Cost-saving approach | Primary evidence | Applies here? |
| --- | --- | --- |
| Guide only selected steps | **Stated:** SMDM Appendix C.3 Table 13 compares equal sampling compute: unguided 256/512 steps gives MT-Bench 1.32/1.35; guided 128/256 steps gives 1.56/1.60. | **Inference:** reducing total denoising steps can offset a second branch, but must be tested for team quality and calibration. This table halves total steps; it does not test skipping guidance on alternating steps. |
| Limit guidance to an interval | **Stated:** the continuous-image guidance-interval paper §3.1 Eqs. 5–6 switches guidance off outside a noise interval. §4.1 Table 1 reports ImageNet-512 Fréchet inception distance 1.81→1.40; its best large-model setting guides only 6 of 32 steps [paper](https://arxiv.org/abs/2404.07724v2). | **Inference:** implementing an on/off interval needs no causal attention or cache. The image noise interval and its quality gain are not portable text settings. With guided fraction `f`, row-equivalent cost is approximately `1+f` times ordinary sampling, or `(1+f)/2` of full CFG, assuming fixed step counts. |
| Switch off after prompt-specific commitment | **Stated:** the 2026 switch-off paper §5.2 Table 1 meets its success margin on 13 subtasks. §5.2 Table 2, on 905 selected handoff survivors at serial width 1, gives success 0.83→0.82 and forward evaluations 128→84 [paper](https://arxiv.org/abs/2608.08082v1). | **Inference:** directly relevant to masked text, but retrospective rollout costs and family calibration matter. These 905 selected prompts do not establish a net deployment speedup on all prompts. The authors explicitly make no wall-clock or net-gain claim over existing schedulers, §5.4. A VGC horizon cannot be chosen by checking final battle win rates online for free. |
| Reuse prefix keys and values exactly | **Stated:** LLaDA §2.2 notes incompatibility with ordinary key/value, KV, caching under its full attention [paper](https://arxiv.org/abs/2502.09992v3). | **Inference:** unchanged prefix token IDs do not imply unchanged hidden states, because the prefix attends to the changing response. Conditional and masked-prompt branches also have different prefix embeddings. Exact autoregressive shared-prefix caching is therefore not directly available. |
| Approximate bidirectional caching | **Stated:** Fast-dLLM §3.2 and Fig. 3 cache approximately stable activations, refreshing across blocks; dual caching also reuses masked suffix activations. This is explicitly approximate for full attention [paper](https://arxiv.org/abs/2505.22618v3). | **Inference:** compatible with bidirectional models as an approximation, but each guided branch would need its own consistent cache. The current `dllm` fast LLaDA sampler does not implement CFG at all; positive `cfg_scale` cannot be used to claim cached guided sampling. |
| Cache the unconditional prediction across steps | **Inference:** stale null logits could save evaluations but change the guidance field as response tokens are revealed. None of the inspected masked-text studies establishes a safe refresh rate for this VGC setup. | **Inference:** an experiment, not an exact optimization. Whole-prompt masking does not make the null input constant, because the response changes. |
| Distill two branches into one | **Stated:** adapter guidance distillation §4 Algorithms 1–2 trains small adapters on cached guided trajectories with a guidance-scale embedding; §5 Table 1 reports comparable image-generation metrics while halving neural function evaluations. For example, DiT image FID 5.30 with CFG versus 5.03 with adapters; the method trains 16 million parameters versus 676 million for full-model distillation, Table 2 [paper](https://arxiv.org/abs/2503.07274v1). | **Inference:** one-pass inference does not depend on causal attention, but the demonstrated targets are continuous image denoisers, not masked-token logits. Adapting the loss to masked token probabilities, collecting teacher trajectories, and validating sequence quality require new work and training. It is not a free library flag or justified expense before establishing a guidance benefit. |

**Inference:** the first useful cost control is a correctly interpreted zero guidance setting. A guidance-interval parameter is a plausible later sampler extension; a distilled guide or new cache implementation is premature for the initial budgeted experiment.

## 5. What the pinned implementations do

**Stated, pins:** `ZHZisZZ/dllm` commit `ca176752fbceec49c6b4777a2c18ae88e4eb10ed` and `ML-GSAI/LLaDA` commit `9182493720ed723ef8031210d85959364e51cbe0`. These are source snapshots inspected for this report, not claims that the project's environment already has those exact versions installed.

### Generic masked sampler actually guides

**Stated:** [`dllm/core/samplers/mdlm.py`, lines 122–127](https://github.com/ZHZisZZ/dllm/blob/ca176752fbceec49c6b4777a2c18ae88e4eb10ed/dllm/core/samplers/mdlm.py#L122) initializes the set of prompt positions once, before response generation:

```python
unmasked_index = (x != mask_id) & attention_mask.bool()
if not (cfg_keep_tokens is None or len(cfg_keep_tokens) == 0):
    keep_mask = torch.isin(
        x, torch.as_tensor(cfg_keep_tokens, device=self.model.device)
    )
    unmasked_index = unmasked_index & ~keep_mask
```

**Stated:** [`MDLMSampler.sample`, lines 165–177](https://github.com/ZHZisZZ/dllm/blob/ca176752fbceec49c6b4777a2c18ae88e4eb10ed/dllm/core/samplers/mdlm.py#L165) then uses:

```python
if cfg_scale > 0.0:
    un_x = x.clone()
    un_x[unmasked_index] = mask_id
    x_ = torch.cat([x, un_x], dim=0)
    logits = self.model(
        x_, attention_mask=attention_mask.repeat(2, 1)
    ).logits
    logits, un_logits = torch.chunk(logits, 2, dim=0)
    logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
else:
    logits = self.model(
        x, attention_mask=attention_mask
    ).logits  # Use attention mask here
```

**Stated:** default `cfg_scale=0.0`, line 27. The in-place filling method repeats the guidance rule at lines 356–368 and defines original known positions at lines 313–318. `cfg_keep_tokens` keeps *token IDs wherever they occur*, rather than a positional prompt component [same file](https://github.com/ZHZisZZ/dllm/blob/ca176752fbceec49c6b4777a2c18ae88e4eb10ed/dllm/core/samplers/mdlm.py#L313).

**Inference:** newly revealed response tokens remain present in both branches because `unmasked_index` was fixed before sampling. Mask tokens remain attention-visible; these are not padding, zero-length inputs, or a learned null token. In filling tasks, the removed information includes all initially known positions selected by that index, not necessarily only a left prefix. The code comment's “non-EOS” wording is less precise than the expression: actual EOS IDs in attention-valid prompt positions are not excluded by `(x != mask_id)` alone. Keeping structural token IDs does not implement replacing only the numeric ask with an unknown label. Negative `cfg_scale` values take the same ordinary branch as zero; this is not an implementation of interpolation toward unconditional generation.

### Fast cached sampler ignores guidance

**Stated:** [`dllm/pipelines/fastdllm/llada/sampler.py`, lines 165–166](https://github.com/ZHZisZZ/dllm/blob/ca176752fbceec49c6b4777a2c18ae88e4eb10ed/dllm/pipelines/fastdllm/llada/sampler.py#L165):

```python
cfg_scale: float = 0.0  # Unused within Fast-dLLM
cfg_keep_tokens: list[int] | None = None
```

**Stated:** `FastdLLMLLaDASampler.sample` reads other configuration values at lines 199–217, but never reads either guidance field. In the uncached path, lines 389–390, it uses:

```python
out = self.model(x, attention_mask=attention_mask)
logits = out.logits
```

**Stated:** the prefix-cache path warms a cache on the full sequence at lines 422–424, trims it at line 458, and evaluates the suffix with `past_key_values` at lines 475–480. The dual-cache path warms at lines 519–521 and evaluates a replaceable block at lines 566–572. Neither duplicates a null branch or mixes logits [complete pinned file](https://github.com/ZHZisZZ/dllm/blob/ca176752fbceec49c6b4777a2c18ae88e4eb10ed/dllm/pipelines/fastdllm/llada/sampler.py).

**Inference:** calling this sampler with `cfg_scale=1` will not reproduce `MDLMSampler` guidance at 1. A config echo is insufficient evidence that a guidance comparison ran. Its cached speed is not evidence that guidance's second branch became free.

### Official LLaDA generator uses the same logit rule

**Stated:** [`generate.py`, line 125](https://github.com/ML-GSAI/LLaDA/blob/9182493720ed723ef8031210d85959364e51cbe0/generate.py#L125) fixes `prompt_index` before denoising; lines 148–158 implement guidance:

```python
prompt_index = (x != mask_id)
```

```python
if cfg_scale > 0.:
    un_x = x.clone()
    un_x[prompt_index] = mask_id
    x_ = torch.cat([x, un_x], dim=0)
    if attention_mask is not None:
        attention_mask_ = torch.cat([attention_mask, attention_mask], dim=0)
    logits = model(x_, attention_mask=attention_mask_).logits
    logits, un_logits = torch.chunk(logits, 2, dim=0)
    logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
else:
    logits = model(x, attention_mask=attention_mask).logits
```

**Stated:** signature lines 95–99 defaults to `cfg_scale=0.`; the example at line 261 also uses zero [file](https://github.com/ML-GSAI/LLaDA/blob/9182493720ed723ef8031210d85959364e51cbe0/generate.py#L95). **Inference, source caveat:** in this particular pin, when guidance is positive and `attention_mask=None`, `attention_mask_` has not been assigned before line 154. That branch would raise an unbound-local error. This static observation was not tested by loading a model; pass an explicit attention mask if using this snapshot. It does not change the intended formula or justify silently repairing the quoted source.

## Sources and what they establish

| Title / source | Pinned link | What it established |
| --- | --- | --- |
| Scaling up Masked Diffusion Models on Text | [2410.18514v3, 2025-02-28](https://arxiv.org/abs/2410.18514v3) | **Stated:** same-length masked-prompt unsupervised CFG; explicit no-dropout versus 10% dropout SFT comparison; equal-compute sampling comparison. |
| Large Language Diffusion Models | [2502.09992v3, 2025-10-18](https://arxiv.org/abs/2502.09992v3) | **Stated:** whole-text masking pretraining, clean-prompt SFT, Eq. 16 and Base guidance ablation, Table 6. |
| LLaDA 1.5: Variance-Reduced Preference Optimization for Large Language Diffusion Models | [2505.19223v2, 2025-10-12](https://arxiv.org/abs/2505.19223v2) | **Stated:** preference-training and evaluation recipe; no identified CFG-specific ablation or reported weight. |
| Simple and Effective Masked Diffusion Language Models | [2406.07524v2, 2024-11-10](https://arxiv.org/abs/2406.07524v2) | **Stated:** original masked-language objective and unconditional generation; no identified original prompt-CFG experiment. |
| Dream 7B: Diffusion Large Language Models | [2508.15487v1, 2025-08-21](https://arxiv.org/abs/2508.15487v1) | **Stated:** Qwen initialization, masked pretraining, response-only SFT; no identified CFG recipe or isolated gain. |
| Simple Guidance Mechanisms for Discrete Diffusion Models | [2412.10193v3, 2025-05-27](https://arxiv.org/abs/2412.10193v3) | **Stated:** transition-probability guidance, learned null class, 10% dropout, guidance-quality trade-offs. Existing scratch research supplies the broader treatment. |
| Adaptive Classifier-Free Guidance via Dynamic Low-Confidence Masking | [2505.20199v1, 2025-05-26](https://arxiv.org/abs/2505.20199v1) | **Stated:** dynamic comparison tokens, two evaluations, LLaDA and Dream results; baseline-description ambiguities retained above. |
| Improving Classifier-Free Guidance in Masked Diffusion: Low-Dim Theoretical Insights with High-Dim Impact | [2507.08965v2, 2026-03-02](https://arxiv.org/abs/2507.08965v2) | **Stated:** normalization of jump rates; low-dimensional schedule analysis; masked-image schedule experiments and LLaDA text normalization experiment. |
| Commitment Before Realization: When Classifier-Free Guidance Becomes Unnecessary in Masked Diffusion Language Models | [2608.08082v1, 2026-08-08](https://arxiv.org/abs/2608.08082v1) | **Stated:** prompt-specific guidance handoff, full-census noninferiority, selected deployment population, calibration/rollout costs and limitations. |
| Applying Guidance in a Limited Interval Improves Sample and Distribution Quality in Diffusion Models | [2404.07724v2, 2024-11-06](https://arxiv.org/abs/2404.07724v2) | **Stated:** continuous-image interval guidance, fewer unconditional evaluations, measured image-quality gains. |
| Efficient Distillation of Classifier-Free Guidance using Adapters | [2503.07274v1, 2025-03-10](https://arxiv.org/abs/2503.07274v1) | **Stated:** single-evaluation guided image denoisers using trained adapters; no masked-text validation. |
| Fast-dLLM: Training-free Acceleration of Diffusion LLM by Enabling KV Cache and Parallel Decoding | [2505.22618v3, 2025-07-03](https://arxiv.org/abs/2505.22618v3) | **Stated:** approximate prefix/dual KV reuse for bidirectional masked models, distinct from exact causal caching. |
| `dllm` generic sampler and trainer | [sampler](https://github.com/ZHZisZZ/dllm/blob/ca176752fbceec49c6b4777a2c18ae88e4eb10ed/dllm/core/samplers/mdlm.py), [trainer](https://github.com/ZHZisZZ/dllm/blob/ca176752fbceec49c6b4777a2c18ae88e4eb10ed/dllm/core/trainers/mdlm.py) | **Stated:** masked-prompt logit guidance, scale offset, keep-token behavior; response-only masks follow labels, no dedicated prompt dropout. |
| `dllm` Fast-dLLM LLaDA sampler | [pinned source](https://github.com/ZHZisZZ/dllm/blob/ca176752fbceec49c6b4777a2c18ae88e4eb10ed/dllm/pipelines/fastdllm/llada/sampler.py) | **Stated:** guidance fields unused in every cache mode. |
| Official LLaDA `generate.py` | [pinned source](https://github.com/ML-GSAI/LLaDA/blob/9182493720ed723ef8031210d85959364e51cbe0/generate.py) | **Stated:** original prompt-position masking and raw-logit extrapolation. **Inference:** positive-guidance missing-attention-mask failure in this snapshot. |
| Project question, parent map, and prompt gate | [#191](https://github.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/issues/191), [#160](https://github.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/issues/160), [#168](https://github.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/issues/168) | **Stated:** decision options, evaluation metrics, data counts, unresolved prompt approval. |
| Existing project guidance and block results | [scratch guidance research](classifier-free-guidance-unlabelled-data-research.md), [text diffusion results](textdiff.md) | **Stated:** unknown-label mixture interpretation, parameter convention, earlier measured block-guidance increment. |
