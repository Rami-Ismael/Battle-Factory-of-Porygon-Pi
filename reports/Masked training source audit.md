# Masked training source audit

| Question | Source-backed answer |
|---|---|
| What runs during training? | One corrupted full sequence per microbatch, with one explicit model call. Prompt and padding positions still run. [loss][u-loss] |
| What is scored? | Sampled masked positions, weighted by the noise schedule and normalized by all eligible positions under generic defaults. [loss][u-loss], [normalization][u-norm] |
| Is there one universal masked recipe? | No. Dream changes alignment and weighting; padding and cropping also vary by example. [Dream][u-dream], [collator][u-dreamcollator] |
| Do our block speed results apply? | No masked speed or dollar comparison was measured in this audit. |

The masked trainer in dllm is a different objective and attention graph from our block port. Generic masked diffusion samples one noise rate per example, masks eligible tokens independently, and reconstructs the original tokens from one corrupted sequence. It does not require complementary noisy copies or a clean teacher segment. Setting our `SHIFT=False` alone would leave the wrong corruption, normalization and attention. The shortest defensible migration is a pinned, ordinary masked recipe first, followed by parity checks, then separately justified optimizations. This report establishes source behavior, not runtime compatibility, convergence, Pokémon quality or savings. ([Masked loss][u-loss], [local block loss][l-loss])

This audit answers [ticket 163](https://github.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/issues/163) under [map 160](https://github.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/issues/160). Upstream links pin dllm commit `ca176752fbceec49c6b4777a2c18ae88e4eb10ed`; local baseline links pin `dc8f91c5f97879d285f64da00d82476b1537edce`. The Modal branch was read with `git show` at `651d31dd92d9187354440308b11dcce521d99ca4`. The external clone was treated as untrusted data in its own new empty directory. Nothing was installed or executed from it; no model, training job, Modal call or paid compute ran. Detailed evidence and issue snapshots are in `research_notes/Masked training source audit/`.

## One noisy sequence replaces the block construction

Let `E[b,i] = labels[b,i] != -100`. Under `MDLMConfig` defaults, `epsilon=0.001`, the alpha schedule is linear, weight type is `scheduler`, normalization is `token`, and shifted logits are disabled. For each example, sample `t = epsilon + (1-epsilon) U`, then independently mask each eligible position with probability `1-alpha(t)=t`. The source replaces selected positions with the tokenizer's mask ID. It forwards the entire corrupted sequence once, without passing `t` into the model. It computes cross entropy against original `input_ids` at every position, then zeros unselected losses. ([Defaults][u-config], [linear schedule][u-alpha], [loss body][u-loss])

```text
eligible = labels != -100
masked[b,i] = eligible[b,i] AND uniform[b,i] < t[b]
x_noisy = where(masked, mask_token_id, input_ids)
logits = model(x_noisy, attention_mask).logits
loss = sum(masked * CE(logits, input_ids) / (t + 1e-6))
       / max(sum(eligible), 1)
```

This is the generic default, after any configured alignment preprocessing. The denominator counts **all eligible tokens**, not just this draw's selected masks. `sequence` normalization instead divides each example's sum by its eligible count and then batch size; `batch` normalization divides only by batch size. `uniform` weighting is an explicit alternative. A draw with no selected masks contributes zero with a clamped denominator. The expected eligible masking fraction under the continuous uniform default is 0.5005, an arithmetic implication, not a measured compute reduction. ([Weight selection][u-weights], [normalization][u-norm])

The standard supervised fine-tuning mapper applies a chat template and, by default, labels the prefix before the final assistant message `-100`. Those prompt tokens stay clean and unscored because corruption uses label eligibility. They remain model inputs. With `mask_prompt_loss=False`, prompt tokens become eligible for both corruption and scoring. The tokenizer/chat-template boundary must therefore be verified on our conditioning prompt and full Showdown answer; simply retaining the current string concatenation is not a proven equivalent mapping. ([Chat mapping][u-data], [eligibility and forward][u-loss], [local encoding][l-encode])

Padding is part of the recipe. LLaDA and converted-model masked examples use right padding, pad labels with the tokenizer's pad ID, and remove the attention mask, deliberately making padded tails visible and eligible targets. Their comments describe padded end-of-sequence training, but the shared loader only substitutes EOS for a missing pad token or vice versa; the selected tokenizer's actual IDs must be inspected. Dream retains its collator attention mask and by default trims long responses to a response length sampled from the batch before padding. These choices change the objective and token count. They must be recorded rather than silently replaced with our current ignored block-aligned padding. ([LLaDA collator][u-llada], [converted collator][u-a2d], [mask removal][u-collator], [tokenizer IDs][u-tokenizer], [Dream cropping][u-dreamcollator], [local collation][l-collate])

### Alignment and weights depend on the family

Generic masked language diffusion, used by the LLaDA, converted-model and ModernBERT supervised examples, defaults to logits at position `i` predicting token `i`. Dream instead forces `right_shift_logits=True`: the trainer changes logits to `[logit_0, logit_0, logit_1, ...]`. For `i>0`, target `i` sees original logit `i-1`. Preprocessing ensures first-position labels are ignored, prepending an ignored beginning-of-sequence token unless every row already starts ignored. This describes exact indexing, not verified pretraining compatibility for any candidate checkpoint. ([Generic defaults][u-config], [entry points][u-llada], [converted example][u-a2d], [encoder example][u-bert], [pre/postprocessing][u-shift], [Dream config][u-dream])

Dream defaults to geometric context weighting, named `cart[geo_p:0.3]` in code. For masked target `i`, its weight sums `0.5*p*(1-p)^(abs(i-j)-1)` over unmasked positions `j != i`; the code clamps the exponent at zero before excluding the diagonal. It uses all unmasked positions, including the clean prompt. It does not use `t` directly. It inherits the shared normalization, so the default denominator is still all eligible positions, not the sum of these context weights. The implementation materializes an `L x L` matrix and a batch-by-length matrix product; that is extra weighting work, not a second transformer pass. ([Dream weighting][u-dream], [shared normalization][u-norm])

Attention also belongs to the selected architecture. Converted Qwen3 explicitly uses a bidirectional padding-only mask, LLaDA attention specifies `is_causal=False`, and Dream does likewise. Stock causal Qwen plus the masked loss does not reproduce those implementations. Our normal block loss conditionally shifts, but its returned-logit path and one verification reference hardcode a shift, so the existing helpers cannot prove same-position equivalence unchanged. ([Converted mask][u-qwen], [LLaDA attention][u-lladaattn], [Dream attention][u-dreamattn], [local alignment][l-align], [local verification][l-verif])

## Configuration proves work structure, not dollars saved

The supervised entry points parse model, data and training dataclasses, load a model/tokenizer, map conversations, postprocess lengths, instantiate a trainer, train and save. LLaDA uses `MDLMTrainer`; Dream uses its subclass and custom collator. `examples/a2d/mdlm/sft.py` uses a local converted Qwen3-0.6B path, while `examples/bert/sft.py` uses ModernBERT-large. The separate block example is not this masked recipe. Default model names and hyperparameters are example configurations, not recommendations for this project. ([LLaDA setup][u-llada], [Dream setup][u-dreamsft], [converted setup][u-a2d], [encoder setup][u-bert])

The shared model defaults are bfloat16, no four-bit loading, no low-rank adaptation (`LoRA`), and, if enabled, adapter rank 32, alpha 64, dropout .05 and `all-linear` targets. The loader tries `AutoModelForMaskedLM` then `AutoModel`; its adapter helper returns the model unchanged when `lora=False`. An explicit unquantized, no-adapter configuration is therefore the full-parameter route in this code, subject to checking the chosen model's actual trainable parameters. Optional four-bit loading uses NF4, double quantization and the configured compute dtype, then prepares the model for quantized training; it does not itself enable adapters. If bitsandbytes is absent, the loader leaves quantization unset. Four-bit without adapters must not be described as full-parameter training. ([Model defaults][u-modelconfig], [loader][u-loader], [adapter helper][u-lora])

A proposed project configuration should pin library and model revisions, explicitly select full parameters or adapters, record resolved modules and trainable counts, and preserve Showdown serialization, conditioning fields and stage identity while adapting records to the chosen chat mapping. The upstream default maximum length is 1024. Its actual `right` truncation retains the first tokens, which can cut a team answer; `filter` removes overlength rows. Count prompt-plus-answer lengths with the selected tokenizer before choosing either policy. ([Length defaults][u-lengthconfig], [length processing][u-length])

| Driver | What source proves | Planning inference, not a measured saving |
|---|---|---|
| Forward calls | Shared supervised loss has one explicit full-sequence model call per microbatch. Dream shares that call. [loss][u-loss], [Dream][u-dream] | Accumulation `A` ordinarily means `A` forwards per device per optimizer update, with a shorter last chunk possible; backward checkpoint recomputation is additional work. [dependency loop][hf-loop], [training step][hf-step] |
| Prompt and mask rate | Ignored prompts remain in forwarded input; full logits and per-position CE are computed before masking loss. [mapping][u-data], [loss][u-loss] | Fewer scored tokens do not imply proportionally cheaper forward work. Longer dense sequences increase attention interactions quadratically, alongside other roughly linear sequence work. No runtime ratio follows. |
| Batch and padding | Example recipes dynamically pad; default training/evaluation batch is four and accumulation one. [collator][u-llada], [arguments][u-trainconfig] | Larger microbatches can change utilization, padding and activation memory. Accumulation changes update size without eliminating microbatch work. |
| Checkpointing | Inherited Transformers default is off; Trainer enables it when requested. LLaDA and Dream contain activation recomputation paths. [default][hf-gcdefault], [enable][hf-gcenable], [LLaDA hook][u-lladagc], [LLaDA loop][u-lladaloop], [Dream loop][u-dreamloop] | It trades memory for recomputation. Fit, seconds/example and dollars/example need measurement. |
| Dtype, adapters and quantization | Load dtype, bf16 training flag, quantization and adapter flags are separate. [model config][u-modelconfig], [training config][u-trainconfig], [loader][u-loader] | Storage, activation, optimizer and loss dtypes must be logged separately. No particular speedup or memory fit was established. |
| Validation and generation | Evaluation calls the same stochastic loss under no-grad; it does not generate teams. [prediction step][u-pred] | Validation frequency adds work. Setup, checkpoints, probes and retries also belong in a cost denominator. |
| Dream context weights | Weighting allocates an `L x L` matrix. [implementation][u-dream] | Its share of wall time is unknown. |

For `N` example visits, microbatch `B`, accumulation `A` and data-parallel devices `D`, approximate updates are `ceil(N/(B*A*D))`; ordinary forward calls across devices are approximately `N/B`, with batching and epoch-edge caveats. These are planning counts, not timings. A future dollar estimate must multiply measured GPU-seconds, including validation and overhead, by the then-approved rate and declare whether the denominator is unique examples or example visits. No dollar estimate, masked/block speed ratio or reuse of block timings is justified here. Earlier documents explicitly measure block workloads. ([Existing block measurements][l-doc], [block speed study][l-speed], [loss][u-loss], [dependency accumulation loop][hf-loop])

The dependency file declares Transformers 4.57.0, Accelerate 1.11.0, DeepSpeed .18.0 and PEFT .17.1; optional bitsandbytes and Flash Attention are declarations, not a tested environment. The cited Transformers loop is pinned separately to that version's source. The masked loss ignores additional normalization kwargs, so exact effective accumulation normalization for unequal token counts or wrapped models remains unverified. ([Dependencies][u-deps], [loss and normalization][u-loss], [dependency training step][hf-step])

## The port needs replacement at every block boundary

The table is a migration proposal derived from the cited definitions. “Carry” means retain a feature or verification pattern, not assume the present implementation is valid for a new model.

| Part of `textdiff_port.py` | Carry, change or drop | Required treatment |
|---|---|---|
| `MODEL_ID`, `MASK_ID`, `SHIFT`, `DEV` | Change | Candidate-specific checkpoint, tokenizer IDs, alignment and device. [14–17][l-globals] |
| `load` | Change | Replace Fast-dLLM architecture/weight swap and block patch with pinned masked loader. [20–50][l-load] |
| `qwen3_folder`, `load_qwen3` | Drop from masked path | Stock causal Qwen3 and `bd_size=32` remain block-baseline code. [53–78][l-qwen] |
| `dense_train_mask`, `segment_mask` | Drop | Remove clean/noisy block visibility rules; use selected noncausal model and padding policy. [81–102][l-masks] |
| `_MODULE`, `type_module`, `patch`; nested attention/mask replacements | Drop block patch | Rotary segment tiling and class mutations are block-specific. Carry scaled dot-product attention only after a separate equivalence check. [105–145][l-patch] |
| `block_mask_rates` | Change | Per-example masked schedule replaces per-block rates. [148–154][l-rates] |
| `diffusion_loss`, nested `noisy` | Change | Keep label-based prompt exclusion; replace complementary/clean concatenation, weighting, denominator and alignment. [157–207][l-loss] |
| `eval_logits` | Change | Replace block-causal evaluation with selected masked path. [210–215][l-eval] |
| `verify` | Carry pattern, change reference | Compare masked loss/logits and gradients to upstream using identical tokens and random draws. [218–228][l-verify] |
| `verify_shared`, `__main__` checks | Drop old comparison; change smoke checks | Shared-clean block equivalence is no longer the reference. [231–260][l-shared] |
| `generate_cfg`, nested `sample`, `stopped` | Change | Replace block prefill cache, first-token autoregression, shifted subblock decoding and ending behavior with candidate sampler. [265–335][l-gen] |
| `_layer`, `_layer_kv`, `functional_forward` | Drop initially | Handwritten Qwen block computation, clean key/value prefix and layer skipping need no place in the reference masked path. Checkpointing remains a transferable technique. [350–399][l-functional] |
| `row_mask`, `_segments`, `_MASKS`, `layout_masks` | Drop | Remove block segment/layout caches. [402–435][l-layout] |
| `noisy_pair` | Drop | Replace complementary pair with one corrupted sequence. [438–446][l-pair] |
| `trim_start` | Drop | Remove this block-layout optimization. It trims redundant noisy-copy prompt rows, with clean prompt context retained in the block graph; a one-sequence bidirectional path still needs the prompt tokens. [449–454][l-trim] |
| `layout_loss_sum` | Drop block layouts; conditionally carry selective projection | Split/detach/reduced-noisy-layer paths rely on the old graph. Selected-position output projection requires new weighted-loss and gradient parity proof. [457–488][l-layoutloss] |
| `PAD_ID` | Change | Use selected tokenizer ID. [491][l-pad] |
| `verify_layouts`, nested `grads`, `released`, `functional` | Carry comparison pattern, change all references | Upstream masked objective and family alignment become the reference, including returned-logit paths. [494–528][l-verif] |

The local trainer also needs a full-parameter option because it always applies LoRA with Qwen projection names. Its block-rounded tokenization/collation, layout and parameter-perturbation optimization (`MeZO`) paths, and block-calibrated pruning should stay outside the initial masked recipe. ([Local adapter wrapper][l-lora], [training call][l-traincall], [collation][l-collate], [parameter-perturbation optimizer][l-mezo], [pruning][l-prune])

Sampling must be evaluated separately from training. Generic masked sampling repeatedly forwards the complete prompt-plus-generation canvas, even when updating one generation window. Its actual forward count is the sum of effective window steps after zero-transfer compaction; nominal steps alone can misstate it. It passes no key/value cache in this loop. Classifier-free guidance doubles rows within one explicit model call and masks prompt tokens in its unconditional row except configured keep IDs. Its coefficient is `uncond + (cfg_scale+1)*(cond-uncond)`, offset by one from our local guidance parameter. Dream instead has its own whole-canvas loop and shifted logits. None of these source facts verifies laptop speed or a cache optimization. ([Generic sampler][u-sampler], [step compaction][u-transfers], [Dream sampler][u-dreamsampler], [local guidance][l-gen])

### The Modal branch already supports CUDA

The launcher on `claude/text-diffusion-modal-training-453ea0` packages the block trainer/port and an external Mask Lab file, hardcodes an H100 and a four-hour timeout, supervises commands, and commits data/cache volumes. Its branch port already supports CUDA and pins Fast-dLLM; the branch trainer already has optional TensorFloat-32 (`TF32`) and bfloat16 autocast. Adding CUDA from scratch is not the migration. ([Image contents][m-image], [wrapper][m-wrapper], [branch loader][m-port], [branch trainer][m-trainer])

The proposed launcher change is to package a reviewed pinned masked workload and compatible dependency lock, removing the block-only and unrelated bundled file from that workload. Record library/model revisions, tokenizer, configuration, data hashes and actual trainable parameters. Retain useful data/cache volumes with candidate-aware paths, but add best and last resumable checkpoints, optimizer/scheduler/RNG/patience state, validation logs and artifact hashes. Volume commits alone are not resumability. Replace unconditional final-checkpoint delivery with best-checkpoint export before GPU exit. Choose hardware, microbatch, accumulation, precision and checkpointing from masked measurements. The old timeout is a duration bound, not a verified dollar cap; the launcher's dated price comment was not reverified. ([Wrapper behavior][m-wrapper], [branch saving][m-trainer], [historical price comment][m-price])

The parent map reserves Modal for training and probes, restricts generation and battles to the laptop, requires one job at a time, and requires stating cost and credit before owner approval. Future commands must respect those constraints; this research authorizes no launch. ([Map Notes](https://github.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/issues/160))

## Fixed validation intervals can stop inside an epoch

The local [early-stopping skill](/Users/ramiismael/.claude/skills/early-stopping/SKILL.md:12) is the policy reference. The required helper search found best-loss, patience, restoration and metadata logic embedded in `diffusion84.Model84.train_new`. Extract or extend that state machine rather than create a divergent one. The text trainer already validates at `step % eval_every == 0` and cap/end conditions, but saves every validation checkpoint and then `final` without patience or best restoration. Its validator resets seed 1 and row order but still measures block loss with its own aggregation. ([Existing stopping logic][l-stopping], [text loop][l-loop], [validator][l-validate])

| Skill requirement | Proposed masked implementation |
|---|---|
| Search before writing | Reuse/extract the existing best/patience state logic. |
| Seeded held-out rows | Audit split provenance; hold out 2% for corpus runs or 20% for a few-hundred-row fine-tune, excluding test/top-50/meta-51–100 teams from both sides. Save row IDs/hash and seed. Existing text preparation reuses older splits and is not proof of this new split. [preparation][l-prep] |
| Small-set rule | If validation would have fewer than about 100 rows, use five-fold cross-validation, median best pass, then all-row training for that count. This is the skill's exception, not an invented tiny holdout. [skill](/Users/ramiismael/.claude/skills/early-stopping/SKILL.md:24) |
| Common random numbers | Fix validation rows/order/batches/padding and the time/mask uniforms. Use separate validation RNG or save/restore training RNG. Set eval mode and restore train mode. Shared upstream prediction redraws noise, so stock evaluation alone does not satisfy this policy. [prediction][u-pred], [random draws][u-loss] |
| Fixed interval in a steps run | Set integer `eval_every_updates=K` before launch, where update means after gradient accumulation. Evaluate the full held-out objective every K updates and at cap. Log update, fraction of pass and validation index. Stock `.1` interval defaults are fractional, so replace them explicitly. [defaults][u-trainconfig], [interval semantics][hf-interval] |
| Patience and best restoration | Strictly smaller finite validation loss saves best and resets counter. Otherwise increment; default patience 3 means three complete evaluation intervals. Check immediately after validation, break the whole epoch loop, then restore best at every normal exit. Fail explicitly on nonfinite loss. |
| Cap and learning-rate schedule | Keep a hard cap and size the schedule to it. Derive actual updates after new filtering/tokenization; old nominal totals can differ from skipped batches and max-step caps. [batching][l-collate], [schedule][l-loop] |
| Metadata and caching | Save best/stopped update and pass fraction, cap, patience, K, split and seeds, mask hash, best loss, both curves, stopping reason and checkpoint/hash. Put the stopping rule in cache key/checkpoint name. Resume also restores optimizer/scheduler/RNG/patience state. |
| Battles judge quality | Checkpoint choice uses validation loss; later laptop battles judge the model under the parent map. No generation or battle ran here. [skill](/Users/ramiismael/.claude/skills/early-stopping/SKILL.md:36) |

For generic token normalization, validation should aggregate weighted token-loss numerators and eligible-token denominators across held-out batches; for another normalization, preserve its actual definition. Do not reuse the block validator's row-weighted average by accident. Fixed masks remove a source of noise but do not make the estimate an exact likelihood. Before an authorized run, validate the new stopping helper with a synthetic loss trace that improves, stalls for three intervals, stops mid-epoch and restores the correct checkpoint; verify resumption preserves its state. These are proposed checks, not completed tests. ([Normalization][u-norm], [current validator][l-validate])

## What remains unverified

No checkpoint was selected or pinned, loaded, instantiated or fine-tuned. Candidate tokenizer lengths, attention backend compatibility, model trainability, adapter module coverage, effective quantization, precision stability, memory fit, gradient-checkpointing behavior on hardware and dependencies remain unverified. No masked port, loss/gradient parity test, deterministic masked-validation implementation, stopping/restoration test or sampler adaptation was run. Exact accumulation normalization for unequal-length microbatches and distributed/wrapped configurations remains unverified. ([Loading branches][u-loader], [loss normalization][u-norm])

No masked seconds/example, dollars/example, convergence rate, GPU requirement, laptop generation speed, quality-per-dollar comparison, or advantage over either block baseline has been established. Current Modal pricing and credits were not queried. The historical block results cannot answer those questions. The validation interval K, cap, stage sizes, split IDs and final padding/cropping policy remain execution decisions. Deterministic validation, best-checkpoint restoration, resume state and final export have not been verified at runtime. No newly found blog entered the evidence, so there is no reading-list candidate from this audit. The deliverable is a source-grounded migration specification for refutation; it does not settle which masked model should win.


| Unverified claim | Evidence still needed |
|---|---|
| A candidate checkpoint is compatible with this recipe | Pin model revision and inspect tokenizer, architecture, shift, padding and loaded parameters. |
| Loss and gradients match upstream | Identical corruption and weighted-normalization parity tests, including family-specific shifts. |
| Accumulation or distributed training preserves the intended denominator | Unequal-length microbatch and wrapped/distributed trainer checks. |
| Validation and stopping deliver the correct best model | Fixed-mask repeatability, synthetic patience trace, restore, resume and export verification. |
| A configuration fits or is cheaper/faster than block baselines | Masked hardware measurements including length, padding, dtype, adapters, checkpointing and total charged overhead; current approved price and credit. |
| The model writes stronger teams | Task validity, diversity, novelty, Stat Point use and laptop battle evaluation under the map's protocol. |

[u-config]: https://github.com/ZHZisZZ/dllm/blob/ca176752fbceec49c6b4777a2c18ae88e4eb10ed/dllm/core/trainers/mdlm.py#L25-L51
[u-alpha]: https://github.com/ZHZisZZ/dllm/blob/ca176752fbceec49c6b4777a2c18ae88e4eb10ed/dllm/core/schedulers/alpha.py#L86-L107
[u-weights]: https://github.com/ZHZisZZ/dllm/blob/ca176752fbceec49c6b4777a2c18ae88e4eb10ed/dllm/core/trainers/mdlm.py#L85-L100
[u-loss]: https://github.com/ZHZisZZ/dllm/blob/ca176752fbceec49c6b4777a2c18ae88e4eb10ed/dllm/core/trainers/mdlm.py#L139-L212
[u-norm]: https://github.com/ZHZisZZ/dllm/blob/ca176752fbceec49c6b4777a2c18ae88e4eb10ed/dllm/core/trainers/mdlm.py#L200-L212
[u-shift]: https://github.com/ZHZisZZ/dllm/blob/ca176752fbceec49c6b4777a2c18ae88e4eb10ed/dllm/core/trainers/mdlm.py#L60-L83
[u-pred]: https://github.com/ZHZisZZ/dllm/blob/ca176752fbceec49c6b4777a2c18ae88e4eb10ed/dllm/core/trainers/mdlm.py#L102-L116
[u-data]: https://github.com/ZHZisZZ/dllm/blob/ca176752fbceec49c6b4777a2c18ae88e4eb10ed/dllm/utils/data.py#L225-L253
[u-llada]: https://github.com/ZHZisZZ/dllm/blob/ca176752fbceec49c6b4777a2c18ae88e4eb10ed/examples/llada/sft.py#L42-L125
[u-a2d]: https://github.com/ZHZisZZ/dllm/blob/ca176752fbceec49c6b4777a2c18ae88e4eb10ed/examples/a2d/mdlm/sft.py#L41-L119
[u-bert]: https://github.com/ZHZisZZ/dllm/blob/ca176752fbceec49c6b4777a2c18ae88e4eb10ed/examples/bert/sft.py#L41-L119
[u-collator]: https://github.com/ZHZisZZ/dllm/blob/ca176752fbceec49c6b4777a2c18ae88e4eb10ed/dllm/utils/collators.py#L58-L69
[u-tokenizer]: https://github.com/ZHZisZZ/dllm/blob/ca176752fbceec49c6b4777a2c18ae88e4eb10ed/dllm/utils/models.py#L118-L131
[u-dream]: https://github.com/ZHZisZZ/dllm/blob/ca176752fbceec49c6b4777a2c18ae88e4eb10ed/dllm/pipelines/dream/trainer.py#L9-L77
[u-dreamsft]: https://github.com/ZHZisZZ/dllm/blob/ca176752fbceec49c6b4777a2c18ae88e4eb10ed/examples/dream/sft.py#L43-L151
[u-dreamcollator]: https://github.com/ZHZisZZ/dllm/blob/ca176752fbceec49c6b4777a2c18ae88e4eb10ed/dllm/pipelines/dream/utils.py#L44-L128
[u-qwen]: https://github.com/ZHZisZZ/dllm/blob/ca176752fbceec49c6b4777a2c18ae88e4eb10ed/dllm/pipelines/a2d/models/qwen3/modeling_qwen3.py#L84-L108
[u-lladaattn]: https://github.com/ZHZisZZ/dllm/blob/ca176752fbceec49c6b4777a2c18ae88e4eb10ed/dllm/pipelines/llada/models/modeling_llada.py#L631-L659
[u-dreamattn]: https://github.com/ZHZisZZ/dllm/blob/ca176752fbceec49c6b4777a2c18ae88e4eb10ed/dllm/pipelines/dream/models/modeling_dream.py#L419-L430
[u-modelconfig]: https://github.com/ZHZisZZ/dllm/blob/ca176752fbceec49c6b4777a2c18ae88e4eb10ed/dllm/utils/configs.py#L11-L24
[u-loader]: https://github.com/ZHZisZZ/dllm/blob/ca176752fbceec49c6b4777a2c18ae88e4eb10ed/dllm/utils/models.py#L28-L79
[u-lora]: https://github.com/ZHZisZZ/dllm/blob/ca176752fbceec49c6b4777a2c18ae88e4eb10ed/dllm/utils/utils.py#L90-L125
[u-lengthconfig]: https://github.com/ZHZisZZ/dllm/blob/ca176752fbceec49c6b4777a2c18ae88e4eb10ed/dllm/utils/configs.py#L32-L47
[u-length]: https://github.com/ZHZisZZ/dllm/blob/ca176752fbceec49c6b4777a2c18ae88e4eb10ed/dllm/utils/data.py#L75-L118
[u-trainconfig]: https://github.com/ZHZisZZ/dllm/blob/ca176752fbceec49c6b4777a2c18ae88e4eb10ed/dllm/utils/configs.py#L50-L69
[u-deps]: https://github.com/ZHZisZZ/dllm/blob/ca176752fbceec49c6b4777a2c18ae88e4eb10ed/pyproject.toml#L14-L35
[u-lladagc]: https://github.com/ZHZisZZ/dllm/blob/ca176752fbceec49c6b4777a2c18ae88e4eb10ed/dllm/pipelines/llada/models/modeling_llada.py#L1052-L1080
[u-lladaloop]: https://github.com/ZHZisZZ/dllm/blob/ca176752fbceec49c6b4777a2c18ae88e4eb10ed/dllm/pipelines/llada/models/modeling_llada.py#L1355-L1384
[u-dreamloop]: https://github.com/ZHZisZZ/dllm/blob/ca176752fbceec49c6b4777a2c18ae88e4eb10ed/dllm/pipelines/dream/models/modeling_dream.py#L657-L713
[u-sampler]: https://github.com/ZHZisZZ/dllm/blob/ca176752fbceec49c6b4777a2c18ae88e4eb10ed/dllm/core/samplers/mdlm.py#L97-L230
[u-transfers]: https://github.com/ZHZisZZ/dllm/blob/ca176752fbceec49c6b4777a2c18ae88e4eb10ed/dllm/core/samplers/utils.py#L28-L69
[u-dreamsampler]: https://github.com/ZHZisZZ/dllm/blob/ca176752fbceec49c6b4777a2c18ae88e4eb10ed/dllm/pipelines/dream/sampler.py#L149-L188
[hf-loop]: https://github.com/huggingface/transformers/blob/76d27aea82449e921e36afdc886c6e5bf34ea526/src/transformers/trainer.py#L2606-L2674
[hf-step]: https://github.com/huggingface/transformers/blob/76d27aea82449e921e36afdc886c6e5bf34ea526/src/transformers/trainer.py#L4014-L4071
[hf-gcdefault]: https://github.com/huggingface/transformers/blob/76d27aea82449e921e36afdc886c6e5bf34ea526/src/transformers/training_args.py#L1405-L1415
[hf-gcenable]: https://github.com/huggingface/transformers/blob/76d27aea82449e921e36afdc886c6e5bf34ea526/src/transformers/trainer.py#L2445-L2447
[hf-interval]: https://github.com/huggingface/transformers/blob/76d27aea82449e921e36afdc886c6e5bf34ea526/src/transformers/training_args.py#L1152-L1158
[l-globals]: https://github.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/blob/dc8f91c5f97879d285f64da00d82476b1537edce/code/src/textdiff_port.py#L14-L17
[l-load]: https://github.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/blob/dc8f91c5f97879d285f64da00d82476b1537edce/code/src/textdiff_port.py#L20-L50
[l-qwen]: https://github.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/blob/dc8f91c5f97879d285f64da00d82476b1537edce/code/src/textdiff_port.py#L53-L78
[l-masks]: https://github.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/blob/dc8f91c5f97879d285f64da00d82476b1537edce/code/src/textdiff_port.py#L81-L102
[l-patch]: https://github.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/blob/dc8f91c5f97879d285f64da00d82476b1537edce/code/src/textdiff_port.py#L105-L145
[l-rates]: https://github.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/blob/dc8f91c5f97879d285f64da00d82476b1537edce/code/src/textdiff_port.py#L148-L154
[l-loss]: https://github.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/blob/dc8f91c5f97879d285f64da00d82476b1537edce/code/src/textdiff_port.py#L157-L207
[l-eval]: https://github.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/blob/dc8f91c5f97879d285f64da00d82476b1537edce/code/src/textdiff_port.py#L210-L215
[l-verify]: https://github.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/blob/dc8f91c5f97879d285f64da00d82476b1537edce/code/src/textdiff_port.py#L218-L228
[l-shared]: https://github.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/blob/dc8f91c5f97879d285f64da00d82476b1537edce/code/src/textdiff_port.py#L231-L260
[l-gen]: https://github.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/blob/dc8f91c5f97879d285f64da00d82476b1537edce/code/src/textdiff_port.py#L265-L335
[l-functional]: https://github.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/blob/dc8f91c5f97879d285f64da00d82476b1537edce/code/src/textdiff_port.py#L350-L399
[l-layout]: https://github.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/blob/dc8f91c5f97879d285f64da00d82476b1537edce/code/src/textdiff_port.py#L402-L435
[l-pair]: https://github.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/blob/dc8f91c5f97879d285f64da00d82476b1537edce/code/src/textdiff_port.py#L438-L446
[l-trim]: https://github.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/blob/dc8f91c5f97879d285f64da00d82476b1537edce/code/src/textdiff_port.py#L449-L454
[l-layoutloss]: https://github.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/blob/dc8f91c5f97879d285f64da00d82476b1537edce/code/src/textdiff_port.py#L457-L488
[l-pad]: https://github.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/blob/dc8f91c5f97879d285f64da00d82476b1537edce/code/src/textdiff_port.py#L491
[l-verif]: https://github.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/blob/dc8f91c5f97879d285f64da00d82476b1537edce/code/src/textdiff_port.py#L494-L528
[l-align]: https://github.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/blob/dc8f91c5f97879d285f64da00d82476b1537edce/code/src/textdiff_port.py#L198-L207
[l-encode]: https://github.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/blob/dc8f91c5f97879d285f64da00d82476b1537edce/code/src/textdiff.py#L99-L111
[l-collate]: https://github.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/blob/dc8f91c5f97879d285f64da00d82476b1537edce/code/src/textdiff.py#L338-L357
[l-lora]: https://github.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/blob/dc8f91c5f97879d285f64da00d82476b1537edce/code/src/textdiff.py#L147-L162
[l-traincall]: https://github.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/blob/dc8f91c5f97879d285f64da00d82476b1537edce/code/src/textdiff.py#L407-L408
[l-mezo]: https://github.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/blob/dc8f91c5f97879d285f64da00d82476b1537edce/code/src/textdiff.py#L223-L246
[l-prune]: https://github.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/blob/dc8f91c5f97879d285f64da00d82476b1537edce/code/src/textdiff.py#L292-L320
[l-loop]: https://github.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/blob/dc8f91c5f97879d285f64da00d82476b1537edce/code/src/textdiff.py#L413-L455
[l-validate]: https://github.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/blob/dc8f91c5f97879d285f64da00d82476b1537edce/code/src/textdiff.py#L458-L478
[l-prep]: https://github.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/blob/dc8f91c5f97879d285f64da00d82476b1537edce/code/src/textdiff.py#L44-L89
[l-stopping]: https://github.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/blob/dc8f91c5f97879d285f64da00d82476b1537edce/code/src/diffusion84.py#L188-L253
[l-doc]: https://github.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/blob/dc8f91c5f97879d285f64da00d82476b1537edce/code/docs/textdiff.md#L13-L53
[l-speed]: https://github.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/blob/dc8f91c5f97879d285f64da00d82476b1537edce/code/docs/textdiff-speed-research.md#L1-L45
[m-image]: https://github.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/blob/651d31dd92d9187354440308b11dcce521d99ca4/code/scripts/textdiff_modal.py#L27-L34
[m-wrapper]: https://github.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/blob/651d31dd92d9187354440308b11dcce521d99ca4/code/scripts/textdiff_modal.py#L39-L63
[m-port]: https://github.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/blob/651d31dd92d9187354440308b11dcce521d99ca4/code/src/textdiff_port.py#L20-L44
[m-trainer]: https://github.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/blob/651d31dd92d9187354440308b11dcce521d99ca4/code/src/textdiff.py#L413-L492
[m-price]: https://github.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/blob/651d31dd92d9187354440308b11dcce521d99ca4/code/scripts/textdiff_modal.py#L13-L14
