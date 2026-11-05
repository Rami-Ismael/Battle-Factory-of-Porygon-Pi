# Specialized dLLM samplers

## Which specialized inference samplers are shipped?

### Takeaway
At source commit `ca176752fbceec49c6b4777a2c18ae88e4eb10ed`, four specialized sampler classes outside the core and Dream families are present. A sampler is the procedure that repeatedly asks a model for predictions and decides which tokens to put into the evolving sequence. These are inference procedures, not the later conversion of token IDs into a Pokémon team.

### Cited Findings
- The exact requested directory `dllm/pipelines/fastdllm/llada` exists. It contains six files: `__init__.py`, `eval.py`, `sampler.py`, `models/__init__.py`, `models/configuration_llada.py`, `models/modeling_llada.py`. The package exports `FastdLLMLLaDASampler` and the cache-aware model, and registers the model with masked-language-model loading. [Directory](https://github.com/ZHZisZZ/dllm/tree/ca176752fbceec49c6b4777a2c18ae88e4eb10ed/dllm/pipelines/fastdllm/llada), [exports and registration, lines 1–20](https://github.com/ZHZisZZ/dllm/blob/ca176752fbceec49c6b4777a2c18ae88e4eb10ed/dllm/pipelines/fastdllm/llada/__init__.py#L1-L19).
- Ordinary LLaDA evaluation uses `MDLMSampler`; it is not an additional independent LLaDA sampler class. [eval.py lines 17–55](https://github.com/ZHZisZZ/dllm/blob/ca176752fbceec49c6b4777a2c18ae88e4eb10ed/dllm/pipelines/llada/eval.py#L17-L55).
- Other specialized concrete classes are `LLaDA2Sampler`, `LLaDA21Sampler`, and `EditFlowSampler`. [LLaDA2 lines 83–107](https://github.com/ZHZisZZ/dllm/blob/ca176752fbceec49c6b4777a2c18ae88e4eb10ed/dllm/pipelines/llada2/sampler.py#L83-L107), [LLaDA2.1 lines 75–111](https://github.com/ZHZisZZ/dllm/blob/ca176752fbceec49c6b4777a2c18ae88e4eb10ed/dllm/pipelines/llada21/sampler.py#L75-L111), [EditFlow lines 99–117](https://github.com/ZHZisZZ/dllm/blob/ca176752fbceec49c6b4777a2c18ae88e4eb10ed/dllm/pipelines/editflow/sampler.py#L99-L117).
- Source includes embedded model `generate()` methods for LLaDA2 and LLaDA2.1. Count these as alternate entry points implementing the same respective sampler families, not two additional algorithms. Defaults and output/EOS handling differ from standalone wrappers: embedded generation length is 2048, standalone defaults to 128. [LLaDA2 model lines 1272–1286 and 1382–1435](https://github.com/ZHZisZZ/dllm/blob/ca176752fbceec49c6b4777a2c18ae88e4eb10ed/dllm/pipelines/llada2/models/modeling_llada2_moe.py#L1272-L1435), [LLaDA2.1 model lines 1240–1257](https://github.com/ZHZisZZ/dllm/blob/ca176752fbceec49c6b4777a2c18ae88e4eb10ed/dllm/pipelines/llada21/models/modeling_llada21_moe.py#L1240-L1257).
- `X0Sampler`, `SampleX0Empty`, and `SampleX0Masks` are EditFlow training-collator source distributions. They return an empty list or a chosen number of mask IDs, not a model-guided inference loop. The empty sampler's docstring says BOS-only but actual return is `[]`. [utils.py lines 14–68](https://github.com/ZHZisZZ/dllm/blob/ca176752fbceec49c6b4777a2c18ae88e4eb10ed/dllm/pipelines/editflow/utils.py#L14-L68).

### Inferences
- Distinguish concrete inference classes, settings that choose modes inside a class, duplicate entry points, and training-data samplers. Listing every class containing “Sampler” as an independent generation algorithm overcounts.

### Gaps
- No execution was performed. Existence and static behavior are verified, runtime correctness is not.

## What happens at one step, and what settings and costs matter?

### Takeaway
Fast-dLLM LLaDA fills selected masks and offers approximate cache reuse; LLaDA2 fills confidence-selected masks in causal blocks; LLaDA2.1 can also replace placed tokens; EditFlow can insert, delete, or replace tokens and therefore change length. Confidence is a model-assigned probability, not a legality check.

### Cited Findings

#### Confidence-selected masked filling with cache choices, Fast-dLLM LLaDA

- Defaults: `max_new_tokens=128`, `max_length=None`, `block_size=128`, `steps=128`, `temperature=0`, `remasking="low_confidence"`, `stochastic_transfer=False`, `use_cache=None`, `threshold=None`, `factor=None`, `right_shift_logits=False`, suppression lists `None`. `cfg_scale=0` and `cfg_keep_tokens=None` exist in config but are not consumed in sampling; config comments explicitly call scale unused. [lines 157–217](https://github.com/ZHZisZZ/dllm/blob/ca176752fbceec49c6b4777a2c18ae88e4eb10ed/dllm/pipelines/fastdllm/llada/sampler.py#L157-L217).
- One forward proposes tokens with greedy or Gumbel-noisy argmax; position scores are either proposed-token softmax probabilities or independent random numbers. Only current masks can change, so “remasking” here names position ranking, not restoration of an already placed token to MASK. Mode priority is `factor` before `threshold` before quota. [lines 32–74](https://github.com/ZHZisZZ/dllm/blob/ca176752fbceec49c6b4777a2c18ae88e4eb10ed/dllm/pipelines/fastdllm/llada/sampler.py#L32-L74).
- Quota mode takes the scheduled number of highest-score masked positions. Threshold mode takes scores at least the threshold and forces one best mask if none qualify. Dynamic-factor mode sorts scores, compares rank `r` to `1-factor/(r+1)`, forces rank one eligible, counts accepted ranks, then takes that many top scores. [lines 76–153](https://github.com/ZHZisZZ/dllm/blob/ca176752fbceec49c6b4777a2c18ae88e4eb10ed/dllm/pipelines/fastdllm/llada/sampler.py#L76-L153).
- No cache recomputes the full canvas per iteration. Prefix cache performs a full warm pass each block, stores preceding keys/values, then forwards the suffix. Dual cache performs a full warm pass then forwards only the current block with a `replace_position` mask. Cache modes require equal prompt lengths in a batch. Dual cache requires model support for `replace_position`, so a generic masked model is not automatically cache-compatible. [lines 291–313](https://github.com/ZHZisZZ/dllm/blob/ca176752fbceec49c6b4777a2c18ae88e4eb10ed/dllm/pipelines/fastdllm/llada/sampler.py#L291-L313), [no cache lines 373–415](https://github.com/ZHZisZZ/dllm/blob/ca176752fbceec49c6b4777a2c18ae88e4eb10ed/dllm/pipelines/fastdllm/llada/sampler.py#L373-L415), [prefix lines 420–512](https://github.com/ZHZisZZ/dllm/blob/ca176752fbceec49c6b4777a2c18ae88e4eb10ed/dllm/pipelines/fastdllm/llada/sampler.py#L420-L512), [dual lines 517–603](https://github.com/ZHZisZZ/dllm/blob/ca176752fbceec49c6b4777a2c18ae88e4eb10ed/dllm/pipelines/fastdllm/llada/sampler.py#L517-L603).
- Cost driver: number of forwards until masks are filled, canvas/suffix/block length per forward, cache memory and warm passes, batching, and confidence settings. `num_blocks=ceil(max_new_tokens/block_size)` and `steps_per_block=ceil(steps/num_blocks)`. No-cache/prefix loops run until masks are absent, while dual-cache uses `range(1,effective_steps)` after the warm pass. [lines 291–294, 361–368, 373–410, 559–563](https://github.com/ZHZisZZ/dllm/blob/ca176752fbceec49c6b4777a2c18ae88e4eb10ed/dllm/pipelines/fastdllm/llada/sampler.py#L291-L563).

Toy step, schematic one-word tokens, invented predictions rather than a model run: `Team: [MASK] [MASK] [MASK]`, proposals `Incineroar Rillaboom Flutter`, scores `.98 .80 .60`. Quota 2 gives `Team: Incineroar Rillaboom [MASK]`; threshold `.9` gives `Team: Incineroar [MASK] [MASK]`; dynamic factor `1` gives the quota-2 result, since rank thresholds are `-1, .667, .75`. With random position scores `.1 .2 .9`, quota 1 instead gives `Team: [MASK] [MASK] Flutter`. These examples follow the cited selection code. Cache variants use the same selection rule but different contexts may produce different proposals.

#### Confidence-threshold block filling, LLaDA2

- Defaults: `max_new_tokens=128`, `block_size=32`, `steps_per_block=32`, `temperature=0`, `top_p=None`, `top_k=None`, `threshold=.95`, `minimal_topk=1`, `eos_early_stop=False`. Equal-length prompts required. Temperature zero is greedy and bypasses top-k/top-p vocabulary filtering. At positive temperature, divide logits by temperature, filter top-k/top-p, sample tokens, and use their filtered distribution probabilities as confidence. [lines 32–92](https://github.com/ZHZisZZ/dllm/blob/ca176752fbceec49c6b4777a2c18ae88e4eb10ed/dllm/pipelines/llada2/sampler.py#L32-L92), [lines 111–142](https://github.com/ZHZisZZ/dllm/blob/ca176752fbceec49c6b4777a2c18ae88e4eb10ed/dllm/pipelines/llada2/sampler.py#L111-L142).
- Uses block-causal attention, meaning a token can see preceding blocks and all of its own block, but no later blocks. Forwards the growing full prefix through the current block, with no explicit cache. Even transfer quotas divide block size across configured steps. Accepts all active positions above the threshold if they reach the quota; otherwise accepts the quota's highest-confidence masks. Placed tokens are unchanged. [lines 144–219](https://github.com/ZHZisZZ/dllm/blob/ca176752fbceec49c6b4777a2c18ae88e4eb10ed/dllm/pipelines/llada2/sampler.py#L144-L219).
- Cost is at most configured effective steps per processed block, with early exit when a block has no masks; each pass sees a growing prefix. Canvas length rounds prompt-plus-request to whole blocks. `eos_early_stop` in standalone code breaks the inner block loop when any EOS appears, so it is not the same behavior as terminating the whole generation method. `infill` is unimplemented. [lines 139–145, 179–239](https://github.com/ZHZisZZ/dllm/blob/ca176752fbceec49c6b4777a2c18ae88e4eb10ed/dllm/pipelines/llada2/sampler.py#L139-L239).

Toy step: current block `Team: [MASK] [MASK]`, predictions `Incineroar Rillaboom` with `.98 .97`, threshold `.95`, scheduled quota 1. Both qualify, yielding `Team: Incineroar Rillaboom` in one pass. This is a mask-filling algorithm, but the shipped model recipe assumes LLaDA2 block attention/training; executing its attention mask on an arbitrary full-sequence masked checkpoint is not evidence of matched training or preserved quality.

#### Mask filling with placed-token editing, LLaDA2.1

- Defaults: `max_new_tokens=128`, `max_length=None`, `block_size=32`, `temperature=0`, `top_p=None`, `top_k=None`, `threshold=.95`, `editing_threshold=.9`, `max_post_steps=16`, `num_to_transfer=1`, `eos_early_stop=False`. Batch size must be one. It also uses block-causal attention and forwards the current growing prefix. [lines 75–86](https://github.com/ZHZisZZ/dllm/blob/ca176752fbceec49c6b4777a2c18ae88e4eb10ed/dllm/pipelines/llada21/sampler.py#L75-L86), [lines 138–174](https://github.com/ZHZisZZ/dllm/blob/ca176752fbceec49c6b4777a2c18ae88e4eb10ed/dllm/pipelines/llada21/sampler.py#L138-L174).
- Accepts high-confidence mask predictions, with top-count fallback. In the same pass it replaces already placed non-prompt tokens if a different proposed token exceeds the editing threshold. No insertion/deletion, so length stays fixed. After no masks remain it permits up to `max_post_steps` editing iterations, ending sooner when no edits occur. The mask-filling phase has no explicit global step cap. [lines 198–278](https://github.com/ZHZisZZ/dllm/blob/ca176752fbceec49c6b4777a2c18ae88e4eb10ed/dllm/pipelines/llada21/sampler.py#L198-L278).

Toy step: `Team: Pikachu [MASK]`, with `Team:` protected prompt, proposals `Incineroar Rillaboom` for the last two positions, confidence `.92 .96`. Default editing and filling thresholds yield `Team: Incineroar Rillaboom`. This requires a checkpoint whose predictions on already filled tokens can perform useful edits. The LLaDA2.1 paper explicitly pairs mask-to-token generation with token-to-token editing; transplanting the loop does not confer that capability on a generic masked-only checkpoint. [Paper abstract](https://arxiv.org/abs/2602.08676v3).

#### Insert delete and substitute editing, EditFlow

- Defaults: `tau=.01`, `time_epsilon=.001`, `mask_length=128`, `temperature=0`, `edit_prompt=False`, `time_independent=True`; batch one only, beginning-of-sequence token required. A mask tail is optional. [lines 99–169](https://github.com/ZHZisZZ/dllm/blob/ca176752fbceec49c6b4777a2c18ae88e4eb10ed/dllm/pipelines/editflow/sampler.py#L99-L169).
- A time step, called a tau leap, asks the model for deletion/substitution/insertion rates and vocabulary predictions for substitutions and insertions, scales rates by a time scheduler, samples event occurrence with probabilities `clamp(rate*tau,0,1-1e-6)`, chooses deletion versus substitution, and applies these edits plus insertions after positions. Thus temperature zero still leaves edit-event randomness. Prompt edits are suppressed by default. [lines 10–95](https://github.com/ZHZisZZ/dllm/blob/ca176752fbceec49c6b4777a2c18ae88e4eb10ed/dllm/pipelines/editflow/sampler.py#L10-L95).
- The model needs additional outputs `del_rate_hat`, `sub_rate_hat`, `ins_rate_hat`, `sub_logits`, `ins_logits`. These are not the ordinary vocabulary logits of an unchanged masked checkpoint. The LLaDA EditFlow implementation adds corresponding heads. [sampler lines 48–52](https://github.com/ZHZisZZ/dllm/blob/ca176752fbceec49c6b4777a2c18ae88e4eb10ed/dllm/pipelines/editflow/sampler.py#L48-L52), [model lines 9–59](https://github.com/ZHZisZZ/dllm/blob/ca176752fbceec49c6b4777a2c18ae88e4eb10ed/dllm/pipelines/editflow/models/llada/modelling_llada.py#L9-L59).
- Cost driver is up to `ceil(1/tau)` time steps at the evolving length, with potential reuse of previous model outputs when no edits occurred and `time_independent=True`. This reuse is output reuse, not a KV cache. [lines 169–197](https://github.com/ZHZisZZ/dllm/blob/ca176752fbceec49c6b4777a2c18ae88e4eb10ed/dllm/pipelines/editflow/sampler.py#L169-L197).

Toy step: `Team: [MASK] junk`. Suppose a substitution event on MASK samples `Incineroar`, deletion fires on `junk`, and insertion after the MASK position samples `Rillaboom`. Result is `Team: Incineroar Rillaboom`. Events are illustrative. No event probabilities or model predictions were measured.

### Inferences
- These examples should be drawn by the parent session; they are not literal tokenizer traces. Real Pokémon names can span several tokens.
- No sampler listed here encodes Showdown species/move/item/format legality. Confidence measures the model distribution, so even `.99` does not certify legal output.
- Smaller quotas or stricter thresholds may reduce simultaneous inconsistent choices but spend more forwards. Lower temperature or more restrictive vocabulary filtering may reduce output variation. Whether this improves legal, novel six-species teams per second is unknown without evaluation.
- Editing can repair or introduce a cross-field contradiction. EditFlow can change paste structure as well as content; that extra flexibility offers no parser or legality guarantee.
- Per-team seconds cannot be inferred from count of diffusion steps alone. Model size, prefix length, cache memory, quantization, backend, and acceptance rate matter on the M4 Pro. The reported GPU throughput claims are not laptop forecasts.

### Gaps
- All runtime behavior, memory use, M4 Pro compatibility, and useful-model applicability outside the named recipes remain unverified.
- Fast LLaDA has implementation hazards requiring a later runtime/static review: factor mode overrides quota in selection but some callers still index quota arrays; dual-cache refinement is bounded by scheduled effective steps rather than running until masks disappear. Mask proposals are not explicitly prohibited from selecting MASK again. Consequently no unconditional completion bound is established.
- LLaDA2 and LLaDA2.1 wrappers round lengths to blocks; standalone results are not truncated in their return statements. Their equivalence to embedded `generate()` methods, especially EOS and output shape, was not proven.
- EditFlow output reuse is conditioned on `time_independent=True`; the quality and correctness of enabling it for a time-dependent checkpoint were not verified.

## What do the papers claim separately from code evidence?

### Takeaway
The papers motivate these procedures and report their own evaluations. Those claims do not establish legality, diversity, or throughput for this project's fine-tunes.

### Cited Findings
- Fast-dLLM describes approximate blockwise cache reuse and confidence-threshold parallel generation; its abstract claims up to 27.6-fold throughput improvement on LLaDA/Dream benchmark configurations with minimal accuracy loss. This is the paper's claim, not a measured property of this repository on Apple silicon. [Fast-dLLM v3 abstract](https://arxiv.org/abs/2505.22618v3).
- LLaDA2.0 describes conversion of autoregressive models through training stages ending in compact-block diffusion and releases 16B and 100B mixture-of-experts models. This supports treating block masking as part of a trained recipe, rather than a free universally valid sampler swap. [LLaDA2.0 v2 abstract](https://arxiv.org/abs/2512.15745v2).
- LLaDA2.1 describes configurable filling/editing thresholds, with faster filling relying on later token edits, and reports its own coding benchmark throughput. No such benchmark was run here. [LLaDA2.1 v3 abstract](https://arxiv.org/abs/2602.08676v3).
- Edit Flows defines sequence generation using insertion, deletion and substitution and reports gains over masked construction on its text/code evaluations. This note verifies the matching operation types in dLLM source, not a faithful reproduction of every paper detail or result. [Edit Flows v3 abstract](https://arxiv.org/abs/2506.09018v3).

### Inferences
- For this project, the correct next research comparison is accepted, diverse teams per second under a fixed checkpoint and clear training compatibility, not advertised raw tokens per second across different models and hardware. This is a proposed comparison, not an executed experiment.

### Gaps
- Paper inspection here is abstract-level only. Full experimental conditions, numerical reproduction, and exact algorithmic equivalence of the dLLM ports to all paper equations were not verified.
- I did not identify a blog in this subtopic and therefore make no reading-list addition recommendation.
