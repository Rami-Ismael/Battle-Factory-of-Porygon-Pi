# DiBO reproduction feasibility and cost

## Which sources and datasets can be trusted for this task?

### Takeaway
The two suggested project notes describe the correct Sun et al. paper, not a different DiBO. They rely substantially on v1 and must not override v3 or the current released code. Data access has improved: the pinned DiBO repository includes training bundles, and its separate public oracle snapshot responds successfully to a read-only HTTP request.

### Cited Findings
- The evidence-and-mechanisms note identifies Sun et al., arXiv 2603.17919, and zpointS/DiBO, explicitly labels its original account v1, and has a September update directing readers to v3. Its old TFBind8 result of .965 is historical, whereas v3 Table 1 reports .946±.043. Sources: [local note lines65–73](../../notes/Papers/LLMs%20as%20black-box%20optimizers%20%E2%80%94%20evidence%20and%20mechanisms.md), [v3 Table1](https://arxiv.org/html/2603.17919v3#S4.T1).
- The diagram identifies the same title and v1 URL at lines11,61. Its line54 says iterative unmasking, whereas v3 §4.3 specifies one forward pass with greedy filling. It correctly distinguishes the earlier dLLM tree-search paper at line56. This is a stale mechanism detail, not evidence of a different DiBO. Sources: [diagram text](../../notes/Papers/DiBO%20-%20visual%20explanation.excalidraw.md), [v3 §4.3](https://arxiv.org/html/2603.17919v3#S4.SS3).
- The prepared task bundles preserve filtering, score alignment, sorting and normalization and avoid Design-Bench during training. Morphology bundles use relabeled exact-oracle scores, with original labels retained. Sources: [data README lines24–34,64–103](https://github.com/zpointS/DiBO/blob/601fef5d502d0088a32841a477cb221beff105ef/data/README.md#L24-L103), [installation lines68–71](https://github.com/zpointS/DiBO/blob/601fef5d502d0088a32841a477cb221beff105ef/docs/installation.md#L68-L71).
- The separate snapshot is for Design-Bench evaluation, including fixed morphology controller policies. TFBind8 uses Design-Bench; Ant additionally requires MuJoCo2.0 and OpenGL. Sources: [data README lines36–62](https://github.com/zpointS/DiBO/blob/601fef5d502d0088a32841a477cb221beff105ef/data/README.md#L36-L62), [installation lines78–84](https://github.com/zpointS/DiBO/blob/601fef5d502d0088a32841a477cb221beff105ef/docs/installation.md#L78-L84).
- Read-only API check on 2026-11-05: tag v1.0.0 resolves to `afc6ac0b2d8425146595d9ffdfeb33f0ba5d3ae3`, public/ungated/active. HEAD with redirect following ends HTTP200 for the ZIP, reports 142,923,783 bytes and linked SHA256 `6a642d8b78f79ffb82ee6d7b73e2c4569c92e2f31a325cea32c4e7ef68e32e81`. These agree with the card's size/hash. Sources: [revision API](https://huggingface.co/api/datasets/zpointsun/DiBO-DesignBench-Snapshot/revision/v1.0.0), [archive URL](https://huggingface.co/datasets/zpointsun/DiBO-DesignBench-Snapshot/resolve/v1.0.0/design_bench_data.zip), [snapshot card](https://huggingface.co/datasets/zpointsun/DiBO-DesignBench-Snapshot). The card claims byte-identical overlap with a community mirror; that comparison was not independently rerun.
- Pinned LLaDA-8B-Instruct revision `08b83a6feb34df1a6011b80c3c00c7563e963b07` has a publicly readable file listing, approximately16GB in six safetensors shards. Source: [base model files](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct/tree/08b83a6feb34df1a6011b80c3c00c7563e963b07).

### Inferences
- Do not use the generic Design-Bench downloader as the only acquisition plan. Use released bundles for training and the exact snapshot for evaluation; this avoids relying on upstream data-host availability.
- A working HTTP header response verifies accessibility, not full archive integrity or functional oracle compatibility.

### Gaps
- No full snapshot or model download, checksum recomputation, installation, archive extraction, simulator execution or oracle test was performed. Upstream Design-Bench downloader health was not independently tested. The Hugging Face dataset viewer displays a streaming error, which does not imply its ZIP download is broken.

## What commands and machine are required?

### Takeaway
Use the CUDA/H100 training recipe for two separately trained boundary formats, each DA→SFT only. The supplied shell wrapper includes RL, so do not run it unchanged for this task. The laptop can perform source review, file transfer, hashing and reporting now; released training is not an Apple-silicon recipe.

### Cited Findings
- Tested core stack: Python3.10.12, torch2.6.0+cu118, transformers4.44.2, tokenizers.19.1, NumPy1.26.4, accelerate1.13.0, bitsandbytes.49.0, BF16 NVIDIA GPU. Oracle stack additionally pins Design-Bench2.0.20, TensorFlow2.17.0, Keras3.12.0, DeepChem2.8.0, Gym.13.1, mujoco-py2.0.2.13, morphing-agents1.5.1 and a ROBEL commit. Source: [installation lines3–44,119–154](https://github.com/zpointS/DiBO/blob/601fef5d502d0088a32841a477cb221beff105ef/docs/installation.md#L3-L154).
- `main.py --stages da sft` excludes RL, `--special_token_type` accepts special/natural, and stage-specific steps/rates are CLI options. Source: [main.py lines15–68](https://github.com/zpointS/DiBO/blob/601fef5d502d0088a32841a477cb221beff105ef/main.py#L15-L68). The wrapper hardcodes da sft rl: [run_pipeline.sh lines16–52](https://github.com/zpointS/DiBO/blob/601fef5d502d0088a32841a477cb221beff105ef/scripts/run_pipeline.sh#L16-L52).
- Sequential stages retain the same model object; pipeline manifests record stage duration and final checkpoint path. Source: [pipeline.py lines237–350](https://github.com/zpointS/DiBO/blob/601fef5d502d0088a32841a477cb221beff105ef/src/pipeline.py#L237-L350).
- Evaluation supports corresponding natural/special rendering, explicit caller-selected seeds, 128 candidates and1000 maximum attempts. The default effective context seed is the square of the input seed; record that transform. Source: [eval.py lines511–565](https://github.com/zpointS/DiBO/blob/601fef5d502d0088a32841a477cb221beff105ef/eval.py#L511-L565).
- The public task checkpoints are final DA+SFT+RL models, not the two Table3 SFT ablation checkpoints. Source: [README lines134–162](https://github.com/zpointS/DiBO/blob/601fef5d502d0088a32841a477cb221beff105ef/README.md#L134-L162).

### Inferences
Commands below are a reviewed proposal, never executed. They assume the audited pinned repository, its documented environment, model access and oracle snapshot have already been prepared. `SEED` and `FORMAT` are supplied explicitly by the future operator. Use eight matched seeds for a paper-scale rerun; exact historical seed IDs are not established.

```bash
# FORMAT is natural or special. SEED is one explicitly recorded integer.
python main.py \
  --use TFBind8-Exact-v0 --stages da sft \
  --seed "$SEED" --special_token_type "$FORMAT" \
  --da_lr 2e-5 --sft_lr 2e-5 --da_steps 1024 --sft_steps 1024 \
  --batch_size 1 --grad_accum_steps 16 --gradient_checkpointing \
  --warmup_steps 100 --lr_schedule warmup_constant \
  --checkpoint_policy last \
  --output_dir "outputs/tfbind8-${FORMAT}-${SEED}" --wandb_mode offline

# Read the exact final_checkpoint value for the SFT stage from pipeline_manifest.json.
python eval.py --tasks TFBind8-Exact-v0 \
  --checkpoint_path "$SFT_CHECKPOINT" \
  --seeds "$SEED" --special_token_type "$FORMAT" \
  --num_candidates 128 --max_attempts 1000 --save_details \
  --output_json "outputs/tfbind8-${FORMAT}-${SEED}-eval.json"
```

For Ant replace task with `AntMorphology-Exact-v0`, DA learning rate with1e-5, DA steps with2048, gradient accumulation with8, and use distinct Ant output paths. Install its MuJoCo dependencies first. Do not reuse one arm's DA checkpoint for the other boundary format: the ablation changes input representation throughout DA and SFT.

The M4Pro24GB laptop can inspect bundles, prepare text, download/hash artifacts and aggregate results. TFBind8 oracle lookup itself is modest CPU work in principle. Running full LLaDA inference on the laptop would require an independently verified Apple-silicon adaptation; the repo documents BF16 NVIDIA evaluation and its evaluator defaults to CUDA or CPU, not MPS. The16GB BF16 weights alone leave less than8GB of24GB unified memory for runtime, model activations and the OS. This is a memory-budget observation, not a measured inference limit or speed estimate. The parent map requires generation/battles on the laptop, so inability to verify that path remains a project gate, not permission to move generation to paid GPUs.

### Gaps
- No command above has been executed or validated end to end. Exact original seed values and whether Table3 seed repetitions retrained weights or only resampled inference contexts were not established. Eight independently trained seeds is a conservative fresh experimental design, not a recovered historical fact.
- No Apple-silicon inference throughput or memory measurement exists in this research. No claim that Ant's legacy simulator installs on this Mac.

## Which Table3 pair costs less, and how many GPU hours?

### Takeaway
Recommend the TFBind8 natural-boundary versus special-delimiter pair after DA→SFT. It avoids Ant's legacy MuJoCo oracle and has short8-base outputs; it is the lower-friction reproduction. Exact training/inference cost differences are unmeasured, and the Table3 gains are smaller than or comparable with seed spread.

### Cited Findings
- Table3 is the delimiter ablation, not a no-DA versus DA comparison. At SFT, TFBind8 natural .918±.043 versus delimiter .939±.031, Ant natural .903±.022 versus delimiter .929±.017. Source: [v3 Table3 and §4.5](https://arxiv.org/html/2603.17919v3#S4.T3).
- The paper reports the entire TFBind8 DA→SFT→RL pipeline within1.5h on a single H10080GB, eight random seeds, but no per-stage or Ant wall times. DA/SFT sample counts are TFBind8 16,384+16,384, Ant16,384+8,192. Source: [v3 §4.3](https://arxiv.org/html/2603.17919v3#S4.SS3).
- Ant designs have60 continuous parameters versus8 categorical bases for TFBind8. Source: [v3 §4.1](https://arxiv.org/html/2603.17919v3#S4.SS1).

### Inferences
All numbers below are estimates/conditional arithmetic, not new measurements.

- One matched TFBind8 pair entails2 training jobs. If each DA→SFT job fits under the paper's1.5h full-pipeline example on the same H100 with comparable implementation/settings, its training-only envelope is2×1.5=3 H100 GPU-hours.
- Eight independently trained matched seeds entail2×8=16 jobs. Under that same assumption,16×1.5=24 H100 GPU-hours of training. This is not an assured upper bound for the released implementation or the longer natural-boundary arm. The paper gives no timing for that ablation, and inference, setup, downloads, validation and failed/repeated jobs are extra.
- Hardware-priced estimate is `training GPU-hours × actual quoted H100 hourly price`. Example only: at an assumed $4/H100h,3h costs$12 and24h costs$96, excluding other charges. This is not a current vendor quote or permission to launch compute. No Modal account or pricing dashboard was used.
- More defensible future estimate: measure training microbatch elapsed times `t_DA` and `t_SFT` in seconds in an authorized probe. TFBind8 per-arm training h ≈ `(1024×16×t_DA +1024×16×t_SFT)/3600 + validation/checkpoint overhead`; Ant ≈ `(2048×8×t_DA +1024×8×t_SFT)/3600 + overhead`. Multiply each arm's distinct timing by chosen training seed count, then add separately measured generation/oracle costs. Natural tokenization can lengthen inputs; Ant has fewer SFT examples but longer sequences. Sample counts alone do not prove TFBind8 trains faster.
- If the historical eight seeds were only inference seeds from one trained model per arm, two jobs plus16 evaluations would be the closer workload. Without the original seed/checkpoint manifest, do not pretend24h is the uniquely correct reproduction total.
- TFBind8 improvement .021 is below each arm's SD and pooled independent-score SD `sqrt(.043²+.031²)=.0530`. Ant improvement .026 is about the pooled SD `sqrt(.022²+.017²)=.0278`. Mean±SD bands overlap in both cases. Neither establishes significance without per-seed paired data/covariance or an appropriate preregistered statistical test.

### Gaps
No actual GPU run, timing, dollar charge, oracle execution, seed-level result recovery or statistical significance test was performed. The paper's1.5h claim was not independently reproduced. Ant total hours and laptop evaluation costs remain unknown. This research found no need to invoke paid compute.
