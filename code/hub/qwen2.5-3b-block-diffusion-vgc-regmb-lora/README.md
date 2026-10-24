---
license: other
license_name: qwen-research
license_link: https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/blob/main/LICENSE
base_model: Qwen/Qwen2.5-3B-Instruct
base_model_relation: adapter
library_name: peft
pipeline_tag: text-generation
language:
- en
tags:
- lora
- peft
- diffusion-language-model
- block-diffusion
- pokemon
- vgc
- pokemon-showdown
---

# Qwen2.5-3B block-diffusion LoRA: Pokémon VGC team writer (Champions Regulation M-B)

**Built with Qwen.** This adapter is a derivative of [Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)
and is released under the [Qwen RESEARCH LICENSE AGREEMENT](LICENSE): research and evaluation use only, no commercial use.
See [NOTICE](NOTICE).

A LoRA adapter that turns Qwen2.5-3B-Instruct into a block-diffusion language model that writes a six-Pokémon team as a
Pokémon Showdown paste for `gen9championsvgc2026regmb` (Champions VGC 2026, Regulation Set M-B), conditioned on a target
win rate. The block-diffusion architecture and training loss are those of
[Fast-dLLM v2](https://huggingface.co/Efficient-Large-Model/Fast_dLLM_v2_1.5B) (Apache-2.0): blocks of 32 tokens, decoded
in parallel within a block and left to right across blocks. Unlike Fast-dLLM v2 1.5B, the 3B checkpoint had no diffusion
pretraining; this fine-tune is its only diffusion training. It comes from the
[Battle Factory of Porygon-Pi](https://github.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi) project, which treats VGC team
building as a search problem. The project's earlier 1.5B writer is
[Rami/fast-dllm-v2-1.5b-vgc-regmb-lora](https://huggingface.co/Rami/fast-dllm-v2-1.5b-vgc-regmb-lora).

## Prompt and output

```
system: You write Pokémon VGC teams as Showdown pastes.
user:   Format: gen9championsvgc2026regmb. Six Pokémon. Target win rate vs top meta teams: 0.50.
```

The target is a win rate to two decimals, or `unknown`. The answer is six blocks of species @ item, ability, nature and
four moves, without Stat Points or level. Add spreads yourself and validate the team with Showdown before use.

## How to use

The adapter needs Fast-dLLM v2's block-diffusion model class with Qwen2.5-3B-Instruct's weights in it.
[`load_vgc_writer.py`](load_vgc_writer.py) builds that, merges the adapter, and includes a batched sampler:

```python
from huggingface_hub import hf_hub_download
import importlib.util
path = hf_hub_download("Rami/qwen2.5-3b-block-diffusion-vgc-regmb-lora", "load_vgc_writer.py")
spec = importlib.util.spec_from_file_location("load_vgc_writer", path); W = importlib.util.module_from_spec(spec); spec.loader.exec_module(W)

tok, model = W.load("cuda")                              # or "mps" on Apple silicon
ids = W.prompt_ids(tok, None, "cuda")                    # target win rate unknown; or e.g. 0.5
teams = [tok.decode(t, skip_special_tokens=True) for t in W.generate_batch(model, ids, 64, block_cache=True)]
print(teams[0])
```

`generate_batch` writes many teams of one prompt per call. With one row, greedy decoding and `exact=True` it reproduces
the released `model.generate` token for token (checked with and without the block cache). The released sampler samples
only its first row, and stops every row at the first stop token it sees, so it writes one team per call.

## Training

- **Data.** 8,000 teams, one epoch: 622 real teams from the project's corpus (Pikalytics, Limitless, VGCPastes),
  5,192 teams from the project's own search runs, and 2,186 synthetic legal teams. Each team's prompt carries its
  battle-measured win rate, or `unknown` for unlabelled teams and for 15% of labelled ones.
- **Recipe.** The Fast-dLLM v2 loss (blocks of 32 tokens, complementary masks, token shift), loss on the answer tokens
  only. LoRA rank 32, alpha 64, dropout 0, on every attention and MLP projection, and the `[MASK]` embedding row
  (id 151665, unused in the Qwen2.5 checkpoint) trained: 59.9M trainable parameters, 1.9%. AdamW, learning rate 2e-4
  with 3% warm-up and linear decay, batch 4, 2,000 steps, bfloat16, no gradient checkpointing. 14 minutes on one
  NVIDIA H100 (Modal).
- **Validation loss** (mean cross-entropy per masked answer token, 256 held-out teams; same data, recipe and seed for
  every row):

| model | step 500 | 1,000 | 1,500 | 2,000 |
|---|---|---|---|---|
| this adapter (Qwen2.5-3B + recipe) | 0.980 | 0.832 | 0.720 | **0.682** |
| Qwen2.5-1.5B + the same recipe | 1.080 | 0.879 | 0.765 | 0.714 |
| Fast-dLLM v2 1.5B (diffusion-pretrained) | 0.949 | 0.803 | 0.707 | 0.667 |

## Evaluation

**Raw legality** (64 samples per setting at target `unknown`, temperature 1, parsed and checked with the Showdown
validator, no repair): 27 to 37 of 64 in every sampler setting below, about 50%. That is the 1.5B writer's rate at the same
target. Failures are mostly invented formes (`Alcremie-Mega`), Item Clause, and moves or abilities a Pokémon cannot have.

**Generation speed** (one NVIDIA H100, bfloat16, 64 teams per setting, about 204 tokens per team):

| sampler | seconds per team | teams per hour |
|---|---|---|
| released `model.generate`, one team per call | 3.65 | 990 |
| `generate_batch`, 1 team per call | 3.30 | 1,090 |
| `generate_batch`, 8 per call | 0.88 | 4,100 |
| `generate_batch`, 32 per call | 0.24 | 14,700 |
| `generate_batch`, 64 per call | 0.17 | 21,100 |
| `generate_batch`, 64 per call, block cache | 0.16 | 22,500 |

All rows use the released confidence threshold, 0.9. At one team per call the H100 spends about 24 ms per forward pass
on a 32-token block, which is launch and synchronisation overhead rather than arithmetic, so batching is the lever. The
block cache (recompute only the current 8-token sub-block) saves arithmetic the H100 is not short of, so it changes
little here. Separate H100 containers differed by up to 25% on the same setting; read smaller gaps as noise.

A lower threshold does not pay in a batch: rows move through each 8-token sub-block together, so a step waits for the
least confident row. At 64 per call, thresholds 0.7 and 0.5 ran at 0.13 to 0.16 s per team, no clear gain over 0.9, and
0.5 broke the paste structure (4 and 10 legal of 64).

**Win rate** (wins over battles against the top-50 meta teams of Regulation M-B, one battle per opponent, the
[VGC-Bench](https://huggingface.co/cameronangliss/vgc-bench-models) behaviour-cloning policy on both sides; 64
Showdown-valid teams per setting, standard error over teams; a top-50 meta team scores 0.531):

| setting | this adapter | 1.5B writer | scratch diffusion model |
|---|---|---|---|
| target `unknown` | 0.442 ± 0.014 | 0.440 ± 0.013 | 0.095 ± 0.016 |
| target 0.50 | 0.472 ± 0.010 | 0.452 ± 0.012 | |
| target 0.60 | 0.452 ± 0.014 | 0.459 ± 0.011 | |
| target 0.50, classifier-free guidance 2 | 0.444 ± 0.013 | 0.463 ± 0.013 | 0.378 ± 0.015 (guided) |

The 3B writer is level with the 1.5B writer (every difference within about 1σ) and far above the scratch model. It
writes fewer distinct species sets (7 to 29 per 64 teams, against 10 to 33).

**On Apple silicon** (M4 Pro, 64 teams): 14.6 s per team with the released sampler, 5.7 s with `generate_batch` at 8 per
call and the block cache. The Mac is compute-bound, so the cache helps there and batching past 8 does not.

## Limitations

- No learnset or legality constraints: about half the samples are not legal teams. Validate every team.
- Trained for one regulation; other formats are out of scope.
- Research and evaluation use only (Qwen RESEARCH LICENSE AGREEMENT).
