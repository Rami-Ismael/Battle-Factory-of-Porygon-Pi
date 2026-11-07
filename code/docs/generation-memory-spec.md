---
title: Memory ceiling for generating teams on the laptop
created_at: 2026-11-06
issues: [169, 170, 172, 183]
---

## Rule

A text generation run for the masked text diffusion map (#160) may use at most **19,069,665,280 bytes (17.76 GiB) of GPU memory**. That is the value of `torch.mps.recommended_max_memory()` on the M4 Pro laptop (24 GiB RAM), read on 2026-11-06 with PyTorch 2.14.1 in `venv-dlm`.

Generation has no speed bar. Record the time, but a run neither passes nor fails on it. The owner set this on 2026-11-06: "don't worry about how long it takes, just generate the thing."

## Scope

This covers every laptop run that draws teams: the benchmark (#170), the prompt-only test (#171) and the comparison's generation (#183). Training on Modal is out of scope.

## Enforcement

Put this first in the generation script, before the model loads:

```python
import torch
torch.mps.set_per_process_memory_fraction(1.0)
CEILING = torch.mps.recommended_max_memory()
```

With fraction 1.0, PyTorch's allocator raises an out-of-memory error when a request would exceed `CEILING`, so the run stops instead of swapping. Read `CEILING` at the start of each run, because Metal can report a different value after a macOS update.

## Measurement

- Read `torch.mps.driver_allocated_memory()` once after the model loads and again after every sampling step.
- Report the largest value as **peak GiB per sampler setting**, the unit the owner set for #169 and #170.
- Store `CEILING`, the peak, the batch size, the prompt length in tokens and the weight precision in the run's output.

`driver_allocated_memory()` includes blocks the allocator has cached but isn't using, so it reads at least as high as live tensors. The measurement errs on the safe side.

## Running alone

- Run one generation process at a time. The ceiling applies to each process, and two processes would each get the full 17.76 GiB.
- Battles never overlap generation. Battles start only after every team is generated.
- If another session's job is running on the laptop, the run waits in the queue behind it.

## When a run goes over

Apply these steps in order:

1. **Quantize the 7–8B candidates.** By the owner's rule, quantize only a model that doesn't fit. In bf16, the checkpoints of DiffuLLaMA (13.48 GB), Dream (15.23 GB) and LLaDA (16.03 GB) leave 5.6, 3.8 and 3.0 GB below the ceiling for everything else. The small four stay in bf16.
2. **Lower the batch size, down to 1.** Speed doesn't count, so a slow batch size is allowed.
3. **Rule the model out** if it still goes over at batch size 1 after the first two steps. #170 reports it as not fitting.

## Why not higher

| Limit | Value | What happens above it |
|---|---|---|
| Metal working set (this spec) | 17.76 GiB | macOS starts pushing memory out to make room for the GPU |
| Physical RAM | 24 GiB | Shared with macOS, T3 and other sessions' jobs |
| PyTorch default cap (1.7 × Metal) | 30.19 GiB | PyTorch itself raises the out-of-memory error |

Without the setting, PyTorch's default lets a run reach 30.19 GiB, about 6 GiB more than the laptop's RAM. The rest would come from swap on a disk with 23 GiB free (2026-11-06). PyTorch's own documentation says turning the cap off "may cause system failure". Time is no longer a reason to stay under 17.76 GiB. These are:

- a failed or killed run;
- other sessions' jobs pushed out to swap.

## Not measured

A run between 17.76 and 24 GiB might finish if nothing else were running. No one has tested this. On 2026-11-06 another session's job was running on the laptop, so no probe was run. A probe would raise the fraction step by step on an idle laptop and record swap use and completion at each step. Until a probe shows otherwise, 17.76 GiB is the ceiling.

## Changes this makes to the issues

- #170 was rewritten on 2026-11-06. It is now "Benchmark each candidate's peak memory on the laptop", with this ceiling as its only bar. The old body ruled out models that couldn't "fit in 24 GB beside the battles" and reported hours that included battling.
- #169's speed bar is gone. Its memory bar is this spec.

## Sources

- `torch.mps.recommended_max_memory()` and `sysctl hw.memsize`, read 2026-11-06.
- The docstring of `torch.mps.set_per_process_memory_fraction`: the range is 0 to 2, and 0 means unlimited.
- The checkpoint sizes come from the quantization research branch `research/quantization-masked-diffusion`, 2026-11-06.
