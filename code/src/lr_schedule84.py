"""Learning-rate schedules shared by the 84-field trainers, diffusion84.py and uniform_diffusion84.py (issue #431).

  cosine        Every run before #431: linear warm-up from (step + 1) / warmup_steps, multiplied by a cosine that
                starts at step 0 and reaches 0 at total_steps. No floor. Kept bit-for-bit, so reruns match.
  gidd_cosine   GIDD's schedule (von Rütte et al., ICML 2025, arXiv:2503.04482, App. E), ported from get_lr in
                dvruette/gidd@f67acc96 gidd/utils.py:18-33: linear warm-up from step / warmup_steps, then a cosine
                over the steps after warm-up that ends on min_lr. GIDD's floor is 10% of the peak.

GIDD_LR_RECIPE is the one recipe #431 tests on both Uniform, no filter and Masked, full filter: peak 5e-4, 1,000
warm-up steps (about 1.5% of 66,960 steps), cosine to the 10% floor 5e-5.
"""
import math

GIDD_LR_RECIPE = dict(lr=5e-4, warmup=1_000, min_lr=5e-5, lr_schedule="gidd_cosine")
SCHEDULES = ("cosine", "gidd_cosine")


def make_lr_at(schedule, *, peak_lr, total_steps, warmup_steps, min_lr=0.0):
    """lr_at(step) for one training run of total_steps optimizer steps."""
    if schedule == "cosine":
        if min_lr:
            raise ValueError("the cosine schedule decays to 0; use gidd_cosine for a floor")
        return lambda s: peak_lr * min(1.0, (s + 1) / warmup_steps) * 0.5 * (
            1 + math.cos(math.pi * min(s, total_steps) / total_steps))
    if schedule == "gidd_cosine":
        floor = min_lr / peak_lr
        decay_steps = total_steps - warmup_steps

        def lr_at(s):
            if s < warmup_steps:
                return peak_lr * s / warmup_steps
            progress = min(s - warmup_steps, decay_steps) / decay_steps
            return peak_lr * (floor + (1 - floor) * (1 + math.cos(math.pi * progress)) / 2)
        return lr_at
    raise ValueError(f"unknown learning-rate schedule {schedule!r}; choose from {SCHEDULES}")
