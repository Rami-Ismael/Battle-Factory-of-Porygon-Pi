"""Per-model sampler settings for masked text diffusion (#392, map #160).

A SamplerSetting names the loop that picks which masked tokens to fill at each step and with which tokens, together
with the model-specific right shift and start token moved in from the candidate table and benchmark code.

Today's shared sampler (SHARED_SAMPLER) is the default for every candidate until #393-#395 switch each model to its
authors' evaluation sampler:
    fill_order      'confidence' (most confident masks first)
    masks_left      'exact_linear' (ours, dLLM) or 'at_most_linear' (DiffuGPT's evaluation sampler, #393)
    steps           positive int, or 'per_mask' (one step per masked token)
    temperature     0.0 (greedy) or positive float (1.0 in today's shared sampler)
    top_p           None (off)
    block_length    None (whole text at once) or positive int (256 in dLLM's evaluation script, #394)
    right_shift     predict position i from position i - 1 (DiffuGPT, DiffuLLaMA, Dream)
    start_token     prepend the tokenizer's start token before the text (DiffuLLaMA, #236)

Pure Python (no torch import), so count_masked_diffusion_text and CPU tests stay lightweight.
"""
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from typing import Any, Literal

FILL_ORDERS = ('confidence',)
MASKS_LEFT = ('exact_linear', 'at_most_linear')
PER_MASK_STEPS = 'per_mask'

FillOrder = Literal['confidence']
MasksLeft = Literal['exact_linear', 'at_most_linear']
StepCount = int | Literal['per_mask']


def _is_positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 1


def _validate_steps(steps: Any) -> None:
    if steps != PER_MASK_STEPS and not _is_positive_int(steps):
        raise ValueError(f'steps must be a positive int or {PER_MASK_STEPS!r}, got {steps!r}')


def _normalize_temperature(temperature: Any) -> float:
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
        raise ValueError(f'temperature must be a non-negative finite number, got {temperature!r}')
    value = float(temperature)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f'temperature must be a non-negative finite number, got {temperature!r}')
    return value


@dataclass(frozen=True)
class SamplerSetting:
    name: str = 'shared'
    fill_order: FillOrder = 'confidence'
    masks_left: MasksLeft = 'exact_linear'
    steps: StepCount = 128
    temperature: float = 1.0
    top_p: None = None
    block_length: int | None = None
    right_shift: bool = False
    start_token: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError(f'name must be a non-empty string, got {self.name!r}')
        if self.fill_order not in FILL_ORDERS:
            raise ValueError(f'fill_order must be one of {FILL_ORDERS}, got {self.fill_order!r}')
        if self.masks_left not in MASKS_LEFT:
            raise ValueError(f'masks_left must be one of {MASKS_LEFT}, got {self.masks_left!r}')
        _validate_steps(self.steps)
        object.__setattr__(self, 'temperature', _normalize_temperature(self.temperature))
        if self.top_p is not None:
            raise ValueError(f'top_p must be None (off), got {self.top_p!r}')
        if self.block_length is not None and not _is_positive_int(self.block_length):
            raise ValueError(f'block_length must be None or a positive int, got {self.block_length!r}')
        if not isinstance(self.right_shift, bool) or not isinstance(self.start_token, bool):
            raise ValueError('right_shift and start_token must be bool')

    def step_count(self, mask_count: int) -> int:
        """Concrete step count for a canvas with mask_count masked positions."""
        if isinstance(mask_count, bool) or not isinstance(mask_count, int) or mask_count < 0:
            raise ValueError(f'mask_count must be a non-negative int, got {mask_count!r}')
        return mask_count if self.steps == PER_MASK_STEPS else self.steps

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> 'SamplerSetting':
        return cls(**data)


SHARED_SAMPLER = SamplerSetting()
SAMPLERS: dict[str, SamplerSetting] = {'shared': SHARED_SAMPLER}


def shared_sampler(*, right_shift: bool = False, start_token: bool = False) -> SamplerSetting:
    """Today's shared sampler with a candidate's own right-shift and start-token flags."""
    return replace(SHARED_SAMPLER, right_shift=right_shift, start_token=start_token)


def resolve_sampler(base: SamplerSetting, name: str | None = None) -> SamplerSetting:
    """base by default, or a named sampler from SAMPLERS keeping base's right_shift and start_token."""
    if name is None:
        return base
    if name not in SAMPLERS:
        raise ValueError(f'unknown sampler {name!r}; expected one of {sorted(SAMPLERS)}')
    return replace(SAMPLERS[name], right_shift=base.right_shift, start_token=base.start_token)
