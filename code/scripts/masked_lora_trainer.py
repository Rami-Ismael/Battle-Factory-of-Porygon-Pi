"""Masked text diffusion Low-Rank Adaptation (LoRA) trainer — DiffuGPT small tracer bullet (#408, parent #177).

Trains LoRA adapters on a frozen masked text diffusion model using the in-place JSON
representation from `masked_diffusion_text.training_target` and the Masked Diffusion Language
Modeling (MDLM) objective from `ZHZisZZ/dllm` at `ca176752fbceec49c6b4777a2c18ae88e4eb10ed`.

Model and adapters
------------------
- DiffuGPT small (`diffusionfamily/diffugpt-s`, revision `391d9850148c0beda355a8874bea958a40164c74`),
  loaded through `masked_diffusion_memory_bench.load_diffugpt`.
- Bidirectional self-attention via an additive 4D mask (all zeros on real tokens; `-inf` /
  `finfo.min` on batch right-padding keys).
- DiffuGPT predicts position `i` from position `i - 1`, so logits are shifted right by one
  (`shift_right`), matching dLLM's `right_shift_logits=True`. Every row starts with the clean
  header (`labels[:, 0] == -100`), so position 0 is never scored.
- LoRA only (owner, 2026-11-15): rank 32, alpha 64 on `c_attn`, `c_proj`, and `c_fc` with
  `fan_in_fan_out=True` (4,718,592 trainable of 129,158,400 weights across 96 adapter tensors).

Pad token and loss-scoring policy (#177 / #408)
-----------------------------------------------
- **Slot pad token:** `tokenizer.pad_token` when the tokenizer defines one, otherwise
  `tokenizer.eos_token` (`<|endoftext|>`, id 50256 in DiffuGPT small).
  Why: DiffuGPT small ships with `pad_token=None` and only two special tokens in its 50,257-token
  vocabulary (`------------` as mask id 10541 and `<|endoftext|>` as id 50256). Adding a brand-new
  token would resize the frozen base embedding table and tied output head, violating LoRA-only
  training. Reusing a printable character such as space would merge across byte-pair encoding
  boundaries and collide with real names containing spaces (for example `"Choice Scarf"`),
  corrupting `masked_diffusion_text.read_filled`. `<|endoftext|>` is a single special token that
  never merges with adjacent text, never appears inside any Regulation M-B field value, and
  matches dLLM's `pad_token = eos_token` fallback.
- **Loss scores slot-pad tokens, and ignores batch right-padding:** Every hidden field slot has
  fixed length equal to the longest Regulation M-B value of that field type in the tokenizer so
  slot length never leaks the answer. During generation the sampler must fill every mask token in
  the slot and `read_filled` strips `pad` before JSON parsing; scoring slot-pad tokens inside
  hidden slots teaches the model to emit `<|endoftext|>` in unused trailing slot positions instead
  of invalid text. By contrast, batch right-padding appended after the closing `]` only aligns
  variable-length sequences in a minibatch: it stays outside all hidden slots, carries
  `labels = -100` (never corrupted and never scored), and is masked out of bidirectional attention.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

import mask_paste
import masked_diffusion_memory_bench as memory_bench
import masked_diffusion_text as diffusion_text

VOCAB_FILE = Path(__file__).resolve().parents[1] / 'data/regmb-vocabulary.json'
TEXT_LIMIT = 1024
MASK_LAB_TASKS: tuple[str, ...] = ('items', 'abilities', 'moves', 'natures', 'stats', 'pokemon')
LORA_TARGETS: tuple[str, ...] = ('c_attn', 'c_proj', 'c_fc')
TIME_EPSILON = 1e-3
WEIGHT_EPS = 1e-6

DIFFUGPT_S_INFO: dict[str, Any] = {
    'key': 'diffugpt_s',
    'label': 'DiffuGPT small',
    'repo': 'diffusionfamily/diffugpt-s',
    'revision': '391d9850148c0beda355a8874bea958a40164c74',
    'mask_id': 10541,
    'mask_token': None,
    'qwen2_files': False,
    'start_token': False,
}


@dataclass(frozen=True)
class RegMBVocab:
    """Regulation M-B token-ID to display-name tables from code/data/regmb-vocabulary.json."""

    itos: dict[str, list[str]]
    names: dict[str, str]


@dataclass(frozen=True)
class EncodedRow:
    """One tokenized in-place training row."""

    input_ids: list[int]
    labels: list[int]               # token id on hidden slot positions, -100 elsewhere
    slot_pad_mask: list[bool]       # True on eligible positions whose target is the slot pad token
    target_text: str
    slots: tuple[diffusion_text.SlotSpan, ...]
    ask: float | None


@dataclass(frozen=True)
class Batch:
    """Right-padded minibatch ready for mdlm_loss."""

    input_ids: torch.Tensor         # [B, L] int64
    labels: torch.Tensor            # [B, L] int64 (-100 outside hidden slots and on batch padding)
    valid_mask: torch.Tensor        # [B, L] bool (True on real tokens, False on batch right-padding)
    slot_pad_mask: torch.Tensor     # [B, L] bool (True on slot-pad tokens inside hidden slots)

    def to(self, device: str | torch.device) -> Batch:
        return Batch(
            self.input_ids.to(device),
            self.labels.to(device),
            self.valid_mask.to(device),
            self.slot_pad_mask.to(device),
        )


@dataclass(frozen=True)
class LossBreakdown:
    """Scalar MDLM loss plus unweighted diagnostics for reporting and parity checks."""

    loss: torch.Tensor
    weighted_nll_sum: float
    eligible_count: int
    masked_count: int
    unweighted_masked_ce_sum: float
    value_masked_ce_sum: float
    value_masked_count: int
    value_weighted_nll_sum: float
    slot_pad_masked_ce_sum: float
    slot_pad_masked_count: int
    slot_pad_weighted_nll_sum: float
    weight_masked_sum: float        # sum(masked * w(t)) for exact draw-matched uniform-random baseline


# ---------- decoding lossdata84_shuffled_labels.npz ----------

def load_regmb_vocab(path: Path = VOCAB_FILE) -> RegMBVocab:
    raw = json.loads(path.read_text())
    itos = {k: ['[MASK]', *sorted(set(v))] for k, v in raw['values'].items()}
    return RegMBVocab(itos=itos, names=raw['names'])


def _display_name(vocab: RegMBVocab, field: str, token_id: int) -> str:
    raw = vocab.itos[field][token_id]
    if not raw:
        return ''
    return vocab.names.get(raw, raw)


def decode_team_84(row: Sequence[int] | np.ndarray, vocab: RegMBVocab) -> mask_paste.Team:
    """Decode one 84-field row from lossdata84_shuffled_labels.npz into a Mask Lab Team."""
    vals = [int(v) for v in row]
    team: mask_paste.Team = []
    for s in range(6):
        base = s * 8
        stat_points = {stat: vals[48 + s * 6 + j] - 1 for j, stat in enumerate(mask_paste.STATS)}
        team.append({
            'slot': s + 1,
            'species': _display_name(vocab, 'species', vals[base + 0]),
            'ability': _display_name(vocab, 'ability', vals[base + 1]),
            'item': _display_name(vocab, 'item', vals[base + 2]),
            'moves': [_display_name(vocab, 'move', vals[base + 3 + j]) for j in range(4)],
            'nature': _display_name(vocab, 'nature', vals[base + 7]),
            'statPoints': stat_points,
            'ivs': dict.fromkeys(mask_paste.STATS, 31),
            'level': 50,
        })
    return team


def _sample_train_arrays(npz_path: Path, count: int, seed: int, *, sort_indices: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Read `count` training-side (`~all_val`) rows and win rates from an 84-field `.npz` table."""
    with np.load(npz_path) as data:
        train_idx = np.flatnonzero(~data['all_val'])
        if count > len(train_idx):
            raise ValueError(f'requested {count} training rows, but {npz_path} has only {len(train_idx)}')
        rng = np.random.default_rng(seed)
        chosen = rng.choice(train_idx, size=count, replace=False)
        if sort_indices:
            chosen = np.sort(chosen)
        return np.ascontiguousarray(data['all_X'][chosen]), np.ascontiguousarray(data['all_W'][chosen])


def load_training_teams(
    npz_path: Path, count: int, seed: int, vocab: RegMBVocab
) -> list[tuple[mask_paste.Team, float | None]]:
    """Sample `count` training-side (`~all_val`) teams and win rates from an 84-field `.npz` table."""
    rows, win_rates = _sample_train_arrays(npz_path, count, seed)
    return [(decode_team_84(row, vocab), float(w) if np.isfinite(w) else None)
            for row, w in zip(rows, win_rates, strict=True)]


def slice_training_npz(source_npz: Path, out_npz: Path, count: int, seed: int) -> dict[str, Any]:
    """Write a compact `.npz` holding `count` training-side rows (`~all_val`) for Modal transfer."""
    source_sha = hashlib.sha256(source_npz.read_bytes()).hexdigest()
    rows, win_rates = _sample_train_arrays(source_npz, count, seed, sort_indices=True)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, all_X=rows, all_W=win_rates, all_val=np.zeros(count, dtype=bool))
    return {
        'source': str(source_npz),
        'source_sha256': source_sha,
        'out': str(out_npz),
        'count': count,
        'seed': seed,
        'unlabelled': int(np.isnan(win_rates).sum()),
    }


# ---------- Mask Lab masks and in-place row encoding ----------

def _all_paths_for_task(task_kind: str) -> list[str]:
    if task_kind == 'pokemon':
        return [f'team[{s}]' for s in range(6)]
    if task_kind == 'moves':
        return [f'team[{s}].moves[{m}]' for s in range(6) for m in range(4)]
    field_map = {'items': 'item', 'abilities': 'ability', 'natures': 'nature', 'stats': 'statPoints'}
    field = field_map[task_kind]
    return [f'team[{s}].{field}' for s in range(6)]


def _apply_masked_path(masked_team: mask_paste.Team, path_str: str) -> None:
    parsed = mask_paste.parse_path(path_str)
    slot_idx = parsed[0]
    if len(parsed) == 1:
        masked_team[slot_idx] = None
    elif len(parsed) == 2:
        mon = masked_team[slot_idx]
        assert mon is not None
        mon[parsed[1]] = None
    else:
        mon = masked_team[slot_idx]
        assert mon is not None
        mon[parsed[1]][parsed[2]] = None


def make_mask_lab_task(team: mask_paste.Team, task_kind: str, k: int, rng: random.Random) -> dict[str, Any]:
    """Construct one Mask Lab task dict hiding `k` distinct paths of `task_kind` on `team`."""
    candidates = _all_paths_for_task(task_kind)
    if not (1 <= k <= len(candidates)):
        raise ValueError(f'{task_kind}: k={k} out of range 1..{len(candidates)}')
    chosen = sorted(rng.sample(candidates, k), key=candidates.index)
    masked_team = copy.deepcopy(team)
    for path_str in chosen:
        _apply_masked_path(masked_team, path_str)
    return {
        'task': task_kind,
        'k': k,
        'original': team,
        'masked_team': masked_team,
        'masked_paths': chosen,
    }


def sample_mask_lab_task(team: mask_paste.Team, rng: random.Random) -> dict[str, Any]:
    """Sample one Mask Lab mask uniformly across the 6 task families and each family's valid k."""
    task_kind = rng.choice(MASK_LAB_TASKS)
    k = rng.randint(1, len(_all_paths_for_task(task_kind)))
    return make_mask_lab_task(team, task_kind, k, rng)


def resolve_pad_token(tokenizer: Any) -> str:
    """Slot pad token: tokenizer.pad_token if set, else tokenizer.eos_token (<|endoftext|> on DiffuGPT)."""
    pad = getattr(tokenizer, 'pad_token', None) or getattr(tokenizer, 'eos_token', None)
    if not pad:
        raise ValueError('tokenizer defines neither pad_token nor eos_token')
    return str(pad)


def resolve_pad_id(tokenizer: Any) -> int:
    pad_token = resolve_pad_token(tokenizer)
    ids = tokenizer(pad_token, add_special_tokens=False)['input_ids']
    if len(ids) != 1:
        raise ValueError(f'pad token {pad_token!r} tokenized to {ids}, expected 1 token')
    return int(ids[0])


def build_training_row(
    task: dict[str, Any],
    ask: float | None,
    tokenizer: Any,
    limit: int = TEXT_LIMIT,
) -> EncodedRow:
    """Build one in-place training row: only hidden slot tokens (including slot pad) are eligible."""
    pad_token = resolve_pad_token(tokenizer)
    pad_id = resolve_pad_id(tokenizer)
    built = diffusion_text.in_place_text(task, ask, tokenizer, limit)
    target_text = diffusion_text.training_target(task, ask, tokenizer, limit, pad_token)
    input_ids = [int(i) for i in tokenizer(target_text, add_special_tokens=False)['input_ids']]
    eligible = {n for slot in built.slots for n in range(slot.first_token, slot.first_token + slot.length)}
    labels = [input_ids[i] if i in eligible else -100 for i in range(len(input_ids))]
    slot_pad_mask = [i in eligible and input_ids[i] == pad_id for i in range(len(input_ids))]
    if labels[0] != -100:
        raise ValueError('position 0 must belong to the clean header (label -100) for shifted logits')
    return EncodedRow(
        input_ids=input_ids,
        labels=labels,
        slot_pad_mask=slot_pad_mask,
        target_text=target_text,
        slots=built.slots,
        ask=ask,
    )


def build_dataset_rows(
    teams: Sequence[tuple[mask_paste.Team, float | None]],
    tokenizer: Any,
    seed: int,
    limit: int = TEXT_LIMIT,
) -> list[EncodedRow]:
    rng = random.Random(seed)
    return [build_training_row(sample_mask_lab_task(team, rng), ask, tokenizer, limit) for team, ask in teams]


def collate_rows(rows: Sequence[EncodedRow], batch_pad_id: int) -> Batch:
    """Right-pad rows to the longest sequence in the batch; batch padding is ignored (-100)."""
    max_len = max(len(r.input_ids) for r in rows)
    b = len(rows)
    input_ids = torch.full((b, max_len), batch_pad_id, dtype=torch.long)
    labels = torch.full((b, max_len), -100, dtype=torch.long)
    valid_mask = torch.zeros((b, max_len), dtype=torch.bool)
    slot_pad_mask = torch.zeros((b, max_len), dtype=torch.bool)
    for i, row in enumerate(rows):
        n = len(row.input_ids)
        input_ids[i, :n] = torch.tensor(row.input_ids, dtype=torch.long)
        labels[i, :n] = torch.tensor(row.labels, dtype=torch.long)
        valid_mask[i, :n] = True
        slot_pad_mask[i, :n] = torch.tensor(row.slot_pad_mask, dtype=torch.bool)
    return Batch(input_ids=input_ids, labels=labels, valid_mask=valid_mask, slot_pad_mask=slot_pad_mask)


# ---------- Bidirectional attention mask & MDLM loss (dLLM ca176752) ----------

def bidirectional_padding_mask(valid_mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """4D additive mask [B, 1, L, L]: 0 everywhere when no batch padding, else finfo.min on pad keys."""
    b, seq = valid_mask.shape
    mask = torch.zeros((b, 1, seq, seq), dtype=dtype, device=valid_mask.device)
    if not bool(valid_mask.all()):
        pad_keys = ~valid_mask[:, None, None, :]
        mask = mask.masked_fill(pad_keys, torch.finfo(dtype).min)
    return mask


def _sample_mdlm_noise(
    b: int,
    seq: int,
    device: torch.device,
    time_epsilon: float,
    generator: torch.Generator | None,
    u_t: torch.Tensor | None,
    u_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if u_t is None:
        u_t = torch.rand(b, generator=generator, device=device, dtype=torch.float32)
    else:
        u_t = u_t.to(device=device, dtype=torch.float32)
    if u_mask is None:
        u_mask = torch.rand((b, seq), generator=generator, device=device, dtype=torch.float32)
    else:
        u_mask = u_mask.to(device=device, dtype=torch.float32)
    t = time_epsilon + (1.0 - time_epsilon) * u_t
    alpha_t = 1.0 - t                                # LinearAlphaScheduler._alpha(t)
    p_mask = 1.0 - alpha_t.unsqueeze(1).expand(b, seq)
    weights = (1.0 / (1.0 - alpha_t + WEIGHT_EPS)).unsqueeze(1).expand(b, seq)
    return p_mask, weights, u_mask


def mdlm_loss(
    logits_fn: Callable[[torch.Tensor, torch.Tensor | None], torch.Tensor],
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    mask_token_id: int,
    *,
    attention_mask: torch.Tensor | None = None,
    slot_pad_mask: torch.Tensor | None = None,
    right_shift_logits: bool = True,
    time_epsilon: float = TIME_EPSILON,
    generator: torch.Generator | None = None,
    u_t: torch.Tensor | None = None,
    u_mask: torch.Tensor | None = None,
) -> LossBreakdown:
    """Masked Diffusion Language Modeling loss matching ZHZisZZ/dllm@ca176752 MDLMTrainer.compute_loss."""
    b, seq = input_ids.shape
    maskable_mask = labels != -100
    if right_shift_logits and not bool((labels[:, 0] == -100).all()):
        raise ValueError('right_shift_logits requires labels[:, 0] == -100 on every row')
    if not bool((input_ids[maskable_mask] == labels[maskable_mask]).all()):
        raise ValueError('input_ids and labels must match on all eligible positions')

    p_mask, weights, drawn_u_mask = _sample_mdlm_noise(
        b, seq, input_ids.device, time_epsilon, generator, u_t, u_mask
    )
    masked_mask = (drawn_u_mask < p_mask) & maskable_mask
    noised_input_ids = torch.where(masked_mask, mask_token_id, input_ids)

    logits = logits_fn(noised_input_ids, attention_mask)
    if right_shift_logits:
        logits = memory_bench.shift_right(logits)

    raw_ce = F.cross_entropy(logits.float().transpose(1, 2), input_ids, reduction='none')
    weighted_nll = raw_ce * weights * masked_mask.to(raw_ce.dtype)
    eligible_count = int(maskable_mask.sum().item())
    denom = maskable_mask.sum().clamp_min(1)
    loss = (weighted_nll / denom).sum()

    return _summarize_loss_breakdown(
        loss, raw_ce, weighted_nll, weights, masked_mask, slot_pad_mask, eligible_count
    )


def _summarize_loss_breakdown(
    loss: torch.Tensor,
    raw_ce: torch.Tensor,
    weighted_nll: torch.Tensor,
    weights: torch.Tensor,
    masked_mask: torch.Tensor,
    slot_pad_mask: torch.Tensor | None,
    eligible_count: int,
) -> LossBreakdown:
    with torch.no_grad():
        pad_sel = (
            masked_mask & slot_pad_mask.to(masked_mask.device)
            if slot_pad_mask is not None
            else torch.zeros_like(masked_mask)
        )
        val_sel = masked_mask & ~pad_sel
        return LossBreakdown(
            loss=loss,
            weighted_nll_sum=float(weighted_nll.sum().item()),
            eligible_count=eligible_count,
            masked_count=int(masked_mask.sum().item()),
            unweighted_masked_ce_sum=float((raw_ce * masked_mask).sum().item()),
            value_masked_ce_sum=float((raw_ce * val_sel).sum().item()),
            value_masked_count=int(val_sel.sum().item()),
            value_weighted_nll_sum=float((weighted_nll * val_sel).sum().item()),
            slot_pad_masked_ce_sum=float((raw_ce * pad_sel).sum().item()),
            slot_pad_masked_count=int(pad_sel.sum().item()),
            slot_pad_weighted_nll_sum=float((weighted_nll * pad_sel).sum().item()),
            weight_masked_sum=float((weights * masked_mask).sum().item()),
        )


def uniform_random_expected_loss(vocab_size: int, time_epsilon: float = TIME_EPSILON) -> dict[str, float]:
    """Expected Masked Diffusion Language Modeling loss of a uniform-random predictor over `vocab_size` tokens."""
    log_v = math.log(vocab_size)
    ratio = 1.0 - (WEIGHT_EPS / (1.0 - time_epsilon)) * math.log((1.0 + WEIGHT_EPS) / (time_epsilon + WEIGHT_EPS))
    return {'log_vocab': log_v, 'expected_mdlm_loss': log_v * ratio}


# ---------- DiffuGPT small + LoRA ----------

def apply_diffugpt_lora(model: torch.nn.Module) -> torch.nn.Module:
    """Wrap DiffuGPT with frozen base weights and LoRA (rank 32, alpha 64) on c_attn, c_proj, c_fc."""
    from peft import LoraConfig, get_peft_model

    config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=list(LORA_TARGETS),
        lora_dropout=0.0,
        bias='none',
        fan_in_fan_out=True,
    )
    return get_peft_model(model, config)


def parameter_counts(model: torch.nn.Module) -> dict[str, int]:
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    frozen_params = [p for p in model.parameters() if not p.requires_grad]
    return {
        'trainable_weights': sum(p.numel() for p in trainable_params),
        'frozen_weights': sum(p.numel() for p in frozen_params),
        'total_weights': sum(p.numel() for p in model.parameters()),
        'trainable_tensors': len(trainable_params),
        'frozen_tensors': len(frozen_params),
    }


def _tensor_sha256(tensor: torch.Tensor) -> str:
    arr = tensor.detach().cpu().contiguous().view(torch.uint8).numpy()
    return hashlib.sha256(arr.tobytes()).hexdigest()


def base_weight_hashes(model: torch.nn.Module) -> dict[str, str]:
    """SHA-256 of every frozen (non-LoRA) parameter tensor in `model`."""
    return {name: _tensor_sha256(p) for name, p in model.named_parameters() if not p.requires_grad}


def lora_weight_hashes(model: torch.nn.Module) -> dict[str, str]:
    """SHA-256 of every trainable (LoRA) parameter tensor in `model`."""
    return {name: _tensor_sha256(p) for name, p in model.named_parameters() if p.requires_grad}


def model_logits_fn(model: torch.nn.Module) -> Callable[[torch.Tensor, torch.Tensor | None], torch.Tensor]:
    return lambda x, mask: model(input_ids=x, attention_mask=mask).logits


def _model_dtype(model: torch.nn.Module) -> torch.dtype:
    return next(model.parameters()).dtype


def train_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: Batch,
    mask_token_id: int,
    *,
    right_shift_logits: bool = True,
    time_epsilon: float = TIME_EPSILON,
    generator: torch.Generator | None = None,
    grad_clip: float = 1.0,
) -> LossBreakdown:
    """One forward, backward, clip, and optimizer step on `batch`."""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    attn = bidirectional_padding_mask(batch.valid_mask, dtype=_model_dtype(model))
    breakdown = mdlm_loss(
        model_logits_fn(model),
        batch.input_ids,
        batch.labels,
        mask_token_id,
        attention_mask=attn,
        slot_pad_mask=batch.slot_pad_mask,
        right_shift_logits=right_shift_logits,
        time_epsilon=time_epsilon,
        generator=generator,
    )
    if not torch.isfinite(breakdown.loss):
        raise RuntimeError(f'non-finite loss encountered: {float(breakdown.loss)}')
    breakdown.loss.backward()
    trainable = [p for p in model.parameters() if p.requires_grad]
    torch.nn.utils.clip_grad_norm_(trainable, grad_clip)
    optimizer.step()
    return breakdown


# ---------- Evaluation, overfit, and tracer-bullet orchestration ----------

@torch.no_grad()
def evaluate_batches(
    model: torch.nn.Module,
    batches: Sequence[Batch],
    mask_token_id: int,
    vocab_size: int,
    *,
    seed: int = 2026,
    right_shift_logits: bool = True,
    time_epsilon: float = TIME_EPSILON,
    device: str | torch.device = 'cpu',
) -> dict[str, Any]:
    """Token-normalized MDLM loss over `batches` in eval mode with a deterministic noise generator."""
    was_training = model.training
    model.eval()
    gen = torch.Generator(device=device).manual_seed(seed)
    totals = {
        'weighted_nll': 0.0, 'eligible': 0, 'masked': 0, 'unweighted_ce': 0.0,
        'val_ce': 0.0, 'val_masked': 0, 'val_weighted_nll': 0.0,
        'pad_ce': 0.0, 'pad_masked': 0, 'pad_weighted_nll': 0.0, 'weight_masked': 0.0,
    }
    try:
        for raw_batch in batches:
            batch = raw_batch.to(device)
            attn = bidirectional_padding_mask(batch.valid_mask, dtype=_model_dtype(model))
            step_out = mdlm_loss(
                model_logits_fn(model), batch.input_ids, batch.labels, mask_token_id,
                attention_mask=attn, slot_pad_mask=batch.slot_pad_mask,
                right_shift_logits=right_shift_logits, time_epsilon=time_epsilon, generator=gen,
            )
            totals['weighted_nll'] += step_out.weighted_nll_sum
            totals['eligible'] += step_out.eligible_count
            totals['masked'] += step_out.masked_count
            totals['unweighted_ce'] += step_out.unweighted_masked_ce_sum
            totals['val_ce'] += step_out.value_masked_ce_sum
            totals['val_masked'] += step_out.value_masked_count
            totals['val_weighted_nll'] += step_out.value_weighted_nll_sum
            totals['pad_ce'] += step_out.slot_pad_masked_ce_sum
            totals['pad_masked'] += step_out.slot_pad_masked_count
            totals['pad_weighted_nll'] += step_out.slot_pad_weighted_nll_sum
            totals['weight_masked'] += step_out.weight_masked_sum
    finally:
        model.train(was_training)

    denom = max(totals['eligible'], 1)
    rand_info = uniform_random_expected_loss(vocab_size, time_epsilon)
    mdlm_val = totals['weighted_nll'] / denom
    matched_rand = rand_info['log_vocab'] * (totals['weight_masked'] / denom)
    return {
        'mdlm_loss': mdlm_val,
        'unweighted_masked_ce': totals['unweighted_ce'] / max(totals['masked'], 1),
        'value_token_ce': totals['val_ce'] / max(totals['val_masked'], 1),
        'slot_pad_token_ce': totals['pad_ce'] / max(totals['pad_masked'], 1),
        'value_mdlm_contribution': totals['val_weighted_nll'] / denom,
        'slot_pad_mdlm_contribution': totals['pad_weighted_nll'] / denom,
        'eligible_tokens': totals['eligible'],
        'masked_tokens': totals['masked'],
        'value_masked_tokens': totals['val_masked'],
        'slot_pad_masked_tokens': totals['pad_masked'],
        'random_log_vocab': rand_info['log_vocab'],
        'random_expected_mdlm_loss': rand_info['expected_mdlm_loss'],
        'random_matched_draw_loss': matched_rand,
        'gap_vs_random_expected': mdlm_val - rand_info['expected_mdlm_loss'],
        'gap_vs_random_matched_draw': mdlm_val - matched_rand,
    }


def _snapshot_lora_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.named_parameters() if v.requires_grad}


def _restore_lora_state(model: torch.nn.Module, state: dict[str, torch.Tensor]) -> None:
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in state:
                param.copy_(state[name].to(device=param.device, dtype=param.dtype))


def run_overfit_check(
    model: torch.nn.Module,
    batch: Batch,
    mask_token_id: int,
    vocab_size: int,
    *,
    steps: int = 250,
    lr: float = 5e-4,
    seed: int = 2026,
    eval_every: int = 25,
    device: str | torch.device = 'cpu',
) -> dict[str, Any]:
    """Overfit one fixed batch (default 8 rows) and report both stochastic and fixed-draw loss curves."""
    dev_batch = batch.to(device)
    init_eval = evaluate_batches(model, [dev_batch], mask_token_id, vocab_size, seed=seed, device=device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    gen = torch.Generator(device=device).manual_seed(seed + 1)
    step_losses: list[float] = []
    eval_curve: list[dict[str, Any]] = [{'step': 0, 'mdlm_loss': init_eval['mdlm_loss'],
                                         'unweighted_masked_ce': init_eval['unweighted_masked_ce']}]
    t0 = time.perf_counter()
    for step in range(1, steps + 1):
        out = train_step(model, optimizer, dev_batch, mask_token_id, generator=gen)
        step_losses.append(float(out.loss.item()))
        if step % eval_every == 0 or step == steps:
            ev = evaluate_batches(model, [dev_batch], mask_token_id, vocab_size, seed=seed, device=device)
            eval_curve.append({'step': step, 'mdlm_loss': ev['mdlm_loss'],
                               'unweighted_masked_ce': ev['unweighted_masked_ce']})
            print(f'[overfit] step {step:3d}/{steps}: train_loss={step_losses[-1]:.4f} '
                  f'fixed_eval_loss={ev["mdlm_loss"]:.4f} masked_cross_entropy={ev["unweighted_masked_ce"]:.4f}',
                  flush=True)
    elapsed = time.perf_counter() - t0
    return {
        'rows': int(batch.input_ids.shape[0]),
        'steps': steps,
        'lr': lr,
        'elapsed_seconds': elapsed,
        'seconds_per_step': elapsed / max(steps, 1),
        'initial_fixed_eval': init_eval,
        'final_fixed_eval': eval_curve[-1],
        'eval_curve': eval_curve,
        'step_losses': step_losses,
    }


def run_few_step_train(
    model: torch.nn.Module,
    batches: Sequence[Batch],
    mask_token_id: int,
    *,
    steps: int = 16,
    lr: float = 3e-4,
    seed: int = 2026,
    device: str | torch.device = 'cpu',
) -> dict[str, Any]:
    """Run `steps` training updates across `batches` of a few hundred in-place rows (#408)."""
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    gen = torch.Generator(device=device).manual_seed(seed + 100)
    losses: list[float] = []
    t0 = time.perf_counter()
    for step in range(steps):
        batch = batches[step % len(batches)].to(device)
        out = train_step(model, optimizer, batch, mask_token_id, generator=gen)
        losses.append(float(out.loss.item()))
        print(f'[train] step {step + 1:3d}/{steps}: loss={losses[-1]:.4f} '
              f'masked={out.masked_count}/{out.eligible_count}', flush=True)
    elapsed = time.perf_counter() - t0
    return {
        'steps': steps,
        'batches_available': len(batches),
        'lr': lr,
        'elapsed_seconds': elapsed,
        'seconds_per_step': elapsed / max(steps, 1),
        'step_losses': losses,
    }


def _resolve_device(device: str) -> str:
    if device != 'auto':
        return device
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def load_diffugpt_small_lora(device: str, dtype_name: str) -> tuple[torch.nn.Module, Any, dict[str, Any]]:
    """Load DiffuGPT small at its pinned revision, check its mask token, and wrap with LoRA."""
    info = dict(DIFFUGPT_S_INFO)
    tokenizer = memory_bench.load_tokenizer(info)
    mask_ids = tokenizer(tokenizer.mask_token, add_special_tokens=False)['input_ids']
    if mask_ids != [info['mask_id']]:
        raise ValueError(f"mask token {tokenizer.mask_token!r} tokenized as {mask_ids}, expected [{info['mask_id']}]")
    dtype = getattr(torch, dtype_name)
    base = memory_bench.load_diffugpt(info, dtype=dtype, device=device)
    model = apply_diffugpt_lora(base)
    return model, tokenizer, info


def _chunk_batches(rows: Sequence[EncodedRow], batch_size: int, pad_id: int) -> list[Batch]:
    return [collate_rows(rows[i:i + batch_size], pad_id) for i in range(0, len(rows), batch_size)]


def run_tracer(args: argparse.Namespace) -> dict[str, Any]:
    """Run initialization-loss measurement, 8-row overfit, and few-step multi-row training."""
    device = _resolve_device(args.device)
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    model, tokenizer, info = load_diffugpt_small_lora(device, args.dtype)
    counts = parameter_counts(model)
    base_hashes_before = base_weight_hashes(model)
    init_lora_state = _snapshot_lora_state(model)

    vocab = load_regmb_vocab(VOCAB_FILE)
    teams = load_training_teams(args.data, args.rows, args.seed, vocab)
    rows = build_dataset_rows(teams, tokenizer, seed=args.seed, limit=TEXT_LIMIT)
    pad_id = resolve_pad_id(tokenizer)
    mask_id = int(info['mask_id'])
    vocab_size = int(getattr(model.config, 'vocab_size', len(tokenizer)))

    batches = _chunk_batches(rows, args.batch_size, pad_id)
    init_eval = evaluate_batches(model, batches, mask_id, vocab_size, seed=args.seed, device=device)

    overfit_batch = collate_rows(rows[:args.overfit_rows], pad_id)
    overfit = run_overfit_check(
        model, overfit_batch, mask_id, vocab_size,
        steps=args.overfit_steps, lr=args.overfit_lr, seed=args.seed,
        eval_every=args.eval_every, device=device,
    )
    overfit_lora_changed = sum(
        h_before != h_after
        for h_before, h_after in zip(
            {k: _tensor_sha256(v) for k, v in init_lora_state.items()}.values(),
            lora_weight_hashes(model).values(),
            strict=True,
        )
    )
    base_unchanged_after_overfit = base_weight_hashes(model) == base_hashes_before

    _restore_lora_state(model, init_lora_state)
    few_step = run_few_step_train(
        model, batches, mask_id, steps=args.train_steps, lr=args.lr, seed=args.seed, device=device
    )
    base_unchanged_after_train = base_weight_hashes(model) == base_hashes_before

    peak_mem_bytes = int(torch.cuda.max_memory_allocated()) if device == 'cuda' else 0
    gpu_name = torch.cuda.get_device_name(0) if device == 'cuda' else None
    result = {
        'candidate': info,
        'device': device,
        'gpu_name': gpu_name,
        'dtype': args.dtype,
        'pad_token': resolve_pad_token(tokenizer),
        'pad_id': pad_id,
        'score_slot_pad': True,
        'score_batch_pad': False,
        'rows': len(rows),
        'seed': args.seed,
        'parameter_counts': counts,
        'init_loss': init_eval,
        'overfit': overfit,
        'overfit_lora_tensors_changed': overfit_lora_changed,
        'base_weights_unchanged_after_overfit': base_unchanged_after_overfit,
        'few_step_train': few_step,
        'base_weights_unchanged_after_train': base_unchanged_after_train,
        'peak_cuda_bytes': peak_mem_bytes,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'tracer_results.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Masked text diffusion Low-Rank Adaptation (LoRA) trainer (#408).')
    sub = parser.add_subparsers(dest='command', required=True)

    sl = sub.add_parser('slice-data', help='Extract a compact training-only .npz slice for Modal transfer.')
    sl.add_argument('--data', type=Path, required=True)
    sl.add_argument('--out', type=Path, required=True)
    sl.add_argument('--count', type=int, default=256)
    sl.add_argument('--seed', type=int, default=2026)

    tr = sub.add_parser('tracer', help='Run initialization loss, 8-row overfit, and few-step training.')
    tr.add_argument('--data', type=Path, required=True)
    tr.add_argument('--out', type=Path, required=True)
    tr.add_argument('--rows', type=int, default=256)
    tr.add_argument('--overfit-rows', type=int, default=8)
    tr.add_argument('--overfit-steps', type=int, default=250)
    tr.add_argument('--overfit-lr', type=float, default=5e-4)
    tr.add_argument('--eval-every', type=int, default=25)
    tr.add_argument('--train-steps', type=int, default=16)
    tr.add_argument('--batch-size', type=int, default=8)
    tr.add_argument('--lr', type=float, default=3e-4)
    tr.add_argument('--seed', type=int, default=2026)
    tr.add_argument('--device', default='auto')
    tr.add_argument('--dtype', choices=('float32', 'bfloat16', 'float16'), default='float32')
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == 'slice-data':
        summary = slice_training_npz(args.data, args.out, args.count, args.seed)
        print(json.dumps(summary, indent=2), flush=True)
        return 0
    result = run_tracer(args)
    init = result['init_loss']
    ov = result['overfit']
    print(
        f"Initialization Masked Diffusion Language Modeling (MDLM) loss: {init['mdlm_loss']:.4f} "
        f"vs uniform random {init['random_expected_mdlm_loss']:.4f} "
        f"(same-draw random {init['random_matched_draw_loss']:.4f}); "
        f"overfit {ov['rows']} rows x {ov['steps']} steps: "
        f"{ov['initial_fixed_eval']['mdlm_loss']:.4f} -> {ov['final_fixed_eval']['mdlm_loss']:.4f}",
        flush=True,
    )
    return 0


if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.exit(main())
