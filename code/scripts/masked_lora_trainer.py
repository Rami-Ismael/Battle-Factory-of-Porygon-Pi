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
import os
import random
import re
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

_SRC_DIR = str(Path(__file__).resolve().parents[1] / 'src')
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import checkpoints as ckpt_io
import lr_schedule84 as lr_sched
import mask_paste
import masked_diffusion_memory_bench as memory_bench
import masked_diffusion_text as diffusion_text
import wandb_tracking as wandb_track
from early_stopping import EarlyStopper

VOCAB_FILE = Path(__file__).resolve().parents[1] / 'data/regmb-vocabulary.json'
EXCLUDED_TEAMS_FILE = Path(__file__).resolve().parents[1] / 'data/masked-lora-excluded-teams.json'
NEUTRAL_NATURES: frozenset[str] = frozenset({'bashful', 'docile', 'hardy', 'quirky', 'serious'})
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


def _sample_train_arrays(npz_path: Path, count: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Read `count` training-side (`~all_val`) rows and win rates from an 84-field `.npz` table."""
    with np.load(npz_path) as data:
        train_idx = np.flatnonzero(~data['all_val'])
        if count > len(train_idx):
            raise ValueError(f'requested {count} training rows, but {npz_path} has only {len(train_idx)}')
        rng = np.random.default_rng(seed)
        chosen = rng.choice(train_idx, size=count, replace=False)
        return np.ascontiguousarray(data['all_X'][chosen]), np.ascontiguousarray(data['all_W'][chosen])


def load_training_teams(
    npz_path: Path, count: int, seed: int, vocab: RegMBVocab
) -> list[tuple[mask_paste.Team, float | None]]:
    """Sample `count` training-side (`~all_val`) teams and win rates from an 84-field `.npz` table."""
    rows, win_rates = _sample_train_arrays(npz_path, count, seed)
    return [(decode_team_84(row, vocab), float(w) if np.isfinite(w) else None)
            for row, w in zip(rows, win_rates, strict=True)]


def slice_training_npz(
    source_npz: Path,
    out_npz: Path,
    count: int,
    seed: int,
    *,
    exclude_groups: dict[str, frozenset[str]] | None = None,
    vocab: RegMBVocab | None = None,
) -> dict[str, Any]:
    """Write a compact `.npz` holding `count` training-side rows (`~all_val`) and original `row_ids` for Modal."""
    source_sha = hashlib.sha256(source_npz.read_bytes()).hexdigest()
    vocab_obj = load_regmb_vocab(VOCAB_FILE) if vocab is None else vocab
    excluded_union = _union_excluded(load_excluded_groups() if exclude_groups is None else exclude_groups)
    with np.load(source_npz) as data:
        train_idx = np.flatnonzero(~data['all_val'])
        all_x = data['all_X']
        eligible = np.array(
            [int(i) for i in train_idx if row_48_key(all_x[i], vocab_obj) not in excluded_union],
            dtype=np.int64,
        )
        if count > len(eligible):
            raise ValueError(f'requested {count} eligible training rows, but {source_npz} has only {len(eligible)}')
        rng = np.random.default_rng(seed)
        chosen = np.sort(rng.choice(eligible, size=count, replace=False))
        rows = np.ascontiguousarray(all_x[chosen])
        win_rates = np.ascontiguousarray(data['all_W'][chosen])
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz,
        all_X=rows,
        all_W=win_rates,
        all_val=np.zeros(count, dtype=bool),
        row_ids=chosen.astype(np.int64),
    )
    return {
        'source': str(source_npz),
        'source_sha256': source_sha,
        'out': str(out_npz),
        'count': count,
        'seed': seed,
        'excluded_from_pool': int(len(train_idx) - len(eligible)),
        'unlabelled': int(np.isnan(win_rates).sum()),
    }


# ---------- 48-field team identity, exclusion sets, and 2% held-out split (#409) ----------

def _norm_field(text: str | None) -> str:
    return re.sub(r'[^a-z0-9]', '', (text or '').lower())


def _canonical_slot_key(species: str, ability: str, item: str, moves: Sequence[str], nature: str) -> tuple[str, ...]:
    ms = sorted(_norm_field(m) for m in moves if _norm_field(m))
    ms += [''] * (4 - len(ms))
    nat = _norm_field(nature)
    return (_norm_field(species), _norm_field(ability), _norm_field(item), *ms[:4],
            'serious' if nat in NEUTRAL_NATURES else nat)


def _hash_slots(slots: list[tuple[str, ...]]) -> str:
    slots.sort(key=lambda s: (s[0], s[2], s[1]))
    return hashlib.sha1('|'.join(x for s in slots for x in s).encode()).hexdigest()


def team_48_key(team: mask_paste.Team) -> str:
    """Canonical 48-field SHA-1 identity matching heldout.grid_key_team (ignores slot/move order & Stat Points)."""
    slots = [
        _canonical_slot_key(mon['species'], mon['ability'], mon['item'], mon['moves'], mon['nature'])
        for mon in team
    ]
    return _hash_slots(slots)


def row_48_key(row: Sequence[int] | np.ndarray, vocab: RegMBVocab) -> str:
    """Canonical 48-field SHA-1 identity of an encoded 84-field row (matches team_48_key(decode_team_84(...)))."""
    itos = vocab.itos
    slots = [
        _canonical_slot_key(
            itos['species'][int(row[s * 8 + 0])],
            itos['ability'][int(row[s * 8 + 1])],
            itos['item'][int(row[s * 8 + 2])],
            [itos['move'][int(row[s * 8 + 3 + m])] for m in range(4)],
            itos['nature'][int(row[s * 8 + 7])],
        )
        for s in range(6)
    ]
    return _hash_slots(slots)


def load_excluded_groups(path: Path = EXCLUDED_TEAMS_FILE) -> dict[str, frozenset[str]]:
    """Load the pinned 48-field SHA-1 exclusion groups: top-50 meta, ranks 51-100, and held-out-grid teams."""
    raw = json.loads(path.read_text())
    return {
        'top50': frozenset(raw.get('top50', ())),
        'ranks51_100': frozenset(raw.get('ranks51_100', ())),
        'heldout_grid': frozenset(raw.get('heldout_grid', ())),
    }


def _union_excluded(groups: dict[str, frozenset[str]]) -> frozenset[str]:
    out: set[str] = set()
    for keys in groups.values():
        out.update(keys)
    return frozenset(out)


@dataclass(frozen=True)
class SplitRecord:
    """Deterministic training/validation split with row IDs, SHA-256 split hash, seed, and exclusion counts."""

    train_positions: tuple[int, ...]
    val_positions: tuple[int, ...]
    train_row_ids: tuple[int, ...]
    val_row_ids: tuple[int, ...]
    split_seed: int
    val_fraction: float
    split_hash: str
    excluded_counts: dict[str, int]
    source_sha256: str | None = None

    def _base_dict(self) -> dict[str, Any]:
        return {
            'split_seed': self.split_seed,
            'val_fraction': self.val_fraction,
            'split_hash': self.split_hash,
            'source_sha256': self.source_sha256,
            'train_count': len(self.train_row_ids),
            'val_count': len(self.val_row_ids),
            'excluded_counts': dict(self.excluded_counts),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self._base_dict(),
            'train_row_ids': list(self.train_row_ids),
            'val_row_ids': list(self.val_row_ids),
        }

    def summary_dict(self, split_file: str = 'split.json') -> dict[str, Any]:
        return {**self._base_dict(), 'split_file': split_file}


def compute_split_hash(
    split_seed: int,
    val_fraction: float,
    train_row_ids: Sequence[int],
    val_row_ids: Sequence[int],
) -> str:
    """SHA-256 fingerprint of a split's seed, fraction, and exact train/validation row IDs."""
    payload = {
        'split_seed': int(split_seed),
        'val_fraction': float(val_fraction),
        'train_row_ids': [int(i) for i in train_row_ids],
        'val_row_ids': [int(i) for i in val_row_ids],
    }
    return hashlib.sha256(json.dumps(payload, separators=(',', ':')).encode()).hexdigest()


def build_train_val_split(
    rows_x: np.ndarray,
    vocab: RegMBVocab,
    *,
    val_fraction: float = 0.02,
    split_seed: int = 2026,
    excluded_groups: dict[str, frozenset[str]] | None = None,
    row_ids: Sequence[int] | None = None,
    source_sha256: str | None = None,
) -> SplitRecord:
    """Exclude top-50, ranks 51-100, and held-out-grid teams from both sides, then hold out seeded `val_fraction`."""
    if not (0.0 < val_fraction < 1.0):
        raise ValueError(f'val_fraction must be in (0, 1), got {val_fraction}')
    groups = load_excluded_groups() if excluded_groups is None else excluded_groups
    all_excluded = _union_excluded(groups)
    ids = list(range(len(rows_x))) if row_ids is None else [int(i) for i in row_ids]
    if len(ids) != len(rows_x):
        raise ValueError(f'row_ids length {len(ids)} != rows_x length {len(rows_x)}')

    counts = {'top50': 0, 'ranks51_100': 0, 'heldout_grid': 0, 'total': 0}
    eligible: list[int] = []
    for pos, row in enumerate(rows_x):
        key = row_48_key(row, vocab)
        if key in all_excluded:
            counts['total'] += 1
            for group_name in ('top50', 'ranks51_100', 'heldout_grid'):
                if key in groups.get(group_name, frozenset()):
                    counts[group_name] += 1
            continue
        eligible.append(pos)

    if len(eligible) < 2:
        raise ValueError(f'need at least 2 eligible rows after exclusions, got {len(eligible)}')
    val_count = max(1, min(len(eligible) - 1, round(val_fraction * len(eligible))))
    perm = np.random.default_rng(split_seed).permutation(len(eligible))
    val_positions = tuple(sorted(eligible[int(i)] for i in perm[:val_count]))
    train_positions = tuple(sorted(eligible[int(i)] for i in perm[val_count:]))
    train_row_ids = tuple(ids[p] for p in train_positions)
    val_row_ids = tuple(ids[p] for p in val_positions)
    if not set(train_row_ids).isdisjoint(val_row_ids):
        raise AssertionError('train and validation splits overlap on row IDs')
    return SplitRecord(
        train_positions=train_positions,
        val_positions=val_positions,
        train_row_ids=train_row_ids,
        val_row_ids=val_row_ids,
        split_seed=int(split_seed),
        val_fraction=float(val_fraction),
        split_hash=compute_split_hash(split_seed, val_fraction, train_row_ids, val_row_ids),
        excluded_counts=counts,
        source_sha256=source_sha256,
    )


def load_split_teams(
    npz_path: Path,
    vocab: RegMBVocab,
    *,
    val_fraction: float = 0.02,
    split_seed: int = 2026,
    excluded_groups: dict[str, frozenset[str]] | None = None,
    max_rows: int | None = None,
) -> tuple[list[tuple[mask_paste.Team, float | None]], list[tuple[mask_paste.Team, float | None]], SplitRecord]:
    """Load an 84-field `.npz`, exclude held-out/meta teams from both sides, and return `(train, val, split)`."""
    source_sha = hashlib.sha256(npz_path.read_bytes()).hexdigest()
    with np.load(npz_path) as data:
        rows_x = np.ascontiguousarray(data['all_X'])
        rows_w = np.ascontiguousarray(data['all_W'])
        row_ids = [int(i) for i in data['row_ids']] if 'row_ids' in data.files else list(range(len(rows_x)))
    if max_rows is not None and max_rows < len(rows_x):
        rows_x = rows_x[:max_rows]
        rows_w = rows_w[:max_rows]
        row_ids = row_ids[:max_rows]
    split = build_train_val_split(
        rows_x,
        vocab,
        val_fraction=val_fraction,
        split_seed=split_seed,
        excluded_groups=excluded_groups,
        row_ids=row_ids,
        source_sha256=source_sha,
    )
    train_teams = [
        (decode_team_84(rows_x[p], vocab), float(rows_w[p]) if np.isfinite(rows_w[p]) else None)
        for p in split.train_positions
    ]
    val_teams = [
        (decode_team_84(rows_x[p], vocab), float(rows_w[p]) if np.isfinite(rows_w[p]) else None)
        for p in split.val_positions
    ]
    return train_teams, val_teams, split


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
    gen_device = generator.device if generator is not None else device
    if u_t is None:
        u_t = torch.rand(b, generator=generator, device=gen_device, dtype=torch.float32).to(device)
    else:
        u_t = u_t.to(device=device, dtype=torch.float32)
    if u_mask is None:
        u_mask = torch.rand((b, seq), generator=generator, device=gen_device, dtype=torch.float32).to(device)
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


def _train_step_with_grad_norm(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: Batch,
    mask_token_id: int,
    *,
    right_shift_logits: bool = True,
    time_epsilon: float = TIME_EPSILON,
    generator: torch.Generator | None = None,
    grad_clip: float = 1.0,
) -> tuple[LossBreakdown, float]:
    """One forward, backward, clip, and optimizer step; returns `(breakdown, grad_norm)` and fails on non-finite loss."""
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
    grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable, grad_clip).detach().item())
    if not math.isfinite(grad_norm):
        raise RuntimeError(f'non-finite gradient norm encountered: {grad_norm}')
    optimizer.step()
    return breakdown, grad_norm


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
    breakdown, _ = _train_step_with_grad_norm(
        model,
        optimizer,
        batch,
        mask_token_id,
        right_shift_logits=right_shift_logits,
        time_epsilon=time_epsilon,
        generator=generator,
        grad_clip=grad_clip,
    )
    return breakdown


# ---------- Evaluation, overfit, and tracer-bullet orchestration ----------

def _summarize_eval_totals(totals: dict[str, Any], vocab_size: int, time_epsilon: float) -> dict[str, Any]:
    denom = max(totals['eligible'], 1)
    rand_info = uniform_random_expected_loss(vocab_size, time_epsilon)
    mdlm_val = totals['weighted_nll'] / denom
    if not math.isfinite(mdlm_val):
        raise RuntimeError(f'non-finite validation loss encountered: {mdlm_val}')
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


def _empty_eval_totals() -> dict[str, Any]:
    return {
        'weighted_nll': 0.0, 'eligible': 0, 'masked': 0, 'unweighted_ce': 0.0,
        'val_ce': 0.0, 'val_masked': 0, 'val_weighted_nll': 0.0,
        'pad_ce': 0.0, 'pad_masked': 0, 'pad_weighted_nll': 0.0, 'weight_masked': 0.0,
    }


def _accumulate_eval_totals(totals: dict[str, Any], step_out: LossBreakdown) -> None:
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


@dataclass(frozen=True)
class FixedValidationBatch:
    """One validation minibatch paired with pre-drawn fixed noise tensors (common random numbers)."""

    batch: Batch
    u_t: torch.Tensor
    u_mask: torch.Tensor


def build_fixed_validation(
    val_rows: Sequence[EncodedRow],
    batch_size: int,
    pad_id: int,
    val_noise_seed: int,
) -> tuple[FixedValidationBatch, ...]:
    """Collate `val_rows` in fixed order and pre-draw fixed MDLM noise from an isolated CPU generator."""
    gen = torch.Generator().manual_seed(int(val_noise_seed))
    fixed: list[FixedValidationBatch] = []
    for raw_batch in _chunk_batches(val_rows, batch_size, pad_id):
        b, seq = raw_batch.input_ids.shape
        u_t = torch.rand(b, generator=gen, dtype=torch.float32)
        u_mask = torch.rand((b, seq), generator=gen, dtype=torch.float32)
        fixed.append(FixedValidationBatch(batch=raw_batch, u_t=u_t, u_mask=u_mask))
    return tuple(fixed)


@torch.no_grad()
def _evaluate_stream(
    model: torch.nn.Module,
    items: Sequence[tuple[Batch, torch.Tensor | None, torch.Tensor | None]],
    mask_token_id: int,
    vocab_size: int,
    *,
    generator: torch.Generator | None = None,
    right_shift_logits: bool = True,
    time_epsilon: float = TIME_EPSILON,
    device: str | torch.device = 'cpu',
) -> dict[str, Any]:
    was_training = model.training
    model.eval()
    totals = _empty_eval_totals()
    try:
        for raw_batch, u_t, u_mask in items:
            batch = raw_batch.to(device)
            attn = bidirectional_padding_mask(batch.valid_mask, dtype=_model_dtype(model))
            step_out = mdlm_loss(
                model_logits_fn(model), batch.input_ids, batch.labels, mask_token_id,
                attention_mask=attn, slot_pad_mask=batch.slot_pad_mask,
                right_shift_logits=right_shift_logits, time_epsilon=time_epsilon,
                generator=generator, u_t=u_t, u_mask=u_mask,
            )
            _accumulate_eval_totals(totals, step_out)
    finally:
        model.train(was_training)
    return _summarize_eval_totals(totals, vocab_size, time_epsilon)


def evaluate_fixed_validation(
    model: torch.nn.Module,
    fixed_batches: Sequence[FixedValidationBatch],
    mask_token_id: int,
    vocab_size: int,
    *,
    right_shift_logits: bool = True,
    time_epsilon: float = TIME_EPSILON,
    device: str | torch.device = 'cpu',
) -> dict[str, Any]:
    """Deterministic validation loss over fixed batches and pre-drawn noise; leaves all training RNG untouched."""
    return _evaluate_stream(
        model,
        [(item.batch, item.u_t, item.u_mask) for item in fixed_batches],
        mask_token_id,
        vocab_size,
        right_shift_logits=right_shift_logits,
        time_epsilon=time_epsilon,
        device=device,
    )


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
    gen = torch.Generator(device=device).manual_seed(seed)
    return _evaluate_stream(
        model,
        [(b, None, None) for b in batches],
        mask_token_id,
        vocab_size,
        generator=gen,
        right_shift_logits=right_shift_logits,
        time_epsilon=time_epsilon,
        device=device,
    )


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


# ---------- Held-out validation, early stopping, checkpoints, and exact resume (#409) ----------

def derive_seeds(base_seed: int, *, split_seed: int | None = None) -> dict[str, int]:
    """Distinct deterministic seeds for every stochastic source so validation never perturbs training."""
    s = int(base_seed)
    return {
        'split_seed': s if split_seed is None else int(split_seed),
        'train_mask_seed': s + 101,
        'val_mask_seed': s + 202,
        'val_noise_seed': s + 303,
        'train_order_seed': s + 404,
        'train_noise_seed': s + 505,
        'model_seed': s + 606,
    }


@dataclass(frozen=True)
class TrainConfig:
    """Hyperparameters and stopping rule for one masked LoRA training run (#409)."""

    max_updates: int
    eval_every: int
    batch_size: int = 8
    eval_batch_size: int = 8
    lr: float = 3e-4
    min_lr: float = 3e-5
    lr_schedule: str = 'gidd_cosine'
    warmup_updates: int = 0
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    patience: int | None = 3
    early_stopping_threshold: float = 0.001
    stop_after_updates: int | None = None
    right_shift_logits: bool = True
    time_epsilon: float = TIME_EPSILON
    device: str = 'cpu'
    seeds: dict[str, int] | None = None

    def resolved_seeds(self, fallback_seed: int = 2026) -> dict[str, int]:
        return dict(self.seeds) if self.seeds is not None else derive_seeds(fallback_seed)

    def to_dict(self) -> dict[str, Any]:
        return {
            'max_updates': self.max_updates,
            'eval_every': self.eval_every,
            'batch_size': self.batch_size,
            'eval_batch_size': self.eval_batch_size,
            'lr': self.lr,
            'min_lr': self.min_lr,
            'lr_schedule': self.lr_schedule,
            'warmup_updates': self.warmup_updates,
            'weight_decay': self.weight_decay,
            'grad_clip': self.grad_clip,
            'patience': self.patience,
            'early_stopping_threshold': self.early_stopping_threshold,
            'stop_after_updates': self.stop_after_updates,
            'right_shift_logits': self.right_shift_logits,
            'time_epsilon': self.time_epsilon,
        }


class _EpochBatchStream:
    """Stateful shuffled epoch stream over `train_rows` with O(1) checkpointable mid-epoch state."""

    def __init__(self, rows: Sequence[EncodedRow], order_gen: torch.Generator) -> None:
        if not rows:
            raise ValueError('train_rows must be non-empty')
        self.rows = rows
        self.order_gen = order_gen
        self.epoch_idx = 0
        self.epoch_cursor = 0
        self.current_epoch_seed: int | None = None
        self.teams_seen = 0
        self._perm: list[int] = []

    def _ensure_epoch_perm(self) -> None:
        if self._perm and self.epoch_cursor < len(self.rows):
            return
        if self.epoch_cursor >= len(self.rows):
            self.epoch_idx += 1
            self.epoch_cursor = 0
        self.current_epoch_seed = int(torch.randint(0, 2**31 - 1, (1,), generator=self.order_gen).item())
        self._perm = torch.randperm(
            len(self.rows), generator=torch.Generator().manual_seed(self.current_epoch_seed)
        ).tolist()

    def next_batch(self, batch_size: int, pad_id: int) -> Batch:
        self._ensure_epoch_perm()
        end = min(self.epoch_cursor + batch_size, len(self.rows))
        chosen = [self.rows[i] for i in self._perm[self.epoch_cursor:end]]
        self.epoch_cursor = end
        self.teams_seen += len(chosen)
        return collate_rows(chosen, pad_id)

    def epoch_share(self) -> float:
        return self.teams_seen / len(self.rows)

    def state_dict(self) -> dict[str, Any]:
        return {
            'epoch_idx': self.epoch_idx,
            'epoch_cursor': self.epoch_cursor,
            'current_epoch_seed': self.current_epoch_seed,
            'teams_seen': self.teams_seen,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.epoch_idx = int(state['epoch_idx'])
        self.epoch_cursor = int(state['epoch_cursor'])
        seed = state['current_epoch_seed']
        self.current_epoch_seed = None if seed is None else int(seed)
        self.teams_seen = int(state['teams_seen'])
        if self.current_epoch_seed is not None:
            self._perm = torch.randperm(
                len(self.rows), generator=torch.Generator().manual_seed(self.current_epoch_seed)
            ).tolist()
        else:
            self._perm = []


def _build_scheduler(optimizer: torch.optim.Optimizer, config: TrainConfig) -> torch.optim.lr_scheduler.LambdaLR:
    if config.lr <= 0.0:
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _step: 1.0)
    effective_warmup = min(config.warmup_updates, max(0, config.max_updates - 1))
    if config.lr_schedule == 'cosine':
        lr_at = lr_sched.make_lr_at(
            'cosine', peak_lr=config.lr, total_steps=max(1, config.max_updates),
            warmup_steps=max(1, effective_warmup) if effective_warmup > 0 else 1, min_lr=0.0,
        )
    else:
        lr_at = lr_sched.make_lr_at(
            config.lr_schedule, peak_lr=config.lr, total_steps=max(1, config.max_updates),
            warmup_steps=effective_warmup, min_lr=config.min_lr,
        )
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: lr_at(step) / config.lr)


def _capture_rng_states(
    noise_gen: torch.Generator, order_gen: torch.Generator, device: str | torch.device
) -> dict[str, Any]:
    cuda_state = torch.cuda.get_rng_state(device) if torch.device(device).type == 'cuda' else None
    return {
        'noise_gen': noise_gen.get_state(),
        'order_gen': order_gen.get_state(),
        'torch_cpu': torch.get_rng_state(),
        'torch_cuda': cuda_state,
    }


def _restore_rng_states(
    rng_states: dict[str, Any],
    noise_gen: torch.Generator,
    order_gen: torch.Generator,
    device: str | torch.device,
) -> None:
    noise_gen.set_state(rng_states['noise_gen'])
    order_gen.set_state(rng_states['order_gen'])
    torch.set_rng_state(rng_states['torch_cpu'])
    if torch.device(device).type == 'cuda' and rng_states.get('torch_cuda') is not None:
        torch.cuda.set_rng_state(rng_states['torch_cuda'], device)


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    torch.save(payload, tmp)
    os.replace(tmp, path)
    return path


def _atomic_write_json(payload: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(payload, indent=2) + '\n')
    os.replace(tmp, path)
    return path


def load_training_checkpoint(path: Path) -> dict[str, Any]:
    """Load a masked LoRA checkpoint safely via src/checkpoints.py (`weights_only=True`)."""
    return ckpt_io.load_checkpoint(path, map_location='cpu')


@dataclass
class _RunState:
    update: int = 0
    val_index: int = 0
    prior_elapsed: float = 0.0
    stopped_early: bool = False
    stop_reason: str | None = None
    train_curve: list[dict[str, Any]] | None = None
    val_curve: list[dict[str, Any]] | None = None
    interval_losses: list[float] | None = None
    interval_grad_norms: list[float] | None = None
    best_lora_state: dict[str, torch.Tensor] | None = None

    def __post_init__(self) -> None:
        self.train_curve = [] if self.train_curve is None else self.train_curve
        self.val_curve = [] if self.val_curve is None else self.val_curve
        self.interval_losses = [] if self.interval_losses is None else self.interval_losses
        self.interval_grad_norms = [] if self.interval_grad_norms is None else self.interval_grad_norms


@dataclass
class _TrainingSession:
    """Bundled mutable state for one masked LoRA training run so helpers share one session object."""

    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LambdaLR
    stopper: EarlyStopper
    stream: _EpochBatchStream
    noise_gen: torch.Generator
    order_gen: torch.Generator
    state: _RunState
    split_record: SplitRecord
    seeds: dict[str, int]
    config: TrainConfig
    out_dir: Path
    best_ckpt_path: Path
    last_ckpt_path: Path
    mask_token_id: int
    pad_id: int
    n_train: int
    n_val: int
    tracker: Any | None = None
    resume_path: Path | None = None
    candidate_info: dict[str, Any] | None = None
    pad_token: str = '<|endoftext|>'
    dtype_name: str = 'float32'
    t0: float = 0.0
    last_saved_update: int = -1

    def elapsed_seconds(self) -> float:
        return self.state.prior_elapsed + (time.perf_counter() - self.t0)

    def checkpoint_payload(self, elapsed_seconds: float) -> dict[str, Any]:
        return {
            'format_version': 1,
            'lora_state_dict': _snapshot_lora_state(self.model),
            'best_lora_state_dict': (
                {k: v.detach().cpu().clone() for k, v in self.state.best_lora_state.items()}
                if self.state.best_lora_state is not None
                else None
            ),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'early_stopper_state': self.stopper.state_dict(),
            'stream_state': self.stream.state_dict(),
            'rng_states': _capture_rng_states(self.noise_gen, self.order_gen, self.config.device),
            'progress': {
                'update': self.state.update,
                'val_index': self.state.val_index,
                'elapsed_seconds': elapsed_seconds,
                'stopped_early': self.state.stopped_early,
                'stop_reason': self.state.stop_reason,
                'interval_losses': list(self.state.interval_losses or ()),
                'interval_grad_norms': list(self.state.interval_grad_norms or ()),
            },
            'curves': {
                'train': list(self.state.train_curve or ()),
                'val': list(self.state.val_curve or ()),
            },
            'split_hash': self.split_record.split_hash,
            'seeds': dict(self.seeds),
            'config': self.config.to_dict(),
        }


def _restore_from_checkpoint(session: _TrainingSession, resume_path: Path) -> None:
    ckpt = load_training_checkpoint(resume_path)
    if ckpt.get('split_hash') != session.split_record.split_hash:
        raise ValueError(
            f'refusing to resume {resume_path}: checkpoint split_hash {ckpt.get("split_hash")!r} '
            f'does not match current split_hash {session.split_record.split_hash!r}'
        )
    _restore_lora_state(session.model, ckpt['lora_state_dict'])
    session.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    session.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    session.stopper.load_state_dict(ckpt['early_stopper_state'])
    session.stream.load_state_dict(ckpt['stream_state'])
    _restore_rng_states(ckpt['rng_states'], session.noise_gen, session.order_gen, session.config.device)

    best_lora = ckpt.get('best_lora_state_dict')
    sibling_best = resume_path.parent / 'best.pt'
    if not session.best_ckpt_path.exists() and sibling_best.is_file():
        session.best_ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        session.best_ckpt_path.write_bytes(sibling_best.read_bytes())
    if best_lora is None and session.best_ckpt_path.is_file():
        best_lora = load_training_checkpoint(session.best_ckpt_path)['lora_state_dict']

    prog = ckpt['progress']
    curves = ckpt['curves']
    session.state = _RunState(
        update=int(prog['update']),
        val_index=int(prog['val_index']),
        prior_elapsed=float(prog['elapsed_seconds']),
        stopped_early=bool(prog['stopped_early']),
        stop_reason=prog['stop_reason'],
        train_curve=list(curves['train']),
        val_curve=list(curves['val']),
        interval_losses=list(prog.get('interval_losses', ())),
        interval_grad_norms=list(prog.get('interval_grad_norms', ())),
        best_lora_state=best_lora,
    )
    session.last_saved_update = session.state.update


def _format_early_stop_reason(stopper: EarlyStopper, val_index: int) -> str:
    meta = stopper.best_meta
    best_update = meta.get('update', 0)
    best_epoch = meta.get('epoch_share', 0.0)
    return (
        f'early_stopping: no validation loss gain > {stopper.threshold:g} for '
        f'{stopper.patience} intervals by validation {val_index} '
        f'(best validation {stopper.best_index} at update {best_update}, epoch {best_epoch:.4f})'
    )


def _record_validation_step(session: _TrainingSession, eval_metrics: dict[str, Any]) -> None:
    st = session.state
    val_loss = float(eval_metrics['mdlm_loss'])
    epoch_share = session.stream.epoch_share()
    elapsed = session.elapsed_seconds()
    meta = {'update': st.update, 'epoch_share': epoch_share, 'teams_seen': session.stream.teams_seen}
    is_best, should_stop = session.stopper.observe(st.val_index, val_loss, meta=meta)
    mean_train = sum(st.interval_losses) / len(st.interval_losses) if st.interval_losses else None
    mean_gnorm = sum(st.interval_grad_norms) / len(st.interval_grad_norms) if st.interval_grad_norms else None
    entry: dict[str, Any] = {
        'val_index': st.val_index,
        'update': st.update,
        'epoch': epoch_share,
        'teams_seen': session.stream.teams_seen,
        'val_loss': val_loss,
        'unweighted_masked_ce': float(eval_metrics.get('unweighted_masked_ce', val_loss)),
        'value_token_ce': float(eval_metrics.get('value_token_ce', val_loss)),
        'slot_pad_token_ce': float(eval_metrics.get('slot_pad_token_ce', val_loss)),
        'best_val_loss': session.stopper.best,
        'is_best': is_best,
        'minutes': round(elapsed / 60.0, 4),
    }
    if mean_train is not None:
        entry['train_loss'] = mean_train
    assert st.val_curve is not None and st.interval_losses is not None and st.interval_grad_norms is not None
    st.val_curve.append(entry)
    if session.tracker is not None:
        session.tracker.log_epoch(
            entry,
            lr=float(session.optimizer.param_groups[0]['lr']),
            step=st.update,
            grad_norm=mean_gnorm,
            best_val_loss=session.stopper.best,
        )
    st.interval_losses.clear()
    st.interval_grad_norms.clear()

    if is_best:
        st.best_lora_state = _snapshot_lora_state(session.model)
    if should_stop:
        st.stopped_early = True
        st.stop_reason = _format_early_stop_reason(session.stopper, st.val_index)

    payload = session.checkpoint_payload(elapsed)
    _atomic_torch_save(payload, session.last_ckpt_path)
    if is_best:
        _atomic_torch_save(payload, session.best_ckpt_path)
    session.last_saved_update = st.update
    print(
        f'[val {st.val_index:3d}] update {st.update:4d}/{session.config.max_updates} '
        f'epoch {epoch_share:.4f} teams {session.stream.teams_seen} '
        f'val_loss={val_loss:.4f} best={session.stopper.best:.4f}'
        + (' *best*' if is_best else ''),
        flush=True,
    )


def _validate_train_config(config: TrainConfig) -> None:
    if config.max_updates < 1:
        raise ValueError(f'max_updates must be >= 1, got {config.max_updates}')
    if config.eval_every < 1:
        raise ValueError(f'eval_every must be >= 1, got {config.eval_every}')


def _finalize_stop_reason(state: _RunState, config: TrainConfig) -> str:
    if state.stop_reason is not None:
        return state.stop_reason
    if (
        config.stop_after_updates is not None
        and state.update >= config.stop_after_updates
        and state.update < config.max_updates
    ):
        return f'stopped_after_updates: stopped at update {state.update} of {config.max_updates}'
    return f'reached_cap: completed {state.update} of {config.max_updates} optimizer updates'


def _build_train_result(
    session: _TrainingSession,
    *,
    stop_reason: str,
    best_restored: bool,
    elapsed_seconds: float,
) -> dict[str, Any]:
    best_meta = session.stopper.best_meta
    best_update = int(best_meta.get('update', 0))
    best_epoch_share = float(best_meta.get('epoch_share', 0.0))
    teams_seen_at_best = int(best_meta.get('teams_seen', 0))
    gpu_name = torch.cuda.get_device_name(0) if torch.device(session.config.device).type == 'cuda' else None
    result: dict[str, Any] = {
        'candidate': session.candidate_info,
        'device': str(session.config.device),
        'gpu_name': gpu_name,
        'dtype': session.dtype_name,
        'pad_token': session.pad_token,
        'pad_id': session.pad_id,
        'score_slot_pad': True,
        'score_batch_pad': False,
        'parameter_counts': parameter_counts(session.model),
        'split': session.split_record.summary_dict('split.json'),
        'split_hash': session.split_record.split_hash,
        'seeds': dict(session.seeds),
        'config': session.config.to_dict(),
        'stop_reason': stop_reason,
        'stopped_early': session.state.stopped_early,
        'best_update': best_update,
        'best_val_index': session.stopper.best_index,
        'best_epoch_share': best_epoch_share,
        'best_val_loss': None if session.stopper.best == math.inf else session.stopper.best,
        'stopped_update': session.state.update,
        'stopped_val_index': session.state.val_index,
        'stopped_epoch_share': session.stream.epoch_share(),
        'teams_seen': session.stream.teams_seen,
        'teams_seen_at_best': teams_seen_at_best,
        'best_restored': best_restored,
        'cap_updates': session.config.max_updates,
        'eval_every': session.config.eval_every,
        'patience': session.config.patience,
        'early_stopping_threshold': session.config.early_stopping_threshold,
        'elapsed_seconds': elapsed_seconds,
        'minutes': round(elapsed_seconds / 60.0, 4),
        'resumed_from': session.resume_path.name if session.resume_path is not None else None,
        'checkpoints': {'best': 'checkpoints/best.pt', 'last': 'checkpoints/last.pt'},
        'curves': {
            'train': list(session.state.train_curve or ()),
            'val': list(session.state.val_curve or ()),
        },
    }
    if session.tracker is not None and (meta := session.tracker.metadata()):
        result['wandb'] = meta
    return result


def _make_val_evaluator(
    val_rows: Sequence[EncodedRow],
    *,
    mask_token_id: int,
    pad_id: int,
    vocab_size: int,
    val_noise_seed: int,
    config: TrainConfig,
    eval_fn: Callable[[torch.nn.Module, int], dict[str, Any]] | None,
) -> Callable[[torch.nn.Module, int], dict[str, Any]]:
    if eval_fn is not None:
        return eval_fn
    fixed_val = build_fixed_validation(val_rows, config.eval_batch_size, pad_id, val_noise_seed)
    return lambda m, _idx: evaluate_fixed_validation(
        m, fixed_val, mask_token_id, vocab_size,
        right_shift_logits=config.right_shift_logits, time_epsilon=config.time_epsilon, device=config.device,
    )


def _seed_torch(seed: int, device: str | torch.device) -> None:
    torch.manual_seed(seed)
    if torch.device(device).type == 'cuda':
        torch.cuda.manual_seed_all(seed)


def _create_training_session(
    model: torch.nn.Module,
    train_rows: Sequence[EncodedRow],
    val_rows: Sequence[EncodedRow],
    *,
    mask_token_id: int,
    pad_id: int,
    out_dir: Path,
    split_record: SplitRecord,
    config: TrainConfig,
    seeds: dict[str, int],
    tracker: Any | None,
    resume_path: Path | None,
    candidate_info: dict[str, Any] | None,
    pad_token: str,
    dtype_name: str,
) -> _TrainingSession:
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=config.lr, weight_decay=config.weight_decay)
    scheduler = _build_scheduler(optimizer, config)
    stopper = EarlyStopper(patience=config.patience, threshold=config.early_stopping_threshold)
    noise_gen = torch.Generator().manual_seed(seeds['train_noise_seed'])
    order_gen = torch.Generator().manual_seed(seeds['train_order_seed'])
    stream = _EpochBatchStream(train_rows, order_gen)
    _seed_torch(seeds['model_seed'], config.device)
    session = _TrainingSession(
        model=model, optimizer=optimizer, scheduler=scheduler, stopper=stopper,
        stream=stream, noise_gen=noise_gen, order_gen=order_gen, state=_RunState(),
        split_record=split_record, seeds=seeds, config=config, out_dir=out_dir,
        best_ckpt_path=out_dir / 'checkpoints' / 'best.pt',
        last_ckpt_path=out_dir / 'checkpoints' / 'last.pt',
        mask_token_id=mask_token_id, pad_id=pad_id,
        n_train=len(train_rows), n_val=len(val_rows),
        tracker=tracker, resume_path=resume_path, candidate_info=candidate_info,
        pad_token=pad_token, dtype_name=dtype_name, t0=time.perf_counter(),
    )
    if resume_path is not None:
        _restore_from_checkpoint(session, resume_path)
    return session


def _run_update_loop(
    session: _TrainingSession,
    run_eval: Callable[[torch.nn.Module, int], dict[str, Any]],
) -> None:
    st, cfg = session.state, session.config
    while st.update < cfg.max_updates and not st.stopped_early:
        if cfg.stop_after_updates is not None and st.update >= cfg.stop_after_updates:
            break
        batch = session.stream.next_batch(cfg.batch_size, session.pad_id)
        lr_used = float(session.optimizer.param_groups[0]['lr'])
        out, grad_norm = _train_step_with_grad_norm(
            session.model, session.optimizer, batch.to(cfg.device), session.mask_token_id,
            right_shift_logits=cfg.right_shift_logits, time_epsilon=cfg.time_epsilon,
            generator=session.noise_gen, grad_clip=cfg.grad_clip,
        )
        session.scheduler.step()
        st.update += 1
        loss_val = float(out.loss.item())
        assert st.train_curve is not None and st.interval_losses is not None and st.interval_grad_norms is not None
        st.train_curve.append({
            'update': st.update,
            'epoch': session.stream.epoch_share(),
            'teams_seen': session.stream.teams_seen,
            'train_loss': loss_val,
            'lr': lr_used,
            'grad_norm': grad_norm,
            'masked_tokens': out.masked_count,
            'eligible_tokens': out.eligible_count,
        })
        st.interval_losses.append(loss_val)
        st.interval_grad_norms.append(grad_norm)

        if st.update % cfg.eval_every == 0 or st.update == cfg.max_updates:
            st.val_index += 1
            _record_validation_step(session, run_eval(session.model, st.val_index))


def _finalize_training_run(session: _TrainingSession) -> dict[str, Any]:
    elapsed_total = session.elapsed_seconds()
    stop_reason = _finalize_stop_reason(session.state, session.config)
    if session.last_saved_update != session.state.update:
        _atomic_torch_save(session.checkpoint_payload(elapsed_total), session.last_ckpt_path)

    best_restored = False
    if session.best_ckpt_path.is_file():
        best_ckpt = load_training_checkpoint(session.best_ckpt_path)
        _restore_lora_state(session.model, best_ckpt['lora_state_dict'])
        best_restored = True
    elif session.state.best_lora_state is not None:
        _restore_lora_state(session.model, session.state.best_lora_state)
        best_payload = session.checkpoint_payload(elapsed_total)
        best_payload['lora_state_dict'] = {
            k: v.detach().cpu().clone() for k, v in session.state.best_lora_state.items()
        }
        _atomic_torch_save(best_payload, session.best_ckpt_path)
        best_restored = True

    result = _build_train_result(
        session, stop_reason=stop_reason, best_restored=best_restored, elapsed_seconds=elapsed_total,
    )
    _atomic_write_json(result, session.out_dir / 'results.json')
    if session.tracker is not None:
        session.tracker.log_summary(
            {
                'best_epoch': result['best_epoch_share'],
                'stopped_epoch': result['stopped_epoch_share'],
                'best_val_loss': result['best_val_loss'],
                'minutes': result['minutes'],
                'n_train': session.n_train,
                'val_rows': session.n_val,
                'curves': result['curves']['val'],
            },
            stop_reason=stop_reason,
            checkpoint_name=session.best_ckpt_path.name,
        )
    return result


def train_masked_lora(
    model: torch.nn.Module,
    train_rows: Sequence[EncodedRow],
    val_rows: Sequence[EncodedRow],
    *,
    mask_token_id: int,
    pad_id: int,
    vocab_size: int,
    out_dir: Path,
    split_record: SplitRecord,
    config: TrainConfig,
    eval_fn: Callable[[torch.nn.Module, int], dict[str, Any]] | None = None,
    tracker: Any | None = None,
    resume_path: Path | None = None,
    candidate_info: dict[str, Any] | None = None,
    pad_token: str = '<|endoftext|>',
    dtype_name: str = 'float32',
) -> dict[str, Any]:
    """Train LoRA adapters with fixed-noise held-out validation, early stopping, exact checkpoints, and W&B mirror."""
    _validate_train_config(config)
    seeds = config.resolved_seeds(split_record.split_seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(split_record.to_dict(), out_dir / 'split.json')
    run_eval = _make_val_evaluator(
        val_rows, mask_token_id=mask_token_id, pad_id=pad_id, vocab_size=vocab_size,
        val_noise_seed=seeds['val_noise_seed'], config=config, eval_fn=eval_fn,
    )
    session = _create_training_session(
        model, train_rows, val_rows,
        mask_token_id=mask_token_id, pad_id=pad_id, out_dir=out_dir,
        split_record=split_record, config=config, seeds=seeds,
        tracker=tracker, resume_path=resume_path, candidate_info=candidate_info,
        pad_token=pad_token, dtype_name=dtype_name,
    )
    own_tracker = tracker is not None and not getattr(tracker, 'active', False)
    if tracker is not None:
        tracker.start(config={
            'candidate': (candidate_info or {}).get('key'),
            'split_hash': split_record.split_hash,
            'n_train': len(train_rows),
            'val_rows': len(val_rows),
            'seeds': seeds,
            **config.to_dict(),
        })
    try:
        _run_update_loop(session, run_eval)
        return _finalize_training_run(session)
    finally:
        if own_tracker:
            tracker.finish()


def run_train(args: argparse.Namespace) -> dict[str, Any]:
    """CLI entry point for `train`: seed RNG before LoRA init, build split, run early-stopped training."""
    device = _resolve_device(args.device)
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    split_seed = args.seed if args.split_seed is None else args.split_seed
    seeds = derive_seeds(args.seed, split_seed=split_seed)
    _seed_torch(seeds['model_seed'], device)
    model, tokenizer, info = load_diffugpt_small_lora(device, args.dtype)
    vocab = load_regmb_vocab(VOCAB_FILE)
    excluded = load_excluded_groups(args.excluded_keys)
    train_teams, val_teams, split_record = load_split_teams(
        args.data, vocab, val_fraction=args.val_fraction, split_seed=seeds['split_seed'],
        excluded_groups=excluded, max_rows=args.max_rows,
    )
    train_rows = build_dataset_rows(train_teams, tokenizer, seed=seeds['train_mask_seed'], limit=TEXT_LIMIT)
    val_rows = build_dataset_rows(val_teams, tokenizer, seed=seeds['val_mask_seed'], limit=TEXT_LIMIT)
    pad_token = resolve_pad_token(tokenizer)
    pad_id = resolve_pad_id(tokenizer)
    vocab_size = int(getattr(model.config, 'vocab_size', len(tokenizer)))
    config = TrainConfig(
        max_updates=args.max_updates,
        eval_every=args.eval_every,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        lr=args.lr,
        min_lr=args.min_lr,
        lr_schedule=args.lr_schedule,
        warmup_updates=args.warmup_updates,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        patience=args.patience,
        early_stopping_threshold=args.early_stopping_threshold,
        stop_after_updates=args.stop_after_updates,
        device=device,
        seeds=seeds,
    )
    tracker = wandb_track.from_cli_args(
        args, name=f"masked_lora_{info['key']}_s{args.seed}", job_type='train',
        tags=('masked-lora', info['key']),
    )
    return train_masked_lora(
        model, train_rows, val_rows,
        mask_token_id=int(info['mask_id']), pad_id=pad_id, vocab_size=vocab_size,
        out_dir=args.out, split_record=split_record, config=config,
        tracker=tracker, resume_path=args.resume, candidate_info=info,
        pad_token=pad_token, dtype_name=args.dtype,
    )


def _train_subprocess_cmd(args: argparse.Namespace, out_dir: Path, extra: Sequence[str]) -> list[str]:
    cmd = [
        sys.executable, str(Path(__file__).resolve()), 'train',
        '--data', str(args.data), '--out', str(out_dir),
        '--val-fraction', str(args.val_fraction), '--split-seed', str(args.seed if args.split_seed is None else args.split_seed),
        '--max-updates', str(args.max_updates), '--eval-every', str(args.eval_every),
        '--batch-size', str(args.batch_size), '--eval-batch-size', str(args.eval_batch_size),
        '--lr', str(args.lr), '--min-lr', str(args.min_lr), '--lr-schedule', str(args.lr_schedule),
        '--warmup-updates', str(args.warmup_updates), '--patience', str(args.patience),
        '--early-stopping-threshold', str(args.early_stopping_threshold),
        '--excluded-keys', str(args.excluded_keys),
        '--seed', str(args.seed), '--device', str(args.device), '--dtype', str(args.dtype),
        '--no-wandb',
    ]
    if args.max_rows is not None:
        cmd.extend(['--max-rows', str(args.max_rows)])
    cmd.extend(extra)
    return cmd


def run_verify_resume(args: argparse.Namespace) -> dict[str, Any]:
    """Run an uninterrupted train process vs a stopped-and-resumed cold process and verify loss parity (#409)."""
    args.out.mkdir(parents=True, exist_ok=True)
    full_dir = args.out / 'uninterrupted'
    resumed_dir = args.out / 'resumed'
    script_dir = str(Path(__file__).resolve().parent)

    subprocess.run(_train_subprocess_cmd(args, full_dir, ()), cwd=script_dir, check=True)
    subprocess.run(
        _train_subprocess_cmd(args, resumed_dir, ('--stop-after-updates', str(args.stop_at))),
        cwd=script_dir, check=True,
    )
    resume_ckpt = resumed_dir / 'checkpoints' / 'last.pt'
    subprocess.run(
        _train_subprocess_cmd(args, resumed_dir, ('--resume', str(resume_ckpt))),
        cwd=script_dir, check=True,
    )

    full_res = json.loads((full_dir / 'results.json').read_text())
    resumed_res = json.loads((resumed_dir / 'results.json').read_text())
    full_after = [e for e in full_res['curves']['train'] if e['update'] > args.stop_at]
    resumed_after = [e for e in resumed_res['curves']['train'] if e['update'] > args.stop_at]
    train_diffs = [
        {
            'update': a['update'],
            'uninterrupted_loss': a['train_loss'],
            'resumed_loss': b['train_loss'],
            'abs_diff': abs(a['train_loss'] - b['train_loss']),
        }
        for a, b in zip(full_after, resumed_after, strict=True)
    ]
    val_diffs = [
        {
            'val_index': a['val_index'],
            'update': a['update'],
            'uninterrupted_val_loss': a['val_loss'],
            'resumed_val_loss': b['val_loss'],
            'abs_diff': abs(a['val_loss'] - b['val_loss']),
        }
        for a, b in zip(full_res['curves']['val'], resumed_res['curves']['val'], strict=True)
    ]
    max_train_diff = max((d['abs_diff'] for d in train_diffs), default=0.0)
    max_val_diff = max((d['abs_diff'] for d in val_diffs), default=0.0)
    summary = {
        'gpu_name': full_res.get('gpu_name'),
        'device': full_res.get('device'),
        'dtype': full_res.get('dtype'),
        'split_hash': full_res['split_hash'],
        'stop_at': args.stop_at,
        'max_updates': args.max_updates,
        'eval_every': args.eval_every,
        'max_train_loss_abs_diff': max_train_diff,
        'max_val_loss_abs_diff': max_val_diff,
        'parity_passed': max_train_diff <= args.atol and max_val_diff <= args.atol,
        'atol': args.atol,
        'train_diffs_after_resume': train_diffs,
        'val_diffs': val_diffs,
        'uninterrupted_stop_reason': full_res['stop_reason'],
        'resumed_stop_reason': resumed_res['stop_reason'],
        'uninterrupted_best_update': full_res['best_update'],
        'resumed_best_update': resumed_res['best_update'],
    }
    _atomic_write_json(summary, args.out / 'resume_verification.json')
    for ckpt in (*full_dir.glob('checkpoints/*.pt'), *resumed_dir.glob('checkpoints/*.pt')):
        ckpt.unlink(missing_ok=True)
    if not summary['parity_passed']:
        raise RuntimeError(
            f'resume parity failed: max_train_diff={max_train_diff:.2e}, '
            f'max_val_diff={max_val_diff:.2e} (atol={args.atol:.2e})'
        )
    return summary


def _add_shared_train_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--val-fraction', type=float, default=0.02)
    parser.add_argument('--split-seed', type=int, default=None)
    parser.add_argument('--max-rows', type=int, default=None)
    parser.add_argument('--max-updates', type=int, default=100)
    parser.add_argument('--eval-every', type=int, default=20)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--early-stopping-threshold', type=float, default=0.001)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--eval-batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--min-lr', type=float, default=3e-5)
    parser.add_argument('--lr-schedule', choices=lr_sched.SCHEDULES, default='gidd_cosine')
    parser.add_argument('--warmup-updates', type=int, default=0)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--grad-clip', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--dtype', choices=('float32', 'bfloat16', 'float16'), default='float32')
    parser.add_argument('--excluded-keys', type=Path, default=EXCLUDED_TEAMS_FILE)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Masked text diffusion Low-Rank Adaptation (LoRA) trainer (#408, #409).')
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

    tn = sub.add_parser('train', help='Train LoRA with held-out validation, early stopping, and checkpoint resume.')
    _add_shared_train_args(tn)
    tn.add_argument('--resume', type=Path, default=None)
    tn.add_argument('--stop-after-updates', type=int, default=None)
    wandb_track.add_cli_args(tn)

    vr = sub.add_parser('verify-resume', help='Verify cold-process checkpoint resume parity against uninterrupted run.')
    _add_shared_train_args(vr)
    vr.add_argument('--stop-at', type=int, default=4)
    vr.add_argument('--atol', type=float, default=1e-5)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == 'slice-data':
        summary = slice_training_npz(args.data, args.out, args.count, args.seed)
        print(json.dumps(summary, indent=2), flush=True)
        return 0
    if args.command == 'train':
        result = run_train(args)
        print(
            f"Training finished ({result['stop_reason']}): best update {result['best_update']} "
            f"(epoch {result['best_epoch_share']:.4f}, val_loss={result['best_val_loss']:.4f}), "
            f"teams seen {result['teams_seen']}",
            flush=True,
        )
        return 0
    if args.command == 'verify-resume':
        summary = run_verify_resume(args)
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
