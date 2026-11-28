"""Shared best-loss and patience tracker extracted from diffusion84.Model84.train_new (#409).

Used by both the 84-field discrete diffusion trainer (`diffusion84.Model84._fit`) and the masked
text diffusion Low-Rank Adaptation (LoRA) trainer (`scripts/masked_lora_trainer.py`) so every
trainer follows one early-stopping rule (`early-stopping` skill):

- Any validation loss strictly lower than the best seen so far updates the kept best checkpoint.
- Only an improvement larger than `threshold` resets the patience counter (`reset_index`).
- Stopping fires once `index - reset_index >= patience` (with `patience` truthy and `index > 0`).
- Non-finite validation losses raise `RuntimeError` immediately.
"""
from __future__ import annotations

import math
from typing import Any


class EarlyStopper:
    """Best-loss and patience state machine with JSON/torch-safe state_dict serialization."""

    def __init__(self, patience: int | None = 3, threshold: float = 0.001) -> None:
        if patience is not None and patience < 0:
            raise ValueError(f'patience must be non-negative or None, got {patience}')
        if threshold < 0:
            raise ValueError(f'threshold must be non-negative, got {threshold}')
        self.patience: int | None = patience
        self.threshold: float = float(threshold)
        self.best: float = math.inf
        self.best_index: int | None = None
        self.reset_index: int = 0
        self.best_meta: dict[str, Any] = {}

    def observe(
        self,
        index: int,
        val_loss: float,
        *,
        meta: dict[str, Any] | None = None,
    ) -> tuple[bool, bool]:
        """Record validation loss at `index` (epoch or validation interval) and return `(is_best, should_stop)`."""
        loss = float(val_loss)
        if not math.isfinite(loss):
            raise RuntimeError(f'non-finite validation loss at index {index}: {val_loss}')
        is_best = loss < self.best
        if is_best:
            if self.best - loss > self.threshold:
                self.reset_index = int(index)
            self.best = loss
            self.best_index = int(index)
            self.best_meta = dict(meta) if meta else {}
        return is_best, self.should_stop(index)

    def should_stop(self, index: int) -> bool:
        """True when patience is enabled and at least `patience` intervals passed since `reset_index`."""
        return bool(self.patience and index > 0 and (int(index) - self.reset_index >= self.patience))

    def state_dict(self) -> dict[str, Any]:
        """Plain Python dict safe for both JSON and `torch.load(..., weights_only=True)`."""
        return {
            'patience': self.patience,
            'threshold': self.threshold,
            'best': self.best,
            'best_index': self.best_index,
            'reset_index': self.reset_index,
            'best_meta': dict(self.best_meta),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore exact patience and best-loss tracking state from `state_dict()`."""
        self.patience = state['patience']
        self.threshold = float(state['threshold'])
        self.best = float(state['best'])
        self.best_index = None if state['best_index'] is None else int(state['best_index'])
        self.reset_index = int(state['reset_index'])
        self.best_meta = dict(state.get('best_meta') or {})
