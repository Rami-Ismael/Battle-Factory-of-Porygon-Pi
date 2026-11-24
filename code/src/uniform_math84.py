"""Pure PyTorch/NumPy mathematical core for 84-field uniform discrete diffusion (UDLM, issue #275).

Loads no game data or /tmp/vgc-pilot modules, so GitHub Actions CI can test the loss, reverse
posterior, posterior-mixing CFG, domain guards, NaN guards, and revision tracker on CPU.

Schedule: self-consistent ideal log-linear schedule alpha(t) = 1 - t, move_chance(t) = t shared
across all 84 fields (Schiff et al. ICLR 2025 Eq. 19; distinct from upstream LogLinearNoise's
internal eps=1e-3 in forward/reverse transitions).
"""
import math

import numpy as np
import torch
import torch.nn.functional as F

import stat_points as SP
import uniform_preflight84 as UP

NSLOT = SP.NSLOT                                         # 6 Pokémon per team
NF = 8                                                   # species, ability, item, 4 moves, Stat Alignment
STAT0 = NSLOT * NF                                       # 48
COLS = STAT0 + SP.COLS                                   # 84
NAN = float("nan")
TRAINING_EPS = UP.TRAINING_EPS                           # 1e-3
SAMPLING_EPS = UP.SAMPLING_EPS                           # 1e-5
DEFAULT_PASSES = 128
CONFIDENT_MARGIN = 0.10                                  # top-1 minus top-2 probability margin for confident mind-changes
PHASE_NAMES = ("high_noise", "mid_noise", "low_noise")   # t > 2/3, 1/3 < t <= 2/3, t <= 1/3

FIELD_KEYS = ("species", "ability", "item", "move", "nature", "stat")
FIELD_COLS = {
    "species": [slot * NF for slot in range(NSLOT)],
    "ability": [slot * NF + 1 for slot in range(NSLOT)],
    "item": [slot * NF + 2 for slot in range(NSLOT)],
    "move": [slot * NF + offset for slot in range(NSLOT) for offset in range(3, 7)],
    "nature": [slot * NF + 7 for slot in range(NSLOT)],
    "stat": list(range(STAT0, COLS)),
}


def table_to_uniform(table_rows):
    """Convert 84-field table tokens (1..N_k; 0 is [MASK]) to 0-based real uniform tokens (0..N_k - 1)."""
    if bool((table_rows < 1).any()):
        raise ValueError("table_to_uniform received a [MASK] or negative token (< 1)")
    return table_rows - 1


def uniform_to_table(uniform_tokens):
    """Convert 0-based real uniform tokens (0..N_k - 1) back to 84-field table tokens (1..N_k)."""
    if bool((uniform_tokens < 0).any()):
        raise ValueError("uniform_to_table received a negative token (< 0)")
    return uniform_tokens + 1


def is_better_val_loss(val_loss_value, best_val_loss):
    """True iff val_loss_value is finite and strictly lower than best_val_loss (rejects NaN/Inf)."""
    return math.isfinite(val_loss_value) and val_loss_value < best_val_loss


def corrupt_uniform(clean_tokens, time_values, vocab_sizes, generator=None):
    """Apply uniform forward corruption q(z_t | x_0) = Cat((1 - t) x_0 + t u) across all 84 fields.

    Corrupted positions are redrawn uniformly from 0..N_k - 1 (which may equal the clean token).
    """
    batch_size = clean_tokens.shape[0]
    device = clean_tokens.device
    noisy_tokens = clean_tokens.clone()
    time_col = time_values.view(batch_size, 1)
    for field_key, cols in FIELD_COLS.items():
        width = len(cols)
        vocab_size = vocab_sizes[field_key]
        if generator is None:
            draws = torch.rand((batch_size, width), device=device)
            random_tokens = torch.randint(0, vocab_size, (batch_size, width), device=device)
        else:
            draws = torch.rand((batch_size, width), generator=generator)
            random_tokens = torch.randint(0, vocab_size, (batch_size, width), generator=generator)
        corrupt_mask = draws < time_col
        noisy_tokens[:, cols] = torch.where(corrupt_mask, random_tokens.to(device), clean_tokens[:, cols])
    return noisy_tokens


def uniform_field_loss(log_clean_probs, noisy_tokens, clean_tokens, time_values, vocab_size):
    """Continuous-time uniform discrete diffusion loss (Schiff et al. Eq. 19, diffusion.py:563-614).

    Requires strictly interior times 0 < t < 1 so alpha(t) = 1 - t > 0 and move_chance = t > 0.
    Computes move_chance = t directly (never 1 - (1 - t)) for float32/MPS numerical hygiene.

    Returns:
        [B, L] per-position continuous-time diffusion loss in nats.
    """
    if bool(((time_values <= 0.0) | (time_values >= 1.0) | ~torch.isfinite(time_values)).any()):
        raise ValueError("uniform_field_loss requires finite time_values strictly inside (0, 1)")
    dtype = log_clean_probs.dtype
    move_chance = time_values.to(dtype).view(-1, 1, 1)
    min_alpha = 1e-6 if dtype == torch.float32 else 1e-12
    alpha_time = (1.0 - move_chance).clamp(min=min_alpha)
    alpha_time_prime = -1.0
    clean_one_hot = F.one_hot(clean_tokens, vocab_size).to(dtype)
    clean_bar = vocab_size * alpha_time * clean_one_hot + move_chance
    pred_bar = vocab_size * alpha_time * log_clean_probs.exp() + move_chance
    coefficient = alpha_time_prime / (vocab_size * alpha_time)
    clean_bar_at_noisy = torch.gather(clean_bar, -1, noisy_tokens[..., None])
    pred_bar_at_noisy = torch.gather(pred_bar, -1, noisy_tokens[..., None])
    first_term = (vocab_size / clean_bar_at_noisy) - (vocab_size / pred_bar_at_noisy)
    log_ratio = pred_bar_at_noisy.log() - pred_bar.log() + clean_bar.log() - clean_bar_at_noisy.log()
    second_term = ((clean_bar / clean_bar_at_noisy) * log_ratio).sum(dim=-1, keepdim=True)
    return (coefficient * (first_term - second_term)).squeeze(-1)


def port_equality_loss(log_clean_probs, noisy_tokens, clean_tokens, time_values, exponent, vocab_size):
    """Callback matching uniform_preflight84.port_equality_check for log-linear noise (exponent == 1.0)."""
    if float(exponent) != 1.0:
        raise ValueError(f"UDLM uses only the log-linear schedule (exponent=1.0), got exponent={exponent!r}")
    return uniform_field_loss(log_clean_probs, noisy_tokens, clean_tokens, time_values, vocab_size)


def reverse_posterior(clean_probs, noisy_tokens, move_chance_s, move_chance_t, vocab_size):
    """Exact reverse uniform posterior p_theta(z_s | z_t) (Schiff et al. Eq. 15, diffusion.py:356-381).

    Computed in Bayes-product form proportional to q(z_t | z_s) * p_theta(z_s) and normalized,
    eliminating float32 cancellation in (alpha_t / alpha_s - alpha_t) on sharp predictors while
    matching Eq. 15 identically (including the exact t = 1.0 boundary where alpha_t = 0).
    """
    alpha_s = 1.0 - move_chance_s
    alpha_t = 1.0 - move_chance_t
    delta_alpha = move_chance_t - move_chance_s
    stay_prob = alpha_t / alpha_s
    jump_prob = delta_alpha / (alpha_s * vocab_size)
    noisy_one_hot = F.one_hot(noisy_tokens, vocab_size).to(clean_probs.dtype)
    forward_transition = stay_prob * noisy_one_hot + jump_prob
    pred_marginal_s = alpha_s * clean_probs + move_chance_s / vocab_size
    unnormalized = forward_transition * pred_marginal_s
    return unnormalized / unnormalized.sum(dim=-1, keepdim=True)


def cfg_posterior(log_clean_cond, log_clean_uncond, noisy_tokens, move_chance_s, move_chance_t, guidance, vocab_size):
    """Posterior-mixing classifier-free guidance (Schiff et al. Eqs. 6-8, diffusion.py:1314-1334)."""
    if guidance == 0.0:
        return reverse_posterior(log_clean_uncond.exp(), noisy_tokens, move_chance_s, move_chance_t, vocab_size)
    if guidance == 1.0:
        return reverse_posterior(log_clean_cond.exp(), noisy_tokens, move_chance_s, move_chance_t, vocab_size)
    log_post_uncond = reverse_posterior(
        log_clean_uncond.exp(), noisy_tokens, move_chance_s, move_chance_t, vocab_size
    ).clamp(min=1e-30).log()
    log_post_cond = reverse_posterior(
        log_clean_cond.exp(), noisy_tokens, move_chance_s, move_chance_t, vocab_size
    ).clamp(min=1e-30).log()
    return F.softmax(guidance * log_post_cond + (1.0 - guidance) * log_post_uncond, dim=-1)


def _phase_for_time(time_value, pass_index, total_passes):
    """Map reverse-step time t in (0, 1] to high_noise (> 2/3), mid_noise (> 1/3), or low_noise (<= 1/3)."""
    t_val = (1.0 - pass_index / max(total_passes, 1)) if time_value is None else float(time_value)
    if t_val > 2.0 / 3.0:
        return "high_noise"
    if t_val > 1.0 / 3.0:
        return "mid_noise"
    return "low_noise"


def _has_duplicate_species(tokens):
    """Boolean [B]: whether any two of the 6 species columns hold the same species token."""
    species = tokens[:, FIELD_COLS["species"]].sort(dim=1).values
    return (species[:, 1:] == species[:, :-1]).any(dim=1)


class RevisionTracker:
    """Logs per-pass top predictions and computes the revision numbers across all 84 fields.

    Pre-registered measures:
      1. value_changes_beyond_first: R_c = max(0, sum_{i=1}^{K} 1[z^{(i)}_c != z^{(i-1)}_c] - 1).
      2. noise_floor: deterministic App. A Eq. 25 fixed-predictor floor (gamma <= 1).
      3. changes_of_mind: M_c = sum_{i=1}^{K-1} 1[top^{(i)}_c != top^{(i-1)}_c] (cond for gamma > 0, uncond for gamma = 0).

    Exploratory diagnostics logged at zero extra network cost:
      - guided_changes_of_mind: flips of argmax(gamma * log x_c + (1 - gamma) * log x_u) under CFG.
      - confident_changes_of_mind: flips where the new top-1 probability margin p_(1) - p_(2) >= 0.10.
      - changes_of_mind_by_phase: changes_of_mind split across high_noise / mid_noise / low_noise passes.
      - mean_fields_changed_per_pass: average number of the 84 fields changing value per pass.
      - duplicate_species_pass_share & final_duplicate_species_share: simultaneous-update species collisions.
    """

    def __init__(self, batch_size, passes, vocab_sizes, *, eps=SAMPLING_EPS, keep_top_predictions=True):
        self.batch_size = int(batch_size)
        self.passes = int(passes)
        self.vocab_sizes = dict(vocab_sizes)
        self.eps = float(eps)
        self.total_value_changes = torch.zeros((batch_size, COLS), dtype=torch.long)
        self.changes_of_mind = torch.zeros((batch_size, COLS), dtype=torch.long)
        self.guided_changes_of_mind = torch.zeros((batch_size, COLS), dtype=torch.long)
        self.confident_changes_of_mind = torch.zeros((batch_size, COLS), dtype=torch.long)
        self.changes_of_mind_by_phase = {
            phase: torch.zeros((batch_size, COLS), dtype=torch.long) for phase in PHASE_NAMES
        }
        self.duplicate_species_passes = torch.zeros(batch_size, dtype=torch.long)
        self.final_duplicate_species = torch.zeros(batch_size, dtype=torch.bool)
        self._prev_top = None
        self._prev_guided_top = None
        self.top_predictions = (
            torch.empty((passes, batch_size, COLS), dtype=torch.int16) if keep_top_predictions else None
        )

    def record_pass(
        self,
        pass_index,
        prev_tokens,
        next_tokens,
        top_tokens,
        guided_top_tokens=None,
        *,
        top_margin=None,
        time_value=None,
    ):
        """Record one reverse step's state transition, species-collision state, and top predictions [B, 84]."""
        self.total_value_changes += (next_tokens != prev_tokens).to(dtype=torch.long, device="cpu")
        dup_species = _has_duplicate_species(next_tokens).cpu()
        self.duplicate_species_passes += dup_species.long()
        self.final_duplicate_species = dup_species
        top_cpu = top_tokens.to(dtype=torch.long, device="cpu")
        guided_cpu = top_cpu if guided_top_tokens is None else guided_top_tokens.to(dtype=torch.long, device="cpu")
        if self.top_predictions is not None:
            self.top_predictions[pass_index] = top_cpu.to(torch.int16)
        if self._prev_top is not None:
            flipped = top_cpu != self._prev_top
            self.changes_of_mind += flipped.long()
            self.guided_changes_of_mind += (guided_cpu != self._prev_guided_top).long()
            phase = _phase_for_time(time_value, pass_index, self.passes)
            self.changes_of_mind_by_phase[phase] += flipped.long()
            if top_margin is not None:
                confident = top_margin.to(device="cpu") >= CONFIDENT_MARGIN
                self.confident_changes_of_mind += (flipped & confident).long()
        self._prev_top = top_cpu
        self._prev_guided_top = guided_cpu

    def value_changes_beyond_first(self):
        """Per-team, per-column value changes beyond the first departure [B, 84]."""
        return (self.total_value_changes - 1).clamp(min=0)

    def per_team_records(self):
        """One compact dict per team with per-field-type means, for ask_grid84.draw aggregation."""
        beyond_first = self.value_changes_beyond_first().float()
        mind_changes = self.changes_of_mind.float()
        guided_mind = self.guided_changes_of_mind.float()
        confident_mind = self.confident_changes_of_mind.float()
        phase_tensors = {phase: tensor.float() for phase, tensor in self.changes_of_mind_by_phase.items()}
        per_pass = self.total_value_changes.float().sum(dim=1) / max(self.passes, 1)
        dup_pass_share = self.duplicate_species_passes.float() / max(self.passes, 1)
        final_dup = self.final_duplicate_species.float()
        records = []
        for row_idx in range(self.batch_size):
            records.append(
                dict(
                    passes=self.passes,
                    eps=self.eps,
                    mean_fields_changed_per_pass=float(per_pass[row_idx]),
                    duplicate_species_pass_share=float(dup_pass_share[row_idx]),
                    final_duplicate_species=float(final_dup[row_idx]),
                    value_changes_beyond_first={
                        key: float(beyond_first[row_idx, cols].mean()) for key, cols in FIELD_COLS.items()
                    },
                    changes_of_mind={
                        key: float(mind_changes[row_idx, cols].mean()) for key, cols in FIELD_COLS.items()
                    },
                    guided_changes_of_mind={
                        key: float(guided_mind[row_idx, cols].mean()) for key, cols in FIELD_COLS.items()
                    },
                    confident_changes_of_mind={
                        key: float(confident_mind[row_idx, cols].mean()) for key, cols in FIELD_COLS.items()
                    },
                    changes_of_mind_by_phase={
                        phase: {key: float(phase_tensors[phase][row_idx, cols].mean()) for key, cols in FIELD_COLS.items()}
                        for phase in PHASE_NAMES
                    },
                )
            )
        return records

    def summary(self):
        """Mean revision metrics by field type alongside the deterministic fixed-predictor noise floor."""
        return summarize_revision_records(self.per_team_records(), self.vocab_sizes, self.passes, eps=self.eps)


def _mean_field_value(records, section_key, field_key, fallback_key=None):
    if not records:
        return float("nan")
    values = []
    for rec in records:
        section = rec.get(section_key) if section_key in rec else (rec.get(fallback_key) if fallback_key else None)
        values.append(float("nan") if section is None else section[field_key])
    return float(np.mean(values))


def _mean_phase_values(records, field_key):
    if not records:
        return {phase: float("nan") for phase in PHASE_NAMES}
    return {
        phase: float(np.mean([rec["changes_of_mind_by_phase"][phase][field_key] for rec in records]))
        if all("changes_of_mind_by_phase" in rec for rec in records)
        else float("nan")
        for phase in PHASE_NAMES
    }


def summarize_revision_records(records, vocab_sizes, passes, *, eps=SAMPLING_EPS):
    """Aggregate per-team revision records into per-field-type means plus the deterministic noise floor."""
    by_field = {}
    for field_key in FIELD_KEYS:
        floor = UP.revision_noise_floor(vocab_sizes[field_key], passes, exponent=1.0, eps=eps)
        by_field[field_key] = dict(
            vocab_size=int(vocab_sizes[field_key]),
            value_changes_beyond_first=_mean_field_value(records, "value_changes_beyond_first", field_key),
            noise_floor=float(floor),
            changes_of_mind=_mean_field_value(records, "changes_of_mind", field_key),
            guided_changes_of_mind=_mean_field_value(records, "guided_changes_of_mind", field_key, "changes_of_mind"),
            confident_changes_of_mind=_mean_field_value(records, "confident_changes_of_mind", field_key),
            changes_of_mind_by_phase=_mean_phase_values(records, field_key),
        )
    mean_changed = float(np.mean([rec["mean_fields_changed_per_pass"] for rec in records])) if records else float("nan")
    dup_pass = (
        float(np.mean([rec["duplicate_species_pass_share"] for rec in records]))
        if records and all("duplicate_species_pass_share" in rec for rec in records)
        else float("nan")
    )
    final_dup = (
        float(np.mean([rec["final_duplicate_species"] for rec in records]))
        if records and all("final_duplicate_species" in rec for rec in records)
        else float("nan")
    )
    return dict(
        teams=len(records),
        passes=int(passes),
        eps=float(eps),
        mean_fields_changed_per_pass=mean_changed,
        duplicate_species_pass_share=dup_pass,
        final_duplicate_species_share=final_dup,
        by_field=by_field,
    )
