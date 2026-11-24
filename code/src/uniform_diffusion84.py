"""Uniform discrete diffusion model (UDLM) for 84-field VGC teams (issue #275, map #270).

Faithful port of Schiff et al. (ICLR 2025, arXiv:2412.10193v3, upstream pinned at
kuleshov-group/discrete-diffusion-guidance@edb0f8c28b7caeb4ea7a06a2fee8d74ab6da1661):
  - Forward uniform corruption (Eq. 1, diffusion.py:473-476): q(z_t | x_0) = Cat(alpha_t x_0 + (1 - alpha_t) u)
    with log-linear schedule alpha(t) = 1 - t, 1 - alpha(t) = t shared across all 84 fields.
  - Continuous-time uniform loss (Eq. 19, diffusion.py:563-614): T = 0, zero_recon_loss = True,
    time_conditioning = True, no legal Pokémon construction filter in the loss.
  - Reverse posterior (Eq. 15, diffusion.py:356-381) and posterior-mixing classifier-free guidance
    (Eqs. 6-8, diffusion.py:1314-1334).
  - Unconstrained reverse sampling on linspace(1, 1e-5, K + 1) with no in-loop legality filter,
    no prediction cache, and no clean-up pass. Every pass logs top clean-field predictions for the
    three revision measures (value changes beyond the first, fixed-predictor noise floor, and changes of mind).

Real vocabulary sizes exclude [MASK] (index 0 in the 84-field table encoding):
  species: 347, ability: 200, item: 149, move: 497, Stat Alignment (nature): 26, Stat Points (stat): 33.
Table-encoded rows (1..N_k) map to 0-based uniform tokens (0..N_k - 1) via table_to_uniform and back
via uniform_to_table so Showdown pastes, validator checks, and ask_grid84 work unchanged.
"""
import copy
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import alignment as AL
import asked_vs_got as X
import diffusion84 as M
import lossdown as L
import stat_points as SP
import team84 as T
import uniform_preflight84 as UP
from checkpoints import load_checkpoint
from encode import NSLOT, Vocab
from epoch_legality import EpochLegality, team_keys
from hps_generate import slot_to_text
from label_noise import noisy_labels

DEV = L.DEV
COLS = T.COLS                                            # 84
STAT0 = T.COLS48                                         # 48
NAN = float("nan")
FIELD_KEYS = ("species", "ability", "item", "move", "nature", "stat")
FIELD_COLS = {**L.COLS_BY_KEY, "stat": list(range(STAT0, COLS))}
TRAINING_EPS = UP.TRAINING_EPS                           # 1e-3 (Schiff et al. training.sampling_eps)
SAMPLING_EPS = UP.SAMPLING_EPS                           # 1e-5 (Schiff et al. sampling endpoint)
DEFAULT_PASSES = 128                                     # Schiff et al. configs/config.yaml default
CHECKPOINT_KIND = "uniform_discrete_diffusion_84"


def real_vocab_sizes(vocab):
    """Real category counts N_k excluding [MASK] for each of the six field types."""
    sizes = {key: vocab.sizes[key] - 1 for key in ("species", "ability", "item", "move", "nature")}
    sizes["stat"] = SP.NVALUES
    return sizes


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


class UniformNet84(nn.Module):
    """Transformer denoiser over 84 fields with real-token embeddings and heads (no [MASK] token).

    Matches diffusion84.Net84's architecture (d=256, nlayer=6, nhead=8, dropout=0.1, learned per-field
    positions, sinusoidal win-rate conditioning, and continuous time conditioning).
    """

    def __init__(self, vocab, d=256, nhead=8, nlayer=6, dropout=0.1, nfreq=8):
        super().__init__()
        self.vocab_sizes = real_vocab_sizes(vocab)
        self.emb = nn.ModuleDict({key: nn.Embedding(size, d) for key, size in self.vocab_sizes.items()})
        self.head = nn.ModuleDict({key: nn.Linear(d, size) for key, size in self.vocab_sizes.items()})
        self.pos = nn.Parameter(torch.randn(COLS, d) * 0.02)
        self.tproj = nn.Sequential(nn.Linear(1, d), nn.SiLU(), nn.Linear(d, d))
        self.register_buffer("freqs", (2.0 ** torch.arange(nfreq)) * math.pi)
        self.wproj = nn.Sequential(nn.Linear(2 * nfreq + 1, d), nn.SiLU(), nn.Linear(d, d))
        self.wnull = nn.Parameter(torch.zeros(d))
        encoder_layer = nn.TransformerEncoderLayer(
            d, nhead, 4 * d, dropout, batch_first=True, norm_first=True, activation="gelu"
        )
        self.tr = nn.TransformerEncoder(encoder_layer, nlayer, enable_nested_tensor=False)
        self.ln = nn.LayerNorm(d)

    def win_rate_vec(self, win_rates):
        """Embed scalar win-rate conditions; NaN maps to the learned null embedding."""
        is_null = torch.isnan(win_rates)
        scaled = torch.nan_to_num(win_rates, 0.0).to(self.pos.dtype).view(-1, 1)
        features = torch.cat([scaled, torch.sin(scaled * self.freqs), torch.cos(scaled * self.freqs)], dim=1)
        embedded = self.wproj(features)
        return torch.where(is_null.view(-1, 1), self.wnull.expand_as(embedded), embedded)

    def forward(self, uniform_tokens, time_values, win_rates):
        """Forward pass on 0-based real tokens [B, 84], noise times [B], and win rates [B]."""
        batch_size, hidden_dim = uniform_tokens.shape[0], self.pos.shape[1]
        hidden = torch.empty(batch_size, COLS, hidden_dim, device=uniform_tokens.device, dtype=self.pos.dtype)
        for field_key, cols in FIELD_COLS.items():
            hidden[:, cols] = self.emb[field_key](uniform_tokens[:, cols])
        time_emb = self.tproj(time_values.view(-1, 1).to(self.pos.dtype)).unsqueeze(1)
        cond_emb = self.win_rate_vec(win_rates).unsqueeze(1)
        hidden = hidden + self.pos.unsqueeze(0) + cond_emb + time_emb
        return self.ln(self.tr(hidden))

    def logits(self, hidden):
        """Return {field_key: [B, L_k, N_k]} clean-token logits over real vocabularies."""
        return {field_key: self.head[field_key](hidden[:, cols]) for field_key, cols in FIELD_COLS.items()}


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

    Uses UDLM's baseline log-linear schedule alpha(t) = 1 - t, alpha'(t) = -1, zero_recon_loss = True.
    Computes move_chance = t directly (never 1 - (1 - t)) for float32/MPS numerical hygiene.

    Returns:
        [B, L] per-position continuous-time diffusion loss in nats.
    """
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


def per_column_ce(logits_by_key, clean_tokens):
    """Cross-entropy [B, 84] of 0-based clean tokens under each field type's logits."""
    ce = torch.zeros(clean_tokens.shape, device=clean_tokens.device, dtype=torch.float32)
    for field_key, cols in FIELD_COLS.items():
        ce[:, cols] = F.cross_entropy(
            logits_by_key[field_key].float().transpose(1, 2), clean_tokens[:, cols], reduction="none"
        )
    return ce


def elbo_terms(net, clean_tokens, noisy_tokens, time_values, labels, vocab_sizes):
    """Per-team continuous-time uniform ELBO [B] in nats per field (averaged across all 84 columns).

    No legal Pokémon construction filter is applied to logits (Uniform, no filter).
    """
    logits_by_key = net.logits(net(noisy_tokens, time_values, labels))
    first_dtype = next(iter(logits_by_key.values())).dtype
    calc_dtype = torch.float64 if first_dtype == torch.float64 else torch.float32
    total_loss = torch.zeros(clean_tokens.shape[0], device=clean_tokens.device, dtype=calc_dtype)
    for field_key, cols in FIELD_COLS.items():
        log_probs = F.log_softmax(logits_by_key[field_key].to(calc_dtype), dim=-1)
        field_loss = uniform_field_loss(
            log_probs, noisy_tokens[:, cols], clean_tokens[:, cols], time_values, vocab_sizes[field_key]
        )
        total_loss = total_loss + field_loss.sum(dim=1)
    return total_loss / COLS


def batch_loss(net, clean_tokens, labels, vocab_sizes, p_uncond=0.15):
    """Mean continuous-time uniform ELBO over one training batch with slot shuffling, stratified times, and ask dropout."""
    batch_size = clean_tokens.shape[0]
    device = clean_tokens.device
    shuffled = M.shuffle_slots(clean_tokens)
    offsets = (torch.rand(1, device=device) + torch.arange(batch_size, device=device) / batch_size) % 1.0
    time_values = offsets[torch.randperm(batch_size, device=device)].clamp(min=TRAINING_EPS, max=1.0 - TRAINING_EPS)
    noisy_tokens = corrupt_uniform(shuffled, time_values, vocab_sizes)
    dropped_labels = labels.clone()
    dropped_labels[torch.rand(batch_size, device=device) < p_uncond] = NAN
    return elbo_terms(net, shuffled, noisy_tokens, time_values, dropped_labels, vocab_sizes).mean()


@torch.no_grad()
def val_loss(net, rows_table, labels, vocab_sizes, reps=4, seed=12345, batch=512):
    """Mean continuous-time uniform ELBO on fixed validation rows with common random numbers (4 stratified draws)."""
    generator = torch.Generator().manual_seed(seed)
    clean_all = table_to_uniform(torch.as_tensor(np.asarray(rows_table, np.int64)))
    labels_all = torch.as_tensor(np.asarray(labels, np.float32))
    num_rows, total = len(clean_all), 0.0
    for rep in range(reps):
        slot_perm = torch.rand(num_rows, NSLOT, generator=generator).argsort(1)
        time_all = ((rep + torch.rand(num_rows, generator=generator)) / reps).clamp(
            min=TRAINING_EPS, max=1.0 - TRAINING_EPS
        )
        shuffled_all = M.shuffle_slots(clean_all, slot_perm)
        noisy_all = corrupt_uniform(shuffled_all, time_all, vocab_sizes, generator=generator)
        for start in range(0, num_rows, batch):
            sl = slice(start, start + batch)
            batch_terms = elbo_terms(
                net,
                shuffled_all[sl].to(DEV),
                noisy_all[sl].to(DEV),
                time_all[sl].to(DEV),
                labels_all[sl].to(DEV),
                vocab_sizes,
            )
            total += float(batch_terms.sum())
    return total / (num_rows * reps)


def reverse_posterior(clean_probs, noisy_tokens, move_chance_s, move_chance_t, vocab_size):
    """Exact reverse uniform posterior p_theta(z_s | z_t) (Schiff et al. Eq. 15, diffusion.py:356-381).

    Uses delta_alpha = move_chance_t - move_chance_s directly to avoid float32 cancellation.
    """
    alpha_s = 1.0 - move_chance_s
    alpha_t = 1.0 - move_chance_t
    alpha_ratio = alpha_t / alpha_s
    delta_alpha = move_chance_t - move_chance_s
    noisy_one_hot = F.one_hot(noisy_tokens, vocab_size).to(clean_probs.dtype)
    numerator = (
        alpha_t * vocab_size * clean_probs * noisy_one_hot
        + (alpha_ratio - alpha_t) * noisy_one_hot
        + delta_alpha * clean_probs
        + (1.0 - alpha_ratio) * move_chance_s / vocab_size
    )
    denominator = alpha_t * vocab_size * torch.gather(clean_probs, -1, noisy_tokens[..., None]) + move_chance_t
    return numerator / denominator


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


class RevisionTracker:
    """Logs per-pass top predictions and computes the three pre-registered revision numbers across all 84 fields.

    1. Value changes beyond the first: R_c = max(0, sum_{i=1}^{K} 1[z^{(i)}_c != z^{(i-1)}_c] - 1).
    2. Fixed-predictor noise floor: deterministic E[max(0, C_c - 1)] from uniform_preflight84.revision_noise_floor.
    3. Changes of mind (headline): M_c = sum_{i=1}^{K-1} 1[top^{(i)}_c != top^{(i-1)}_c].
    """

    def __init__(self, batch_size, passes, vocab_sizes, *, eps=SAMPLING_EPS, keep_top_predictions=True):
        self.batch_size = int(batch_size)
        self.passes = int(passes)
        self.vocab_sizes = dict(vocab_sizes)
        self.eps = float(eps)
        self.total_value_changes = torch.zeros((batch_size, COLS), dtype=torch.long)
        self.changes_of_mind = torch.zeros((batch_size, COLS), dtype=torch.long)
        self._prev_top = None
        self.top_predictions = (
            torch.empty((passes, batch_size, COLS), dtype=torch.int16) if keep_top_predictions else None
        )

    def record_pass(self, pass_index, prev_tokens, next_tokens, top_tokens):
        """Record one reverse step's state transition and top clean-field predictions [B, 84]."""
        changed = (next_tokens != prev_tokens).to(dtype=torch.long, device="cpu")
        self.total_value_changes += changed
        top_cpu = top_tokens.to(dtype=torch.long, device="cpu")
        if self.top_predictions is not None:
            self.top_predictions[pass_index] = top_cpu.to(torch.int16)
        if self._prev_top is not None:
            self.changes_of_mind += (top_cpu != self._prev_top).long()
        self._prev_top = top_cpu

    def value_changes_beyond_first(self):
        """Per-team, per-column value changes beyond the first departure [B, 84]."""
        return (self.total_value_changes - 1).clamp(min=0)

    def per_team_records(self):
        """One compact dict per team with per-field-type means, for ask_grid84.draw aggregation."""
        beyond_first = self.value_changes_beyond_first().float()
        mind_changes = self.changes_of_mind.float()
        records = []
        for row_idx in range(self.batch_size):
            records.append(
                dict(
                    passes=self.passes,
                    eps=self.eps,
                    value_changes_beyond_first={
                        key: float(beyond_first[row_idx, cols].mean()) for key, cols in FIELD_COLS.items()
                    },
                    changes_of_mind={
                        key: float(mind_changes[row_idx, cols].mean()) for key, cols in FIELD_COLS.items()
                    },
                )
            )
        return records

    def summary(self):
        """Mean revision metrics by field type alongside the deterministic fixed-predictor noise floor."""
        return summarize_revision_records(self.per_team_records(), self.vocab_sizes, self.passes, eps=self.eps)


def summarize_revision_records(records, vocab_sizes, passes, *, eps=SAMPLING_EPS):
    """Aggregate per-team revision records into per-field-type means plus the deterministic noise floor."""
    by_field = {}
    for field_key in FIELD_KEYS:
        floor = UP.revision_noise_floor(vocab_sizes[field_key], passes, exponent=1.0, eps=eps)
        if records:
            mean_beyond = float(np.mean([rec["value_changes_beyond_first"][field_key] for rec in records]))
            mean_mind = float(np.mean([rec["changes_of_mind"][field_key] for rec in records]))
        else:
            mean_beyond = float("nan")
            mean_mind = float("nan")
        by_field[field_key] = dict(
            vocab_size=int(vocab_sizes[field_key]),
            value_changes_beyond_first=mean_beyond,
            noise_floor=float(floor),
            changes_of_mind=mean_mind,
        )
    return dict(teams=len(records), passes=int(passes), eps=float(eps), by_field=by_field)


def _uniform_prior(batch_size, vocab_sizes, device):
    """Independent uniform random 0-based real tokens [B, 84] across all six field types."""
    tokens = torch.empty((batch_size, COLS), dtype=torch.long, device=device)
    for field_key, cols in FIELD_COLS.items():
        tokens[:, cols] = torch.randint(0, vocab_sizes[field_key], (batch_size, len(cols)), device=device)
    return tokens


def _predict_log_clean(model, tokens, time_tensor, cond_tensor, temperature):
    """Run model once and return {field_key: log_softmax(logits / temperature)} without any legality mask."""
    logits_by_key = model._logits_eval(tokens, time_tensor, cond_tensor)
    if temperature == 1.0:
        return {key: F.log_softmax(val.float(), dim=-1) for key, val in logits_by_key.items()}
    return {key: F.log_softmax(val.float() / temperature, dim=-1) for key, val in logits_by_key.items()}


def _top_predictions_row(log_clean_by_key, batch_size, device):
    """Extract [B, 84] argmax clean-field token indices from per-field-type log-probabilities."""
    top_tokens = torch.empty((batch_size, COLS), dtype=torch.long, device=device)
    for field_key, cols in FIELD_COLS.items():
        top_tokens[:, cols] = log_clean_by_key[field_key].argmax(dim=-1)
    return top_tokens


def _step_sample_tokens(log_cond, log_uncond, tokens, move_s, move_t, guidance, vocab_sizes):
    """Compute guided/unguided reverse posterior for every field type and draw next tokens [B, 84]."""
    batch_size = tokens.shape[0]
    next_tokens = torch.empty_like(tokens)
    for field_key, cols in FIELD_COLS.items():
        vocab_size = vocab_sizes[field_key]
        if log_cond is None:
            probs = reverse_posterior(log_uncond[field_key].exp(), tokens[:, cols], move_s, move_t, vocab_size)
        elif log_uncond is None:
            probs = reverse_posterior(log_cond[field_key].exp(), tokens[:, cols], move_s, move_t, vocab_size)
        else:
            probs = cfg_posterior(
                log_cond[field_key], log_uncond[field_key], tokens[:, cols], move_s, move_t, guidance, vocab_size
            )
        drawn = torch.multinomial(probs.clamp(min=0.0).reshape(-1, vocab_size), 1).reshape(batch_size, len(cols))
        next_tokens[:, cols] = drawn
    return next_tokens


@torch.no_grad()
def sample_uniform(
    model,
    n,
    *,
    ask=None,
    guidance=1.0,
    passes=DEFAULT_PASSES,
    temperature=1.0,
    eps=SAMPLING_EPS,
    seed=None,
    revisions=None,
    keep_top_predictions=True,
):
    """Unconstrained UDLM reverse sampling on linspace(1, eps, passes + 1) with posterior-mixing CFG.

    No in-loop legal Pokémon construction filter, no prediction cache, and no clean-up pass.
    Returns [n, 84] table-encoded tokens (1..N_k, zero [MASK]).
    """
    if passes < 1:
        raise ValueError(f"passes must be >= 1, got {passes!r}")
    if seed is not None:
        torch.manual_seed(seed)
    device = next(model.nets[0][0].parameters()).device
    vocab_sizes = model.vocab_sizes
    effective_guidance = 0.0 if ask is None else float(guidance)
    tokens = _uniform_prior(n, vocab_sizes, device)
    ask_tensor = torch.full((n,), NAN if ask is None else float(ask), device=device)
    null_tensor = torch.full((n,), NAN, device=device)
    times = torch.linspace(1.0, eps, passes + 1, dtype=torch.float32, device=device)
    tracker = (
        RevisionTracker(n, passes, vocab_sizes, eps=eps, keep_top_predictions=keep_top_predictions)
        if revisions is None
        else revisions
    )

    for step in range(passes):
        move_t = times[step]
        move_s = times[step + 1]
        time_batch = torch.full((n,), float(move_t), device=device)
        need_cond = effective_guidance != 0.0
        need_uncond = effective_guidance != 1.0
        log_cond = _predict_log_clean(model, tokens, time_batch, ask_tensor, temperature) if need_cond else None
        log_uncond = _predict_log_clean(model, tokens, time_batch, null_tensor, temperature) if need_uncond else None
        top_source = log_cond if effective_guidance > 0.0 else log_uncond
        top_tokens = _top_predictions_row(top_source, n, device)
        next_tokens = _step_sample_tokens(log_cond, log_uncond, tokens, move_s, move_t, effective_guidance, vocab_sizes)
        tracker.record_pass(step, tokens, next_tokens, top_tokens)
        tokens = next_tokens

    return uniform_to_table(tokens), tracker


def _print_entry(entry):
    print(
        "  "
        + " · ".join(
            f"{key} {val:.4f}" if isinstance(val, float) else f"{key} {val}"
            for key, val in entry.items()
            if key != "rejection_reasons"
        ),
        flush=True,
    )


def _train_epoch(net, avg, optimizer, clean_tokens, labels, vocab_sizes, *, batch, lr_at, step, ema, label_noise, noise_rng):
    """Run one training epoch of continuous-time uniform diffusion and update EMA weights."""
    net.train()
    loss_sum, grad_norm_sum, num_batches = 0.0, 0.0, 0
    for indices in torch.randperm(len(clean_tokens), device=DEV).split(batch):
        noisy_w = noisy_labels(labels[indices], label_noise, noise_rng)
        loss = batch_loss(net, clean_tokens[indices], noisy_w, vocab_sizes)
        current_lr = lr_at(step)
        for group in optimizer.param_groups:
            group["lr"] = current_lr
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm_sum += float(nn.utils.clip_grad_norm_(net.parameters(), 1.0))
        optimizer.step()
        step += 1
        if avg is not None:
            with torch.no_grad():
                decay = min(ema, (1 + step) / (10 + step))
                for avg_param, net_param in zip(avg.parameters(), net.parameters(), strict=True):
                    avg_param.lerp_(net_param, 1.0 - decay)
        loss_sum += float(loss.detach())
        num_batches += 1
    return loss_sum / num_batches, grad_norm_sum / num_batches, step


class UniformModel84:
    """Trained 84-field uniform discrete diffusion model (UDLM) with Model84-compatible interface."""

    def __init__(self, vocab, corpus, nets, info=None, *, default_passes=DEFAULT_PASSES):
        self.V = vocab
        self.vocab_sizes = real_vocab_sizes(vocab)
        self.nets = [(net.eval(), kw) for net, kw in nets]
        self.info = info or {}
        self.default_passes = int(default_passes)
        self.tables = L.Tables(vocab, corpus)
        _, _, _, look, _ = X.setup()
        look.update(vocab.regulation["names"])
        self.look = look
        _, _, _, self.modes, self.natures = AL.build_tables(corpus)
        self.last_revisions = None

    @property
    def net(self):
        assert len(self.nets) == 1, "expected a single-member uniform diffusion model"
        return self.nets[0][0]

    def _logits_eval(self, uniform_tokens, time_values, win_rates):
        """Evaluate the single network (or log-mean ensemble if multiple) and return clean logits."""
        if len(self.nets) == 1:
            network = self.nets[0][0]
            return network.logits(network(uniform_tokens, time_values, win_rates))
        log_probs_list = [
            {k: F.log_softmax(v.float(), dim=-1) for k, v in net.logits(net(uniform_tokens, time_values, win_rates)).items()}
            for net, _ in self.nets
        ]
        return {
            key: torch.logsumexp(torch.stack([lp[key] for lp in log_probs_list]), dim=0) - math.log(len(log_probs_list))
            for key in FIELD_KEYS
        }

    @classmethod
    def train_new(
        cls,
        vocab,
        corpus,
        rows,
        labels=None,
        epochs=80,
        seed=0,
        batch=256,
        lr=4e-4,
        wd=0.01,
        warmup=500,
        ema=0.999,
        log=None,
        val=None,
        patience=3,
        early_stopping_threshold=0.0,
        on_best=None,
        label_noise=0.0,
        stop_after=None,
        on_epoch=None,
        legality_teams=0,
        legality_passes=DEFAULT_PASSES,
        legality_guidance=1.0,
        legality_copies=True,
        tracker=None,
        **net_kw,
    ):
        """Train a uniform discrete diffusion model from scratch on 84-field rows."""
        torch.manual_seed(seed)
        tables = L.Tables(vocab, corpus)
        x_table, w_tensor, _row_weights, keep = M._training_tensors(tables, rows, labels)
        clean_tokens = table_to_uniform(x_table)
        net = UniformNet84(vocab, **net_kw).to(DEV)
        vocab_sizes = net.vocab_sizes
        info = dict(
            kind=CHECKPOINT_KIND,
            noise_process="uniform",
            schedule="log_linear",
            vocab_real_uniform=vocab_sizes,
            n_train=len(clean_tokens),
            dropped_by_tables=int((~keep).sum()),
            cap=epochs,
            patience=patience,
            early_stopping_threshold=early_stopping_threshold,
            lr=lr,
            batch=batch,
            seed=seed,
            net_kw=net_kw,
            label_noise=label_noise,
            stop_after=stop_after,
            legality_teams=legality_teams,
            legality_passes=legality_passes,
            legality_guidance=legality_guidance,
            table_loss=False,
            legality_tables=False,
        )
        known = team_keys(x_table.cpu().numpy()) if legality_teams and legality_copies else None
        legality = EpochLegality(legality_teams, guidance=legality_guidance, tables=False, known=known)
        return cls._fit(
            vocab,
            corpus,
            net,
            clean_tokens,
            w_tensor,
            vocab_sizes,
            info,
            epochs=epochs,
            seed=seed,
            batch=batch,
            lr=lr,
            wd=wd,
            warmup=warmup,
            ema=ema,
            log=log,
            val=val,
            patience=patience,
            threshold=early_stopping_threshold,
            on_best=on_best,
            label_noise=label_noise,
            stop_after=stop_after,
            on_epoch=on_epoch,
            net_kw=net_kw,
            legality=legality,
            legality_passes=legality_passes,
            tracker=tracker,
        )

    @classmethod
    def _fit(
        cls,
        vocab,
        corpus,
        net,
        clean_tokens,
        labels,
        vocab_sizes,
        info,
        *,
        epochs,
        seed,
        batch,
        lr,
        wd,
        warmup,
        ema,
        log,
        val,
        patience,
        threshold,
        on_best,
        label_noise,
        stop_after,
        on_epoch,
        net_kw,
        legality,
        legality_passes,
        tracker,
    ):
        make = lambda network, extra: cls(
            vocab, corpus, [(network, net_kw)], dict(info, **extra), default_passes=legality_passes
        )
        avg = copy.deepcopy(net).eval() if ema else None
        total_steps = math.ceil(len(clean_tokens) / batch) * epochs
        warm_steps = min(warmup, max(1, total_steps // 10))
        optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd, betas=(0.9, 0.98))
        lr_at = lambda s: lr * min(1.0, (s + 1) / warm_steps) * 0.5 * (1 + math.cos(math.pi * min(s, total_steps) / total_steps))
        noise_rng = torch.Generator().manual_seed(seed + 1)
        curves = [] if log is None else log
        state = dict(best=math.inf, best_epoch=0, best_sd=None, reset_epoch=0, step=0)
        t_start = time.perf_counter()
        checked = []
        own_tracker = tracker is not None and not getattr(tracker, "active", False)
        if tracker is not None:
            tracker.start(
                config=dict(
                    info,
                    wd=wd,
                    warmup=warmup,
                    ema=ema,
                    device=str(DEV),
                    n_params=sum(param.numel() for param in net.parameters()),
                    val_rows=0 if val is None else len(val[0]),
                )
            )
        try:
            for epoch_idx in range(epochs):
                train_loss, grad_norm, state["step"] = _train_epoch(
                    net,
                    avg,
                    optimizer,
                    clean_tokens,
                    labels,
                    vocab_sizes,
                    batch=batch,
                    lr_at=lr_at,
                    step=state["step"],
                    ema=ema,
                    label_noise=label_noise,
                    noise_rng=noise_rng,
                )
                eval_net = (avg if avg is not None else net).eval()
                entry = _evaluate_epoch(
                    epoch_idx + 1,
                    train_loss,
                    eval_net,
                    val,
                    vocab_sizes,
                    legality,
                    checked,
                    make,
                    t_start,
                )
                _update_best(state, entry, eval_net, threshold, on_best, make)
                if on_epoch is not None:
                    on_epoch(make(eval_net, dict(epoch=epoch_idx + 1)), epoch_idx + 1)
                curves.append(entry)
                if tracker is not None:
                    tracker.log_epoch(
                        entry,
                        lr=optimizer.param_groups[0]["lr"],
                        step=state["step"],
                        grad_norm=grad_norm,
                        best_val_loss=state["best"],
                    )
                _print_entry(entry)
                if _should_stop(epoch_idx + 1, state["reset_epoch"], patience, stop_after):
                    break
            final_net = avg if avg is not None else net
            if patience and state["best_sd"] is not None:
                final_net.load_state_dict(state["best_sd"])
            info.update(
                stopped_epoch=len(curves),
                best_epoch=state["best_epoch"] or None,
                best_val_loss=None if state["best"] == math.inf else state["best"],
                curves=curves,
            )
            if tracker is not None:
                tracker.log_summary(info)
            return make(final_net, {})
        finally:
            if legality is not None:
                legality.close()
            if own_tracker:
                tracker.finish()

    def save(self, path):
        """Save model weights, network settings, regulation snapshot, checkpoint kind, and run info."""
        torch.save(
            dict(
                kind=CHECKPOINT_KIND,
                members=[dict(sd=net.state_dict(), net_kw=kw) for net, kw in self.nets],
                regulation=self.V.regulation,
                info=self.info,
            ),
            path,
        )

    @classmethod
    def load(cls, path, corpus=None):
        checkpoint = load_checkpoint(path, map_location="cpu")
        if checkpoint.get("kind") != CHECKPOINT_KIND:
            raise ValueError(f"{path}: expected checkpoint kind {CHECKPOINT_KIND!r}, got {checkpoint.get('kind')!r}")
        vocab = Vocab.from_regulation(checkpoint["regulation"])
        corpus = X.setup()[0] if corpus is None else corpus
        nets = []
        for member in checkpoint["members"]:
            network = UniformNet84(vocab, **member["net_kw"])
            network.load_state_dict(member["sd"])
            nets.append((network.to(DEV), member["net_kw"]))
        return cls(vocab, corpus, nets, checkpoint.get("info"))

    @torch.no_grad()
    def sample(
        self,
        n,
        ask=None,
        guidance=1.0,
        passes=None,
        temperature=1.0,
        seed=None,
        eps=SAMPLING_EPS,
        tables=False,
        revisions=None,
        keep_top_predictions=True,
    ):
        """Draw n teams via unconstrained UDLM sampling; returns [n, 84] table-encoded tokens."""
        if tables:
            raise ValueError("UniformModel84 uses unconstrained sampling with no in-loop legal Pokémon construction filter")
        chosen_passes = self.default_passes if passes is None else int(passes)
        rows, tracker = sample_uniform(
            self,
            n,
            ask=ask,
            guidance=guidance,
            passes=chosen_passes,
            temperature=temperature,
            eps=eps,
            seed=seed,
            revisions=revisions,
            keep_top_predictions=keep_top_predictions,
        )
        self.last_revisions = tracker
        return rows

    def to_paste(self, row, repair=False):
        """(Showdown paste, slot dicts) from an 84-field table-encoded row; repair=True aligns Stat Alignment."""
        slots = []
        for slot_raw in T.decode(self.V, row):
            lookup = lambda value: self.look.get(value, value)
            slot = dict(
                species=lookup(slot_raw["species"]),
                ability=lookup(slot_raw["ability"]),
                item=lookup(slot_raw["item"]),
                nature=lookup(slot_raw["nature"]),
                moves=[lookup(move) for move in slot_raw["moves"]],
                evs=slot_raw["evs"],
            )
            if repair:
                slot["nature"] = AL.repair_nature(slot, self.modes, self.natures)
            slots.append(slot)
        return "\n\n".join(slot_to_text(slot) for slot in slots) + "\n", slots


def _evaluate_epoch(epoch_num, train_loss, eval_net, val, vocab_sizes, legality, checked, make, t_start):
    entry = dict(epoch=epoch_num, train_loss=train_loss)
    if legality is not None and legality.teams:
        if not checked:
            checked.append(make(eval_net, {}))
        entry.update(legality(checked[0]))
    if val is not None:
        entry["val_loss"] = val_loss(eval_net, val[0], val[1], vocab_sizes)
    entry["minutes"] = round((time.perf_counter() - t_start) / 60, 3)
    return entry


def _update_best(state, entry, eval_net, threshold, on_best, make):
    if "val_loss" not in entry or entry["val_loss"] >= state["best"]:
        return
    if state["best"] - entry["val_loss"] > threshold:
        state["reset_epoch"] = entry["epoch"]
    state["best"] = entry["val_loss"]
    state["best_epoch"] = entry["epoch"]
    state["best_sd"] = {key: param.detach().clone() for key, param in eval_net.state_dict().items()}
    if on_best is not None:
        on_best(make(eval_net, dict(best_epoch=state["best_epoch"])), state["best_epoch"])


def _should_stop(epoch_num, reset_epoch, patience, stop_after):
    if patience and epoch_num - reset_epoch >= patience:
        return True
    return bool(stop_after and epoch_num >= stop_after)


class UniformSampler84:
    """Configured unconstrained uniform discrete diffusion sampler for ask_grid84.draw and sweep scripts."""

    def __init__(self, model, passes=DEFAULT_PASSES, *, guidance=1.0, eps=SAMPLING_EPS, keep_top_predictions=True):
        self.w = self.model = model
        self.passes = int(passes)
        self.guidance = float(guidance)
        self.eps = float(eps)
        self.keep_top_predictions = bool(keep_top_predictions)
        self.last_revisions = None

    @torch.no_grad()
    def sample(self, n, ask=None, guidance=None, temperature=1.0, seed=None, passes=None):
        chosen_guidance = self.guidance if guidance is None else float(guidance)
        chosen_passes = self.passes if passes is None else int(passes)
        rows = self.model.sample(
            n,
            ask=ask,
            guidance=chosen_guidance,
            passes=chosen_passes,
            temperature=temperature,
            seed=seed,
            eps=self.eps,
            tables=False,
            keep_top_predictions=self.keep_top_predictions,
        )
        self.last_revisions = self.model.last_revisions
        return rows

    def to_paste(self, row, repair=False):
        return self.model.to_paste(row, repair=repair)
