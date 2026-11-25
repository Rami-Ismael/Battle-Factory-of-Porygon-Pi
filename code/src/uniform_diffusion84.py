"""Uniform discrete diffusion model (UDLM) for 84-field VGC teams (issue #275, map #270).

Faithful port of Schiff et al. (ICLR 2025, arXiv:2412.10193v3, upstream pinned at
kuleshov-group/discrete-diffusion-guidance@edb0f8c28b7caeb4ea7a06a2fee8d74ab6da1661):
  - Forward uniform corruption (Eq. 1, diffusion.py:473-476): q(z_t | x_0) = Cat(alpha_t x_0 + (1 - alpha_t) u)
    with self-consistent ideal log-linear schedule alpha(t) = 1 - t, move_chance(t) = t shared across all 84 fields.
  - Continuous-time uniform loss (Eq. 19, diffusion.py:563-614): T = 0, zero_recon_loss = True,
    time_conditioning = True (raw scalar t into tproj matching diffusion84.Net84), no legal Pokémon construction filter.
  - Reverse posterior (Eq. 15, diffusion.py:356-381, Bayes-product form) and posterior-mixing classifier-free guidance
    (Eqs. 6-8, diffusion.py:1314-1334).
  - Unconstrained reverse sampling on linspace(1, 1e-5, K + 1) with no in-loop legality filter,
    no prediction cache, and no clean-up pass. Every pass logs top clean-field predictions for the
    revision measures.

Pure math and revision tracking live in uniform_math84.py (CI-testable on CPU) and are re-exported here.
"""
import contextlib
import copy
import math
import os
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
from checkpoints import load_checkpoint
from encode import NSLOT, Vocab
from epoch_legality import EpochLegality, team_keys
from hps_generate import slot_to_text
from label_noise import noisy_labels
import lr_schedule84 as LS
from uniform_math84 import (
    COLS,
    DEFAULT_PASSES,
    FIELD_COLS,
    FIELD_KEYS,
    NAN,
    SAMPLING_EPS,
    STAT0,
    TRAINING_EPS,
    RevisionTracker,
    cfg_posterior,
    corrupt_uniform,
    is_better_val_loss,
    port_equality_loss,
    reverse_posterior,
    summarize_revision_records,
    table_to_uniform,
    uniform_field_loss,
    uniform_to_table,
    vocab_vector,
)

DEV = L.DEV
CHECKPOINT_KIND = "uniform_discrete_diffusion_84"
FINITE_CHECK_EVERY = 32


def _resolve_amp(amp, device):
    """Enable bfloat16 autocast on MPS/CUDA by default unless disabled via amp=False or VGC_AMP=0."""
    if torch.device(device).type not in ("mps", "cuda"):
        return False
    if amp is not None:
        return bool(amp)
    return os.environ.get("VGC_AMP", "1").strip().lower() not in ("0", "false", "no", "off")


def _amp_context(use_amp, device):
    dev_type = torch.device(device).type
    if use_amp and dev_type in ("mps", "cuda"):
        return torch.amp.autocast(dev_type, dtype=torch.bfloat16)
    return contextlib.nullcontext()


def _make_adamw(params, *, lr, wd, device):
    """Build AdamW with fused Metal/CUDA kernels when supported (exact same update, ~4x faster on MPS)."""
    if torch.device(device).type in ("mps", "cuda"):
        try:
            return torch.optim.AdamW(params, lr=lr, weight_decay=wd, betas=(0.9, 0.98), fused=True), True
        except (RuntimeError, TypeError, ValueError):
            pass
    return torch.optim.AdamW(params, lr=lr, weight_decay=wd, betas=(0.9, 0.98)), False

__all__ = [
    "CHECKPOINT_KIND",
    "COLS",
    "DEFAULT_PASSES",
    "DEV",
    "FIELD_COLS",
    "FIELD_KEYS",
    "NAN",
    "SAMPLING_EPS",
    "STAT0",
    "TRAINING_EPS",
    "RevisionTracker",
    "UniformModel84",
    "UniformNet84",
    "UniformSampler84",
    "batch_loss",
    "cfg_posterior",
    "corrupt_uniform",
    "elbo_terms",
    "is_better_val_loss",
    "per_column_ce",
    "port_equality_loss",
    "real_vocab_sizes",
    "reverse_posterior",
    "sample_uniform",
    "summarize_revision_records",
    "table_to_uniform",
    "uniform_field_loss",
    "uniform_to_table",
    "val_loss",
]


def real_vocab_sizes(vocab):
    """Real category counts N_k excluding [MASK] for each of the six field types."""
    sizes = {key: vocab.sizes[key] - 1 for key in ("species", "ability", "item", "move", "nature")}
    sizes["stat"] = SP.NVALUES
    return sizes


class UniformNet84(nn.Module):
    """Transformer denoiser over 84 fields with real-token embeddings and heads (no [MASK] token).

    Matches diffusion84.Net84's architecture (d=256, nlayer=6, nhead=8, dropout=0.1, learned per-field
    positions, sinusoidal win-rate conditioning, and scalar continuous time conditioning).
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


def per_column_ce(logits_by_key, clean_tokens):
    """Cross-entropy [B, 84] of 0-based clean tokens under each field type's logits."""
    ce = torch.zeros(clean_tokens.shape, device=clean_tokens.device, dtype=torch.float32)
    for field_key, cols in FIELD_COLS.items():
        ce[:, cols] = F.cross_entropy(
            logits_by_key[field_key].float().transpose(1, 2), clean_tokens[:, cols], reduction="none"
        )
    return ce


def elbo_terms(net, clean_tokens, noisy_tokens, time_values, labels, vocab_sizes, *, check_bounds=True, amp=False):
    """Per-team continuous-time uniform diffusion loss [B] in nats per field (averaged across all 84 columns).

    No legal Pokémon construction filter is applied to logits (Uniform, no filter).
    When amp=True on MPS/CUDA, runs the transformer backbone in bfloat16 while computing log_softmax
    and the Eq. 19 uniform diffusion loss in float32.
    """
    with _amp_context(amp, clean_tokens.device):
        logits_by_key = net.logits(net(noisy_tokens, time_values, labels))
    first_dtype = next(iter(logits_by_key.values())).dtype
    calc_dtype = torch.float64 if first_dtype == torch.float64 else torch.float32
    total_loss = torch.zeros(clean_tokens.shape[0], device=clean_tokens.device, dtype=calc_dtype)
    for field_key, cols in FIELD_COLS.items():
        log_probs = F.log_softmax(logits_by_key[field_key].to(calc_dtype), dim=-1)
        field_loss = uniform_field_loss(
            log_probs,
            noisy_tokens[:, cols],
            clean_tokens[:, cols],
            time_values,
            vocab_sizes[field_key],
            check_bounds=check_bounds,
        )
        total_loss = total_loss + field_loss.sum(dim=1)
    return total_loss / COLS


def batch_loss(net, clean_tokens, labels, vocab_sizes, p_uncond=0.15, *, amp=False, vocab_vec=None):
    """Mean continuous-time uniform loss over one training batch with slot shuffling, stratified times, and ask dropout."""
    batch_size = clean_tokens.shape[0]
    device = clean_tokens.device
    shuffled = M.shuffle_slots(clean_tokens)
    offsets = (torch.rand(1, device=device) + torch.arange(batch_size, device=device) / batch_size) % 1.0
    time_values = offsets[torch.randperm(batch_size, device=device)].clamp(min=TRAINING_EPS, max=1.0 - TRAINING_EPS)
    noisy_tokens = corrupt_uniform(shuffled, time_values, vocab_sizes, vocab_vec=vocab_vec)
    dropped_labels = labels.clone()
    dropped_labels[torch.rand(batch_size, device=device) < p_uncond] = NAN
    return elbo_terms(
        net, shuffled, noisy_tokens, time_values, dropped_labels, vocab_sizes, check_bounds=False, amp=amp
    ).mean()


@torch.no_grad()
def val_loss(net, rows_table, labels, vocab_sizes, reps=4, seed=12345, batch=512):
    """Clipped-time continuous-time diffusion loss estimate on fixed validation rows (4 stratified CRN draws)."""
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


def _top_predictions_and_margin(log_clean_by_key, batch_size, device):
    """Extract [B, 84] argmax clean-field token indices and top-1 minus top-2 probability margin."""
    top_tokens = torch.empty((batch_size, COLS), dtype=torch.long, device=device)
    top_margin = torch.empty((batch_size, COLS), dtype=torch.float32, device=device)
    for field_key, cols in FIELD_COLS.items():
        probs = log_clean_by_key[field_key].exp()
        top2 = torch.topk(probs, k=2, dim=-1)
        top_tokens[:, cols] = top2.indices[..., 0]
        top_margin[:, cols] = top2.values[..., 0] - top2.values[..., 1]
    return top_tokens, top_margin


def _guided_top_predictions_row(log_cond, log_uncond, guidance, batch_size, device):
    """Extract [B, 84] argmax of the tilted clean distribution gamma * log x_c + (1 - gamma) * log x_u."""
    top_tokens = torch.empty((batch_size, COLS), dtype=torch.long, device=device)
    for field_key, cols in FIELD_COLS.items():
        tilted = guidance * log_cond[field_key] + (1.0 - guidance) * log_uncond[field_key]
        top_tokens[:, cols] = tilted.argmax(dim=-1)
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


def _validate_sampler_args(passes, eps, temperature):
    if passes < 1:
        raise ValueError(f"passes must be >= 1, got {passes!r}")
    if not (0.0 < eps < 1.0):
        raise ValueError(f"eps must be strictly inside (0, 1), got {eps!r}")
    if temperature <= 0.0 or not math.isfinite(temperature):
        raise ValueError(f"temperature must be finite and > 0, got {temperature!r}")


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
    _validate_sampler_args(passes, eps, temperature)
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
    need_cond = effective_guidance != 0.0
    need_uncond = effective_guidance != 1.0

    for step in range(passes):
        move_t = times[step]
        move_s = times[step + 1]
        t_scalar = float(move_t)
        time_batch = torch.full((n,), t_scalar, device=device)
        log_cond = _predict_log_clean(model, tokens, time_batch, ask_tensor, temperature) if need_cond else None
        log_uncond = _predict_log_clean(model, tokens, time_batch, null_tensor, temperature) if need_uncond else None
        top_tokens, top_margin = _top_predictions_and_margin(
            log_cond if effective_guidance > 0.0 else log_uncond, n, device
        )
        guided_top = (
            _guided_top_predictions_row(log_cond, log_uncond, effective_guidance, n, device)
            if (need_cond and need_uncond)
            else top_tokens
        )
        next_tokens = _step_sample_tokens(log_cond, log_uncond, tokens, move_s, move_t, effective_guidance, vocab_sizes)
        tracker.record_pass(
            step,
            tokens,
            next_tokens,
            top_tokens,
            guided_top_tokens=guided_top,
            top_margin=top_margin,
            time_value=t_scalar,
        )
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


def _train_epoch(
    net,
    avg,
    optimizer,
    clean_tokens,
    labels,
    vocab_sizes,
    *,
    epoch_num,
    batch,
    lr_at,
    step,
    ema,
    label_noise,
    noise_rng,
    amp=False,
    vocab_vec=None,
):
    """Run one training epoch of continuous-time uniform diffusion with batched device-side accumulation."""
    net.train()
    device = clean_tokens.device
    net_params = list(net.parameters())
    avg_params = list(avg.parameters()) if avg is not None else None
    loss_accum = torch.zeros((), device=device, dtype=torch.float32)
    grad_norm_accum = torch.zeros((), device=device, dtype=torch.float32)
    num_batches = 0
    for indices in torch.randperm(len(clean_tokens), device=device).split(batch):
        noisy_w = noisy_labels(labels[indices], label_noise, noise_rng)
        loss = batch_loss(net, clean_tokens[indices], noisy_w, vocab_sizes, amp=amp, vocab_vec=vocab_vec)
        current_lr = lr_at(step)
        for group in optimizer.param_groups:
            group["lr"] = current_lr
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm_t = nn.utils.clip_grad_norm_(net_params, 1.0)
        optimizer.step()
        step += 1
        if avg_params is not None:
            with torch.no_grad():
                decay = min(ema, (1 + step) / (10 + step))
                alpha = 1.0 - decay
                for avg_param, net_param in zip(avg_params, net_params, strict=True):
                    avg_param.lerp_(net_param, alpha)
        loss_accum = loss_accum + loss.detach().float()
        grad_norm_accum = grad_norm_accum + grad_norm_t.detach().float()
        num_batches += 1
        if num_batches % FINITE_CHECK_EVERY == 0:
            if not math.isfinite(float(loss_accum)):
                raise RuntimeError(f"non-finite training loss at epoch {epoch_num}, step {step}")
            if not math.isfinite(float(grad_norm_accum)):
                raise RuntimeError(f"non-finite gradient norm at epoch {epoch_num}, step {step}")
    loss_total = float(loss_accum)
    grad_total = float(grad_norm_accum)
    if not math.isfinite(loss_total):
        raise RuntimeError(f"non-finite training loss at epoch {epoch_num}, step {step}: {loss_total}")
    if not math.isfinite(grad_total):
        raise RuntimeError(f"non-finite gradient norm at epoch {epoch_num}, step {step}: {grad_total}")
    return loss_total / num_batches, grad_total / num_batches, step


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
        amp=None,
        lr_schedule="cosine",
        min_lr=0.0,
        **net_kw,
    ):
        """Train a uniform discrete diffusion model from scratch on 84-field rows.

        lr_schedule, min_lr: the learning-rate schedule and its floor (lr_schedule84.py); "cosine" decays to 0 as
        every run before issue #431 did, "gidd_cosine" is GIDD's warm-up then cosine to min_lr."""
        torch.manual_seed(seed)
        tables = L.Tables(vocab, corpus)
        x_table, w_tensor, _row_weights, keep = M._training_tensors(tables, rows, labels)
        clean_tokens = table_to_uniform(x_table)
        net = UniformNet84(vocab, **net_kw).to(DEV)
        vocab_sizes = net.vocab_sizes
        use_amp = _resolve_amp(amp, DEV)
        info = dict(
            kind=CHECKPOINT_KIND,
            noise_process="uniform",
            schedule="log_linear",
            schedule_details=dict(
                alpha="1 - t",
                move_chance="t",
                time_input="raw_t",
                training_time_bounds=[TRAINING_EPS, 1.0 - TRAINING_EPS],
                sampling_eps=SAMPLING_EPS,
            ),
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
            amp_dtype="bfloat16" if use_amp else "float32",
        )
        if lr_schedule != "cosine" or min_lr:
            info.update(lr_schedule=lr_schedule, min_lr=min_lr, warmup=warmup)   # only when not the old default
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
            use_amp=use_amp,
            lr_schedule=lr_schedule,
            min_lr=min_lr,
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
        use_amp=False,
        lr_schedule="cosine",
        min_lr=0.0,
    ):
        make = lambda network, extra: cls(
            vocab, corpus, [(network, net_kw)], dict(info, **extra), default_passes=legality_passes
        )
        avg = copy.deepcopy(net).eval() if ema else None
        total_steps = math.ceil(len(clean_tokens) / batch) * epochs
        warm_steps = min(warmup, max(1, total_steps // 10))
        optimizer, fused_adamw = _make_adamw(net.parameters(), lr=lr, wd=wd, device=clean_tokens.device)
        info["fused_adamw"] = fused_adamw
        vocab_vec = vocab_vector(vocab_sizes, clean_tokens.device)
        lr_at = LS.make_lr_at(lr_schedule, peak_lr=lr, total_steps=total_steps, warmup_steps=warm_steps, min_lr=min_lr)
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
                epoch_num = epoch_idx + 1
                train_loss, grad_norm, state["step"] = _train_epoch(
                    net,
                    avg,
                    optimizer,
                    clean_tokens,
                    labels,
                    vocab_sizes,
                    epoch_num=epoch_num,
                    batch=batch,
                    lr_at=lr_at,
                    step=state["step"],
                    ema=ema,
                    label_noise=label_noise,
                    noise_rng=noise_rng,
                    amp=use_amp,
                    vocab_vec=vocab_vec,
                )
                eval_net = (avg if avg is not None else net).eval()
                entry = _evaluate_epoch(
                    epoch_num,
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
                    on_epoch(make(eval_net, dict(epoch=epoch_num)), epoch_num)
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
                if _should_stop(epoch_num, state["reset_epoch"], patience, stop_after):
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
        val_value = val_loss(eval_net, val[0], val[1], vocab_sizes)
        if not math.isfinite(val_value):
            raise RuntimeError(f"non-finite validation loss at epoch {epoch_num}: {val_value}")
        entry["val_loss"] = val_value
    entry["minutes"] = round((time.perf_counter() - t_start) / 60, 3)
    return entry


def _update_best(state, entry, eval_net, threshold, on_best, make):
    if "val_loss" not in entry or not is_better_val_loss(entry["val_loss"], state["best"]):
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
