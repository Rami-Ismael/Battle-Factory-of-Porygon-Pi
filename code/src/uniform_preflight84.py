"""Pre-registration probes, port-equality oracle, revision noise floor, and Pareto trade-off
summary for the uniform discrete diffusion comparison (issue #274, map #270;
docs/uniform-discrete-diffusion.md).

Run all battle-free pre-training probes and save the JSON report:
    /tmp/vgc-pilot/.venv/bin/python code/src/uniform_preflight84.py

Upstream reference pinned at kuleshov-group/discrete-diffusion-guidance@edb0f8c28b7caeb4ea7a06a2fee8d74ab6da1661:
  - Continuous-time uniform diffusion loss (paper Eq. 19): diffusion.py:563-614
  - Reverse uniform posterior (paper Eq. 15 / App. A Eq. 25): diffusion.py:356-381
  - Uniform classifier-free guidance posterior mixing (paper Eqs. 6-8): diffusion.py:1314-1334
"""
from collections import Counter
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
UPSTREAM_COMMIT = "edb0f8c28b7caeb4ea7a06a2fee8d74ab6da1661"
UPSTREAM_REPO = "kuleshov-group/discrete-diffusion-guidance"
TABLE_NAME = "lossdata84_shuffled_only.npz"
OUT_PATH = Path.home() / "vgc-data/results/uniform_discrete_diffusion_preflight.json"

PER_FIELD_EXPONENTS = dict(species=16.0, ability=4.0, item=4.0, move=1.5, nature=1.0, stat=1.0)
LOG_LINEAR_EXPONENTS = dict(species=1.0, ability=1.0, item=1.0, move=1.0, nature=1.0, stat=1.0)
PASS_COUNTS = (84, 128, 256)
SAMPLING_EPS = 1e-5
TRAINING_EPS = 1e-3
DEFAULT_ERROR_REF = 0.025
DEFAULT_METRIC_REF = 0.0
MIN_OVERLAP_WIDTH = 0.002


def upstream_uniform_loss(log_clean_probs, noisy_tokens, clean_tokens, time_values, vocab_size):
    """Verbatim continuous-time log-linear uniform loss from upstream diffusion.py:572-597.

    Args:
        log_clean_probs: [B, L, V] normalized log-probabilities over real tokens (excluding [MASK]).
        noisy_tokens: [B, L] integer token indices in 0..V-1.
        clean_tokens: [B, L] integer token indices in 0..V-1.
        time_values: [B] noise times in (0, 1).
        vocab_size: integer V, number of real categories for this field type.

    Returns:
        [B, L] per-position continuous-time diffusion loss (zero_recon_loss=True).
    """
    alpha_time_prime = -1.0
    alpha_time = 1.0 - time_values[..., None, None]
    clean_one_hot = F.one_hot(clean_tokens, vocab_size).to(log_clean_probs.dtype)
    clean_bar = vocab_size * alpha_time * clean_one_hot + 1.0 - alpha_time
    pred_bar = vocab_size * alpha_time * log_clean_probs.exp() + 1.0 - alpha_time
    coefficient = alpha_time_prime / (vocab_size * alpha_time)
    clean_bar_at_noisy = torch.gather(clean_bar, -1, noisy_tokens[..., None])
    pred_bar_at_noisy = torch.gather(pred_bar, -1, noisy_tokens[..., None])
    first_term = (vocab_size / clean_bar_at_noisy) - (vocab_size / pred_bar_at_noisy)
    second_term = (
        (clean_bar / clean_bar_at_noisy)
        * (pred_bar_at_noisy.log() - pred_bar.log() + clean_bar.log() - clean_bar_at_noisy.log())
    ).sum(dim=-1, keepdim=True)
    return (coefficient * (first_term - second_term)).squeeze(-1)


def generalized_uniform_loss(log_clean_probs, noisy_tokens, clean_tokens, time_values, exponents, vocab_size):
    """Generalized Eq. 19 continuous-time uniform loss for alpha_c(t) = 1 - t^w_c.

    Uses move_chance = t^w_c directly rather than 1 - (1 - t^w_c) so high exponents such as
    species w = 16 do not underflow to zero in float32 for t < 0.36. Clamping move_chance at
    1e-30 keeps 1 / move_chance well below float32 max (3.4e38) while staying exact to float32
    precision for log-linear noise (w = 1) on t in [1e-3, 1 - 1e-3].
    """
    dtype = log_clean_probs.dtype
    time_column = time_values.to(dtype).view(-1, 1, 1)
    exponent_tensor = torch.as_tensor(exponents, dtype=dtype, device=log_clean_probs.device)
    if exponent_tensor.ndim == 0:
        exponent_tensor = exponent_tensor.view(1, 1, 1)
    elif exponent_tensor.ndim == 1:
        exponent_tensor = exponent_tensor.view(1, -1, 1)
    min_move = 1e-30 if dtype == torch.float32 else 1e-300
    move_chance = time_column.pow(exponent_tensor).clamp(min=min_move)
    alpha_time = (1.0 - move_chance).clamp(min=1e-6 if dtype == torch.float32 else 1e-12)
    alpha_time_prime = -exponent_tensor * time_column.pow(exponent_tensor - 1.0)
    clean_one_hot = F.one_hot(clean_tokens, vocab_size).to(dtype)
    clean_bar = vocab_size * alpha_time * clean_one_hot + move_chance
    pred_bar = vocab_size * alpha_time * log_clean_probs.exp() + move_chance
    coefficient = alpha_time_prime / (vocab_size * alpha_time)
    clean_bar_at_noisy = torch.gather(clean_bar, -1, noisy_tokens[..., None])
    pred_bar_at_noisy = torch.gather(pred_bar, -1, noisy_tokens[..., None])
    first_term = (vocab_size / clean_bar_at_noisy) - (vocab_size / pred_bar_at_noisy)
    second_term = (
        (clean_bar / clean_bar_at_noisy)
        * (pred_bar_at_noisy.log() - pred_bar.log() + clean_bar.log() - clean_bar_at_noisy.log())
    ).sum(dim=-1, keepdim=True)
    return (coefficient * (first_term - second_term)).squeeze(-1)


def upstream_reverse_posterior(clean_probs, noisy_tokens, alpha_s, alpha_t, vocab_size):
    """Verbatim alpha-form reverse posterior from upstream diffusion.py:356-381 (commit edb0f8c2)."""
    alpha_ratio = alpha_t / alpha_s
    noisy_one_hot = F.one_hot(noisy_tokens, vocab_size).to(clean_probs.dtype)
    numerator = (
        alpha_t * vocab_size * clean_probs * noisy_one_hot
        + (alpha_ratio - alpha_t) * noisy_one_hot
        + (alpha_s - alpha_t) * clean_probs
        + (1.0 - alpha_ratio) * (1.0 - alpha_s) / vocab_size
    )
    denominator = alpha_t * vocab_size * torch.gather(clean_probs, -1, noisy_tokens[..., None]) + (1.0 - alpha_t)
    return numerator / denominator


def upstream_cfg_posterior(log_clean_cond, log_clean_uncond, noisy_tokens, alpha_s, alpha_t, gamma, vocab_size):
    """Verbatim posterior-mixing CFG from upstream diffusion.py:1314-1334 (commit edb0f8c2)."""
    if gamma == 0.0:
        return upstream_reverse_posterior(log_clean_uncond.exp(), noisy_tokens, alpha_s, alpha_t, vocab_size)
    if gamma == 1.0:
        return upstream_reverse_posterior(log_clean_cond.exp(), noisy_tokens, alpha_s, alpha_t, vocab_size)
    log_post_uncond = upstream_reverse_posterior(
        log_clean_uncond.exp(), noisy_tokens, alpha_s, alpha_t, vocab_size
    ).clamp(min=1e-30).log()
    log_post_cond = upstream_reverse_posterior(
        log_clean_cond.exp(), noisy_tokens, alpha_s, alpha_t, vocab_size
    ).clamp(min=1e-30).log()
    return F.softmax(gamma * log_post_cond + (1.0 - gamma) * log_post_uncond, dim=-1)


def bayes_reverse_posterior(clean_probs, noisy_tokens, move_chance_s, move_chance_t, vocab_size):
    """Independent Bayes-rule calculation of p_theta(z_s | z_t) proportional to q(z_t | z_s) * p_theta(z_s)."""
    alpha_s = 1.0 - move_chance_s
    alpha_t = 1.0 - move_chance_t
    delta_alpha = move_chance_t - move_chance_s
    stay_prob = alpha_t / alpha_s
    jump_prob = delta_alpha / (alpha_s * vocab_size)
    noisy_one_hot = F.one_hot(noisy_tokens, vocab_size).to(clean_probs.dtype)
    q_t_given_s = stay_prob * noisy_one_hot + jump_prob
    p_s = alpha_s * clean_probs + move_chance_s / vocab_size
    unnormalized = q_t_given_s * p_s
    return unnormalized / unnormalized.sum(dim=-1, keepdim=True)


def uniform_reverse_posterior(clean_probs, noisy_tokens, move_chance_s, move_chance_t, vocab_size):
    """Exact reverse posterior p_theta(z_s | z_t) (Schiff et al. Eq. 15 / diffusion.py:356-381).

    Takes move_chance_s = 1 - alpha(s) and move_chance_t = 1 - alpha(t) directly so
    delta_alpha = alpha(s) - alpha(t) = move_chance_t - move_chance_s avoids cancellation.
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


def uniform_cfg_posterior(log_clean_cond, log_clean_uncond, noisy_tokens, move_chance_s, move_chance_t, gamma, vocab_size):
    """Classifier-free guidance posterior mixing for uniform discrete diffusion (diffusion.py:1314-1334)."""
    if gamma == 0.0:
        return uniform_reverse_posterior(log_clean_uncond.exp(), noisy_tokens, move_chance_s, move_chance_t, vocab_size)
    if gamma == 1.0:
        return uniform_reverse_posterior(log_clean_cond.exp(), noisy_tokens, move_chance_s, move_chance_t, vocab_size)
    log_post_uncond = uniform_reverse_posterior(
        log_clean_uncond.exp(), noisy_tokens, move_chance_s, move_chance_t, vocab_size
    ).clamp(min=1e-30).log()
    log_post_cond = uniform_reverse_posterior(
        log_clean_cond.exp(), noisy_tokens, move_chance_s, move_chance_t, vocab_size
    ).clamp(min=1e-30).log()
    return F.softmax(gamma * log_post_cond + (1.0 - gamma) * log_post_uncond, dim=-1)


def port_equality_check(candidate_loss_fn, vocab_sizes, *, batch_size=8, seed=274, dtype=torch.float32):
    """Verify a candidate uniform loss matches upstream diffusion.py:572-597 on one identical batch
    across every field type at log-linear noise (exponent = 1.0), in both value and gradient.

    candidate_loss_fn(log_clean_probs, noisy_tokens, clean_tokens, time_values, exponent, vocab_size) -> [B, L]
    """
    generator = torch.Generator().manual_seed(seed)
    time_values = torch.linspace(TRAINING_EPS, 1.0 - TRAINING_EPS, batch_size, dtype=dtype)
    atol = 1e-3 if dtype == torch.float32 else 1e-12
    rtol = 1e-3 if dtype == torch.float32 else 1e-10
    grad_atol = 1e-4 if dtype == torch.float32 else 1e-10
    grad_rtol = 1e-3 if dtype == torch.float32 else 1e-8
    per_field = {}
    all_passed = True
    for field_name, vocab_size in vocab_sizes.items():
        seq_len = 24 if field_name == "move" else 36 if field_name == "stat" else 6
        clean_tokens = torch.randint(0, vocab_size, (batch_size, seq_len), generator=generator)
        corrupt_mask = torch.rand((batch_size, seq_len), generator=generator, dtype=dtype) < time_values[:, None]
        random_tokens = torch.randint(0, vocab_size, (batch_size, seq_len), generator=generator)
        noisy_tokens = torch.where(corrupt_mask, random_tokens, clean_tokens)
        base_logits = torch.randn((batch_size, seq_len, vocab_size), generator=generator, dtype=dtype)
        logits_ref = base_logits.clone().requires_grad_(True)
        logits_cand = base_logits.clone().requires_grad_(True)
        loss_ref = upstream_uniform_loss(F.log_softmax(logits_ref, dim=-1), noisy_tokens, clean_tokens, time_values, vocab_size)
        loss_cand = candidate_loss_fn(F.log_softmax(logits_cand, dim=-1), noisy_tokens, clean_tokens, time_values, 1.0, vocab_size)
        loss_ref.mean().backward()
        loss_cand.mean().backward()
        val_diff = float((loss_ref - loss_cand).abs().max().item())
        grad_diff = float((logits_ref.grad - logits_cand.grad).abs().max().item())
        val_ok = bool(torch.allclose(loss_ref, loss_cand, atol=atol, rtol=rtol))
        grad_ok = bool(torch.allclose(logits_ref.grad, logits_cand.grad, atol=grad_atol, rtol=grad_rtol))
        passed = val_ok and grad_ok
        all_passed = all_passed and passed
        per_field[field_name] = dict(
            vocab_size=vocab_size,
            max_abs_loss_diff=val_diff,
            max_abs_grad_diff=grad_diff,
            passed=passed,
        )
    return dict(dtype=str(dtype), batch_size=batch_size, seed=seed, passed=all_passed, per_field=per_field)


def revision_noise_floor(vocab_size, num_passes, exponent=1.0, eps=SAMPLING_EPS, schedule_eps=0.0):
    """Exact expected value changes beyond the first departure from uniform initialization under
    a fixed one-hot predictor (Schiff et al. App. A Eq. 25, linearly spaced grid from 1 to eps).
    """
    times = np.linspace(1.0, eps, num_passes + 1, dtype=np.float64)
    move_chances = (1.0 - schedule_eps) * (times ** exponent)
    alphas = 1.0 - move_chances
    prob_at_target = 1.0 / vocab_size
    expected_total_changes = 0.0
    log_stay_at_target = 0.0
    log_stay_away = 0.0
    for step_idx in range(num_passes):
        alpha_t, alpha_s = alphas[step_idx], alphas[step_idx + 1]
        move_t, move_s = move_chances[step_idx], move_chances[step_idx + 1]
        delta_alpha = move_t - move_s
        depart_prob = (delta_alpha / alpha_s) * ((vocab_size - 1.0) / vocab_size) * move_s / (1.0 + (vocab_size - 1.0) * alpha_t)
        away_change_prob = (delta_alpha / alpha_s) * (1.0 - move_s / vocab_size) / move_t if move_t > 0.0 else 0.0
        enter_target_prob = (delta_alpha / alpha_s) * (alpha_s + move_s / vocab_size) / move_t if move_t > 0.0 else 0.0
        expected_total_changes += prob_at_target * depart_prob + (1.0 - prob_at_target) * away_change_prob
        log_stay_at_target += math.log(max(1e-300, 1.0 - depart_prob))
        log_stay_away += math.log(max(1e-300, 1.0 - away_change_prob))
        prob_at_target = prob_at_target * (1.0 - depart_prob) + (1.0 - prob_at_target) * enter_target_prob
    prob_any_change = 1.0 - ((1.0 / vocab_size) * math.exp(log_stay_at_target) + (1.0 - 1.0 / vocab_size) * math.exp(log_stay_away))
    return float(expected_total_changes - prob_any_change)


def pareto_upper_envelope(points):
    """Return Pareto-optimal (error, metric) vertices sorted by ascending error (lower error, higher metric)."""
    sorted_pts = sorted(((float(err), float(val)) for err, val in points), key=lambda pair: (pair[0], -pair[1]))
    envelope = []
    best_metric = -math.inf
    for err, val in sorted_pts:
        if val > best_metric:
            if envelope and envelope[-1][0] == err:
                envelope[-1] = (err, val)
            else:
                envelope.append((err, val))
            best_metric = val
    return envelope


def dominated_area(points, *, error_ref=DEFAULT_ERROR_REF, metric_ref=DEFAULT_METRIC_REF):
    """2D dominated hypervolume relative to (error_ref, metric_ref) for lower-is-better error and higher-is-better metric."""
    clipped = [(err, val) for err, val in points if err <= error_ref and val >= metric_ref]
    envelope = pareto_upper_envelope(clipped)
    if not envelope:
        return 0.0
    area = 0.0
    for idx, (err, val) in enumerate(envelope):
        next_err = envelope[idx + 1][0] if idx + 1 < len(envelope) else error_ref
        area += max(0.0, next_err - err) * max(0.0, val - metric_ref)
    return float(area)


def equal_error_advantage(points_a, points_b, *, min_overlap=MIN_OVERLAP_WIDTH):
    """Mean metric advantage of frontier A minus frontier B over their shared error overlap [e_lo, e_hi].

    Each arm's upper Pareto envelope is linearly interpolated between adjacent Pareto vertices.
    Returns None when the overlap width is narrower than min_overlap.
    """
    env_a, env_b = pareto_upper_envelope(points_a), pareto_upper_envelope(points_b)
    if not env_a or not env_b:
        return None
    err_lo = max(env_a[0][0], env_b[0][0])
    err_hi = min(env_a[-1][0], env_b[-1][0])
    if err_hi - err_lo < min_overlap:
        return None
    knots = sorted({err_lo, err_hi} | {e for e, _ in env_a + env_b if err_lo < e < err_hi})
    xs_a, ys_a = [e for e, _ in env_a], [v for _, v in env_a]
    xs_b, ys_b = [e for e, _ in env_b], [v for _, v in env_b]
    vals_a = np.interp(knots, xs_a, ys_a)
    vals_b = np.interp(knots, xs_b, ys_b)
    integral = float(np.trapezoid(vals_a - vals_b, knots))
    return dict(
        overlap_low=float(err_lo),
        overlap_high=float(err_hi),
        overlap_width=float(err_hi - err_lo),
        mean_advantage=float(integral / (err_hi - err_lo)),
        signed_area=integral,
    )


def is_usable_strength(cells_complete, within_ask_unique_fraction, largest_share, error, best_lower_error):
    """Pre-registered usability gate for a guidance strength on asks 1/24 to 17/24.

    1. Every scored cell fills all 64 validator-accepted non-held-out teams within MAX_TRIES (1,280).
    2. Within-ask line-up uniqueness >= 0.20 (12.8 distinct line-ups per 64 teams) and largest_share <= 0.50.
    3. Ask-versus-got error does not degrade by more than +0.005 above the best error at lower guidance.
    """
    if not cells_complete:
        return False
    if within_ask_unique_fraction < 0.20 or largest_share > 0.50:
        return False
    if best_lower_error is not None and (error - best_lower_error) > 0.005:
        return False
    return True


def bend_points_to_add(coarse_records, *, max_added=2):
    """Pre-registered bend-point and upper-extension rule from coarse guidance results.

    Each record is a dict with keys: weight, error, within_ask_unique_fraction, usable.
    Returns up to max_added new guidance weights to add, ordered by largest gap / bracket priority.
    """
    ordered = sorted(coarse_records, key=lambda rec: rec["weight"])
    candidates = []
    for left, right in itertools.pairwise(ordered):
        bracket = left["usable"] and not right["usable"]
        err_gap = abs(right["error"] - left["error"])
        div_gap = abs(right["within_ask_unique_fraction"] - left["within_ask_unique_fraction"])
        if bracket or err_gap > 0.003 or div_gap > 0.10:
            midpoint = round(math.sqrt(left["weight"] * right["weight"]), 1)
            priority = (1 if bracket else 0, err_gap / 0.003 + div_gap / 0.10)
            candidates.append((priority, midpoint))
    if len(ordered) >= 2 and ordered[-1]["usable"]:
        top_improvement = ordered[-2]["error"] - ordered[-1]["error"]
        if top_improvement >= 0.001:
            candidates.append(((0, top_improvement / 0.001), round(ordered[-1]["weight"] * 2.0, 1)))
    candidates.sort(key=lambda item: item[0], reverse=True)
    chosen = []
    existing = {rec["weight"] for rec in ordered}
    for _, weight in candidates:
        if weight not in existing and weight not in chosen:
            chosen.append(weight)
        if len(chosen) >= max_added:
            break
    return sorted(chosen)


def _mean_species_replacement_distance(compositions):
    count = len(compositions)
    if count < 2:
        return float("nan")
    species_counts = Counter(species for comp in compositions for species in comp)
    shared_pairs = sum(freq * (freq - 1) / 2 for freq in species_counts.values())
    total_pairs = count * (count - 1) / 2
    return float(6.0 - shared_pairs / total_pairs)


def _load_project_modules():
    sys.path.insert(0, "/tmp/vgc-pilot/src")
    sys.path.insert(0, str(HERE))
    import ask_line
    import asked_vs_got as X
    import grid_novelty
    import heldout as HO
    import lossdown as L
    import stat_points as SP
    import team84 as T
    import team_diversity as TD
    from corpus import parse_team_text

    return dict(ask_line=ask_line, X=X, grid_novelty=grid_novelty, HO=HO, L=L, SP=SP, T=T, TD=TD, parse_team_text=parse_team_text)


def _probe_table_and_vocab(mods):
    X, L, SP, T, HO = mods["X"], mods["L"], mods["SP"], mods["T"], mods["HO"]
    vocab = T.regulation_vocab()
    table_path = X.RES / TABLE_NAME
    data = np.load(table_path)
    labelled = ~np.isnan(data["all_W"])
    train_mask = ~data["all_val"] & labelled
    val_mask = data["all_val"] & labelled
    train_rows = data["all_X"][train_mask].astype(np.int64)
    val_rows = data["all_X"][val_mask].astype(np.int64)
    tables = L.Tables(vocab, [])
    train_rejects = int(tables.rejects(train_rows[:, :T.COLS48]).sum())
    val_rejects = int(tables.rejects(val_rows[:, :T.COLS48]).sum())
    real_vocab = {key: vocab.sizes[key] - 1 for key in ("species", "ability", "item", "move", "nature")}
    real_vocab["stat"] = SP.NVALUES
    used_train = {key: len(np.unique(train_rows[:, cols])) for key, cols in L.COLS_BY_KEY.items()}
    used_train["stat"] = len(np.unique(train_rows[:, T.COLS48:]))
    train_keys = {b"".join(sorted(np.concatenate([row[:T.COLS48].reshape(6, -1)[i], row[T.COLS48:].reshape(6, -1)[i]]).tobytes() for i in range(6))) for row in train_rows}
    heldout_overlap = 0
    heldout_encoded = 0
    for path in HO.files():
        team = mods["parse_team_text"](Path(path).read_text())
        try:
            encoded = T.encode(vocab, team)
        except ValueError:
            continue
        heldout_encoded += 1
        key = b"".join(sorted(np.concatenate([encoded[:T.COLS48].reshape(6, -1)[i], encoded[T.COLS48:].reshape(6, -1)[i]]).tobytes() for i in range(6)))
        heldout_overlap += int(key in train_keys)
    return dict(
        table=TABLE_NAME,
        sha256=hashlib.sha256(table_path.read_bytes()).hexdigest(),
        total_rows=len(data["all_X"]),
        labelled_total=int(labelled.sum()),
        labelled_train=int(train_mask.sum()),
        labelled_val=int(val_mask.sum()),
        unlabelled=int((~labelled).sum()),
        min_battles_labelled=int(data["all_NB"][labelled].min()),
        wrong_stone_rejects=dict(train=train_rejects, val=val_rejects, passed=(train_rejects == 0 and val_rejects == 0)),
        separation=dict(heldout_encoded=heldout_encoded, overlap=heldout_overlap, passed=heldout_overlap == 0),
        vocab_total_with_mask={**vocab.sizes, "stat": SP.NVALUES + 1},
        vocab_real_uniform=real_vocab,
        vocab_used_in_train=used_train,
    )


def _probe_stability_and_kl(real_vocab):
    eq_f32 = port_equality_check(generalized_uniform_loss, real_vocab, dtype=torch.float32)
    eq_f64 = port_equality_check(generalized_uniform_loss, real_vocab, dtype=torch.float64)
    torch.manual_seed(274)
    times = torch.tensor([1e-3, 0.01, 0.1, 0.25, 0.5, 0.9, 1.0 - 1e-3], dtype=torch.float32)
    stability = {}
    for field_name, exponent in PER_FIELD_EXPONENTS.items():
        vocab_size = real_vocab[field_name]
        clean = torch.randint(0, vocab_size, (len(times), 6))
        move = torch.rand(len(times), 6) < times[:, None].pow(exponent)
        noisy = torch.where(move, torch.randint(0, vocab_size, (len(times), 6)), clean)
        logits = torch.randn(len(times), 6, vocab_size, requires_grad=True)
        loss = generalized_uniform_loss(F.log_softmax(logits, dim=-1), noisy, clean, times, exponent, vocab_size)
        loss.mean().backward()
        stability[field_name] = dict(
            exponent=exponent,
            finite_loss=bool(torch.isfinite(loss).all()),
            finite_grad=bool(torch.isfinite(logits.grad).all()),
            passed=bool(torch.isfinite(loss).all() and torch.isfinite(logits.grad).all()),
        )
    dt = 1e-7
    t_kl = torch.linspace(0.3, 0.9, 8, dtype=torch.float64)
    kl_check = {}
    for exponent in (1.0, 1.5, 4.0, 16.0):
        vocab_size = 33
        clean = torch.randint(0, vocab_size, (8, 6))
        noisy = torch.randint(0, vocab_size, (8, 6))
        log_pred = F.log_softmax(torch.randn(8, 6, vocab_size, dtype=torch.float64), dim=-1)
        analytic = generalized_uniform_loss(log_pred, noisy, clean, t_kl, exponent, vocab_size)
        move_t = t_kl[:, None, None].pow(exponent)
        move_s = (t_kl - dt)[:, None, None].pow(exponent)
        q_true = uniform_reverse_posterior(F.one_hot(clean, vocab_size).double(), noisy, move_s, move_t, vocab_size)
        p_model = uniform_reverse_posterior(log_pred.exp(), noisy, move_s, move_t, vocab_size)
        kl_val = (q_true * (q_true.log() - p_model.log())).sum(dim=-1)
        max_rel = float(((analytic - kl_val / dt).abs() / analytic.abs().clamp_min(1e-8)).max().item())
        kl_check[f"w={exponent:g}"] = dict(max_rel_diff=max_rel, passed=max_rel < 5e-4)
    return dict(
        port_equality_float32=eq_f32,
        port_equality_float64=eq_f64,
        float32_exponent_stability=stability,
        finite_diff_kl_identity=kl_check,
        passed=eq_f32["passed"] and eq_f64["passed"] and all(v["passed"] for v in stability.values()) and all(v["passed"] for v in kl_check.values()),
    )


def _probe_baseline_frontier(mods):
    res_dir = Path.home() / "vgc-data/results"
    train_lineups = mods["grid_novelty"].training_line_ups(TABLE_NAME)
    grid_pairs = [
        (1, "stat_points_84_sched_linear_species_first_g1.json", "stat_points_84_sched_linear_species_first_g1_seed795783370.json"),
        (2, "stat_points_84_sched_linear_species_first.json", "stat_points_84_sched_linear_species_first_seed940135810.json"),
        (4, "stat_points_84_sched_linear_species_first_g4.json", "stat_points_84_sched_linear_species_first_g4_seed630821980.json"),
        (8, "stat_points_84_sched_linear_species_first_g8.json", "stat_points_84_sched_linear_species_first_g8_seed111619920.json"),
    ]
    rows = []
    for guidance, battle_file, redraw_file in grid_pairs:
        if not (res_dir / battle_file).exists() or not (res_dir / redraw_file).exists():
            continue
        bdata = json.loads((res_dir / battle_file).read_text())
        rdata = json.loads((res_dir / redraw_file).read_text())
        suffix = "" if guidance == 2 else f"_g{guidance}"
        by_seed = {}
        for seed in (1, 2, 3):
            ask_rows = [
                dict(numerator=k, win=float(np.mean(bdata["cells"][f"{bdata['model']}{suffix}_a{k:02d}of24_s{seed}"]["y"])))
                for k in range(1, 18)
            ]
            err = mods["ask_line"].line_measures(ask_rows)["mse_scored_asks"]
            all_comps, wa_uniq, wa_srd, wa_top = [], [], [], []
            for k in range(1, 18):
                cell_key = next(ck for ck in rdata["cells"] if ck.endswith(f"_a{k:02d}of24_s{seed}"))
                comps = [tuple(sorted(slot["species"] for slot in mods["TD"].slots_from_paste(p))) for p in rdata["cells"][cell_key]["pastes"]]
                all_comps.extend(comps)
                counts = Counter(comps)
                wa_uniq.append(len(counts) / len(comps))
                wa_srd.append(_mean_species_replacement_distance(comps))
                wa_top.append(max(counts.values()) / len(comps))
            by_seed[str(seed)] = dict(
                error=float(err),
                novel_share=float(np.mean([cp not in train_lineups for cp in all_comps])),
                within_ask_unique_fraction=float(np.mean(wa_uniq)),
                within_ask_species_replacement_distance=float(np.mean(wa_srd)),
                within_ask_largest_share=float(np.mean(wa_top)),
            )
        rows.append(dict(guidance=guidance, by_seed=by_seed))
    hv_by_metric = {}
    for metric in ("novel_share", "within_ask_unique_fraction", "within_ask_species_replacement_distance"):
        seed_hvs = [dominated_area([(r["by_seed"][str(s)]["error"], r["by_seed"][str(s)][metric]) for r in rows]) for s in (1, 2, 3)]
        hv_by_metric[metric] = dict(by_seed=seed_hvs, mean=float(np.mean(seed_hvs)), min=float(min(seed_hvs)), max=float(max(seed_hvs)))
    return dict(provisional_model="sched_linear_species_first", training_distinct_lineups=len(train_lineups), points=rows, dominated_area=hv_by_metric)


def run_preflight(save_path=OUT_PATH):
    mods = _load_project_modules()
    table_vocab = _probe_table_and_vocab(mods)
    real_vocab = table_vocab["vocab_real_uniform"]
    math_probes = _probe_stability_and_kl(real_vocab)
    noise_floors = {}
    for schedule_name, exp_map in (("log_linear", LOG_LINEAR_EXPONENTS), ("per_field", PER_FIELD_EXPONENTS)):
        noise_floors[schedule_name] = {
            field_name: {str(k): round(revision_noise_floor(real_vocab[field_name], k, exp_map[field_name]), 6) for k in PASS_COUNTS}
            for field_name in PER_FIELD_EXPONENTS
        }
    baseline = _probe_baseline_frontier(mods)
    report = dict(
        upstream=dict(repo=UPSTREAM_REPO, commit=UPSTREAM_COMMIT),
        table_and_vocab=table_vocab,
        math_and_port_equality=math_probes,
        revision_noise_floor_ideal=noise_floors,
        provisional_baseline_frontier=baseline,
        passed=bool(table_vocab["wrong_stone_rejects"]["passed"] and table_vocab["separation"]["passed"] and math_probes["passed"]),
    )
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        Path(save_path).write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    result = run_preflight()
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit("UNIFORM_PREFLIGHT84 FAILED")
    print("UNIFORM_PREFLIGHT84 PASS")
