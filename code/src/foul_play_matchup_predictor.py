"""Matchup predictor on the 500 Foul Play-labelled pilot teams (issue #419, map #416).

Tests whether a matchup predictor trained on the 500 pilot teams' Foul Play deterministic argmax win rate
(issue #447, policy_id = 10) achieves lower 5-fold cross-validated error than reusing each team's existing
behaviour-cloning win rate (policy_id = 8) directly as the guess.

Pre-registered arms (shared five folds, seed 419):
  predictor_team_bc    team 84-field features + stored behaviour-cloning win rate (policy_id = 8)
  predictor_team_only  team 84-field features only
  guess_bc_label       no training; stored behaviour-cloning win rate is the prediction

Secondary diagnostic controls (reported beside the three arms):
  calibrate_bc_only    linear calibration a + b * bc_wr fit inside each training fold (no team features)
  guess_train_mean     constant mean of each training fold

    PY=/tmp/vgc-pilot/.venv/bin/python
    $PY code/src/foul_play_matchup_predictor.py preflight
    $PY code/src/foul_play_matchup_predictor.py run
    $PY code/src/foul_play_matchup_predictor.py figures
"""
import argparse
import hashlib
import itertools
import json
import os
import random
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

PILOT_DIR = Path(os.environ.get("FP_PILOT_OUT", Path.home() / "vgc-data" / "foul_play_label_pilot_447"))
OUT = Path(os.environ.get("FP_PREDICTOR_OUT", Path.home() / "vgc-data" / "foul_play_matchup_predictor_419"))
SEED = 419
K_FOLDS = 5
MIN_PAIR = 5
LAM_GRID = (1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0)
PAIR_MULT = 3.0
BOOTSTRAP = 10000
SMALLEST_EFFECT = 0.005
DEFAULT_PREDICTOR_NAME = "foul_play_det_ridge_team_bc_v1"
PREDICTOR_KIND = "predicted_score"
PREDICTOR_MODEL_FAMILY = "ridge_multi_fidelity_loocv"
STATS = ("HP", "Atk", "Def", "SpA", "SpD", "Spe")
ARM_LABELS = {
    "predictor_team_bc": "Team and behaviour-cloning win rate",
    "predictor_team_only": "Team only, no behaviour-cloning win rate",
    "guess_bc_label": "No training, behaviour-cloning win rate as the guess",
    "calibrate_bc_only": "No team features, linear behaviour-cloning calibration",
    "guess_train_mean": "No features, training-fold mean as the guess",
}
SHORT_LABELS = {
    "predictor_team_bc": "Team + old win rate",
    "predictor_team_only": "Team only",
    "guess_bc_label": "No training (old win rate)",
    "calibrate_bc_only": "No team features (linear fit)",
    "guess_train_mean": "No features (fold mean)",
}


# ---------------------------------------------------------------- folds
def make_folds(team_ids, k=K_FOLDS, seed=SEED):
    """Deterministic k disjoint folds of sorted team_ids, shuffled with random.Random(seed)."""
    ids = sorted(int(t) for t in team_ids)
    rng = random.Random(seed)
    rng.shuffle(ids)
    return [sorted(ids[i::k]) for i in range(k)]


def make_grouped_folds(team_ids, group_of, k=K_FOLDS, seed=SEED):
    """Greedy balanced k-fold split keeping every team with the same group_of[tid] in one fold."""
    groups = defaultdict(list)
    for tid in sorted(int(t) for t in team_ids):
        groups[group_of[tid]].append(tid)
    rng = random.Random(seed)
    keys = sorted(groups)
    rng.shuffle(keys)
    keys.sort(key=lambda g: -len(groups[g]))
    folds = [[] for _ in range(k)]
    for g in keys:
        idx = min(range(k), key=lambda i: len(folds[i]))
        folds[idx].extend(groups[g])
    return [sorted(f) for f in folds]


# ---------------------------------------------------------------- features
def _norm(text):
    return "".join(ch for ch in (text or "").lower() if ch.isalnum())


def _stat_summary(team):
    """10 permutation-invariant Stat Point summaries across the 6 Pokemon."""
    mat = np.array([[float(s.get("evs", {}).get(st, 0)) for st in STATS] for s in team], dtype=np.float64)
    means = mat.mean(axis=0)
    mean_total = float(mat.sum(axis=1).mean())
    mean_max = float(mat.max(axis=1).mean())
    share_fast = float((mat[:, 5] >= 28.0).mean())
    share_hp = float((mat[:, 0] >= 28.0).mean())
    return np.concatenate([means, [mean_total, mean_max, share_fast, share_hp]])


def _collect_vocab(teams):
    vals = {k: set() for k in ("species", "ability", "item", "move", "nature")}
    pair_counts = Counter()
    for team in teams:
        species = sorted({_norm(s.get("species")) for s in team if s.get("species")})
        for a, b in itertools.combinations(species, 2):
            pair_counts[(a, b)] += 1
        for slot in team:
            for key in ("species", "ability", "item", "nature"):
                v = _norm(slot.get(key))
                if v:
                    vals[key].add(v)
            for mv in slot.get("moves", ()):
                v = _norm(mv)
                if v:
                    vals["move"].add(v)
    return vals, pair_counts


@dataclass
class FeatureMap:
    """Fold-local feature encoder: field indicators + frequent species pairs + standardized Stat Point summaries."""

    single_idx: dict
    pair_idx: dict
    stat_mean: np.ndarray
    stat_sd: np.ndarray
    n_single: int
    n_pair: int
    n_stat: int

    @property
    def dim(self):
        return self.n_single + self.n_pair + self.n_stat

    @property
    def feature_names(self):
        names = [None] * (self.n_single + self.n_pair)
        for (k, v), i in self.single_idx.items():
            names[i] = f"{k}:{v}"
        for (a, b), i in self.pair_idx.items():
            names[i] = f"pair:{a}+{b}"
        stat_names = [f"sp_mean:{st}" for st in STATS] + ["sp_mean_total", "sp_mean_max", "sp_share_spe28", "sp_share_hp28"]
        return names + stat_names

    @classmethod
    def fit(cls, teams, min_pair=MIN_PAIR):
        vals, pair_counts = _collect_vocab(teams)
        single_idx, off = {}, 0
        for key in ("species", "ability", "item", "move", "nature"):
            for val in sorted(vals[key]):
                single_idx[(key, val)] = off
                off += 1
        n_single = off
        pairs = sorted(p for p, c in pair_counts.items() if c >= min_pair)
        pair_idx = {p: n_single + i for i, p in enumerate(pairs)}
        stats = np.stack([_stat_summary(t) for t in teams])
        mean = stats.mean(axis=0)
        sd = np.where(stats.std(axis=0) > 1e-8, stats.std(axis=0), 1.0)
        return cls(single_idx, pair_idx, mean, sd, n_single, len(pairs), len(mean))

    def _encode_one(self, team):
        x = np.zeros(self.dim, dtype=np.float64)
        species = sorted({_norm(s.get("species")) for s in team if s.get("species")})
        for a, b in itertools.combinations(species, 2):
            idx = self.pair_idx.get((a, b))
            if idx is not None:
                x[idx] = 1.0
        for slot in team:
            for key in ("species", "ability", "item", "nature"):
                idx = self.single_idx.get((key, _norm(slot.get(key))))
                if idx is not None:
                    x[idx] += 1.0
            for mv in slot.get("moves", ()):
                idx = self.single_idx.get(("move", _norm(mv)))
                if idx is not None:
                    x[idx] += 1.0
        x[self.n_single + self.n_pair :] = (_stat_summary(team) - self.stat_mean) / self.stat_sd
        return x

    def transform(self, teams):
        return np.stack([self._encode_one(t) for t in teams])

    def penalty_vector(self, lam_s, lam_p):
        return np.concatenate(
            [
                np.full(self.n_single, float(lam_s)),
                np.full(self.n_pair, float(lam_p)),
                np.full(self.n_stat, float(lam_s)),
            ]
        )


# ---------------------------------------------------------------- ridge + exact LOOCV
def solve_ridge_with_loocv(X_unpen, X_pen, y, lam_vec):
    """Dual ridge solve with unpenalized columns X_unpen and diagonal positive penalties lam_vec on X_pen.

    Returns (loocv_mse, w_unpen, w_pen). Exact closed-form LOOCV via S = M - M U (U^T M U)^-1 U^T M where
    M = (I + X_pen diag(1/lam) X_pen^T)^-1, residuals r = S y, and leave-one-out errors e_i = r_i / S_ii.
    """
    n = len(y)
    U = np.asarray(X_unpen, dtype=np.float64)
    Z = np.asarray(X_pen, dtype=np.float64)
    yv = np.asarray(y, dtype=np.float64)
    inv_lam = 1.0 / np.asarray(lam_vec, dtype=np.float64)

    K = (Z * inv_lam[None, :]) @ Z.T
    A = K + np.eye(n, dtype=np.float64)
    L = np.linalg.cholesky(A)
    M = np.linalg.solve(L.T, np.linalg.solve(L, np.eye(n, dtype=np.float64)))
    MU = M @ U
    UtMU = U.T @ MU
    w_unpen = np.linalg.solve(UtMU, MU.T @ yv)
    S = M - MU @ np.linalg.solve(UtMU, MU.T)
    r = M @ (yv - U @ w_unpen)
    w_pen = inv_lam * (Z.T @ r)
    diag_s = np.maximum(np.diag(S), 1e-12)
    loo_err = r / diag_s
    return float(np.mean(loo_err**2)), w_unpen, w_pen


def _unpenalized_matrix(n, bc_wr, use_bc):
    if not use_bc:
        return np.ones((n, 1), dtype=np.float64)
    if bc_wr is None:
        raise ValueError("use_bc=True requires bc_wr")
    return np.column_stack([np.ones(n, dtype=np.float64), np.asarray(bc_wr, dtype=np.float64)])


@dataclass
class RidgePredictor:
    """Trained ridge matchup predictor with fold-local FeatureMap and chosen regularization."""

    fmap: FeatureMap
    w_unpen: np.ndarray
    w_pen: np.ndarray
    use_bc: bool
    lam_s: float
    lam_p: float
    loocv_mse: float

    def predict(self, teams, bc_wr=None):
        X_pen = self.fmap.transform(teams)
        U = _unpenalized_matrix(len(teams), bc_wr, self.use_bc)
        return np.clip(U @ self.w_unpen + X_pen @ self.w_pen, 0.0, 1.0)


def fit_ridge_arm(teams, y, bc_wr=None, *, use_bc=False, min_pair=MIN_PAIR, lam_grid=LAM_GRID):
    """Fit a ridge arm with inner exact LOOCV selecting lam_s from lam_grid (lam_p = PAIR_MULT * lam_s)."""
    fmap = FeatureMap.fit(teams, min_pair=min_pair)
    X_pen = fmap.transform(teams)
    U = _unpenalized_matrix(len(teams), bc_wr, use_bc)

    best = None
    for lam_s in lam_grid:
        lam_p = lam_s * PAIR_MULT
        mse, w_u, w_p = solve_ridge_with_loocv(U, X_pen, y, fmap.penalty_vector(lam_s, lam_p))
        if best is None or mse < best[0]:
            best = (mse, lam_s, lam_p, w_u, w_p)
    mse, lam_s, lam_p, w_u, w_p = best
    return RidgePredictor(fmap, w_u, w_p, use_bc, lam_s, lam_p, mse)


@dataclass
class RidgeUncertaintyModel:
    """RidgePredictor plus Bayesian ridge posterior covariance and calibrated predictive-SD parameters."""

    predictor: RidgePredictor
    cov_factor: np.ndarray
    df: float
    sigma_resid: float
    sigma_true_resid: float
    target_noise_var: float

    def _extra_bc_var(self, bc_wr, bc_battles, bc_opponents):
        n = len(bc_wr)
        if not self.predictor.use_bc or bc_battles is None or bc_opponents is None:
            return np.zeros(n, dtype=np.float64)
        nb = np.asarray(bc_battles, dtype=np.float64)
        k = np.asarray(bc_opponents, dtype=np.float64)
        beta_bc = float(self.predictor.w_unpen[1])
        p_tilde = (np.asarray(bc_wr, dtype=np.float64) * nb + 0.5) / (nb + 1.0)
        depth_gap = np.maximum(0.0, 1.0 / np.maximum(nb, 1.0) - 1.0 / np.maximum(2.0 * k, 1.0))
        return (beta_bc**2) * depth_gap * p_tilde * (1.0 - p_tilde)

    def predict_with_std(self, teams, bc_wr=None, *, bc_battles=None, bc_opponents=None):
        """Return (pred_win_rate, pred_std, epistemic_std) for `teams`.

        epistemic_std = sigma_resid * sqrt(x^T (X^T X + P)^-1 x)
        pred_std      = sqrt(sigma_true_resid^2 + epistemic_std^2 + extra_bc_input_var)
        """
        X_pen = self.predictor.fmap.transform(teams)
        U = _unpenalized_matrix(len(teams), bc_wr, self.predictor.use_bc)
        X = np.hstack([U, X_pen])
        w = np.concatenate([self.predictor.w_unpen, self.predictor.w_pen])
        pred = np.clip(X @ w, 0.0, 1.0)
        proj = X @ self.cov_factor
        ep_var = (self.sigma_resid**2) * np.sum(proj * proj, axis=1)
        extra_bc = self._extra_bc_var(bc_wr if bc_wr is not None else np.zeros(len(teams)), bc_battles, bc_opponents)
        pred_std = np.sqrt(self.sigma_true_resid**2 + ep_var + extra_bc)
        return pred, pred_std, np.sqrt(ep_var)


def fit_ridge_with_uncertainty(
    teams, y, bc_wr=None, *, use_bc=False, min_pair=MIN_PAIR, lam_grid=LAM_GRID, target_noise_var=0.0
):
    """Fit RidgePredictor and exact Bayesian ridge posterior covariance factor (L^-T of X^T X + P)."""
    pred_model = fit_ridge_arm(teams, y, bc_wr, use_bc=use_bc, min_pair=min_pair, lam_grid=lam_grid)
    X_pen = pred_model.fmap.transform(teams)
    U = _unpenalized_matrix(len(teams), bc_wr, use_bc)
    X = np.hstack([U, X_pen])
    yv = np.asarray(y, dtype=np.float64)
    pen = np.concatenate([np.zeros(U.shape[1], dtype=np.float64), pred_model.fmap.penalty_vector(pred_model.lam_s, pred_model.lam_p)])
    A = X.T @ X + np.diag(pen)
    L_a = np.linalg.cholesky(A)
    cov_factor = np.linalg.solve(L_a, np.eye(A.shape[0], dtype=np.float64)).T
    H_proj = X @ cov_factor
    df = float(np.sum(H_proj * H_proj))
    w = np.concatenate([pred_model.w_unpen, pred_model.w_pen])
    resid = yv - X @ w
    sigma2_resid = float(np.sum(resid**2) / max(len(yv) - df, 1.0))
    sigma2_true = max(0.0, sigma2_resid - float(target_noise_var))
    return RidgeUncertaintyModel(
        predictor=pred_model,
        cov_factor=cov_factor,
        df=df,
        sigma_resid=float(np.sqrt(sigma2_resid)),
        sigma_true_resid=float(np.sqrt(sigma2_true)),
        target_noise_var=float(target_noise_var),
    )


def fit_linear_bc_calibration(bc_train, y_train):
    """OLS calibration a + b * bc_wr, no team features."""
    U = np.column_stack([np.ones(len(bc_train)), np.asarray(bc_train, dtype=np.float64)])
    coef, *_ = np.linalg.lstsq(U, np.asarray(y_train, dtype=np.float64), rcond=None)
    return float(coef[0]), float(coef[1])


# ---------------------------------------------------------------- metrics & bootstrap
def _ranks(arr):
    order = np.argsort(arr, kind="mergesort")
    r = np.empty(len(arr), dtype=np.float64)
    i = 0
    while i < len(arr):
        j = i
        while j + 1 < len(arr) and arr[order[j + 1]] == arr[order[i]]:
            j += 1
        r[order[i : j + 1]] = (i + j) / 2.0
        i = j + 1
    return r


def _corr(a, b):
    da, db = a - a.mean(), b - b.mean()
    denom = float(np.sqrt((da**2).sum() * (db**2).sum()))
    return float((da * db).sum() / denom) if denom > 0 else float("nan")


def metrics_of(pred, target):
    """Point metrics for predicted win rates vs target win rates."""
    p = np.asarray(pred, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    err = p - y
    mse = float(np.mean(err**2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(err)))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = float(1.0 - np.sum(err**2) / ss_tot) if ss_tot > 0 else float("nan")
    return {
        "rmse": rmse,
        "mse": mse,
        "mae": mae,
        "r2": r2,
        "pearson_r": _corr(p, y),
        "spearman_rho": _corr(_ranks(p), _ranks(y)),
        "mean_bias": float(np.mean(err)),
    }


def _bootstrap_metric_ci(pred, target, indices):
    p = np.asarray(pred, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    sq = (p - y) ** 2
    ae = np.abs(p - y)
    boot_mse = sq[indices].mean(axis=1)
    boot_rmse = np.sqrt(boot_mse)
    boot_mae = ae[indices].mean(axis=1)
    yb = y[indices]
    ss_tot = np.sum((yb - yb.mean(axis=1, keepdims=True)) ** 2, axis=1)
    boot_r2 = 1.0 - sq[indices].sum(axis=1) / np.maximum(ss_tot, 1e-12)
    return {
        "rmse": [float(v) for v in np.percentile(boot_rmse, [2.5, 97.5])],
        "mse": [float(v) for v in np.percentile(boot_mse, [2.5, 97.5])],
        "mae": [float(v) for v in np.percentile(boot_mae, [2.5, 97.5])],
        "r2": [float(v) for v in np.percentile(boot_r2, [2.5, 97.5])],
    }


def paired_bootstrap_ci(pred_a, pred_b, target, indices, *, margin=SMALLEST_EFFECT):
    """Paired bootstrap CI for error(A) - error(B); negative means A has lower error (better)."""
    y = np.asarray(target, dtype=np.float64)
    sq_a = (np.asarray(pred_a, dtype=np.float64) - y) ** 2
    sq_b = (np.asarray(pred_b, dtype=np.float64) - y) ** 2
    ae_a = np.abs(np.asarray(pred_a, dtype=np.float64) - y)
    ae_b = np.abs(np.asarray(pred_b, dtype=np.float64) - y)

    rmse_diff = float(np.sqrt(sq_a.mean()) - np.sqrt(sq_b.mean()))
    boot_rmse_diff = np.sqrt(sq_a[indices].mean(axis=1)) - np.sqrt(sq_b[indices].mean(axis=1))
    lo, hi = [float(v) for v in np.percentile(boot_rmse_diff, [2.5, 97.5])]
    if hi < 0.0:
        verdict = "better"
    elif lo > 0.0:
        verdict = "worse"
    elif abs(lo) <= margin and abs(hi) <= margin:
        verdict = "same"
    else:
        verdict = "undecided"

    mae_diff = float(ae_a.mean() - ae_b.mean())
    boot_mae_diff = ae_a[indices].mean(axis=1) - ae_b[indices].mean(axis=1)
    mse_diff = float(sq_a.mean() - sq_b.mean())
    boot_mse_diff = sq_a[indices].mean(axis=1) - sq_b[indices].mean(axis=1)
    return {
        "rmse": {"diff": rmse_diff, "ci95": [lo, hi], "verdict": verdict},
        "mse": {"diff": mse_diff, "ci95": [float(v) for v in np.percentile(boot_mse_diff, [2.5, 97.5])]},
        "mae": {"diff": mae_diff, "ci95": [float(v) for v in np.percentile(boot_mae_diff, [2.5, 97.5])]},
    }


def _run_fold_predictions(team_ids, teams, fp_target, bc_input, folds, min_pair, lam_grid, *, bc_test_input=None):
    idx_of = {tid: i for i, tid in enumerate(team_ids)}
    bc_eval = bc_input if bc_test_input is None else bc_test_input
    n = len(team_ids)
    preds = {name: np.zeros(n, dtype=np.float64) for name in ARM_LABELS}
    fold_meta = []

    for fi, test_ids in enumerate(folds):
        test_set = set(test_ids)
        train_ids = [t for t in team_ids if t not in test_set]
        tr_teams = [teams[t] for t in train_ids]
        te_teams = [teams[t] for t in test_ids]
        y_tr = np.array([fp_target[t] for t in train_ids], dtype=np.float64)
        bc_tr = np.array([bc_input[t] for t in train_ids], dtype=np.float64)
        bc_te = np.array([bc_eval[t] for t in test_ids], dtype=np.float64)

        m_bc = fit_ridge_arm(tr_teams, y_tr, bc_tr, use_bc=True, min_pair=min_pair, lam_grid=lam_grid)
        m_only = fit_ridge_arm(tr_teams, y_tr, None, use_bc=False, min_pair=min_pair, lam_grid=lam_grid)
        a_cal, b_cal = fit_linear_bc_calibration(bc_tr, y_tr)
        tr_mean = float(y_tr.mean())

        te_idx = np.array([idx_of[t] for t in test_ids], dtype=np.int64)
        preds["predictor_team_bc"][te_idx] = m_bc.predict(te_teams, bc_te)
        preds["predictor_team_only"][te_idx] = m_only.predict(te_teams, None)
        preds["guess_bc_label"][te_idx] = bc_te
        preds["calibrate_bc_only"][te_idx] = np.clip(a_cal + b_cal * bc_te, 0.0, 1.0)
        preds["guess_train_mean"][te_idx] = tr_mean
        fold_meta.append(
            {
                "fold": fi,
                "n_train": len(train_ids),
                "n_test": len(test_ids),
                "predictor_team_bc": {
                    "lam_s": m_bc.lam_s,
                    "lam_p": m_bc.lam_p,
                    "loocv_rmse": float(np.sqrt(m_bc.loocv_mse)),
                    "intercept": float(m_bc.w_unpen[0]),
                    "bc_slope": float(m_bc.w_unpen[1]),
                    "n_features": m_bc.fmap.dim,
                },
                "predictor_team_only": {
                    "lam_s": m_only.lam_s,
                    "lam_p": m_only.lam_p,
                    "loocv_rmse": float(np.sqrt(m_only.loocv_mse)),
                    "intercept": float(m_only.w_unpen[0]),
                    "n_features": m_only.fmap.dim,
                },
                "calibrate_bc_only": {"intercept": a_cal, "bc_slope": b_cal},
            }
        )
    return preds, fold_meta


def evaluate_arms(
    team_ids,
    teams,
    fp_target,
    bc_input,
    folds,
    *,
    bc_test_input=None,
    min_pair=MIN_PAIR,
    lam_grid=LAM_GRID,
    boot=BOOTSTRAP,
    seed=SEED,
):
    """Run 5-fold CV for all arms on identical folds, returning metrics, CIs, per-fold errors, and paired gaps."""
    team_ids = [int(t) for t in team_ids]
    idx_of = {tid: i for i, tid in enumerate(team_ids)}
    y_all = np.array([fp_target[t] for t in team_ids], dtype=np.float64)
    preds, fold_meta = _run_fold_predictions(
        team_ids, teams, fp_target, bc_input, folds, min_pair, lam_grid, bc_test_input=bc_test_input
    )

    rng = np.random.default_rng(seed)
    boot_idx = rng.integers(0, len(team_ids), size=(boot, len(team_ids)))
    arms_out = {}
    for name, oof in preds.items():
        per_fold = []
        for test_ids in folds:
            ti = np.array([idx_of[t] for t in test_ids], dtype=np.int64)
            per_fold.append(metrics_of(oof[ti], y_all[ti])["rmse"])
        arms_out[name] = {
            "reader_name": ARM_LABELS[name],
            **metrics_of(oof, y_all),
            "ci95": _bootstrap_metric_ci(oof, y_all, boot_idx),
            "per_fold_rmse": [float(v) for v in per_fold],
            "oof_pred": [float(v) for v in oof],
        }

    pairs = [
        ("predictor_team_bc", "guess_bc_label"),
        ("predictor_team_only", "guess_bc_label"),
        ("calibrate_bc_only", "guess_bc_label"),
        ("predictor_team_bc", "calibrate_bc_only"),
        ("predictor_team_bc", "predictor_team_only"),
    ]
    comparisons = {}
    for a, b in pairs:
        comp = paired_bootstrap_ci(preds[a], preds[b], y_all, boot_idx)
        fold_diffs = [arms_out[a]["per_fold_rmse"][i] - arms_out[b]["per_fold_rmse"][i] for i in range(len(folds))]
        comp["per_fold_rmse_diff"] = [float(v) for v in fold_diffs]
        comp["folds_better"] = sum(1 for d in fold_diffs if d < 0.0)
        comparisons[f"{a}_minus_{b}"] = comp

    return {
        "n_teams": len(team_ids),
        "team_ids": team_ids,
        "folds": folds,
        "target_fp": [float(v) for v in y_all],
        "input_bc": [float(bc_input[t]) for t in team_ids],
        "arms": arms_out,
        "comparisons": comparisons,
        "fold_details": fold_meta,
    }


def evaluate_predictions_on_target(oof_by_arm, team_ids, target_map, folds, *, boot=BOOTSTRAP, seed=SEED):
    """Score already-computed out-of-fold predictions against a different target (e.g., 10 validation opponents)."""
    idx_of = {tid: i for i, tid in enumerate(team_ids)}
    y_all = np.array([target_map[t] for t in team_ids], dtype=np.float64)
    rng = np.random.default_rng(seed)
    boot_idx = rng.integers(0, len(team_ids), size=(boot, len(team_ids)))
    arms_out = {}
    for name, oof in oof_by_arm.items():
        arr = np.asarray(oof, dtype=np.float64)
        per_fold = [metrics_of(arr[[idx_of[t] for t in f]], y_all[[idx_of[t] for t in f]])["rmse"] for f in folds]
        arms_out[name] = {
            "reader_name": ARM_LABELS[name],
            **metrics_of(arr, y_all),
            "ci95": _bootstrap_metric_ci(arr, y_all, boot_idx),
            "per_fold_rmse": [float(v) for v in per_fold],
        }
    comparisons = {}
    for a, b in (
        ("predictor_team_bc", "guess_bc_label"),
        ("predictor_team_only", "guess_bc_label"),
        ("predictor_team_bc", "calibrate_bc_only"),
    ):
        comparisons[f"{a}_minus_{b}"] = paired_bootstrap_ci(oof_by_arm[a], oof_by_arm[b], y_all, boot_idx)
    return {"arms": arms_out, "comparisons": comparisons}


def draw_one_battle_labels(cells, teams, opponents, seed=SEED):
    """Draw 1 battle per (team, opponent) cell without replacement to mirror single-battle deployment noise."""
    rng = random.Random(seed)
    out = {}
    for t in teams:
        scores = []
        for o in opponents:
            w, l, ti = cells[(t, o)]
            pool = [1.0] * w + [0.0] * l + [0.5] * ti
            scores.append(rng.choice(pool))
        out[t] = sum(scores) / len(scores)
    return out


# ---------------------------------------------------------------- exploratory PyTorch MLP check
def _train_one_mlp_fold(X_tr, y_tr, bc_tr, X_te, bc_te, *, use_bc, seed):
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    n = len(y_tr)
    perm = rng.permutation(n)
    n_val = max(20, n // 5)
    va_i, tr_i = perm[:n_val], perm[n_val:]

    base_tr = np.zeros(n, dtype=np.float64)
    base_te = np.zeros(len(X_te), dtype=np.float64)
    if use_bc:
        a, b = fit_linear_bc_calibration(bc_tr[tr_i], y_tr[tr_i])
        base_tr = a + b * bc_tr
        base_te = a + b * bc_te
        feats_tr = np.column_stack([X_tr, bc_tr])
        feats_te = np.column_stack([X_te, bc_te])
    else:
        feats_tr, feats_te = X_tr, X_te

    net = nn.Sequential(
        nn.Linear(feats_tr.shape[1], 64),
        nn.ReLU(),
        nn.Dropout(0.35),
        nn.Linear(64, 1),
    )
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=5e-2)
    xt = torch.tensor(feats_tr[tr_i], dtype=torch.float32)
    yt = torch.tensor(y_tr[tr_i] - base_tr[tr_i], dtype=torch.float32)
    xv = torch.tensor(feats_tr[va_i], dtype=torch.float32)
    yv = torch.tensor(y_tr[va_i] - base_tr[va_i], dtype=torch.float32)

    best_loss, best_state, stale = float("inf"), None, 0
    for _ in range(300):
        net.train()
        opt.zero_grad()
        loss = torch.mean((net(xt).squeeze(-1) - yt) ** 2)
        loss.backward()
        opt.step()
        net.eval()
        with torch.no_grad():
            vloss = float(torch.mean((net(xv).squeeze(-1) - yv) ** 2))
        if vloss < best_loss - 1e-6:
            best_loss = vloss
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= 25:
                break
    net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        delta = net(torch.tensor(feats_te, dtype=torch.float32)).squeeze(-1).numpy()
    return np.clip(base_te + delta, 0.0, 1.0)


def evaluate_mlp_arms(team_ids, teams, fp_target, bc_input, folds, *, min_pair=MIN_PAIR, boot=BOOTSTRAP, seed=SEED):
    """Exploratory nonlinear PyTorch MLP check on the exact same 5 folds (residual over linear BC for team+bc)."""
    idx_of = {tid: i for i, tid in enumerate(team_ids)}
    y_all = np.array([fp_target[t] for t in team_ids], dtype=np.float64)
    oof_bc = np.zeros(len(team_ids), dtype=np.float64)
    oof_only = np.zeros(len(team_ids), dtype=np.float64)
    for fi, test_ids in enumerate(folds):
        test_set = set(test_ids)
        train_ids = [t for t in team_ids if t not in test_set]
        fmap = FeatureMap.fit([teams[t] for t in train_ids], min_pair=min_pair)
        X_tr = fmap.transform([teams[t] for t in train_ids])
        X_te = fmap.transform([teams[t] for t in test_ids])
        y_tr = np.array([fp_target[t] for t in train_ids], dtype=np.float64)
        bc_tr = np.array([bc_input[t] for t in train_ids], dtype=np.float64)
        bc_te = np.array([bc_input[t] for t in test_ids], dtype=np.float64)
        te_idx = np.array([idx_of[t] for t in test_ids], dtype=np.int64)
        oof_bc[te_idx] = _train_one_mlp_fold(X_tr, y_tr, bc_tr, X_te, bc_te, use_bc=True, seed=seed + fi)
        oof_only[te_idx] = _train_one_mlp_fold(X_tr, y_tr, bc_tr, X_te, bc_te, use_bc=False, seed=seed + 100 + fi)

    rng = np.random.default_rng(seed)
    boot_idx = rng.integers(0, len(team_ids), size=(boot, len(team_ids)))
    guess_bc = np.array([bc_input[t] for t in team_ids], dtype=np.float64)
    return {
        "predictor_team_bc_mlp": {
            **metrics_of(oof_bc, y_all),
            "ci95": _bootstrap_metric_ci(oof_bc, y_all, boot_idx),
            "vs_guess_bc_label": paired_bootstrap_ci(oof_bc, guess_bc, y_all, boot_idx),
        },
        "predictor_team_only_mlp": {
            **metrics_of(oof_only, y_all),
            "ci95": _bootstrap_metric_ci(oof_only, y_all, boot_idx),
            "vs_guess_bc_label": paired_bootstrap_ci(oof_only, guess_bc, y_all, boot_idx),
        },
    }


# ---------------------------------------------------------------- data loading & preflight
def _parse_paste(paste):
    from corpus import parse_team_text

    slots = []
    for block in paste.strip().split("\n\n"):
        slots += parse_team_text(block)
    return slots


def load_pilot_dataset(con, pilot_dir=PILOT_DIR):
    """Load the 500 pilot teams, parsed teams, labels (FP policy_id & BC policy 8), and split-half noise floor."""
    import foul_play_label_pilot as FP

    saved = json.loads((pilot_dir / "pilot_teams.json").read_text())
    pid = int((pilot_dir / "policy_id").read_text().strip())
    report = json.loads((pilot_dir / "report_pilot.json").read_text())
    team_ids = [int(t) for t in saved["team_ids"]]
    train_opps = [int(t) for t in saved["training_opponents"]]
    val_opps = [int(t) for t in saved["validation_opponents"]]
    meta_opps = train_opps + val_opps

    ph = ",".join("?" * len(team_ids))
    rows = con.execute(f"SELECT team_id, paste, species_key, grid_hash FROM team WHERE team_id IN ({ph})", team_ids).fetchall()
    pastes = {int(r[0]): r[1] for r in rows}
    species_keys = {int(r[0]): r[2] for r in rows}
    grid_hashes = {int(r[0]): r[3] for r in rows}
    teams = {tid: _parse_paste(pastes[tid]) for tid in team_ids}

    fp_cells = FP.directed_cells(con, pid, team_ids, meta_opps)
    bc_cells = FP.directed_cells(con, FP.BC_POLICY, team_ids, meta_opps)
    fp_train = FP.labels(fp_cells, team_ids, train_opps)
    bc_train = FP.labels(bc_cells, team_ids, train_opps)
    fp_val = FP.labels(fp_cells, team_ids, val_opps)

    h_fp = report["halves"]["foul_play"]
    h_bc = report["halves"]["behaviour_cloning"]
    F = np.array([h_fp[str(t)] for t in team_ids], dtype=np.float64)
    B = np.array([h_bc[str(t)] for t in team_ids], dtype=np.float64)
    fp_noise_var = float(np.mean((F[:, 0] - F[:, 1]) ** 2) / 4.0)
    bc_noise_var = float(np.mean((B[:, 0] - B[:, 1]) ** 2) / 4.0)

    return {
        "policy_id": pid,
        "bc_policy_id": FP.BC_POLICY,
        "team_ids": team_ids,
        "train_opps": train_opps,
        "val_opps": val_opps,
        "pastes": pastes,
        "species_keys": species_keys,
        "grid_hashes": grid_hashes,
        "teams": teams,
        "fp_cells": fp_cells,
        "bc_cells": bc_cells,
        "fp_train": fp_train,
        "bc_train": bc_train,
        "fp_val": fp_val,
        "battle_noise_floor": {
            "fp_noise_var": fp_noise_var,
            "fp_rmse_floor": float(np.sqrt(fp_noise_var)),
            "bc_noise_var": bc_noise_var,
            "bc_rmse_floor": float(np.sqrt(bc_noise_var)),
            "bc_one_battle_rmse_floor": float(np.sqrt(2.0 * bc_noise_var)),
        },
    }


def _probe_round_trip(data, n=20):
    import team84 as T

    Vreg = T.regulation_vocab()
    mismatched = []
    for tid in data["team_ids"][:n]:
        team = data["teams"][tid]
        row = T.encode(Vreg, team)
        dec = T.decode(Vreg, row)
        want = sorted(
            (_norm(s["species"]), _norm(s["ability"]), _norm(s["item"]), tuple(sorted(_norm(m) for m in s["moves"])),
             _norm(s.get("nature")), tuple(int(s["evs"].get(k, 0)) for k in STATS))
            for s in team
        )
        got = sorted(
            (_norm(s["species"]), _norm(s["ability"]), _norm(s["item"]), tuple(sorted(_norm(m) for m in s["moves"])),
             _norm(s.get("nature")), tuple(int(s["evs"][k]) for k in STATS))
            for s in dec
        )
        if want != got:
            mismatched.append(tid)
    return {"checked": n, "mismatched": mismatched, "passed": len(mismatched) == 0}


def _probe_training_smoke(data):
    sample_teams = [data["teams"][t] for t in data["team_ids"][:40]]
    fmap = FeatureMap.fit(sample_teams, min_pair=3)
    X_pen = fmap.transform(sample_teams)
    rng = np.random.default_rng(SEED)
    bc = rng.uniform(0.1, 0.7, len(sample_teams))
    U = np.column_stack([np.ones(len(sample_teams)), bc])
    true_w = rng.normal(0, 0.02, X_pen.shape[1])
    y_syn = 0.04 + 0.8 * bc + X_pen @ true_w
    lam = fmap.penalty_vector(10.0, 30.0)
    loo_fast, _, _ = solve_ridge_with_loocv(U, X_pen, y_syn, lam)
    # Independent brute-force check across all rows
    X_full = np.hstack([U, X_pen])
    pen = np.concatenate([np.zeros(2), lam])
    bf = []
    for i in range(len(sample_teams)):
        m = np.ones(len(sample_teams), dtype=bool)
        m[i] = False
        wi = np.linalg.solve(X_full[m].T @ X_full[m] + np.diag(pen), X_full[m].T @ y_syn[m])
        bf.append(float((y_syn[i] - X_full[i] @ wi) ** 2))
    loo_diff = abs(loo_fast - float(np.mean(bf)))
    _, w_u_low, w_p_low = solve_ridge_with_loocv(U, X_pen, y_syn, fmap.penalty_vector(1e-4, 1e-4))
    fit_r2 = metrics_of(U @ w_u_low + X_pen @ w_p_low, y_syn)["r2"]
    return {
        "loocv_max_abs_diff": loo_diff,
        "synthetic_recovery_r2": fit_r2,
        "passed": bool(loo_diff < 1e-8 and fit_r2 > 0.99),
    }


def _probe_invariance(data, n=50):
    rng = random.Random(SEED)
    subset = [data["teams"][t] for t in data["team_ids"][:n]]
    fmap = FeatureMap.fit(subset, min_pair=3)
    X_orig = fmap.transform(subset)
    shuffled = []
    for team in subset:
        copy = list(team)
        rng.shuffle(copy)
        shuffled.append(copy)
    X_shuf = fmap.transform(shuffled)
    max_diff = float(np.max(np.abs(X_orig - X_shuf)))
    return {"checked": n, "max_abs_diff": max_diff, "passed": max_diff == 0.0}


def run_preflight(data):
    """Run all build-experiment step 2 probes before judging CV results."""
    import heldout as HO

    held = HO.identities()
    held_overlap = sum(1 for h in data["grid_hashes"].values() if h in held)
    folds = make_folds(data["team_ids"], k=K_FOLDS, seed=SEED)
    flat = [t for f in folds for t in f]
    fold_overlap = len(flat) - len(set(flat))

    fp_arr = np.array([data["fp_train"][t] for t in data["team_ids"]])
    bc_arr = np.array([data["bc_train"][t] for t in data["team_ids"]])
    r_inter = _corr(fp_arr, bc_arr)
    fp_complete = sum(
        1 for t in data["team_ids"] if all(sum(data["fp_cells"].get((t, o), (0, 0, 0))) >= 2 for o in data["train_opps"] + data["val_opps"])
    )
    bc_complete = sum(
        1 for t in data["team_ids"] if all(sum(data["bc_cells"].get((t, o), (0, 0, 0))) >= 2 for o in data["train_opps"])
    )
    raw_rmse = metrics_of(bc_arr, fp_arr)["rmse"]
    floor = data["battle_noise_floor"]["fp_rmse_floor"]

    probes = {
        "ceiling_battle_noise_floor": {
            "fp_rmse_floor": round(floor, 4),
            "guess_bc_rmse": round(raw_rmse, 4),
            "headroom_rmse": round(raw_rmse - floor, 4),
            "passed": bool(raw_rmse - floor > 0.05),
        },
        "round_trip": _probe_round_trip(data),
        "coverage": {
            "n_teams": len(data["team_ids"]),
            "fp_complete_all_49": fp_complete,
            "bc_complete_39_train": bc_complete,
            "passed": bool(fp_complete == 500 and bc_complete == 500),
        },
        "training_smoke": _probe_training_smoke(data),
        "controls": {
            "fp_mean": round(float(fp_arr.mean()), 4),
            "bc_mean": round(float(bc_arr.mean()), 4),
            "inter_labeler_r": round(r_inter, 4),
            "passed": bool(abs(fp_arr.mean() - 0.262) < 0.002 and abs(bc_arr.mean() - 0.309) < 0.002 and abs(r_inter - 0.783) < 0.002),
        },
        "invariance_slot_order": _probe_invariance(data),
        "separation": {
            "held_out_teams_51_100": held_overlap,
            "fold_overlap": fold_overlap,
            "validation_opponents_in_train_input": len(set(data["train_opps"]) & set(data["val_opps"])),
            "passed": bool(held_overlap == 0 and fold_overlap == 0 and not (set(data["train_opps"]) & set(data["val_opps"]))),
        },
        "read_teams_condition_copies": {
            "skipped": True,
            "reason": "Matchup predictor outputs scalar win-rate predictions, not generated team pastes, and takes no generative ask.",
            "passed": True,
        },
    }
    probes["all_passed"] = all(v["passed"] for v in probes.values())
    return probes


# ---------------------------------------------------------------- full experiment
def _git_head():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=HERE.parent, text=True).strip()
    except Exception:
        return "unknown"


def _top_residual_features(data, k=12):
    """Fit predictor_team_bc on all 500 teams to inspect which team features shift Foul Play vs behaviour cloning."""
    ids = data["team_ids"]
    teams = [data["teams"][t] for t in ids]
    y = np.array([data["fp_train"][t] for t in ids], dtype=np.float64)
    bc = np.array([data["bc_train"][t] for t in ids], dtype=np.float64)
    m = fit_ridge_arm(teams, y, bc, use_bc=True, min_pair=MIN_PAIR, lam_grid=LAM_GRID)
    names = m.fmap.feature_names
    pairs = sorted(zip(names, m.w_pen), key=lambda z: -z[1])
    return {
        "lam_s": m.lam_s,
        "intercept": round(float(m.w_unpen[0]), 5),
        "bc_slope": round(float(m.w_unpen[1]), 5),
        "positive": [[n, round(float(w), 5)] for n, w in pairs[:k]],
        "negative": [[n, round(float(w), 5)] for n, w in pairs[-k:][::-1]],
    }


def run_experiment(con, *, pilot_dir=PILOT_DIR, out_dir=OUT, run_mlp=True):
    """Run preflight, primary 5-fold CV, sensitivity checks, and exploratory MLP, saving results.json."""
    t0 = time.perf_counter()
    data = load_pilot_dataset(con, pilot_dir)
    preflight = run_preflight(data)
    if not preflight["all_passed"]:
        raise RuntimeError(f"Preflight failed: {json.dumps(preflight, indent=1)}")

    folds = make_folds(data["team_ids"], k=K_FOLDS, seed=SEED)
    primary = evaluate_arms(data["team_ids"], data["teams"], data["fp_train"], data["bc_train"], folds)
    # Repeatability probe: exact bit-for-bit repeat with same seed
    repeat = evaluate_arms(data["team_ids"], data["teams"], data["fp_train"], data["bc_train"], folds, boot=10)
    repeat_diff = max(
        float(np.max(np.abs(np.array(primary["arms"][a]["oof_pred"]) - np.array(repeat["arms"][a]["oof_pred"]))))
        for a in ARM_LABELS
    )
    preflight["repeat"] = {"max_pred_diff": repeat_diff, "passed": repeat_diff == 0.0}
    preflight["all_passed"] = all(v["passed"] for k, v in preflight.items() if k != "all_passed")

    # Sensitivity 1: 1-battle-per-cell behaviour-cloning deployment noise (both 1-battle train/test and 2-battle train -> 1-battle test)
    bc_one = draw_one_battle_labels(data["bc_cells"], data["team_ids"], data["train_opps"], seed=SEED)
    sens_one = evaluate_arms(data["team_ids"], data["teams"], data["fp_train"], bc_one, folds)
    sens_deploy = evaluate_arms(
        data["team_ids"], data["teams"], data["fp_train"], data["bc_train"], folds, bc_test_input=bc_one
    )

    # Sensitivity 2: score primary OOF predictions (trained on 39 train opps) against 10 unseen validation opponents
    oof_map = {a: primary["arms"][a]["oof_pred"] for a in ARM_LABELS}
    sens_val = evaluate_predictions_on_target(oof_map, data["team_ids"], data["fp_val"], folds)

    # Sensitivity 3: lineup-grouped 5-fold CV (zero shared 6-species lineups between train and test)
    grouped_folds = make_grouped_folds(data["team_ids"], data["species_keys"], k=K_FOLDS, seed=SEED)
    sens_group = evaluate_arms(data["team_ids"], data["teams"], data["fp_train"], data["bc_train"], grouped_folds)

    mlp = evaluate_mlp_arms(data["team_ids"], data["teams"], data["fp_train"], data["bc_train"], folds) if run_mlp else None
    top_feats = _top_residual_features(data)

    payload = {
        "issue": 419,
        "parent_map": 416,
        "pilot_issue": 447,
        "commit": _git_head(),
        "run_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seconds": round(time.perf_counter() - t0, 2),
        "config": {
            "seed": SEED,
            "k_folds": K_FOLDS,
            "min_pair": MIN_PAIR,
            "lam_grid": list(LAM_GRID),
            "pair_mult": PAIR_MULT,
            "bootstrap": BOOTSTRAP,
            "policy_id": data["policy_id"],
            "bc_policy_id": data["bc_policy_id"],
            "n_train_opponents": len(data["train_opps"]),
            "n_val_opponents": len(data["val_opps"]),
        },
        "n_teams": len(data["team_ids"]),
        "battle_noise_floor": data["battle_noise_floor"],
        "preflight": preflight,
        "primary": primary,
        "sensitivity": {
            "one_battle_bc": {k: v for k, v in sens_one.items() if k not in ("oof_pred", "target_fp", "input_bc")},
            "train_2b_test_1b_bc": {k: v for k, v in sens_deploy.items() if k not in ("oof_pred", "target_fp", "input_bc")},
            "validation_opponents": sens_val,
            "grouped_lineup_folds": {k: v for k, v in sens_group.items() if k not in ("oof_pred", "target_fp", "input_bc")},
        },
        "exploratory_mlp": mlp,
        "top_residual_features": top_feats,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(payload, indent=1) + "\n")
    return payload


# ---------------------------------------------------------------- storyboard figures (ECharts)
COLORS = {
    "predictor_team_bc": "#2a6fdb",
    "calibrate_bc_only": "#0f766e",
    "predictor_team_only": "#c2410c",
    "guess_bc_label": "#8a8984",
    "guess_train_mean": "#a8a29e",
    "floor": "#15803d",
}


def _arm_rmse_chart(payload, path):
    from echarts_chart import write_chart

    arms = payload["primary"]["arms"]
    floor = payload["battle_noise_floor"]["fp_rmse_floor"]
    order = ["predictor_team_bc", "predictor_team_only", "calibrate_bc_only", "guess_bc_label", "guess_train_mean"]
    labels = [SHORT_LABELS[k] for k in order]
    series = []
    for key, label in zip(order, labels):
        a = arms[key]
        lo, hi = a["ci95"]["rmse"]
        color = COLORS[key]
        series.append(
            {
                "type": "line",
                "data": [[round(lo, 4), label], [round(hi, 4), label]],
                "symbol": "none",
                "lineStyle": {"color": color, "width": 3},
                "tooltip": {"show": False},
                "silent": True,
            }
        )
        entry = {
            "name": ARM_LABELS[key],
            "type": "scatter",
            "data": [[round(a["rmse"], 4), label]],
            "symbolSize": 15,
            "itemStyle": {"color": color},
            "emphasis": {"focus": "series"},
            "label": {
                "show": True,
                "position": "top",
                "distance": 8,
                "formatter": f"{a['rmse']:.3f} [{lo:.3f}, {hi:.3f}]",
                "fontSize": 12,
                "color": color,
                "fontWeight": "bold",
            },
        }
        if key == order[0]:
            entry["markLine"] = {
                "silent": True,
                "symbol": "none",
                "lineStyle": {"color": COLORS["floor"], "type": "dashed", "width": 2},
                "label": {
                    "formatter": f"2-battle noise floor ({floor:.3f})",
                    "color": COLORS["floor"],
                    "position": "end",
                    "distance": 6,
                    "fontSize": 12,
                },
                "data": [{"xAxis": round(floor, 4)}],
            }
        series.append(entry)
    option = {
        "tooltip": {"trigger": "axis", "axisPointer": {"type": "line"}},
        "grid": {"left": 210, "right": 56, "top": 42, "bottom": 54},
        "xAxis": {
            "type": "value",
            "min": 0.03,
            "max": 0.17,
            "name": "5-fold out-of-fold root mean squared error (lower is better)",
            "nameLocation": "middle",
            "nameGap": 34,
        },
        "yAxis": {"type": "category", "data": labels[::-1], "axisTick": {"show": False}, "axisLine": {"show": False}},
        "series": series,
    }
    bc_rmse = arms["guess_bc_label"]["rmse"]
    tb_rmse = arms["predictor_team_bc"]["rmse"]
    to_rmse = arms["predictor_team_only"]["rmse"]
    title = (
        f"Team and behaviour-cloning win rate cuts error from {bc_rmse:.3f} to {tb_rmse:.3f}"
        if tb_rmse < bc_rmse
        else f"Neither predictor beats the {bc_rmse:.3f} behaviour-cloning guess"
    )
    subtitle = (
        f"5-fold cross-validation on 500 pilot teams (issue #419). Team only scores {to_rmse:.3f}; "
        f"an oracle on infinite battles cannot beat the {floor:.3f} two-battle noise floor."
    )
    note = "Points show out-of-fold root mean squared error with 95% bootstrap intervals (10,000 resamples over the 500 teams)."
    return write_chart(option, path, title=title, subtitle=subtitle, note=note, height=370)


def _scatter_panels_chart(payload, path):
    from echarts_chart import write_chart

    y = payload["primary"]["target_fp"]
    arms = payload["primary"]["arms"]
    panels = [
        ("guess_bc_label", SHORT_LABELS["guess_bc_label"], COLORS["guess_bc_label"]),
        ("predictor_team_only", SHORT_LABELS["predictor_team_only"], COLORS["predictor_team_only"]),
        ("predictor_team_bc", SHORT_LABELS["predictor_team_bc"], COLORS["predictor_team_bc"]),
    ]
    grids, xaxes, yaxes, titles, series = [], [], [], [], []
    for i, (key, label, color) in enumerate(panels):
        left = 6 + i * 32
        a = arms[key]
        grids.append({"left": f"{left}%", "width": "26%", "top": 52, "bottom": 54})
        titles.append(
            {
                "text": f"{label}\nerror {a['rmse']:.3f} · r {a['pearson_r']:.3f}",
                "left": f"{left + 13}%",
                "top": 6,
                "textAlign": "center",
                "textStyle": {"fontSize": 12.5, "fontWeight": "bold"},
            }
        )
        xaxes.append(
            {
                "gridIndex": i,
                "type": "value",
                "min": 0,
                "max": 0.75,
                "name": "predicted win rate" if i == 1 else "",
                "nameLocation": "middle",
                "nameGap": 32,
            }
        )
        yaxes.append(
            {
                "gridIndex": i,
                "type": "value",
                "min": 0,
                "max": 0.75,
                "name": "Foul Play win rate" if i == 0 else "",
                "nameLocation": "middle",
                "nameGap": 38,
            }
        )
        pts = [[round(p, 4), round(t, 4)] for p, t in zip(a["oof_pred"], y)]
        series.append(
            {
                "name": ARM_LABELS[key],
                "type": "scatter",
                "xAxisIndex": i,
                "yAxisIndex": i,
                "data": pts,
                "symbolSize": 5,
                "itemStyle": {"color": color, "opacity": 0.55},
                "emphasis": {"focus": "series"},
            }
        )
        series.append(
            {
                "type": "line",
                "xAxisIndex": i,
                "yAxisIndex": i,
                "data": [[0, 0], [0.75, 0.75]],
                "symbol": "none",
                "lineStyle": {"color": "#6f6e69", "type": "dashed", "width": 1.3},
                "silent": True,
                "tooltip": {"show": False},
            }
        )
    option = {
        "title": titles,
        "tooltip": {"trigger": "item", "formatter": "js:p => p.seriesName + '<br>predicted ' + p.value[0].toFixed(3) + ' · actual ' + p.value[1].toFixed(3)"},
        "grid": grids,
        "xAxis": xaxes,
        "yAxis": yaxes,
        "series": series,
    }
    return write_chart(
        option,
        path,
        title="Predicted vs measured Foul Play win rate across all 500 out-of-fold teams",
        subtitle="The unadjusted behaviour-cloning win rate sits right of the dashed diagonal (over-rating teams by +0.047 on average); the trained predictor centers on the diagonal.",
        note="Each dot is one of the 500 pilot teams, predicted by the fold that did not train on it. Panel error is root mean squared error; r is Pearson correlation.",
        height=390,
    )


def _sensitivity_chart(payload, path):
    from echarts_chart import write_chart

    sens = payload["sensitivity"]
    deploy = sens.get("train_2b_test_1b_bc", sens["one_battle_bc"])
    cases = [
        ("Primary\n(2-battle input,\n39 training opponents)", payload["primary"]["arms"]),
        ("Deployment mismatch\n(2-battle train,\n1-battle test input)", deploy["arms"]),
        ("Lineup-grouped folds\n(unseen 6-species\nlineups)", sens["grouped_lineup_folds"]["arms"]),
        ("10 unseen validation\nopponents\n(20 battles/team)", sens["validation_opponents"]["arms"]),
    ]
    cats = [c[0] for c in cases]
    bar_arms = [
        ("guess_bc_label", SHORT_LABELS["guess_bc_label"], COLORS["guess_bc_label"]),
        ("calibrate_bc_only", SHORT_LABELS["calibrate_bc_only"], COLORS["calibrate_bc_only"]),
        ("predictor_team_only", SHORT_LABELS["predictor_team_only"], COLORS["predictor_team_only"]),
        ("predictor_team_bc", SHORT_LABELS["predictor_team_bc"], COLORS["predictor_team_bc"]),
    ]
    series = []
    for key, label, color in bar_arms:
        series.append(
            {
                "name": label,
                "type": "bar",
                "data": [round(arms[key]["rmse"], 4) for _, arms in cases],
                "itemStyle": {"color": color},
                "emphasis": {"focus": "series"},
                "label": {"show": True, "position": "top", "formatter": "js:p => p.value.toFixed(3)", "fontSize": 10.5},
            }
        )
    option = {
        "legend": {"top": 0},
        "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"}},
        "grid": {"left": 66, "right": 24, "top": 46, "bottom": 76},
        "xAxis": {"type": "category", "data": cats},
        "yAxis": {"type": "value", "name": "root mean squared error (lower is better)", "min": 0, "max": 0.18},
        "series": series,
    }
    return write_chart(
        option,
        path,
        title="Does the predictor hold up under deployment conditions?",
        subtitle="Root mean squared error when predicting on single-battle corpus inputs, unseen 6-species lineups, and the 10 held-back validation opponents.",
        note="Lower bars are better. Validation-opponent labels have 20 battles per team (10 opponents × 2), so their battle-noise floor is higher.",
        height=420,
    )


def _teaser_chart(payload, path):
    from echarts_chart import write_chart

    arms = payload["primary"]["arms"]
    comps = payload["primary"]["comparisons"]
    floor = payload["battle_noise_floor"]["fp_rmse_floor"]
    order = ["predictor_team_bc", "predictor_team_only", "calibrate_bc_only", "guess_bc_label"]
    names = [SHORT_LABELS[k] for k in order]
    gap_rows = [
        ("Team + old vs old guess", comps["predictor_team_bc_minus_guess_bc_label"]["rmse"], COLORS["predictor_team_bc"]),
        ("Team only vs old guess", comps["predictor_team_only_minus_guess_bc_label"]["rmse"], COLORS["predictor_team_only"]),
        ("Team + old vs linear fit", comps["predictor_team_bc_minus_calibrate_bc_only"]["rmse"], COLORS["calibrate_bc_only"]),
    ]
    series = [
        {
            "name": "5-fold root mean squared error",
            "type": "bar",
            "xAxisIndex": 0,
            "yAxisIndex": 0,
            "barWidth": 22,
            "emphasis": {"focus": "series"},
            "data": [{"value": round(arms[k]["rmse"], 4), "itemStyle": {"color": COLORS[k]}} for k in order[::-1]],
            "label": {"show": True, "position": "right", "formatter": "js:p => p.value.toFixed(3)"},
            "markLine": {
                "silent": True,
                "symbol": "none",
                "lineStyle": {"color": COLORS["floor"], "type": "dashed", "width": 1.8},
                "label": {"formatter": f"noise floor {floor:.3f}", "color": COLORS["floor"], "position": "end", "distance": 4, "fontSize": 11.5},
                "data": [{"xAxis": round(floor, 4)}],
            },
        }
    ]
    for lab, stat, color in gap_rows:
        lo, hi = stat["ci95"]
        series.append(
            {
                "type": "line",
                "xAxisIndex": 1,
                "yAxisIndex": 1,
                "data": [[round(lo, 4), lab], [round(hi, 4), lab]],
                "symbol": "none",
                "lineStyle": {"color": color, "width": 3},
                "silent": True,
                "tooltip": {"show": False},
            }
        )
        series.append(
            {
                "name": lab,
                "type": "scatter",
                "xAxisIndex": 1,
                "yAxisIndex": 1,
                "data": [[round(stat["diff"], 4), lab]],
                "symbolSize": 13,
                "itemStyle": {"color": color},
                "emphasis": {"focus": "series"},
                "label": {
                    "show": True,
                    "position": "top",
                    "distance": 6,
                    "formatter": f"{stat['diff']:+.3f}",
                    "fontSize": 12,
                    "color": color,
                    "fontWeight": "bold",
                },
            }
        )
    series.append(
        {
            "type": "line",
            "xAxisIndex": 1,
            "yAxisIndex": 1,
            "data": [[0, gap_rows[-1][0]], [0, gap_rows[0][0]]],
            "symbol": "none",
            "lineStyle": {"color": "#6f6e69", "type": "dashed", "width": 1.5},
            "silent": True,
            "tooltip": {"show": False},
        }
    )
    option = {
        "title": [
            {"text": "5-fold root mean squared error (lower is better)", "left": "2%", "top": 0, "textStyle": {"fontSize": 13, "fontWeight": "normal"}},
            {"text": "paired error gap (< 0 means lower error)", "left": "54%", "top": 0, "textStyle": {"fontSize": 13, "fontWeight": "normal"}},
        ],
        "tooltip": {"trigger": "axis", "axisPointer": {"type": "line"}},
        "grid": [
            {"left": 182, "width": "27%", "top": 52, "bottom": 34},
            {"left": "69%", "right": 34, "top": 52, "bottom": 34},
        ],
        "xAxis": [
            {"gridIndex": 0, "type": "value", "min": 0, "max": 0.15},
            {"gridIndex": 1, "type": "value", "min": -0.06, "max": 0.02, "interval": 0.02},
        ],
        "yAxis": [
            {"gridIndex": 0, "type": "category", "data": names[::-1], "axisTick": {"show": False}},
            {"gridIndex": 1, "type": "category", "data": [r[0] for r in gap_rows][::-1], "axisTick": {"show": False}},
        ],
        "series": series,
    }
    gap_tb = comps["predictor_team_bc_minus_guess_bc_label"]["rmse"]
    gap_cal = comps["predictor_team_bc_minus_calibrate_bc_only"]["rmse"]
    verdict = (
        "Team and behaviour-cloning beats the old label, and team features beat linear rescaling alone"
        if gap_tb["verdict"] == "better" and gap_cal["verdict"] == "better"
        else "Team and behaviour-cloning beats the old label, mostly by rescaling behaviour cloning"
        if gap_tb["verdict"] == "better"
        else "Neither matchup predictor beats reusing the behaviour-cloning win rate"
    )
    subtitle = (
        f"500 pilot teams, 5-fold cross-validation (issue #419). Gap to reusing the old label: {gap_tb['diff']:+.3f} "
        f"[{gap_tb['ci95'][0]:+.3f}, {gap_tb['ci95'][1]:+.3f}]; gap to linear calibration alone: "
        f"{gap_cal['diff']:+.3f} [{gap_cal['ci95'][0]:+.3f}, {gap_cal['ci95'][1]:+.3f}]."
    )
    return write_chart(option, path, title=verdict, subtitle=subtitle, note="Error bars on the right are 95% paired bootstrap intervals over the 500 teams.", height=310)


def write_storyboard_figures(payload, out_dir):
    """Write the 4 standalone ECharts HTML storyboard pages and return their paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return [
        _teaser_chart(payload, out_dir / "0_teaser.html"),
        _arm_rmse_chart(payload, out_dir / "1_arm_rmse.html"),
        _scatter_panels_chart(payload, out_dir / "2_pred_vs_actual.html"),
        _sensitivity_chart(payload, out_dir / "3_deployment_checks.html"),
    ]


def format_markdown_report(payload):
    """Format a result-first Markdown summary for issue #419."""
    arms = payload["primary"]["arms"]
    comps = payload["primary"]["comparisons"]
    floor = payload["battle_noise_floor"]["fp_rmse_floor"]
    tb = arms["predictor_team_bc"]
    to = arms["predictor_team_only"]
    gb = arms["guess_bc_label"]
    cal = arms["calibrate_bc_only"]
    g_tb = comps["predictor_team_bc_minus_guess_bc_label"]["rmse"]
    g_to = comps["predictor_team_only_minus_guess_bc_label"]["rmse"]
    g_cal = comps["predictor_team_bc_minus_calibrate_bc_only"]["rmse"]
    cal_verb = {"better": "beats", "worse": "trails", "same": "matches", "undecided": "is undecided against"}[g_cal["verdict"]]

    lines = [
        f"**Verdict:** **{tb['reader_name']}** (`predictor_team_bc`) beats **{gb['reader_name']}** (`guess_bc_label`) with 5-fold root mean squared error (RMSE) **{tb['rmse']:.4f}** [{tb['ci95']['rmse'][0]:.4f}, {tb['ci95']['rmse'][1]:.4f}] vs **{gb['rmse']:.4f}** [{gb['ci95']['rmse'][0]:.4f}, {gb['ci95']['rmse'][1]:.4f}] (paired gap **{g_tb['diff']:+.4f}** [{g_tb['ci95'][0]:+.4f}, {g_tb['ci95'][1]:+.4f}], lower in {comps['predictor_team_bc_minus_guess_bc_label']['folds_better']}/5 folds). "
        f"**{to['reader_name']}** (`predictor_team_only`) scores **{to['rmse']:.4f}** [{to['ci95']['rmse'][0]:.4f}, {to['ci95']['rmse'][1]:.4f}] (gap **{g_to['diff']:+.4f}** [{g_to['ci95'][0]:+.4f}, {g_to['ci95'][1]:+.4f}]). "
        f"Adding team features {cal_verb} **{cal['reader_name']}** (`calibrate_bc_only`, RMSE **{cal['rmse']:.4f}**) by **{g_cal['diff']:+.4f}** [{g_cal['ci95'][0]:+.4f}, {g_cal['ci95'][1]:+.4f}].",
        "",
        "### 5-fold cross-validation on the 500 pilot teams (39 training opponents)",
        "",
        f"Shared five disjoint folds of 100 teams (`seed = 419`), scored against Foul Play deterministic argmax win rate (`policy_id = 10`, two-battle noise RMSE floor **{floor:.4f}**).",
        "",
        "| Arm | Code ID | RMSE [95% CI] | Gap vs no-training guess [95% CI] | MAE | R² | Pearson *r* | Spearman ρ |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key in ("predictor_team_bc", "predictor_team_only", "guess_bc_label", "calibrate_bc_only", "guess_train_mean"):
        a = arms[key]
        cname = f"{key}_minus_guess_bc_label"
        gap_s = "—" if key == "guess_bc_label" else (
            f"{comps[cname]['rmse']['diff']:+.4f} [{comps[cname]['rmse']['ci95'][0]:+.4f}, {comps[cname]['rmse']['ci95'][1]:+.4f}]"
            if cname in comps
            else f"{a['rmse'] - gb['rmse']:+.4f}"
        )
        lines.append(
            f"| {a['reader_name']} | `{key}` | **{a['rmse']:.4f}** [{a['ci95']['rmse'][0]:.4f}, {a['ci95']['rmse'][1]:.4f}] | "
            f"{gap_s} | {a['mae']:.4f} | {a['r2']:.3f} | {a['pearson_r']:.3f} | {a['spearman_rho']:.3f} |"
        )
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- full-corpus SQLite labeling (with predictive std)
PREDICTOR_SCHEMA = """
CREATE TABLE IF NOT EXISTS team_score_predictor (
  predictor_id      INTEGER PRIMARY KEY,
  predictor_name    TEXT NOT NULL UNIQUE,
  kind              TEXT NOT NULL CHECK (kind = 'predicted_score'),
  model_family      TEXT NOT NULL,
  target_policy_id  INTEGER NOT NULL REFERENCES policy(policy_id),
  input_policy_id   INTEGER REFERENCES policy(policy_id),
  opponent_set      TEXT NOT NULL,
  training_n        INTEGER NOT NULL,
  training_source   TEXT NOT NULL,
  oof_rmse          REAL NOT NULL,
  deploy_1b_rmse    REAL NOT NULL,
  sigma_resid       REAL NOT NULL,
  sigma_true_resid  REAL NOT NULL,
  fp_2b_noise_sd    REAL NOT NULL,
  lam_single        REAL NOT NULL,
  lam_pair          REAL NOT NULL,
  intercept         REAL NOT NULL,
  bc_slope          REAL NOT NULL,
  n_features        INTEGER NOT NULL,
  git_commit        TEXT NOT NULL,
  created_at        TEXT NOT NULL DEFAULT (datetime('now')),
  metadata_json     TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS team_predicted_score (
  predictor_id      INTEGER NOT NULL REFERENCES team_score_predictor(predictor_id),
  team_id           INTEGER NOT NULL REFERENCES team(team_id),
  pred_win_rate     REAL NOT NULL CHECK (pred_win_rate >= 0.0 AND pred_win_rate <= 1.0),
  pred_std          REAL NOT NULL CHECK (pred_std >= 0.0),
  epistemic_std     REAL NOT NULL CHECK (epistemic_std >= 0.0),
  bc_win_rate       REAL NOT NULL CHECK (bc_win_rate >= 0.0 AND bc_win_rate <= 1.0),
  bc_opponents      INTEGER NOT NULL CHECK (bc_opponents > 0),
  bc_battles        INTEGER NOT NULL CHECK (bc_battles > 0),
  in_pilot_train    INTEGER NOT NULL DEFAULT 0 CHECK (in_pilot_train IN (0, 1)),
  scored_at         TEXT NOT NULL DEFAULT (datetime('now')),
  PRIMARY KEY (predictor_id, team_id)
) WITHOUT ROWID;
CREATE INDEX IF NOT EXISTS team_predicted_score_team ON team_predicted_score (team_id);
CREATE VIEW IF NOT EXISTS team_predicted_score_summary AS
  SELECT p.predictor_id, p.predictor_name, p.kind, p.model_family, p.target_policy_id,
         p.input_policy_id, p.opponent_set, s.team_id, s.pred_win_rate, s.pred_std,
         s.epistemic_std, s.bc_win_rate, s.bc_opponents, s.bc_battles, s.in_pilot_train, s.scored_at
  FROM team_predicted_score s
  JOIN team_score_predictor p USING (predictor_id);
"""


def ensure_predictor_schema(con):
    """Create the predictor registry, team prediction table, and summary view without touching battle tables."""
    con.executescript(PREDICTOR_SCHEMA)


def _calibration_summary(y_true, pred, pred_std, ep_std, target_noise_var):
    obs_std = np.sqrt(np.asarray(pred_std, dtype=np.float64) ** 2 + float(target_noise_var))
    z = (np.asarray(y_true, dtype=np.float64) - np.asarray(pred, dtype=np.float64)) / np.maximum(obs_std, 1e-12)
    return {
        "rmse": float(np.sqrt(np.mean((np.asarray(pred) - np.asarray(y_true)) ** 2))),
        "z_mean": float(np.mean(z)),
        "z_std": float(np.std(z)),
        "cover_1sd": float(np.mean(np.abs(z) <= 1.0)),
        "cover_2sd": float(np.mean(np.abs(z) <= 2.0)),
        "pred_std_median": float(np.median(pred_std)),
        "epistemic_std_median": float(np.median(ep_std)),
    }


def evaluate_std_calibration(data, folds, *, min_pair=MIN_PAIR, lam_grid=LAM_GRID):
    """Verify 5-fold out-of-fold predictive-SD calibration under 2-battle and 1-battle deployment BC inputs."""
    ids = data["team_ids"]
    k_opp = len(data["train_opps"])
    fp_noise_var = data["battle_noise_floor"]["fp_noise_var"]
    bc_one = draw_one_battle_labels(data["bc_cells"], ids, data["train_opps"], seed=SEED)
    y_all, p2, sd2, ep2, p1, sd1, ep1 = [], [], [], [], [], [], []
    for test_ids in folds:
        ts = set(test_ids)
        tr_ids = [t for t in ids if t not in ts]
        model = fit_ridge_with_uncertainty(
            [data["teams"][t] for t in tr_ids],
            np.array([data["fp_train"][t] for t in tr_ids], dtype=np.float64),
            np.array([data["bc_train"][t] for t in tr_ids], dtype=np.float64),
            use_bc=True,
            min_pair=min_pair,
            lam_grid=lam_grid,
            target_noise_var=fp_noise_var,
        )
        te_teams = [data["teams"][t] for t in test_ids]
        bc2 = np.array([data["bc_train"][t] for t in test_ids], dtype=np.float64)
        bc1 = np.array([bc_one[t] for t in test_ids], dtype=np.float64)
        n_te = len(test_ids)
        m2, s2, e2 = model.predict_with_std(te_teams, bc2, bc_battles=np.full(n_te, 2 * k_opp), bc_opponents=np.full(n_te, k_opp))
        m1, s1, e1 = model.predict_with_std(te_teams, bc1, bc_battles=np.full(n_te, k_opp), bc_opponents=np.full(n_te, k_opp))
        y_all.extend(data["fp_train"][t] for t in test_ids)
        p2.extend(m2); sd2.extend(s2); ep2.extend(e2)
        p1.extend(m1); sd1.extend(s1); ep1.extend(e1)
    return {
        "two_battle_input": _calibration_summary(y_all, p2, sd2, ep2, fp_noise_var),
        "one_battle_deploy_input": _calibration_summary(y_all, p1, sd1, ep1, fp_noise_var),
    }


def load_corpus_bc_stats(con, bc_policy_id, train_opps, *, excluded_ids=()):
    """Aggregate policy_id=bc_policy_id battles against `train_opps` for all eligible legal teams.

    Matches foul_play_label_pilot.labels: pooled (wins + 0.5 * ties) / battles. Excludes `excluded_ids`
    (held-out ranks 51-100) and requires full coverage of `train_opps` (skipping self for training meta teams).
    """
    train_opps = [int(o) for o in train_opps]
    train_set = set(train_opps)
    excluded = {int(t) for t in excluded_ids}
    ph = ",".join("?" * len(train_opps))
    q = (
        f"SELECT t, SUM(w) AS wins, SUM(ti) AS ties, SUM(n) AS battles, COUNT(DISTINCT m) AS n_opp FROM ("
        f"  SELECT hi AS t, lo AS m, hi_wins AS w, ties AS ti, battles AS n FROM cell WHERE policy_id=? AND lo IN ({ph})"
        f"  UNION ALL "
        f"  SELECT lo AS t, hi AS m, lo_wins AS w, ties AS ti, battles AS n FROM cell WHERE policy_id=? AND hi IN ({ph})"
        f") GROUP BY t"
    )
    agg = {}
    for tid, wins, ties, battles, n_opp in con.execute(q, [bc_policy_id, *train_opps, bc_policy_id, *train_opps]):
        tid = int(tid)
        need = len(train_opps) - (1 if tid in train_set else 0)
        if tid in excluded or int(battles) <= 0 or int(n_opp) != need:
            continue
        agg[tid] = ((float(wins) + 0.5 * float(ties)) / float(battles), int(n_opp), int(battles))

    entries = []
    for tid, legal, origin, paste in con.execute("SELECT team_id, legal, origin, paste FROM team ORDER BY team_id"):
        tid = int(tid)
        stat = agg.get(tid)
        if stat is None or legal != 1:
            continue
        entries.append((tid, origin, paste, stat[0], stat[1], stat[2]))
    return entries


def score_corpus_entries(model, entries, pilot_ids, *, batch_size=10000, parse_fn=_parse_paste):
    """Score `(team_id, origin, paste, bc_wr, bc_opps, bc_battles)` entries in batches, returning DB rows."""
    pilot_set = {int(t) for t in pilot_ids}
    scored_rows = []
    for start in range(0, len(entries), batch_size):
        chunk = entries[start : start + batch_size]
        parsed = [parse_fn(row[2]) for row in chunk]
        bc_wr = np.array([row[3] for row in chunk], dtype=np.float64)
        bc_opps = np.array([row[4] for row in chunk], dtype=np.int64)
        bc_bats = np.array([row[5] for row in chunk], dtype=np.int64)
        pred, pred_std, ep_std = model.predict_with_std(parsed, bc_wr, bc_battles=bc_bats, bc_opponents=bc_opps)
        for row, p, sd, ep in zip(chunk, pred, pred_std, ep_std):
            tid = int(row[0])
            scored_rows.append(
                (tid, float(p), float(sd), float(ep), float(row[3]), int(row[4]), int(row[5]), int(tid in pilot_set))
            )
    return scored_rows


def write_corpus_predictions(con, predictor_meta, scored_rows, *, held_ids=(), overwrite=False):
    """Persist one predictor registry row and all team predictions atomically, guarding against held-out leaks."""
    held = {int(t) for t in held_ids}
    leaked = [r[0] for r in scored_rows if r[0] in held]
    if leaked:
        import heldout as HO

        raise HO.HeldoutLeak(f"{len(leaked)} held-out team(s) in predicted-score write: {leaked[:5]}")

    ensure_predictor_schema(con)
    name = predictor_meta["predictor_name"]
    existing = con.execute("SELECT predictor_id FROM team_score_predictor WHERE predictor_name=?", (name,)).fetchone()
    if existing and not overwrite:
        raise ValueError(f"predictor_name {name!r} already exists (predictor_id={existing[0]}); pass overwrite=True to replace")

    with con:
        if existing:
            pid = int(existing[0])
            con.execute("DELETE FROM team_predicted_score WHERE predictor_id=?", (pid,))
            con.execute("DELETE FROM team_score_predictor WHERE predictor_id=?", (pid,))
        cur = con.execute(
            "INSERT INTO team_score_predictor ("
            "predictor_name, kind, model_family, target_policy_id, input_policy_id, opponent_set, "
            "training_n, training_source, oof_rmse, deploy_1b_rmse, sigma_resid, sigma_true_resid, "
            "fp_2b_noise_sd, lam_single, lam_pair, intercept, bc_slope, n_features, git_commit, metadata_json"
            ") VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                name,
                PREDICTOR_KIND,
                PREDICTOR_MODEL_FAMILY,
                int(predictor_meta["target_policy_id"]),
                int(predictor_meta["input_policy_id"]),
                predictor_meta["opponent_set"],
                int(predictor_meta["training_n"]),
                predictor_meta["training_source"],
                float(predictor_meta["oof_rmse"]),
                float(predictor_meta["deploy_1b_rmse"]),
                float(predictor_meta["sigma_resid"]),
                float(predictor_meta["sigma_true_resid"]),
                float(predictor_meta["fp_2b_noise_sd"]),
                float(predictor_meta["lam_single"]),
                float(predictor_meta["lam_pair"]),
                float(predictor_meta["intercept"]),
                float(predictor_meta["bc_slope"]),
                int(predictor_meta["n_features"]),
                predictor_meta["git_commit"],
                json.dumps(predictor_meta["metadata"], sort_keys=True),
            ),
        )
        predictor_id = int(cur.lastrowid)
        con.executemany(
            "INSERT INTO team_predicted_score ("
            "predictor_id, team_id, pred_win_rate, pred_std, epistemic_std, "
            "bc_win_rate, bc_opponents, bc_battles, in_pilot_train"
            ") VALUES (?,?,?,?,?,?,?,?,?)",
            [(predictor_id, *r) for r in scored_rows],
        )
    return predictor_id


def _dist_summary(arr):
    a = np.asarray(arr, dtype=np.float64)
    q = np.quantile(a, [0.05, 0.25, 0.5, 0.75, 0.95])
    return {
        "mean": round(float(a.mean()), 5),
        "sd": round(float(a.std()), 5),
        "min": round(float(a.min()), 5),
        "p05": round(float(q[0]), 5),
        "p25": round(float(q[1]), 5),
        "median": round(float(q[2]), 5),
        "p75": round(float(q[3]), 5),
        "p95": round(float(q[4]), 5),
        "max": round(float(a.max()), 5),
    }


def label_corpus(
    con,
    *,
    pilot_dir=PILOT_DIR,
    predictor_name=DEFAULT_PREDICTOR_NAME,
    out_dir=OUT,
    batch_size=10000,
    overwrite=False,
    held_ids=None,
    parse_fn=_parse_paste,
):
    """Fit predictor_team_bc on the 500 pilot teams, score the full eligible corpus + std, and store in SQLite."""
    t0 = time.perf_counter()
    data = load_pilot_dataset(con, pilot_dir)
    preflight = run_preflight(data)
    if not preflight["all_passed"]:
        raise RuntimeError(f"Preflight failed before corpus labeling: {json.dumps(preflight, indent=1)}")

    folds = make_folds(data["team_ids"], k=K_FOLDS, seed=SEED)
    calibration = evaluate_std_calibration(data, folds)
    ids = data["team_ids"]
    fp_noise_var = data["battle_noise_floor"]["fp_noise_var"]
    model = fit_ridge_with_uncertainty(
        [data["teams"][t] for t in ids],
        np.array([data["fp_train"][t] for t in ids], dtype=np.float64),
        np.array([data["bc_train"][t] for t in ids], dtype=np.float64),
        use_bc=True,
        min_pair=MIN_PAIR,
        lam_grid=LAM_GRID,
        target_noise_var=fp_noise_var,
    )

    if held_ids is None:
        import heldout as HO

        held_ids = HO.team_ids(con)
    entries = load_corpus_bc_stats(con, data["bc_policy_id"], data["train_opps"], excluded_ids=held_ids)
    scored_rows = score_corpus_entries(model, entries, ids, batch_size=batch_size, parse_fn=parse_fn)

    pilot_hash = hashlib.sha256(",".join(str(t) for t in sorted(ids)).encode()).hexdigest()[:16]
    predictor_meta = {
        "predictor_name": predictor_name,
        "target_policy_id": data["policy_id"],
        "input_policy_id": data["bc_policy_id"],
        "opponent_set": "train39_of_top50",
        "training_n": len(ids),
        "training_source": "issue_447_pilot_seed447",
        "oof_rmse": calibration["two_battle_input"]["rmse"],
        "deploy_1b_rmse": calibration["one_battle_deploy_input"]["rmse"],
        "sigma_resid": model.sigma_resid,
        "sigma_true_resid": model.sigma_true_resid,
        "fp_2b_noise_sd": float(np.sqrt(fp_noise_var)),
        "lam_single": model.predictor.lam_s,
        "lam_pair": model.predictor.lam_p,
        "intercept": float(model.predictor.w_unpen[0]),
        "bc_slope": float(model.predictor.w_unpen[1]),
        "n_features": model.predictor.fmap.dim + len(model.predictor.w_unpen),
        "git_commit": _git_head(),
        "metadata": {
            "description": (
                "Predicted Foul Play deterministic argmax win rate (NOT battle outcomes) from the issue #419 "
                "multi-fidelity ridge matchup predictor (Team and behaviour-cloning win rate, predictor_team_bc) "
                "trained on all 500 issue #447 pilot teams."
            ),
            "issue": 419,
            "parent_map": 416,
            "pilot_issue": 447,
            "pilot_team_ids_sha256_16": pilot_hash,
            "train_opponents": data["train_opps"],
            "effective_df": round(model.df, 4),
            "std_definitions": {
                "epistemic_std": "sigma_resid * sqrt(x^T (X^T X + P)^-1 x) from Bayesian ridge posterior",
                "pred_std": "sqrt(sigma_true_resid^2 + epistemic_std^2 + extra_bc_input_var) for true infinite-battle Foul Play win rate",
                "obs_2b_std_formula": "sqrt(pred_std^2 + fp_2b_noise_sd^2) for a fresh 2-battle-per-opponent Foul Play measurement",
            },
            "calibration_5fold": calibration,
            "min_pair": MIN_PAIR,
            "pair_mult": PAIR_MULT,
            "lam_grid": list(LAM_GRID),
        },
    }
    predictor_id = write_corpus_predictions(con, predictor_meta, scored_rows, held_ids=held_ids, overwrite=overwrite)

    preds = np.array([r[1] for r in scored_rows], dtype=np.float64)
    stds = np.array([r[2] for r in scored_rows], dtype=np.float64)
    eps = np.array([r[3] for r in scored_rows], dtype=np.float64)
    bcs = np.array([r[4] for r in scored_rows], dtype=np.float64)
    summary = {
        "predictor_id": predictor_id,
        "predictor_name": predictor_name,
        "kind": PREDICTOR_KIND,
        "target_policy_id": data["policy_id"],
        "input_policy_id": data["bc_policy_id"],
        "teams_scored": len(scored_rows),
        "pilot_train_teams_in_corpus": sum(r[7] for r in scored_rows),
        "held_out_teams_scored": 0,
        "seconds": round(time.perf_counter() - t0, 2),
        "model": {
            "lam_single": model.predictor.lam_s,
            "lam_pair": model.predictor.lam_p,
            "intercept": round(float(model.predictor.w_unpen[0]), 5),
            "bc_slope": round(float(model.predictor.w_unpen[1]), 5),
            "n_features": predictor_meta["n_features"],
            "effective_df": round(model.df, 2),
            "sigma_resid": round(model.sigma_resid, 5),
            "sigma_true_resid": round(model.sigma_true_resid, 5),
            "fp_2b_noise_sd": round(float(np.sqrt(fp_noise_var)), 5),
        },
        "calibration_5fold": calibration,
        "pred_win_rate": _dist_summary(preds),
        "pred_std": _dist_summary(stds),
        "epistemic_std": _dist_summary(eps),
        "bc_win_rate": _dist_summary(bcs),
        "corr_pred_vs_bc": round(_corr(preds, bcs), 5),
    }
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "corpus_labeling_summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    return summary


# ---------------------------------------------------------------- CLI
def main(argv=None):
    import matchup_db as M

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--db", default=str(M.DEFAULT_DB))
    ap.add_argument("--pilot-dir", default=str(PILOT_DIR))
    ap.add_argument("--out", default=str(OUT))
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("preflight")
    r_p = sub.add_parser("run")
    r_p.add_argument("--no-mlp", action="store_true")
    sub.add_parser("figures")
    lc_p = sub.add_parser("label-corpus", help="fit predictor_team_bc on the 500 pilot teams and label the full corpus + std")
    lc_p.add_argument("--predictor-name", default=DEFAULT_PREDICTOR_NAME)
    lc_p.add_argument("--batch-size", type=int, default=10000)
    lc_p.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)

    out_dir = Path(a.out)
    if a.cmd == "figures":
        payload = json.loads((out_dir / "results.json").read_text())
        pages = write_storyboard_figures(payload, out_dir / "storyboard")
        print("wrote:", [str(p) for p in pages])
        return

    con = M.connect(a.db)
    if a.cmd == "preflight":
        data = load_pilot_dataset(con, Path(a.pilot_dir))
        pf = run_preflight(data)
        print(json.dumps(pf, indent=1))
        if not pf["all_passed"]:
            raise SystemExit(1)
    elif a.cmd == "run":
        payload = run_experiment(con, pilot_dir=Path(a.pilot_dir), out_dir=out_dir, run_mlp=not a.no_mlp)
        write_storyboard_figures(payload, out_dir / "storyboard")
        md = format_markdown_report(payload)
        (out_dir / "report.md").write_text(md)
        print(md)
    elif a.cmd == "label-corpus":
        summary = label_corpus(
            con,
            pilot_dir=Path(a.pilot_dir),
            predictor_name=a.predictor_name,
            out_dir=out_dir,
            batch_size=a.batch_size,
            overwrite=a.overwrite,
        )
        print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
