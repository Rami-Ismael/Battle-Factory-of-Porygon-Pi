"""Three candidate values of t for a fixed-order sampler under per-field masking rates.

Pass k (0..84) has the first k fields of ORDER filled and the rest hidden. Field i is
hidden with probability t**w_i. Rules:

  (a) linear:      t = 1 - k/84                         (today's sampler)
  (b) expected:    solve f(t) = (84 - k)/84, f(t) = sum_i t**w_i / 84
  (c) most likely: argmax_t sum_hidden log(t**w_i) + sum_visible log(1 - t**w_i)

Numpy only. Run: python corrected_t_rules.py
"""

import numpy as np

# Species first, then abilities, items, moves, Stat Alignment, Stat Points (see code/src/diffusion84.py).
GROUPS = [("species", 6, 16.0), ("ability", 6, 4.0), ("item", 6, 4.0),
          ("move", 24, 1.5), ("stat_alignment", 6, 1.0), ("stat_points", 36, 1.0)]
W = np.concatenate([np.full(n, w) for _, n, w in GROUPS])
N = W.size
assert N == 84

CHECK_KS = [0, 6, 12, 18, 24, 42, 48, 60, 78, 83]


def rule_linear(k):
    return 1.0 - k / N


def expected_share(t):
    return np.sum(t ** W) / N


def rule_expected(k, iters=200):
    target = (N - k) / N
    if target <= 0.0:
        return 0.0
    if target >= 1.0:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if expected_share(mid) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def log_lik_pattern(t, hidden):
    visible = ~hidden
    return np.sum(W[hidden] * np.log(t)) + np.sum(np.log1p(-(t ** W[visible])))


def rule_most_likely(k, iters=200):
    """Root of dL/dt = sum_hidden w/t - sum_visible w t^(w-1)/(1-t^w); each term is monotone in t,
    so the derivative falls strictly and has at most one root. Bisect it."""
    hidden = np.arange(N) >= k
    visible = ~hidden
    if not hidden.any():
        return 0.0
    if not visible.any():
        return 1.0

    def dL(t):
        return np.sum(W[hidden] / t) - np.sum(W[visible] * t ** (W[visible] - 1) / (1 - t ** W[visible]))

    lo, hi = 1e-12, 1.0 - 1e-12
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if dL(mid) > 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def check_most_likely(k, t_star):
    """Brute-force grid check that t_star maximises the log-likelihood."""
    hidden = np.arange(N) >= k
    grid = np.linspace(1e-4, 1 - 1e-4, 20001)
    ll = np.array([log_lik_pattern(t, hidden) for t in grid])
    return abs(grid[np.argmax(ll)] - t_star) < 1e-3


def main():
    print(f"{'k':>3} {'filled':>7} {'(a) linear':>11} {'(b) expected':>13} {'(c) most-likely':>16}  grid-ok")
    rows = []
    for k in range(N + 1):
        a = rule_linear(k)
        b = rule_expected(k)
        c = rule_most_likely(k)
        ok = check_most_likely(k, c) if 0 < k < N else True
        rows.append((k, a, b, c, ok))
        if k in CHECK_KS:
            print(f"{k:>3} {k:>3}/84 {a:>11.4f} {b:>13.4f} {c:>16.4f}  {ok}")
    print()
    print("all 85 passes:")
    for k, a, b, c, ok in rows:
        print(f"{k:>3} {a:.4f} {b:.4f} {c:.4f} {ok}")
    assert all(ok for *_, ok in rows), "most-likely root disagrees with grid search"


if __name__ == "__main__":
    main()
