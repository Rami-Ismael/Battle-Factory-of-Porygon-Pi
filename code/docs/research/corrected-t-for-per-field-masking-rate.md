# Corrected t for a per-field masking rate, fixed-order decoding

Issue #293. Script: `code/docs/research/corrected_t_rules.py` (numpy only; run `python3 -I corrected_t_rules.py`).

## Question

The sampler fills 84 fields in a fixed ORDER (species first, then abilities, items, moves, Stat Alignment, Stat Points; see `code/src/diffusion84.py` line 39). Field i is hidden with probability t^{w_i}, where w_i is its ORDER weight: species 16, abilities 4, items 4, moves 1.5, Stat Alignment 1, Stat Points 1. The sampler today uses t = 1 − k/84 at pass k, which ignores the weights. Which t should pass k use so the model sees the masking rate it was trained on?

## What the sources say

- **MD4** (Shi et al., arXiv 2406.04329). Training draws each token's mask independently. Algorithm 2 (Appendix F.2) samples each masked position from a categorical distribution over a time grid: "for n ∈ [N], if x_t^(n) = m, draw x_s^(n) ∼ Cat(...)". Section 6 (GenMD4) generalizes the schedule to a per-token α_t: the loss in Eq. (12) uses "α_t is the elementwise derivative of α_t", and "This generalizes the MD4 loss (4), which is recovered when α_t is a scalar schedule times a vector of ones." So a per-token α_{t,i} is in the paper's own framework. The form is in the paper: "we parametrize the m + 1 dimensional function α_t using the polynomial schedule (see Fig. 1) as α_t,i = 1 − t^{w_i}" (§6), with w_i learned per token value (App. I.2). Ours fixes w per field type.
- **MD4 has no fixed-order rule.** Its sampler unmasks tokens at random times, not in a fixed order. Any rule for a fixed order is ours.
- **LLaDA** (arXiv 2502.09992). It uses a linear masking probability, which it calls "analogous to but distinct from, the noise schedule in continuous diffusion models", motivated by "the information in the text is proportional to the number of tokens on average". Its Eq. (11) gives the time-free posterior: q_{0|t}(x_s^i | x_t) = p_data(x_0^i | x_t^UM) for every i with x_t^i = M. The extracted text writes x_s^i on the left; x_0^i is probably meant (**UNVERIFIED**, could be an extraction artifact).

Read together: both papers set the masked share of each token from the schedule, and neither gives a rule for a fixed order.

## Three candidate rules

Let f(t) = Σ_i t^{w_i} / 84 be the expected hidden share at rate t. Pass k has the first k fields visible and 84 − k hidden.

- **(a) Linear.** t = 1 − k/84. Today's rule. Ignores weights.
- **(b) Expected share.** Solve f(t) = (84 − k)/84 by bisection. Matches the expected number of hidden fields to the number the pass leaves hidden.
- **(c) Most likely.** Maximize the log-likelihood of the pattern (first k visible, rest hidden): L(t) = Σ_hidden w_i log t + Σ_visible log(1 − t^{w_i}). Its derivative falls strictly, so the root is unique; bisect on [1e-12, 1 − 1e-12]. The script checks each root against a 20001-point grid search for every k.

## Values

| k (fields visible) | (a) linear | (b) expected share | (c) most likely |
|---|---|---|---|
| 0 | 1.0000 | 1.0000 | 1.0000 |
| 6 | 0.9286 | 0.9701 | 0.9652 |
| 12 | 0.8571 | 0.9338 | 0.9238 |
| 18 | 0.7857 | 0.8901 | 0.8721 |
| 24 | 0.7143 | 0.8391 | 0.8188 |
| 42 | 0.5000 | 0.6497 | 0.6169 |
| 48 | 0.4286 | 0.5759 | 0.5446 |
| 60 | 0.2857 | 0.4120 | 0.3880 |
| 78 | 0.0714 | 0.1193 | 0.1128 |
| 83 | 0.0119 | 0.0220 | 0.0212 |

All 85 passes are in the script output; every grid check passes.

## Recommendation

Use rule (b), the expected share. MD4's training and sampling both set each token's masked probability from the schedule, so the right target for a pass is the expected number of masked fields at that pass. Rule (a) hides fewer fields than the pass calls for: at k = 42 it gives t = 0.5, which hides 36% of fields, not the 50% the pass leaves hidden. Rule (b) corrects this. Rule (c) is the alternative if the goal is the single most probable pattern; it is not what MD4's per-token schedule computes.

This recommendation is ours. MD4 and LLaDA set the masking rate per token or uniformly over tokens, not over a weighted fixed order, so no paper prescribes the fixed-order t.

## Caveats

- The per-token form α_{t,i} = 1 − t^{w_i} is MD4 §6's (verified 2026-11-11 by the main session); the fixed per-field-type w is ours.
- The weights are the project's ORDER weights from `code/src/diffusion84.py`, not values from a paper.
- Nothing here is tested against model output. Compare the sampler's quality under each rule before adopting one.
