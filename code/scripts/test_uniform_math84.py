"""CPU-only unit tests for the uniform discrete diffusion math and revision kernel (issue #275).

Needs only torch and numpy (no /tmp/vgc-pilot or laptop data), so GitHub Actions CI runs this on every PR.
Verifies:
  - Non-circular port-equality against verbatim upstream diffusion.py:572-597 (edb0f8c2) in float64 and float32,
    forcing both changed and unchanged tokens near the time bounds t = 1e-3 and t = 0.999.
  - Non-circular reverse-posterior and CFG equality against BOTH verbatim upstream alpha-form
    (diffusion.py:356-381, 1314-1334) AND independent Bayes-rule calculation (q(z_t | z_s) * p(z_s) / Z),
    including sharp-predictor float32 stress tests and the 84/128/256 sampling grids through gamma = 16.
  - Domain guards (out-of-range loss times rejected) and NaN/Inf best-checkpoint guards.
  - RevisionTracker metrics (value changes beyond first, noise floor, changes of mind, guided changes of mind,
    and mean fields changed per pass).

    python scripts/test_uniform_math84.py   (from code/)
"""
import math
import sys
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import uniform_math84 as UMath
import uniform_preflight84 as UP

REAL_VOCAB = dict(species=347, ability=200, item=149, move=497, nature=26, stat=33)


class UniformMathPortEqualityTests(unittest.TestCase):
    def test_port_equality_forced_changed_and_unchanged_near_time_bounds(self):
        for dtype, atol, rtol, grad_atol, grad_rtol in (
            (torch.float64, 1e-12, 1e-10, 1e-10, 1e-8),
            (torch.float32, 1e-3, 1e-3, 1e-4, 1e-3),
        ):
            with self.subTest(dtype=dtype):
                gen = torch.Generator().manual_seed(275)
                times = torch.tensor([1e-3, 2e-3, 0.25, 0.5, 0.75, 0.998, 0.999], dtype=dtype)
                for field_key, vocab_size in REAL_VOCAB.items():
                    clean = torch.randint(0, vocab_size, (len(times), 6), generator=gen)
                    # Force half of columns unchanged (noisy == clean) and half strictly changed (noisy != clean)
                    shifted = (clean + 1) % vocab_size
                    noisy = torch.cat([clean[:, :3], shifted[:, 3:]], dim=1)
                    base_logits = torch.randn((len(times), 6, vocab_size), generator=gen, dtype=dtype)
                    logits_ref = base_logits.clone().requires_grad_(True)
                    logits_cand = base_logits.clone().requires_grad_(True)

                    loss_ref = UP.upstream_uniform_loss(
                        F.log_softmax(logits_ref, dim=-1), noisy, clean, times, vocab_size
                    )
                    loss_cand = UMath.port_equality_loss(
                        F.log_softmax(logits_cand, dim=-1), noisy, clean, times, 1.0, vocab_size
                    )
                    loss_ref.mean().backward()
                    loss_cand.mean().backward()
                    self.assertTrue(torch.isfinite(loss_cand).all(), field_key)
                    self.assertTrue(torch.isfinite(logits_cand.grad).all(), field_key)
                    self.assertTrue(torch.allclose(loss_ref, loss_cand, atol=atol, rtol=rtol), field_key)
                    self.assertTrue(torch.allclose(logits_ref.grad, logits_cand.grad, atol=grad_atol, rtol=grad_rtol), field_key)

    def test_reverse_posterior_and_cfg_match_verbatim_upstream_and_independent_bayes(self):
        for dtype, tol in ((torch.float64, 1e-11), (torch.float32, 1e-5)):
            with self.subTest(dtype=dtype):
                gen = torch.Generator().manual_seed(77)
                for field_key, vocab_size in REAL_VOCAB.items():
                    log_cond = F.log_softmax(torch.randn(4, 6, vocab_size, generator=gen, dtype=dtype), dim=-1)
                    log_uncond = F.log_softmax(torch.randn(4, 6, vocab_size, generator=gen, dtype=dtype), dim=-1)
                    noisy = torch.randint(0, vocab_size, (4, 6), generator=gen)
                    for move_t_val, move_s_val in ((1.0, 1.0 - 1.0 / 128), (0.5, 0.5 - 1.0 / 128), (1.0 / 128, 1e-5)):
                        move_t = torch.full((4, 1, 1), move_t_val, dtype=dtype)
                        move_s = torch.full((4, 1, 1), move_s_val, dtype=dtype)
                        alpha_t = 1.0 - move_t
                        alpha_s = 1.0 - move_s
                        cand = UMath.reverse_posterior(log_cond.exp(), noisy, move_s, move_t, vocab_size)
                        ref_upstream = UP.upstream_reverse_posterior(log_cond.exp(), noisy, alpha_s, alpha_t, vocab_size)
                        ref_bayes = UP.bayes_reverse_posterior(log_cond.exp(), noisy, move_s, move_t, vocab_size)
                        self.assertTrue(torch.allclose(cand, ref_upstream, atol=tol, rtol=tol), (field_key, move_t_val))
                        self.assertTrue(torch.allclose(cand, ref_bayes, atol=tol, rtol=tol), (field_key, move_t_val))
                        self.assertTrue(torch.allclose(cand.sum(dim=-1), torch.ones(4, 6, dtype=dtype), atol=tol))

                        for gamma in (0.0, 1.0, 2.0, 4.0, 8.0, 16.0):
                            cfg_cand = UMath.cfg_posterior(log_cond, log_uncond, noisy, move_s, move_t, gamma, vocab_size)
                            cfg_up = UP.upstream_cfg_posterior(log_cond, log_uncond, noisy, alpha_s, alpha_t, gamma, vocab_size)
                            self.assertTrue(torch.isfinite(cfg_cand).all(), (field_key, gamma))
                            self.assertTrue(bool((cfg_cand >= 0).all()))
                            self.assertTrue(torch.allclose(cfg_cand.sum(dim=-1), torch.ones(4, 6, dtype=dtype), atol=tol))
                            self.assertTrue(torch.allclose(cfg_cand, cfg_up, atol=tol, rtol=tol), (field_key, gamma))

    def test_sharp_predictor_float32_accuracy_and_full_pass_grids_up_to_gamma_16(self):
        # Sharp predictor stress test: Bayes-product form keeps float32 error << 1e-4 vs float64 Bayes reference
        vocab_size = 497
        gen = torch.Generator().manual_seed(99)
        logits_64 = torch.randn((4, 6, vocab_size), generator=gen, dtype=torch.float64) * 8.0
        log_cond_64 = F.log_softmax(logits_64, dim=-1)
        log_uncond_64 = F.log_softmax(torch.randn((4, 6, vocab_size), generator=gen, dtype=torch.float64) * 4.0, dim=-1)
        noisy = torch.randint(0, vocab_size, (4, 6), generator=gen)
        for num_passes in (84, 128, 256):
            times = torch.linspace(1.0, UMath.SAMPLING_EPS, num_passes + 1, dtype=torch.float64)
            for idx in (0, 1, num_passes // 2, num_passes - 2, num_passes - 1):
                move_t_64 = times[idx].view(1, 1, 1)
                move_s_64 = times[idx + 1].view(1, 1, 1)
                ref_64 = UP.bayes_reverse_posterior(log_cond_64.exp(), noisy, move_s_64, move_t_64, vocab_size)
                cand_32 = UMath.reverse_posterior(
                    log_cond_64.float().exp(), noisy, move_s_64.float(), move_t_64.float(), vocab_size
                )
                max_err = float((cand_32.double() - ref_64).abs().max().item())
                self.assertLess(max_err, 5e-5, f"K={num_passes}, step={idx}, max_err={max_err}")
                cfg_16 = UMath.cfg_posterior(
                    log_cond_64.float(), log_uncond_64.float(), noisy, move_s_64.float(), move_t_64.float(), 16.0, vocab_size
                )
                self.assertTrue(bool(torch.isfinite(cfg_16).all()))
                self.assertTrue(torch.allclose(cfg_16.sum(dim=-1), torch.ones(4, 6), atol=1e-5))

    def test_loss_domain_guards_and_nan_best_val_guard(self):
        log_probs = F.log_softmax(torch.randn(2, 6, 33), dim=-1)
        tokens = torch.randint(0, 33, (2, 6))
        for bad_t in (0.0, 1.0, -0.1, 1.1, float("nan"), float("inf")):
            with self.subTest(bad_t=bad_t), self.assertRaises(ValueError):
                UMath.uniform_field_loss(log_probs, tokens, tokens, torch.tensor([0.5, bad_t]), 33)

        self.assertTrue(UMath.is_better_val_loss(0.9, 1.0))
        self.assertFalse(UMath.is_better_val_loss(1.0, 1.0))
        self.assertFalse(UMath.is_better_val_loss(float("nan"), 1.0))
        self.assertFalse(UMath.is_better_val_loss(float("nan"), math.inf))
        self.assertFalse(UMath.is_better_val_loss(float("inf"), math.inf))
        self.assertFalse(UMath.is_better_val_loss(float("-inf"), 1.0))

    def test_revision_tracker_logs_pre_registered_and_exploratory_metrics(self):
        tracker = UMath.RevisionTracker(batch_size=2, passes=4, vocab_sizes=REAL_VOCAB)
        z0 = torch.zeros((2, UMath.COLS), dtype=torch.long)
        z1 = torch.ones((2, UMath.COLS), dtype=torch.long)
        z2 = torch.full((2, UMath.COLS), 2, dtype=torch.long)
        # 4 passes: state changes on passes 0, 1, 2 -> total_changes=3 -> beyond_first=2; mean fields/pass = 3*84/4 = 63
        # cond top: [1, 1, 2, 2] -> 1 change of mind; guided top: [1, 2, 3, 4] -> 3 guided changes of mind
        seq = [(z0, z1, 1, 1), (z1, z2, 1, 2), (z2, z0, 2, 3), (z0, z0, 2, 4)]
        for idx, (prev_t, next_t, top_val, guided_val) in enumerate(seq):
            tracker.record_pass(
                idx,
                prev_t,
                next_t,
                torch.full((2, UMath.COLS), top_val, dtype=torch.long),
                guided_top_tokens=torch.full((2, UMath.COLS), guided_val, dtype=torch.long),
            )
        summary = tracker.summary()
        self.assertAlmostEqual(summary["mean_fields_changed_per_pass"], 63.0, places=6)
        for field_key in UMath.FIELD_KEYS:
            stats = summary["by_field"][field_key]
            self.assertAlmostEqual(stats["value_changes_beyond_first"], 2.0, places=6)
            self.assertAlmostEqual(stats["changes_of_mind"], 1.0, places=6)
            self.assertAlmostEqual(stats["guided_changes_of_mind"], 3.0, places=6)
            self.assertGreater(stats["noise_floor"], 0.0)


if __name__ == "__main__":
    unittest.main()
