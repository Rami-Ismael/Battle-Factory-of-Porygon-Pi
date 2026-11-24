"""Unit tests for the uniform discrete diffusion pre-registration probes and metrics (issue #274)."""
import sys
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import uniform_preflight84 as UP


class UniformPreflightTests(unittest.TestCase):
    def test_port_equality_matches_upstream_diffusion_py_in_float32_and_float64(self):
        vocab_sizes = dict(species=347, ability=200, item=149, move=497, nature=26, stat=33)
        for dtype in (torch.float32, torch.float64):
            report = UP.port_equality_check(UP.generalized_uniform_loss, vocab_sizes, batch_size=6, dtype=dtype)
            self.assertTrue(report["passed"], report)

    def test_generalized_loss_finite_at_high_species_exponent_in_float32(self):
        torch.manual_seed(0)
        times = torch.tensor([1e-3, 0.05, 0.2, 0.35, 0.7, 0.999], dtype=torch.float32)
        clean = torch.randint(0, 347, (len(times), 6))
        move = torch.rand(len(times), 6) < times[:, None].pow(16.0)
        noisy = torch.where(move, torch.randint(0, 347, (len(times), 6)), clean)
        logits = torch.randn(len(times), 6, 347, requires_grad=True)
        loss = UP.generalized_uniform_loss(F.log_softmax(logits, dim=-1), noisy, clean, times, 16.0, 347)
        loss.mean().backward()
        self.assertTrue(torch.isfinite(loss).all())
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_generalized_loss_matches_finite_difference_kl_derivative(self):
        torch.manual_seed(7)
        dt = 1e-7
        times = torch.linspace(0.3, 0.85, 6, dtype=torch.float64)
        vocab_size = 26
        clean = torch.randint(0, vocab_size, (6, 4))
        noisy = torch.randint(0, vocab_size, (6, 4))
        log_pred = F.log_softmax(torch.randn(6, 4, vocab_size, dtype=torch.float64), dim=-1)
        for exponent in (1.0, 1.5, 4.0, 16.0):
            analytic = UP.generalized_uniform_loss(log_pred, noisy, clean, times, exponent, vocab_size)
            move_t = times[:, None, None].pow(exponent)
            move_s = (times - dt)[:, None, None].pow(exponent)
            q_true = UP.uniform_reverse_posterior(F.one_hot(clean, vocab_size).double(), noisy, move_s, move_t, vocab_size)
            p_model = UP.uniform_reverse_posterior(log_pred.exp(), noisy, move_s, move_t, vocab_size)
            kl_val = (q_true * (q_true.log() - p_model.log())).sum(dim=-1)
            rel_err = ((analytic - kl_val / dt).abs() / analytic.abs().clamp_min(1e-8)).max().item()
            self.assertLess(rel_err, 5e-4, f"exponent={exponent}, rel_err={rel_err}")

    def test_cfg_posterior_endpoints_and_normalization(self):
        torch.manual_seed(11)
        vocab_size = 33
        log_cond = F.log_softmax(torch.randn(4, 6, vocab_size), dim=-1)
        log_uncond = F.log_softmax(torch.randn(4, 6, vocab_size), dim=-1)
        noisy = torch.randint(0, vocab_size, (4, 6))
        move_t = torch.full((4, 1, 1), 0.5)
        move_s = torch.full((4, 1, 1), 0.4)
        q0 = UP.uniform_cfg_posterior(log_cond, log_uncond, noisy, move_s, move_t, 0.0, vocab_size)
        q1 = UP.uniform_cfg_posterior(log_cond, log_uncond, noisy, move_s, move_t, 1.0, vocab_size)
        q2 = UP.uniform_cfg_posterior(log_cond, log_uncond, noisy, move_s, move_t, 2.0, vocab_size)
        for posterior in (q0, q1, q2):
            self.assertTrue((posterior >= 0).all())
            self.assertTrue(torch.allclose(posterior.sum(dim=-1), torch.ones(4, 6), atol=1e-5))
        self.assertFalse(torch.allclose(q1, q2))

    def test_revision_noise_floor_reproduces_research_golden(self):
        # Research ticket #273 golden: N=189, K=128, ideal alpha(t)=1-t -> 4.404443
        golden = UP.revision_noise_floor(189, 128, exponent=1.0, eps=1e-5, schedule_eps=0.0)
        self.assertAlmostEqual(golden, 4.404443, places=5)
        # Per-field species exponent 16 lowers the noise floor well below log-linear
        pf_species = UP.revision_noise_floor(347, 128, exponent=16.0)
        ll_species = UP.revision_noise_floor(347, 128, exponent=1.0)
        self.assertLess(pf_species, ll_species / 2.0)

    def test_dominated_area_and_equal_error_advantage(self):
        # Two points: (0.005, 0.5) and (0.015, 0.8), error_ref=0.025, metric_ref=0.0
        # Area = (0.015 - 0.005) * 0.5 + (0.025 - 0.015) * 0.8 = 0.005 + 0.008 = 0.013
        pts_a = [(0.005, 0.5), (0.015, 0.8), (0.010, 0.4)]  # (0.010, 0.4) is dominated
        pts_b = [(0.005, 0.4), (0.015, 0.7)]
        self.assertAlmostEqual(UP.dominated_area(pts_a, error_ref=0.025, metric_ref=0.0), 0.013, places=10)
        adv = UP.equal_error_advantage(pts_a, pts_b)
        self.assertIsNotNone(adv)
        self.assertAlmostEqual(adv["mean_advantage"], 0.1, places=10)
        self.assertIsNone(UP.equal_error_advantage([(0.002, 0.5)], [(0.010, 0.8)]))

    def test_usability_and_bend_point_rules(self):
        self.assertTrue(UP.is_usable_strength(True, 0.50, 0.15, 0.006, 0.005))
        self.assertFalse(UP.is_usable_strength(False, 0.50, 0.15, 0.006, 0.005))
        self.assertFalse(UP.is_usable_strength(True, 0.19, 0.15, 0.006, 0.005))
        self.assertFalse(UP.is_usable_strength(True, 0.50, 0.51, 0.006, 0.005))
        self.assertFalse(UP.is_usable_strength(True, 0.50, 0.15, 0.011, 0.005))
        coarse = [
            dict(weight=1.0, error=0.018, within_ask_unique_fraction=0.90, usable=True),
            dict(weight=2.0, error=0.008, within_ask_unique_fraction=0.88, usable=True),
            dict(weight=4.0, error=0.005, within_ask_unique_fraction=0.85, usable=True),
            dict(weight=8.0, error=0.012, within_ask_unique_fraction=0.10, usable=False),
        ]
        # Must prioritize bracketing the usability transition (4..8 -> 5.7) and large error jump (1..2 -> 1.4), capped at 2
        self.assertEqual(UP.bend_points_to_add(coarse, max_added=2), [1.4, 5.7])


if __name__ == "__main__":
    unittest.main()
