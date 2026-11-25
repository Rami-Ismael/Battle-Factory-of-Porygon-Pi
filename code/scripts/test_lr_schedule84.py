"""Tests for the learning-rate schedules the 84-field trainers share (issue #431).

The GIDD cosine expected values are worked by hand from GIDD's get_lr (dvruette/gidd@f67acc96,
gidd/utils.py:18-33): linear warm-up lr * step / warmup_steps, then
lr * (0.1 + 0.9 * (1 + cos(pi * (step - warmup) / (total - warmup))) / 2).
Clean-machine safe.

    python scripts/test_lr_schedule84.py   (from code/)
"""
import math
import unittest

import repo_paths  # noqa: F401  (src/ first on sys.path)
import lr_schedule84 as LS


class CosineToZeroTests(unittest.TestCase):
    """The schedule every run before #431 used must stay bit-for-bit the same."""

    def test_matches_the_inline_formula_the_trainers_used(self):
        lr, warm, steps = 4e-4, 500, 66_960
        old = lambda s: lr * min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps))
        lr_at = LS.make_lr_at("cosine", peak_lr=lr, total_steps=steps, warmup_steps=warm)
        for step in (0, 1, 499, 500, 33_480, 66_959, 66_960, 70_000):
            self.assertEqual(lr_at(step), old(step))

    def test_refuses_a_floor(self):
        with self.assertRaises(ValueError):
            LS.make_lr_at("cosine", peak_lr=4e-4, total_steps=100, warmup_steps=10, min_lr=1e-5)


class GiddCosineTests(unittest.TestCase):
    def setUp(self):
        # 1,000 warm-up steps, 11,000 total: the cosine runs over the 10,000 steps after warm-up
        self.lr_at = LS.make_lr_at("gidd_cosine", peak_lr=5e-4, total_steps=11_000, warmup_steps=1_000, min_lr=5e-5)

    def test_warm_up_is_linear_from_zero(self):
        self.assertEqual(self.lr_at(0), 0.0)
        self.assertAlmostEqual(self.lr_at(500), 2.5e-4, places=12)

    def test_peak_at_the_end_of_warm_up(self):
        self.assertAlmostEqual(self.lr_at(1_000), 5e-4, places=12)

    def test_halfway_through_the_cosine_sits_midway_between_peak_and_floor(self):
        self.assertAlmostEqual(self.lr_at(6_000), 2.75e-4, places=12)        # 5e-4 * (0.1 + 0.9 * 0.5)

    def test_ends_on_the_floor_and_stays_there(self):
        self.assertAlmostEqual(self.lr_at(11_000), 5e-5, places=12)
        self.assertAlmostEqual(self.lr_at(20_000), 5e-5, places=12)

    def test_floor_is_a_free_parameter(self):
        lr_at = LS.make_lr_at("gidd_cosine", peak_lr=5e-4, total_steps=11_000, warmup_steps=1_000, min_lr=0.0)
        self.assertAlmostEqual(lr_at(11_000), 0.0, places=12)


class SharedRecipeTests(unittest.TestCase):
    def test_gidd_recipe_is_the_ticket_recipe(self):
        self.assertEqual(LS.GIDD_LR_RECIPE, dict(lr=5e-4, warmup=1_000, min_lr=5e-5, lr_schedule="gidd_cosine"))


if __name__ == "__main__":
    unittest.main()
