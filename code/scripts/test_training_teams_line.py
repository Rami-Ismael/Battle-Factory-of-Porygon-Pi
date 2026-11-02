"""The arithmetic behind the training-teams line (issue #72). No game data, checkpoints or Showdown, so CI can run it.

    .venv/bin/python scripts/test_training_teams_line.py   (from code/)
"""
import math
import unittest

import numpy as np

import training_teams_line as L


class NearestAsk(unittest.TestCase):
    def test_a_label_goes_to_the_ask_it_is_closest_to(self):
        self.assertEqual(L.nearest_ask([0.5, 0.26, 0.24]).tolist(), [12, 6, 6])

    def test_a_label_just_past_the_midpoint_moves_up(self):
        self.assertEqual(L.nearest_ask([12.5 / 24 + 1e-6, 12.5 / 24 - 1e-6]).tolist(), [13, 12])

    def test_an_unlabelled_team_gets_no_ask(self):
        self.assertEqual(L.nearest_ask([float("nan")]).tolist(), [-1])


class LineError(unittest.TestCase):
    def test_on_the_line_is_zero(self):
        self.assertEqual(L.line_error([0.25, 0.5], [0.25, 0.5]), 0.0)

    def test_a_typical_miss_of_0_03_is_just_under_the_near_line_bar(self):
        self.assertLess(L.line_error([0.5], [0.53]), L.NEAR_LINE)
        self.assertGreater(L.line_error([0.5], [0.54]), L.NEAR_LINE)

    def test_noise_correction_removes_the_variance_of_the_mean(self):
        y = np.array([0.4, 0.6, 0.4, 0.6])
        expected = L.line_error([0.5], [0.5]) - y.var(ddof=1) / 4
        self.assertAlmostEqual(L.noise_corrected_error([0.5], [y]), expected)


class Scatter(unittest.TestCase):
    def test_teams_differ_only_by_what_battle_noise_leaves(self):
        y = np.array([0.4, 0.6, 0.4, 0.6])
        s = L.scatter(y, [y.var(ddof=1)] * 4)
        self.assertEqual(s["team_sd"], 0.0)
        self.assertAlmostEqual(s["total_sd"], s["battle_noise_sd"])

    def test_noise_above_the_observed_spread_floors_at_zero(self):
        self.assertEqual(L.scatter(np.array([0.5, 0.5]), [0.01, 0.01])["team_sd"], 0.0)

    def test_one_battle_noise_is_the_binomial_variance_and_not_the_scorers_zero(self):
        self.assertAlmostEqual(L.one_battle_noise(0.5, 49), 0.25 / 49)
        self.assertEqual(L.one_battle_noise(0.0, 49), 0.0)

    def test_real_differences_remain(self):
        y = np.array([0.2, 0.8])
        self.assertAlmostEqual(L.scatter(y, [0.0, 0.0])["team_sd"], math.sqrt(y.var(ddof=1)))


class Bootstrap(unittest.TestCase):
    def test_identical_arms_differ_by_exactly_zero(self):
        y = [np.array([0.2, 0.3, 0.4, 0.5]), np.array([0.5, 0.6, 0.4, 0.7])]
        b = L.bootstrap([0.25, 0.5], y, y, reps=200)
        self.assertEqual(b["redrawn_minus_stored"], [0.0, 0.0])
        self.assertEqual(b["stored"], b["redrawn"])

    def test_a_constant_shift_shows_up_in_got(self):
        y = [np.array([0.2, 0.3, 0.4, 0.5])]
        b = L.bootstrap([0.35], y, [v + 0.1 for v in y], reps=200)
        self.assertAlmostEqual(b["got_redrawn_minus_stored"][0], 0.1)
        self.assertAlmostEqual(b["got_redrawn_minus_stored"][1], 0.1)


if __name__ == "__main__":
    unittest.main()
