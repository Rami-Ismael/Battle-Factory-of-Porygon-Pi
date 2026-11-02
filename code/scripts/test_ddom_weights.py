"""Tests for the paper's bin weights. Run: python scripts/test_ddom_weights.py"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ddom_weights import bin_weights


class BinWeights(unittest.TestCase):
    def test_a_bin_of_higher_labels_outweighs_an_equal_bin_of_lower_labels(self):
        labels = np.array([0.1] * 50 + [0.9] * 50)
        weights = bin_weights(labels, tau=0.1, bins=4)
        self.assertGreater(weights[-1], weights[0] * 100)

    def test_a_nearly_empty_top_bin_is_held_back_by_its_size(self):
        labels = np.array([0.5] * 1000 + [1.0])
        weights = bin_weights(labels, tau=10.0, k_share=0.1, bins=2)
        self.assertLess(weights[-1], weights[0] / 50)

    def test_weights_average_one(self):
        labels = np.random.default_rng(0).uniform(size=1000)
        self.assertAlmostEqual(bin_weights(labels, tau=0.3).mean(), 1.0)

    def test_an_unlabelled_team_is_refused(self):
        with self.assertRaises(ValueError):
            bin_weights(np.array([0.2, np.nan, 0.8]), tau=0.1)


if __name__ == "__main__":
    unittest.main()
