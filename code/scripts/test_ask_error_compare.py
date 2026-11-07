"""The ask-versus-got error of a sampling arm, and its paired difference from a reference arm (src/ask_error_compare.py).
Needs only numpy, so CI can run it.

    .venv/bin/python scripts/test_ask_error_compare.py   (from code/)
"""
import unittest

import numpy as np

import repo_paths  # noqa: F401  (src/ first on sys.path)
from ask_error_compare import ask_error, paired_difference


class AskError(unittest.TestCase):
    def test_worked_example(self):
        # (0.25 - 4/24)^2 = 0.0069444, (1/3 - 8/24)^2 = 0, (0.5 - 16/24)^2 = 0.0277778; mean 0.0115741
        self.assertAlmostEqual(ask_error({4: 0.25, 8: 1 / 3, 16: 0.5}, [4, 8, 16]), 0.0115741, places=6)

    def test_only_the_named_asks_count(self):
        self.assertAlmostEqual(ask_error({4: 4 / 24, 12: 0.9, 16: 16 / 24}, [4, 16]), 0.0)


def draws(offset, asks=(4, 8, 16), n=2000, seed=0):
    """Bootstrap draws of the mean got per ask: the line plus a fixed offset plus a little noise."""
    rng = np.random.default_rng(seed)
    return {k: k / 24 + offset + rng.normal(0, 0.005, n) for k in asks}


class PairedDifference(unittest.TestCase):
    def test_an_arm_on_the_line_beats_one_off_it(self):
        p = paired_difference(draws(0.0), draws(0.1, seed=1), [4, 8, 16])
        self.assertLess(p["interval"][1], 0)
        self.assertEqual(p["share_closer"], 1.0)

    def test_an_arm_off_the_line_loses(self):
        p = paired_difference(draws(0.1), draws(0.0, seed=1), [4, 8, 16])
        self.assertGreater(p["interval"][0], 0)
        self.assertEqual(p["share_closer"], 0.0)


if __name__ == "__main__":
    unittest.main()
