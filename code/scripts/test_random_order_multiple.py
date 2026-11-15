"""The verdict of issue #323: did a weaker model cause the 2.8 multiple?
The rule uses hand-made numbers. The bootstrap test reads this laptop's stored grids and skips without them.

    /tmp/vgc-pilot/.venv/bin/python scripts/test_random_order_multiple.py   (from code/)
"""
import unittest

import numpy as np

import repo_paths  # noqa: F401  (src/ first on sys.path)
import masking_rate_rows as MRR
import random_order_multiple as R

OLD, LINEAR, LINEAR_TOP = 2.78, 4.76, 7.10


class Call(unittest.TestCase):
    """interval: the 95% interval of random order's error over the control's, both on the control's checkpoint."""
    def test_confirmed_when_the_interval_holds_the_linear_multiple_and_sits_above_the_old_one(self):
        self.assertEqual(R.call([3.1, 6.2], OLD, LINEAR, LINEAR_TOP), "confirmed")

    def test_not_separated_when_the_interval_also_holds_the_old_multiple(self):
        self.assertEqual(R.call([2.5, 6.2], OLD, LINEAR, LINEAR_TOP), "not separated")

    def test_cleared_when_the_interval_sits_above_the_top_of_the_linear_interval(self):
        self.assertEqual(R.call([7.3, 11.0], OLD, LINEAR, LINEAR_TOP), "cleared: the linear masking schedule helped random order")

    def test_cleared_the_other_way_when_the_interval_sits_below_the_linear_multiple(self):
        self.assertEqual(R.call([3.0, 4.5], OLD, LINEAR, LINEAR_TOP), "cleared: the linear masking schedule made random order relatively worse")

    def test_an_interval_above_the_linear_multiple_but_not_its_top_is_not_separated(self):
        self.assertEqual(R.call([5.0, 9.0], OLD, LINEAR, LINEAR_TOP), "not separated")


class Bootstrap(unittest.TestCase):
    """The control's-checkpoint draws must be #297's: the same function gives the linear pair's stored interval."""
    def setUp(self):
        if not all(MRR.opponents_path(row, 2).exists() for row in MRR.ROWS): self.skipTest("stored grids not on this machine")

    def test_row_draws_equal_the_masking_schedule_bootstrap_for_a_row_it_knows(self):
        counts = MRR.opponent_draws(len(np.load(MRR.opponents_path(MRR.CONTROL, 2))["weights"]))
        ours, theirs = R.row_draws(MRR.CONTROL, 2, counts), MRR.bootstrap(MRR.CONTROL, 2, counts)
        for key in theirs: np.testing.assert_array_equal(ours[key], theirs[key])

    def test_the_linear_multiple_and_interval_match_the_masking_schedule_summary(self):
        if not MRR.SUMMARY.exists(): self.skipTest("no masking-schedule summary")
        stored = __import__("json").load(open(MRR.SUMMARY))["random_order_against_species_first"]["guidance_2"]
        counts = MRR.opponent_draws(len(np.load(MRR.opponents_path(MRR.CONTROL, 2))["weights"]))
        m = R.shuffled_only_multiple(R.LINEAR_RANDOM, R.LINEAR_SPECIES_FIRST, 2, counts)
        self.assertAlmostEqual(m["multiple"], stored["error_multiple"], places=9)
        np.testing.assert_allclose(m["interval"], stored["error_multiple_interval"], rtol=1e-12)


if __name__ == "__main__":
    unittest.main()
