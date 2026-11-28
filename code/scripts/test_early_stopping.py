"""CPU tests for the shared EarlyStopper extracted from diffusion84.Model84.train_new (#409).

Verifies best-loss tracking, threshold-gated patience resets, mid-epoch/interval stopping,
non-finite loss failure, and exact state_dict save/restore for checkpoint resume.
"""
import math
import unittest

import repo_paths  # noqa: F401
from early_stopping import EarlyStopper


class EarlyStopperTest(unittest.TestCase):
    def test_improves_stalls_three_intervals_and_stops_identifying_best(self):
        stopper = EarlyStopper(patience=3, threshold=0.001)
        # Index 0 baseline, improves at 1 and 2, stalls at 3, 4, 5 -> stops at 5 with best at 2.
        trace = [
            (0, 9.50, True, False),
            (1, 5.00, True, False),
            (2, 4.00, True, False),
            (3, 4.10, False, False),
            (4, 4.05, False, False),
            (5, 4.02, False, True),
        ]
        for idx, loss, exp_best, exp_stop in trace:
            is_best, should_stop = stopper.observe(idx, loss, meta={'update': idx * 5})
            self.assertEqual((is_best, should_stop), (exp_best, exp_stop), f'index {idx}')
        self.assertEqual(stopper.best_index, 2)
        self.assertAlmostEqual(stopper.best, 4.00)
        self.assertEqual(stopper.best_meta, {'update': 10})

    def test_gain_below_threshold_updates_best_without_resetting_patience(self):
        stopper = EarlyStopper(patience=3, threshold=0.01)
        self.assertEqual(stopper.observe(1, 5.00), (True, False))   # best=5.00, reset=1
        self.assertEqual(stopper.observe(2, 4.995), (True, False))  # smaller than 0.01 -> best=4.995, reset stays 1
        self.assertEqual(stopper.observe(3, 4.994), (True, False))  # best=4.994, reset stays 1
        self.assertEqual(stopper.observe(4, 4.993), (True, True))   # 4 - 1 >= 3 -> stops, keeping index 4 as best
        self.assertEqual(stopper.best_index, 4)
        self.assertAlmostEqual(stopper.best, 4.993)

    def test_zero_threshold_resets_patience_on_any_strict_improvement(self):
        stopper = EarlyStopper(patience=2, threshold=0.0)
        self.assertEqual(stopper.observe(1, 5.00), (True, False))
        self.assertEqual(stopper.observe(2, 4.9999), (True, False))
        self.assertEqual(stopper.observe(3, 5.01), (False, False))
        self.assertEqual(stopper.observe(4, 5.02), (False, True))
        self.assertEqual(stopper.best_index, 2)

    def test_patience_none_tracks_best_without_stopping(self):
        stopper = EarlyStopper(patience=None, threshold=0.001)
        for idx, loss in enumerate([4.0, 5.0, 6.0, 7.0, 8.0], start=1):
            is_best, should_stop = stopper.observe(idx, loss)
            self.assertEqual(is_best, idx == 1)
            self.assertFalse(should_stop)
        self.assertEqual(stopper.best_index, 1)

    def test_non_finite_loss_raises(self):
        stopper = EarlyStopper(patience=3, threshold=0.001)
        for bad in (math.nan, math.inf, -math.inf):
            with self.assertRaises(RuntimeError):
                stopper.observe(1, bad)

    def test_state_dict_roundtrip_matches_uninterrupted_trace(self):
        full = EarlyStopper(patience=3, threshold=0.01)
        resumed = EarlyStopper(patience=3, threshold=0.01)
        first_half = [(0, 6.0), (1, 4.0), (2, 4.05)]
        second_half = [(3, 4.04), (4, 4.03)]

        for idx, loss in first_half:
            full.observe(idx, loss, meta={'update': idx * 4})
            resumed.observe(idx, loss, meta={'update': idx * 4})

        saved = resumed.state_dict()
        restored = EarlyStopper()
        restored.load_state_dict(saved)

        for idx, loss in second_half:
            self.assertEqual(
                full.observe(idx, loss, meta={'update': idx * 4}),
                restored.observe(idx, loss, meta={'update': idx * 4}),
            )
        self.assertEqual(full.state_dict(), restored.state_dict())


if __name__ == '__main__':
    unittest.main()
