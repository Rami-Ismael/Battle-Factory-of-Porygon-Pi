"""The corrected t and the linear masking schedule of the masking-schedule rows (src/masking_rate_preflight.py; docs/masking-rate-rows.md).

Needs this laptop's /tmp/vgc-pilot data; skipped without it.

    /tmp/vgc-pilot/.venv/bin/python scripts/test_masking_rate_preflight.py   (from code/)
"""
import sys
import unittest
from itertools import pairwise
from pathlib import Path

import repo_paths  # noqa: F401  (src/ first on sys.path)

HERE = Path("/tmp/vgc-pilot/data/pokedex.json").exists()
if HERE:
    sys.path.insert(1, "/tmp/vgc-pilot/src")
    import diffusion84 as M
    import masking_rate_preflight as P


@unittest.skipUnless(HERE, "needs /tmp/vgc-pilot data")
class CorrectedT(unittest.TestCase):
    def setUp(self):
        self.per_field = M.sched84(M.SCHED).double().numpy()
        self.linear = M.sched84(P.LINEAR).double().numpy()

    def test_linear_masking_rate_hides_every_field_alike(self):
        self.assertEqual(len(self.linear), 84)
        self.assertTrue((self.linear == 1).all())

    def test_linear_masking_rate_needs_no_correction(self):
        for filled in range(84):
            self.assertAlmostEqual(P.corrected_t(self.linear, filled), 1 - filled / 84, places=9)

    def test_per_field_values_match_the_corrected_t_ticket(self):
        for filled, published in P.PUBLISHED_CORRECTED_T.items():
            self.assertAlmostEqual(P.corrected_t(self.per_field, filled), published, delta=5e-4)

    def test_expected_hidden_share_equals_the_share_still_hidden(self):
        for filled in (1, 6, 18, 42, 60, 83):
            t = P.corrected_t(self.per_field, filled)
            self.assertAlmostEqual(float((t ** self.per_field).mean()), 1 - filled / 84, places=9)

    def test_t_falls_as_fields_fill_and_sits_above_the_uncorrected_t(self):
        values = [P.corrected_t(self.per_field, filled) for filled in range(84)]
        self.assertEqual(values[0], 1.0)
        self.assertTrue(all(a > b for a, b in pairwise(values)))
        self.assertTrue(all(v > 1 - filled / 84 for filled, v in enumerate(values) if filled))


@unittest.skipUnless(HERE, "needs /tmp/vgc-pilot data")
class CorrectedTGate(unittest.TestCase):
    """The reading that decides whether Per-field, corrected t is battled (docs/masking-rate-rows.md)."""

    @staticmethod
    def readings(**by_ask):
        return {ask: dict(total_variation_by_rule_and_field_type=dict(corrected=fields, always_1=dict(species=0.9))) for ask, fields in by_ask.items()}

    def test_one_ask_above_the_bar_is_enough_even_when_the_mean_over_asks_is_below(self):
        by_ask = self.readings(low=dict(species=0.001, moves=0.002), mid=dict(species=0.001, moves=0.003), high=dict(species=0.001, moves=0.012))
        self.assertAlmostEqual(P.largest_reading(by_ask), 0.012)
        self.assertLess((0.002 + 0.003 + 0.012) / 3, P.MIN_TOTAL_VARIATION)
        self.assertGreaterEqual(P.largest_reading(by_ask), P.MIN_TOTAL_VARIATION)

    def test_only_the_corrected_rule_is_judged(self):
        by_ask = self.readings(low=dict(species=0.001), high=dict(species=0.004))
        self.assertAlmostEqual(P.largest_reading(by_ask), 0.004)
        self.assertAlmostEqual(P.largest_reading(by_ask, rule="always_1"), 0.9)


if __name__ == "__main__":
    unittest.main()
