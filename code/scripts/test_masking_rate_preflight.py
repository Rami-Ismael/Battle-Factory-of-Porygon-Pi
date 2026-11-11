"""The corrected t and the linear masking rate of the masking-rate rows (src/masking_rate_preflight.py; docs/masking-rate-rows.md).

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


if __name__ == "__main__":
    unittest.main()
