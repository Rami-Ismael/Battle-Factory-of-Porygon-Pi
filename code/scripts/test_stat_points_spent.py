"""Stat Points spent (stat_points_spent.py) on hand-built Stat Point values with known answers. No game data.

    /tmp/vgc-pilot/.venv/bin/python scripts/test_stat_points_spent.py   (from code/)
"""
import unittest

import numpy as np

import stat_points_spent as S


class StatPointValues(unittest.TestCase):
    def test_tokens_become_values_one_row_per_pokemon(self):
        row = np.array([0] * 48 + [v + 1 for slot in range(6) for v in (slot, 0, 0, 0, 0, 32)], dtype=np.int64)
        values = S.stat_point_values(row[None])
        self.assertEqual(values.shape, (6, 6))
        self.assertEqual(values[2].tolist(), [2, 0, 0, 0, 0, 32])


class Measures(unittest.TestCase):
    def setUp(self):
        self.m = S.measures(np.array([[32, 2, 0, 0, 0, 32],       # 66, Speed took its room of 32
                                      [32, 32, 2, 0, 0, 0],       # 66, first five reach 66, Speed room 0
                                      [20, 20, 0, 0, 0, 10],      # 50, Speed room 26, 16 unspent
                                      [0, 0, 0, 0, 0, 0]]))       # 0, Speed room 32

    def test_totals(self):
        self.assertAlmostEqual(self.m["share_spending_66"], 0.5)
        self.assertAlmostEqual(self.m["mean_total"], (66 + 66 + 50 + 0) / 4)
        self.assertAlmostEqual(self.m["share_total_66"], 0.5)
        self.assertAlmostEqual(self.m["share_total_40_to_59"], 0.25)
        self.assertAlmostEqual(self.m["share_total_0_to_39"], 0.25)
        self.assertAlmostEqual(self.m["share_total_60_to_65"], 0.0)

    def test_speed_room(self):
        self.assertAlmostEqual(self.m["mean_Spe"], (32 + 0 + 10 + 0) / 4)
        self.assertAlmostEqual(self.m["mean_speed_room"], (32 + 0 + 26 + 32) / 4)
        self.assertAlmostEqual(self.m["share_speed_below_room"], 0.5)
        self.assertAlmostEqual(self.m["share_first_five_reach_66"], 0.25)

    def test_bins_cover_every_pokemon(self):
        self.assertAlmostEqual(sum(self.m[f"share_total_{name}"] for name, _, _ in S.TOTAL_BINS), 1.0)


if __name__ == "__main__":
    unittest.main()
