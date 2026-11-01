"""The 84-field training table (issue #58): which teams it keeps, its Stat Point check, and byte-identical saves.
Needs this laptop's /tmp/vgc-pilot runtime (lossdata84 imports the data modules); skipped without it.

    /tmp/vgc-pilot/.venv/bin/python scripts/test_lossdata84.py   (from code/)
"""
import tempfile
import unittest
from pathlib import Path

import numpy as np

import repo_paths  # noqa: F401  (src/ first on sys.path)

LD84 = None
if Path("/tmp/vgc-pilot/data/pokedex.json").exists():
    import lossdata84 as LD84

ROW = np.array([1] * 48 + [28, 1, 3, 6, 33, 1] + [1] * 30, np.int16)      # Archaludon's 66 Stat Points, then zeros


@unittest.skipUnless(LD84 is not None, "the /tmp/vgc-pilot runtime is not available")
class WhichTeamsAreKept(unittest.TestCase):
    def test_a_team_seen_with_one_set_of_stat_points_keeps_its_row(self):
        self.assertIs(LD84.single({(27, 0, 2, 5, 32, 0): ROW}), ROW)

    def test_a_team_battled_with_two_sets_of_stat_points_is_dropped(self):
        self.assertIsNone(LD84.single({(27,) * 1: ROW, (26,) * 1: ROW}))

    def test_a_team_with_no_paste_is_dropped(self):
        self.assertIsNone(LD84.single(None))
        self.assertIsNone(LD84.single({}))

    def test_a_team_that_does_not_encode_is_dropped(self):
        self.assertIsNone(LD84.single({(27, 0, 2, 5, 32, 0): None}))


@unittest.skipUnless(LD84 is not None, "the /tmp/vgc-pilot runtime is not available")
class StatPointLimits(unittest.TestCase):
    def test_legal_rows_pass(self):
        self.assertTrue(LD84.legal_stat_points(np.stack([ROW, ROW])))

    def test_a_masked_stat_point_fails(self):
        bad = ROW.copy(); bad[50] = 0
        self.assertFalse(LD84.legal_stat_points(bad[None]))

    def test_a_pokemon_over_66_fails(self):
        bad = ROW.copy(); bad[53] = 2                                  # Archaludon's Speed 0 -> 1: total 67
        self.assertFalse(LD84.legal_stat_points(bad[None]))


@unittest.skipUnless(LD84 is not None, "the /tmp/vgc-pilot runtime is not available")
class ByteIdenticalSaves(unittest.TestCase):
    def test_saving_the_same_arrays_twice_gives_the_same_bytes_and_loads_back(self):
        arrays = dict(all_X=np.stack([ROW, ROW]), all_W=np.array([0.5, np.nan], np.float32))
        with tempfile.TemporaryDirectory() as d:
            a, b = Path(d) / "a.npz", Path(d) / "b.npz"
            LD84.save_npz(a, arrays); LD84.save_npz(b, dict(reversed(list(arrays.items()))))
            self.assertEqual(a.read_bytes(), b.read_bytes())
            back = np.load(a)
            np.testing.assert_array_equal(back["all_X"], arrays["all_X"])
            np.testing.assert_array_equal(back["all_W"], arrays["all_W"])


if __name__ == "__main__":
    unittest.main()
