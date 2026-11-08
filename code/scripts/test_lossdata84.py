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


def entry(battles=0, dropped=(), fam=1):
    e = dict(x=None, wins=0.5 * battles, battles=battles, tags=set(), fam=fam)
    if dropped: e["dropped"] = set(dropped)
    return e


@unittest.skipUnless(LD84 is not None, "the /tmp/vgc-pilot runtime is not available")
class ShuffledSlotTable(unittest.TestCase):
    """Which teams leave the shuffled-slot table (issue #229), and its counts read back from the saved files."""

    def test_a_team_whose_only_labels_were_dropped_leaves(self):
        self.assertEqual(LD84.fate(entry(dropped={LD84.LD.LISTED_FALLBACK}), None), LD84.NOT_SHUFFLED)
        self.assertEqual(LD84.fate(entry(dropped={LD84.LD.RUN_FILE}), None), LD84.NOT_SHUFFLED)

    def test_a_team_with_a_shuffled_label_stays_beside_dropped_ones(self):
        self.assertIsNone(LD84.fate(entry(battles=49, dropped={LD84.LD.RUN_FILE}), None))

    def test_a_team_never_labelled_stays_unlabelled(self):
        self.assertIsNone(LD84.fate(entry(), None))

    def test_a_team_the_validator_rejects_leaves(self):
        self.assertEqual(LD84.fate(entry(battles=49), "Archaludon can't learn Hyper Beam"), LD84.ILLEGAL)
        self.assertEqual(LD84.fate(entry(), "Archaludon can't learn Hyper Beam"), LD84.ILLEGAL)

    def test_lost_mask_names_each_dropped_rule(self):
        self.assertEqual(LD84.lost_mask(entry()), 0)
        self.assertEqual(LD84.lost_mask(entry(dropped={LD84.LD.LISTED_FALLBACK, LD84.LD.RUN_FILE})), 3)

    def test_tables_and_counts_from_the_saved_files(self):
        rows = {bytes([i]): ROW.copy() for i in range(5)}
        for i, r in enumerate(rows.values()): r[0] = i + 1
        pool = {bytes([0]): entry(battles=49),                                   # shuffled label: stays labelled
                bytes([1]): entry(battles=49, dropped={LD84.LD.RUN_FILE}),       # stays, run-file label lost
                bytes([2]): entry(dropped={LD84.LD.LISTED_FALLBACK}, fam=2),    # leaves: listed only
                bytes([3]): entry(),                                             # stays unlabelled
                bytes([4]): entry(battles=49)}                                   # leaves: illegal
        sets = {k: {(0,): r} for k, r in rows.items()}
        checked = []
        def check(row):
            checked.append(int(row[0])); return "illegal" if row[0] == 5 else None
        table, gone, errors, kept = LD84.shuffled_tables(dict(pool=pool), sets, check)
        self.assertEqual(kept, [bytes([0]), bytes([1]), bytes([3])])
        self.assertNotIn(3, checked)                                             # a team leaving anyway is not validated
        self.assertEqual(errors, ["illegal"])
        with tempfile.TemporaryDirectory() as d:
            t, r = Path(d) / "t.npz", Path(d) / "r.npz"
            LD84.save_npz(t, table); LD84.save_npz(r, gone)
            c = LD84.counts_from_files(t, r)
        self.assertEqual(c["table"]["rows"], 3)
        self.assertEqual(c["table"]["labelled"], 2)
        self.assertEqual(c["table"]["unlabelled"], 1)
        self.assertEqual(c["removed"][LD84.NOT_SHUFFLED]["teams"], 1)
        self.assertEqual(c["removed"][LD84.NOT_SHUFFLED]["by_family"]["oldpool"], 1)
        self.assertEqual(c["removed"][LD84.ILLEGAL]["teams"], 1)
        self.assertEqual(c["lost_label"][LD84.LD.RUN_FILE]["teams"], 1)
        self.assertEqual(c["lost_label"][LD84.LD.LISTED_FALLBACK]["teams"], 1)


if __name__ == "__main__":
    unittest.main()
