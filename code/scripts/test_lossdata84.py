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


NONE_WRONG = lambda rows: np.zeros(len(rows), bool)                    # no row holds a wrong Mega Stone


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
        table, gone, errors, kept = LD84.shuffled_tables(dict(pool=pool), sets, check, NONE_WRONG)
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


# A legal team in the regulation vocabulary (lossdata84_shuffled_only.npz): Basculegion, Excadrill, Raichu, Sinistcha,
# Staraptor holding Staraptite and Tyranitar holding Tyranitarite.
LEGAL = np.array([45, 1, 32, 18, 165, 243, 482, 2, 102, 138, 53, 123, 235, 320, 350, 2, 233, 83, 126, 68, 131, 143, 476, 26,
                  270, 60, 129, 249, 264, 334, 466, 6, 284, 75, 135, 52, 75, 320, 438, 13, 310, 140, 141, 239, 254, 320, 350, 2,
                  2, 19, 15, 1, 2, 33, 2, 33, 2, 1, 1, 33, 18, 1, 18, 1, 1, 33, 33, 1, 5, 1, 31, 1, 33, 5, 1, 1, 1, 31, 28, 27,
                  2, 1, 1, 13], np.int16)


@unittest.skipUnless(LD84 is not None, "the /tmp/vgc-pilot runtime is not available")
class WrongMegaStone(unittest.TestCase):
    """Issue #368: a team holding a Mega Stone on a Pokémon it does not belong to leaves both sides of the split,
    and every other team keeps its side."""

    @classmethod
    def setUpClass(cls):
        cls.V = LD84.T.regulation_vocab()
        cls.rejects = staticmethod(LD84.table_rejects(cls.V))

    def swapped_stones(self):
        """LEGAL with Staraptor and Tyranitar trading stones: each holds the other's Mega Stone."""
        row, nf = LEGAL.copy(), LD84.T.NF
        row[4 * nf + 2], row[5 * nf + 2] = LEGAL[5 * nf + 2], LEGAL[4 * nf + 2]
        names = [s["item"] for s in LD84.T.decode(self.V, row)]
        self.assertEqual(names[4:], ["tyranitarite", "staraptite"])
        return row

    def test_a_legal_team_passes_the_tables(self):
        self.assertFalse(self.rejects(LEGAL[None])[0])

    def test_a_stone_on_the_wrong_pokemon_is_rejected(self):
        self.assertTrue(self.rejects(self.swapped_stones()[None])[0])

    def test_the_tables_check_counts_each_side(self):
        rows = np.stack([LEGAL, self.swapped_stones()])
        self.assertEqual(LD84.tables_check_by_side(self.V, rows[:1], np.array([False])), dict(training=0, validation=0))
        self.assertEqual(LD84.tables_check_by_side(self.V, rows, np.array([False, True]))["training"], 0)

    def test_wrong_stone_teams_leave_and_every_other_team_keeps_its_side(self):
        n = 200                                                                   # 2% validation: 4 teams
        rows = {i.to_bytes(2, "big"): ROW.copy() for i in range(n)}
        for i, r in enumerate(rows.values()): r[0] = i + 1
        pool = {k: entry(battles=49) for k in rows}
        sets = {k: {(0,): r} for k, r in rows.items()}
        legal = lambda row: None
        before, *_ = LD84.shuffled_tables(dict(pool=pool), sets, legal, NONE_WRONG)
        side = dict(zip(before["all_X"][:, 0].tolist(), before["all_val"].tolist()))
        val_id, train_id = next(i for i in side if side[i]), next(i for i in side if not side[i])
        wrong = lambda X: np.isin(np.asarray(X)[:, 0], [val_id, train_id])     # one team from each side holds a wrong stone
        table, gone, _, kept = LD84.shuffled_tables(dict(pool=pool), sets, legal, wrong)
        self.assertEqual(len(kept), n - 2)
        self.assertEqual(sorted(gone["X"][:, 0].tolist()), sorted([val_id, train_id]))
        self.assertEqual(gone["reason"].tolist(), [LD84.REMOVED.index(LD84.WRONG_STONE)] * 2)
        self.assertEqual(dict(zip(table["all_X"][:, 0].tolist(), table["all_val"].tolist())),
                         {i: v for i, v in side.items() if i not in (val_id, train_id)})
        with tempfile.TemporaryDirectory() as d:
            t, r = Path(d) / "t.npz", Path(d) / "r.npz"
            LD84.save_npz(t, table); LD84.save_npz(r, gone)
            c = LD84.counts_from_files(t, r)
        self.assertEqual(c["removed"][LD84.WRONG_STONE]["teams"], 2)
        self.assertEqual(c["table"]["val"], int(before["all_val"].sum()) - 1)

    def test_without_wrong_stones_keeps_the_drawn_split(self):
        val = np.array([True, False, True, False])
        keys, kept_val, _, wrong = LD84.without_wrong_stones(list("abcd"), val, np.arange(4)[:, None],
                                                                 lambda X: X[:, 0] % 2 == 0)
        self.assertEqual(keys, ["b", "d"])
        self.assertEqual(kept_val.tolist(), [False, False])
        self.assertEqual(wrong.tolist(), [True, False, True, False])


@unittest.skipUnless(LD84 is not None, "the /tmp/vgc-pilot runtime is not available")
class DropFromAStoredTable(unittest.TestCase):
    """A stored table loses only its wrong-stone rows (owner, 2026-11-17): no rebuild, no new teams, same split."""

    def stored(self):
        X = np.stack([ROW.copy() for _ in range(5)]); X[:, 0] = np.arange(1, 6)
        return dict(all_X=X, all_W=np.array([0.5, np.nan, 0.4, 0.3, 0.2], np.float32), all_NB=np.arange(5, dtype=np.int32),
                    all_val=np.array([False, True, True, False, False]), all_S=np.arange(5, dtype=np.int32),
                    all_lost=np.zeros(5, np.int8), loop_X=X[:2], loop_W=np.array([0.1, 0.2], np.float32))

    def test_every_array_of_a_group_loses_the_same_rows_and_the_rest_keep_their_side(self):
        a = self.stored()
        kept, wrong = LD84.drop_wrong_stones(a, lambda X: np.isin(X[:, 0], [2, 4]))
        self.assertEqual(kept["all_X"][:, 0].tolist(), [1, 3, 5])
        self.assertEqual(kept["all_val"].tolist(), [False, True, False])
        self.assertEqual(kept["all_NB"].tolist(), [0, 2, 4])
        self.assertEqual(kept["loop_X"][:, 0].tolist(), [1])
        self.assertEqual(kept["loop_W"].tolist(), [np.float32(0.1)])
        self.assertEqual({p: w.tolist() for p, w in wrong.items()}, {"all": [False, True, False, True, False], "loop": [False, True]})

    def test_the_dropped_rows_join_the_removed_teams_under_their_reason(self):
        a = self.stored()
        _, wrong = LD84.drop_wrong_stones(a, lambda X: X[:, 0] == 2)
        gone = dict(X=a["all_X"][:1], reason=np.array([0], np.int8), S=np.array([7], np.int32), lost=np.array([1], np.int8))
        r = LD84.with_removed(gone, a, wrong)
        self.assertEqual(r["X"][:, 0].tolist(), [1, 2])
        self.assertEqual(r["reason"].tolist(), [0, LD84.REMOVED.index(LD84.WRONG_STONE)])
        self.assertEqual(r["S"].tolist(), [7, 1])

    def test_a_table_without_wrong_stones_is_unchanged(self):
        a = self.stored()
        kept, wrong = LD84.drop_wrong_stones(a, NONE_WRONG)
        self.assertFalse(any(w.any() for w in wrong.values()))
        for k in a: np.testing.assert_array_equal(kept[k], a[k])


if __name__ == "__main__":
    unittest.main()
