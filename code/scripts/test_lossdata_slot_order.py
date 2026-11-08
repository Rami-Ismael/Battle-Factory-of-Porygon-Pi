"""Which battles give a team in the matchup database its win rate in the training table (lossdata.gather).

Shuffled slot order has been the battle default since 2026-10-28, so a team battled only that way must still get a
win rate. Each team's label comes from one slot order: shuffled when it covers enough top-50 opponents, else listed.
With shuffled_only (issue #229) only shuffled-slot battles label a team, and every other label is recorded as dropped.
Needs this laptop's /tmp/vgc-pilot runtime (lossdata imports the data modules); skipped without it.

    /tmp/vgc-pilot/.venv/bin/python scripts/test_lossdata_slot_order.py   (from code/)
"""
import tempfile
import unittest
from pathlib import Path

import numpy as np

import repo_paths  # noqa: F401  (src/ first on sys.path)

LD = None
if Path("/tmp/vgc-pilot/data/pokedex.json").exists():
    import lossdata as LD

TOP = list(range(1, 42))                       # 41 top-50 teams, so a team outside them can cover 41 opponents
LISTED, SHUFFLED = "bc_100.zip:abc", "bc_100.zip:abc+slots-shuffled"


class FakeDatabase:
    """A matchup database with shuffled-only, listed-only and mixed teams against 41 top-50 teams."""
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.con = LD.MDB.connect(Path(self.tmp.name) / "m.sqlite")
        with self.con:
            for tid in [*TOP, 100, 101, 102, 103, 104]:
                self.con.execute("INSERT INTO team (team_id, format_id, canon_hash, paste, species_key, base_key, legal, origin) "
                                 "VALUES (?, 'f', ?, 'p', 's', 'b', 1, 'test')", (tid, f"h{tid}"))
            self.con.executemany("INSERT INTO team_set (set_name, team_id) VALUES ('top50', ?)", [(t,) for t in TOP])
            self.con.execute("INSERT INTO policy VALUES (1, 'f', ?, ?, 'showdown:a', NULL)", (LISTED, LISTED))
            self.con.execute("INSERT INTO policy VALUES (8, 'f', ?, ?, 'showdown:b', NULL)", (SHUFFLED, SHUFFLED))
        self.battle(100, 8, TOP, wins=3, n=4)            # shuffled only
        self.battle(101, 1, TOP, wins=1, n=4)            # listed only
        self.battle(102, 1, TOP, wins=4, n=4)            # both: listed says 1.0, shuffled says 0.0
        self.battle(102, 8, TOP, wins=0, n=4)
        self.battle(103, 1, TOP, wins=2, n=4)            # listed covers 41, shuffled only 10
        self.battle(103, 8, TOP[:10], wins=4, n=4)
        self.battle(104, 8, TOP[:10], wins=4, n=4)       # too few opponents in either order

    def tearDown(self):
        self.con.close(); self.tmp.cleanup()

    def battle(self, team, policy_id, opponents, wins, n):
        with self.con:
            self.con.executemany("INSERT INTO cell VALUES (?, ?, ?, ?, ?, 0, ?, 'now')",
                                 [(policy_id, op, team, n - wins, wins, n) for op in opponents])



@unittest.skipUnless(LD is not None, "the /tmp/vgc-pilot runtime is not available")
class LabelSlotOrder(FakeDatabase, unittest.TestCase):
    def label(self, team):
        rows = LD.top50_rows(self.con, TOP)
        return LD.matrix_label(rows[team], None, np.ones(len(TOP), np.float32)) if team in rows else (None, None)

    def test_a_team_battled_only_in_shuffled_order_gets_its_win_rate(self):
        self.assertEqual(self.label(100), (0.75, 164))

    def test_a_team_battled_only_in_listed_order_keeps_its_win_rate(self):
        self.assertEqual(self.label(101), (0.25, 164))

    def test_shuffled_battles_win_over_listed_when_both_cover_enough_opponents(self):
        self.assertEqual(self.label(102), (0.0, 164))

    def test_listed_battles_are_used_when_shuffled_ones_cover_too_few_opponents(self):
        self.assertEqual(self.label(103), (0.5, 164))

    def test_a_team_with_too_few_opponents_in_either_order_has_no_win_rate(self):
        self.assertEqual(self.label(104), (None, None))

    def test_the_two_orders_are_never_pooled_into_one_row(self):
        row = LD.top50_rows(self.con, TOP)[103]
        self.assertEqual(row[:, 1].sum(), 164)

    def test_each_row_records_the_slot_order_it_came_from(self):
        sources = {}
        LD.top50_rows(self.con, TOP, sources)
        self.assertEqual({t: sources[t] for t in (100, 101, 102, 103)},
                         {100: LD.SHUFFLED, 101: LD.LISTED_FALLBACK, 102: LD.SHUFFLED, 103: LD.LISTED_FALLBACK})


@unittest.skipUnless(LD is not None, "the /tmp/vgc-pilot runtime is not available")
class ShuffledOnlyLabels(FakeDatabase, unittest.TestCase):
    """gather(shuffled_only=True) credits each team through top50_labels and credit, as entry() does here."""

    def entry(self, team, run_file_label=None):
        e = dict(wins=0.0, battles=0)
        labels = LD.top50_labels(self.con, TOP, np.ones(len(TOP), np.float32))
        if team in labels:
            _, y, n, source = labels[team]
            LD.credit(e, y, n, source, shuffled_only=True)
        if run_file_label is not None: LD.credit(e, *run_file_label, LD.RUN_FILE, shuffled_only=True)
        return e

    def test_a_team_battled_only_in_shuffled_order_keeps_its_win_rate(self):
        e = self.entry(100)
        self.assertEqual((e["wins"] / e["battles"], e["battles"]), (0.75, 164))
        self.assertNotIn("dropped", e)

    def test_a_team_battled_only_in_listed_order_has_no_label_and_its_listed_label_is_dropped(self):
        e = self.entry(101)
        self.assertEqual(e["battles"], 0)
        self.assertEqual(e["dropped"], {LD.LISTED_FALLBACK})

    def test_a_mixed_team_with_enough_shuffled_battles_keeps_only_the_shuffled_win_rate(self):
        e = self.entry(102)
        self.assertEqual((e["wins"], e["battles"]), (0.0, 164))
        self.assertNotIn("dropped", e)

    def test_a_mixed_team_with_too_few_shuffled_battles_loses_its_listed_label(self):
        e = self.entry(103)
        self.assertEqual(e["battles"], 0)
        self.assertEqual(e["dropped"], {LD.LISTED_FALLBACK})

    def test_a_team_with_too_few_opponents_has_nothing_to_drop(self):
        self.assertEqual(self.entry(104), dict(wins=0.0, battles=0))

    def test_a_run_file_label_is_dropped_even_beside_a_shuffled_label(self):
        e = self.entry(100, run_file_label=(0.1, 24))
        self.assertEqual((e["wins"] / e["battles"], e["battles"]), (0.75, 164))
        self.assertEqual(e["dropped"], {LD.RUN_FILE})

    def test_every_label_left_equals_the_shuffled_battles_alone(self):
        shuffled = LD.policy_rows(self.con, TOP, [8])[8]
        for team in (100, 101, 102, 103, 104):
            e = self.entry(team, run_file_label=(1.0, 24))
            r = shuffled.get(team)
            y, n = LD.matrix_label(r, None, np.ones(len(TOP), np.float32)) if r is not None else (None, None)
            self.assertEqual((e["wins"] / e["battles"], e["battles"]) if e["battles"] else (None, None), (y, n), team)

    def test_without_shuffled_only_the_listed_fallback_still_labels(self):
        e = dict(wins=0.0, battles=0)
        _, y, n, source = LD.top50_labels(self.con, TOP, np.ones(len(TOP), np.float32))[101]
        LD.credit(e, y, n, source)
        self.assertEqual((e["wins"] / e["battles"], e["battles"]), (0.25, 164))


if __name__ == "__main__":
    unittest.main()
