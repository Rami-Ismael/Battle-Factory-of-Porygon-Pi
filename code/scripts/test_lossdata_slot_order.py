"""Which battles give a team in the matchup database its win rate in the training table (lossdata.gather).

Shuffled slot order has been the battle default since 2026-10-28, so a team battled only that way must still get a
win rate. Each team's label comes from one slot order: shuffled when it covers enough top-50 opponents, else listed.
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


@unittest.skipUnless(LD is not None, "the /tmp/vgc-pilot runtime is not available")
class LabelSlotOrder(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
