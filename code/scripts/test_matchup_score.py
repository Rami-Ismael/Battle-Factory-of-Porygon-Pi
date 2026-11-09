"""Run: python scripts/test_matchup_score.py   (no battles, no laptop data; CI runs it).

matchup_db.score is a team's win rate against a team set, the number every result rests on.
These tests use uneven records and uneven weights, so a wrong version gives a different number:
the row team credited with its opponent's wins, set weights ignored, or a standard error that
does not shrink with the number of battles. Each test fails on one such planted bug."""
import math
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import matchup_db as M


class Score(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.TemporaryDirectory()
        self.con = M.connect(Path(self.d.name) / "t.sqlite")
        self.pid = M.get_policy(self.con, "p", "p", "sim")
        with self.con:
            self.b, self.c, self.team = [self.con.execute(
                "INSERT INTO team (format_id, canon_hash, paste, species_key, base_key, origin) "
                "VALUES ('f', ?, ?, ?, ?, 'test')", (name, name, name, name)).lastrowid for name in "bcz"]
        # The scored team has the highest id, so it is the hi side of both cells: 6-2 against b, 5-3 against c.
        M.record(self.con, self.pid, [(self.team, self.b, 6, 2, 0), (self.team, self.c, 5, 3, 0)])

    def tearDown(self):
        self.con.close(); self.d.cleanup()

    def set_weights(self, wb, wc):
        with self.con:
            self.con.executemany("INSERT INTO team_set VALUES ('top50', ?, ?)", [(self.b, wb), (self.c, wc)])

    def score(self):
        return M.score(self.con, self.pid, self.team, "top50", held=frozenset())

    def test_the_row_team_is_credited_with_its_own_wins(self):
        self.set_weights(1, 1)
        self.assertAlmostEqual(self.score()["score"], (6 / 8 + 5 / 8) / 2)     # 0.6875; its opponents' wins give 0.3125

    def test_set_weights_count(self):
        self.set_weights(3, 1)
        self.assertAlmostEqual(self.score()["score"], (3 * 6 / 8 + 1 * 5 / 8) / 4)   # 0.71875; unweighted gives 0.6875

    def test_standard_error_is_that_of_a_weighted_mean_of_binomial_rates(self):
        self.set_weights(1, 1)
        se = math.sqrt(0.5 ** 2 * (6 / 8) * (2 / 8) / 8 + 0.5 ** 2 * (5 / 8) * (3 / 8) / 8)
        self.assertAlmostEqual(self.score()["se"], se)


if __name__ == "__main__":
    unittest.main(verbosity=2)
