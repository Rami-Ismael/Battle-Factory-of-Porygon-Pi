"""Run: /tmp/vgc-pilot/.venv/bin/python scripts/test_rust_matchup_live.py   (laptop only: real battles).

One live tracer batch for the Rust search labeler (issue #418): 4 VGCPastes teams x 4 top-50 meta
teams x 2 games = 32 battles of Foul Play deterministic argmax (30,000 iterations, 1 sample) on a
temporary database that holds copies of those teams. The real matchup_regmb.sqlite is only read.
Needs the built engine (cargo build --release in rust/poke-engine-doubles), the Showdown runtime,
the pinned validator and the VGCPastes pool under /tmp/vgc-pilot. Battle evidence stays under
~/vgc-data/rust-matchup/."""
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import heldout as H
import matchup_db as M
import rust_matchup_runner as R

COLS = ("team_id", "format_id", "canon_hash", "grid_hash", "paste", "species_key", "base_key", "legal", "legal_error", "origin")


class LiveTracer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d = tempfile.TemporaryDirectory()
        src = sqlite3.connect(f"file:{M.DEFAULT_DB}?mode=ro", uri=True)
        held = set(H.identities())
        top = [t for (t,) in src.execute("SELECT team_id FROM team_set WHERE set_name='top50' ORDER BY team_id")]
        cols = top[:4]
        rows = [t for t, g in src.execute(
            "SELECT team_id, grid_hash FROM team WHERE legal=1 AND origin='vgcpastes' AND team_id NOT IN "
            "(SELECT team_id FROM team_set) ORDER BY team_id") if g not in held][:4]
        heldout = [t for t, g in src.execute("SELECT team_id, grid_hash FROM team") if g in held][:1]
        cls.con = M.connect(Path(cls.d.name) / "tracer.sqlite")
        with cls.con:
            for p in src.execute("SELECT * FROM policy ORDER BY policy_id"):   # keep policy_id 1-8 taken, as on the laptop
                cls.con.execute("INSERT INTO policy VALUES (?,?,?,?,?,?)", p)
            for t in rows + cols + heldout:
                v = src.execute(f"SELECT {','.join(COLS)} FROM team WHERE team_id=?", (t,)).fetchone()
                cls.con.execute(f"INSERT INTO team ({','.join(COLS)}) VALUES ({','.join('?' * len(COLS))})", v)
            cls.con.executemany("INSERT INTO team_set VALUES ('top50', ?, 1)", [(c,) for c in cols])
        src.close()
        cls.rows, cls.cols, cls.heldout = rows, cols, heldout
        cls.runner = R.RustMatchupRunner("rust_foulplay_det", samples=1, iterations=30000, workers=8, concurrency=1)
        cls.pid = cls.runner.register(cls.con)
        cls.first = M.ensure(cls.con, cls.pid, rows, cols, 2, seed=418, runner=cls.runner, note="tracer #418")
        cls.stats = dict(cls.runner.last)
        print(f"\ntracer: policy {cls.pid}, {cls.first} battles, {cls.stats}", flush=True)

    @classmethod
    def tearDownClass(cls):
        cls.con.close(); cls.d.cleanup()

    def test_tracer_ran_32_battles_with_no_failures_or_fallbacks(self):
        self.assertEqual((len(self.rows), len(self.cols)), (4, 4))
        self.assertEqual(self.first, 32)
        self.assertEqual((self.stats["errors"], self.stats["fallbacks"]), (0, 0))
        batches = self.con.execute("SELECT battles, unattributed, note FROM batch WHERE policy_id=?", (self.pid,)).fetchall()
        self.assertEqual(sum(b for b, _, _ in batches), 32)
        self.assertTrue(all(u == 0 and "fallbacks=0" in n and "errors=0" in n for _, u, n in batches))

    def test_cells_and_contributions_are_consistent(self):
        cells = self.con.execute("SELECT lo, hi, lo_wins + hi_wins + ties, battles FROM cell WHERE policy_id=?", (self.pid,)).fetchall()
        self.assertEqual(len(cells), 16)
        self.assertTrue(all(s == n == 2 for _, _, s, n in cells))
        contributed = self.con.execute(
            "SELECT SUM(c.battles) FROM contribution c JOIN batch b USING (batch_id) WHERE b.policy_id=?", (self.pid,)).fetchone()[0]
        self.assertEqual(contributed, 32)

    def test_policy_is_new_and_carries_provenance(self):
        self.assertGreater(self.pid, 8)
        row, sim, note = self.con.execute("SELECT row_policy, simulator, note FROM policy WHERE policy_id=?", (self.pid,)).fetchone()
        self.assertTrue(row.startswith("rust_foulplay_det:iterations=30000:samples=1:engine="))
        self.assertTrue(row.endswith("+slots-shuffled"))
        self.assertTrue(sim.startswith("showdown-runtime:"))
        self.assertIn('"engine"', note)

    def test_second_call_is_fully_cached(self):
        self.assertEqual(M.ensure(self.con, self.pid, self.rows, self.cols, 2, seed=418, runner=self.runner), 0)

    def test_score_pastes_reads_the_rust_cells(self):
        paste = self.con.execute("SELECT paste FROM team WHERE team_id=?", (self.rows[0],)).fetchone()[0]
        out = M.score_pastes(self.con, [paste], per_cell=2, seed=418, runner=self.runner)
        self.assertEqual((out[0]["team_id"], out[0]["coverage"], out[0]["battles"]), (self.rows[0], 4, 8))

    def test_heldout_team_raises(self):
        self.assertEqual(len(self.heldout), 1)
        with self.assertRaises(H.HeldoutLeak):
            M.ensure(self.con, self.pid, self.heldout, self.cols, 2, runner=self.runner)


if __name__ == "__main__":
    unittest.main(verbosity=2)
