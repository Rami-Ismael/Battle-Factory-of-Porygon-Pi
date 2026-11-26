"""Run: python scripts/test_rust_matchup_runner.py   (no battles, no laptop data; CI runs it).

The Rust search labeler (src/rust_matchup_runner.py, issue #418) behind matchup_db.ensure, with
the node battle driver replaced by a fake that reports outcomes. Each test fails on one planted
bug: battles recorded under a behaviour-cloning policy, a cached cell battled again, a top-up that
replays earlier seeds, sides that never alternate, finished battles lost when a run dies, or a
failed battle that passes silently."""
import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import heldout as H
import matchup_db as M
import rust_matchup_runner as R

PROV = {k: k * 32 for k in ("engine", "dex", "spreads", "agent", "runtime")}   # 64-character stand-in hashes
STATS = {"decisions": 9, "iterations": 270000, "fallbacks": 0, "failedSamples": 0}


def paste(name):
    return "\n\n".join(f"{name}{i} @ Item\nAbility: A\n- Move" for i in range(6)) + "\n"


class FakePlay:
    """Stands in for the node processes: p1 wins every battle unless told otherwise."""
    def __init__(self, die_after=None, error_ids=(), fallbacks=0):
        self.games, self.die_after, self.error_ids, self.fallbacks = [], die_after, set(error_ids), fallbacks

    def __call__(self, games, spec, on_done):
        for g in games:
            if self.die_after is not None and len(self.games) == self.die_after:
                raise RuntimeError("runner killed")
            self.games.append(g)
            if g["id"] in self.error_ids:
                on_done(g, None, "RustRunnerError: poke-engine-doubles exited with code 1")
            else:
                stats = {"p1": dict(STATS, fallbacks=self.fallbacks), "p2": dict(STATS, rejected=["|error|[Unavailable"])}
                on_done(g, {"winner": "p1", "turns": 7, "ms": 250, "stats": stats}, None)


class RustRunner(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.TemporaryDirectory()
        self.con = M.connect(Path(self.d.name) / "t.sqlite")
        for i in range(8):                      # the behaviour-cloning rows that already exist on the laptop
            M.get_policy(self.con, f"bc_100.zip:{i}", f"bc_100.zip:{i}", "showdown:unknown")
        with self.con:
            self.ids = [self.con.execute(
                "INSERT INTO team (format_id, canon_hash, paste, species_key, base_key, origin, grid_hash, legal) "
                "VALUES ('f', ?, ?, ?, ?, 'test', ?, 1)", (n, paste(n), n, n, f"grid-{n}")).lastrowid for n in "abcdef"]
        self.rows, self.cols = self.ids[:2], self.ids[2:5]
        self.held = mock.patch.object(H, "identities", lambda: frozenset({f"grid-{'f'}"}))
        self.held.start()

    def tearDown(self):
        self.held.stop(); self.con.close(); self.d.cleanup()

    def runner(self, play, **kw):
        return R.RustMatchupRunner(play=play, prov=PROV, evidence=Path(self.d.name) / "ev", **kw)

    def test_runner_takes_shuffled_slots_and_its_own_isolated_policy(self):
        r = self.runner(FakePlay())
        self.assertEqual(r.slot_order, "shuffled")
        pid = r.register(self.con)
        self.assertNotIn(pid, range(1, 9))
        row, col, sim = self.con.execute("SELECT row_policy, col_policy, simulator FROM policy WHERE policy_id=?", (pid,)).fetchone()
        self.assertEqual(row, col)
        for part in ("rust_foulplay_det", "iterations=30000", "samples=1", "engine=" + "engine" * 2 + "engi",
                     "dex=", "spreads=", "+slots-shuffled"):
            self.assertIn(part, row)
        self.assertEqual(sim, "showdown-runtime:" + PROV["runtime"])
        self.assertNotEqual(pid, R.get_rust_policy(self.con, "rust_uct_rollout", prov=PROV))
        self.assertEqual(pid, r.register(self.con))

    def test_listed_slot_order_is_refused(self):
        with self.assertRaises(ValueError):
            R.get_rust_policy(self.con, slot_order="listed", prov=PROV)

    def test_second_ensure_runs_no_battles(self):
        play = FakePlay(); r = self.runner(play); pid = r.register(self.con)
        self.assertEqual(M.ensure(self.con, pid, self.rows, self.cols, 2, seed=5, runner=r), 2 * 3 * 2)
        self.assertEqual(M.ensure(self.con, pid, self.rows, self.cols, 2, seed=5, runner=r), 0)
        self.assertEqual(len(play.games), 12)
        n, cells = self.con.execute("SELECT SUM(battles), COUNT(*) FROM cell WHERE policy_id=?", (pid,)).fetchone()
        self.assertEqual((n, cells), (12, 6))
        self.assertEqual(self.con.execute("SELECT SUM(battles) FROM contribution").fetchone()[0], 12)

    def test_heldout_team_still_raises(self):
        r = self.runner(FakePlay()); pid = r.register(self.con)
        with self.assertRaises(H.HeldoutLeak):
            M.ensure(self.con, pid, [self.ids[5]], self.cols, 2, runner=r)

    def test_sides_alternate_and_top_ups_play_new_seeds(self):
        play = FakePlay(); r = self.runner(play); pid = r.register(self.con)
        M.ensure(self.con, pid, self.rows[:1], self.cols[:1], 2, seed=5, runner=r)
        M.ensure(self.con, pid, self.rows[:1], self.cols[:1], 4, seed=5, runner=r)
        lo, hi = sorted((self.rows[0], self.cols[0]))
        self.assertEqual([g["game"] for g in play.games], [0, 1, 2, 3])
        self.assertEqual([g["p1_team"] for g in play.games], [lo, hi, lo, hi])
        self.assertEqual(len({g["seed"] for g in play.games}), 4)
        # p1 won every game, so each team won the two it played as p1
        self.assertEqual(self.con.execute("SELECT lo_wins, hi_wins, ties, battles FROM cell WHERE policy_id=?", (pid,)).fetchone(),
                         (2, 2, 0, 4))
        again = R.make_games([{"row": self.rows[0], "schedule": [self.cols[0]]}], dict(self.con.execute("SELECT team_id, paste FROM team")),
                             {(lo, hi): 0}, 5)[0]
        self.assertEqual((again["seed"], again["a"]), (play.games[0]["seed"], play.games[0]["a"]))

    def test_slots_are_shuffled_per_game(self):
        play = FakePlay(); r = self.runner(play); pid = r.register(self.con)
        M.ensure(self.con, pid, self.rows[:1], self.cols[:1], 6, seed=5, runner=r)
        firsts = {g["a"].split(" @")[0] for g in play.games}
        self.assertGreater(len(firsts), 1)

    def test_finished_battles_survive_a_killed_run(self):
        r = self.runner(FakePlay(die_after=5), flush_every=2); pid = r.register(self.con)
        with self.assertRaises(RuntimeError):
            M.ensure(self.con, pid, self.rows, self.cols, 2, seed=5, runner=r)
        self.assertEqual(self.con.execute("SELECT SUM(battles) FROM cell WHERE policy_id=?", (pid,)).fetchone()[0], 5)
        self.assertEqual([b for (b,) in self.con.execute("SELECT battles FROM batch WHERE policy_id=? ORDER BY batch_id", (pid,))],
                         [2, 2, 1])
        play = FakePlay(); r2 = self.runner(play)
        self.assertEqual(M.ensure(self.con, pid, self.rows, self.cols, 2, seed=5, runner=r2), 7)

    def test_a_failed_game_index_is_never_reused(self):
        lo, hi = sorted((self.rows[0], self.cols[0]))
        r = self.runner(FakePlay(error_ids={f"{lo}-{hi}-0"})); pid = r.register(self.con)
        with self.assertRaises(R.RustRunnerError):           # game 0 fails, game 1 is recorded
            M.ensure(self.con, pid, self.rows[:1], self.cols[:1], 2, seed=5, runner=r)
        play = FakePlay(); M.ensure(self.con, pid, self.rows[:1], self.cols[:1], 2, seed=5, runner=self.runner(play))
        self.assertEqual([g["game"] for g in play.games], [2])     # index 1 again would replay a recorded game

    def test_failed_battle_is_recorded_as_unattributed_and_raises(self):
        lo, hi = sorted((self.rows[0], self.cols[0]))
        r = self.runner(FakePlay(error_ids={f"{lo}-{hi}-0"}, fallbacks=1)); pid = r.register(self.con)
        with self.assertRaises(R.RustRunnerError):
            M.ensure(self.con, pid, self.rows, self.cols, 2, seed=5, runner=r, note="tracer")
        battles, unattributed, note = self.con.execute(
            "SELECT battles, unattributed, note FROM batch WHERE policy_id=?", (pid,)).fetchone()
        self.assertEqual((battles, unattributed), (11, 1))
        self.assertTrue(note.startswith("tracer; rust_foulplay_det"))
        self.assertIn("fallbacks=11", note); self.assertIn("rejected=11", note); self.assertIn("errors=1", note)

    def test_a_rust_policy_takes_no_other_runner_and_a_runner_no_other_policy(self):
        r = self.runner(FakePlay()); pid = r.register(self.con)
        bc_like = lambda con, jobs, seed=None: ([], 0, 0.0)
        bc_like.slot_order = "shuffled"
        with self.assertRaises(ValueError):
            M.ensure(self.con, pid, self.rows, self.cols, 2, runner=bc_like)
        with self.assertRaises(ValueError):
            M.ensure(self.con, 8, self.rows, self.cols, 2, runner=r)
        self.assertIsNone(self.con.execute("SELECT 1 FROM cell").fetchone())

    def test_missing_engine_fails_before_any_battle(self):
        with mock.patch.object(R, "BIN", Path(self.d.name) / "no-such-binary"):
            with self.assertRaises(FileNotFoundError):
                R.RustMatchupRunner(evidence=Path(self.d.name) / "ev").register(self.con)

    def test_score_pastes_uses_the_runner_policy(self):
        play = FakePlay(); r = self.runner(play)
        names = {paste(n): t for n, t in zip("abcdef", self.ids)}
        with mock.patch.object(M, "add_team", lambda con, p, origin, val=None: (names[p], False)), \
             mock.patch.object(M, "Validator", lambda: mock.Mock()):
            with self.con:
                for t in self.cols: self.con.execute("INSERT INTO team_set VALUES ('top50', ?, 1)", (t,))
            out = M.score_pastes(self.con, [paste("a"), paste("f")], per_cell=2, seed=5, runner=r)
        pid = r.register(self.con)
        self.assertEqual(out[0]["coverage"], 3); self.assertTrue(out[1]["heldout"])
        self.assertEqual(self.con.execute("SELECT DISTINCT policy_id FROM cell").fetchall(), [(pid,)])
        self.assertEqual(len(play.games), 6)

    def test_evidence_lists_every_battle(self):
        r = self.runner(FakePlay()); pid = r.register(self.con)
        M.ensure(self.con, pid, self.rows, self.cols, 1, seed=5, runner=r)
        rows = [json.loads(x) for x in (Path(self.d.name) / "ev" / "battles.jsonl").read_text().splitlines()]
        self.assertEqual(len(rows), 6)
        self.assertEqual(json.loads((Path(self.d.name) / "ev" / "summary.json").read_text())["battles"], 6)


FAKE_WORKER = r"""
import json, os, sys, time
print(json.dumps({"kind": "ready"}), flush=True)
for line in sys.stdin:
    m = json.loads(line)
    if m["id"].startswith("crash"): os._exit(3)
    if m["id"].startswith("hang"): time.sleep(60); continue
    print("not json", flush=True)                      # stray stdout line
    print(json.dumps({"kind": "result", "id": m["id"], "winner": "p1", "turns": 3, "stats": {}}), flush=True)
"""


class NodeDriver(unittest.TestCase):
    """play_node against a fake worker process that answers, crashes or hangs by game id."""
    def play(self, ids, workers=2, timeout=30):
        d = tempfile.TemporaryDirectory(); self.addCleanup(d.cleanup)
        worker = Path(d.name) / "worker.py"; worker.write_text(FAKE_WORKER)
        done = {}
        games = [{"id": i, "a": "", "b": "", "seed": 1} for i in ids]
        import asyncio
        asyncio.run(R.play_node(games, {}, lambda g, r, e: done.__setitem__(g["id"], e or r["winner"]),
                                workers=workers, timeout=timeout, command=(sys.executable, str(worker))))
        return done

    def test_a_crashing_worker_fails_only_its_own_battle(self):
        done = self.play(["crash-0"] + [f"ok-{i}" for i in range(6)])
        self.assertIn("stopped", done["crash-0"])
        self.assertEqual({k: v for k, v in done.items() if k != "crash-0"}, {f"ok-{i}": "p1" for i in range(6)})

    def test_a_timed_out_worker_is_retired_and_the_rest_play_on_live_workers(self):
        t0 = __import__("time").perf_counter()
        done = self.play(["hang-0"] + [f"ok-{i}" for i in range(6)], timeout=1)
        self.assertIn("TimeoutError", done["hang-0"])
        self.assertEqual({k: v for k, v in done.items() if k != "hang-0"}, {f"ok-{i}": "p1" for i in range(6)})
        self.assertLess(__import__("time").perf_counter() - t0, 15)     # one timeout, not one per queued game

    def test_battles_left_when_every_worker_is_gone_fail_at_once(self):
        done = self.play(["crash-0", "crash-1", "ok-0"], workers=2)
        self.assertIn("no live battle runner", done["ok-0"])


class Provenance(unittest.TestCase):
    def test_the_format_rules_are_in_the_runtime_hash(self):
        with tempfile.TemporaryDirectory() as d:
            rt = Path(d)
            for f in ("package.json", "dist/sim/battle.js", "dist/data/moves.js", "dist/config/formats.js"):
                (rt / f).parent.mkdir(parents=True, exist_ok=True); (rt / f).write_text("x")
            before = R.runtime_hash(rt)
            (rt / "dist/config/formats.js").write_text("y")
            self.assertNotEqual(before, R.runtime_hash(rt))

    def test_two_jobs_cannot_claim_the_same_game_index(self):
        with tempfile.TemporaryDirectory() as d:
            con = M.connect(Path(d) / "t.sqlite")
            other = sqlite3.connect(Path(d) / "t.sqlite", timeout=0.1)
            def write_meanwhile(*a):
                other.execute("INSERT INTO rust_next_game VALUES (1, 1, 2, 0)")    # a second job claiming now
                return 0
            with mock.patch.object(M, "battles_done", write_meanwhile):
                with self.assertRaises(sqlite3.OperationalError):          # locked: the claim holds the write lock
                    R.claim_games(con, 1, {(1, 2): 2})
            con.close(); other.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
