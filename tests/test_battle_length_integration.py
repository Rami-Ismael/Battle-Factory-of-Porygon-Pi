"""Opt-in real Showdown battles through the existing evaluator and SQLite cache."""

import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import unittest
from unittest.mock import patch

from test_battle_length_example import load_summarizer, result


ROOT = Path(__file__).resolve().parents[1]
PIPELINE = ROOT / "notes" / "team_completion_pipeline"
RUNTIME = os.environ.get("VGC_SHOWDOWN_RUNTIME")


@unittest.skipUnless(RUNTIME, "Set VGC_SHOWDOWN_RUNTIME to a built Champions Showdown runtime")
class RealBattleTurnCountTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not shutil.which("node"):
            raise RuntimeError("The requested integration test requires Node.js on PATH")
        runtime = Path(RUNTIME).expanduser().resolve()
        if not (runtime / "dist" / "sim" / "index.js").is_file():
            raise RuntimeError(f"Missing compiled Showdown simulator: {runtime}")
        spec = importlib.util.spec_from_file_location("battle_length_pipeline", PIPELINE / "pipeline.py")
        cls.pipeline = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.pipeline)
        cls.config = json.loads((PIPELINE / "smoke.json").read_text())
        cls.config["showdown_runtime"] = str(runtime)
        cls.config["timeout_seconds"] = 30
        cls.team = cls.pipeline.load_teams(
            ROOT / "notes" / "Blog" / "visuals" / "search-loop" / "team-example.json"
        )[0]["team"]
        cls.fingerprint = cls.pipeline.bridge(cls.config, "fingerprint")["fingerprint"]
        cls.summarize = staticmethod(load_summarizer())

    def test_real_battle_turns_match_log_and_survive_cache_reload(self):
        measured = []
        expected_turns = []
        with tempfile.TemporaryDirectory() as folder:
            database = Path(folder) / "battles.sqlite3"
            for seed in (101, 202):
                with self.subTest(seed=seed):
                    cache = self.pipeline.BattleCache(database)
                    try:
                        recorded, reused = cache.evaluate(
                            self.config, self.team, self.team, seed, self.fingerprint)
                        self.assertFalse(reused, "First evaluation must execute a real battle")
                    finally:
                        cache.db.close()

                    self.assertIn(recorded["outcome"], ("win", "loss", "draw"))
                    self.assertRegex(recorded["log"], r"(?m)^\|(?:win\|(?:Candidate|Opponent)|tie\|)$")
                    turns = [int(value) for value in re.findall(
                        r"(?m)^\|turn\|(\d+)$", recorded["log"])]
                    self.assertTrue(turns, "Expected a real battle with turn events")
                    self.assertEqual(turns, list(range(1, turns[-1] + 1)))
                    self.assertIs(type(recorded["turns"]), int)
                    self.assertEqual(recorded["turns"], turns[-1])
                    self.assertGreater(recorded["turns"], 0)
                    expected_turns.append(turns[-1])

                    # Reopen the database and forbid execution: this must use persisted data.
                    cache = self.pipeline.BattleCache(database)
                    try:
                        with patch.object(self.pipeline, "bridge", side_effect=AssertionError(
                                "A cache hit must not rerun the simulator")):
                            restored, reused = cache.evaluate(
                                self.config, self.team, self.team, seed, self.fingerprint)
                        self.assertTrue(reused)
                        self.assertEqual(restored, recorded)
                        self.assertEqual(cache.db.execute("SELECT COUNT(*) FROM battles").fetchone()[0],
                                         len(expected_turns))
                    finally:
                        cache.db.close()
                    measured.append(result(restored["turns"], outcome=restored["outcome"]))

        self.assertEqual(len(measured), 2, "Both seeded battles must pass before aggregation")
        summary, = self.summarize(measured)
        self.assertEqual(summary["n_measured"], 2)
        self.assertEqual(summary["mean_turns"], sum(expected_turns) / 2)
        print(f"Real Showdown turns verified against logs and cache: {expected_turns}")

    def test_turn_limit_failure_is_not_cached_as_a_completed_battle(self):
        config = {**self.config, "battle": {**self.config["battle"], "max_turns": 0}}
        with tempfile.TemporaryDirectory() as folder:
            cache = self.pipeline.BattleCache(Path(folder) / "battles.sqlite3")
            try:
                with self.assertRaisesRegex(RuntimeError, "Turn limit exceeded"):
                    cache.evaluate(config, self.team, self.team, 101, self.fingerprint)
                self.assertEqual(cache.db.execute("SELECT COUNT(*) FROM battles").fetchone()[0], 0)
            finally:
                cache.db.close()


if __name__ == "__main__":
    unittest.main()
