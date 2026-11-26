"""Foul Play for doubles (src/foul_play_doubles.cjs) on this laptop's pinned Showdown runtime.

Needs ~/vgc-data (runtime and matchup database), so it stays off scripts/ci_tests.txt.
Run: /tmp/vgc-pilot/.venv/bin/python scripts/test_foul_play_doubles.py
"""
import json
import sqlite3
import subprocess
import tempfile
import unittest
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]
SRC = CODE / "src"
DB = Path.home() / "vgc-data" / "matchup_regmb.sqlite"
RUNTIME = Path.home() / "vgc-data" / "battle-evaluations" / "runtime"


def meta_teams(n):
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    rows = con.execute("SELECT t.paste FROM team t JOIN team_set s ON s.team_id = t.team_id "
                       "WHERE s.set_name = 'top50' ORDER BY t.team_id LIMIT ?", (n,)).fetchall()
    con.close()
    return [r[0] for r in rows]


@unittest.skipUnless(RUNTIME.exists() and DB.exists(), "needs this laptop's Showdown runtime and matchup database")
class FoulPlayDoubles(unittest.TestCase):
    def test_illusion_break_keeps_the_hp_the_disguise_showed(self):
        """Showdown's replace line (Illusion broken) carries no HP: the view must not parse one, and the revealed
        Zoroark keeps the HP, boosts and switch turn its disguise showed. 61 of 98 battles of pilot team 385255
        (#447) crashed here."""
        script = """
const {View} = require(process.argv[1]);
const v = new View('p1', process.argv[2]);
for (const l of ['|turn|1', '|switch|p2a: Wyrdeer|Wyrdeer, L50, M|100/100', '|-damage|p2a: Wyrdeer|43/100',
                 '|-boost|p2a: Wyrdeer|atk|1', '|turn|2', '|replace|p2a: Zoroark|Zoroark-Hisui, L50, M',
                 '|-end|p2a: Zoroark|Illusion']) v.line(l);
const z = v.active.p2[0];
console.log(JSON.stringify({species: z.species, frac: z.frac, atk: z.boosts.atk, switched: z.switchedTurn}));
"""
        paste = meta_teams(1)[0]
        out = subprocess.run(["node", "-e", script, str(SRC / "foul_play_doubles.cjs"), paste],
                             capture_output=True, text=True, timeout=60)
        self.assertEqual(out.returncode, 0, out.stderr[-800:])
        got = json.loads(out.stdout.strip().splitlines()[-1])
        self.assertEqual(got["species"], "Zoroark-Hisui")
        self.assertAlmostEqual(got["frac"], 0.435, places=3)
        self.assertEqual((got["atk"], got["switched"]), (1, 1))

    def test_rebuilt_battle_matches_real_battle(self):
        """Every visible field of the searched battle equals the real one at every decision."""
        with tempfile.TemporaryDirectory() as tmp:
            teams = Path(tmp) / "teams.json"
            teams.write_text(json.dumps(meta_teams(8)))
            out = subprocess.run(["node", str(SRC / "foul_play_doubles_audit.cjs"), str(teams), "8", "3"],
                                 capture_output=True, text=True, timeout=600, check=True)
        audit = json.loads(out.stdout)
        self.assertGreater(audit["decisions"], 100)
        self.assertEqual(audit["errors"], [])
        self.assertEqual(audit["unbuilt"], 0)
        checks = sum(audit["checks"].values())
        wrong = sum(audit["mismatches"].values())
        # Mid-turn switch decisions can miss a Protect counter (documented gap); nothing else may differ.
        other = {k: v for k, v in audit["mismatches"].items() if k not in ("volatile stall", "fake out window")}
        self.assertEqual(other, {}, audit["examples"][:5])
        self.assertLess(wrong, checks * 0.001)

    def test_foul_play_battle_completes_without_fallbacks(self):
        """A short Foul Play battle against random play finishes with every choice accepted."""
        a, b = meta_teams(2)
        proc = subprocess.Popen(["node", str(SRC / "foul_play_doubles_runner.cjs")], stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE, text=True)
        self.assertEqual(json.loads(proc.stdout.readline())["kind"], "ready")
        job = dict(kind="start", id="t", a=a, b=b, seed=5, p1={"policy": "foulplay", "samples": 2, "ms": 40,
                   "previewMs": 80}, p2={"policy": "random"})
        proc.stdin.write(json.dumps(job) + "\n")
        proc.stdin.flush()
        while True:
            message = json.loads(proc.stdout.readline())
            if message["kind"] in ("result", "error"):
                break
        proc.stdin.close()
        proc.wait(timeout=30)
        self.assertEqual(message["kind"], "result", message)
        self.assertIn(message["winner"], ("p1", "p2", None))
        stats = message["stats"]["p1"]
        self.assertGreater(stats["iterations"], 0)
        self.assertEqual(stats["fallbacks"], 0, stats.get("rejected"))

    def test_rust_doubles_engine_battle_and_audit(self):
        """Rust Reg M-B doubles engine completes battles with zero fallbacks and passes differential audit."""
        rust_bin = CODE / "rust" / "poke-engine-doubles" / "target" / "release" / "poke-engine-doubles"
        if not rust_bin.exists():
            self.skipTest("Rust poke-engine-doubles binary not built")
        audit_out = subprocess.run(["node", str(CODE / "scripts" / "audit_rust_vs_showdown.cjs")],
                                   cwd=CODE, capture_output=True, text=True, timeout=120, check=True)
        audit = json.loads(audit_out.stdout)
        self.assertGreaterEqual(audit["turns"], 100)
        self.assertLess(audit["mean_abs_hp_frac_error"], 0.08)
        self.assertGreater(audit["faint_agreement"], 0.93)
        a, b = meta_teams(2)
        for pol in ("rust_foulplay_doubles", "rust_foulplay_joint", "rust_uct_rollout", "rust_montecarlo"):
            proc = subprocess.Popen(["node", str(SRC / "foul_play_doubles_runner.cjs")], stdin=subprocess.PIPE,
                                    stdout=subprocess.PIPE, text=True)
            self.assertEqual(json.loads(proc.stdout.readline())["kind"], "ready")
            job = dict(kind="start", id="r", a=a, b=b, seed=9,
                       p1={"policy": pol, "samples": 1, "ms": 30, "previewMs": 50}, p2={"policy": "random"})
            proc.stdin.write(json.dumps(job) + "\n")
            proc.stdin.flush()
            while True:
                message = json.loads(proc.stdout.readline())
                if message["kind"] in ("result", "error"):
                    break
            proc.stdin.close()
            proc.wait(timeout=30)
            self.assertEqual(message["kind"], "result", message)
            self.assertGreater(message["stats"]["p1"]["iterations"], 5000)
            self.assertEqual(message["stats"]["p1"]["fallbacks"], 0)


if __name__ == "__main__":
    unittest.main()
