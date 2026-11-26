"""Run: python scripts/test_foul_play_label_pilot.py   (no battles, no laptop data; CI runs it).

The issue #447 pilot (src/foul_play_label_pilot.py) on a small fake matchup database, with the Rust runner's node
processes replaced by a fake. Each test fails on one planted bug: a meta, held-out or VGCPastes pool team drawn as a
pilot team, a redraw that changes the saved teams, halves read from the wrong side of a cell, a failed game counted
as a half, an ICC that does not separate decisive pairs from coin flips, or a behaviour-cloning lookup that picks
the Rust policy because it carries the shuffled-slot tag too."""
import hashlib
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import foul_play_label_pilot as P
import heldout as H
import matchup_db as M
import rust_matchup_runner as R

PROV = {k: k * 32 for k in ("engine", "dex", "spreads", "agent", "runtime")}
STATS = {"decisions": 9, "iterations": 270000, "fallbacks": 0, "failedSamples": 0}


def paste(name):
    return "\n\n".join(f"{name}{i} @ Item\nAbility: A\n- Move" for i in range(6)) + "\n"


class StrengthPlay:
    """Fake node processes: the team with the higher strength wins, except a seeded coin flip on `noise` of games."""
    def __init__(self, strength, noise=0.3, error_ids=()):
        self.strength, self.noise, self.error_ids, self.games = strength, noise, set(error_ids), []

    def __call__(self, games, spec, on_done):
        for g in games:
            self.games.append(g)
            if g["id"] in self.error_ids:
                on_done(g, None, "RustRunnerError: poke-engine-doubles exited with code 1"); continue
            u = int(hashlib.sha256(g["id"].encode()).hexdigest()[:8], 16) / 2 ** 32
            p1_strong = self.strength[g["p1_team"]] >= self.strength[g["p2_team"]]
            p1_wins = p1_strong if u >= self.noise else u < self.noise / 2
            on_done(g, {"winner": "p1" if p1_wins else "p2", "turns": 7, "ms": 250, "stats": {"p1": STATS, "p2": STATS}}, None)


class Pilot(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.TemporaryDirectory()
        self.con = M.connect(Path(self.d.name) / "t.sqlite")
        for i in range(1, 9):                   # policy 8 is behaviour cloning with shuffled slots, as on the laptop
            tag = "+slots-shuffled" if i == 8 else ""
            M.get_policy(self.con, f"bc_100.zip:{i}{tag}", f"bc_100.zip:{i}{tag}", "showdown:unknown")
        with self.con:
            def team(n, origin="search:test"):
                return self.con.execute("INSERT INTO team (format_id, canon_hash, paste, species_key, base_key, origin, "
                                        "grid_hash, legal) VALUES ('f', ?, ?, ?, ?, ?, ?, 1)",
                                        (n, paste(n), n, n, origin, f"grid-{n}")).lastrowid
            self.meta = [team(f"m{i}", "vgcpastes") for i in range(4)]
            self.teams = [team(f"t{i}") for i in range(12)]
            self.pool_team, self.held_team, self.short_team = team("pool", "vgcpastes"), team("held"), team("short")
        self.held = mock.patch.object(H, "identities", lambda: frozenset({"grid-held"}))
        self.held.start()
        rows = [*self.teams, self.pool_team, self.held_team]
        M.record(self.con, 8, [(r, c, 1, 0, 0) for r in rows for c in self.meta])
        M.record(self.con, 8, [(self.short_team, c, 0, 1, 0) for c in self.meta[:3]])     # misses one meta team
        self.strength = {t: i for i, t in enumerate(self.meta + self.teams)}

    def tearDown(self):
        self.held.stop(); self.con.close(); self.d.cleanup()

    def runner(self, play):
        return R.RustMatchupRunner(play=play, prov=PROV, evidence=Path(self.d.name) / "evidence")

    def test_eligible_leaves_out_meta_held_out_pool_and_uncovered_teams(self):
        got = P.eligible(self.con, self.meta, H.team_ids(self.con))
        self.assertEqual(got, sorted(self.teams))

    def test_draw_is_fixed_by_the_seed_and_saved_draw_is_reused(self):
        pool = P.eligible(self.con, self.meta, H.team_ids(self.con))
        self.assertEqual(P.draw(pool, 5, 1), P.draw(list(reversed(pool)), 5, 1))
        self.assertNotEqual(P.draw(pool, 5, 1), P.draw(pool, 5, 2))
        split = (self.meta[:3], self.meta[3:])
        out = Path(self.d.name) / "out"
        with mock.patch.object(P, "meta_split", lambda con: split):
            first = P.sample(self.con, out, n=5, seed=1)
            with self.con: self.con.execute("DELETE FROM cell WHERE lo=? OR hi=?", (first["team_ids"][0],) * 2)
            again = P.sample(self.con, out, n=5, seed=1)           # the pool changed; the saved teams do not
            self.assertEqual(first["team_ids"], again["team_ids"])
            with self.assertRaises(ValueError): P.sample(self.con, out, n=5, seed=2)

    def test_first_two_reads_the_row_side_and_skips_failed_games(self):
        games = {(3, 9): {0: 1.0, 2: 0.0, 3: 0.5}}                # game 1 failed
        self.assertEqual(P.first_two(games, 3, 9), (1.0, 0.0))
        self.assertEqual(P.first_two(games, 9, 3), (0.0, 1.0))
        self.assertIsNone(P.first_two({(3, 9): {0: 1.0}}, 3, 9))

    def test_split_half_is_one_for_identical_halves_and_low_for_coin_flips(self):
        same = {t: (t / 10, t / 10) for t in range(10)}
        r = P.split_half(same, boot=50)
        self.assertAlmostEqual(r["r"], 1.0); self.assertAlmostEqual(r["spearman_brown"], 1.0)
        import random
        rng = random.Random(0)
        flips = {t: (rng.random(), rng.random()) for t in range(400)}
        self.assertLess(abs(P.split_half(flips, boot=50)["r"]), 0.15)

    def test_icc_separates_decisive_pairs_from_coin_flips(self):
        decisive = {(i, i + 100): (8, 0, 0) if i % 2 else (0, 8, 0) for i in range(40)}
        coin = {(i, i + 100): (4, 4, 0) for i in range(40)}
        self.assertGreater(P.matchup_icc(decisive)["icc"], 0.95)
        self.assertEqual(P.matchup_icc(coin)["icc"], 0.0)

    def test_battles_and_report_end_to_end(self):
        play = StrengthPlay(self.strength, error_ids={f"{min(self.teams[0], self.meta[0])}-{max(self.teams[0], self.meta[0])}-1"})
        runner = self.runner(play)
        runner.allow_errors = True
        pid = runner.register(self.con)
        self.assertNotEqual(pid, 8)
        P.battle_meta(self.con, runner, self.meta, pid)
        saved = {"team_ids": self.teams[:6], "training_opponents": self.meta[:3], "validation_opponents": self.meta[3:]}
        P.battle_pilot(self.con, runner, saved["team_ids"], self.meta, pid, chunk=4, log=lambda m: None)
        P.battle_pilot(self.con, runner, saved["team_ids"], self.meta, pid, chunk=4, log=lambda m: None)   # tops up the failed game
        self.assertTrue(all(M.battles_done(self.con, pid, t, m) == 2 for t in saved["team_ids"] for m in self.meta))
        self.assertTrue(all(M.battles_done(self.con, 8, t, m) == 1 for t in saved["team_ids"] for m in self.meta))
        M.record(self.con, 8, [(t, m, 0, 1, 0) for t in saved["team_ids"] for m in self.meta])   # the top-up's second battle
        r = P.report_pilot(self.con, pid, saved)
        self.assertEqual((r["teams_with_2_in_all_49"], r["held_out_cells"], r["cells_where_evidence_differs_from_db"]), (6, 0, 0))
        fp = r["halves"]["foul_play"]
        self.assertEqual(len(fp), 6)
        self.assertEqual(r["split_half"]["behaviour_cloning"]["teams"], 6)
        self.assertGreater(r["label"]["foul_play"]["mean"], 0.5)   # pilot teams are stronger than every meta team
        m = P.report_meta(self.con, pid, self.meta)
        self.assertEqual((m["pairs_expected"], m["pairs_short_of_8"], m["foul_play"]["battles"]), (6, 0, 48))
        self.assertIn("Matchup ICC", P.meta_markdown(m)); self.assertIn("Split-half r", P.pilot_markdown(r))

    def test_behaviour_cloning_lookups_skip_the_rust_policy(self):
        R.get_rust_policy(self.con, prov=PROV)
        ids = [p for (p,) in self.con.execute("SELECT policy_id FROM policy WHERE row_policy LIKE '%+slots-shuffled' "
                                              "AND row_policy NOT LIKE 'rust%'")]
        self.assertEqual(ids, [8])
        try:
            import lossdata65 as L65
        except ImportError:
            self.skipTest("lossdata65 needs the /tmp/vgc-pilot runtime")
        self.assertEqual(L65.shuffled_policy(self.con), 8)


if __name__ == "__main__":
    unittest.main()
