"""Deep evaluation of elite line-ups: the re-battle depth on line_up_yield.py and the flip count (docs/weakness-pressure-deep-evaluation.md part A, issue #345).

line_up_yield imports the Showdown validator from /tmp/vgc-pilot/src, so these tests need this laptop's staging; they are skipped without it.

    /tmp/vgc-pilot/.venv/bin/python scripts/test_deep_evaluation.py   (from code/)
"""
import importlib
import os
import sqlite3
import unittest
from pathlib import Path

import numpy as np

import repo_paths  # noqa: F401  (src/ first on sys.path)

# line_up_yield imports the Showdown validator, which reads the pilot staging folder at import.
PILOT_STAGED = Path("/tmp/vgc-pilot/src/hps_generate.py").exists() and Path("/tmp/vgc-pilot/learnset_true.json").exists()


def load(**env):
    """line_up_yield reads its settings from the environment at import."""
    for k in ("LY_PER_CELL", "LY_SEED", "LY_DB", "LY_MODEL", "LY_SMOKE"):
        os.environ.pop(k, None)
    os.environ.update(env)
    import line_up_yield
    return importlib.reload(line_up_yield)


@unittest.skipUnless(PILOT_STAGED, "needs the /tmp/vgc-pilot staging")
class Depth(unittest.TestCase):
    def test_default_depth_keeps_the_stored_names_seed_and_database(self):
        L = load(LY_MODEL="sched_species_first")
        self.assertEqual((L.DEPTH, L.SEED), (4, 7520))
        self.assertEqual(L.fresh_path("sched_species_first").name, "line_up_yield_sched_species_first.json")
        self.assertEqual(L.DB.name, "fresh.sqlite")

    def test_depth_16_gets_its_own_file_seed_and_database(self):
        L = load(LY_MODEL="sched_species_first", LY_PER_CELL="16")
        self.assertEqual(L.DEPTH, 16)
        self.assertEqual(L.fresh_path("sched_species_first").name, "line_up_yield_sched_species_first_d16.json")
        self.assertEqual(L.fresh_path("sched_species_first", 4).name, "line_up_yield_sched_species_first.json")
        self.assertNotEqual(L.SEED, 7520)
        self.assertGreater(abs(L.SEED - 7520), 10 ** 5)      # worker seeds are seed + job id; the arms' ranges must not overlap
        self.assertEqual(L.DB.name, "fresh_d16.sqlite")

    def test_only_4_and_16_battles_per_meta_team(self):
        for depth in ("8", "1"):
            with self.assertRaises(SystemExit):
                load(LY_PER_CELL=depth)

    def test_a_smoke_at_depth_16_never_claims_the_real_database(self):
        L = load(LY_PER_CELL="16", LY_SMOKE="1")
        self.assertEqual(L.DB.name, "fresh_d16_smoke.sqlite")

    def test_flips_needs_another_depth(self):
        L = load()
        with self.assertRaises(SystemExit):
            L.flips("sched_species_first")

    def test_only_the_all_23_view_is_battled_at_another_depth(self):
        L = load(LY_PER_CELL="16")
        a = dict(ask=1, seed=1, pos=0, paste="A", y=0.6, line_up=("a",))
        b = dict(ask=23, seed=1, pos=0, paste="B", y=0.7, line_up=("a",))       # best of line-up a in all 23 asks only
        c = dict(ask=2, seed=1, pos=0, paste="C", y=0.55, line_up=("c",))
        self.assertEqual(L.teams_to_battle([a, b, c], depth=16), ["B", "C"])
        self.assertEqual(L.teams_to_battle([a, b, c], depth=4), ["A", "B", "C"])


@unittest.skipUnless(PILOT_STAGED, "needs the /tmp/vgc-pilot staging")
class OneDepthPerDatabase(unittest.TestCase):
    """A 16-battle cell topped up from 4 stored battles would share those 4 with the other arm."""

    def con(self):
        con = sqlite3.connect(":memory:")
        con.executescript("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);"
                          "CREATE TABLE cell (policy_id INTEGER, lo INTEGER, hi INTEGER, battles INTEGER);")
        return con

    def test_an_empty_database_takes_the_depth_and_keeps_it(self):
        L = load(); con = self.con()
        L.claim_depth(con, 16); L.claim_depth(con, 16)
        with self.assertRaises(SystemExit):
            L.claim_depth(con, 4)

    def test_a_database_with_battles_and_no_recorded_depth_is_the_4_battle_arm(self):
        L = load(); con = self.con()
        con.execute("INSERT INTO cell VALUES (1, 1, 2, 4)")
        L.claim_depth(con, 4)
        con2 = self.con(); con2.execute("INSERT INTO cell VALUES (1, 1, 2, 4)")
        with self.assertRaises(SystemExit):
            L.claim_depth(con2, 16)


@unittest.skipUnless(PILOT_STAGED, "needs the /tmp/vgc-pilot staging")
class Flips(unittest.TestCase):
    def test_flip_share_counts_and_directions(self):
        L = load()
        v4 = np.array([1, 1, 0, 0, 1, 0, 0, 0, 1, 1], bool)
        v16 = np.array([1, 0, 0, 1, 1, 0, 0, 0, 1, 0], bool)
        s = L.flip_summary(v4, v16, boot=500, seed=0)
        self.assertEqual((s["line_ups"], s["flips"], s["held_to_not_held"], s["not_held_to_held"]), (10, 3, 2, 1))
        self.assertAlmostEqual(s["flip_share"], 0.3)
        self.assertEqual((s["held_at_4"], s["held_at_16"]), (5, 4))
        lo, hi = s["flip_share_interval"]
        self.assertLessEqual(lo, 0.3); self.assertGreaterEqual(hi, 0.3)

    def test_the_rule_fixed_on_the_ticket(self):
        L = load()
        self.assertEqual(L.depth_verdict([0.12, 0.30]), "16")
        self.assertEqual(L.depth_verdict([0.02, 0.08]), "4")
        self.assertEqual(L.depth_verdict([0.05, 0.15]), "undecided")

    def test_the_prediction_flips_more_near_the_cut(self):
        L = load()
        near = L.predicted_flip_share(np.full(200, L.CUT + 0.005), np.full(200, 196), 784, draws=200, seed=0, prior=(0.53, 0.04))
        self.assertAlmostEqual(L.effective_columns(), 50 ** 2 / 52, places=6)   # 48 meta teams once, one twice
        far = L.predicted_flip_share(np.full(200, L.CUT + 0.15), np.full(200, 196), 784, draws=200, seed=0, prior=(0.53, 0.04))
        self.assertGreater(near["mean"], 0.3); self.assertLess(far["mean"], near["mean"])


@unittest.skipUnless(PILOT_STAGED, "needs the /tmp/vgc-pilot staging")
class Provenance(unittest.TestCase):
    """flips refuses results that did not run at the planned depth and seed, or whose teams changed."""

    def deep(self, L, **over):
        team = "Pikachu @ Light Ball\nAbility: Static\n- Protect\n\nGarchomp @ Life Orb\nAbility: Rough Skin\n- Protect"
        return dict(per_cell=16, seed=L.DEEP_SEED, fresh=[dict(paste=team, fresh=0.55, battles=16 * 49)], **over)

    def test_the_planned_deep_re_battle_passes(self):
        L = load(LY_PER_CELL="16")
        self.assertEqual(len(L.deep_rows("m", self.deep(L))), 1)

    def test_wrong_depth_seed_or_battle_count_is_refused(self):
        L = load(LY_PER_CELL="16")
        for bad in (dict(per_cell=4), dict(seed=L.SHALLOW_SEED)):
            d = self.deep(L); d.update(bad)
            with self.assertRaises(SystemExit):
                L.deep_rows("m", d)
        d = self.deep(L); d["fresh"][0]["battles"] = 4 * 49 + 16 * 49        # topped up from the stored 4
        with self.assertRaises(SystemExit):
            L.deep_rows("m", d)


@unittest.skipUnless(PILOT_STAGED, "needs the /tmp/vgc-pilot staging")
class Comment(unittest.TestCase):
    def block(self, share, lo, hi, verdict):
        return dict(line_ups=272, flips=round(share * 272), flip_share=share, flip_share_interval=[lo, hi], held_to_not_held=40,
                    not_held_to_held=25, held_at_4=114, held_at_16=99, depth=16, verdict=verdict, correlation=0.6,
                    predicted_from_4=dict(mean=0.24), scores=[[0.54, 0.52], [0.52, 0.54], [0.60, 0.61], [0.45, 0.44]])

    def test_the_comment_leads_with_the_verdict_and_lists_every_guidance(self):
        load(LY_PER_CELL="16")
        import deep_evaluation_post as P
        P = importlib.reload(P)
        text = P.comment([(2, self.block(0.22, 0.17, 0.27, "16")), (4, self.block(0.21, 0.17, 0.26, "16"))])
        first = text.splitlines()[0]
        self.assertTrue(first.startswith("**Deep evaluation, guidance 4:")); self.assertIn("16 battles per meta team becomes", first)
        self.assertIn("| 2 | 272 |", text); self.assertIn("| 4 | 272 |", text)
        self.assertIn("Held at 4 battles per meta team", text); self.assertIn("Flip share (95% interval)", text)
        self.assertNotIn("| 16 |", text)                       # verdicts in words, not codes
        self.assertIn('<picture><source media="(prefers-color-scheme: dark)" srcset="u-scores-dark.png">', P.comment([(2, self.block(0.22, 0.17, 0.27, "16"))], "u-{name}-{theme}.png"))

    def test_the_figure_draws_interval_measured_prediction_and_rule(self):
        load(LY_PER_CELL="16")
        import deep_evaluation_post as P
        o = P.flip_share_option([(2, self.block(0.22, 0.17, 0.27, "16")), (4, self.block(0.21, 0.17, 0.26, "16"))])
        interval, measured, predicted = o["series"]
        self.assertEqual(interval["data"], [[0, 0.17, 0.27], [1, 0.17, 0.26]])
        self.assertEqual((measured["data"], predicted["data"]), ([0.22, 0.21], [0.24, 0.24]))
        self.assertEqual(measured["markLine"]["data"], [{"yAxis": 0.10}])
        self.assertEqual(o["tooltip"]["trigger"], "axis")

    def test_the_other_two_figures_and_titles_follow_the_data(self):
        load(LY_PER_CELL="16")
        import deep_evaluation_post as P
        r = [(2, self.block(0.22, 0.17, 0.27, "16"))]
        dots = {x["name"]: x["data"] for x in P.scores_option(r)["series"] if x["type"] == "scatter"}
        self.assertEqual([len(v) for v in dots.values()], [1, 1, 1, 1])        # both, lost, gained, neither
        self.assertTrue(P.scores_title(r).startswith("Flipped line-ups sat near the cut: 100%"))
        held = P.held_option(r)["series"]
        self.assertEqual([x["data"] for x in held], [[74, 74], [40, None], [None, 25]])   # 114 held at 4 = 74 at both + 40 lost
        self.assertEqual(P.held_title(r), "The held count moves by up to 13% between the depths")   # 114 at 4, 99 at 16
        r2 = [(2, {**self.block(0.22, 0.17, 0.27, "16"), "held_at_16": 109, "not_held_to_held": 35})]
        self.assertIn("35% of the line-ups held at 4 are not held at 16", P.held_title(r2))

    def test_a_low_share_stops_the_run(self):
        load(LY_PER_CELL="16")
        import deep_evaluation_post as P
        self.assertIn("the run stops here", P.comment([(2, self.block(0.04, 0.02, 0.07, "4"))]).splitlines()[0])
        later = P.comment([(2, self.block(0.22, 0.17, 0.27, "16")), (4, self.block(0.04, 0.02, 0.07, "4"))]).splitlines()[0]
        self.assertNotIn("stops", later); self.assertIn("below the rule at this guidance", later)


if __name__ == "__main__":
    unittest.main()
