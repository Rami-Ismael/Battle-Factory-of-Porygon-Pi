"""The DDOM comparison's error, top-ask win rate and line-up counts (src/ddom_error_compare.py). Needs only numpy, so CI can run it.

    .venv/bin/python scripts/test_ddom_error_compare.py   (from code/)
"""
import unittest

import numpy as np

import repo_paths  # noqa: F401  (src/ first on sys.path)
import ddom_error_compare as D


def grid(got, teams=8, seeds=3):
    """Every team at ask k scores got(k / 24) exactly, so the bootstrap has no spread."""
    return {k: [dict(y=[got(k / 24)] * teams, pastes=[f"{k}-{s}-{i}" for i in range(teams)]) for s in range(seeds)] for k in D.ASKS}


class Error(unittest.TestCase):
    def test_top_asks_score_apart_from_the_rest(self):
        win = {k: (k / 24 if k <= 17 else 0.5) for k in D.ASKS}
        self.assertAlmostEqual(D.ask_error(win, D.SCORED), 0)
        self.assertGreater(D.ask_error(win, D.TOP), 0.04)
        self.assertEqual(D.SCORED, list(range(1, 18)))
        self.assertEqual(D.TOP, list(range(18, 24)))

    def test_weighted_arm_further_from_the_line_has_a_positive_difference(self):
        grids = {D.CONTROL: grid(lambda a: a), "gen84_ddom_tau03": grid(lambda a: a - 0.1)}
        arms = D.judge(grids, np.random.default_rng(0))
        d = arms["gen84_ddom_tau03"]["scored"]["minus_control"]
        self.assertAlmostEqual(d["difference"], 0.01)
        self.assertAlmostEqual(d["interval"][0], 0.01)
        self.assertTrue(arms[D.CONTROL]["scored"]["near_line"])
        self.assertNotIn("minus_control", arms[D.CONTROL]["scored"])
        self.assertAlmostEqual(arms["gen84_ddom_tau03"]["top_win"]["minus_control"]["difference"], -0.1)

    def test_bootstrap_pools_seeds_by_team(self):
        g = {k: [dict(y=[0.0] * 2), dict(y=[1.0] * 6)] for k in D.ASKS}
        boot = D.bootstrap(g, np.random.default_rng(0), draws=10)
        self.assertTrue(np.allclose(boot[1], 0.75))


class LineUps(unittest.TestCase):
    def test_new_line_ups_leave_out_both_references(self):
        g = {k: [dict(pastes=["a", "b", "c", "c"])] for k in D.ASKS}
        n = D.line_ups(g, lambda p: p, training={"a"}, records={"b"})
        self.assertEqual(n, dict(line_ups=3, new_vs_training=2, new_vs_both=1))


if __name__ == "__main__":
    unittest.main()
