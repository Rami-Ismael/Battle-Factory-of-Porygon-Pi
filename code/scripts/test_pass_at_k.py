"""The Pass@k reader of issue #331: the estimator against brute-force enumeration, Potential@k, and the success rule on
hand-made grids, so no data, database or checkpoint is needed.

    /tmp/vgc-pilot/.venv/bin/python scripts/test_pass_at_k.py   (from code/)
"""
import itertools
import unittest

import numpy as np

import repo_paths  # noqa: F401  (src/ first on sys.path)
import pass_at_k as P


class Estimator(unittest.TestCase):
    def test_matches_brute_force_enumeration_on_every_small_case(self):
        for n in range(1, 9):
            for c in range(n + 1):
                outcomes = [True] * c + [False] * (n - c)
                for k in range(1, n + 1):
                    self.assertAlmostEqual(P.pass_at_k(n, c, k), P.brute_force(outcomes, k), places=12, msg=f"n={n} c={c} k={k}")

    def test_brute_force_does_not_depend_on_where_the_successes_sit(self):
        for outcomes in set(itertools.permutations([True, True, False, False, False])):
            self.assertAlmostEqual(P.brute_force(outcomes, 2), P.pass_at_k(5, 2, 2))

    def test_edges(self):
        self.assertEqual(P.pass_at_k(192, 0, 48), 0.0)
        self.assertEqual(P.pass_at_k(192, 192, 1), 1.0)
        self.assertAlmostEqual(P.pass_at_k(192, 3, 1), 3 / 192)
        self.assertEqual(P.pass_at_k(10, 3, 8), 1.0)                 # 8 of 10 always include one of the 3
        with self.assertRaises(ValueError): P.pass_at_k(10, 11, 1)

    def test_table_rows_are_k_and_columns_are_c(self):
        t = P.pass_table(12, 3)
        self.assertEqual(t.shape, (3, 13))
        self.assertAlmostEqual(t[1, 4], P.pass_at_k(12, 4, 2))

    def test_curves_take_arrays_of_counts(self):
        c, n = np.array([[0, 5], [10, 192]]), np.full((2, 2), 192)
        got = P.pass_curves(c, n)
        self.assertEqual(got.shape, (P.K_MAX, 2, 2))
        self.assertAlmostEqual(got[47, 0, 1], P.pass_at_k(192, 5, 48))

    def test_curves_refuse_fewer_than_four_k_teams(self):
        with self.assertRaises(SystemExit): P.pass_curves(np.array([1]), np.array([4 * P.K_MAX - 1]))


class Potential(unittest.TestCase):
    def test_weights_each_ask_by_what_the_first_sample_leaves_unsolved(self):
        pass_k = np.array([[0.5, 0.9], [0.2, 0.4]])                  # [asks, K]
        pass_1 = np.array([0.5, 0.2])
        want = (pass_k[0] * 0.5 + pass_k[1] * 0.8) / 1.3
        np.testing.assert_allclose(P.potential_at_k(pass_k, pass_1), want)

    def test_draws_broadcast(self):
        pass_k = np.random.default_rng(0).random((3, 4, 5)); pass_1 = pass_k[:, 0, :]
        got = P.potential_at_k(pass_k, pass_1)
        self.assertEqual(got.shape, (4, 5))
        np.testing.assert_allclose(got[:, 2], P.potential_at_k(pass_k[:, :, 2], pass_1[:, 2]))

    def test_every_ask_solved_by_one_sample_has_no_potential(self):
        self.assertTrue(np.isnan(P.potential_at_k(np.ones((2, 3)), np.ones(2))).all())


A, B, C = ("a",) * 6, ("b",) * 6, ("c",) * 6


def grid(definition=P.HELD):
    """Two asks, two seeds, two opponents of equal weight. Line-up A held in its re-battle, B did not, C was never re-battled."""
    team = lambda win: [win, win]
    cells = {13: {1: dict(rate=np.array([team(0.6), team(0.6), team(0.2)]), line_ups=[A, B, C], tried=4, kept=3),
                  2: dict(rate=np.array([team(0.6), team(0.4), team(0.7)]), line_ups=[A, A, C], tried=3, kept=3)},
             14: {1: dict(rate=np.array([team(0.1), team(0.1), team(0.1)]), line_ups=[A, B, C], tried=3, kept=3),
                  2: dict(rate=np.array([team(0.1), team(0.9), team(0.1)]), line_ups=[B, B, C], tried=3, kept=3)}}
    return dict(name="toy", reader="Toy", guidance=2.0, guided=True, weights=np.ones(2), cells=cells, seeds=[1, 2], definition=definition,
                candidates={A: np.array([0.6, 0.6]), B: np.array([0.4, 0.5])})


class Successes(unittest.TestCase):
    def test_a_team_succeeds_only_when_it_clears_the_cut_and_its_line_up_held(self):
        got = P.successes(grid(), P.HELD)
        self.assertEqual(got[(13, 1)].tolist(), [True, False, False])   # B cleared the cut but did not hold
        self.assertEqual(got[(13, 2)].tolist(), [True, False, False])   # C cleared the cut and was never re-battled
        self.assertEqual(got[(14, 2)].tolist(), [False, False, False])

    def test_the_one_score_elite_is_the_cut_alone(self):
        got = P.successes(grid(), P.ONE_SCORE)
        self.assertEqual(got[(13, 1)].tolist(), [True, True, False])
        self.assertEqual(got[(13, 2)].tolist(), [True, False, True])

    def test_opponent_draws_move_the_one_score_and_the_re_battle_together(self):
        g = grid(); g["candidates"][A] = np.array([0.9, 0.2])            # holds only when the first opponent counts more
        g["cells"][13][1]["rate"][0] = [0.9, 0.2]
        counts = np.array([[2.0, 0.0], [0.0, 2.0], [1.0, 1.0]])
        got = P.successes(g, P.HELD, counts)[(13, 1)]
        self.assertEqual(got.shape, (3, 3))
        self.assertEqual(got[0].tolist(), [True, False, True])

    def test_bootstrap_counts_keep_the_team_total(self):
        rng = np.random.default_rng(1); counts = np.ones((P.DRAWS, 2))
        (c, n), (cd, nd) = P.counts_at(grid(), [13, 14], P.HELD, counts, rng)
        self.assertEqual(c.tolist(), [2, 0]); self.assertEqual(n.tolist(), [6, 6])
        self.assertTrue((nd == 6).all())
        self.assertTrue((cd[1] == 0).all())
        self.assertTrue(0 < cd[0].mean() < 6)
        self.assertAlmostEqual(cd[0].mean() / 6, 2 / 6, delta=0.03)       # the bootstrap is centred on the point estimate

    def test_network_calls_count_every_try_and_both_guided_passes(self):
        self.assertAlmostEqual(P.calls_per_legal_team(grid(), [13]), 84 * 2 * 7 / 6)
        g = grid(); g["cells"][13][1]["tried"] = None
        self.assertIsNone(P.calls_per_legal_team(g, [13]))


class Rule(unittest.TestCase):
    def test_pays_off_only_where_the_intervals_separate(self):
        K = P.K_MAX; rng = np.random.default_rng(2)
        first = np.full((K, 400), 0.5) + rng.normal(0, 0.01, (K, 400))
        random = first.copy(); random[40:] += 0.2
        iv = lambda d: [[float(np.quantile(x, .025)), float(np.quantile(x, .975))] for x in d]
        got = P.rule(random.mean(1), first.mean(1), random, first, iv(random), iv(first))
        self.assertTrue(got["pays_off"]); self.assertEqual(got["k_where_random_interval_lies_above"], list(range(41, K + 1)))
        self.assertFalse(P.rule(first.mean(1), random.mean(1), first, random, iv(first), iv(random))["pays_off"])
        self.assertAlmostEqual(got["difference"][45], 0.2, places=2)


if __name__ == "__main__":
    unittest.main()
