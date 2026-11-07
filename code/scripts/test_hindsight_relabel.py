"""The hindsight-relabeled fine-tune's relabel set and its comparison (src/hindsight_relabel.py, src/hindsight_error_compare.py).
Needs only numpy, so CI can run it.

    .venv/bin/python scripts/test_hindsight_relabel.py   (from code/)
"""
import unittest

import numpy as np

import repo_paths  # noqa: F401  (src/ first on sys.path)
import ddom_error_compare as D
import hindsight_error_compare as C
import hindsight_relabel as H


def cell(numerator, seed, y):
    return dict(numerator=numerator, seed=seed, pastes=[f"team{numerator}-{seed}-{i}" for i in range(len(y))], y=y)


def grid(got, teams=8, seeds=3):
    """Every team at ask k scores got(k / 24) exactly, so the bootstrap has no spread."""
    return {k: [dict(y=[got(k / 24)] * teams, pastes=[f"{k}-{s}-{i}" for i in range(teams)]) for s in range(seeds)] for k in D.ASKS}


class Relabels(unittest.TestCase):
    def test_each_team_takes_its_measured_win_rate_not_its_ask(self):
        out = dict(cells=dict(a=cell(12, 1, [0.1, 0.9]), b=cell(3, 2, [0.4])))
        teams = H.relabels(out)
        self.assertEqual(sorted((t[1], t[2], t[3]) for t in teams), [(0.1, 12, 1), (0.4, 3, 2), (0.9, 12, 1)])

    def test_no_elite_filter_keeps_every_battled_team(self):
        out = dict(cells={f"c{k}": cell(k, 1, [0.0, 0.2, 1.0]) for k in range(1, 24)})
        self.assertEqual(len(H.relabels(out)), 69)

    def test_a_cell_without_battles_is_refused(self):
        for bad in (dict(numerator=1, seed=1, pastes=["x"]), cell(1, 1, [None]), dict(cell(1, 1, [0.5]), pastes=["x", "y"])):
            with self.assertRaises(ValueError):
                H.relabels(dict(cells=dict(a=bad)))

    def test_split_is_seeded_and_holds_a_fifth(self):
        a, b = H.split(4416), H.split(4416)
        self.assertTrue((a == b).all())
        self.assertEqual(a.sum(), 883)
        self.assertFalse((a == H.split(4416, seed=1)).all())

    def test_relabels_count_by_nearest_twenty_fourth(self):
        n = H.per_ask_count([0.0, 0.5, 0.51, 1.0, 1 / 48 + 1e-9])
        self.assertEqual((n[0], n[1], n[12], n[24]), (1, 1, 2, 1))
        self.assertEqual(sum(n.values()), 5)


class Compare(unittest.TestCase):
    def test_response_to_the_ask_on_the_line_is_fourteen_twenty_fourths(self):
        self.assertAlmostEqual(C.ask_response({k: k / 24 for k in D.ASKS}), 14 / 24)
        self.assertAlmostEqual(C.ON_LINE_RESPONSE, 14 / 24)
        self.assertAlmostEqual(C.ask_response({k: 0.4 for k in D.ASKS}), 0)

    def test_a_flatter_fine_tune_has_a_negative_response_difference(self):
        grids = {C.CONTROL: grid(lambda a: a), C.FINE_TUNE: grid(lambda a: 0.3 + a / 2)}
        r = C.response_rows(grids, np.random.default_rng(0))
        self.assertAlmostEqual(r[C.FINE_TUNE]["minus_control"]["difference"], -7 / 24)
        self.assertNotIn("minus_control", r[C.CONTROL])

    def test_scatter_is_the_spread_within_an_ask(self):
        g = {k: [dict(y=[k / 24 - 0.1, k / 24 + 0.1])] for k in D.ASKS}
        self.assertAlmostEqual(C.scatter(g), 0.1)

    def test_a_reordered_team_is_the_same_team(self):
        self.assertEqual(C.team_key("A\nx\n\nB\ny\n"), C.team_key("B\ny\n\nA\nx"))
        self.assertNotEqual(C.team_key("A\nx\n\nB\ny"), C.team_key("A\nx\n\nB\nz"))

    def test_repeats_count_identical_teams_and_shared_line_ups(self):
        g = {1: [dict(pastes=["A\n\nB", "B\n\nA", "A\n\nC"])]}
        r = C.repeats(g, ["A\n\nB", "A\n\nD"], species_of=lambda p: tuple(sorted(p.split("\n\n"))))
        self.assertEqual(r, dict(teams=3, identical_to_relabelled=2, line_ups=2, line_ups_in_relabelled=1))

    def test_verdict_needs_the_interval_and_the_smallest_effect(self):
        self.assertEqual(C.verdict(dict(minus_control=dict(difference=-0.002, interval=[-0.003, -0.001]))), "wins")
        self.assertEqual(C.verdict(dict(minus_control=dict(difference=-0.0005, interval=[-0.0008, -0.0002]))), "matches")
        self.assertEqual(C.verdict(dict(minus_control=dict(difference=0.002, interval=[0.001, 0.003]))), "loses")
        self.assertEqual(C.verdict(dict(minus_control=dict(difference=-0.002, interval=[-0.004, 0.0001]))), "matches")
        self.assertEqual(C.verdict({}), "control")

    def test_judge_takes_this_comparison_s_names_and_control(self):
        grids = {C.CONTROL: grid(lambda a: a - 0.1), C.FINE_TUNE: grid(lambda a: a)}
        arms = D.judge(grids, np.random.default_rng(0), names=C.NAMES, control=C.CONTROL)
        self.assertEqual(arms[C.FINE_TUNE]["name"], "Hindsight-relabeled fine-tune")
        self.assertAlmostEqual(arms[C.FINE_TUNE]["scored"]["minus_control"]["difference"], -0.01)
        self.assertEqual(C.verdict(arms[C.FINE_TUNE]["scored"]), "wins")


if __name__ == "__main__":
    unittest.main()
