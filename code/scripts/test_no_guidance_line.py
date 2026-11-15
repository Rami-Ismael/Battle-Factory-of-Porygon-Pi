"""The reader of issue #73 (docs/no-guidance-line.md): the rule, line-ups per 64 teams, the map's three-part split and
the comment. Hand-made numbers, so no grid, checkpoint or battle is needed.

    /tmp/vgc-pilot/.venv/bin/python scripts/test_no_guidance_line.py   (from code/)
"""
import unittest

import repo_paths  # noqa: F401  (src/ first on sys.path)
import no_guidance_line as N


def paste(*species): return "\n\n".join(f"{s} @ Leftovers\nAbility: X\n- Protect" for s in species)


A, B = paste("Garchomp", "Charizard", "Kingambit", "Basculegion", "Floette-Eternal", "Tyranitar"), paste("Garchomp", "Charizard", "Kingambit", "Basculegion", "Floette-Eternal", "Staraptor")
A_REORDERED = paste("Tyranitar", "Garchomp", "Charizard", "Kingambit", "Basculegion", "Floette-Eternal")


class Rule(unittest.TestCase):
    def test_a_drop_of_the_bar_with_the_interval_below_zero_is_nearer(self):
        self.assertEqual(N.judge(-0.001, [-0.002, -0.0001]), "nearer")

    def test_a_drop_smaller_than_the_bar_is_no_measurable_change_even_when_its_interval_is_below_zero(self):
        self.assertEqual(N.judge(-0.0009, [-0.0015, -0.0001]), "no measurable change")

    def test_an_interval_that_crosses_zero_is_no_measurable_change(self):
        self.assertEqual(N.judge(-0.004, [-0.009, 0.0002]), "no measurable change")

    def test_an_interval_above_zero_is_farther(self):
        self.assertEqual(N.judge(0.0004, [0.0001, 0.0008]), "farther")

    def test_near_the_line_is_strictly_under_the_bar(self):
        self.assertTrue(N.near_line(0.00099)); self.assertFalse(N.near_line(0.001))


class LineUps(unittest.TestCase):
    def test_line_ups_are_counted_within_each_cell_then_averaged_over_scored_asks(self):
        cells = {(1, 1): [A, A_REORDERED, B], (1, 2): [A, A], (2, 1): [B], (18, 1): [A, B]}
        got = N.line_ups_per_cell(cells)
        # ask 1 seed 1: A and A in another slot order are one line-up, B another -> 2; ask 1 seed 2 -> 1; ask 2 seed 1 -> 1; ask 18 is not scored
        self.assertAlmostEqual(got["scored_asks"], 4 / 3)
        self.assertEqual(got["by_ask"], {1: 1.5, 2: 1.0, 18: 2.0})


class Split(unittest.TestCase):
    def test_the_miss_splits_into_data_limit_fit_and_guidance(self):
        got = N.split(no_guidance=0.02, guidance_2=0.0075)
        self.assertAlmostEqual(got["data_limit"], 0.0038)
        self.assertAlmostEqual(got["fit"], 0.02 - 0.0038)
        self.assertAlmostEqual(got["guidance_2_removes"], 0.0125)


class Comment(unittest.TestCase):
    RESULT = dict(near_line=False, arms={
        "1": dict(reader="Guidance 1, no push", error=0.0201, error_interval=[0.015, 0.026], error_asks_18_to_23=0.05, line_ups_per_64=40.1, got_by_ask={"1": 0.1}),
        "2": dict(reader="Guidance 2", error=0.0075, error_interval=[0.0039, 0.0126], error_asks_18_to_23=0.01, line_ups_per_64=37.0, got_by_ask={"1": 0.03},
                  against_no_guidance=dict(difference=-0.0126, interval=[-0.018, -0.007], verdict="nearer"))},
        split=dict(data_limit=0.0038, fit=0.0163, guidance_2_removes=0.0126), prediction_held=True)

    def test_the_first_line_states_the_error_against_the_bar_and_what_guidance_2_does(self):
        first = N.comment(self.RESULT, "https://x/{theme}.png").splitlines()[0]
        self.assertIn("0.0201", first); self.assertIn("0.001", first); self.assertIn("nearer", first); self.assertIn("0.0075", first)

    def test_the_picture_follows_light_and_dark(self):
        text = N.comment(self.RESULT, "https://x/{theme}.png")
        self.assertIn("https://x/dark.png", text); self.assertIn("https://x/light.png", text)

    def test_what_guidance_2_takes_away_has_no_plus_sign(self):
        text = N.comment(self.RESULT, "https://x/{theme}.png")
        self.assertIn("takes away 0.0126", text); self.assertNotIn("+0.0126", text)


if __name__ == "__main__":
    unittest.main()
