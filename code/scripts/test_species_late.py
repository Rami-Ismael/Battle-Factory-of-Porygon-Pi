"""The stored-teams step of issue #322: random-order and species-first teams with the same six species at the same ask,
compared on win rate. Hand-made teams, so no data or checkpoint is needed.

    /tmp/vgc-pilot/.venv/bin/python scripts/test_species_late.py   (from code/)
"""
import unittest

import numpy as np

import repo_paths  # noqa: F401  (src/ first on sys.path)
import species_late as S

WEIGHTS = np.array([1.0, 1.0])
L1, L2, L3 = ("A", "B", "C", "D", "E", "F"), ("A", "B", "C", "D", "E", "G"), ("H", "I", "J", "K", "L", "M")


def team(lineup, win): return lineup, np.array([win, win])


RANDOM = {13: [team(L1, 0.4), team(L1, 0.6), team(L2, 0.1)], 14: [team(L3, 0.2)]}
FIRST = {13: [team(L1, 0.7), team(L3, 0.9)], 14: [team(L1, 0.5)]}


class StoredComparison(unittest.TestCase):
    def test_teams_with_the_same_line_up_at_the_same_ask_are_compared(self):
        got = S.gaps(RANDOM, FIRST, WEIGHTS)
        # ask 13 shares L1 (random 0.4 and 0.6, species-first 0.7); L3 sits at ask 14 in one row and ask 13 in the other, so it is not matched
        self.assertAlmostEqual(got["same_line_up"], -0.2)
        self.assertAlmostEqual(got["random_matched"], 0.5); self.assertAlmostEqual(got["species_first_matched"], 0.7)
        self.assertEqual((got["matched_random_teams"], got["matched_line_ups"]), (2, 1))

    def test_the_total_gap_averages_each_ask_over_all_its_teams(self):
        got = S.gaps(RANDOM, FIRST, WEIGHTS)
        # ask 13: 0.3667 against 0.8; ask 14: 0.2 against 0.5
        self.assertAlmostEqual(got["total"], -0.3666667, places=6)

    def test_a_line_up_matched_only_at_another_ask_is_not_matched(self):
        got = S.gaps({14: [team(L1, 0.2)]}, {13: [team(L1, 0.9)], 14: [team(L3, 0.5)]}, WEIGHTS)
        self.assertEqual(got["matched_random_teams"], 0)
        self.assertTrue(np.isnan(got["same_line_up"]))

    def test_teams_that_score_alike_give_the_same_gap_in_every_bootstrap_draw(self):
        counts = np.random.default_rng(0).multinomial(2, [0.5, 0.5], 50).astype(float)
        got = S.gaps({13: [team(L1, 0.3), team(L1, 0.3)]}, {13: [team(L1, 0.6)]}, WEIGHTS, counts=counts, rng=np.random.default_rng(1))
        self.assertEqual(got["same_line_up"].shape, (50,))
        np.testing.assert_allclose(got["same_line_up"], -0.3)

    def test_the_bootstrap_resamples_teams_within_a_line_up(self):
        counts = np.ones((400, 2))
        got = S.gaps(RANDOM, FIRST, WEIGHTS, counts=counts, rng=np.random.default_rng(2))
        draws = got["same_line_up"]
        self.assertEqual(set(np.round(np.unique(draws), 6)), {-0.3, -0.2, -0.1})   # random's L1 resampled: both 0.4, one each, both 0.6
        self.assertAlmostEqual(float(draws.mean()), -0.2, delta=0.02)

    def test_weights_decide_a_team_s_score(self):
        rate = np.array([1.0, 0.0])
        got = S.gaps({13: [(L1, rate)]}, {13: [(L1, np.array([0.0, 0.0]))]}, np.array([3.0, 1.0]))
        self.assertAlmostEqual(got["same_line_up"], 0.75)


class StepTwoVerdict(unittest.TestCase):
    """The ticket's rule. Got: mean win rate at asks 13/24 to 17/24. Error: ask-versus-got error on asks 1/24 to 17/24.
    Linear, species-first reads got 0.48 and error 0.0075; Linear, random order reads got 0.33."""
    def verdict(self, got, error): return S.verdict(got, got_first=0.48, got_random=0.33, error=error, error_first=0.0075)

    def test_got_back_beside_species_first_and_error_within_training_luck_confirms(self):
        self.assertEqual(self.verdict(0.46, 0.0090), "confirmed")

    def test_got_beside_species_first_but_error_past_training_luck_is_partial(self):
        self.assertEqual(self.verdict(0.46, 0.0100), "partial")

    def test_got_beside_random_order_clears(self):
        self.assertEqual(self.verdict(0.35, 0.0300), "cleared")

    def test_got_between_the_two_is_partial(self):
        self.assertEqual(self.verdict(0.41, 0.0150), "partial")


class GridAtEachGuidance(unittest.TestCase):
    """Issue #329: the step-two grid read at guidance 4 and 8 beside the guidance-2 grid of #322."""
    def test_guidance_2_keeps_the_results_file_of_issue_322(self):
        self.assertEqual(S.grid_path(2).name, "species_late_grid.json")

    def test_guidance_4_and_8_each_get_their_own_results_file(self):
        self.assertEqual((S.grid_path(4).name, S.grid_path(8).name), ("species_late_grid_g4.json", "species_late_grid_g8.json"))

    def result(self, guidance):
        row = lambda reader, err, got: dict(reader=reader, error=err, error_interval=[err - 0.001, err + 0.001], error_by_seed={"1": err, "2": err, "3": err},
                                            got_asks_13_to_17=got, got_asks_13_to_17_interval=[got - 0.03, got + 0.03])
        gap = dict(difference=-0.0003, interval=[-0.0018, 0.0011])
        return dict(guidance=guidance, verdict="confirmed", rule="the rule", error_against_species_first=gap, error_against_random=gap,
                    got_against_species_first=gap, got_against_random=gap,
                    rows={S.SPECIES_THEN_RANDOM: row(S.READER[S.SPECIES_THEN_RANDOM], 0.0040, 0.50), S.SPECIES_FIRST: row(S.READER[S.SPECIES_FIRST], 0.0049, 0.51),
                          S.RANDOM: row(S.READER[S.RANDOM], 0.0290, 0.35)})

    def test_the_comment_at_guidance_4_names_guidance_4_and_never_guidance_2(self):
        text = S.comment_text(self.result(4), "https://example.org/{theme}.png")
        self.assertIn("Guidance 4", text.splitlines()[0])
        self.assertIn("at guidance 4", text)
        self.assertNotIn("guidance 2", text.lower())

    def test_share_recovered_is_the_part_of_random_order_s_got_gap_that_species_first_drawing_wins_back(self):
        # species-first 0.50, random order 0.30: a row at 0.45 wins back 0.15 of the 0.20 gap
        self.assertAlmostEqual(S.share_recovered(got=0.45, got_first=0.50, got_random=0.30), 0.75)
        self.assertAlmostEqual(S.share_recovered(got=0.50, got_first=0.50, got_random=0.30), 1.0)

    def test_the_comment_at_guidance_2_still_reads_as_step_two_of_issue_322(self):
        self.assertTrue(S.comment_text(self.result(2), "https://example.org/{theme}.png").startswith("**Step two, "))


if __name__ == "__main__":
    unittest.main()
