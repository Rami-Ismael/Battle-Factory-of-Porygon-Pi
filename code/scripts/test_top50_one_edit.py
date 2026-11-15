"""Tests for the statistics of scripts/top50_one_edit.py, on made-up scores. No battles, no validator, no files.

Run from code/:  /tmp/vgc-pilot/.venv/bin/python -m unittest scripts/test_top50_one_edit.py
"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import top50_one_edit as study


def made_up_teams(count, edit_move, noise, rng, base=0.5):
    """Teams whose edits move the true score by +-edit_move and whose every measurement carries Gaussian noise."""
    teams = []
    for i in range(count):
        true = base + 0.2 * (i / count - 0.5)      # true scores from 0.4 to 0.6
        measure = lambda value: float(value + rng.normal(0, noise))
        teams.append({"id": f"team {i}", "f": measure(true), "n": 384, "replays": [measure(true) for _ in range(3)],
                      "edits": [{"kind": "A_move_swap", "f": measure(true + edit_move * rng.choice([-1, 1])), "n": 384, "change": ""} for _ in range(3)]})
    return teams


class EditSize(unittest.TestCase):
    def test_noise_is_removed_from_the_size_of_an_edit(self):
        teams = made_up_teams(4000, edit_move=0.06, noise=0.025, rng=np.random.default_rng(0))
        size, share, correlation = study.edit_size(teams, "A_move_swap", variance=0.01)
        self.assertAlmostEqual(size, 0.06, delta=0.004)
        self.assertAlmostEqual(share, size / 0.1)                   # the spread is the square root of the variance
        self.assertAlmostEqual(correlation, 1 - share ** 2 / 2)

    def test_an_edit_that_changes_nothing_has_size_near_zero(self):
        teams = made_up_teams(4000, edit_move=0.0, noise=0.025, rng=np.random.default_rng(1))
        size, _, correlation = study.edit_size(teams, "A_move_swap", variance=0.01)
        self.assertLess(size, 0.012)
        self.assertGreater(correlation, 0.99)

    def test_share_beyond_noise_is_higher_for_real_edits_than_for_replays(self):
        teams = made_up_teams(2000, edit_move=0.10, noise=0.0255, rng=np.random.default_rng(2))
        edits, replays = study.share_beyond_noise(teams, "A_move_swap")
        self.assertGreater(edits, 0.5)
        self.assertLess(replays, 0.10)


class BenchmarkStep(unittest.TestCase):
    def test_a_step_is_a_share_of_the_spread_inside_its_own_kind_of_starting_point(self):
        rng = np.random.default_rng(3)
        low = made_up_teams(3000, edit_move=0.03, noise=0.01, rng=rng, base=0.3)
        high = made_up_teams(3000, edit_move=0.03, noise=0.01, rng=rng, base=0.8)
        own_spread = 0.2 / np.sqrt(12)                                # true scores are uniform over a range of 0.2
        self.assertAlmostEqual(study.step_share(low), 0.03 / own_spread, delta=0.03)
        self.assertAlmostEqual(study.step_share(high), 0.03 / own_spread, delta=0.03)
        pooled = study.edit_size(low, None, study.score_variance({"low": low, "high": high}))[1]
        self.assertLess(pooled, study.step_share(low) / 3)           # pooling the two kinds lets the gap between them shrink the step


class Readings(unittest.TestCase):
    def test_an_edit_is_above_zero_only_when_its_interval_excludes_zero(self):
        self.assertTrue(study.readings([0.4, 2.1], [[0.1, 0.3]])[0])
        self.assertFalse(study.readings([0.0, 2.6], [[0.1, 0.3]])[0])

    def test_against_a_pest_control_step_needs_every_interval_on_one_side(self):
        self.assertEqual(study.readings([1, 2], [[0.1, 0.3], [0.2, 0.4]])[1], "larger")
        self.assertEqual(study.readings([1, 2], [[-0.3, -0.1], [-0.2, -0.05]])[1], "smaller")
        self.assertEqual(study.readings([1, 2], [[0.1, 0.3], [-0.1, 0.4]])[1], "undecided")

    def test_no_size_cutoff_is_left_in_the_study(self):
        self.assertFalse(hasattr(study, "SIZE_CUTOFF"))


class Rank(unittest.TestCase):
    def test_rank_uses_the_replays_and_not_the_first_score(self):
        low_first = {"id": "wins most", "f": 0.10, "n": 384, "replays": [0.70, 0.70, 0.70], "edits": [{"kind": "A_move_swap", "f": 0.10, "n": 384, "change": ""}]}
        high_first = {"id": "wins least", "f": 0.90, "n": 384, "replays": [0.30, 0.30, 0.30], "edits": [{"kind": "A_move_swap", "f": 0.90, "n": 384, "change": ""}]}
        rows = study.per_team({"top50": [high_first, low_first]})
        self.assertEqual([row["id"] for row in rows], ["wins most", "wins least"])
        self.assertEqual(rows[0]["rank"], 1)

    def test_rank_correlation_is_negative_when_teams_that_win_more_change_more(self):
        rows = [{"rank": rank, "edit_move_points": 10 - rank} for rank in range(1, 9)]
        self.assertAlmostEqual(study.rank_correlation(rows), -1.0)


class Words(unittest.TestCase):
    def test_a_change_is_described_by_what_was_replaced(self):
        slot = {"species": "Raichu", "moves": ["Nuzzle", "Fake Out"], "ability": "Static", "item": "Focus Sash", "nature": "Timid", "evs": {"HP": 2}}
        other = dict(slot, moves=["Volt Tackle", "Fake Out"])
        self.assertEqual(study.describe_change([slot], [other]), "Raichu: move Nuzzle -> Volt Tackle")
        self.assertEqual(study.describe_change([slot], [dict(slot, species="Mawile")]), "Raichu and its whole set -> Mawile and its whole set")

    def test_the_round_trip_check_ignores_only_gender_and_shiny(self):
        self.assertEqual(study.comparable("Garchomp (F) @ Life Orb\nShiny: Yes\nAbility: Rough Skin"), study.comparable("Garchomp @ Life Orb\nAbility: Rough Skin"))
        self.assertNotEqual(study.comparable("Garchomp @ Life Orb"), study.comparable("Garchomp @ Choice Scarf"))


if __name__ == "__main__":
    unittest.main()
