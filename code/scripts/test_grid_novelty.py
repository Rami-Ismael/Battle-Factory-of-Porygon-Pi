"""Novelty and diversity of an ask grid's teams, reported by default for every new model or decoding row. No model, no data.

    /tmp/vgc-pilot/.venv/bin/python scripts/test_grid_novelty.py   (from code/)
"""
import unittest

import repo_paths  # noqa: F401  (src/ first on sys.path)
import grid_novelty as N

TRAIN = {("a", "b", "c", "d", "e", "f"), ("a", "b", "c", "d", "e", "g")}


def paste(*species):
    return "\n\n".join(f"{s} @ Leftovers\nAbility: X\n- Protect" for s in species)


class Swaps(unittest.TestCase):
    def test_a_copied_line_up_is_zero_swaps_and_one_new_species_is_one(self):
        self.assertEqual(list(N.swaps([("a", "b", "c", "d", "e", "f"), ("a", "b", "c", "d", "e", "h")], TRAIN)), [0, 1])

    def test_six_new_species_are_six_swaps(self):
        self.assertEqual(list(N.swaps([("m", "n", "o", "p", "q", "r")], TRAIN)), [6])


class Measures(unittest.TestCase):
    def setUp(self):
        self.lus = [("a", "b", "c", "d", "e", "f")] * 2 + [("a", "b", "c", "d", "e", "h"), ("a", "b", "c", "d", "h", "i")]
        self.m = N.measures(self.lus, TRAIN)

    def test_novel_share_counts_teams_whose_line_up_is_not_a_training_line_up(self):
        self.assertEqual(self.m["teams"], 4)
        self.assertAlmostEqual(self.m["novel_share"], 0.5)

    def test_swap_shares_add_up_to_one(self):
        s = self.m["swaps"]
        self.assertAlmostEqual(s["copied"] + s["one"] + s["two"] + s["three_or_more"], 1.0)
        self.assertAlmostEqual(s["copied"], 0.5)
        self.assertAlmostEqual(s["two"], 0.25)

    def test_line_ups_count_distinct_six_species_sets_and_the_novel_ones_among_them(self):
        self.assertEqual(self.m["line_ups"], 3)
        self.assertEqual(self.m["novel_line_ups"], 2)



class FromCells(unittest.TestCase):
    def test_cells_are_read_through_their_pastes(self):
        cells = [dict(numerator=12, pastes=[paste("A", "B", "C", "D", "E", "F"), paste("A", "B", "C", "D", "E", "H")])]
        m = N.from_cells(cells, TRAIN)
        self.assertEqual(m["teams"], 2)
        self.assertAlmostEqual(m["novel_share"], 0.5)
        self.assertAlmostEqual(m["within_ask_line_ups"], 2.0)

    def test_effective_species_is_the_mean_of_the_cells_5_team_subset_measure(self):
        cells = [dict(numerator=1, pastes=[paste("A", "B", "C", "D", "E", "F")], div=dict(effective_species=14.0)),
                 dict(numerator=2, pastes=[paste("A", "B", "C", "D", "E", "F")], div=dict(effective_species=16.0)),
                 dict(numerator=3, pastes=[], div=None)]
        self.assertAlmostEqual(N.from_cells(cells, TRAIN)["effective_species"], 15.0)

    def test_an_empty_grid_reports_no_teams_instead_of_failing(self):
        self.assertEqual(N.from_cells([], TRAIN), dict(teams=0))

    def test_the_summary_line_names_each_measure(self):
        line = N.summary_line(N.from_cells([dict(numerator=1, pastes=[paste("A", "B", "C", "D", "E", "H")], div=dict(effective_species=6.0))], TRAIN))
        for word in ("novel", "copied", "line-ups", "effective species"):
            self.assertIn(word, line)


if __name__ == "__main__":
    unittest.main()
