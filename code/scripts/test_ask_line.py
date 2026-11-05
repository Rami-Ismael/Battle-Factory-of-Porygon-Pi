"""The ask-versus-got error, not the slope, judges an ask grid (src/ask_line.py). Needs only numpy, so CI can run it.

    .venv/bin/python scripts/test_ask_line.py   (from code/)
"""
import unittest

import repo_paths  # noqa: F401  (src/ first on sys.path)
from ask_line import headline, line_measures


def grid(got):
    return [dict(numerator=k, win=got(k / 24)) for k in range(1, 24)]


class LineMeasures(unittest.TestCase):
    def test_on_the_line(self):
        m = line_measures(grid(lambda a: a))
        self.assertAlmostEqual(m["mse_scored_asks"], 0)
        self.assertAlmostEqual(m["slope"], 1)
        self.assertTrue(m["near_line"])

    def test_a_steeper_line_can_be_further_away(self):
        shallow = line_measures(grid(lambda a: 0.9 * a))
        steep_but_lower = line_measures(grid(lambda a: a - 0.1))
        self.assertGreater(steep_but_lower["slope"], shallow["slope"])
        self.assertGreater(steep_but_lower["mse_scored_asks"], shallow["mse_scored_asks"])

    def test_top_asks_stay_out_of_the_score(self):
        low = lambda a: a if a <= 17 / 24 else 0.0
        m = line_measures(grid(low))
        self.assertAlmostEqual(m["mse_scored_asks"], 0)
        self.assertGreater(m["mse_asks_18_to_23"], 0.3)

    def test_empty(self):
        self.assertEqual(line_measures([]), {})
        self.assertIn("no asks", headline({}))

    def test_a_smoke_grid_has_no_top_asks(self):
        m = line_measures([dict(numerator=12, win=0.4)])
        self.assertIsNone(m["mse_asks_18_to_23"])
        self.assertIsNone(m["slope"])
        self.assertIn("no asks on 18/24 to 23/24", headline(m))


if __name__ == "__main__":
    unittest.main()
