"""Run: python scripts/test_type_coverage.py (standard library and data/ only)."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from type_coverage import coverage_failures, defensive_coverage


def team(*species):
    return [dict(species=s, item="", ability="", moves=[], nature="") for s in species]


# Counted by hand from the type chart: three slots are weak to Rock, nothing resists Ice.
EXAMPLE = team("Charizard", "Garchomp", "Dragonite", "Gengar", "Snorlax", "Gyarados")


class TypeCoverageTest(unittest.TestCase):
    def test_counts_match_a_hand_count(self):
        cov = defensive_coverage(EXAMPLE)
        self.assertEqual(len(cov), 18)
        self.assertEqual(cov["Ice"], (2, 0))
        self.assertEqual(cov["Rock"], (3, 1))
        self.assertEqual(cov["Fairy"], (2, 2))

    def test_an_immunity_counts_as_resisting(self):
        # Garchomp's Ground typing is immune to Electric; Dragonite's 2x and 0.5x cancel to neutral.
        self.assertEqual(defensive_coverage(EXAMPLE)["Electric"], (2, 1))

    def test_failures_name_the_broken_types(self):
        fails = coverage_failures(EXAMPLE)
        self.assertEqual(fails["too_many_weak"], ["Rock"])
        self.assertIn("Ice", fails["too_few_resist"])
        self.assertNotIn("Electric", fails["too_few_resist"] + fails["too_many_weak"])

    def test_thresholds_are_parameters(self):
        self.assertEqual(coverage_failures(EXAMPLE, max_weak=3)["too_many_weak"], [])

    def test_unknown_species_raises(self):
        with self.assertRaises(ValueError):
            defensive_coverage(team("Charizard", "Not A Pokemon"))


if __name__ == "__main__":
    unittest.main()
