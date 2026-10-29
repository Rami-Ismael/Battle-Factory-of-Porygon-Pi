import unittest
from collections import Counter

import repo_paths  # noqa: F401 -- puts src/ first on sys.path
import mercury_dspy_grid as G
from team_masks import masked


def first(kind, k=2):
    return next(t for t in G.tasks() if t['task'] == kind and t['k'] == k)


class ConversionTest(unittest.TestCase):
    def test_paths(self):
        self.assertEqual(G.to_path('team[3].statPoints'), [3, 'evs'])
        self.assertEqual(G.to_path('team[0].moves[2]'), [0, 'moves', 2])
        self.assertEqual(G.to_path('team[4]'), [4])

    def test_every_mask_hides_the_same_fields(self):
        for kind in ('stats', 'moves', 'pokemon', 'items', 'abilities', 'natures'):
            task = first(kind)
            paths = [G.to_path(p) for p in task['masked_paths']]
            self.assertEqual(masked(G.to_llm(task['original']), paths), G.to_llm(task['masked_team']))

    def test_a_filled_team_returns_to_the_original_slots(self):
        task = first('pokemon')
        team = G.to_mask_lab(G.to_llm(task['original']), task['original'])
        self.assertEqual(team, task['original'])


class DiversityTest(unittest.TestCase):
    def test_filled_reads_only_the_masked_values(self):
        task = first('pokemon')
        written = G.filled(task, task['original'])
        self.assertEqual(list(written), ['species'])
        self.assertEqual(len(written['species']), 2)

    def test_effective_values(self):
        self.assertEqual(G.effective(Counter(a=3)), 1.0)
        self.assertEqual(G.effective(Counter(a=1, b=1, c=1, d=1)), 4.0)
        self.assertIsNone(G.effective(Counter()))


if __name__ == '__main__':
    unittest.main()
