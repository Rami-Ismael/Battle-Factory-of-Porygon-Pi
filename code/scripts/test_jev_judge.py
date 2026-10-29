"""Experiment 6's judge: team collection, predictions, resume and rank correlation, with the fake chooser."""
import json
import tempfile
import unittest
from pathlib import Path

import jev_fill as J
import jev_judge as G
import jev_run as R

TEAM = [{'slot': 1, 'species': 'Kleavor', 'item': 'Choice Scarf', 'ability': 'Sharpness',
         'moves': ['Stone Axe', 'X-Scissor', 'Close Combat', 'Night Slash'], 'nature': 'Jolly',
         'statPoints': {'HP': 2, 'Atk': 32, 'Def': 0, 'SpA': 0, 'SpD': 0, 'Spe': 32}, 'level': 50}]


class JudgeTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.grid = root / 'grid'
        self.grid.mkdir()
        cands = [{'label': 'orig:MB1', 'paste': 'A\n', 'team': TEAM}, {'label': 'jev:top:abc', 'paste': 'A\n', 'team': TEAM},
                 {'label': 'random:abc', 'paste': 'B\n', 'team': TEAM}, {'label': 'ling:x:abc', 'paste': '', 'team': None},
                 {'label': 'textdiff:abc', 'paste': 'C\n', 'team': TEAM}]
        (self.grid / 'candidates.json').write_text(json.dumps({'candidates': cands}))
        panels = {'orig:MB1': {'score': 0.5}, 'jev:top:abc': {'score': 0.5}, 'random:abc': {'score': 0.2},
                  'textdiff:abc': {'score': None}}
        (self.grid / 'panels.json').write_text(json.dumps({'panels': panels}))
        self.old = R.DATA, dict(J.GRIDS)
        R.DATA = root / 'data'
        J.GRIDS['test'] = self.grid

    def tearDown(self):
        R.DATA, grids = self.old
        J.GRIDS.clear()
        J.GRIDS.update(grids)
        self.tmp.cleanup()

    def test_teams_are_distinct_battled_pastes_with_every_label(self):
        ts = sorted(G.teams('test'), key=lambda t: t['win_rate'])
        self.assertEqual([t['win_rate'] for t in ts], [0.2, 0.5])
        self.assertEqual(ts[1]['labels'], ['orig:MB1', 'jev:top:abc'])

    def test_the_two_questions_fit_jev(self):
        qs = G.questions()
        self.assertEqual(qs['band']['type'], 'score')
        self.assertEqual(len(qs['band']['criteria']), 10)
        self.assertEqual(len(G.MIDPOINTS), 10)
        self.assertEqual(list(qs['percent']['criteria'])[:2], ['0', '1'])
        self.assertLessEqual(len(qs['percent']['criteria']), J.OPTION_LIMIT)

    def test_predictions_are_probability_weighted(self):
        answers = {'band': {'choice': '9', 'confidence': 1.0, 'probabilities': {'0': 0.5, '9': 0.5}},
                   'percent': {'choice': '40', 'confidence': 0.5, 'probabilities': {'40': 0.5, '60': 0.5}}}
        p = G.predict(answers)
        self.assertAlmostEqual(p['band'], (0.05 + 0.65) / 2)
        self.assertAlmostEqual(p['percent'], 0.5)

    def test_every_team_is_judged_once_and_a_rerun_sends_nothing(self):
        todo = G.pending('test')
        self.assertEqual(G.execute('test', todo, J.FakeChooser, ['meta paste'], workers=2), {'judged': 2, 'transport': 0})
        self.assertEqual(G.pending('test'), [])
        rows = [json.loads(line) for line in G.results_path('test').read_text().splitlines()]
        self.assertEqual({r['win_rate'] for r in rows}, {0.2, 0.5})
        self.assertTrue(all(0 <= r['predicted']['band'] <= 1 and 0 <= r['predicted']['percent'] <= 1 for r in rows))

    def test_spearman_handles_ties(self):
        self.assertAlmostEqual(G.spearman([1, 2, 3, 4], [10, 20, 30, 40]), 1.0)
        self.assertAlmostEqual(G.spearman([1, 2, 3, 4], [40, 30, 20, 10]), -1.0)
        self.assertAlmostEqual(G.spearman([1, 1, 2, 2], [1, 1, 2, 2]), 1.0)


if __name__ == '__main__':
    unittest.main()
