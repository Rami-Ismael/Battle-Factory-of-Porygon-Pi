import json
import tempfile
import unittest
from pathlib import Path

import repo_paths  # noqa: F401 -- puts src/ first on sys.path
import jev_fill as J
import jev_report as Rep
import jev_run as R

MON = {'slot': 1, 'species': 'Garchomp', 'item': 'Life Orb', 'ability': 'Rough Skin', 'nature': 'Jolly',
       'moves': ['Earthquake', 'Dragon Claw', 'Rock Slide', 'Protect'],
       'statPoints': {'HP': 2, 'Atk': 32, 'Def': 0, 'SpA': 0, 'SpD': 0, 'Spe': 32}}
TEAM = [dict(MON, slot=i + 1) for i in range(6)]
TASK = {'id': 'abc', 'task': 'items', 'k': 1, 'source': 'MB1'}


class ReportTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.grid = root / 'grid'
        self.grid.mkdir()
        self.saved = (dict(J.GRIDS), R.DATA, J.tasks)
        J.GRIDS['test'], R.DATA = self.grid, root / 'data'
        J.tasks = lambda grid: [TASK]
        R.DATA.mkdir()
        self.existing = [{'label': 'orig:MB1', 'kind': 'original', 'source': 'MB1', 'team': TEAM, 'paste': 'x'},
                         {'label': 'random:abc', 'kind': 'random', 'task_id': 'abc', 'team': TEAM, 'paste': 'y'}]
        (self.grid / 'candidates.json').write_text(json.dumps({'candidates': self.existing}))
        rows = [{'task_id': 'abc', 'run': run, 'source': 'MB1', 'status': 'legal' if run != 'sample3' else 'illegal',
                 'team': TEAM} for run in R.RUNS]
        R.results_path('test').write_text(''.join(json.dumps(r) + '\n' for r in rows))

    def tearDown(self):
        J.GRIDS.clear()
        J.GRIDS.update(self.saved[0])
        R.DATA, J.tasks = self.saved[1], self.saved[2]
        self.tmp.cleanup()

    def test_collect_appends_legal_fills_once_and_keeps_the_rest(self):
        self.assertEqual(Rep.collect('test')['appended'], 3)
        cands = json.loads((self.grid / 'candidates.json').read_text())['candidates']
        self.assertEqual(cands[:2], self.existing)
        self.assertEqual({c['label'] for c in cands[2:]}, {'jev:top:abc', 'jev:sample1:abc', 'jev:sample2:abc'})
        self.assertEqual(len(list(self.grid.glob('candidates.before-jev-*.json'))), 1)
        self.assertEqual(Rep.collect('test')['appended'], 0)

    def test_report_compares_jev_with_the_original_and_random_fill(self):
        panels = {'orig:MB1': {'score': 0.50}, 'random:abc': {'score': 0.40}, 'jev:top:abc': {'score': 0.55},
                  'jev:sample1:abc': {'score': 0.45}, 'jev:sample2:abc': {'score': 0.35}}
        (self.grid / 'panels.json').write_text(json.dumps({'panels': panels}))
        out = Rep.report('test')
        g = out['groups']['items']
        self.assertAlmostEqual(g['top_minus_original']['mean'], 0.05)
        self.assertAlmostEqual(g['top_minus_random']['mean'], 0.15)
        self.assertAlmostEqual(g['sampled_minus_original']['mean'], -0.10)
        self.assertTrue((R.DATA / 'test-report.json').exists())


if __name__ == '__main__':
    unittest.main()
