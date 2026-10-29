import json
import tempfile
import unittest
from pathlib import Path

import repo_paths  # noqa: F401 -- puts src/ first on sys.path
import jev_fields  # noqa: F401 -- registers the one-question field builders
import jev_species  # noqa: F401 -- registers the species tournament
import jev_fill as J
import jev_run as R

META = ['Torkoal @ Heat Rock\nAbility: Drought\nLevel: 50\nQuiet Nature\n- Eruption']


class Unreachable:
    """A chooser whose every call fails the way an outage would."""

    def choose(self, state, questions):
        raise J.TransportError('APIConnectionError: no route')


class RunnerTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.old, R.DATA = R.DATA, Path(self.tmp.name)
        self.pipeline = J.load_pipeline()

    def tearDown(self):
        R.DATA = self.old
        self.tmp.cleanup()

    def rows(self):
        return [json.loads(line) for line in R.results_path('meta').read_text().splitlines()]

    def test_one_top_run_per_mask_and_nothing_is_sent_twice(self):
        pairs = R.pending('meta', limit=2)
        self.assertEqual(len(pairs), 2 * len(R.RUNS))
        R.execute('meta', pairs, J.FakeChooser, META, self.pipeline, workers=2)
        self.assertEqual(R.pending('meta', limit=2), [])
        self.assertEqual(sorted({r['run'] for r in self.rows()}), sorted(R.RUNS))
        self.assertTrue(all(r['status'] == 'legal' for r in self.rows()))

    def test_a_transport_failure_is_recorded_and_retried(self):
        pairs = R.pending('meta', limit=1)
        R.execute('meta', pairs, Unreachable, META, self.pipeline, workers=1)
        self.assertTrue(all(r['status'] == 'transport' for r in self.rows()))
        self.assertEqual(len(R.pending('meta', limit=1)), len(R.RUNS))

    def test_seeds_are_stable_and_differ_between_runs(self):
        task = J.tasks('meta')[0]
        self.assertEqual(R.run_seed(task, 'sample1'), R.run_seed(task, 'sample1'))
        self.assertEqual(len({R.run_seed(task, run) for run in ('top', *R.SAMPLED)}), 4)

    def test_the_estimate_counts_one_question_per_hidden_field(self):
        task = next(t for t in J.tasks('meta') if t['task'] == 'items' and t['k'] == 3)
        est = R.estimate([(task, 'top')], META)
        self.assertEqual(est['questions'], 3)
        self.assertGreater(est['tokens'], 0)


if __name__ == '__main__':
    unittest.main()
