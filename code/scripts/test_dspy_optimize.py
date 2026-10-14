import json
import tempfile
import unittest
from unittest.mock import Mock

import llm_baseline as B

try:
    import dspy
    from dspy.utils import DummyLM
    import dspy_optimize as O
    import dspy_structure as S
    from test_dspy_structure import FILLS, MANIFEST, TRIALS
except ImportError:  # the runtime without DSPy still runs the other suites
    O = None


@unittest.skipIf(O is None, 'DSPy is not installed')
class OptimizeTests(unittest.TestCase):
    def setUp(self):
        self.m = dict(MANIFEST, structure=S.config('chat'), optimization=dict(split={'0': 'train', '1': 'dev'}))
        self.h = Mock(slot_to_text=lambda slot: slot['species'])
        self.validator = Mock(return_value=None)

    def run_program(self, fills, example):
        with dspy.context(lm=DummyLM([{'fills': json.dumps(fills)}]), adapter=S.adapter_for(self.m['structure'])):
            return O.Completion(self.m)(**example.inputs())

    def test_splits_keep_starters_apart(self):
        train, dev = O.examples(self.m, 'train'), O.examples(self.m, 'dev')
        self.assertEqual([e.trial_id for e in train], ['s0-item-2', 's0-pokemon-1'])
        self.assertEqual([e.trial_id for e in dev], ['s1-move-2'])
        self.assertEqual(set(train[0].inputs()), {'format', 'opponents', 'masked_team', 'requested_paths'})

    def test_metric_scores_legality_with_the_validator_message(self):
        metric = O.Metric(self.m, tempfile.mkdtemp(), validator=self.validator, h=self.h)
        example = O.examples(self.m, 'train')[0]
        self.assertEqual(metric(example, self.run_program(FILLS['s0-item-2'], example)).score, 1.0)
        rejected = metric(example, self.run_program(FILLS['s1-move-2'], example))
        self.assertEqual((rejected.score, rejected.feedback), (0.0, 'Rejected: duplicate or unmasked path'))
        self.validator.return_value = 'Garchomp has 76 total Stat Points'
        self.assertIn('76 total', metric(example, self.run_program(FILLS['s0-item-2'], example)).feedback)

    def test_battle_metric_caches_each_team_panel(self):
        battle = Mock(return_value=.1)
        metric = O.Metric(self.m, tempfile.mkdtemp(), battles=True, alpha=.5,
                          validator=self.validator, h=self.h, battle=battle)
        example = O.examples(self.m, 'train')[0]
        pred = self.run_program(FILLS['s0-item-2'], example)
        self.assertAlmostEqual(metric(example, pred).score, .55)
        self.assertAlmostEqual(metric(example, pred).score, .55)
        self.assertEqual(battle.call_count, 1)

    def test_exported_program_reaches_the_phase_one_request(self):
        program = O.Completion(self.m)
        example = O.examples(self.m, 'train')[0]
        program.predict.signature = program.predict.signature.with_instructions('Fill each null with a legal value.')
        program.predict.demos = [dspy.Example(**example.inputs(), fills=FILLS['s0-item-2'])]
        exported = O.export(program, self.m, {'name': 'test'})
        m = dict(MANIFEST, structure=S.config('chat', exported['instructions'], exported['demos']))
        system, *demo, user = B.completion_payload(m, TRIALS[2])['messages']
        self.assertIn('Fill each null with a legal value.', system['content'])
        self.assertNotIn('Complete only the null fields', system['content'])
        self.assertEqual(len(demo), 2)  # one demonstration: a user turn and an assistant turn
        self.assertIn('Focus Sash', demo[1]['content'])
        S.check_information(m, TRIALS[2])


if __name__ == '__main__':
    unittest.main()
