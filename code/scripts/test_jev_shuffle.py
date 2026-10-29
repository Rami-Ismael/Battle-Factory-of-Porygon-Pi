"""Experiment 2's shuffle: the order changes, the options do not, and the two fakes bracket the measurement."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import jev_shuffle as S

QUESTION = {'type': 'choice', 'instructions': 'Which move?', 'criteria': {m: {'power': 1} for m in
            ['Aqua Jet', 'Brick Break', 'Crunch', 'Dig', 'Earthquake', 'Facade', 'Giga Impact', 'Hail']}}
Q = {'id': 'abc:team[0].moves[1]', 'grid': 'test', 'task_id': 'abc', 'source': 'MB1', 'path': 'team[0].moves[1]',
     'team': [], 'state': {'team': []}, 'question': QUESTION, 'main': 'Crunch'}


class ShuffleTest(unittest.TestCase):
    def test_the_shuffle_keeps_every_option_and_only_reorders(self):
        _, qs = S.request(Q)
        a, s = list(qs['alphabetical']['criteria']), list(qs['shuffled']['criteria'])
        self.assertEqual(a, sorted(a))
        self.assertEqual(sorted(s), a)
        self.assertNotEqual(s, a)
        self.assertEqual(qs['shuffled']['criteria']['Dig'], QUESTION['criteria']['Dig'])

    def test_the_shuffle_is_seeded_by_question(self):
        self.assertEqual(list(S.shuffled(QUESTION['criteria'], 'x')), list(S.shuffled(QUESTION['criteria'], 'x')))
        self.assertNotEqual(list(S.shuffled(QUESTION['criteria'], 'x')), list(S.shuffled(QUESTION['criteria'], 'y')))

    def test_the_fakes_bracket_the_measurement(self):
        _, qs = S.request(Q)
        first, name = S.FirstOption().choose({}, qs), S.ByName().choose({}, qs)
        self.assertNotEqual(first['alphabetical']['choice'], first['shuffled']['choice'])
        self.assertEqual(name['alphabetical']['choice'], name['shuffled']['choice'])

    def test_an_answer_row_records_the_position_in_its_own_order(self):
        _, qs = S.request(Q)
        row = S.answer_row(Q, qs, S.FirstOption().choose({}, qs))
        self.assertEqual(row['alphabetical'], {**row['alphabetical'], 'choice': 'Aqua Jet', 'position': 0})
        self.assertEqual(row['shuffled']['position'], 0)
        self.assertEqual(row['options'], 8)

    def test_a_swapped_team_changes_only_the_asked_move(self):
        team = [{'species': 'Kleavor', 'moves': ['Stone Axe', 'X-Scissor', 'Close Combat', 'Night Slash']}]
        out = S.swapped_team(team, 'team[0].moves[1]', 'Crunch')
        self.assertEqual(out[0]['moves'], ['Stone Axe', 'Crunch', 'Close Combat', 'Night Slash'])
        self.assertEqual(team[0]['moves'][1], 'X-Scissor')

    def test_collect_appends_only_swapped_teams_the_showdown_check_passes(self):
        team = [{'species': 'Kleavor', 'moves': ['Stone Axe', 'X-Scissor', 'Close Combat', 'Night Slash']}]
        answer = lambda task, path, choice: {'task_id': task, 'source': 'MB1', 'path': path,
                                             'alphabetical': {'choice': 'Aqua Jet'}, 'shuffled': {'choice': choice}}

        def check_legal(pipeline, task, swapped):
            if 'Hail' in swapped[0]['moves']:
                raise ValueError('Kleavor cannot learn Hail')

        with tempfile.TemporaryDirectory() as folder:
            grid = Path(folder)
            results = grid / 'top.jsonl'
            results.write_text(json.dumps({'task_id': 'abc', 'run': 'top', 'team': team}) + '\n')
            (grid / 'candidates.json').write_text(json.dumps({'candidates': []}))
            with mock.patch.object(S.jev_run, 'results_path', return_value=results), \
                 mock.patch.object(S, 'rows', return_value=[answer('abc', 'team[0].moves[1]', 'Crunch'),
                                                            answer('abc', 'team[0].moves[2]', 'Hail')]), \
                 mock.patch.dict(S.jev_fill.GRIDS, {'test': grid}), \
                 mock.patch.object(S.jev_fill, 'tasks', return_value=[{'id': 'abc'}]), \
                 mock.patch.object(S.jev_fill, 'load_pipeline'), \
                 mock.patch.object(S.jev_fill, 'check_legal', side_effect=check_legal), \
                 mock.patch.object(S.jev_fill, 'team_paste', return_value=''), \
                 contextlib.redirect_stdout(io.StringIO()):
                out = S.collect('test')
            cands = json.loads((grid / 'candidates.json').read_text())['candidates']
        self.assertEqual((out['appended'], out['rejected']), (1, 1))
        self.assertEqual([c['team'][0]['moves'][1] for c in cands], ['Crunch'])


if __name__ == '__main__':
    unittest.main()
