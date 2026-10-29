import unittest

import repo_paths  # noqa: F401 -- puts src/ first on sys.path
import jev_fields  # noqa: F401 -- registers the item question
import jev_fill as J

META = ['Torkoal @ Heat Rock\nAbility: Drought\nLevel: 50\nQuiet Nature\n- Eruption']


def item_mask(k):
    """The meta grid's first item mask that hides k items."""
    return next(t for t in J.tasks('meta') if t['task'] == 'items' and t['k'] == k)


class PathsTest(unittest.TestCase):
    def test_paths_round_trip(self):
        for text in ('team[1]', 'team[3].item', 'team[0].moves[2]', 'team[5].statPoints'):
            self.assertEqual(J.path_text(J.parse_path(text)), text)

    def test_a_whole_pokemon_is_asked_in_slot_order(self):
        team = [None] * 6
        paths = J.expand([2], team)
        self.assertEqual([p[1] for p in paths],
                         ['species', 'ability', 'item', 'moves', 'moves', 'moves', 'moves', 'nature', 'statPoints'])
        self.assertEqual([p[2] for p in paths if p[1] == 'moves'], [0, 1, 2, 3])
        self.assertEqual(team[2]['slot'], 3)


class ItemMaskTest(unittest.TestCase):
    def test_every_answer_is_in_the_next_questions_state(self):
        task, chooser = item_mask(6), J.FakeChooser(1)
        team, records = J.fill_mask(task, chooser, META)
        self.assertEqual(len(chooser.requests), 6)
        for n, request in enumerate(chooser.requests):
            asked = request['state']['team']
            for record in records[:n]:
                slot = J.parse_path(record['path'])[0]
                self.assertEqual(asked[slot]['item'], record['value'])
        self.assertTrue(all(mon['item'] for mon in team))

    def test_the_state_carries_format_rules_meta_and_the_field_asked(self):
        chooser = J.FakeChooser()
        J.fill_mask(item_mask(1), chooser, META)
        state = chooser.requests[0]['state']
        self.assertEqual(set(state), {'format', 'rules', 'meta', 'team', 'asking_about'})
        self.assertEqual(state['meta'], META)
        question = next(iter(chooser.requests[0]['questions'].values()))
        self.assertEqual(question['type'], 'choice')
        self.assertTrue(all(set(d) == {'effect'} and d['effect'] for d in question['criteria'].values()))

    def test_no_two_slots_get_the_same_item(self):
        team, _ = J.fill_mask(item_mask(6), J.FakeChooser(3), META)
        items = [mon['item'] for mon in team]
        self.assertEqual(len(items), len(set(items)))

    def test_a_filled_team_passes_the_showdown_validator(self):
        task = item_mask(6)
        team, _ = J.fill_mask(task, J.FakeChooser(5), J.meta_pastes())
        J.check_legal(J.load_pipeline(), task, team)

    def test_a_sampled_run_takes_a_value_from_the_probabilities(self):
        _, records = J.fill_mask(item_mask(2), J.FakeChooser(), META, run='sample', seed=9)
        self.assertTrue(all(r['value'] in r['probabilities'] for r in records))

    def test_a_question_over_the_token_limit_is_refused(self):
        huge = ['x' * int(J.TOKEN_LIMIT * J.CHARS_PER_TOKEN + 1)]
        with self.assertRaises(J.QuestionTooLarge):
            J.fill_mask(item_mask(1), J.FakeChooser(), huge)

    def test_a_field_with_no_builder_says_so(self):
        task = dict(item_mask(1), masked_paths=['team[0].unknownField'])
        with self.assertRaises(NotImplementedError):
            J.fill_mask(task, J.FakeChooser(), META)


if __name__ == '__main__':
    unittest.main()
