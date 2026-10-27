import copy
import random
import unittest

import repo_paths  # noqa: F401 -- puts src/ first on sys.path
import jev_species as S
import jev_fill as J
import legal_options
from mb_prompt import norm


def species_mask():
    """The meta grid's first pokemon mask that hides one whole Pokémon."""
    return next(t for t in J.tasks('meta') if t['task'] == 'pokemon' and t['k'] == 1)


def hidden_fill(seed=None):
    """A Fill whose hidden slot is blank, and that slot's index."""
    task = species_mask()
    i = J.parse_path(task['masked_paths'][0])[0]
    team = copy.deepcopy(task['masked_team'])
    team[i] = J.blank_slot(i + 1)
    return J.Fill(task=task, team=team, meta=['x'], rng=random.Random(seed)), i


class SpeciesTest(unittest.TestCase):
    def setUp(self):
        self.fill, self.i = hidden_fill(seed=1)
        self.chooser = J.FakeChooser(1)
        self.species = S.ask_species(self.chooser, self.fill, [self.i, 'species'])
        self.legal = legal_options.fillable_species_options(self.fill.team)
        self.groups = self.fill.records[-1]['groups']
        self.group_records = self.fill.records[:-1]

    def test_no_question_has_more_than_255_options(self):
        for request in self.chooser.requests:
            for question in request['questions'].values():
                self.assertLessEqual(len(question['criteria']), 255)

    def test_group_questions_then_one_final(self):
        self.assertGreaterEqual(len(self.group_records), 2)
        self.assertEqual(len(self.chooser.requests), len(self.fill.records))
        self.assertTrue(all(r['round'].startswith('group ') for r in self.group_records))
        self.assertEqual(self.fill.records[-1]['round'], 'final')

    def test_final_options_are_exactly_the_group_winners(self):
        final = self.chooser.requests[-1]['questions']
        winners = [r['value'] for r in self.group_records]
        (question,) = final.values()
        self.assertEqual(set(question['criteria']), set(winners))
        self.assertEqual(len(winners), len(self.group_records))

    def test_groups_hold_every_legal_species_exactly_once(self):
        flat = [name for group in self.groups for name in group]
        self.assertEqual(len(flat), len(set(flat)))
        self.assertEqual(sorted(flat), sorted(self.legal))

    def test_returned_species_is_legal(self):
        self.assertIn(self.species, self.legal)

    def test_every_legal_species_has_a_pokedex_entry(self):
        for name in self.legal:
            self.assertIn(norm(name), S.pokedex())
            self.assertIn('types', S._entry(name))

    def test_same_seed_gives_same_groups(self):
        first, i = hidden_fill(seed=5)
        second, _ = hidden_fill(seed=5)
        S.ask_species(J.FakeChooser(2), first, [i, 'species'])
        S.ask_species(J.FakeChooser(2), second, [i, 'species'])
        self.assertEqual(first.records[-1]['groups'], second.records[-1]['groups'])

    def test_a_small_list_is_one_group(self):
        names = self.legal[:10]
        groups = S.split(names, random.Random(0), self.fill, [self.i, 'species'], 'x')
        self.assertEqual(len(groups), 1)
        self.assertEqual(sorted(groups[0]), sorted(names))


if __name__ == '__main__':
    unittest.main()
