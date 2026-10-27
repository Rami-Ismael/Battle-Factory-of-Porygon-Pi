import unittest

import repo_paths  # noqa: F401 -- puts src/ first on sys.path
import jev_fields
import jev_fill as J
import legal_options
from mb_prompt import norm

META = ['x']


def first_mask(task_type):
    """The meta grid's first mask of one task type."""
    return next(t for t in J.tasks('meta') if t['task'] == task_type)


def fill_first(task_type):
    chooser = J.FakeChooser()
    team, records = J.fill_mask(first_mask(task_type), chooser, META)
    return team, records, chooser


class OfferedOptionsTest(unittest.TestCase):
    def assert_every_value_was_offered(self, task_type):
        _, records, chooser = fill_first(task_type)
        self.assertTrue(records)
        self.assertEqual(len(records), len(chooser.requests))
        for request, record in zip(chooser.requests, records):
            criteria = next(iter(request['questions'].values()))['criteria']
            self.assertIn(record['choice'], criteria)
            self.assertIn(record['value'], criteria)

    def test_abilities_are_offered_options(self):
        self.assert_every_value_was_offered('abilities')

    def test_moves_are_offered_options(self):
        self.assert_every_value_was_offered('moves')

    def test_natures_are_offered_options(self):
        self.assert_every_value_was_offered('natures')

    def test_stat_points_are_offered_spreads(self):
        task = first_mask('stats')
        team, records, chooser = fill_first('stats')
        self.assertEqual(len(records), len(task['masked_paths']))
        for request, record in zip(chooser.requests, records):
            self.assertIn(record['value'], next(iter(request['questions'].values()))['criteria'])
        for text in task['masked_paths']:
            i = J.parse_path(text)[0]
            menu = jev_fields.spread_menu_for(team[i]['species'])
            self.assertIn(team[i]['statPoints'], menu)


class MoveTableTest(unittest.TestCase):
    def test_every_legal_move_name_has_a_move_table_entry(self):
        table = jev_fields.move_table()
        for task in J.tasks('meta'):
            if task['task'] != 'moves':
                continue
            for text in task['masked_paths']:
                for name in legal_options.options(task['masked_team'], J.parse_path(text)):
                    self.assertIn(norm(name), table, name)


class StatPointsTest(unittest.TestCase):
    def test_a_filled_stats_mask_holds_six_stats_within_limits(self):
        task = first_mask('stats')
        team, _ = J.fill_mask(task, J.FakeChooser(), META)
        for text in task['masked_paths']:
            spread = team[J.parse_path(text)[0]]['statPoints']
            self.assertEqual(set(spread), set(J.STATS))
            self.assertTrue(all(v <= 32 for v in spread.values()))
            self.assertLessEqual(sum(spread.values()), 66)

    def test_every_menu_spread_is_within_limits(self):
        menu = jev_fields.spread_menu()
        for spreads in [menu['generic'], *menu['species'].values()]:
            for spread in spreads:
                self.assertEqual(set(spread), set(J.STATS))
                self.assertTrue(all(v <= 32 for v in spread.values()))
                self.assertLessEqual(sum(spread.values()), 66)

    def test_a_species_with_its_own_spreads_uses_them_and_others_use_the_generic_list(self):
        menu = jev_fields.spread_menu()
        self.assertIs(jev_fields.spread_menu_for('Abomasnow'), menu['species']['abomasnow'])
        self.assertIs(jev_fields.spread_menu_for('Not A Real Pokemon'), menu['generic'])


if __name__ == '__main__':
    unittest.main()
