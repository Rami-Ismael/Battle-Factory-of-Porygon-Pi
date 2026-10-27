import unittest

import legal_options as L
from mb_prompt import norm, regulation

TEAM = [
    dict(species='Garchomp', item='Life Orb', ability='Rough Skin', nature='Jolly',
         moves=['Earthquake', 'Dragon Claw', 'Protect', None]),
    dict(species='Incineroar', item='Sitrus Berry', ability='Intimidate', nature='Careful',
         moves=['Fake Out', 'Parting Shot', 'Flare Blitz', 'Protect']),
    dict(species='Kingambit', item=None, ability='Defiant', nature='Adamant',
         moves=['Kowtow Cleave', 'Sucker Punch', 'Protect', 'Iron Head']),
    None,
    dict(species='Rotom-Wash', item='Leftovers', ability='Levitate', nature='Bold',
         moves=['Hydro Pump', 'Thunderbolt', 'Will-O-Wisp', 'Protect']),
    dict(species='Pelipper', item='Focus Sash', ability='Drizzle', nature='Modest',
         moves=['Hurricane', 'Weather Ball', 'Tailwind', 'Protect']),
]


class LegalOptionsTest(unittest.TestCase):
    def test_a_move_list_excludes_the_slots_other_moves(self):
        moves = L.options(TEAM, [0, 'moves', 3])
        self.assertIn('Rock Slide', moves)
        for known in ('Earthquake', 'Dragon Claw', 'Protect'):
            self.assertNotIn(known, moves)

    def test_a_move_list_holds_only_learnable_moves(self):
        learnable = set(regulation()['slots'][norm('Garchomp')]['moves'])
        self.assertTrue(all(norm(m) in learnable for m in L.options(TEAM, [0, 'moves', 3])))
        self.assertNotIn('Thunderbolt', L.options(TEAM, [0, 'moves', 3]))

    def test_an_item_list_excludes_teammates_items(self):
        items = L.options(TEAM, [2, 'item'])
        self.assertIn('Black Glasses', items)
        for held in ('Life Orb', 'Sitrus Berry', 'Leftovers', 'Focus Sash'):
            self.assertNotIn(held, items)

    def test_an_ability_list_belongs_to_the_species(self):
        self.assertIn('Defiant', L.options(TEAM, [2, 'ability']))
        self.assertNotIn('Intimidate', L.options(TEAM, [2, 'ability']))

    def test_a_species_list_excludes_families_on_the_team(self):
        species = L.species_options(TEAM)
        self.assertIn('Dragonite', species)
        for taken in ('Garchomp', 'Incineroar', 'Kingambit', 'Rotom-Wash', 'Pelipper', 'Rotom-Heat'):
            self.assertNotIn(taken, species)

    def test_a_hidden_species_widens_the_other_fields(self):
        hidden = [dict(TEAM[0], species=None), *TEAM[1:]]
        self.assertEqual(L.options(hidden, [0, 'moves', 3]), L.options(hidden, [0, 'moves', 0]))
        self.assertGreater(len(L.options(hidden, [0, 'moves', 3])), len(L.options(TEAM, [0, 'moves', 3])))

    def test_there_are_25_stat_alignments(self):
        self.assertEqual(len(L.options(TEAM, [0, 'nature'])), 25)

    def test_every_list_is_sorted_and_unique(self):
        for path in ([0, 'moves', 3], [2, 'item'], [2, 'ability'], [0, 'nature']):
            values = L.options(TEAM, path)
            self.assertEqual(values, sorted(set(values)))

    def test_stat_points_have_no_option_list(self):
        with self.assertRaises(ValueError):
            L.options(TEAM, [0, 'evs'])

    def test_a_fillable_species_can_finish_its_slot(self):
        species = L.fillable_species_options(TEAM)
        self.assertNotIn('Ditto', species)
        self.assertIn('Dragonite', species)

    def test_a_mega_whose_stone_a_teammate_holds_is_not_offered(self):
        team = [dict(TEAM[0], item='Feraligite'), *TEAM[1:]]
        self.assertIn('Feraligatr-Mega', L.fillable_species_options(TEAM))
        self.assertNotIn('Feraligatr-Mega', L.fillable_species_options(team))

    def test_an_ability_with_every_item_held_is_not_offered(self):
        team = [dict(TEAM[0], item='Feraligite'), *TEAM[1:3], dict(species='Feraligatr-Mega', item=None, ability=None,
                nature=None, moves=[None] * 4), *TEAM[4:]]
        self.assertEqual(L.options(team, [3, 'ability']), [])

    def test_a_megas_own_stone_is_kept_for_it(self):
        team = [*TEAM[:3], dict(species='Tyranitar-Mega', item=None, ability='Sand Stream', nature='Adamant',
                moves=['Rock Slide', 'Knock Off', 'Protect', 'Low Kick']), *TEAM[4:]]
        team[2] = dict(team[2], item=None)
        self.assertNotIn('Tyranitarite', L.options(team, [2, 'item']))
        self.assertEqual(L.options(team, [3, 'item']), ['Tyranitarite'])


if __name__ == '__main__':
    unittest.main()
