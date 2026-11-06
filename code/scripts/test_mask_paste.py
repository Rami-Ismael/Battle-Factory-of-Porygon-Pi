"""The mask and paste helpers the Jev fill and the masked diffusion prompt share."""
import unittest

import mask_paste as M

NINETALES = {'slot': 4, 'species': 'Ninetales-Alola', 'item': 'Light Clay', 'ability': 'Snow Warning',
             'moves': ['Aurora Veil', 'Blizzard', 'Moonblast', 'Protect'], 'nature': 'Timid',
             'statPoints': {'HP': 2, 'Atk': 0, 'Def': 0, 'SpA': 32, 'SpD': 0, 'Spe': 32},
             'ivs': dict.fromkeys(M.STATS, 31), 'level': 50}


class SharedHelpersTest(unittest.TestCase):
    def test_paths_round_trip(self):
        for text in ('team[1]', 'team[3].item', 'team[0].moves[2]', 'team[5].statPoints'):
            self.assertEqual(M.path_text(M.parse_path(text)), text)

    def test_a_path_that_is_not_a_mask_path_is_refused(self):
        with self.assertRaises(ValueError):
            M.parse_path('team[1].moves[')

    def test_fill_order_is_species_ability_item_moves_alignment_stat_points(self):
        self.assertEqual(M.SLOT_ORDER, ('species', 'ability', 'item', 'moves', 'nature', 'statPoints'))

    def test_a_blank_slot_has_every_field_hidden_and_the_fixed_ivs_and_level(self):
        slot = M.blank_slot(3)
        self.assertEqual(slot['slot'], 3)
        self.assertEqual([slot[name] for name in ('species', 'item', 'ability', 'nature', 'statPoints')], [None] * 5)
        self.assertEqual(slot['moves'], [None] * 4)
        self.assertEqual((slot['ivs'], slot['level']), (dict.fromkeys(M.STATS, 31), 50))

    def test_team_paste_writes_the_battle_steps_text_and_skips_zero_stat_points(self):
        self.assertEqual(M.team_paste([NINETALES, None]),
                         'Ninetales-Alola @ Light Clay\nAbility: Snow Warning\nLevel: 50\n'
                         'EVs: 2 HP / 32 SpA / 32 Spe\nTimid Nature\n'
                         '- Aurora Veil\n- Blizzard\n- Moonblast\n- Protect\n')

    def test_format_and_rules_are_the_ones_jev_is_sent(self):
        self.assertEqual(M.FORMAT_NAME, 'Champions VGC 2026 Reg Set M-B')
        self.assertEqual(M.RULES, 'doubles, bring 6 pick 4, level 50, Species Clause, Item Clause')


if __name__ == '__main__':
    unittest.main()
