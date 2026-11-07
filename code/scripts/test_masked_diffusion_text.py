"""The in-place masked diffusion text (#199), on real Mask Lab tasks, with a fake tokenizer and no model.

Reads the meta grid's tasks from the vault pipeline, as the Jev fill tests do, so it is not in ci_tests.txt.
"""
import json
import re
import unittest

import jev_fill as J
import mask_paste as M
import masked_diffusion_text as D

SINGLE_FIELD = ('items', 'abilities', 'moves', 'natures', 'stats')


class FakeTokenizer:
    """Mask and pad markers are one token each, a run of letters is one token, anything else is one per character."""
    mask_token, pad_token = '<|mask|>', '<|pad|>'
    pattern = r'<\|mask\|>|<\|pad\|>|[A-Za-z]+|.'

    def __call__(self, text, *, add_special_tokens, return_offsets_mapping):
        return {'offset_mapping': [m.span() for m in re.finditer(self.pattern, text, re.S)]}


class QuoteJoiningTokenizer(FakeTokenizer):
    """Joins an opening quote to the letters after it, so a value's tokens cross its quotes."""
    pattern = r'<\|mask\|>|<\|pad\|>|"?[A-Za-z]+|.'


FAKE = FakeTokenizer()
LIMIT = 10_000


def task(mask, k, source='MB158'):
    return next(t for t in J.tasks('meta') if (t['source'], t['task'], t['k']) == (source, mask, k))


def single_field_tasks():
    return [t for t in J.tasks('meta') if t['task'] in SINGLE_FIELD]


def unmasked(built, task):
    """The in-place text with each slot replaced by the unmasked team's value."""
    real = {p[0]: p[1] for p in M.json_pieces(task['original']) if not isinstance(p, str)}
    text = built.text
    for slot, path in reversed(list(zip(built.slots, D.hidden_values(task), strict=True))):
        text = text[:slot.start] + M.value_text(real[path]) + text[slot.end:]
    return text


class InPlaceTextTest(unittest.TestCase):
    def test_only_the_header_and_the_team_are_in_the_text(self):
        for t in single_field_tasks():
            lines = D.in_place_text(t, 0.55, FAKE, LIMIT).text.split('\n')
            self.assertEqual([line.split(':')[0] for line in lines],
                             ['Format', 'Rules', 'Target win rate vs top meta teams', 'Team'])
            self.assertEqual(len(json.loads(lines[3].removeprefix('Team: ').replace(FAKE.mask_token, '1'))), 6)

    def test_no_meta_team_paste_appears(self):
        text = D.in_place_text(task('items', 6), 0.55, FAKE, LIMIT).text
        for paste in J.meta_pastes():
            for line in paste.split('\n'):
                if '@' in line or line.startswith(('Ability:', 'EVs:', '- ')) or line.endswith(' Nature'):
                    self.assertNotIn(line, text)

    def test_format_rules_ask_and_team_are_present(self):
        text = D.in_place_text(task('items', 1), 0.55, FAKE, LIMIT).text
        self.assertTrue(text.startswith(f'Format: {M.FORMAT_NAME}.\nRules: {M.RULES}.\n'
                                        'Target win rate vs top meta teams: 0.55.\nTeam: [{"species":"Blaziken-Mega"'))

    def test_the_ask_is_two_decimals_and_no_ask_is_unknown(self):
        for ask, written in ((0.5, '0.50'), (0.531, '0.53'), (1, '1.00'), (None, 'unknown')):
            text = D.in_place_text(task('items', 1), ask, FAKE, LIMIT).text
            self.assertIn(f'Target win rate vs top meta teams: {written}.\n', text)

    def test_each_hidden_field_is_its_types_slot_length_of_mask_tokens(self):
        lengths = D.slot_lengths(FAKE)
        for t in single_field_tasks():
            built = D.in_place_text(t, 0.55, FAKE, LIMIT)
            self.assertEqual(len(built.slots), len(D.hidden_values(t)))
            for slot in built.slots:
                self.assertEqual(built.text[slot.start:slot.end], FAKE.mask_token * lengths[slot.field])
                self.assertEqual(slot.length, lengths[slot.field])
            self.assertEqual(built.text.count(FAKE.mask_token), sum(s.length for s in built.slots))

    def test_slot_lengths_are_the_longest_value_not_the_hidden_one(self):
        lengths = D.slot_lengths(FAKE)
        self.assertEqual(set(lengths), {'species', 'item', 'ability', 'nature', 'moves', 'statPoints'})
        self.assertEqual(lengths['statPoints'], 2)      # '32', one token per digit
        t = task('items', 6)
        real = D.value_tokens(FAKE, M.json_pieces(t['original']))
        self.assertGreater(len({real[(i, 'item')] for i in range(6)}), 1)      # the six items differ in length
        built = D.in_place_text(t, 0.55, FAKE, LIMIT)
        self.assertEqual({slot.length for slot in built.slots}, {lengths['item']})

    def test_a_stat_points_mask_hides_all_six_numbers(self):
        built = D.in_place_text(task('stats', 1), 0.55, FAKE, LIMIT)
        self.assertEqual([s.path for s in built.slots], [f'team[5].statPoints.{stat}' for stat in M.STATS])
        self.assertIn('"statPoints":{"HP":<|mask|><|mask|>,"Atk":<|mask|><|mask|>', built.text)

    def test_no_hidden_value_from_the_unmasked_team_appears(self):
        for t in (t for t in single_field_tasks() if t['task'] != 'stats'):    # numbers recur in Stat Points
            original = M.json_pieces(t['original'])
            hidden = set(D.hidden_values(t))
            shown = [p[1] for p in original if not isinstance(p, str) and p[0] not in hidden]
            text = D.in_place_text(t, 0.55, FAKE, LIMIT).text
            for path in hidden:
                value = dict(p for p in original if not isinstance(p, str))[path]
                self.assertEqual(text.count(f'"{value}"'), shown.count(value), f'{t["id"]} {path}')

    def test_every_field_not_hidden_reads_as_in_the_unmasked_team(self):
        for t in single_field_tasks():
            built = D.in_place_text(t, 0.55, FAKE, LIMIT)
            header = ''.join(D.header_parts(0.55).values()) + D.TEAM_LABEL
            self.assertEqual(unmasked(built, t), header + M.team_json(t['original']))

    def test_slots_point_at_their_mask_tokens(self):
        built = D.in_place_text(task('moves', 3), 0.55, FAKE, LIMIT)
        spans = D.offsets(FAKE, built.text)
        for slot in built.slots:
            covered = spans[slot.first_token:slot.first_token + slot.length]
            self.assertEqual((covered[0][0], covered[-1][1]), (slot.start, slot.end))

    def test_a_tokenizer_whose_tokens_cross_the_quotes_is_refused(self):
        with self.assertRaisesRegex(ValueError, 'crosses the edge'):
            D.in_place_text(task('items', 1), 0.55, QuoteJoiningTokenizer(), LIMIT)

    def test_text_over_the_limit_is_refused_with_each_parts_size(self):
        built = D.in_place_text(task('items', 1), 0.55, FAKE, LIMIT)
        total = sum(built.part_tokens.values())
        self.assertEqual(total, len(D.offsets(FAKE, built.text)))
        D.in_place_text(task('items', 1), 0.55, FAKE, total)
        with self.assertRaises(D.TextTooLong) as refused:
            D.in_place_text(task('items', 1), 0.55, FAKE, total - 1)
        for name, n in built.part_tokens.items():
            self.assertIn(f'{name} {n}', str(refused.exception))

    def test_the_same_inputs_give_the_same_output(self):
        t = task('moves', 5)
        self.assertEqual(D.in_place_text(t, 0.55, FAKE, LIMIT), D.in_place_text(json.loads(json.dumps(t)), 0.55,
                                                                                 FakeTokenizer(), LIMIT))
        self.assertEqual(D.training_target(t, 0.55, FAKE, LIMIT, FAKE.pad_token),
                         D.training_target(t, 0.55, FAKE, LIMIT, FAKE.pad_token))

    def test_the_version_names_the_in_place_text_without_meta_teams(self):
        self.assertEqual(D.in_place_text(task('items', 1), 0.55, FAKE, LIMIT).version, 'in-place-json-v1-no-meta-teams')

    def test_a_whole_hidden_pokemon_is_left_to_issue_200(self):
        with self.assertRaisesRegex(NotImplementedError, '#200'):
            D.in_place_text(task('pokemon', 1), 0.55, FAKE, LIMIT)


class TrainingTargetTest(unittest.TestCase):
    def test_the_target_parses_back_to_the_unmasked_team(self):
        for t in single_field_tasks():
            target = D.training_target(t, 0.55, FAKE, LIMIT, FAKE.pad_token)
            self.assertEqual(D.read_filled(target, FAKE.pad_token, FAKE.mask_token), t['original'])

    def test_the_target_pads_each_value_to_its_slot_length(self):
        target = D.training_target(task('items', 1), 0.55, FAKE, LIMIT, FAKE.pad_token)
        pads = D.slot_lengths(FAKE)['item'] - 5         # Never, -, Melt, ' ', Ice
        self.assertIn('"item":"Never-Melt Ice' + FAKE.pad_token * pads + '"', target)
        self.assertEqual(len(D.offsets(FAKE, target)),
                         len(D.offsets(FAKE, D.in_place_text(task('items', 1), 0.55, FAKE, LIMIT).text)))

    def test_stat_points_carry_six_numbers_and_the_battle_paste_is_unchanged(self):
        t = task('stats', 6)
        target = D.training_target(t, 0.55, FAKE, LIMIT, FAKE.pad_token)
        team = D.read_filled(target, FAKE.pad_token, FAKE.mask_token)
        self.assertTrue(all(tuple(mon['statPoints']) == M.STATS for mon in team))
        self.assertIn('"Atk":0', M.team_json(team))
        self.assertEqual(M.team_paste(team), M.team_paste(t['original']))
        self.assertNotIn('0 Atk', M.team_paste(team))

    def test_an_unfilled_text_is_refused(self):
        built = D.in_place_text(task('items', 1), 0.55, FAKE, LIMIT)
        with self.assertRaisesRegex(ValueError, 'mask tokens'):
            D.read_filled(built.text, FAKE.pad_token, FAKE.mask_token)


if __name__ == '__main__':
    unittest.main()
