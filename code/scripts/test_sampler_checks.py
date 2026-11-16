"""The three sampler checks of #236 with a character-level fake tokenizer and a fake model on the CPU.

No checkpoint, GPU, dllm or vault file is read, so it is in ci_tests.txt.
"""
import unittest

import torch

import masked_diffusion_memory_bench as B
import sampler_checks as S

MASK_TOKEN, MASK_ID = '\x00', 0


class Characters:
    """One token per character, its id the character's code point; the mask token is code point 0."""
    mask_token = MASK_TOKEN

    def __call__(self, text, *, add_special_tokens=False, return_offsets_mapping=False):
        out = {'input_ids': [ord(c) for c in text]}
        if return_offsets_mapping:
            out['offset_mapping'] = [(n, n + 1) for n in range(len(text))]
        return out


def decode(ids):
    return ''.join(chr(i) for i in ids)


TEAM = [{'species': 'Pikachu', 'item': 'Light Ball', 'ability': 'Static', 'nature': 'Timid',
         'moves': ['Protect', 'Thunderbolt', 'Volt Switch', 'Fake Out'],
         'statPoints': {'HP': 2, 'Atk': 0, 'Def': 0, 'SpA': 32, 'SpD': 0, 'Spe': 32}}]


class ValuesTest(unittest.TestCase):
    def setUp(self):
        self.canvas_text = f'Team: {{"species":"{MASK_TOKEN * 3}","item":"{MASK_TOKEN * 2}"}}'
        self.ids = [ord(c) for c in self.canvas_text]
        self.slots = [n for n, i in enumerate(self.ids) if i == MASK_ID]

    def test_slot_runs_split_at_gaps(self):
        self.assertEqual(S.slot_runs([3, 4, 5, 9, 10, 12]), [[3, 4, 5], [9, 10], [12]])

    def test_values_read_back_from_the_text(self):
        filled = 'Team: {"species":"Mew","item":"Or"}'
        self.assertEqual(S.values_in_text(self.canvas_text, MASK_TOKEN, filled), ['Mew', 'Or'])

    def test_structure_written_into_a_value_is_kept_with_the_value(self):
        filled = 'Team: {"species":"a",","item":"Or"}'
        self.assertEqual(S.values_in_text(self.canvas_text, MASK_TOKEN, filled), ['a",', 'Or'])

    def test_text_whose_fixed_part_changed_is_unreadable(self):
        self.assertIsNone(S.values_in_text(self.canvas_text, MASK_TOKEN, 'Team: {"name":"Mew","item":"Or"}'))

    def test_values_from_ids_decode_each_run(self):
        filled = list(self.ids)
        for n, c in zip(self.slots, 'Mew:,'):
            filled[n] = ord(c)
        self.assertEqual(S.values_from_ids(self.slots, filled, decode), ['Mew', ':,'])

    def test_a_value_stays_inside_its_quotes_only_without_quote_colon_or_comma(self):
        self.assertEqual([S.stays_inside(v) for v in ['Mew', '1-1,000', 'a"', 'item:', '', 'Spe SpA']],
                         [True, False, False, False, True, True])

    def test_field_order_is_the_json_order_of_one_pokemon(self):
        self.assertEqual(S.FIELDS[:5], ('species', 'item', 'ability', 'nature', 'move'))
        self.assertEqual(len(S.FIELDS), 14)


class CopyCasesTest(unittest.TestCase):
    def setUp(self):
        self.cases = S.copy_cases(TEAM, Characters(), MASK_ID)

    def test_one_case_per_value_of_the_second_copy(self):
        self.assertEqual(len(self.cases), 14)
        self.assertEqual(self.cases[0]['path'], 'team[0].species')

    def test_the_masked_positions_hold_the_answer_and_it_is_visible_earlier(self):
        case = self.cases[1]                                    # item: Light Ball
        self.assertEqual(decode(case['answer']), 'Light Ball')
        original = list(case['ids'])
        for n, i in zip(case['positions'], case['answer']):
            original[n] = i
        text = decode(original)
        self.assertEqual(text.count('Light Ball'), 2)
        self.assertLess(text.index('Light Ball'), case['positions'][0])
        self.assertTrue(all(case['ids'][n] == MASK_ID for n in case['positions']))
        self.assertEqual(case['ids'].count(MASK_ID), len(case['positions']))

    def test_copied_counts_exact_values_and_tokens(self):
        case = self.cases[0]
        filled = list(case['ids'])
        for n, i in zip(case['positions'], case['answer']):
            filled[n] = i
        self.assertEqual(S.copied(filled, case), (True, 1.0))
        filled[case['positions'][0]] = ord('X')
        exact, share = S.copied(filled, case)
        self.assertFalse(exact)
        self.assertAlmostEqual(share, 6 / 7)


class Copier(torch.nn.Module):
    """Predicts, at position i + 1, the token at position i - 1 of the earlier copy: a model whose logits need
    the right shift, so it copies only when shifted."""

    def __init__(self, vocab, source, target, length):
        super().__init__()
        self.vocab, self.source, self.target, self.length = vocab, source, target, length

    def forward(self, x):
        logits = torch.zeros(*x.shape, self.vocab)
        for k in range(self.length):                            # logits at target + k - 1 predict target + k
            logits[:, self.target + k - 1, x[0, self.source + k]] = 10.0
        logits[..., MASK_ID] = -1e9
        return logits


class ShiftTest(unittest.TestCase):
    def test_a_model_that_predicts_the_next_position_copies_only_with_the_shift(self):
        ids = [5, 6, 7, 8, 1, 1, 0, 0, 0, 0, 2]
        case = {'ids': ids, 'positions': [6, 7, 8, 9], 'answer': [5, 6, 7, 8]}
        model = Copier(vocab=12, source=0, target=6, length=4)
        results = {shift: S.copied(S.fill_case(lambda x: model(x), case, MASK_ID, shift), case)
                   for shift in (True, False)}
        self.assertEqual(results[True], (True, 1.0))
        self.assertFalse(results[False][0])


class RecordingTest(unittest.TestCase):
    def test_states_are_each_canvas_the_model_saw_then_the_final_one(self):
        x = torch.tensor([[1, MASK_ID, MASK_ID, 3]])
        slots = torch.tensor([1, 2])

        def forward(canvas):
            logits = torch.zeros(*canvas.shape, 6)
            logits[:, 1, 4], logits[:, 2, 5] = 2.0, 1.0
            return logits

        recorded, states = S.recording(forward)
        final = B.fill(recorded, x, slots, MASK_ID, steps=2, temperature=0)
        states.append(final)
        self.assertEqual([s[0].tolist() for s in states],
                         [[1, 0, 0, 3], [1, 4, 0, 3], [1, 4, 5, 3]])

    def test_differences_find_the_first_step_the_samplers_part(self):
        same = [[1, 0, 0], [1, 4, 0], [1, 4, 5]]
        other = [[1, 0, 0], [1, 4, 0], [1, 4, 2]]
        self.assertEqual(S.differences(same, same), {'differing_tokens': 0, 'first_step': None, 'steps': 2})
        self.assertEqual(S.differences(same, other), {'differing_tokens': 1, 'first_step': 2, 'steps': 2})

    def test_states_of_unequal_length_differ_at_the_shorter_end(self):
        self.assertEqual(S.differences([[0], [1]], [[0], [1], [2]])['first_step'], 2)


if __name__ == '__main__':
    unittest.main()
