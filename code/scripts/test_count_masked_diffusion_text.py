"""The count over every mask (#201), with a fake tokenizer and no model.

Reads the meta grid's tasks from the vault pipeline, as the in-place text tests do, so it is not in ci_tests.txt.
"""
import unittest

import count_masked_diffusion_text as C
import masked_diffusion_text as D
from test_masked_diffusion_text import FAKE

FAKE_CANDIDATE = C.Candidate('fake', 'Fake tokenizer', 'none/fake', '0' * 40, 0)
BROKEN_CANDIDATE = C.Candidate('broken', 'Broken tokenizer', 'none/broken', '0' * 40, 0)


def load(candidate):
    if candidate is BROKEN_CANDIDATE:
        raise OSError('no such repository')
    return FAKE


class CountTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tasks = C.all_tasks(['meta'])
        cls.report = C.count([BROKEN_CANDIDATE, FAKE_CANDIDATE], cls.tasks, load=load, limit=C.NO_LIMIT)
        cls.broken, cls.fake = cls.report['candidates']

    def test_a_tokenizer_that_fails_to_load_is_not_counted(self):
        self.assertFalse(self.broken['counted'])
        self.assertIn('no such repository', self.broken['reason'])
        self.assertNotIn('tokens', self.broken)
        self.assertNotIn('largest', self.broken)

    def test_every_mask_is_counted_with_the_builders_own_length(self):
        self.assertEqual(self.fake['masks'], len(self.tasks))
        self.assertEqual(self.fake['refused'], [])
        ask = None if self.fake['ask'] == 'unknown' else float(self.fake['ask'])
        for grid, task in self.tasks[::37]:
            built = D.in_place_text(task, ask, FAKE, C.NO_LIMIT)
            self.assertEqual(self.fake['tokens'][grid][task['id']][0], sum(built.part_tokens.values()))

    def test_the_ask_counted_is_the_longest(self):
        ask = None if self.fake['ask'] == 'unknown' else float(self.fake['ask'])
        task = self.tasks[0][1]
        counted = sum(D.in_place_text(task, ask, FAKE, C.NO_LIMIT).part_tokens.values())
        for other in C.ASKS:
            self.assertLessEqual(sum(D.in_place_text(task, other, FAKE, C.NO_LIMIT).part_tokens.values()), counted)

    def test_the_largest_is_the_longest_mask_and_names_it(self):
        counts = [n for grid in self.fake['tokens'].values() for n, _ in grid.values()]
        big = self.fake['largest']
        self.assertEqual(big['tokens'], max(counts))
        self.assertEqual(big['ties'], counts.count(max(counts)))
        self.assertEqual(self.fake['tokens'][big['grid']][big['id']][0], big['tokens'])
        self.assertTrue(big['name'].startswith(f"meta grid {big['source']}, "))

    def test_masks_over_the_limit_are_counted_strictly_above_it(self):
        counts = sorted(n for grid in self.fake['tokens'].values() for n, _ in grid.values())
        limit = counts[len(counts) // 2]
        report = C.count([FAKE_CANDIDATE], self.tasks, load=load, limit=limit)
        self.assertEqual(report['candidates'][0]['over_limit'], sum(n > limit for n in counts))

    def test_the_summary_lists_a_failed_tokenizer_as_not_counted(self):
        table = C.summary(self.report)
        self.assertIn('| Broken tokenizer | not counted | OSError: no such repository |', table)
        self.assertIn(f"| Fake tokenizer | {self.fake['largest']['tokens']:,} |", table)

    def test_every_candidate_of_162_is_listed_once_at_a_pinned_revision(self):
        self.assertEqual([c.label for c in C.CANDIDATES],
                         ['Masked Qwen3 0.6B', 'Masked Qwen2.5 Coder 0.5B', 'DiffuGPT small', 'DiffuGPT medium',
                          'DiffuLLaMA 7B', 'Dream 7B', 'LLaDA 8B'])
        for c in C.CANDIDATES:
            self.assertRegex(c.revision, r'^[0-9a-f]{40}$')

    def test_a_mask_token_other_than_the_one_162_found_is_refused(self):
        class Wrong:
            mask_token = '<|mask|>'

            def __call__(self, text, add_special_tokens):
                return type('Encoded', (), {'input_ids': [1, 2]})()
        with self.assertRaisesRegex(ValueError, 'not \\[151669\\]'):
            C.check_mask_token(Wrong(), C.CANDIDATES[0])


if __name__ == '__main__':
    unittest.main()
