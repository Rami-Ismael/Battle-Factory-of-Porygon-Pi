"""The Modal first-try fills (#238, inplace_prompt_fill_modal.py): which masks are left to fill, and the check
that compares today's-prompt fills on Modal with #237's laptop fills. No Modal, GPU or Showdown is used."""
import json
import tempfile
import unittest
from pathlib import Path

import inplace_prompt_fill_modal as M


def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(r) + '\n' for r in rows))


class PendingTest(unittest.TestCase):
    def test_each_prompt_fills_every_mask_not_yet_drawn_in_prompt_order(self):
        tasks = [{'id': 'a'}, {'id': 'b'}, {'id': 'c'}]
        left = M.pending(['inplace_header', 'inplace_instructions'], tasks, {'inplace_header': {'a', 'c'}})
        self.assertEqual([(p, t['id']) for p, t in left],
                         [('inplace_header', 'b'), ('inplace_instructions', 'a'), ('inplace_instructions', 'b'),
                          ('inplace_instructions', 'c')])

    def test_nothing_is_left_once_every_mask_is_drawn(self):
        tasks = [{'id': 'a'}]
        self.assertEqual(M.pending(['inplace_header'], tasks, {'inplace_header': {'a'}}), [])


class SameAsLaptopTest(unittest.TestCase):
    def test_counts_token_for_token_matches_and_each_runs_legal_masks(self):
        with tempfile.TemporaryDirectory() as d:
            modal, laptop = Path(d) / 'modal', Path(d) / 'laptop'
            write(modal / 'inplace_header/qwen3/fills.jsonl',
                  [{'id': 'a', 'slot_ids': [[1, 2]], 'legal': True}, {'id': 'b', 'slot_ids': [[3]], 'legal': False}])
            write(laptop / 'inplace_header/qwen3/fills.jsonl',
                  [{'id': 'a', 'slot_ids': [[1, 2]], 'legal': True}, {'id': 'b', 'slot_ids': [[4]], 'legal': True}])
            self.assertEqual(M.same_as_laptop('qwen3', modal, laptop),
                             {'masks': 2, 'same_tokens': 1, 'legal_modal': 1, 'legal_laptop': 2})


if __name__ == '__main__':
    unittest.main()
