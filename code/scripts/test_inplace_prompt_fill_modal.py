"""The Modal first-try fills (#238, inplace_prompt_fill_modal.py): which masks are left to fill, and the check
that compares today's-prompt fills on Modal with #237's laptop fills. No Modal, GPU or Showdown is used."""
import json
import tempfile
import unittest
import unittest.mock
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

    def test_a_prompt_with_a_mask_list_fills_only_its_masks(self):
        tasks = [{'id': 'a'}, {'id': 'b'}, {'id': 'c'}]
        left = M.pending(['inplace_header', 'inplace_example'], tasks, {'inplace_example': {'a'}},
                         {'inplace_example': {'a', 'c'}})
        self.assertEqual([(p, t['id']) for p, t in left],
                         [('inplace_header', 'a'), ('inplace_header', 'b'), ('inplace_header', 'c'),
                          ('inplace_example', 'c')])

    def test_the_frozen_lists_read_as_sets_and_a_missing_file_as_none(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'fitting-ids.json'
            self.assertEqual(M.fitting(path), {})
            path.write_text(json.dumps({'inplace_example': ['b', 'a']}))
            self.assertEqual(M.fitting(path), {'inplace_example': {'a', 'b'}})

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


class GpuTest(unittest.TestCase):
    def test_every_model_fills_on_the_l40s_unless_it_names_its_own_card(self):
        with unittest.mock.patch.dict('os.environ', clear=False) as env:
            env.pop('INPLACE_GPU', None)
            self.assertEqual({M.gpu_for(k) for k in M.KEYS}, {'L40S'})
            with unittest.mock.patch.object(M, 'GPU_BY_KEY', {'qwen3': 'A10'}):
                self.assertEqual((M.gpu_for('qwen3'), M.gpu_for('diffugpt_m')), ('A10', 'L40S'))

    def test_inplace_gpu_overrides_every_model(self):
        with unittest.mock.patch.dict('os.environ', {'INPLACE_GPU': 'L4'}):
            self.assertEqual(M.gpu_for('diffugpt_m'), 'L4')

    def test_a_resume_on_another_card_is_named(self):
        rows = [{'id': 'a', 'device': 'cuda NVIDIA L4'}, {'id': 'b', 'device': 'cuda NVIDIA L4'}, {'id': 'c'}]
        self.assertEqual(M.other_devices(rows, 'cuda NVIDIA A10'), ['cuda NVIDIA L4'])
        self.assertEqual(M.other_devices(rows, 'cuda NVIDIA L4'), [])
        self.assertEqual(M.other_devices([], 'cuda NVIDIA A10'), [])


    def test_illada_gets_three_hours_and_the_small_models_one(self):
        self.assertEqual(M.timeout_for('illada'), 3 * 60 * 60)
        self.assertEqual({M.timeout_for(k) for k in M.KEYS if k != 'illada'}, {M.TIMEOUT_SECONDS})

if __name__ == '__main__':
    unittest.main()
