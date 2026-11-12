"""The memory benchmark's sampler, schedule and batch ladder (#170), with a fake model on the CPU.

No checkpoint, GPU or vault file is read, so it is in ci_tests.txt.
"""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import torch

import masked_diffusion_memory_bench as B

MASK, VOCAB = 7, 11


class Fixed(torch.nn.Module):
    """Logits that prefer token (position + row) % VOCAB, never the mask token, with a confidence that rises
    with position, so confidence order is right to left."""

    def forward(self, x):
        b, n = x.shape
        positions = torch.arange(n).expand(b, n)
        best = (positions + torch.arange(b)[:, None]) % VOCAB
        best = torch.where(best == MASK, (best + 1) % VOCAB, best)
        logits = torch.zeros(b, n, VOCAB)
        logits.scatter_(2, best[..., None], (1.0 + positions[..., None]).float())
        logits[..., MASK] = -1e9
        return logits


def canvas(batch=2, length=12, slots=(3, 4, 5, 9, 10)):
    x = torch.arange(length).expand(batch, length).clone() % 5
    slots = torch.tensor(slots)
    x[:, slots] = MASK
    return x, slots


class ScheduleTest(unittest.TestCase):
    def test_every_mask_is_committed_once(self):
        for masks, steps in [(354, 128), (5, 128), (128, 128), (59, 7), (1, 1)]:
            counts = B.schedule(masks, steps)
            self.assertEqual(sum(counts), masks)
            self.assertLessEqual(len(counts), steps)
            self.assertTrue(all(c > 0 for c in counts))

    def test_linear_schedule_spreads_masks_evenly(self):
        self.assertEqual(B.schedule(10, 5), [2, 2, 2, 2, 2])
        self.assertEqual(B.schedule(3, 6), [1, 1, 1])
        self.assertEqual(max(B.schedule(354, 128)) - min(B.schedule(354, 128)), 1)


class FillTest(unittest.TestCase):
    def test_fills_every_slot_and_leaves_the_rest(self):
        x, slots = canvas()
        out = B.fill(Fixed(), x, slots, MASK, steps=3, temperature=0)
        self.assertFalse((out == MASK).any())
        keep = torch.ones(x.shape[1], dtype=torch.bool)
        keep[slots] = False
        self.assertTrue(torch.equal(out[:, keep], x[:, keep]))
        expected = Fixed()(x).argmax(-1)
        self.assertTrue(torch.equal(out[:, slots], expected[:, slots]))

    def test_most_confident_slots_commit_first(self):
        x, slots = canvas(batch=1)
        seen = []
        B.fill(lambda c: (seen.append(c.clone()), Fixed()(c))[1], x, slots, MASK, steps=5, temperature=0)
        first_committed = (seen[1] != MASK)[0, slots]
        self.assertEqual(first_committed.tolist(), [False, False, False, False, True])

    def test_slot_scope_gives_the_same_team_as_scoring_every_position(self):
        x, slots = canvas()
        every = B.fill(Fixed(), x, slots, MASK, steps=2, temperature=0)
        only = B.fill(lambda c: Fixed()(c)[:, slots], x, slots, MASK, steps=2, temperature=0)
        self.assertTrue(torch.equal(every, only))

    def test_temperature_draws_differ_by_seed_and_repeat_with_it(self):
        x, slots = canvas(batch=4)
        flat = lambda c: torch.zeros(c.shape[0], c.shape[1], VOCAB)
        draw = lambda seed: B.fill(flat, x, slots, MASK, 2, 1.0, torch.Generator().manual_seed(seed))
        self.assertTrue(torch.equal(draw(1), draw(1)))
        self.assertFalse(torch.equal(draw(1), draw(2)))

    def test_no_activations_are_kept_for_gradients(self):
        """Autograd would keep every layer's activations alive through each step and inflate the peak (#170)."""
        x, slots = canvas()
        weight = torch.nn.Parameter(torch.ones(1))
        seen = []
        B.fill(lambda c: (seen.append(torch.is_grad_enabled()), Fixed()(c) * weight)[1], x, slots, MASK, 2, 0)
        self.assertEqual(seen, [False, False])

    def test_gumbel_noise_is_float32_and_never_overflows(self):
        logits = torch.full((2, 3, VOCAB), 500.0)
        logits[..., 4] = 600.0
        picked = B.gumbel_argmax(logits, 1.0, torch.Generator().manual_seed(0))
        self.assertTrue((picked == 4).all())

    def test_shift_right_moves_each_prediction_one_position(self):
        logits = torch.arange(4.0).view(1, 4, 1)
        self.assertEqual(B.shift_right(logits).flatten().tolist(), [0.0, 0.0, 1.0, 2.0])


class LadderTest(unittest.TestCase):
    def test_stops_at_the_first_batch_size_over_the_ceiling_and_skips_finished_runs(self):
        with tempfile.TemporaryDirectory() as d:
            directory, launched = Path(d), []

            def launch(arguments):
                batch, scope = int(arguments[arguments.index('--batch') + 1]), arguments[arguments.index('--scope') + 1]
                launched.append((scope, batch))
                over = batch >= (8 if scope == 'all' else 32)
                path = Path(arguments[arguments.index('--out') + 1])
                path.write_text(json.dumps({'status': 'over ceiling' if over else 'fits'}))
                return B.OVER if over else 0

            fitted = B.ladder(directory, 'qwen3', launch)
            self.assertEqual(fitted, {'all': [1, 2, 4], 'slots': [1, 2, 4, 8, 16]})
            self.assertEqual(launched[:4], [('all', 1), ('all', 2), ('all', 4), ('all', 8)])
            launched.clear()
            self.assertEqual(B.ladder(directory, 'qwen3', launch), fitted)
            self.assertEqual(launched, [])

    def test_a_run_that_crashed_stops_its_scope(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(B.ladder(Path(d), 'dream', lambda _: 1, scopes=['all']), {'all': []})
            self.assertEqual(B.crashed_settings(Path(d), 'dream', {'all': []}, scopes=['all']), ['all batch 1'])

    def test_a_setting_over_the_ceiling_is_not_a_crash(self):
        with tempfile.TemporaryDirectory() as d:
            path = B.result_path(Path(d), 'qwen3', 'all', 2)
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps({'status': 'over ceiling'}))
            self.assertEqual(B.crashed_settings(Path(d), 'qwen3', {'all': [1]}, scopes=['all']), [])
            self.assertEqual(B.crashed_settings(Path(d), 'qwen3', {'all': list(B.BATCHES)}, scopes=['all']), [])


class Int8Test(unittest.TestCase):
    def test_int8_layer_matches_the_float_layer(self):
        torch.manual_seed(0)
        linear = torch.nn.Linear(64, 96).to(torch.bfloat16)
        x = torch.randn(2, 5, 64, dtype=torch.bfloat16)
        q = B.Int8Linear(linear, 'cpu')
        self.assertEqual(q.weight.dtype, torch.int8)
        self.assertEqual(q(x).shape, (2, 5, 96))
        error = (q(x).float() - linear(x).float()).abs().max() / linear(x).float().abs().max()
        self.assertLess(float(error), 0.02)

    def test_every_linear_is_replaced_except_one_tied_to_an_embedding(self):
        model = torch.nn.Sequential(torch.nn.Embedding(10, 8), torch.nn.Linear(8, 8), torch.nn.ReLU(),
                                    torch.nn.Sequential(torch.nn.Linear(8, 8, bias=False)), torch.nn.Linear(8, 10))
        tied = torch.nn.Linear(8, 10, bias=False)
        tied.weight = model[0].weight
        both = torch.nn.ModuleList([model, tied])
        self.assertEqual(B.quantize_linears(both, 'cpu'), 3)
        self.assertIsInstance(model[3][0], B.Int8Linear)
        self.assertNotIsInstance(both[1], B.Int8Linear)
        self.assertEqual(model(torch.tensor([[1, 2]])).shape, (1, 2, 10))

    def test_agreement_of_identical_and_of_different_predictions(self):
        a = torch.log_softmax(torch.tensor([[2.0, 0.0, 0.0], [0.0, 3.0, 0.0]]), -1)
        same = B.agreement(a, a)
        self.assertEqual(same['top1_same'], 1.0)
        self.assertAlmostEqual(same['kl_mean'], 0.0, places=6)
        b = torch.log_softmax(torch.tensor([[2.0, 0.0, 0.0], [0.0, 0.0, 3.0]]), -1)
        other = B.agreement(a, b)
        self.assertEqual(other['top1_same'], 0.5)
        self.assertGreater(other['kl_max'], other['kl_mean'])


class RemoteCodeTest(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec('transformers'), 'transformers is not in code/requirements.txt')
    def test_import_check_skips_dllm_only(self):
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / 'modeling.py'
            f.write_text('import torch\nimport json\nif __name__ == "__main__":\n    import dllm\n')
            from transformers import dynamic_module_utils as dm
            with B.ignoring_main_only_imports():
                self.assertNotIn('dllm', dm.get_imports(str(f)))
                self.assertIn('torch', dm.get_imports(str(f)))
            self.assertIn('dllm', dm.get_imports(str(f)))

    def test_out_of_memory_is_recognised(self):
        self.assertTrue(B.is_over_ceiling(RuntimeError('MPS backend out of memory (MPS allocated: 17.2 GiB)')))
        self.assertFalse(B.is_over_ceiling(RuntimeError('shape mismatch')))
        self.assertFalse(B.is_over_ceiling(ValueError('out of memory')))


if __name__ == '__main__':
    unittest.main()
