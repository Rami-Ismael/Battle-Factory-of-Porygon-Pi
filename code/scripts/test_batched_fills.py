"""Batched in-place fills (#443) on the CPU, with fake models that ignore padding.

Checks:
  1. fill_rows fills rows of different lengths and mask counts in one batch, and every row gets exactly the tokens
     fill gives it alone with the same seed, whatever the batch order.
  2. make_row_forward's padding mask hides padding from real tokens for DiffuGPT (GPT-2, shifted) and the masked
     Qwen models (2D padding mask), so a padded row's logits are the logits it gets alone.
  3. largest_batch climbs the ladder until the ceiling or the headroom.
  4. The held-out grid filled batch at a time keeps every visible value, fills only the slots, and gives each mask
     the fill it gets one at a time; redraws batched round by round match the one-at-a-time loop.

No checkpoint, GPU or vault file is read, so this test is in ci_tests.txt.
"""
import json
import tempfile
import unittest
import unittest.mock
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import torch

import inplace_prompt_grid as G
import masked_diffusion_memory_bench as B
import masked_diffusion_sampler as MS
from test_inplace_prompt_grid import PROMPT, TASKS, FakeTokenizer, FakeValidator

MASK_ID, VOCAB = 0, 17


def padding_blind_logits(x: torch.Tensor, attention: torch.Tensor, vocab: int = VOCAB) -> torch.Tensor:
    """Logits [B, L, V] that depend on every real token of a row and on no padded one."""
    length = x.shape[1]
    pos = torch.arange(length, dtype=torch.float32)
    tok = torch.arange(vocab, dtype=torch.float32)
    real = x.float() * attention.float()
    row_mix = (real * (pos[None, :] + 1.0)).sum(dim=1, keepdim=True)
    logits = torch.sin(pos[None, :, None] * 0.7 + tok[None, None, :] * 0.5 + row_mix[:, :, None] * 0.03) * 2.0
    logits[..., MASK_ID] = -1e9
    return logits


def alone_forward(slots: torch.Tensor):
    """fill's forward for one row at batch size 1, scope slots."""
    return lambda x: padding_blind_logits(x, torch.ones_like(x, dtype=torch.bool))[:, slots]


def row_forward(x: torch.Tensor, attention: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
    logits = padding_blind_logits(x, attention)
    return logits.gather(1, keep[..., None].expand(-1, -1, logits.shape[-1]))


def make_rows() -> tuple[list[list[int]], list[list[int]]]:
    """Five texts of different lengths and mask counts; tokens 1 to VOCAB - 1, masks at the slots."""
    rng = torch.Generator().manual_seed(7)
    rows, slots = [], []
    for length, masks in ((9, 2), (23, 11), (14, 5), (31, 4), (18, 9)):
        ids = torch.randint(1, VOCAB, (length,), generator=rng).tolist()
        where = sorted(torch.randperm(length - 1, generator=rng)[:masks].add(1).tolist())
        for p in where:
            ids[p] = MASK_ID
        rows.append(ids)
        slots.append(where)
    return rows, slots


def fill_alone(ids: list[int], where: list[int], seed: int, setting: MS.SamplerSetting) -> list[int]:
    slots = torch.tensor(where)
    out = B.fill(alone_forward(slots), torch.tensor([ids]), slots, MASK_ID,
                 generator=torch.Generator().manual_seed(seed), setting=setting)
    return out[0].tolist()


class FillRowsTest(unittest.TestCase):
    SETTINGS = (MS.SHARED_SAMPLER, replace(MS.SHARED_SAMPLER, steps=6), replace(MS.SHARED_SAMPLER, steps='per_mask'),
                replace(MS.SHARED_SAMPLER, steps=6, temperature=0.0))

    def test_each_row_gets_the_fill_it_gets_alone(self):
        rows, slots = make_rows()
        seeds = [101, 202, 303, 404, 505]
        for setting in self.SETTINGS:
            with self.subTest(steps=setting.steps, temperature=setting.temperature):
                alone = [fill_alone(r, s, seed, setting) for r, s, seed in zip(rows, slots, seeds, strict=True)]
                generators = [torch.Generator().manual_seed(seed) for seed in seeds]
                batched = B.fill_rows(row_forward, rows, slots, MASK_ID, setting, generators)
                self.assertEqual(batched, alone)
                for ids, filled, where in zip(rows, batched, slots, strict=True):
                    self.assertEqual(len(filled), len(ids))
                    self.assertEqual([t for n, t in enumerate(filled) if n not in where],
                                     [t for n, t in enumerate(ids) if n not in where])
                    self.assertNotIn(MASK_ID, filled)

    def test_batch_order_does_not_change_a_row(self):
        rows, slots = make_rows()
        seeds = [11, 22, 33, 44, 55]
        forward = B.fill_rows(row_forward, rows, slots, MASK_ID, MS.SHARED_SAMPLER,
                              [torch.Generator().manual_seed(s) for s in seeds])
        backward = B.fill_rows(row_forward, rows[::-1], slots[::-1], MASK_ID, MS.SHARED_SAMPLER,
                               [torch.Generator().manual_seed(s) for s in seeds[::-1]])
        self.assertEqual(forward, backward[::-1])

    def test_finished_rows_leave_the_batch(self):
        rows, slots = make_rows()
        served = []
        B.fill_rows(row_forward, rows, slots, MASK_ID, replace(MS.SHARED_SAMPLER, steps='per_mask'),
                    [None] * len(rows), on_step=served.append)
        self.assertEqual(len(served), max(map(len, slots)))
        self.assertEqual([sum(n in live for live in served) for n in range(len(rows))], [len(s) for s in slots])

    def test_refuses_slots_that_are_not_the_masks(self):
        rows, slots = make_rows()
        with self.assertRaises(ValueError):
            B.fill_rows(row_forward, rows[:1], [slots[0][:-1]], MASK_ID, MS.SHARED_SAMPLER, [None])
        with self.assertRaises(ValueError):
            B.fill_rows(row_forward, rows[:2], slots[:2], MASK_ID, MS.SHARED_SAMPLER, [None])


class MaskingCore(torch.nn.Module):
    """A fake transformer whose output at each position sums the real tokens it is allowed to see: a key position
    counts when the additive 4D mask is zero there (GPT-2, Llama) or the 2D mask is one (masked Qwen)."""

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> SimpleNamespace:
        seen = (attention_mask[:, 0, 0, :] == 0) if attention_mask.ndim == 4 else attention_mask.bool()
        return SimpleNamespace(last_hidden_state=padding_blind_logits(input_ids, seen))


class FakeModel:
    dtype = torch.float32

    def __init__(self, family: str):
        core = MaskingCore()
        self.lm_head = torch.nn.Identity()
        if family == 'gpt2':
            self.transformer = core
        else:
            self.model = core


class RowForwardTest(unittest.TestCase):
    def test_padded_row_scores_as_it_does_alone(self):
        rows, slots = make_rows()
        for family, shifted in (('gpt2', True), ('a2d', False), ('llama', True)):
            with self.subTest(family=family):
                forward = B.make_row_forward(FakeModel(family), family, right_shift=shifted)
                width, depth = max(map(len, rows)), max(map(len, slots))
                x = torch.zeros(len(rows), width, dtype=torch.long)
                attention = torch.zeros(x.shape, dtype=torch.bool)
                keep = torch.zeros(len(rows), depth, dtype=torch.long)
                for n, (ids, where) in enumerate(zip(rows, slots, strict=True)):
                    x[n, :len(ids)], attention[n, :len(ids)] = torch.tensor(ids), True
                    keep[n, :len(where)] = torch.tensor(where)
                batched = forward(x, attention, keep)
                # a padded key that leaked in would move a logit by about 0.03 x its token id x its position; the
                # tolerance only absorbs float rounding of the fake's sin on a different shape
                for n, (ids, where) in enumerate(zip(rows, slots, strict=True)):
                    alone = forward(torch.tensor([ids]), torch.ones(1, len(ids), dtype=torch.bool),
                                    torch.tensor([where]))
                    torch.testing.assert_close(batched[n, :len(where)], alone[0], rtol=0, atol=1e-6)
                    full = padding_blind_logits(torch.tensor([ids]), torch.ones(1, len(ids), dtype=torch.bool))[0]
                    expected = full[[p - 1 for p in where]] if shifted else full[where]
                    torch.testing.assert_close(alone[0], expected, rtol=0, atol=1e-6)

    def test_padding_mask_blocks_only_padded_keys(self):
        attention = torch.tensor([[True, True, False], [True, True, True]])
        mask = B.padding_mask(FakeModel('gpt2'), attention)
        self.assertEqual(tuple(mask.shape), (2, 1, 3, 3))
        self.assertTrue((mask[1] == 0).all())
        self.assertTrue((mask[0, 0, :, :2] == 0).all())
        self.assertTrue((mask[0, 0, :, 2] == torch.finfo(torch.float32).min).all())

    def test_families_without_a_padding_path_refuse_padded_rows(self):
        x, keep = torch.ones(2, 4, dtype=torch.long), torch.zeros(2, 1, dtype=torch.long)
        attention = torch.tensor([[True] * 4, [True, True, True, False]])
        for family in ('dream', 'llada'):
            with self.assertRaises(NotImplementedError):
                B.make_row_forward(FakeModel('llama'), family)(x, attention, keep)


class LargestBatchTest(unittest.TestCase):
    def test_stops_at_the_ceiling(self):
        def peak_at(batch):
            if batch >= 32:
                raise torch.OutOfMemoryError('CUDA out of memory')
            return batch * 10
        result = B.largest_batch(peak_at, capacity=10_000)
        self.assertEqual(result['batch'], 16)
        self.assertEqual(result['ladder'][-1], {'batch': 32, 'status': 'over ceiling'})

    def test_stops_above_the_headroom(self):
        result = B.largest_batch(lambda b: b * 100, capacity=1000, headroom=0.9)
        self.assertEqual(result['batch'], 8)
        self.assertEqual(result['ladder'][-1]['status'], 'over headroom')

    def test_other_errors_and_no_fit_raise(self):
        with self.assertRaises(ValueError):
            B.largest_batch(lambda b: (_ for _ in ()).throw(ValueError('bad')), capacity=10)
        with self.assertRaises(RuntimeError):
            B.largest_batch(lambda b: 100, capacity=10)


def truth_row_forward(tokenizer, tasks, prompt=PROMPT):
    """A row forward that writes each mask's true value: it finds which task a row is by its visible tokens."""
    import masked_diffusion_text as D
    known = []
    for task in tasks:
        c = G.canvas(task, prompt, tokenizer, MASK_ID)
        target = D.training_target(task, G.ASK, tokenizer, G.LIMIT, tokenizer.pad_token, prompt.statement,
                                   prompt.example)
        known.append((c['ids'], tokenizer(target, add_special_tokens=False)['input_ids']))

    def forward(x, attention, keep):
        logits = torch.full((x.shape[0], keep.shape[1], len(tokenizer) + 5), -1e9)
        for n in range(x.shape[0]):
            ids = x[n, :int(attention[n].sum())].tolist()
            matches = [truth for text, truth in known if len(text) == len(ids)
                       and all(t == MASK_ID or t == i for t, i in zip(text, ids, strict=True))]
            assert len(matches) == 1, 'a row matched no task or several'
            for s, position in enumerate(keep[n].tolist()):
                logits[n, s, matches[0][position]] = 0.0
        return logits
    return forward


def noisy_pair(tokenizer):
    """The same padding-blind fake model as fill's forward_for and as fill_rows' row forward."""
    vocab = len(tokenizer) + 5

    def forward_for(slots):
        return lambda x: padding_blind_logits(x, torch.ones_like(x, dtype=torch.bool), vocab)[:, slots]

    def rows(x, attention, keep):
        logits = padding_blind_logits(x, attention, vocab)
        return logits.gather(1, keep[..., None].expand(-1, -1, vocab))
    return forward_for, rows


class BatchedGridTest(unittest.TestCase):
    def test_batched_fills_keep_visible_values_and_fill_only_the_slots(self):
        tokenizer = FakeTokenizer()
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / 'fills.jsonl'
            filled = G.fill_grid(TASKS, PROMPT, tokenizer, None, MASK_ID, out, FakeValidator(),
                                 row_forward=truth_row_forward(tokenizer, TASKS), batch=2)
            rows = G.read_rows(out)
        self.assertEqual(filled, len(TASKS))
        self.assertEqual(sorted(r['id'] for r in rows), sorted(t['id'] for t in TASKS))
        self.assertGreater(len({r['prompt_tokens'] for r in rows}), 1)            # masks of different lengths
        for r in rows:
            self.assertTrue(r['visible_kept'], r['id'])
            self.assertTrue(r['legal'], r['id'])
            self.assertTrue(r['recovered'], r['id'])
            self.assertEqual(r['batch_size'], 2)
            self.assertIn(r['batch_rows'], (1, 2))

    def test_batched_grid_gives_each_mask_its_one_at_a_time_fill(self):
        tokenizer = FakeTokenizer()
        forward_for, rows_forward = noisy_pair(tokenizer)
        with tempfile.TemporaryDirectory() as d:
            G.fill_grid(TASKS, PROMPT, tokenizer, forward_for, MASK_ID, Path(d) / 'one.jsonl', FakeValidator())
            G.fill_grid(TASKS, PROMPT, tokenizer, None, MASK_ID, Path(d) / 'many.jsonl', FakeValidator(),
                        row_forward=rows_forward, batch=3)
            one = {r['id']: r for r in G.read_rows(Path(d) / 'one.jsonl')}
            many = {r['id']: r for r in G.read_rows(Path(d) / 'many.jsonl')}
        self.assertEqual(set(one), set(many))
        for task_id, r in one.items():
            self.assertEqual(many[task_id]['slot_ids'], r['slot_ids'], task_id)
            self.assertEqual(many[task_id]['seed'], r['seed'])
            self.assertEqual(many[task_id]['forwards'], r['forwards'])

    def test_batched_redraws_match_the_one_at_a_time_loop(self):
        tokenizer = FakeTokenizer()
        forward_for, rows_forward = noisy_pair(tokenizer)
        with tempfile.TemporaryDirectory() as d:
            first = Path(d) / 'fills.jsonl'
            G.fill_grid(TASKS, PROMPT, tokenizer, forward_for, MASK_ID, first, FakeValidator())
            self.assertFalse(any(r['legal'] for r in G.read_rows(first)))
            G.redraw_grid(TASKS, first, PROMPT, tokenizer, forward_for, MASK_ID, Path(d) / 'one.jsonl',
                          FakeValidator())
            G.redraw_grid(TASKS, first, PROMPT, tokenizer, None, MASK_ID, Path(d) / 'many.jsonl', FakeValidator(),
                          row_forward=rows_forward, batch=2)
            one = {(r['id'], r['draw']): r['slot_ids'] for r in G.read_rows(Path(d) / 'one.jsonl')}
            many = {(r['id'], r['draw']): r['slot_ids'] for r in G.read_rows(Path(d) / 'many.jsonl')}
        # the noisy fake writes a blank item (legal) on some redraws, so the loop stops early for those masks
        self.assertLess(len(one), len(TASKS) * G.REDRAWS)
        self.assertGreater(len(one), len(TASKS))
        self.assertEqual(many, one)

    def test_batched_redraws_stop_at_the_first_legal_draw(self):
        self.assertEqual(G.next_redraw({}), 1)
        self.assertEqual(G.next_redraw({1: False, 2: False}), 3)
        self.assertIsNone(G.next_redraw({1: False, 2: True}))
        self.assertIsNone(G.next_redraw(dict.fromkeys(range(1, G.REDRAWS + 1), False)))

    def test_batches_and_length_order(self):
        self.assertEqual(G.batches(list(range(5)), 2), [[0, 1], [2, 3], [4]])
        with self.assertRaises(ValueError):
            G.batches([1], 0)
        tokenizer = FakeTokenizer()
        order = G.length_order(TASKS, PROMPT, tokenizer, MASK_ID)
        lengths = [len(G.canvas({t['id']: t for t in TASKS}[i], PROMPT, tokenizer, MASK_ID)['ids']) for i in order]
        self.assertEqual(lengths, sorted(lengths))

    def test_cpu_batch_needs_no_ladder(self):
        tokenizer = FakeTokenizer()
        longest = G.longest_canvas(TASKS, [PROMPT], tokenizer, MASK_ID, None)
        self.assertEqual(len(longest['ids']), max(len(G.canvas(t, PROMPT, tokenizer, MASK_ID)['ids']) for t in TASKS))
        self.assertEqual(G.choose_batch(None, longest, MASK_ID, MS.SHARED_SAMPLER, 'cpu')['batch'], G.CPU_BATCH)


class ContainerModel:
    """A masked-Qwen-shaped fake for the Modal container code: model(input_ids, logits_to_keep) for fill_one and the
    probes, model.model(input_ids, attention_mask) and lm_head for fill_rows. Padding-blind, so batched and alone
    agree."""
    dtype = torch.float32

    def __init__(self, vocab: int):
        self.vocab = vocab
        self.lm_head = torch.nn.Identity()
        self.model = lambda input_ids, attention_mask: SimpleNamespace(
            last_hidden_state=padding_blind_logits(input_ids, attention_mask.bool(), vocab))

    def __call__(self, input_ids, logits_to_keep=None, **_kwargs):
        logits = padding_blind_logits(input_ids, torch.ones_like(input_ids, dtype=torch.bool), self.vocab)
        return SimpleNamespace(logits=logits if logits_to_keep is None else logits.index_select(1, logits_to_keep))


class ContainerTest(unittest.TestCase):
    """The Modal fill and redraw containers' own code, run on the CPU with the fake model in place of the GPU one."""

    def setUp(self):
        self.tokenizer = FakeTokenizer()
        for task in TASKS:                                   # every token id the texts use exists before sampling
            G.canvas(task, G.PROMPTS['inplace_instructions'], self.tokenizer, MASK_ID)
        candidate = SimpleNamespace(sampler=MS.SHARED_SAMPLER, mask_id=MASK_ID, start_token=False)
        self.model = model = ContainerModel(len(self.tokenizer))
        patcher = unittest.mock.patch.object(G, 'load_candidate',
                                             lambda _key, _device: (candidate, self.tokenizer, model, 'a2d'))
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_fill_container_batches_every_prompt_alike_and_passes_its_probe(self):
        import inplace_prompt_fill_modal as F
        prompts = ['inplace_header', 'inplace_instructions']
        lines = [json.loads(x) for x in F.fill_remote('qwen3', prompts, TASKS, {}, None, gpu='cpu')]
        probe, rows = lines[0]['probe'], lines[1:]
        self.assertTrue(probe['pass'], probe)
        self.assertEqual(probe['repeat_same_tokens'], 1.0)
        self.assertEqual(probe['batch1_same_tokens'], 1.0)
        self.assertEqual(probe['padding']['top1_same'], 1.0)
        self.assertEqual(probe['batching']['batch'], G.CPU_BATCH)
        self.assertEqual(sorted((r['prompt'], r['id']) for r in rows),
                         sorted((p, t['id']) for p in prompts for t in TASKS))
        self.assertEqual({r['batch_size'] for r in rows}, {G.CPU_BATCH})
        order = [[r['id'] for r in rows if r['prompt'] == p] for p in prompts]
        self.assertEqual(order[0], order[1])
        with tempfile.TemporaryDirectory() as d:          # each row is the fill fill_grid gives it one at a time
            forward_for = lambda slots: (lambda x: self.model(x, logits_to_keep=slots).logits)
            for p in prompts:
                out = Path(d) / f'{p}.jsonl'
                G.fill_grid(TASKS, G.PROMPTS[p], self.tokenizer, forward_for, MASK_ID, out, FakeValidator())
                alone = {r['id']: r['slot_ids'] for r in G.read_rows(out)}
                self.assertEqual({r['id']: r['slot_ids'] for r in rows if r['prompt'] == p}, alone)

    def test_fill_container_given_batch_and_resume(self):
        import inplace_prompt_fill_modal as F
        lines = [json.loads(x) for x in F.fill_remote('qwen3', ['inplace_header'], TASKS, {'inplace_header': ['a1']},
                                                      None, batch=2, gpu='cpu')]
        self.assertEqual(lines[0]['probe']['batching'], {'batch': 2, 'ladder': [], 'chosen_by': 'given'})
        self.assertEqual(sorted(r['id'] for r in lines[1:]), ['b2', 'c3'])
        self.assertEqual(F.batching([{'batch_size': 2}] * 2, lines[0]['probe'])['batch'], 2)
        with self.assertRaises(ValueError):
            F.batching([{'batch_size': 2}, {}], lines[0]['probe'])

    def test_redraw_container_draws_every_redraw_with_its_own_seed(self):
        import inplace_prompt_redraw_modal as R
        first = [{'id': 'a1', 'draw': 0, 'legal': True}, {'id': 'b2', 'draw': 0, 'legal': False},
                 {'id': 'c3', 'draw': 0, 'legal': False}]
        lines = [json.loads(x) for x in R.draw_remote('qwen3', TASKS, first, [], None, batch=3, gpu='cpu')]
        probe, rows = lines[0]['probe'], lines[1:]
        self.assertEqual(probe['repeat_same_tokens'], 1.0)
        self.assertEqual(sorted((r['id'], r['draw']) for r in rows),
                         [(i, d) for i in ('b2', 'c3') for d in range(1, R.REDRAWS + 1)])
        for r in rows:
            self.assertEqual(r['seed'], G.redraw_seed(G.SETTING['seed'], r['id'], r['draw']))
            self.assertEqual(r['batch_size'], 3)


class ReportTest(unittest.TestCase):
    def test_pairs_masks_counts_speed_and_noise(self):
        import batched_fills_report as R

        def rows(legal, seconds, batch):
            return [{'id': f'm{n}', 'legal': ok, 'seconds': seconds, 'batch_size': batch} for n, ok in enumerate(legal)]
        before = rows([True, True, False, False, True], 3.0, 1)
        extra = {'id': 'extra', 'legal': True, 'seconds': 1.0, 'batch_size': 16}
        after = [*rows([True, False, True, False, True], 0.5, 16), extra]
        result = R.one_pair(before, after)
        self.assertEqual((result['masks'], result['legal_batch1'], result['legal_batched']), (5, 3, 3))
        self.assertEqual((result['gained'], result['lost']), (1, 1))
        self.assertTrue(result['within_noise'])
        self.assertAlmostEqual(result['batch1']['fills_per_second'], 1 / 3)
        self.assertAlmostEqual(result['batched']['fills_per_second'], 2.0)
        self.assertAlmostEqual(result['batched']['dollars_per_1000'], 1000 / 2.0 / 3600 * R.L4_DOLLARS_PER_HOUR)
        self.assertEqual(result['batched']['batch'], 16)
        lost = R.one_pair(rows([True] * 12, 1.0, 1), rows([False] * 12, 1.0, 8))
        self.assertFalse(lost['within_noise'])
        with self.assertRaises(ValueError):
            R.batch_size([{'batch_size': 1}, {'batch_size': 4}])


if __name__ == '__main__':
    unittest.main()
