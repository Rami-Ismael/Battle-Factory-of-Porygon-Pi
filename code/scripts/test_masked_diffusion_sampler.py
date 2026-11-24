"""Per-model sampler settings (#392) on the CPU, with a fake model and a fixed seed.

Checks every acceptance criterion of #392:
  1. Every entry in CANDIDATES names its sampler setting, and the default reproduces today's shared sampler.
  2. Every existing fill path gives token-identical output under the default compared with an independent copy of
     today's shared sampler loop, with a fake model and a fixed seed (exercising make_forward, right_shift, and
     start_token).
  3. Every result file a fill writes records the sampler setting it used.

No checkpoint, GPU or vault file is read, so this test is in ci_tests.txt.
"""
import json
import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import torch

import count_masked_diffusion_text as C
import inplace_prompt_grid as G
import masked_diffusion_memory_bench as B
import masked_diffusion_sampler as MS
import sampler_checks as S
from test_inplace_prompt_grid import PROMPT, TASKS, FakeTokenizer, FakeValidator

MASK_ID, VOCAB = 0, 17


def reference_schedule(masks: int, steps: int) -> list[int]:
    """Independent frozen copy of today's shared linear schedule (pre-#392 masked_diffusion_memory_bench.schedule)."""
    counts, remaining = [], masks
    for j in range(steps):
        t = torch.tensor((steps - j) / steps, dtype=torch.float32)
        s = torch.tensor((steps - j - 1) / steps, dtype=torch.float32)
        kept = ((1 - (1 - s)) / (1 - (1 - t))).item()
        n = min(round(remaining * (1 - kept)), remaining)
        remaining -= n
        if n:
            counts.append(n)
        if not remaining:
            break
    return counts


@torch.no_grad()
def reference_shared_fill(forward, canvas: torch.Tensor, slots: torch.Tensor, mask_id: int, steps: int,
                          temperature: float, generator: torch.Generator | None = None) -> torch.Tensor:
    """Independent frozen copy of today's shared sampler loop (pre-#392 masked_diffusion_memory_bench.fill)."""
    x = canvas.clone()
    for k in reference_schedule(len(slots), steps):
        logits = forward(x).float()
        if temperature == 0:
            x0 = logits.argmax(-1)
        else:
            u = torch.rand(logits.shape, generator=generator, device=logits.device, dtype=torch.float32)
            u.clamp_(min=torch.finfo(torch.float32).tiny).log_().neg_().log_().mul_(-temperature).add_(logits)
            x0 = u.argmax(-1)
        p = torch.softmax(logits, -1).gather(-1, x0.unsqueeze(-1)).squeeze(-1)
        if x0.shape[1] != len(slots):
            x0, p = x0[:, slots], p[:, slots]
        current = x[:, slots]
        p = torch.where(current == mask_id, p, -math.inf)
        chosen = p.topk(k, dim=1).indices
        x[:, slots] = current.scatter(1, chosen, x0.gather(1, chosen))
    return x


class ContextModel(torch.nn.Module):
    """Fake model whose logits depend on every token in x, so any change in fill order, shift or schedule changes
    later tokens."""

    def __init__(self, vocab: int = VOCAB, shifted: bool = False):
        super().__init__()
        self.vocab = vocab
        self.shifted = shifted

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, length = x.shape
        pos = torch.arange(length, device=x.device, dtype=torch.float32)
        tok = torch.arange(self.vocab, device=x.device, dtype=torch.float32)
        row_mix = (x.float() * (pos[None, :] + 1.0)).sum(dim=1, keepdim=True)
        signal = torch.sin(pos[None, :, None] * 0.7 + tok[None, None, :] * 0.5 + row_mix[:, :, None] * 0.03)
        logits = signal * 2.0
        logits[..., MASK_ID] = -1e9
        if self.shifted:
            logits = torch.cat([logits[:, 1:], logits[:, -1:]], dim=1)
        return logits


class FakeFamilyModel(torch.nn.Module):
    """Wraps ContextModel with the attributes B.make_forward expects for a given family (a2d, gpt2, llama, dream, llada)."""

    dtype = torch.float32

    def __init__(self, family: str, vocab: int = VOCAB, shifted: bool = False):
        super().__init__()
        self.family = family
        self.core = ContextModel(vocab=vocab, shifted=shifted)
        self.lm_head = torch.nn.Identity()
        if family == 'gpt2':
            self.transformer = lambda input_ids, attention_mask=None: SimpleNamespace(last_hidden_state=self.core(input_ids))
        elif family == 'llama':
            self.model = lambda input_ids, attention_mask=None: SimpleNamespace(last_hidden_state=self.core(input_ids))
        elif family == 'dream':
            self.model = lambda input_ids, attention_mask=None: (self.core(input_ids),)
        elif family == 'llada':
            ln_f = torch.nn.Identity()
            self.model = SimpleNamespace(transformer=SimpleNamespace(ln_f=ln_f))

    def forward(self, input_ids: torch.Tensor, *_args, logits_to_keep: torch.Tensor | None = None,
                **_kwargs) -> SimpleNamespace:
        logits = self.core(input_ids)
        if self.family == 'llada':
            logits = self.model.transformer.ln_f(logits)
        elif logits_to_keep is not None:
            logits = logits.index_select(1, logits_to_keep)
        return SimpleNamespace(logits=logits)


class SamplerSettingTest(unittest.TestCase):
    def test_default_setting_matches_todays_shared_sampler(self):
        s = MS.SHARED_SAMPLER
        self.assertEqual(s.name, 'shared')
        self.assertEqual(s.fill_order, 'confidence')
        self.assertEqual(s.masks_left, 'exact_linear')
        self.assertEqual(s.steps, 128)
        self.assertEqual(s.temperature, 1.0)
        self.assertIsNone(s.top_p)
        self.assertIsNone(s.block_length)
        self.assertFalse(s.right_shift)
        self.assertFalse(s.start_token)
        self.assertIs(MS.SAMPLERS['shared'], MS.SHARED_SAMPLER)

    def test_setting_represents_every_value_needed_by_later_tickets(self):
        diffugpt_eval = MS.SamplerSetting(
            name='diffugpt_eval', fill_order='confidence', masks_left='at_most_linear',
            steps='per_mask', temperature=1.0, top_p=None, block_length=None, right_shift=True, start_token=False,
        )
        dllm_eval = MS.SamplerSetting(
            name='dllm_eval', fill_order='confidence', masks_left='exact_linear',
            steps='per_mask', temperature=0, top_p=None, block_length=256, right_shift=False, start_token=False,
        )
        self.assertEqual(diffugpt_eval.step_count(47), 47)
        self.assertEqual(dllm_eval.step_count(47), 47)
        self.assertEqual(dllm_eval.temperature, 0.0)
        for setting in (MS.SHARED_SAMPLER, diffugpt_eval, dllm_eval):
            encoded = json.loads(json.dumps(setting.to_dict()))
            self.assertEqual(MS.SamplerSetting.from_dict(encoded), setting)

    def test_invalid_settings_are_rejected(self):
        for kwargs in (
            {'name': ''},
            {'fill_order': 'random'},
            {'masks_left': 'linear'},
            {'steps': 0},
            {'steps': True},
            {'steps': 'half'},
            {'temperature': -0.1},
            {'temperature': float('nan')},
            {'top_p': 0.9},
            {'block_length': 0},
            {'block_length': True},
        ):
            with self.assertRaises(ValueError, msg=str(kwargs)):
                MS.SamplerSetting(**kwargs)

    def test_unimplemented_future_modes_refuse_rather_than_silently_running_shared(self):
        x = torch.tensor([[1, MASK_ID, MASK_ID, 2]])
        slots = torch.tensor([1, 2])
        model = ContextModel()
        for kwargs in ({'masks_left': 'at_most_linear'}, {'block_length': 256}):
            setting = replace(MS.SHARED_SAMPLER, **kwargs)
            with self.assertRaises(NotImplementedError, msg=str(kwargs)):
                B.fill(model, x, slots, MASK_ID, setting=setting)


class CandidatesSamplerTest(unittest.TestCase):
    def test_every_candidate_names_the_shared_sampler_with_its_shift_and_start_token(self):
        expected_shifted = {'diffugpt_s', 'diffugpt_m', 'diffullama', 'dream'}
        expected_start = {'diffullama'}
        for c in C.CANDIDATES:
            s = c.sampler
            self.assertIsInstance(s, MS.SamplerSetting)
            self.assertEqual(s.name, 'shared')
            self.assertEqual(s.fill_order, 'confidence')
            self.assertEqual(s.masks_left, 'exact_linear')
            self.assertEqual(s.steps, 128)
            self.assertEqual(s.temperature, 1.0)
            self.assertIsNone(s.top_p)
            self.assertIsNone(s.block_length)
            self.assertEqual(s.right_shift, c.key in expected_shifted, c.key)
            self.assertEqual(s.start_token, c.key in expected_start, c.key)
            self.assertEqual(c.right_shift, s.right_shift)
            self.assertEqual(c.start_token, s.start_token)

    def test_candidate_info_and_canvas_sampler_round_trip(self):
        for c in C.CANDIDATES:
            info = B.candidate_info(c)
            self.assertEqual(info['sampler_setting'], c.sampler.to_dict())
            canvas = {'candidate': info, 'family': B.FAMILY[c.key]}
            self.assertEqual(B.canvas_sampler(canvas), c.sampler)

    def test_legacy_canvas_without_sampler_field_reconstructs_shared_setting(self):
        for c in C.CANDIDATES:
            legacy_info = {k: v for k, v in B.candidate_info(c).items() if k != 'sampler_setting'}
            canvas = {'candidate': legacy_info, 'family': B.FAMILY[c.key]}
            self.assertEqual(B.canvas_sampler(canvas), c.sampler)


class TokenIdenticalDefaultTest(unittest.TestCase):
    def test_fill_through_make_forward_under_each_candidates_default_matches_reference(self):
        slots = torch.tensor([2, 3, 4, 7, 8, 9, 13, 14, 15, 16])
        x = torch.arange(20).unsqueeze(0).repeat(2, 1) % 6 + 1
        x[:, slots] = MASK_ID
        for c in C.CANDIDATES:
            family = B.FAMILY[c.key]
            for scope in ('all', 'slots'):
                model = FakeFamilyModel(family, shifted=c.sampler.right_shift)
                forward = B.make_forward(model, family, scope, slots, right_shift=c.sampler.right_shift)
                ref_core = ContextModel(shifted=c.sampler.right_shift)

                def ref_forward(inp, m=ref_core, sh=c.sampler.right_shift, sc=scope):
                    logits = B.shift_right(m(inp)) if sh else m(inp)
                    return logits if sc == 'all' else logits[:, slots]

                ref = reference_shared_fill(ref_forward, x, slots, MASK_ID, 128, 1.0,
                                            torch.Generator().manual_seed(2026))
                got = B.fill(forward, x, slots, MASK_ID, generator=torch.Generator().manual_seed(2026),
                             setting=c.sampler)
                self.assertTrue(torch.equal(got, ref), (c.key, scope))

    def test_fill_greedy_and_per_mask_steps_match_reference(self):
        slots = torch.tensor([1, 2, 5, 6, 7])
        x = torch.tensor([[3, MASK_ID, MASK_ID, 4, 5, MASK_ID, MASK_ID, MASK_ID, 6]])
        model = ContextModel(shifted=False)
        ref = reference_shared_fill(model, x, slots, MASK_ID, len(slots), 0.0)
        got = B.fill(model, x, slots, MASK_ID, setting=replace(MS.SHARED_SAMPLER, steps='per_mask', temperature=0.0))
        self.assertTrue(torch.equal(got, ref))

    def test_inplace_prompt_grid_fill_is_token_identical_under_default_sampler(self):
        tokenizer = FakeTokenizer()
        vocab = len(tokenizer) + 25
        start_token_id = len(tokenizer) + 10

        for key in ('qwen3', 'diffugpt_s', 'diffullama'):
            candidate = C.candidate_by_key(key)
            family = B.FAMILY[key]
            model = FakeFamilyModel(family, vocab=vocab, shifted=candidate.sampler.right_shift)
            forward_for = lambda slots, m=model, f=family, s=candidate.sampler: B.make_forward(
                m, f, 'slots', slots, right_shift=s.right_shift
            )
            task = TASKS[0]
            seed = G.mask_seed(G.SETTING['seed'], task['id'])
            start_id = start_token_id if candidate.sampler.start_token else None
            c = G.canvas(task, PROMPT, tokenizer, 0, start_id=start_id)
            x, slots = torch.tensor([c['ids']]), torch.tensor(c['slots'])
            expected = reference_shared_fill(forward_for(slots), x, slots, 0, 128, 1.0,
                                             torch.Generator().manual_seed(seed))[0].tolist()
            row_default = G.fill_one(task, PROMPT, tokenizer, forward_for, 0, FakeValidator(), G.vocabulary(), seed,
                                     start_id=start_id)
            row_explicit = G.fill_one(task, PROMPT, tokenizer, forward_for, 0, FakeValidator(), G.vocabulary(), seed,
                                      start_id=start_id, sampler=candidate.sampler)
            expected_slot_ids = [[expected[n] for n in span] for span in c['spans']]
            self.assertEqual(row_default['slot_ids'], expected_slot_ids, key)
            self.assertEqual(row_explicit['slot_ids'], expected_slot_ids, key)
            self.assertEqual(row_explicit['values'], row_default['values'], key)

    def test_copy_fill_case_matches_reference(self):
        ids = [5, 6, 7, 8, 1, 1, 0, 0, 0, 0, 2]
        case = {'ids': ids, 'positions': [6, 7, 8, 9], 'answer': [5, 6, 7, 8]}
        for shift in (True, False):
            model = ContextModel(vocab=12, shifted=shift)
            forward = (lambda t, m=model: B.shift_right(m(t))) if shift else model
            expected = reference_shared_fill(forward, torch.tensor([ids]), torch.tensor(case['positions']), 0,
                                             len(case['positions']), 0.0)[0].tolist()
            got = S.fill_case(model, case, 0, shift)
            self.assertEqual(got, expected, shift)


class RecordedSamplerSettingTest(unittest.TestCase):
    def test_memory_bench_run_header_records_sampler_setting(self):
        candidate = C.candidate_by_key('diffullama')
        canvas = {'candidate': B.candidate_info(candidate), 'family': B.FAMILY[candidate.key], 'version': 'v1',
                  'mask': {'grid': 'meta', 'id': 'm1', 'source': 'MB1', 'task': 'pokemon', 'k': 6},
                  'ask': 'unknown', 'tokens': 20, 'slots': [2, 3, 4, 5]}
        active = B.run_sampler(canvas)
        header = B.run_header(canvas, batch=1, scope='slots', setting=active, seed=2026, precision='native', ceiling=1)
        self.assertEqual(header['sampler_setting'], candidate.sampler.to_dict())
        self.assertEqual(header['steps'], 128)
        self.assertEqual(header['temperature'], 1.0)

        overridden = B.run_sampler(canvas, steps=64, temperature=0.0)
        header_over = B.run_header(canvas, batch=1, scope='slots', setting=overridden, seed=2026,
                                   precision='native', ceiling=1)
        self.assertEqual(header_over['sampler_setting']['steps'], 64)
        self.assertEqual(header_over['sampler_setting']['temperature'], 0.0)

    def test_inplace_prompt_grid_fills_redraws_and_run_meta_record_sampler_setting(self):
        tokenizer = FakeTokenizer()
        candidate = C.candidate_by_key('diffugpt_s')
        model = FakeFamilyModel('gpt2', vocab=len(tokenizer) + 25, shifted=True)
        forward_for = lambda slots: B.make_forward(model, 'gpt2', 'slots', slots, right_shift=True)
        with tempfile.TemporaryDirectory() as d:
            fills_path = Path(d) / 'fills.jsonl'
            redraws_path = Path(d) / 'redraws.jsonl'
            G.fill_grid(TASKS[:1], PROMPT, tokenizer, forward_for, 0, fills_path, FakeValidator(),
                        sampler=candidate.sampler)
            G.redraw_grid(TASKS[:1], fills_path, PROMPT, tokenizer, forward_for, 0, redraws_path, FakeValidator(),
                          sampler=candidate.sampler, redraws=1)
            fill_row = G.read_rows(fills_path)[0]
            redraw_row = G.read_rows(redraws_path)[0]
        meta = G.run_meta(PROMPT, candidate, TASKS[:1], tasks_sha256='0' * 64)
        self.assertEqual(meta['sampler_setting'], candidate.sampler.to_dict())
        self.assertEqual(fill_row['sampler_setting'], candidate.sampler.to_dict())
        self.assertEqual(redraw_row['sampler_setting'], candidate.sampler.to_dict())

    def test_sampler_checks_results_record_sampler_setting(self):
        candidate = C.candidate_by_key('qwen3')
        canvas = {'candidate': B.candidate_info(candidate), 'family': B.FAMILY[candidate.key], 'slots': [1, 2]}
        setting = replace(candidate.sampler, steps=16, temperature=0.0)
        result = S._fill_result('ours, logits scored at all', canvas, 16, [[1, 2, 3]], lambda ids: str(ids), 0.1,
                                sampler_setting=setting, scope='all')
        self.assertEqual(result['sampler_setting'], setting.to_dict())

        llama = C.candidate_by_key('diffullama')
        copy_res = S.copy_header(B.candidate_info(llama), B.FAMILY[llama.key], 'cpu', 1, 'sdpa',
                                 llama.sampler.right_shift)
        self.assertEqual(copy_res['sampler_setting']['shift'], S.copy_setting(True, True).to_dict())
        self.assertEqual(copy_res['sampler_setting']['no_shift'], S.copy_setting(False, True).to_dict())
        self.assertEqual(MS.SamplerSetting.from_dict(copy_res['sampler_setting']['shift']), S.copy_setting(True, True))
        self.assertTrue(copy_res['trained_shifted'])

        dream = C.candidate_by_key('dream')
        dream_canvas = {'candidate': B.candidate_info(dream), 'family': B.FAMILY[dream.key]}
        dream_setting = S.dream_own_setting(dream_canvas, 'maskgit_plus', 64, 1.0)
        self.assertEqual(dream_setting, {'name': 'dream_own:maskgit_plus', 'alg': 'maskgit_plus', 'steps': 64,
                                         'temperature': 1.0, 'right_shift': True, 'start_token': False})


if __name__ == '__main__':
    unittest.main()
