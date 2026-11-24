"""CPU tests for the masked LoRA trainer and its minimal Modal launcher (#408).

No pretrained weights, GPU, Modal account, or vault files are used, so this file
runs on CPU in CI (listed in code/scripts/ci_tests.txt). Every test that invokes a
training or launcher entry point mocks real side effects (model/tokenizer loading,
dataset reads, subprocess/Modal execution) to raise so no test can start a real run.
"""
import importlib.util
import json
import random
import re
import subprocess
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
import torch.nn.functional as F

import count_masked_diffusion_text as C
import mask_paste as M
import masked_diffusion_memory_bench as B
import masked_diffusion_text as D
import masked_lora_modal as LM
import masked_lora_trainer as T

HAS_TRANSFORMERS_PEFT = bool(importlib.util.find_spec('transformers') and importlib.util.find_spec('peft'))
VOCAB_PATH = Path(__file__).resolve().parents[1] / 'data/regmb-vocabulary.json'


def _forbid(*_args, **_kwargs):
    raise AssertionError('real side effect is forbidden in unit tests')


class FakeTokenizer:
    """Fast-tokenizer stand-in matching test_masked_diffusion_text.FakeTokenizer, plus stable token IDs."""

    mask_token = '<|mask|>'
    eos_token = '<|endoftext|>'
    pad_token = None
    pattern = r'<\|mask\|>|<\|endoftext\|>|[A-Za-z]+|.'

    def __init__(self, max_vocab: int = 2048) -> None:
        self._max_vocab = max_vocab
        self._stoi: dict[str, int] = {self.mask_token: 0, self.eos_token: 1}
        self._itos: list[str] = [self.mask_token, self.eos_token]

    def __len__(self) -> int:
        return self._max_vocab

    @property
    def mask_token_id(self) -> int:
        return self._stoi[self.mask_token]

    @property
    def eos_token_id(self) -> int:
        return self._stoi[self.eos_token]

    @property
    def pad_token_id(self) -> int | None:
        return None

    def _id(self, piece: str) -> int:
        if piece not in self._stoi:
            self._stoi[piece] = len(self._itos)
            self._itos.append(piece)
        return self._stoi[piece]

    def __call__(self, text: str, *, add_special_tokens: bool = False, return_offsets_mapping: bool = False):
        del add_special_tokens
        matches = list(re.finditer(self.pattern, text, re.S))
        ids = [self._id(m.group(0)) for m in matches]
        out: dict[str, object] = {'input_ids': ids}
        if return_offsets_mapping:
            out['offset_mapping'] = [m.span() for m in matches]
        return out


def sample_team_row(offset: int = 0) -> np.ndarray:
    """One synthetic valid 84-field row in code/data/regmb-vocabulary.json."""
    reg = json.loads(VOCAB_PATH.read_text())
    itos = {k: ['[MASK]', *sorted(set(v))] for k, v in reg['values'].items()}
    row = np.zeros(84, dtype=np.int16)
    for s in range(6):
        row[s * 8 + 0] = 1 + (s + offset) % (len(itos['species']) - 1)
        row[s * 8 + 1] = 1 + (s + offset) % (len(itos['ability']) - 1)
        row[s * 8 + 2] = 1 + (s + offset) % (len(itos['item']) - 1)
        for m in range(4):
            row[s * 8 + 3 + m] = 1 + (s * 4 + m + offset) % (len(itos['move']) - 1)
        row[s * 8 + 7] = 1 + (s + offset) % (len(itos['nature']) - 1)
        spread = [10, 10, 10, 10, 10, 16]
        for j, v in enumerate(spread):
            row[48 + s * 6 + j] = v + 1
    return row


def dllm_ca176752_mdlm_loss(
    logits_fn,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: torch.Tensor | None,
    mask_token_id: int,
    u_t: torch.Tensor,
    u_mask: torch.Tensor,
    *,
    time_epsilon: float = 1e-3,
    right_shift_logits: bool = True,
) -> torch.Tensor:
    """Verbatim reference of ZHZisZZ/dllm@ca176752 MDLMTrainer.compute_loss + LinearAlphaScheduler.

    Source lines pinned at ca176752fbceec49c6b4777a2c18ae88e4eb10ed:
      - dllm/core/schedulers/alpha.py:86-107 (LinearAlphaScheduler: alpha=1-t, alpha'=-1, weight=-alpha'/(1-alpha+1e-6))
      - dllm/core/trainers/mdlm.py:60-83 (_preprocess_inputs, _postprocess_outputs)
      - dllm/core/trainers/mdlm.py:152-216 (compute_loss with loss_weight_type='scheduler', loss_norm_type='token')
    """
    b, seq = input_ids.shape
    maskable_mask = labels != -100
    t = time_epsilon + (1.0 - time_epsilon) * u_t
    alpha_t = 1.0 - t
    p_mask = 1.0 - alpha_t.unsqueeze(1).expand(b, seq)
    masked_mask = (u_mask < p_mask) & maskable_mask
    noised_input_ids = torch.where(masked_mask, mask_token_id, input_ids)
    logits = logits_fn(noised_input_ids, attention_mask)
    if right_shift_logits:
        logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
    weights = (1.0 / (1.0 - alpha_t + 1e-6)).unsqueeze(1).repeat(1, seq)
    assert (input_ids[maskable_mask] == labels[maskable_mask]).all()
    token_nll = F.cross_entropy(logits.transpose(1, 2), input_ids, reduction='none')
    token_nll = token_nll * weights * masked_mask.to(token_nll.dtype)
    token_nll = token_nll / maskable_mask.sum().clamp_min(1)
    return token_nll.sum()


class TinyLoRALM(torch.nn.Module):
    """Minimal pure-PyTorch LM with frozen base weights and PEFT-named LoRA adapters for CI CPU tests."""

    def __init__(self, vocab_size: int = 64, dim: int = 16, rank: int = 4) -> None:
        super().__init__()
        self.config = types.SimpleNamespace(vocab_size=vocab_size)
        self.embed = torch.nn.Embedding(vocab_size, dim)
        self.base_proj = torch.nn.Linear(dim, dim)
        self.lora_A = torch.nn.Linear(dim, rank, bias=False)
        self.lora_B = torch.nn.Linear(rank, dim, bias=False)
        self.lm_head = torch.nn.Linear(dim, vocab_size, bias=False)
        torch.nn.init.normal_(self.lora_A.weight, std=0.05)
        torch.nn.init.normal_(self.lora_B.weight, std=0.05)
        for name, param in self.named_parameters():
            param.requires_grad = 'lora_' in name

    @property
    def dtype(self) -> torch.dtype:
        return self.base_proj.weight.dtype

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None):
        del attention_mask
        h = self.embed(input_ids)
        h = self.base_proj(h) + self.lora_B(self.lora_A(h))
        return types.SimpleNamespace(logits=self.lm_head(h))


class RowAndMaskTest(unittest.TestCase):
    def test_decode_84_field_row_uses_display_names_and_stat_points(self):
        vocab = T.load_regmb_vocab(VOCAB_PATH)
        team = T.decode_team_84(sample_team_row(), vocab)
        self.assertEqual(len(team), 6)
        for i, mon in enumerate(team):
            self.assertEqual(mon['slot'], i + 1)
            self.assertEqual(tuple(mon['statPoints']), M.STATS)
            self.assertEqual(sum(mon['statPoints'].values()), 66)
            self.assertEqual(len(mon['moves']), 4)

    def test_training_row_marks_only_hidden_slots_eligible_including_slot_padding(self):
        vocab = T.load_regmb_vocab(VOCAB_PATH)
        team = T.decode_team_84(sample_team_row(), vocab)
        tok = FakeTokenizer()
        task = T.make_mask_lab_task(team, 'items', 2, random.Random(7))
        row = T.build_training_row(task, 0.55, tok, limit=10_000)

        self.assertEqual(row.labels[0], -100)
        eligible = [i for i, y in enumerate(row.labels) if y != -100]
        expected_eligible = [i for s in row.slots for i in range(s.first_token, s.first_token + s.length)]
        self.assertEqual(eligible, expected_eligible)
        slot_pad_positions = [i for i in eligible if row.input_ids[i] == tok.eos_token_id]
        self.assertTrue(len(slot_pad_positions) > 0)
        for i in slot_pad_positions:
            self.assertEqual(row.labels[i], tok.eos_token_id)
        self.assertEqual(D.read_filled(row.target_text, T.resolve_pad_token(tok), tok.mask_token), team)

    def test_unknown_win_rate_writes_unknown_in_header(self):
        vocab = T.load_regmb_vocab(VOCAB_PATH)
        team = T.decode_team_84(sample_team_row(), vocab)
        tok = FakeTokenizer()
        task = T.make_mask_lab_task(team, 'abilities', 1, random.Random(3))
        row = T.build_training_row(task, None, tok, limit=10_000)
        self.assertIn('Target win rate vs top meta teams: unknown.\n', row.target_text)

    def test_collate_right_pads_with_ignored_labels_and_valid_mask(self):
        r1 = T.EncodedRow([10, 11, 12], [-100, 11, -100], [False, True, False], '', (), 0.5)
        r2 = T.EncodedRow([10, 11, 12, 13, 14], [-100, -100, 12, 13, -100], [False, False, True, True, False], '', (), None)
        batch = T.collate_rows([r1, r2], batch_pad_id=1)
        self.assertEqual(batch.input_ids.tolist(), [[10, 11, 12, 1, 1], [10, 11, 12, 13, 14]])
        self.assertEqual(batch.labels.tolist(), [[-100, 11, -100, -100, -100], [-100, -100, 12, 13, -100]])
        self.assertEqual(batch.valid_mask.tolist(), [[True, True, True, False, False], [True, True, True, True, True]])


class MDLMLossParityTest(unittest.TestCase):
    def test_matches_dllm_ca176752_to_float32_precision(self):
        torch.manual_seed(42)
        vocab, b, seq = 47, 4, 15
        model = TinyLoRALM(vocab_size=vocab, dim=12, rank=4)
        input_ids = torch.randint(2, vocab, (b, seq))
        labels = torch.full((b, seq), -100, dtype=torch.long)
        labels[0, 4:9] = input_ids[0, 4:9]
        labels[1, 2:11] = input_ids[1, 2:11]
        labels[2, 6:10] = input_ids[2, 6:10]
        labels[3, 1:5] = input_ids[3, 1:5]
        valid_mask = torch.ones((b, seq), dtype=torch.bool)
        valid_mask[0, 12:] = False
        attn = T.bidirectional_padding_mask(valid_mask, dtype=torch.float32)

        u_t = torch.tensor([0.12, 0.55, 0.87, 0.03], dtype=torch.float32)
        u_mask = torch.rand((b, seq), generator=torch.Generator().manual_seed(99))

        for shift in (True, False):
            ref = dllm_ca176752_mdlm_loss(
                lambda x, m: model(input_ids=x, attention_mask=m).logits,
                input_ids,
                labels,
                attn,
                mask_token_id=0,
                u_t=u_t,
                u_mask=u_mask,
                time_epsilon=1e-3,
                right_shift_logits=shift,
            )
            ours = T.mdlm_loss(
                lambda x, m: model(input_ids=x, attention_mask=m).logits,
                input_ids,
                labels,
                mask_token_id=0,
                attention_mask=attn,
                right_shift_logits=shift,
                time_epsilon=1e-3,
                u_t=u_t,
                u_mask=u_mask,
            )
            torch.testing.assert_close(ours.loss, ref, rtol=1e-6, atol=1e-6)

    def test_zero_masked_tokens_gives_zero_loss_without_nan(self):
        model = TinyLoRALM(vocab_size=20, dim=8, rank=2)
        input_ids = torch.randint(2, 20, (2, 6))
        labels = torch.full((2, 6), -100, dtype=torch.long)
        labels[:, 2:4] = input_ids[:, 2:4]
        out = T.mdlm_loss(
            lambda x, m: model(input_ids=x, attention_mask=m).logits,
            input_ids,
            labels,
            mask_token_id=0,
            right_shift_logits=True,
            u_t=torch.tensor([0.5, 0.5]),
            u_mask=torch.ones((2, 6)),
        )
        self.assertEqual(float(out.loss.detach()), 0.0)
        self.assertEqual(out.masked_count, 0)


class LoRAWeightInvarianceTest(unittest.TestCase):
    def test_only_lora_weights_change_and_base_weights_keep_sha256(self):
        torch.manual_seed(0)
        model = TinyLoRALM(vocab_size=32, dim=16, rank=4)
        before_base = T.base_weight_hashes(model)
        before_lora = T.lora_weight_hashes(model)
        self.assertTrue(len(before_base) > 0 and len(before_lora) > 0)

        input_ids = torch.randint(2, 32, (2, 10))
        labels = torch.full((2, 10), -100, dtype=torch.long)
        labels[:, 3:8] = input_ids[:, 3:8]
        batch = T.Batch(input_ids, labels, torch.ones((2, 10), dtype=torch.bool), torch.zeros((2, 10), dtype=torch.bool))

        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-2)
        T.train_step(
            model,
            opt,
            batch,
            mask_token_id=0,
            right_shift_logits=True,
            generator=torch.Generator().manual_seed(123),
        )
        self.assertEqual(T.base_weight_hashes(model), before_base)
        after_lora = T.lora_weight_hashes(model)
        for name, digest in before_lora.items():
            self.assertNotEqual(after_lora[name], digest, f'LoRA tensor {name} did not update')

    @unittest.skipUnless(HAS_TRANSFORMERS_PEFT, 'transformers and peft are not in code/requirements.txt')
    def test_real_gpt2_peft_lora_and_bidirectional_padding_mask(self):
        from transformers import GPT2Config, GPT2LMHeadModel
        torch.manual_seed(0)
        cfg = GPT2Config(vocab_size=64, n_positions=32, n_embd=32, n_layer=2, n_head=4)
        cfg._attn_implementation = 'sdpa'
        base = GPT2LMHeadModel(cfg)
        model = T.apply_diffugpt_lora(base)

        trainable = [n for n, p in model.named_parameters() if p.requires_grad]
        self.assertEqual(len(trainable), 2 * 4 * 2)  # 2 layers * 4 projections * (A, B)
        before_base = T.base_weight_hashes(model)
        before_lora = T.lora_weight_hashes(model)

        # Bidirectional attention check: changing a later real token changes position 0's logits,
        # while changing a batch-padded token (valid_mask=False) leaves all real positions' logits identical.
        model.eval()
        x = torch.tensor([[5, 6, 7, 8, 9, 10, 1, 1]])
        valid = torch.tensor([[True, True, True, True, True, True, False, False]])
        mask = T.bidirectional_padding_mask(valid, torch.float32)
        with torch.no_grad():
            base_logits = model(input_ids=x, attention_mask=mask).logits
            x_later_real = x.clone()
            x_later_real[0, 5] = 22
            later_logits = model(input_ids=x_later_real, attention_mask=mask).logits
            self.assertGreater(float((base_logits[0, 0] - later_logits[0, 0]).abs().max()), 0.0)
            x_pad_changed = x.clone()
            x_pad_changed[0, 6:] = 33
            pad_logits = model(input_ids=x_pad_changed, attention_mask=mask).logits
            torch.testing.assert_close(base_logits[0, :6], pad_logits[0, :6], rtol=1e-5, atol=1e-5)

        input_ids = torch.randint(2, 64, (2, 12))
        labels = torch.full((2, 12), -100, dtype=torch.long)
        labels[:, 4:10] = input_ids[:, 4:10]
        batch = T.Batch(input_ids, labels, torch.ones((2, 12), dtype=torch.bool), torch.zeros((2, 12), dtype=torch.bool))
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-2)
        T.train_step(model, opt, batch, mask_token_id=0, right_shift_logits=True,
                     generator=torch.Generator().manual_seed(7))

        self.assertEqual(T.base_weight_hashes(model), before_base)
        after_step1 = T.lora_weight_hashes(model)
        # On step 1, lora_B starts at zero so gradients flow into lora_B first.
        for name, digest in before_lora.items():
            if 'lora_B' in name:
                self.assertNotEqual(after_step1[name], digest, f'{name} did not update on step 1')

        # After step 2, every adapter tensor (both lora_A and lora_B) has updated while base weights remain frozen.
        T.train_step(model, opt, batch, mask_token_id=0, right_shift_logits=True,
                     generator=torch.Generator().manual_seed(8))
        self.assertEqual(T.base_weight_hashes(model), before_base)
        after_step2 = T.lora_weight_hashes(model)
        for name, digest in before_lora.items():
            self.assertNotEqual(after_step2[name], digest, f'PEFT LoRA tensor {name} did not update')


class EntryPointGuardTest(unittest.TestCase):
    """Every entry-point test mocks real side effects to raise so no test can start a real run."""

    def test_main_tracer_dispatch_never_touches_real_model_or_data(self):
        captured: dict[str, object] = {}

        def fake_runner(args):
            captured['data'] = args.data
            captured['rows'] = args.rows
            return {
                'init_loss': {'mdlm_loss': 9.5, 'random_expected_mdlm_loss': 10.8, 'random_matched_draw_loss': 10.7},
                'overfit': {'rows': 8, 'steps': 10, 'initial_fixed_eval': {'mdlm_loss': 9.5},
                            'final_fixed_eval': {'mdlm_loss': 0.2}},
            }

        with (
            mock.patch.object(B, 'load_diffugpt', side_effect=_forbid),
            mock.patch.object(C, 'load_tokenizer', side_effect=_forbid),
            mock.patch.object(T, 'load_training_teams', side_effect=_forbid),
            mock.patch.object(T, 'slice_training_npz', side_effect=_forbid),
        ):
            rc = T.main(['tracer', '--data', '/nonexistent.npz', '--out', '/tmp/unused', '--rows', '16'],
                        tracer_runner=fake_runner, slicer=_forbid)
        self.assertEqual(rc, 0)
        self.assertEqual(captured, {'data': Path('/nonexistent.npz'), 'rows': 16})

    def test_run_tracer_end_to_end_with_mocked_side_effects_and_tiny_model(self):
        vocab = T.load_regmb_vocab(VOCAB_PATH)
        synthetic_teams = [(T.decode_team_84(sample_team_row(i), vocab), 0.5 + 0.01 * i if i else None)
                           for i in range(6)]
        tok = FakeTokenizer(max_vocab=4096)
        torch.manual_seed(0)
        tiny = TinyLoRALM(vocab_size=4096, dim=16, rank=4)
        fake_info = {'key': 'diffugpt_s', 'label': 'Fake DiffuGPT', 'repo': 'fake/repo',
                     'revision': '00000000', 'mask_id': tok.mask_token_id}

        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / 'out'
            args = T.build_parser().parse_args([
                'tracer', '--data', '/forbidden/real.npz', '--out', str(out_dir),
                '--rows', '6', '--overfit-rows', '4', '--overfit-steps', '10',
                '--eval-every', '5', '--train-steps', '3', '--batch-size', '2',
                '--overfit-lr', '2e-2', '--lr', '1e-2', '--device', 'cpu',
            ])
            with (
                mock.patch.object(B, 'load_diffugpt', side_effect=_forbid),
                mock.patch.object(C, 'load_tokenizer', side_effect=_forbid),
                mock.patch.object(T, 'load_training_teams', side_effect=_forbid),
                mock.patch.object(T, 'slice_training_npz', side_effect=_forbid),
            ):
                result = T.run_tracer(
                    args,
                    loader=lambda _dev, _dt: (tiny, tok, fake_info),
                    team_loader=lambda _p, count, _seed, _v: synthetic_teams[:count],
                )
            saved = json.loads((out_dir / 'tracer_results.json').read_text())
            self.assertEqual(saved['rows'], 6)
            self.assertLess(saved['dllm_parity_max_abs_diff'], 1e-6)
            self.assertTrue(saved['base_weights_unchanged_after_overfit'])
            self.assertTrue(saved['base_weights_unchanged_after_smoke'])
            self.assertGreater(saved['overfit_lora_tensors_changed'], 0)
            self.assertEqual(result['score_slot_pad'], True)
            self.assertEqual(result['score_batch_pad'], False)

    def test_slice_data_keeps_only_training_rows_and_forbids_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / 'full.npz'
            dst = Path(tmp) / 'slice.npz'
            all_x = np.stack([sample_team_row(i) for i in range(10)])
            all_w = np.array([0.1 * i for i in range(9)] + [np.nan], dtype=np.float32)
            all_val = np.array([True, False, True, False, False, False, True, False, False, False])
            np.savez(src, all_X=all_x, all_W=all_w, all_val=all_val)
            with (
                mock.patch.object(B, 'load_diffugpt', side_effect=_forbid),
                mock.patch.object(T, 'run_tracer', side_effect=_forbid),
            ):
                rc = T.main(['slice-data', '--data', str(src), '--out', str(dst), '--count', '5', '--seed', '1'],
                            tracer_runner=_forbid)
            self.assertEqual(rc, 0)
            with np.load(dst) as sliced:
                self.assertEqual(sliced['all_X'].shape, (5, 84))
                self.assertFalse(bool(sliced['all_val'].any()))
                # None of the validation rows (0, 2, 6) may appear in the slice
                val_rows = {all_x[i].tobytes() for i in (0, 2, 6)}
                for row in sliced['all_X']:
                    self.assertNotIn(row.tobytes(), val_rows)


class ModalLauncherTest(unittest.TestCase):
    def test_packaged_files_exist_and_exclude_block_diffusion_and_masklab(self):
        for rel in LM.PACKAGED_FILES:
            self.assertTrue((LM.CODE_DIR / rel).is_file(), f'missing packaged file: {rel}')
        joined = ' '.join(LM.PACKAGED_FILES)
        for forbidden in ('textdiff.py', 'textdiff_port.py', 'mask_lab_textdiff.py'):
            self.assertNotIn(forbidden, joined)

    def test_build_remote_command_preserves_trainer_cli(self):
        cmd = LM.build_remote_command('tracer --data /tmp/rows.npz --out /tmp/out --rows 256')
        self.assertEqual(cmd, ['python', LM.REMOTE_TRAINER, 'tracer', '--data', '/tmp/rows.npz',
                               '--out', '/tmp/out', '--rows', '256'])
        with self.assertRaises(ValueError):
            LM.build_remote_command('   ')

    def test_run_trainer_subprocess_stages_data_collects_outputs_and_raises_on_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dest = Path(tmp) / 'rows.npz'
            out_dir = Path(tmp) / 'out'
            seen_cmd: list[str] = []

            def fake_run(cmd, *, cwd, check):
                seen_cmd.extend(cmd)
                self.assertEqual(cwd, f'{LM.REMOTE_CODE}/scripts')
                self.assertFalse(check)
                (out_dir / 'tracer_results.json').write_text('{"ok": true}')
                return types.SimpleNamespace(returncode=0)

            with mock.patch.object(subprocess, 'run', side_effect=_forbid):
                res = LM.run_trainer_subprocess(
                    'tracer --data /tmp/rows.npz --out /tmp/out',
                    data_bytes=b'npz-bytes',
                    data_dest=data_dest,
                    out_dir=out_dir,
                    runner=fake_run,
                )
            self.assertEqual(data_dest.read_bytes(), b'npz-bytes')
            self.assertEqual(seen_cmd[:2], ['python', LM.REMOTE_TRAINER])
            self.assertEqual(res['files'], {'tracer_results.json': b'{"ok": true}'})

            local_out = Path(tmp) / 'local_out'
            written = LM.write_output_files(local_out, res['files'])
            self.assertEqual([p.name for p in written], ['tracer_results.json'])
            self.assertEqual((local_out / 'tracer_results.json').read_bytes(), b'{"ok": true}')

            with self.assertRaises(ValueError):
                LM.write_output_files(local_out, {'../escape.txt': b'bad'})

            with mock.patch.object(subprocess, 'run', side_effect=_forbid), self.assertRaises(RuntimeError):
                LM.run_trainer_subprocess(
                    'tracer --data /tmp/rows.npz --out /tmp/out',
                    data_dest=data_dest,
                    out_dir=out_dir,
                    runner=lambda *_a, **_kw: types.SimpleNamespace(returncode=2),
                )


if __name__ == '__main__':
    unittest.main()
