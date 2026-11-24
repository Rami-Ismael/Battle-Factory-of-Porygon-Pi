"""CPU tests for the masked Low-Rank Adaptation (LoRA) trainer and its minimal Modal launcher (#408).

No pretrained weights, GPU, Modal account, or vault files are used, so this file
runs on CPU in CI (listed in code/scripts/ci_tests.txt). Every test that invokes a
training or launcher entry point mocks real side effects (model/tokenizer loading,
dataset reads, subprocess/Modal execution) to raise so no test can start a real run.
"""
import importlib.util
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
import torch.nn.functional as F

import mask_paste
import masked_diffusion_memory_bench as memory_bench
import masked_diffusion_text as diffusion_text
import masked_lora_modal as launcher
import masked_lora_trainer as trainer

HAS_TRANSFORMERS_PEFT = bool(importlib.util.find_spec('transformers') and importlib.util.find_spec('peft'))
DLLM_CHECKOUT = Path(os.environ.get('DLLM_CA176752_DIR', '/tmp/dllm-ca176752'))
HAS_UPSTREAM_DLLM = HAS_TRANSFORMERS_PEFT and (DLLM_CHECKOUT / 'dllm/core/trainers/mdlm.py').is_file()
VOCAB_PATH = Path(__file__).resolve().parents[1] / 'data/regmb-vocabulary.json'


def _forbid(*_args, **_kwargs):
    raise AssertionError('real side effect is forbidden in unit tests')


class FakeTokenizer:
    """Fast-tokenizer stand-in matching test_masked_diffusion_text.FakeTokenizer, plus stable token IDs."""

    mask_token = '<|mask|>'
    eos_token = '<|endoftext|>'
    pad_token = None
    pattern = r'<\|mask\|>|<\|endoftext\|>|[A-Za-z]+|.'

    def __init__(self, max_vocab: int = 4096) -> None:
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
    itos = trainer.load_regmb_vocab(VOCAB_PATH).itos
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
        vocab = trainer.load_regmb_vocab(VOCAB_PATH)
        team = trainer.decode_team_84(sample_team_row(), vocab)
        self.assertEqual(len(team), 6)
        for i, mon in enumerate(team):
            self.assertEqual(mon['slot'], i + 1)
            self.assertEqual(tuple(mon['statPoints']), mask_paste.STATS)
            self.assertEqual(sum(mon['statPoints'].values()), 66)
            self.assertEqual(len(mon['moves']), 4)

    def test_training_row_marks_only_hidden_slots_eligible_including_slot_padding(self):
        vocab = trainer.load_regmb_vocab(VOCAB_PATH)
        team = trainer.decode_team_84(sample_team_row(), vocab)
        tok = FakeTokenizer()
        task = trainer.make_mask_lab_task(team, 'items', 2, random.Random(7))
        row = trainer.build_training_row(task, 0.55, tok, limit=10_000)

        self.assertEqual(row.labels[0], -100)
        eligible = [i for i, y in enumerate(row.labels) if y != -100]
        expected_eligible = [i for s in row.slots for i in range(s.first_token, s.first_token + s.length)]
        self.assertEqual(eligible, expected_eligible)
        slot_pad_positions = [i for i in eligible if row.input_ids[i] == tok.eos_token_id]
        self.assertTrue(len(slot_pad_positions) > 0)
        for i in slot_pad_positions:
            self.assertEqual(row.labels[i], tok.eos_token_id)
        self.assertEqual(diffusion_text.read_filled(row.target_text, trainer.resolve_pad_token(tok), tok.mask_token), team)

    def test_unknown_win_rate_writes_unknown_in_header(self):
        vocab = trainer.load_regmb_vocab(VOCAB_PATH)
        team = trainer.decode_team_84(sample_team_row(), vocab)
        tok = FakeTokenizer()
        task = trainer.make_mask_lab_task(team, 'abilities', 1, random.Random(3))
        row = trainer.build_training_row(task, None, tok, limit=10_000)
        self.assertIn('Target win rate vs top meta teams: unknown.\n', row.target_text)

    def test_collate_right_pads_with_ignored_labels_and_valid_mask(self):
        r1 = trainer.EncodedRow([10, 11, 12], [-100, 11, -100], [False, True, False], '', (), 0.5)
        r2 = trainer.EncodedRow([10, 11, 12, 13, 14], [-100, -100, 12, 13, -100], [False, False, True, True, False], '', (), None)
        batch = trainer.collate_rows([r1, r2], batch_pad_id=1)
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
        attn = trainer.bidirectional_padding_mask(valid_mask, dtype=torch.float32)

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
            ours = trainer.mdlm_loss(
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

    @unittest.skipUnless(HAS_UPSTREAM_DLLM, 'upstream ZHZisZZ/dllm@ca176752 checkout not present locally')
    def test_matches_upstream_dllm_mdlm_trainer_compute_loss_directly(self):
        """Execute unmodified ZHZisZZ/dllm@ca176752 MDLMTrainer.compute_loss on the exact same batch and draws."""
        import transformers
        from transformers import GPT2Config, GPT2LMHeadModel

        # Load dllm's LinearAlphaScheduler and MDLMTrainer directly without triggering dllm.core.eval (lm_eval).
        alpha_spec = importlib.util.spec_from_file_location('dllm_alpha', DLLM_CHECKOUT / 'dllm/core/schedulers/alpha.py')
        alpha_mod = importlib.util.module_from_spec(alpha_spec)
        with mock.patch.dict(sys.modules, {'dllm_alpha': alpha_mod}):
            alpha_spec.loader.exec_module(alpha_mod)

        # Stub only the package namespace entries mdlm.py imports from its own tree.
        fake_pkg = types.ModuleType('dllm')
        fake_core = types.ModuleType('dllm.core')
        fake_sched = types.ModuleType('dllm.core.schedulers')
        fake_sched.BaseAlphaScheduler = alpha_mod.BaseAlphaScheduler
        fake_sched.LinearAlphaScheduler = alpha_mod.LinearAlphaScheduler
        fake_utils = types.ModuleType('dllm.utils')
        fake_configs = types.ModuleType('dllm.utils.configs')
        fake_configs.TrainingArguments = transformers.TrainingArguments
        fake_data = types.ModuleType('dllm.utils.data')
        fake_data.prepend_bos = _forbid
        fake_trainers = types.ModuleType('dllm.core.trainers')
        fake_tutils = types.ModuleType('dllm.core.trainers.utils')
        fake_tutils.NLLMetric = lambda: None
        fake_tutils.PPLMetric = lambda: None
        fake_tutils.OnEvaluateMetricsCallback = lambda **_kw: types.SimpleNamespace(update=lambda **_u: None)

        mdlm_spec = importlib.util.spec_from_file_location(
            'dllm.core.trainers.mdlm', DLLM_CHECKOUT / 'dllm/core/trainers/mdlm.py'
        )
        mdlm_mod = importlib.util.module_from_spec(mdlm_spec)
        stubs = {
            'dllm': fake_pkg, 'dllm.core': fake_core, 'dllm.core.schedulers': fake_sched,
            'dllm.utils': fake_utils, 'dllm.utils.configs': fake_configs, 'dllm.utils.data': fake_data,
            'dllm.core.trainers': fake_trainers, 'dllm.core.trainers.utils': fake_tutils,
            'dllm.core.trainers.mdlm': mdlm_mod,
        }
        with mock.patch.dict(sys.modules, stubs):
            mdlm_spec.loader.exec_module(mdlm_mod)

        torch.manual_seed(7)
        cfg = GPT2Config(vocab_size=64, n_positions=32, n_embd=32, n_layer=2, n_head=4)
        cfg._attn_implementation = 'sdpa'
        model = GPT2LMHeadModel(cfg).eval()

        b, seq = 4, 14
        input_ids = torch.randint(2, 64, (b, seq))
        labels = torch.full((b, seq), -100, dtype=torch.long)
        labels[0, 3:9] = input_ids[0, 3:9]
        labels[1, 2:10] = input_ids[1, 2:10]
        labels[2, 5:11] = input_ids[2, 5:11]
        labels[3, 1:6] = input_ids[3, 1:6]
        valid = torch.ones((b, seq), dtype=torch.bool)
        valid[0, 11:] = False
        attn = trainer.bidirectional_padding_mask(valid, torch.float32)

        # Instantiate a lightweight MDLMTrainer object carrying the exact attributes compute_loss reads.
        upstream = object.__new__(mdlm_mod.MDLMTrainer)
        upstream.processing_class = types.SimpleNamespace(padding_side='right', mask_token_id=0, bos_token_id=1)
        upstream.scheduler = alpha_mod.LinearAlphaScheduler()
        upstream.time_epsilon = 1e-3
        upstream.loss_weight_type = 'scheduler'
        upstream.loss_norm_type = 'token'
        upstream.right_shift_logits = True
        upstream.meter = types.SimpleNamespace(update=lambda **_kw: None)

        # Run upstream MDLMTrainer.compute_loss with a fixed global torch seed and feed the same draws to ours.
        torch.manual_seed(2026)
        u_t = torch.rand(b)
        u_mask = torch.rand((b, seq))

        torch.manual_seed(2026)
        upstream_loss = upstream.compute_loss(
            model, {'input_ids': input_ids, 'labels': labels, 'attention_mask': attn}
        )
        ours = trainer.mdlm_loss(
            trainer.model_logits_fn(model),
            input_ids,
            labels,
            mask_token_id=0,
            attention_mask=attn,
            right_shift_logits=True,
            time_epsilon=1e-3,
            u_t=u_t,
            u_mask=u_mask,
        ).loss
        torch.testing.assert_close(ours, upstream_loss, rtol=1e-6, atol=1e-6)

    def test_zero_masked_tokens_gives_zero_loss_without_nan(self):
        model = TinyLoRALM(vocab_size=20, dim=8, rank=2)
        input_ids = torch.randint(2, 20, (2, 6))
        labels = torch.full((2, 6), -100, dtype=torch.long)
        labels[:, 2:4] = input_ids[:, 2:4]
        out = trainer.mdlm_loss(
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
        before_base = trainer.base_weight_hashes(model)
        before_lora = trainer.lora_weight_hashes(model)
        self.assertTrue(len(before_base) > 0 and len(before_lora) > 0)

        input_ids = torch.randint(2, 32, (2, 10))
        labels = torch.full((2, 10), -100, dtype=torch.long)
        labels[:, 3:8] = input_ids[:, 3:8]
        batch = trainer.Batch(
            input_ids, labels, torch.ones((2, 10), dtype=torch.bool), torch.zeros((2, 10), dtype=torch.bool)
        )

        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-2)
        trainer.train_step(
            model,
            opt,
            batch,
            mask_token_id=0,
            right_shift_logits=True,
            generator=torch.Generator().manual_seed(123),
        )
        self.assertEqual(trainer.base_weight_hashes(model), before_base)
        after_lora = trainer.lora_weight_hashes(model)
        for name, digest in before_lora.items():
            self.assertNotEqual(after_lora[name], digest, f'LoRA tensor {name} did not update')

    @unittest.skipUnless(HAS_TRANSFORMERS_PEFT, 'transformers and peft are not in code/requirements.txt')
    def test_real_gpt2_peft_lora_and_bidirectional_padding_mask(self):
        from transformers import GPT2Config, GPT2LMHeadModel
        torch.manual_seed(0)
        cfg = GPT2Config(vocab_size=64, n_positions=32, n_embd=32, n_layer=2, n_head=4)
        cfg._attn_implementation = 'sdpa'
        base = GPT2LMHeadModel(cfg)
        model = trainer.apply_diffugpt_lora(base)

        trainable = [n for n, p in model.named_parameters() if p.requires_grad]
        self.assertEqual(len(trainable), 2 * 4 * 2)  # 2 layers * 4 projections * (A, B)
        before_base = trainer.base_weight_hashes(model)
        before_lora = trainer.lora_weight_hashes(model)

        model.eval()
        x = torch.tensor([[5, 6, 7, 8, 9, 10, 1, 1]])
        valid = torch.tensor([[True, True, True, True, True, True, False, False]])
        mask = trainer.bidirectional_padding_mask(valid, torch.float32)
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
        batch = trainer.Batch(
            input_ids, labels, torch.ones((2, 12), dtype=torch.bool), torch.zeros((2, 12), dtype=torch.bool)
        )
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-2)
        trainer.train_step(model, opt, batch, mask_token_id=0, right_shift_logits=True,
                           generator=torch.Generator().manual_seed(7))

        self.assertEqual(trainer.base_weight_hashes(model), before_base)
        after_step1 = trainer.lora_weight_hashes(model)
        for name, digest in before_lora.items():
            if 'lora_B' in name:
                self.assertNotEqual(after_step1[name], digest, f'{name} did not update on step 1')

        trainer.train_step(model, opt, batch, mask_token_id=0, right_shift_logits=True,
                           generator=torch.Generator().manual_seed(8))
        self.assertEqual(trainer.base_weight_hashes(model), before_base)
        after_step2 = trainer.lora_weight_hashes(model)
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
            mock.patch.object(memory_bench, 'load_diffugpt', side_effect=_forbid),
            mock.patch.object(memory_bench, 'load_tokenizer', side_effect=_forbid),
            mock.patch.object(trainer, 'load_training_teams', side_effect=_forbid),
            mock.patch.object(trainer, 'slice_training_npz', side_effect=_forbid),
            mock.patch.object(trainer, 'run_tracer', side_effect=fake_runner),
        ):
            rc = trainer.main(['tracer', '--data', '/nonexistent.npz', '--out', '/tmp/unused', '--rows', '16'])
        self.assertEqual(rc, 0)
        self.assertEqual(captured, {'data': Path('/nonexistent.npz'), 'rows': 16})

    def test_run_tracer_end_to_end_with_mocked_side_effects_and_tiny_model(self):
        vocab = trainer.load_regmb_vocab(VOCAB_PATH)
        synthetic_teams = [(trainer.decode_team_84(sample_team_row(i), vocab), 0.5 + 0.01 * i if i else None)
                           for i in range(6)]
        tok = FakeTokenizer(max_vocab=4096)
        torch.manual_seed(0)
        tiny = TinyLoRALM(vocab_size=4096, dim=16, rank=4)
        fake_info = {'key': 'diffugpt_s', 'label': 'Fake DiffuGPT', 'repo': 'fake/repo',
                     'revision': '00000000', 'mask_id': tok.mask_token_id}

        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / 'out'
            args = trainer.build_parser().parse_args([
                'tracer', '--data', '/forbidden/real.npz', '--out', str(out_dir),
                '--rows', '6', '--overfit-rows', '4', '--overfit-steps', '10',
                '--eval-every', '5', '--train-steps', '3', '--batch-size', '2',
                '--overfit-lr', '2e-2', '--lr', '1e-2', '--device', 'cpu',
            ])
            with (
                mock.patch.object(memory_bench, 'load_diffugpt', side_effect=_forbid),
                mock.patch.object(memory_bench, 'load_tokenizer', side_effect=_forbid),
                mock.patch.object(trainer, 'slice_training_npz', side_effect=_forbid),
                mock.patch.object(trainer, 'load_diffugpt_small_lora', return_value=(tiny, tok, fake_info)),
                mock.patch.object(trainer, 'load_training_teams',
                                  side_effect=lambda _p, count, _seed, _v: synthetic_teams[:count]),
            ):
                result = trainer.run_tracer(args)
            saved = json.loads((out_dir / 'tracer_results.json').read_text())
            self.assertEqual(saved['rows'], 6)
            self.assertTrue(saved['base_weights_unchanged_after_overfit'])
            self.assertTrue(saved['base_weights_unchanged_after_train'])
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
                mock.patch.object(memory_bench, 'load_diffugpt', side_effect=_forbid),
                mock.patch.object(trainer, 'run_tracer', side_effect=_forbid),
            ):
                rc = trainer.main(['slice-data', '--data', str(src), '--out', str(dst), '--count', '5', '--seed', '1'])
            self.assertEqual(rc, 0)
            with np.load(dst) as sliced:
                self.assertEqual(sliced['all_X'].shape, (5, 84))
                self.assertFalse(bool(sliced['all_val'].any()))
                val_rows = {all_x[i].tobytes() for i in (0, 2, 6)}
                for row in sliced['all_X']:
                    self.assertNotIn(row.tobytes(), val_rows)


class ModalLauncherTest(unittest.TestCase):
    def test_packaged_files_are_self_contained_and_exclude_block_diffusion_and_masklab(self):
        for rel in launcher.PACKAGED_FILES:
            self.assertTrue((launcher.CODE_DIR / rel).is_file(), f'missing packaged file: {rel}')
        joined = ' '.join(launcher.PACKAGED_FILES)
        for forbidden in ('textdiff.py', 'textdiff_port.py', 'mask_lab_textdiff.py', 'jev_fill.py'):
            self.assertNotIn(forbidden, joined)

        # Copy ONLY PACKAGED_FILES into an isolated directory and prove the trainer imports cleanly there.
        with tempfile.TemporaryDirectory() as tmp:
            stage = Path(tmp) / 'code'
            for rel in launcher.PACKAGED_FILES:
                dest = stage / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(launcher.CODE_DIR / rel, dest)
            env = {k: v for k, v in os.environ.items() if k != 'PYTHONPATH'}
            proc = subprocess.run(
                [sys.executable, str(stage / 'scripts/masked_lora_trainer.py'), '--help'],
                cwd=str(stage / 'scripts'),
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertIn('tracer', proc.stdout)

    def test_build_remote_command_and_extract_out_dir(self):
        cmd = launcher.build_remote_command('tracer --data /tmp/rows.npz --out /tmp/custom_out --rows 256')
        self.assertEqual(cmd, ['python', launcher.REMOTE_TRAINER, 'tracer', '--data', '/tmp/rows.npz',
                               '--out', '/tmp/custom_out', '--rows', '256'])
        self.assertEqual(launcher.extract_out_dir('tracer --out /tmp/custom_out'), Path('/tmp/custom_out'))
        self.assertEqual(launcher.extract_out_dir('tracer --out=/tmp/eq_out'), Path('/tmp/eq_out'))
        self.assertEqual(launcher.extract_out_dir('tracer --rows 8'), Path(launcher.REMOTE_OUT))
        with self.assertRaises(ValueError):
            launcher.build_remote_command('   ')

    def test_run_trainer_subprocess_respects_custom_out_and_raises_on_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dest = Path(tmp) / 'rows.npz'
            custom_out = Path(tmp) / 'custom_out'
            seen_cmd: list[str] = []

            def fake_run(cmd, *, cwd, check):
                seen_cmd.extend(cmd)
                self.assertEqual(cwd, f'{launcher.REMOTE_CODE}/scripts')
                self.assertFalse(check)
                (custom_out / 'tracer_results.json').write_text('{"ok": true}')
                return types.SimpleNamespace(returncode=0)

            with mock.patch.object(subprocess, 'run', side_effect=fake_run):
                res = launcher.run_trainer_subprocess(
                    f'tracer --data /tmp/rows.npz --out {custom_out}',
                    data_bytes=b'npz-bytes',
                    data_dest=data_dest,
                )
            self.assertEqual(data_dest.read_bytes(), b'npz-bytes')
            self.assertEqual(seen_cmd[:2], ['python', launcher.REMOTE_TRAINER])
            self.assertEqual(res['files'], {'tracer_results.json': b'{"ok": true}'})

            local_out = Path(tmp) / 'local_out'
            written = launcher.write_output_files(local_out, res['files'])
            self.assertEqual([p.name for p in written], ['tracer_results.json'])
            self.assertEqual((local_out / 'tracer_results.json').read_bytes(), b'{"ok": true}')

            with self.assertRaises(ValueError):
                launcher.write_output_files(local_out, {'../escape.txt': b'bad'})

            with (
                mock.patch.object(subprocess, 'run', return_value=types.SimpleNamespace(returncode=2)),
                self.assertRaises(RuntimeError),
            ):
                launcher.run_trainer_subprocess(
                    f'tracer --data /tmp/rows.npz --out {custom_out}',
                    data_dest=data_dest,
                )


if __name__ == '__main__':
    unittest.main()
