"""Tests for the 84-field uniform discrete diffusion training, sampler, and port-equality gate (issue #275).

Verifies:
  1. Port-equality against verbatim upstream diffusion.py:563-614 (edb0f8c2) in float64 and float32
     (both per-field loss and full 84-field elbo_terms + gradients).
  2. Reverse posterior (diffusion.py:356-381) and posterior-mixing CFG (diffusion.py:1314-1334)
     equality across guidance weights {0, 1, 2, 4} and boundary times (t = 1.0, s = 1e-5).
  3. Unconstrained sampler behavior: uniform start, exact K passes (K for gamma in {0, 1}, 2K for CFG),
     no in-loop legality filter, valid table tokens (1..N_k), Showdown paste round-trip, and exact
     revision tracking (value changes beyond first, fixed-predictor noise floor, changes of mind).
  4. Training smoke (init CE near ln N_k, 32-team overfit >= 40%, MPS finiteness, checkpoint save/load,
     EpochLegality integration) and train_uniform84 / ask_grid84 wiring with non-legacy grid_seeds.

    /tmp/vgc-pilot/.venv/bin/python scripts/test_uniform_diffusion84.py   (from code/)
"""
import io
import math
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

import repo_paths  # noqa: F401  (src/ first on sys.path)

HERE = Path("/tmp/vgc-pilot/data/pokedex.json").exists()
if HERE:
    sys.path.insert(1, "/tmp/vgc-pilot/src")
    import numpy as np
    import torch
    import torch.nn.functional as F
    import ask_grid84 as AG
    import asked_vs_got as X
    import grid_seeds
    import team84 as T
    import train_uniform84 as TU
    import uniform_diffusion84 as UM
    import uniform_preflight84 as UP
    from corpus import parse_team_text
    HERE = TU.SHUFFLED_ONLY.exists()

TINY = dict(d=32, nlayer=2, nhead=2)
EXPECTED_REAL_VOCAB = dict(species=347, ability=200, item=149, move=497, nature=26, stat=33)


if HERE:
    VOCAB = T.regulation_vocab()
    CORPUS = X.setup()[0]
    _DATA = np.load(TU.SHUFFLED_ONLY)
    _LABELLED = ~np.isnan(_DATA["all_W"]) & ~_DATA["all_val"]
    SAMPLE_ROWS = _DATA["all_X"][_LABELLED][:64].astype(np.int64)
    SAMPLE_LABELS = _DATA["all_W"][_LABELLED][:64].astype(np.float32)


class _ConstantOneHotModel:
    """Deterministic fake model that always predicts one fixed clean token per column."""

    def __init__(self, target_tokens, vocab_sizes, device):
        self.vocab_sizes = vocab_sizes
        self.target = target_tokens.to(device)
        self.nets = [(torch.nn.Linear(1, 1).to(device), {})]

    def _logits_eval(self, uniform_tokens, time_values, win_rates):
        batch_size = uniform_tokens.shape[0]
        out = {}
        for field_key, cols in UM.FIELD_COLS.items():
            vocab_size = self.vocab_sizes[field_key]
            logits = torch.full((batch_size, len(cols), vocab_size), -1e9, device=uniform_tokens.device)
            targets = self.target[:batch_size, cols].unsqueeze(-1)
            logits.scatter_(-1, targets, 0.0)
            out[field_key] = logits
        return out


class _ScriptedTopModel:
    """Fake model whose argmax clean token follows a scripted sequence per pass, split by cond/uncond."""

    def __init__(self, cond_schedule, uncond_schedule, vocab_sizes, device):
        self.cond_schedule = cond_schedule
        self.uncond_schedule = uncond_schedule
        self.vocab_sizes = vocab_sizes
        self.nets = [(torch.nn.Linear(1, 1).to(device), {})]
        self.cond_calls = 0
        self.uncond_calls = 0

    def _logits_eval(self, uniform_tokens, time_values, win_rates):
        is_uncond = bool(torch.isnan(win_rates).all())
        if is_uncond:
            target_id = self.uncond_schedule[self.uncond_calls]
            self.uncond_calls += 1
        else:
            target_id = self.cond_schedule[self.cond_calls]
            self.cond_calls += 1
        batch_size = uniform_tokens.shape[0]
        out = {}
        for field_key, cols in UM.FIELD_COLS.items():
            vocab_size = self.vocab_sizes[field_key]
            logits = torch.full((batch_size, len(cols), vocab_size), -5.0, device=uniform_tokens.device)
            logits[..., target_id % vocab_size] = 5.0
            out[field_key] = logits
        return out


@unittest.skipUnless(HERE, "needs /tmp/vgc-pilot and lossdata84_shuffled_only.npz")
class PortEqualityTests(unittest.TestCase):
    def test_port_equality_gate_passes_in_float64_and_float32(self):
        real_vocab = UM.real_vocab_sizes(VOCAB)
        self.assertEqual(real_vocab, EXPECTED_REAL_VOCAB)
        report = TU.verify_port_equality(VOCAB)
        self.assertTrue(report["passed"], report)
        for precision in ("float64", "float32"):
            self.assertTrue(report[precision]["passed"], report[precision])
            for field_name, stats in report[precision]["per_field"].items():
                self.assertTrue(stats["passed"], f"{precision} {field_name}: {stats}")

    def test_elbo_terms_matches_upstream_loss_averaged_over_84_fields(self):
        real_vocab = UM.real_vocab_sizes(VOCAB)
        for dtype, atol, rtol in ((torch.float64, 1e-12, 1e-10), (torch.float32, 1e-3, 1e-3)):
            with self.subTest(dtype=dtype):
                torch.manual_seed(275)
                net = UM.UniformNet84(VOCAB, **TINY).to(dtype=dtype).eval()
                clean = UM.table_to_uniform(torch.as_tensor(SAMPLE_ROWS[:6], dtype=torch.long))
                times = torch.linspace(UM.TRAINING_EPS, 1.0 - UM.TRAINING_EPS, 6, dtype=dtype)
                noisy = UM.corrupt_uniform(clean, times, real_vocab, generator=torch.Generator().manual_seed(9))
                labels = torch.as_tensor(SAMPLE_LABELS[:6], dtype=dtype)

                # Reference: verbatim upstream_uniform_loss per field type averaged over 84 columns
                logits = net.logits(net(noisy, times, labels))
                ref_total = torch.zeros(6, dtype=dtype)
                for field_key, cols in UM.FIELD_COLS.items():
                    log_probs = F.log_softmax(logits[field_key], dim=-1)
                    ref_total = ref_total + UP.upstream_uniform_loss(
                        log_probs, noisy[:, cols], clean[:, cols], times, real_vocab[field_key]
                    ).sum(dim=1)
                ref_elbo = ref_total / UM.COLS

                # Candidate: elbo_terms directly on the same net, batch, times, and labels
                cand_elbo = UM.elbo_terms(net, clean, noisy, times, labels, real_vocab)
                self.assertEqual(cand_elbo.dtype, dtype)
                self.assertTrue(torch.allclose(ref_elbo, cand_elbo, atol=atol, rtol=rtol))

        # Also check elbo_terms directly on default float32 / DEV with backward gradients
        torch.manual_seed(275)
        net = UM.UniformNet84(VOCAB, **TINY).to(UM.DEV).train()
        clean = UM.table_to_uniform(torch.as_tensor(SAMPLE_ROWS[:8], dtype=torch.long, device=UM.DEV))
        times = torch.linspace(UM.TRAINING_EPS, 1.0 - UM.TRAINING_EPS, 8, device=UM.DEV)
        noisy = UM.corrupt_uniform(clean, times, real_vocab)
        labels = torch.as_tensor(SAMPLE_LABELS[:8], device=UM.DEV)
        terms = UM.elbo_terms(net, clean, noisy, times, labels, real_vocab)
        self.assertEqual(tuple(terms.shape), (8,))
        self.assertTrue(bool(torch.isfinite(terms).all()))
        terms.mean().backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None))

    def test_port_equality_loss_rejects_non_log_linear_exponent(self):
        log_probs = F.log_softmax(torch.randn(2, 6, 33), dim=-1)
        tokens = torch.randint(0, 33, (2, 6))
        times = torch.tensor([0.2, 0.8])
        with self.assertRaises(ValueError):
            UM.port_equality_loss(log_probs, tokens, tokens, times, 16.0, 33)

    def test_reverse_posterior_and_cfg_match_upstream_reference(self):
        real_vocab = UM.real_vocab_sizes(VOCAB)
        for dtype, tol in ((torch.float64, 1e-11), (torch.float32, 1e-5)):
            with self.subTest(dtype=dtype):
                gen = torch.Generator().manual_seed(42)
                for field_key, vocab_size in real_vocab.items():
                    log_cond = F.log_softmax(torch.randn(4, 6, vocab_size, generator=gen, dtype=dtype), dim=-1)
                    log_uncond = F.log_softmax(torch.randn(4, 6, vocab_size, generator=gen, dtype=dtype), dim=-1)
                    noisy = torch.randint(0, vocab_size, (4, 6), generator=gen)
                    for move_t_val, move_s_val in ((1.0, 0.99), (0.5, 0.49), (0.01, 1e-5)):
                        move_t = torch.full((4, 1, 1), move_t_val, dtype=dtype)
                        move_s = torch.full((4, 1, 1), move_s_val, dtype=dtype)
                        alpha_t, alpha_s = 1.0 - move_t, 1.0 - move_s
                        post_cand = UM.reverse_posterior(log_cond.exp(), noisy, move_s, move_t, vocab_size)
                        post_ref = UP.upstream_reverse_posterior(log_cond.exp(), noisy, alpha_s, alpha_t, vocab_size)
                        post_bayes = UP.bayes_reverse_posterior(log_cond.exp(), noisy, move_s, move_t, vocab_size)
                        self.assertTrue(torch.allclose(post_cand, post_ref, atol=tol, rtol=tol), field_key)
                        self.assertTrue(torch.allclose(post_cand, post_bayes, atol=tol, rtol=tol), field_key)
                        self.assertTrue(torch.allclose(post_cand.sum(dim=-1), torch.ones(4, 6, dtype=dtype), atol=tol))
                        for gamma in (0.0, 1.0, 2.0, 4.0, 8.0, 16.0):
                            cfg_cand = UM.cfg_posterior(log_cond, log_uncond, noisy, move_s, move_t, gamma, vocab_size)
                            cfg_ref = UP.upstream_cfg_posterior(log_cond, log_uncond, noisy, alpha_s, alpha_t, gamma, vocab_size)
                            self.assertTrue(torch.allclose(cfg_cand, cfg_ref, atol=tol, rtol=tol), (field_key, gamma))
                            self.assertTrue(bool((cfg_cand >= 0).all()))
                            self.assertTrue(torch.allclose(cfg_cand.sum(dim=-1), torch.ones(4, 6, dtype=dtype), atol=tol))


@unittest.skipUnless(HERE, "needs /tmp/vgc-pilot and lossdata84_shuffled_only.npz")
class SamplerAndRevisionsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        cls.net = UM.UniformNet84(VOCAB, **TINY).to(UM.DEV).eval()
        cls.model = UM.UniformModel84(VOCAB, CORPUS, [(cls.net, dict(TINY))], info=dict(test=True))
        cls.real_vocab = UM.real_vocab_sizes(VOCAB)

    def test_network_has_no_mask_embeddings_or_outputs(self):
        for field_key, expected_size in EXPECTED_REAL_VOCAB.items():
            self.assertEqual(self.net.emb[field_key].num_embeddings, expected_size)
            self.assertEqual(self.net.head[field_key].out_features, expected_size)
        with self.assertRaises(ValueError):
            UM.table_to_uniform(torch.zeros((2, UM.COLS), dtype=torch.long))
        with self.assertRaises(ValueError):
            UM.uniform_to_table(torch.full((2, UM.COLS), -1, dtype=torch.long))

    def test_sampler_runs_exact_passes_refuses_tables_and_round_trips_pastes(self):
        with self.assertRaises(ValueError):
            self.model.sample(2, ask=0.5, guidance=1.0, passes=8, tables=True)

        call_counts = {}
        original = self.model._logits_eval
        for label, ask, guidance, expected_calls in (
            ("unguided_none", None, 2.0, 10),
            ("unguided_zero", 0.5, 0.0, 10),
            ("conditional_one", 0.5, 1.0, 10),
            ("guided_two", 0.5, 2.0, 20),
        ):
            count = [0]
            self.model._logits_eval = lambda *a, _c=count: (_c.__setitem__(0, _c[0] + 1), original(*a))[1]
            try:
                rows = self.model.sample(3, ask=ask, guidance=guidance, passes=10, seed=5)
            finally:
                self.model._logits_eval = original
            call_counts[label] = count[0]
            self.assertEqual(count[0], expected_calls, label)
            self.assertEqual(tuple(rows.shape), (3, UM.COLS))
            self.assertTrue(bool((rows >= 1).all()))
            for field_key, cols in UM.FIELD_COLS.items():
                self.assertTrue(bool((rows[:, cols] <= self.real_vocab[field_key]).all()))

        # Deterministic under seed and pastes round-trip (both unconstrained rows and legal-Stat-Point rows)
        a = self.model.sample(4, ask=0.5, guidance=2.0, passes=12, seed=17)
        b = self.model.sample(4, ask=0.5, guidance=2.0, passes=12, seed=17)
        self.assertTrue(torch.equal(a, b))
        rt_raw = AG.round_trip(self.model, a.cpu().numpy())
        self.assertTrue(rt_raw["passed"], rt_raw)
        budgeted = a.cpu().numpy().copy()
        budgeted[:, UM.STAT0:] = np.minimum(budgeted[:, UM.STAT0:], 11)   # <= 60 Stat Points per Pokemon
        for row in budgeted:
            paste, _ = self.model.to_paste(row, repair=True)
            decoded_orig = T.decode(VOCAB, row)
            decoded_back = T.decode(VOCAB, T.encode(VOCAB, parse_team_text(paste, blank_ends_slot=True)))
            self.assertEqual(
                sorted(str(AG.slot_key(s)) for s in decoded_orig),
                sorted(str(AG.slot_key(s)) for s in decoded_back),
            )

    def test_revision_tracker_exact_changes_of_mind_and_cond_vs_uncond_source(self):
        # 5 passes: cond argmax schedule [1, 1, 2, 2, 3] -> 2 changes of mind; uncond [4, 5, 4, 5, 4] -> 4 changes of mind
        cond_sched = [1, 1, 2, 2, 3]
        uncond_sched = [4, 5, 4, 5, 4]
        fake_guided = _ScriptedTopModel(cond_sched, uncond_sched, self.real_vocab, torch.device("cpu"))
        _, tracker_guided = UM.sample_uniform(fake_guided, 2, ask=0.5, guidance=2.0, passes=5, seed=0)
        self.assertEqual(fake_guided.cond_calls, 5)
        self.assertEqual(fake_guided.uncond_calls, 5)
        self.assertEqual(tuple(tracker_guided.top_predictions.shape), (5, 2, UM.COLS))
        self.assertTrue(bool((tracker_guided.changes_of_mind == 2).all()))
        self.assertTrue(bool((tracker_guided.guided_changes_of_mind == 2).all()))

        # Unguided (gamma = 0) must use uncond predictions -> 4 changes of mind
        fake_uncond = _ScriptedTopModel(cond_sched, uncond_sched, self.real_vocab, torch.device("cpu"))
        _, tracker_uncond = UM.sample_uniform(fake_uncond, 2, ask=0.5, guidance=0.0, passes=5, seed=0)
        self.assertEqual(fake_uncond.cond_calls, 0)
        self.assertEqual(fake_uncond.uncond_calls, 5)
        self.assertTrue(bool((tracker_uncond.changes_of_mind == 4).all()))
        self.assertTrue(bool((tracker_uncond.guided_changes_of_mind == 4).all()))

    def test_constant_one_hot_predictor_matches_theoretical_noise_floor(self):
        # Verify Monte Carlo value_changes_beyond_first matches uniform_preflight84.revision_noise_floor
        batch_size = 512
        targets = torch.zeros((batch_size, UM.COLS), dtype=torch.long)
        fake = _ConstantOneHotModel(targets, self.real_vocab, torch.device("cpu"))
        _, tracker = UM.sample_uniform(fake, batch_size, ask=None, guidance=0.0, passes=84, seed=123, keep_top_predictions=False)
        summary = tracker.summary()
        for field_key in UM.FIELD_KEYS:
            field_stats = summary["by_field"][field_key]
            self.assertEqual(field_stats["changes_of_mind"], 0.0)
            self.assertAlmostEqual(
                field_stats["value_changes_beyond_first"],
                field_stats["noise_floor"],
                delta=0.15,
                msg=f"{field_key}: {field_stats}",
            )


@unittest.skipUnless(HERE, "needs /tmp/vgc-pilot and lossdata84_shuffled_only.npz")
class TrainingSmokeAndWiringTests(unittest.TestCase):
    def test_training_smoke_init_ce_and_32_team_overfit_pass(self):
        report = TU.run_training_smoke(steps=60, seed=0)
        self.assertTrue(report["port_equality"]["passed"], report["port_equality"])
        self.assertTrue(report["port_equality"]["posterior_and_cfg"]["passed"], report["port_equality"])
        self.assertTrue(report["init_ce"]["passed"], report["init_ce"])
        self.assertTrue(report["overfit_32_teams"]["passed"], report["overfit_32_teams"])
        self.assertGreaterEqual(report["overfit_32_teams"]["drop_fraction"], 0.40)
        self.assertTrue(report["production_batch_step"]["passed"], report["production_batch_step"])
        self.assertEqual(report["production_batch_step"]["batch_size"], 256)
        self.assertTrue(report["sampler_smoke"]["passed"], report["sampler_smoke"])
        self.assertTrue(report["passed"], report)

    def test_two_epoch_training_save_load_and_epoch_legality(self):
        model = UM.UniformModel84.train_new(
            VOCAB,
            CORPUS,
            SAMPLE_ROWS[:48],
            SAMPLE_LABELS[:48],
            epochs=2,
            batch=24,
            val=(SAMPLE_ROWS[48:], SAMPLE_LABELS[48:]),
            patience=3,
            early_stopping_threshold=0.0,
            label_noise=1 / 48,
            legality_teams=4,
            legality_passes=12,
            legality_guidance=1.0,
            **TINY,
        )
        curves = model.info["curves"]
        self.assertEqual([e["epoch"] for e in curves], [1, 2])
        for entry in curves:
            self.assertTrue(math.isfinite(entry["train_loss"]))
            self.assertTrue(math.isfinite(entry["val_loss"]))
            self.assertEqual(entry["legality_teams"], 4)
            self.assertIn("illegal_share", entry)
            self.assertIn("rule_shares", entry)
            self.assertIn("copied_share", entry)

        rows = model.sample(4, ask=0.5, guidance=2.0, passes=12, seed=9)
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "udlm.pt"
            model.save(ckpt)
            reloaded = UM.UniformModel84.load(ckpt, CORPUS)
        self.assertTrue(torch.equal(rows, reloaded.sample(4, ask=0.5, guidance=2.0, passes=12, seed=9)))

    def test_train_uniform84_wiring_matches_preregistration(self):
        self.assertEqual(TU.NAMES["udlm_loglinear"], "stat84_udlm_loglinear_es3_cap80")
        self.assertEqual(TU.DATA.name, "lossdata84_shuffled_only.npz")
        fake_table = dict(
            all_X=SAMPLE_ROWS[:6],
            all_W=np.array([0.5, np.nan, 0.25, 0.75, np.nan, 0.1]),
            all_val=np.array([False, False, False, True, True, False]),
            sha256="abc",
            table="lossdata84_shuffled_only.npz",
        )
        with (
            mock.patch.object(TU, "train") as mock_train,
            mock.patch.object(TU, "load", return_value=fake_table) as mock_load,
            mock.patch.object(TU, "check_rows"),
            redirect_stdout(io.StringIO()),
        ):
            TU.main(["udlm_loglinear"])
        mock_load.assert_called_once_with(TU.SHUFFLED_ONLY)
        mock_train.assert_called_once()
        arm, rows, labels, (val_rows, val_labels) = mock_train.call_args.args
        self.assertEqual(arm, "udlm_loglinear")
        self.assertEqual((len(rows), len(val_rows)), (3, 1))
        self.assertEqual(labels.tolist(), [0.5, 0.25, 0.1])
        self.assertEqual(val_labels.tolist(), [0.75])
        self.assertEqual(
            mock_train.call_args.kwargs,
            dict(epochs=80, patience=3, early_stopping_threshold=0.0, label_noise=1 / 48, every_epoch=True),
        )

    def test_ask_grid84_wiring_uses_non_legacy_seed_and_records_revisions(self):
        row = AG.MODELS["udlm_loglinear_nofilter"]
        self.assertEqual(row["checkpoint"], "stat84_udlm_loglinear_es3_cap80")
        self.assertEqual(row["reader"], "Uniform")
        self.assertEqual(row["decoder"], "uniform_nofilter")
        self.assertTrue(row["uniform"])
        self.assertEqual(AG.table_of(row), "lossdata84_shuffled_only.npz")
        self.assertNotIn("seed0", row)
        with tempfile.TemporaryDirectory() as tmp:
            seed_base = grid_seeds.seed0("udlm_loglinear_nofilter_g1", Path(tmp) / "missing.json")
        self.assertNotEqual(seed_base, grid_seeds.LEGACY)
        self.assertGreaterEqual(seed_base, 100_000)

        torch.manual_seed(0)
        net = UM.UniformNet84(VOCAB, **TINY).to(UM.DEV).eval()
        model = UM.UniformModel84(VOCAB, CORPUS, [(net, dict(TINY))])
        sampler = UM.UniformSampler84(model, passes=8)

        class _AcceptAllValidator:
            def __call__(self, paste):
                return None

            def close(self):
                return None

        with (
            mock.patch.object(AG, "Validator", _AcceptAllValidator),
            mock.patch.object(AG, "KEEP", 4),
            mock.patch.object(AG, "BATCH", 4),
            mock.patch.object(AG, "MAX_TRIES", 8),
            mock.patch.object(AG, "GUIDANCE", 1.0),
            mock.patch.object(AG, "SEED0", seed_base),
        ):
            kept, _stats, counts = AG.draw(sampler, 0.5, seed=1)
        self.assertEqual(len(kept), 4)
        self.assertIn("revisions", counts)
        for bucket in ("all_tries", "kept"):
            rev = counts["revisions"][bucket]
            self.assertEqual(rev["teams"], 4)
            self.assertEqual(rev["passes"], 8)
            self.assertIn("mean_fields_changed_per_pass", rev)
            self.assertEqual(set(rev["by_field"]), set(UM.FIELD_KEYS))

    def test_ask_grid84_pass_counts_have_distinct_identities_and_reject_mismatched_files(self):
        # Different AG84_PASSES values must produce distinct suffixes, tags, filenames, and seed bases
        suffixes = [AG.grid_suffix(False, 1.0, str(k)) for k in (84, 128, 256)]
        self.assertEqual(suffixes, ["_g1_k84", "_g1_k128", "_g1_k256"])
        self.assertEqual(AG.grid_suffix(False, 2.0, None), "")
        self.assertEqual(AG.grid_suffix(True, 1.0, "128"), "_unguided_k128")
        with tempfile.TemporaryDirectory() as tmp:
            seeds = {grid_seeds.seed0(f"udlm_loglinear_nofilter{s}", Path(tmp) / f"{s}.json") for s in suffixes}
            self.assertEqual(len(seeds), 3)
            ckpt = Path(tmp) / "fake.pt"
            ckpt.write_bytes(b"ckpt")
            with (
                mock.patch.object(AG, "checkpoint_path", return_value=ckpt),
                mock.patch.object(AG, "PASSES", 256),
                self.assertRaises(SystemExit),
            ):
                AG.require_same_checkpoint({"cells": {"c": {}}, "passes": 84})
        # load_sampler refuses udlm_loglinear_nofilter when AG84_PASSES is unset (until #277 locks K*)
        with (
            mock.patch.object(AG, "MODEL", "udlm_loglinear_nofilter"),
            mock.patch.object(AG, "PASSES", None),
            self.assertRaises(SystemExit),
        ):
            AG.load_sampler(w=object())

    def test_nan_guards_protect_best_checkpoint_and_abort_non_finite_steps(self):
        net = UM.UniformNet84(VOCAB, **TINY).to(UM.DEV)
        prior_sd = {k: v.detach().clone() for k, v in net.state_dict().items()}
        state = dict(best=1.0, best_epoch=1, best_sd=prior_sd, reset_epoch=1, step=10)
        called = []
        UM._update_best(state, dict(epoch=2, val_loss=float("nan")), net, 0.0, lambda *a: called.append(a), lambda m, e: m)
        self.assertEqual(state["best"], 1.0)
        self.assertEqual(state["best_epoch"], 1)
        self.assertIs(state["best_sd"], prior_sd)
        self.assertEqual(called, [])

        # Non-finite training loss raises RuntimeError before stepping AdamW or updating EMA
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
        clean = UM.table_to_uniform(torch.as_tensor(SAMPLE_ROWS[:8], dtype=torch.long, device=UM.DEV))
        labels = torch.as_tensor(SAMPLE_LABELS[:8], device=UM.DEV)
        with (
            mock.patch.object(UM, "batch_loss", return_value=torch.tensor(float("nan"), device=UM.DEV, requires_grad=True)),
            self.assertRaises(RuntimeError),
        ):
            UM._train_epoch(
                net,
                None,
                opt,
                clean,
                labels,
                net.vocab_sizes,
                epoch_num=1,
                batch=8,
                lr_at=lambda _s: 1e-3,
                step=0,
                ema=0.0,
                label_noise=0.0,
                noise_rng=torch.Generator().manual_seed(0),
            )

    def test_sampler_domain_guards_and_field_cols_match_lossdown(self):
        import lossdown as L
        self.assertEqual(UM.FIELD_COLS, {**L.COLS_BY_KEY, "stat": list(range(UM.STAT0, UM.COLS))})
        torch.manual_seed(0)
        model = UM.UniformModel84(VOCAB, CORPUS, [(UM.UniformNet84(VOCAB, **TINY).to(UM.DEV).eval(), dict(TINY))])
        for bad_kw in (dict(passes=0), dict(eps=0.0), dict(eps=1.0), dict(temperature=0.0)):
            with self.subTest(bad_kw=bad_kw), self.assertRaises(ValueError):
                model.sample(2, ask=0.5, **bad_kw)

    def test_full_width_batch_256_step_is_finite_on_device(self):
        torch.manual_seed(0)
        net = UM.UniformNet84(VOCAB, **TU.NET).to(UM.DEV).train()
        repeated_rows = np.tile(SAMPLE_ROWS, (4, 1))[:256]
        repeated_labels = np.tile(SAMPLE_LABELS, 4)[:256]
        clean = UM.table_to_uniform(torch.as_tensor(repeated_rows, dtype=torch.long, device=UM.DEV))
        labels = torch.as_tensor(repeated_labels, device=UM.DEV)
        loss = UM.batch_loss(net, clean, labels, net.vocab_sizes)
        self.assertTrue(bool(torch.isfinite(loss)))
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0))
        self.assertTrue(math.isfinite(grad_norm))


if __name__ == "__main__":
    unittest.main()
