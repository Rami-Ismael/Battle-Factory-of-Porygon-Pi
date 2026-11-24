"""Weights & Biases experiment tracking for 84-field training runs (src/wandb_tracking.py, diffusion84.py, train84.py).

Unit tests run on the CPU with an injected fake wandb module (no network, no /tmp/vgc-pilot data), so CI runs them.
End-to-end tests against Model84.train_new, Model84.fine_tuned and train84.py run when /tmp/vgc-pilot data is present.

    /tmp/vgc-pilot/.venv/bin/python scripts/test_wandb_tracking.py   (from code/)
"""
import argparse
import io
import math
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

import numpy as np

import repo_paths  # noqa: F401  (src/ first on sys.path)
import wandb_tracking as WT

HERE = None
if Path("/tmp/vgc-pilot/data/pokedex.json").exists():
    sys.path.insert(1, "/tmp/vgc-pilot/src")
    import torch
    import asked_vs_got as X
    import diffusion84 as M
    import team84 as T
    import train84 as TR
    if (X.RES / "lossdata84.npz").exists():
        HERE = True


class FakeConfig(dict):
    def update(self, values, allow_val_change=False):
        self._allow_val_change = allow_val_change
        super().update(values)


class FakeRun:
    def __init__(self, *, project, entity=None, name=None, mode="online", config=None, fail_log=False):
        self.project = project
        self.entity = entity
        self.name = name or "run-1"
        self.id = "fake1234"
        self.mode = mode
        self.url = None if mode == "offline" else f"https://wandb.ai/{entity or 'user'}/{project}/runs/{self.id}"
        self.config = FakeConfig(config or {})
        self.summary = {}
        self.metrics_defined = []
        self.logged = []
        self.finished = 0
        self._fail_log = fail_log

    def define_metric(self, name, step_metric=None):
        self.metrics_defined.append((name, step_metric))

    def log(self, data):
        if self._fail_log:
            raise RuntimeError("simulated log failure")
        self.logged.append(dict(data))

    def finish(self):
        self.finished += 1


class FakeWandb:
    def __init__(self, *, fail_modes=(), fail_log=False):
        self.calls = []
        self.runs = []
        self.fail_modes = set(fail_modes)
        self.fail_log = fail_log

    def init(self, **kwargs):
        self.calls.append(dict(kwargs))
        if kwargs.get("mode") in self.fail_modes:
            raise RuntimeError(f"init failed for {kwargs.get('mode')}")
        run = FakeRun(project=kwargs["project"], entity=kwargs.get("entity"), name=kwargs.get("name"),
                      mode=kwargs.get("mode", "online"), config=kwargs.get("config"), fail_log=self.fail_log)
        self.runs.append(run)
        return run


class WandbHelpers(unittest.TestCase):
    def test_resolve_mode_precedence_and_validation(self):
        self.assertEqual(WT.resolve_mode(env={}), "online")
        self.assertEqual(WT.resolve_mode(smoke=True, env={}), "disabled")
        self.assertEqual(WT.resolve_mode(smoke=True, env={"WANDB_MODE": "offline"}), "offline")
        self.assertEqual(WT.resolve_mode("online", smoke=True, env={"WANDB_MODE": "offline"}), "online")
        self.assertEqual(WT.resolve_mode("online", disabled=True, env={"WANDB_MODE": "online"}), "disabled")
        with self.assertRaises(ValueError):
            WT.resolve_mode("invalid", env={})

    def test_resolve_project_and_entity(self):
        self.assertEqual(WT.resolve_project(env={}), WT.DEFAULT_PROJECT)
        self.assertEqual(WT.resolve_project(env={"WANDB_PROJECT": "env-proj"}), "env-proj")
        self.assertEqual(WT.resolve_project("cli-proj", env={"WANDB_PROJECT": "env-proj"}), "cli-proj")
        self.assertIsNone(WT.resolve_entity(env={}))
        self.assertEqual(WT.resolve_entity(env={"WANDB_ENTITY": "team-a"}), "team-a")
        self.assertEqual(WT.resolve_entity("team-b", env={"WANDB_ENTITY": "team-a"}), "team-b")

    def test_epoch_metrics_flattens_scalars_rule_shares_and_drops_non_scalars(self):
        entry = {
            "epoch": 3,
            "train_loss": 1.82,
            "val_loss": 1.91,
            "labelled_val_loss": 1.95,
            "illegal_share": 0.25,
            "errors_per_team": 0.5,
            "legality_teams": 64,
            "legality_seconds": 1.2,
            "rejection_reasons": [["wrong Mega Stone", 4]],
            "rule_shares": {"wrong Mega Stone": 0.0625, "Stat Points over 66": 0.0},
            "minutes": 2.5,
        }
        got = WT.epoch_metrics(entry, lr=3e-4, step=300, grad_norm=0.75, best_val_loss=math.inf)
        self.assertEqual(got["epoch"], 3)
        self.assertAlmostEqual(got["train_loss"], 1.82)
        self.assertAlmostEqual(got["val_loss"], 1.91)
        self.assertAlmostEqual(got["labelled_val_loss"], 1.95)
        self.assertAlmostEqual(got["illegal_share"], 0.25)
        self.assertAlmostEqual(got["legal_share"], 0.75)
        self.assertAlmostEqual(got["rule_shares/wrong Mega Stone"], 0.0625)
        self.assertAlmostEqual(got["rule_shares/Stat Points over 66"], 0.0)
        self.assertAlmostEqual(got["learning_rate"], 3e-4)
        self.assertEqual(got["optimizer_step"], 300)
        self.assertAlmostEqual(got["grad_norm"], 0.75)
        self.assertNotIn("rejection_reasons", got)
        self.assertNotIn("best_val_loss", got)              # math.inf is excluded

        with_best = WT.epoch_metrics({"epoch": 4, "val_loss": 1.80}, best_val_loss=1.80)
        self.assertAlmostEqual(with_best["best_val_loss"], 1.80)

    def test_clean_config_strips_large_keys_and_private_paths(self):
        raw = {
            "lr": np.float32(4e-4),
            "batch": np.int64(256),
            "curves": [{"epoch": 1}],
            "base": {"n_train": 100},
            "table_path": Path("/Users/alice/vgc-data/lossdata84.npz"),
            "home_path": "/home/alice/secret/model.pt",
            "net_kw": {"d": 256, "nlayer": 6},
        }
        cleaned = WT.clean_config(raw)
        self.assertNotIn("curves", cleaned)
        self.assertNotIn("base", cleaned)
        self.assertAlmostEqual(cleaned["lr"], 4e-4, places=6)
        self.assertEqual(cleaned["batch"], 256)
        self.assertEqual(cleaned["table_path"], "lossdata84.npz")
        self.assertEqual(cleaned["home_path"], "model.pt")
        self.assertEqual(cleaned["net_kw"], {"d": 256, "nlayer": 6})

    def test_summary_metrics_extracts_final_values_and_checkpoint_basename(self):
        info = {
            "best_epoch": 5,
            "stopped_epoch": 8,
            "best_val_loss": 1.42,
            "minutes": 12.3,
            "n_train": 1000,
            "curves": [{"epoch": 1, "train_loss": 2.0, "val_loss": 2.1},
                       {"epoch": 8, "train_loss": 1.35, "val_loss": 1.44, "illegal_share": 0.125}],
        }
        got = WT.summary_metrics(info, stop_reason="no new lowest validation loss for 3 epochs after epoch 5",
                                 checkpoint_name="/Users/alice/ck/stat84_test.pt")
        self.assertEqual(got["best_epoch"], 5)
        self.assertEqual(got["stopped_epoch"], 8)
        self.assertAlmostEqual(got["best_val_loss"], 1.42)
        self.assertAlmostEqual(got["final_train_loss"], 1.35)
        self.assertAlmostEqual(got["final_val_loss"], 1.44)
        self.assertAlmostEqual(got["final_illegal_share"], 0.125)
        self.assertEqual(got["checkpoint"], "stat84_test.pt")
        self.assertIn("patience" not in got and "3 epochs", got["stop_reason"])

    def test_cli_args_build_tracker_with_tags_and_group(self):
        p = WT.add_cli_args(argparse.ArgumentParser())
        args = p.parse_args(["--wandb-mode", "offline", "--wandb-project", "demo", "--wandb-entity", "org",
                             "--wandb-group", "masking", "--wandb-name", "custom", "--wandb-tags", "exp1,linear"])
        tracker = WT.from_cli_args(args, name="default-name", tags=("84-field", "linear"))
        self.assertEqual(tracker.mode, "offline")
        self.assertEqual(tracker.project, "demo")
        self.assertEqual(tracker.entity, "org")
        self.assertEqual(tracker.group, "masking")
        self.assertEqual(tracker.name, "custom")
        self.assertEqual(tracker.tags, ["84-field", "linear", "exp1"])
        self.assertEqual(WT.from_cli_args(None, name="x").mode, "disabled")


class WandbTrackerLifecycle(unittest.TestCase):
    def test_context_manager_defines_epoch_axis_logs_and_finishes(self):
        fake = FakeWandb()
        with WT.WandbTracker(name="run-a", project="proj", group="grp", tags=["84-field"],
                             config={"arm": "shuffled_linear"}, mode="online", dir="/tmp", wandb_module=fake) as tracker:
            tracker.start(config={"lr": 4e-4, "n_params": 1234})
            tracker.log_epoch({"epoch": 1, "train_loss": 2.1, "val_loss": 2.0, "minutes": 0.5},
                              lr=4e-4, step=10, grad_norm=0.4, best_val_loss=2.0)
            tracker.log_summary({"best_epoch": 1, "stopped_epoch": 1, "best_val_loss": 2.0,
                                 "curves": [{"epoch": 1, "train_loss": 2.1, "val_loss": 2.0}]},
                                stop_reason="reached the cap of 1 epochs", checkpoint_name="m.pt")
            meta = tracker.metadata()
        self.assertEqual(len(fake.runs), 1)
        run = fake.runs[0]
        self.assertEqual(run.metrics_defined, [("epoch", None), ("*", "epoch")])
        self.assertEqual(run.config["arm"], "shuffled_linear")
        self.assertEqual(run.config["lr"], 4e-4)
        self.assertEqual(run.config["n_params"], 1234)
        self.assertEqual(len(run.logged), 1)
        self.assertEqual(run.logged[0]["epoch"], 1)
        self.assertAlmostEqual(run.logged[0]["learning_rate"], 4e-4)
        self.assertEqual(run.summary["checkpoint"], "m.pt")
        self.assertEqual(run.finished, 1)
        self.assertEqual(meta["run_id"], "fake1234")
        self.assertEqual(meta["mode"], "online")

    def test_online_init_failure_falls_back_to_offline(self):
        fake = FakeWandb(fail_modes={"online"})
        err = io.StringIO()
        with redirect_stderr(err):
            with WT.WandbTracker(name="run-b", mode="online", dir="/tmp", wandb_module=fake) as tracker:
                tracker.log_epoch({"epoch": 1, "train_loss": 1.5})
                meta = tracker.metadata()
        self.assertEqual([c["mode"] for c in fake.calls], ["online", "offline"])
        self.assertEqual(tracker.mode, "offline")
        self.assertEqual(meta["mode"], "offline")
        self.assertEqual(len(fake.runs[0].logged), 1)
        self.assertIn("falling back to offline", err.getvalue())

    def test_total_init_failure_disables_without_crashing(self):
        fake = FakeWandb(fail_modes={"online", "offline"})
        err = io.StringIO()
        with redirect_stderr(err):
            with WT.WandbTracker(name="run-c", mode="online", dir="/tmp", wandb_module=fake) as tracker:
                tracker.log_epoch({"epoch": 1, "train_loss": 1.5})
                self.assertIsNone(tracker.metadata())
        self.assertEqual(tracker.mode, "disabled")
        self.assertEqual(fake.runs, [])

    def test_disabled_tracker_never_touches_wandb(self):
        fake = FakeWandb()
        with WT.WandbTracker(name="run-d", disabled=True, wandb_module=fake) as tracker:
            tracker.log_epoch({"epoch": 1, "train_loss": 1.5})
            self.assertIsNone(tracker.metadata())
        self.assertEqual(fake.calls, [])

    def test_log_failure_warns_once_and_does_not_abort(self):
        fake = FakeWandb(fail_log=True)
        err = io.StringIO()
        with redirect_stderr(err):
            with WT.WandbTracker(name="run-e", mode="online", dir="/tmp", wandb_module=fake) as tracker:
                tracker.log_epoch({"epoch": 1, "train_loss": 1.5})
                tracker.log_epoch({"epoch": 2, "train_loss": 1.4})
        self.assertEqual(err.getvalue().count("[wandb]"), 1)
        self.assertEqual(fake.runs[0].finished, 1)


@unittest.skipUnless(HERE, "needs /tmp/vgc-pilot and lossdata84.npz")
class Diffusion84Integration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        d = np.load(X.RES / "lossdata84.npz")
        has = ~np.isnan(d["all_W"])
        x, w = d["all_X"][has][:128].astype(np.int64), d["all_W"][has][:128]
        cls.V, cls.corpus = T.regulation_vocab(), X.setup()[0]
        cls.train_args = (x[:96], w[:96])
        cls.val = (x[96:], w[96:])
        cls.tiny = dict(d=32, nlayer=1, nhead=2, batch=32, epochs=2, seed=0)

    def test_train_new_tracks_every_epoch_without_changing_weights(self):
        fake = FakeWandb()
        tracker = WT.WandbTracker(name="tiny-train", mode="online", dir="/tmp", wandb_module=fake)
        with redirect_stdout(io.StringIO()):
            plain = M.Model84.train_new(self.V, self.corpus, *self.train_args, val=self.val, **self.tiny)
            tracked = M.Model84.train_new(self.V, self.corpus, *self.train_args, val=self.val, tracker=tracker, **self.tiny)
        self.assertEqual(len(fake.runs), 1)
        run = fake.runs[0]
        self.assertEqual(run.finished, 1)
        self.assertEqual([e["epoch"] for e in run.logged], [1, 2])
        self.assertIn("n_params", run.config)
        self.assertIn("learning_rate", run.logged[0])
        self.assertIn("grad_norm", run.logged[0])
        self.assertIn("optimizer_step", run.logged[0])
        self.assertEqual(run.summary["stopped_epoch"], 2)
        for a, b in zip(plain.info["curves"], tracked.info["curves"], strict=True):
            self.assertAlmostEqual(a["train_loss"], b["train_loss"], delta=1e-5)
            self.assertAlmostEqual(a["val_loss"], b["val_loss"], delta=1e-5)
        for k, v in plain.net.state_dict().items():
            self.assertTrue(torch.allclose(v, tracked.net.state_dict()[k], atol=1e-5), k)

    def test_fine_tuned_tracks_epoch_zero_and_watched_validation_sets(self):
        fake = FakeWandb()
        with redirect_stdout(io.StringIO()):
            base = M.Model84.train_new(self.V, self.corpus, *self.train_args, d=32, nlayer=1, nhead=2, batch=32, epochs=1, seed=0)
            tracker = WT.WandbTracker(name="tiny-ft", job_type="fine-tune", mode="online", dir="/tmp", wandb_module=fake)
            base.fine_tuned(*self.train_args, epochs=1, val=self.val, batch=32, watch=dict(labelled=self.val), tracker=tracker)
        run = fake.runs[0]
        self.assertEqual([e["epoch"] for e in run.logged], [0, 1])
        self.assertIn("labelled_val_loss", run.logged[0])
        self.assertIn("train_loss", run.logged[1])
        self.assertNotIn("train_loss", run.logged[0])
        self.assertTrue(run.config["fine_tune"])
        self.assertEqual(run.finished, 1)

    def test_train84_wires_cli_flags_and_records_wandb_metadata(self):
        fake = FakeWandb()
        args = TR.parse_args(["shuffled_linear", "--wandb-mode", "online", "--wandb-project", "porygon-test", "--wandb-group", "g1"])
        model = mock.MagicMock()
        model.info = dict(best_epoch=1, stopped_epoch=1, cap=80, patience=3, best_val_loss=1.5,
                          curves=[dict(epoch=1, train_loss=1.6, val_loss=1.5)])
        rows = np.zeros((4, 84), dtype=np.int64)
        labels = np.array([0.5, 0.4, 0.6, 0.3], dtype=np.float32)
        orig_from_cli = WT.from_cli_args
        with (tempfile.TemporaryDirectory() as tmp, mock.patch.object(TR, "CK", Path(tmp)), mock.patch.object(TR.X, "RES", Path(tmp)),
              mock.patch.object(TR.X, "setup", return_value=[None]), mock.patch.object(TR.T, "regulation_vocab", return_value=None),
              mock.patch.object(TR, "WANDB_ARGS", args), mock.patch.object(TR, "DATA_TABLE", "lossdata84_shuffled_only.npz"),
              mock.patch.object(TR, "DATA_SHA", "deadbeef"),
              mock.patch.object(TR.WT, "from_cli_args", side_effect=lambda a, **kw: orig_from_cli(a, wandb_module=fake, **kw)),
              mock.patch.object(TR.M.Model84, "train_new", return_value=model) as train_new, redirect_stdout(io.StringIO())):
            TR.train("shuffled_linear", rows[:3], labels[:3], (rows[3:], labels[3:]), epochs=80, patience=3)
        sent_tracker = train_new.call_args.kwargs["tracker"]
        self.assertIsInstance(sent_tracker, WT.WandbTracker)
        self.assertEqual(len(fake.runs), 1)
        run = fake.runs[0]
        self.assertEqual(run.project, "porygon-test")
        self.assertEqual(run.finished, 1)
        self.assertEqual(run.summary["checkpoint"], "stat84_shuffled_noise_linear_es3_cap80.pt")
        self.assertEqual(model.info["wandb"]["project"], "porygon-test")

    def test_train65_and_hindsight_relabel_parse_wandb_flags(self):
        import hindsight_relabel as HR
        import train65 as T65
        t65_args = T65.parse_args(["subset_10_39", "--wandb-mode", "offline", "--wandb-group", "subsets"])
        self.assertEqual((t65_args.arm, t65_args.wandb_mode, t65_args.wandb_group), ("subset_10_39", "offline", "subsets"))
        hr_args = HR.parse_args(["train", "--no-wandb", "--wandb-project", "ft-proj"])
        self.assertEqual((hr_args.cmd, hr_args.no_wandb, hr_args.wandb_project), ("train", True, "ft-proj"))

    def test_real_wandb_offline_run_records_epochs(self):
        try:
            import wandb  # noqa: F401
        except ImportError:
            self.skipTest("wandb not installed")
        with tempfile.TemporaryDirectory() as tmp:
            tracker = WT.WandbTracker(name="offline-smoke", project="test-offline", mode="offline", dir=tmp)
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                with tracker:
                    m = M.Model84.train_new(self.V, self.corpus, *self.train_args, val=self.val, tracker=tracker, **self.tiny)
                    meta = tracker.metadata()
            self.assertEqual(len(m.info["curves"]), 2)
            self.assertEqual(meta["mode"], "offline")
            self.assertTrue(meta["run_id"])
            self.assertTrue(list((Path(tmp) / "wandb").glob("offline-run-*")))


if __name__ == "__main__":
    unittest.main()
