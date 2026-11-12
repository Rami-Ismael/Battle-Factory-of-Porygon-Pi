"""Training reads the shuffled-only table by default (owner, 2026-11-11): src/train84.py and src/ask_grid84.py.

The older table pools labels from listed-order battles and run files. An arm that reproduces a model trained on it
runs only when asked for by name, with --older-table.

Needs this laptop's /tmp/vgc-pilot data; skipped without it.

    /tmp/vgc-pilot/.venv/bin/python scripts/test_train84_tables.py   (from code/)
"""
import io
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import repo_paths  # noqa: F401  (src/ first on sys.path)

HERE = Path("/tmp/vgc-pilot/data/pokedex.json").exists()
if HERE:
    sys.path.insert(1, "/tmp/vgc-pilot/src")
    import train84 as TR
    import ask_grid84 as A


@unittest.skipUnless(HERE, "needs /tmp/vgc-pilot data")
class TrainingTable(unittest.TestCase):
    def test_the_default_table_is_the_shuffled_only_one(self):
        self.assertEqual(TR.DATA, TR.SHUFFLED_ONLY)
        self.assertEqual(TR.DATA.name, "lossdata84_shuffled_only.npz")
        self.assertEqual(TR.load.__defaults__, (TR.SHUFFLED_ONLY,))

    def test_the_older_table_keeps_its_own_name(self):
        self.assertEqual(TR.OLDER_TABLE.name, "lossdata84.npz")

    def test_an_arm_of_the_older_table_refuses_without_the_flag(self):
        """train, load and train_position_arms are replaced first: main() is the real training entry point, and a missing
        guard must fail this test, never start a training that overwrites a stored checkpoint."""
        started = AssertionError("an arm of the older table loaded a table or started training without --older-table")
        for arm in (name.split() for name in TR.OLDER_TABLE_ARMS):
            with (self.subTest(arm=arm), mock.patch.object(TR, "train", side_effect=started), mock.patch.object(TR, "load", side_effect=started),
                  mock.patch.object(TR, "train_position_arms", side_effect=started), redirect_stdout(io.StringIO()),
                  self.assertRaises(SystemExit) as stop):
                TR.main(arm)
            self.assertIn("--older-table", str(stop.exception))

    def test_the_flag_is_parsed(self):
        self.assertTrue(TR.parse_args(["byepoch", "--older-table"]).older_table)
        self.assertFalse(TR.parse_args(["byepoch"]).older_table)

    def test_a_checkpoint_saved_during_training_names_its_table(self):
        with mock.patch.object(TR, "DATA_TABLE", "lossdata84_shuffled_only.npz"), mock.patch.object(TR, "DATA_SHA", "abc"):
            saved = TR.stamped(SimpleNamespace(info={}), "linear")
        self.assertEqual(saved.info["table"], "lossdata84_shuffled_only.npz")
        self.assertEqual(A.table_from_info(saved.info), "lossdata84_shuffled_only.npz")

    def test_no_arm_is_in_both_groups(self):
        self.assertFalse(set(TR.OLDER_TABLE_ARMS) & set(TR.POSITION_ARMS))


@unittest.skipUnless(HERE, "needs /tmp/vgc-pilot data")
class MaskingRateArms(unittest.TestCase):
    """The two trainings of the masking-rate rows (docs/masking-rate-rows.md, issue #296): the same table, rows and recipe,
    and only the masking rate differs. train and load are replaced first, so no test starts a real training."""

    def fake_table(self):
        import numpy as np
        labels = np.array([0.5, np.nan, 0.25, 0.75, np.nan, 0.1])
        return dict(all_X=np.arange(6 * 84).reshape(6, 84), all_W=labels, all_val=np.array([False, False, False, True, True, False]),
                    sha256="abc", table="lossdata84_shuffled_only.npz")

    def run_arm(self, arm):
        with (mock.patch.object(TR, "train") as train, mock.patch.object(TR, "load", return_value=self.fake_table()) as load,
              mock.patch.object(TR, "check_rows"), redirect_stdout(io.StringIO())):
            TR.main([arm])
        return train, load

    def test_checkpoint_names_follow_the_pre_registration(self):
        self.assertEqual(TR.NAMES["shuffled_per_field"], "stat84_shuffled_noise_per_field_es3_cap80")
        self.assertEqual(TR.NAMES["shuffled_linear"], "stat84_shuffled_noise_linear_es3_cap80")

    def test_the_two_arms_differ_only_in_the_masking_rate(self):
        calls = {}
        for arm in TR.MASKING_RATE_ARMS:
            train, load = self.run_arm(arm)
            load.assert_called_once_with(TR.SHUFFLED_ONLY)
            train.assert_called_once()
            calls[arm] = train.call_args
        per_field, linear = calls["shuffled_per_field"], calls["shuffled_linear"]
        self.assertEqual(per_field.kwargs.pop("sched"), "16,4,4,1.5,1")
        self.assertEqual(linear.kwargs.pop("sched"), "1,1,1,1,1")
        self.assertEqual(per_field.kwargs, linear.kwargs)
        for mine, theirs in zip(per_field.args[1:], linear.args[1:], strict=True):
            self.assertEqual(repr(mine), repr(theirs))

    def test_the_recipe_is_the_default_models_with_threshold_zero_and_every_epoch_saved(self):
        train, _load = self.run_arm("shuffled_linear")
        self.assertEqual(train.call_args.kwargs, dict(epochs=80, patience=3, early_stopping_threshold=0.0, label_noise=1 / 48,
                                                      every_epoch=True, sched="1,1,1,1,1"))

    def test_only_labelled_teams_train_and_validate(self):
        train, _load = self.run_arm("shuffled_per_field")
        _arm, rows, labels, (val_rows, val_labels) = train.call_args.args
        self.assertEqual(labels.tolist(), [0.5, 0.25, 0.1])
        self.assertEqual(val_labels.tolist(), [0.75])
        self.assertEqual((len(rows), len(val_rows)), (3, 1))

    def test_train_hands_the_arms_masking_rate_to_the_model(self):
        """train() merges the arm's sched into the shared recipe; passing it twice would raise before any epoch."""
        import tempfile
        model = mock.MagicMock(); model.info = dict(best_epoch=1, stopped_epoch=1)
        with (tempfile.TemporaryDirectory() as tmp, mock.patch.object(TR, "CK", Path(tmp)), mock.patch.object(TR.X, "RES", Path(tmp)),
              mock.patch.object(TR.X, "setup", return_value=[None]), mock.patch.object(TR.T, "regulation_vocab", return_value=None),
              mock.patch.object(TR.M.Model84, "train_new", return_value=model) as train_new, redirect_stdout(io.StringIO())):
            table = self.fake_table()
            TR.train("shuffled_linear", table["all_X"][:3], table["all_W"][:3], (table["all_X"][3:4], table["all_W"][3:4]),
                     epochs=80, patience=3, early_stopping_threshold=0.0, label_noise=1 / 48, every_epoch=True, sched="1,1,1,1,1")
        sent = train_new.call_args.kwargs
        self.assertEqual((sent["sched"], sent["early_stopping_threshold"], sent["seed"], sent["batch"], sent["lr"]), ("1,1,1,1,1", 0.0, 0, 256, 4e-4))
        self.assertTrue(callable(sent["on_epoch"]))

    def test_under_the_linear_rate_the_share_hidden_at_t_is_t(self):
        import torch
        exponents = TR.M.sched84("1,1,1,1,1")
        self.assertTrue((exponents == 1).all())
        generator = torch.Generator().manual_seed(0)
        for t in (0.1, 0.5, 0.9):
            hidden = torch.rand(20000, len(exponents), generator=generator) < torch.tensor(t) ** exponents
            self.assertLess((hidden.float().mean(0) - t).abs().max().item(), 0.015)


@unittest.skipUnless(HERE, "needs /tmp/vgc-pilot data")
class AskGridTable(unittest.TestCase):
    """The ask grid compares a model with its own training teams, so every model row names its table."""

    def test_a_new_row_defaults_to_the_shuffled_only_table(self):
        self.assertEqual(A.DEFAULT_TABLE, "lossdata84_shuffled_only.npz")
        self.assertEqual(A.table_of(dict(checkpoint="anything")), "lossdata84_shuffled_only.npz")

    def test_models_trained_on_the_older_table_say_so(self):
        for name in ("gen84_all_ep21", "gen84_all_noise_ep21", "gen84_same", "gen84_noise_byepoch_ep80", "gen84_ddom_tau03", "ft_hindsight_relabel"):
            self.assertEqual(A.table_of(A.MODELS[name]), "lossdata84.npz", name)

    def test_a_checkpoint_named_by_hand_takes_the_table_its_run_recorded(self):
        self.assertEqual(A.table_from_info(dict(table="lossdata84_shuffled_only.npz")), "lossdata84_shuffled_only.npz")
        self.assertEqual(A.table_from_info({}), "lossdata84.npz")       # saved before run info carried a table
        self.assertEqual(A.table_from_info(None), "lossdata84.npz")

    def test_models_trained_on_the_shuffled_only_table_say_so(self):
        for name in ("pos_per_field_s0", "pos_no_slot"):
            self.assertEqual(A.table_of(A.MODELS[name]), "lossdata84_shuffled_only.npz", name)


if __name__ == "__main__":
    unittest.main()
