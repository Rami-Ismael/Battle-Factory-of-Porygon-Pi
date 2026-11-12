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
