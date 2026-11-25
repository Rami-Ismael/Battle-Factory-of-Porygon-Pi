"""Tests for issue #431: the two arms that train one shared GIDD-style learning-rate recipe.

Each new arm must call its trainer exactly as its control does, except for the learning-rate fields of
lr_schedule84.GIDD_LR_RECIPE, and must save under a new checkpoint name.
  Uniform, no filter    udlm_loglinear            -> udlm_loglinear_gidd_lr
  Masked, full filter   shuffled_linear_filtered  -> shuffled_linear_filtered_gidd_lr
Laptop-only (skipped without /tmp/vgc-pilot): the trainers import the pinned data at import time.

    /tmp/vgc-pilot/.venv/bin/python scripts/test_gidd_lr_arms.py   (from code/)
"""
import io
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

import numpy as np

import repo_paths  # noqa: F401  (src/ first on sys.path)
import lr_schedule84 as LS

HERE = Path("/tmp/vgc-pilot/data/pokedex.json").exists()
if HERE:
    sys.path.insert(1, "/tmp/vgc-pilot/src")
    import train84 as TR
    import train_uniform84 as TU

LR_FIELDS = set(LS.GIDD_LR_RECIPE)


def fake_table():
    return dict(all_X=np.arange(6 * 84).reshape(6, 84) % 5 + 1, all_W=np.array([0.5, np.nan, 0.25, 0.75, np.nan, 0.1]),
                all_val=np.array([False, False, False, True, True, False]), sha256="e58af7689e62",
                table="lossdata84_shuffled_only.npz")


@unittest.skipUnless(HERE, "needs this laptop's /tmp/vgc-pilot data")
class MaskedFullFilterArmTests(unittest.TestCase):
    def call_for(self, arm):
        with (mock.patch.object(TR, "train") as train, mock.patch.object(TR, "load", return_value=fake_table()),
              mock.patch.object(TR, "check_rows"),
              mock.patch.object(TR, "assert_no_wrong_stones",
                                return_value=dict(method="mock", train_rows=3, val_rows=1, train_rejected=0, val_rejected=0)),
              redirect_stdout(io.StringIO())):
            TR.main([arm])
        train.assert_called_once()
        return train.call_args

    def test_name_is_new_and_says_the_recipe(self):
        self.assertEqual(TR.NAMES["shuffled_linear_filtered_gidd_lr"], "stat84_shuffled_noise_linear_filtered_lr5e4_min5e5_es3_cap80")

    def test_only_the_learning_rate_fields_differ_from_the_control(self):
        control, new = self.call_for("shuffled_linear_filtered"), self.call_for("shuffled_linear_filtered_gidd_lr")
        self.assertEqual(new.kwargs, dict(control.kwargs, **LS.GIDD_LR_RECIPE))
        for a, b in zip(control.args[1:], new.args[1:], strict=True):
            self.assertEqual(repr(a), repr(b))

    def test_recipe_for_moves_the_learning_rate_fields_into_the_recipe(self):
        net_kw = dict(sched=TR.LINEAR_MASKING_RATE, label_noise=1 / 48, **LS.GIDD_LR_RECIPE)
        recipe = TR.recipe_for("shuffled_linear_filtered_gidd_lr", net_kw)
        self.assertEqual(recipe, dict(TR.RECIPE, **LS.GIDD_LR_RECIPE))
        self.assertEqual(net_kw, dict(label_noise=1 / 48))

    def test_old_arms_keep_their_recipe(self):
        self.assertEqual(TR.recipe_for("shuffled_linear_filtered", dict(sched=TR.LINEAR_MASKING_RATE)), TR.RECIPE)


@unittest.skipUnless(HERE, "needs this laptop's /tmp/vgc-pilot data")
class UniformNoFilterArmTests(unittest.TestCase):
    def call_for(self, arm):
        with (mock.patch.object(TU, "train") as train, mock.patch.object(TU, "load", return_value=fake_table()),
              mock.patch.object(TU, "check_rows")):
            TU.train_arm(arm)
        train.assert_called_once()
        return train.call_args

    def test_name_is_new_and_says_the_recipe(self):
        self.assertEqual(TU.NAMES["udlm_loglinear_gidd_lr"], "stat84_udlm_loglinear_lr5e4_min5e5_es3_cap80")

    def test_only_the_learning_rate_recipe_differs_from_the_control(self):
        control, new = self.call_for("udlm_loglinear"), self.call_for("udlm_loglinear_gidd_lr")
        self.assertEqual(new.kwargs, dict(control.kwargs, recipe=LS.GIDD_LR_RECIPE))
        for a, b in zip(control.args[1:], new.args[1:], strict=True):
            self.assertEqual(repr(a), repr(b))

    def test_recipe_for_overrides_only_the_learning_rate_fields(self):
        self.assertEqual(TU.recipe_for(LS.GIDD_LR_RECIPE), dict(TU.RECIPE, **LS.GIDD_LR_RECIPE))
        self.assertEqual(TU.recipe_for(None), TU.RECIPE)
        changed = {k for k in TU.recipe_for(LS.GIDD_LR_RECIPE) if TU.recipe_for(LS.GIDD_LR_RECIPE)[k] != TU.RECIPE.get(k)}
        self.assertEqual(changed, LR_FIELDS)


if __name__ == "__main__":
    unittest.main()
