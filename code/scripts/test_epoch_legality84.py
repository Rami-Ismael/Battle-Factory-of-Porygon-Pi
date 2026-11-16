"""First-try legality at the end of every training epoch (issue #248): a tiny 84-field model trained two epochs on 96
labelled teams logs, per epoch, the illegal share of a small batch of teams checked by Showdown's validator. Needs this
laptop's /tmp/vgc-pilot data, lossdata84.npz and the Showdown checkout; skipped without them.

    /tmp/vgc-pilot/.venv/bin/python scripts/test_epoch_legality84.py   (from code/)
"""
import sys
import unittest
from pathlib import Path

import repo_paths  # noqa: F401  (src/ first on sys.path)

HERE = None
if Path("/tmp/vgc-pilot/data/pokedex.json").exists():
    sys.path.insert(1, "/tmp/vgc-pilot/src")
    import numpy as np
    import torch
    import asked_vs_got as X
    import diffusion84 as M
    import team84 as T
    if (X.RES / "lossdata84.npz").exists(): HERE = True

TINY = dict(d=32, nlayer=1, nhead=2, batch=32, epochs=2)
LEGALITY_FIELDS = {"illegal_share", "legality_teams", "rejection_reasons", "legality_seconds"}


@unittest.skipUnless(HERE, "needs /tmp/vgc-pilot and lossdata84.npz")
class EpochLegality(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        d = np.load(X.RES / "lossdata84.npz"); has = ~np.isnan(d["all_W"])
        x, w = d["all_X"][has][:128].astype(np.int64), d["all_W"][has][:128]
        cls.V, cls.corpus = T.regulation_vocab(), X.setup()[0]
        cls.args = (x[:96], w[:96]); cls.val = (x[96:], w[96:])

    def train(self, teams):
        return M.Model84.train_new(self.V, self.corpus, *self.args, val=self.val, legality_teams=teams, **TINY)

    def test_every_epoch_logs_its_illegal_share_reasons_and_seconds(self):
        curves = self.train(8).info["curves"]
        self.assertEqual([e["epoch"] for e in curves], [1, 2])
        for e in curves:
            self.assertLessEqual(LEGALITY_FIELDS, set(e))
            self.assertEqual(e["legality_teams"], 8)
            self.assertTrue(0.0 <= e["illegal_share"] <= 1.0)
            self.assertGreater(e["legality_seconds"], 0.0)
            rejected = round(e["illegal_share"] * 8)
            self.assertLessEqual(sum(n for _, n in e["rejection_reasons"]), rejected)
            if rejected: self.assertTrue(e["rejection_reasons"])

    def test_with_zero_teams_training_logs_no_legality_and_the_check_does_not_change_training(self):
        off, on = self.train(0), self.train(8)
        self.assertFalse(any(LEGALITY_FIELDS & set(e) for e in off.info["curves"]))
        # Two runs with the check off already differ by about 1e-7 on this GPU; other batches or masks would move the loss by far more.
        for a, b in zip(off.info["curves"], on.info["curves"], strict=True):
            self.assertAlmostEqual(a["train_loss"], b["train_loss"], delta=1e-5)
            self.assertAlmostEqual(a["val_loss"], b["val_loss"], delta=1e-5)
        weights = lambda m: m.net.state_dict()
        self.assertTrue(all(torch.allclose(v, weights(on)[k], atol=1e-5) for k, v in weights(off).items()))

    def test_a_fine_tune_checks_its_starting_weights_too(self):
        base = M.Model84.train_new(self.V, self.corpus, *self.args, **TINY)
        curves = base.fine_tuned(*self.args, epochs=1, val=self.val, batch=32, legality_teams=4).info["curves"]
        self.assertEqual([e["epoch"] for e in curves], [0, 1])
        self.assertTrue(all(e["legality_teams"] == 4 for e in curves))


if __name__ == "__main__":
    unittest.main()
