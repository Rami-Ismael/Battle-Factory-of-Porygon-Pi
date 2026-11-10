"""Model84.fine_tuned (src/diffusion84.py), the hindsight-relabeled fine-tune's training step: a tiny model trained one epoch on
96 labelled teams, then fine-tuned. Needs this laptop's /tmp/vgc-pilot data and lossdata84.npz; skipped without them.

    /tmp/vgc-pilot/.venv/bin/python scripts/test_fine_tuned84.py   (from code/)
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

TINY = dict(d=32, nlayer=1, nhead=2)


@unittest.skipUnless(HERE, "needs /tmp/vgc-pilot and lossdata84.npz")
class FineTuned(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        d = np.load(X.RES / "lossdata84.npz"); has = ~np.isnan(d["all_W"])
        x, w = d["all_X"][has][:160].astype(np.int64), d["all_W"][has][:160]
        cls.x, cls.w = x, w
        cls.base = M.Model84.train_new(T.regulation_vocab(), X.setup()[0], x[:96], w[:96], epochs=1, batch=32, **TINY)

    def weights(self, m):
        return {k: v.detach().cpu().clone() for k, v in m.net.state_dict().items()}

    def test_with_no_learning_the_starting_weights_are_the_best_epoch(self):
        before = self.weights(self.base)
        m = self.base.fine_tuned(self.x[96:128], self.w[96:128], epochs=5, val=(self.x[128:], self.w[128:]), lr=0.0, batch=16,
                                 watch=dict(labelled=(self.x[:32], self.w[:32])))
        self.assertEqual(m.info["best_epoch"], 0)
        self.assertEqual(m.info["stopped_epoch"], 3)                 # patience 3 after epoch 0
        self.assertEqual([e["epoch"] for e in m.info["curves"]], [0, 1, 2, 3])
        self.assertIn("labelled_val_loss", m.info["curves"][0])
        after = self.weights(m)
        self.assertTrue(all(torch.equal(before[k], after[k]) for k in before))

    def test_fine_tuning_leaves_the_base_model_unchanged(self):
        before = self.weights(self.base)
        m = self.base.fine_tuned(self.x[96:128], self.w[96:128], epochs=2, val=(self.x[128:], self.w[128:]), lr=1e-3, batch=16, patience=None)
        self.assertTrue(all(torch.equal(before[k], v) for k, v in self.weights(self.base).items()))
        self.assertFalse(all(torch.equal(before[k], v) for k, v in self.weights(m).items()))
        self.assertIs(m.V, self.base.V)
        self.assertEqual(m.info["base"], self.base.info)
        self.assertEqual(m.sample(2, ask=0.5, guidance=2.0, seed=0).shape, (2, M.COLS))


if __name__ == "__main__":
    unittest.main()
