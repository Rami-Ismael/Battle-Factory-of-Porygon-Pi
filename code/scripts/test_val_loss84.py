"""The 84-field validation-loss function (issue #247), the one test seam: mean, standard error and teams scored.
Real corpus teams and a tiny model trained for one epoch. Needs this laptop's /tmp/vgc-pilot data and the v3
loop-teams checkpoint (for the Regulation M-B vocabulary); skipped without them.

    /tmp/vgc-pilot/.venv/bin/python scripts/test_val_loss84.py   (from code/)
"""
import unittest

from pathlib import Path

import repo_paths  # noqa: F401  (src/ first on sys.path)

VOCAB = None
if Path("/tmp/vgc-pilot/data/pokedex.json").exists():
    import numpy as np
    import torch
    import team84 as T
    import asked_vs_got as X
    VOCAB = T.regulation_vocab()
    CORPUS = X.setup()[0]


def todays_val_loss(M, net, tables, x, w, sched, reps=4, seed=12345, batch=512):
    """diffusion84.val_loss as it was before issue #247, kept here to prove the default estimator did not move."""
    g = torch.Generator().manual_seed(seed)
    xs, ws, sc = torch.as_tensor(np.asarray(x, np.int64)), torch.as_tensor(np.asarray(w, np.float32)), sched.cpu()
    N, tot = len(xs), 0.0
    with torch.no_grad():
        for r in range(reps):
            ps = torch.rand(N, M.NSLOT, generator=g).argsort(1)
            tt = ((r + torch.rand(N, generator=g)) / reps).clamp(min=1e-3)
            u = torch.rand(N, M.COLS, generator=g)
            for i in range(0, N, batch):
                sl = slice(i, i + batch)
                m = (u[sl] < tt[sl].view(-1, 1) ** sc.view(1, -1)).to(M.DEV)
                xb = M.shuffle_slots(xs[sl], ps[sl]).to(M.DEV)
                tot += float(M.elbo_terms(net, tables, xb, m, tt[sl].to(M.DEV), ws[sl].to(M.DEV), sched).sum())
    return tot / (N * reps)


@unittest.skipUnless(VOCAB is not None, "the /tmp/vgc-pilot runtime or the v3 checkpoint is not available")
class ValidationLoss(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import diffusion84 as M
        cls.M = M
        rows = []
        for team in CORPUS:
            try: rows.append(T.encode(VOCAB, team))
            except ValueError: continue
            if len(rows) == 352: break
        rows = np.stack(rows)
        model = M.Model84.train_new(VOCAB, CORPUS, rows[:256], epochs=1, seed=0, d=64, nlayer=2, nhead=4)
        cls.net, cls.tables, cls.sched = model.net, model.tables, M.sched84().to(M.DEV)
        held = rows[256:]
        cls.val = held[~model.tables.check(held[:, :M.STAT0])]            # no team the rule tables exclude
        cls.val_w = np.full(len(cls.val), np.nan, np.float32)

    def estimate(self, **kw):
        return self.M.val_loss_estimate(self.net, self.tables, self.val, self.val_w, self.sched, **kw)

    def test_the_validation_teams_are_clean_and_counted(self):
        self.assertGreater(len(self.val), 64)
        self.assertEqual(self.estimate().n, len(self.val))

    def test_the_same_seed_gives_the_same_mean_and_another_seed_a_different_one(self):
        self.assertEqual(self.estimate(seed=7).mean, self.estimate(seed=7).mean)
        self.assertNotEqual(self.estimate(seed=7).mean, self.estimate(seed=8).mean)

    def test_the_standard_error_is_positive_and_shrinks_with_more_draws(self):
        few, many = self.estimate(reps=1).se, self.estimate(reps=16).se
        self.assertGreater(few, 0.0)
        self.assertGreater(many, 0.0)
        self.assertLess(many, few)

    def test_the_default_estimator_matches_the_old_function_on_clean_teams(self):
        old = todays_val_loss(self.M, self.net, self.tables, self.val, self.val_w, self.sched)
        self.assertAlmostEqual(self.estimate().mean, old, delta=1e-6)
        self.assertAlmostEqual(self.M.val_loss(self.net, self.tables, self.val, self.val_w, self.sched), old, delta=1e-6)

    def test_every_estimator_option_scores_every_team(self):
        for kind in self.M.VAL_ESTIMATORS:
            terms = self.M.val_loss_terms(self.net, self.tables, self.val, self.val_w, self.sched, reps=4, estimator=kind)
            self.assertEqual(tuple(terms.shape), (4, len(self.val)), kind)
            self.assertTrue(bool(torch.isfinite(terms).all()), kind)
        with self.assertRaises(ValueError):
            self.estimate(estimator="nearest")

    def test_validation_drops_a_planted_team_the_tables_reject_and_keeps_the_rest(self):
        import lossdown as L
        bad = self.val[0].copy()
        col = L.COLS_BY_KEY["item"][0]                                 # the first Pokémon's item
        rejected = False
        for v in range(1, VOCAB.sizes["item"]):                         # first item the tables reject on this Pokémon
            bad[col] = v
            if bool(self.tables.check(bad[None, :self.M.STAT0])[0]): rejected = True; break
        self.assertTrue(rejected, "no item value made the team break the tables")
        rows = np.concatenate([self.val, bad[None]]); labels = np.full(len(rows), np.nan, np.float32)
        kept, kept_w, dropped = self.M.validation_rows(self.tables, rows, labels)
        self.assertEqual(dropped, 1)
        self.assertTrue((kept == self.val).all())
        self.assertEqual(len(kept_w), len(self.val))


if __name__ == "__main__":
    unittest.main()
