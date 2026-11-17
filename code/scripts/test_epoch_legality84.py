"""First-try legality at the end of every training epoch (issue #248): a tiny 84-field model trained two epochs on 96
labelled teams logs, per epoch, the illegal share of a small batch of teams checked by Showdown's validator. Then
No tables in the loss (issue #361): the switch that leaves the tables' mask out of the loss, the draw without the
tables, and the script that checks saved epoch checkpoints. Needs this laptop's /tmp/vgc-pilot data, lossdata84.npz and
the Showdown checkout; skipped without them.

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
    import tempfile
    import asked_vs_got as X
    import checkpoint_legality as CL
    import diffusion84 as M
    import epoch_legality as E
    import team84 as T
    if (X.RES / "lossdata84.npz").exists(): HERE = True

TINY = dict(d=32, nlayer=1, nhead=2, batch=32, epochs=2)
LEGALITY_FIELDS = {"illegal_share", "errors_per_team", "legality_teams", "rejection_reasons", "legality_seconds"}


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
            self.assertTrue(all(n <= rejected for _, n in e["rejection_reasons"]))   # each kind counts once per team
            self.assertGreaterEqual(e["errors_per_team"], e["illegal_share"])
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


NET = dict(d=32, nlayer=1, nhead=2)


@unittest.skipUnless(HERE, "needs /tmp/vgc-pilot and lossdata84.npz")
class NoTablesInTheLoss(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        EpochLegality.setUpClass.__func__(cls)
        torch.manual_seed(0)
        cls.untrained = M.Model84(cls.V, cls.corpus, [(M.Net84(cls.V, **NET).to(M.DEV), NET)])

    def terms(self, **kw):
        net, x = self.untrained.net, torch.as_tensor(self.args[0][:32], device=M.DEV)
        g = torch.Generator().manual_seed(1)
        t = torch.rand(32, generator=g).clamp(min=0.05).to(M.DEV); m = (torch.rand(32, M.COLS, generator=g).to(M.DEV) < t.view(-1, 1))
        w = torch.as_tensor(self.args[1][:32], dtype=torch.float32, device=M.DEV)
        with torch.no_grad(): return M.elbo_terms(net.eval(), self.untrained.tables, x, m, t, w, M.sched84().to(M.DEV), **kw)

    def test_the_default_loss_is_the_masked_one_and_the_unmasked_one_is_larger(self):
        default, masked, unmasked = self.terms(), self.terms(table_loss=True), self.terms(table_loss=False)
        self.assertTrue(torch.equal(default, masked))
        # The mask moves forbidden values' probability onto the allowed ones, the true value among them, so it only lowers the loss
        self.assertTrue(bool((unmasked >= masked - 1e-5).all()))
        self.assertGreater(float((unmasked - masked).mean()), 0.01)

    def test_without_the_tables_an_untrained_network_draws_illegal_teams(self):
        check = E.EpochLegality(16, tables=False)
        try: share = check(self.untrained)["illegal_share"]
        finally: check.close()
        self.assertGreater(share, 0.0)

    def test_a_rejection_splits_into_kinds_without_names_or_numbers(self):
        err = ("Milotic can't learn Heat Wave.; Floette has 71 total Stat Points, which is more than this format's limit of 66.; "
               "Feraligatr-Mega transforms in-battle with Feraligite, please fix its item.; Chandelure has multiple copies of Will-O-Wisp.")
        self.assertEqual([E.kind(e) for e in E.errors(err)],
                         ["can't learn ...", "has N total Stat Points, which is more than this format's limit of N",
                          "transforms in-battle with ..., please fix its item","has multiple copies of ..."])
        self.assertEqual(E.reason(err), "can't learn ...")

    def test_a_team_key_ignores_slot_order_and_move_order(self):
        row = np.array(self.args[0][0]); other = M.shuffle_slots(torch.as_tensor(row[None]), torch.tensor([[5, 4, 3, 2, 1, 0]]))[0].numpy()
        other[3:7] = other[3:7][::-1]
        self.assertEqual(E.team_key(row), E.team_key(other))
        changed = row.copy(); changed[T.COLS48] = changed[T.COLS48] % 33 + 1
        self.assertNotEqual(E.team_key(row), E.team_key(changed))

    def test_the_copied_share_counts_drawn_teams_found_among_known_teams(self):
        drawn = self.untrained.sample(4, ask=E.ASK, guidance=E.GUIDANCE, seed=E.SEED, tables=False).cpu().numpy()
        for known, share in ((E.team_keys(drawn), 1.0), (E.team_keys(self.args[0]), 0.0)):
            check = E.EpochLegality(4, tables=False, known=known)
            try: self.assertEqual(check(self.untrained)["copied_share"], share)
            finally: check.close()
        m = M.Model84.train_new(self.V, self.corpus, *self.args, legality_teams=4, legality_tables=False, legality_copies=True, **TINY)
        self.assertTrue(all("copied_share" in e for e in m.info["curves"]))

    def test_the_lookahead_needs_the_tables(self):
        with self.assertRaises(ValueError): self.untrained.sample(2, ask=0.5, tables=False, lookahead=True)

    def test_a_two_epoch_training_without_the_loss_mask_logs_its_illegal_share_every_epoch(self):
        m = M.Model84.train_new(self.V, self.corpus, *self.args, val=self.val, legality_teams=8, table_loss=False,
                                legality_tables=False, **TINY)
        self.assertEqual([e["epoch"] for e in m.info["curves"]], [1, 2])
        self.assertTrue(all(0.0 <= e["illegal_share"] <= 1.0 and e["legality_teams"] == 8 for e in m.info["curves"]))
        self.assertIs(m.info["table_loss"], False); self.assertIs(m.info["legality_tables"], False)
        default = M.Model84.train_new(self.V, self.corpus, *self.args, val=self.val, **TINY)
        self.assertNotIn("table_loss", default.info); self.assertNotIn("legality_tables", default.info)
        self.assertGreater(m.info["curves"][0]["val_loss"], default.info["curves"][0]["val_loss"])

    def test_the_checkpoint_script_checks_every_epoch_once_and_resumes(self):
        with tempfile.TemporaryDirectory() as root:
            folder = Path(root) / "tiny"; folder.mkdir()
            for epoch in (1, 2): self.untrained.save(folder / f"tiny_ep{epoch:02d}.pt")
            out = Path(root) / "out.json"
            first = CL.run("tiny", teams=4, root=root, out=out, epochs={1})
            self.assertEqual([e["epoch"] for e in first["epochs"]], [1])
            both = CL.run("tiny", teams=4, root=root, out=out)
            self.assertEqual([e["epoch"] for e in both["epochs"]], [1, 2])
            self.assertEqual(both["epochs"][0], first["epochs"][0])
            self.assertEqual(both["epochs"][0]["illegal_share"], both["epochs"][1]["illegal_share"])   # same weights, same seed
            with self.assertRaises(SystemExit): CL.run("tiny", teams=8, root=root, out=out)


if __name__ == "__main__":
    unittest.main()
