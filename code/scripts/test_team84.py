"""The 84-field team (issue #57) on real corpus teams: the 48 existing fields plus six Stat Points per Pokémon.
Needs this laptop's /tmp/vgc-pilot data and the v3 loop-teams checkpoint (for the Regulation M-B vocabulary);
skipped without them.

    /tmp/vgc-pilot/.venv/bin/python scripts/test_team84.py   (from code/)
"""
import unittest

from pathlib import Path

import repo_paths  # noqa: F401  (src/ first on sys.path)

VOCAB = None
if Path("/tmp/vgc-pilot/data/pokedex.json").exists():
    import team84 as T
    import asked_vs_got as X
    VOCAB = T.regulation_vocab()
    CORPUS = X.setup()[0]


@unittest.skipUnless(VOCAB is not None, "the /tmp/vgc-pilot runtime or the v3 checkpoint is not available")
class EightyFourFields(unittest.TestCase):
    def teams(self, n=200):
        out = []
        for team in CORPUS:
            try: out.append((team, T.encode(VOCAB, team)))
            except ValueError: continue
            if len(out) == n: return out
        return out

    def test_a_corpus_team_encodes_to_84_tokens_whose_first_48_are_the_old_encoding(self):
        team, row = self.teams(1)[0]
        self.assertEqual(len(row), 84)
        self.assertEqual(list(row[:48]), list(VOCAB.encode(team)))

    def test_every_pokemon_keeps_its_own_stat_points(self):
        for team, row in self.teams():
            slots = T.decode(VOCAB, row)
            got = sorted((s["species"], tuple(s["evs"][k] for k in T.SP.STATS)) for s in slots)
            want = sorted((T.norm(s["species"]), tuple(s["evs"].get(k, 0) for k in T.SP.STATS)) for s in team)
            self.assertEqual(got, want)


@unittest.skipUnless(VOCAB is not None, "the /tmp/vgc-pilot runtime or the v3 checkpoint is not available")
class TinyModelEndToEnd(unittest.TestCase):
    """A model trained for two passes on 256 corpus teams: quality does not matter here, the plumbing does."""

    @classmethod
    def setUpClass(cls):
        import diffusion84 as M
        from hps_generate import Validator
        rows = []
        for team in CORPUS:
            try: rows.append(T.encode(VOCAB, team))
            except ValueError: continue
            if len(rows) == 256: break
        cls.model = M.Model84.train_new(VOCAB, CORPUS, rows, labels=None, epochs=2, seed=0)
        cls.rows = cls.model.sample(16, ask=0.5, guidance=2.0, seed=0).cpu().numpy()
        cls.pastes = [cls.model.to_paste(r) for r in cls.rows]
        val = Validator()
        try: cls.errors = [val(p) for p, _ in cls.pastes]
        finally: val.close()

    def test_every_sampled_pokemon_has_legal_stat_points(self):
        for r in self.rows:
            for slot in T.decode(VOCAB, r):
                self.assertTrue(all(0 <= v <= 32 for v in slot["evs"].values()))
                self.assertLessEqual(sum(slot["evs"].values()), 66)

    def test_the_paste_carries_the_generated_stat_points(self):
        for r, (_, slots) in zip(self.rows, self.pastes):
            want = sorted(tuple(s["evs"][k] for k in T.SP.STATS) for s in T.decode(VOCAB, r))
            got = sorted(tuple(s["evs"].get(k, 0) for k in T.SP.STATS) for s in slots)
            self.assertEqual(got, want)

    def test_showdown_accepts_every_sampled_team(self):
        self.assertEqual([e for e in self.errors if e is not None], [])


@unittest.skipUnless(VOCAB is not None, "the /tmp/vgc-pilot runtime or the v3 checkpoint is not available")
class TrainingStoppingAndSaving(unittest.TestCase):
    """Issues #59 and #60 on tiny models: early stopping, the matchup-row member, the mixture, save and load."""

    @classmethod
    def setUpClass(cls):
        import numpy as np
        import diffusion84 as M
        cls.M, cls.np = M, np
        rows = []
        for team in CORPUS:
            try: rows.append(T.encode(VOCAB, team))
            except ValueError: continue
            if len(rows) == 320: break
        cls.rows, cls.val = np.stack(rows[:256]), np.stack(rows[256:])
        cls.val_w = np.full(len(cls.val), np.nan, np.float32)
        cls.log = []
        cls.stopped = M.Model84.train_new(VOCAB, CORPUS, cls.rows, epochs=6, seed=0, val=(cls.val, cls.val_w),
                                          patience=1, log=cls.log, d=64, nlayer=2, nhead=4)

    def test_validation_loss_is_the_same_on_every_call(self):
        f = lambda: self.M.val_loss(self.stopped.net, self.stopped.tables, self.val, self.val_w, self.M.sched84().to(self.M.DEV))
        self.assertEqual(f(), f())

    def test_early_stopping_keeps_the_best_epoch(self):
        info = self.stopped.info
        curve = [e["val_loss"] for e in self.log]
        self.assertEqual(info["best_epoch"], 1 + int(self.np.argmin(curve)))
        self.assertTrue(info["stopped_epoch"] == 6 or info["stopped_epoch"] - info["best_epoch"] == 1)
        restored = self.M.val_loss(self.stopped.net, self.stopped.tables, self.val, self.val_w, self.M.sched84().to(self.M.DEV))
        self.assertAlmostEqual(restored, info["best_val_loss"], places=5)

    def test_a_rising_validation_curve_stops_after_patience_and_restores_the_best_epoch(self):
        from unittest import mock
        curve = iter([3.0, 2.0, 2.5, 2.6, 1.0, 0.5])                   # best at pass 2; passes 3 and 4 fail to beat it
        best_weights = {}
        def on_best(model, pass_number):
            best_weights[pass_number] = {k: v.clone() for k, v in model.net.state_dict().items()}
        with mock.patch.object(self.M, "val_loss", lambda *a, **k: next(curve)):
            m = self.M.Model84.train_new(VOCAB, CORPUS, self.rows, epochs=6, seed=0, val=(self.val, self.val_w),
                                         patience=2, on_best=on_best, d=64, nlayer=2, nhead=4)
        self.assertEqual((m.info["best_epoch"], m.info["stopped_epoch"]), (2, 4))
        for k, v in m.net.state_dict().items():
            self.assertTrue(bool((v == best_weights[2][k]).all()), k)

    def test_a_fixed_stop_keeps_the_last_epoch_even_when_an_earlier_one_scored_better(self):
        from unittest import mock
        curve = iter([1.0, 2.0, 3.0])
        saved = {}
        def on_epoch(model, epoch):
            saved[epoch] = {k: v.clone() for k, v in model.net.state_dict().items()}
        with mock.patch.object(self.M, "val_loss", lambda *a, **k: next(curve)):
            m = self.M.Model84.train_new(VOCAB, CORPUS, self.rows, epochs=6, seed=0, val=(self.val, self.val_w),
                                         stop_after=3, on_epoch=on_epoch, d=64, nlayer=2, nhead=4)
        self.assertEqual(sorted(saved), [1, 2, 3])
        self.assertEqual(m.info["stopped_epoch"], 3)
        for k, v in m.net.state_dict().items():
            self.assertTrue(bool((v == saved[3][k]).all()), k)

    def test_a_saved_model_loads_and_samples_the_same_teams(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            self.stopped.save(f"{d}/m.pt")
            back = self.M.Model84.load(f"{d}/m.pt", corpus=CORPUS)
        a = self.stopped.sample(8, ask=0.5, guidance=2.0, seed=3).cpu().numpy()
        b = back.sample(8, ask=0.5, guidance=2.0, seed=3).cpu().numpy()
        self.assertTrue((a == b).all())
        self.assertEqual(back.info["best_epoch"], self.stopped.info["best_epoch"])

    def test_a_mixture_with_a_matchup_row_member_samples_legal_teams(self):
        np = self.np
        rng = np.random.default_rng(0)
        wm = rng.random((len(self.rows), 51)).astype(np.float32)
        wm[rng.random(wm.shape) < 0.5] = np.nan
        records = self.M.Model84.train_new(VOCAB, CORPUS, self.rows, wm, epochs=1, seed=1, mrow=50, d=64, nlayer=2, nhead=4)
        both = self.M.Model84.mixture([self.stopped, records])
        self.assertEqual(len(both.nets), 2)
        for r in both.sample(8, ask=0.5, guidance=2.0, seed=0).cpu().numpy():
            for slot in T.decode(VOCAB, r):
                self.assertLessEqual(sum(slot["evs"].values()), 66)


if __name__ == "__main__":
    unittest.main()
