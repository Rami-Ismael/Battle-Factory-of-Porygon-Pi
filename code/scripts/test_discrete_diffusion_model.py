"""The discrete diffusion model trained from scratch, against the code it replaces: foundation.Gen("v3") for scoring, sampling and
pastes (bit-for-bit, same device and seeds, except that the model never samples a Pokémon with no
moves, which Showdown rejects and Gen writes in about 2% of teams), cem_v3.members and cem_dbbo.members for fine-tuning.
Training on the Mac's GPU is not bit-deterministic: two identical runs of cem_v3.members differ by about
1e-3 in log-probability after one epoch. So a fine-tune must start bit-identical (zero budget) and then
land within three times the old code's own run-to-run spread.
Needs the v3 member checkpoints; runs in about three minutes on the M4 Pro.

    /tmp/vgc-pilot/.venv/bin/python -m unittest discover -s scripts -p 'test_discrete_diffusion_model.py'
"""
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

import repo_paths  # (src/ first on sys.path)
import discrete_diffusion_model  # before foundation: a module it imports puts /tmp/vgc-pilot/src first on sys.path
from discrete_diffusion_model import DiscreteDiffusionModel, _lowest_confidence
import foundation as FD
import lossdown as L
from encode import NF

TAG = f"test_discrete_diffusion_model_{os.getpid()}"


def corpus_fixture(model, n=96):
    """n encodable corpus teams, a label for each, and a half-masked batch (inputs, t, labels) to score."""
    rows = []
    for team in model.corpus:
        try: rows.append(model.V.encode(team))
        except ValueError: continue
        if len(rows) == n: break
    rows = np.stack(rows)
    labels = np.random.default_rng(0).uniform(0.3, 0.7, len(rows)).astype(np.float32)
    g = torch.Generator().manual_seed(1)
    x = torch.as_tensor(rows, device=L.DEV)
    batch = (x.masked_fill((torch.rand(x.shape, generator=g) < 0.5).to(L.DEV), 0),
             torch.full((len(x),), 0.5, device=L.DEV), torch.as_tensor(labels, device=L.DEV))
    return rows, labels, batch


class DiscreteDiffusionModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gen = FD.Gen("v3")
        cls.model = DiscreteDiffusionModel.load()
        cls.rows, cls.labels, (cls.xin, cls.t, cls.w) = corpus_fixture(cls.model)
        cls.tmp = tempfile.TemporaryDirectory()

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()
        for f in L.CK.glob(f"{TAG}*"): f.unlink()

    def assert_same_logprobs(self, f_new, f_old):
        a, b = f_new(self.xin, self.t, self.w), f_old(self.xin, self.t, self.w)
        self.assertEqual(set(a), set(b))
        for k in a: self.assertTrue(torch.equal(a[k], b[k]), k)

    def gap(self, f1, f2):
        a, b = f1(self.xin, self.t, self.w), f2(self.xin, self.t, self.w)
        return max(float((a[k] - b[k]).abs().max()) for k in a)

    def assert_within_old_spread(self, new, old_twice):
        spread = self.gap(*old_twice)
        self.assertLessEqual(self.gap(new, old_twice[0]), max(3 * spread, 1e-4), f"run-to-run spread of the replaced fine-tune: {spread:.2e}")

    def test_logprobs_match_gen(self):
        self.assert_same_logprobs(self.model.logprobs, self.gen.f)

    def test_sample_matches_gen(self):
        sur = FD.Surrogate(self.model.V)
        settings = [dict(args=(8,), kw={}),
                    dict(args=(8, 0.5, 2.0, 0.9, 0.95), kw=dict(g_species=1.5, cads=0.2)),
                    dict(args=(8, 0.5, 2.0, 1.0, 1.0, sur, sur.lam), kw={})]
        # The model forbids a slot with no moves (Showdown rejects it); Gen("v3") allows one. Each row draws
        # its own tokens, so every team must match Gen's except a team where Gen left a slot moveless.
        moveless = lambda r: any(int(r[NF * i + 3]) == self.model.tables.EMV for i in range(6))
        for i, s in enumerate(settings):
            kw = dict(s["kw"]); torch.manual_seed(100 + i); old = self.gen.sample(*s["args"], **kw)
            if "g_species" in kw: kw["species_guidance"] = kw.pop("g_species")
            torch.manual_seed(100 + i); new = self.model.sample(*s["args"], **kw)
            for a, b in zip(old, new):
                self.assertFalse(moveless(b), s)
                if not torch.equal(a, b): self.assertTrue(moveless(a), s)
        self.assertIsNone(L.COND_NOISE)

    def test_to_paste_matches_gen(self):
        for repair in (False, True):
            for r in self.rows[:8]:
                a = self.gen.to_paste(r, np.random.default_rng(3), repair)
                b = self.model.to_paste(r, np.random.default_rng(3), repair)
                self.assertEqual(a, b)

    def test_epoch_fine_tune_matches_cem_v3_members(self):
        import cem_v3 as CV
        saved = CV.TAG, CV.FT_EPOCHS
        elites = [(r, float(y), "test") for r, y in zip(self.rows, self.labels)]
        old = {}
        try:
            for epochs, run in [(0, "z"), (2, "a"), (2, "b")]:
                CV.TAG, CV.FT_EPOCHS = f"{TAG}_cv{run}", epochs
                old[run] = CV.members(self.model.V, self.model.tables, elites, 0)
        finally:
            CV.TAG, CV.FT_EPOCHS = saved
        tune = lambda epochs: self.model.fine_tuned(self.rows, self.labels, epochs=epochs, batch=CV.FT_BATCH,
                                                    lr=CV.FT_LR, seed=CV.SEED, cache_dir=self.tmp.name)
        self.assert_same_logprobs(tune(0).logprobs, old["z"])
        self.assert_within_old_spread(tune(2).logprobs, (old["a"], old["b"]))

    def test_mixed_step_fine_tune_matches_cem_dbbo_members(self):
        import cem_dbbo as CD
        saved = CD.TAG, CD.FT_STEPS
        lab = [(r, float(y)) for r, y in zip(self.rows, self.labels)]
        old = {}
        try:
            for run in "ab":                                   # cem_dbbo divides by FT_STEPS: no zero-budget run
                CD.TAG, CD.FT_STEPS = f"{TAG}_cd{run}", 3
                old[run] = CD.members(self.model.V, self.model.tables, lab, 0)
        finally:
            CD.TAG, CD.FT_STEPS = saved
        new = self.model.fine_tuned(self.rows, self.labels, steps=3, batch=CD.FT_BATCH, lr=CD.FT_LR,
                                     mix_original=True, seed=CD.SEED, cache_dir=self.tmp.name)
        self.assert_within_old_spread(new.logprobs, (old["a"], old["b"]))

    def test_cache_key_covers_every_setting(self):
        d = Path(self.tmp.name) / "key"
        a = self.model.fine_tuned(self.rows[:64], self.labels[:64], epochs=1, seed=5, cache_dir=d)
        self.assertEqual(len(list(d.glob("*.pt"))), 2)
        b = self.model.fine_tuned(self.rows[:64], self.labels[:64], epochs=1, seed=5, cache_dir=d)
        self.assertEqual(len(list(d.glob("*.pt"))), 2)
        self.assert_same_logprobs(a.logprobs, b.logprobs)
        self.model.fine_tuned(self.rows[:64], self.labels[:64], epochs=2, seed=5, cache_dir=d)
        self.model.fine_tuned(self.rows[:64], self.labels[:64] + 0.01, epochs=1, seed=5, cache_dir=d)
        self.assertEqual(len(list(d.glob("*.pt"))), 6)
        self.assertNotEqual(a.key, self.model.key)

    def test_fine_tune_leaves_the_model_unchanged(self):
        before = self.model.logprobs(self.xin, self.t, self.w)
        self.model.fine_tuned(self.rows[:64], self.labels[:64], steps=2, seed=9, cache_dir=Path(self.tmp.name) / "keep")
        after = self.model.logprobs(self.xin, self.t, self.w)
        for k in before: self.assertTrue(torch.equal(before[k], after[k]), k)

    def test_bad_budget_is_refused(self):
        with self.assertRaises(ValueError): self.model.fine_tuned(self.rows, self.labels, cache_dir=self.tmp.name)
        with self.assertRaises(ValueError): self.model.fine_tuned(self.rows, self.labels, epochs=1, mix_original=True,
                                                                   cache_dir=self.tmp.name)



class DiscreteDiffusionModelSpec(unittest.TestCase):
    """What the model promises through its interface, checked against independent truths (the Showdown
    validator, the paste parser, the training loss). These outlive the equivalence tests above."""

    @classmethod
    def setUpClass(cls):
        cls.model = DiscreteDiffusionModel.load()
        cls.rows, cls.labels, cls.batch = corpus_fixture(cls.model)
        cls.tmp = tempfile.TemporaryDirectory(); cls.dir = Path(cls.tmp.name)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_zero_budget_keeps_the_parent_weights(self):
        for budget in (dict(epochs=0), dict(steps=0)):
            child = self.model.fine_tuned(self.rows, self.labels, seed=1, cache_dir=self.dir / "zero", **budget)
            parent, kept = self.model.logprobs(*self.batch), child.logprobs(*self.batch)
            for k in parent: self.assertTrue(torch.equal(parent[k], kept[k]), (budget, k))

    def test_cache_hit_returns_without_training(self):
        d = self.dir / "hit"
        first = self.model.fine_tuned(self.rows, self.labels, epochs=1, seed=2, cache_dir=d)
        stamps = {f.name: f.stat().st_mtime_ns for f in d.glob("*.pt")}
        again = self.model.fine_tuned(self.rows, self.labels, epochs=1, seed=2, cache_dir=d)
        self.assertEqual({f.name: f.stat().st_mtime_ns for f in d.glob("*.pt")}, stamps)
        a, b = first.logprobs(*self.batch), again.logprobs(*self.batch)
        for k in a: self.assertTrue(torch.equal(a[k], b[k]), k)
        self.assertEqual(first.key, again.key)

    def test_fine_tuning_lowers_the_loss_on_its_rows(self):
        loss = lambda w: L.evaluate(w.logprobs, self.rows, self.labels, reps=4)[0]["elbo"]
        before = loss(self.model)
        for budget in (dict(epochs=10), dict(steps=20), dict(steps=20, mix_original=True)):
            after = loss(self.model.fine_tuned(self.rows, self.labels, seed=3, cache_dir=self.dir / "lower", **budget))
            self.assertLess(after, before, budget)

    def test_a_paste_parses_back_to_the_sampled_fields(self):
        from corpus import parse_team_text
        # A team is six slots in no particular order; the encoder sorts them, the sampler does not.
        slots = lambda r, keep: sorted(tuple(int(v) for v in r[8 * i:8 * i + 8][keep]) for i in range(6))
        torch.manual_seed(4); rows = self.model.sample(200, 0.5, 2.0).cpu().numpy()
        rng = np.random.default_rng(4)
        for row in rows:
            for repair in (False, True):
                text, _ = self.model.to_paste(row, rng, repair)
                parsed = parse_team_text(text, blank_ends_slot=True)
                self.assertEqual(len(parsed), 6, text)
                keep = np.ones(8, bool)
                if repair: keep[7] = False                           # repair may change the nature, nothing else
                self.assertEqual(slots(self.model.V.encode(parsed), keep), slots(row, keep), text)

    def test_every_sampled_team_passes_the_showdown_validator(self):
        from showdown_validator import Validator
        rng = np.random.default_rng(5); failures = []
        v = Validator()
        try:
            for i, settings in enumerate([(64,), (64, 0.5, 2.0)]):
                torch.manual_seed(500 + i)
                for row in self.model.sample(*settings).cpu().numpy():
                    text, _ = self.model.to_paste(row, rng)
                    err = v(text)
                    if err: failures.append((settings, err[:160]))
        finally:
            v.close()
        self.assertEqual(failures, [], f"{len(failures)} of 128 sampled teams rejected")

    # Remasking (docs/remask-ask-gap.md).
    def test_remasking_off_reproduces_the_plain_sampler(self):
        for i, (args, kw) in enumerate([((8,), {}), ((8, 0.5, 2.0), {}),
                                        ((8, 0.5, 2.0, 0.9, 0.95), dict(species_guidance=1.5, cads=0.2))]):
            torch.manual_seed(700 + i); plain = self.model.sample(*args, **kw)
            torch.manual_seed(700 + i); stats = {}; same = self.model.sample(*args, **kw, remask=0.0, steps=48, stats=stats)
            self.assertTrue(torch.equal(plain, same), (args, kw))
            self.assertEqual(stats, dict(passes=48, reverted=0, remasked={}, changed={}))
        self.assertIsNone(L.COND_NOISE)

    def assert_every_column_legal(self, x, msg):
        """Each column's value is allowed given the other 47, by the tables themselves (no fallback)."""
        for c in range(x.shape[1]):
            hidden = x.clone(); hidden[:, c] = 0
            legal, empty = self.model._legal(hidden, c)
            ok = legal.gather(1, x[:, c:c + 1]).squeeze(1) & ~empty
            self.assertTrue(bool(ok.all()), (msg, c, int((~ok).sum())))

    def test_remasked_teams_satisfy_every_legality_table(self):
        for remask, steps in ((0.2, 48), (0.1, 96)):
            torch.manual_seed(710); stats = {}
            x = self.model.sample(64, 0.5, 2.0, remask=remask, steps=steps, stats=stats)
            self.assertGreater(sum(stats["remasked"].values()), 0)
            self.assert_every_column_legal(x, (remask, steps))

    def test_a_remasking_dead_end_is_undone(self):
        # GPT-6-Astra's case (2026-10-30 review): slot 1 Arcanine (Intimidate; Flamethrower, Protect, Rest, Roar) and
        # slot 2 Incineroar (Blaze; Darkest Lariat, Fake Out, Parting Shot, Protect), both species remasked in one
        # pass. Slot 1 may also take Incineroar, and then Species Clause leaves slot 2 no legal species.
        V, n, wr = self.model.V, 256, self.model
        bad = {V.stoi["species"][s] for s in ("arcanine", "arcaninehisui", "incineroar")}
        torch.manual_seed(750)
        row = next(r for r in wr.sample(64, 0.5, 2.0) if not any(int(r[s * NF]) in bad for s in range(2, 6)))
        x = row.repeat(n, 1)
        for s, (species, ability, moves) in enumerate([("arcanine", "intimidate", ["flamethrower", "protect", "rest", "roar"]),
                                                      ("incineroar", "blaze", ["darkestlariat", "fakeout", "partingshot", "protect"])]):
            x[:, s * NF] = V.stoi["species"][species]; x[:, s * NF + 1] = V.stoi["ability"][ability]
            for j, m in enumerate(sorted(V.stoi["move"][m] for m in moves)): x[:, s * NF + 3 + j] = m
            x[:, s * NF + 2] = 0; x[:, s * NF + 2] = int(wr.legal_values(x[:1], s * NF + 2)[0].nonzero()[-1])
        self.assert_every_column_legal(x, "the starting team")
        due = 5 * NF + 7                                      # the last slot's nature, still masked: this pass's new column
        conf = torch.full(x.shape, 0.9, device=L.DEV); x[:, due] = 0; conf[:, due] = float("inf")
        old, old_conf = x.clone(), conf.clone()               # the legal partial team before the pass
        redo = torch.zeros_like(x, dtype=torch.bool); redo[:, [0, NF]] = True
        x[redo] = 0; conf[redo] = float("inf")
        real, seen = wr.logprobs, []
        def uniform(xin, t, w):                                        # uniform draws; records each network input
            seen.append(xin.clone()); return {k: torch.zeros_like(v) for k, v in real(xin, t, w).items()}
        wr.logprobs = uniform
        try:
            torch.manual_seed(751)
            due_mask = torch.zeros_like(x, dtype=torch.bool); due_mask[:, due] = True
            reverted = wr._fill_pass(x, conf, [0, NF, due], due_mask, redo, torch.full((n,), 0.5, device=L.DEV), None,
                                     dict(guidance=1.0, species_guidance=None, temperature=1.0, top_p=1.0, surrogate=None, lam=0.0),
                                     old, old_conf)
        finally:
            del wr.logprobs
        self.assertGreater(reverted, 0)
        self.assertLess(reverted, n)
        self.assertTrue(bool((x[:, due] != 0).all()), "every team, reverted or not, draws the pass's new column")
        self.assert_every_column_legal(x, "after the pass")
        back = (conf[:, [0, NF]] == 0.9).all(1)               # uniform redraws never have probability 0.9
        self.assertEqual(int(back.sum()), reverted)
        self.assertTrue(bool((x[back][:, [0, NF]] == old[back][:, [0, NF]]).all()), "reverted teams keep their old species")
        self.assertTrue(bool((x[~back][:, [0, NF]] != 0).all()))
        restored = old[back].clone(); restored[:, due] = 0
        self.assertTrue(any(s.shape == restored.shape and bool((s == restored).all()) for s in seen),
                        "reverted teams redraw from predictions on the restored team")

    def test_confidence_order_with_remasking_fills_every_column(self):
        # a remasked column is masked again but not new; it must not use up a new-column slot
        for seed in (760, 761, 762):
            torch.manual_seed(seed); x = self.model.sample(16, 0.5, 2.0, order="confidence", remask=0.2)
            self.assertTrue(bool((x != 0).all()))

    def test_remasking_keeps_the_number_of_network_passes(self):
        calls, real = [0], self.model.logprobs
        def counting(*a):
            calls[0] += 1; return real(*a)
        self.model.logprobs = counting
        try:
            def passes(**kw):
                calls[0] = 0; torch.manual_seed(720); self.model._sample_remasking(16, 0.5, 2.0, 1.0, 1.0, None, 0.0, None, None, stats={}, **kw)
                return calls[0]
            fixed, remasked, doubled = passes(remask=0.0, steps=48), passes(remask=0.2, steps=48), passes(remask=0.1, steps=96)
        finally:
            del self.model.logprobs
        self.assertEqual(fixed, 2 * 48)                              # one conditioned + one unconditioned pass per step
        self.assertEqual(remasked, fixed)
        self.assertLessEqual(doubled, 2 * 96)

    def test_remasking_redraws_and_changes_some_values(self):
        torch.manual_seed(730); plain = self.model.sample(64, 0.5, 2.0)
        torch.manual_seed(730); stats = {}; remasked = self.model.sample(64, 0.5, 2.0, remask=0.2, stats=stats)
        self.assertFalse(torch.equal(plain, remasked))
        self.assertGreater(sum(stats["changed"].values()), 0)
        self.assertLessEqual(sum(stats["changed"].values()), sum(stats["remasked"].values()))

    def test_remasked_teams_pass_the_showdown_validator(self):
        from showdown_validator import Validator
        rng = np.random.default_rng(6); failures = []
        v = Validator()
        try:
            for i, (remask, steps) in enumerate(((0.2, 48), (0.1, 96))):
                torch.manual_seed(740 + i)
                for row in self.model.sample(64, 0.5, 2.0, remask=remask, steps=steps).cpu().numpy():
                    text, _ = self.model.to_paste(row, rng)
                    err = v(text)
                    if err: failures.append((remask, steps, err[:160]))
        finally:
            v.close()
        self.assertEqual(failures, [], f"{len(failures)} of 128 remasked teams rejected")

    def test_bad_remasking_settings_are_refused(self):
        for kw in (dict(remask=0.1, steps=0), dict(remask=1.0), dict(remask=-0.1), dict(order="alphabetical")):
            with self.assertRaises(ValueError, msg=kw): self.model.sample(2, 0.5, 2.0, **kw)

    # Decoding orders (docs/decoding-order.md).
    def trace_orders(self, order, steps=48, n=16):
        """Run the sampler with order, recording each pass's due mask; returns (teams, list of [n, 48] masks)."""
        wr, real, seen = self.model, self.model._fill_pass, []
        def spy(x, conf, targets, due, *a):
            self.assertFalse(bool((due & (x != 0)).any()), "a due column must still be masked")
            seen.append(due.clone()); return real(x, conf, targets, due, *a)
        wr._fill_pass = spy
        try:
            torch.manual_seed(760); x = wr.sample(n, 0.5, 2.0, order=order, steps=steps, stats={})   # stats: the scheduled path even for species-first
        finally:
            del wr._fill_pass
        return x, seen

    def test_every_order_fills_every_field_once(self):
        for order, steps in (("species", 48), ("pokemon", 48), ("random", 48), ("random", 24), ("confidence", 48)):
            x, seen = self.trace_orders(order, steps)
            self.assertTrue(bool((x != 0).all()), order)
            total = torch.stack(seen).sum(0)
            self.assertTrue(bool((total == 1).all()), (order, steps, "each field due exactly once per team"))
            self.assertEqual(len(seen), steps, (order, steps))
            self.assertTrue(all(int(d.sum(1).max()) == 48 // steps for d in seen), (order, steps, "fields per pass"))

    def test_fixed_orders_follow_their_sequence(self):
        from discrete_diffusion_model import POKEMON_ORDER
        import diffusion as Dm
        for order, seq in (("species", Dm.ORDER), ("pokemon", POKEMON_ORDER)):
            _, seen = self.trace_orders(order)
            self.assertEqual([int(d[0].nonzero()[0]) for d in seen], list(seq), order)
            self.assertTrue(all(bool((d == d[0]).all()) for d in seen), (order, "every team shares the order"))

    def test_random_order_differs_between_teams(self):
        _, seen = self.trace_orders("random")
        first = [d.float().argmax(1) for d in seen]                    # the column each team filled at each pass
        orders = torch.stack(first, 1)
        self.assertGreater(len({tuple(r.tolist()) for r in orders}), 1)

    def test_confidence_order_takes_the_most_peaked_field(self):
        wr, real = self.model, self.model.logprobs
        def peaked(xin, t, w):                                         # flat everywhere except one value of the last field
            out = {k: torch.zeros_like(v) for k, v in real(xin, t, w).items()}
            key, i = L.KEY_OF_COL[47], L.COLS_BY_KEY[L.KEY_OF_COL[47]].index(47)
            out[key][:, i, 3] = 50.0
            return out
        wr.logprobs = peaked
        try:
            x0 = torch.zeros(8, L.COLS, dtype=torch.long, device=L.DEV)
            due, _ = wr._most_confident(x0, torch.full((8,), 0.5, device=L.DEV), None,
                                        dict(guidance=1.0, species_guidance=None, temperature=1.0, top_p=1.0, surrogate=None, lam=0.0), 1)
        finally:
            del wr.logprobs
        self.assertTrue(bool(due[:, 47].all()) and int(due.sum()) == 8)

    def test_lookahead_keeps_species_first_unchanged(self):
        # species-first order draws every species before its set, so the lookahead never removes a value
        torch.manual_seed(740); plain = self.model.sample(32, 0.5, 2.0)
        torch.manual_seed(740); ahead = self.model.sample(32, 0.5, 2.0, lookahead=True)
        self.assertTrue(torch.equal(plain, ahead))

    def species_dead_ends(self, lookahead):
        wr, real, hits = self.model, self.model._legal, [0]
        def counting(x, c, lookahead=False):
            o, empty = real(x, c, lookahead)
            if c % NF == 0: hits[0] += int(empty.sum())
            return o, empty
        wr._legal = counting
        try:
            torch.manual_seed(741); wr.sample(64, 0.5, 2.0, order="random", lookahead=lookahead)
        finally:
            del wr._legal
        return hits[0]

    def test_lookahead_removes_most_species_dead_ends_in_random_order(self):
        without, ahead = self.species_dead_ends(False), self.species_dead_ends(True)
        self.assertGreater(without, 50)
        self.assertLess(ahead, without / 5)


class LowestConfidenceSpec(unittest.TestCase):
    def test_the_checkouts_model_is_under_test(self):
        self.assertEqual(Path(discrete_diffusion_model.__file__).resolve().parent, (repo_paths.ROOT / "src").resolve())

    def test_picks_the_least_likely_filled_columns(self):
        inf = float("inf")
        conf = torch.tensor([[0.7, 0.2, 0.9, 0.4] + [inf] * 44, [inf] * 48])
        torch.manual_seed(0); picked = _lowest_confidence(conf, 0.5)             # 0.5 x 4 filled = exactly 2
        self.assertEqual(picked[0].nonzero().flatten().tolist(), [1, 3])
        self.assertFalse(bool(picked[1].any()))

    def test_stochastic_rounding_keeps_the_mean_fraction(self):
        conf = torch.rand(4000, 48); conf[:, 10:] = float("inf")              # 10 filled columns per team
        torch.manual_seed(1); k = _lowest_confidence(conf, 0.25).sum(1).float()
        self.assertTrue(set(k.unique().tolist()) <= {2.0, 3.0})
        self.assertAlmostEqual(float(k.mean()), 2.5, delta=0.05)


if __name__ == "__main__":
    unittest.main()
