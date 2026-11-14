"""The two looping options of the 84-field network (docs/looped-84-fields.md), both off by default.

    Loop block      loop_layers=2, loop_start=1, loop_max=6   layers 1 and 2 run up to 6 times with shared weights (LoopMDM)
    Carried state   carry=True                               each pass hands its hidden state to the next (Loopholing)

Needs this laptop's /tmp/vgc-pilot data and the shuffled-only table; skipped without them.

    /tmp/vgc-pilot/.venv/bin/python scripts/test_looped84.py   (from code/)
"""
import sys
import tempfile
import unittest
from pathlib import Path

import repo_paths  # noqa: F401  (src/ first on sys.path)

HERE = Path("/tmp/vgc-pilot/data/pokedex.json").exists()
if HERE:
    sys.path.insert(1, "/tmp/vgc-pilot/src")
    import numpy as np
    import torch
    import diffusion84 as M
    import lossdown as L
    import asked_vs_got as X
    import team84 as T
    import train84 as TR
    HERE = TR.SHUFFLED_ONLY.exists()

TINY = dict(d=32, nlayer=3, nhead=2)
LOOP = dict(loop_start=1, loop_layers=1, loop_max=3)

# Recorded 2026-11-12 on this laptop (device mps) from diffusion84.py before the looping options existed (main at 94678bdb,
# plus the masking-schedule arms 03e4ed4f): torch.manual_seed(0), Net84(V, **TINY).train(); torch.manual_seed(5), then
# batch_loss on the first 64 rows of the shuffled-only table, then one torch.rand(1); val_loss with reps=2 in eval mode.
OLD_BATCH_LOSS, OLD_NEXT_RANDOM, OLD_VAL_LOSS = 3.4148964881896973, 0.8302518725395203, 3.1996374130249023

if HERE:
    VOCAB = T.regulation_vocab()
    TABLES = L.Tables(VOCAB, X.setup()[0])
    _d = np.load(TR.SHUFFLED_ONLY)
    ROWS, LABELS = _d["all_X"][:64].astype(np.int64), _d["all_W"][:64].astype(np.float32)


def net(seed=0, **kw):
    torch.manual_seed(seed)
    return M.Net84(VOCAB, **TINY, **kw).to(M.DEV)


def batch():
    x = torch.as_tensor(ROWS).to(M.DEV)
    return x, torch.full((len(x),), 0.6, device=M.DEV), torch.as_tensor(LABELS).to(M.DEV)


def close(a, b, tol=1e-5):
    return float((a - b).abs().max()) < tol


@unittest.skipUnless(HERE, "needs /tmp/vgc-pilot and the shuffled-only table")
class PlainNetworkIsUnchanged(unittest.TestCase):
    def test_loss_random_stream_and_validation_loss_match_the_old_code(self):
        m = net().train(); x, _, w = batch()
        torch.manual_seed(5)
        self.assertAlmostEqual(float(M.batch_loss(m, TABLES, x, w, M.sched84().to(M.DEV))), OLD_BATCH_LOSS, places=4)
        self.assertAlmostEqual(float(torch.rand(1)), OLD_NEXT_RANDOM, places=6)      # no extra draw for the plain network
        m.eval()
        self.assertAlmostEqual(M.val_loss(m, TABLES, ROWS, LABELS, M.sched84().to(M.DEV), reps=2), OLD_VAL_LOSS, places=4)

    def test_options_change_no_starting_weight(self):
        plain = net().state_dict()
        looped = net(**LOOP).state_dict()
        carried = net(carry=True).state_dict()
        self.assertEqual(plain.keys(), looped.keys())
        self.assertEqual(set(carried) - set(plain), {"carry_ln.weight", "carry_ln.bias"})
        for k, v in plain.items():
            self.assertTrue(torch.equal(v, looped[k]), k); self.assertTrue(torch.equal(v, carried[k]), k)


@unittest.skipUnless(HERE, "needs /tmp/vgc-pilot and the shuffled-only table")
class LoopBlock(unittest.TestCase):
    def test_one_loop_is_the_plain_network(self):
        plain, looped = net().eval(), net(**LOOP).eval(); x, t, w = batch()
        with torch.no_grad(): self.assertTrue(close(looped(x, t, w, loops=1), plain(x, t, w)))

    def test_loops_repeat_the_block_with_shared_weights(self):
        m = net(**LOOP).eval(); x, t, w = batch()
        with torch.no_grad():
            got = m(x, t, w, loops=3)
            default = m(x, t, w)
            one = m(x, t, w, loops=1)
            layers = list(m.tr.layers)
            # unroll by hand: layer 0, layer 1 three times, layer 2
            ref = net(**LOOP).eval(); ref.load_state_dict(m.state_dict()); ref.loop_layers = 0
            ref.tr.layers = torch.nn.ModuleList([layers[0], layers[1], layers[1], layers[1], layers[2]])
            unrolled = ref(x, t, w)
        self.assertTrue(close(got, default))                  # a pass runs loop_max times unless told otherwise
        self.assertTrue(close(got, unrolled, 1e-4))
        self.assertFalse(close(got, one, 1e-3))

    def test_training_draws_a_count_from_one_to_the_maximum(self):
        m = net(**LOOP).train(); x, _, w = batch(); seen = set()
        original = m.forward
        m.forward = lambda *a, loops=None, state=None: (seen.add(loops), original(*a, loops=loops, state=state))[1]
        for s in range(40):
            torch.manual_seed(s); M.batch_loss(m, TABLES, x[:8], w[:8], M.sched84().to(M.DEV))
        self.assertEqual(seen, {1, 2, 3})

    def test_a_block_that_does_not_fit_is_refused(self):
        with self.assertRaises(ValueError): net(loop_start=2, loop_layers=2, loop_max=2)


@unittest.skipUnless(HERE, "needs /tmp/vgc-pilot and the shuffled-only table")
class CarriedState(unittest.TestCase):
    def test_no_state_is_a_zero_state_and_a_state_changes_the_output(self):
        m = net(carry=True).eval(); x, t, w = batch()
        with torch.no_grad():
            torch.nn.init.normal_(m.carry_ln.weight)          # the starting layer norm maps every state to its zero bias
            a, b = m(x, t, w), m(x, t, w, state=torch.zeros(len(x), M.COLS, TINY["d"], device=M.DEV))
            c = m(x, t, w, state=m(x, t, w))
        self.assertTrue(close(a, b)); self.assertFalse(close(a, c, 1e-3))

    def test_self_conditioned_loss_uses_the_first_pass_state_without_gradient(self):
        m = net(carry=True).eval(); x, t, w = batch()
        hidden = torch.rand(len(x), M.COLS, device=M.DEV) < 0.5
        with torch.no_grad():
            torch.nn.init.normal_(m.carry_ln.weight)
            one = M.elbo_terms(m, TABLES, x, hidden, t, w, M.sched84().to(M.DEV), self_condition=False)
            two = M.elbo_terms(m, TABLES, x, hidden, t, w, M.sched84().to(M.DEV))
            xin = x.masked_fill(hidden, 0)
            lg = m.logits(m(xin, t, w, state=m(xin, t, w)))
        self.assertFalse(close(one, two, 1e-4))
        al = M.allowed(TABLES, xin)
        ref = (M.per_column_ce({k: v.masked_fill(~al[k], -1e4) for k, v in lg.items()}, x) * hidden
               * M.sched84().to(M.DEV).view(1, -1) / t.view(-1, 1)).sum(1) / M.COLS
        self.assertTrue(close(two, ref, 1e-3))
        m.train(); M.batch_loss(m, TABLES, x[:8], w[:8], M.sched84().to(M.DEV)).backward()
        self.assertIsNotNone(m.carry_ln.weight.grad)

    def test_a_carried_state_network_cannot_join_a_mixture(self):
        a = M.Model84(VOCAB, X.setup()[0], [(net(carry=True), dict(TINY, carry=True))])
        with self.assertRaises(ValueError): M.Model84.mixture([a, a])
        two = M.Model84(VOCAB, X.setup()[0], [(net(carry=True), dict(TINY, carry=True)), (net(carry=True), dict(TINY, carry=True))])
        with self.assertRaises(ValueError): two.sample(2, ask=0.5, guidance=2.0)


@unittest.skipUnless(HERE, "needs /tmp/vgc-pilot and the shuffled-only table")
class SamplingAndCheckpoints(unittest.TestCase):
    def test_both_options_sample_legal_shaped_teams_and_survive_a_save(self):
        corpus = X.setup()[0]
        for kw in (LOOP, dict(carry=True)):
            with self.subTest(kw=kw):
                m = M.Model84(VOCAB, corpus, [(net(**kw).eval(), dict(TINY, **kw))])
                rows = m.sample(4, ask=0.5, guidance=2.0, seed=3)
                self.assertEqual(tuple(rows.shape), (4, M.COLS)); self.assertTrue(bool((rows > 0).all()))
                self.assertTrue(torch.equal(rows, m.sample(4, ask=0.5, guidance=2.0, seed=3)))
                with tempfile.TemporaryDirectory() as tmp:
                    m.save(Path(tmp) / "m.pt"); back = M.Model84.load(Path(tmp) / "m.pt", corpus)
                self.assertTrue(torch.equal(rows, back.sample(4, ask=0.5, guidance=2.0, seed=3)))

    def test_carried_sampler_hands_each_pass_its_previous_hidden_state(self):
        m = M.Model84(VOCAB, X.setup()[0], [(net(carry=True).eval(), dict(TINY, carry=True))])
        torch.nn.init.normal_(m.net.carry_ln.weight)
        seen = []; original = m.net.forward
        m.net.forward = lambda x, t, w, loops=None, state=None: (seen.append(state), original(x, t, w, loops=loops, state=state))[1]
        m.sample(2, ask=0.5, guidance=2.0, seed=1)
        self.assertEqual(len(seen), 2 * M.COLS)               # one guided and one unguided pass per field
        self.assertIsNone(seen[0]); self.assertIsNone(seen[1])
        self.assertTrue(all(s is not None for s in seen[2:]))


@unittest.skipUnless(HERE, "needs /tmp/vgc-pilot and the shuffled-only table")
class Compute(unittest.TestCase):
    def test_compute_ratios_and_equal_compute_caps(self):
        base = M.training_macs(VOCAB, **TR.NET)
        block = M.training_macs(VOCAB, **TR.NET, **TR.LOOP_BLOCK) / base
        carry = M.training_macs(VOCAB, **TR.NET, carry=True) / base
        self.assertAlmostEqual(carry, 1 + M.CARRY_P / 3)
        self.assertTrue(1.48 < block < 1.51)                   # 6 layers -> on average 1 + 2 x 2.5 + 3 = 9, plus the heads
        self.assertEqual(TR.equal_compute_cap(TR.LOOP_BLOCK), 53)
        self.assertEqual(TR.equal_compute_cap(dict(carry=True)), 61)
        for k, v in TR.LOOP_NETS.items():
            cap = TR.equal_compute_cap(v); r = M.training_macs(VOCAB, **TR.NET, **v) / base
            self.assertLessEqual(cap * r, TR.CONTROL_EPOCHS); self.assertGreater((cap + 1) * r, TR.CONTROL_EPOCHS)


if __name__ == "__main__":
    unittest.main()
