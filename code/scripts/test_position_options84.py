"""Position options of the 84-field network (issue #81; spec #233): how a column's position vector is built.

    No split               pos="per_field"     one learned vector per column (the default, unchanged)
    Learned slot vectors   pos="slot_learned"  learned field-type vector + learned team-position vector
    Fixed slot vectors     pos="slot_fixed"    learned field-type vector + random team-position vector that never trains
    No slot vectors        pos="no_slot"       learned field-type vector only

Needs this laptop's /tmp/vgc-pilot data; skipped without it.

    /tmp/vgc-pilot/.venv/bin/python scripts/test_position_options84.py   (from code/)
"""
import sys
import tempfile
import unittest
from pathlib import Path

import repo_paths  # noqa: F401  (src/ first on sys.path)

HERE = Path("/tmp/vgc-pilot/data/pokedex.json").exists()
if HERE:
    sys.path.insert(1, "/tmp/vgc-pilot/src")
    import torch
    import diffusion84 as M
    import team84 as T

TINY = dict(d=32, nlayer=1, nhead=2)

# Recorded 2026-11-07 from the network before position options existed (main at 5df3b063): torch.manual_seed(0),
# then Net84(V, **TINY); inputs drawn from torch.Generator().manual_seed(1). Independent of the new code.
OLD_NAMES = 43
OLD_POS_SUM = -0.2495521
OLD_POS_FIRST3 = -0.04198825
OLD_OUT_ABS_SUM = 8814.3164
OLD_H_0_5 = [-0.25293, -1.12427, -0.16244]
OLD_H_3_80 = [-0.702, -0.15684, 0.34099]
OLD_CE = 4.81942


def seeded_net(**kw):
    torch.manual_seed(0)
    return M.Net84(VOCAB, **TINY, **kw).eval()


def inputs(n=4):
    g = torch.Generator().manual_seed(1)
    x = torch.zeros(n, M.COLS, dtype=torch.long)
    t, w = torch.rand(n, generator=g), torch.rand(n, generator=g)
    sizes = [VOCAB.sizes[M.L.KEY_OF_COL[c]] for c in range(M.STAT0)]
    xs = torch.stack([torch.randint(1, s, (n,), generator=g) for s in sizes]
                     + [torch.randint(1, 33, (n,), generator=g) for _ in range(M.COLS - M.STAT0)], 1)
    return x, t, w, xs


VOCAB = T.regulation_vocab() if HERE else None


@unittest.skipUnless(HERE, "needs /tmp/vgc-pilot")
class NoSplitIsUnchanged(unittest.TestCase):
    def test_parameters_starting_values_outputs_and_loss_match_the_old_network(self):
        for kw in ({}, {"pos": "per_field"}):
            with self.subTest(kw=kw):
                net = seeded_net(**kw); sd = net.state_dict()
                self.assertEqual(len(sd), OLD_NAMES)
                self.assertEqual(tuple(sd["pos"].shape), (84, 32))
                self.assertAlmostEqual(float(sd["pos"].sum()), OLD_POS_SUM, places=6)
                self.assertAlmostEqual(float(sd["pos"][0, :3].sum()), OLD_POS_FIRST3, places=7)
                x, t, w, xs = inputs()
                with torch.no_grad():
                    h = net(x, t, w)
                    ce = M.per_column_ce(net.logits(net(xs, t, w)), xs)
                self.assertAlmostEqual(float(h.abs().sum()), OLD_OUT_ABS_SUM, places=2)
                for got, want in ((h[0, 5, :3], OLD_H_0_5), (h[3, 80, :3], OLD_H_3_80)):
                    for g, v in zip(got.tolist(), want, strict=True): self.assertAlmostEqual(g, v, places=4)
                self.assertAlmostEqual(float(ce.mean()), OLD_CE, places=4)


@unittest.skipUnless(HERE, "needs /tmp/vgc-pilot")
class ColumnLayout(unittest.TestCase):
    def test_each_column_has_its_field_type_and_team_position(self):
        # 48 columns = 6 Pokémon x 8 fields (species, ability, item, 4 moves, Stat Alignment), then 6 x 6 Stat Points
        for col, field_type, position in ((0, 0, 0), (7, 7, 0), (8, 0, 1), (47, 7, 5), (48, 8, 0), (53, 13, 0),
                                          (54, 8, 1), (83, 13, 5)):
            with self.subTest(col=col):
                self.assertEqual(int(M.COLUMN_TYPE[col]), field_type)
                self.assertEqual(int(M.COLUMN_SLOT[col]), position)
        self.assertEqual(int(M.COLUMN_TYPE.max()) + 1, 14)
        self.assertEqual(int(M.COLUMN_SLOT.max()) + 1, 6)


def pairs_across_positions(v):
    """The difference between team positions 1 and 2 for a species, a move and a Stat Points column."""
    return [v[c + M.NF] - v[c] for c in (0, 4)] + [v[M.STAT0 + 6 + 5] - v[M.STAT0 + 5]]


@unittest.skipUnless(HERE, "needs /tmp/vgc-pilot")
class LearnedSlotVectors(unittest.TestCase):
    def test_a_column_is_its_field_type_plus_its_team_position(self):
        v = seeded_net(pos="slot_learned").position_vectors().detach()
        a, b, c = pairs_across_positions(v)
        self.assertTrue(torch.allclose(a, b, atol=1e-6) and torch.allclose(a, c, atol=1e-6))
        self.assertGreater(float(a.abs().sum()), 0.0)
        self.assertFalse(torch.allclose(v[0], v[4]))                    # different field types, same position

    def test_it_learns_20_vectors_that_start_at_scale_0_02(self):
        net = seeded_net(pos="slot_learned")
        pos = {k: p for k, p in net.named_parameters() if k.startswith("pos")}
        self.assertEqual(sorted(p.shape[0] for p in pos.values()), [6, 14])
        for p in pos.values(): self.assertAlmostEqual(float(p.std()), 0.02, delta=0.004)
        self.assertNotIn("pos", dict(net.named_parameters()))


def one_step(net):
    """One AdamW step on a seeded loss; returns the position vectors before and after."""
    net.train(); opt = torch.optim.AdamW(net.parameters(), lr=1e-2)
    _, t, w, xs = inputs()
    before = net.position_vectors().detach().clone()
    M.per_column_ce(net.logits(net(xs.masked_fill(torch.rand(xs.shape) < .5, 0), t, w)), xs).mean().backward()
    opt.step()
    return before, net.position_vectors().detach()


@unittest.skipUnless(HERE, "needs /tmp/vgc-pilot")
class FixedSlotVectors(unittest.TestCase):
    def test_the_team_position_part_is_not_trained(self):
        net = seeded_net(pos="slot_fixed")
        names = dict(net.named_parameters())
        self.assertIn("pos_type", names)
        self.assertNotIn("pos_slot", names)
        self.assertIn("pos_slot", net.state_dict())                     # saved with the weights
        slot_before = net.state_dict()["pos_slot"].clone()
        before, after = one_step(net)
        self.assertTrue(torch.equal(net.state_dict()["pos_slot"], slot_before))
        self.assertFalse(torch.equal(before, after))                     # the field-type part did train
        self.assertTrue(torch.allclose(*[torch.stack(pairs_across_positions(v)) for v in (before, after)], atol=1e-6))

    def test_it_starts_at_scale_0_02_and_differs_per_position(self):
        slot = seeded_net(pos="slot_fixed").state_dict()["pos_slot"]
        self.assertEqual(tuple(slot.shape), (6, 32))
        self.assertAlmostEqual(float(slot.std()), 0.02, delta=0.004)
        self.assertFalse(torch.allclose(slot[0], slot[1]))


def swap_columns(a=1, b=4):
    """Column order with Pokémon a and b swapped, in both the 48-field block and the Stat Points."""
    slot = M.COLUMN_SLOT.clone()
    slot[M.COLUMN_SLOT == a], slot[M.COLUMN_SLOT == b] = b, a
    return torch.tensor([int(torch.nonzero((M.COLUMN_TYPE == M.COLUMN_TYPE[c]) & (M.COLUMN_SLOT == slot[c]))[0])
                         for c in range(M.COLS)])


@unittest.skipUnless(HERE, "needs /tmp/vgc-pilot")
class NoSlotVectors(unittest.TestCase):
    def test_every_column_of_a_field_type_gets_the_same_vector(self):
        net = seeded_net(pos="no_slot")
        v = net.position_vectors().detach()
        for c in range(M.COLS): self.assertTrue(torch.equal(v[c], v[int(torch.nonzero(M.COLUMN_TYPE == M.COLUMN_TYPE[c])[0])]))
        self.assertEqual(sorted(k for k, _ in net.named_parameters() if k.startswith("pos")), ["pos_type"])

    def test_swapping_two_pokemon_changes_the_other_fields_only_by_rounding(self):
        _, t, w, xs = inputs()
        for option, most in (("no_slot", 1e-5), ("slot_learned", None)):
            with self.subTest(option=option):
                net = seeded_net(pos=option)
                perm = swap_columns()
                with torch.no_grad():
                    h, hw = net(xs, t, w), net(xs[:, perm], t, w)
                gap = float((hw[:, perm] - h).abs().max())
                if most: self.assertLess(gap, most)
                else: self.assertGreater(gap, 1e-3)                    # with slot information the outputs move


@unittest.skipUnless(HERE, "needs /tmp/vgc-pilot")
class SaveAndLoad(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import asked_vs_got as X
        cls.corpus = X.setup()[0]

    def round_trip(self, net, kw, path):
        M.Model84(VOCAB, self.corpus, [(net, kw)]).save(path)
        loaded, loaded_kw = M.Model84.load(path, self.corpus).nets[0]
        return loaded.cpu().eval(), loaded_kw

    def outputs(self, net):
        _, t, w, xs = inputs()
        with torch.no_grad(): return net(xs, t, w)

    def test_every_option_loads_back_as_itself(self):
        for option in M.POSITION_OPTIONS:
            with self.subTest(option=option), tempfile.TemporaryDirectory() as tmp:
                kw = dict(TINY, pos=option)
                net = seeded_net(pos=option)
                one_step(net); net.eval()
                loaded, loaded_kw = self.round_trip(net, kw, Path(tmp) / "m.pt")
                self.assertEqual(loaded.pos_option, option)
                self.assertEqual(loaded_kw["pos"], option)
                self.assertTrue(torch.allclose(self.outputs(loaded), self.outputs(net), atol=1e-5))

    def test_a_checkpoint_saved_before_position_options_loads_as_no_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            net = seeded_net()
            loaded, _ = self.round_trip(net, dict(TINY), Path(tmp) / "old.pt")    # net settings with no "pos", as before
            self.assertEqual(loaded.pos_option, "per_field")
            self.assertTrue(torch.allclose(self.outputs(loaded), self.outputs(net), atol=1e-5))


if __name__ == "__main__":
    unittest.main()
