"""Entropy first, then confidence order (issue #333): pass 1 fills the masked field with the highest entropy and draws
its value at the first-pass temperature; every later pass follows confidence order at temperature 1. Tested through
the arm object the ask grid takes, on a tiny model trained for two epochs on 256 corpus teams, as in
test_confidence_order84.py. The edit-distance diagnostic is tested against a plain Levenshtein. The model tests need
this laptop's /tmp/vgc-pilot data and are skipped without it.

    /tmp/vgc-pilot/.venv/bin/python scripts/test_entropy_init84.py   (from code/)
"""
import itertools
import unittest
from pathlib import Path

import numpy as np

import repo_paths  # noqa: F401  (src/ first on sys.path)

ASK, GUIDANCE, N = 0.5, 2.0, 48

VOCAB = None
if Path("/tmp/vgc-pilot/data/pokedex.json").exists():
    import team84 as T
    import asked_vs_got as X
    VOCAB = T.regulation_vocab()
    CORPUS = X.setup()[0]


class Watched:
    """The model, passed to the arm unchanged, keeping a copy of the team each network pass reads."""
    def __init__(self, model): self.model, self.seen = model, []

    def _logprobs(self, x, *args, **kwargs):
        self.seen.append(x.clone()); return self.model._logprobs(x, *args, **kwargs)

    def __getattr__(self, name): return getattr(self.model, name)


def levenshtein(a, b):
    d = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        prev, d[0] = d[0], i
        for j, y in enumerate(b, 1): prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (x != y))
    return d[-1]


class EditDistance(unittest.TestCase):
    def test_matches_a_plain_levenshtein(self):
        import entropy_init84 as E
        rng = np.random.default_rng(0)
        orders = np.array([rng.permutation(9) for _ in range(7)] + [np.arange(9), np.arange(9)])
        got = E.edit_distances(orders)
        want = [levenshtein(list(a), list(b)) for a, b in itertools.combinations(orders, 2)]
        self.assertEqual(got.tolist(), want)
        self.assertEqual(got[-1], 0, "two identical orders are 0 apart")

    def test_normalized_by_84_and_pooled_by_ask(self):
        import entropy_init84 as E
        same = [dict(numerator=3, refusals=[dict(fill_order=list(range(84)))] * 2)]
        flip = [dict(numerator=4, refusals=[dict(fill_order=list(range(84)))]), dict(numerator=4, refusals=[dict(fill_order=list(range(84))[::-1])])]
        r = E.fill_order_edit_distance(same + flip)
        self.assertEqual(r["by_ask"]["3"], 0.0)
        self.assertAlmostEqual(r["by_ask"]["4"], levenshtein(list(range(84)), list(range(84))[::-1]) / 84)


@unittest.skipUnless(VOCAB is not None, "the /tmp/vgc-pilot runtime is not available")
class EntropyFirst(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        import diffusion84 as M
        import entropy_init84 as E
        cls.torch, cls.M, cls.E = torch, M, E
        rows = []
        for team in CORPUS:
            try: rows.append(T.encode(VOCAB, team))
            except ValueError: continue
            if len(rows) == 256: break
        cls.model = M.Model84.train_new(VOCAB, CORPUS, rows, labels=None, epochs=2, seed=0)

    def logits(self, x, step, c):
        """Guided log-probabilities of column c for teams x at pass `step`, legal values only, built without the arm's code."""
        torch, M, n = self.torch, self.M, len(x)
        t = torch.full((n,), 1.0 - step / M.COLS, device=M.DEV)
        lc = self.model._column(self.model._logprobs(x, t, torch.full((n,), ASK, device=M.DEV)), c)
        lc = GUIDANCE * lc + (1 - GUIDANCE) * self.model._column(self.model._logprobs(x, t, torch.full((n,), M.NAN, device=M.DEV)), c)
        return lc.masked_fill(~self.model._legal(x, c, lookahead=True), -float("inf"))

    def draw(self, model, n=N, seed=1):
        self.torch.manual_seed(seed)
        arm = self.E.arm(model)
        return arm.sample(n, ask=ASK, guidance=GUIDANCE), arm

    def empty(self): return self.torch.zeros(1, self.M.COLS, dtype=self.torch.long, device=self.M.DEV)

    def test_pass_one_fills_the_highest_entropy_field(self):
        torch, M = self.torch, self.M
        h = torch.stack([torch.special.entr(torch.softmax(self.logits(self.empty(), 0, c), -1)).sum(1)[0] for c in range(M.COLS)])
        top = torch.stack([torch.softmax(self.logits(self.empty(), 0, c), -1).max() for c in range(M.COLS)])
        self.assertNotEqual(int(h.argmax()), int(top.argmax()), "entropy and confidence pick the same first field: the test cannot tell them apart")
        _, arm = self.draw(self.model)
        firsts = {r["fill_order"][0] for r in arm.last_refusals}
        self.assertEqual(firsts, {int(h.argmax())}, "pass 1 did not fill the highest-entropy field")

    def test_pass_one_draws_at_the_first_pass_temperature(self):
        """The probabilities handed to the draw: pass 1's are the legal logits over 1.5, every later pass's over 1."""
        from unittest import mock
        torch = self.torch
        seen, real = [], torch.multinomial
        def spy(p, k, *a, **kw):
            seen.append(p.detach().clone()); return real(p, k, *a, **kw)
        watched = Watched(self.model)
        with mock.patch.object(self.E.C.torch, "multinomial", side_effect=spy):
            _, arm = self.draw(watched, n=6, seed=3)
        c = arm.last_refusals[0]["fill_order"][0]
        lg = self.logits(self.empty().expand(6, -1).contiguous(), 0, c)
        hot, base = torch.softmax(lg / self.E.FIRST_TEMPERATURE, -1), torch.softmax(lg, -1)
        self.assertTrue(torch.allclose(seen[0], hot, atol=1e-5), "pass 1 did not draw from the temperature-1.5 distribution")
        self.assertFalse(torch.allclose(seen[0], base, atol=1e-5), "temperatures 1 and 1.5 give the same distribution here")
        before = watched.seen[2]                                # the team read by pass 2's guided call
        c2 = arm.last_refusals[0]["fill_order"][1]
        rows = torch.tensor([r["fill_order"][1] == c2 for r in arm.last_refusals], device=before.device)
        want = torch.softmax(self.logits(before, 1, c2), -1)[rows]
        columns = sorted({r["fill_order"][1] for r in arm.last_refusals})   # pass 2 draws one column at a time, in this order
        got = seen[1 + columns.index(c2)]
        self.assertTrue(torch.allclose(got, want, atol=1e-5), "pass 2 did not draw at temperature 1")

    def test_every_later_pass_follows_confidence_order(self):
        torch, M = self.torch, self.M
        watched = Watched(self.model)
        x, arm = self.draw(watched, n=12)
        passes = [*watched.seen[::2], x]                       # the guided pass of each step reads the team before its draw
        for step in (1, 2, 5, 20, 47, 70, 83):
            before, after = passes[step], passes[step + 1]
            top = torch.full(before.shape, -1.0, device=before.device)
            for c in range(M.COLS):
                masked = before[:, c] == 0
                if masked.any(): top[masked, c] = torch.softmax(self.logits(before, step, c), -1).max(1).values[masked]
            filled = (before == 0) & (after != 0)
            self.assertTrue(bool((filled.sum(1) == 1).all()), f"pass {step} filled other than one field per team")
            self.assertTrue(bool((filled.float().argmax(1) == top.argmax(1)).all()), f"pass {step} skipped a field with a higher top probability")
        for r in arm.last_refusals:
            self.assertEqual(sorted(r["fill_order"]), list(range(M.COLS)), "the fill order is not one field per column")

    def test_probe_reads_the_same_first_field_as_the_grid(self):
        _, arm = self.draw(self.model, n=8)
        p = self.E.first_pass(self.model, ASK, GUIDANCE)
        self.assertEqual(p["column"], arm.last_refusals[0]["fill_order"][0])
        self.assertEqual(p["field_type"], self.E.field_type(p["column"]))
        self.assertAlmostEqual(p["entropy"], max(p["entropy_by_field_type"].values()), places=5)

    def test_confidence_first_pick_is_unchanged(self):
        import confidence_order84 as C
        self.torch.manual_seed(9); a = C.ConfidenceOrderArm(self.model).sample(16, ask=ASK, guidance=GUIDANCE)
        self.torch.manual_seed(9); b = C.ConfidenceOrderArm(self.model, first_pick="confidence", first_temperature=None).sample(16, ask=ASK, guidance=GUIDANCE)
        self.assertTrue(bool((a == b).all()))


if __name__ == "__main__":
    unittest.main()
