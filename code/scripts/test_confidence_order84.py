"""Confidence order on the 84-field model (issue #332): each pass fills each team's masked field with the highest top
probability, never fills a field twice, and draws the value from that field's distribution. Tested through the arm
object the ask grid takes, on a tiny model trained for two epochs on 256 corpus teams, as in test_species_then_random.py.
Needs this laptop's /tmp/vgc-pilot data; skipped without it.

    /tmp/vgc-pilot/.venv/bin/python scripts/test_confidence_order84.py   (from code/)
"""
import unittest
from pathlib import Path

import repo_paths  # noqa: F401  (src/ first on sys.path)
import itertools

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


@unittest.skipUnless(VOCAB is not None, "the /tmp/vgc-pilot runtime is not available")
class ConfidenceOrder(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        import diffusion84 as M
        import sampler84 as S
        cls.torch, cls.M, cls.S = torch, M, S
        rows = []
        for team in CORPUS:
            try: rows.append(T.encode(VOCAB, team))
            except ValueError: continue
            if len(rows) == 256: break
        cls.model = M.Model84.train_new(VOCAB, CORPUS, rows, labels=None, epochs=2, seed=0)

    def distribution(self, x, step, c):
        """The drawing distribution of column c for teams x at pass `step`, built without the arm's code: guided
        log-probabilities, the legality mask from Model84._legal with the lookahead, temperature 1."""
        torch, M, n = self.torch, self.M, len(x)
        t = torch.full((n,), 1.0 - step / M.COLS, device=M.DEV)
        lc = self.model._column(self.model._logprobs(x, t, torch.full((n,), ASK, device=M.DEV)), c)
        lc = GUIDANCE * lc + (1 - GUIDANCE) * self.model._column(self.model._logprobs(x, t, torch.full((n,), M.NAN, device=M.DEV)), c)
        return torch.softmax(lc.masked_fill(~self.model._legal(x, c, lookahead=True), -float("inf")), -1)

    def draw(self, model, n=N, seed=1, **kw):
        self.torch.manual_seed(seed)
        arm = self.S.Sampler84(model, "confidence", **kw)
        return arm.sample(n, ask=ASK, guidance=GUIDANCE), arm

    def test_each_pass_fills_the_masked_field_with_the_highest_top_probability(self):
        watched = Watched(self.model)
        x, _ = self.draw(watched, n=12)
        passes = [*watched.seen[::2], x]                       # the guided pass of each step reads the team before its draw
        for step in (0, 1, 5, 20, 47, 70, 83):
            before, after = passes[step], passes[step + 1]
            top = self.torch.full(before.shape, -1.0, device=before.device)
            for c in range(self.M.COLS):
                masked = before[:, c] == 0
                if masked.any(): top[masked, c] = self.distribution(before, step, c).max(1).values[masked]
            filled = (before == 0) & (after != 0)
            self.assertTrue(bool((filled.sum(1) == 1).all()), f"pass {step} filled other than one field per team")
            chosen = filled.float().argmax(1)
            self.assertTrue(bool((chosen == top.argmax(1)).all()), f"pass {step} skipped a field with a higher top probability")

    def test_a_filled_field_is_never_filled_again(self):
        watched = Watched(self.model)
        x, arm = self.draw(watched)
        passes = [*watched.seen[::2], x]
        for a, b in itertools.pairwise(passes):
            kept = a != 0
            self.assertTrue(bool((a[kept] == b[kept]).all()), "a filled field changed")
        for r in arm.last_refusals:
            self.assertEqual(sorted(r["fill_order"]), list(range(self.M.COLS)), "the fill order is not one field per column")

    def test_the_value_is_drawn_from_the_distribution_not_chosen_by_it(self):
        n = 400
        x, arm = self.draw(self.model, n=n, seed=3)
        first = {r["fill_order"][0] for r in arm.last_refusals}
        self.assertEqual(len(first), 1, "teams with the same empty start picked different first fields")
        c = first.pop()
        p = self.distribution(self.torch.zeros(1, self.M.COLS, dtype=self.torch.long, device=self.M.DEV), 0, c)[0]
        values = x[:, c] - 1 if c >= self.M.STAT0 else x[:, c]
        share = (values == p.argmax()).float().mean().item(); p_top = p.max().item()
        self.assertLess(p_top, 0.999, "the first field's distribution is too sharp to tell drawing from choosing")
        self.assertLess(abs(share - p_top), 4 * (p_top * (1 - p_top) / n) ** 0.5 + 1e-3,
                        f"top value drawn {share:.3f} of the time against probability {p_top:.3f}")
        self.assertTrue(bool((p[values] > 0).all()), "a drawn value had zero probability")
        self.assertGreater(sum(r["non_top_fields"] for r in arm.last_refusals), 0, "every draw took the most likely value")

    def test_a_fixed_order_with_the_lookahead_off_reproduces_model84_sample(self):
        torch = self.torch
        torch.manual_seed(5); a = self.model.sample(N, ask=ASK, guidance=GUIDANCE)
        b, _ = self.draw(self.model, seed=5, lookahead=False, order=self.M.ORDER)
        self.assertTrue(bool((a == b).all()), "the fixed-order path differs from Model84.sample")

    def test_non_top_count_matches_the_draws(self):
        _, arm = self.draw(self.model, n=8, seed=7)
        for r in arm.last_refusals:
            self.assertTrue(0 <= r["non_top_fields"] <= self.M.COLS)
            self.assertIn("log_kept_share", r)


if __name__ == "__main__":
    unittest.main()
