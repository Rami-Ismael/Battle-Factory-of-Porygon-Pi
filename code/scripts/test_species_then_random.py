"""Species first, then random order (issue #322): the six species in species-first order, then the other 78 fields as
random order fills them. Tested through the arm object the ask grid takes, on a tiny model trained for two epochs on
256 corpus teams, as in test_lookahead84.py. Needs this laptop's /tmp/vgc-pilot data; skipped without it.

    /tmp/vgc-pilot/.venv/bin/python scripts/test_species_then_random.py   (from code/)
"""
import unittest
from pathlib import Path

import repo_paths  # noqa: F401  (src/ first on sys.path)

ASK, GUIDANCE, SEEDS, N = 0.5, 2.0, (1, 2), 48

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
class SpeciesThenRandom(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        import diffusion84 as M
        import remask84 as R
        cls.torch, cls.M, cls.R = torch, M, R
        rows = []
        for team in CORPUS:
            try: rows.append(T.encode(VOCAB, team))
            except ValueError: continue
            if len(rows) == 256: break
        cls.model = M.Model84.train_new(VOCAB, CORPUS, rows, labels=None, epochs=2, seed=0)
        cls.species = [c for c in range(M.STAT0) if c % 8 == 0]

    def draw(self, model, seed, lookahead=True):
        self.torch.manual_seed(seed)
        return self.R.RandomDecodingArm(model, "species_then_random", lookahead=lookahead).sample(N, ask=ASK, guidance=GUIDANCE)

    def test_every_species_is_filled_before_any_other_field(self):
        watched = Watched(self.model)
        self.draw(watched, SEEDS[0])
        species = self.torch.zeros(self.M.COLS, dtype=self.torch.bool); species[self.species] = True
        for x in watched.seen:
            others_filled = (x[:, ~species] != 0).any(1)
            self.assertTrue(bool((x[others_filled][:, species] != 0).all()), "a set field was filled while a species was still masked")
        self.assertTrue(any(bool((x[:, ~species] != 0).any()) for x in watched.seen), "no pass ever read a filled set field")

    def test_species_are_drawn_one_per_pass_in_species_first_order(self):
        watched = Watched(self.model)
        self.draw(watched, SEEDS[0])
        first = self.M.ORDER[:6]
        self.assertEqual(sorted(first), self.species)
        for i, c in enumerate(first):            # the guided and unguided pass of draw i both read species 1..i filled and the rest masked
            for x in watched.seen[2 * i: 2 * i + 2]:
                self.assertTrue(bool((x[:, first[:i]] != 0).all()) and bool((x[:, first[i:]] == 0).all()), f"species draw {i}")

    def test_the_lookahead_changes_no_draw(self):
        for seed in SEEDS:
            on, off = self.draw(self.model, seed, lookahead=True), self.draw(self.model, seed, lookahead=False)
            self.assertTrue(bool((on == off).all()), f"seed {seed}: the lookahead changed a draw")

    def test_every_team_is_complete(self):
        x = self.draw(self.model, SEEDS[1])
        self.assertTrue(bool((x != 0).all()))


if __name__ == "__main__":
    unittest.main()
