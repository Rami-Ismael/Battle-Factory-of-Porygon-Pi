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
        cls.model = M.Model84.train_new(VOCAB, CORPUS, rows, labels=None, passes=2, seed=0)
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


if __name__ == "__main__":
    unittest.main()
