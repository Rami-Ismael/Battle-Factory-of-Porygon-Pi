"""legality_skew.SkewLog splits a draw's removed probability into the four sampling-only rules. No checkpoint, no sampling.

Needs this laptop's /tmp/vgc-pilot data (diffusion84 imports it); skipped without it.

    /tmp/vgc-pilot/.venv/bin/python scripts/test_legality_skew.py   (from code/)
"""
import math
import sys
import unittest
from pathlib import Path

import repo_paths  # noqa: F401  (src/ first on sys.path)

HERE = Path("/tmp/vgc-pilot/data/pokedex.json").exists()
if HERE:
    sys.path.insert(1, "/tmp/vgc-pilot/src")
    import torch
    import diffusion84 as M
    import legality_skew as S
    import lossdown as L

# Five values for a first-move field: 0 [MASK], 1 the empty move, 2 and 3 real moves, 4 a move the tables forbid.
P = [0.1, 0.2, 0.3, 0.3, 0.1]
TAB = [True, True, True, True, False]


class FakeTables:
    EMV = 1

    def __init__(self, rows):
        self.rows = rows

    def allowed(self, x):
        n_moves = len(L.COLS_BY_KEY["move"])
        return {"move": torch.tensor(self.rows).unsqueeze(1).expand(len(self.rows), n_moves, len(P)).clone()}


class FakeModel:
    def __init__(self, rows): self.tables = FakeTables(rows)


@unittest.skipUnless(HERE, "needs /tmp/vgc-pilot data")
class SkewLogShares(unittest.TestCase):
    column = M.FIRST_MOVE                                  # Pokémon 1's first move

    def draw(self, kept_row, fallback_row, n=1):
        w = FakeModel([TAB] * n)
        scores = torch.log(torch.tensor([P] * n, dtype=torch.float64))
        plain = torch.tensor([[False, False, True, True, False]] * n)
        kept = torch.tensor([kept_row] * n)
        x = torch.zeros(n, M.COLS, dtype=torch.long)
        log = S.SkewLog(w, n)
        log.add(scores, plain, kept, torch.tensor([fallback_row] * n), column=self.column, x=x)
        return log

    def test_each_rule_gets_its_own_share(self):
        log = self.draw([False, False, True, False, False], False)
        out, _ = log.shares(torch.log(torch.tensor([P], dtype=torch.float64)), torch.tensor([[False, False, True, True, False]]),
                            torch.tensor([[False, False, True, False, False]]), torch.tensor([False]), self.column,
                            torch.zeros(1, M.COLS, dtype=torch.long))
        for rule, want in (("mask_token", 0.1), ("no_moves", 0.2), ("tables", 0.1), ("lookahead", 0.3)):
            self.assertAlmostEqual(float(out[rule][0]), want, places=9, msg=rule)

    def test_team_log_ratio_is_minus_log_of_the_kept_share(self):
        log = self.draw([False, False, True, False, False], False)
        self.assertAlmostEqual(log.team_log_ratio[0].item(), -math.log(0.3), places=9)
        s = log.sums["move"]
        self.assertEqual((s["draws"], s["fallback"]), (1, 0))

    def test_a_fallback_draw_removes_only_the_mask_value(self):
        log = self.draw([False, False, True, True, False], True)
        s = log.sums["move"]
        self.assertEqual(s["fallback"], 1)
        self.assertAlmostEqual(s["mask_token"], 0.1, places=9)
        for rule in ("no_moves", "tables", "lookahead"): self.assertEqual(s[rule], 0.0, rule)
        self.assertEqual(log.team_fallbacks[0].item(), 1)

    def test_shares_that_do_not_add_up_raise(self):
        with self.assertRaises(AssertionError):            # kept lets in value 4, which the tables forbid
            self.draw([False, False, True, True, True], False)

    def test_a_stat_point_draw_counts_what_the_budget_now_forbids(self):
        log = S.SkewLog(FakeModel([TAB]), 1)               # a Stat Point drawn earlier in the pass closed values 3 and 4
        plain = torch.tensor([[True, True, True, False, False]])
        out, _ = log.shares(torch.log(torch.tensor([P], dtype=torch.float64)), plain, plain, torch.tensor([False]),
                            M.STAT0 + 34, torch.zeros(1, M.COLS, dtype=torch.long))
        self.assertAlmostEqual(float(out["tables"][0]), 0.4, places=9)
        for rule in ("mask_token", "no_moves", "lookahead"): self.assertEqual(float(out[rule][0]), 0.0, rule)


if __name__ == "__main__":
    unittest.main()
