"""Stat Points as six generated fields per Pokémon (issue #57): the legal values a sampler may draw.

A Pokémon's six Stat Points are each 0 to 32 and total at most 66. A Stat Point token is the value plus one;
token 0 is [MASK]. These tests need no game data, checkpoints or Showdown, so CI runs them.

    .venv/bin/python scripts/test_stat_points.py   (from code/)
"""
import unittest

import torch

import repo_paths  # noqa: F401  (src/ first on sys.path)
import stat_points as SP

MASK = 0


def tok(*values):
    """Stat Point tokens for one Pokémon: a value v becomes v + 1, None stays masked."""
    return [MASK if v is None else v + 1 for v in values]


def team(*pokemon):
    """[1, 36] tokens: the given Pokémon first, every other Pokémon fully masked."""
    rows = list(pokemon) + [tok(*[None] * 6)] * (6 - len(pokemon))
    return torch.tensor([[t for row in rows for t in row]])


def allowed_values(tokens, pokemon, stat):
    """The Stat Point values (not tokens) the mask allows for one stat of one Pokémon."""
    ok = SP.allowed_stat_points(tokens)[0, pokemon * 6 + stat]
    return [v for v in range(33) if ok[v]]


class StatPointBudget(unittest.TestCase):
    def test_a_pokemon_with_nothing_drawn_may_take_any_value_up_to_32(self):
        self.assertEqual(allowed_values(team(), 0, 0), list(range(33)))

    def test_two_maxed_stats_leave_2_points_for_the_rest(self):
        x = team(tok(32, 32, None, None, None, None))
        self.assertEqual(allowed_values(x, 0, 2), [0, 1, 2])
        self.assertEqual(allowed_values(x, 0, 5), [0, 1, 2])

    def test_a_full_budget_allows_only_zero(self):
        x = team(tok(32, 2, 32, None, 0, None))
        self.assertEqual(allowed_values(x, 0, 3), [0])

    def test_the_budget_is_per_pokemon(self):
        x = team(tok(32, 32, 2, 0, 0, None), tok(None, 10, None, None, None, None))
        self.assertEqual(allowed_values(x, 0, 5), [0])
        self.assertEqual(allowed_values(x, 1, 0), list(range(33)))

    def test_a_drawn_stat_keeps_its_own_value_allowed(self):
        x = team(tok(30, 30, 6, None, None, None))
        self.assertIn(6, allowed_values(x, 0, 2))

    def test_shape_is_36_columns_by_33_values_for_a_batch(self):
        x = torch.zeros(5, 36, dtype=torch.long)
        self.assertEqual(tuple(SP.allowed_stat_points(x).shape), (5, 36, 33))


ARCHALUDON = {"HP": 27, "Def": 2, "SpA": 5, "SpD": 32}          # a paste lists only the non-zero stats
BASCULEGION = {"HP": 2, "Atk": 32, "Spe": 32}


class StatPointTokens(unittest.TestCase):
    def test_stat_points_become_value_plus_one_in_hp_atk_def_spa_spd_spe_order(self):
        tokens = SP.tokens_from_stat_points([ARCHALUDON, BASCULEGION] + [{}] * 4)
        self.assertEqual(tokens[:12], [28, 1, 3, 6, 33, 1, 3, 33, 1, 1, 1, 33])
        self.assertEqual(len(tokens), 36)

    def test_tokens_round_trip_to_the_same_stat_points(self):
        tokens = SP.tokens_from_stat_points([ARCHALUDON, BASCULEGION] + [{}] * 4)
        back = SP.stat_points_from_tokens(tokens)
        self.assertEqual(back[0], {"HP": 27, "Atk": 0, "Def": 2, "SpA": 5, "SpD": 32, "Spe": 0})
        self.assertEqual(back[1], {"HP": 2, "Atk": 32, "Def": 0, "SpA": 0, "SpD": 0, "Spe": 32})

    def test_a_masked_token_cannot_become_stat_points(self):
        with self.assertRaises(ValueError):
            SP.stat_points_from_tokens([0] + [1] * 35)

    def test_illegal_stat_points_are_refused(self):
        with self.assertRaises(ValueError):
            SP.tokens_from_stat_points([{"HP": 33}] + [{}] * 5)
        with self.assertRaises(ValueError):
            SP.tokens_from_stat_points([{"HP": 32, "Atk": 32, "Def": 3}] + [{}] * 5)


if __name__ == "__main__":
    unittest.main()
