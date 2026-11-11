"""The diversity measures behind diversity_vs_training.py, on made-up teams. No game data, checkpoints or Showdown.

    .venv/bin/python scripts/test_diversity_vs_training.py   (from code/)
"""
import math
import unittest

import numpy as np

import diversity_vs_training as D

STATS = D.STATS


def slot(sp, item="i", ability="a", nature="n", moves=("m1", "m2", "m3", "m4"), spe=0):
    evs = {s: 0 for s in STATS}; evs["Spe"] = spe
    return dict(species=sp, item=item, ability=ability, nature=nature, moves=list(moves), evs=evs)


def team(prefix, **kw): return [slot(f"{prefix}{j}", **kw) for j in range(6)]


class Anchors(unittest.TestCase):
    def test_copies_of_one_team_count_as_one_team(self):
        teams = [team("x")] * 50
        for lv in D.LEVELS:
            K = D.kernel(teams, lv)
            self.assertAlmostEqual(D.vendi(K), 1.0, places=6)
            self.assertAlmostEqual(D.dcscore(K), 1.0, places=6)
            self.assertAlmostEqual(D.mean_distance(K), 0.0, places=6)

    def test_teams_sharing_nothing_count_as_n_teams(self):
        teams = [team(f"t{i}_") for i in range(20)]
        K = D.kernel(teams, "team")
        self.assertAlmostEqual(D.vendi(K), 20.0, places=4)
        self.assertAlmostEqual(D.mean_distance(K), 1.0, places=6)
        self.assertGreater(D.dcscore(K), 19.9)

    def test_two_copies_and_one_stranger_match_effective_number_of_frequencies(self):
        teams = [team("x"), team("x"), team("y")]
        expected = math.exp(-(2 / 3 * math.log(2 / 3) + 1 / 3 * math.log(1 / 3)))
        self.assertAlmostEqual(D.vendi(D.kernel(teams, "team")), expected, places=6)


class Levels(unittest.TestCase):
    def test_a_different_item_changes_sets_but_not_the_line_up(self):
        a, b = team("x"), team("x", item="other")
        self.assertEqual(D.kernel([a, b], "lineup")[0, 1], 1.0)
        self.assertLess(D.kernel([a, b], "sets")[0, 1], 1.0)
        self.assertEqual(D.kernel([a, b], "statpoints")[0, 1], 1.0)

    def test_different_stat_points_change_only_the_stat_points_level(self):
        a, b = team("x"), team("x", spe=32)
        self.assertEqual(D.kernel([a, b], "sets")[0, 1], 1.0)
        self.assertLess(D.kernel([a, b], "statpoints")[0, 1], 1.0)
        self.assertNotEqual(D.key84(a), D.key84(b)); self.assertEqual(D.key48(a), D.key48(b))

    def test_the_same_item_on_another_species_is_not_shared(self):
        a, b = team("x"), team("y")
        self.assertEqual(D.kernel([a, b], "sets")[0, 1], 0.0)

    def test_slot_order_does_not_matter(self):
        a = team("x"); b = list(reversed(a))
        self.assertAlmostEqual(float(D.kernel([a, b], "team")[0, 1]), 1.0, places=6)
        self.assertEqual(D.key84(a), D.key84(b))


class Counts(unittest.TestCase):
    def test_effective_number_of_equal_frequencies_is_their_count(self):
        self.assertAlmostEqual(D.effective({"a": 3, "b": 3, "c": 3}), 3.0)

    def test_one_line_up_has_fifteen_species_pairs(self):
        self.assertEqual(D.counts([team("x")])["distinct_pairs"], 15)


class Swaps(unittest.TestCase):
    def test_a_copied_line_up_is_zero_swaps_and_shares_sum_to_one(self):
        shares = D.species_swaps([team("x")], [D.lineup(team("x"))])
        self.assertEqual(shares["swaps_0"], 1.0)
        self.assertAlmostEqual(sum(shares.values()), 1.0)

    def test_one_changed_species_is_one_swap(self):
        changed = team("x"); changed[0] = slot("new")
        shares = D.species_swaps([changed], [D.lineup(team("x"))])
        self.assertEqual(shares["swaps_1"], 1.0)

    def test_sharing_no_species_is_three_or_more_swaps(self):
        shares = D.species_swaps([team("y")], [D.lineup(team("x"))])
        self.assertEqual(shares["swaps_3plus"], 1.0)

    def test_exclude_self_drops_a_line_up_that_appears_once(self):
        ref = [D.lineup(team("x"))]
        self.assertEqual(D.species_swaps([team("x")], ref)["swaps_0"], 1.0)
        self.assertEqual(D.species_swaps([team("x")], ref, exclude_self=True)["swaps_3plus"], 1.0)

    def test_exclude_self_keeps_a_line_up_that_appears_twice(self):
        ref = [D.lineup(team("x"))] * 2
        self.assertEqual(D.species_swaps([team("x")], ref, exclude_self=True)["swaps_0"], 1.0)

    def test_exclude_self_falls_back_to_the_next_closest_line_up(self):
        near = team("x"); near[0] = slot("new"); near[1] = slot("new2")
        ref = [D.lineup(team("x")), D.lineup(near), D.lineup(team("y"))]
        self.assertEqual(D.species_swaps([team("x")], ref, exclude_self=True)["swaps_2"], 1.0)

    def test_closest_of_several_reference_line_ups_wins(self):
        one_off = team("x"); one_off[5] = slot("new")
        ref = [D.lineup(team("y")), D.lineup(one_off)]
        self.assertEqual(D.species_swaps([team("x")], ref)["swaps_1"], 1.0)

    def test_shares_are_per_team(self):
        shares = D.species_swaps([team("x"), team("y")], [D.lineup(team("x"))])
        self.assertEqual((shares["swaps_0"], shares["swaps_3plus"]), (0.5, 0.5))


class Archetypes(unittest.TestCase):
    def test_a_setting_ability_puts_a_team_in_its_archetype_only(self):
        shares = D.archetype_shares([team("x", ability="Drought")])
        self.assertEqual(shares["sun"], 1.0)
        self.assertEqual(shares["none"], 0.0)

    def test_a_team_with_two_setters_is_in_both_archetypes(self):
        shares = D.archetype_shares([team("x", moves=("Trick Room", "Tailwind", "m3", "m4"))])
        self.assertEqual(shares["Trick Room"], 1.0)
        self.assertEqual(shares["Tailwind"], 1.0)

    def test_a_mega_stone_whose_mega_ability_sets_weather_counts_with_the_base_ability_stored(self):
        t = team("x"); t[0] = slot("charizard", item="charizarditey", ability="blaze")
        self.assertEqual(D.archetype_shares([t])["sun"], 1.0)

    def test_names_match_in_the_vocabulary_spelling(self):
        shares = D.archetype_shares([team("x", moves=("trickroom", "m2", "m3", "m4"), ability="sandspit")])
        self.assertEqual((shares["Trick Room"], shares["sand"]), (1.0, 1.0))

    def test_a_plain_team_is_only_in_none(self):
        shares = D.archetype_shares([team("x")])
        self.assertEqual(shares["none"], 1.0)
        self.assertEqual(sum(v for k, v in shares.items() if k != "none"), 0.0)


class Builds(unittest.TestCase):
    def test_one_build_used_ten_times_gives_one_build_at_five_uses(self):
        teams = [[slot("s"), *team(f"f{i}_")[:5]] for i in range(10)]
        out = D.builds_at_equal_uses(teams, 5, np.random.default_rng(0))
        self.assertEqual(out["builds_per_species"], 1.0)
        self.assertEqual(out["species_counted"], 1)

    def test_ten_builds_used_ten_times_give_five_builds_at_five_uses(self):
        teams = [[slot("s", item=f"item{i}"), *team(f"f{i}_")[:5]] for i in range(10)]
        out = D.builds_at_equal_uses(teams, 5, np.random.default_rng(0))
        self.assertEqual(out["builds_per_species"], 5.0)

    def test_a_species_used_fewer_than_uses_times_is_not_counted(self):
        teams = [[slot("s"), *team(f"f{i}_")[:5]] for i in range(4)]
        teams += [[slot("z"), *team(f"g{i}_")[:5]] for i in range(10)]
        out = D.builds_at_equal_uses(teams, 5, np.random.default_rng(0))
        self.assertEqual(out["species_counted"], 1)
        self.assertEqual(out["builds_per_species"], 1.0)

    def test_species_restriction_compares_samples_on_the_same_species(self):
        teams = [[slot("s", item=f"item{i}"), slot("z"), *team(f"f{i}_")[:4]] for i in range(10)]
        out = D.builds_at_equal_uses(teams, 5, np.random.default_rng(0), species=["z"])
        self.assertEqual((out["builds_per_species"], out["species_counted"]), (1.0, 1))

    def test_shared_species_drops_a_species_only_one_side_uses_enough(self):
        old = D.EQUAL_USES; D.EQUAL_USES = 5
        try:
            a = [[slot("s", item=f"item{i}"), slot("z"), *team(f"a{i}_")[:4]] for i in range(10)]
            b = [[slot("s"), *team(f"b{i}_")[:5]] for i in range(10)]
            model, training, species = D.builds_on_shared_species(a, b, np.random.default_rng(0))
        finally:
            D.EQUAL_USES = old
        self.assertEqual((model, training, species), (5.0, 1.0, 1))

    def test_no_qualifying_species_gives_nan(self):
        out = D.builds_at_equal_uses([team("x")], 5, np.random.default_rng(0))
        self.assertTrue(math.isnan(out["builds_per_species"]))
        self.assertEqual(out["species_counted"], 0)


if __name__ == "__main__":
    np.seterr(all="raise")
    unittest.main()
