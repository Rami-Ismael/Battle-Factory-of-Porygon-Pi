"""The diversity-novelty frontier's dominance rule: a point beats another only when it is at least as high on every axis
on each of the three sampling seeds, seed for seed, and higher somewhere. And its elite line-ups that held, per seed.

    python scripts/test_diversity_novelty_frontier.py   (from code/)
"""
import unittest

import repo_paths  # noqa: F401  (src/ first on sys.path)
import diversity_novelty_frontier as F


def pt(arm, line_ups, novel, win=(0.3, 0.3, 0.3)):
    seeds = lambda v: {str(s): x for s, x in zip((1, 2, 3), v)}
    return dict(arm=arm, guidance=2, line_ups_by_seed=seeds(line_ups), novel_share_by_seed=seeds(novel), win_by_seed=seeds(win))


AXES = ("line_ups", "novel_share")


class Dominance(unittest.TestCase):
    def test_higher_on_every_seed_dominates(self):
        self.assertTrue(F.dominates(pt("a", (10, 10, 10), (.5, .5, .5)), pt("b", (9, 9, 9), (.5, .5, .5)), AXES))

    def test_one_seed_lower_does_not(self):
        self.assertFalse(F.dominates(pt("a", (10, 10, 8), (.5, .5, .5)), pt("b", (9, 9, 9), (.5, .5, .5)), AXES))

    def test_equal_everywhere_does_not(self):
        same = pt("a", (9, 9, 9), (.5, .5, .5))
        self.assertFalse(F.dominates(same, pt("b", (9, 9, 9), (.5, .5, .5)), AXES))

    def test_a_trade_leaves_both_on_the_front(self):
        broad, novel = pt("broad", (20, 20, 20), (.3, .3, .3)), pt("novel", (10, 10, 10), (.9, .9, .9))
        worse = pt("worse", (9, 9, 9), (.2, .2, .2))
        on, beaten = F.front([broad, novel, worse], AXES)
        self.assertEqual(sorted(on), ["broad, guidance 2", "novel, guidance 2"])
        self.assertEqual(sorted(beaten["worse, guidance 2"]), ["broad, guidance 2", "novel, guidance 2"])

    def test_a_third_axis_can_rescue_a_point(self):
        narrow_winner = pt("winner", (9, 9, 9), (.2, .2, .2), win=(.4, .4, .4))
        broad = pt("broad", (20, 20, 20), (.9, .9, .9))
        self.assertIn("winner, guidance 2", F.front([narrow_winner, broad], F.AXES)[0])
        self.assertNotIn("winner, guidance 2", F.front([narrow_winner, broad], AXES)[0])

    def test_a_grid_never_rebattled_is_left_out_of_elite_fronts(self):
        held = dict(pt("held", (10, 10, 10), (.5, .5, .5)), elite_held=5, elite_held_by_seed={"1": 5, "2": 5, "3": 5})
        unbattled = dict(pt("unbattled", (20, 20, 20), (.9, .9, .9)), elite_held=None)
        on, _ = F.front([held, unbattled], ("line_ups", "elite_held"))
        self.assertEqual(on, ["held, guidance 2"])


def team(*species):
    return "\n\n".join(f"{s} @ Leftovers\nAbility: Pressure" for s in species)


class EliteHeld(unittest.TestCase):
    A, B, C = ("Garchomp", "Charizard", "Pikachu", "Snorlax", "Gengar", "Lucario"), \
              ("Garchomp", "Charizard", "Pikachu", "Snorlax", "Gengar", "Dragonite"), \
              ("Garchomp", "Charizard", "Pikachu", "Snorlax", "Gengar", "Gyarados")

    def cells(self):
        # seed 1 draws A elite and B below the cut; seed 2 draws A and B elite; seed 3 draws C elite (C did not hold)
        return [dict(seed=1, pastes=[team(*self.A), team(*self.B)], y=[0.6, 0.4]),
                dict(seed=2, pastes=[team(*self.A), team(*self.B)], y=[0.55, 0.7]),
                dict(seed=3, pastes=[team(*self.C)], y=[0.9])]

    def held(self):
        import line_up_yield as LY
        return [list(LY.line_up(team(*x))) for x in (self.A, self.B)]

    def test_each_seed_counts_the_held_line_ups_it_drew_as_elite(self):
        import line_up_yield as LY
        out = F.elite(self.cells(), self.held(), training={LY.line_up(team(*self.A))})
        self.assertEqual(out["elite_held"], 2)
        self.assertEqual(out["elite_held_by_seed"], {"1": 1, "2": 2, "3": 0})
        self.assertEqual(out["novel_elite_held"], 1)
        self.assertEqual(out["novel_elite_held_by_seed"], {"1": 0, "2": 1, "3": 0})

    def test_not_rebattled_gives_no_value(self):
        self.assertEqual(F.elite(self.cells(), None, set()), dict(elite_held=None, novel_elite_held=None))


if __name__ == "__main__":
    unittest.main()
