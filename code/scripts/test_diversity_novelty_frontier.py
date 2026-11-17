"""The diversity-novelty frontier's dominance rule: a point beats another only when it is at least as high on every axis
on each of the three sampling seeds, seed for seed, and higher somewhere.

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


if __name__ == "__main__":
    unittest.main()
