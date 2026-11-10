"""The position-embedding split report (src/position_split_verdict.py; issues #80, #84, spec #233). Needs only numpy, so CI can run it.

    .venv/bin/python scripts/test_position_split_verdict.py   (from code/)
"""
import unittest

import numpy as np

import repo_paths  # noqa: F401  (src/ first on sys.path)
import position_split_verdict as P

FIRST = P.FIRST_SEED


def grid(got, teams=8, seeds=3, species=20.0, legal=1.0):
    """Every team at ask k scores got(k / 24) exactly; each cell's effective species and legal share are given."""
    return {k: [dict(numerator=k, seed=s, y=[got(k / 24)] * teams, div=dict(effective_species=species), legal=legal)
                for s in range(1, seeds + 1)] for k in P.ASKS}


def compare(**arms):
    return P.compare(arms, np.random.default_rng(0))


class AgainstTheFirstSeed(unittest.TestCase):
    def test_an_arm_equal_to_the_first_seed_differs_by_nothing(self):
        out = compare(**{FIRST: grid(lambda a: 0.4), "pos_slot_learned": grid(lambda a: 0.4)})["pos_slot_learned"]
        for measure in ("win", "effective_species", "error", "legal"):
            with self.subTest(measure=measure):
                d = out[measure]["minus_first_seed"]
                self.assertAlmostEqual(d["difference"], 0.0)
                self.assertLessEqual(d["interval"][0], 1e-12)
                self.assertGreaterEqual(d["interval"][1], -1e-12)


    def test_a_win_rate_gain_and_more_species_show_as_differences_with_intervals_above_zero(self):
        out = compare(**{FIRST: grid(lambda a: 0.3), "pos_slot_fixed": grid(lambda a: 0.4, species=21.5)})["pos_slot_fixed"]
        self.assertAlmostEqual(out["win"]["minus_first_seed"]["difference"], 0.1)
        self.assertGreater(out["win"]["minus_first_seed"]["interval"][0], 0.099)
        self.assertAlmostEqual(out["effective_species"]["minus_first_seed"]["difference"], 1.5)
        self.assertEqual(out["effective_species"]["minus_first_seed"]["cells"], 69)

    def test_an_arm_on_the_line_has_a_lower_error_than_one_off_it(self):
        out = compare(**{FIRST: grid(lambda a: a - 0.1), "pos_slot_learned": grid(lambda a: a)})["pos_slot_learned"]
        self.assertAlmostEqual(out["error"]["value"], 0.0)
        self.assertAlmostEqual(out["error"]["minus_first_seed"]["difference"], -0.01)
        self.assertLess(out["error"]["minus_first_seed"]["interval"][1], 0)

    def test_pooled_win_rate_weights_every_team_the_same(self):
        g = grid(lambda a: 0.0)
        g[1] = [dict(c, y=[1.0] * 32) for c in g[1]]                   # ask 1/24 has four times the teams
        out = compare(**{FIRST: g})[FIRST]
        self.assertAlmostEqual(out["win"]["value"], 96 / (96 + 22 * 24))


class SeedGap(unittest.TestCase):
    def test_the_second_no_split_seed_gives_the_size_of_training_luck(self):
        out = P.seed_gap(compare(**{FIRST: grid(lambda a: 0.40), P.SECOND_SEED: grid(lambda a: 0.42, species=20.3)}))
        self.assertAlmostEqual(out["win"], 0.02)
        self.assertAlmostEqual(out["effective_species"], 0.3)
        self.assertEqual(P.seed_gap(compare(**{FIRST: grid(lambda a: 0.4)})), {})


class Expectations(unittest.TestCase):
    """The expectations written on #80 before any run, against the first seed on the 23 asks pooled."""

    def check(self, win, species=20.0, learned_win=None):
        arms = {FIRST: grid(lambda a: 0.4), "pos_no_slot": grid(lambda a: 0.4 + win, species=species)}
        if learned_win is not None:
            arms["pos_slot_learned"] = grid(lambda a: 0.4 + learned_win)
            arms["pos_slot_fixed"] = arms.pop("pos_no_slot")
        return P.expectations(compare(**arms))

    def test_no_slot_vectors_were_expected_to_lose_at_least_0_05(self):
        self.assertTrue(self.check(-0.06, species=20.5)["pos_no_slot"]["win"]["right"])
        self.assertFalse(self.check(-0.01)["pos_no_slot"]["win"]["right"])
        self.assertTrue(self.check(-0.06, species=20.5)["pos_no_slot"]["effective_species"]["right"])   # same or higher
        self.assertFalse(self.check(-0.06, species=19.0)["pos_no_slot"]["effective_species"]["right"])

    def test_fixed_slot_vectors_were_expected_to_tie_learned_slot_vectors(self):
        self.assertTrue(self.check(0.005, learned_win=0.0)["pos_slot_fixed"]["win"]["right"])
        self.assertFalse(self.check(0.03, learned_win=0.0)["pos_slot_fixed"]["win"]["right"])
        self.assertFalse(self.check(0.0, species=21.0, learned_win=0.0)["pos_slot_fixed"]["effective_species"]["right"])

    def test_every_expectation_names_its_wording_from_the_plan(self):
        out = self.check(-0.06)["pos_no_slot"]
        self.assertEqual(out["win"]["expected"], "loses at least 0.05")
        self.assertEqual(out["chance_right"], 0.85)


class Luck(unittest.TestCase):
    def test_luck_is_the_larger_of_the_new_no_split_gap_and_the_earlier_one(self):
        arms = compare(**{FIRST: grid(lambda a: 0.40), P.SECOND_SEED: grid(lambda a: 0.41, species=20.01)})
        luck = P.luck(arms)
        self.assertAlmostEqual(luck["win"], 0.01)                                # new gap 0.01 beats the earlier 0.0035
        self.assertAlmostEqual(luck["effective_species"], P.EARLIER_GAP["effective_species"])   # earlier 0.07 beats 0.01
        self.assertAlmostEqual(luck["error"], P.EARLIER_GAP["error"])

    def test_before_the_second_seed_is_battled_luck_is_the_earlier_gap(self):
        self.assertEqual(P.luck(compare(**{FIRST: grid(lambda a: 0.4)})), P.EARLIER_GAP)


class VerdictRule(unittest.TestCase):
    """The rule on #80 and in docs/position-embedding-split.md (owner, 2026-11-07)."""

    def verdict(self, **arms):
        base = {FIRST: grid(lambda a: a * 0.8), P.SECOND_SEED: grid(lambda a: a * 0.8)}
        return P.verdict(compare(**base, **arms))

    def test_an_arm_that_wins_beyond_luck_with_species_held_is_a_switch_waiting_on_its_second_seed(self):
        v = self.verdict(pos_slot_learned=grid(lambda a: a * 0.8 + 0.02))
        self.assertTrue(v["arms"]["pos_slot_learned"]["switch_on_win_rate"])
        self.assertEqual(v["outcome"], "provisional switch")
        self.assertEqual(v["choice"], "pos_slot_learned")
        self.assertIn("second seed", v["next"])

    def test_a_species_gain_beyond_luck_switches_while_win_rate_stays_above_minus_0_01(self):
        v = self.verdict(pos_no_slot=grid(lambda a: a * 0.8 - 0.005, species=21.0))
        self.assertTrue(v["arms"]["pos_no_slot"]["switch_on_species"])
        self.assertEqual(v["choice"], "pos_no_slot")
        v = self.verdict(pos_no_slot=grid(lambda a: a * 0.8 - 0.02, species=21.0))
        self.assertFalse(v["arms"]["pos_no_slot"]["switch_on_species"])

    def test_a_gain_inside_luck_is_no_evidence_of_a_gain(self):
        v = self.verdict(pos_slot_learned=grid(lambda a: a * 0.8 + 0.002))
        self.assertEqual(v["outcome"], P.NO_GAIN)
        self.assertIsNone(v["choice"])

    def test_a_switch_fails_its_guard_when_asks_are_followed_worse_beyond_luck(self):
        v = self.verdict(pos_slot_learned=grid(lambda a: 0.5))           # pooled win rate up 0.1, but every ask gets 0.5
        row = v["arms"]["pos_slot_learned"]
        self.assertTrue(row["switch_on_win_rate"])
        self.assertFalse(row["guard"])
        self.assertFalse(row["passes"])

    def test_two_arms_passing_within_luck_of_each_other_take_fixed_slot_vectors(self):
        v = self.verdict(pos_slot_learned=grid(lambda a: a * 0.8 + 0.02), pos_slot_fixed=grid(lambda a: a * 0.8 + 0.0201))
        self.assertEqual(v["choice"], "pos_slot_fixed")
        v = self.verdict(pos_slot_learned=grid(lambda a: a * 0.8 + 0.03), pos_slot_fixed=grid(lambda a: a * 0.8 + 0.0201))
        self.assertIsNone(v["choice"])
        self.assertEqual(v["outcome"], "several pass, not within luck: the owner decides")


class ReportedOnly(unittest.TestCase):
    def test_validation_loss_and_stat_point_shares_come_from_the_training_and_grid_records(self):
        grid_out = dict(summary=dict(stat_points=dict(share_matching_corpus_set=0.037, share_under_four_moves=0.157, pokemon=26496)))
        training = dict(best_val_loss=1.816, best_pass=21, stopped_pass=21, seed=0, net_kw=dict(d=256, pos="no_slot"))
        self.assertEqual(P.reported(grid_out, training), dict(validation_loss=1.816, validation_epoch=21, stopped_epoch=21,
                                                              training_seed=0, position_option="no_slot",
                                                              share_matching_corpus_set=0.037, share_under_four_moves=0.157))

    def test_a_checkpoint_saved_before_position_options_reads_as_no_split(self):
        out = P.reported(dict(summary=dict(stat_points={})), dict(best_val_loss=1.8, seed=0, net_kw=dict(d=256)))
        self.assertEqual(out["position_option"], "per_field")
        self.assertIsNone(out["share_matching_corpus_set"])


if __name__ == "__main__":
    unittest.main()
