"""Novelty within a copied line-up (novelty_within_lineup.py) on hand-built 84-token rows with known answers.
No game data.

    /tmp/vgc-pilot/.venv/bin/python scripts/test_novelty_within_lineup.py   (from code/)
"""
import unittest

import numpy as np

import novelty_within_lineup as W

BLANK_MOVE = 0


def build_row(slots):
    """slots: six (species, ability, item, moves[4], stat_alignment, stat_points[6]) -> one 84-token row."""
    first_48, stat_points = [], []
    for species, ability, item, moves, stat_alignment, points in slots:
        first_48 += [species, ability, item, *moves, stat_alignment]
        stat_points += list(points)
    return np.array(first_48 + stat_points, dtype=np.int64)


def base_slots(species_offset=0):
    """Six distinct Pokémon; species tokens 10..15 (plus species_offset), every other field unique to its slot."""
    return [[10 + slot + species_offset, 20 + slot, 30 + slot, (40 + slot, 50 + slot, 60 + slot, 70 + slot), 80 + slot,
             (11, 11, 11, 11, 11, 11)] for slot in range(6)]


def with_change(slots, slot, field, value):
    changed = [list(s) for s in slots]
    changed[slot][field] = value
    return changed


SPECIES, ABILITY, ITEM, MOVES, STAT_ALIGNMENT, STAT_POINTS = range(6)


def compare(query_slots, reference_slot_lists, self_reference_id=-1):
    index = W.ReferenceIndex(np.stack([build_row(s) for s in reference_slot_lists]))
    return W.compare_team(index, build_row(query_slots), self_reference_id)


class TestReferenceCount(unittest.TestCase):
    def test_counts_only_teams_with_the_same_six_species(self):
        base = base_slots()
        references = [base, with_change(base, 2, ITEM, 99), base_slots(species_offset=1)]
        record = compare(base, references)
        self.assertEqual(record["reference_count"], 2)

    def test_no_shared_line_up_gives_no_within_line_up_measures(self):
        record = compare(base_slots(), [base_slots(species_offset=1)])
        self.assertEqual(record["reference_count"], 0)
        self.assertNotIn("distance_78", record)

    def test_slot_order_does_not_matter(self):
        base = base_slots()
        shuffled = [base[i] for i in (3, 0, 5, 1, 4, 2)]
        record = compare(shuffled, [base])
        self.assertEqual(record["reference_count"], 1)
        self.assertEqual(record["distance_78"], 0)

    def test_leave_self_out(self):
        base = base_slots()
        alone = compare(base, [base, base_slots(species_offset=1)], self_reference_id=0)
        self.assertEqual(alone["reference_count"], 0)
        duplicated = compare(base, [base, base], self_reference_id=0)
        self.assertEqual(duplicated["reference_count"], 1)
        self.assertEqual(duplicated["distance_78"], 0)


class TestClosestDistance(unittest.TestCase):
    def test_distance_to_closest_same_line_up_team(self):
        base = base_slots()
        two_away = with_change(with_change(base, 0, ITEM, 99), 1, ABILITY, 98)
        one_away = with_change(base, 4, STAT_ALIGNMENT, 97)
        record = compare(base, [two_away, one_away])
        self.assertEqual(record["distance_78"], 1)

    def test_exact_copy_is_zero(self):
        base = base_slots()
        self.assertEqual(compare(base, [base])["distance_78"], 0)


class TestFieldTypeBreakdown(unittest.TestCase):
    def test_each_field_type_counted_against_the_closest(self):
        base = base_slots()
        reference = with_change(base, 0, ITEM, 99)
        reference = with_change(reference, 1, ABILITY, 98)
        reference = with_change(reference, 2, MOVES, (40 + 2, 50 + 2, 91, 92))
        reference = with_change(reference, 3, STAT_POINTS, (11, 11, 11, 11, 12, 10))
        record = compare(base, [reference])
        self.assertEqual(record["item_fields"], 1)
        self.assertEqual(record["ability_fields"], 1)
        self.assertEqual(record["stat_alignment_fields"], 0)
        self.assertEqual(record["move_fields"], 2)
        self.assertEqual(record["stat_point_fields"], 2)
        self.assertEqual(record["distance_78"], 6)
        self.assertAlmostEqual(record["points_moved_per_pokemon"], 1 / 6)

    def test_moves_compared_as_sets(self):
        base = base_slots()
        reordered = with_change(base, 0, MOVES, (70, 60, 50, 40))
        self.assertEqual(compare(reordered, [base])["move_fields"], 0)

    def test_blank_moves_count_as_one_shared_blank(self):
        base = with_change(base_slots(), 0, MOVES, (40, 50, 60, BLANK_MOVE))
        reference = with_change(base_slots(), 0, MOVES, (40, 50, BLANK_MOVE, BLANK_MOVE))
        self.assertEqual(compare(base, [reference])["move_fields"], 1)

    def test_points_moved_is_half_the_absolute_difference(self):
        base = with_change(base_slots(), 0, STAT_POINTS, (33, 33, 1, 1, 1, 1))
        reference = with_change(base_slots(), 0, STAT_POINTS, (1, 33, 1, 1, 1, 33))
        record = compare(base, [reference])
        self.assertEqual(record["stat_point_fields"], 2)
        self.assertAlmostEqual(record["points_moved_per_pokemon"], 32 / 6)


class TestPerFieldNovelty(unittest.TestCase):
    def test_new_for_line_up_but_seen_on_species_elsewhere(self):
        base = base_slots()
        query = with_change(base, 0, ITEM, 99)
        other_line_up_with_item = with_change(base_slots(), 0, ITEM, 99)
        other_line_up_with_item[5][SPECIES] = 77                           # same species 10, different line-up
        record = compare(query, [base, other_line_up_with_item])
        self.assertEqual(record["reference_count"], 1)
        self.assertEqual(record["new_item_for_line_up"], 1 / 6)
        self.assertEqual(record["new_item_for_species"], 0)
        self.assertEqual(record["new_build_for_line_up"], 1 / 6)
        self.assertEqual(record["new_build_for_species"], 0)
        self.assertEqual(record["new_stat_points_for_line_up"], 0)

    def test_novelty_looks_at_all_same_line_up_teams_not_only_the_closest(self):
        base = base_slots()
        query = with_change(with_change(base, 0, ITEM, 99), 1, ITEM, 98)
        closest = with_change(base, 0, ITEM, 99)
        far = with_change(with_change(with_change(base, 1, ITEM, 98), 2, ABILITY, 1), 3, ABILITY, 2)
        record = compare(query, [closest, far])
        self.assertEqual(record["item_fields"], 1)
        self.assertEqual(record["new_item_for_line_up"], 0)

    def test_new_stat_points_for_species(self):
        query = with_change(base_slots(), 2, STAT_POINTS, (33, 33, 1, 1, 1, 1))
        record = compare(query, [base_slots()])
        self.assertEqual(record["new_stat_points_for_line_up"], 1 / 6)
        self.assertEqual(record["new_stat_points_for_species"], 1 / 6)

    def test_species_novelty_leaves_self_out(self):
        query = with_change(base_slots(), 0, ITEM, 99)
        record = compare(query, [query, base_slots()], self_reference_id=0)
        self.assertEqual(record["reference_count"], 1)
        self.assertEqual(record["new_item_for_species"], 1 / 6)


class TestTeamCombinations(unittest.TestCase):
    def test_items_each_seen_but_combination_new(self):
        base = base_slots()
        query = with_change(with_change(base, 0, ITEM, 99), 1, ITEM, 98)
        references = [with_change(base, 0, ITEM, 99), with_change(base, 1, ITEM, 98)]
        record = compare(query, references)
        self.assertEqual(record["new_item_for_line_up"], 0)
        self.assertEqual(record["new_item_combination"], 1)
        self.assertEqual(record["new_build_combination"], 1)

    def test_same_items_different_stat_points_is_not_a_new_combination(self):
        base = base_slots()
        query = with_change(base, 0, STAT_POINTS, (33, 33, 1, 1, 1, 1))
        record = compare(query, [base])
        self.assertEqual(record["new_item_combination"], 0)
        self.assertEqual(record["new_build_combination"], 0)
        self.assertEqual(record["distance_78"], 6)                        # all six values differ from 11

    def test_fewest_differing_items_and_builds_over_all_same_line_up_teams(self):
        base = base_slots()
        query = with_change(with_change(base, 0, ITEM, 99), 1, ABILITY, 98)
        three_items_off = with_change(with_change(with_change(base, 2, ITEM, 1), 3, ITEM, 2), 4, ITEM, 3)
        record = compare(query, [three_items_off, base])
        self.assertEqual(record["fewest_differing_items"], 1)                # base differs in slot 0 only
        self.assertEqual(record["fewest_differing_builds"], 2)               # base differs in slots 0 and 1


class TestStatPointTotals(unittest.TestCase):
    def test_total_per_pokemon_and_share_spending_66(self):
        query = with_change(base_slots(), 0, STAT_POINTS, (33, 33, 3, 1, 1, 1))     # values 32, 32, 2: total 66
        record = compare(query, [base_slots()])                                       # others: six tens, total 60
        self.assertAlmostEqual(record["stat_point_total_per_pokemon"], (66 + 5 * 60) / 6)
        self.assertAlmostEqual(record["share_pokemon_spending_66"], 1 / 6)


class TestWithinLineUpBuilds(unittest.TestCase):
    def test_counts_pokemon_whose_build_no_same_line_up_team_gives(self):
        base = base_slots()
        query = with_change(with_change(base, 0, ITEM, 99), 3, MOVES, (43, 53, 63, 91))
        record = compare(query, [base, with_change(base, 5, ABILITY, 98)])
        self.assertEqual(record["within_line_up_new_builds"], 2)
        self.assertEqual((record["copied_whole"], record["recombined"], record["has_new_build"]), (0, 0, 1))

    def test_stat_points_do_not_make_a_new_build(self):
        query = with_change(base_slots(), 0, STAT_POINTS, (33, 33, 1, 1, 1, 1))
        record = compare(query, [base_slots()])
        self.assertEqual(record["within_line_up_new_builds"], 0)
        self.assertEqual((record["copied_whole"], record["recombined"], record["has_new_build"]), (1, 0, 0))

    def test_recombined_when_every_build_is_seen_but_no_team_holds_all_six(self):
        base = base_slots()
        query = with_change(with_change(base, 0, ITEM, 99), 1, ITEM, 98)
        references = [with_change(base, 0, ITEM, 99), with_change(base, 1, ITEM, 98)]
        record = compare(query, references)
        self.assertEqual(record["within_line_up_new_builds"], 0)
        self.assertEqual((record["copied_whole"], record["recombined"], record["has_new_build"]), (0, 1, 0))

    def test_a_build_seen_only_on_another_line_up_is_still_new(self):
        base = base_slots()
        query = with_change(base, 0, ITEM, 99)
        other_line_up = with_change(with_change(base, 0, ITEM, 99), 5, SPECIES, 77)
        self.assertEqual(compare(query, [base, other_line_up])["within_line_up_new_builds"], 1)


class TestStandaloneDistances(unittest.TestCase):
    def test_42_and_36_field_distances_each_take_their_own_closest_team(self):
        base = base_slots()
        three_items_off = with_change(with_change(with_change(base, 0, ITEM, 1), 1, ITEM, 2), 2, ITEM, 3)
        one_item_off_and_stat_points = with_change(with_change(base, 0, ITEM, 1), 4, STAT_POINTS, (1, 1, 1, 1, 1, 1))
        record = compare(base, [three_items_off, one_item_off_and_stat_points])
        self.assertEqual(record["distance_78"], 3)                          # closest on 78 fields: three items off
        self.assertEqual(record["item_fields"], 3)                          # breakdown stays against that team
        self.assertEqual(record["non_stat_point_distance_42"], 1)
        self.assertEqual(record["stat_point_distance_36"], 0)

    def test_42_fields_cover_items_abilities_alignments_and_moves(self):
        base = base_slots()
        reference = with_change(with_change(base, 0, ABILITY, 1), 1, STAT_ALIGNMENT, 2)
        reference = with_change(with_change(reference, 2, MOVES, (91, 92, 62, 72)), 3, STAT_POINTS, (12, 10, 11, 11, 11, 11))
        record = compare(base, [reference])
        self.assertEqual(record["non_stat_point_distance_42"], 4)
        self.assertEqual(record["stat_point_distance_36"], 2)


def headline_ready(record):
    """A summary-test record: every per-team measure zero unless given."""
    filled = {k: 0.0 for k in W.PER_TEAM_MEASURES + W.BUILD_MEASURES}
    filled.update(record)
    return filled


class TestHeadlineSummary(unittest.TestCase):
    def setUp(self):
        self.records = [dict(reference_count=0)] + [headline_ready(r) for r in (
            dict(reference_count=1, within_line_up_new_builds=0, copied_whole=1, non_stat_point_distance_42=0,
                 stat_point_distance_36=6),
            dict(reference_count=3, within_line_up_new_builds=0, recombined=1, non_stat_point_distance_42=2),
            dict(reference_count=7, within_line_up_new_builds=2, has_new_build=1, non_stat_point_distance_42=5),
            dict(reference_count=40, within_line_up_new_builds=6, has_new_build=1, non_stat_point_distance_42=20))]

    def test_three_shares_sum_to_one_over_copied_line_ups(self):
        summary = W.summarize(self.records)
        shares = [summary[f"share_{k}"] for k in ("copied_whole", "recombined", "has_new_build")]
        self.assertEqual(shares, [0.25, 0.25, 0.5])
        self.assertAlmostEqual(sum(shares), 1.0)

    def test_mean_and_distribution_of_within_line_up_new_builds(self):
        summary = W.summarize(self.records)
        self.assertAlmostEqual(summary["mean_within_line_up_new_builds"], 2.0)
        distribution = [summary[f"share_within_line_up_new_builds_{k}"] for k in range(7)]
        self.assertEqual(distribution, [0.5, 0, 0.25, 0, 0, 0, 0.25])
        self.assertAlmostEqual(summary["mean_non_stat_point_distance_42"], 27 / 4)
        self.assertAlmostEqual(summary["mean_stat_point_distance_36"], 6 / 4)

    def test_headlines_split_by_how_many_training_teams_share_the_line_up(self):
        summary = W.summarize(self.records)
        self.assertAlmostEqual(summary["line_up_shared_by_1/share_of_teams"], 1 / 5)
        self.assertEqual(summary["line_up_shared_by_1/share_copied_whole"], 1.0)
        self.assertEqual(summary["line_up_shared_by_2_to_5/share_recombined"], 1.0)
        self.assertAlmostEqual(summary["line_up_shared_by_6_plus/share_of_teams"], 2 / 5)
        self.assertAlmostEqual(summary["line_up_shared_by_6_plus/mean_within_line_up_new_builds"], 4.0)
        self.assertEqual(summary["line_up_shared_by_6_plus/share_within_line_up_new_builds_6"], 0.5)
        for measure in W.HEADLINE_MEASURES:
            self.assertIn(measure, summary)
            for bucket, _, _ in W.LINE_UP_POPULARITY_BUCKETS:
                self.assertIn(f"{bucket}/{measure}", summary)

    def test_headlines_never_sum_stat_points_with_other_fields(self):
        self.assertNotIn("mean_distance_78", W.HEADLINE_MEASURES)
        self.assertIn("mean_stat_point_distance_36", W.HEADLINE_MEASURES)
        self.assertIn("mean_non_stat_point_distance_42", W.HEADLINE_MEASURES)


class TestLowAnchor(unittest.TestCase):
    def test_one_reference_team_repeated_scores_zero_on_every_measure(self):
        base = base_slots()
        references = [base, with_change(base, 0, ITEM, 99), with_change(base, 2, STAT_POINTS, (1, 1, 1, 1, 1, 67)),
                      base_slots(species_offset=1)]
        index = W.ReferenceIndex(np.stack([build_row(s) for s in references]))
        repeated = np.stack([build_row(references[1])] * 5)
        summary = W.summarize(W.measure_sample(index, repeated))
        self.assertEqual(summary["share_copied_line_up"], 1.0)
        self.assertEqual(summary["share_copied_whole"], 1.0)
        novelty_measures = [m for m in W.HEADLINE_MEASURES if m != "share_copied_whole"] + [
            f"mean_{m}" for m in W.PER_TEAM_MEASURES if m not in W.OWN_STAT_POINT_MEASURES]
        for measure in novelty_measures:
            self.assertEqual(summary[measure], 0, measure)


def toy_sampler():
    """Two ordinary species and one mega forme; items 'a' to 'd' plus the mega's stone."""
    tables = dict(items=["a", "b", "c", "d", "stone"], is_mega={"x": False, "y": False, "xmega": True},
                  stone_of={"xmega": "stone"}, base_of={"x": "x", "y": "y", "xmega": "x"},
                  abil={"x": ["ab1", "ab2"], "y": ["ab3"], "xmega": ["ab1"]}, disp={}, species=["x", "y", "xmega"])
    return dict(tables=tables, legal_moves={sp: ["m1", "m2", "m3", "m4", "m5"] for sp in ("x", "y", "xmega")},
                mega_stone_of={"stone": "x"}, natures=["n1", "n2"],
                draw_stat_points=lambda rng: dict(HP=66, Atk=0, Def=0, SpA=0, SpD=0, Spe=0))


class TestRandomSetsOnTrainingLineUps(unittest.TestCase):
    def test_keeps_species_draws_legal_fields_and_distinct_items(self):
        sampler = toy_sampler()
        rng = np.random.default_rng(0)
        for _ in range(50):
            used_items = set()
            slots = [W.random_set_for_species(sampler, rng, sp, used_items) for sp in ("y", "xmega")]
            self.assertEqual([s["species"] for s in slots], ["y", "xmega"])
            self.assertEqual(slots[1]["item"], "stone")
            self.assertNotEqual(slots[0]["item"], "stone")                  # the stone belongs to another species
            self.assertEqual(len(set(slots[0]["moves"])), 4)
            self.assertLessEqual(set(slots[0]["moves"]), set(sampler["legal_moves"]["y"]))

    def test_species_with_fewer_than_four_legal_moves_gets_all_of_them(self):
        sampler = toy_sampler()
        sampler["legal_moves"]["y"] = ["transform"]                        # Ditto's case
        slot = W.random_set_for_species(sampler, np.random.default_rng(0), "y", set())
        self.assertEqual(slot["moves"], ["transform"])

    def test_refill_redraws_rejected_teams_and_counts_them(self):
        verdicts_in_order = iter(["illegal", None, None])
        rows, info = W.refill_line_ups([("x", "y"), ("y", "xmega")], toy_sampler(), np.random.default_rng(1),
                                       validate=lambda slots: next(verdicts_in_order),
                                       encode=lambda slots: [s["species"] for s in slots])
        self.assertEqual(rows, [["x", "y"], ["y", "xmega"]])
        self.assertEqual(info["rejected_by_showdown"], 1)
        self.assertEqual(info["tries"], 3)
        self.assertEqual(info["teams_not_refilled"], 0)

    def test_is_deterministic_under_a_seed(self):
        draw = lambda: W.refill_line_ups([("x", "y")] * 5, toy_sampler(), np.random.default_rng(7),
                                         validate=lambda slots: None, encode=lambda slots: slots)[0]
        self.assertEqual(draw(), draw())


class TestSummaryAndVerdict(unittest.TestCase):
    def test_summary_conditions_on_copied_line_ups_and_splits_by_count(self):
        records = [dict(reference_count=0), dict(reference_count=1, distance_78=0), dict(reference_count=12, distance_78=4)]
        for r in records[1:]:
            r.update({k: 0.0 for k in W.PER_TEAM_MEASURES + W.BUILD_MEASURES if k != "distance_78"})
        summary = W.summarize(records)
        self.assertAlmostEqual(summary["share_copied_line_up"], 2 / 3)
        self.assertAlmostEqual(summary["share_exact_copy_of_all_teams"], 1 / 3)
        self.assertAlmostEqual(summary["share_exact_copy"], 1 / 2)
        self.assertAlmostEqual(summary["mean_distance_78"], 2.0)
        self.assertEqual(summary["median_reference_count"], 6.5)
        self.assertAlmostEqual(summary["count_1/share_of_teams"], 1 / 3)
        self.assertAlmostEqual(summary["count_10_plus/mean_distance_78"], 4.0)
        self.assertTrue(np.isnan(summary["count_2_to_9/mean_distance_78"]))

    def test_verdict_needs_all_seeds_outside_the_floor_with_one_sign(self):
        floor_differences = [{"m": x} for x in np.linspace(-1, 1, 20)]
        model = {s: {"m": 5.0} for s in (1, 2, 3)}
        training = {s: {"m": 0.0} for s in (1, 2, 3)}
        verdict = W.verdicts(floor_differences, model, training)["m"]
        self.assertTrue(verdict["differs"])
        self.assertEqual(verdict["sign"], 1)
        model[2] = {"m": 0.5}
        self.assertFalse(W.verdicts(floor_differences, model, training)["m"]["differs"])


if __name__ == "__main__":
    unittest.main()
