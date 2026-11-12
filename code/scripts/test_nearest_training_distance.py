"""Nearest training distance (nearest_training_distance.py) on hand-built and random 84-token rows. No game data.

    .venv/bin/python scripts/test_nearest_training_distance.py   (from code/)
"""
import itertools
import sys
import unittest
from collections import Counter
from pathlib import Path

import numpy as np

import nearest_training_distance as N


def build_row(slots):
    """slots: six (species, ability, item, moves[4], stat_alignment, stat_points[6]) -> one 84-token row."""
    first_48, stat_points = [], []
    for species, ability, item, moves, stat_alignment, points in slots:
        first_48 += [species, ability, item, *moves, stat_alignment]
        stat_points += list(points)
    return np.array(first_48 + stat_points, dtype=np.int64)


def base_slots():
    return [(10 + s, 20 + s, 30 + s, (40 + s, 50 + s, 60 + s, 70 + s), 80 + s, (s + 1,) * 6) for s in range(6)]


def brute_force_distance(query_row, reference_row):
    """Plain loop over all 720 slot permutations; moves compared as multisets."""
    def slot(row, index):
        first = row[index * 8:(index + 1) * 8]
        points = row[48 + index * 6:48 + (index + 1) * 6]
        return [first[0], first[1], first[2], first[7], *points], Counter(first[3:7].tolist())
    query_slots = [slot(query_row, i) for i in range(6)]
    reference_slots = [slot(reference_row, j) for j in range(6)]
    best = 84
    for permutation in itertools.permutations(range(6)):
        total = 0
        for i, j in enumerate(permutation):
            (query_single, query_moves), (reference_single, reference_moves) = query_slots[i], reference_slots[j]
            total += sum(a != b for a, b in zip(query_single, reference_single, strict=True))
            total += 4 - sum((query_moves & reference_moves).values())
        best = min(best, total)
    return best


def random_rows(generator, count, value_count):
    return generator.integers(0, value_count, size=(count, 84))


class HandBuilt(unittest.TestCase):
    def setUp(self):
        self.team = build_row(base_slots())
        self.filler = build_row([(90 + s, 91, 92, (93, 94, 95, 96), 97, (0,) * 6) for s in range(6)])

    def distance(self, query, references, **options):
        return int(N.nearest_training_distance(query[None], np.stack(references), **options)[0])

    def test_identical_team_is_zero(self):
        self.assertEqual(self.distance(self.team, [self.filler, self.team]), 0)

    def test_one_changed_item_is_one(self):
        slots = base_slots(); species, ability, _, moves, alignment, points = slots[3]
        slots[3] = (species, ability, 999, moves, alignment, points)
        self.assertEqual(self.distance(build_row(slots), [self.team, self.filler]), 1)

    def test_same_slots_in_another_order_is_zero(self):
        self.assertEqual(self.distance(build_row(base_slots()[::-1]), [self.filler, self.team]), 0)

    def test_moves_in_another_order_is_zero(self):
        slots = base_slots(); species, ability, item, moves, alignment, points = slots[0]
        slots[0] = (species, ability, item, moves[::-1], alignment, points)
        self.assertEqual(self.distance(build_row(slots), [self.team]), 0)

    def test_swapped_species_costs_that_slots_best_match(self):
        # Slot 5 becomes a new Pokémon sharing only its item and two moves with old slot 5: species, ability,
        # Stat Alignment, six Stat Points and two moves differ = 11, and no other pairing does better.
        slots = base_slots()
        slots[5] = (500, 501, 35, (45, 55, 502, 503), 504, (9,) * 6)
        query = build_row(slots)
        self.assertEqual(self.distance(query, [self.team, self.filler]), 11)
        self.assertEqual(brute_force_distance(query, self.team), 11)

    def test_exclude_identical_once_with_one_copy(self):
        near = base_slots(); species, ability, _, moves, alignment, points = near[0]
        near[0] = (species, ability, 999, moves, alignment, points)
        references = [self.filler, self.team, build_row(near)]
        self.assertEqual(self.distance(self.team, references), 0)
        self.assertEqual(self.distance(self.team, references, exclude_identical_once=True), 1)

    def test_exclude_identical_once_with_two_copies(self):
        references = [self.team, self.filler, build_row(base_slots()[::-1])]
        self.assertEqual(self.distance(self.team, references, exclude_identical_once=True), 0)

    def test_exclude_keeps_the_nearest_when_no_copy(self):
        query = build_row([(s, s, s, (s, s, s, s), s, (s,) * 6) for s in range(6)])
        expected = min(brute_force_distance(query, self.team), brute_force_distance(query, self.filler))
        self.assertEqual(self.distance(query, [self.team, self.filler], exclude_identical_once=True), expected)

    def test_summary_shares(self):
        summary = N.summarize_nearest_training_distance([0, 0, 3, 6, 20, 21, 84, 5])
        self.assertEqual(summary["share_copied_zero_fields"], 0.25)
        self.assertEqual(summary["share_one_to_five_fields"], 0.25)
        self.assertEqual(summary["share_six_to_twenty_fields"], 0.25)
        self.assertEqual(summary["share_over_twenty_fields"], 0.25)
        self.assertEqual(summary["median"], 5.5)


class AgainstBruteForce(unittest.TestCase):
    def test_pair_distance_matches_brute_force_on_random_rows(self):
        generator = np.random.default_rng(0)
        for value_count in (2, 3, 6):
            queries, references = random_rows(generator, 100, value_count), random_rows(generator, 100, value_count)
            for query, reference in zip(queries, references, strict=True):
                fast = int(N.exact_team_distances(query, reference[None])[0])
                self.assertEqual(fast, brute_force_distance(query, reference))

    def test_lower_bound_never_exceeds_distance(self):
        generator = np.random.default_rng(1)
        queries, references = random_rows(generator, 30, 3), random_rows(generator, 40, 3)
        bounds = N.lower_bounds(queries, references, reference_chunk=7)
        for query_index, query in enumerate(queries):
            self.assertTrue(np.all(bounds[query_index] <= N.exact_team_distances(query, references)))

    def test_pruned_nearest_matches_brute_force(self):
        # References are near copies of each other so many rows tie on the bound; first_candidates=2 forces
        # the pruning cutoff to do the work.
        generator = np.random.default_rng(2)
        base = random_rows(generator, 6, 8)
        references = np.repeat(base, 8, axis=0)
        flips = generator.random(references.shape) < 0.15
        references[flips] = generator.integers(0, 8, size=int(flips.sum()))
        queries = np.concatenate([references[:12], random_rows(generator, 8, 8)])
        brute = np.array([[brute_force_distance(q, r) for r in references] for q in queries])
        fast = N.nearest_training_distance(queries, references, first_candidates=2)
        np.testing.assert_array_equal(fast, brute.min(1))
        excluded = N.nearest_training_distance(queries, references, exclude_identical_once=True, first_candidates=2)
        sorted_brute = np.sort(brute, axis=1)
        expected = np.where(sorted_brute[:, 0] == 0, sorted_brute[:, 1], sorted_brute[:, 0])
        np.testing.assert_array_equal(excluded, expected)


class LayoutMatchesSource(unittest.TestCase):
    def test_constants_match_encode_and_stat_points(self):
        source = Path(__file__).resolve().parents[1] / "src"
        sys.path.insert(0, str(source))
        try:
            import encode
            import stat_points
        except ImportError as error:
            self.skipTest(f"src modules not importable here: {error}")
        self.assertEqual(encode.FIELDS, ["species", "ability", "item", "m0", "m1", "m2", "m3", "nature"])
        self.assertEqual((encode.NSLOT, encode.NF), (N.SLOT_COUNT, N.FIELDS_PER_SLOT_IN_FIRST_48))
        self.assertEqual(stat_points.COLS, N.SLOT_COUNT * N.STAT_POINTS_PER_SLOT)


if __name__ == "__main__":
    unittest.main()
