"""Nearest training distance (issue #264): how many of a team's 84 fields differ from the closest team in the
model's training data, slots matched to give the fewest. Zero means the team was copied.

Row layout (src/team84.py, src/encode.py, src/stat_points.py): columns 0..47 are six slots of eight fields
(species, ability, item, move, move, move, move, Stat Alignment), slots sorted by species; columns 48..83 are
six Stat Point tokens per slot (HP, Atk, Def, SpA, SpD, Spe) in the same slot order. The four move columns share
one move vocabulary. encode.team_fields sorts a source team's moves by name, but generated rows need not be
sorted, so two slots' moves are compared as multisets: differing move fields = 4 - shared moves.

    .venv/bin/python scripts/nearest_training_distance.py   (from code/) times the measure on the training table
"""
import itertools
import time
from pathlib import Path

import numpy as np

SLOT_COUNT = 6
FIELDS_PER_SLOT_IN_FIRST_48 = 8
FIRST_48_COLUMNS = SLOT_COUNT * FIELDS_PER_SLOT_IN_FIRST_48
STAT_POINTS_PER_SLOT = 6
TEAM_COLUMNS = FIRST_48_COLUMNS + SLOT_COUNT * STAT_POINTS_PER_SLOT          # 84
FIELDS_PER_SLOT = FIELDS_PER_SLOT_IN_FIRST_48 + STAT_POINTS_PER_SLOT         # 14
MOVE_OFFSETS_IN_SLOT = (3, 4, 5, 6)
SINGLE_VALUE_OFFSETS_IN_SLOT = (0, 1, 2, 7)                                  # species, ability, item, Stat Alignment
SLOT_PERMUTATIONS = np.array(list(itertools.permutations(range(SLOT_COUNT))), dtype=np.int64)   # [720, 6]
TRAINING_TABLE = Path.home() / "Documents/code/vgc-team-generator-pilot/results/lossdata84.npz"


def slot_views(rows):
    """Split [N, 84] rows into per-slot single-value fields [N, 6, 10] and per-slot moves [N, 6, 4]."""
    rows = np.asarray(rows, dtype=np.int64)
    first_48 = rows[:, :FIRST_48_COLUMNS].reshape(-1, SLOT_COUNT, FIELDS_PER_SLOT_IN_FIRST_48)
    stat_points = rows[:, FIRST_48_COLUMNS:].reshape(-1, SLOT_COUNT, STAT_POINTS_PER_SLOT)
    single_values = np.concatenate([first_48[:, :, list(SINGLE_VALUE_OFFSETS_IN_SLOT)], stat_points], axis=2)
    moves = first_48[:, :, list(MOVE_OFFSETS_IN_SLOT)]
    return single_values, moves


def slot_cost_matrices(query_row, reference_rows):
    """[B, 6, 6]: fields that differ between query slot i and reference slot j, moves compared as multisets."""
    query_single, query_moves = slot_views(query_row[None])
    reference_single, reference_moves = slot_views(reference_rows)
    single_differences = (query_single[:, :, None, :] != reference_single[:, None, :, :]).sum(-1)
    # Multiset intersection of two 4-move lists: each query move a with copies_in_query[a] copies of its value
    # in its own slot adds min(copies_in_query, copies_in_reference) / copies_in_query, so each distinct value
    # contributes min(count in query, count in reference) in total.
    copies_in_query = (query_moves[0][:, :, None] == query_moves[0][:, None, :]).sum(-1)              # [6, 4]
    copies_in_reference = (query_moves[0][None, :, None, :, None]
                           == reference_moves[:, None, :, None, :]).sum(-1)                           # [B, 6, 6, 4]
    shared_moves = (np.minimum(copies_in_query[None, :, None, :], copies_in_reference)
                    / copies_in_query[None, :, None, :]).sum(-1)
    move_differences = len(MOVE_OFFSETS_IN_SLOT) - np.rint(shared_moves).astype(np.int64)
    return single_differences + move_differences


def exact_team_distances(query_row, reference_rows):
    """[B] true minimum over all 720 one-to-one slot matchings of the summed slot costs."""
    costs = slot_cost_matrices(query_row, reference_rows)
    matched = costs[:, np.arange(SLOT_COUNT)[None, :], SLOT_PERMUTATIONS]                 # [B, 720, 6]
    return matched.sum(-1).min(-1)


# ---------------------------------------------------------------- the pruning bound
# Field groups: species, ability, item, Stat Alignment and each of the six Stat Points are one value per slot
# (six values per team); moves are 24 values per team. Under any slot matching, the slots whose field f agree
# pair equal values of f one to one, so at most |multiset intersection of the two teams' f values| of them
# agree. For moves, the shared moves of each matched slot pair are disjoint sub-multisets of the two teams'
# 24 moves, so their sum is at most the multiset intersection of all 24. Summing over groups:
#     distance >= 84 - sum over groups of |multiset intersection|  =  lower_bound.
# The intersection is a dot product of 0/1 vectors with one column per (group, value, occurrence index):
# a value seen c times in a group lights its occurrences 1..c, and two such vectors share min(c, c') ones.
GROUP_COLUMNS = ([[slot * FIELDS_PER_SLOT_IN_FIRST_48 + offset for slot in range(SLOT_COUNT)]
                  for offset in SINGLE_VALUE_OFFSETS_IN_SLOT]
                 + [[FIRST_48_COLUMNS + slot * STAT_POINTS_PER_SLOT + stat for slot in range(SLOT_COUNT)]
                    for stat in range(STAT_POINTS_PER_SLOT)]
                 + [[slot * FIELDS_PER_SLOT_IN_FIRST_48 + offset for slot in range(SLOT_COUNT)
                     for offset in MOVE_OFFSETS_IN_SLOT]])
OCCURRENCE_LIMIT = 32                                    # more than the 24 values of the largest group


def occurrence_keys(rows, value_limit):
    """[N, 84] int64 keys (group, value, occurrence index), the 0/1 columns of the lower bound; every token
    must be below value_limit."""
    rows = np.asarray(rows, dtype=np.int64)
    keys = []
    for group, columns in enumerate(GROUP_COLUMNS):
        values = np.sort(rows[:, columns], axis=1)
        positions = np.broadcast_to(np.arange(values.shape[1]), values.shape)
        run_starts = np.ones(values.shape, dtype=bool)
        run_starts[:, 1:] = values[:, 1:] != values[:, :-1]
        occurrence = positions - np.maximum.accumulate(np.where(run_starts, positions, 0), axis=1)
        keys.append((group * value_limit + values) * OCCURRENCE_LIMIT + occurrence)
    return np.concatenate(keys, axis=1)


def lower_bounds(query_rows, reference_rows, reference_chunk=8192):
    """[Q, R] uint8 lower bounds on the team distance (proof above), by matrix products in reference chunks."""
    value_limit = int(max(np.max(query_rows), np.max(reference_rows))) + 1
    query_keys = occurrence_keys(query_rows, value_limit)
    reference_keys_all = occurrence_keys(reference_rows, value_limit)
    columns = np.unique(query_keys)                     # only columns some query lights can be shared
    query_matrix = np.zeros((len(query_keys), len(columns)), dtype=np.float32)
    np.put_along_axis(query_matrix, np.searchsorted(columns, query_keys), 1.0, axis=1)
    bounds = np.empty((len(query_keys), len(reference_keys_all)), dtype=np.uint8)
    for start in range(0, len(reference_keys_all), reference_chunk):
        reference_keys = reference_keys_all[start:start + reference_chunk]
        column_ids = np.minimum(np.searchsorted(columns, reference_keys), len(columns) - 1)
        present = columns[column_ids] == reference_keys
        reference_matrix = np.zeros((len(reference_keys), len(columns)), dtype=np.float32)
        row_ids = np.broadcast_to(np.arange(len(reference_keys))[:, None], reference_keys.shape)
        reference_matrix[row_ids[present], column_ids[present]] = 1.0
        shared = np.rint(query_matrix @ reference_matrix.T).astype(np.int64)
        bounds[:, start:start + len(reference_keys)] = TEAM_COLUMNS - shared
    return bounds


# ---------------------------------------------------------------- the measure
def nearest_training_distance(query_rows, reference_rows, exclude_identical_once=False, first_candidates=64):
    """[Q] int64, 0..84: fields of each query team that differ from its closest reference team, slots matched
    to give the fewest. exclude_identical_once is for queries drawn from the reference: a query's first
    distance-0 reference row (itself) is skipped, a second identical copy still counts as 0.

    Exact: each query first scores its first_candidates rows with the smallest lower bound, giving a cutoff
    (the smallest distance found, or the second smallest when excluding). A row whose lower bound is at least
    the cutoff cannot be closer, so scoring every row with a bound below the cutoff finds the true minimum."""
    query_rows = np.asarray(query_rows, dtype=np.int64)
    reference_rows = np.asarray(reference_rows, dtype=np.int64)
    needed = 2 if exclude_identical_once else 1
    if len(reference_rows) < needed:
        raise ValueError(f"need at least {needed} reference rows")
    bounds = lower_bounds(query_rows, reference_rows)
    distances = np.empty(len(query_rows), dtype=np.int64)
    for query_index, query_row in enumerate(query_rows):
        query_bounds = bounds[query_index]
        first_count = min(first_candidates, len(reference_rows))
        first = np.argpartition(query_bounds, first_count - 1)[:first_count]
        found = np.sort(exact_team_distances(query_row, reference_rows[first]))[:needed]
        cutoff = found[needed - 1]
        unchecked = np.ones(len(reference_rows), dtype=bool)
        unchecked[first] = False
        rest = np.flatnonzero(unchecked & (query_bounds < cutoff))
        for start in range(0, len(rest), 4096):
            more = exact_team_distances(query_row, reference_rows[rest[start:start + 4096]])
            found = np.sort(np.concatenate([found, more]))[:needed]
        distances[query_index] = found[1] if exclude_identical_once and found[0] == 0 else found[0]
    return distances


def summarize_nearest_training_distance(distances):
    """Mean, median and the shares of teams 0, 1 to 5, 6 to 20 and over 20 fields from their nearest
    training team."""
    distances = np.asarray(distances)
    return dict(mean=float(distances.mean()), median=float(np.median(distances)),
                share_copied_zero_fields=float(np.mean(distances == 0)),
                share_one_to_five_fields=float(np.mean((distances >= 1) & (distances <= 5))),
                share_six_to_twenty_fields=float(np.mean((distances >= 6) & (distances <= 20))),
                share_over_twenty_fields=float(np.mean(distances > 20)))


def main():
    table = np.load(TRAINING_TABLE)
    labelled_training = ~table["all_val"] & np.isfinite(table["all_W"])
    reference_rows = table["all_X"][labelled_training].astype(np.int64)
    query_rows = reference_rows[:1088]
    started = time.perf_counter()
    distances = nearest_training_distance(query_rows, reference_rows, exclude_identical_once=True)
    print(f"{len(query_rows)} queries x {len(reference_rows)} references: {time.perf_counter() - started:.1f} s")
    print(summarize_nearest_training_distance(distances))


if __name__ == "__main__":
    main()
