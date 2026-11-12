"""Novelty within a copied line-up (issue #264 follow-up): when a team's six species are an exact copy of a training
line-up, how else does the team differ from the training teams with that line-up?

Samples are those of diversity_vs_training.py, redrawn with its seeds: the default model's teams per sampling seed
(model_s1..3), training teams matched to the asks (train_s1..3) and training teams drawn over all win rates
(train_all_s1..3: 1,088 rows without replacement from the labelled non-validation training teams, seed 20261109 +
100 + s). The reference is those 170,785 training teams. A training team is left out of its own reference.

Every team is its 84 tokens (src/team84.py), slots sorted by species. Two teams with the same line-up therefore align
slot by slot by species, so no matching is needed. Per team whose line-up appears in the reference:
  reference_count             same-line-up reference teams
  distance_78                 fields out of the 78 non-species fields that differ from the closest same-line-up team
                              (ties: the lowest reference row), split into item_fields (of 6), ability_fields (6),
                              stat_alignment_fields (6), move_fields (24; per Pokémon 4 minus the shared moves,
                              compared as multisets of four tokens so a blank move matches a blank) and
                              stat_point_fields (36); points_moved_per_pokemon is half the summed absolute Stat Point
                              difference, averaged over the six Pokémon
  new_*_for_line_up           share of the six Pokémon whose item, build (ability, item, Stat Alignment, four moves)
                              or exact six Stat Points no same-line-up reference team gives that species
  new_*_for_species           the same against that species anywhere in the reference
  new_item_combination        1 if no same-line-up reference team holds the same six items on the same species
  new_build_combination       the same for the six builds
  fewest_differing_items      0 to 6: items that differ from the same-line-up team sharing the most of them
  fewest_differing_builds     the same for builds. The training table holds no two teams with the same 48 fields,
                              so for a training team distance_78 >= 1 and new_build_combination = 1 by construction;
                              these two counts are the informative versions
  stat_point_total_per_pokemon, share_pokemon_spending_66: the team's own Stat Point totals, which separate points
                              placed differently from points left unspent in points_moved_per_pokemon
Added 2026-11-11, second pass. A build is a set without its Stat Points: item, ability, four moves, Stat Alignment.
  within_line_up_new_builds   0 to 6: Pokémon whose build no same-line-up reference team gives that species
  copied_whole                1 if one same-line-up reference team holds all six builds (Stat Points may differ)
  recombined                  1 if every build appears in some same-line-up team but no single team holds all six
  has_new_build               1 if within_line_up_new_builds >= 1; the three are exclusive and sum to 1. No two
                              training teams share all 48 non-Stat-Point fields, so a training team (itself left
                              out) is never copied_whole; it splits between recombined and has_new_build
  non_stat_point_distance_42  fields of the 42 (items, abilities, Stat Alignments, moves) that differ from the
                              same-line-up team closest on those 42 fields
  stat_point_distance_36      Stat Point fields of 36 that differ from the same-line-up team closest on those 36.
                              Each distance takes its own closest team; distance_78 is kept but is not a headline,
                              because it sums Stat Points with the other field types
Headlines (HEADLINE_MEASURES, over copied-line-up teams): mean_within_line_up_new_builds with its distribution
share_within_line_up_new_builds_0..6, share_copied_whole, share_recombined, share_has_new_build,
mean_non_stat_point_distance_42, mean_stat_point_distance_36. Each is also split by reference_count into
line_up_shared_by_1, line_up_shared_by_2_to_5 and line_up_shared_by_6_plus (keys "<bucket>/<headline>").
Anchors, measured against the whole reference with nothing left out:
  anchor_copied_one_reference_team         diversity_vs_training's copied training team repeated 1,088 times: 0 on
                                           every novelty measure, share_copied_whole 1 (low anchor)
  anchor_random_sets_on_training_line_ups  the line-ups of train_s1, each Pokémon given a random legal set for its
                                           species (hps_generate.sample_slot with the species fixed), every team
                                           checked by Showdown (high anchor)
Noise floor and verdict as in diversity_vs_training.py: 20 pairs of disjoint ask-matched training draws, the 2.5% to
97.5% range of their differences; the model differs on a measure when all three seed differences share a sign and
lie outside that range. headline_verdict and headline_verdict_all_win_rates hold the headline keys only.

    python scripts/novelty_within_lineup.py run [--out PATH]      (with /tmp/vgc-pilot/.venv/bin/python, from code/)
"""
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

SLOT_COUNT = 6
FIELDS_PER_SLOT_IN_FIRST_48 = 8
FIRST_48_COLUMNS = SLOT_COUNT * FIELDS_PER_SLOT_IN_FIRST_48
STAT_POINTS_PER_SLOT = 6
SPECIES_OFFSET, ABILITY_OFFSET, ITEM_OFFSET, STAT_ALIGNMENT_OFFSET = 0, 1, 2, 7
MOVE_OFFSETS = (3, 4, 5, 6)
SEEDS = (1, 2, 3)
FLOOR_PAIRS = 20
SEED0 = 20261109
RESULTS = Path.home() / "vgc-data/results"
OUT = RESULTS / "novelty_within_lineup_2026-11-11.json"
DIVERSITY_RESULT = RESULTS / "diversity_vs_training_2026-11-11.json"
# Cut points on reference_count, chosen before the run from the owner's example (one training team versus 500).
REFERENCE_COUNT_BUCKETS = (("count_1", 1, 1), ("count_2_to_9", 2, 9), ("count_10_plus", 10, None))
PER_TEAM_MEASURES = ("distance_78", "item_fields", "ability_fields", "stat_alignment_fields", "move_fields",
                     "stat_point_fields", "points_moved_per_pokemon",
                     "new_item_for_line_up", "new_build_for_line_up", "new_stat_points_for_line_up",
                     "new_item_for_species", "new_build_for_species", "new_stat_points_for_species",
                     "new_item_combination", "new_build_combination", "fewest_differing_items",
                     "fewest_differing_builds", "stat_point_total_per_pokemon", "share_pokemon_spending_66")
STAT_POINT_TOKEN_OFFSET = 1                             # a Stat Point token is the value plus one (src/stat_points.py)
FULL_STAT_POINT_BUDGET = 66
OWN_STAT_POINT_MEASURES = ("stat_point_total_per_pokemon", "share_pokemon_spending_66")   # the team itself, no reference
# Added 2026-11-11 (second pass). Per-team build measures; HEADLINE_MEASURES are the summary keys the report leads with.
BUILD_MEASURES = ("within_line_up_new_builds", "copied_whole", "recombined", "has_new_build",
                  "non_stat_point_distance_42", "stat_point_distance_36")
HEADLINE_MEASURES = ("mean_within_line_up_new_builds", "share_copied_whole", "share_recombined", "share_has_new_build",
                     "mean_non_stat_point_distance_42", "mean_stat_point_distance_36")
# Same-line-up teams in the team's reference (a training team's own row left out), as fixed in the task brief.
LINE_UP_POPULARITY_BUCKETS = (("line_up_shared_by_1", 1, 1), ("line_up_shared_by_2_to_5", 2, 5),
                              ("line_up_shared_by_6_plus", 6, None))
REFILL_SEED = SEED0 + 2222
ITEM_DRAW_TRIES = 40                                    # as in hps_generate.sample_slot
TEAM_REFILL_TRIES = 40


def slot_column(slot, offset):
    return slot * FIELDS_PER_SLOT_IN_FIRST_48 + offset


SPECIES_COLUMNS = [slot_column(s, SPECIES_OFFSET) for s in range(SLOT_COUNT)]
ABILITY_COLUMNS = [slot_column(s, ABILITY_OFFSET) for s in range(SLOT_COUNT)]
ITEM_COLUMNS = [slot_column(s, ITEM_OFFSET) for s in range(SLOT_COUNT)]
STAT_ALIGNMENT_COLUMNS = [slot_column(s, STAT_ALIGNMENT_OFFSET) for s in range(SLOT_COUNT)]
MOVE_COLUMNS = [[slot_column(s, m) for m in MOVE_OFFSETS] for s in range(SLOT_COUNT)]
STAT_POINT_COLUMNS = [[FIRST_48_COLUMNS + s * STAT_POINTS_PER_SLOT + k for k in range(STAT_POINTS_PER_SLOT)]
                      for s in range(SLOT_COUNT)]


# ---------------------------------------------------------------- pure measures on encoded rows
def canonical_rows(rows):
    """[N, 84] rows with slots reordered by species token, so two teams with one line-up align slot by slot."""
    rows = np.asarray(rows, dtype=np.int64)
    order = np.argsort(rows[:, SPECIES_COLUMNS], axis=1, kind="stable")
    first_48 = rows[:, :FIRST_48_COLUMNS].reshape(-1, SLOT_COUNT, FIELDS_PER_SLOT_IN_FIRST_48)
    stat_points = rows[:, FIRST_48_COLUMNS:].reshape(-1, SLOT_COUNT, STAT_POINTS_PER_SLOT)
    first_48 = np.take_along_axis(first_48, order[:, :, None], axis=1)
    stat_points = np.take_along_axis(stat_points, order[:, :, None], axis=1)
    return np.concatenate([first_48.reshape(len(rows), -1), stat_points.reshape(len(rows), -1)], axis=1)


def species_keys(row):
    """Per slot: (species, item), (species, build) and (species, six Stat Points) keys of one canonical row."""
    keys = []
    for slot in range(SLOT_COUNT):
        species = int(row[SPECIES_COLUMNS[slot]])
        moves = tuple(sorted(int(m) for m in row[MOVE_COLUMNS[slot]]))
        build = (int(row[ABILITY_COLUMNS[slot]]), int(row[ITEM_COLUMNS[slot]]), int(row[STAT_ALIGNMENT_COLUMNS[slot]]), moves)
        keys.append(((species, build[1]), (species, build), (species, tuple(int(v) for v in row[STAT_POINT_COLUMNS[slot]]))))
    return keys


class ReferenceIndex:
    """Reference rows grouped by line-up, plus per-species counts of items, builds and Stat Points."""

    def __init__(self, reference_rows):
        self.rows = canonical_rows(reference_rows)
        groups = {}
        for row_id, species in enumerate(map(tuple, self.rows[:, SPECIES_COLUMNS].tolist())):
            groups.setdefault(species, []).append(row_id)
        self.groups = {key: np.array(ids, dtype=np.int64) for key, ids in groups.items()}
        self.species_counts = (Counter(), Counter(), Counter())          # item, build, Stat Points
        for row in self.rows:
            for slot_keys in species_keys(row):
                for counter, key in zip(self.species_counts, slot_keys):
                    counter[key] += 1

    def same_line_up(self, canonical_row, self_reference_id=-1):
        ids = self.groups.get(tuple(canonical_row[SPECIES_COLUMNS].tolist()), np.empty(0, dtype=np.int64))
        return ids[ids != self_reference_id]


def move_differences(query_row, group_rows):
    """[k, 6]: per slot, 4 minus the multiset intersection of the query's and each reference team's four moves."""
    query_moves = query_row[MOVE_COLUMNS]                                                    # [6, 4]
    reference_moves = group_rows[:, MOVE_COLUMNS]                                            # [k, 6, 4]
    copies_in_query = (query_moves[:, :, None] == query_moves[:, None, :]).sum(-1)          # [6, 4]
    copies_in_reference = (query_moves[None, :, :, None] == reference_moves[:, :, None, :]).sum(-1)   # [k, 6, 4]
    shared = (np.minimum(copies_in_query[None], copies_in_reference) / copies_in_query[None]).sum(-1)
    return len(MOVE_OFFSETS) - np.rint(shared).astype(np.int64)


def per_slot_differences(query_row, group_rows):
    """Dict of [k, 6] arrays: differing fields per slot and field type, plus Stat Points moved."""
    stat_point_gap = np.abs(group_rows[:, STAT_POINT_COLUMNS] - query_row[STAT_POINT_COLUMNS][None])   # [k, 6, 6]
    return dict(item_fields=(group_rows[:, ITEM_COLUMNS] != query_row[ITEM_COLUMNS]).astype(np.int64),
                ability_fields=(group_rows[:, ABILITY_COLUMNS] != query_row[ABILITY_COLUMNS]).astype(np.int64),
                stat_alignment_fields=(group_rows[:, STAT_ALIGNMENT_COLUMNS]
                                       != query_row[STAT_ALIGNMENT_COLUMNS]).astype(np.int64),
                move_fields=move_differences(query_row, group_rows),
                stat_point_fields=(stat_point_gap > 0).sum(-1),
                points_moved=stat_point_gap.sum(-1) / 2)


def species_novelty(index, query_row, leave_self_out):
    """Share of the six Pokémon whose item, build and Stat Points the species never has in the reference."""
    own_copy = 1 if leave_self_out else 0
    new = np.zeros(3)
    for slot_keys in species_keys(query_row):
        for kind, (counter, key) in enumerate(zip(index.species_counts, slot_keys)):
            new[kind] += counter[key] - own_copy <= 0
    return dict(zip(("new_item_for_species", "new_build_for_species", "new_stat_points_for_species"), new / SLOT_COUNT))


def compare_team(index, query_row, self_reference_id=-1):
    """Per-team record (module docstring). self_reference_id: the query's own reference row, left out."""
    query_row = canonical_rows(np.asarray(query_row)[None])[0]
    ids = index.same_line_up(query_row, self_reference_id)
    record = dict(reference_count=len(ids))
    if not len(ids):
        return record
    group_rows = index.rows[ids]
    differences = per_slot_differences(query_row, group_rows)
    field_types = ("item_fields", "ability_fields", "stat_alignment_fields", "move_fields", "stat_point_fields")
    totals = sum(differences[name].sum(1) for name in field_types)
    closest = int(np.argmin(totals))
    record["distance_78"] = int(totals[closest])
    record.update({name: int(differences[name][closest].sum()) for name in field_types})
    record["points_moved_per_pokemon"] = float(differences["points_moved"][closest].mean())
    item_same = differences["item_fields"] == 0
    build_same = item_same & (differences["ability_fields"] == 0) & (differences["stat_alignment_fields"] == 0) \
        & (differences["move_fields"] == 0)
    stat_points_same = differences["stat_point_fields"] == 0
    record["new_item_for_line_up"] = float(np.mean(~item_same.any(0)))
    record["new_build_for_line_up"] = float(np.mean(~build_same.any(0)))
    record["new_stat_points_for_line_up"] = float(np.mean(~stat_points_same.any(0)))
    record.update(species_novelty(index, query_row, self_reference_id >= 0))
    record["new_item_combination"] = int(not item_same.all(1).any())
    record["new_build_combination"] = int(not build_same.all(1).any())
    record["fewest_differing_items"] = int((~item_same).sum(1).min())
    record["fewest_differing_builds"] = int((~build_same).sum(1).min())
    totals_per_pokemon = (query_row[STAT_POINT_COLUMNS] - STAT_POINT_TOKEN_OFFSET).sum(1)
    record["stat_point_total_per_pokemon"] = float(totals_per_pokemon.mean())
    record["share_pokemon_spending_66"] = float(np.mean(totals_per_pokemon == FULL_STAT_POINT_BUDGET))
    record.update(build_measures(differences, build_same))
    return record


def build_measures(differences, build_same):
    """BUILD_MEASURES of one team from its [k, 6] differences to the k same-line-up reference teams."""
    within_line_up_new_builds = int((~build_same.any(0)).sum())
    copied_whole = int(build_same.all(1).any())
    non_stat_point_fields = sum(differences[name].sum(1) for name in
                                ("item_fields", "ability_fields", "stat_alignment_fields", "move_fields"))
    return dict(within_line_up_new_builds=within_line_up_new_builds, copied_whole=copied_whole,
                recombined=int(within_line_up_new_builds == 0 and not copied_whole),
                has_new_build=int(within_line_up_new_builds > 0),
                non_stat_point_distance_42=int(non_stat_point_fields.min()),
                stat_point_distance_36=int(differences["stat_point_fields"].sum(1).min()))


def mean_or_nan(values):
    return float(np.mean(values)) if len(values) else float("nan")


def summarize(records):
    """Flat dict of headline numbers for one sample: shares over all teams, the rest over copied line-ups."""
    copied = [r for r in records if r["reference_count"] > 0]
    counts = np.array([r["reference_count"] for r in copied])
    summary = dict(teams=len(records), share_copied_line_up=len(copied) / len(records),
                   median_reference_count=float(np.median(counts)) if len(counts) else float("nan"),
                   share_exact_copy_of_all_teams=sum(r["distance_78"] == 0 for r in copied) / len(records),
                   share_exact_copy=mean_or_nan([r["distance_78"] == 0 for r in copied]))
    for q in (10, 25, 75, 90):
        summary[f"reference_count_percentile_{q}"] = float(np.percentile(counts, q)) if len(counts) else float("nan")
    summary.update({f"mean_{m}": mean_or_nan([r[m] for r in copied]) for m in PER_TEAM_MEASURES})
    for name, low, high in REFERENCE_COUNT_BUCKETS:
        inside = in_bucket(copied, low, high)
        summary[f"{name}/share_of_teams"] = len(inside) / len(records)
        summary[f"{name}/share_exact_copy"] = mean_or_nan([r["distance_78"] == 0 for r in inside])
        summary.update({f"{name}/mean_{m}": mean_or_nan([r[m] for r in inside]) for m in PER_TEAM_MEASURES})
    summary.update(headline_numbers(copied))
    for name, low, high in LINE_UP_POPULARITY_BUCKETS:
        inside = in_bucket(copied, low, high)
        summary[f"{name}/share_of_teams"] = len(inside) / len(records)
        summary.update({f"{name}/{k}": v for k, v in headline_numbers(inside).items()})
    return summary


def in_bucket(records, low, high):
    return [r for r in records if r["reference_count"] >= low and (high is None or r["reference_count"] <= high)]


def headline_numbers(copied):
    """HEADLINE_MEASURES plus the 0-to-6 distribution of within-line-up new builds, over copied-line-up records."""
    numbers = dict(mean_within_line_up_new_builds=mean_or_nan([r["within_line_up_new_builds"] for r in copied]),
                   share_copied_whole=mean_or_nan([r["copied_whole"] for r in copied]),
                   share_recombined=mean_or_nan([r["recombined"] for r in copied]),
                   share_has_new_build=mean_or_nan([r["has_new_build"] for r in copied]),
                   mean_non_stat_point_distance_42=mean_or_nan([r["non_stat_point_distance_42"] for r in copied]),
                   mean_stat_point_distance_36=mean_or_nan([r["stat_point_distance_36"] for r in copied]))
    for count in range(SLOT_COUNT + 1):
        numbers[f"share_within_line_up_new_builds_{count}"] = mean_or_nan(
            [r["within_line_up_new_builds"] == count for r in copied])
    return numbers


# ---------------------------------------------------------------- high anchor: training line-ups, random legal sets
def random_set_for_species(sampler, rng, species, used_items):
    """One random legal set for a given species: hps_generate.sample_slot with the species fixed instead of drawn.
    Item from the corpus items (a mega forme gets its stone; another species' stone is skipped), four distinct moves
    from the species' learnset (all of them when it has fewer, as Ditto does; hps_generate leaves such species out), an ability from its legal list, a nature, and 66 Stat Points placed at random.
    Adds the item to used_items (Item Clause); None when no free item was found."""
    tables = sampler["tables"]
    if tables["is_mega"][species]:
        item = tables["stone_of"][species]
        if item in used_items:
            return None
    else:
        for _ in range(ITEM_DRAW_TRIES):
            item = tables["items"][rng.integers(0, len(tables["items"]))]
            if item in used_items:
                continue
            stone_species = sampler["mega_stone_of"].get(item)
            if stone_species is not None and stone_species != tables["base_of"][species]:
                continue
            break
        else:
            return None
    legal_moves = sampler["legal_moves"][species]
    move_count = min(len(MOVE_OFFSETS), len(legal_moves))      # Ditto knows only Transform
    moves = [legal_moves[i] for i in rng.choice(len(legal_moves), move_count, replace=False)]
    abilities = tables["abil"][species]
    ability = abilities[rng.integers(0, len(abilities))]
    display = tables["disp"]
    used_items.add(item)
    return dict(species=display.get(species, species), item=display.get(item, item),
                ability=display.get(ability, ability), nature=sampler["natures"][rng.integers(0, len(sampler["natures"]))],
                moves=[display.get(m, m) for m in moves], evs=sampler["draw_stat_points"](rng))


def refill_line_ups(line_ups, sampler, rng, validate, encode):
    """Per line-up, six random legal sets on its species, redrawn until validate(slots) returns None (legal) and
    encode(slots) succeeds, at most TEAM_REFILL_TRIES times. Returns the encoded teams and the counts of each outcome."""
    teams, outcomes = [], Counter(tries=0, dead_draw=0, rejected_by_showdown=0, outside_vocabulary=0,
                                  teams_not_refilled=0)
    for line_up in line_ups:
        for _ in range(TEAM_REFILL_TRIES):
            outcomes["tries"] += 1
            used_items = set()
            slots = [random_set_for_species(sampler, rng, species, used_items) for species in line_up]
            if any(slot is None for slot in slots):
                outcomes["dead_draw"] += 1
                continue
            if validate(slots) is not None:
                outcomes["rejected_by_showdown"] += 1
                continue
            try:
                teams.append(encode(slots))
                break
            except ValueError:
                outcomes["outside_vocabulary"] += 1
        else:
            outcomes["teams_not_refilled"] += 1
    return teams, dict(outcomes)


def random_sets_on_training_line_ups(T, V, line_ups):
    """The high anchor's rows: each line-up refilled with random legal sets, every team checked by Showdown."""
    import hps_generate as H
    from corpus import MEGA_STONE_OF, NATURES
    sampler = dict(tables=H.build_tables(), legal_moves=H.TRUE, mega_stone_of=MEGA_STONE_OF, natures=NATURES,
                   draw_stat_points=H.sample_spread)
    validator = H.Validator()
    try:
        teams, outcomes = refill_line_ups(
            line_ups, sampler, np.random.default_rng(REFILL_SEED),
            validate=lambda slots: validator("\n\n".join(H.slot_to_text(s) for s in slots) + "\n"),
            encode=lambda slots: T.encode(V, slots))
    finally:
        validator.close()
    return np.stack(teams), outcomes


def verdicts(floor_differences, model_by_seed, training_by_seed):
    """Per measure: the floor range, the three model-minus-training differences, and whether they differ."""
    out = {}
    for measure in floor_differences[0]:
        values = np.array([d[measure] for d in floor_differences], dtype=float)
        values = values[np.isfinite(values)]
        differences = [model_by_seed[s][measure] - training_by_seed[s][measure] for s in SEEDS]
        if len(values) < len(floor_differences) or not np.all(np.isfinite(differences)):
            out[measure] = dict(diffs=differences, floor_pairs_defined=len(values), differs=False, sign=0)
            continue
        low, high = np.percentile(values, [2.5, 97.5])
        outside = all(x < low or x > high for x in differences)
        same_sign = all(x > 0 for x in differences) or all(x < 0 for x in differences)
        out[measure] = dict(floor_lo=float(low), floor_hi=float(high), diffs=differences,
                            differs=bool(outside and same_sign), sign=int(np.sign(np.mean(differences))) if same_sign else 0)
    return out


# ---------------------------------------------------------------- run
def measure_sample(index, rows, self_reference_ids=None):
    if self_reference_ids is None:
        self_reference_ids = np.full(len(rows), -1)
    return [compare_team(index, row, int(own)) for row, own in zip(rows, self_reference_ids)]


def reference_count_histogram(records):
    return dict(sorted(Counter(min(r["reference_count"], 1000) for r in records).items()))


def sanity_checks(summaries):
    """Copied-line-up share against novelty_swaps_0 and exact-copy share against nearest_share_copied_zero_fields."""
    earlier = json.loads(DIVERSITY_RESULT.read_text())["measures"]
    checks = {}
    for name, summary in summaries.items():
        swaps_0, copied_zero = earlier[name]["novelty_swaps_0"], earlier[name]["nearest_share_copied_zero_fields"]
        checks[name] = dict(share_copied_line_up=summary["share_copied_line_up"], novelty_swaps_0=swaps_0,
                            line_up_matches=abs(summary["share_copied_line_up"] - swaps_0) < 1e-12,
                            share_exact_copy_of_all_teams=summary["share_exact_copy_of_all_teams"],
                            nearest_share_copied_zero_fields=copied_zero,
                            exact_copy_matches=abs(summary["share_exact_copy_of_all_teams"] - copied_zero) < 1e-12)
    return checks


def draw_samples(rng, train, tgroups, vgroups):
    """The ask-matched and all-win-rate training rows of diversity_vs_training.run, drawn in its order, leaving rng
    where its noise-floor pairs start; plus the row of its copied anchor."""
    import diversity_vs_training as D
    n = len(D.ASKS) * D.PER_ASK
    rows_of = {f"train_s{s}": D.stratified(tgroups, rng) for s in SEEDS}
    copied_anchor_row = int(rng.choice(train))
    D.stratified(vgroups, rng)                            # the validation sample
    for s in SEEDS:
        rows_of[f"train_all_s{s}"] = np.random.default_rng(SEED0 + 100 + s).choice(train, n, replace=False)
    return rows_of, copied_anchor_row


def headline_keys(summary):
    """Every summary key built from a HEADLINE_MEASURES entry, its distribution, or a popularity bucket of them."""
    names = set(HEADLINE_MEASURES) | {f"share_within_line_up_new_builds_{k}" for k in range(SLOT_COUNT + 1)}
    return [key for key in summary if key.split("/")[-1] in names
            or (key.endswith("/share_of_teams") and key.startswith("line_up_shared_by"))]


def run(out_path=OUT):
    started = time.perf_counter()
    import diversity_vs_training as D
    T, V, X, train, tgroups, vgroups, decode_row, model, dropped = D.load()
    index = ReferenceIndex(X[train])
    print(f"reference: {len(train)} teams, {len(index.groups)} line-ups ({time.perf_counter() - started:.0f} s)", flush=True)
    position_in_reference = lambda rows: np.searchsorted(train, rows)
    rng = np.random.default_rng(SEED0)
    training_rows, copied_anchor_row = draw_samples(rng, train, tgroups, vgroups)
    records = {f"model_s{s}": measure_sample(index, np.stack([T.encode(V, t) for t in model[s]])) for s in SEEDS}
    for name, rows in training_rows.items():
        records[name] = measure_sample(index, X[rows], position_in_reference(rows))
    sample_size = len(training_rows["train_s1"])
    # Anchors are measured against the whole reference, nothing left out.
    records["anchor_copied_one_reference_team"] = measure_sample(index, np.repeat(X[[copied_anchor_row]], sample_size, 0))
    anchor_line_ups = [[slot["species"] for slot in decode_row(row)] for row in training_rows["train_s1"]]
    refilled_rows, refill_outcomes = random_sets_on_training_line_ups(T, V, anchor_line_ups)
    records["anchor_random_sets_on_training_line_ups"] = measure_sample(index, refilled_rows)
    print(f"random-set anchor: {len(refilled_rows)} teams, {refill_outcomes} ({time.perf_counter() - started:.0f} s)",
          flush=True)
    summaries = {name: summarize(r) for name, r in records.items()}
    for name, s in summaries.items():
        print(f"{name:40s} copied line-up {s['share_copied_line_up']:.3f} new builds "
              f"{s['mean_within_line_up_new_builds']:.2f} whole/recombined/new {s['share_copied_whole']:.3f}/"
              f"{s['share_recombined']:.3f}/{s['share_has_new_build']:.3f} distance 42 "
              f"{s['mean_non_stat_point_distance_42']:.2f} 36 {s['mean_stat_point_distance_36']:.2f}", flush=True)
    floor_differences = []
    for pair in range(FLOOR_PAIRS):
        a = D.stratified(tgroups, rng); b = D.stratified(tgroups, rng, exclude=a)
        sa = summarize(measure_sample(index, X[a], position_in_reference(a)))
        sb = summarize(measure_sample(index, X[b], position_in_reference(b)))
        floor_differences.append({m: sa[m] - sb[m] for m in sa})
        print(f"floor pair {pair + 1}/{FLOOR_PAIRS}", flush=True)
    model_by_seed = {s: summaries[f"model_s{s}"] for s in SEEDS}
    floor_check = json.loads(DIVERSITY_RESULT.read_text())["floor"]["novelty_swaps_0"]
    own_floor = np.percentile([d["share_copied_line_up"] for d in floor_differences], [2.5, 97.5])
    verdict = verdicts(floor_differences, model_by_seed, {s: summaries[f"train_s{s}"] for s in SEEDS})
    verdict_all_win_rates = verdicts(floor_differences, model_by_seed, {s: summaries[f"train_all_s{s}"] for s in SEEDS})
    headlines = headline_keys(summaries["model_s1"])
    out = dict(reference_teams=len(train), reference_line_ups=len(index.groups), dropped=dict(dropped),
               reference_count_buckets=REFERENCE_COUNT_BUCKETS, line_up_popularity_buckets=LINE_UP_POPULARITY_BUCKETS,
               headline_measures=HEADLINE_MEASURES, summaries=summaries,
               anchors=dict(anchor_copied_one_reference_team=dict(training_row=copied_anchor_row),
                            anchor_random_sets_on_training_line_ups=dict(line_ups_from="train_s1", seed=REFILL_SEED,
                                                                         outcomes=refill_outcomes)),
               reference_count_histogram={k: reference_count_histogram(r) for k, r in records.items()},
               verdict=verdict, verdict_all_win_rates=verdict_all_win_rates,
               headline_verdict={key: verdict[key] for key in headlines},
               headline_verdict_all_win_rates={key: verdict_all_win_rates[key] for key in headlines},
               sanity=dict(per_sample=sanity_checks({k: v for k, v in summaries.items() if not k.startswith("anchor")}),
                           floor_pairs_match=dict(own=own_floor.tolist(), diversity_vs_training=[floor_check["lo"], floor_check["hi"]],
                                                  matches=bool(np.allclose(own_floor, [floor_check["lo"], floor_check["hi"]])))))
    Path(out_path).write_text(json.dumps(out, indent=1, default=str))
    print(f"wrote {out_path} in {time.perf_counter() - started:.0f} s")


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args or args[0] != "run" or len(args) not in (1, 3) or (len(args) == 3 and args[1] != "--out"):
        raise SystemExit(__doc__)
    run(Path(args[2]) if len(args) == 3 else OUT)
