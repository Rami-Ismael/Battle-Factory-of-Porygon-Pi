"""Is the 80-epoch labelled model's output as diverse as its training teams? (plan: notes/Diversity.md and its page)

Five samples of n = 1,088 teams each (17 asks x 64 teams), every team as its 84 fields:
  model    the saved ask-grid teams of gen84_noise_byepoch_ep80, asks 1/24 to 17/24, one sample per seed (1, 2, 3)
  train    training teams (lossdata84 all_X, labelled, not in the 2% validation split), 64 per ask group, a team's
           group being the ask nearest its label, so the label mix matches the model's; one sample per seed
  copied   one training team repeated n times (low anchor: every measure at its minimum)
  random   n random legal teams from hps_generate.one_team (random legal sets, 66 Stat Points placed at random),
           each checked by Showdown (high anchor)
  val      labelled validation-split teams, the same 64 per ask group (real teams the model never trained on)
Noise floor: 20 pairs of disjoint training samples drawn the same way; a measure's floor is the 2.5% to 97.5% range of
the 20 differences. The model differs from its training teams on a measure when all three seed differences share a
sign and each lies outside that range.

Measures per team level (line-up = six species; sets = species, ability, item, nature, moves, 48 fields;
Stat Points = 36 fields; team = all 84). Similarity of two teams at a level is the cosine of their feature sets, every
feature tied to its species, so the same item on two different species is not shared:
  vendi      exp(entropy of the eigenvalues of K/n): the effective number of different teams, 1 to n (Friedman & Dieng)
  dcscore    trace(softmax(K/0.1)) row-wise, 1 to n (Zhu et al. 2025); tau fixed before any run
  distance   1 - mean off-diagonal similarity (embedding diversity, Tevet & Berant 2021)
plus exact counts (distinct and effective line-ups, species, species pairs, sets, Stat Points per Pokémon, teams) and
novelty against the full training set (share of teams whose line-up, 48 fields or 84 fields appear in it).

Added 2026-11-11 (issue #264), on every sample and every floor pair:
  train_all  training teams drawn at random over all win rates (labelled, not validation), one sample per seed
  builds per species at EQUAL_USES uses, also on the species both sides use that often (builds_shared_species)
  Novelty: share of teams 0, 1, 2 or 3+ species swaps from the closest training line-up
  Nearest training distance: fields out of 84 that differ from the closest training team, slots matched
  archetype shares, setters only (ability, move, or a mega stone held by the species it mega-evolves, whose mega form
  has the setting ability)
A sample drawn from the training teams leaves each team out of its own reference once.

    python diversity_vs_training.py run [--out PATH]      (with /tmp/vgc-pilot/.venv/bin/python)
"""
import functools
import hashlib
import itertools
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "src"
POKEDEX = HERE.parent / "data/pokedex.json"
RES = Path.home() / "vgc-data/results"
GEN = RES / "stat_points_84_gen84_noise_byepoch_ep80.json"
OUT = RES / "diversity_vs_training_2026-11-11.json"     # the 11-09 result is diversity_vs_training.json
EQUAL_USES = 30          # every model seed has 36 to 40 species used at least 30 times out of 6,528 slots
TABLE = Path.home() / "Documents/code/vgc-team-generator-pilot/results/lossdata84.npz"
TABLE_SHA = "02463cbd812499e7bd94ffb4faff505318f86e58d2fc41ba57476d21b03e11c7"
GRID, ASKS, PER_ASK, SEEDS, FLOOR_PAIRS, TAU, SEED0 = 24, range(1, 18), 64, (1, 2, 3), 20, 0.1, 20261109
LEVELS = ("lineup", "sets", "statpoints", "team")
STATS = ("HP", "Atk", "Def", "SpA", "SpD", "Spe")


# ---------------------------------------------------------------- measures (pure: a team is six slot dicts)
def features(team, level):
    """Species-anchored features of one team at one level."""
    team_features = set()
    for slot in team:
        species = slot["species"]
        if level in ("lineup", "sets", "team"): team_features.add((species,))
        if level in ("sets", "team"):
            team_features |= {(species, "ability", slot["ability"]), (species, "item", slot["item"]),
                              (species, "nature", slot["nature"])}
            team_features |= {(species, "move", move) for move in slot["moves"]}
        if level in ("statpoints", "team"):
            team_features |= {(species, stat, slot["evs"][stat]) for stat in STATS}
    return team_features


def kernel(teams, level):
    """n x n cosine similarity of the teams' feature sets."""
    features_per_team = [features(team, level) for team in teams]
    column_of_feature = {}
    distinct_features = {feature for team_features in features_per_team for feature in team_features}
    presence = np.zeros((len(features_per_team), len(distinct_features)))
    for row, team_features in enumerate(features_per_team):
        presence[row, [column_of_feature.setdefault(feature, len(column_of_feature)) for feature in team_features]] = 1
    presence /= np.sqrt(presence.sum(1, keepdims=True))
    return np.clip(np.round(presence @ presence.T, 12), 0, 1)


def vendi(similarity):
    eigenvalues = np.clip(np.linalg.eigvalsh(similarity / len(similarity)), 0, None)
    eigenvalues = eigenvalues[eigenvalues > 1e-12]
    return float(np.exp(-(eigenvalues * np.log(eigenvalues)).sum()))


def dcscore(similarity, tau=TAU):
    scaled = (similarity - similarity.max(1, keepdims=True)) / tau
    softmax_rows = np.exp(scaled); softmax_rows /= softmax_rows.sum(1, keepdims=True)
    return float(np.trace(softmax_rows))


def mean_distance(similarity):
    team_count = len(similarity)
    return float(1 - (similarity.sum() - np.trace(similarity)) / (team_count * (team_count - 1)))


def effective(counter):
    frequencies = np.array(list(counter.values()), float); shares = frequencies / frequencies.sum()
    return float(np.exp(-(shares * np.log(shares)).sum()))


def lineup(team): return tuple(sorted(slot["species"] for slot in team))
def set_key(slot): return (slot["species"], slot["ability"], slot["item"], slot["nature"], tuple(sorted(slot["moves"])))
def statpoints_key(slot): return (slot["species"], *tuple(slot["evs"][stat] for stat in STATS))
def key48(team): return tuple(sorted(set_key(slot) for slot in team))
def key84(team): return tuple(sorted(set_key(slot) + statpoints_key(slot)[1:] for slot in team))


def counts(teams):
    lineup_counts = Counter(lineup(team) for team in teams)
    species_counts = Counter(slot["species"] for team in teams for slot in team)
    pairs = Counter(pair for team in teams for pair in itertools.combinations(lineup(team), 2))
    sets = Counter(set_key(slot) for team in teams for slot in team)
    statpoints = Counter(statpoints_key(slot) for team in teams for slot in team)
    top_lineup, top_lineup_count = lineup_counts.most_common(1)[0]
    return dict(distinct_lineups=len(lineup_counts), effective_lineups=effective(lineup_counts),
                largest_lineup_share=max(lineup_counts.values()) / len(teams),
                distinct_species=len(species_counts), effective_species=effective(species_counts),
                distinct_pairs=len(pairs), effective_pairs=effective(pairs),
                distinct_sets=len(sets), sets_per_species=len(sets) / len(species_counts), effective_sets=effective(sets),
                distinct_statpoints=len(statpoints), effective_statpoints=effective(statpoints),
                top_lineup=list(top_lineup), top_lineup_count=top_lineup_count,
                distinct_teams48=len({key48(team) for team in teams}), distinct_teams84=len({key84(team) for team in teams}))


def measure(teams):
    out = counts(teams)
    for level in LEVELS:
        similarity = kernel(teams, level)
        out.update({f"vendi_{level}": vendi(similarity), f"dcscore_{level}": dcscore(similarity),
                    f"distance_{level}": mean_distance(similarity)})
    return out


def novelty(teams, ref):
    team_count = len(teams)
    return dict(lineup_in_train=sum(lineup(team) in ref["lineup"] for team in teams) / team_count,
                sets48_in_train=sum(key48(team) in ref["k48"] for team in teams) / team_count,
                team84_in_train=sum(key84(team) in ref["k84"] for team in teams) / team_count)


def species_swaps(teams, ref_lineups, exclude_self=False):
    """Share of teams at each count of species swaps to the nearest reference line-up (six minus shared species).

    ref_lineups must keep its duplicates (one entry per reference team, never a set). With exclude_self, a reference
    line-up identical to the team's own counts only if it appears at least twice (one copy is the team itself).
    A line-up never repeats a species (species clause), so shared species = size of the intersection.
    """
    copies = Counter(ref_lineups)
    refs = list(copies)
    species = sorted({name for ref in refs for name in ref} | {slot["species"] for team in teams for slot in team})
    column_of_species = {name: column for column, name in enumerate(species)}
    reference_presence = np.zeros((len(refs), len(species)), np.float32)
    for row, ref in enumerate(refs): reference_presence[row, [column_of_species[name] for name in ref]] = 1
    row_of = {ref: row for row, ref in enumerate(refs)}
    buckets = Counter()
    for start in range(0, len(teams), 256):
        chunk = teams[start:start + 256]
        chunk_presence = np.zeros((len(chunk), len(species)), np.float32)
        for row, team in enumerate(chunk): chunk_presence[row, [column_of_species[slot["species"]] for slot in team]] = 1
        shared = chunk_presence @ reference_presence.T
        for row, team in enumerate(chunk):
            own = row_of.get(lineup(team))
            if exclude_self and own is not None and copies[refs[own]] == 1: shared[row, own] = -1
            swaps = 6 - max(int(np.rint(shared[row].max())), 0) if len(refs) else 6
            buckets["swaps_3plus" if swaps >= 3 else f"swaps_{swaps}"] += 1
    team_count = len(teams)
    return {name: buckets[name] / team_count for name in ("swaps_0", "swaps_1", "swaps_2", "swaps_3plus")}


# A set stores its base ability, so a mega stone whose mega form has a setting ability is listed as an item: in the
# training teams 4,925 of 5,202 Charizard holding Charizardite Y store Blaze, not Drought (sample of 20,000, 11-11).
# Mega abilities from code/data/pokedex.json; Meganiumite's Mega Sol is left out (it sets no weather on the field).
ARCHETYPE_SETTERS = {
    "sun": dict(abilities=("Drought", "Orichalcum Pulse", "Desolate Land"), moves=("Sunny Day",), items=("Charizardite Y",)),
    "rain": dict(abilities=("Drizzle", "Primordial Sea"), moves=("Rain Dance",), items=()),
    "sand": dict(abilities=("Sand Stream", "Sand Spit"), moves=("Sandstorm",), items=("Tyranitarite",)),
    "snow": dict(abilities=("Snow Warning",), moves=("Snowscape", "Chilly Reception", "Hail"),
                 items=("Abomasite", "Froslassite")),
    "electric terrain": dict(abilities=("Electric Surge", "Hadron Engine"), moves=("Electric Terrain",),
                             items=("Raichunite X",)),
    "grassy terrain": dict(abilities=("Grassy Surge", "Seed Sower"), moves=("Grassy Terrain",), items=()),
    "misty terrain": dict(abilities=("Misty Surge",), moves=("Misty Terrain",), items=()),
    "psychic terrain": dict(abilities=("Psychic Surge",), moves=("Psychic Terrain",), items=()),
    "Trick Room": dict(abilities=(), moves=("Trick Room",), items=()),
    "Tailwind": dict(abilities=(), moves=("Tailwind",), items=()),
}


def comparable(name):
    """Lowercased name with every character that is not a letter or digit removed."""
    return "".join(character for character in name.lower() if character.isalnum())


@functools.cache
def mega_stone_species():
    """{stone: species that mega-evolve with it}, comparable spellings, read off each mega form's requiredItem in
    code/data/pokedex.json as src/corpus.py MEGA_STONE_OF does. The species are the base species (changesFrom, else
    baseSpecies) and the mega form itself; a regional form such as Raichu-Alola is a different species and is not
    listed, as Showdown does not let it mega-evolve with Raichunite X."""
    holders = {}
    for entry in json.loads(POKEDEX.read_text()).values():
        stone = entry.get("requiredItem")
        base = entry.get("changesFrom") or entry.get("baseSpecies")
        if stone and base and "mega" in comparable(entry.get("forme", "")):
            holders.setdefault(comparable(stone), set()).update({comparable(base), comparable(entry["name"])})
    return holders


def archetype_shares(teams):
    """Share of teams in each archetype (a team with any setting ability or move, or a listed mega stone held by a
    species it mega-evolves, mega_stone_species), plus "none"."""
    stone_species = mega_stone_species()
    setters = {archetype: ({comparable(name) for name in parts["abilities"] + parts["moves"]},
                           {comparable(name) for name in parts["items"]})
               for archetype, parts in ARCHETYPE_SETTERS.items()}
    hits_per_team = Counter()
    for team in teams:
        hits = set()
        for slot in team:
            ability, item = comparable(slot["ability"]), comparable(slot["item"])
            moves = {comparable(move) for move in slot["moves"]}
            holds_own_stone = comparable(slot["species"]) in stone_species.get(item, ())
            for archetype, (words, stones) in setters.items():
                if ability in words or moves & words or (item in stones and holds_own_stone): hits.add(archetype)
        hits_per_team.update(hits or {"none"})
    team_count = len(teams)
    return {name: hits_per_team[name] / team_count for name in [*ARCHETYPE_SETTERS, "none"]}


def uses_per_species(teams):
    return Counter(slot["species"] for team in teams for slot in team)


def builds_at_equal_uses(teams, uses, rng, draws=20, species=None):
    """Distinct builds (set_key) among `uses` sets drawn without replacement, per species used at least `uses` times.

    Averaged over `draws` repeats and then over species, so species with more uses are not favoured. Species are
    visited in sorted order, so the result is reproducible for a given rng. `species` restricts the average to a
    given list, so two samples can be compared on the same species (otherwise the species mix differs between them).
    """
    sets_of = {}
    for team in teams:
        for slot in team: sets_of.setdefault(slot["species"], []).append(set_key(slot))
    allowed = set(sets_of) if species is None else set(species) & set(sets_of)
    qualifying = [name for name in sorted(allowed) if len(sets_of[name]) >= uses]
    if not qualifying: return dict(builds_per_species=float("nan"), species_counted=0)
    per_species = []
    for name in qualifying:
        sets = sets_of[name]
        distinct = [len({sets[position] for position in rng.choice(len(sets), uses, replace=False)})
                    for _ in range(draws)]
        per_species.append(np.mean(distinct))
    return dict(builds_per_species=float(np.mean(per_species)), species_counted=len(qualifying))


# ---------------------------------------------------------------- data
def modules():
    sys.path.insert(0, "/tmp/vgc-pilot/src")
    sys.path.insert(0, str(SRC))
    import lossdown  # noqa: F401  puts /tmp/vgc-pilot/src first again
    sys.path.insert(0, str(SRC))


def nearest_ask(win_rates):
    out = np.full(win_rates.shape, -1, np.int64); labelled = np.isfinite(win_rates)
    out[labelled] = np.floor(win_rates[labelled] * GRID + 0.5).astype(np.int64)
    return out


def stratified(groups, rng, exclude=()):
    """PER_ASK row indices per ask group, without replacement, avoiding `exclude`."""
    picked = []
    for ask in ASKS:
        pool = np.setdiff1d(groups[ask], exclude) if len(exclude) else groups[ask]
        picked.append(rng.choice(pool, min(PER_ASK, len(pool)), replace=False))
    return np.concatenate(picked)


def load():
    modules()
    import team84
    from corpus import parse_team_text
    if hashlib.sha256(TABLE.read_bytes()).hexdigest() != TABLE_SHA: raise SystemExit("lossdata84.npz changed")
    vocabulary = team84.regulation_vocab()
    table = np.load(TABLE)
    table_rows, win_rates, in_validation = table["all_X"], table["all_W"], table["all_val"]
    ask = nearest_ask(win_rates)
    train = np.flatnonzero(~in_validation & np.isfinite(win_rates))
    validation_rows = np.flatnonzero(in_validation & np.isfinite(win_rates))
    tgroups = {ask_index: train[ask[train] == ask_index] for ask_index in ASKS}
    vgroups = {ask_index: validation_rows[ask[validation_rows] == ask_index] for ask_index in ASKS}
    decode_row = lambda row: team84.decode(vocabulary, table_rows[row])
    for row in train[:200]:                                  # decode/encode round trip on real rows
        if not np.array_equal(team84.encode(vocabulary, decode_row(row)), table_rows[row].astype(np.int64)):
            raise SystemExit(f"row {row} does not round-trip")

    def from_paste(paste):
        return team84.decode(vocabulary, team84.encode(vocabulary, parse_team_text(paste)))

    generated = json.load(open(GEN))
    model, dropped = {}, Counter()
    for seed in SEEDS:
        teams = []
        for ask_index in ASKS:
            cell = generated["cells"][f"gen84_noise_byepoch_ep80_a{ask_index:02d}of24_s{seed}"]
            if len(cell["pastes"]) != PER_ASK: raise SystemExit(f"cell a{ask_index}s{seed} has {len(cell['pastes'])} teams")
            for paste in cell["pastes"]:
                try: teams.append(from_paste(paste))
                except ValueError: dropped[f"model_s{seed}"] += 1
        model[seed] = teams
    return team84, vocabulary, table_rows, train, tgroups, vgroups, decode_row, model, dropped


def random_teams(vocabulary, team84, team_count, seed):
    import hps_generate
    rng = np.random.default_rng(seed); tables = hps_generate.build_tables(); validator = hps_generate.Validator()
    out, tries, outside = [], 0, 0
    try:
        while len(out) < team_count and tries < 40 * team_count:
            tries += 1
            slots = hps_generate.one_team(tables, rng)
            if slots is None or validator("\n\n".join(hps_generate.slot_to_text(slot) for slot in slots) + "\n") is not None:
                continue
            try: out.append(team84.decode(vocabulary, team84.encode(vocabulary, slots)))
            except ValueError: outside += 1                # a field outside the Regulation M-B vocabulary
    finally:
        validator.close()
    return out, dict(tries=tries, outside_vocab=outside)


def added_measures(teams, rows, from_training, ref_lineups, ref_rows, rng):
    """The 11-11 additions for one sample, flat: builds per species at equal uses, Novelty shares, Nearest training
    distance summary, archetype shares. from_training: the sample was drawn from the reference, so each team is
    left out of its own reference once (exclude_self, exclude_identical_once)."""
    from nearest_training_distance import nearest_training_distance, summarize_nearest_training_distance
    builds = builds_at_equal_uses(teams, EQUAL_USES, rng)
    out = dict(builds_per_species=builds["builds_per_species"], builds_species_counted=builds["species_counted"])
    out.update({f"novelty_{name}": share for name, share
                in species_swaps(teams, ref_lineups, exclude_self=from_training).items()})
    distances = nearest_training_distance(rows, ref_rows, exclude_identical_once=from_training)
    out.update({f"nearest_{name}": value for name, value in summarize_nearest_training_distance(distances).items()})
    out.update({f"archetype_{name}": share for name, share in archetype_shares(teams).items()})
    return out


def builds_on_shared_species(first_sample, second_sample, rng):
    """Builds per species at EQUAL_USES for two samples, over the species both use at least EQUAL_USES times."""
    first_uses, second_uses = uses_per_species(first_sample), uses_per_species(second_sample)
    common = sorted(name for name in first_uses
                    if first_uses[name] >= EQUAL_USES and second_uses.get(name, 0) >= EQUAL_USES)
    first_builds = builds_at_equal_uses(first_sample, EQUAL_USES, rng, species=common)
    second_builds = builds_at_equal_uses(second_sample, EQUAL_USES, rng, species=common)
    return first_builds["builds_per_species"], second_builds["builds_per_species"], len(common)


def setter_vocabulary(vocabulary, teams_by_sample):
    """Which listed setters exist in the Regulation M-B vocabulary, how often a mega stone holder stores the mega
    form's ability, and how many holders are another species, so their stone does not count (mega_stone_species)."""
    in_vocabulary = {field: {comparable(name) for name in vocabulary.itos[field]} for field in ("ability", "move", "item")}
    field_of_kind = dict(abilities="ability", moves="move", items="item")
    listed = {archetype: {kind: {name: comparable(name) in in_vocabulary[field_of_kind[kind]] for name in parts[kind]}
                          for kind in parts} for archetype, parts in ARCHETYPE_SETTERS.items()}
    mega_ability = {"charizarditey": "drought", "tyranitarite": "sandstream", "abomasite": "snowwarning",
                    "froslassite": "snowwarning", "raichunitex": "electricsurge"}
    stored = {}
    for sample_name, teams in teams_by_sample.items():
        held = Counter(); same = Counter(); another_species = Counter()
        for team in teams:
            for slot in team:
                stone = comparable(slot["item"])
                if stone in mega_ability:
                    held[stone] += 1; same[stone] += comparable(slot["ability"]) == mega_ability[stone]
                    another_species[stone] += comparable(slot["species"]) not in mega_stone_species()[stone]
        stored[sample_name] = {stone: dict(holders=held[stone], stores_mega_ability=same[stone],
                                           holders_of_another_species=another_species[stone]) for stone in sorted(held)}
    return dict(listed_in_vocabulary=listed, mega_stone_holders=stored)


# ---------------------------------------------------------------- run
def run(out_path=OUT):
    team84, vocabulary, table_rows, train, tgroups, vgroups, decode_row, model, dropped = load()
    team_count = len(ASKS) * PER_ASK
    rng = np.random.default_rng(SEED0)
    rng_added = np.random.default_rng(SEED0 + 1111)     # the additions draw here, so the 11-09 samples are unchanged
    ref = dict(lineup=set(), k48=set(), k84=set())
    ref_lineups = []
    for row in train:
        team = decode_row(row); ref["lineup"].add(lineup(team)); ref["k48"].add(key48(team)); ref["k84"].add(key84(team))
        ref_lineups.append(lineup(team))
    ref_rows = table_rows[train].astype(np.int64)
    print(f"training reference: {len(train)} teams, {len(ref['lineup'])} line-ups", flush=True)

    samples, rows_of = {f"model_s{seed}": model[seed] for seed in SEEDS}, {}
    for seed in SEEDS:
        picked = stratified(tgroups, rng)
        samples[f"train_s{seed}"] = [decode_row(row) for row in picked]; rows_of[f"train_s{seed}"] = picked
    copied_row = rng.choice(train)
    samples["copied"] = [decode_row(copied_row)] * team_count; rows_of["copied"] = np.full(team_count, copied_row)
    picked = stratified(vgroups, rng); samples["val"] = [decode_row(row) for row in picked]; rows_of["val"] = picked
    samples["random"], random_info = random_teams(vocabulary, team84, team_count, SEED0)
    for seed in SEEDS:                                  # training teams drawn over all win rates, not ask-matched
        picked = np.random.default_rng(SEED0 + 100 + seed).choice(train, team_count, replace=False)
        samples[f"train_all_s{seed}"] = [decode_row(row) for row in picked]; rows_of[f"train_all_s{seed}"] = picked
    from_training = {name for name in samples if name.startswith("train") or name == "copied"}
    out = dict(n=team_count, tau=TAU, sizes={name: len(teams) for name, teams in samples.items()}, dropped=dict(dropped),
               random=random_info, equal_uses=EQUAL_USES,
               self_excluded=dict(samples=sorted(from_training), note=(
                   "Rows drawn from the training teams leave each team out of its own reference once: Novelty "
                   "(species_swaps exclude_self) and Nearest training distance (exclude_identical_once). The 11-09 "
                   "novelty block does not exclude.")),
               measures={}, novelty={}, floor={}, verdict={}, verdict_all_win_rates={}, builds_shared_species={})
    for name, teams in samples.items():
        rows = (table_rows[rows_of[name]].astype(np.int64) if name in rows_of
                else np.stack([team84.encode(vocabulary, team) for team in teams]))
        out["measures"][name] = measure(teams)
        out["measures"][name].update(added_measures(teams, rows, name in from_training, ref_lineups, ref_rows, rng_added))
        if not name.startswith("train"): out["novelty"][name] = novelty(teams, ref)
        measures = out["measures"][name]
        print(f"{name:12s} n={len(teams)} vendi team {measures['vendi_team']:.1f} line-up {measures['vendi_lineup']:.1f} "
              f"builds/species {measures['builds_per_species']:.2f} nearest {measures['nearest_mean']:.1f}", flush=True)
    for seed in SEEDS:
        for other in (f"train_s{seed}", f"train_all_s{seed}"):
            model_builds, training_builds, species_count = builds_on_shared_species(
                samples[f"model_s{seed}"], samples[other], rng_added)
            out["builds_shared_species"][f"model_s{seed}_vs_{other}"] = dict(
                model=model_builds, training=training_builds, diff=model_builds - training_builds, species=species_count)
    out["setter_vocabulary"] = setter_vocabulary(vocabulary, samples)

    floor_differences = []
    for pair in range(FLOOR_PAIRS):
        first_rows = stratified(tgroups, rng); second_rows = stratified(tgroups, rng, exclude=first_rows)
        first_teams, second_teams = [decode_row(row) for row in first_rows], [decode_row(row) for row in second_rows]
        first_measures, second_measures = measure(first_teams), measure(second_teams)
        first_measures.update(added_measures(first_teams, table_rows[first_rows].astype(np.int64), True,
                                             ref_lineups, ref_rows, rng_added))
        second_measures.update(added_measures(second_teams, table_rows[second_rows].astype(np.int64), True,
                                              ref_lineups, ref_rows, rng_added))
        difference = {key: first_measures[key] - second_measures[key] for key in first_measures
                      if not isinstance(first_measures[key], list)}
        first_builds, second_builds, _ = builds_on_shared_species(first_teams, second_teams, rng_added)
        difference["builds_shared_species"] = first_builds - second_builds
        floor_differences.append(difference)
        print(f"floor pair {pair + 1}/{FLOOR_PAIRS}", flush=True)
    verdicts(out, floor_differences)
    Path(out_path).write_text(json.dumps(out, indent=1, default=str))
    print(f"wrote {out_path}")


def verdicts(out, floor_differences):
    """Noise floor per measure from the training pair differences, then the model's verdict against the ask-matched
    training rows (verdict) and the all-win-rate training rows (verdict_all_win_rates)."""
    for key in floor_differences[0]:
        floor_values = np.array([difference[key] for difference in floor_differences])
        low, high = np.percentile(floor_values, [2.5, 97.5])
        out["floor"][key] = dict(lo=float(low), hi=float(high), sd=float(floor_values.std(ddof=1)))
        for verdict_name, training_prefix in (("verdict", "train_s"), ("verdict_all_win_rates", "train_all_s")):
            if key == "builds_shared_species":
                seed_differences = [out["builds_shared_species"][f"model_s{seed}_vs_{training_prefix}{seed}"]["diff"]
                                    for seed in SEEDS]
            else:
                seed_differences = [out["measures"][f"model_s{seed}"][key] - out["measures"][f"{training_prefix}{seed}"][key]
                                    for seed in SEEDS]
            outside = all(value < low or value > high for value in seed_differences)
            same_sign = all(value > 0 for value in seed_differences) or all(value < 0 for value in seed_differences)
            out[verdict_name][key] = dict(diffs=seed_differences, differs=bool(outside and same_sign),
                                          sign=int(np.sign(np.mean(seed_differences))) if same_sign else 0)
            if key != "builds_shared_species":
                copied, random_anchor, training = (out["measures"][name][key]
                                                   for name in ("copied", "random", f"{training_prefix}1"))
                out[verdict_name][key]["anchors_ordered"] = bool(min(copied, random_anchor) < training
                                                                 < max(copied, random_anchor))


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args or args[0] != "run" or len(args) not in (1, 3) or (len(args) == 3 and args[1] != "--out"):
        raise SystemExit(__doc__)
    run(Path(args[2]) if len(args) == 3 else OUT)
