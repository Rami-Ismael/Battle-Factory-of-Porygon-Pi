"""The species question for a whole-Pokémon mask: a tournament over the legal species.

Jev takes at most 255 options per question, and a regulation has 347 species, so one question cannot
list them all. The legal species (legal_options.fillable_species_options) are shuffled with the run's seeded rng
and cut into near-equal groups of at most 255. Each group is asked as its own question, with each
species' types, base stats and legal abilities as its criteria. The group winners are then asked together
in a final question, and that answer is the species. If a group's request is still over the token limit
(jev_fill.request_for), the cut is made finer until every group fits.
"""
import json
import math
import random
from functools import lru_cache
from typing import Any

import repo_paths
import jev_fill
import legal_options
from mb_prompt import norm, regulation

MAX_OPTIONS = 255


@lru_cache(maxsize=1)
def pokedex() -> dict[str, Any]:
    """data/pokedex.json: {normalised species name: entry}."""
    return json.loads((repo_paths.ROOT / 'data/pokedex.json').read_text())


def _entry(name: str) -> dict[str, Any]:
    """The pokedex entry for a species. A cosmetic forme (Alcremie-Ruby-Swirl) has no types or stats
    of its own, so it takes its base species' entry."""
    entry = pokedex()[norm(name)]
    if 'types' in entry:
        return entry
    return pokedex()[norm(entry['baseSpecies'])]


def species_text(name: str) -> dict[str, Any]:
    """Game facts for one species: its types, base stats and legal abilities."""
    entry = _entry(name)
    names = regulation()['names']
    abilities = regulation()['slots'][norm(name)]['abilities']
    return {'types': entry['types'], 'base_stats': entry['baseStats'],
            'abilities': [names.get(a, a) for a in abilities]}


def _criteria(group: list[str]) -> jev_fill.Criteria:
    return {name: species_text(name) for name in group}


def _cut(items: list[str], n: int) -> list[list[str]]:
    """n groups in order whose sizes differ by at most one."""
    return [items[k * len(items) // n:(k + 1) * len(items) // n] for k in range(n)]


def _fits(group: list[str], fill: jev_fill.Fill, path: jev_fill.FieldPath, instructions: str) -> bool:
    """True when the group is at most MAX_OPTIONS names and its question is under the token limit."""
    if len(group) > MAX_OPTIONS:
        return False
    try:
        jev_fill.request_for(fill, path, instructions, _criteria(group))
    except jev_fill.QuestionTooLarge:
        return False
    return True


def split(names: list[str], rng: random.Random, fill: jev_fill.Fill, path: jev_fill.FieldPath,
          instructions: str) -> list[list[str]]:
    """Shuffled groups of species, each small enough to ask as one question.

    Starts at ceil(len(names) / 255) groups and adds one group at a time until every group fits.
    """
    if not names:
        raise ValueError('no legal species to ask about')
    shuffled = list(names)
    rng.shuffle(shuffled)
    n = math.ceil(len(shuffled) / MAX_OPTIONS)
    groups = _cut(shuffled, n)
    while not all(_fits(group, fill, path, instructions) for group in groups):
        n += 1
        if n > len(shuffled):
            raise jev_fill.QuestionTooLarge(f'{jev_fill.path_text(path)}: one species alone is over the token limit')
        groups = _cut(shuffled, n)
    return groups


def _ask_group(chooser: jev_fill.Chooser, fill: jev_fill.Fill, path: jev_fill.FieldPath, instructions: str,
               group: list[str], number: int, total: int) -> str:
    value = jev_fill.ask_choice(chooser, fill, path, instructions, _criteria(group))
    fill.records[-1]['round'] = f'group {number} of {total}'
    return value


def _ask_final(chooser: jev_fill.Chooser, fill: jev_fill.Fill, path: jev_fill.FieldPath, instructions: str,
               winners: list[str], groups: list[list[str]]) -> str:
    value = jev_fill.ask_choice(chooser, fill, path, instructions, _criteria(winners))
    fill.records[-1]['round'] = 'final'
    fill.records[-1]['groups'] = groups
    return value


@jev_fill.register('species')
def ask_species(chooser: jev_fill.Chooser, fill: jev_fill.Fill, path: jev_fill.FieldPath) -> str:
    instructions = f'Which Pokémon should fill slot {path[0] + 1} to do best against the meta?'
    groups = split(legal_options.fillable_species_options(fill.team), fill.rng, fill, path, instructions)
    winners = [_ask_group(chooser, fill, path, instructions, group, g + 1, len(groups))
               for g, group in enumerate(groups)]
    if len(winners) == 1:
        return winners[0]
    return _ask_final(chooser, fill, path, instructions, winners, groups)
