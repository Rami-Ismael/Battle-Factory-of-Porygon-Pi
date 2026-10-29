"""Question builders for hidden fields that take one Choice question each.

Each builder registers itself in jev_fill.FIELDS under the mask's field name and returns the value
the run takes. Options come from legal_options; each option's description is a JSON object of game
facts, never prose written for the prompt.
"""
import json
from functools import lru_cache
from typing import Any

import repo_paths
import jev_fill
import legal_options
from mb_prompt import norm, regulation


@lru_cache(maxsize=1)
def option_text() -> dict[str, Any]:
    """{'item': {name: effect}, 'ability': {name: effect}} from data/regmb-option-text.json."""
    return json.loads((repo_paths.ROOT / 'data/regmb-option-text.json').read_text())


def slot_label(fill: jev_fill.Fill, path: jev_fill.FieldPath) -> str:
    return f"{jev_fill.slot_at(fill.team, path)['species']} in slot {path[0] + 1}"


@jev_fill.register('item')
def ask_item(chooser: jev_fill.Chooser, fill: jev_fill.Fill, path: jev_fill.FieldPath) -> str:
    text = option_text()['item']
    criteria = {name: {'effect': text[name]} for name in legal_options.options(fill.team, path)}
    instructions = f'Which item should {slot_label(fill, path)} hold to do best against the meta?'
    return jev_fill.ask_choice(chooser, fill, path, instructions, criteria)


@lru_cache(maxsize=1)
def move_table() -> dict[str, Any]:
    """{normalized move name: entry} from data/moves.json."""
    moves = json.loads((repo_paths.ROOT / 'data/moves.json').read_text())
    return {norm(m['name']): m for m in moves.values()}


@lru_cache(maxsize=1)
def spread_menu() -> dict[str, Any]:
    """{'generic': [spread], 'species': {species id: [spread]}} from data/regmb-spread-menu.json."""
    return json.loads((repo_paths.ROOT / 'data/regmb-spread-menu.json').read_text())


def move_facts(move: dict[str, Any]) -> dict[str, Any]:
    return {'type': move['type'], 'category': move['category'], 'power': move['basePower'],
            'accuracy': move['accuracy'], 'priority': move['priority'], 'target': move['target'],
            'effect': move.get('shortDesc') or move.get('desc', '')}


def spread_menu_for(species: str) -> list[dict[str, int]]:
    """The species' own spreads if the menu has them, else its family's, else the generic list."""
    menu, key = spread_menu(), norm(species)
    base = regulation()['slots'].get(key, {}).get('base')
    for name in (key, base):
        if name in menu['species']:
            return menu['species'][name]
    return menu['generic']


@jev_fill.register('ability')
def ask_ability(chooser: jev_fill.Chooser, fill: jev_fill.Fill, path: jev_fill.FieldPath) -> str:
    text = option_text()['ability']
    criteria = {name: {'effect': text[name]} for name in legal_options.options(fill.team, path)}
    instructions = f'Which ability should {slot_label(fill, path)} have to do best against the meta?'
    return jev_fill.ask_choice(chooser, fill, path, instructions, criteria)


@jev_fill.register('moves')
def ask_move(chooser: jev_fill.Chooser, fill: jev_fill.Fill, path: jev_fill.FieldPath) -> str:
    table = move_table()
    criteria = {name: move_facts(table[norm(name)]) for name in legal_options.options(fill.team, path)}
    instructions = f'Which move should {slot_label(fill, path)} use as move {path[2] + 1} to do best against the meta?'
    return jev_fill.ask_choice(chooser, fill, path, instructions, criteria)


@jev_fill.register('nature')
def ask_nature(chooser: jev_fill.Chooser, fill: jev_fill.Fill, path: jev_fill.FieldPath) -> str:
    text = option_text()['nature']
    criteria = {name: text[name] for name in legal_options.options(fill.team, path)}
    instructions = f'Which Stat Alignment should {slot_label(fill, path)} use to do best against the meta?'
    return jev_fill.ask_choice(chooser, fill, path, instructions, criteria)


@jev_fill.register('statPoints')
def ask_stat_points(chooser: jev_fill.Chooser, fill: jev_fill.Fill, path: jev_fill.FieldPath) -> dict[str, int]:
    menu = spread_menu_for(jev_fill.slot_at(fill.team, path)['species'])
    criteria = {f'spread {n + 1}': spread for n, spread in enumerate(menu)}
    instructions = f'Which Stat Point spread should {slot_label(fill, path)} use to do best against the meta?'
    label = jev_fill.ask_choice(chooser, fill, path, instructions, criteria)
    return dict(criteria[label])
