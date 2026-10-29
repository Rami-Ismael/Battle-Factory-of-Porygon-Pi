"""Legal option names for one hidden field of a partial team, under the pinned Reg M-B vocabulary.

The DSPy experiment's legal-choice answer types and the Jev decision-model baseline both take
their options from here, so both offer the same choices for every single-field question. A whole
hidden Pokémon differs: DSPy answers it as one structured value whose per-species rules are stated
in descriptions, while Jev asks it field by field from species_options and fillable_species_options.

A partial team is six slots. A slot is None when the whole Pokémon is hidden, otherwise a dict
with species, item, ability, nature and moves, where a hidden field is None. A path names one
hidden field: [i, 'species'], [i, 'ability'], [i, 'item'], [i, 'moves', j] or [i, 'nature'].

Rules (data from data/regmb-vocabulary.json):
- species: no family already on the team. For a whole hidden Pokémon, also four learnable moves
  and an ability with an item no teammate holds (fillable_species_options).
- ability: one of the species'. Fixed item: legal with it. Hidden item: leaves an item free.
- item: legal for the species and ability, held by no teammate, and not the only stone another
  hidden-item Mega needs.
- move: the species learns it and the slot lacks it.
- Stat Alignment: any of the 25.
Species hidden: the other fields get the regulation-wide lists.
"""
from collections.abc import Iterable, Sequence
from typing import Any

from mb_prompt import norm, regulation

Slot = dict[str, Any]           # species, item, ability, nature, moves; a hidden field is None
Team = Sequence[Slot | None]    # six slots; None is a whole hidden Pokémon

FIELD_POOLS = {'species': 'species', 'ability': 'ability', 'item': 'item', 'moves': 'move', 'nature': 'nature'}


def _show(ids: Iterable[str]) -> list[str]:
    names = regulation()['names']
    return sorted({names.get(i, i) for i in ids if i})


def _rules(slot: Slot | None) -> dict[str, Any] | None:
    """The vocabulary entry for a slot's species, or None when the slot or its species is hidden."""
    if not slot or not slot.get('species'):
        return None
    return regulation()['slots'].get(norm(slot['species']))


def _held_items(masked_team: Team) -> set[str]:
    return {norm(s['item']) for s in masked_team if s and s.get('item')}


def _families(masked_team: Team) -> set[str]:
    return {r['base'] for r in map(_rules, masked_team) if r}


def species_options(masked_team: Team) -> list[str]:
    """Every species whose family is not already on the team."""
    reg, families = regulation(), _families(masked_team)
    return _show(s for s in reg['values']['species'] if s and reg['slots'][s]['base'] not in families)


def _completable(species_id: str, held: set[str]) -> bool:
    """True when a slot of this species can still get four moves and an item no teammate holds."""
    rules = regulation()['slots'][species_id]
    return len(rules['moves']) >= 4 and any(set(items) - held for items in rules['abilityItems'].values())


def fillable_species_options(masked_team: Team) -> list[str]:
    """species_options, minus species the rest of the slot could not be filled for.

    A whole hidden Pokémon is filled species first, so the species must leave every later question
    with an option: Ditto learns one move, and a Mega form's only item is its own stone, which a
    teammate may already hold.
    """
    reg, families, held = regulation(), _families(masked_team), _held_items(masked_team)
    return _show(s for s in reg['values']['species']
                 if s and reg['slots'][s]['base'] not in families and _completable(s, held))


def regulation_wide(field: str, masked_team: Team = ()) -> list[str]:
    """The whole regulation's list for a field, minus teammates' items for an item."""
    values = regulation()['values'][FIELD_POOLS[field]]
    if field == 'item':
        held = _held_items(masked_team)
        values = [v for v in values if v not in held]
    return _show(values)


def _ability_options(masked_team: Team, i: int, slot: Slot, rules: dict[str, Any]) -> list[str]:
    """With the item fixed, abilities legal with it; with the item hidden, abilities that leave an item free."""
    item = slot.get('item')
    if item is not None:
        return _show(a for a, items in rules['abilityItems'].items() if norm(item) in items)
    blocked = _held_items(masked_team) | _reserved_items(masked_team, i)
    return _show(a for a, items in rules['abilityItems'].items() if set(items) - blocked)


def _legal_items(slot: Slot, rules: dict[str, Any]) -> set[str]:
    ability = slot.get('ability')
    return {it for a, items in rules['abilityItems'].items() if ability is None or a == norm(ability) for it in items}


def _reserved_items(masked_team: Team, i: int) -> set[str]:
    """Items that another slot with a hidden item can hold and nothing else (a Mega form's own stone)."""
    reserved: set[str] = set()
    for j, slot in enumerate(masked_team):
        rules = _rules(slot)
        if j != i and rules and slot is not None and slot.get('item') is None:
            legal = _legal_items(slot, rules)
            if len(legal) == 1:
                reserved |= legal
    return reserved


def _item_options(masked_team: Team, i: int, rules: dict[str, Any]) -> list[str]:
    slot = masked_team[i]
    assert slot is not None  # rules exist only for a slot whose species is known
    legal = _legal_items(slot, rules)
    return _show(legal - _held_items(masked_team) - _reserved_items(masked_team, i))


def _move_options(slot: Slot, rules: dict[str, Any]) -> list[str]:
    known = {norm(m) for m in slot['moves'] if m}
    return _show(m for m in rules['moves'] if m not in known)


def options(masked_team: Team, path: list[Any]) -> list[str]:
    """The legal option names for one hidden field, sorted by name."""
    i, field = path[0], path[1]
    if field not in FIELD_POOLS:
        raise ValueError(f'no option list for path {path}')
    slot, rules = masked_team[i], _rules(masked_team[i])
    if field == 'nature':
        return regulation_wide('nature')
    if rules is None:  # the species is hidden too, so nothing narrows the list
        return _show(regulation()['values'][FIELD_POOLS[field]])
    assert slot is not None  # rules exist only for a slot whose species is known
    if field == 'species':
        return species_options(masked_team)
    if field == 'ability':
        return _ability_options(masked_team, i, slot, rules)
    if field == 'item':
        return _item_options(masked_team, i, rules)
    return _move_options(slot, rules)
