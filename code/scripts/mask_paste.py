"""What a Mask Lab mask means and how a team is written, shared by every prompt built from a mask.

The Jev fill (jev_fill, which asks Jev, TypeSafe AI's decision model, one question per hidden field)
and the masked diffusion prompt (issues #199 and #200) both read a mask and write teams as paste text.
Both take these pieces from here, so the two prompts can't drift apart:
- the format name and the rules line a prompt states,
- mask paths ('team[1].moves[2]') parsed and printed,
- the fill order of a whole Pokémon's fields, and the blank slot that stands for one,
- the paste writer, which writes the same text the battle step stores,
- the compact JSON writer and reader, the team form the masked diffusion model fills in place.
"""
import json
import re
from typing import Any

FORMAT_NAME = 'Champions VGC 2026 Reg Set M-B'
RULES = 'doubles, bring 6 pick 4, level 50, Species Clause, Item Clause'
SLOT_ORDER = ('species', 'ability', 'item', 'moves', 'nature', 'statPoints')    # nature is Stat Alignment
STATS = ('HP', 'Atk', 'Def', 'SpA', 'SpD', 'Spe')

Slot = dict[str, Any]           # species, item, ability, moves, nature, statPoints, ivs, level, slot
Team = list[Slot | None]        # six slots; None is a whole hidden Pokémon
FieldPath = list[Any]           # [i], [i, 'item'] or [i, 'moves', j]


def parse_path(text: str) -> FieldPath:
    """'team[1].moves[2]' -> [1, 'moves', 2]; 'team[1]' -> [1]; 'team[1].item' -> [1, 'item']."""
    m = re.fullmatch(r'team\[(\d+)\](?:\.(\w+)(?:\[(\d+)\])?)?', text)
    if not m:
        raise ValueError(f'not a mask path: {text}')
    i, name, j = m.groups()
    return [int(i)] + ([name] if name else []) + ([int(j)] if j is not None else [])


def path_text(path: FieldPath) -> str:
    text = f'team[{path[0]}]'
    if len(path) > 1:
        text += f'.{path[1]}'
    if len(path) > 2:
        text += f'[{path[2]}]'
    return text


def blank_slot(number: int) -> Slot:
    return {'slot': number, 'species': None, 'item': None, 'ability': None, 'moves': [None] * 4, 'nature': None,
            'statPoints': None, 'ivs': dict.fromkeys(STATS, 31), 'level': 50}


def paste(mon: Slot) -> str:
    """One Pokémon as Showdown paste text, as mask_lab_prepare.to_paste writes it."""
    lines = [mon['species'] + (f" @ {mon['item']}" if mon.get('item') else ''), f"Ability: {mon['ability']}",
             'Level: 50']
    spread = ' / '.join(f'{mon["statPoints"][k]} {k}' for k in STATS if mon['statPoints'].get(k))
    if spread:
        lines.append(f'EVs: {spread}')
    lines.append(f"{mon['nature']} Nature")
    lines += [f'- {move}' for move in mon['moves'] if move]
    return '\n'.join(lines)


def team_paste(team: Team) -> str:
    """A whole team as mask_lab_prepare.to_paste writes it, the text the battle step stores."""
    return '\n\n'.join(paste(mon) for mon in team if mon is not None) + '\n'


# The team as compact JSON, the text a masked diffusion model fills in place (#199). One object per
# Pokémon in this key order, no spaces after ',' and ':', none of Jev's slot, ivs and level keys, and
# all six Stat Point numbers, zeros included. The battle step's paste above is not written from this.
JSON_FIELDS = ('species', 'item', 'ability', 'nature', 'moves', 'statPoints')
ValuePath = tuple[Any, ...]     # (i, 'item'), (i, 'moves', j) or (i, 'statPoints', 'HP'): one value in the JSON
Piece = str | tuple[ValuePath, Any]     # fixed JSON text, or one value with its path


def _mon_pieces(i: int, mon: Slot) -> list[Piece]:
    pieces: list[Piece] = []
    for n, name in enumerate(JSON_FIELDS):
        pieces.append(f'{"," if n else ""}"{name}":')
        if name == 'moves':
            for j, move in enumerate(mon['moves']):
                pieces += ['[' if j == 0 else ',', '"', ((i, 'moves', j), move), '"']
            pieces.append(']')
        elif name == 'statPoints':
            points = mon['statPoints']
            for k, stat in enumerate(STATS):
                pieces += [f'{"," if k else "{"}"{stat}":', ((i, 'statPoints', stat), None if points is None else points.get(stat, 0))]
            pieces.append('}')
        else:
            pieces += ['"', ((i, name), mon[name]), '"']
    return pieces


def json_pieces(team: Team) -> list[Piece]:
    """The compact JSON team split into fixed text and values. A string value's quotes are fixed text,
    so the value piece is exactly the text between them; a hidden value is None."""
    pieces: list[Piece] = ['[']
    for i, mon in enumerate(team):
        if mon is None:
            raise NotImplementedError(f'team[{i}] is a whole hidden Pokémon; its JSON object is issue #200')
        pieces += ['{' if i == 0 else ',{', *_mon_pieces(i, mon), '}']
    pieces.append(']')
    return pieces


def value_text(value: Any) -> str:
    """A value as it reads in the JSON: a string without its quotes, a number as digits."""
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)[1:-1]
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    raise ValueError(f'not a team value: {value!r}')


def team_json(team: Team) -> str:
    """A complete team as compact JSON."""
    return ''.join(piece if isinstance(piece, str) else value_text(piece[1]) for piece in json_pieces(team))


def team_from_json(text: str) -> Team:
    """Read compact JSON back into Mask Lab slots, with the fixed IVs and level the JSON leaves out."""
    rows = json.loads(text)
    if not isinstance(rows, list) or len(rows) != 6:
        raise ValueError('a team is a JSON list of six Pokémon')
    team: Team = []
    for number, row in enumerate(rows, 1):
        if not isinstance(row, dict) or tuple(row) != JSON_FIELDS:
            raise ValueError(f'Pokémon {number}: keys must be {", ".join(JSON_FIELDS)} in that order')
        if len(row['moves']) != 4 or tuple(row['statPoints']) != STATS:
            raise ValueError(f'Pokémon {number}: needs four moves and all six Stat Points')
        team.append({'slot': number, **row, 'ivs': dict.fromkeys(STATS, 31), 'level': 50})
    return team
