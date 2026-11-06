"""What a Mask Lab mask means and how a team is written, shared by every prompt built from a mask.

The Jev fill (jev_fill, which asks Jev, TypeSafe AI's decision model, one question per hidden field)
and the masked diffusion prompt (issues #199 and #200) both read a mask and write teams as paste text.
Both take these pieces from here, so the two prompts can't drift apart:
- the format name and the rules line a prompt states,
- mask paths ('team[1].moves[2]') parsed and printed,
- the fill order of a whole Pokémon's fields, and the blank slot that stands for one,
- the paste writer, which writes the same text the battle step stores.
"""
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
