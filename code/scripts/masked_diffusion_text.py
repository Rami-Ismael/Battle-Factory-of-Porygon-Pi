"""The text a masked diffusion model fills in place, for any Mask Lab mask: single fields (#199), whole
Pokémon and all six (#200).

The model does not answer a prompt. It gets a short header (format, rules, win-rate ask) and the team
as compact JSON (mask_paste.team_json), with each hidden field's value replaced by mask tokens between
its quotes (a Stat Point number has no quotes). It fills only those tokens, so fields that are not
hidden cannot change. No meta teams, legal option lists or example teams are in the text.

A whole hidden Pokémon keeps its JSON object as fixed text: every key, quote, bracket and the six stat
names. Each of its fourteen values becomes slots of its field type's length, 9 + 5 + 6 + 4 * 6 + 3 + 6 * 2
= 59 mask tokens with Masked Qwen3 0.6B. A mask hiding all six is the text for writing a team from scratch.

Every hidden field of one type gets the same number of mask tokens, its slot length: the longest
Reg M-B value of that type in the given tokenizer, never the hidden value's own length, which would
leak the answer. Masked Qwen3 0.6B gives species 9, ability 5, item 6, move 6, Stat Alignment 3 and
each Stat Point number 2. A slot is exactly the text between a value's quotes, which holds only if no
value's tokens cross its quotes; slot_lengths asserts that for the tokenizer it is given.

    in_place_text(task, ask, tokenizer, limit)        the text, where its slots sit, its parts' sizes
    training_target(task, ask, tokenizer, limit, pad) the same text with the real values, padded
    read_filled(text, pad, mask_token)                a filled text back into a team, by a JSON parser

The tokenizer is a Hugging Face fast tokenizer, or anything with its mask_token and
`tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)['offset_mapping']`.
Which pad token fills the rest of a slot, and whether the loss scores it, belongs to #177.

Demo, Masked Qwen3 0.6B's tokenizer on meta grid mask MB158, Alolan Ninetales hidden whole, then all six:
    ~/.local/share/vgc-pilot-runtime/venv-dlm/bin/python code/scripts/masked_diffusion_text.py
Any other mask by source, task and k, such as Alolan Ninetales's item alone:
    ~/.local/share/vgc-pilot-runtime/venv-dlm/bin/python code/scripts/masked_diffusion_text.py MB158 items 1
"""
import json
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Protocol

from mask_paste import (FORMAT_NAME, RULES, STATS, Piece, Team, ValuePath, json_pieces, parse_path, team_from_json,
                        value_text)

VERSION = 'in-place-json-v1-no-meta-teams'      # stored beside every generated team
ASK_LINE = 'Target win rate vs top meta teams: {ask}.'     # the earlier text fine-tunes' wording
TEAM_LABEL = 'Team: '
VOCABULARY = Path(__file__).resolve().parents[1] / 'data/regmb-vocabulary.json'
VOCABULARY_FIELD = {'species': 'species', 'item': 'item', 'ability': 'ability', 'nature': 'nature', 'moves': 'move'}
STAT_POINT_VALUES = range(33)       # each Stat Point number is 0 through 32
PROBE = {'species': 'Pikachu', 'item': 'Light Ball', 'ability': 'Static', 'nature': 'Timid',
         'moves': ['Protect', 'Thunderbolt', 'Volt Switch', 'Fake Out'],
         'statPoints': {'HP': 2, 'Atk': 0, 'Def': 0, 'SpA': 32, 'SpD': 0, 'Spe': 32}}   # context for measuring


class Tokenizer(Protocol):
    mask_token: str

    def __call__(self, text: str, *, add_special_tokens: bool, return_offsets_mapping: bool) -> Any: ...


class TextTooLong(ValueError):
    pass


@dataclass(frozen=True)
class SlotSpan:
    """One hidden value's mask tokens: text[start:end], tokens first_token to first_token + length."""
    path: str           # 'team[3].item', 'team[4].moves[2]' or 'team[5].statPoints.HP'
    field: str          # its field type, which sets length
    start: int
    end: int
    first_token: int
    length: int


@dataclass(frozen=True)
class InPlaceText:
    text: str
    slots: tuple[SlotSpan, ...]
    part_tokens: dict[str, int]     # format and rules, ask, team
    version: str = VERSION


def ask_text(ask: float | None) -> str:
    return 'unknown' if ask is None else f'{ask:.2f}'


def header_parts(ask: float | None) -> dict[str, str]:
    return {'format and rules': f'Format: {FORMAT_NAME}.\nRules: {RULES}.\n', 'ask': ASK_LINE.format(ask=ask_text(ask)) + '\n'}


def value_label(path: ValuePath) -> str:
    text = f'team[{path[0]}].{path[1]}'
    if len(path) > 2:
        text += f'[{path[2]}]' if path[1] == 'moves' else f'.{path[2]}'
    return text


def offsets(tokenizer: Tokenizer, text: str) -> list[tuple[int, int]]:
    return [tuple(o) for o in tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)['offset_mapping']]


def tokens_within(spans: list[tuple[int, int]], start: int, end: int, what: str) -> list[int]:
    """Indices of the tokens inside text[start:end]; refuses a token that crosses either edge."""
    inside = []
    for n, (a, b) in enumerate(spans):
        if a < start < b or a < end < b:
            raise ValueError(f'{what}: a token crosses the edge of its value, so a slot would not be the text '
                             f'between its quotes')
        if start <= a and b <= end and b > a:
            inside.append(n)
    return inside


def render(pieces: list[Piece], write: Any) -> tuple[str, dict[ValuePath, tuple[int, int]]]:
    """Join the JSON pieces, writing each value with write(path, value), and note where each value sits."""
    text, where = '', {}
    for piece in pieces:
        if isinstance(piece, str):
            text += piece
        else:
            path, value = piece
            start = len(text)
            text += write(path, value)
            where[path] = (start, len(text))
    return text, where


def value_tokens(tokenizer: Tokenizer, pieces: list[Piece], prefix: str = '') -> dict[ValuePath, int]:
    """Each value's token count in its place in the JSON (after prefix)."""
    text, where = render(pieces, lambda _, value: value_text(value))
    spans = offsets(tokenizer, prefix + text)
    return {path: len(tokens_within(spans, len(prefix) + a, len(prefix) + b, f'{value_label(path)} = {text[a:b]!r}'))
            for path, (a, b) in where.items()}


def _field_values() -> dict[str, list[Any]]:
    vocabulary = json.loads(VOCABULARY.read_text())
    named = {field: [vocabulary['names'][v] for v in vocabulary['values'][key] if v]
             for field, key in VOCABULARY_FIELD.items()}
    return {**named, 'statPoints': list(STAT_POINT_VALUES)}


def _probe(field: str, value: Any, position: Any) -> list[Piece]:
    mon = json.loads(json.dumps(PROBE))
    if field == 'moves':
        mon['moves'][position] = value
    elif field == 'statPoints':
        mon['statPoints'][position] = value
    else:
        mon[field] = value
    return json_pieces([mon])


@lru_cache(maxsize=8)
def slot_lengths(tokenizer: Tokenizer) -> dict[str, int]:
    """Each field type's slot length: its longest Reg M-B value's tokens, measured in every place the
    type sits in the JSON. Refuses a tokenizer whose tokens cross a value's quotes."""
    positions = {'moves': range(4), 'statPoints': STATS}
    lengths = {}
    for field, values in _field_values().items():
        longest = 0
        for value in values:
            for position in positions.get(field, [None]):
                counts = value_tokens(tokenizer, _probe(field, value, position))
                place = (0, field) if position is None else (0, field, position)
                longest = max(longest, counts[place])
        lengths[field] = longest
    return lengths


def hidden_values(task: dict[str, Any]) -> list[ValuePath]:
    """The value paths a mask hides, in the order the JSON writes them."""
    hidden = set()
    for text in task['masked_paths']:
        path = parse_path(text)
        if len(path) == 1:
            hidden |= {piece[0] for piece in json_pieces([None] * 6)
                       if not isinstance(piece, str) and piece[0][0] == path[0]}
        elif path[1] == 'statPoints':
            hidden |= {(path[0], 'statPoints', stat) for stat in STATS}
        else:
            hidden.add(tuple(path))
    order = [piece[0] for piece in json_pieces(task['masked_team']) if not isinstance(piece, str)]
    return [path for path in order if path in hidden]


def _check_masked_team(task: dict[str, Any], hidden: list[ValuePath]) -> None:
    for path, value in (p for p in json_pieces(task['masked_team']) if not isinstance(p, str)):
        if (value is None) != (path in hidden):
            raise ValueError(f'{value_label(path)}: the masked team and its masked paths disagree')


def _count_parts(spans: list[tuple[int, int]], parts: dict[str, str]) -> dict[str, int]:
    counts, start = {}, 0
    for name, part in parts.items():
        counts[name] = sum(1 for a, b in spans if start <= a < start + len(part) and b > a)
        start += len(part)
    return counts


def in_place_text(task: dict[str, Any], ask: float | None, tokenizer: Tokenizer, limit: int) -> InPlaceText:
    """The text the model fills for one Mask Lab task. Refuses text over limit tokens, naming each part's size."""
    hidden = hidden_values(task)
    _check_masked_team(task, hidden)
    lengths = slot_lengths(tokenizer)
    parts = header_parts(ask)
    team, where = render(json_pieces(task['masked_team']),
                         lambda path, value: tokenizer.mask_token * lengths[path[1]] if path in hidden else value_text(value))
    parts['team'] = TEAM_LABEL + team
    prefix = len(parts['format and rules']) + len(parts['ask']) + len(TEAM_LABEL)
    text = ''.join(parts.values())
    spans = offsets(tokenizer, text)
    part_tokens = _count_parts(spans, parts)
    if len(spans) > limit:
        sizes = ', '.join(f'{name} {n}' for name, n in part_tokens.items())
        raise TextTooLong(f'{len(spans)} tokens, over the limit of {limit}: {sizes}')
    slots = []
    for path in hidden:
        start, end = (prefix + a for a in where[path])
        inside = tokens_within(spans, start, end, value_label(path))
        if len(inside) != lengths[path[1]]:
            raise ValueError(f'{value_label(path)}: {len(inside)} tokens, not {lengths[path[1]]}; '
                             f'{tokenizer.mask_token!r} is not one token')
        slots.append(SlotSpan(value_label(path), path[1], start, end, inside[0], len(inside)))
    return InPlaceText(text, tuple(slots), part_tokens)


def training_target(task: dict[str, Any], ask: float | None, tokenizer: Tokenizer, limit: int, pad: str) -> str:
    """The in-place text with each hidden field's real value in its slot, followed by pad up to the slot length.
    Checked to have the same tokens as the in-place text everywhere outside the slots."""
    built = in_place_text(task, ask, tokenizer, limit)
    original = json_pieces(task['original'])
    real = {piece[0]: piece[1] for piece in original if not isinstance(piece, str)}
    counts = value_tokens(tokenizer, original)
    text, cut = built.text, 0
    for slot, path in zip(built.slots, hidden_values(task), strict=True):
        if counts[path] > slot.length:
            raise ValueError(f'{slot.path} = {real[path]!r}: {counts[path]} tokens, over its slot of {slot.length}')
        filled = value_text(real[path]) + pad * (slot.length - counts[path])
        text = text[:slot.start + cut] + filled + text[slot.end + cut:]
        cut += len(filled) - (slot.end - slot.start)
    _check_target_tokens(built, text, tokenizer)
    return text


def _check_target_tokens(built: InPlaceText, target: str, tokenizer: Tokenizer) -> None:
    masked, filled = offsets(tokenizer, built.text), offsets(tokenizer, target)
    if len(masked) != len(filled):
        raise ValueError(f'the target is {len(filled)} tokens and the in-place text {len(masked)}: a value and its '
                         f'padding did not keep their slot length')
    in_slots = {n for slot in built.slots for n in range(slot.first_token, slot.first_token + slot.length)}
    for n, (a, b) in enumerate(masked):
        c, d = filled[n]
        if n not in in_slots and built.text[a:b] != target[c:d]:
            raise ValueError(f'token {n} outside the slots reads {target[c:d]!r} in the target, not {built.text[a:b]!r}')


def read_filled(text: str, pad: str, mask_token: str) -> Team:
    """A filled in-place text back into a team: drop the header and padding, then parse the JSON."""
    _, label, team = text.partition('\n' + TEAM_LABEL)
    if not label:
        raise ValueError(f'no {TEAM_LABEL.strip()!r} line in the text')
    if mask_token in team:
        raise ValueError('the text still has mask tokens: a slot was not filled')
    return team_from_json(team.replace(pad, ''))


def _demo(tokenizer: Tokenizer, source: str, mask: str, k: int) -> None:
    import jev_fill
    task = next(t for t in jev_fill.tasks('meta') if (t['source'], t['task'], t['k']) == (source, mask, k))
    built = in_place_text(task, 0.55, tokenizer, 1024)
    print(f'Meta grid mask {source}, {mask} k={k}, hides {", ".join(task["masked_paths"])}; version {built.version}')
    print('Slot lengths in Masked Qwen3 0.6B tokens:', slot_lengths(tokenizer))
    print('Tokens per part:', built.part_tokens, 'total', sum(built.part_tokens.values()), 'of 1,024\n')
    marked, cut = built.text, 0
    for slot in built.slots:
        mark = f'⟦{slot.path}: {slot.length} × {tokenizer.mask_token}, tokens {slot.first_token}–{slot.first_token + slot.length - 1}⟧'
        marked = marked[:slot.start + cut] + mark + marked[slot.end + cut:]
        cut += len(mark) - (slot.end - slot.start)
    print('In-place text, slots marked:\n' + marked + '\n')
    print('In-place text as the model reads it:\n' + built.text + '\n')
    target = training_target(task, 0.55, tokenizer, 1024, tokenizer.pad_token)
    print(f'Training target, padded with {tokenizer.pad_token}:\n' + target + '\n')
    assert read_filled(target, tokenizer.pad_token, tokenizer.mask_token) == task['original']
    print('The training target reads back to the unmasked team.\n')


if __name__ == '__main__':
    from transformers import AutoTokenizer
    qwen = AutoTokenizer.from_pretrained('dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1')
    if len(sys.argv) > 1:
        _demo(qwen, sys.argv[1], sys.argv[2], int(sys.argv[3]))
    else:
        _demo(qwen, 'MB158', 'pokemon', 1)
        _demo(qwen, 'MB158', 'pokemon', 6)
