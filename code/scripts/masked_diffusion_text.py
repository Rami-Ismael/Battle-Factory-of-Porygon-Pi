"""The text a masked diffusion model fills in place, for any Mask Lab mask: single fields (#199), whole
Pokémon and all six (#200), and the check that reads a filled text back and validates it (#196).

The model does not answer a prompt. It gets a short header (format, rules, win-rate ask) and the team
as compact JSON (mask_paste.team_json), with each hidden field's value replaced by mask tokens between
its quotes (a Stat Point number has no quotes). It fills only those tokens, so fields that are not
hidden cannot change. No meta teams or legal option lists are in the text; #239's prompts add one fixed example
team (example_text), never one of the mask's own teams.

A whole hidden Pokémon keeps its JSON object as fixed text: every key, quote, bracket and the six stat
names. Each of its fourteen values becomes slots of its field type's length, 9 + 5 + 6 + 4 * 6 + 3 + 6 * 2
= 59 mask tokens with Masked Qwen3 0.6B. A mask hiding all six is the text for writing a team from scratch.

Every hidden field of one type gets the same number of mask tokens, its slot length: the longest
Reg M-B value of that type in the given tokenizer, never the hidden value's own length, which would
leak the answer. Masked Qwen3 0.6B gives species 9, ability 5, item 6, move 6, Stat Alignment 3 and
each Stat Point number 2. A slot is exactly the text between a value's quotes, which holds only if no
value's tokens cross its quotes; slot_lengths asserts that for the tokenizer it is given.

    in_place_text(task, ask, tokenizer, limit)        the text, where its slots sit, its parts' sizes
    in_place_text(..., statement=TASK_STATEMENT)      the same text with a task statement before the team (#238)
    in_place_text(..., example=example_text())        the same text with one fixed example team before the team (#239)
    in_place_text(..., rules=RULES_LINE)              the same text with three team-building rules spelled out (#445)
    in_place_text(..., filler=FILLER_LINE)            the same text with a neutral line of matching token count (#445)
    training_target(task, ask, tokenizer, limit, pad) the same text with the real values, padded
    read_filled(text, pad, mask_token)                a filled text back into a team, by a JSON parser

A chat model (CHAT_TEMPLATES, keyed by its tokenizer's repo) gets every prompt inside its own chat template: the
header as the user turn and the team, slots and all, as the assistant turn, so its masks sit where its instruction
tuning put the answers. iLLaDA 8B Instruct is the first (owner, 2026-11-27).
    check_filled(text, pad, mask_token, validate)     a filled text's legality, with the reason it failed

The tokenizer is a Hugging Face fast tokenizer, or anything with its mask_token and
`tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)['offset_mapping']`.
Which pad token fills the rest of a slot, and whether the loss scores it, belongs to #177.

Demo, Masked Qwen3 0.6B's tokenizer on meta grid mask MB158, Alolan Ninetales hidden whole, then all six:
    ~/.local/share/vgc-pilot-runtime/venv-dlm/bin/python code/scripts/masked_diffusion_text.py
Any other mask by source, task and k, such as Alolan Ninetales's item alone:
    ~/.local/share/vgc-pilot-runtime/venv-dlm/bin/python code/scripts/masked_diffusion_text.py MB158 items 1
"""
import copy
import json
import re
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Protocol

from mask_paste import (FORMAT_NAME, JSON_FIELDS, RULES, STATS, Piece, Team, ValuePath, blank_slot, json_pieces,
                        parse_path, team_from_json, team_json, team_paste, value_text)

VERSION = 'in-place-json-v1-no-meta-teams'      # stored beside every generated team
# #238: one task statement before the team, worded once for every model. It names no model's mask token.
TASK_STATEMENT = ('Task: each run of mask tokens in the team below is one hidden value of its field. Fill it with a '
                  'legal Reg M-B value: a species, item, ability, move, Stat Alignment (nature) or Stat Point number. '
                  'Write names as Showdown spells them. Leave the JSON keys, quotes and commas as they are.\n')
INSTRUCTIONS_VERSION = VERSION + '+task-statement-v1'
# #239: one complete legal team, the same in every prompt, shown before the masked team in the same compact JSON.
EXAMPLE_TEAM = Path(__file__).resolve().parents[1] / 'data/inplace-example-team.json'
EXAMPLE_LABEL = 'Example team: '
EXAMPLE_SUFFIX = '+example-team-v1'
# #445: three team-building rules spelled out in plain words and numbers, and a neutral prose line with no numbers
# or domain terms that has the exact same token count in every small-four tokenizer (39 in DiffuGPT, 43 in Qwen).
RULES_LINE = ("Team rules: each item appears at most once on the team. Each Pokémon's six Stat Points add up to "
              "66 or less. Each Stat Point is a whole number from 0 to 32.\n")
FILLER_LINE = ("Note: morning clouds drift across the quiet farmland toward the old lighthouse. A flickering "
               "lantern sits beside a worn satchel under the tall sycamore. Warm bread rests on the wooden shelf.\n")
RULES_SUFFIX = '+rules-line-v1'
FILLER_SUFFIX = '+filler-line-v1'
# #240: free-text answer layout. Hidden values appear as JSON null inside Team:, and all mask tokens sit in one
# fixed-length answer region after Answer:\n (or the chat assistant turn). Newline avoids BPE merges with '['.
FREETEXT_VERSION = 'freetext-answer-v2-no-meta-teams'
ANSWER_LABEL = 'Answer:\n'
ANSWER_SLACK = 8            # room past worst-case Reg M-B compact JSON: 1 end token + up to 7 formatting tokens
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
    tail: str = ''                  # fixed text after the team's JSON: a chat template's end of turn
    mode: str = 'inplace'           # 'inplace' (one slot per hidden value) or 'freetext' (one answer region)
    answer_length: int | None = None
    answer_prefix: str = ''         # fixed answer-opening prefix (e.g. '[' for iLLaDA to bypass <think>)


@dataclass(frozen=True)
class ChatTemplate:
    """A chat model's turn markers around the prompt text. In-place mode puts the header in the user turn and
    'Team: ' with the masked team in the assistant turn (closed with end). Free-text mode puts header + format +
    'Team: ' in the user turn, opens the assistant turn (plus optional freetext_prefix), and places all mask tokens
    in the assistant turn with no fixed end token."""
    name: str           # stored in the text's version
    user: str           # before the header
    assistant: str      # between the user turn and the assistant turn
    end: str            # after the team in in-place mode
    freetext_prefix: str = ''       # fixed prefix at the start of the free-text assistant turn
    trim_user: bool = True          # whether the tokenizer's chat template strips trailing whitespace on user turn


# iLLaDA 8B Instruct's chat_template.jinja at revision 5769f049: bos_token + role + '\n' + content | trim + eos_token
# for each turn, with bos <[BOS]> and eos <[EOS]> (special_tokens_map.json). No system turn.
# In free-text mode, fixed '[' bypasses learned <think> prose (#240 diagnostic, 2026-11-27).
ILLADA_CHAT = ChatTemplate('illada-chat-v1', '<[BOS]>user\n', '<[EOS]><[BOS]>assistant\n', '<[EOS]>',
                           freetext_prefix='[')
# Masked Qwen3 0.6B (dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1, rev c8d24a3f): trained on chat-templated instruction
# data with enable_thinking=False (<|im_start|>assistant\n<think>\n\n</think>\n\n), evaluated with --apply_chat_template.
QWEN3_CHAT = ChatTemplate('qwen3-chat-v1', '<|im_start|>user\n',
                          '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n', '<|im_end|>\n')
# Masked Qwen2.5 Coder 0.5B (dllm-hub/Qwen2.5-Coder-0.5B-Instruct-diffusion-mdlm-v0.1, rev a284e895): chat template
# supplies the default system turn when omitted and opens <|im_start|>assistant\n.
QWEN25C_CHAT = ChatTemplate(
    'qwen25c-chat-v1',
    '<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n',
    '<|im_end|>\n<|im_start|>assistant\n', '<|im_end|>\n',
)
CHAT_TEMPLATES = {
    'GSAI-ML/iLLaDA-8B-Instruct': ILLADA_CHAT,
    'dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1': QWEN3_CHAT,
    'dllm-hub/Qwen2.5-Coder-0.5B-Instruct-diffusion-mdlm-v0.1': QWEN25C_CHAT,
}


def chat_template(tokenizer: Any) -> ChatTemplate | None:
    """The chat template the in-place text is wrapped in for this tokenizer's model, or None for a base model."""
    return CHAT_TEMPLATES.get(getattr(tokenizer, 'name_or_path', None))


def ask_text(ask: float | None) -> str:
    return 'unknown' if ask is None else f'{ask:.2f}'


def header_parts(ask: float | None, statement: str | None = None, example: str | None = None,
                 rules: str | None = None, filler: str | None = None) -> dict[str, str]:
    if rules is not None and filler is not None:
        raise ValueError('pass rules or filler, not both')
    parts = {'format and rules': f'Format: {FORMAT_NAME}.\nRules: {RULES}.\n', 'ask': ASK_LINE.format(ask=ask_text(ask)) + '\n'}
    for name, text in (('task statement', statement), ('rules line', rules), ('filler line', filler),
                       ('example team', example)):
        if text is not None:
            if not text.endswith('\n') or TEAM_LABEL in text:
                raise ValueError(f'a {name} ends its own line and never holds the team label')
            parts[name] = text
    return parts


@lru_cache(maxsize=1)
def example_team() -> Team:
    """#239's fixed example team, from EXAMPLE_TEAM."""
    return json.loads(EXAMPLE_TEAM.read_text())['team']


def example_text() -> str:
    """The example team's line: its label and the team as compact JSON, as the masked team is written."""
    return EXAMPLE_LABEL + team_json(example_team()) + '\n'


def text_version(statement: str | None, example: str | None,
                 rules: str | None = None, filler: str | None = None, chat: ChatTemplate | None = None) -> str:
    version = INSTRUCTIONS_VERSION if statement is not None else VERSION
    if rules is not None:
        version += RULES_SUFFIX
    if filler is not None:
        version += FILLER_SUFFIX
    if example is not None:
        version += EXAMPLE_SUFFIX
    if chat is not None:
        version += '+' + chat.name
    return version


def in_chat(parts: dict[str, str], chat: ChatTemplate) -> dict[str, str]:
    """The header parts as a chat's user turn, trimmed as the template trims it, and the assistant turn opened."""
    names = list(parts)
    trimmed = {name: text.rstrip() if name == names[-1] else text for name, text in parts.items()}
    return {'chat user': chat.user, **trimmed, 'chat assistant': chat.assistant}


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


def in_place_text(task: dict[str, Any], ask: float | None, tokenizer: Tokenizer, limit: int,
                  statement: str | None = None, example: str | None = None,
                  rules: str | None = None, filler: str | None = None) -> InPlaceText:
    """The text the model fills for one Mask Lab task, with optional lines (statement, rules or filler, example)
    between the header and the team when given. Refuses text over limit tokens, naming each part's size."""
    hidden = hidden_values(task)
    _check_masked_team(task, hidden)
    lengths = slot_lengths(tokenizer)
    parts = header_parts(ask, statement, example, rules, filler)
    chat = chat_template(tokenizer)
    if chat is not None:
        parts = in_chat(parts, chat)
    prefix = sum(map(len, parts.values())) + len(TEAM_LABEL)
    team, where = render(json_pieces(task['masked_team']),
                         lambda path, value: tokenizer.mask_token * lengths[path[1]] if path in hidden else value_text(value))
    parts['team'] = TEAM_LABEL + team
    if chat is not None:
        parts['chat end'] = chat.end
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
    return InPlaceText(text, tuple(slots), part_tokens, text_version(statement, example, rules, filler, chat),
                       chat.end if chat is not None else '')


def team_text(text: str, tail: str = '') -> tuple[str, bool]:
    """The team's JSON in an in-place text, filled or not, and whether the text has a 'Team: ' line: what follows
    the line's label, less the fixed tail after the team."""
    _, label, team = text.partition('\n' + TEAM_LABEL)
    if tail and team.endswith(tail):
        team = team[:-len(tail)]
    return team, bool(label)


def training_target(task: dict[str, Any], ask: float | None, tokenizer: Tokenizer, limit: int, pad: str,
                    statement: str | None = None, example: str | None = None,
                    rules: str | None = None, filler: str | None = None) -> str:
    """The in-place text with each hidden field's real value in its slot, followed by pad up to the slot length.
    Checked to have the same tokens as the in-place text everywhere outside the slots."""
    built = in_place_text(task, ask, tokenizer, limit, statement, example, rules, filler)
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


def read_filled(text: str, pad: str, mask_token: str, tail: str = '') -> Team:
    """A filled in-place text back into a team: drop the header, the tail and padding, then parse the JSON."""
    team, label = team_text(text, tail)
    if not label:
        raise ValueError(f'no {TEAM_LABEL.strip()!r} line in the text')
    if mask_token in team:
        raise ValueError('the text still has mask tokens: a slot was not filled')
    return team_from_json(team.replace(pad, ''))


@dataclass(frozen=True)
class FilledCheck:
    """One filled text's legality, stored beside the generated team. A text fails at one of two steps, and
    the prompt-only test (#171) counts them apart: unreadable says why the filled JSON is not a team, and
    validator_errors holds each error Showdown's validator gave the team's paste."""
    version: str                        # the in-place text's VERSION, so teams from later text versions stay apart
    team: Team | None                   # None when unreadable
    paste: str | None                   # the battle step's paste, written by the shared paste writer
    unreadable: str | None = None
    validator_errors: tuple[str, ...] = ()

    @property
    def legal(self) -> bool:
        return self.unreadable is None and not self.validator_errors


def _check_value_types(team: Team) -> None:
    """team_from_json checks the keys; a filled slot can still hold the wrong JSON type, which the paste
    writer would turn into a crash or a misleading paste."""
    for i, mon in enumerate(team):
        texts = {f'team[{i}].{name}': mon[name] for name in ('species', 'item', 'ability', 'nature')}
        texts |= {f'team[{i}].moves[{j}]': move for j, move in enumerate(mon['moves'])}
        for where, value in texts.items():
            if not isinstance(value, str):
                raise ValueError(f'{where} is {value!r}, not text')
        for stat, value in mon['statPoints'].items():
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f'team[{i}].statPoints.{stat} is {value!r}, not a whole number')


def check_filled(text: str, pad: str, mask_token: str, validate: Callable[[str], str | None],
                 tail: str = '') -> FilledCheck:
    """Read a filled in-place text into a team, write its paste and validate it. validate takes paste text and
    returns None for a legal team, else its errors joined with '; ', as showdown_validator.Validator does.
    Legality is only measured here: nothing in the text or the decoder enforces it."""
    try:
        team = read_filled(text, pad, mask_token, tail)
        _check_value_types(team)
    except ValueError as error:         # json.JSONDecodeError is a ValueError
        return FilledCheck(VERSION, None, None, unreadable=str(error))
    paste = team_paste(team)
    errors = validate(paste)
    return FilledCheck(VERSION, team, paste, validator_errors=tuple(errors.split('; ')) if errors else ())


# ---------- #240: free-text answer layout ----------

FREETEXT_TASK_LABELS = {'items': ('item', 'items'), 'abilities': ('ability', 'abilities'),
                        'natures': ('nature (Stat Alignment)', 'natures (Stat Alignments)'),
                        'moves': ('move', 'moves')}
_CODE_FENCE = re.compile(r'^```(?:json)?\s*\n(.*?)\n?```$', re.DOTALL | re.IGNORECASE)


def ordered_masked_paths(task: dict[str, Any]) -> list[list[Any]]:
    """The task's masked_paths parsed and sorted in canonical team JSON order (mon index, field order, move index)."""
    def sort_key(path: list[Any]) -> tuple[int, int, int]:
        if len(path) == 1:
            return (path[0], -1, -1)
        return (path[0], JSON_FIELDS.index(path[1]), path[2] if len(path) > 2 else -1)

    return sorted((parse_path(p) for p in task['masked_paths']), key=sort_key)


def masked_team_null_json(task: dict[str, Any]) -> str:
    """Compact JSON of the masked team with JSON null for every hidden value (or whole hidden Pokémon). Contains no
    mask tokens so all masks sit in the free-text answer region."""
    out: list[Any] = []
    for mon in task['masked_team']:
        out.append(None if mon is None else {f: mon[f] for f in JSON_FIELDS})
    return json.dumps(out, ensure_ascii=False, separators=(',', ':'))


def true_answer_items(task: dict[str, Any]) -> list[Any]:
    """The true hidden values in ordered_masked_paths order: strings for scalar/move masks, statPoints dicts for
    stats masks, and compact Pokémon dicts for whole-Pokémon masks."""
    orig = task['original']
    items: list[Any] = []
    for path in ordered_masked_paths(task):
        if len(path) == 1:
            mon = orig[path[0]]
            items.append({f: mon[f] for f in JSON_FIELDS})
        elif len(path) == 2:
            items.append(orig[path[0]][path[1]])
        else:
            items.append(orig[path[0]][path[1]][path[2]])
    return items


def true_answer_json(task: dict[str, Any]) -> str:
    """The true free-text answer as a compact JSON list."""
    return json.dumps(true_answer_items(task), ensure_ascii=False, separators=(',', ':'))


def freetext_instruction(task: dict[str, Any]) -> str:
    """One fixed-template format line per (task, k), stating the exact JSON list shape without naming any mask token.
    Uses 'Replace each null ... (never output null or a Pokémon name)' to avoid literal ["null"] or species-name echoes."""
    kind, k = task['task'], task['k']
    if kind in FREETEXT_TASK_LABELS:
        singular, _plural = FREETEXT_TASK_LABELS[kind]
        if k == 1:
            return (f'Replace the null {singular} in Team with a legal Champions VGC Reg M-B {singular} '
                    '(never output null or a Pokémon name). Output only a compact JSON list of 1 string.\n')
        return (f'Replace each null {singular} in Team with a legal Champions VGC Reg M-B {singular} '
                f'(never output null or a Pokémon name). Output only a compact JSON list of {k} strings in order.\n')
    if kind == 'stats':
        target = 'the null statPoints' if k == 1 else 'each null statPoints'
        return (f'Replace {target} in Team with a JSON object with keys HP, Atk, Def, SpA, SpD, Spe '
                f'(never output null). Output only a compact JSON list of '
                f'{"1 object" if k == 1 else f"{k} objects in order"}.\n')
    if kind == 'pokemon':
        target = 'the null Pokémon' if k == 1 else 'each null Pokémon'
        return (f'Replace {target} in Team with a legal Champions VGC Reg M-B Pokémon object with keys species, '
                f'item, ability, nature, moves, statPoints (never output null). Output only a compact JSON list of '
                f'{"1 object" if k == 1 else f"{k} objects in order"}.\n')
    raise ValueError(f'unknown task kind: {kind!r}')


@lru_cache(maxsize=8)
def _worst_vocabulary_items(tokenizer: Tokenizer) -> dict[str, Any]:
    """Longest Reg M-B value per field in this tokenizer (measured in JSON context), for deterministic answer lengths."""
    def tok_len(text: str) -> int:
        return len(tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)['offset_mapping'])

    longest: dict[str, Any] = {}
    for field, values in _field_values().items():
        if field == 'statPoints':
            longest[field] = max(values, key=lambda n: tok_len(f':{n},'))
        else:
            longest[field] = max(values, key=lambda s: tok_len(json.dumps(s, ensure_ascii=False)))
    worst_stats = dict.fromkeys(STATS, longest['statPoints'])
    worst_mon = {'species': longest['species'], 'item': longest['item'], 'ability': longest['ability'],
                 'nature': longest['nature'], 'moves': [longest['moves']] * 4, 'statPoints': worst_stats}
    return {'items': longest['item'], 'abilities': longest['ability'], 'natures': longest['nature'],
            'moves': longest['moves'], 'stats': worst_stats, 'pokemon': worst_mon}


def answer_length(task_kind: str, k: int, tokenizer: Tokenizer) -> int:
    """Fixed mask-token count for the free-text answer region of one (task_kind, k) cell: the worst-case legal
    Reg M-B compact JSON list's token count in this tokenizer plus ANSWER_SLACK (1 end token + formatting slack).
    Depends only on (task_kind, k) and the vocabulary, never on the hidden answer."""
    worst = _worst_vocabulary_items(tokenizer)
    if task_kind not in worst:
        raise ValueError(f'unknown task kind: {task_kind!r}')
    sample = json.dumps([worst[task_kind]] * k, ensure_ascii=False, separators=(',', ':'))
    n = len(tokenizer(sample, add_special_tokens=False, return_offsets_mapping=True)['offset_mapping'])
    return n + ANSWER_SLACK


def end_token_ids(tokenizer: Any) -> frozenset[int]:
    """Token IDs that end a free-text answer: eos_token_id only (matches dLLM eval trimming at <|im_end|>
    rather than SFT pad <|endoftext|>)."""
    ids: set[int] = set()
    if getattr(tokenizer, 'eos_token_id', None) is not None:
        ids.add(int(tokenizer.eos_token_id))
    elif getattr(tokenizer, 'eos_token', None):
        encoded = tokenizer(tokenizer.eos_token, add_special_tokens=False)['input_ids']
        if len(encoded) == 1:
            ids.add(int(encoded[0]))
    return frozenset(ids)


def eos_and_pad_counts(tokenizer: Any, ids: Sequence[int]) -> tuple[int, int]:
    """Counts of EOS tokens and distinct pad tokens in an answer region's token IDs."""
    ends = end_token_ids(tokenizer)
    pad_id = getattr(tokenizer, 'pad_token_id', None)
    eos_n = sum(t in ends for t in ids)
    pad_n = sum(t == pad_id for t in ids) if isinstance(pad_id, int) and pad_id not in ends else 0
    return eos_n, pad_n


def trim_answer_ids(tokenizer: Any, ids: Sequence[int]) -> tuple[str, int, bool, int]:
    """Trim answer-region token ids at the first end token and decode. Returns (text, answer_tokens,
    hit_length_limit, special_dropped). hit_length_limit is True iff no end token appeared in the answer region."""
    ends = end_token_ids(tokenizer)
    special = set(getattr(tokenizer, 'all_special_ids', []) or ())
    cut = next((i for i, t in enumerate(ids) if t in ends), None)
    hit_limit = cut is None
    kept = list(ids if cut is None else ids[:cut])
    dropped = sum(t in special for t in kept)
    text = tokenizer.decode(kept, skip_special_tokens=True)
    return text, len(kept), hit_limit, dropped


def freetext_version(example: str | None = None, chat: ChatTemplate | None = None) -> str:
    version = FREETEXT_VERSION
    if example is not None:
        version += EXAMPLE_SUFFIX
    if chat is not None:
        version += '+' + chat.name
    return version


def freetext_text(task: dict[str, Any], ask: float | None, tokenizer: Tokenizer, limit: int,
                  example: str | None = None) -> InPlaceText:
    """#240: the free-text answer prompt. Visible fields stay in Team: as compact JSON with null for hidden values;
    all mask tokens sit in one fixed-length answer region after Answer:\\n (or the chat assistant turn, plus any
    fixed chat.freetext_prefix such as '[' for iLLaDA)."""
    hidden = hidden_values(task)
    _check_masked_team(task, hidden)
    ans_len = answer_length(task['task'], task['k'], tokenizer)
    parts = header_parts(ask, example=example)
    parts['answer format'] = freetext_instruction(task)
    parts['team'] = TEAM_LABEL + masked_team_null_json(task) + '\n'
    chat = chat_template(tokenizer)
    answer_prefix = ''
    if chat is not None:
        parts = in_chat(parts, chat)
        if chat.freetext_prefix:
            answer_prefix = chat.freetext_prefix
            parts['answer prefix'] = answer_prefix
    else:
        parts['answer label'] = ANSWER_LABEL
    prefix_tokens = len(offsets(tokenizer, answer_prefix)) if answer_prefix else 0
    mask_len = ans_len - prefix_tokens
    if mask_len < 1:
        raise ValueError(f'answer_prefix {answer_prefix!r} ({prefix_tokens} tokens) leaves no masks in {ans_len}')
    prefix_len = sum(map(len, parts.values()))
    parts['answer'] = tokenizer.mask_token * mask_len
    text = ''.join(parts.values())
    spans = offsets(tokenizer, text)
    part_tokens = _count_parts(spans, parts)
    if len(spans) > limit:
        sizes = ', '.join(f'{name} {n}' for name, n in part_tokens.items())
        raise TextTooLong(f'{len(spans)} tokens, over the limit of {limit}: {sizes}')
    inside = tokens_within(spans, prefix_len, len(text), 'answer')
    if len(inside) != mask_len:
        raise ValueError(f'answer: {len(inside)} tokens, not {mask_len}; {tokenizer.mask_token!r} is not one token')
    slot = SlotSpan('answer', 'answer', prefix_len, len(text), inside[0], len(inside))
    return InPlaceText(text, (slot,), part_tokens, freetext_version(example, chat), tail='',
                       mode='freetext', answer_length=mask_len, answer_prefix=answer_prefix)


def freetext_training_target(task: dict[str, Any], ask: float | None, tokenizer: Tokenizer, limit: int, pad: str,
                             example: str | None = None) -> str:
    """The free-text prompt with the true compact JSON answer + eos_token + pad up to answer_length."""
    built = freetext_text(task, ask, tokenizer, limit, example)
    slot = built.slots[0]
    eos = getattr(tokenizer, 'eos_token', None) or pad
    body = true_answer_json(task)
    if built.answer_prefix:
        if not body.startswith(built.answer_prefix):
            raise ValueError(f"{task['id']}: true answer {body[:16]!r} does not start with {built.answer_prefix!r}")
        body = body[len(built.answer_prefix):]
        prefix_spans = offsets(tokenizer, built.answer_prefix)
        joined_spans = offsets(tokenizer, built.answer_prefix + body)
        if joined_spans[:len(prefix_spans)] != prefix_spans:
            body = ' ' + body
            joined_spans = offsets(tokenizer, built.answer_prefix + body)
            if joined_spans[:len(prefix_spans)] != prefix_spans:
                raise ValueError(f"{task['id']}: answer_prefix {built.answer_prefix!r} merges with true answer")
        body_tokens = len(joined_spans) - len(prefix_spans)
    else:
        body_tokens = len(offsets(tokenizer, body))
    eos_tokens = len(offsets(tokenizer, eos))
    rem = slot.length - body_tokens - eos_tokens
    if rem < 0:
        raise ValueError(f"{task['id']}: true answer ({body_tokens} + {eos_tokens} eos) exceeds answer_length {slot.length}")
    filled = body + eos + pad * rem
    target = built.text[:slot.start] + filled + built.text[slot.end:]
    _check_target_tokens(built, target, tokenizer)
    return target


def _unwrap_code_fence(text: str) -> tuple[str, bool]:
    stripped = text.strip()
    m = _CODE_FENCE.match(stripped)
    return (m.group(1).strip(), True) if m else (stripped, False)


def _check_stat_points_obj(obj: Any, where: str) -> dict[str, int]:
    if not isinstance(obj, dict) or set(obj) != set(STATS):
        raise ValueError(f'{where}: needs all six Stat Points')
    out: dict[str, int] = {}
    for stat in STATS:
        v = obj[stat]
        if isinstance(v, bool) or not isinstance(v, int):
            raise ValueError(f'{where}.{stat} is {v!r}, not a whole number')
        out[stat] = v
    return out


def _check_pokemon_obj(obj: Any, slot_number: int) -> dict[str, Any]:
    if not isinstance(obj, dict) or set(obj) != set(JSON_FIELDS):
        raise ValueError(f'Pokémon {slot_number}: keys must be {", ".join(JSON_FIELDS)}')
    if not isinstance(obj['moves'], list) or len(obj['moves']) != 4:
        raise ValueError(f'Pokémon {slot_number}: needs four moves and all six Stat Points')
    mon = blank_slot(slot_number)
    for field in ('species', 'item', 'ability', 'nature'):
        if not isinstance(obj[field], str):
            raise ValueError(f'team[{slot_number - 1}].{field} is {obj[field]!r}, not text')
        mon[field] = obj[field]
    for j, mv in enumerate(obj['moves']):
        if not isinstance(mv, str):
            raise ValueError(f'team[{slot_number - 1}].moves[{j}] is {mv!r}, not text')
    mon['moves'] = list(obj['moves'])
    mon['statPoints'] = _check_stat_points_obj(obj['statPoints'], f'team[{slot_number - 1}].statPoints')
    return mon


def parse_freetext_answer(task: dict[str, Any], raw_answer: str) -> tuple[Team, list[str], bool, bool]:
    """Parse a trimmed free-text answer back into a full 6-Pokémon team and flat hidden-value list in
    hidden_values(task) order. Returns (team, values, had_fence, strict_json_parses); raises ValueError if unreadable."""
    cleaned, had_fence = _unwrap_code_fence(raw_answer)
    strict_ok = False
    try:
        json.loads(raw_answer.strip())
        strict_ok = True
    except ValueError:
        pass
    parsed = json.loads(cleaned)
    paths = ordered_masked_paths(task)
    if not isinstance(parsed, list) or len(parsed) != len(paths):
        raise ValueError(f'answer must be a JSON list of {len(paths)} items')
    team: Team = copy.deepcopy(task['masked_team'])
    for path, item in zip(paths, parsed, strict=True):
        if len(path) == 1:
            team[path[0]] = _check_pokemon_obj(item, path[0] + 1)
        elif path[1] == 'statPoints':
            team[path[0]]['statPoints'] = _check_stat_points_obj(item, f'team[{path[0]}].statPoints')
        elif len(path) == 2:
            if not isinstance(item, str):
                raise ValueError(f'team[{path[0]}].{path[1]} is {item!r}, not text')
            team[path[0]][path[1]] = item
        else:
            if not isinstance(item, str):
                raise ValueError(f'team[{path[0]}].moves[{path[2]}] is {item!r}, not text')
            team[path[0]]['moves'][path[2]] = item
    _check_value_types(team)
    by_path = {piece[0]: piece[1] for piece in json_pieces(team) if not isinstance(piece, str)}
    values = [value_text(by_path[p]) for p in hidden_values(task)]
    return team, values, had_fence, strict_ok


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
    print('The training target reads back to the unmasked team.')
    validate = Validator()
    try:
        checked = check_filled(target, tokenizer.pad_token, tokenizer.mask_token, validate)
        guess = built.text
        for slot in reversed(built.slots):          # every hidden text value Pikachu, every number 1
            guess = guess[:slot.start] + ('1' if slot.field == 'statPoints' else 'Pikachu') + guess[slot.end:]
        guessed = check_filled(guess, tokenizer.pad_token, tokenizer.mask_token, validate)
    finally:
        validate.close()
    print(f'Showdown validator on the training target: legal {checked.legal}, errors {list(checked.validator_errors)}')
    print(f'Every hidden text value Pikachu and number 1: legal {guessed.legal}, unreadable {guessed.unreadable!r}, '
          f'errors {list(guessed.validator_errors)[:3]}\n')


if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
    from showdown_validator import Validator
    from transformers import AutoTokenizer
    qwen = AutoTokenizer.from_pretrained('dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1')
    if len(sys.argv) > 1:
        _demo(qwen, sys.argv[1], sys.argv[2], int(sys.argv[3]))
    else:
        _demo(qwen, 'MB158', 'pokemon', 1)
        _demo(qwen, 'MB158', 'pokemon', 6)
