"""Fill the held-out grid with one named prompt and one masked diffusion candidate, first try only, and report how
good the filled teams are (#237). Pre-registration: code/docs/inplace-prompt-heldout-baseline.md.

The held-out grid is 270 Mask Lab masks: 5 unmasked teams (ranks 51-100) x 54 cells, items, abilities, Stat
Alignments, Stat Points and whole Pokemon at 1 to 6 hidden, moves at 1 to 24. Each mask is filled once with its own
seed derived from the run seed and the mask's id, so a mask's draw does not depend on which masks ran before it and
a stopped run resumes where it stopped. #237 to #239 filled one mask per forward pass. From #443 (owner, 2026-11-25:
batch size 1 is no longer required) run fills a batch of masks per forward pass, the largest batch that fits the
device: rows are right-padded and grouped shortest text first, and each keeps its own seed, slots and schedule
(fill_batch, masked_diffusion_memory_bench.fill_rows). The batch size is in run.json's setting.

Prompts are stacked (#237 to #240); each later ticket adds one entry to PROMPTS:
    inplace_header      Header only, values filled in place: the #168 text, masked_diffusion_text.in_place_text
    inplace_instructions  Task statement added (#238): the same text with masked_diffusion_text.TASK_STATEMENT
                        between the header and the team
    inplace_header_example  Example team added to the header-only prompt (#239): one fixed legal team
                        (masked_diffusion_text.example_text) between the header and the team
    inplace_example     Example team added to the task-statement prompt (#239, the ticket's own arm): the statement,
                        then the example team, then the team
An example team costs about 420 tokens, so with it some masks pass the 1,024-token limit; fitting_ids names the masks
whose text fits for all four small models, and every run, probe and comparison with that prompt uses those alone.

The sampler is #170's (masked_diffusion_memory_bench.fill), configured by each candidate's SamplerSetting (defaulting
to today's shared sampler: one window, dLLM's linear schedule, most confident masks first, 128 steps, temperature 1.0,
float32 Gumbel-max; #392). Every result file records the sampler_setting it used.

A filled mask is judged in four steps; the first that fails is its reason:
    1. parses      the team part is valid JSON with six Pokemon, the six keys in order, and the right value types
    2. visible     every value the mask did not hide reads as in the masked team (a value that broke out of its
                   quotes can overwrite a visible one)
    3. Reg M-B     every filled name (species, item, ability, Stat Alignment, move) is a Reg M-B value of its field
                   in code/data/regmb-vocabulary.json, by Showdown's ID (lower case, letters and digits). Blank is
                   allowed where the vocabulary allows it: item, move and Stat Alignment
    4. Showdown    Showdown's validator accepts the team's paste
Legal means all four pass. Step 3 exists because Showdown's validator accepts some junk: a Stat Alignment of "00"
passes it (checked 2026-11-18). The report also counts teams Showdown alone accepts (steps 1, 2 and 4), the count
#169's bar was written against.

    preflight --out F [--prompt P]            the CPU probes: ceiling, round trip, coverage, controls, floor; with a
                                              task statement, also its token cost per tokenizer
    run --prompt P --key K --dir D [--limit N] [--device mps]
                                              fill the grid; appends one line per mask to D/P/K/fills.jsonl
    redraw --prompt P --key K --dir D         #169's validator loop for #171: each mask that failed its first try is
                                              drawn up to 4 more times with its own seed per draw, stopping at the
                                              first legal draw; appends to D/P/K/redraws.jsonl
    teams --prompt P --key K --dir D          each mask's first legal draw as a paste, judged again, in teams.json
    gpu-preflight --key K --dir D [--prompt P]  the model probes: sees later tokens, repeat, run order
    report --dir D [--prompt P]               per prompt and model, as JSON (D/P/report.json) and a Markdown table
    compare --dir D --before P --after Q      the same masks under two prompts, per model, paired by mask (McNemar),
                                              as JSON (D/Q/compare.json) and a Markdown table

Run with venv-dlm: ~/.local/share/vgc-pilot-runtime/venv-dlm/bin/python code/scripts/inplace_prompt_grid.py ...
"""
import argparse
import functools
import hashlib
import json
import math
import platform
import random
import re
import sys
import time
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import masked_diffusion_text as D
from mask_paste import STATS, Team, json_pieces, team_from_json, team_paste, value_text
from masked_diffusion_sampler import SAMPLERS, SHARED_SAMPLER, SamplerSetting, resolve_sampler

GRID = 'heldout'
GRID_SIZE = 270
SMALL_FOUR = ('diffugpt_s', 'diffugpt_m', 'qwen25c', 'qwen3')    # #237: fixed for every prompt (owner, 2026-11-07)
ASK: float | None = None            # the ask line reads 'unknown': no base model was trained on the ask
SETTING = {'steps': 128, 'temperature': 1.0, 'seed': 2026, 'scope': 'slots', 'precision': 'native', 'batch': 1}
LIMIT = 1024
REDRAWS = 4                 # #169: a failing mask is drawn up to 4 more times, 5 draws at most
OUT = Path.home() / 'vgc-data/inplace_prompt_grid'
NAME_FIELDS = ('species', 'item', 'ability', 'nature', 'moves')
VOCABULARY_KEY = {'species': 'species', 'item': 'item', 'ability': 'ability', 'nature': 'nature', 'moves': 'move'}
FIELD_NAMES = {'species': 'species', 'item': 'item', 'ability': 'ability', 'nature': 'Stat Alignment',
               'moves': 'move', 'statPoints': 'Stat Point number'}
STEPS = ('parses', 'visible', 'regmb', 'showdown')
# Every rule a filled mask can break, judged on each fill (owner, 2026-11-25: "keep track of that feature for each
# sample"). The team-building rules follow #361's rule-by-rule table (epoch_legality.RULES), judged here on names.
# A rule reads None when it cannot be judged: the fill does not parse, the mask hides no value of that field, or
# Showdown has not run yet (a Modal fill before merge).
NAME_RULES = {f: f'not a Reg M-B {FIELD_NAMES[f]}' for f in NAME_FIELDS}
RULES = ('does not parse as a team', 'changed a visible value', *NAME_RULES.values(), 'item used more than once',
         'wrong Mega Stone', 'species used more than once', "ability the species can't have",
         "move the species can't learn", 'same move twice on one Pokémon', 'Stat Points over 66',
         'a Stat Point over 32', 'a name Showdown does not know', 'other Showdown error')
SHOWDOWN_RULES = ((r"can't have", "ability the species can't have"), (r"can't learn", "move the species can't learn"),
                  (r'is an invalid|does not exist', 'a name Showdown does not know'),
                  (r'Item Clause', 'item used more than once'), (r'Species Clause', 'species used more than once'),
                  (r'multiple copies of', 'same move twice on one Pokémon'),
                  (r'total Stat Points', 'Stat Points over 66'), (r'more than \d+ Stat Points in', 'a Stat Point over 32'))
SHOWDOWN_ONLY = ("ability the species can't have", "move the species can't learn", 'a name Showdown does not know',
                 'other Showdown error')
DEX = Path(__file__).resolve().parents[1] / 'data/pokedex.json'
STEP_NAMES = {'parses': 'does not parse as a team', 'visible': 'changed a visible value',
              'regmb': 'wrote a name that is not Reg M-B', 'showdown': 'Showdown rejects it', None: 'legal'}


@dataclass(frozen=True)
class Prompt:
    code: str
    name: str
    build: Callable[..., D.InPlaceText]     # (task, ask, tokenizer, limit) -> the text and its slots
    statement: str | None = None            # the task statement build puts before the team, if any
    example: str | None = None              # the example team's line build puts before the team, if any
    base: str | None = None                 # for an example prompt, the prompt it adds the example to


PROMPTS = {'inplace_header': Prompt('inplace_header', 'Header only, values filled in place', D.in_place_text),
           'inplace_instructions': Prompt('inplace_instructions', 'Task statement added, values filled in place',
                                          functools.partial(D.in_place_text, statement=D.TASK_STATEMENT),
                                          D.TASK_STATEMENT),
           'inplace_header_example': Prompt('inplace_header_example', 'Example team, header only',
                                            functools.partial(D.in_place_text, example=D.example_text()),
                                            None, D.example_text(), 'inplace_header'),
           'inplace_example': Prompt('inplace_example', 'Example team, task statement',
                                     functools.partial(D.in_place_text, statement=D.TASK_STATEMENT,
                                                       example=D.example_text()),
                                     D.TASK_STATEMENT, D.example_text(), 'inplace_instructions')}


def to_id(text: str) -> str:
    """Showdown's toID: lower case, letters and digits only."""
    return re.sub(r'[^a-z0-9]', '', text.lower())


def vocabulary() -> dict[str, frozenset[str]]:
    values = json.loads(D.VOCABULARY.read_text())['values']
    return {field: frozenset(values[key]) for field, key in VOCABULARY_KEY.items()}


def mask_seed(seed: int, task_id: str) -> int:
    """A mask's own seed: the same for a mask in every run with this seed, whatever ran before it."""
    return int.from_bytes(hashlib.sha256(f'{seed}:{task_id}'.encode()).digest()[:8], 'big') >> 1


# ---------- the canvas a model fills ----------

def canvas(task: dict[str, Any], prompt: Prompt, tokenizer: Any, mask_id: int, start_id: int | None = None,
           ask: float | None = ASK) -> dict[str, Any]:
    """The prompt's text for one mask as token ids, with each slot's token positions. start_id goes first when the
    candidate needs its start token (DiffuLLaMA, #236); every position then moves up by one."""
    built = prompt.build(task, ask, tokenizer, LIMIT)
    ids = list(tokenizer(built.text, add_special_tokens=False)['input_ids'])
    shift = 0 if start_id is None else 1
    if start_id is not None:
        ids = [start_id, *ids]
    spans = [list(range(s.first_token + shift, s.first_token + shift + s.length)) for s in built.slots]
    flat = [n for span in spans for n in span]
    if not all(ids[n] == mask_id for n in flat) or ids.count(mask_id) != len(flat):
        raise ValueError(f"{task['id']}: the mask tokens are not exactly the slots")
    return {'built': built, 'ids': ids, 'spans': spans, 'slots': flat}


def fits(task: dict[str, Any], prompt: Prompt, tokenizer: Any, mask_id: int, start_id: int | None) -> bool:
    """Whether the prompt's text for this mask, start token included, is at most LIMIT tokens in this tokenizer."""
    try:
        return len(canvas(task, prompt, tokenizer, mask_id, start_id)['ids']) <= LIMIT
    except D.TextTooLong:
        return False


@functools.lru_cache(maxsize=8)
def fitting_ids(prompt_code: str) -> frozenset[str] | None:
    """The masks whose text fits under LIMIT for every small-four tokenizer with this prompt, or None when the prompt
    has no example team and every mask fits (#239). One list for all four models, so each arm is the same masks."""
    prompt = PROMPTS[prompt_code]
    if prompt.example is None:
        return None
    import count_masked_diffusion_text as C
    tasks, keep = grid_tasks(), None
    for key in SMALL_FOUR:
        candidate = C.candidate_by_key(key)
        tokenizer = C.load_tokenizer(candidate)
        start_id = tokenizer.bos_token_id if candidate.sampler.start_token else None
        ok = {t['id'] for t in tasks if fits(t, prompt, tokenizer, candidate.mask_id, start_id)}
        keep = ok if keep is None else keep & ok
    return frozenset(keep or ())


def prompt_tasks(prompt_code: str, tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The masks a prompt is filled on, in grid order: all of them, or for an example prompt those that fit."""
    keep = fitting_ids(prompt_code)
    return tasks if keep is None else [t for t in tasks if t['id'] in keep]


def slot_values(tokenizer: Any, ids: Sequence[int], spans: Sequence[Sequence[int]]) -> tuple[list[str], int]:
    """Each slot's filled text, decoded from its own tokens with special tokens dropped (they stand for padding),
    and how many special tokens were dropped."""
    special = set(getattr(tokenizer, 'all_special_ids', []))
    values, dropped = [], 0
    for span in spans:
        tokens = [ids[n] for n in span]
        dropped += sum(t in special for t in tokens)
        values.append(tokenizer.decode(tokens, skip_special_tokens=True))
    return values, dropped


def splice(built: D.InPlaceText, values: Sequence[str]) -> str:
    """The prompt's text with each slot's mask tokens replaced by its filled text, exactly as written."""
    text, cut = built.text, 0
    for slot, value in zip(built.slots, values, strict=True):
        text = text[:slot.start + cut] + value + text[slot.end + cut:]
        cut += len(value) - (slot.end - slot.start)
    return text


# ---------- judging one filled mask ----------

def visible_values(team: Team) -> dict[tuple[Any, ...], Any]:
    return {piece[0]: piece[1] for piece in json_pieces(team) if not isinstance(piece, str) and piece[1] is not None}


def _value_at(team: Team, path: tuple[Any, ...]) -> Any:
    value: Any = team[path[0]]
    for key in path[1:]:
        value = value[key]
    return value


def reason_group(error: str) -> str:
    """A validator or parser message with its names and positions taken out, so like errors count together."""
    rules = [(r'does not exist', 'a species, ability or form that does not exist'),
             (r'is an invalid item', 'an item that does not exist'),
             (r'is an invalid ability', 'an ability that does not exist'),
             (r'is an invalid move', 'a move that does not exist'),
             (r'is an invalid nature', 'a Stat Alignment that does not exist'),
             (r"can't learn", "a move its species can't learn"),
             (r"can't have", "an ability its species can't have"),
             (r'needs to have an ability', 'no ability'),
             (r'has no moves', 'no moves'),
             (r'multiple copies of', 'the same move twice'),
             (r'Species Clause', 'two of one species (Species Clause)'),
             (r'Item Clause', 'two of one item (Item Clause)'),
             (r'more than 32 Stat Points', 'over 32 Stat Points in one stat'),
             (r'total Stat Points', 'over 66 Stat Points in total'),
             (r'can only be obtained|must hold|requires', 'a form or item requirement')]
    for pattern, group in rules:
        if re.search(pattern, error):
            return group
    error = re.sub(r'"[^"]*"', '"…"', error)
    error = re.sub(r': line \d+ column \d+ \(char \d+\)', '', error)
    return re.sub(r'\d+', 'N', error)


@functools.lru_cache(maxsize=1)
def dex() -> tuple[dict[str, str], dict[str, str]]:
    """(Mega Stone ID -> the base species ID it belongs to, species ID -> base species ID), from Showdown's
    pokedex.json in code/data, as corpus.MEGA_STONE_OF builds it."""
    entries = json.loads(DEX.read_text())
    stones, base = {}, {}
    for key, entry in entries.items():
        if not isinstance(entry, dict):
            continue
        base[key] = to_id(entry.get('baseSpecies') or entry.get('name', key))
        owner = entry.get('changesFrom') or entry.get('baseSpecies')
        if 'mega' in to_id(entry.get('forme', '')) and entry.get('requiredItem') and owner:
            stones[to_id(entry['requiredItem'])] = to_id(owner)
    return stones, base


def _twice(ids: Iterable[str]) -> bool:
    seen = [i for i in ids if i]
    return len(seen) != len(set(seen))


def rule_breaks(fields: Sequence[str], values: Sequence[str], names: dict[str, frozenset[str]],
                verdict: dict[str, Any], team: Team | None) -> dict[str, bool | None]:
    """Which of RULES one filled mask breaks. The Reg M-B name rules read the filled values, so they are judged even
    when the fill does not parse; every other rule needs the parsed team. Six team rules are checked on the team
    itself; Showdown's messages add the ability, learnset and unknown-name rules, and confirm the others."""
    out: dict[str, bool | None] = dict.fromkeys(RULES)
    out['does not parse as a team'] = not verdict['reads_as_team']
    for field, rule in NAME_RULES.items():
        filled = [v for f, v in zip(fields, values, strict=True) if f == field]
        out[rule] = any(to_id(v) not in names[field] for v in filled) if filled else None
    if team is None:
        return out
    stones, base = dex()
    species = [to_id(mon['species']) for mon in team]
    out['changed a visible value'] = not verdict['visible_kept']
    out['item used more than once'] = _twice(to_id(mon['item']) for mon in team)
    out['wrong Mega Stone'] = any(to_id(mon['item']) in stones and stones[to_id(mon['item'])] != base.get(s, s)
                                  for mon, s in zip(team, species, strict=True))
    out['species used more than once'] = _twice(base.get(s, s) for s in species)
    out['same move twice on one Pokémon'] = any(_twice(to_id(m) for m in mon['moves']) for mon in team)
    out['Stat Points over 66'] = any(sum(mon['statPoints'].values()) > 66 for mon in team)
    out['a Stat Point over 32'] = any(v > 32 for mon in team for v in mon['statPoints'].values())
    errors = verdict['showdown_errors'] or []
    if any(e.startswith('deferred') for e in errors):
        return out
    for rule in SHOWDOWN_ONLY:
        out[rule] = False
    for error in errors:
        hits = [rule for pattern, rule in SHOWDOWN_RULES if re.search(pattern, error)]
        for rule in hits or ['other Showdown error']:
            out[rule] = True
    return out


def judge(task: dict[str, Any], built: D.InPlaceText, values: Sequence[str], validate: Callable[[str], str | None],
          names: dict[str, frozenset[str]]) -> dict[str, Any]:
    """One filled mask through the four steps. Every step that can run is recorded; failed is the first that fails."""
    text = splice(built, values)
    out: dict[str, Any] = {'json_parses': False, 'reads_as_team': False, 'visible_kept': None, 'regmb_names': None,
                           'showdown_errors': None, 'showdown_accepts': False, 'legal': False, 'failed': None,
                           'reasons': []}
    fields = [slot.field for slot in built.slots]
    bad = [(f, v) for f, v in zip(fields, values, strict=True) if f in NAME_FIELDS and to_id(v) not in names[f]]
    out['regmb_names'] = not bad
    _, _, team_text = text.partition('\n' + D.TEAM_LABEL)
    try:
        json.loads(team_text)
        out['json_parses'] = True
        team = team_from_json(team_text)
        D._check_value_types(team)
        out['reads_as_team'] = True
    except ValueError as error:
        out['failed'], out['reasons'] = 'parses', [reason_group(str(error))]
        out['rules'] = rule_breaks(fields, values, names, out, None)
        return out
    changed = [D.value_label(path) for path, value in visible_values(task['masked_team']).items()
               if _value_at(team, path) != value]
    out['visible_kept'] = not changed
    errors = validate(team_paste(team))
    out['showdown_errors'] = [] if errors is None else [e for e in errors.split('; ') if not e.startswith('(')]
    out['showdown_accepts'] = errors is None and not changed
    out['team'] = team
    if changed:
        out['failed'], out['reasons'] = 'visible', ['changed a visible value']
    elif bad:
        out['failed'], out['reasons'] = 'regmb', sorted({f'not a Reg M-B {FIELD_NAMES[f]}' for f, _ in bad})
        out['not_regmb'] = [f'{FIELD_NAMES[f]} {v!r}' for f, v in bad]
    elif errors is not None:
        out['failed'] = 'showdown'
        out['reasons'] = sorted({reason_group(e) for e in out['showdown_errors']})
    out['legal'] = out['failed'] is None
    out['rules'] = rule_breaks(fields, values, names, out, team)
    return out


def copies(task: dict[str, Any], built: D.InPlaceText, values: Sequence[str]) -> dict[str, Any]:
    """Whether the fill gives back the unmasked team's values, and how many filled names repeat a visible value
    of the same field in the same text."""
    seen: dict[str, set[str]] = {}
    for path, value in visible_values(task['masked_team']).items():
        if path[1] in NAME_FIELDS:
            seen.setdefault(path[1], set()).add(to_id(value))
    named = [(s.field, v) for s, v in zip(built.slots, values, strict=True) if s.field in NAME_FIELDS]
    return {'recovered': list(values) == true_values(task),
            'copied_visible': sum(to_id(v) in seen.get(f, set()) and to_id(v) != '' for f, v in named)}


def example_copies(rows: Sequence[dict[str, Any]], team: Team | None = None) -> dict[str, dict[str, int]]:
    """#239: per field, how many filled values equal the example team's value in the same place (same Pokémon number,
    move number or stat), and, for names, how many equal any of the example's values of that field. Blank fills are
    not counted. Run on a prompt without the example, the same counts are the chance rate."""
    from mask_paste import parse_path
    team = team if team is not None else D.example_team()
    anywhere: dict[str, set[str]] = {}
    for path, value in visible_values(team).items():
        if path[1] in NAME_FIELDS:
            anywhere.setdefault(path[1], set()).add(to_id(value))
    out: dict[str, dict[str, int]] = {}
    for row in rows:
        if 'paths' not in row:
            continue
        for field, path_label, value in zip(row['fields'], row['paths'], row['values'], strict=True):
            if value.strip() == '':
                continue
            parts = path_label.split('.')
            path = tuple(parse_path('.'.join(parts[:2]))) + ((parts[2],) if len(parts) > 2 else ())
            here = out.setdefault(field, {'same_place': 0, 'same_field': 0, 'filled': 0})
            here['filled'] += 1
            mine = _value_at(team, path)
            if field in NAME_FIELDS:
                here['same_place'] += to_id(value) == to_id(str(mine))
                here['same_field'] += to_id(value) in anywhere.get(field, set())
            else:
                here['same_place'] += value.strip() == str(mine)
    return out


# ---------- filling the grid ----------

def environment() -> dict[str, Any]:
    import torch
    try:
        import transformers
        version = transformers.__version__
    except ImportError:
        version = None
    return {'torch': torch.__version__, 'transformers': version, 'python': platform.python_version(),
            'macos': platform.mac_ver()[0], 'machine': platform.machine()}


def done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {json.loads(line)['id'] for line in path.read_text().splitlines() if line.strip()}


def start_token_id(tokenizer: Any, sampler: SamplerSetting) -> int | None:
    """The tokenizer's start token id when sampler.start_token is True, else None."""
    return tokenizer.bos_token_id if sampler.start_token else None


def _fill_sampler(sampler: SamplerSetting | None, setting: dict[str, Any],
                  start_id: int | None) -> SamplerSetting:
    if sampler is not None:
        if sampler.start_token != (start_id is not None):
            raise ValueError(f'sampler.start_token ({sampler.start_token}) disagrees with start_id ({start_id!r})')
        return sampler
    return replace(SHARED_SAMPLER, steps=setting['steps'], temperature=setting['temperature'],
                   start_token=start_id is not None)


def fill_one(task: dict[str, Any], prompt: Prompt, tokenizer: Any, forward_for: Callable[[Any], Any], mask_id: int,
             validate: Callable[[str], str | None], names: dict[str, frozenset[str]], seed: int, device: str = 'cpu',
             start_id: int | None = None, setting: dict[str, Any] = SETTING,
             memory: Callable[[], int] = lambda: 0, sampler: SamplerSetting | None = None) -> dict[str, Any]:
    """One draw for one mask with the given seed, judged; a legal draw carries its paste."""
    import torch

    import masked_diffusion_memory_bench as B
    active = _fill_sampler(sampler, setting, start_id)
    c = canvas(task, prompt, tokenizer, mask_id, start_id)
    x = torch.tensor([c['ids']], device=device)
    slots = torch.tensor(c['slots'], device=device)
    generator = torch.Generator(device=device).manual_seed(seed)
    forwards = 0

    def count_step(_step: int) -> None:
        nonlocal forwards
        forwards += 1

    begin = time.perf_counter()
    result = B.fill(forward_for(slots), x, slots, mask_id, generator=generator, on_step=count_step, setting=active)
    if device == 'mps':
        torch.mps.synchronize()
    elif device.startswith('cuda'):
        torch.cuda.synchronize()
    seconds = time.perf_counter() - begin
    return judged_row(task, prompt, tokenizer, c, result[0].cpu().tolist(), seed, active, forwards, seconds, memory(),
                      validate, names)


def judged_row(task: dict[str, Any], prompt: Prompt, tokenizer: Any, c: dict[str, Any], ids: list[int], seed: int,
               active: SamplerSetting, forwards: int, seconds: float, peak: int,
               validate: Callable[[str], str | None], names: dict[str, frozenset[str]]) -> dict[str, Any]:
    """One filled mask's row: its slot values, the verdict, and how it was drawn."""
    values, dropped = slot_values(tokenizer, ids, c['spans'])
    verdict = judge(task, c['built'], values, validate, names)
    team = verdict.pop('team', None)
    verdict |= copies(task, c['built'], values)
    row = {'id': task['id'], 'source': task['source'], 'task': task['task'], 'k': task['k'],
           'prompt': prompt.code, 'version': c['built'].version, 'ask': D.ask_text(ASK), 'seed': seed,
           'sampler_setting': active.to_dict(), 'prompt_tokens': len(c['ids']), 'mask_tokens': len(c['slots']),
           'forwards': forwards, 'seconds': seconds,
           'peak_bytes': peak, 'fields': [s.field for s in c['built'].slots],
           'paths': [s.path for s in c['built'].slots], 'values': values, 'special_dropped': dropped,
           'slot_ids': [[ids[n] for n in span] for span in c['spans']], **verdict}
    if verdict['legal']:
        row['paste'] = team_paste(team)
    return row


def fill_batch(draws: Sequence[tuple[dict[str, Any], int]], prompt: Prompt, tokenizer: Any, row_forward: Any,
               mask_id: int, validate: Callable[[str], str | None], names: dict[str, frozenset[str]],
               device: str = 'cpu', start_id: int | None = None, setting: dict[str, Any] = SETTING,
               memory: Callable[[], int] = lambda: 0, sampler: SamplerSetting | None = None,
               batch_size: int | None = None) -> list[dict[str, Any]]:
    """Many (task, seed) draws in one batch, one forward pass per step (#443), each judged as fill_one judges it.
    Each row keeps its own seed, slots and schedule, so a row's tokens are the ones fill_one draws for it, up to the
    model's arithmetic on a padded batch. A row's seconds are its share of the batch's time."""
    import torch

    import masked_diffusion_memory_bench as B
    active = _fill_sampler(sampler, setting, start_id)
    canvases = [canvas(task, prompt, tokenizer, mask_id, start_id) for task, _ in draws]
    generators = [torch.Generator(device=device).manual_seed(seed) for _, seed in draws]
    forwards = [0] * len(draws)

    def count_step(live: list[int]) -> None:
        for n in live:
            forwards[n] += 1

    if device.startswith('cuda'):
        # The ladder sized the batch on an empty cache; blocks cached by earlier batch shapes would not be reusable
        # by this batch's shape and ran DiffuGPT medium out of memory at batch 142 (#443, 2026-11-25).
        torch.cuda.empty_cache()
    begin = time.perf_counter()
    filled = B.fill_rows(row_forward, [c['ids'] for c in canvases], [c['slots'] for c in canvases], mask_id, active,
                         generators, device, on_step=count_step)
    if device == 'mps':
        torch.mps.synchronize()
    elif device.startswith('cuda'):
        torch.cuda.synchronize()
    seconds = time.perf_counter() - begin
    peak = memory()
    rows = []
    for (task, seed), c, ids, n in zip(draws, canvases, filled, forwards, strict=True):
        row = judged_row(task, prompt, tokenizer, c, ids, seed, active, n, seconds / len(draws), peak, validate, names)
        rows.append(row | {'batch_size': batch_size or len(draws), 'batch_rows': len(draws), 'batch_seconds': seconds})
    return rows


def length_order(tasks: Iterable[dict[str, Any]], prompt: Prompt, tokenizer: Any, mask_id: int,
                 start_id: int | None = None) -> list[str]:
    """Task ids from the shortest text to the longest under this prompt (ties by id), so a batch of neighbours needs
    little padding. mask_groups orders every prompt by the header-only text, so a pair gets the same batches."""
    return [t['id'] for t in sorted(tasks, key=lambda t: (len(canvas(t, prompt, tokenizer, mask_id, start_id)['ids']),
                                                          t['id']))]


def batches(items: Sequence[Any], size: int) -> list[list[Any]]:
    if size < 1:
        raise ValueError(f'batch size must be at least 1, got {size}')
    return [list(items[i:i + size]) for i in range(0, len(items), size)]


ORDER_PROMPT = 'inplace_header'     # its text fits every mask of the grid (#237), so it orders all of them


def mask_groups(tasks: Iterable[dict[str, Any]], tokenizer: Any, mask_id: int, start_id: int | None,
                size: int) -> list[list[str]]:
    """The batches a run's masks fall into: task ids ordered shortest first by their header-only text, cut into
    groups of size. Every prompt fills each group's masks that it has, so a mask two prompts share sits beside the
    same masks under both, and an example prompt, whose text is too long for some masks, still orders cleanly."""
    return batches(length_order(tasks, PROMPTS[ORDER_PROMPT], tokenizer, mask_id, start_id), size)


def fill_grid(tasks: Iterable[dict[str, Any]], prompt: Prompt, tokenizer: Any, forward_for: Callable[[Any], Any] | None,
              mask_id: int, out: Path, validate: Callable[[str], str | None], device: str = 'cpu',
              start_id: int | None = None, setting: dict[str, Any] = SETTING,
              memory: Callable[[], int] = lambda: 0, sampler: SamplerSetting | None = None,
              row_forward: Any = None, batch: int = 1, group_tasks: Sequence[dict[str, Any]] | None = None) -> int:
    """Fill every task not yet in out (one JSON line each) and return how many were filled. forward_for(slots) is
    the sampler's forward for that mask's slot positions, one mask per forward. With row_forward
    (masked_diffusion_memory_bench.make_row_forward) the masks are filled batch at a time instead, in mask_groups
    of group_tasks (default: tasks), so prompts filled on different mask lists of one grid share their groups (#443)."""
    tasks = list(tasks)
    names, seen, filled = vocabulary(), done_ids(out), 0
    todo = [t for t in tasks if t['id'] not in seen]
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('a') as handle:
        if row_forward is None:
            for task in todo:
                row = fill_one(task, prompt, tokenizer, forward_for, mask_id, validate, names,
                               mask_seed(setting['seed'], task['id']), device, start_id, setting, memory, sampler)
                handle.write(json.dumps(row, ensure_ascii=False) + '\n')
                handle.flush()
                filled += 1
            return filled
        by_id = {t['id']: t for t in todo}
        for group in mask_groups(group_tasks or tasks, tokenizer, mask_id, start_id, batch):
            chunk = [by_id[i] for i in group if i in by_id]
            if not chunk:
                continue
            draws = [(t, mask_seed(setting['seed'], t['id'])) for t in chunk]
            for row in fill_batch(draws, prompt, tokenizer, row_forward, mask_id, validate, names, device, start_id,
                                  setting, memory, sampler, batch):
                handle.write(json.dumps(row, ensure_ascii=False) + '\n')
                filled += 1
            handle.flush()
    return filled


def redraw_seed(seed: int, task_id: str, draw: int) -> int:
    """Draw 0 is the first try's seed; redraw d (1 to REDRAWS) gets its own, so no redraw repeats another."""
    return mask_seed(seed, task_id) if draw == 0 else mask_seed(seed, f'{task_id}#redraw{draw}')


def next_redraw(drawn: dict[int, bool], redraws: int = REDRAWS) -> int | None:
    """The redraw a failing mask needs next, given {draw: legal} for the redraws it has: the first one missing, or
    None once one is legal or all are made."""
    for draw in range(1, redraws + 1):
        if draw not in drawn:
            return draw
        if drawn[draw]:
            return None
    return None


def redraw_rounds(groups: Sequence[Sequence[str]], failing: Sequence[dict[str, Any]],
                  done: dict[str, dict[int, bool]], redraws: int) -> Iterator[tuple[int, list[dict[str, Any]]]]:
    """(draw, the batch to draw) round by round: round d holds each group's masks whose redraws before d all failed.
    Reads done as it is updated, so a mask legal at draw d leaves every later round."""
    by_id = {t['id']: t for t in failing}
    for draw in range(1, redraws + 1):
        for group in groups:
            chunk = [by_id[i] for i in group if i in by_id and next_redraw(done.get(i, {}), redraws) == draw]
            if chunk:
                yield draw, chunk


def redraw_grid(tasks: Iterable[dict[str, Any]], first: Path, prompt: Prompt, tokenizer: Any,
                forward_for: Callable[[Any], Any] | None, mask_id: int, out: Path,
                validate: Callable[[str], str | None], device: str = 'cpu', start_id: int | None = None,
                setting: dict[str, Any] = SETTING, memory: Callable[[], int] = lambda: 0, redraws: int = REDRAWS,
                sampler: SamplerSetting | None = None, row_forward: Any = None, batch: int = 1,
                group_tasks: Sequence[dict[str, Any]] | None = None) -> int:
    """#169's validator loop: a mask whose first try (in first) failed is drawn again, up to redraws more times,
    stopping at its first legal draw. One JSON line per redraw in out, with its draw number; resumes. With
    row_forward the redraws go round by round, each round's draws batch at a time (#443): round d draws every mask
    whose draws before d all failed, so each mask gets the draws, seeds and stopping point the one-at-a-time loop
    gives it."""
    names = vocabulary()
    legal = {r['id']: r['legal'] for r in read_rows(first)}
    done: dict[str, dict[int, bool]] = {}
    for r in read_rows(out) if out.exists() else []:
        done.setdefault(r['id'], {})[r['draw']] = r['legal']
    tasks = list(tasks)
    missing = [t['id'] for t in tasks if t['id'] not in legal]
    if missing:
        raise ValueError(f'{missing[0]} has no first try in {first}')
    failing = [t for t in tasks if not legal[t['id']]]
    filled = 0
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('a') as handle:
        def write(row: dict[str, Any], draw: int) -> None:
            nonlocal filled
            row['draw'] = draw
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')
            handle.flush()
            done.setdefault(row['id'], {})[draw] = row['legal']
            filled += 1

        if row_forward is None:
            for task in failing:
                while (draw := next_redraw(done.get(task['id'], {}), redraws)) is not None:
                    write(fill_one(task, prompt, tokenizer, forward_for, mask_id, validate, names,
                                   redraw_seed(setting['seed'], task['id'], draw), device, start_id, setting, memory,
                                   sampler), draw)
            return filled
        groups = mask_groups(group_tasks or tasks, tokenizer, mask_id, start_id, batch)
        for draw, chunk in redraw_rounds(groups, failing, done, redraws):
            draws = [(t, redraw_seed(setting['seed'], t['id'], draw)) for t in chunk]
            for row in fill_batch(draws, prompt, tokenizer, row_forward, mask_id, validate, names, device,
                                  start_id, setting, memory, sampler, batch):
                write(row, draw)
    return filled


CPU_BATCH = 8               # the CPU has no memory ladder; it is used only by tests and smoke runs


def device_memory(device: str) -> tuple[Callable[[], None], Callable[[], int], int]:
    """(reset the peak, read the peak, the ceiling) for a device. On the laptop the ceiling is
    torch.mps.recommended_max_memory() (#170) and the reading is the driver's allocation, which holds the peak
    because the allocator keeps what it has taken."""
    import torch
    if device.startswith('cuda'):
        return (torch.cuda.reset_peak_memory_stats, torch.cuda.max_memory_allocated,
                torch.cuda.get_device_properties(torch.device(device)).total_memory)
    if device == 'mps':
        return (lambda: None), torch.mps.driver_allocated_memory, torch.mps.recommended_max_memory()
    return (lambda: None), (lambda: 0), 0


def choose_batch(row_forward: Any, longest: dict[str, Any], mask_id: int, active: SamplerSetting, device: str,
                 headroom: float = 0.9, most_masks: dict[str, Any] | None = None,
                 cap: int | None = None) -> dict[str, Any]:
    """The largest batch that fits the device (#443): masked_diffusion_memory_bench.largest_batch, one step each, at
    the worst shape a batch of the run can take: as wide as the longest text and as deep as the text with the most
    masks (b - 1 copies of the longest beside the deepest; at b = 1 each alone). cap is the number of fills to make.
    On the CPU, CPU_BATCH."""
    import torch

    import masked_diffusion_memory_bench as B
    deepest = most_masks or longest
    if device == 'cpu':
        return {'batch': CPU_BATCH if cap is None else min(CPU_BATCH, cap), 'ladder': [],
                'longest_tokens': len(longest['ids']), 'most_masks': len(deepest['slots'])}
    reset, peak, capacity = device_memory(device)
    one_step = replace(active, steps=1)

    def one(texts: list[dict[str, Any]]) -> int:
        if device.startswith('cuda'):
            torch.cuda.empty_cache()
        reset()
        B.fill_rows(row_forward, [c['ids'] for c in texts], [c['slots'] for c in texts], mask_id, one_step,
                    [None] * len(texts), device)
        if device == 'mps':
            torch.mps.synchronize()
        return peak()

    def peak_at(batch: int) -> int:
        if batch == 1:
            return max(one([longest]), one([deepest]))
        return one([longest] * (batch - 1) + [deepest])

    try:
        result = B.largest_batch(peak_at, capacity, headroom=headroom, cap=cap)
    finally:
        if device.startswith('cuda'):
            torch.cuda.empty_cache()
        elif device == 'mps':
            torch.mps.empty_cache()
        reset()                             # the rows' peak readings start after the ladder, not at its last rung
    return result | {'longest_tokens': len(longest['ids']), 'most_masks': len(deepest['slots'])}


def extreme_texts(texts: Iterable[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """(the longest text, the text with the most masks, the shortest text) among a run's canvases: the first two
    size the batch, the first and last are the padding probe's pair."""
    texts = list(texts)
    return (max(texts, key=lambda c: (len(c['ids']), len(c['slots']))),
            max(texts, key=lambda c: (len(c['slots']), len(c['ids']))),
            min(texts, key=lambda c: (len(c['ids']), len(c['slots']))))


def repeat_draw(task: dict[str, Any], seed: int, prompt: Prompt, tokenizer: Any, row_forward: Any, mask_id: int,
                validate: Callable[[str], str | None], names: dict[str, frozenset[str]], device: str,
                start_id: int | None, active: SamplerSetting, batch: int) -> tuple[dict[str, Any], dict[str, Any]]:
    """One mask drawn twice with one seed, for the determinism probe: side by side in one batch, or one after the
    other when only batch 1 fits."""
    if batch >= 2:
        a, b = fill_batch([(task, seed), (task, seed)], prompt, tokenizer, row_forward, mask_id, validate, names,
                          device, start_id, sampler=active)
        return a, b
    return tuple(fill_batch([(task, seed)], prompt, tokenizer, row_forward, mask_id, validate, names, device,
                            start_id, sampler=active)[0] for _ in range(2))


PADDING_TOP1 = 0.95         # the padding probe: share of a padded row's slots whose most likely token is unchanged


def token_share(a: Sequence[Sequence[int]], b: Sequence[Sequence[int]]) -> float:
    """The share of slot tokens two fills of one mask agree on."""
    same = [x == y for sa, sb in zip(a, b, strict=True) for x, y in zip(sa, sb, strict=True)]
    return sum(same) / len(same) if same else 1.0


def padding_probe(row_forward: Any, short: dict[str, Any], long: dict[str, Any], device: str) -> dict[str, Any]:
    """The first step's logits for the short text alone and right-padded beside the long one: how often the most
    likely token at a slot is the same, and the largest logit difference. A padding mask that let the padding in
    would move them far more than the arithmetic of a wider batch does."""
    import torch

    def first_step(texts: list[dict[str, Any]]) -> torch.Tensor:
        width, depth = max(len(c['ids']) for c in texts), max(len(c['slots']) for c in texts)
        x = torch.zeros(len(texts), width, dtype=torch.long)
        attention = torch.zeros(x.shape, dtype=torch.bool)
        keep = torch.zeros(len(texts), depth, dtype=torch.long)
        for n, c in enumerate(texts):
            x[n, :len(c['ids'])], attention[n, :len(c['ids'])] = torch.tensor(c['ids']), True
            keep[n, :len(c['slots'])] = torch.tensor(c['slots'])
        with torch.no_grad():
            return row_forward(x.to(device), attention.to(device), keep.to(device))[0, :len(texts[0]['slots'])].float()

    alone, padded = first_step([short]), first_step([short, long])
    return {'short_tokens': len(short['ids']), 'long_tokens': len(long['ids']),
            'top1_same': float((alone.argmax(-1) == padded.argmax(-1)).float().mean()),
            'max_logit_difference': float((alone - padded).abs().max())}


def final_draws(first: Path, redraws: Path | None) -> dict[str, dict[str, Any]]:
    """Each mask's first legal draw (draw 0 is the first try), or its last failed one: {task id: row}."""
    rows = [dict(r, draw=0) for r in read_rows(first)]
    rows += read_rows(redraws) if redraws is not None and redraws.exists() else []
    out: dict[str, dict[str, Any]] = {}
    for r in sorted(rows, key=lambda r: r['draw']):
        if r['id'] not in out or not out[r['id']]['legal']:
            out[r['id']] = r
    return out


def load_candidate(key: str, device: str) -> tuple[Any, Any, Any, str]:
    """(candidate, tokenizer, model, family) at #162's pinned revision, as the #170 benchmark loads it."""
    import count_masked_diffusion_text as C
    import masked_diffusion_memory_bench as B
    candidate = C.candidate_by_key(key)
    info = B.candidate_info(candidate)
    tokenizer = B.load_tokenizer(info)
    family = B.FAMILY[key]
    model = B.load_model(info, family, SETTING['precision'], device)
    return candidate, tokenizer, model, family


def run_meta(prompt: Prompt, candidate: Any, tasks: list[dict[str, Any]],
             sampler: SamplerSetting | None = None, tasks_sha256: str | None = None,
             batch: dict[str, Any] | None = None) -> dict[str, Any]:
    """run.json: the prompt, model, grid and sampler setting. batch is how the batch size was chosen (#443): the
    size, and the memory ladder that chose it; the setting's batch is that size, 1 for a one-at-a-time run."""
    import masked_diffusion_memory_bench as B
    if tasks_sha256 is None:
        import jev_fill
        tasks_sha256 = hashlib.sha256((jev_fill.GRIDS[GRID] / 'eval2.tasks.json').read_bytes()).hexdigest()
    active = candidate.sampler if sampler is None else sampler
    run_setting = {**SETTING, 'steps': active.steps, 'temperature': active.temperature,
                   'batch': SETTING['batch'] if batch is None else batch['batch']}
    return {'prompt': prompt.code, 'prompt_name': prompt.name, 'candidate': B.candidate_info(candidate),
            'grid': GRID, 'tasks': len(tasks), 'tasks_sha256': tasks_sha256,
            'ask': D.ask_text(ASK), 'setting': run_setting, 'sampler': B.SAMPLER,
            'sampler_setting': active.to_dict(), 'environment': environment(),
            'started': time.strftime('%Y-%m-%dT%H:%M:%S%z'), **({} if batch is None else {'batching': batch})}


def check_run_meta(meta_path: Path, active: SamplerSetting, right_shift: bool) -> None:
    """Refuse to resume or redraw into a directory whose run.json was produced with a different sampler setting."""
    if not meta_path.exists():
        return
    existing = json.loads(meta_path.read_text())
    if 'sampler_setting' in existing:
        recorded = SamplerSetting.from_dict(existing['sampler_setting'])
    else:
        s = existing.get('setting', SETTING)
        recorded = replace(
            SHARED_SAMPLER, steps=s['steps'], temperature=s['temperature'], right_shift=right_shift,
            start_token=bool(existing.get('candidate', {}).get('start_token', False)),
        )
    if recorded != active:
        raise ValueError(f'{meta_path}: existing run used {recorded.to_dict()}, refusing to mix with '
                         f'{active.to_dict()}')


def check_existing_rows(paths: Sequence[Path], active: SamplerSetting, right_shift: bool,
                        start_token: bool) -> None:
    """Refuse to resume or redraw when saved rows (even without run.json) came from a different sampler setting."""
    legacy_default = replace(SHARED_SAMPLER, steps=SETTING['steps'], temperature=SETTING['temperature'],
                             right_shift=right_shift, start_token=start_token)
    for path in paths:
        if not path.exists():
            continue
        for row in read_rows(path):
            recorded = (SamplerSetting.from_dict(row['sampler_setting']) if 'sampler_setting' in row
                        else legacy_default)
            if recorded != active:
                raise ValueError(f'{path}: row {row.get("id")!r} used {recorded.to_dict()}, refusing to mix with '
                                 f'{active.to_dict()}')


def grid_tasks() -> list[dict[str, Any]]:
    import jev_fill
    tasks = jev_fill.tasks(GRID)
    if len(tasks) != GRID_SIZE:
        raise ValueError(f'the held-out grid has {len(tasks)} masks, not {GRID_SIZE}')
    return tasks


def resumed_batch(paths: Sequence[Path], batch: int | None) -> int | None:
    """The batch size a resumed run keeps: the one its rows were filled at (rows from before #443 count as 1), the
    given size, or None (find it) when nothing is filled yet. Refuses a different given size and mixed sizes."""
    sizes = {r.get('batch_size', 1) for path in paths if path.exists() for r in read_rows(path)}
    if len(sizes) > 1:
        raise ValueError(f'{paths[0].parent}: rows filled at batch sizes {sorted(sizes)}')
    if sizes and batch is not None and batch not in sizes:
        raise ValueError(f'{paths[0].parent}: rows were filled at batch {sizes.pop()}, not {batch}')
    return sizes.pop() if sizes else batch


def record_batching(meta: dict[str, Any], sizing: dict[str, Any], redraw: bool) -> dict[str, Any]:
    """run.json after a resumed fill or a redraw pass: the pass's batching appended to batches, and for redraws the
    size in the setting as redraw_batch beside the first try's batch."""
    meta.setdefault('batches', []).append(sizing | {'when': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'redraw': redraw})
    if redraw:
        meta['setting']['redraw_batch'] = sizing['batch']
    return meta


def run(prompt_code: str, key: str, directory: Path, limit: int | None, device: str, redraw: bool = False,
        sampler_name: str | None = None, batch: int | None = None) -> int:
    """Fill (or redraw) the grid with one prompt and model, batch at a time: batch masks per forward pass, or with
    batch None the largest batch that fits the device (#443)."""
    import torch

    import masked_diffusion_memory_bench as B
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
    from showdown_validator import Validator
    prompt, tasks = PROMPTS[prompt_code], prompt_tasks(prompt_code, grid_tasks())
    if device == 'mps':
        torch.mps.set_per_process_memory_fraction(1.0)
    candidate, tokenizer, model, family = load_candidate(key, device)
    active = resolve_sampler(candidate.sampler, sampler_name)
    out = directory / prompt.code / key
    out.mkdir(parents=True, exist_ok=True)
    meta_path = out / 'run.json'
    check_run_meta(meta_path, active, candidate.sampler.right_shift)
    check_existing_rows((out / 'fills.jsonl', out / 'redraws.jsonl'), active,
                        candidate.sampler.right_shift, candidate.sampler.start_token)
    start_id = start_token_id(tokenizer, active)
    row_forward = B.make_row_forward(model, family, right_shift=active.right_shift)
    batch = resumed_batch([out / 'fills.jsonl'] + ([out / 'redraws.jsonl'] if redraw else []), batch)
    if batch is None:
        # every prompt's texts for these masks, so each prompt of a pair is sized alike on this device
        every = [canvas(t, other, tokenizer, candidate.mask_id, start_id) for other in PROMPTS.values()
                 for t in prompt_tasks(other.code, grid_tasks())[:limit]]
        longest, most_masks, _ = extreme_texts(every)
        sizing = choose_batch(row_forward, longest, candidate.mask_id, active, device, most_masks=most_masks,
                              cap=len(tasks[:limit]))
    else:
        sizing = {'batch': batch, 'ladder': [], 'chosen_by': 'given or resumed'}
    print(f"{candidate.label}: batch {sizing['batch']} ({json.dumps(sizing['ladder'])})", flush=True)
    meta = run_meta(prompt, candidate, tasks, active, batch=sizing)
    if meta_path.exists():
        meta = record_batching(json.loads(meta_path.read_text()), sizing, redraw)
    meta_path.write_text(json.dumps(meta, indent=1) + '\n')
    memory = (lambda: torch.mps.driver_allocated_memory()) if device == 'mps' else (lambda: 0)
    validate = Validator()
    try:
        if redraw:
            filled = redraw_grid(tasks[:limit], out / 'fills.jsonl', prompt, tokenizer, None, candidate.mask_id,
                                 out / 'redraws.jsonl', validate, device, start_id, memory=memory, sampler=active,
                                 row_forward=row_forward, batch=sizing['batch'], group_tasks=grid_tasks())
        else:
            filled = fill_grid(tasks[:limit], prompt, tokenizer, None, candidate.mask_id, out / 'fills.jsonl',
                               validate, device, start_id, memory=memory, sampler=active, row_forward=row_forward,
                               batch=sizing['batch'], group_tasks=grid_tasks())
    finally:
        validate.close()
    if redraw:
        final = final_draws(out / 'fills.jsonl', out / 'redraws.jsonl')
        print(f'{candidate.label}, {prompt.name}: {filled} redraws this run; legal after redraws '
              f'{sum(r["legal"] for r in final.values())} of {len(final)}', flush=True)
    else:
        total = len(done_ids(out / 'fills.jsonl'))
        print(f'{candidate.label}, {prompt.name}: filled {filled} masks this run, {total} of {len(tasks)} in all',
              flush=True)
    return 0


def legal_teams(prompt_code: str, key: str, directory: Path) -> dict[str, Any]:
    """Each mask's first legal draw as a paste, for the battles (#171), in D/P/K/teams.json. Every legal row is
    judged again from its saved values with the tokenizer alone; a row that no longer reads as legal stops it."""
    import masked_diffusion_memory_bench as B
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
    import count_masked_diffusion_text as C
    from showdown_validator import Validator
    prompt, tasks = PROMPTS[prompt_code], {t['id']: t for t in grid_tasks()}
    candidate = C.candidate_by_key(key)
    tokenizer = B.load_tokenizer(B.candidate_info(candidate))
    start_id = tokenizer.bos_token_id if candidate.sampler.start_token else None
    out = directory / prompt.code / key
    final, names, validate = final_draws(out / 'fills.jsonl', out / 'redraws.jsonl'), vocabulary(), Validator()
    teams = {}
    try:
        for task_id, row in final.items():
            if not row['legal']:
                continue
            c = canvas(tasks[task_id], prompt, tokenizer, candidate.mask_id, start_id)
            again = judge(tasks[task_id], c['built'], row['values'], validate, names)
            if not again['legal']:
                raise ValueError(f'{key} {task_id} draw {row["draw"]}: legal when filled, not when judged again')
            teams[task_id] = {'draw': row['draw'], 'source': row['source'], 'task': row['task'], 'k': row['k'],
                              'recovered': row['recovered'], 'paste': team_paste(again['team'])}
    finally:
        validate.close()
    draws = Counter(r['draw'] for r in final.values() if r['legal'])
    result = {'candidate': key, 'prompt': prompt.code, 'masks': len(final), 'legal': len(teams),
              'legal_first_try': draws.get(0, 0), 'legal_by_draw': {str(d): draws[d] for d in sorted(draws)},
              'draws_made': len(final) + (len(read_rows(out / 'redraws.jsonl')) if (out / 'redraws.jsonl').exists() else 0),
              'teams': teams}
    (out / 'teams.json').write_text(json.dumps(result, indent=1, ensure_ascii=False) + '\n')
    print(f"{candidate.label}: {len(teams)} of {len(final)} masks legal after redraws "
          f"({result['legal_by_draw']}), {result['draws_made']} draws", flush=True)
    return result


# ---------- the report ----------

def binomial_cdf(k: int, n: int, p: float) -> float:
    if p <= 0:
        return 1.0
    if p >= 1:
        return 0.0 if k < n else 1.0
    return sum(math.exp(math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
                        + i * math.log(p) + (n - i) * math.log1p(-p)) for i in range(k + 1))


def exact_interval(k: int, n: int, level: float = 0.95) -> tuple[float, float]:
    """Clopper-Pearson interval for k of n, by bisection on the binomial tail."""
    alpha = (1 - level) / 2

    def solve(f: Callable[[float], float]) -> float:
        lo, hi = 0.0, 1.0
        for _ in range(60):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if f(mid) > 0 else (lo, mid)
        return (lo + hi) / 2

    low = 0.0 if k == 0 else solve(lambda p: alpha - (1 - binomial_cdf(k - 1, n, p)))
    high = 1.0 if k == n else solve(lambda p: binomial_cdf(k, n, p) - alpha)
    return low, high


def summarize(rows: list[dict[str, Any]], names: dict[str, frozenset[str]] | None = None) -> dict[str, Any]:
    names = names or vocabulary()
    n = len(rows)
    legal = sum(r['legal'] for r in rows)
    field_values: dict[str, list[str]] = {}
    for r in rows:
        for field, value in zip(r['fields'], r['values'], strict=True):
            field_values.setdefault(field, []).append(value)
    real = {f: (sum(to_id(v) in names[f] for v in vs), len(vs)) for f, vs in field_values.items() if f in names}
    species = [v for v in field_values.get('species', []) if to_id(v) in names['species']]
    first_failed = Counter(STEP_NAMES[r['failed']] for r in rows)
    reasons = Counter(reason for r in rows for reason in r['reasons'])
    showdown_reasons = Counter(reason_group(e) for r in rows for e in set(r['showdown_errors'] or []))
    peak = max((r['peak_bytes'] for r in rows), default=0)
    return {'masks': n, 'legal': legal, 'legal_share': legal / n if n else 0.0,
            'legal_interval': exact_interval(legal, n) if n else (0.0, 1.0),
            'passes_169_bar': legal >= 3,
            'showdown_accepts': sum(r['showdown_accepts'] for r in rows),
            'json_parses': sum(r['json_parses'] for r in rows), 'reads_as_team': sum(r['reads_as_team'] for r in rows),
            'real_values': {f: {'real': a, 'of': b} for f, (a, b) in real.items()},
            'distinct_real_species': len({to_id(s) for s in species}),
            'recovered_unmasked_team': sum(r['recovered'] for r in rows),
            'names_copied_from_visible': sum(r['copied_visible'] for r in rows),
            'values_with_a_quote': sum('"' in v for vs in field_values.values() for v in vs),
            'values': sum(map(len, field_values.values())),
            'legal_by_task': {kind: {'legal': sum(r['legal'] for r in rows if r['task'] == kind),
                                     'of': sum(r['task'] == kind for r in rows)}
                              for kind in sorted({r['task'] for r in rows})},
            'first_failed_step': dict(first_failed.most_common()), 'reasons': dict(reasons.most_common()),
            'showdown_reasons': dict(showdown_reasons.most_common()),
            'prompt_tokens_mean': sum(r['prompt_tokens'] for r in rows) / n if n else 0,
            'prompt_tokens_max': max((r['prompt_tokens'] for r in rows), default=0),
            'seconds_per_team': sum(r['seconds'] for r in rows) / n if n else 0,
            'seconds_total': sum(r['seconds'] for r in rows), 'peak_gib': peak / 2**30,
            'special_tokens_dropped': sum(r['special_dropped'] for r in rows),
            'example_copies': example_copies(rows),
            'rules': rule_counts(rows)}


def rule_counts(rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, int]]:
    """For each rule, the fills that break it out of the fills where it could be judged. Rows filled before
    rule_breaks existed carry no 'rules' and are left out; rules_for fills them in."""
    judged = [r['rules'] for r in rows if r.get('rules')]
    return {rule: {'broken': sum(bool(j[rule]) for j in judged), 'of': sum(j[rule] is not None for j in judged)}
            for rule in RULES}


def read_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def read_fills(path: Path) -> list[dict[str, Any]]:
    """A fills.jsonl's rows, each with its rule breaks: from the row, or from rules.jsonl beside it for fills judged
    before rule_breaks existed."""
    rows = read_rows(path)
    side = path.with_name('rules.jsonl')
    if side.exists():
        rules = {r['id']: r['rules'] for r in read_rows(side)}
        rows = [r if r.get('rules') else {**r, 'rules': rules.get(r['id'])} for r in rows]
    return rows


def rules_for(directory: Path, prompt_code: str, key: str, tasks: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Judge every saved fill of one run again from its values and write its rule breaks to rules.jsonl beside
    fills.jsonl, leaving fills.jsonl as it is. Refuses a fill whose verdict comes out different from the saved one."""
    import count_masked_diffusion_text as C

    import masked_diffusion_memory_bench as B
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
    from showdown_validator import Validator
    out = directory / prompt_code / key
    by_id = {t['id']: t for t in (tasks or grid_tasks())}
    candidate = C.candidate_by_key(key)
    tokenizer = B.load_tokenizer(B.candidate_info(candidate))
    start_id = start_token_id(tokenizer, candidate.sampler)
    names, validate, lines = vocabulary(), Validator(), []
    try:
        for row in read_rows(out / 'fills.jsonl'):
            c = canvas(by_id[row['id']], PROMPTS[prompt_code], tokenizer, candidate.mask_id, start_id)
            again = judge(by_id[row['id']], c['built'], row['values'], validate, names)
            if (again['legal'], again['failed']) != (row['legal'], row['failed']):
                raise ValueError(f"{out} {row['id']}: judged {again['failed']!r} now, {row['failed']!r} when filled")
            lines.append(json.dumps({'id': row['id'], 'rules': again['rules']}, ensure_ascii=False))
    finally:
        validate.close()
    (out / 'rules.jsonl').write_text('\n'.join(lines) + '\n')
    return {'prompt': prompt_code, 'key': key, **rule_counts(read_fills(out / 'fills.jsonl'))}


def report(directory: Path, prompt_code: str) -> str:
    import count_masked_diffusion_text as C
    labels = {c.key: c.label for c in C.CANDIDATES}
    results = {}
    for key in SMALL_FOUR:
        path = directory / prompt_code / key / 'fills.jsonl'
        if path.exists():
            results[key] = summarize(read_fills(path))
    (directory / prompt_code).mkdir(parents=True, exist_ok=True)
    (directory / prompt_code / 'report.json').write_text(json.dumps(results, indent=1) + '\n')
    lines = [f'Prompt: {PROMPTS[prompt_code].name} (`{prompt_code}`). Held-out grid, first try only.', '',
             '| Model | Legal, first try | 95% interval | Showdown accepts | Parses as a team | Real species | '
             'Prompt tokens, mean | Seconds per team |', '|---|---|---|---|---|---|---|---|']
    for key, s in results.items():
        lo, hi = s['legal_interval']
        sp = s['real_values'].get('species', {'real': 0, 'of': 0})
        lines.append(f"| {labels[key]} | {s['legal']}/{s['masks']} | {lo:.2%} to {hi:.2%} | "
                     f"{s['showdown_accepts']}/{s['masks']} | {s['reads_as_team']}/{s['masks']} | "
                     f"{sp['real']}/{sp['of']} | {s['prompt_tokens_mean']:.0f} | {s['seconds_per_team']:.1f} |")
    return '\n'.join(lines)


# ---------- probes ----------

def true_values(task: dict[str, Any]) -> list[str]:
    real = {piece[0]: piece[1] for piece in json_pieces(task['original']) if not isinstance(piece, str)}
    return [value_text(real[path]) for path in D.hidden_values(task)]


def random_token_values(tokenizer: Any, c: dict[str, Any], rng: random.Random) -> list[str]:
    """The floor: every slot token drawn uniformly from the tokenizer's vocabulary, special tokens excluded."""
    special = set(getattr(tokenizer, 'all_special_ids', []))
    vocab = [i for i in range(len(tokenizer)) if i not in special]
    return [tokenizer.decode([rng.choice(vocab) for _ in span]) for span in c['spans']]


def round_trip_values(tokenizer: Any, task: dict[str, Any], c: dict[str, Any], start_id: int | None,
                      prompt: Prompt = PROMPTS['inplace_header']) -> list[str]:
    """The true values written into the slots as the training target writes them (padded), encoded, then decoded
    from the slot tokens as a model's fill is."""
    pad = tokenizer.pad_token or tokenizer.eos_token
    target = D.training_target(task, ASK, tokenizer, LIMIT, pad, prompt.statement, prompt.example)
    ids = list(tokenizer(target, add_special_tokens=False)['input_ids'])
    if start_id is not None:
        ids = [start_id, *ids]
    return slot_values(tokenizer, ids, c['spans'])[0]


def statement_tokens(tasks: Sequence[dict[str, Any]], prompt: Prompt, tokenizer: Any, mask_id: int,
                     start_id: int | None) -> dict[str, int]:
    """What the prompt's task statement costs in this tokenizer: its tokens alone, and the fewest and most tokens it
    adds to a mask's text over today's prompt (#238)."""
    alone = len(tokenizer(prompt.statement or '', add_special_tokens=False)['input_ids'])
    added = [len(canvas(t, prompt, tokenizer, mask_id, start_id)['ids'])
             - len(canvas(t, PROMPTS['inplace_header'], tokenizer, mask_id, start_id)['ids']) for t in tasks]
    return {'statement_alone': alone, 'added_min': min(added), 'added_max': max(added), 'masks': len(added)}


def example_tokens(tasks: Sequence[dict[str, Any]], prompt: Prompt, tokenizer: Any, mask_id: int,
                   start_id: int | None) -> dict[str, int]:
    """What the example team costs in this tokenizer: its line alone, and the fewest and most tokens it adds to a
    mask's text over the prompt it extends (#239)."""
    alone = len(tokenizer(prompt.example or '', add_special_tokens=False)['input_ids'])
    base = PROMPTS[prompt.base or 'inplace_header']
    added = [len(canvas(t, prompt, tokenizer, mask_id, start_id)['ids'])
             - len(canvas(t, base, tokenizer, mask_id, start_id)['ids']) for t in tasks]
    return {'example_alone': alone, 'added_min': min(added), 'added_max': max(added), 'masks': len(added)}


def example_team_check(validate: Callable[[str], str | None], names: dict[str, frozenset[str]]) -> dict[str, Any]:
    """#239: the example team passes the validator and the Reg M-B check, and is none of the grid's unmasked teams
    and no held-out team."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
    import heldout as H
    team = D.example_team()
    named = [(path[1], value) for path, value in visible_values(team).items() if path[1] in NAME_FIELDS]
    not_regmb = [f'{field} {value}' for field, value in named if value and to_id(value) not in names[field]]
    error = validate(team_paste(team))
    sources = {t['source']: t['original'] for t in grid_tasks()}
    in_grid = any(H.grid_key_team(team) == H.grid_key_team(o) for o in sources.values())
    out = {'source': json.loads(D.EXAMPLE_TEAM.read_text())['source'], 'species': [m['species'] for m in team],
           'showdown_error': error, 'not_regmb': not_regmb, 'is_a_grid_team': in_grid,
           'is_heldout': bool(H.is_heldout_team(team)),
           'species_shared_with_grid_teams': sorted({m['species'] for m in team}
                                                    & {m['species'] for o in sources.values() for m in o})}
    out['pass'] = error is None and not not_regmb and not in_grid and not out['is_heldout']
    return out


def preflight(out: Path, prompt_code: str = 'inplace_header') -> dict[str, Any]:
    """The probes that need no model: run on every mask (for an example prompt, every mask that fits) with every
    small-four tokenizer."""
    import count_masked_diffusion_text as C
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
    from showdown_validator import Validator
    every, names, prompt = grid_tasks(), vocabulary(), PROMPTS[prompt_code]
    tasks = prompt_tasks(prompt_code, every)
    validate = Validator()
    result: dict[str, Any] = {'date': time.strftime('%Y-%m-%d'), 'prompt': prompt.code, 'masks': len(tasks),
                              'environment': environment(), 'candidates': {}}
    if prompt.example is not None:
        kept = {t['id'] for t in tasks}
        result['too_long'] = {'masks': len(every) - len(tasks),
                              'cells': dict(Counter(f"{t['task']} {t['k']}" for t in every if t['id'] not in kept))}
    try:
        for key in SMALL_FOUR:
            candidate = C.candidate_by_key(key)
            tokenizer = C.load_tokenizer(candidate)
            start_id = tokenizer.bos_token_id if candidate.sampler.start_token else None
            rng = random.Random(SETTING['seed'])
            row: dict[str, Any] = {'slot_lengths': D.slot_lengths(tokenizer)}
            ceiling = trip = trip_paste = floor = floor_showdown = built = 0
            unknown, lengths, trip_misses = 0, [], []
            for task in tasks:
                c = canvas(task, prompt, tokenizer, candidate.mask_id, start_id)
                built += 1
                lengths.append(len(c['ids']))
                if tokenizer.unk_token_id is not None:
                    unknown += c['ids'].count(tokenizer.unk_token_id)
                ceiling += judge(task, c['built'], true_values(task), validate, names)['legal']
                values = round_trip_values(tokenizer, task, c, start_id, prompt)
                verdict = judge(task, c['built'], values, validate, names)
                trip += verdict['legal']
                same = verdict.get('team') is not None and team_paste(verdict['team']) == team_paste(task['original'])
                trip_paste += same
                if not same:
                    trip_misses.append(task['id'])
                noise = judge(task, c['built'], random_token_values(tokenizer, c, rng), validate, names)
                floor += noise['legal']
                floor_showdown += noise['showdown_accepts']
            row |= {'masks_built': built, 'masks_dropped': len(tasks) - built, 'unknown_tokens': unknown,
                    'prompt_tokens_min': min(lengths), 'prompt_tokens_max': max(lengths),
                    'ceiling_legal': ceiling, 'round_trip_legal': trip, 'round_trip_same_paste': trip_paste,
                    'round_trip_misses': trip_misses[:10], 'floor_random_tokens_legal': floor,
                    'floor_random_tokens_showdown_accepts': floor_showdown}
            if prompt.statement is not None:
                row['statement_tokens'] = statement_tokens(tasks, prompt, tokenizer, candidate.mask_id, start_id)
            if prompt.example is not None:
                row['example_tokens'] = example_tokens(tasks, prompt, tokenizer, candidate.mask_id, start_id)
            row['pass'] = (ceiling == trip == trip_paste == len(tasks)) and floor < 3 and unknown == 0
            result['candidates'][key] = row
            print(key, json.dumps(row), flush=True)
        result['controls'] = controls(validate, names)
        if prompt.example is not None:
            result['example_team'] = example_team_check(validate, names)
    finally:
        validate.close()
    result['pass'] = (all(r['pass'] for r in result['candidates'].values()) and result['controls']['pass']
                      and result.get('example_team', {'pass': True})['pass'])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1) + '\n')
    return result


def controls(validate: Callable[[str], str | None], names: dict[str, frozenset[str]]) -> dict[str, Any]:
    """The checker on known teams: the five unmasked teams and 20 uniform-legal teams must be legal; five corrupted
    copies must not, including the junk Stat Alignment Showdown alone accepts."""
    import copy
    originals, seen = [], set()
    for task in grid_tasks():
        if task['source'] not in seen:
            seen.add(task['source'])
            originals.append(task['original'])

    def strict(team: Team) -> bool:
        ok = all(to_id(mon[f]) in names[f] for mon in team for f in ('species', 'item', 'ability', 'nature'))
        ok &= all(to_id(m) in names['moves'] for mon in team for m in mon['moves'])
        return ok and validate(team_paste(team)) is None

    uniform = []
    path = Path.home() / 'vgc-data/teams/hps_reg_mb_100k.jsonl'
    with path.open() as handle:
        lines = [next(handle) for _ in range(2000)]
    for line in random.Random(SETTING['seed']).sample(lines, 20):
        team = team_from_paste_row(json.loads(line))
        if team is not None:
            uniform.append(team)
    corrupt = []
    for change in (lambda t: t[0].__setitem__('nature', '00'), lambda t: t[1].__setitem__('species', t[0]['species']),
                   lambda t: t[2].__setitem__('item', 'SpA'), lambda t: t[3]['moves'].__setitem__(0, 'Japan'),
                   lambda t: t[4]['statPoints'].__setitem__('HP', 40)):
        team = copy.deepcopy(originals[0])
        change(team)
        corrupt.append(team)
    showdown_only = [validate(team_paste(t)) is None for t in corrupt]
    out = {'unmasked_legal': sum(map(strict, originals)), 'unmasked': len(originals),
           'uniform_legal': sum(map(strict, uniform)), 'uniform': len(uniform),
           'corrupt_rejected': sum(not strict(t) for t in corrupt), 'corrupt': len(corrupt),
           'corrupt_showdown_alone_accepts': sum(showdown_only)}
    out['pass'] = (out['unmasked_legal'] == out['unmasked'] and out['uniform_legal'] == out['uniform'] >= 15
                   and out['corrupt_rejected'] == out['corrupt'])
    return out


def team_from_paste_row(row: dict[str, Any]) -> Team | None:
    """A uniform-legal team row from hps_reg_mb_100k.jsonl (its 'team' is paste text) as Mask Lab slots, or None
    when it is not six Pokemon."""
    blocks = [b for b in row['team'].strip().split('\n\n') if b.strip()]
    if len(blocks) != 6:
        return None
    team = []
    for i, block in enumerate(blocks, 1):
        lines = block.splitlines()
        species, _, item = lines[0].partition(' @ ')
        mon: dict[str, Any] = {'slot': i, 'species': species.strip(), 'item': item.strip(), 'ability': '',
                               'nature': '', 'moves': [], 'statPoints': dict.fromkeys(STATS, 0),
                               'ivs': dict.fromkeys(STATS, 31), 'level': 50}
        for line in lines[1:]:
            if line.startswith('Ability: '):
                mon['ability'] = line.removeprefix('Ability: ').strip()
            elif line.startswith('EVs: '):
                for part in line.removeprefix('EVs: ').split(' / '):
                    number, stat = part.split()
                    mon['statPoints'][stat] = int(number)
            elif line.endswith(' Nature'):
                mon['nature'] = line.removesuffix(' Nature').strip()
            elif line.startswith('- '):
                mon['moves'].append(line[2:].strip())
        mon['moves'] = (mon['moves'] + [''] * 4)[:4]
        team.append(mon)
    return team


def gpu_preflight(key: str, directory: Path, device: str = 'mps', sampler_name: str | None = None,
                  prompt_code: str = 'inplace_header') -> dict[str, Any]:
    """The probes that need the model, on three masks: attention sees later tokens, the same seed refills the same
    values, running the masks in reverse order changes nothing, and a redraw's seed gives a different fill (#171:
    a redraw that repeats the first try would make the validator loop a no-op)."""
    import torch

    import masked_diffusion_memory_bench as B
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
    torch.mps.set_per_process_memory_fraction(1.0)
    candidate, tokenizer, model, family = load_candidate(key, device)
    active = resolve_sampler(candidate.sampler, sampler_name)
    tasks = grid_tasks()
    picks = [next(t for t in tasks if t['task'] == kind and t['k'] == k)
             for kind, k in (('items', 1), ('moves', 8), ('pokemon', 2))]
    start_id = start_token_id(tokenizer, active)

    def draw(task: dict[str, Any], seed: int | None = None) -> list[int]:
        c = canvas(task, PROMPTS[prompt_code], tokenizer, candidate.mask_id, start_id)
        x, slots = torch.tensor([c['ids']], device=device), torch.tensor(c['slots'], device=device)
        seed = mask_seed(SETTING['seed'], task['id']) if seed is None else seed
        generator = torch.Generator(device=device).manual_seed(seed)
        forward = B.make_forward(model, family, SETTING['scope'], slots, right_shift=active.right_shift)
        filled = B.fill(forward, x, slots, candidate.mask_id, generator=generator, setting=active)
        return filled[0, slots].cpu().tolist()

    c0 = canvas(picks[0], PROMPTS[prompt_code], tokenizer, candidate.mask_id, start_id)
    sees = B.sees_later_tokens(model, family, torch.tensor([c0['ids']], device=device),
                               torch.tensor(c0['slots'], device=device), right_shift=active.right_shift)
    first = [draw(t) for t in picks]
    again = [draw(t) for t in picks]
    reverse = [draw(t) for t in reversed(picks)][::-1]
    redrawn = [draw(t, redraw_seed(SETTING['seed'], t['id'], 1)) for t in picks]

    def same(a: list[list[int]], b: list[list[int]]) -> float:
        pairs = [(x, y) for p, q in zip(a, b, strict=True) for x, y in zip(p, q, strict=True)]
        return sum(x == y for x, y in pairs) / len(pairs)

    out = {'candidate': key, 'prompt': prompt_code, 'sampler_setting': active.to_dict(), 'masks': [t['id'] for t in picks],
           'sees_later_tokens': sees, 'repeat_same_tokens': same(first, again),
           'reverse_order_same_tokens': same(first, reverse), 'redraw_same_tokens': same(first, redrawn),
           'environment': environment()}
    redraw_ok = (out['redraw_same_tokens'] == 1.0) if active.temperature == 0.0 else (out['redraw_same_tokens'] < 1.0)
    out['pass'] = (sees and out['repeat_same_tokens'] >= 0.9 and out['reverse_order_same_tokens'] >= 0.9
                   and redraw_ok)
    path = gpu_preflight_path(directory, key, prompt_code)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps(out), flush=True)
    return out


def gpu_preflight_path(directory: Path, key: str, prompt_code: str = 'inplace_header') -> Path:
    """Today's prompt keeps #237's file name; a later prompt's probes sit beside it under the prompt's code."""
    name = f'gpu-{key}.json' if prompt_code == 'inplace_header' else f'gpu-{prompt_code}-{key}.json'
    return directory / 'preflight' / name


# ---------- two prompts on the same masks (#238 to #240) ----------

def paired(before: Sequence[dict[str, Any]], after: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """First-try legality of the same masks under two prompts, paired by mask id: masks legal under both, neither,
    only after (gained) and only before (lost), with McNemar's exact two-sided p-value on gained against lost."""
    for side, rows in (('before', before), ('after', after)):
        if len({r['id'] for r in rows}) != len(rows):
            raise ValueError(f'the {side} run has a mask more than once')
    was, now = {r['id']: r['legal'] for r in before}, {r['id']: r['legal'] for r in after}
    if set(was) != set(now):
        raise ValueError(f'the two runs cover different masks: {len(set(was) ^ set(now))} in only one')
    gained = sum(now[i] and not was[i] for i in was)
    lost = sum(was[i] and not now[i] for i in was)
    n = gained + lost
    p_value = 1.0 if n == 0 else min(1.0, 2 * binomial_cdf(min(gained, lost), n, 0.5))
    return {'both': sum(was[i] and now[i] for i in was), 'neither': sum(not was[i] and not now[i] for i in was),
            'gained': gained, 'lost': lost, 'p_value': p_value}


def compare_rows(before: list[dict[str, Any]], after: list[dict[str, Any]],
                 names: dict[str, frozenset[str]] | None = None) -> dict[str, Any]:
    """The ticket's before-and-after measures for one model: first-try legal, JSON parse share, real-species share,
    seconds per team, the most common reasons, and the paired test."""
    a, b = summarize(before, names), summarize(after, names)

    def pair(key: str) -> dict[str, int]:
        return {'before': a[key], 'after': b[key], 'of': a['masks']}

    species = {s: x['real_values'].get('species', {'real': 0, 'of': 0}) for s, x in (('before', a), ('after', b))}
    return {'legal': pair('legal'), 'json_parses': pair('json_parses'), 'reads_as_team': pair('reads_as_team'),
            'showdown_accepts': pair('showdown_accepts'),
            'real_species': {'before': species['before']['real'], 'after': species['after']['real'],
                             'of': species['before']['of']},
            'seconds_per_team': {'before': a['seconds_per_team'], 'after': b['seconds_per_team']},
            'prompt_tokens_mean': {'before': a['prompt_tokens_mean'], 'after': b['prompt_tokens_mean']},
            'legal_by_task': {'before': a['legal_by_task'], 'after': b['legal_by_task']},
            'first_failed_step': {'before': a['first_failed_step'], 'after': b['first_failed_step']},
            'reasons': {s: [list(x) for x in Counter(r for row in rows for r in row['reasons']).most_common(5)]
                        for s, rows in (('before', before), ('after', after))},
            'rules': {'before': a['rules'], 'after': b['rules']},
            'paired': paired(before, after)}


def shared_ids(*prompt_codes: str) -> frozenset[str] | None:
    """The masks every one of these prompts is filled on, or None when that is the whole grid."""
    sets = [k for k in map(fitting_ids, prompt_codes) if k is not None]
    return frozenset.intersection(*sets) if sets else None


def compare(directory: Path, before_code: str, after_code: str) -> str:
    import count_masked_diffusion_text as C
    labels = {c.key: c.label for c in C.CANDIDATES}
    results = {}
    keep = shared_ids(before_code, after_code)
    for key in SMALL_FOUR:
        paths = [directory / code / key / 'fills.jsonl' for code in (before_code, after_code)]
        if all(p.exists() for p in paths):
            before, after = ([r for r in read_fills(p) if keep is None or r['id'] in keep] for p in paths)
            results[key] = compare_rows(before, after)
    (directory / after_code).mkdir(parents=True, exist_ok=True)
    name = 'compare.json' if before_code in ('inplace_header', PROMPTS[after_code].base) else f'compare-{before_code}.json'
    masks = GRID_SIZE if keep is None else len(keep)
    (directory / after_code / name).write_text(
        json.dumps({'before': before_code, 'after': after_code, 'masks': masks, 'models': results}, indent=1) + '\n')
    lines = [f'{PROMPTS[before_code].name} (`{before_code}`) → {PROMPTS[after_code].name} (`{after_code}`). '
             f'Held-out grid, the same {masks} masks and mask seeds, first try only.', '',
             '| Model | Legal, first try | Gained / lost (McNemar p) | Parses as JSON | Real species | '
             'Seconds per team | Most common reasons, after |', '|---|---|---|---|---|---|---|']
    for key, c in results.items():
        p = c['paired']
        reasons = '; '.join(f'{r} ({n})' for r, n in c['reasons']['after'][:3])
        lines.append(f"| {labels[key]} | {c['legal']['before']} → {c['legal']['after']} of {c['legal']['of']} | "
                     f"{p['gained']} / {p['lost']} (p = {p['p_value']:.3g}) | "
                     f"{c['json_parses']['before']} → {c['json_parses']['after']} | "
                     f"{c['real_species']['before']} → {c['real_species']['after']} of {c['real_species']['of']} | "
                     f"{c['seconds_per_team']['before']:.1f} → {c['seconds_per_team']['after']:.1f} | {reasons} |")
    return '\n'.join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = parser.add_subparsers(dest='command', required=True)
    pre = sub.add_parser('preflight')
    pre.add_argument('--out', type=Path, help='default preflight/cpu.json, or cpu-<prompt>.json for a later prompt')
    pre.add_argument('--prompt', choices=PROMPTS, default='inplace_header')
    one = sub.add_parser('run')
    one.add_argument('--prompt', choices=PROMPTS, required=True)
    one.add_argument('--key', choices=SMALL_FOUR, required=True)
    one.add_argument('--dir', type=Path, default=OUT)
    one.add_argument('--limit', type=int)
    one.add_argument('--device', default='mps')
    again = sub.add_parser('redraw')
    again.add_argument('--prompt', choices=PROMPTS, required=True)
    again.add_argument('--key', choices=SMALL_FOUR, required=True)
    again.add_argument('--dir', type=Path, default=OUT)
    again.add_argument('--limit', type=int)
    again.add_argument('--device', default='mps')
    legal = sub.add_parser('teams')
    legal.add_argument('--prompt', choices=PROMPTS, required=True)
    legal.add_argument('--key', choices=SMALL_FOUR, required=True)
    legal.add_argument('--dir', type=Path, default=OUT)
    gpu = sub.add_parser('gpu-preflight')
    gpu.add_argument('--key', choices=SMALL_FOUR, required=True)
    gpu.add_argument('--dir', type=Path, default=OUT)
    gpu.add_argument('--prompt', choices=PROMPTS, default='inplace_header')
    for p in (one, again):
        p.add_argument('--batch', type=int, default=None,
                       help='masks per forward pass (default: the largest that fits the device, #443)')
    for p in (one, again, gpu):
        p.add_argument('--sampler', choices=tuple(SAMPLERS), default=None,
                       help="named sampler setting (default: the candidate's own)")
    rep = sub.add_parser('report')
    rep.add_argument('--dir', type=Path, default=OUT)
    rep.add_argument('--prompt', choices=PROMPTS, default='inplace_header')
    rul = sub.add_parser('rules', help='write rules.jsonl beside a run judged before rule_breaks existed')
    rul.add_argument('--dir', type=Path, default=OUT)
    rul.add_argument('--prompt', choices=PROMPTS, required=True)
    rul.add_argument('--key', choices=SMALL_FOUR, required=True)
    both = sub.add_parser('compare')
    both.add_argument('--dir', type=Path, default=OUT)
    both.add_argument('--before', choices=PROMPTS, default='inplace_header')
    both.add_argument('--after', choices=PROMPTS, required=True)
    args = parser.parse_args(argv)
    if args.command == 'preflight':
        name = 'cpu.json' if args.prompt == 'inplace_header' else f'cpu-{args.prompt}.json'
        return 0 if preflight(args.out or OUT / 'preflight' / name, args.prompt)['pass'] else 1
    if args.command == 'rules':
        print(json.dumps(rules_for(args.dir, args.prompt, args.key)))
        return 0
    if args.command == 'compare':
        print(compare(args.dir, args.before, args.after))
        return 0
    if args.command == 'run':
        return run(args.prompt, args.key, args.dir, args.limit, args.device, sampler_name=args.sampler,
                   batch=args.batch)
    if args.command == 'redraw':
        return run(args.prompt, args.key, args.dir, args.limit, args.device, redraw=True, sampler_name=args.sampler,
                   batch=args.batch)
    if args.command == 'teams':
        legal_teams(args.prompt, args.key, args.dir)
        return 0
    if args.command == 'gpu-preflight':
        return 0 if gpu_preflight(args.key, args.dir, sampler_name=args.sampler, prompt_code=args.prompt)['pass'] else 1
    print(report(args.dir, args.prompt))
    return 0


if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.exit(main())
