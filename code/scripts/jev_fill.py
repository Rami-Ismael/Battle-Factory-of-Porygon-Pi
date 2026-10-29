"""Fill a Mask Lab mask with a decision model, one Choice question per hidden field.

A mask hides fields of a starting team. Each hidden field becomes one Choice question whose
options are the legal values for that field (legal_options). Questions are asked one at a time,
and each answer is written into the team before the next question, so every question sees the
earlier answers. A whole hidden Pokémon is asked in SLOT_ORDER: species, ability, item, four
moves, Stat Alignment, Stat Points.

The decision model sits behind the Chooser interface. JevChooser (later) calls TypeSafe's SDK;
FakeChooser answers with uniform probabilities and records every request, for tests and dry runs.

A run is 'top' (take the model's top option at every question) or 'sample' (draw each answer from
the model's probabilities). Question builders live in jev_fields (one hidden field, one question)
and jev_species (the species tournament); each registers itself in FIELDS.
"""
import copy
import json
import random
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType
from typing import Any, NotRequired, Protocol, TypedDict

PIPELINE = Path.home() / ('Documents/yakumsi-vault/Personal Project/Using Advance method search in the whole search of '
                          'possible vgc pokemon format to find the right counter to a pokemon team/team_completion_pipeline')
GRIDS = {'random': PIPELINE / 'runs/prompt-ab', 'heldout': PIPELINE / 'runs/heldout-starters',
         'meta': PIPELINE / 'runs/meta-starters'}
FORMAT_NAME = 'Champions VGC 2026 Reg Set M-B'
RULES = 'doubles, bring 6 pick 4, level 50, Species Clause, Item Clause'
SLOT_ORDER = ('species', 'ability', 'item', 'moves', 'nature', 'statPoints')
STATS = ('HP', 'Atk', 'Def', 'SpA', 'SpD', 'Spe')
TOKEN_LIMIT = 32_000            # Jev: state plus one question
OPTION_LIMIT = 255              # Jev: options in one Choice question
CHARS_PER_TOKEN = 2.3           # measured 2.54 on the first live call (2026-10-27, 62,548 chars, 24,644 tokens); 10% margin

Slot = dict[str, Any]           # species, item, ability, moves, nature, statPoints, ivs, level, slot
Team = list[Slot | None]        # six slots; None is a whole hidden Pokémon
FieldPath = list[Any]           # [i], [i, 'item'] or [i, 'moves', j]
Criteria = dict[str, Any]       # option name -> its description (a JSON-able value)
Task = dict[str, Any]           # one row of a grid's eval2.tasks.json


class Question(TypedDict):
    type: str
    instructions: str
    criteria: Criteria


class Answer(TypedDict):
    choice: str
    confidence: float
    probabilities: dict[str, float]
    score: NotRequired[float]       # a Score question's expected level


class Chooser(Protocol):
    def choose(self, state: dict[str, Any], questions: dict[str, Question]) -> dict[str, Answer]:
        """One answer per question name."""
        ...


class FakeChooser:
    """Uniform probabilities over every option, a seeded pick, and a log of every request."""

    def __init__(self, seed: int = 0) -> None:
        self.rng = random.Random(seed)
        self.requests: list[dict[str, Any]] = []

    def choose(self, state: dict[str, Any], questions: dict[str, Question]) -> dict[str, Answer]:
        self.requests.append({'state': state, 'questions': questions})
        answers: dict[str, Answer] = {}
        for name, question in questions.items():
            names = list(question['criteria'])
            p = 1 / len(names)
            answers[name] = {'choice': self.rng.choice(names), 'confidence': p,
                             'probabilities': dict.fromkeys(names, p)}
        return answers


class QuestionTooLarge(ValueError):
    pass


class TransportError(RuntimeError):
    """The decision model could not be reached or refused the request; the pair is worth retrying."""


@dataclass
class Fill:
    """One mask being filled in one run."""
    task: Task
    team: Team
    meta: list[str]
    run: str = 'top'
    rng: random.Random = field(default_factory=random.Random)
    records: list[dict[str, Any]] = field(default_factory=list)


Builder = Callable[[Chooser, Fill, FieldPath], Any]
FIELDS: dict[str, Builder] = {}


def register(name: str) -> Callable[[Builder], Builder]:
    """Decorator: the function that fills one kind of hidden field."""
    def add(function: Builder) -> Builder:
        FIELDS[name] = function
        return function
    return add


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


def expand(path: FieldPath, team: Team) -> list[FieldPath]:
    """The field paths a masked path asks, in order. A whole hidden Pokémon becomes a blank slot."""
    if len(path) > 1:
        return [path]
    i = path[0]
    team[i] = blank_slot(i + 1)
    paths: list[FieldPath] = []
    for name in SLOT_ORDER:
        paths += [[i, 'moves', j] for j in range(4)] if name == 'moves' else [[i, name]]
    return paths


def slot_at(team: Team, path: FieldPath) -> Slot:
    """The slot a field path points into; a whole hidden Pokémon has been expanded by then."""
    slot = team[path[0]]
    assert slot is not None, f'{path_text(path)} is in a hidden Pokémon that was never expanded'
    return slot


def set_value(team: Team, path: FieldPath, value: Any) -> None:
    slot = slot_at(team, path)
    if len(path) == 3:
        slot[path[1]][path[2]] = value
    else:
        slot[path[1]] = value


def state_for(fill: Fill, path: FieldPath) -> dict[str, Any]:
    return {'format': FORMAT_NAME, 'rules': RULES, 'meta': fill.meta, 'team': fill.team,
            'asking_about': path_text(path)}


def estimate_tokens(obj: Any) -> int:
    return int(len(json.dumps(obj, ensure_ascii=False)) / CHARS_PER_TOKEN)


def request_for(fill: Fill, path: FieldPath, instructions: str,
                criteria: Criteria) -> tuple[dict[str, Any], dict[str, Question], int]:
    """The (state, questions) pair sent for one hidden field, checked against Jev's limits."""
    if not criteria:
        raise ValueError(f'{path_text(path)}: no legal option to offer')
    if len(criteria) > OPTION_LIMIT:
        raise QuestionTooLarge(f'{path_text(path)}: {len(criteria)} options, over {OPTION_LIMIT}')
    question: Question = {'type': 'choice', 'instructions': instructions, 'criteria': criteria}
    state, questions = state_for(fill, path), {path_text(path): question}
    tokens = estimate_tokens({'state': state, 'questions': questions})
    if tokens > TOKEN_LIMIT:
        raise QuestionTooLarge(f'{path_text(path)}: about {tokens} tokens, over {TOKEN_LIMIT}')
    return state, questions, tokens


def draw(probabilities: dict[str, float], rng: random.Random) -> str:
    names = list(probabilities)
    return rng.choices(names, weights=[probabilities[n] for n in names])[0]


def ask_choice(chooser: Chooser, fill: Fill, path: FieldPath, instructions: str, criteria: Criteria) -> str:
    """Ask one Choice question, record the answer, and return the value this run takes."""
    state, questions, tokens = request_for(fill, path, instructions, criteria)
    answer = chooser.choose(copy.deepcopy(state), questions)[path_text(path)]
    if answer['choice'] not in criteria or not set(answer['probabilities']) <= set(criteria):
        raise ValueError(f'{path_text(path)}: the answer names an option that was not offered')
    value = answer['choice'] if fill.run == 'top' else draw(answer['probabilities'], fill.rng)
    fill.records.append({'path': path_text(path), 'options': len(criteria), 'tokens': tokens,
                         'choice': answer['choice'], 'confidence': answer['confidence'],
                         'probabilities': answer['probabilities'], 'value': value})
    return value


def fill_mask(task: Task, chooser: Chooser, meta: list[str], run: str = 'top',
              seed: int = 0) -> tuple[Team, list[dict[str, Any]]]:
    """(filled team, records) for one mask. Fields with no registered builder raise NotImplementedError."""
    fill = Fill(task=task, team=copy.deepcopy(task['masked_team']), meta=meta, run=run, rng=random.Random(seed))
    for text in task['masked_paths']:
        for path in expand(parse_path(text), fill.team):
            if path[1] not in FIELDS:
                raise NotImplementedError(f'no question builder for {path[1]} yet')
            set_value(fill.team, path, FIELDS[path[1]](chooser, fill, path))
    return fill.team, fill.records


def load_pipeline() -> ModuleType:
    """The vault pipeline module (Showdown validation, meta teams), loaded from its file."""
    spec = spec_from_file_location('mask_lab_pipeline', PIPELINE / 'pipeline.py')
    assert spec is not None and spec.loader is not None, f'cannot load {PIPELINE / "pipeline.py"}'
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ling_config() -> dict[str, Any]:
    return json.loads((PIPELINE / 'benchmark.ling.json').read_text())


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


def meta_pastes(pipeline: ModuleType | None = None) -> list[str]:
    """The top-50 meta teams Ling's prompt gets, as paste text (same teams, compact form)."""
    pipeline = pipeline or load_pipeline()
    rows = pipeline.load_teams(str(PIPELINE / ling_config()['meta_teams']))
    return ['\n\n'.join(paste(mon) for mon in row['team']) for row in rows]


def tasks(grid: str) -> list[Task]:
    return json.loads((GRIDS[grid] / 'eval2.tasks.json').read_text())['tasks']


def check_legal(pipeline: ModuleType, task: Task, team: Team) -> None:
    """Raise ValueError unless the team is legal and only the masked fields changed."""
    pipeline.check_completion(task['masked_team'], team, task['task'], task['slots'])
    pipeline.validate_team(ling_config(), team)
