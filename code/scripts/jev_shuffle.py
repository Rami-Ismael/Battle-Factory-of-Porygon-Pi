"""Experiment 2 (ticket 10): Shuffle the list. Does the order of the options change Jev's pick?

Every move question of the main run's top-pick fills is replayed exactly: a ReplayChooser feeds
fill_mask the recorded answers, so each request is rebuilt byte for byte at no cost, and the rebuilt
team must equal the recorded one. The main run offered moves in alphabetical order. Each replayed
question then goes to Jev once more as one request with two questions on the same state:

- alphabetical: the main run's order again. Main-run pick against this pick is Jev's own noise.
- shuffled: the same options in a seeded random order (seed from the mask and the field).

Alphabetical against shuffled is the order effect, read against that noise. Where the shuffled pick
differs, it is swapped into the top-pick team (the rest unchanged) and that team is battled, so its win
rate pairs with the top-pick team's. One line per question goes to ~/vgc-data/jev/shuffle-<grid>.jsonl.

    /tmp/vgc-pilot/.venv/bin/python scripts/jev_shuffle.py verify --grid heldout      # fake choosers, no calls
    /tmp/vgc-pilot/.venv/bin/python scripts/jev_shuffle.py run --grid all --dry-run
    /tmp/vgc-pilot/.venv/bin/python scripts/jev_shuffle.py run --grid all --workers 6 --max-calls 6000
    /tmp/vgc-pilot/.venv/bin/python scripts/jev_shuffle.py collect --grid heldout     # then battle the grid
    /tmp/vgc-pilot/.venv/bin/python scripts/jev_shuffle.py report
"""
import argparse
import concurrent.futures as cf
import copy
import hashlib
import json
import random
import shutil
import statistics
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import jev_fields  # noqa: F401 -- registers the one-question field builders the replay rebuilds
import jev_fill
import jev_run
import jev_species  # noqa: F401 -- registers the species tournament

GRIDS = ('heldout', 'meta', 'random')
KIND = 'jev_shuffle'


class ReplayChooser:
    """Answers each question with the main run's recorded answer, in order, and keeps every move request."""

    def __init__(self, records: list[dict[str, Any]]) -> None:
        self.records, self.i = records, 0
        self.moves: list[dict[str, Any]] = []

    def choose(self, state: dict[str, Any], questions: dict[str, jev_fill.Question]) -> dict[str, jev_fill.Answer]:
        (name, question), = questions.items()
        rec = self.records[self.i]
        self.i += 1
        assert rec['path'] == name, f'replay out of step: asked {name}, recorded {rec["path"]}'
        if '.moves[' in name:
            self.moves.append({'path': name, 'state': state, 'question': question, 'main': rec['choice']})
        return {name: {'choice': rec['choice'], 'confidence': rec['confidence'], 'probabilities': rec['probabilities']}}


def results_path(grid: str) -> Path:
    return jev_run.DATA / f'shuffle-{grid}.jsonl'


def question_id(task_id: str, path: str) -> str:
    return f'{task_id}:{path}'


def shuffled(criteria: jev_fill.Criteria, key: str) -> jev_fill.Criteria:
    names = list(criteria)
    random.Random(int(hashlib.sha256(key.encode()).hexdigest()[:12], 16)).shuffle(names)
    return {n: criteria[n] for n in names}


def replay(grid: str, meta: list[str]) -> list[dict[str, Any]]:
    """Every move question of the grid's top-pick fills, rebuilt exactly; checks each rebuilt team."""
    tasks = {t['id']: t for t in jev_fill.tasks(grid)}
    out = []
    for line in jev_run.results_path(grid).read_text().splitlines():
        row = json.loads(line)
        if row['run'] != 'top' or row['status'] != 'legal':
            continue
        chooser = ReplayChooser(row['records'])
        team, _ = jev_fill.fill_mask(tasks[row['task_id']], chooser, meta, run='top', seed=row['seed'])
        assert team == row['team'], f"{row['task_id']}: the replay did not rebuild the recorded team"
        for m in chooser.moves:
            out.append({'id': question_id(row['task_id'], m['path']), 'grid': grid, 'task_id': row['task_id'],
                        'source': row['source'], 'team': row['team'], **m})
    return out


def request(q: dict[str, Any]) -> tuple[dict[str, Any], dict[str, jev_fill.Question]]:
    base = q['question']
    alpha = dict(sorted(base['criteria'].items()))
    assert list(alpha) == list(base['criteria']), f"{q['id']}: the main run's options were not alphabetical"
    mixed = shuffled(alpha, q['id'])
    def ask(criteria: jev_fill.Criteria) -> jev_fill.Question:
        return {'type': base['type'], 'instructions': base['instructions'], 'criteria': criteria}
    return q['state'], {'alphabetical': ask(alpha), 'shuffled': ask(mixed)}


def pending(grid: str, meta: list[str]) -> list[dict[str, Any]]:
    path = results_path(grid)
    done = {json.loads(line)['id'] for line in path.read_text().splitlines() if line.strip()} if path.exists() else set()
    return [q for q in replay(grid, meta) if q['id'] not in done]


def answer_row(q: dict[str, Any], questions: dict[str, jev_fill.Question],
               answers: dict[str, jev_fill.Answer]) -> dict[str, Any]:
    row = {'id': q['id'], 'grid': q['grid'], 'task_id': q['task_id'], 'source': q['source'], 'path': q['path'],
           'options': len(questions['alphabetical']['criteria']), 'main': q['main']}
    for name, question in questions.items():
        a = answers[name]
        order = list(question['criteria'])
        row[name] = {'choice': a['choice'], 'position': order.index(a['choice']), 'confidence': a['confidence'],
                     'probabilities': a['probabilities']}
    return row


def execute(grid: str, todo: list[dict[str, Any]], make: Callable[[], jev_fill.Chooser], workers: int = 6) -> dict[str, int]:
    lock, local, counts = threading.Lock(), threading.local(), {'asked': 0, 'transport': 0}
    path = results_path(grid)
    path.parent.mkdir(parents=True, exist_ok=True)

    def one(q: dict[str, Any]) -> None:
        if not hasattr(local, 'chooser'):
            local.chooser = make()
        state, questions = request(q)
        try:
            answers = local.chooser.choose(copy.deepcopy(state), questions)
        except jev_fill.TransportError:
            with lock:
                counts['transport'] += 1
            return
        row = answer_row(q, questions, answers)
        with lock, path.open('a') as f:
            f.write(json.dumps(row) + '\n')
            counts['asked'] += 1
            if counts['asked'] % 200 == 0:
                print(f"{grid}: {counts['asked']}/{len(todo)} asked", flush=True)

    with cf.ThreadPoolExecutor(workers) as pool:
        list(pool.map(one, todo))
    return counts


def rows(grid: str) -> list[dict[str, Any]]:
    path = results_path(grid)
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def swapped_team(team: jev_fill.Team, path: str, move: str) -> jev_fill.Team:
    out = copy.deepcopy(team)
    jev_fill.set_value(out, jev_fill.parse_path(path), move)
    return out


def collect(grid: str) -> dict[str, int]:
    """Append the top-pick team with the shuffled pick swapped in, for every question whose shuffled pick differs.

    Each swapped team passes the Showdown check first, as every other fill does; a team that fails is counted, not appended.
    """
    teams = {json.loads(line)['task_id']: json.loads(line)['team'] for line in jev_run.results_path(grid).read_text().splitlines()
             if json.loads(line)['run'] == 'top'}
    tasks = {t['id']: t for t in jev_fill.tasks(grid)}
    pipeline = jev_fill.load_pipeline()
    cpath = jev_fill.GRIDS[grid] / 'candidates.json'
    cands = json.loads(cpath.read_text())['candidates']
    have = {c['label'] for c in cands}
    new, rejected = [], 0
    for r in rows(grid):
        label = f"{KIND}:{r['path']}:{r['task_id']}"
        if r['shuffled']['choice'] == r['alphabetical']['choice'] or label in have:
            continue
        team = swapped_team(teams[r['task_id']], r['path'], r['shuffled']['choice'])
        try:
            jev_fill.check_legal(pipeline, tasks[r['task_id']], team)
        except ValueError as error:
            print(f"{label}: not appended, {error}", flush=True)
            rejected += 1
            continue
        new.append({'label': label, 'kind': KIND, 'task_id': r['task_id'], 'source': r['source'], 'team': team,
                    'paste': jev_fill.team_paste(team)})
        have.add(label)
    if new:
        shutil.copy(cpath, cpath.with_name(f"candidates.before-{KIND}-{time.strftime('%Y%m%d-%H%M%S')}.json"))
        cpath.write_text(json.dumps({'candidates': cands + new}, indent=1) + '\n')
    return {'questions': len(rows(grid)), 'appended': len(new), 'rejected': rejected, 'candidates': len(cands) + len(new)}


def share(xs: list[bool]) -> float:
    return round(sum(xs) / len(xs), 4) if xs else 0.0


def grid_report(grid: str) -> dict[str, Any]:
    rs = rows(grid)
    if not rs:
        return {'questions': 0}
    out: dict[str, Any] = {
        'questions': len(rs),
        'noise_change': share([r['main'] != r['alphabetical']['choice'] for r in rs]),
        'order_change': share([r['alphabetical']['choice'] != r['shuffled']['choice'] for r in rs]),
        'first_option': {k: share([r[k]['position'] == 0 for r in rs]) for k in ('alphabetical', 'shuffled')},
        'chance_first': round(statistics.mean(1 / r['options'] for r in rs), 4),
        'relative_position': {k: round(statistics.mean(r[k]['position'] / max(1, r['options'] - 1) for r in rs), 4)
                              for k in ('alphabetical', 'shuffled')}}
    panels_path = jev_fill.GRIDS[grid] / 'panels.json'
    if panels_path.exists():
        scores = {k: p.get('score') for k, p in json.loads(panels_path.read_text())['panels'].items()}
        diffs = [(r['source'], s - t) for r in rs
                 if (s := scores.get(f"{KIND}:{r['path']}:{r['task_id']}")) is not None
                 and (t := scores.get(f"jev:top:{r['task_id']}")) is not None]
        out['battled'] = len(diffs)
        out['shuffled_minus_top'] = round(statistics.mean(d for _, d in diffs), 4) if diffs else None
    return out


def report() -> dict[str, Any]:
    out = {g: grid_report(g) for g in GRIDS}
    (jev_run.DATA / 'shuffle-report.json').write_text(json.dumps(out, indent=1) + '\n')
    return out


class FirstOption:
    """A fake that always takes the first option: it must follow the shuffle almost every time."""

    def choose(self, state: dict[str, Any], questions: dict[str, jev_fill.Question]) -> dict[str, jev_fill.Answer]:  # noqa: ARG002 -- the Chooser signature
        return {n: {'choice': next(iter(q['criteria'])), 'confidence': 1.0, 'probabilities': {next(iter(q['criteria'])): 1.0}}
                for n, q in questions.items()}


class ByName:
    """A fake that takes the option whose name sorts last: the shuffle must never move it."""

    def choose(self, state: dict[str, Any], questions: dict[str, jev_fill.Question]) -> dict[str, jev_fill.Answer]:  # noqa: ARG002 -- the Chooser signature
        return {n: {'choice': max(q['criteria']), 'confidence': 1.0, 'probabilities': {max(q['criteria']): 1.0}}
                for n, q in questions.items()}


def verify(grid: str, meta: list[str], limit: int = 300) -> dict[str, Any]:
    """The ticket's Verify step on real replayed questions, with no call to Jev."""
    qs = replay(grid, meta)[:limit]
    checks: dict[str, Any] = {'questions': len(qs)}
    orders = set()
    for q in qs:
        _, questions = request(q)
        a, s = list(questions['alphabetical']['criteria']), list(questions['shuffled']['criteria'])
        assert sorted(a) == sorted(s) and len(set(s)) == len(s), f"{q['id']}: the shuffle dropped or repeated an option"
        assert list(request(q)[1]['shuffled']['criteria']) == s, f"{q['id']}: the shuffle is not reproducible"
        orders.add(tuple(s))
    checks['not_alphabetical'] = share([list(request(q)[1]['shuffled']['criteria']) != sorted(q['question']['criteria']) for q in qs])
    checks['distinct_orders'] = len(orders)
    for name, fake in (('first_option', FirstOption()), ('by_name', ByName())):
        changed = [fake.choose({}, request(q)[1])['alphabetical']['choice'] != fake.choose({}, request(q)[1])['shuffled']['choice'] for q in qs]
        checks[f'{name}_change'] = share(changed)
    checks['chance_same_first'] = round(statistics.mean(1 / len(q['question']['criteria']) for q in qs), 4)
    return checks


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    for name in ('verify', 'run', 'collect'):
        p = sub.add_parser(name)
        p.add_argument('--grid', choices=[*GRIDS, 'all'], default='heldout' if name != 'run' else 'all')
        if name == 'run':
            p.add_argument('--dry-run', action='store_true')
            p.add_argument('--max-calls', type=int, default=0)
            p.add_argument('--workers', type=int, default=6)
    sub.add_parser('report')
    a = ap.parse_args()
    if a.cmd == 'report':
        print(json.dumps(report(), indent=1))
        return
    grids = GRIDS if a.grid == 'all' else (a.grid,)
    if a.cmd == 'collect':
        for g in grids:
            print(g, collect(g))
        return
    meta = jev_fill.meta_pastes()
    if a.cmd == 'verify':
        for g in grids:
            print(g, json.dumps(verify(g, meta)))
        return
    run_grids(grids, meta, a.dry_run, a.max_calls, a.workers)


def run_grids(grids: tuple[str, ...], meta: list[str], dry_run: bool, max_calls: int, workers: int) -> None:
    """Price the pending questions, then (unless a dry run) ask Jev and report what it cost."""
    todo = {g: pending(g, meta) for g in grids}
    calls = sum(map(len, todo.values()))
    tokens = sum(jev_fill.estimate_tokens(list(request(q))) for ts in todo.values() for q in ts)
    print(f"{', '.join(f'{g} {len(t)}' for g, t in todo.items())} questions, {calls} requests, "
          f"~{tokens:,} input tokens, ~${tokens * 0.042 / 1e6:.2f} at $0.042/M (estimate)", flush=True)
    if dry_run or not calls:
        return
    if max_calls and calls > max_calls:
        raise SystemExit(f'{calls} requests is over --max-calls {max_calls}')
    import jev_client
    made: list[jev_client.JevChooser] = []

    def make() -> jev_client.JevChooser:
        c = jev_client.JevChooser(jev_run.DATA / 'shuffle-calls.jsonl')
        made.append(c)
        return c
    for g, ts in todo.items():
        print(g, execute(g, ts, make, workers), flush=True)
    print(f"{sum(c.usage['calls'] for c in made)} requests, ${sum(c.dollars() for c in made):.4f}", flush=True)
    for c in made:
        c.close()


if __name__ == '__main__':
    main()
