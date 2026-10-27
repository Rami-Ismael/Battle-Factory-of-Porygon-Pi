"""Run Jev over a Mask Lab grid: four runs per mask, resumable, priced before any spend.

Every mask is filled four times: 'top' takes Jev's top option at every question, and 'sample1',
'sample2', 'sample3' draw each answer from Jev's probabilities. A run's seed comes from the mask id
and the run name, so a rerun draws the same samples. Because later questions see earlier answers,
each run asks its own questions.

One line per (mask, run) goes to ~/vgc-data/jev/<grid>.jsonl: the filled team, its legality, and
every question's choice, confidence and probabilities. A rerun skips pairs that finished and retries
pairs that failed on transport. The per-call log (JevChooser) is ~/vgc-data/jev/<grid>-calls.jsonl.

estimate() prices a run before it starts by filling every pending pair with FakeChooser, which
asks exactly the questions the real run would ask for the top run, and nearly so for sampled runs
(a sampled species changes the later questions' option lists, not their number).
"""
import concurrent.futures as cf
import hashlib
import json
import threading
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import jev_fill

RUNS = ('top', 'sample1', 'sample2', 'sample3')
DATA = Path.home() / 'vgc-data/jev'
FINISHED = ('legal', 'illegal', 'too large', 'no option')      # a pair with one of these is never sent again


def run_seed(task: jev_fill.Task, run: str) -> int:
    return int(hashlib.sha256(f"{task['id']}:{run}".encode()).hexdigest()[:12], 16)


def mode(run: str) -> str:
    return 'top' if run == 'top' else 'sample'


def results_path(grid: str) -> Path:
    return DATA / f'{grid}.jsonl'


def finished(grid: str) -> set[tuple[str, str]]:
    path = results_path(grid)
    if not path.exists():
        return set()
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return {(r['task_id'], r['run']) for r in rows if r['status'] in FINISHED}


def pending(grid: str, limit: int = 0) -> list[tuple[jev_fill.Task, str]]:
    """(mask, run) pairs not finished yet, masks in grid order; limit counts masks, not pairs."""
    done, pairs = finished(grid), []
    tasks = jev_fill.tasks(grid)
    for task in tasks[:limit] if limit else tasks:
        pairs += [(task, run) for run in RUNS if (task['id'], run) not in done]
    return pairs


def estimate(pairs: list[tuple[jev_fill.Task, str]], meta: list[str]) -> dict[str, int]:
    """Questions and estimated input tokens the pairs would send, from a FakeChooser dry fill."""
    questions = tokens = 0
    for task, run in pairs:
        _, records = jev_fill.fill_mask(task, jev_fill.FakeChooser(run_seed(task, run)), meta, mode(run),
                                        run_seed(task, run))
        questions += len(records)
        tokens += sum(r['tokens'] for r in records)
    return {'pairs': len(pairs), 'questions': questions, 'tokens': tokens}


def attempt(task: jev_fill.Task, run: str, chooser: jev_fill.Chooser, meta: list[str],
            pipeline: ModuleType) -> dict[str, Any]:
    """Fill one (mask, run) and check it; transport failures are recorded, not raised."""
    row: dict[str, Any] = {'task_id': task['id'], 'task': task['task'], 'k': task['k'], 'source': task['source'],
                           'run': run, 'seed': run_seed(task, run)}
    try:
        team, records = jev_fill.fill_mask(task, chooser, meta, mode(run), run_seed(task, run))
        row.update(team=team, records=records)
        jev_fill.check_legal(pipeline, task, team)
        row['status'] = 'legal'
    except jev_fill.QuestionTooLarge as error:
        row.update(status='too large', error=str(error)[:300])
    except ValueError as error:
        row.update(status='no option' if 'no legal option' in str(error) else 'illegal', error=str(error)[:300])
    except jev_fill.TransportError as error:   # logged now, retried on the next run
        row.update(status='transport', error=f'{type(error).__name__}: {str(error)[:300]}')
    return row


def execute(grid: str, pairs: list[tuple[jev_fill.Task, str]], make_chooser: Callable[[], jev_fill.Chooser],
            meta: list[str], pipeline: ModuleType, workers: int) -> dict[str, int]:
    """Send every pair, append each finished row to the grid's results file, and count statuses."""
    lock, local = threading.Lock(), threading.local()
    counts: dict[str, int] = {}
    results_path(grid).parent.mkdir(parents=True, exist_ok=True)

    def work(pair: tuple[jev_fill.Task, str]) -> None:
        if not hasattr(local, 'chooser'):
            local.chooser = make_chooser()
        row = attempt(pair[0], pair[1], local.chooser, meta, pipeline)
        with lock:
            with results_path(grid).open('a') as f:
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
            counts[row['status']] = counts.get(row['status'], 0) + 1
            if sum(counts.values()) % 25 == 0:
                print(f'{grid}: {sum(counts.values())}/{len(pairs)} {counts}', flush=True)

    with cf.ThreadPoolExecutor(workers) as pool:
        list(pool.map(work, pairs))
    return counts
