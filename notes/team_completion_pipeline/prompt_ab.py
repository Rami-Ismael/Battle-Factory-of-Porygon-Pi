#!/usr/bin/env python3
"""Legal-output rate of LLM completions under different prompts. No battles, one attempt each.

Every (prompt, masked task) pair is sent once. A completion is legal only if it is valid
JSON, keeps every unmasked field, and passes the Showdown team validator for the format.
Results append to a JSONL file; rerunning skips finished pairs and retries transport errors.
"""
import argparse
import concurrent.futures as cf
import json
import os
import threading
import time
from pathlib import Path

import pipeline as P


def load_config(path):
    path = Path(path).resolve()
    config = P.read_json(path)
    for field in ('starting_teams', 'meta_teams', 'showdown_runtime'):
        config[field] = str((path.parent / config[field]).resolve())
    return config


def build_tasks(config, starts, seed):
    tasks = []
    for row in starts:
        for task, k, rep in P.task_specs(config):
            s = int(P.digest([seed, row['id'], task, k, rep])[:12], 16)
            masked, slots = P.mask_team(row['team'], task, k, s)
            tasks.append({'id': P.digest([row['id'], task, k, rep, s])[:16], 'source': row['id'],
                          'task': task, 'k': k, 'seed': s, 'slots': slots, 'masked_team': masked,
                          'masked_paths': P.masked_paths(task, slots), 'original': row['team']})
    return tasks


def attempt(config, method, meta, task):
    """Returns a record whose 'stage' says where a failed completion failed."""
    record = {'arm': method['arm'], 'prompt': method['prompt'], 'max_tokens': method['max_tokens'],
              'task_id': task['id'], 'source': task['source'],
              'task': task['task'], 'k': task['k']}
    request = {'format': config['format'], 'spread_system': config['spread_system'],
               'task': {key: task[key] for key in ('task', 'k', 'seed', 'slots')},
               'masked_team': task['masked_team'], 'meta_teams': meta, 'feedback': None}
    request['task']['masked_paths'] = P.masked_paths(task['task'], task['slots'])
    started = time.monotonic()
    try:
        response, metadata = P.generate(method, request, [], task['seed'], config['timeout_seconds'])
        record['usage'] = metadata.get('usage')
    except (RuntimeError, OSError) as error:
        record.update(status='transport', stage='transport', error=str(error))
        return record
    except (ValueError, KeyError, TypeError, IndexError) as error:
        record.update(status='illegal', stage='json', error=f'{type(error).__name__}: {error}',
                      usage=getattr(error, 'usage', None))
        return record
    finally:
        record['seconds'] = round(time.monotonic() - started, 1)
    try:
        team = response['completed_team']
        P.check_structure(team)
    except (ValueError, KeyError, TypeError) as error:
        record.update(status='illegal', stage='structure', error=str(error))
        return record
    try:
        P.check_completion(task['masked_team'], team, task['task'], task['slots'])
    except ValueError as error:
        record.update(status='illegal', stage='preservation', error=str(error))
        return record
    try:
        P.validate_team(config, team)
    except ValueError as error:
        record.update(status='illegal', stage='validator', error=str(error))
        return record
    except (RuntimeError, OSError) as error:
        record.update(status='transport', stage='validator-bridge', error=str(error))
        return record
    record.update(status='legal', stage=None, error=None, team=team)
    return record


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='benchmark.ling.json')
    ap.add_argument('--arms', default='baseline:4096,baseline:16384,regmb:16384',
                    help='comma list of prompt:max_tokens')
    ap.add_argument('--teams', type=int, default=1, help='number of starting teams (54 masked tasks each)')
    ap.add_argument('--skip-teams', type=int, default=0, help='skip this many starting teams first')
    ap.add_argument('--reps', type=int, default=1, help='masks per task and k (config default is for the full grid)')
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--workers', type=int, default=12)
    ap.add_argument('--max-calls', type=int, default=700)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    if not os.environ.get('OPENROUTER_API_KEY'):
        raise SystemExit('Set OPENROUTER_API_KEY first')
    config = load_config(a.config)
    config['mask_repetitions'] = a.reps
    base = config['methods'][0]
    arms = {}
    for spec in a.arms.split(','):
        name, tokens = spec.split(':')
        if name not in P.PROMPTS:
            raise SystemExit(f'Unknown prompt {name}; known: {sorted(P.PROMPTS)}')
        arms[f'{name}@{tokens}'] = dict(base, prompt=name, max_tokens=int(tokens), arm=f'{name}@{tokens}')
    meta_rows = P.load_teams(config['meta_teams'])
    meta = [row['team'] for row in meta_rows]
    starts = P.load_teams(config['starting_teams'])[a.skip_teams:a.skip_teams + a.teams]
    tasks = build_tasks(config, starts, a.seed)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    P.save_json(out.with_suffix('.tasks.json'), {'format': config['format'], 'seed': a.seed, 'tasks': tasks,
                                                 'starting_teams': starts})
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            r = json.loads(line)
            if r['status'] != 'transport':
                done.add((r.get('arm', r['prompt']), r['task_id']))
    todo = [(p, t) for t in tasks for p in arms if (p, t['id']) not in done]
    print(f'{len(tasks)} tasks x {len(arms)} arms; {len(done)} done; {len(todo)} to send', flush=True)
    if len(todo) > a.max_calls:
        raise SystemExit(f'{len(todo)} calls exceeds --max-calls {a.max_calls}')

    lock = threading.Lock()
    counts = {}

    def work(item):
        name, task = item
        record = attempt(config, arms[name], meta, task)
        with lock:
            with out.open('a') as handle:
                handle.write(json.dumps(record, sort_keys=True) + '\n')
            counts[name] = counts.get(name, 0) + 1
            if sum(counts.values()) % 25 == 0:
                print('sent', dict(counts), flush=True)
        return record

    with cf.ThreadPoolExecutor(a.workers) as pool:
        list(pool.map(work, todo))
    print('finished', dict(counts), flush=True)


if __name__ == '__main__':
    main()
