"""Parallelize independent exact battle panels, optionally as generation arrives."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import fcntl
import json
import multiprocessing
import os
from pathlib import Path
import time
from types import SimpleNamespace

import llm_baseline as B
import reasoning_experiment as E


def battle_design(m):
    return {k: m.get(k) for k in ('format', 'seed', 'battles', 'opponents',
                                 'seed_limit', 'exclude_starter_team_ids')}


def plan(runs, canonical):
    """One group per canonical team AND schedule across every arm/repeat."""
    groups = {}
    design = battle_design(runs[0][2])
    for entry, folder, m in runs:
        if battle_design(m) != design:
            raise ValueError('battle design differs across runs')
        for trial in m['trials']:
            row = E.completion(folder, entry, trial)
            if row is None or not row['valid']:
                continue
            key = B.digest(dict(team=canonical(row['team']),
                                schedule=B.schedule_for(m, trial['start'])))
            group = groups.setdefault(key, dict(key=key, team=row['team'], start=trial['start'], destinations=[]))
            group['destinations'].append(str(folder / 'labels' / ('llm-' + trial['id'] + '.json')))
    return [g for g in groups.values() if any(not Path(p).exists() for p in g['destinations'])]


def publish(group, result, config):
    """Serialize config publication and never replace an existing result."""
    if result['battles'] != 50 or not 0 <= result['wins'] <= 50:
        raise ValueError('incomplete or invalid panel')
    for destination in group['destinations']:
        destination = Path(destination)
        folder = destination.parent.parent
        with (folder / '.battle-publish.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            config_path = folder / 'battle-config.json'
            old_config = E.read(config_path)
            if old_config is not None and old_config != config:
                raise ValueError(f'battle configuration changed: {folder}')
            old = E.read(destination)
            if old is not None and old != result:
                raise ValueError(f'refusing to replace saved battle result: {destination}')
            if old_config is None:
                B.save(config_path, config)
            if old is None:
                B.save(destination, result)


def worker(output, index, manifest, groups):
    folder = Path(output) / 'battle-workers' / str(index)
    B.save(folder / 'manifest.json', manifest)
    by_label = {f"p{g['key']}-s{g['start']}": g for g in groups}
    # Separate log per worker; publishing happens after each complete panel.
    with (folder / 'worker.log').open('a', buffering=1) as log:
        os.dup2(log.fileno(), 1)
        os.dup2(log.fileno(), 2)
        def finished(label, result):
            publish(by_label[label], result, E.read(folder / 'battle-config.json'))
        B.battle(SimpleNamespace(output=folder),
                 candidates_override=[(label, g['team']) for label, g in by_label.items()],
                 on_result=finished)
    return len(groups)


def score(output, workers):
    _, runs = E.load_runs(output)
    h, db = B.runtime()
    groups = plan(runs, lambda team: db.canonical_paste(B.paste(team, h)))
    # Register teams and paste files before spawning: identical teams with
    # different schedules may be assigned to different workers.
    con = db.connect()
    validator = h.Validator()
    try:
        for group in groups:
            tid = db.add_team(con, B.paste(group['team'], h), 'reasoning-parallel:' + output.name, validator)[0]
            db.paste_file(con, tid)
        for opponent in runs[0][2]['opponents']:
            db.paste_file(con, opponent['team_id'])
    finally:
        validator.close()
        con.close()
    buckets = [[] for _ in range(workers)]
    for group in groups:
        buckets[int(group['key'], 16) % workers].append(group)
    # Spawn avoids sharing SQLite connections, Torch state or simulator sockets.
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        jobs = [pool.submit(worker, str(output), i, runs[0][2], bucket)
                for i, bucket in enumerate(buckets) if bucket]
        for job in jobs:
            job.result()
    return len(groups)


def progress(output):
    study, runs = E.load_runs(output)
    returned = legal = scored = 0
    for entry, folder, m in runs:
        for trial in m['trials']:
            row = E.completion(folder, entry, trial)
            if row is not None:
                returned += 1
                if row['valid']:
                    legal += 1
                    scored += (folder / 'labels' / ('llm-' + trial['id'] + '.json')).exists()
    return dict(returned=returned, planned=study['planned_calls'], legal=legal, scored=scored,
                complete=returned == study['planned_calls'] and scored == legal)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--watch', action='store_true', help='score new legal responses as generation continues')
    args = parser.parse_args()
    if not 1 <= args.workers <= 32:
        parser.error('workers must be between 1 and 32')
    args.output = args.output.resolve()
    # Separate from the generation lock; only one scorer can publish at a time.
    with (args.output / '.parallel-battles.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            while True:
                state = dict(progress(args.output), workers=args.workers, stage='scoring', updated_at=time.time())
                B.save(args.output / 'parallel-battles.json', state)
                panels = score(args.output, args.workers)
                state = dict(progress(args.output), workers=args.workers, updated_at=time.time())
                state['stage'] = 'complete' if state['complete'] else 'waiting_for_generation'
                B.save(args.output / 'parallel-battles.json', state)
                print(json.dumps(dict(state, new_panels=panels)), flush=True)
                if state['complete'] or not args.watch:
                    # An older generation process may still write its end-of-run
                    # report. Wait for it before publishing the final report.
                    with (args.output.parent / ('.' + args.output.name + '.reasoning.lock')).open('a') as generation_lock:
                        fcntl.flock(generation_lock, fcntl.LOCK_EX)
                        E.report(SimpleNamespace(output=args.output))
                    break
                time.sleep(30)
        except Exception as exc:
            B.save(args.output / 'parallel-battles.json', dict(stage='failed', error=f'{type(exc).__name__}: {exc}', updated_at=time.time()))
            raise


if __name__ == '__main__':
    main()
