#!/usr/bin/env python3
"""Mask Lab battles: every candidate against the 49 distinct top-50 meta teams, behaviour-cloning policy on both sides.

Run with /tmp/vgc-pilot/.venv/bin/python. It never uses random play: the policy is bc_100.zip (hash checked),
the same policy 1 as the shared matchup matrix. It starts its own Showdown servers and works on a private copy of
the matrix (~/vgc-data/mask_lab_matrix.sqlite), so the shared database and servers are not touched.
Writes runs/prompt-ab/panels.json.

    python mask_lab_battle.py --limit 6      # six persistent workers, 24 battles each
    python mask_lab_battle.py                # all candidates, same optimized pool
"""
import argparse
from contextlib import ExitStack
import json
import signal
import socket
import sqlite3
import subprocess
import sys
import time
import uuid
from pathlib import Path

REPO = Path('/Users/ramiismael/Documents/code/vgc-team-generator-pilot')
RUNTIME = Path.home() / '.local/share/vgc-pilot-runtime'
VALIDATOR = RUNTIME / 'validator-913da36'
BENCH = RUNTIME / 'unpacked/vgc-bench-d79f9532947ac114dce1dda2456a590afcd375b2'
HERE = Path(__file__).resolve().parent
RUN = HERE / 'runs/prompt-ab'
PRIVATE_DB = Path.home() / 'vgc-data/mask_lab_matrix.sqlite'
POLICY, BC_HASH, SEED = 1, '57f5edcab415cf6c', 20261008

sys.path.insert(0, '/tmp/vgc-pilot/src')
sys.path.insert(0, str(REPO / 'src'))
import hps_generate  # noqa: E402
import pool  # noqa: E402
hps_generate.SHOWDOWN = VALIDATOR
import matchup_db as MDB  # noqa: E402
MDB.SHOWDOWN = VALIDATOR
pool.REPO = str(BENCH)
from fast_battle_runner import FastBattleRunner


def private_copy(destination=PRIVATE_DB):
    if destination.resolve() == Path(MDB.DEFAULT_DB).resolve():
        raise ValueError('Use a private database, not the shared matchup matrix')
    if destination.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect('file:'+str(MDB.DEFAULT_DB)+'?mode=ro', uri=True, timeout=60) as src:
        dst = sqlite3.connect(destination)
        try:
            src.backup(dst)
        finally:
            dst.close()


def start_servers(n):
    procs, ports = [], []
    for _ in range(n):
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            port = sock.getsockname()[1]
        log = open(f'/tmp/mask_lab_showdown_{port}.log', 'w')
        procs.append(subprocess.Popen(['node', 'pokemon-showdown', 'start', str(port), '--no-security'],
                                      cwd=VALIDATOR, stdout=log, stderr=subprocess.STDOUT))
        ports.append(port)
    for port, proc in zip(ports, procs):
        for _ in range(300):
            try:
                with socket.create_connection(('127.0.0.1', port), timeout=.2):
                    break
            except OSError:
                if proc.poll() is not None:
                    raise RuntimeError(f'Showdown on {port} exited; see /tmp/mask_lab_showdown_{port}.log')
                time.sleep(.1)
        else:
            raise RuntimeError(f'Showdown on {port} did not start')
    return procs, ports


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', '--servers', dest='workers', type=int, default=6)
    ap.add_argument('--concurrency', type=int, default=24)
    ap.add_argument('--backend', choices=['fast', 'legacy'], default='fast')
    ap.add_argument('--batch', type=int, default=24, help='teams battled per saved batch')
    ap.add_argument('--per-cell', type=int, default=1)
    ap.add_argument('--limit', type=int, help='only the first N candidates (smoke test)')
    ap.add_argument('--database', type=Path, default=PRIVATE_DB)
    ap.add_argument('--run-dir', type=Path, default=RUN, help='candidate input and panels output directory')
    ap.add_argument('--evidence-dir', type=Path, help='fresh evaluator provenance/results directory')
    a = ap.parse_args(argv)
    if min(a.workers, a.concurrency, a.batch, a.per_cell, a.limit or 1) < 1 or a.limit == 0:
        ap.error('Counts must be positive')
    return a


def stop_servers(procs):
    for proc in procs:
        proc.terminate()
    for proc in procs:
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)


def evaluate(con, rows, cols, a):
    """Keep the runner alive across database deficit batches; return cache identity."""
    evidence = a.evidence_dir or HERE/'runs/battle-evaluations'/(
        time.strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:8])
    with ExitStack() as stack:
        if a.backend == 'fast':
            runner = stack.enter_context(FastBattleRunner(evidence, a.workers, a.concurrency))
            policy_id = runner.policy_id(con, MDB)
        else:
            policy_id = POLICY
            runner = None
            if MDB.plan(con, policy_id, rows, cols, a.per_cell):
                procs, ports = start_servers(a.workers)
                stack.callback(stop_servers, procs)
                original_ports = pool.live_ports
                stack.callback(setattr, pool, 'live_ports', original_ports)
                pool.live_ports = lambda *args, **kwargs: ports
                runner = lambda con, jobs, seed=None: MDB.run_jobs(con, jobs, seed=seed, conc=a.concurrency)
        started = time.time()
        for i in range(0, len(rows), a.batch):
            t0 = time.time()
            n = MDB.ensure(con, policy_id, rows[i:i + a.batch], cols, a.per_cell,
                           seed=SEED, runner=runner, note='maskLab:'+a.backend)
            print(f'batch {i // a.batch + 1}/{-(-len(rows) // a.batch)}: {n} battles in {time.time() - t0:.1f}s '
                  f'({time.time() - started:.1f}s total)', flush=True)
    return policy_id, str(evidence) if a.backend == 'fast' else None


def main(argv=None):
    a = parse_args(argv)

    if MDB.sha256_file(MDB.POLICY_FILE) != BC_HASH:
        raise SystemExit('bc_100.zip differs from the verified behaviour-cloning checkpoint; refusing to battle')
    private_copy(a.database)
    con = MDB.connect(a.database)
    cands = json.loads((a.run_dir / 'candidates.json').read_text())['candidates']
    cands = [c for c in cands if c['paste']]
    if a.limit:
        cands = cands[:a.limit]
    validator = hps_generate.Validator()
    try:
        with con:
            ids = {c['label']: MDB.add_team(con, c['paste'], 'llm-completion:maskLab', validator)[0] for c in cands}
    finally:
        validator.close()
    marks = ','.join('?' * len(ids))
    legal = {t for (t,) in con.execute(f'SELECT team_id FROM team WHERE legal=1 AND team_id IN ({marks})', list(ids.values()))}
    cols = MDB.set_members(con, 'top50')
    rows = sorted(legal)
    print(f'{len(cands)} candidates -> {len(rows)} distinct legal teams; {len(cols)} meta opponents; '
          f'planned battles <= {len(rows) * len(cols)}', flush=True)

    previous = signal.getsignal(signal.SIGTERM)
    def interrupted(*_):
        raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM, interrupted)
    try:
        policy_id, evidence = evaluate(con, rows, cols, a)
    finally:
        signal.signal(signal.SIGTERM, previous)

    out = {'policy_id': policy_id, 'policy': 'bc_100.zip:' + BC_HASH + ' on both sides', 'seed': SEED,
           'evaluator': {'backend':a.backend, 'workers':a.workers, 'concurrency':a.concurrency, 'evidence':evidence},
           'opponents': [], 'panels': {}}
    for t in cols:
        paste = con.execute('SELECT paste FROM team WHERE team_id=?', (t,)).fetchone()[0]
        weight = con.execute("SELECT weight FROM team_set WHERE set_name='top50' AND team_id=?", (t,)).fetchone()[0]
        out['opponents'].append({'team_id': t, 'weight': weight,
                                 'species': [s['species'] for s in MDB.canonical_slots(paste)]})
    for c in cands:
        t = ids[c['label']]
        if t not in legal:
            out['panels'][c['label']] = {'team_id': t, 'score': None, 'illegal': True}
            continue
        s = MDB.score(con, policy_id, t)
        cells = MDB.row_cells(con, policy_id, t, cols)
        out['panels'][c['label']] = {'team_id': t, 'score': s['score'], 'coverage': s['coverage'],
                                     'battles': s['battles'], 'se': s['se'],
                                     'by_opponent': {str(o): list(v) for o, v in cells.items()}}
    (a.run_dir / 'panels.json').write_text(json.dumps(out, indent=1) + '\n')
    scored = sum(1 for p in out['panels'].values() if p.get('score') is not None)
    print(f'saved {scored} scored panels to {a.run_dir / "panels.json"}', flush=True)
    con.close()


if __name__ == '__main__':
    main()
