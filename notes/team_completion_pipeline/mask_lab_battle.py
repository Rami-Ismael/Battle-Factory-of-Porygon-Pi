#!/usr/bin/env python3
"""Mask Lab battles: every candidate against the 49 distinct top-50 meta teams, behaviour-cloning policy on both sides.

Run with /tmp/vgc-pilot/.venv/bin/python. It never uses random play: the policy is bc_100.zip (hash checked),
the same policy 1 as the shared matchup matrix. It starts its own Showdown servers and works on a private copy of
the matrix (~/vgc-data/mask_lab_matrix.sqlite), so the shared database and servers are not touched.
Writes runs/prompt-ab/panels.json.

    nice -n 10 python mask_lab_battle.py --servers 3 --limit 6      # smoke test
    nice -n 10 python mask_lab_battle.py --servers 3                 # everything
"""
import argparse
import json
import socket
import sqlite3
import subprocess
import sys
import time
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


def private_copy():
    if PRIVATE_DB.exists():
        return
    src = sqlite3.connect(MDB.DEFAULT_DB, timeout=60)
    dst = sqlite3.connect(PRIVATE_DB)
    src.backup(dst)
    dst.close()
    src.close()


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--servers', type=int, default=3)
    ap.add_argument('--batch', type=int, default=24, help='teams battled per saved batch')
    ap.add_argument('--limit', type=int, help='only the first N candidates (smoke test)')
    a = ap.parse_args()

    if MDB.sha256_file(MDB.POLICY_FILE) != BC_HASH:
        raise SystemExit('bc_100.zip differs from the verified behaviour-cloning checkpoint; refusing to battle')
    private_copy()
    con = MDB.connect(PRIVATE_DB)
    cands = json.loads((RUN / 'candidates.json').read_text())['candidates']
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

    procs, ports = start_servers(a.servers)
    pool.live_ports = lambda *args, **kwargs: ports
    started = time.time()
    try:
        for i in range(0, len(rows), a.batch):
            t0 = time.time()
            n = MDB.ensure(con, POLICY, rows[i:i + a.batch], cols, 1, seed=SEED, note='maskLab')
            print(f'batch {i // a.batch + 1}/{-(-len(rows) // a.batch)}: {n} battles in {time.time() - t0:.0f}s '
                  f'({time.time() - started:.0f}s total)', flush=True)
    finally:
        for proc in procs:
            proc.terminate()
        for proc in procs:
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()

    out = {'policy_id': POLICY, 'policy': 'bc_100.zip:' + BC_HASH + ' on both sides', 'seed': SEED,
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
        s = MDB.score(con, POLICY, t)
        cells = MDB.row_cells(con, POLICY, t, cols)
        out['panels'][c['label']] = {'team_id': t, 'score': s['score'], 'coverage': s['coverage'],
                                     'battles': s['battles'], 'se': s['se'],
                                     'by_opponent': {str(o): list(v) for o, v in cells.items()}}
    (RUN / 'panels.json').write_text(json.dumps(out, indent=1) + '\n')
    scored = sum(1 for p in out['panels'].values() if p.get('score') is not None)
    print(f'saved {scored} scored panels to {RUN / "panels.json"}', flush=True)


if __name__ == '__main__':
    main()
