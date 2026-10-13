"""Persistent optimized BC runner implementing matchup_db.ensure's runner contract.

Database access stays on the caller thread. Results are returned only after every
requested battle is complete and attributed; ensure commits them atomically.
"""
import concurrent.futures
import hashlib
import importlib.metadata
import json
import shutil
import sys
import threading
import time
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from battle_speed_lab import run as lab

FORMAT = 'gen9championsvgc2026regmb'
CHECKPOINT = Path('/tmp/bc_100.zip')
CHECKPOINT_SHA256 = '57f5edcab415cf6ccc1b6231923c8b66d3b1b7249b6b2562b97531471e4ca60b'


def python_source_digest(root):
    digest = hashlib.sha256()
    for file in sorted(Path(root).rglob('*.py')):
        digest.update(str(file.relative_to(root)).encode()+b'\0')
        digest.update(file.read_bytes())
    return digest.hexdigest()


def base_key(species):
    from corpus import dex_entry
    values = []
    for name in species:
        entry = dex_entry(name)
        base = entry.get('baseSpecies', name) if entry else name
        values.append(''.join(c for c in base.lower() if c.isalnum()))
    return ','.join(sorted(values))


def build_requests(con, jobs, chunk_size=24):
    if chunk_size < 1:
        raise ValueError('chunk_size must be positive')
    needed = {j['row'] for j in jobs} | {c for j in jobs for c in j['schedule']}
    teams = {}
    for identity in needed:
        row = con.execute('SELECT paste,base_key,format_id,legal FROM team WHERE team_id=?', (identity,)).fetchone()
        if row is None or row[2] != FORMAT or row[3] != 1:
            raise ValueError(f'Team {identity} is missing, illegal, or in another format')
        teams[identity] = row
    requests = []
    for job in jobs:
        candidate = job['row']
        mapping = {}
        for opponent in job['schedule']:
            if candidate == opponent:
                raise ValueError('A team cannot battle itself')
            key = teams[opponent][1]
            if key in mapping and mapping[key] != opponent:
                raise ValueError('Ambiguous opponent base-species identity; use matchup_db.jobs_for')
            mapping[key] = opponent
        for offset in range(0, len(job['schedule']), chunk_size):
            pairs = [{'id':i, 'a_id':candidate, 'b_id':opponent,
                      'a':teams[candidate][0], 'b':teams[opponent][0],
                      'a_base':teams[candidate][1], 'b_base':teams[opponent][1]}
                     for i, opponent in enumerate(job['schedule'][offset:offset + chunk_size])]
            requests.append({'job_id':len(requests), 'row':candidate, 'format':FORMAT, 'pairs':pairs})
    return requests


def assign_requests(requests, workers):
    """Balance large chunks first so repeated short remainders cannot strand cores."""
    shards = [[] for _ in range(workers)]
    loads = [0]*workers
    for request in sorted(requests, key=lambda r: (-len(r['pairs']), r['job_id'])):
        index = min(range(workers), key=lambda i: (loads[i], i))
        shards[index].append(request)
        loads[index] += len(request['pairs'])
    return shards


def attribute(request, outcome):
    pairs = request['pairs']
    battles = outcome.get('battles', [])
    if outcome.get('completed') != len(pairs) or len(battles) != len(pairs):
        raise RuntimeError('Incomplete battle chunk; nothing may be committed')
    expected = Counter(p['b_base'] for p in pairs)
    opponents = {p['b_base']:p['b_id'] for p in pairs}
    observed = Counter()
    totals = defaultdict(lambda: [0, 0, 0])
    tags = set()
    for battle in battles:
        if not battle.get('finished') or not isinstance(battle.get('tag'), str) or battle['tag'] in tags:
            raise RuntimeError('Unfinished or duplicated battle')
        tags.add(battle.get('tag'))
        if len(battle['team']) != 6 or base_key(battle['team']) != pairs[0]['a_base']:
            raise RuntimeError('Candidate attribution mismatch')
        key = base_key(battle['opponent'])
        if len(battle['opponent']) != 6 or key not in opponents:
            raise RuntimeError('Opponent attribution mismatch')
        observed[key] += 1
        won = battle['won']
        if won is not True and won is not False and won is not None:
            raise RuntimeError('Invalid battle outcome')
        totals[opponents[key]][0 if won is True else 1 if won is False else 2] += 1
    if observed != expected:
        raise RuntimeError('Opponent battle counts differ from requested schedule')
    return [(request['row'], opponent, *counts) for opponent, counts in sorted(totals.items())]


class FastBattleRunner:
    def __init__(self, folder, workers=6, concurrency=24, runtime=None):
        if workers < 1 or concurrency < 1:
            raise ValueError('Worker and concurrency counts must be positive')
        if lab.digest(CHECKPOINT) != CHECKPOINT_SHA256:
            raise ValueError('BC checkpoint differs from the verified policy')
        self.folder = Path(folder).resolve()
        self.folder.mkdir(parents=True, exist_ok=False)
        self.runtime = Path(runtime).resolve() if runtime else self.folder/'runtime'
        self.worker_count = workers
        self.config = {'batcher':'original', 'threads':1, 'concurrency':concurrency,
                       'shared':True, 'encoder':'fast', 'actor_only':True, 'inference_mode':True}
        self.servers, self.workers = [], []
        self.closed = False
        self.calls = 0
        sources = lab.snapshot_sources(self.folder)
        shutil.copy2(__file__, self.folder/'source/fast_battle_runner.py')
        relevant = {name:sources[name] for name in ['worker.py', 'encoder.py', 'inference.py', 'run.py']}
        relevant['fast_battle_runner.py'] = lab.digest(Path(__file__))
        self.provenance = {'configuration':self.config, 'workers':workers,
                           'policy_sha256':CHECKPOINT_SHA256, 'implementation':relevant,
                           'simulator_sha256':lab.tree_digest(lab.SOURCE/'dist'),
                           'formats_sha256':lab.digest(lab.SOURCE/'config/formats.ts'),
                           'bench_source_sha256':python_source_digest(Path.home()/'.local/share/vgc-pilot-runtime/unpacked/vgc-bench-d79f9532947ac114dce1dda2456a590afcd375b2/vgc_bench/src'),
                           'bench_data_sha256':lab.tree_digest(Path.home()/'.local/share/vgc-pilot-runtime/unpacked/vgc-bench-d79f9532947ac114dce1dda2456a590afcd375b2/data'),
                           'packages':{name:importlib.metadata.version(name) for name in
                                       ['numpy', 'torch', 'poke-env', 'stable-baselines3']}}
        self.identity = hashlib.sha256(json.dumps(self.provenance, sort_keys=True).encode()).hexdigest()
        (self.folder/'metadata.json').write_text(json.dumps({**self.provenance, 'identity':self.identity,
            'source_hashes':sources, 'note':'Policy seeds do not pin live simulator trajectories.'}, indent=2)+'\n')

    def policy_id(self, con, mdb):
        return mdb.get_policy(con, row_policy='bc-fast:'+self.identity,
                              simulator='showdown:'+self.provenance['simulator_sha256'],
                              note='Optimized BC; full provenance in '+str(self.folder))

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def _start(self):
        if self.closed:
            raise RuntimeError('Runner is closed')
        if self.workers:
            return
        try:
            lab.prepare_runtime(self.runtime)
            if lab.tree_digest(self.runtime/'dist') != self.provenance['simulator_sha256']:
                raise RuntimeError('Private simulator differs from the cache identity')
            for i in range(self.worker_count):
                self.servers.append(lab.Server(self.runtime, self.folder))
                self.workers.append(lab.Worker(self.config, self.folder, str(i)))
                lab.emit(self.folder/'startup.jsonl', {'worker':i, 'worker_pid':self.workers[-1].proc.pid,
                         'server_pid':self.servers[-1].proc.pid, **self.workers[-1].ready})
        except BaseException:
            self.close()
            raise

    def close(self):
        if self.closed:
            return
        self.closed = True
        for process in [*self.workers, *self.servers]:
            process.close()

    def __call__(self, con, jobs, seed=None):
        if self.closed:
            raise RuntimeError('Runner is closed')
        requests = build_requests(con, jobs, self.config['concurrency'])
        if not requests:
            return [], 0, 0.
        seed = 20261013 if seed is None else seed
        if not isinstance(seed, int) or seed < 0 or seed + len(requests) >= 2**32:
            raise ValueError('Seed range is outside NumPy limits')
        self.calls += 1
        serial = self.calls
        lab.emit(self.folder/'requests.jsonl', {'call':serial, 'seed':seed, 'jobs':requests})
        start = time.perf_counter()
        self._start()
        shards = assign_requests(requests, self.worker_count)
        lock = threading.Lock()
        def consume(index):
            results = []
            for request in shards[index]:
                result = self.workers[index].trial({**request, 'seed':seed + request['job_id'],
                    'config':self.config, 'port':self.servers[index].port, 'timeout':180})
                attributed = attribute(request, result)
                with lock:
                    lab.emit(self.folder/'results.jsonl', {'call':serial, 'job_id':request['job_id'],
                             'worker':index, 'row':request['row'], **result})
                results.extend(attributed)
            return results
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.worker_count)
        try:
            futures = [executor.submit(consume, i) for i in range(self.worker_count)]
            results = [entry for future in concurrent.futures.as_completed(futures) for entry in future.result()]
        except BaseException as exc:
            lab.emit(self.folder/'failures.jsonl', {'call':serial, 'error':type(exc).__name__+': '+str(exc)})
            self.close()
            raise
        finally:
            executor.shutdown(wait=True, cancel_futures=True)
        seconds = time.perf_counter() - start
        lab.emit(self.folder/'calls.jsonl', {'call':serial, 'battles':sum(sum(r[2:]) for r in results),
                                           'seconds_including_lazy_startup':seconds})
        return results, 0, seconds
