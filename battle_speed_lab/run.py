"""Isolated live-server benchmark with repeated paired blocks and load telemetry."""
import argparse
import hashlib
import json
import math
import os
import random
import resource
import selectors
import shutil
import signal
import socket
import sqlite3
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = Path.home()/'.local/share/vgc-pilot-runtime/validator-913da36'
BASE = {'batcher': 'original', 'threads': 8, 'concurrency': 24, 'shared': False}
VARIANTS = {
    'baseline': BASE,
    'threads1': {**BASE, 'threads': 1},
    'threads2': {**BASE, 'threads': 2},
    'deadline0': {**BASE, 'batcher': 'deadline', 'wait_ms': 0},
    'deadline05': {**BASE, 'batcher': 'deadline', 'wait_ms': .5},
    'deadline1': {**BASE, 'batcher': 'deadline', 'wait_ms': 1},
    'deadline2': {**BASE, 'batcher': 'deadline', 'wait_ms': 2},
    'shared': {**BASE, 'shared': True},
    'combined': {**BASE, 'batcher': 'deadline', 'wait_ms': 1, 'threads': 1, 'shared': True},
    'encoder': {**BASE, 'encoder': 'fast'},
    'encoder_verify': {**BASE, 'encoder': 'fast', 'verify_encoder': True},
    'encoder_shared': {**BASE, 'encoder': 'fast', 'shared': True, 'threads': 1},
    'inference_mode': {**BASE, 'inference_mode':True},
    'actor_only': {**BASE, 'actor_only':True,'inference_mode':True},
    'logit_cache': {**BASE, 'actor_only':True,'inference_mode':True,'logit_cache':True},
    'fast_stack': {**BASE, 'encoder':'fast','actor_only':True,'inference_mode':True,'shared':True,'threads':1},
    'fast_cache': {**BASE, 'encoder':'fast','actor_only':True,'inference_mode':True,'shared':True,'threads':1,'logit_cache':True},
    'cold_workers': {**BASE,'cold_worker':True},
    'upstream': {**BASE,'upstream':True},
    'production': {**BASE,'upstream':True,'preserve_training':True},
    'concurrency8': {**BASE,'concurrency_override':8},
    'concurrency50': {**BASE,'concurrency_override':50},
    'student': {**BASE,'encoder':'fast','actor_only':True,'inference_mode':True,'shared':True,'threads':1},
}


def emit(path, value):
    with path.open('a') as f:
        f.write(json.dumps(value)+'\n')
        f.flush()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot_sources(folder):
    target = folder/'source'
    target.mkdir(exist_ok=True)
    hashes = {}
    for file in sorted([*HERE.glob('*.py'),*HERE.glob('*.cjs')]):
        shutil.copy2(file,target/file.name)
        hashes[file.name] = digest(file)
    return hashes


def tree_digest(path):
    hasher=hashlib.sha256()
    for file in sorted(path.rglob('*')):
        if file.is_file():
            hasher.update(str(file.relative_to(path)).encode()+b'\0')
            hasher.update(file.read_bytes())
    return hasher.hexdigest()


def prepare_runtime(destination):
    """Private compiled code/config/log paths; dependencies alone are shared."""
    if (destination/'READY').exists():
        return
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copytree(SOURCE/'dist', destination/'dist', dirs_exist_ok=True)
    for name in ['pokemon-showdown', 'package.json']:
        shutil.copy2(SOURCE/name, destination/name)
    dependencies = destination/'node_modules'
    if not dependencies.exists():
        dependencies.symlink_to(SOURCE/'node_modules', target_is_directory=True)
    (destination/'config').mkdir(exist_ok=True)
    shutil.copy2(SOURCE/'config/config-example.js', destination/'config/config-example.js')
    shutil.copy2(SOURCE/'config/formats.ts', destination/'config/formats.ts')
    (destination/'config/config.js').write_text(
        "exports.port = Number(process.env.SPEED_LAB_PORT);\n"
        "exports.bindaddress = '127.0.0.1';\n"
        "exports.repl = false;\nexports.nofswriting = true;\n"
        "exports.watchconfig = false;\nexports.logchat = false;\n"
        "exports.logchallenges = false;\nexports.reportjoins = false;\n"
        "exports.subprocesses = { simulator: 1, validator: 1 };\n"
    )
    (destination/'logs').mkdir(exist_ok=True)
    (destination/'READY').write_text('private benchmark runtime\n')


def stop_group(proc):
    # This process group was created with start_new_session=True by this harness.
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    except PermissionError:
        if proc.poll() is None:
            proc.terminate()
    try:
        proc.wait(timeout=8)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait(timeout=8)


class Server:
    def __init__(self, runtime, folder):
        with socket.socket() as s:
            s.bind(('127.0.0.1', 0))
            self.port = s.getsockname()[1]
        self.log = (folder/f'server-{self.port}.log').open('w')
        self.proc = subprocess.Popen(['node', 'pokemon-showdown', 'start', '--skip-build', str(self.port), '--no-security'],
                                     cwd=runtime, env={**os.environ, 'SPEED_LAB_PORT': str(self.port)},
                                     stdout=self.log, stderr=subprocess.STDOUT, start_new_session=True)
        deadline = time.monotonic()+60
        try:
            while time.monotonic() < deadline:
                if self.proc.poll() is not None:
                    raise RuntimeError('Private Showdown exited; inspect '+str(self.log.name))
                try:
                    with socket.create_connection(('127.0.0.1', self.port), timeout=.2):
                        return
                except OSError:
                    time.sleep(.1)
            raise TimeoutError('Private Showdown startup timed out')
        except BaseException:
            self.close()
            raise

    def close(self):
        stop_group(self.proc)
        self.log.close()


class Worker:
    def __init__(self, config, folder, name):
        self.log = (folder/f'worker-{name}.log').open('w')
        start = time.perf_counter()
        script = folder/'source/worker.py'
        if not script.exists():script = HERE/'worker.py'
        self.proc = subprocess.Popen([sys.executable, str(script), json.dumps(config)],
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log,
                                     text=True, bufsize=1, start_new_session=True,
                                     cwd=Path.home()/'.local/share/vgc-pilot-runtime/unpacked/vgc-bench-d79f9532947ac114dce1dda2456a590afcd375b2')
        try:
            self.ready = self.read(90)
            self.ready['startup_seconds'] = time.perf_counter()-start
        except BaseException:
            self.close()
            raise

    def read(self, timeout):
        with selectors.DefaultSelector() as selector:
            selector.register(self.proc.stdout, selectors.EVENT_READ)
            if not selector.select(timeout):
                raise TimeoutError('Worker response timeout; see '+str(self.log.name))
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError('Worker exited; see '+str(self.log.name))
        return json.loads(line)

    def trial(self, request):
        self.proc.stdin.write(json.dumps(request)+'\n')
        self.proc.stdin.flush()
        result = self.read(request.get('timeout', 180)+20)
        if 'error' in result:
            raise RuntimeError(result['error']+'; see '+str(self.log.name))
        return result

    def close(self):
        stop_group(self.proc)
        for stream in (self.proc.stdin, self.proc.stdout, self.proc.stderr):
            if stream is not None:
                stream.close()
        self.log.close()


class Monitor:
    """ps CPU percentages are smoothed OS estimates, not instant utilization."""
    def __init__(self, roots):
        self.roots = set(roots)
        self.samples = []
        self.done = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.clock = {}

    def run(self):
        while not self.done.is_set():
            start = time.monotonic()
            try:
                raw = subprocess.check_output(['ps', '-axo', 'pid=,ppid=,%cpu=,rss='], text=True)
                rows = [line.split() for line in raw.splitlines()]
                rows = [(int(p), int(pp), float(cpu), int(rss)) for p, pp, cpu, rss in rows]
                own = set(self.roots)
                for _ in range(8):
                    expanded = own | {p for p, pp, _, _ in rows if pp in own}
                    if expanded == own:
                        break
                    own = expanded
                self.samples.append({'t': time.time(), 'monotonic':time.perf_counter(),'loadavg': os.getloadavg(),
                    'experiment_cpu_percent': sum(cpu for p, _, cpu, _ in rows if p in own),
                    'external_cpu_percent': sum(cpu for p, _, cpu, _ in rows if p not in own),
                    'experiment_rss_kib': sum(rss for p, _, _, rss in rows if p in own)})
            except (subprocess.SubprocessError, ValueError, OSError) as exc:
                self.samples.append({'monitor_error': str(exc)})
            self.done.wait(max(.05, .5-(time.monotonic()-start)))

    def __enter__(self):
        self.wall_start=time.time();self.monotonic_start=time.perf_counter()
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.clock={'wall_seconds':time.time()-self.wall_start,
                    'monotonic_seconds':time.perf_counter()-self.monotonic_start}
        self.clock['pause_or_clock_jump']=self.clock['wall_seconds']-self.clock['monotonic_seconds']>2
        self.done.set()
        self.thread.join(timeout=5)


def workload(count,row_index=0):
    db = Path.home()/'vgc-data/mask_lab_matrix.sqlite'
    with sqlite3.connect('file:'+str(db)+'?mode=ro', uri=True) as con:
        rows = con.execute("SELECT t.team_id,t.paste,t.species_key,t.format_id,t.base_key FROM team t JOIN team_set s USING(team_id) WHERE s.set_name='top50' AND t.legal=1 ORDER BY t.team_id").fetchall()
    if len(rows) < 5:
        raise RuntimeError('Insufficient legal top50 teams')
    if not 0<=row_index<len(rows):raise ValueError('row_index outside legal top50 workload')
    a = rows[row_index]
    opponents = rows[:row_index]+rows[row_index+1:]
    # A fixed row means challenge arrival cannot change candidate/opponent pairing.
    pairs = [{'id': i, 'a_id': a[0], 'b_id': opponents[i%len(opponents)][0], 'a': a[1],
              'b': opponents[i%len(opponents)][1], 'a_species': a[2],
              'b_species': opponents[i%len(opponents)][2], 'a_base':a[4],
              'b_base':opponents[i%len(opponents)][4]} for i in range(count)]
    return {'format': a[3], 'pairs': pairs, 'database': str(db)}


def analyze(rows):
    by_block = {}
    invalid=set()
    for r in rows:
        times=[s['t'] for s in r.get('telemetry',[]) if 't'in s]
        legacy_gap=len(times)>1 and times[-1]-times[0]-r.get('seconds',0)>30
        if r.get('clock',{}).get('pause_or_clock_jump',False) or legacy_gap:
            invalid.add((r.get('treatment'),r.get('block')))
    for r in rows:
        if r.get('phase') != 'measure':
            continue
        if (r['treatment'],r['block']) in invalid:
            continue
        by_block.setdefault((r['treatment'], r['block']), {}).setdefault(r['variant'], []).append(r)
    output = {}
    for treatment in sorted({key[0] for key in by_block}):
        ratios, turn_ratios, accepted_blocks = [], [], []
        for (name, block), groups in by_block.items():
            if name != treatment or len(groups.get('baseline', [])) != 2 or len(groups.get(treatment, [])) != 2:
                continue
            def rate(label, numerator):
                return sum(x[numerator] for x in groups[label])/sum(x['seconds'] for x in groups[label])
            ratios.append(rate(treatment, 'completed')/rate('baseline', 'completed'))
            turn_ratios.append(rate(treatment, 'turns')/rate('baseline', 'turns'))
            accepted_blocks.append(block)
        if not ratios:
            continue
        rng = random.Random(991)
        bootstrap = sorted(math.exp(statistics.mean(math.log(rng.choice(ratios)) for _ in ratios)) for _ in range(5000))
        relevant=[r for r in rows if r.get('phase')=='measure' and r.get('treatment')==treatment
                  and r.get('block') in accepted_blocks]
        trace_equivalent=bool(relevant) and all(r.get('trace_equivalent',False) for r in relevant)
        output[treatment] = {'blocks': len(ratios), 'block_speedups': ratios,
            'accepted_block_ids':accepted_blocks,
            'invalid_clock_blocks':sorted(b for t,b in invalid if t==treatment and b>=0),
            'geometric_mean_speedup': math.exp(statistics.mean(map(math.log, ratios))),
            'bootstrap95': [bootstrap[125], bootstrap[4874]] if len(ratios) >= 3 else None,
            'turn_speedup': math.exp(statistics.mean(map(math.log, turn_ratios))),
            'trace_equivalent':trace_equivalent,
            'status': 'preliminary' if len(ratios) < 5 else
                ('repeated comparison with normalized exact trace agreement' if trace_equivalent else
                 'repeated live-server comparison; stochastic workloads, not trace-equivalent')}
    return output


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variants', nargs='+', choices=sorted(set(VARIANTS)-{'baseline'}), default=['threads1','deadline1','shared','combined'])
    ap.add_argument('--blocks', type=int, default=5)
    ap.add_argument('--battles', type=int, default=96)
    ap.add_argument('--concurrency', type=int, default=24)
    ap.add_argument('--run-id', default=time.strftime('%Y%m%d-%H%M%S'))
    ap.add_argument('--timeout', type=int, default=180)
    ap.add_argument('--capture',action='store_true',help='Capture real policy inputs in warmup only, outside measured comparisons')
    ap.add_argument('--row-indices',nargs='+',type=int,default=[0],help='Rotate candidate teams between complete paired blocks')
    ap.add_argument('--baseline',choices=['instrumented','upstream','production','fast_stack'],default='instrumented')
    ap.add_argument('--student-file',type=Path)
    args = ap.parse_args()
    if 'student' in args.variants and (not args.student_file or not args.student_file.is_file()):
        ap.error('student variant requires an existing --student-file')
    if min(args.blocks, args.battles, args.concurrency) < 1:
        ap.error('counts must be positive')
    folder = HERE/'runs'/args.run_id
    folder.mkdir(parents=True, exist_ok=False)
    runtime = HERE/'runs'/'private-runtime'
    prepare_runtime(runtime)
    works = [workload(args.battles,index) for index in args.row_indices]
    (folder/'workload.json').write_text(json.dumps(works[0] if len(works)==1 else works, indent=2)+'\n')
    meta = {'args': vars(args), 'policy_sha256': digest(Path('/tmp/bc_100.zip')),
            'workload_sha256': digest(folder/'workload.json'), 'python': sys.version,
            'worker_sha256': digest(HERE/'worker.py'), 'controller_sha256': digest(Path(__file__)),
            'simulator_source': str(SOURCE), 'platform': dict(zip(['sysname','nodename','release','version','machine'], os.uname())),
            'cpu': subprocess.check_output(['sysctl','-n','machdep.cpu.brand_string'],text=True).strip(),
            'source_hashes': snapshot_sources(folder),
            'private_runtime_dist_sha256':tree_digest(runtime/'dist'),
            'private_runtime_config_sha256':digest(runtime/'config/config.js'),
            'note': 'Fresh simulator randomness per battle; policy sampling is job-seeded, not per-battle deterministic.'}
    if args.student_file:meta['student_sha256']=digest(args.student_file)
    (folder/'metadata.json').write_text(json.dumps(meta, indent=2,default=str)+'\n')
    server = Server(runtime, folder)
    workers = {}
    rows = []
    def configuration(name):
        selected=VARIANTS[name]
        if name=='baseline' and args.baseline!='instrumented':selected=VARIANTS[args.baseline]
        config={**selected,'concurrency':selected.get('concurrency_override',args.concurrency)}
        if name=='student':config['student_file']=str(args.student_file.resolve())
        return config
    try:
        for name in ['baseline', *args.variants]:
            config = configuration(name)
            workers[name] = Worker(config, folder, name)
            emit(folder/'startup.jsonl', {'variant': name, **workers[name].ready})
        roots = [server.proc.pid, *[w.proc.pid for w in workers.values()]]
        rng = random.Random(20261013)
        def execute(name, phase, treatment, block, position):
            work=works[max(block,0)%len(works)]
            config = configuration(name)
            request = {**work, 'port': server.port, 'seed': 20261013+block, 'config': config, 'timeout': args.timeout}
            if phase == 'warmup':
                request['pairs'] = work['pairs'][:min(48, len(work['pairs']))]
                if args.capture:
                    request['config'] = {**config,'capture':True}
                    request['capture_path'] = str(folder/(name+'-observations.npz'))
            with Monitor(roots) as monitor:
                if config.get('cold_worker'):
                    start=time.perf_counter()
                    fresh=Worker(config,folder,name+'-fresh')
                    monitor.roots.add(fresh.proc.pid)
                    try:
                        result=fresh.trial(request)
                        result['warm_seconds']=result['seconds']
                        result['seconds']=time.perf_counter()-start
                        result['cold_startup']=fresh.ready
                    finally:
                        fresh.close()
                else:
                    result = workers[name].trial(request)
            row = {**result, 'variant': name, 'config': config, 'phase': phase, 'treatment': treatment,
                   'block': block, 'position': position,'candidate_id':work['pairs'][0]['a_id'],
                   'telemetry': monitor.samples,'clock':monitor.clock, 'finished_at': time.time()}
            emit(folder/'trials.jsonl', row)
            rows.append(row)
            print(json.dumps({'phase':phase,'variant':name,'block':block,'battles':result['completed'],
                              'seconds':round(result['seconds'],3),'bps':round(result['completed']/result['seconds'],2)}), flush=True)
        for name in workers:
            execute(name, 'warmup', name, -1, -1)
        # Rotate treatment order between blocks to spread changing background load.
        for block in range(args.blocks):
            order = list(args.variants)
            rng.shuffle(order)
            for treatment in order:
                sequence = ['baseline', treatment, treatment, 'baseline']
                if rng.getrandbits(1):
                    sequence.reverse()  # ABBA is palindromic; explicitly swap labels below.
                    sequence = [treatment, 'baseline', 'baseline', treatment]
                for position, name in enumerate(sequence):
                    execute(name, 'measure', treatment, block, position)
                (folder/'summary.json').write_text(json.dumps(analyze(rows), indent=2)+'\n')
    finally:
        for worker in workers.values():
            worker.close()
        server.close()
    print('RESULTS '+str(folder), flush=True)


if __name__ == '__main__':
    main()
