"""Trace-equivalent direct-engine experiments, with seeded random policies."""
import argparse
import json
import os
import random
import selectors
import subprocess
import time
from pathlib import Path
from run import HERE, SOURCE, Monitor, analyze, digest, emit, stop_group, workload, snapshot_sources


class Direct:
    def __init__(self, folder, label):
        self.log = (folder/(label+'.log')).open('a')
        script=folder/'source/direct.cjs'
        if not script.exists():script=HERE/'direct.cjs'
        self.proc = subprocess.Popen(['node',str(script),str(SOURCE)],
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log,
            text=True,bufsize=1,start_new_session=True)
        try:
            self.ready = self.read()
        except BaseException:
            self.close()
            raise

    def read(self):
        with selectors.DefaultSelector() as sel:
            sel.register(self.proc.stdout, selectors.EVENT_READ)
            if not sel.select(180):
                raise TimeoutError('direct simulator response timed out')
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError('direct simulator exited: '+str(self.log.name))
        result = json.loads(line)
        if result.get('error'):
            raise RuntimeError(result['error'])
        return result

    def run(self, request):
        start = time.perf_counter()
        self.proc.stdin.write(json.dumps(request)+'\n')
        self.proc.stdin.flush()
        result = self.read()
        result['engine_seconds'] = result['seconds']
        result['seconds'] = time.perf_counter()-start
        result['log_bytes'] = sum(len(r.pop('log','').encode()) for r in result['results'])
        return result

    def close(self):
        stop_group(self.proc)
        self.log.close()


def signatures(result):
    return [(r['id'],r['outcome'],r['turns'],r['trace']) for r in result['results']]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--blocks',type=int,default=5)
    ap.add_argument('--battles',type=int,default=48)
    ap.add_argument('--variants',nargs='+',default=['minimal','concurrent','team_cache','combined'],
                    choices=['minimal','concurrent','team_cache','combined','cold'])
    ap.add_argument('--run-id',default='direct-'+time.strftime('%Y%m%d-%H%M%S'))
    args = ap.parse_args()
    folder = HERE/'runs'/args.run_id
    folder.mkdir(parents=True,exist_ok=False)
    work = workload(args.battles)
    (folder/'workload.json').write_text(json.dumps(work,indent=2)+'\n')
    meta = {'args':vars(args),'policy':'showdown-random-v1','source':str(SOURCE),
            'direct_sha256':digest(HERE/'direct.cjs'),'controller_sha256':digest(Path(__file__)),
            'workload_sha256':digest(folder/'workload.json'),
            'source_hashes':snapshot_sources(folder),
            'note':'Both policy RNG streams and simulator RNG independently seeded per battle. Exact trace agreement required after removing wall-clock |t:| lines only.'}
    (folder/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    configs = {'baseline':{'concurrency':1,'cacheTeams':False,'returnLogs':True},
               'minimal':{'concurrency':1,'cacheTeams':False,'returnLogs':False},
               'concurrent':{'concurrency':24,'cacheTeams':False,'returnLogs':True},
               'team_cache':{'concurrency':1,'cacheTeams':True,'returnLogs':True},
               'combined':{'concurrency':24,'cacheTeams':True,'returnLogs':False},
               'cold':{'concurrency':1,'cacheTeams':False,'returnLogs':True}}
    clients = {}
    rows = []
    references = {}
    try:
        for name in ['baseline',*args.variants]:
            if name != 'cold':
                clients[name] = Direct(folder,name)
        def execute(name, phase, treatment, block, position):
            jobs = [{**p,'seed':20261013+10000*block+p['id']} for p in work['pairs']]
            request = {**configs[name],'format':work['format'],'jobs':jobs}
            roots = [c.proc.pid for c in clients.values()]
            with Monitor(roots) as monitor:
                if name == 'cold':
                    start = time.perf_counter()
                    collected = []
                    for job in jobs:
                        client = Direct(folder,'cold')
                        monitor.roots.add(client.proc.pid)
                        try:
                            collected.append(client.run({**request,'jobs':[job]}))
                        finally:
                            client.close()
                    result = {'seconds':time.perf_counter()-start,'completed':len(collected),
                              'turns':sum(r['turns'] for r in collected),
                              'engine_seconds':sum(r['engine_seconds'] for r in collected),
                              'results':[r['results'][0] for r in collected],
                              'log_bytes':sum(r['log_bytes'] for r in collected)}
                else:
                    result = clients[name].run(request)
            sig = signatures(result)
            key = (phase,block)
            if key not in references:
                references[key] = sig
            if sig != references[key]:
                emit(folder/'mismatches.jsonl',{'variant':name,'block':block,'expected':references[key],'actual':sig})
                raise RuntimeError('Trace mismatch: performance comparison rejected')
            result.update({'phase':phase,'variant':name,'treatment':treatment,'block':block,
                           'position':position,'telemetry':monitor.samples,'clock':monitor.clock,'trace_equivalent':True})
            emit(folder/'trials.jsonl',result)
            rows.append(result)
            print(json.dumps({'variant':name,'phase':phase,'block':block,'bps':round(result['completed']/result['seconds'],2),
                              'trace_equivalent':True}),flush=True)
        for name in clients:
            execute(name,'warmup',name,-1,-1)
        rng = random.Random(20261013)
        for block in range(args.blocks):
            order = args.variants[:]
            rng.shuffle(order)
            for treatment in order:
                sequence = ['baseline',treatment,treatment,'baseline']
                if rng.getrandbits(1):
                    sequence = [treatment,'baseline','baseline',treatment]
                for pos,name in enumerate(sequence):
                    execute(name,'measure',treatment,block,pos)
                (folder/'summary.json').write_text(json.dumps(analyze(rows),indent=2)+'\n')
    finally:
        for client in clients.values():
            client.close()
    print('RESULTS '+str(folder),flush=True)


if __name__ == '__main__':
    main()
