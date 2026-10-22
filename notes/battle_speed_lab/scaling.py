"""Independent simulator scaling and static versus dynamic work allocation."""
import argparse
import concurrent.futures
import json
import queue
import random
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from run import HERE, Monitor, Server, Worker, analyze, digest, emit, prepare_runtime, workload, snapshot_sources
from direct_bench import Direct


class ActorServer(Direct):
    def __init__(self,folder,wait_ms=1.):
        self.log=(folder/'remote-actor.log').open('w')
        self.socket=str(Path(tempfile.gettempdir())/('vgc-'+uuid.uuid4().hex[:12]+'.sock'))
        self.proc=subprocess.Popen([sys.executable,str(folder/'source/remote_inference.py'),
            '--socket',self.socket,'--wait-ms',str(wait_ms),'--stats',str(folder/'remote-actor-stats.json')],
            stdout=subprocess.PIPE,stderr=self.log,text=True,start_new_session=True)
        try:self.ready=self.read()
        except BaseException:self.close();raise


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', nargs='+', type=int, default=[2,4])
    ap.add_argument('--blocks',type=int,default=5)
    ap.add_argument('--jobs',type=int,default=12)
    ap.add_argument('--battles-per-job',type=int,default=24)
    ap.add_argument('--concurrency',type=int,default=24)
    ap.add_argument('--compare-concurrency',nargs='+',type=int,
                    help='Compare these in-flight limits against --concurrency at each fixed worker count')
    ap.add_argument('--shared-inference',action='store_true',help='Compare central actor against local actor at fixed worker counts')
    ap.add_argument('--fast-stack',action='store_true',help='Use the validated fast encoder and local actor')
    ap.add_argument('--skip-dynamic',action='store_true')
    ap.add_argument('--inference-wait-ms',type=float,default=1.)
    ap.add_argument('--run-id',default='scaling-'+time.strftime('%Y%m%d-%H%M%S'))
    args = ap.parse_args()
    if min(args.workers+[args.blocks,args.jobs,args.battles_per_job,args.concurrency])<1:
        ap.error('all counts must be positive')
    if args.compare_concurrency:
        if args.shared_inference or min(args.compare_concurrency)<1:
            ap.error('Concurrency comparisons require positive limits and local inference')
        if args.battles_per_job<max([args.concurrency]+args.compare_concurrency):
            ap.error('Each job must contain enough battles to exercise every concurrency limit')
    folder = HERE/'runs'/args.run_id
    folder.mkdir(parents=True,exist_ok=False)
    runtime = HERE/'runs/private-runtime'
    prepare_runtime(runtime)
    base = workload(args.battles_per_job)
    jobs = []
    for i in range(args.jobs):
        # Rotate both the row and the ordered opponents, keeping each job's row fixed.
        row = base['pairs'][i%len(base['pairs'])]
        pairs = [{**p,'a':row['b'],'a_id':row['b_id'],'a_species':row['b_species'],'a_base':row['b_base']}
                 for p in base['pairs']]
        jobs.append({'job_id':i,'format':base['format'],'pairs':pairs,'seed':9000+i})
    (folder/'workload.json').write_text(json.dumps(jobs,indent=2)+'\n')
    config = {'batcher':'original','threads':1,'concurrency':args.concurrency,'shared':False}
    if args.shared_inference or args.fast_stack:
        config.update(encoder='fast',actor_only=True,inference_mode=True,shared=True)
    (folder/'metadata.json').write_text(json.dumps({'args':vars(args),'config':config,
        'source_hashes':snapshot_sources(folder),
        'worker_sha256':digest(HERE/'worker.py'),'controller_sha256':digest(Path(__file__)),
        'note':'Fixed jobs; stochastic live-server trajectories. Idle pool processes remain present in all treatments.'},indent=2)+'\n')
    servers, workers, rows = [], [], []
    remote=None
    try:
        if args.shared_inference:remote=ActorServer(folder,args.inference_wait_ms)
        for i in range(max(args.workers)):
            servers.append(Server(runtime,folder))
            workers.append(Worker(config,folder,str(i)))
        roots = [s.proc.pid for s in servers]+[w.proc.pid for w in workers]
        if remote:roots.append(remote.proc.pid)
        def execute(count,dynamic,block,variant,treatment,position,phase='measure',use_remote=False,trial_concurrency=None):
            trial_config={**config}
            if trial_concurrency is not None:trial_config['concurrency']=trial_concurrency
            if use_remote:trial_config['inference_socket']=remote.socket
            todo = queue.Queue()
            for j in jobs:
                todo.put(j)
            def run_one(i):
                results = []
                fixed = iter(jobs[i::count])
                while True:
                    try:
                        job = todo.get_nowait() if dynamic else next(fixed)
                    except (queue.Empty,StopIteration):
                        return results
                    request = {**job,'config':trial_config,'seed':job['seed']+10000*(block+1),'port':servers[i].port,'timeout':180}
                    result = workers[i].trial(request)
                    results.append({'worker':i,'job_id':job['job_id'],**result})
            with Monitor(roots) as monitor:
                start = time.perf_counter()
                with concurrent.futures.ThreadPoolExecutor(max_workers=count) as pool:
                    nested = list(pool.map(run_one,range(count)))
                elapsed = time.perf_counter()-start
            results = [r for group in nested for r in group]
            if sorted(r['job_id'] for r in results) != list(range(len(jobs))):
                raise RuntimeError('Missing or duplicated jobs')
            row = {'phase':phase,'block':block,'variant':variant,'treatment':treatment,'position':position,
                'workers':count,'dynamic':dynamic,'seconds':elapsed,'completed':sum(r['completed'] for r in results),
                'remote_inference':use_remote,
                'concurrency':trial_config['concurrency'],
                'turns':sum(r['turns'] for r in results),'jobs':results,'telemetry':monitor.samples,'clock':monitor.clock,
                'worker_busy_seconds':[sum(r['seconds'] for r in group) for group in nested]}
            rows.append(row);emit(folder/'trials.jsonl',row)
            print(json.dumps({k:row[k] for k in ['variant','phase','block','seconds','completed']}),flush=True)
        execute(max(args.workers),False,-1,'warmup','warmup',-1,'warmup')
        if remote:execute(max(args.workers),False,-1,'remote-warmup','warmup',-1,'warmup',use_remote=True)
        rng = random.Random(572)
        treatments = [(f'workers{n}',n,False,1,False) for n in args.workers]
        if not args.skip_dynamic:treatments += [(f'dynamic{n}',n,True,n,False) for n in args.workers if n>1]
        if remote:treatments=[(f'shared_actor{n}',n,False,n,True) for n in args.workers]
        treatments=[(*t,args.concurrency) for t in treatments]
        if args.compare_concurrency:
            treatments=[(f'workers{n}_concurrency{c}',n,False,n,False,c)
                        for n in args.workers for c in args.compare_concurrency if c!=args.concurrency]
        for block in range(args.blocks):
            rng.shuffle(treatments)
            for name,count,dynamic,baseline_count,use_remote,trial_concurrency in treatments:
                sequence = [False,True,True,False]
                if rng.getrandbits(1):sequence=[True,False,False,True]
                for pos,is_treatment in enumerate(sequence):
                    execute(count if is_treatment else baseline_count,dynamic if is_treatment else False,
                            block,name if is_treatment else 'baseline',name,pos,use_remote=use_remote and is_treatment,
                            trial_concurrency=trial_concurrency if is_treatment else args.concurrency)
                (folder/'summary.json').write_text(json.dumps(analyze(rows),indent=2)+'\n')
    finally:
        for w in workers:w.close()
        for s in servers:s.close()
        if remote:remote.close()
    print('RESULTS '+str(folder),flush=True)


if __name__ == '__main__':
    main()
