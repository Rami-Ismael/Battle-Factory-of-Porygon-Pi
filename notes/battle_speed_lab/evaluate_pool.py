"""Reusable fast evaluator for candidates in the read-only top50 team pool.

Keeps policy workers and private simulator servers alive across chunks. Results
are separate artifacts; this program never writes to the matchup database.
"""
import argparse
import concurrent.futures
import json
import signal
import sqlite3
import threading
import time
from pathlib import Path
from run import HERE,Server,Worker,Monitor,digest,emit,prepare_runtime,snapshot_sources,workload


def candidate_rows():
    database=Path.home()/'vgc-data/mask_lab_matrix.sqlite'
    with sqlite3.connect('file:'+str(database)+'?mode=ro',uri=True) as db:
        return db.execute("SELECT t.team_id,t.species_key FROM team t JOIN team_set s USING(team_id) WHERE s.set_name='top50' AND t.legal=1 ORDER BY t.team_id").fetchall()


def make_jobs(candidates,battles,chunk_size,seed):
    rows=candidate_rows();indices={identity:i for i,(identity,_) in enumerate(rows)}
    if len(set(candidates))!=len(candidates):raise ValueError('Duplicate candidate IDs')
    jobs=[]
    for candidate in candidates:
        if candidate not in indices:raise ValueError(f'Candidate {candidate} is not in the legal top50 set')
        work=workload(battles,indices[candidate])
        for offset in range(0,battles,chunk_size):
            identity=len(jobs)
            jobs.append({'job_id':identity,'candidate_id':candidate,'format':work['format'],
                         'pairs':work['pairs'][offset:offset+chunk_size],'seed':seed+identity})
    return jobs


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--candidate-ids',nargs='+',type=int)
    ap.add_argument('--list-candidates',action='store_true')
    ap.add_argument('--battles-per-candidate',type=int,default=96)
    ap.add_argument('--workers',type=int,default=4)
    ap.add_argument('--chunk-size',type=int,default=24)
    ap.add_argument('--concurrency',type=int,default=24)
    ap.add_argument('--seed',type=int,default=20261013)
    ap.add_argument('--run-id',default='evaluation-'+time.strftime('%Y%m%d-%H%M%S'))
    args=ap.parse_args()
    if args.list_candidates:
        print(json.dumps([{'id':i,'species':species} for i,species in candidate_rows()],indent=2));return
    if not args.candidate_ids:ap.error('Choose --candidate-ids or --list-candidates')
    if min(args.battles_per_candidate,args.workers,args.chunk_size,args.concurrency)<1:ap.error('Counts must be positive')
    jobs=make_jobs(args.candidate_ids,args.battles_per_candidate,args.chunk_size,args.seed)
    if not 0<=args.seed or args.seed+len(jobs)>=2**32:ap.error('Seed range is outside NumPy limits')
    folder=HERE/'runs'/args.run_id;folder.mkdir(parents=True,exist_ok=False)
    hashes=snapshot_sources(folder)
    (folder/'workload.json').write_text(json.dumps(jobs,indent=2)+'\n')
    config={'batcher':'original','threads':1,'concurrency':args.concurrency,'shared':True,
            'encoder':'fast','actor_only':True,'inference_mode':True}
    (folder/'metadata.json').write_text(json.dumps({'args':vars(args),'config':config,'source_hashes':hashes,
        'policy_sha256':digest(Path('/tmp/bc_100.zip')),'workload_sha256':digest(folder/'workload.json'),
        'note':'Fresh server simulator randomness; policy seeds are job-level, not complete trace reproducibility. Results remain outside the source database.'},indent=2)+'\n')
    runtime=HERE/'runs/private-runtime';prepare_runtime(runtime)
    servers=[];workers=[];results=[];lock=threading.Lock();stop=threading.Event();pool=None
    previous=signal.getsignal(signal.SIGTERM)
    def interrupted(*_):raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM,interrupted)
    start=time.perf_counter()
    try:
        count=min(args.workers,len(jobs))
        for i in range(count):
            servers.append(Server(runtime,folder));workers.append(Worker(config,folder,str(i)))
            emit(folder/'startup.jsonl',{'worker':i,**workers[-1].ready})
        def consume(index):
            for job in jobs[index::count]:
                if stop.is_set():return
                outcome=workers[index].trial({**job,'config':config,'port':servers[index].port,'timeout':180})
                if outcome['completed']!=len(job['pairs']):raise RuntimeError('Incomplete evaluation chunk')
                record={'job_id':job['job_id'],'candidate_id':job['candidate_id'],'worker':index,**outcome}
                with lock:
                    emit(folder/'results.jsonl',record);results.append(record)
                    print(json.dumps({'completed_chunks':len(results),'total_chunks':len(jobs),
                                      'completed_battles':sum(r['completed'] for r in results)}),flush=True)
        with Monitor([s.proc.pid for s in servers]+[w.proc.pid for w in workers]) as monitor:
            pool=concurrent.futures.ThreadPoolExecutor(max_workers=count)
            futures=[pool.submit(consume,i) for i in range(count)]
            for future in concurrent.futures.as_completed(futures):future.result()
        if sorted(r['job_id'] for r in results)!=list(range(len(jobs))):raise RuntimeError('Missing or duplicated job results')
        summaries=[]
        for candidate in args.candidate_ids:
            selected=[r for r in results if r['candidate_id']==candidate]
            completed=sum(r['completed'] for r in selected)
            wins=sum(r['wins'] for r in selected);ties=sum(r['ties'] for r in selected)
            summaries.append({'candidate_id':candidate,'battles':completed,'wins':wins,'ties':ties,
                              'score':(wins+.5*ties)/completed})
        (folder/'summary.json').write_text(json.dumps({'status':'complete','candidates':summaries,
            'total_seconds_including_startup':time.perf_counter()-start,'clock':monitor.clock,
            'note':'Evaluation output, not a paired performance benchmark or high-confidence ranking.'},indent=2)+'\n')
        print('RESULTS '+str(folder),flush=True)
    except BaseException as exc:
        (folder/'failure.json').write_text(json.dumps({'status':'incomplete','error':type(exc).__name__+': '+str(exc),
            'completed_chunks':len(results),'expected_chunks':len(jobs)},indent=2)+'\n')
        raise
    finally:
        stop.set()
        if pool:pool.shutdown(wait=False,cancel_futures=True)
        for worker in workers:worker.close()
        for server in servers:server.close()
        signal.signal(signal.SIGTERM,previous)


if __name__=='__main__':main()
