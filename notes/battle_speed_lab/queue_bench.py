"""Compare static direct-engine chunks with an HTTP durable leased queue.

All worker engines are local independent processes. This tests transport,
durability and scheduling overhead, not network latency or multi-host scaling.
"""
import argparse
import concurrent.futures
import json
import os
import random
import secrets
import threading
import time
from direct_bench import Direct,signatures
from leased_queue import LeaseQueue,Client,sha,http_server
from run import HERE,SOURCE,Monitor,analyze,emit,snapshot_sources,tree_digest,workload


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--workers',type=int,default=4)
    ap.add_argument('--chunks',type=int,default=16)
    ap.add_argument('--battles-per-chunk',type=int,default=12)
    ap.add_argument('--blocks',type=int,default=5)
    ap.add_argument('--run-id',default='queue-'+time.strftime('%Y%m%d-%H%M%S'))
    args=ap.parse_args()
    if min(args.workers,args.chunks,args.battles_per_chunk,args.blocks)<1:ap.error('Positive counts required')
    folder=HERE/'runs'/args.run_id;folder.mkdir(parents=True,exist_ok=False)
    hashes=snapshot_sources(folder)
    runtime=sha({'compiled_engine':tree_digest(SOURCE/'dist'),'driver':hashes['direct.cjs'],
                 'policy':'showdown-random-v1'})
    work=workload(args.chunks*args.battles_per_chunk)
    (folder/'workload.json').write_text(json.dumps(work,indent=2)+'\n')
    (folder/'metadata.json').write_text(json.dumps({'args':vars(args),'source_hashes':hashes,'runtime':runtime,
        'limitation':'Local worker engines and authenticated loopback HTTP; not a multi-host performance test.'},indent=2)+'\n')
    workers=[];rows=[];references={}
    try:
        for i in range(args.workers):workers.append(Direct(folder,f'engine-{i}'))
        def execute(queued,phase,block,position):
            chunks=[]
            for i in range(args.chunks):
                jobs=[{**job,'seed':20261013+10000*block+job['id']}
                      for job in work['pairs'][i*args.battles_per_chunk:(i+1)*args.battles_per_chunk]]
                chunks.append({'chunk':i,'request':{'format':work['format'],'jobs':jobs,
                    'concurrency':1,'cacheTeams':True,'returnLogs':False}})
            queue=server=thread=None
            token=secrets.token_urlsafe(32)
            if queued:
                queue=LeaseQueue(folder/f'queue-{block}-{position}.sqlite',runtime)
                server=http_server(queue,token)
                thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
                url=f'http://127.0.0.1:{server.server_port}'
            def one_worker(index):
                output=[]
                if not queued:
                    for chunk in chunks[index::args.workers]:
                        output.append(workers[index].run(chunk['request']))
                    return output
                client=Client(url,token)
                while True:
                    job=client.call('claim',worker=f'worker-{index}',runtime=runtime,seconds=180)
                    if job is None:return output
                    result=workers[index].run(job['payload']['request'])
                    client.call('complete',identity=job['id'],token=job['token'],result=result)
            try:
                with Monitor([os.getpid(),*[w.proc.pid for w in workers]]) as monitor:
                    start=time.perf_counter()
                    if queued:
                        client=Client(url,token)
                        for chunk in chunks:client.call('submit',payload=chunk)
                    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
                        nested=list(pool.map(one_worker,range(args.workers)))
                    if queued:
                        stored=queue.results()
                        if sorted(x['payload']['chunk'] for x in stored)!=list(range(args.chunks)):
                            raise RuntimeError('Missing or duplicate committed chunks')
                        result_list=[x['result'] for x in stored]
                    else:result_list=[x for group in nested for x in group]
                    seconds=time.perf_counter()-start
                results=sorted([x for r in result_list for x in r['results']],key=lambda x:x['id'])
                if len(results)!=len(work['pairs']):raise RuntimeError('Incorrect battle count')
                signature=signatures({'results':results})
                if block not in references:references[block]=signature
                if signature!=references[block]:raise RuntimeError('Queue execution changed battle traces')
                row={'phase':phase,'block':block,'position':position,'variant':'leased_queue' if queued else 'baseline',
                    'treatment':'leased_queue','seconds':seconds,'completed':len(results),
                    'turns':sum(x['turns'] for x in results),'results':results,'trace_equivalent':True,
                    'clock':monitor.clock,'telemetry':monitor.samples,
                    'queue':queue.status() if queue else None}
                rows.append(row);emit(folder/'trials.jsonl',row)
                print(json.dumps({k:row[k] for k in ['phase','block','variant','seconds','completed']}),flush=True)
            finally:
                if server:server.shutdown();server.server_close();thread.join()
        execute(False,'warmup',-1,0);execute(True,'warmup',-1,1)
        rng=random.Random(853)
        for block in range(args.blocks):
            order=[False,True,True,False] if rng.getrandbits(1) else [True,False,False,True]
            for position,queued in enumerate(order):execute(queued,'measure',block,position)
            (folder/'summary.json').write_text(json.dumps(analyze(rows),indent=2)+'\n')
    finally:
        for worker in workers:worker.close()


if __name__=='__main__':main()
