"""Matched-workload comparison of live-server and direct BC job adapters.

Both use the same fast policy path. Whole job-call latency is measured, including
the direct adapter's per-job engine launch. This is not transport-only overhead
and server simulator seeds are not controllably paired with direct seeds.
"""
import argparse
import asyncio
import json
import os
import random
import time
import direct_bc
from direct_bc import w
from run import HERE,Server,Worker,Monitor,analyze,emit,prepare_runtime,snapshot_sources,workload


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--battles',type=int,default=96)
    ap.add_argument('--blocks',type=int,default=5)
    ap.add_argument('--run-id',default='transport-'+time.strftime('%Y%m%d-%H%M%S'))
    args=ap.parse_args();folder=HERE/'runs'/args.run_id;folder.mkdir(parents=True,exist_ok=False)
    hashes=snapshot_sources(folder);runtime=HERE/'runs/private-runtime';prepare_runtime(runtime)
    config={'batcher':'original','shared':True,'deterministic':False,'per_battle_rng':True,
        'encoder':'fast','concurrency':24,'timeout':180,'actor_only':True,'inference_mode':True,'threads':1}
    works=[workload(args.battles,i) for i in [0,7,14,21,28]]
    (folder/'workload.json').write_text(json.dumps(works,indent=2)+'\n')
    (folder/'metadata.json').write_text(json.dumps({'args':vars(args),'config':config,'source_hashes':hashes,
        'note':'Whole job latency includes direct per-job Node startup; policy loads excluded for both. Server RNG not paired. This does not isolate WebSocket cost.'},indent=2)+'\n')
    w.torch.set_num_threads(1);policy=w.PPO.load('/tmp/bc_100.zip',device='cpu').policy;policy.set_training_mode(False)
    server=worker=None;rows=[]
    try:
        server=Server(runtime,folder);worker=Worker(config,folder,'server')
        def execute(direct,phase,block,position):
            work=works[max(block,0)%len(works)]
            trial_config={**config,'policy_seed':220000+10000*(block+1),'simulator_seed':330000+10000*(block+1)}
            with Monitor([os.getpid(),server.proc.pid,worker.proc.pid]) as monitor:
                start=time.perf_counter()
                if direct:
                    target=folder/f'{phase}-{block}-{position}';target.mkdir()
                    result=asyncio.run_coroutine_threadsafe(direct_bc.evaluate(policy,trial_config,work,target),w.POKE_LOOP).result()
                else:
                    result=worker.trial({**work,'config':trial_config,'port':server.port,
                                         'seed':440000+10000*(block+1),'timeout':180})
                seconds=time.perf_counter()-start
            row={**result,'battle_seconds':result['seconds'],'seconds':seconds,'phase':phase,'block':block,
                'position':position,'variant':'direct' if direct else 'baseline','treatment':'direct',
                'trace_equivalent':False,'telemetry':monitor.samples,'clock':monitor.clock}
            if result['completed']!=len(work['pairs']):raise RuntimeError('Incomplete matched workload')
            rows.append(row);emit(folder/'trials.jsonl',row)
            print(json.dumps({'variant':row['variant'],'phase':phase,'block':block,'seconds':seconds}),flush=True)
        execute(False,'warmup',-1,0);execute(True,'warmup',-1,1)
        rng=random.Random(1601)
        for block in range(args.blocks):
            order=[False,True,True,False] if rng.getrandbits(1) else [True,False,False,True]
            for position,direct in enumerate(order):execute(direct,'measure',block,position)
            (folder/'summary.json').write_text(json.dumps(analyze(rows),indent=2)+'\n')
    finally:
        if worker:worker.close()
        if server:server.close()


if __name__=='__main__':main()
