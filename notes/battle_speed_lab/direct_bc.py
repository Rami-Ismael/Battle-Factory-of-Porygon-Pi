"""Direct Showdown→poke-env→BC adapter with reference player-view parsing.

Initial validation uses a deterministic BC policy, explicitly distinct from the
stochastic production evaluator. No live-server speedup claim follows until
observations, OTS, legality, and outcomes are compared on a common seeded trace.
"""
import argparse
import asyncio
import json
import os
import time
from pathlib import Path
_original_cwd=os.getcwd()
os.chdir(Path.home()/'.local/share/vgc-pilot-runtime/unpacked/vgc-bench-d79f9532947ac114dce1dda2456a590afcd375b2')
import worker as w
os.chdir(_original_cwd)
from run import HERE,SOURCE,workload


class Transport:
    def __init__(self,original,send,identity,side):
        self.original,self.send,self.identity,self.side=original,send,identity,side
    def __getattr__(self,name):return getattr(self.original,name)
    async def send_message(self,message,room=None):
        await self.send({'kind':'choice','id':self.identity,'side':self.side,'choice':message})


async def evaluate(policy,config,work,folder):
    log=(folder/'bridge.log').open('w')
    events=(folder/'events.jsonl').open('w') if config.get('events') else None
    bridge_arguments=['node',str(HERE/'bridge.cjs'),str(SOURCE)]
    if config.get('node_actor_model'):
        bridge_arguments=['node',str(HERE/'actor_bridge.cjs'),str(SOURCE),str(config['node_actor_model']),
                          config.get('node_actor_runtime','/tmp/vgc-node-actor')]
    proc=await asyncio.create_subprocess_exec(*bridge_arguments,
        stdin=asyncio.subprocess.PIPE,stdout=asyncio.subprocess.PIPE,stderr=log)
    ready=json.loads(await proc.stdout.readline())
    if ready.get('kind')!='ready':raise RuntimeError('Bridge did not initialize')
    metrics={'embedding_seconds':0.,'mask_seconds':0.,'forward_seconds':0.,'stack_seconds':0.,
             'queue_seconds':[],'batch_sizes':[]}
    service=w.Inference(policy,config,metrics)
    channels={}
    traces={}
    inference_futures={}
    results=[]
    async def send(message):
        if events:
            events.write(json.dumps({'direction':'out',**message})+'\n');events.flush()
        proc.stdin.write((json.dumps(message)+'\n').encode())
        await proc.stdin.drain()
    if config.get('node_actor_model'):
        if not config.get('actor_only'):raise ValueError('Node actor requires actor-only sampling')
        from bridge_actor import BridgeActor
        service.remote=BridgeActor(send,inference_futures)
    async def read():
        while True:
            line=await proc.stdout.readline()
            if not line:raise RuntimeError('Bridge unexpectedly exited')
            item=json.loads(line)
            if events:
                events.write(json.dumps({'direction':'in',**item})+'\n');events.flush()
            if item['kind']=='error':raise RuntimeError(item['error'])
            if item['kind']=='logits':
                future=inference_futures.get(item['id'])
                if future is None:raise RuntimeError('Unexpected actor response')
                if not future.done():future.set_result(item)
            elif item['kind']=='trace':
                traces[item['id']]=item['trace']
            else:
                await channels[(item['id'],item['side'])].put(item)
    async def battle(job):
        identity=str(job['id'])
        tag=f"battle-{work['format']}-{identity}"
        players=[]
        for side,name,paste in [('p1','Candidate',job['a']),('p2','Opponent',job['b'])]:
            q=channels[(identity,side)]=asyncio.Queue()
            player=w.InstrumentedPlayer(service,metrics,policy=policy,
                account_configuration=w.AccountConfiguration(name,None),battle_format=work['format'],
                max_concurrent_battles=1,start_listening=False,accept_open_team_sheet=True,
                deterministic=config.get('deterministic',True),team=w.SequenceTeams([paste]),log_level=50)
            player.ps_client=Transport(player.ps_client,send,identity,side)
            players.append(player)
        async def consume(player,side):
            while True:
                item=await channels[(identity,side)].get()
                if item['kind']=='end':break
                lines=[[f'>{tag}']]+[line.split('|') for line in item['chunk'].split('\n')]
                await player._handle_battle_message(lines)
        consumers=[asyncio.create_task(consume(p,s)) for p,s in zip(players,['p1','p2'])]
        await send({'kind':'start',**job,'id':identity,'format':work['format'],
                    'seed':config.get('simulator_seed',20261013)+job['id']})
        try:
            await asyncio.gather(*consumers)
            b=players[0].battles[tag]
            if not b.finished:raise RuntimeError('Nonterminal direct battle')
            results.append({'id':identity,'turns':b.turn,'won':b.won,'finished':b.finished})
        finally:
            for task in consumers:
                if not task.done():task.cancel()
            await asyncio.gather(*consumers,return_exceptions=True)
    reader=asyncio.create_task(read())
    semaphore=asyncio.Semaphore(config['concurrency'])
    async def limited(job):
        async with semaphore:await battle(job)
    start=time.perf_counter()
    tasks=asyncio.gather(*(limited(job) for job in work['pairs']))
    try:
        done,_=await asyncio.wait([tasks,reader],timeout=config.get('timeout',180),return_when=asyncio.FIRST_COMPLETED)
        if reader in done:await reader
        if tasks not in done:
            stacks=[]
            for task in asyncio.all_tasks():
                chain=[];coro=task.get_coro()
                for _ in range(15):
                    if coro is None:break
                    frame=getattr(coro,'cr_frame',None)
                    chain.append({'coroutine':str(getattr(coro,'__qualname__',type(coro).__name__)),
                                  'line':frame.f_lineno if frame else None})
                    coro=getattr(coro,'cr_await',None)
                stacks.append(chain)
            (folder/'timeout-stacks.json').write_text(json.dumps(stacks,indent=2)+'\n')
            raise TimeoutError('Direct BC battles timed out')
        await tasks
        # Trace messages can be queued immediately after the player streams close.
        deadline=time.monotonic()+5
        while len(traces)<len(results) and time.monotonic()<deadline:await asyncio.sleep(.001)
        if len(traces)!=len(results):raise RuntimeError('Missing simulator trace')
        return {'seconds':time.perf_counter()-start,'completed':len(results),'turns':sum(r['turns'] for r in results),
                'results':sorted(results,key=lambda x:int(x['id'])),'traces':traces,
                'bridge_ready':ready,
                'metrics':{k:v for k,v in metrics.items() if k not in ('queue_seconds','batch_sizes')},
                'policy':('bc_100 stochastic with independent per-battle/side policy RNG' if config.get('per_battle_rng')
                          else 'bc_100 deterministic; direct adapter validation, not production-equivalent benchmark')}
    finally:
        tasks.cancel();reader.cancel()
        await asyncio.gather(tasks,reader,return_exceptions=True)
        await service.close()
        proc.terminate()
        await proc.wait()
        log.close()
        if events:events.close()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--battles',type=int,default=4)
    ap.add_argument('--concurrency',type=int,default=4)
    ap.add_argument('--encoder',choices=['original','fast'],default='original')
    ap.add_argument('--timeout',type=int,default=30)
    ap.add_argument('--stochastic',action='store_true')
    ap.add_argument('--actor-only',action='store_true')
    ap.add_argument('--verify-encoder',action='store_true')
    ap.add_argument('--events',action='store_true')
    ap.add_argument('--run-id',default='direct-bc-'+time.strftime('%Y%m%d-%H%M%S'))
    args=ap.parse_args()
    folder=HERE/'runs'/args.run_id;folder.mkdir(parents=True,exist_ok=False)
    w.torch.set_num_threads(1)
    policy=w.PPO.load('/tmp/bc_100.zip',device=w.torch.device('cpu')).policy
    policy.set_training_mode(False)
    config={'concurrency':args.concurrency,'batcher':'original','shared':True,'deterministic':not args.stochastic,
            'encoder':args.encoder,'verify_encoder':args.verify_encoder,'timeout':args.timeout,
            'per_battle_rng':args.stochastic,'actor_only':args.actor_only,'events':args.events}
    result=asyncio.run_coroutine_threadsafe(evaluate(policy,config,workload(args.battles),folder),w.POKE_LOOP).result()
    (folder/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('results','traces')}),flush=True)


if __name__=='__main__':main()
