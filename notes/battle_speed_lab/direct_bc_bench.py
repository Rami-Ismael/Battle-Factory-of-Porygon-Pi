"""Repeated trace-equivalent BC comparisons on the direct simulator adapter."""
import argparse
import asyncio
import json
import random
import time
from pathlib import Path
import direct_bc
from direct_bc import w
from run import HERE,Monitor,analyze,emit,snapshot_sources,workload,digest


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--blocks',type=int,default=5)
    ap.add_argument('--battles',type=int,default=48)
    ap.add_argument('--concurrency',type=int,default=16)
    ap.add_argument('--node-actor-model')
    ap.add_argument('--validate-only',action='store_true')
    ap.add_argument('--run-id',default='direct-bc-bench-'+time.strftime('%Y%m%d-%H%M%S'))
    args=ap.parse_args()
    folder=HERE/'runs'/args.run_id;folder.mkdir(parents=True,exist_ok=False)
    hashes=snapshot_sources(folder)
    (folder/'metadata.json').write_text(json.dumps({'args':vars(args),'source_hashes':hashes,
        'node_model_sha256':digest(Path(args.node_actor_model)) if args.node_actor_model else None,
        'note':'Independent per-battle and side policy RNGs; simulator RNGs independently seeded; normalized exact trace matching required. Both variants use the direct adapter.'},indent=2)+'\n')
    w.torch.set_num_threads(1)
    policy=w.PPO.load('/tmp/bc_100.zip',device='cpu').policy
    policy.set_training_mode(False)
    work=workload(args.battles)
    (folder/'workload.json').write_text(json.dumps(work,indent=2)+'\n')
    base={'batcher':'original','shared':True,'deterministic':False,'per_battle_rng':True,
          'encoder':'original','concurrency':args.concurrency,'timeout':180}
    configs={'baseline':base,'fast':{**base,'encoder':'fast','actor_only':True,'inference_mode':True}}
    alternative='fast'
    if args.node_actor_model:
        base=configs['fast']
        configs={'baseline':base,'node_actor':{**base,'node_actor_model':str(Path(args.node_actor_model).resolve())}}
        alternative='node_actor'
    rows=[];reference={}
    def execute(name,phase,block,position):
        target=folder/f'{phase}-{block}-{position}-{name}';target.mkdir()
        config={**configs[name],'simulator_seed':20261013+10000*block,'policy_seed':20261013+10000*block}
        with Monitor([__import__('os').getpid()]) as monitor:
            result=asyncio.run_coroutine_threadsafe(direct_bc.evaluate(policy,config,work,target),w.POKE_LOOP).result()
        key=(phase,block)
        if key not in reference:reference[key]=(result['results'],result['traces'])
        if reference[key]!=(result['results'],result['traces']):
            (target/'mismatch.json').write_text(json.dumps({'reference':reference[key],'result':result},indent=2)+'\n')
            raise RuntimeError('Seeded BC trace mismatch; timing not accepted')
        row={**result,'phase':phase,'variant':name,'treatment':alternative,'block':block,'position':position,
             'trace_equivalent':True,'telemetry':monitor.samples,'clock':monitor.clock}
        rows.append(row);emit(folder/'trials.jsonl',row)
        print(json.dumps({'phase':phase,'variant':name,'block':block,'bps':round(result['completed']/result['seconds'],2),
                          'trace_equivalent':True}),flush=True)
    execute('baseline','warmup',-1,0);execute(alternative,'warmup',-1,1)
    if args.validate_only:
        (folder/'validation.json').write_text(json.dumps({'passed':True,'battles_per_variant':args.battles,
            'comparison':alternative,'full_normalized_trace_agreement':True},indent=2)+'\n')
        return
    rng=random.Random(531)
    for block in range(args.blocks):
        sequence=['baseline',alternative,alternative,'baseline'] if rng.getrandbits(1) else [alternative,'baseline','baseline',alternative]
        for pos,name in enumerate(sequence):execute(name,'measure',block,pos)
        (folder/'summary.json').write_text(json.dumps(analyze(rows),indent=2)+'\n')
    print('RESULTS '+str(folder),flush=True)


if __name__=='__main__':main()
