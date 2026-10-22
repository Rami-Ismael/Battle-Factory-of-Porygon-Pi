"""Independent seeded BC outcome grid for allocation and paired-RNG experiments.

All candidates face the same disjoint opponent pool and seed schedule. The first
half of replicates are exploration data; the second half are held out. Outcomes
remain actual completed battles, never surrogate-generated pseudo-observations.
"""
import argparse
import asyncio
import json
import time
from pathlib import Path
from run import HERE,Monitor,digest,emit,snapshot_sources,workload
from direct_bc import evaluate,w


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--rows',nargs='+',type=int,default=[0,7,14,21,28,35,42,48])
    ap.add_argument('--opponents',type=int,default=8)
    ap.add_argument('--replicates',type=int,default=128)
    ap.add_argument('--run-id',default='quality-'+time.strftime('%Y%m%d-%H%M%S'))
    args=ap.parse_args()
    if args.replicates<4 or args.replicates%2 or args.opponents<1:
        ap.error('Use an even replicate count >=4 and positive opponent count')
    if len(set(args.rows))!=len(args.rows):ap.error('Candidate rows must be unique')
    candidates=[workload(1,i)['pairs'][0] for i in args.rows]
    candidate_ids={c['a_id'] for c in candidates}
    base=workload(48,0)
    available=[p for p in base['pairs'] if p['b_id'] not in candidate_ids]
    if len(available)<args.opponents:ap.error('Not enough disjoint opponents')
    # Spread the fixed pool across available opponents; equal-weight objective.
    opponents=[available[i*len(available)//args.opponents] for i in range(args.opponents)]
    folder=HERE/'runs'/args.run_id;folder.mkdir(parents=True,exist_ok=False)
    config={'concurrency':24,'batcher':'original','shared':True,'deterministic':False,
            'encoder':'fast','timeout':600,'per_battle_rng':True,'actor_only':True,
            'inference_mode':True,'simulator_seed':49000000,'policy_seed':71000000}
    metadata={'args':vars(args),'config':config,'source_hashes':snapshot_sources(folder),
              'policy_sha256':digest(Path('/tmp/bc_100.zip')),
              'candidate_ids':sorted(candidate_ids),'opponent_ids':[o['b_id'] for o in opponents],
              'candidate_species':{str(c['a_id']):c['a_species'] for c in candidates},
              'objective':'Equal-weight win score over this fixed opponent pool; draw=0.5.',
              'split':'replicate < replicates/2 is exploration; remainder is held-out evaluation',
              'limitation':'Finite held-out scores have uncertainty; they are not known true team strengths.'}
    (folder/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    w.torch.set_num_threads(1)
    policy=w.PPO.load('/tmp/bc_100.zip',device='cpu').policy;policy.set_training_mode(False)
    for row,candidate in zip(args.rows,candidates):
        pairs=[];identities={}
        for oi,opponent in enumerate(opponents):
            for replicate in range(args.replicates):
                identity=oi*args.replicates+replicate
                pairs.append({**opponent,'id':identity,'a':candidate['a'],'a_id':candidate['a_id'],
                              'a_species':candidate['a_species'],'a_base':candidate['a_base']})
                identities[str(identity)]={'candidate':candidate['a_id'],'opponent':opponent['b_id'],
                    'replicate':replicate,'split':'exploration' if replicate<args.replicates//2 else 'heldout'}
        part=folder/f'row-{row}';part.mkdir()
        (part/'workload.json').write_text(json.dumps({'format':base['format'],'pairs':pairs})+'\n')
        # This is a corpus-generation duration, not a paired throughput result.
        with Monitor([]) as monitor:
            result=asyncio.run_coroutine_threadsafe(evaluate(policy,config,{'format':base['format'],'pairs':pairs},part),w.POKE_LOOP).result()
        if result['completed']!=len(pairs):raise RuntimeError('Incomplete corpus row')
        for outcome in result['results']:
            emit(folder/'outcomes.jsonl',{**identities[outcome['id']],**outcome,
                'trace':result['traces'][outcome['id']],
                'score':.5 if outcome['won'] is None else float(outcome['won'])})
        emit(folder/'rows.jsonl',{'row':row,'candidate':candidate['a_id'],'completed':result['completed'],
            'seconds':result['seconds'],'clock':monitor.clock})
        print(json.dumps({'row':row,'battles':result['completed']}),flush=True)


if __name__=='__main__':main()
