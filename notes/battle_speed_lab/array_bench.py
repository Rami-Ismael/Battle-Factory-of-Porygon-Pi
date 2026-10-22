"""Differential validation and paired timing for a restricted array simulator.

Never mix these fixed-policy custom battles with ordinary VGC/BC throughput.
"""
import argparse
import json
import math
import os
import random
import subprocess
import time
import numpy as np
from array_kernel import simulate,scalar
from direct_bench import Direct
from run import HERE,SOURCE,Monitor,analyze,emit,snapshot_sources,workload


class Reference(Direct):
    def __init__(self,folder):
        self.log=(folder/'reference.log').open('w')
        self.proc=subprocess.Popen(['node',str(folder/'source/restricted.cjs'),str(SOURCE)],
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log,text=True,start_new_session=True)
        try:self.ready=self.read()
        except BaseException:self.close();raise


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--battles',type=int,default=256)
    ap.add_argument('--blocks',type=int,default=5)
    ap.add_argument('--validate-only',action='store_true')
    ap.add_argument('--run-id',default='array-'+time.strftime('%Y%m%d-%H%M%S'))
    args=ap.parse_args()
    if min(args.battles,args.blocks)<1:ap.error('Positive counts required')
    folder=HERE/'runs'/args.run_id;folder.mkdir(parents=True,exist_ok=False)
    hashes=snapshot_sources(folder)
    rng=random.Random(1018)
    jobs=[]
    for i in range(args.battles):
        speeds=[0,80,160,240];rng.shuffle(speeds)
        jobs.append({'id':i,'hpEVs':[rng.randrange(64)*4 for _ in range(4)],'speedEVs':speeds})
    (folder/'workload.json').write_text(json.dumps(jobs,indent=2)+'\n')
    (folder/'metadata.json').write_text(json.dumps({'args':vars(args),'source_hashes':hashes,
        'scope':'Restricted custom doubles, fixed attack policy; not VGC/BC throughput.',
        'fidelity':'Every damage target, post-hit HP and turn plus final HP/winner must match Showdown.',
        'timing':'Array dispatch excludes reference stat preparation; preparation time reported separately.'},indent=2)+'\n')
    reference=Reference(folder);rows=[]
    try:
        preparation=reference.run({'jobs':jobs})
        initial_hp=np.array([[p['hp'] for p in r['initial']] for r in preparation['results']])
        speeds=np.array([[p['speed'] for p in r['initial']] for r in preparation['results']])
        checked=simulate(initial_hp,speeds,record_events=True)
        scalar_checked=scalar(initial_hp,speeds,record_events=True)
        if scalar_checked['events']!=checked['events']:raise AssertionError('Scalar/vectorized events differ')
        for i,result in enumerate(preparation['results']):
            if (checked['events'][i]!=result['events'] or checked['hp'][i].tolist()!=result['hp']
                or int(checked['turns'][i])!=result['turns'] or int(checked['winner'][i])!=result['winner']):
                (folder/'mismatch.json').write_text(json.dumps({'reference':result,'array_events':checked['events'][i],
                    'array_hp':checked['hp'][i].tolist(),'array_turns':int(checked['turns'][i])},indent=2)+'\n')
                raise AssertionError('Restricted kernel differs from Showdown')
        # Each real candidate has six Pokemon, outside the two-per-side kernel.
        # This is a hard fallback condition, not a claim to support their moves.
        real=workload(48,0)
        unique={real['pairs'][0]['a_id']:real['pairs'][0]['a_species']}
        unique.update({p['b_id']:p['b_species'] for p in real['pairs']})
        coverage={'real_teams':len(unique),'eligible':sum(len(s.split(','))==2 for s in unique.values()),
                  'first_rejection':'Kernel requires exactly two Pokemon per side; real team previews have six.'}
        (folder/'validation.json').write_text(json.dumps({'passed':True,'battles':len(jobs),
            'damage_events':sum(len(r['events']) for r in preparation['results']),
            'preparation_seconds':preparation['seconds'],'real_pool_coverage':coverage},indent=2)+'\n')
        if args.validate_only:
            print(json.dumps({'passed':True,'battles':len(jobs),'real_pool_coverage':coverage}),flush=True)
            return
        repetitions={}
        for method in ['showdown','scalar','array']:
            start=time.perf_counter()
            if method=='showdown':reference.run({'jobs':jobs})
            else:(simulate if method=='array' else scalar)(initial_hp,speeds)
            repetitions[method]=max(1,min(10000,math.ceil(.5/max(time.perf_counter()-start,1e-6))))
        (folder/'timing-calibration.json').write_text(json.dumps({'repetitions':repetitions,
            'target_seconds':.5,'note':'Fixed repetitions per method after warm calibration; repeated inputs are throughput work, not new independent evidence.'},indent=2)+'\n')
        def execute(method,phase,block,position,variant,treatment):
            with Monitor([os.getpid(),reference.proc.pid]) as monitor:
                start=time.perf_counter()
                if method!='showdown':
                    for _ in range(repetitions[method]):result=(simulate if method=='array' else scalar)(initial_hp,speeds)
                    turns=int(result['turns'].sum())*repetitions[method]
                    if not np.array_equal(result['hp'],checked['hp']) or not np.array_equal(result['winner'],checked['winner']):
                        raise AssertionError('Array result changed during timing')
                else:
                    turns=0
                    for _ in range(repetitions[method]):
                        result=reference.run({'jobs':jobs})
                        if result['results']!=preparation['results']:raise AssertionError('Reference changed during timing')
                        turns+=sum(r['turns'] for r in result['results'])
                seconds=time.perf_counter()-start
            row={'phase':phase,'block':block,'position':position,'variant':variant,
                'treatment':treatment,'seconds':seconds,'completed':len(jobs)*repetitions[method],'turns':turns,
                'repetitions':repetitions[method],
                'clock':monitor.clock,'telemetry':monitor.samples,
                'trace_equivalent':False,'restricted_damage_events_verified':True}
            rows.append(row);emit(folder/'trials.jsonl',row)
            print(json.dumps({k:row[k] for k in ['phase','block','variant','seconds']}),flush=True)
        for method in ['showdown','scalar','array']:execute(method,'warmup',-1,0,method,'warmup')
        for block in range(args.blocks):
            comparisons=[('specialized','showdown','scalar'),('array','showdown','array'),('vectorization','scalar','array')]
            rng.shuffle(comparisons)
            for treatment,baseline,alternative in comparisons:
                sequence=[False,True,True,False] if rng.getrandbits(1) else [True,False,False,True]
                for pos,fast in enumerate(sequence):
                    execute(alternative if fast else baseline,'measure',block,pos,treatment if fast else 'baseline',treatment)
            summary=analyze(rows)
            for result in summary.values():result['status']='Restricted custom-battle comparison; damage events verified, no full protocol replay equivalence claim'
            (folder/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    finally:reference.close()


if __name__=='__main__':main()
