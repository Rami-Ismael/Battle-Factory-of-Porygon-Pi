"""Live equal-wall-clock strategy comparison with separate held-out scoring.

All completed exploration battles are real BC battles. Results arriving after
the deadline are recorded but cannot influence that trial's selected candidate.
"""
import argparse
import json
import math
import random
import time
from pathlib import Path
import numpy as np
from allocation import load_grid
from direct_bc import w
from run import HERE,Server,Worker,Monitor,emit,prepare_runtime,snapshot_sources,digest


def base_key(names):
    normalized=[]
    for name in names:
        entry=w.dex_entry(name)
        base=entry.get('baseSpecies',name) if entry else name
        normalized.append(''.join(c for c in base.lower() if c.isalnum()))
    return ','.join(sorted(normalized))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('corpus',type=Path)
    ap.add_argument('--seconds',type=float,default=15.)
    ap.add_argument('--blocks',type=int,default=5)
    ap.add_argument('--methods',nargs='+',choices=['race','stratified','surrogate'],default=['race','stratified','surrogate'])
    ap.add_argument('--run-id',default='live-allocation-'+time.strftime('%Y%m%d-%H%M%S'))
    args=ap.parse_args()
    if args.seconds<=0 or args.blocks<1:ap.error('Positive budget and block count required')
    grid,candidates,opponents=load_grid(args.corpus)
    heldout=np.split(grid,2,axis=2)[1].mean(axis=(1,2))
    profiles={};opponent_profiles={};formats=set()
    for file in sorted(args.corpus.glob('row-*/workload.json')):
        work=json.loads(file.read_text());formats.add(work['format'])
        first=work['pairs'][0];profiles[first['a_id']]=first
        opponent_profiles.update({p['b_id']:p for p in work['pairs']})
    if len(formats)!=1:raise ValueError('Mixed formats in corpus')
    format_id=formats.pop()
    opposing_keys={opponent_profiles[o]['b_base']:j for j,o in enumerate(opponents)}
    if len(opposing_keys)!=len(opponents):raise ValueError('Opponent base-species keys are ambiguous')
    sets=[set(profiles[c]['a_species'].split(',')) for c in candidates]
    vocabulary=sorted(set.union(*sets))
    features=np.array([[float(s in team) for s in vocabulary] for team in sets])
    x=np.column_stack([np.ones(len(candidates)),features]);penalty=np.eye(x.shape[1])*.5;penalty[0,0]=1e-8
    folder=HERE/'runs'/args.run_id;folder.mkdir(parents=True,exist_ok=False)
    hashes=snapshot_sources(folder)
    (folder/'metadata.json').write_text(json.dumps({'args':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        'source_hashes':hashes,'corpus_sha256':digest(args.corpus/'outcomes.jsonl'),
        'candidate_ids':candidates,'opponent_ids':opponents,'heldout_scores':heldout.tolist(),
        'note':'Whole exploration job latency and strategy computation count toward the budget; model/server startup and prior warmup excluded. After-deadline results never influence selection.'},indent=2)+'\n')
    runtime=HERE/'runs/private-runtime';prepare_runtime(runtime)
    config={'batcher':'original','threads':1,'concurrency':24,'shared':True,'encoder':'fast',
            'actor_only':True,'inference_mode':True,'per_battle_rng':True}
    server=worker=None;rows=[];trial_serial=0
    try:
        server=Server(runtime,folder);worker=Worker(config,folder,'live')
        def batch(candidate_index,opponent_indices,seed):
            candidate=profiles[candidates[candidate_index]]
            pairs=[]
            for identity,j in enumerate(opponent_indices):
                opponent=opponent_profiles[opponents[j]]
                pairs.append({**opponent,'id':identity,'a':candidate['a'],'a_id':candidate['a_id'],
                              'a_base':candidate['a_base'],'a_species':candidate['a_species']})
            return worker.trial({'format':format_id,'pairs':pairs,'seed':seed,
                'config':{**config,'policy_seed':seed},'port':server.port,'timeout':180})
        for i in range(len(candidates)):batch(i,list(range(len(opponents)))*2,770000+i)
        def execute(method,treatment,block,position):
            nonlocal trial_serial
            serial=trial_serial;trial_serial+=1
            rng=np.random.default_rng(850000+serial)
            c,o=len(candidates),len(opponents)
            counts=np.zeros((c,o),dtype=int);sums=np.zeros((c,o));squares=np.zeros((c,o))
            physical_battles=0;late_battles=0;calls=0;active=list(range(c));last_selected=None
            with Monitor([server.proc.pid,worker.proc.pid]) as monitor:
                start=time.perf_counter();deadline=start+args.seconds
                def get(i,js):
                    nonlocal physical_battles,late_battles,calls,last_selected,counts,sums,squares
                    result=batch(i,js,880000+serial*10000+calls);calls+=1
                    physical_battles+=result['completed']
                    proposed_counts=counts.copy();proposed_sums=sums.copy();proposed_squares=squares.copy();entries=[]
                    for battle in result['battles']:
                        key=base_key(battle['opponent'])
                        if key not in opposing_keys:raise RuntimeError('Unexpected opponent identity')
                        j=opposing_keys[key];score=.5 if battle['won'] is None else float(battle['won'])
                        proposed_counts[i,j]+=1;proposed_sums[i,j]+=score;proposed_squares[i,j]+=score*score
                        entries.append({'trial':serial,'candidate':candidates[i],
                            'opponent':opponents[j],'tag':battle['tag'],'turns':battle['turns'],
                            'score':score})
                    proposed_selected=None
                    if (proposed_counts>=2).all():
                        scores=(proposed_sums/proposed_counts).mean(1)
                        choices=active if method=='race' else list(range(c))
                        proposed_selected=max(choices,key=lambda i:(scores[i],-i))
                    eligible=time.perf_counter()<=deadline
                    if eligible:
                        counts,sums,squares=proposed_counts,proposed_sums,proposed_squares
                        last_selected=proposed_selected
                    else:late_battles+=result['completed']
                    for entry in entries:emit(folder/'exploration.jsonl',{**entry,'eligible_before_deadline':eligible})
                    return eligible
                for i in range(c):
                    if time.perf_counter()>=deadline:break
                    if not get(i,list(range(o))*2):break
                pilot_complete=bool((counts>=2).all())
                if pilot_complete:
                    means=sums/counts;variance=np.maximum(squares/counts-means**2,.05)
                    if method=='surrogate':
                        predictions=x@np.linalg.solve(x.T@x+penalty,x.T@means.mean(1))
                        preferred=list(np.argsort(-predictions)[:math.ceil(c/2)])
                    next_elimination=time.perf_counter()+max(0,(deadline-time.perf_counter())/3)
                    while time.perf_counter()<deadline:
                        if method=='race' and time.perf_counter()>=next_elimination and len(active)>1:
                            scores=(sums/counts).mean(1)
                            active=sorted(active,key=lambda i:(-scores[i],i))[:math.ceil(len(active)/2)]
                            next_elimination=time.perf_counter()+(deadline-time.perf_counter())/2
                        available=active if method=='race' else list(range(c))
                        if method=='surrogate' and rng.random()<.2:i=int(rng.choice(available))
                        elif method=='surrogate':i=min(preferred,key=lambda i:(counts[i].sum(),i))
                        else:i=min(available,key=lambda i:(counts[i].sum(),i))
                        temporary=counts[i].copy();js=[]
                        for _ in range(16):
                            if method=='stratified':j=int(np.argmax(variance[i]/(temporary*(temporary+1))))
                            else:j=int(np.argmin(temporary))
                            temporary[j]+=1;js.append(j)
                        if not get(i,js):break
                elapsed=time.perf_counter()-start
            selected=last_selected
            row={'trial':serial,'block':block,'position':position,'treatment':treatment,'method':method,
                'budget_seconds':args.seconds,'actual_seconds_including_drain':elapsed,
                'eligible_battles':int(counts.sum()),'physical_battles':physical_battles,'late_battles':late_battles,
                'pilot_complete':pilot_complete,'selected_candidate':candidates[selected] if selected is not None else None,
                'heldout_score':float(heldout[selected]) if selected is not None else None,
                'counts':counts.tolist(),'clock':monitor.clock,'telemetry':monitor.samples}
            rows.append(row);emit(folder/'trials.jsonl',row)
            print(json.dumps({k:row[k] for k in ['block','method','eligible_battles','pilot_complete','heldout_score']}),flush=True)
        rng=random.Random(1652)
        for block in range(args.blocks):
            treatments=args.methods[:];rng.shuffle(treatments)
            for treatment in treatments:
                sequence=['uniform',treatment,treatment,'uniform'] if rng.getrandbits(1) else [treatment,'uniform','uniform',treatment]
                for position,method in enumerate(sequence):execute(method,treatment,block,position)
        summary={}
        for treatment in args.methods:
            subset=[r for r in rows if r['treatment']==treatment]
            invalid={r['block'] for r in subset if r['clock']['pause_or_clock_jump']}
            accepted=[r for r in subset if r['block'] not in invalid]
            details={}
            for method in ['uniform',treatment]:
                chosen=[r for r in accepted if r['method']==method]
                scores=[r['heldout_score'] for r in chosen if r['pilot_complete']]
                details[method]={'trials':len(chosen),'pilot_failures':sum(not r['pilot_complete'] for r in chosen),
                    'mean_heldout_score_when_selectable':float(np.mean(scores)) if scores else None,
                    'mean_eligible_battles':float(np.mean([r['eligible_battles'] for r in chosen])) if chosen else None}
            summary[treatment]={'invalid_clock_blocks':sorted(invalid),'methods':details}
        (folder/'summary.json').write_text(json.dumps({'comparisons':summary,
            'limitations':['Fixed finite held-out labels have uncertainty; no unique best team was certified.',
                'Late results are discarded for selection; their compute cost and drain time remain recorded.',
                'Pilot failures remain explicit and are never silently dropped from trial counts.',
                'Small repeated screen under observed background load; no general quality improvement is presumed.']},indent=2)+'\n')
    finally:
        if worker:worker.close()
        if server:server.close()


if __name__=='__main__':main()
