"""Verify a completed integration benchmark and derive paired stage timings."""
import argparse
import hashlib
import json
import sqlite3
import statistics
from collections import defaultdict
from pathlib import Path
from benchmark_fast_pipeline import analyze
from fast_battle_runner import attribute


def read_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def audit(folder):
    metadata=json.loads((folder/'metadata.json').read_text())
    completion=json.loads((folder/'completion.json').read_text())
    trials=read_lines(folder/'trials.jsonl')
    expected_trials=metadata['args']['blocks']*4
    assert completion['status']=='complete' and len(trials)==expected_trials
    assert completion['trials']==len(trials)
    assert completion['battles']==sum(r['completed'] for r in trials)
    for name,expected in metadata['source_hashes'].items():
        assert hashlib.sha256((folder/'source'/name).read_bytes()).hexdigest()==expected,name
    assert hashlib.sha256((folder/'fixture/matrix.sqlite').read_bytes()).hexdigest()==metadata['fixture_sha256']
    identities=set();fast_jobs=0
    for i,trial in enumerate(trials):
        current=folder/f'trial-{i:03d}'
        with sqlite3.connect('file:'+str(current/'matrix.sqlite')+'?mode=ro',uri=True) as db:
            cells={(lo,hi):(w,l,t,n) for lo,hi,w,l,t,n in db.execute(
                'SELECT lo,hi,lo_wins,hi_wins,ties,battles FROM cell WHERE policy_id=?',(trial['policy_id'],))}
            assert sum(v[3] for v in cells.values())==trial['completed']
            assert len(cells)==len(metadata['rows'])*len(metadata['columns'])
            assert all(sum(v[:3])==v[3]==1 for v in cells.values())
            assert sum(b['battles'] for b in trial['batches'])==trial['completed']
        if trial['variant']!='fast':continue
        evidence=current/'evaluator'
        provenance=json.loads((evidence/'metadata.json').read_text())
        identities.add(provenance['identity'])
        assert provenance['workers']==6 and provenance['configuration']['concurrency']==24
        for name,expected in provenance['source_hashes'].items():
            assert hashlib.sha256((evidence/'source'/name).read_bytes()).hexdigest()==expected,name
        assert hashlib.sha256((evidence/'source/fast_battle_runner.py').read_bytes()).hexdigest()==provenance['implementation']['fast_battle_runner.py']
        requests={(call['call'],job['job_id']):job for call in read_lines(evidence/'requests.jsonl') for job in call['jobs']}
        seen=set();counts=defaultdict(lambda:[0,0,0]);loads=defaultdict(int)
        for result in read_lines(evidence/'results.jsonl'):
            key=(result['call'],result['job_id']);assert key not in seen;seen.add(key)
            for row,col,w,l,t in attribute(requests[key],result):
                if row>col:row,col,w,l=col,row,l,w
                totals=counts[row,col]
                for j,value in enumerate([w,l,t]):totals[j]+=value
            loads[result['worker']]+=result['completed']
        assert seen==set(requests)
        assert {k:tuple(v) for k,v in counts.items()}=={k:v[:3] for k,v in cells.items()}
        assert set(loads)==set(range(6))
        fast_jobs+=len(seen)
    assert len(identities)==1,'Implementation/configuration changed between fast trials'
    later=[]
    for trial in trials:
        if len(trial['batches'])>1:
            later.append({**trial,'seconds':sum(b['seconds'] for b in trial['batches'][1:]),
                          'completed':sum(b['battles'] for b in trial['batches'][1:])})
    external=[s['external_cpu_percent'] for trial in trials for s in trial['telemetry'] if 'external_cpu_percent' in s]
    return {'audit':'passed','trials':len(trials),'measured_battles':sum(r['completed'] for r in trials),
        'verified_fast_jobs':fast_jobs,'whole_battle_stage':analyze(trials),
        'later_saved_batches':analyze(later),
        'external_cpu_percent':{'min':min(external),'median':statistics.median(external),'max':max(external)},
        'note':'Later-batch timing excludes the first saved batch; legacy model reloads still count. It is a persistent-pipeline comparison, not isolated engine speed. Database wins/losses/ties were reconciled with every fast battle result.'}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('folders',type=Path,nargs='+')
    parser.add_argument('--output',type=Path);parser.add_argument('--min-blocks',type=int,default=5)
    args=parser.parse_args()
    audits=[];trials=[];offset=0;origins={};hashes=None;workload=None
    for path in args.folders:
        folder=path.resolve();audits.append({'run':str(folder),**audit(folder)})
        metadata=json.loads((folder/'metadata.json').read_text())
        current_hashes=metadata['source_hashes']
        if hashes is not None:assert current_hashes==hashes,'Source changed between benchmark runs'
        hashes=current_hashes
        with sqlite3.connect('file:'+str(folder/'fixture/matrix.sqlite')+'?mode=ro',uri=True) as db:
            current_workload=[metadata['rows'],metadata['columns'],metadata['args']['batch'],
                              metadata['baseline'],metadata['treatment'],
                              db.execute('SELECT team_id,format_id,canon_hash,paste,base_key,legal FROM team ORDER BY team_id').fetchall()]
        if workload is not None:assert current_workload==workload,'Workload changed between benchmark runs'
        workload=current_workload
        rows=read_lines(folder/'trials.jsonl')
        for trial in rows:
            block=trial['block']+offset
            origins[block]={'run':str(folder),'original_block':trial['block']}
            trials.append({**trial,'block':block})
        offset+=1+max(r['block'] for r in rows)
    later=[{**r,'seconds':sum(b['seconds'] for b in r['batches'][1:]),
            'completed':sum(b['battles'] for b in r['batches'][1:])} for r in trials if len(r['batches'])>1]
    whole=analyze(trials)
    assert len(whole['accepted_blocks'])>=args.min_blocks,'Not enough accepted complete blocks'
    result={'audit':'passed','runs':audits,'block_origins':origins,'whole_battle_stage':whole,
        'later_saved_batches':analyze(later),'attempted_battles':sum(r['completed'] for r in trials),
        'accepted_battles':sum(r['completed'] for r in trials if r['block'] in whole['accepted_blocks']),
        'note':'Replacement blocks are added only for clock-invalid blocks; entire invalid blocks remain excluded.'}
    destination=args.output or args.folders[0]/'analysis.json'
    destination.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
