"""Paired benchmark of the real database pipeline with isolated uncached fixtures."""
import argparse
import json
import math
import os
import random
import shutil
import sqlite3
import statistics
import time
from pathlib import Path
from unittest.mock import patch
import mask_lab_battle as pipeline
from fast_battle_runner import lab
from battle_integration_fixture import create_fixture


def analyze(trials):
    ratios=[];accepted=[];invalid=[]
    for block in sorted({r['block'] for r in trials}):
        group=[r for r in trials if r['block']==block]
        if any(r['clock']['pause_or_clock_jump'] for r in group):
            invalid.append(block);continue
        baseline=[r for r in group if r['variant']=='baseline']
        fast=[r for r in group if r['variant']=='fast']
        if len(baseline)!=2 or len(fast)!=2 or {r['position'] for r in group}!={0,1,2,3}:continue
        rate=lambda rows:sum(r['completed'] for r in rows)/sum(r['seconds'] for r in rows)
        ratios.append(rate(fast)/rate(baseline));accepted.append(block)
    if not ratios:return {'accepted_blocks':[], 'invalid_clock_blocks':invalid}
    rng=random.Random(991)
    bootstrap=sorted(math.exp(statistics.mean(math.log(rng.choice(ratios)) for _ in ratios)) for _ in range(5000))
    return {'accepted_blocks':accepted,'invalid_clock_blocks':invalid,'block_speedups':ratios,
        'geometric_mean_speedup':math.exp(statistics.mean(map(math.log,ratios))),
        'bootstrap95':[bootstrap[125],bootstrap[4874]] if len(ratios)>=3 else None,
        'aggregate_battles_per_second':{name:sum(r['completed'] for r in trials if r['variant']==name and r['block'] in accepted)/
            sum(r['seconds'] for r in trials if r['variant']==name and r['block'] in accepted) for name in ['baseline','fast']},
        'limitations':['Stochastic live trajectories are not trace-paired.',
            'Legacy output does not expose turns; no turns-normalized comparison is claimed.',
            'The complete configuration changes from three legacy workers at concurrency50 to six fast workers at concurrency24.',
            'Exploratory block-bootstrap interval under observed background load; no guarantee for other workloads.']}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--blocks',type=int,default=5)
    parser.add_argument('--candidates',type=int,default=24)
    parser.add_argument('--opponents',type=int,default=49)
    parser.add_argument('--batch',type=int,default=12)
    parser.add_argument('--run-id',default='fast-integration-'+time.strftime('%Y%m%d-%H%M%S'))
    args=parser.parse_args()
    if min(args.blocks,args.candidates,args.opponents,args.batch)<1:parser.error('Counts must be positive')
    folder=Path(__file__).resolve().parent/'runs'/args.run_id
    folder.mkdir(parents=True,exist_ok=False)
    rows,cols=create_fixture(folder/'fixture',args.candidates,args.opponents)
    runtime=folder/'legacy-private-runtime';lab.prepare_runtime(runtime)
    source=folder/'source';source.mkdir()
    hashes={}
    for file in [Path(__file__),Path(pipeline.__file__),Path(__file__).with_name('fast_battle_runner.py'),
                 Path(__file__).with_name('battle_integration_fixture.py'),Path(pipeline.MDB.__file__),
                 Path('/tmp/vgc-pilot/src/pair_shard.py')]:
        shutil.copy2(file,source/file.name);hashes[file.name]=lab.digest(file)
    (folder/'metadata.json').write_text(json.dumps({'args':vars(args),'rows':rows,'columns':cols,
        'source_hashes':hashes,'fixture_sha256':lab.digest(folder/'fixture/matrix.sqlite'),
        'baseline':{'backend':'legacy','workers':3,'concurrency':50},
        'treatment':{'backend':'fast','workers':6,'concurrency':24},
        'note':'Fresh per-trial database; elapsed time includes evaluator provenance, startup, missing-cell planning, two saved batches by default, scores, and cleanup. Candidate generation/validation and copying the fixture database are excluded. Legacy servers use the same isolated compiled runtime configuration as the fast path; legacy runner code is unchanged. Live trajectories are stochastic.'},indent=2)+'\n')
    trials=[];rng=random.Random(20261013)
    for block in range(args.blocks):
        sequence=['baseline','fast','fast','baseline']
        if rng.getrandbits(1):sequence=['fast','baseline','baseline','fast']
        for position,variant in enumerate(sequence):
            trial_folder=folder/f'trial-{len(trials):03d}';trial_folder.mkdir()
            src=sqlite3.connect(folder/'fixture/matrix.sqlite');con=sqlite3.connect(trial_folder/'matrix.sqlite')
            src.backup(con);src.close();con.close()
            con=pipeline.MDB.connect(trial_folder/'matrix.sqlite')
            servers=[]
            def start_private(count):
                for _ in range(count):servers.append(lab.Server(runtime,trial_folder))
                return [s.proc for s in servers],[s.port for s in servers]
            options=pipeline.parse_args(['--backend','fast' if variant=='fast' else 'legacy',
                '--workers','6' if variant=='fast' else '3','--concurrency','24' if variant=='fast' else '50',
                '--batch',str(args.batch),'--evidence-dir',str(trial_folder/'evaluator')])
            try:
                with lab.Monitor([os.getpid()]) as monitor:
                    started=time.perf_counter()
                    with patch.object(pipeline,'start_servers',side_effect=start_private),patch.object(pipeline.MDB,'PASTES',trial_folder/'pastes'):
                        policy,evidence=pipeline.evaluate(con,rows,cols,options)
                    scores=[pipeline.MDB.score(con,policy,row) for row in rows]
                    completed=con.execute('SELECT COALESCE(SUM(battles),0) FROM cell WHERE policy_id=?',(policy,)).fetchone()[0]
                    if completed!=len(rows)*len(cols):raise RuntimeError('Incomplete pipeline workload')
                    if any(s['coverage']!=len(cols) for s in scores):raise RuntimeError('Incomplete score coverage')
                    batches=[dict(zip(['seconds','battles','unattributed'],r)) for r in con.execute('SELECT seconds,battles,unattributed FROM batch ORDER BY batch_id')]
                    if any(b['unattributed'] for b in batches):raise RuntimeError('Unattributed pipeline battle')
                    for server in servers:server.close()
                    elapsed=time.perf_counter()-started
                record={'phase':'measure','variant':variant,'treatment':'fast','position':position,'block':block,
                    'seconds':elapsed,'completed':completed,'batches':batches,'scores':scores,
                    'clock':monitor.clock,'telemetry':monitor.samples,'policy_id':policy,'evidence':evidence}
                trials.append(record);lab.emit(folder/'trials.jsonl',record)
                summary=analyze(trials)
                (folder/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
                print(json.dumps({k:record[k] for k in ['block','position','variant','seconds','completed']}),flush=True)
            finally:
                for server in servers:
                    if not server.log.closed:server.close()
                con.close()
    (folder/'completion.json').write_text(json.dumps({'status':'complete','trials':len(trials),
        'battles':sum(t['completed'] for t in trials)},indent=2)+'\n')
    print('RESULTS '+str(folder),flush=True)


if __name__=='__main__':main()
