"""Paired live benchmark of request chunk sizes; no production matrix writes."""
import json
import sqlite3
import time
from pathlib import Path
import statistics
import argparse
import mask_lab_battle as B
import fast_battle_runner as F

ROOT = B.HERE/'runs/label-speed-20261013'
DB = B.HERE/'runs/all-labels-20261013/battles.sqlite'


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--workers',action='store_true')
    args=ap.parse_args()
    ROOT.mkdir(exist_ok=True,parents=True)
    con = sqlite3.connect('file:'+str(DB)+'?mode=ro',uri=True)
    rows = [r[0] for r in con.execute("SELECT DISTINCT team_id FROM label_job WHERE status='ready' AND row_id NOT IN (SELECT row_id FROM label_value) ORDER BY team_id LIMIT " + ('48' if args.workers else '36'))]
    cols = B.MDB.set_members(con,'top50')
    jobs = B.MDB.jobs_for(con,[(r,c,1) for r in rows for c in cols if r!=c])
    original = F.build_requests
    results = []
    mode='workers' if args.workers else 'chunks'
    with F.FastBattleRunner(ROOT/(mode+'-evaluator-'+str(time.time_ns())),8 if args.workers else 6,24) as runner:
        # Cold-start cost is excluded from the repeated steady-state comparison.
        runner._start()
        order=[6,8,8,6,8,6,6,8] if args.workers else [24,96,96,24,96,24,24,96]
        for n,setting in enumerate(order):
            size=96 if args.workers else setting
            runner.worker_count=setting if args.workers else 6
            def build(c,j,unused=24):
                return original(c,j,size)
            F.build_requests=build
            start=time.perf_counter()
            outcomes, unattributed, seconds=runner(con,jobs,seed=20261013+n//4)
            count=sum(sum(r[2:]) for r in outcomes)
            assert count==sum(len(j['schedule']) for j in jobs) and not unattributed
            row=dict(trial=n,block=n//4,chunk=size,setting=setting,workers=runner.worker_count,battles=count,seconds=time.perf_counter()-start,
                     evaluator_seconds=seconds,evidence=str(runner.folder),call=runner.calls)
            results.append(row)
            with open(ROOT/(mode+'-trials.jsonl'),'a') as f:f.write(json.dumps(row)+'\n')
            print(json.dumps(row),flush=True)
    F.build_requests=original
    ratios=[]
    for block in [0,1]:
        group=[r for r in results if r['block']==block]
        base,treatment=(6,8) if args.workers else (24,96)
        ratios.append(sum(r['seconds'] for r in group if r['setting']==base)/sum(r['seconds'] for r in group if r['setting']==treatment))
    summary=dict(block_speedups=ratios,geomean_speedup=statistics.geometric_mean(ratios),trials=results)
    (ROOT/(mode+'-summary.json')).write_text(json.dumps(summary,indent=2)+'\n')
    print('SUMMARY',json.dumps({k:v for k,v in summary.items() if k!='trials'}),flush=True)


if __name__=='__main__':main()
