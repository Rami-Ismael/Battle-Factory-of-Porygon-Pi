"""Validate the synchronous checkpoint driver against the original stream driver.

This is correctness evidence only; validation timings are not benchmark results.
"""
import argparse
import json
import time
from direct_bench import Direct, signatures
from snapshot_bench import Snapshot
from run import HERE, snapshot_sources, workload


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--battles',type=int,default=12)
    ap.add_argument('--run-id',default='snapshot-validation-'+time.strftime('%Y%m%d-%H%M%S'))
    args=ap.parse_args()
    folder=HERE/'runs'/args.run_id
    folder.mkdir(parents=True,exist_ok=False)
    hashes=snapshot_sources(folder)
    clients=[]
    try:
        original=Direct(folder,'reference');clients.append(original)
        snapshot=Snapshot(folder);clients.append(snapshot)
        checks=[]
        for row in [0,21,48]:
            work=workload(args.battles,row_index=row)
            jobs=[{**j,'seed':720000+row*1000+j['id']} for j in work['pairs']]
            request={'format':work['format'],'jobs':jobs,'concurrency':1,'cacheTeams':True}
            expected=signatures(original.run(request))
            for turn in [1,3,8]:
                for mode in ['full','build','resume']:
                    result=snapshot.run({**request,'mode':mode,'prefixTurn':turn})
                    actual=signatures(result)
                    if actual!=expected:
                        (folder/'mismatch.json').write_text(json.dumps({'row':row,'turn':turn,'mode':mode,
                            'expected':expected,'actual':actual},indent=2)+'\n')
                        raise AssertionError('Checkpoint driver differs from original stream driver')
                    checks.append({'row':row,'turn':turn,'mode':mode,'battles':len(actual),
                                   'hits':result['hits'],'misses':result['misses']})
        report={'status':'pass','source_hashes':hashes,'checks':checks,
                'note':'Full normalized traces, outcomes and turn counts agree; only wall-clock timestamps excluded.'}
        (folder/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({'status':'pass','checks':len(checks),'battles_checked':sum(c['battles'] for c in checks)}))
    finally:
        for client in reversed(clients):client.close()


if __name__=='__main__':main()
