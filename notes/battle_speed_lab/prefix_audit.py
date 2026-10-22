"""Check whether a legal single-nature mutation preserves an exact checkpoint."""
import argparse
import json
import re
import time
from run import HERE,snapshot_sources,workload
from snapshot_bench import Snapshot


def mutate_nature(paste):
    match=re.search(r'^([A-Za-z]+) Nature$',paste,re.MULTILINE)
    if not match:raise ValueError('Expected an explicit nature in candidate paste')
    replacement='Modest' if match.group(1)=='Adamant' else 'Adamant'
    return paste[:match.start(1)]+replacement+paste[match.end(1):]


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run-id',default='prefix-audit-'+time.strftime('%Y%m%d-%H%M%S'))
    args=ap.parse_args();folder=HERE/'runs'/args.run_id;folder.mkdir(parents=True,exist_ok=False)
    hashes=snapshot_sources(folder);client=Snapshot(folder);checks=[]
    try:
        for row in [0,7,14,21,28,35,42,48]:
            work=workload(8,row)
            originals=[{**job,'seed':902000+job['id']} for job in work['pairs']]
            mutants=[{**job,'a':mutate_nature(job['a'])} for job in originals]
            request={'format':work['format'],'mode':'build','prefixTurn':3}
            a=client.run({**request,'jobs':originals})['results']
            repeat=client.run({**request,'jobs':originals})['results']
            b=client.run({**request,'jobs':mutants})['results']
            for original,same,mutant in zip(a,repeat,b):
                if not original['stateHash'] or original['stateHash']!=same['stateHash']:
                    raise AssertionError('Identical seeded queries did not reproduce a checkpoint')
                checks.append({'row':row,'battle':original['id'],
                    'exact_state_equal':original['stateHash']==mutant['stateHash'],
                    'visible_board_equal':original['boardHash']==mutant['boardHash'],
                    'full_outcome_trace_equal':original['trace']==mutant['trace']})
        report={'source_hashes':hashes,'checks':checks,'mutated_queries':len(checks),
            'exact_state_matches':sum(c['exact_state_equal'] for c in checks),
            'visible_board_matches':sum(c['visible_board_equal'] for c in checks),
            'limitation':'Only nature mutations and turn-three random-policy checkpoints tested. Exact state includes bench, queues, history and RNG. Board similarity never authorizes reuse.'}
        (folder/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({k:v for k,v in report.items() if k not in ['checks','source_hashes']}))
    finally:client.close()


if __name__=='__main__':main()
