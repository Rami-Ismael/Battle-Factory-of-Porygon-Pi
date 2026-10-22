"""Run independent experimental stages serially, retaining failed-stage evidence."""
import argparse
import json
import subprocess
import sys
import time
from run import HERE,emit,snapshot_sources


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--run-id',default='campaign-'+time.strftime('%Y%m%d-%H%M%S'))
    ap.add_argument('--stages',nargs='+',choices=['direct-bc','direct-engine','cold-engine','cold-worker','more-settings','concurrency','scaling','snapshot-validate','snapshot-bench','remote-validate','remote-bench','cache-isolation','quality-corpus','capture-corpus','queue-validate','queue-bench'])
    args=ap.parse_args()
    dependencies={'snapshot-bench':'snapshot-validate','remote-bench':'remote-validate','queue-bench':'queue-validate'}
    for stage,dependency in dependencies.items():
        if args.stages and stage in args.stages and dependency not in args.stages:
            ap.error(f'{stage} requires {dependency} in the same campaign')
    folder=HERE/'runs'/args.run_id;folder.mkdir(parents=True,exist_ok=False)
    snapshot_sources(folder)
    stages=[
      ('direct-bc',['direct_bc_bench.py','--blocks','5','--battles','48']),
      ('direct-engine',['direct_bench.py','--blocks','5','--battles','96']),
      ('cold-engine',['direct_bench.py','--blocks','5','--battles','12','--variants','cold']),
      ('cold-worker',['run.py','--blocks','5','--battles','48','--variants','cold_workers']),
      ('more-settings',['run.py','--blocks','5','--battles','96','--variants','threads2','deadline0','deadline2','logit_cache','inference_mode']),
      ('concurrency',['run.py','--blocks','5','--battles','96','--variants','concurrency8','concurrency50']),
      ('scaling',['scaling.py','--blocks','5','--workers','2','4','--jobs','8','--battles-per-job','24']),
      ('snapshot-validate',['validate_snapshot.py','--battles','12']),
      ('snapshot-bench',['snapshot_bench.py','--blocks','5','--battles','48']),
      ('remote-validate',['validate_remote.py']),
      ('remote-bench',['scaling.py','--shared-inference','--blocks','5','--workers','2','4','--jobs','8','--battles-per-job','24']),
      ('cache-isolation',['run.py','--baseline','fast_stack','--variants','fast_cache','--blocks','5','--battles','96']),
      ('quality-corpus',['quality_corpus.py']),
      ('capture-corpus',['collect.py']),
      ('queue-validate',['test_leased_queue.py']),
      ('queue-bench',['queue_bench.py']),
    ]
    if args.stages:stages=[s for s in stages if s[0] in args.stages]
    manifest=[]
    for name,arguments in stages:
        if name in dependencies and not any(s['stage']==dependencies[name] and s.get('exit_code')==0 for s in manifest):
            item={'stage':name,'status':'skipped','reason':f'Independent correctness gate {dependencies[name]} failed'}
            manifest.append(item);emit(folder/'events.jsonl',item)
            (folder/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
            print(json.dumps(item),flush=True)
            continue
        run_id=args.run_id+'-'+name
        command=[sys.executable,str(HERE/arguments[0]),*arguments[1:]]
        if name!='queue-validate':command+=['--run-id',run_id]
        start=time.time()
        print(json.dumps({'stage':name,'status':'starting','run_id':run_id}),flush=True)
        with (folder/(name+'.log')).open('w') as log:
            child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
            emit(folder/'events.jsonl',{'stage':name,'status':'running','pid':child.pid,'time':start,'command':command})
            code=child.wait()
        item={'stage':name,'exit_code':code,'seconds':time.time()-start,'run_id':run_id}
        manifest.append(item);emit(folder/'events.jsonl',item)
        (folder/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        print(json.dumps(item),flush=True)
    print('CAMPAIGN '+str(folder),flush=True)


if __name__=='__main__':main()
