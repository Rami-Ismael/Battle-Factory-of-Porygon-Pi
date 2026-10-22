"""Serial second-phase prototypes, training, timing, and allocation analysis."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from run import HERE,emit,snapshot_sources


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--actor-model',type=Path,required=True)
    ap.add_argument('--capture-corpus',type=Path,required=True)
    ap.add_argument('--quality-corpus',type=Path,required=True)
    ap.add_argument('--run-id',default='extended-'+time.strftime('%Y%m%d-%H%M%S'))
    args=ap.parse_args()
    args.actor_model=args.actor_model.resolve();args.capture_corpus=args.capture_corpus.resolve()
    args.quality_corpus=args.quality_corpus.resolve()
    folder=HERE/'runs'/args.run_id;folder.mkdir(parents=True,exist_ok=False)
    snapshot_sources(folder)
    student=folder/'student.pt'
    train=[str(args.capture_corpus/f'row-{i}.npz') for i in [0,7,14,21,28]]
    validation=[str(args.capture_corpus/f'row-{i}.npz') for i in [35,42,48]]
    stages=[
        ('queue-validation',['test_leased_queue.py'],None),
        ('queue',['queue_bench.py','--run-id',args.run_id+'-queue'],'queue-validation'),
        ('array-validation',['array_bench.py','--validate-only','--battles','128','--run-id',args.run_id+'-array-validation'],None),
        ('array',['array_bench.py','--battles','128','--run-id',args.run_id+'-array'],'array-validation'),
        ('node-actor',['direct_bc_bench.py','--node-actor-model',str(args.actor_model),
            '--run-id',args.run_id+'-node-actor'],None),
        ('apprentice',['apprentice.py','--train',*train,'--validation',*validation,'--output',str(student)],None),
        ('student',['run.py','--baseline','fast_stack','--variants','student','--student-file',str(student),
            '--row-indices','35','42','48','--blocks','5','--battles','96','--run-id',args.run_id+'-student'],'apprentice'),
        ('allocation',['allocation.py',str(args.quality_corpus),'--repeats','50'],None),
    ]
    (folder/'inputs.json').write_text(json.dumps({k:str(v) for k,v in vars(args).items()},indent=2)+'\n')
    manifest=[]
    for name,arguments,dependency in stages:
        if dependency and not any(x['stage']==dependency and x.get('exit_code')==0 for x in manifest):
            entry={'stage':name,'status':'skipped','reason':dependency+' failed'}
        else:
            command=[sys.executable,str(HERE/arguments[0]),*arguments[1:]]
            print(json.dumps({'stage':name,'status':'starting'}),flush=True)
            start=time.time()
            with (folder/(name+'.log')).open('w') as log:
                child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
                emit(folder/'events.jsonl',{'stage':name,'pid':child.pid,'status':'running','command':command})
                code=child.wait()
            entry={'stage':name,'exit_code':code,'seconds':time.time()-start}
        manifest.append(entry);emit(folder/'events.jsonl',entry)
        (folder/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        print(json.dumps(entry),flush=True)


if __name__=='__main__':main()
