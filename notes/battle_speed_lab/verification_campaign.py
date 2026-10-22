"""Serial matched-adapter and live wall-clock allocation verification."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from run import HERE,emit,snapshot_sources


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--quality-corpus',type=Path,required=True)
    ap.add_argument('--run-id',default='verification-'+time.strftime('%Y%m%d-%H%M%S'))
    args=ap.parse_args();folder=HERE/'runs'/args.run_id;folder.mkdir(parents=True,exist_ok=False)
    snapshot_sources(folder)
    stages=[('transport',['transport_bench.py']),
            ('live-allocation',['live_allocation.py',str(args.quality_corpus.resolve()),'--seconds','15','--blocks','5'])]
    manifest=[]
    for name,arguments in stages:
        command=[sys.executable,str(HERE/arguments[0]),*arguments[1:],'--run-id',args.run_id+'-'+name]
        start=time.time();print(json.dumps({'stage':name,'status':'starting'}),flush=True)
        with (folder/(name+'.log')).open('w') as log:
            child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
            emit(folder/'events.jsonl',{'stage':name,'status':'running','pid':child.pid,'command':command})
            code=child.wait()
        record={'stage':name,'exit_code':code,'seconds':time.time()-start}
        manifest.append(record);emit(folder/'events.jsonl',record)
        (folder/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        print(json.dumps(record),flush=True)


if __name__=='__main__':main()
