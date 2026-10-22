"""Measure exact prefix restoration, including preparation cost and miss cases."""
import argparse
import json
import random
import subprocess
import time
from run import HERE,SOURCE,Monitor,analyze,emit,snapshot_sources,stop_group,workload
from direct_bench import Direct,signatures


class Snapshot(Direct):
    def __init__(self,folder):
        self.log=(folder/'snapshot.log').open('w')
        self.proc=subprocess.Popen(['node',str(folder/'source/snapshot.cjs'),str(SOURCE)],
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log,text=True,bufsize=1,start_new_session=True)
        self.ready=self.read()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--blocks',type=int,default=5)
    ap.add_argument('--battles',type=int,default=48);ap.add_argument('--prefix-turn',type=int,default=3)
    ap.add_argument('--run-id',default='snapshot-'+time.strftime('%Y%m%d-%H%M%S'));args=ap.parse_args()
    folder=HERE/'runs'/args.run_id;folder.mkdir(parents=True,exist_ok=False);hashes=snapshot_sources(folder)
    work=workload(args.battles)
    (folder/'metadata.json').write_text(json.dumps({'args':vars(args),'source_hashes':hashes,
        'note':'Random-policy exact snapshots; includes both policy RNGs. Resume queries repeat the same seeds; not new independent simulations.'},indent=2)+'\n')
    client=Snapshot(folder);rows=[];reference={};preparations=[]
    try:
        def execute(mode,phase,block,position):
            jobs=[{**j,'seed':20261013+block*10000+j['id']} for j in work['pairs']]
            with Monitor([client.proc.pid]) as monitor:
                result=client.run({'format':work['format'],'mode':mode,'prefixTurn':args.prefix_turn,'jobs':jobs})
            sig=signatures(result)
            if block not in reference:reference[block]=sig
            if reference[block]!=sig:raise RuntimeError('Snapshot continuation changed trace/outcome')
            row={**result,'phase':phase,'block':block,'variant':'resume' if mode=='resume' else 'baseline',
                 'treatment':'resume','position':position,'trace_equivalent':True,'telemetry':monitor.samples,'clock':monitor.clock}
            rows.append(row);emit(folder/'trials.jsonl',row)
            print(json.dumps({'mode':mode,'phase':phase,'block':block,'seconds':result['seconds'],
                              'hits':result['hits'],'misses':result['misses']}),flush=True)
            return result
        execute('full','warmup',-1,0)
        rng=random.Random(912)
        for block in range(args.blocks):
            built=execute('build','preparation',block,-1);preparations.append(built)
            sequence=['full','resume','resume','full'] if rng.getrandbits(1) else ['resume','full','full','resume']
            for pos,mode in enumerate(sequence):execute(mode,'measure',block,pos)
            (folder/'summary.json').write_text(json.dumps(analyze(rows),indent=2)+'\n')
        fresh=[{**j,'seed':912000+j['id']} for j in work['pairs']]
        miss=client.run({'format':work['format'],'mode':'resume','prefixTurn':args.prefix_turn,'jobs':fresh})
        if miss['hits']!=0:raise AssertionError('Unseen seeds reused a cached prefix')
        (folder/'reuse-audit.json').write_text(json.dumps({'new_seed_hits':miss['hits'],'new_seed_misses':miss['misses'],
            'preparation_seconds':sum(r['seconds'] for r in preparations),
            'snapshot_bytes':sum(r['snapshotBytes'] for r in preparations),
            'limitation':'No speedup for unseen independent seeds is established. All first-use preparation costs are recorded separately.'},indent=2)+'\n')
    finally:client.close()


if __name__=='__main__':main()
