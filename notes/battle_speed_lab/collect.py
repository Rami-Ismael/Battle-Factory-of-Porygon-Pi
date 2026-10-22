"""Collect real BC inputs with battle identities for distillation/cache studies."""
import argparse
import json
import time
from run import HERE,Server,Worker,prepare_runtime,snapshot_sources,workload


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--row-indices',nargs='+',type=int,default=[0,7,14,21,28,35,42,48])
    ap.add_argument('--battles',type=int,default=96)
    ap.add_argument('--max-samples',type=int,default=4096)
    ap.add_argument('--run-id',default='corpus-'+time.strftime('%Y%m%d-%H%M%S'))
    args=ap.parse_args()
    folder=HERE/'runs'/args.run_id;folder.mkdir(parents=True,exist_ok=False)
    hashes=snapshot_sources(folder)
    runtime=HERE/'runs/private-runtime';prepare_runtime(runtime)
    config={'batcher':'original','threads':1,'concurrency':24,'shared':True,'encoder':'fast',
            'actor_only':True,'inference_mode':True,'capture':True,'capture_stride':1,'capture_limit':args.max_samples}
    server=Server(runtime,folder)
    worker=None
    manifest=[]
    try:
        worker=Worker(config,folder,'capture')
        for index in args.row_indices:
            work=workload(args.battles,index)
            file=folder/f'row-{index}.npz'
            result=worker.trial({**work,'config':config,'port':server.port,'seed':20261013+index,
                                 'capture_path':str(file),'timeout':180})
            entry={'row_index':index,'candidate_id':work['pairs'][0]['a_id'],'capture':result['capture'],
                   'battles':result['completed'],'turns':result['turns']}
            manifest.append(entry)
            (folder/'manifest.json').write_text(json.dumps({'args':vars(args),'source_hashes':hashes,'files':manifest},indent=2)+'\n')
            print(json.dumps(entry),flush=True)
    finally:
        if worker:worker.close()
        server.close()


if __name__=='__main__':main()
