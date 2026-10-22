"""Check concurrent remote actor routing, numerical logits and seeded actions."""
import argparse
import asyncio
import json
import time
from pathlib import Path
from run import HERE,digest,snapshot_sources
from scaling import ActorServer
from remote_inference import RemoteActor


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--corpus',type=Path,default=HERE/'runs/encoder-verify-001/baseline-observations.npz')
    ap.add_argument('--run-id',default='remote-validation-'+time.strftime('%Y%m%d-%H%M%S'))
    args=ap.parse_args()
    # Reuse the known pinned import setup, without invoking its main function.
    from validate_inference import np,torch,PPO,actor_logits,sample_actions
    torch.set_num_threads(1)
    data=np.load(args.corpus)
    obs,masks=data['observations'],data['masks']
    folder=HERE/'runs'/args.run_id;folder.mkdir(parents=True,exist_ok=False)
    hashes=snapshot_sources(folder)
    policy=PPO.load('/tmp/bc_100.zip',device='cpu').policy;policy.set_training_mode(False)
    server=ActorServer(folder,wait_ms=5.)
    async def validate():
        clients=[RemoteActor(server.socket) for _ in range(4)]
        maximum=0.;cases=0;actions=0
        try:
            for size in [1,8,24]:
                for start in range(0,len(obs),size*len(clients)):
                    batches=[(obs[i:i+size],masks[i:i+size])
                             for i in range(start,min(start+size*len(clients),len(obs)),size)]
                    # No Torch context spans an await: its thread-local mode
                    # would otherwise affect unrelated coroutine execution.
                    expected=[]
                    with torch.inference_mode():
                        for x,m in batches:
                            expected.append(actor_logits(policy,{'observation':torch.as_tensor(x),
                                'action_mask':torch.as_tensor(m)}))
                    results=await asyncio.gather(*(c.logits(x,m) for c,(x,m) in zip(clients,batches)))
                    with torch.inference_mode():
                        for local,remote,(_,m) in zip(expected,results,batches):
                            actual=torch.as_tensor(remote);mask=torch.as_tensor(m)
                            torch.testing.assert_close(actual,local,rtol=1e-5,atol=1e-5)
                            maximum=max(maximum,float((actual-local).abs().max()))
                            for seed in [91,271]:
                                torch.manual_seed(seed);a=sample_actions(policy,local,mask)
                                torch.manual_seed(seed);b=sample_actions(policy,actual,mask)
                                if not torch.equal(a,b):raise AssertionError('Remote sampled action differs')
                                actions+=len(m)
                            cases+=len(m)
            return {'passed':True,'logit_cases':cases,'seeded_action_cases':actions,
                    'max_abs_logit_error':maximum,'concurrent_clients':len(clients),
                    'limitation':'Tolerance-based numerical checks on captured states; not exhaustive game equivalence.'}
        finally:
            for client in clients:await client.close()
    try:result=asyncio.run(validate())
    finally:server.close()
    stats=json.loads((folder/'remote-actor-stats.json').read_text())
    if stats['requests']<=stats['batches']:
        raise AssertionError('Validation did not exercise cross-client batching')
    result.update(source_hashes=hashes,corpus_sha256=digest(args.corpus),server_stats=stats)
    (folder/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['source_hashes','server_stats']}),flush=True)


if __name__=='__main__':main()
