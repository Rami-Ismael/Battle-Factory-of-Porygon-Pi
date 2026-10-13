"""Validate actor-only/cached logits and fresh sampling on captured real inputs."""
import argparse
import json
import os
import sys
from pathlib import Path
BENCH=Path.home()/'.local/share/vgc-pilot-runtime/unpacked/vgc-bench-d79f9532947ac114dce1dda2456a590afcd375b2'
sys.path.insert(0,str(BENCH))
ORIGINAL_CWD=os.getcwd()
os.chdir(BENCH)
import numpy as np
import torch
from stable_baselines3 import PPO
from inference import actor_logits,sample_actions,LogitCache
from vgc_bench.src import policy as _policy_module
os.chdir(ORIGINAL_CWD)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('corpus',type=Path)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    torch.set_num_threads(1)
    data=np.load(args.corpus)
    obs,masks=data['observations'],data['masks']
    policy=PPO.load('/tmp/bc_100.zip',device='cpu').policy
    policy.set_training_mode(False)
    maximum=0.;tested=0;action_tests=0
    cache=LogitCache(limit=4096)
    with torch.inference_mode():
        for size in [1,8,24]:
            for i in range(0,len(obs),size):
                arrays=obs[i:i+size];mask_arrays=masks[i:i+size]
                inputs={'observation':torch.as_tensor(arrays),'action_mask':torch.as_tensor(mask_arrays)}
                reference,_=policy.get_logits(inputs,actor_grad=False)
                actor=actor_logits(policy,inputs)
                torch.testing.assert_close(actor,reference,rtol=1e-5,atol=1e-6)
                maximum=max(maximum,float((actor-reference).abs().max()))
                cached=cache.compute(policy,inputs,arrays,mask_arrays)
                # Cache entries produced at a different batch size can differ in
                # floating-point rounding. Treat this as numerical equivalence,
                # not bitwise identity, and still demand tested action agreement.
                torch.testing.assert_close(cached,reference,rtol=1e-5,atol=1e-5)
                maximum=max(maximum,float((cached-reference).abs().max()))
                for seed in [91,271]:
                    torch.manual_seed(seed)
                    expected=policy.forward(inputs)[0]
                    torch.manual_seed(seed)
                    actual=sample_actions(policy,actor,inputs['action_mask'])
                    if not torch.equal(expected,actual):raise AssertionError('Actor-only sampled action mismatch')
                    torch.manual_seed(seed)
                    hit=sample_actions(policy,cached,inputs['action_mask'])
                    if not torch.equal(expected,hit):raise AssertionError('Cached sampled action mismatch')
                    action_tests+=len(arrays)
                tested+=len(arrays)
    result={'passed':True,'corpus':str(args.corpus),'observations':len(obs),'batch_sizes':[1,8,24],
            'logit_cases':tested,'seeded_action_cases':action_tests,'max_abs_logit_error':maximum,
            'cache_hits':cache.hits,'cache_misses':cache.misses,
            'cache_logit_atol':1e-5,'cache_logit_rtol':1e-5,
            'limitation':'Coverage of captured states only; tolerance-based logits, exact tested sampled actions. Cross-batch cache is numerically equivalent, not bitwise exact.'}
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
