"""Small approximate policy distilled from the frozen BC actor.

Uses legal-action KL targets and a file-disjoint validation split. This changes
the evaluator; speed alone never establishes competitive-policy equivalence.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn


class Apprentice(nn.Module):
    def __init__(self,width,actions,hidden=128):
        super().__init__()
        self.register_buffer('mean',torch.zeros(width))
        self.register_buffer('scale',torch.ones(width))
        self.net=nn.Sequential(nn.Linear(width,hidden),nn.GELU(),nn.Linear(hidden,actions))
    def forward(self,observations):
        return self.net((observations-self.mean)/self.scale)


def load_student(path):
    data=torch.load(path,map_location='cpu',weights_only=True)
    model=Apprentice(data['width'],data['actions'],data['hidden'])
    model.load_state_dict(data['state_dict']);model.eval()
    return model


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--train',nargs='+',type=Path,required=True)
    ap.add_argument('--validation',nargs='+',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--epochs',type=int,default=30)
    ap.add_argument('--hidden',type=int,default=128)
    args=ap.parse_args()
    if set(p.resolve() for p in args.train)&set(p.resolve() for p in args.validation):
        raise ValueError('Training and validation files overlap')
    def read(paths):
        arrays=[];masks=[];groups=set()
        for path in paths:
            data=np.load(path)
            if 'battle_ids' not in data:raise ValueError('Corpus lacks battle identities')
            arrays.append(data['observations']);masks.append(data['masks'])
            # All files in one collected corpus share a private server; battle IDs
            # are consequently comparable across those files.
            groups.update(str(path.parent.resolve())+':'+str(x) for x in data['battle_ids'])
        return torch.as_tensor(np.concatenate(arrays)),torch.as_tensor(np.concatenate(masks)),groups
    train,train_mask,train_groups=read(args.train)
    val,val_mask,val_groups=read(args.validation)
    if train_groups&val_groups:raise ValueError('Battle-level leakage across split')
    bench=Path.home()/'.local/share/vgc-pilot-runtime/unpacked/vgc-bench-d79f9532947ac114dce1dda2456a590afcd375b2'
    sys.path.insert(0,str(bench));old=os.getcwd();os.chdir(bench)
    from stable_baselines3 import PPO
    from vgc_bench.src import policy as _policy
    os.chdir(old)
    from inference import actor_logits,sample_actions
    torch.set_num_threads(1);torch.manual_seed(820)
    teacher=PPO.load('/tmp/bc_100.zip',device='cpu').policy;teacher.set_training_mode(False)
    def targets(x,mask):
        with torch.inference_mode():
            result=torch.cat([actor_logits(teacher,{'observation':x[i:i+64],'action_mask':mask[i:i+64]})
                              for i in range(0,len(x),64)])
        return result.clone()
    train_targets=targets(train,train_mask);val_targets=targets(val,val_mask)
    model=Apprentice(train.shape[1],train_mask.shape[1],args.hidden)
    model.mean.copy_(train.mean(0));model.scale.copy_(train.std(0).clamp_min(.01))
    optimizer=torch.optim.Adam(model.parameters(),lr=3e-4)
    def kl(logits,target,mask):
        values=[]
        for a,b in [(0,logits.shape[1]//2),(logits.shape[1]//2,logits.shape[1])]:
            teacher_log=torch.log_softmax(target[:,a:b].masked_fill(mask[:,a:b]!=1,-1e9),dim=-1)
            student_log=torch.log_softmax(logits[:,a:b].masked_fill(mask[:,a:b]!=1,-1e9),dim=-1)
            values.append((teacher_log.exp()*(teacher_log-student_log)).sum(-1).mean())
        return sum(values)/2
    history=[];start=time.perf_counter()
    for epoch in range(args.epochs):
        order=torch.randperm(len(train));model.train()
        for ids in order.split(128):
            optimizer.zero_grad();loss=kl(model(train[ids]),train_targets[ids],train_mask[ids]);loss.backward();optimizer.step()
        model.eval()
        with torch.no_grad():
            valid=kl(model(val),val_targets,val_mask).item()
        history.append({'epoch':epoch+1,'validation_kl':valid})
    with torch.inference_mode():
        predicted=model(val)
        expected_actions=sample_actions(teacher,val_targets,val_mask,True)
        predicted_actions=sample_actions(teacher,predicted,val_mask,True)
        agreement=(expected_actions==predicted_actions).all(dim=1).float().mean().item()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    torch.save({'state_dict':model.state_dict(),'width':train.shape[1],'actions':train_mask.shape[1],
                'hidden':args.hidden},args.output)
    report={'training_inputs':len(train),'validation_inputs':len(val),'training_battles':len(train_groups),
            'validation_battles':len(val_groups),'validation_joint_greedy_agreement':agreement,
            'history':history,'seconds':time.perf_counter()-start,'parameters':sum(p.numel() for p in model.parameters()),
            'limitation':'Approximate student. KL evaluates initial marginal masks; action agreement applies the conditional second-action mask. Battle strength and throughput still require live evaluation.'}
    args.output.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='history'}),flush=True)


if __name__=='__main__':main()
