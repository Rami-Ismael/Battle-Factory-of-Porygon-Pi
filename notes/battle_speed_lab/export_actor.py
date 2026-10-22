"""Export the frozen BC actor and validate ONNX Runtime against real inputs.

This prepares the colocated-policy experiment. Export alone does not remove
Python preprocessing or demonstrate a colocated battle implementation.
"""
import argparse
import copy
import json
import time
from pathlib import Path
from run import HERE,digest,snapshot_sources


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--corpus',type=Path,default=HERE/'runs/encoder-verify-001/baseline-observations.npz')
    ap.add_argument('--run-id',default='export-'+time.strftime('%Y%m%d-%H%M%S'))
    args=ap.parse_args()
    from validate_inference import np,torch,PPO,actor_logits,sample_actions
    import onnx
    import onnxruntime as ort
    torch.set_num_threads(1)
    policy=PPO.load('/tmp/bc_100.zip',device='cpu').policy;policy.set_training_mode(False)
    frozen=copy.deepcopy(policy)
    # Embedding max_norm mutates weights on access. Renormalize every row once,
    # then freeze this constraint for a pure, portable inference graph.
    with torch.no_grad():
        for module in frozen.modules():
            if isinstance(module,torch.nn.Embedding) and module.max_norm is not None:
                module(torch.arange(module.num_embeddings));module.max_norm=None
    class Actor(torch.nn.Module):
        def __init__(self):
            super().__init__();self.features=frozen.pi_features_extractor
            self.mlp=frozen.mlp_extractor;self.output=frozen.action_net
        def forward(self,observations):
            features=self.features({'observation':observations.float()})
            return self.output(self.mlp.forward_actor(features))
    model=Actor().eval()
    data=np.load(args.corpus);obs=data['observations'];masks=data['masks']
    folder=HERE/'runs'/args.run_id;folder.mkdir(parents=True,exist_ok=False)
    hashes=snapshot_sources(folder)
    target=folder/'actor.onnx'
    # Disable fused Transformer fastpath for export; reference validation below
    # returns to the original setting and checks the resulting differences.
    fastpath=torch.backends.mha.get_fastpath_enabled()
    try:
        torch.backends.mha.set_fastpath_enabled(False)
        torch.onnx.export(model,(torch.as_tensor(obs[:2],dtype=torch.float32),),str(target),
            input_names=['observations'],output_names=['logits'],opset_version=17,
            dynamic_axes={'observations':{0:'batch'},'logits':{0:'batch'}},dynamo=False)
    finally:torch.backends.mha.set_fastpath_enabled(fastpath)
    onnx.checker.check_model(str(target))
    options=ort.SessionOptions();options.intra_op_num_threads=1;options.inter_op_num_threads=1
    session=ort.InferenceSession(str(target),sess_options=options,providers=['CPUExecutionProvider'])
    maximum=0.;cases=0;actions=0
    with torch.inference_mode():
        for size in [1,8,24,48]:
            for i in range(0,len(obs),size):
                observations=torch.as_tensor(obs[i:i+size]);mask=torch.as_tensor(masks[i:i+size])
                expected=actor_logits(policy,{'observation':observations,'action_mask':mask})
                frozen_logits=model(observations)
                torch.testing.assert_close(frozen_logits,expected,rtol=1e-5,atol=1e-5)
                actual=torch.as_tensor(session.run(['logits'],{'observations':obs[i:i+size].astype(np.float32)})[0])
                torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-4)
                maximum=max(maximum,float((actual-expected).abs().max()))
                for seed in [91,271]:
                    torch.manual_seed(seed);reference=sample_actions(policy,expected,mask)
                    torch.manual_seed(seed);exported=sample_actions(policy,actual,mask)
                    if not torch.equal(reference,exported):raise AssertionError('Export changed sampled actions')
                    actions+=len(observations)
                cases+=len(observations)
    result={'passed':True,'source_hashes':hashes,'model_sha256':digest(target),
        'teacher_sha256':digest(Path('/tmp/bc_100.zip')),'corpus_sha256':digest(args.corpus),
        'onnx_version':onnx.__version__,'onnxruntime_version':ort.__version__,'torch_version':torch.__version__,
        'logit_cases':cases,'sampled_action_cases':actions,'max_abs_logit_error':maximum,
        'rtol':1e-5,'atol':1e-4,'observation_width':obs.shape[1],'action_width':masks.shape[1],
        'limitation':'Frozen actor export only. Tested numerical/action agreement is not exhaustive equivalence or a colocated speed measurement.'}
    (folder/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='source_hashes'}),flush=True)


if __name__=='__main__':main()
