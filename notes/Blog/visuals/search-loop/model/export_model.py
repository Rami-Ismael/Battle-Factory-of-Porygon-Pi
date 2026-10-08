"""Export the existing pilot checkpoint; no training and no battle evaluation."""
import hashlib, json, sys, types, time, shutil
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
import onnx
import onnxruntime as ort

REPO = Path('/Users/ramiismael/Documents/code/vgc-team-generator-pilot')
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO/'src'))
def source_module(name, replacements=()):
    module = types.ModuleType(name); module.__file__ = str(REPO/'src'/f'{name}.py')
    sys.modules[name] = module
    source = Path(module.__file__).read_text()
    for old,new in replacements: source = source.replace(old,new)
    exec(compile(source, module.__file__, 'exec'), module.__dict__)
    return module
C = source_module('corpus', [('/tmp/vgc-pilot/data', str(REPO/'data')),('/tmp/vgc-pilot/vgc-bench/teams/reg_mb', str(REPO/'teams/reg_mb'))])
E = source_module('encode', [('/tmp/vgc-pilot/learnset_true.json', str(REPO/'results/learnset_true.json'))])
D = source_module('diffusion')
H = source_module('hpsdiffusion')
torch.set_num_threads(1)
torch.backends.mha.set_fastpath_enabled(False)
meta = json.loads((REPO/'results/temperature_p0.training.json').read_text())
checkpoint = REPO/'results/temperature_p0.pt'
assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == meta['checkpoint_sha256']
teams=[]
for original,sha in meta['corpus'].items():
    p=REPO/'teams/reg_mb'/original.split('/teams/reg_mb/')[-1]
    assert hashlib.sha256(p.read_bytes()).hexdigest()==sha, p
    t=C.parse_team(p)
    assert len(t)==6
    teams.append(t)
V=E.Vocab(teams); constraints=D.Constraints(V,E.Legality(teams))
model=H.TeamDiffusionHPS(V).cpu().eval()
model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True)['sd'])
width=max(V.sizes.values())
class ExportModel(torch.nn.Module):
    def __init__(self,model):super().__init__();self.model=model
    def forward(self,x,t,w):
        h=self.model(x,t,w)
        return torch.stack([F.pad(self.model.logits(h,c),(0,width-V.sizes[V.key(c)]),value=-1e9) for c in range(48)],dim=1)
wrapped=ExportModel(model).eval()
x=torch.zeros((1,48),dtype=torch.int64);t=torch.ones(1);w=torch.tensor([H.WNULL],dtype=torch.int64)
start=time.perf_counter()
with torch.inference_mode():
    torch.onnx.export(wrapped,(x,t,w),str(OUT/'model.onnx'),input_names=['x','t','w'],output_names=['logits'],opset_version=17,dynamo=False)
print('export seconds',time.perf_counter()-start,flush=True)
onnx.checker.check_model(str(OUT/'model.onnx'))
options=ort.SessionOptions();options.intra_op_num_threads=1
session=ort.InferenceSession(str(OUT/'model.onnx'),options,providers=['CPUExecutionProvider'])
look={};spreads=defaultdict(list)
for team in teams:
    for slot in team:
        for value in [slot['species'],slot['ability'],slot['item'],slot['nature'],*slot['moves']]:
            if value:look[C.norm(value)]=value
        spreads[C.norm(slot['species'])].append(slot['evs'])
manifest=dict(schemaVersion=1,modelFile='model.onnx',vocab=V.itos,sizes=V.sizes,columns=[V.key(c) for c in range(48)],order=D.ORDER,baseOf=constraints.base_of,abilityAllowed={k:torch.where(v)[0].tolist() for k,v in constraints.ab_ok.items()},moveAllowed={k:torch.where(v)[0].tolist() for k,v in constraints.mv_ok.items()},megaStoneOf=C.MEGA_STONE_OF,displayNames=look,spreads=dict(spreads),condition=H.WNULL,outputWidth=width,format='gen9championsvgc2026regmb',parameters=sum(p.numel() for p in model.parameters()),checkpointSha256=meta['checkpoint_sha256'],modelSha256=hashlib.sha256((OUT/'model.onnx').read_bytes()).hexdigest(),training=dict(corpusSize=len(teams),epochs=meta['epochs'],seed=meta['seed'],condition='Unconditional/null token; this checkpoint was trained on real teams without win-rate labels'),notes=['This is the pretrained initial generator, not a model proven to counter the meta.','48 categorical fields are sampled; Stat Points are copied from same-species training examples.','Constraint projection is not a substitute for the complete Showdown validator.'])
(OUT/'manifest.json').write_text(json.dumps(manifest,separators=(',',':'))+'\n')
fixtures=[];errors=[];times=[]
row=torch.zeros((1,48),dtype=torch.int64)
torch.manual_seed(713)
with torch.inference_mode():
    for step,col in enumerate(D.ORDER):
        tt=torch.tensor([1-step/48],dtype=torch.float32)
        pt=wrapped(row,tt,w).numpy()
        before=time.perf_counter();ov=session.run(None,{'x':row.numpy(),'t':tt.numpy(),'w':w.numpy()})[0];times.append(time.perf_counter()-before)
        err=float(np.max(np.abs(pt-ov)));errors.append(err)
        assert np.allclose(pt,ov,rtol=2e-4,atol=3e-5), (step,err)
        allowed=constraints.mask_for(col,row[0],'cpu')
        if step in (0,6,18,42,47):
            fixtures.append(dict(step=step,col=col,x=row[0].tolist(),t=float(tt[0]),w=H.WNULL,allowed=torch.where(allowed)[0].tolist(),logits=pt[0,col,:V.sizes[V.key(col)]].tolist()))
        logits=torch.tensor(pt[0,col,:V.sizes[V.key(col)]]);logits[~allowed]=-1e9
        row[0,col]=torch.multinomial(torch.softmax(logits,dim=-1),1)
(OUT/'fixtures.json').write_text(json.dumps(dict(fixtures=fixtures,finalRow=row[0].tolist()),separators=(',',':'))+'\n')
report=dict(onnxBytes=(OUT/'model.onnx').stat().st_size,parameters=manifest['parameters'],maxAbsoluteError=max(errors),torchOnnxComparedSteps=48,ortCpuSecondsTotal=sum(times),ortCpuMedianStep=float(np.median(times)),corpusHashesVerified=len(teams),checkpointSha256=manifest['checkpointSha256'],modelSha256=manifest['modelSha256'],versions=dict(torch=torch.__version__,onnx=onnx.__version__,onnxruntime=ort.__version__),cpuThreads=1)
(OUT/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report),flush=True)
