"""Run original trained sampler with observational hooks; write an auditable replay.
No model retraining or repository source modifications. CPU, seed 91826 onward.
"""
from pathlib import Path
import sys,types,json,hashlib,random,subprocess
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent
REPO=Path('/Users/ramiismael/Documents/code/vgc-team-generator-pilot')
CORPUS=REPO/'teams/reg_mb'
CK=REPO/'results/temperature_p0.pt'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
sys.path.insert(0,str(REPO/'src'))
# Relocate retired /tmp data paths in memory; retain all parsing logic.
mod=types.ModuleType('corpus');mod.__file__=str(REPO/'src/corpus.py');sys.modules['corpus']=mod
source=(REPO/'src/corpus.py').read_text().replace('Path("/tmp/vgc-pilot/data")',repr(REPO/'data')).replace('PosixPath(', 'Path(').replace('Path("/tmp/vgc-pilot/vgc-bench/teams/reg_mb")',repr(CORPUS)).replace('PosixPath(', 'Path(')
exec(compile(source,mod.__file__,'exec'),mod.__dict__)
import encode as E, diffusion as D,hpsdiffusion as H
manifest=json.loads((REPO/'results/temperature_p0.training.json').read_text())
assert sha(CK)==manifest['checkpoint_sha256']
files=sorted(CORPUS.rglob('*.txt'));teams=[];source_records=[]
for p in files:
 team=mod.parse_team(p)
 if len(team)==6:
  key='/tmp/vgc-pilot/vgc-bench/teams/reg_mb/'+str(p.relative_to(CORPUS))
  assert sha(p)==manifest['corpus'][key],key
  teams.append(team);source_records.append(p)
assert len(teams)==manifest['corpus_size']
V=E.Vocab(teams);L=E.Legality(teams);L._true={k:set(v) for k,v in json.loads((REPO/'results/learnset_true.json').read_text()).items()}
C=D.Constraints(V,L);torch.set_num_threads(4)
model=H.TeamDiffusionHPS(V).to('cpu');model.load_state_dict(torch.load(CK,map_location='cpu',weights_only=True)['sd']);model.eval()
look={};spreads={}
for team in teams:
 for member in team:
  for val in [member['species'],member['ability'],member['item'],member['nature'],*member['moves']]:look[mod.norm(val)]=val
  spreads.setdefault(mod.norm(member['species']),[]).append(member['evs'])
def display(c,v):
 value=V.decode_field(c,int(v));return look.get(value,value) if v else None
original_sample=torch.multinomial
attempts=[]
for seed in range(91826,91838):
 torch.manual_seed(seed);rng=np.random.default_rng(seed);events=[];state=[0]*48
 def observe(probs,*args,**kwargs):
  result=original_sample(probs,*args,**kwargs)
  c=D.ORDER[len(events)];chosen=int(result.item());p=probs.detach().cpu();values,ids=torch.topk(p,min(5,len(p)))
  state[c]=chosen
  events.append({'step':len(events)+1,'column':c,'slot':c//8,'field':E.FIELDS[c%8],'value':display(c,chosen),'probability':float(p[chosen]),'top':[{'value':display(c,int(i)),'probability':float(v)} for v,i in zip(values,ids)],'state':[display(i,v) for i,v in enumerate(state)]})
  return result
 torch.multinomial=observe
 try:row=H.sample_constrained(model,C,1,H.WNULL,1.,1.,device='cpu')[0].tolist()
 finally:torch.multinomial=original_sample
 assert row==state and len(events)==48
 # Replay the untouched sampler at the same seed to demonstrate observational parity.
 torch.manual_seed(seed)
 assert H.sample_constrained(model,C,1,H.WNULL,1.,1.,device='cpu')[0].tolist()==row
 final=[];pastes=[]
 for i in range(6):
  vals=[display(i*8+j,row[i*8+j]) for j in range(8)];sp=V.decode_field(i*8,row[i*8]);options=spreads[sp];spread_index=int(rng.integers(0,len(options)));evs=options[spread_index]
  member={'species':vals[0],'ability':vals[1],'item':vals[2],'moves':vals[3:7],'nature':vals[7],'statPoints':evs,'spreadSourceIndex':spread_index}
  final.append(member);lines=[vals[0]+(' @ '+vals[2] if vals[2] else ''),'Ability: '+vals[1],'Level: 50','EVs: '+' / '.join(str(evs[s])+' '+s for s in mod.STATS if evs.get(s)),vals[7]+' Nature',*['- '+m for m in vals[3:7]]];pastes.append('\n'.join(lines))
 paste='\n\n'.join(pastes)+'\n'
 bundle=ROOT.parent/'search-loop/runtime/showdown-validator.js'
 js="const fs=require('fs'),vm=require('vm');vm.runInThisContext(fs.readFileSync(process.argv[1],'utf8'));console.log(JSON.stringify(PokemonTeamValidator.validateTeam(fs.readFileSync(0,'utf8'))));"
 errors=json.loads(subprocess.check_output(['node','-e',js,str(bundle)],input=paste,text=True));attempts.append({'seed':seed,'errors':errors});print('seed',seed,'validation',errors,flush=True)
 if not errors:break
else:raise RuntimeError('No accepted trace; inspect attempts')
trace={'kind':'Recorded trained-model inference','checkpoint':str(CK),'checkpointSha256':sha(CK),'seed':seed,'device':'cpu','trainingCorpusSize':len(teams),'conditioning':'unconditional WNULL','guidance':1,'temperature':1,'sampler':'hpsdiffusion.sample_constrained','note':'Masked-diffusion-trained Transformer; dependency-order constrained decoding. Stats copied separately from species-matched corpus sets. No battles evaluated.','format':'gen9championsvgc2026regmb','sourceHashes':{n:sha(REPO/'src'/n) for n in ['corpus.py','encode.py','diffusion.py','hpsdiffusion.py']},'validatorSha256':sha(bundle),'attempts':attempts,'samplerParity':True,'events':events,'finalTeam':final,'paste':paste,'validationErrors':errors}
(ROOT/'trace.json').write_text(json.dumps(trace,indent=2));(ROOT/'generated-team.txt').write_text(paste)
print('RECORDED',seed,[m['species'] for m in final],flush=True)
