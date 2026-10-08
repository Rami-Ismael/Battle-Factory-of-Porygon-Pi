"""Record the data behind the archetype-diffusion animation.

Samples N teams from the saved temperature_p0.pt checkpoint with the untouched
hpsdiffusion.sample_constrained, then, for every partly masked state along each
team's decode, samples K completions from that state with the same sampler loop
(separate RNG). A state's 2D position is the mean projected position of its
completions; its spread is their standard deviation; its colour is the share of
completions landing in each archetype. CPU only, no retraining, no battles.
"""
from pathlib import Path
import sys,types,json,hashlib,os
import numpy as np
import torch
import multiprocessing as mp
ROOT=Path(__file__).resolve().parent
REPO=Path('/Users/ramiismael/Documents/code/vgc-team-generator-pilot')
CORPUS=REPO/'teams/reg_mb'
CK=REPO/'results/temperature_p0.pt'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
sys.path.insert(0,str(REPO/'src'))
mod=types.ModuleType('corpus');mod.__file__=str(REPO/'src/corpus.py');sys.modules['corpus']=mod
source=(REPO/'src/corpus.py').read_text().replace('Path("/tmp/vgc-pilot/data")',repr(REPO/'data')).replace('Path("/tmp/vgc-pilot/vgc-bench/teams/reg_mb")',repr(CORPUS)).replace('PosixPath(','Path(')
exec(compile(source,mod.__file__,'exec'),mod.__dict__)
import encode as E, diffusion as D,hpsdiffusion as H
manifest=json.loads((REPO/'results/temperature_p0.training.json').read_text())
assert sha(CK)==manifest['checkpoint_sha256']
teams=[]
for p in sorted(CORPUS.rglob('*.txt')):
 team=mod.parse_team(p)
 if len(team)==6:
  assert sha(p)==manifest['corpus']['/tmp/vgc-pilot/vgc-bench/teams/reg_mb/'+str(p.relative_to(CORPUS))]
  teams.append(team)
assert len(teams)==manifest['corpus_size']
V=E.Vocab(teams);L=E.Legality(teams);L._true={k:set(v) for k,v in json.loads((REPO/'results/learnset_true.json').read_text()).items()}
C=D.Constraints(V,L)
model=H.TeamDiffusionHPS(V).to('cpu');model.load_state_dict(torch.load(CK,map_location='cpu',weights_only=True)['sd']);model.eval()
look={}
for team in teams:
 for m in team:
  for v in [m['species'],m['ability'],m['item'],m['nature'],*m['moves']]:look[mod.norm(v)]=v

N,K,K0,REF=[int(v) for v in os.environ.get('ARCH_SIZES','300,24,512,1500').split(',')]
SEED_MAIN,SEED_REF=20260928,20260929
ORDER=list(D.ORDER);LABEL_STEPS=42           # species, ability, item, moves; natures never change the label
ARCH=['Trick Room','Weather','Tailwind','Other']
# Weather setters: abilities, plus Megas whose Showdown ability sets weather (checked in the bundled dex).
WEATHER_AB={'drizzle','drought','sandstream','snowwarning'}
WEATHER_IT={'charizarditey','tyranitarite','froslassite','abomasite'}
WEATHER_SP={'charizardmegay','tyranitarmega','froslassmega','abomasnowmega'}
KEYS=['species','ability','item','move']
OFF=np.cumsum([0]+[V.sizes[k] for k in KEYS])
dec=lambda c,v:V.decode_field(c,int(v))

def label(row):
 mv={dec(c,row[c]) for c in range(48) if 3<=c%8<=6}
 if 'trickroom' in mv:return 0
 for i in range(6):
  if dec(i*8,row[i*8]) in WEATHER_SP or dec(i*8+1,row[i*8+1]) in WEATHER_AB or dec(i*8+2,row[i*8+2]) in WEATHER_IT:return 1
 return 2 if 'tailwind' in mv else 3

def feats(rows):
 X=np.zeros((len(rows),OFF[-1]),np.float32)
 for n,row in enumerate(rows):
  for c in range(48):
   j=c%8
   if j==7 or not row[c]:continue
   X[n,OFF[min(j,3)]+int(row[c])]=1
 return X

def sample(n,seed):
 torch.manual_seed(seed);out=[]
 while n>0:
  b=min(n,100);out.append(H.sample_constrained(model,C,b,H.WNULL,1.,1.,device='cpu'));n-=b
 return torch.cat(out)

@torch.no_grad()
def complete(state,s,k,g):
 """sample_constrained's loop resumed at step s, stopping once the label is fixed."""
 x=state.unsqueeze(0).repeat(k,1);w=torch.full((k,),H.WNULL,dtype=torch.long)
 for step in range(s,LABEL_STEPS):
  c=ORDER[step];lg=H._step_logits(model,x,1.0-step/len(ORDER),w,1.)[c]
  for b in range(k):
   ok=C.mask_for(c,x[b],'cpu');l=lg[b].clone();l[~ok]=-1e9
   x[b,c]=torch.multinomial(torch.softmax(l,-1),1,generator=g).item()
 return x

def masked(row,s):
 x=row.clone();x[ORDER[s:]]=0;return x

def summary(rows,W,mu):
 P=(feats(rows)-mu)@W;lab=np.bincount([label(r) for r in rows],minlength=4)/len(rows);S=np.cov(P.T,bias=True) if len(P)>1 else np.zeros((3,3))
 return [*np.round(P.mean(0),4).tolist(),*np.round([S[0,0],S[1,1],S[2,2],S[0,1],S[0,2],S[1,2]],5).tolist(),*np.round(lab,4).tolist()]

def worker(args):
 idx,row,W,mu=args;torch.set_num_threads(1);g=torch.Generator().manual_seed(SEED_MAIN+1+idx);out=[]
 for s in range(1,LABEL_STEPS):out.append(summary(complete(masked(row,s),s,K,g),W,mu))
 return idx,out

if __name__=='__main__':
 torch.set_num_threads(4)
 # 1. Reference sample -> archetype labels -> shrinkage LDA (all 3 axes).
 ref=sample(REF,SEED_REF);y=np.array([label(r) for r in ref]);X=feats(ref);mu=X.mean(0)
 Sw=sum(np.cov((X[y==k]-mu).T,bias=True)*(y==k).sum() for k in range(4))/len(X)+0.05*np.eye(X.shape[1])
 M=np.stack([X[y==k].mean(0)-mu for k in range(4)]);Sb=(M.T*np.bincount(y,minlength=4)/len(y))@M
 Lc=np.linalg.cholesky(Sw);Li=np.linalg.inv(Lc);ev,U=np.linalg.eigh(Li@Sb@Li.T);W=(Li.T@U[:,::-1][:,:3])
 P=(X-mu)@W;scale=np.percentile(np.abs(P),98,axis=0);W=W/scale
 cent=np.stack([((X[y==k]-mu)@W).mean(0) for k in range(4)])
 print('reference labels',np.bincount(y,minlength=4),'centroids',cent.round(2).tolist(),flush=True)
 # 2. Main teams, with sampler parity.
 main=sample(N,SEED_MAIN);assert torch.equal(main,sample(N,SEED_MAIN))
 # 3. Fully masked state is shared by every team: one large completion batch.
 s0=summary(complete(torch.zeros(48,dtype=torch.long),0,K0,torch.Generator().manual_seed(SEED_MAIN)),W,mu)
 with mp.get_context('spawn').Pool(11) as pool:
  res={}
  for i,out in pool.imap_unordered(worker,[(i,main[i],W,mu) for i in range(N)]):
   res[i]=out
   if len(res)%10==0:print('completed',len(res),'/',N,flush=True)
 steps=[];labels=[]
 for i in range(N):
  fin=summary(main[i:i+1],W,mu);labels.append(label(main[i]))
  steps.append([s0]+res[i]+[fin]*(len(ORDER)-LABEL_STEPS+1))
 fin=np.array([st[-1][:3] for st in steps]);rng=np.random.default_rng(SEED_MAIN);featured=[]
 for k in range(4):
  ids=[i for i in range(N) if labels[i]==k]
  if ids:featured.append(min(ids,key=lambda i:np.linalg.norm(fin[i]-cent[k])))
 featured+=rng.choice([i for i in range(N) if i not in featured],2,replace=False).tolist()
 show=lambda c,v:(look.get(dec(c,v),dec(c,v)) if v else None)
 data={'kind':'Recorded masked-diffusion inference with sampled completions','checkpoint':str(CK),'checkpointSha256':sha(CK),
  'sampler':'hpsdiffusion.sample_constrained, unconditional, temperature 1, guidance 1','samplerParity':True,'seedMain':SEED_MAIN,'seedReference':SEED_REF,
  'teams':N,'completionsPerState':K,'completionsFullyMasked':K0,'referenceTeams':REF,'referenceLabelCounts':np.bincount(y,minlength=4).tolist(),
  'projection':'shrinkage LDA (lambda 0.05, all 3 discriminant axes) on species/ability/item/move multi-hot of the reference sample; axes scaled to the 98th percentile; build.py maps 3D to 2D',
  'archetypes':ARCH,'archetypeRule':'first match: Trick Room move; weather-setting ability or Mega (Drizzle, Drought, Sand Stream, Snow Warning); Tailwind move; otherwise Other',
  'centroids':np.round(cent,4).tolist(),'order':[{'slot':c//8,'field':E.FIELDS[c%8]} for c in ORDER],
  'stepFormat':'[mean1, mean2, mean3, cov11, cov22, cov33, cov12, cov13, cov23, p(Trick Room), p(Weather), p(Tailwind), p(Other)] per step 0..48, LDA space',
  'featured':[int(i) for i in featured],'sourceHashes':{n:sha(REPO/'src'/n) for n in ['corpus.py','encode.py','diffusion.py','hpsdiffusion.py']},
  'records':[{'label':labels[i],'fields':[show(c,main[i][c]) for c in range(48)],'speciesId':[dec(j*8,main[i][j*8]) for j in range(6)],'steps':steps[i]} for i in range(N)]}
 (ROOT/'trace.json').write_text(json.dumps(data,separators=(',',':')))
 print('labels',np.bincount(labels,minlength=4).tolist(),'featured',featured,flush=True)
