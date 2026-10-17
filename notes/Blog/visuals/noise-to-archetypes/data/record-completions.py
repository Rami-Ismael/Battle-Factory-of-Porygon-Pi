"""Dump the individual completions ("futures") that trace.json only summarises.

Loads ../../archetype-diffusion/record-archetypes.py as a module and calls its own
sample / complete / masked / feats / label / summary, so the checkpoint, sampler loop,
seeds, per-team generators, archetype rule and LABEL_STEPS are the recorder's, not copies.
The LDA block below is the recorder's main block, line for line.

Part 1: trace records 0, 246, 298, 86, 294, 59, 158 -> the same 24 completions per step
        s = 1..41 the original run drew, plus the shared 512 of the fully masked state.
Part 2: MB607 (teams/reg_mb/MB607.txt), slots in file order, revealed in D.ORDER,
        24 completions per step on a new seed, written in trace.json's record schema.

Writes completions.json and mb607-record.json next to this file. CPU only, no battles.
"""
from pathlib import Path
import sys,json,hashlib,time,importlib.util
import numpy as np
import torch
import multiprocessing as mp

HERE=Path(__file__).resolve().parent
SRC=HERE.parents[1]/'archetype-diffusion'
sys.dont_write_bytecode=True             # read-only on the recorder's folder: no __pycache__ beside it
spec=importlib.util.spec_from_file_location('record_archetypes',SRC/'record-archetypes.py')
R=importlib.util.module_from_spec(spec);sys.modules['record_archetypes']=R;spec.loader.exec_module(R)
assert (R.N,R.K,R.K0,R.REF)==(300,24,512,1500),'ARCH_SIZES must be unset: parity needs the recorded sizes'

RECORDS=[0,246,298,86,294,59,158]
FULL_SHEET={'0','MB607'}                 # teams whose futures also carry all 48 fields (`f`)
SEED_MB607=20261011                      # new: MB607's completions, one generator carried over s = 1..41
SEED_NATURE={'0':20261012,'MB607':20261013}   # new: natures of `f` only, separate generators
TOL=1e-4
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
show=lambda c,v:(R.look.get(R.dec(c,v),R.dec(c,v)) if v else None)

@torch.no_grad()
def natures(x,g):
 """R.complete's loop continued over the six nature steps (42..47). Its own generator,
 so the recorder's random stream is untouched; natures move neither label nor position."""
 x=x.clone();w=torch.full((len(x),),R.H.WNULL,dtype=torch.long)
 for step in range(R.LABEL_STEPS,len(R.ORDER)):
  c=R.ORDER[step];lg=R.H._step_logits(R.model,x,1.0-step/len(R.ORDER),w,1.)[c]
  for b in range(len(x)):
   ok=R.C.mask_for(c,x[b],'cpu');l=lg[b].clone();l[~ok]=-1e9
   x[b,c]=torch.multinomial(torch.softmax(l,-1),1,generator=g).item()
 return x

def worker(args):
 """record-archetypes.py's worker, returning the completions instead of their summary."""
 key,row,seed,nseed=args;torch.set_num_threads(1);row=torch.tensor(row,dtype=torch.long)
 g=torch.Generator().manual_seed(seed);g2=torch.Generator().manual_seed(nseed) if nseed is not None else None;out=[]
 for s in range(1,R.LABEL_STEPS):
  x=R.complete(R.masked(row,s),s,R.K,g)
  out.append((x.numpy(),natures(x,g2).numpy() if g2 is not None else None))
 return key,out

def trigger(row):
 """(label, slot of the first Pokemon that triggers it) under the recorder's rule; -1 for Other."""
 lab=R.label(row);has=lambda i,m:any(R.dec(i*8+j,row[i*8+j])==m for j in range(3,7))
 if lab==0:return lab,next(i for i in range(6) if has(i,'trickroom'))
 if lab==1:return lab,next(i for i in range(6) if R.dec(i*8,row[i*8]) in R.WEATHER_SP or R.dec(i*8+1,row[i*8+1]) in R.WEATHER_AB or R.dec(i*8+2,row[i*8+2]) in R.WEATHER_IT)
 if lab==2:return lab,next(i for i in range(6) if has(i,'tailwind'))
 return lab,-1

def trigger_field(row,lab,slot):
 """Column of the field on that slot that fires the label, first in decode order."""
 for c in R.ORDER[:R.LABEL_STEPS]:
  j=c%8;v=R.dec(c,row[c])
  if c//8!=slot:continue
  if lab==0 and 3<=j<=6 and v=='trickroom':return c
  if lab==1 and ((j==0 and v in R.WEATHER_SP) or (j==1 and v in R.WEATHER_AB) or (j==2 and v in R.WEATHER_IT)):return c
  if lab==2 and 3<=j<=6 and v=='tailwind':return c
 return None

def encode_mb607():
 """File slot order (no canonical sort); each slot through encode.team_fields, so its
 moves are in the encoder's alphabetical order, as in every team the model trained on."""
 path=R.CORPUS/'MB607.txt';team=R.mod.parse_team(path);fields=[]
 for slot in team:fields+=R.E.team_fields([slot])
 assert len(team)==6 and len(fields)==48
 where=lambda c:f"slot {c//8} ({team[c//8]['species']}) {R.E.FIELDS[c%8]} = {fields[c]!r}"
 oov=[where(c) for c in range(48) if fields[c] not in R.V.stoi[R.V.key(c)]]
 if oov:raise SystemExit('MB607 outside the vocabulary: '+'; '.join(oov))
 row=torch.tensor([R.V.stoi[R.V.key(c)][fields[c]] for c in range(48)],dtype=torch.long)
 x=torch.zeros(48,dtype=torch.long);bad=[]
 for c in R.ORDER:                      # would the constrained sampler be allowed to emit this value here?
  if not bool(R.C.mask_for(c,x,'cpu')[row[c]]):bad.append(where(c))
  x[c]=row[c]
 if bad:raise SystemExit('MB607 fails the sampler constraints: '+'; '.join(bad))
 viol=R.L.violations(fields)
 if viol:raise SystemExit(f'MB607 fails encode.Legality: {viol}')
 return row,path

if __name__=='__main__':
 t0=time.time();torch.set_num_threads(4)
 trace=json.loads((SRC/'trace.json').read_text());recs=trace['records']
 # 1. Reference sample -> archetype labels -> shrinkage LDA. Verbatim from record-archetypes.py.
 ref=R.sample(R.REF,R.SEED_REF);y=np.array([R.label(r) for r in ref]);X=R.feats(ref);mu=X.mean(0)
 Sw=sum(np.cov((X[y==k]-mu).T,bias=True)*(y==k).sum() for k in range(4))/len(X)+0.05*np.eye(X.shape[1])
 M=np.stack([X[y==k].mean(0)-mu for k in range(4)]);Sb=(M.T*np.bincount(y,minlength=4)/len(y))@M
 Lc=np.linalg.cholesky(Sw);Li=np.linalg.inv(Lc);ev,U=np.linalg.eigh(Li@Sb@Li.T);W=(Li.T@U[:,::-1][:,:3])
 P=(X-mu)@W;scale=np.percentile(np.abs(P),98,axis=0);W=W/scale
 cent=np.stack([((X[y==k]-mu)@W).mean(0) for k in range(4)])
 print('reference labels',np.bincount(y,minlength=4).tolist(),'centroids',cent.round(4).tolist(),flush=True)
 # 2. Main teams and the shared fully masked batch, same calls as the recorder.
 main=R.sample(R.N,R.SEED_MAIN)
 x0=R.complete(torch.zeros(48,dtype=torch.long),0,R.K0,torch.Generator().manual_seed(R.SEED_MAIN))
 mb,mbpath=encode_mb607()
 print('sampled reference, main, shared',round(time.time()-t0,1),'s',flush=True)
 # 3. Per-team completions, one spawned single-thread process each (as the recorder's pool).
 jobs=[(str(i),main[i].tolist(),R.SEED_MAIN+1+i,SEED_NATURE.get(str(i))) for i in RECORDS]+[('MB607',mb.tolist(),SEED_MB607,SEED_NATURE['MB607'])]
 with mp.get_context('spawn').Pool(len(jobs)) as pool:res=dict(pool.imap_unordered(worker,jobs))
 print('completions done',round(time.time()-t0,1),'s',flush=True)

 # 4. Parity against trace.json, on the unrounded completions (check-completions.py redoes it on the file).
 def diff(a,b):d=np.abs(np.array(a)-np.array(b));return [float(d[:3].max()),float(d[3:9].max()),float(d[9:].max())]
 s0=R.summary(x0,W,mu);par={'centroids':float(np.abs(np.round(cent,4)-np.array(trace['centroids'])).max()),
  'referenceLabelCounts':np.bincount(y,minlength=4).tolist()==trace['referenceLabelCounts'],'step0':diff(s0,recs[0]['steps'][0]),
  'mainTeams':sum([show(c,main[i][c]) for c in range(48)]==recs[i]['fields'] for i in range(R.N)),'records':{}}
 for i in RECORDS:
  d=np.array([diff(R.summary(torch.from_numpy(res[str(i)][s-1][0]),W,mu),recs[i]['steps'][s]) for s in range(1,R.LABEL_STEPS)])
  par['records'][str(i)]=d.max(0).tolist()
 parity=bool(par['centroids']<=TOL and par['referenceLabelCounts'] and max(par['step0'])<=TOL and par['mainTeams']==R.N and all(max(v)<=TOL for v in par['records'].values()))
 print('parity',parity,json.dumps(par),flush=True)

 # 5. Tables, then the two files.
 rows={k:[torch.from_numpy(a) for a,_ in v] for k,v in res.items()};full={k:[torch.from_numpy(b) for _,b in res[k]] for k in FULL_SHEET}
 own={str(i):main[i] for i in RECORDS};own['MB607']=mb
 species=sorted({R.dec(j*8,r[j*8]) for rs in [[x0],*rows.values(),[torch.stack(list(own.values()))]] for x in rs for r in x for j in range(6)});sidx={s:n for n,s in enumerate(species)}
 vocab=sorted({show(c,r[c]) for k in FULL_SHEET for x in full[k] for r in x for c in range(48)});assert None not in vocab;vidx={s:n for n,s in enumerate(vocab)}
 r4=lambda a:(np.round(a,4)+0.0).tolist()
 def pack(x,xf=None):
  tr=[trigger(r) for r in x];d={'p':r4((R.feats(x)-mu)@W),'lab':[t[0] for t in tr],'sp':[[sidx[R.dec(j*8,r[j*8])] for j in range(6)] for r in x],'ev':[t[1] for t in tr]}
  if xf is not None:assert torch.equal(xf[:,R.ORDER[:R.LABEL_STEPS]],x[:,R.ORDER[:R.LABEL_STEPS]]);d['f']=[[vidx[show(c,r[c])] for c in range(48)] for r in xf]
  return d
 names={'0':'Team #1','MB607':'MB607',**{str(i):f'Followed team {n+1}' for n,i in enumerate(trace['featured'])}}
 Cn=np.round(cent,4);TGT=np.array([[-1,0],[0,1],[1,0],[0,-1]],float)*0.8;A=np.linalg.lstsq(Cn,TGT,rcond=None)[0].T   # build.py's fit
 teams={}
 for k,row in own.items():
  teams[k]={'name':names[k],'fields':[show(c,row[c]) for c in range(48)],'speciesId':[R.dec(j*8,row[j*8]) for j in range(6)],'label':R.label(row),
   'steps':{str(s):pack(rows[k][s-1],full[k][s-1] if k in FULL_SHEET else None) for s in range(1,R.LABEL_STEPS)}}
 sh=pack(x0);del sh['ev']
 # MB607 in trace.json's records[i] schema: shared step 0, 41 sampled steps, then the finished team.
 fin=R.summary(mb.unsqueeze(0),W,mu);lab,slot=trigger(mb)
 hit=trigger_field(mb,lab,slot)
 mbrec={'label':teams['MB607']['label'],'fields':teams['MB607']['fields'],'speciesId':teams['MB607']['speciesId'],
  'steps':[s0]+[R.summary(x,W,mu) for x in rows['MB607']]+[fin]*(len(R.ORDER)-R.LABEL_STEPS+1),'order':trace['order'],'stepFormat':trace['stepFormat'],
  'meta':{'name':'MB607','source':str(mbpath),'sourceSha256':sha(mbpath),'seed':SEED_MB607,'completionsPerState':R.K,'archetype':R.ARCH[lab],
   'trigger':None if hit is None else {'slot':slot,'field':R.E.FIELDS[hit%8],'value':show(hit,mb[hit]),'revealedAtStep':R.ORDER.index(hit)+1},
   'note':'Slots in file order; moves within a slot in the encoder order (alphabetical), as in training. Step 0 is the shared 512-completion row of trace.json; steps 1-41 are 24 completions each; steps 42-48 are the finished team. Stat Points are not sampled.'}}
 note=('Parity with trace.json holds: these are the completions the original run drew.' if parity else 'PARITY FAILED: these completions are NOT the ones summarised in trace.json; see parityDetail and `rows`.')+' `ev` takes the lowest slot when several qualify. `f` natures come from the same loop continued over steps 42-47 on separate generators (seeds.natures); they change no label or position. Stat Points are not sampled.'
 data={'meta':{'checkpoint':str(R.CK),'checkpointSha256':sha(R.CK),'recorderSha256':sha(SRC/'record-archetypes.py'),'traceSha256':sha(SRC/'trace.json'),
   'seeds':{'reference':R.SEED_REF,'main':R.SEED_MAIN,'sharedFullyMasked':R.SEED_MAIN,'perRecord':'main + 1 + record index','MB607':SEED_MB607,'natures':SEED_NATURE},
   'completionsPerState':R.K,'completionsFullyMasked':R.K0,'labelSteps':R.LABEL_STEPS,'parity':parity,'parityDetail':par,'tolerance':TOL,
   'sourceHashes':{n:sha(R.REPO/'src'/n) for n in ['corpus.py','encode.py','diffusion.py','hpsdiffusion.py']},'torch':torch.__version__,'numpy':np.__version__,
   'wallClockSeconds':round(time.time()-t0,1),'note':note},
  'archetypes':R.ARCH,'centroids3d':r4(cent),'map2d':{'matrix':A.tolist(),'offset':[0.0,0.0],'use':'xy = matrix @ xyz + offset; least-squares fit of the four rounded centroids3d onto 0.8 * [left, top, right, bottom], as in build.py'},
  'centroids2d':r4((A@Cn.T).T),'speciesTable':species,'vocabTable':vocab,'shared0':sh,'teams':teams}
 if not parity:                          # self-consistent summaries for pages, in trace.json's step format
  for k in teams:teams[k]['rows']=[s0]+[R.summary(x,W,mu) for x in rows[k]]+[R.summary(own[k].unsqueeze(0),W,mu)]*(len(R.ORDER)-R.LABEL_STEPS+1)
 (HERE/'completions.json').write_text(json.dumps(data,separators=(',',':')))
 (HERE/'mb607-record.json').write_text(json.dumps(mbrec,separators=(',',':')))
 print('wrote completions.json',(HERE/'completions.json').stat().st_size,'bytes, mb607-record.json',(HERE/'mb607-record.json').stat().st_size,'bytes;',round(time.time()-t0,1),'s',flush=True)
