"""Acceptance test for completions.json and mb607-record.json, read back from disk.

Recomputes each state's mean, population covariance (divide by 24) and archetype shares
from the dumped completions and compares them with ../../archetype-diffusion/trace.json.
Needs numpy only: no model, no torch. Exits 1 if anything fails.
"""
from pathlib import Path
import json,re,sys
import numpy as np
HERE=Path(__file__).resolve().parent;SRC=HERE.parents[1]/'archetype-diffusion'
T=json.loads((SRC/'trace.json').read_text());C=json.loads((HERE/'completions.json').read_text());M=json.loads((HERE/'mb607-record.json').read_text())
TOL=1e-4;RECORDS=['0','246','298','86','294','59','158'];FIELDS=['species','ability','item','m0','m1','m2','m3','nature']
ORDER=[o['slot']*8+FIELDS.index(o['field']) for o in T['order']];norm=lambda s:re.sub(r'[^a-z0-9]','',s.lower())
fails=[]
def ok(name,cond,detail=''):
 print(('PASS ' if cond else 'FAIL ')+name+(' — '+str(detail) if detail!='' else ''))
 if not cond:fails.append(name)
def summ(p,lab):
 P=np.array(p);S=np.cov(P.T,bias=True);return [*P.mean(0),S[0,0],S[1,1],S[2,2],S[0,1],S[0,2],S[1,2],*(np.bincount(lab,minlength=4)/len(lab))]
def worst(rows_a,rows_b):
 d=np.abs(np.array(rows_a)-np.array(rows_b));return [float(d[:,:3].max()),float(d[:,3:9].max()),float(d[:,9:].max())]

print(f'tolerance {TOL}; worst absolute difference vs trace.json over steps 1..41, as [mean, covariance, shares]')
for k in RECORDS:
 st=C['teams'][k]['steps'];assert sorted(st,key=int)==[str(s) for s in range(1,42)] and all(len(st[s]['p'])==24 for s in st)
 w=worst([summ(st[str(s)]['p'],st[str(s)]['lab']) for s in range(1,42)],[T['records'][int(k)]['steps'][s] for s in range(1,42)])
 ok(f'record {k:>3}',max(w)<=TOL,[f'{v:.2e}' for v in w])
 r=T['records'][int(k)];ok(f'record {k:>3} fields/speciesId/label equal trace',(C['teams'][k]['fields'],C['teams'][k]['speciesId'],C['teams'][k]['label'])==(r['fields'],r['speciesId'],r['label']))
s0=C['shared0'];cnt=np.bincount(s0['lab'],minlength=4).tolist()
ok('step 0 counts are 178/153/83/98 of 512',cnt==[178,153,83,98] and len(s0['p'])==512,cnt)
w=worst([summ(s0['p'],s0['lab'])],[T['records'][0]['steps'][0]]);ok('step 0 mean/covariance/shares',max(w)<=TOL,[f'{v:.2e}' for v in w])
ok('all 300 trace records share that step-0 row',all(r['steps'][0]==T['records'][0]['steps'][0] for r in T['records']))
d=float(np.abs(np.array(C['centroids3d'])-np.array(T['centroids'])).max());ok('centroids3d equal trace centroids',d<=TOL,f'{d:.2e}')
A=np.linalg.lstsq(np.array(T['centroids']),np.array([[-1,0],[0,1],[1,0],[0,-1]],float)*0.8,rcond=None)[0].T;A2=np.array(C['map2d']['matrix'])
d=float(np.abs(A-A2).max());ok('map2d equals the build.py fit on trace centroids',d<=1e-12 and C['map2d']['offset']==[0,0],f'{d:.2e}')
d=float(np.abs((A@np.array(T['centroids']).T).T-np.array(C['centroids2d'])).max());ok('centroids2d',d<=TOL,f'{d:.2e}')
ok('meta: parity flag, 24 and 512',C['meta']['parity'] is True and C['meta']['completionsPerState']==24 and C['meta']['completionsFullyMasked']==512)
ok('meta: checkpoint hash equals trace',C['meta']['checkpointSha256']==T['checkpointSha256'])

# Structure of every team, MB607 included.
sp,vt=C['speciesTable'],C['vocabTable'];bad=[];blank={}
for k,t in C['teams'].items():
 for s in range(1,42):
  e=t['steps'][str(s)]
  if not (len(e['p'])==len(e['lab'])==len(e['sp'])==len(e['ev'])==24):bad.append((k,s,'length'))
  for b in range(24):
   if [sp[i] for i in e['sp'][b]][:min(s,6)]!=t['speciesId'][:min(s,6)]:bad.append((k,s,b,'revealed species changed'))
   if (e['ev'][b]==-1)!=(e['lab'][b]==3) or not -1<=e['ev'][b]<=5:bad.append((k,s,b,'ev/lab'))
   if ('f' in e)!=(k in ('0','MB607')):bad.append((k,s,'f presence'))
   if 'f' in e:
    f=[vt[i] for i in e['f'][b]]
    if any(f[ORDER[i]]!=t['fields'][ORDER[i]] for i in range(s)):bad.append((k,s,b,'revealed field changed'))
    if [norm(f[j*8]) for j in range(6)]!=[sp[i] for i in e['sp'][b]]:bad.append((k,s,b,'f species != sp'))
    if any(not v for c,v in enumerate(f) if c%8!=7):bad.append((k,s,b,'empty field in f'))
    blank[k]=blank.get(k,0)+sum(not f[c] for c in range(7,48,8))
ok('structure: 8 teams x 41 steps x 24, revealed fields kept, ev matches lab, f only on 0 and MB607',not bad and list(C['teams'])==RECORDS+['MB607'],bad[:5])
print('     note: "" in a nature column of f is the vocabulary token for a set with no nature line; count per team',blank,'| trace.json itself has',sum(v=='' for r in T['records'] for v in r['fields']))
size=(HERE/'completions.json').stat().st_size;ok('completions.json under 3 MB',size<3e6,f'{size} bytes')
known=json.loads((HERE.parents[1]/'search-loop/model/sprites.json').read_text())['sprites']
miss=[s for s in sp if s not in known and (s.split('mega')[0] if 'mega' in s else s) not in known];ok('every speciesTable key has a sprite (build.py fallback)',not miss,miss)

# MB607 record.
mt=C['teams']['MB607'];st=M['steps']
ok('MB607 record keys',set(M)=={'label','fields','speciesId','steps','order','stepFormat','meta'} and M['order']==T['order'] and M['stepFormat']==T['stepFormat'])
ok('MB607 record: 49 rows of 13, 48 fields, 6 species',len(st)==49 and all(len(r)==13 for r in st) and len(M['fields'])==48 and len(M['speciesId'])==6 and all(M['fields']))
ok('MB607 record equals its completions.json entry',(M['fields'],M['speciesId'],M['label'])==(mt['fields'],mt['speciesId'],mt['label']))
ok('MB607 step 0 is the shared trace row',st[0]==T['records'][0]['steps'][0])
w=worst([summ(mt['steps'][str(s)]['p'],mt['steps'][str(s)]['lab']) for s in range(1,42)],st[1:42]);ok('MB607 steps 1..41 equal its dumped completions',max(w)<=TOL,[f'{v:.2e}' for v in w])
one=[float(i==M['label']) for i in range(4)];ok('MB607 steps 42..48 are the finished team',all(r==st[42] and r[3:9]==[0]*6 and r[9:]==one for r in st[42:]))

print('\nMB607:',M['meta']['archetype'],'| trigger',M['meta']['trigger'],'| seed',M['meta']['seed'])
print(' fields:',' / '.join(' '.join(str(v) for v in M['fields'][j*8:j*8+8]) for j in range(6)))
print(' step  TR  We  Tw  Ot     x2d      y2d')
for s in [0,1,2,3,4,5,6,12,18,24,30,36,41,42]:
 n=512 if s==0 else 24 if s<42 else 1;c=[round(v*n) for v in st[s][9:]];xy=A2@np.array(st[s][:3])
 print(f' {s:>4} {c[0]:>3} {c[1]:>3} {c[2]:>3} {c[3]:>3}  {xy[0]:+.4f}  {xy[1]:+.4f}'+('   (of 512)' if s==0 else '   (finished team)' if s==42 else ''))
lock=next(s for s in range(1,50) if all(st[q][9+M['label']]==1.0 for q in range(s,49)));print(' final archetype holds every completion from step',lock,'to the end')
print(' steps 1..41 where it does not:',[s for s in range(1,42) if st[s][9+M['label']]<1.0])
print('\n'+('ALL PASS' if not fails else f'{len(fails)} FAILED: {fails}'));sys.exit(1 if fails else 0)
