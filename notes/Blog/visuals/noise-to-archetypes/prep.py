"""Write data/figdata.js (window.FIGDATA) and data/sprites.js (window.SPRITES) for build.py.

Reads ../archetype-diffusion/trace.json (300 teams), data/completions.json and data/mb607-record.json
(MB607's futures), and sprites from ../search-loop. Needs numpy."""
import json,base64
from pathlib import Path
import numpy as np
H=Path(__file__).resolve().parent; W=H; V=H.parent
t=json.loads((V/'archetype-diffusion/trace.json').read_text())
c=json.loads((W/'data/completions.json').read_text())
mb=json.loads((W/'data/mb607-record.json').read_text())
A=np.array(c['map2d']['matrix'])
def st2(st,n):
 m=A@np.array(st[:3]);q=st[3:9];S=np.array([[q[0],q[3],q[4]],[q[3],q[1],q[5]],[q[4],q[5],q[2]]]);S2=A@S@A.T
 return [round(float(m[0]),4),round(float(m[1]),4),round(float(S2[0,0]),5),round(float(S2[1,1]),5),round(float(S2[0,1]),5)]+[int(round(p*n)) for p in st[9:13]]
def rec(r):return {'label':r['label'],'speciesId':r['speciesId'],'fields':r['fields'],'steps':[st2(s,512 if i==0 else 24) for i,s in enumerate(r['steps'])]}
ST=c['speciesTable']
def fut(step):return [{'x':round(float((A@np.array(p))[0]),4),'y':round(float((A@np.array(p))[1]),4),'lab':l,'ev':e,'sp':[ST[k] for k in s],'f':[c['vocabTable'][k] for k in f]} for p,l,s,e,f in zip(step['p'],step['lab'],step['sp'],step['ev'],step['f'])]
m=rec(mb);T=c['teams']['MB607']
m['name']='MB607';m['futures']={s:fut(v) for s,v in T['steps'].items()}
s0=c['shared0'];m['futures0']=[{'x':round(float((A@np.array(p))[0]),4),'y':round(float((A@np.array(p))[1]),4),'lab':l,'sp':[ST[k] for k in sp]} for p,l,sp in zip(s0['p'],s0['lab'],s0['sp'])]
D={'note':'Step row = [x,y,varX,varY,covXY,countTR,countWeather,countTailwind,countOther]; counts out of 512 at step 0, else 24 (finished teams: exactly one count is nonzero... see label). Steps 0..48; 1-6 species, 7-18 ability/item, 19-42 moves, 43-48 natures (move nothing).',
 'archetypes':c['archetypes'],'centroids':c['centroids2d'],'order':t['order'],'featured':t['featured'],
 'teams':[rec(r) for r in t['records']],'mb607':m}
species=sorted({s for r in t['records'] for s in r['speciesId']}|set(mb['speciesId'])|{s for v in m['futures'].values() for f in v for s in f['sp']}|{s for f in m['futures0'] for s in f['sp']})
known=json.loads((V/'search-loop/model/sprites.json').read_text())['sprites']
spr={}
for sp in species:
 for k in (sp,sp.split('mega')[0]):
  if k in known: spr[sp]='data:image/png;base64,'+base64.b64encode((V/'search-loop'/known[k]).read_bytes()).decode();break
(H/'data/figdata.js').write_text('window.FIGDATA='+json.dumps(D,separators=(',',':'))+';')
(H/'data/sprites.js').write_text('window.SPRITES='+json.dumps(spr,separators=(',',':'))+';')
print('species',len(species),'missing',[s for s in species if s not in spr],'figdata KB',(H/'data/figdata.js').stat().st_size//1024,'sprites KB',(H/'data/sprites.js').stat().st_size//1024)
print('MB607 step6',m['steps'][6],'step41',m['steps'][41])
