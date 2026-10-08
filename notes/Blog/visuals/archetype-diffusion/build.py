"""Embed trace.json and the needed sprites into one self-contained HTML page."""
import json,base64
from pathlib import Path
from collections import Counter
ROOT=Path(__file__).resolve().parent
SPR=ROOT.parent/'search-loop'
t=json.loads((ROOT/'trace.json').read_text())
known=json.loads((SPR/'model/sprites.json').read_text())['sprites']

def sprite_file(sp):
 # Megas without their own gen5 sprite fall back to the base forme.
 for key in (sp,sp.split('mega')[0] if 'mega' in sp else sp):
  if key in known:return SPR/known[key]
 return None

recs=t['records']
# 3D discriminant space -> 2D: the linear map that best puts the four archetype centres
# at left / top / right / bottom. Linear, so means and covariances carry over exactly.
import numpy as np
Cn=np.array(t['centroids']);TGT=np.array([[-1,0],[0,1],[1,0],[0,-1]],float)*0.8
A=np.linalg.lstsq(Cn,TGT,rcond=None)[0].T
def to2d(st):
 m=A@np.array(st[:3]);c=st[3:9];S=np.array([[c[0],c[3],c[4]],[c[3],c[1],c[5]],[c[4],c[5],c[2]]]);S2=A@S@A.T
 l11=np.sqrt(max(S2[0,0],0));l21=S2[0,1]/l11 if l11>1e-9 else 0.;l22=np.sqrt(max(S2[1,1]-l21**2,0))
 return [m[0],m[1],l11,l21,l22,*st[9:13]]
cent2=(A@Cn.T).T
species=sorted({s for r in recs for s in r['speciesId']})
sprites={}
for sp in species:
 f=sprite_file(sp)
 if f:sprites[sp]='data:image/png;base64,'+base64.b64encode(f.read_bytes()).decode()
missing=[s for s in species if s not in sprites]

# Representative species per archetype: highest share among that archetype's teams relative to all teams.
alls=Counter(s for r in recs for s in set(r['speciesId']));reps=[]
for k in range(4):
 ks=[r for r in recs if r['label']==k];c=Counter(s for r in ks for s in set(r['speciesId']))
 score={s:(c[s]/len(ks))*(c[s]/alls[s]) for s in c if c[s]>=max(3,len(ks)//10) and s in sprites}
 reps.append(sorted(score,key=score.get,reverse=True)[:3])

method=f"""<p>Every state is a real point in the model's decode: the first <i>s</i> fields in dependency order are revealed, the rest masked. From that state the recorder draws {t['completionsPerState']} completions with the same constrained sampler ({t['completionsFullyMasked']} for the fully masked state, which all teams share).</p>
<p>Each completed team is projected with shrinkage linear discriminant analysis, fitted on {t['referenceTeams']} separate model samples using species, abilities, items and moves. Its three axes are viewed through one fixed linear 2D map, chosen so the four archetype centres sit left, top, right and bottom. A dot sits at the <b>mean</b> of its completions' positions. Its offset from that mean is a fixed random Gaussian direction per team, scaled by the completions' <b>standard deviation</b>: the cloud's width is the model's real uncertainty, and it reaches zero once the team is complete. Its colour is the share of completions in each archetype; grey means undecided.</p>
<p>Archetype rule, first match wins: a Trick Room user; else a weather setter (Drizzle, Drought, Sand Stream, Snow Warning, including Mega Charizard Y, Tyranitar, Froslass, Abomasnow); else a Tailwind user; else Other. Natures never change the label, so the last six steps do not move the dots.</p>
<p>{t['teams']} teams, seed {t['seedMain']}, unconditional, temperature 1. Stat Points are not sampled here. No battles were run: this shows what the model generates, not how well it plays.</p>"""

steps=[[round(float(v),3) for st in r['steps'] for v in to2d(st)] for r in recs]
data={'teams':[{'label':r['label'],'fields':r['fields'],'speciesId':r['speciesId'],'steps':s} for r,s in zip(recs,steps)],
 'order':t['order'],'archetypes':t['archetypes'],'centroids':np.round(cent2,3).tolist(),'featured':t['featured'],'sprites':sprites,
 'representatives':reps,'labelCounts':[sum(r['label']==k for r in recs) for k in range(4)],'methodHTML':method}
html=(ROOT/'template.html').read_text().replace('__DATA__',json.dumps(data,separators=(',',':')).replace('</','<\\/'))
(ROOT/'archetype-diffusion.html').write_text(html)
print('centroids 2D',np.round(cent2,2).tolist(),'fit residual',np.round(cent2-TGT,2).tolist());print('species',len(species),'missing sprites',missing,'reps',reps,'bytes',len(html))
