"""Embed trace.json and the needed sprites into one self-contained HTML page."""
import json,base64
from pathlib import Path
from collections import Counter
from itertools import combinations
from strategies import classify,title
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
sprites={};sprite_sources={}
for sp in species:
 f=sprite_file(sp)
 if f:
  sprites[sp]='data:image/png;base64,'+base64.b64encode(f.read_bytes()).decode()
  sprite_sources[sp]=f.stem
missing=[s for s in species if s not in sprites]
assert not missing, f'Missing sprites: {missing}'

# Representative species per archetype: highest share among that archetype's teams relative to all teams.
alls=Counter(s for r in recs for s in set(r['speciesId']));reps=[]
for k in range(4):
 ks=[r for r in recs if r['label']==k];c=Counter(s for r in ks for s in set(r['speciesId']))
 score={s:(c[s]/len(ks))*(c[s]/alls[s]) for s in c if c[s]>=max(3,len(ks)//10) and s in sprites}
 reps.append(sorted(score,key=score.get,reverse=True)[:3])

# Human-readable identities are descriptive, never invented tournament results.
# Prefer a less common pair; select another real pair if that name is already used.
used_names=set();names=[]
for r in recs:
 members={sp:r['fields'][slot*8] for slot,sp in enumerate(r['speciesId'])}
 pairs=sorted(combinations(sorted(members),2),key=lambda pair:(sum(alls[s] for s in pair),pair))
 for pair in pairs:
  name=' + '.join(members[s] for s in pair)
  if name not in used_names:
   used_names.add(name);names.append(name);break
 else:raise ValueError('No unique composition pair remains; add evidence-based set descriptors instead of numeric aliases.')

method=f"""<p><b>Actual model states.</b> This is a recorded CPU run of a masked-diffusion-trained Transformer, decoded in 48 dependency-ordered categorical draws. At state <i>s</i>, the first <i>s</i> fields are known. This sampler starts with MASK tokens, not Gaussian noise. Nothing is resampled in the browser.</p>
<p><b>One fixed map.</b> Species, abilities, items and moves become an unordered multi-hot vector. Shrinkage LDA (λ = 0.05) is fitted once on {t['referenceTeams']:,} separate model samples and the rule labels below. All three discriminant axes feed one fixed linear 2D map, fitted toward left / top / right / bottom anchors. Labels, axes and scale never refit during playback.</p>
<p><b>Partly masked teams.</b> For each state, {t['completionsPerState']} conditional completions estimate a projected mean μ and covariance Σ ({t['completionsFullyMasked']} draws for the shared fully masked state). The animated dot is μ + Lz, with LLᵀ = Σ and one seeded Gaussian vector z per team, reused at every step. Mean-only view removes this illustrative offset. A mean does not necessarily correspond to a valid team. Covariance need not shrink monotonically; 24 completions give a noisy estimate.</p>
<p><b>What motion means.</b> Integer steps show recorded summaries. Smooth movement between them interpolates coordinates and covariance factors for continuity; it is not an observed model trajectory. Composition and completion shares change only at integer steps. Natures are excluded from the projection, so steps 42–48 do not move the dots. The Gaussian approximation can hide multiple modes and is not a posterior sample or calibrated confidence region.</p>
<p><b>Why these labels?</b> First match wins: a Trick Room move; else a weather-setting ability (Drizzle, Drought, Sand Stream, Snow Warning) or weather Mega species / stone; else Tailwind; else Other. Hybrid teams are common: the selected-team evidence lists all detected cues. These are rule-based buckets, not battle-validated strategies. Other means no matching cue. Supervised LDA deliberately emphasizes these labels: separation is not evidence of naturally discovered clusters, strategic similarity or team quality.</p>
<p><b>Provenance.</b> {t['teams']} teams; main seed {t['seedMain']}; reference seed {t['seedReference']}; unconditional, temperature 1, guidance 1. Checkpoint <code>temperature_p0.pt</code>, SHA-256 <code class="hash">{t['checkpointSha256']}</code>. Sprites are embedded from the repository’s Pokémon Showdown sprite assets; unavailable Mega artwork uses a disclosed base-form sprite. Stat Points are absent and no battles or full-team legality validation were run for this figure.</p>"""

steps=[[round(float(v),3) for st in r['steps'] for v in to2d(st)] for r in recs]
data={'teams':[{'name':name,'modes':classify(r),'strategyTitle':title(classify(r)),'label':r['label'],'fields':r['fields'],'speciesId':r['speciesId'],'steps':s} for r,s,name in zip(recs,steps,names)],
 'order':t['order'],'archetypes':t['archetypes'],'centroids':np.round(cent2,3).tolist(),'featured':t['featured'],'sprites':sprites,
 'representatives':reps,'labelCounts':[sum(r['label']==k for r in recs) for k in range(4)],'methodHTML':method,
 'spriteSources':sprite_sources,'projectionMatrix':A.tolist(), 'checkpointSha256':t['checkpointSha256'],
 'completionsPerState':t['completionsPerState'],'completionsFullyMasked':t['completionsFullyMasked']}
html=(ROOT/'template.html').read_text().replace('__DATA__',json.dumps(data,separators=(',',':')).replace('</','<\\/'))
(ROOT/'archetype-diffusion.html').write_text(html)
print('centroids 2D',np.round(cent2,2).tolist(),'fit residual',np.round(cent2-TGT,2).tolist());print('species',len(species),'missing sprites',missing,'reps',reps,'bytes',len(html))
