"""Recompute exact integer construction counts; render explanatory paper figures.

Run extract.cjs first. Requires matplotlib and Pillow. No external assets.
"""
from pathlib import Path
from collections import defaultdict
from math import comb, perm, log10
import json
import itertools
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, PathPatch
from matplotlib.path import Path as MPath

ROOT=Path(__file__).resolve().parent
D=json.loads((ROOT/'domain.json').read_text())
ROSTER=D['roster']

def elementary(weights,k=6):
    dp=[1]+[0]*k
    for weight in weights:
        for i in range(k,0,-1): dp[i]+=weight*dp[i-1]
    return dp[k]

def roster_count(weight):
    groups=defaultdict(int)
    for r in ROSTER: groups[r['num']]+=weight(r)
    return elementary(groups.values())

def spreads(total=66,cap=32,exact=True):
    # Coefficient count by independent dynamic programming.
    dp=[1]+[0]*total
    for _ in range(6):
        dp=[sum(dp[max(0,i-cap):i+1]) for i in range(total+1)]
    value=dp[total] if exact else sum(dp)
    # Independent inclusion-exclusion cross-check.
    degree=5 if exact else 6
    closed=sum((-1)**k*comb(6,k)*comb(total-k*(cap+1)+degree,degree)
               for k in range(min(6,total//(cap+1))+1))
    assert value==closed
    return value

N=len(set(r['num'] for r in ROSTER)); ITEMS=len(D['items']); A=len(D['alignments'])
ITEM_FACTOR=sum(comb(6,k)*perm(ITEMS,k) for k in range(7))
FORM=roster_count(lambda r:1)
ABILITY=roster_count(lambda r:len(r['pairs']))
MOVE=roster_count(lambda r:sum(comb(len(p['moves']),min(4,len(p['moves']))) for p in r['pairs']))
SP=spreads(); SP_PARTIAL=spreads(exact=False)
NUMBERS=[comb(N,6),FORM,ABILITY,ABILITY*ITEM_FACTOR,MOVE*ITEM_FACTOR,MOVE*ITEM_FACTOR*A**6,MOVE*ITEM_FACTOR*A**6*SP**6]
PARTIAL=NUMBERS[-1]//SP**6*SP_PARTIAL**6

# Meaningful small-domain verification: DP equals direct enumeration with forms.
toy=[[2,3],[7],[11,13],[17]]
assert elementary([sum(g) for g in toy],2)==sum(sum(toy[i])*sum(toy[j]) for i,j in itertools.combinations(range(4),2))
for n in range(2,6):
    for k in range(1,4):
        brute=sum(len([x for x in tup if x>=0])==len(set(x for x in tup if x>=0)) for tup in itertools.product(range(-1,n),repeat=k))
        assert brute==sum(comb(k,h)*perm(n,h) for h in range(k+1))
assert not D['validation']['itemErrors'] and not D['validation']['teamErrors']
assert all(r['sourceKinds']==['9M'] for r in ROSTER)
assert all(len(p['moves'])>=4 or r['id']=='ditto' for r in ROSTER for p in r['pairs'])

LABELS=['Six distinct species','Choose their forms','Add abilities','Assign held items','Choose movesets','Choose stat alignments','Allocate Stat Points']
DETAILS=[f'Choose 6 from {N}; roster order excluded',f'{len(ROSTER)} form representatives; one per species',
         'Use each form’s eligible abilities','148 items + none; no duplicate held items',
         'Four distinct moves per member; Ditto has Transform',
         '21 distinct alignments per member','10,008,272 fully spent spreads per member']
result={'format':D['format'],'revision':D['revision'],'species':N,'forms':len(ROSTER),'items':ITEMS,'alignments':A,
        'itemAssignments':str(ITEM_FACTOR),'fullySpentSpreads':SP,'partialSpreads':SP_PARTIAL,
        'stages':[{'name':s,'count':str(n),'log10':log10(n)} for s,n in zip(LABELS,NUMBERS)],
        'withPartialSpreads':str(PARTIAL),'scope':'Conservative model: direct 9M moves, exactly four where possible, 66 SP spent, fixed gender/defaults; ordering and cosmetic variants omitted.'}
(ROOT/'counts.json').write_text(json.dumps(result,indent=2)+'\n')

BG='#F8F7F2'; INK='#172C35'; MUTED='#52656B'; GRID='#D8DEDB'; TEAL='#007E80'; LIME='#DAE9CB'; PALE='#E5EFEB'; ORANGE='#BC603C'
plt.rcParams.update({'font.family':'Avenir Next','font.size':11,'text.color':INK,'axes.labelcolor':MUTED,
                     'xtick.color':MUTED,'ytick.color':MUTED,'svg.fonttype':'none','pdf.fonttype':42,
                     'mathtext.fontset':'dejavusans','figure.facecolor':BG,'axes.facecolor':BG})

def scientific(n,digits=2):
    e=int(log10(n));return rf'${n/10**e:.{digits}f}\times10^{{{e}}}$'

def text(ax,x,y,s,size=11,color=INK,weight='normal',ha='left',va='center',**kw):
    return ax.text(x,y,s,fontsize=size,color=color,weight=weight,ha=ha,va=va,**kw)

def box(ax,x,y,w,h,title,subtitle=None,selected=False,size=11):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.012,rounding_size=0.09',
                 linewidth=1.05,edgecolor=TEAL if selected else GRID,facecolor=PALE if selected else BG,zorder=3))
    text(ax,x+w/2,y+h*(.64 if subtitle else .5),title,size,weight='demibold' if selected else 'normal',ha='center',zorder=4)
    if subtitle:text(ax,x+w/2,y+h*.28,subtitle,9.5,MUTED,ha='center',zorder=4)

def link(ax,x1,y1,x2,y2,highlight=False):
    mid=(x1+x2)/2
    p=MPath([(x1,y1),(mid,y1),(mid,y2),(x2,y2)],[MPath.MOVETO,MPath.CURVE4,MPath.CURVE4,MPath.CURVE4])
    ax.add_patch(PathPatch(p,fill=False,linewidth=1.6 if highlight else .9,edgecolor=TEAL if highlight else GRID,zorder=1))

def tree(ax):
    ax.set(xlim=(-.06,16),ylim=(0,6.25));ax.axis('off')
    text(ax,0,6.0,'A  /  INSIDE ONE OF SIX TEAM MEMBERS',11,TEAL,'demibold')
    text(ax,0,5.56,'Choosing a species opens another search space.',22,weight='demibold')
    for x,s in [(0,'SPECIES'),(3.0,'ABILITY'),(6.15,'ITEM'),(9.35,'MOVESET')]:text(ax,x,4.86,s,9.5,MUTED,'demibold')
    box(ax,0,2.60,2.10,.95,'Garchomp','Example member',True,15)
    for y,name,sel in [(3.88,'Sand Veil',False),(2.08,'Rough Skin',True)]:
        box(ax,3,y,2.02,.68,name,selected=sel)
        link(ax,2.1,3.075,3,y+.34,sel)
    # Other ability branch also expands, but only one path is expanded in detail.
    for y in [4.48,4.10,3.72]:link(ax,5.02,4.22,5.74,y)
    text(ax,5.86,4.10,'…',18,MUTED)
    for y,name,sel in [(3.0,'Clear Amulet',True),(1.98,'Life Orb',False),(.96,'No item',False)]:
        box(ax,6.15,y,2.08,.66,name,selected=sel)
        link(ax,5.02,2.42,6.15,y+.33,sel)
    text(ax,6.15,.47,'+146 other items',10,MUTED)
    # Move sets on the illustrated branch are checked against the actual snapshot.
    g=next(r for r in ROSTER if r['id']=='garchomp')
    pool=set(g['pairs'][0]['moves'])
    sets=[['earthquake','dragonclaw','protect','swordsdance'],['stompingtantrum','rockslide','protect','swordsdance']]
    assert all(set(s)<=pool for s in sets)
    for y,title in [(3.23,'Earthquake · Dragon Claw\nProtect · Swords Dance'),(1.77,'Stomping Tantrum · Rock Slide\nProtect · Swords Dance')]:
        box(ax,9.35,y,4.60,.94,title,selected=y>3)
        link(ax,8.23,3.33,9.35,y+.47,y>3)
    link(ax,8.23,3.33,9.35,1.02)
    text(ax,9.52,1.02,'… 424,268 more four-move sets',11,MUTED)
    text(ax,14.27,3.7,'…',24,TEAL)
    text(ax,9.35,.45,'58 eligible moves yield 424,270 four-move sets',10,TEAL,'demibold')
    # Compact continuation makes the suppressed branching explicit.
    text(ax,0,-.02,'EVERY PATH CONTINUES',9.5,MUTED,'demibold')
    text(ax,3.0,-.02,'21 stat alignments',12,weight='demibold')
    text(ax,6.12,-.02,'×',14,MUTED)
    text(ax,6.62,-.02,'10,008,272 Stat Point spreads',12,weight='demibold')
    text(ax,12.40,-.02,'Repeat for all six members',11,TEAL,'demibold')

def growth(ax):
    # Coordinates on x are log10(number of configurations); every +20 = ×10^20.
    ax.set_xlim(0,122); ax.set_ylim(-.90,7.25)
    ys=list(reversed(range(7)))
    ax.set_yticks([])
    ax.set_xticks([0,20,40,60,80,100,120],[r'$1$',r'$10^{20}$',r'$10^{40}$',r'$10^{60}$',r'$10^{80}$',r'$10^{100}$',r'$10^{120}$'])
    ax.tick_params(axis='x',length=0,pad=12,labelsize=10)
    for spine in ax.spines.values():spine.set_visible(False)
    for x in [0,20,40,60,80,100,120]:ax.plot([x,x],[-.48,6.45],color=GRID,lw=.7,zorder=0)
    for i,(y,n,label,detail) in enumerate(zip(ys,NUMBERS,LABELS,DETAILS)):
        value=log10(n);ax.barh(y,value,height=.24,color=TEAL if i==6 else '#92B5AD',zorder=2)
        ax.scatter([value],[y],s=30,color=TEAL if i==6 else '#548F86',zorder=3)
        text(ax,-44,y+.08,label,12,weight='demibold',clip_on=False)
        text(ax,-44,y-.22,detail,8.9,MUTED,clip_on=False)
        text(ax,value+1.5,y,scientific(n),12,weight='demibold' if i==6 else 'normal')
    text(ax,0,7.03,'B  /  HOW THE NUMBER OF CONFIGURATIONS GROWS',11,TEAL,'demibold')
    ax.set_xlabel('Cumulative team configurations  ·  logarithmic scale',labelpad=15,fontsize=11)
    text(ax,0,-.73,'Each row keeps the earlier choices and adds one more configuration layer.',10,MUTED)

def save(fig,name):
    for ext in ['png','svg','pdf']:
        fig.savefig(ROOT/f'{name}.{ext}',dpi=180,facecolor=BG,bbox_inches=None)
    plt.close(fig)

def footer(fig,y=.027):
    fig.text(.06,y,'COUNTING SCOPE   Six distinct species; four moves where possible; all 66 Stat Points spent; 32 per-stat cap.',fontsize=9,color=MUTED)
    fig.text(.06,y-.016,'Fixed gender/defaults. Roster order, move order and cosmetic variants excluded. Frozen Showdown M-B revision 913da3602a3a.',fontsize=9,color=MUTED)

# Main figure: one integrated page, suitable for the introduction or search-space section.
fig=plt.figure(figsize=(18,16),facecolor=BG)
fig.text(.06,.955,'POKÉMON TEAM BUILDING  /  SEARCH-SPACE ATLAS',fontsize=11,color=TEAL,weight='demibold')
fig.text(.06,.915,'Six species are only the beginning.',fontsize=34,weight='demibold')
fig.text(.06,.881,'A conservative construction model reaches',fontsize=17,color=MUTED)
fig.text(.475,.880,scientific(NUMBERS[-1])+' configurations.',fontsize=23,weight='demibold',color=TEAL)
fig.text(.06,.853,'Pokémon Champions · Regulation M-B · simulator-defined subset, not a count of strategically distinct or strong teams',fontsize=11,color=MUTED)
tree(fig.add_axes([.06,.465,.88,.355]))
growth(fig.add_axes([.305,.145,.635,.280]))
fig.text(.06,.079,'Even with the same species, the configuration choices dominate the count.',fontsize=17,weight='demibold')
footer(fig,.049)
save(fig,'team-search-space')

fig=plt.figure(figsize=(18,8.8),facecolor=BG)
tree(fig.add_axes([.06,.15,.88,.76]))
fig.text(.06,.055,'Branching illustration: only a few paths are shown. Item choices are coupled across the six members by Item Clause.',fontsize=11,color=MUTED)
save(fig,'decision-tree')

fig=plt.figure(figsize=(18,10.6),facecolor=BG)
fig.text(.06,.93,'From six species to a fully configured team',fontsize=29,weight='demibold')
fig.text(.06,.885,'Conservative construction model · Champions Regulation M-B · all 66 Stat Points spent',fontsize=12,color=MUTED)
growth(fig.add_axes([.305,.195,.635,.605]))
footer(fig,.058)
save(fig,'search-space-growth')
print(json.dumps({'final':f'{NUMBERS[-1]:.6e}','partial':f'{PARTIAL:.6e}','SP':SP,'files':['team-search-space','decision-tree','search-space-growth']},indent=2))
