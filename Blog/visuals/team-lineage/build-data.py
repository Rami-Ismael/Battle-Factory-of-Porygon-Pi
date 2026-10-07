import json,re
R='/Users/ramiismael/Documents/code/vgc-team-generator-pilot/results/'
d=json.load(open(R+'gradloop.json')); rb=json.load(open(R+'rebattle_top.json'))
dex=json.load(open('/tmp/vgc-pilot/data/pokedex.json'))
def toid(s): return re.sub(r'[^a-z0-9]','',s.lower())
def sprite(name):
    e=dex.get(toid(name),{})
    if e.get('forme'): return toid(e['baseSpecies'])+'-'+toid(e['forme'])
    return toid(name)
CORE={'garchomp','kingambit','basculegion','whimsicott','charizard','floetteeternal'}
def parse(p):
    mons=[]
    for blk in p.strip().split('\n\n'):
        L=blk.strip().split('\n'); sp,_,it=L[0].partition(' @ ')
        ab=next((l[9:] for l in L if l.startswith('Ability: ')),'')
        nat=next((l.split()[0] for l in L if l.endswith(' Nature')),'')
        ev=next((l[5:] for l in L if l.startswith('EVs: ')),'')
        mv=[l[2:] for l in L if l.startswith('- ')]
        mons.append(dict(s=sp.strip(),id=sprite(sp.strip()),i=it.strip(),a=ab,n=nat,e=ev,m=mv))
    return mons
rbmap={t['paste'].strip():t for t in rb['teams']}
nodes=[dict(n=0,g=0,r=0,mons=[],parent=None)]
cols=[]
for g in range(1,12):
    G=d['gens'][f'combined_g{g}']; st=G['stats']
    cols.append(dict(g=g,mean=round(st['mean'],3),se=round(st['se'],3),n=st['n']))
    order=sorted(range(len(G['y'])),key=lambda i:-G['y'][i])[:3]
    for r,i in enumerate(order):
        mons=parse(G['pastes'][i]); t=rbmap.get(G['pastes'][i].strip())
        node=dict(n=len(nodes),g=g,r=r+1,y=G['y'][i],wins=round(G['y'][i]*24),mons=mons,
          core=sum(toid(m['s']) in CORE for m in mons))
        if t: node['rb']=dict(wr=t['win_rate'],se=round(t['se'],3),b=t['battles'],rank=t['rank'])
        prev=[x for x in nodes if x['g']==g-1]
        S=set(toid(m['s']) for m in mons)
        best=max(prev,key=lambda x:(len(S&set(toid(m['s']) for m in x['mons'])),-x['r']))
        node['parent']=best['n']; node['shared']=len(S&set(toid(m['s']) for m in best['mons']))
        nodes.append(node)
print(sum('rb' in x for x in nodes),'rebattled among nodes')
for x in nodes[1:]: print(x['g'],x['r'],x['y'],x['core'],x['parent'],x.get('rb',{}).get('wr'),[m['id'] for m in x['mons']])
json.dump(dict(nodes=nodes,cols=cols,real=rb['anchor_real_mean']),open('lineage.json','w'),separators=(',',':'))
