# PROTOTYPE — throwaway. Builds data.js for teams-differ-prototype/index.html
# from the combined loop's 11 generations (gradloop.json, 128 teams x 24 battles each).
import json, re, statistics
from collections import Counter

R = '/Users/ramiismael/Documents/code/vgc-team-generator-pilot/results/'
d = json.load(open(R + 'gradloop.json'))
toid = lambda s: re.sub(r'[^a-z0-9]', '', s.lower())

strings, sidx = [], {}
def S(x):
    if x not in sidx:
        sidx[x] = len(strings); strings.append(x)
    return sidx[x]

species = {}  # name -> sprite id

def parse(p):
    mons = []
    for blk in p.strip().split('\n\n'):
        L = blk.strip().split('\n'); sp, _, it = L[0].partition(' @ ')
        sp = sp.strip(); species[sp] = toid(sp)
        ab = next((l[9:] for l in L if l.startswith('Ability: ')), '')
        nat = next((l.split()[0] for l in L if l.endswith(' Nature')), '')
        ev = next((l[5:] for l in L if l.startswith('EVs: ')), '')
        mv = sorted(l[2:] for l in L if l.startswith('- '))
        # [species, item, ability, stat alignment, stat points, moves]
        mons.append([S(sp), S(it.strip()), S(ab), S(nat), S(ev), S(', '.join(mv))])
    return mons

gens = []
for g in range(1, 12):
    G = d['gens'][f'combined_g{g}']
    teams = [dict(y=round(y, 4), m=parse(p)) for p, y in zip(G['pastes'], G['y'])]
    gens.append(dict(g=g, mean=round(G['stats']['mean'], 3), teams=teams))

# sanity + the numbers the copy quotes
for G in gens:
    c = Counter('.'.join(sorted(strings[m[0]] for m in t['m'])) for t in G['teams'])
    print(G['g'], len(c), c.most_common(1)[0][1])
G = gens[-1]
c = Counter('.'.join(sorted(strings[m[0]] for m in t['m'])) for t in G['teams'])
top = c.most_common(1)[0][0]
ys = [t['y'] for t in G['teams'] if '.'.join(sorted(strings[m[0]] for m in t['m'])) == top]
p = statistics.mean(ys)
print('largest g11 group n', len(ys), 'range', min(ys), max(ys), 'mean', round(p, 3),
      'observed var', round(statistics.pvariance(ys), 4), 'binomial var at 24', round(p * (1 - p) / 24, 4))

out = dict(strings=strings, sprites={species_name: sid for species_name, sid in species.items()},
           battles=24, realMean=0.4625, gens=gens)
open('data.js', 'w').write('window.TEAMS_DIFFER=' + json.dumps(out, separators=(',', ':')) + ';\n')
