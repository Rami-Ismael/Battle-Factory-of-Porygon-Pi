"""Corpus density map + cross-entropy loop paths (plain vs taboo), embedded into one HTML page.

Map: shrinkage linear discriminant analysis (lambda 0.05, 3 axes) on species/ability/item/move
multi-hot vectors of the Reg M-B corpus, labelled by the archetype rule of ../archetype-diffusion,
then one fixed linear map to 2D that puts the archetype centres left/top/right/bottom.
Density: Gaussian kernel density of the corpus teams on that map.
Paths: results/diversity_methods_all.json, arms control (plain) and taboo, seeds 1-3, generations 1-6.
Each generation's 128 battled teams are projected; the path visits the best-labelled team of each.
"""
import json, re, base64
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
REPO = Path('/Users/ramiismael/Documents/code/vgc-team-generator-pilot')
SPR = ROOT.parent / 'search-loop'
known = {f.stem: f'sprites/{f.name}' for f in (SPR / 'sprites').glob('*.png')}
toid = lambda s: re.sub(r'[^a-z0-9]', '', s.lower())

def parse(paste):
    mons = []
    for blk in paste.strip().split('\n\n'):
        L = [l.strip() for l in blk.strip().split('\n') if l.strip()]
        if not L: continue
        head, _, item = L[0].partition(' @ ')
        head = re.sub(r'\s*\((M|F)\)\s*$', '', head.strip())
        m = re.search(r'\(([^()]+)\)\s*$', head)
        sp = m.group(1) if m else head
        ab = next((l[9:] for l in L if l.startswith('Ability: ')), '')
        mv = [l[2:].strip() for l in L if l.startswith('- ')]
        mons.append(dict(name=sp.strip(), s=toid(sp), a=toid(ab), i=toid(item), m=[toid(x) for x in mv]))
    return mons

# Archetype rule, first match wins (same as Blog/visuals/archetype-diffusion).
ARCH = ['Trick Room', 'Weather', 'Tailwind', 'Other']
W_AB = {'drizzle', 'drought', 'sandstream', 'snowwarning'}
W_IT = {'charizarditey', 'tyranitarite', 'froslassite', 'abomasite'}
W_SP = {'charizardmegay', 'tyranitarmega', 'froslassmega', 'abomasnowmega'}
def label(t):
    mv = {x for p in t for x in p['m']}
    if 'trickroom' in mv: return 0
    if any(p['s'] in W_SP or p['a'] in W_AB or p['i'] in W_IT for p in t): return 1
    return 2 if 'tailwind' in mv else 3

# The same rule per Pokemon: the job its own set does.
from collections import Counter
NAMES = {}
def role(p):
    if 'trickroom' in p['m']: return 0
    if p['s'] in W_SP or p['a'] in W_AB or p['i'] in W_IT: return 1
    return 2 if 'tailwind' in p['m'] else 3
def role_stats(teams, top=4):
    n = len(teams); out = []
    for k in range(4):
        has = sum(any(role(p) == k for p in t) for t in teams)
        c = Counter(sp for t in teams for sp in {p['s'] for p in t if role(p) == k})
        out.append(dict(cov=round(has / n, 3), top=[[sp, round(v / n, 3)] for sp, v in c.most_common(top)]))
    for t in teams:
        for p in t: NAMES.setdefault(p['s'], p['name'])
    return out

corpus = []
seen = set()
for p in sorted((REPO / 'teams/reg_mb').rglob('*.txt')):
    t = parse(p.read_text())
    if len(t) == 6 and p.stem not in seen: seen.add(p.stem); corpus.append((p.stem, t))
y = np.array([label(t) for _, t in corpus])
corpus_roles = role_stats([t for _, t in corpus])

tokens = sorted({f'{k}:{v}' for _, t in corpus for p in t
                 for k, v in [('s', p['s']), ('a', p['a']), ('i', p['i'])] + [('m', x) for x in p['m']] if v})
ix = {k: n for n, k in enumerate(tokens)}
def feats(teams):
    X = np.zeros((len(teams), len(tokens)), np.float64)
    for n, t in enumerate(teams):
        for p in t:
            for k in [f"s:{p['s']}", f"a:{p['a']}", f"i:{p['i']}"] + [f'm:{x}' for x in p['m']]:
                if k in ix: X[n, ix[k]] = 1
    return X

X = feats([t for _, t in corpus]); mu = X.mean(0)
Sw = sum(np.cov((X[y == k] - mu).T, bias=True) * (y == k).sum() for k in range(4)) / len(X) + 0.05 * np.eye(X.shape[1])
M = np.stack([X[y == k].mean(0) - mu for k in range(4)]); Sb = (M.T * np.bincount(y, minlength=4) / len(y)) @ M
Li = np.linalg.inv(np.linalg.cholesky(Sw)); ev, U = np.linalg.eigh(Li @ Sb @ Li.T); W = Li.T @ U[:, ::-1][:, :3]
P3 = (X - mu) @ W; W = W / np.percentile(np.abs(P3), 98, axis=0)
cent3 = np.stack([((X[y == k] - mu) @ W).mean(0) for k in range(4)])
A = np.linalg.lstsq(cent3, np.array([[-1, 0], [0, 1], [1, 0], [0, -1]], float) * 0.8, rcond=None)[0]
proj = lambda Xn: ((Xn - mu) @ W) @ A
P = proj(X); cent = cent3 @ A

# Map extent from the corpus, padded; density by Gaussian kernel (Scott bandwidth, narrowed).
lo = np.percentile(P, 0.5, axis=0) - 0.25; hi = np.percentile(P, 99.5, axis=0) + 0.25
GX, GY = 220, 160
gx = np.linspace(lo[0], hi[0], GX); gy = np.linspace(lo[1], hi[1], GY)
h = P.std(0) * len(P) ** (-1 / 6) * 0.4
dens = np.zeros((GY, GX))
for px, py in P:
    dens += np.exp(-0.5 * ((gy[:, None] - py) / h[1]) ** 2) * np.exp(-0.5 * ((gx[None, :] - px) / h[0]) ** 2)
dens = np.round(255 * np.sqrt(dens / dens.max())).astype(int)   # sqrt so the small pools stay visible

# Loop runs.
D = json.loads((REPO / 'results/diversity_methods_all.json').read_text())
ARMS = {'plain': 'control', 'taboo': 'taboo'}
species = {sp for r in corpus_roles for sp, _ in r['top']}; runs = {}
for arm, key in ARMS.items():
    for s in D['config']['seeds']:
        gens = []
        for g in range(1, D['config']['gens'] + 1):
            G = D['gens'][f'{key}_s{s}_g{g}']
            teams = [parse(x) for x in G['battled']]
            ok = [n for n, t in enumerate(teams) if len(t) == 6]
            Q = proj(feats([teams[n] for n in ok]))
            ys = [G['y'][n] for n in ok]
            b = int(np.argmax(ys))
            best = teams[ok[b]]
            for t in (teams[n] for n in ok):
                species.update(p['s'] for p in t)
            gens.append(dict(
                pts=[[round(float(q[0]), 3), round(float(q[1]), 3), round(float(v), 2)] for q, v in zip(Q, ys)],
                cloud=[[p['s'] for p in teams[n]] for n in ok],
                best=dict(x=round(float(Q[b, 0]), 3), y=round(float(Q[b, 1]), 3), wr=round(float(ys[b]), 2),
                          arch=label(best), mons=[dict(s=p['s'], name=p['name']) for p in best],
                          paste=G['battled'][ok[b]].strip()),
                roles=role_stats([teams[n] for n in ok]),
                mean=round(float(np.mean(ys)), 3), comps=G['div']['distinct_compositions'], accepted=G['div']['accepted'],
                cx=round(float(Q[:, 0].mean()), 3), cy=round(float(Q[:, 1].mean()), 3)))
        runs[f'{arm}_s{s}'] = gens

dex = json.loads((REPO / 'data/pokedex.json').read_text())
def sprite_file(sp):
    # Formes and Megas without their own sprite fall back to the base species.
    base = toid(dex.get(sp, {}).get('baseSpecies', sp))
    for k in (sp, sp.split('mega')[0] if 'mega' in sp and sp != 'meganium' else sp, base):
        if k in known: return SPR / known[k]
    return None
sprites = {}
for sp in sorted(species):
    f = sprite_file(sp)
    if f: sprites[sp] = 'data:image/png;base64,' + base64.b64encode(f.read_bytes()).decode()
missing = sorted(species - set(sprites))

data = dict(extent=[float(lo[0]), float(hi[0]), float(lo[1]), float(hi[1])], grid=[GX, GY], density=dens.flatten().tolist(),
            corpus=[[round(float(a), 3), round(float(b), 3), int(k)] for (a, b), k in zip(P, y)],
            centroids=np.round(cent, 3).tolist(), archetypes=ARCH, labelCounts=np.bincount(y, minlength=4).tolist(),
            nCorpus=len(corpus), corpusRoles=corpus_roles, names=NAMES, runs=runs, sprites=sprites, seeds=D['config']['seeds'], gens=D['config']['gens'])
tmpl = ROOT / 'template.html'
if tmpl.exists():
    html = tmpl.read_text().replace('__DATA__', json.dumps(data, separators=(',', ':')).replace('</', '<\\/'))
    (ROOT / 'team-map-path.html').write_text(html)
    print('bytes', len(html))
print('corpus', len(corpus), 'labels', np.bincount(y, minlength=4).tolist(), 'tokens', len(tokens))
print('centroids 2D', np.round(cent, 2).tolist(), 'extent', np.round(lo, 2).tolist(), np.round(hi, 2).tolist())
for k, gens in runs.items():
    print(k, [(g['best']['wr'], g['mean'], g['comps'], ARCH[g['best']['arch']][:2], round(g['best']['x'], 2), round(g['best']['y'], 2)) for g in gens])
print('species', len(species), 'missing sprites', missing)
