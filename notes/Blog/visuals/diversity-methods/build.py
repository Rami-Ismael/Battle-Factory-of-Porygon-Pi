"""Embed the diversity-methods run (2026-10-08) and the sprites it needs into diversity-methods.html.

Reads ~/Documents/code/vgc-team-generator-pilot/results/diversity_methods_all.json (3 seeds x 6 arms x
6 generations, merged) and cem_v3.json (the shared generation 0). Crowd figure = seed 1. No new battles.
"""
import base64, json, pathlib, re
from collections import Counter

here = pathlib.Path(__file__).parent
res = pathlib.Path.home() / "Documents/code/vgc-team-generator-pilot/results"
sprites_dir = here.parent / "search-loop" / "sprites"
out = json.load(open(res / "diversity_methods_all.json"))
c3 = json.load(open(res / "cem_v3.json"))

ARMS = ["control", "taboo", "peel", "island", "big", "dpp"]
SEEDS, GENS = [1, 2, 3], 6
norm = lambda s: re.sub(r"[^a-z0-9]", "", s.lower())


def species_of(paste):
    names = []
    for block in paste.strip().split("\n\n"):
        first = block.strip().split("\n")[0].split(" @ ")[0].strip()
        m = re.search(r"\(([^()]{3,})\)\s*(\([MF]\))?$", first)
        if m and m.group(1) not in ("M", "F"): first = m.group(1)
        first = re.sub(r"\s*\([MF]\)$", "", first)
        names.append(norm(first))
    return sorted(names)


# ---- generation 0 (shared by every arm): the cem_v3 proposals
g0_comps = [species_of(p) for p in c3["gens"]["0"]["proposals"]]
g0_cnt = Counter(tuple(c) for c in g0_comps)
# the run counted 176: one proposal has an item-less slot that its parser merged (known bug), giving a 5-species key
assert abs(len(g0_cnt) - out["g0"]["sets"]) <= 1, (len(g0_cnt), out["g0"]["sets"])

vocab = {}
def idx(sp):
    return vocab.setdefault(sp, len(vocab))

def crowd(counter):
    rows = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))
    return [[n, [idx(s) for s in comp]] for comp, n in rows]

CROWD = {a: {"0": crowd(g0_cnt)} for a in ARMS}
for a in ARMS:
    for g in range(1, GENS + 1):
        c = out["gens"][f"{a}_s1_g{g}"]
        CROWD[a][str(g)] = crowd(Counter(tuple(x) for x in c["stream_comps"]))

# ---- 3-seed means for the curves, and per-seed points at generation 6
CURVE = {}
for a in ARMS:
    pts = [[out["g0"]["sets"], max(g0_cnt.values()) / 512, out["g0"]["stats"]["mean"]]]
    for g in range(1, GENS + 1):
        cs = [out["gens"][f"{a}_s{s}_g{g}"] for s in SEEDS]
        pts.append([sum(c["div"]["distinct_compositions"] for c in cs) / 3,
                    sum(c["div"]["largest_share"] for c in cs) / 3,
                    sum(c["stats"]["mean"] for c in cs) / 3])
    CURVE[a] = pts
SEEDPTS = {a: [[out["gens"][f"{a}_s{s}_g{GENS}"]["div"]["distinct_compositions"],
                out["gens"][f"{a}_s{s}_g{GENS}"]["stats"]["mean"]] for s in SEEDS] for a in ARMS}

# ---- distinct rosters among the 128 battled teams (g0 from the cem_v3 stats key 'battled_sets')
BATTLED = {a: [out["g0"]["stats"]["battled_sets"]] + [sum(out["gens"][f"{a}_s{s}_g{g}"]["stats"]["battled_compositions"] for s in SEEDS) / 3
                                                       for g in range(1, GENS + 1)] for a in ARMS}

# ---- table
summ = out["summary"]
TABLE = {}
for a in ARMS:
    best = [max(r["score"] for r in out["rebattle"][f"{a}_s{s}"]) for s in SEEDS]
    row = dict(best=best, comps=CURVE[a][-1][0], share=CURVE[a][-1][1], win=CURVE[a][-1][2])
    if a in summ:
        s = summ[a]
        row.update(diff=s["win_diff"], ci=s["ci"], passes=s["passes"], cores=sum(s["qualifying_cores"].values()) / 3,
                   off=sum(s["off_core_cores"].values()) / 3)
    else:
        cores = []
        for s in SEEDS:
            cores.append(len({tuple(species_of(p)) for g in range(1, GENS + 1)
                              for p, v in zip(out["gens"][f"control_s{s}_g{g}"]["battled"], out["gens"][f"control_s{s}_g{g}"]["y"])
                              if v is not None and v >= 0.531}))
        row.update(cores=sum(cores) / 3)
    TABLE[a] = row

# ---- mechanism figures use real seed-1 data
peel = {str(g): out["gens"][f"peel_s1_g{g}"]["banned"] for g in range(1, GENS + 1)}
isl = [species_of(i["seed"]["paste"]) for i in out["gens"]["island_s1_g1"]["_isl"]]
for comp in isl:
    for s in comp: idx(s)
for g in peel.values():
    for s in g: idx(s)
top_ctrl = out["gens"]["control_s1_g6"]["div"]["top_compositions"][0]
for s in top_ctrl["species"]: idx(s)


# ---- up-close figures (all seed 1 unless stated) ----
CUT = 0.531
def tiles_for(pastes, ys):
    cnt = Counter(tuple(species_of(p)) for p in pastes)
    rank = {c: i for i, (c, _) in enumerate(sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0])))}
    t = sorted(((rank[tuple(species_of(p))], int(y is not None and y >= CUT)) for p, y in zip(pastes, ys)))
    top = [[n, [idx(x) for x in c]] for c, n in sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0]))[:4]]
    return t, top, len(cnt), sum(h for _, h in t)

LOOP = {}
g0b, g0y = c3["gens"]["0"]["battled"], c3["gens"]["0"]["y"]
t, top, dist, above = tiles_for(g0b, g0y)
LOOP["0"] = dict(tiles=t, top=top, distinct=dist, above=above, elites=None)
for g in range(1, GENS + 1):
    c = out["gens"][f"control_s1_g{g}"]
    t, top, dist, above = tiles_for(c["battled"], c["y"])
    LOOP[str(g)] = dict(tiles=t, top=top, distinct=dist, above=above, elites=c["elites"]["n"],
                        anchors=c["elites"]["anchors"], cut=round(c["elites"]["cut"], 3))

TAB = {}
for g in range(1, GENS + 1):
    cs = [out["gens"][f"taboo_s{s}_g{g}"] for s in SEEDS]
    m = lambda f: sum(f(c) for c in cs) / 3
    TAB[str(g)] = dict(raw=m(lambda c: c["div_raw"]["distinct_compositions"]), rawshare=m(lambda c: c["div_raw"]["largest_share"]),
                       kept=m(lambda c: c["div"]["distinct_compositions"]), attempts=m(lambda c: c["div"]["attempts"]),
                       rejected=m(lambda c: c["taboo_rejected"]), keptshare=m(lambda c: c["div"]["largest_share"]),
                       top=[[t_["count"], [idx(x) for x in t_["species"]]] for t_ in out["gens"][f"taboo_s1_g{g}"]["div_raw"]["top_compositions"]])

ISL = {}
for g in range(1, GENS + 1):
    c = out["gens"][f"island_s{1}_g{g}"]
    per = []
    for k in range(K := 4):
        comps = Counter(tuple(x) for x in c["stream_comps"][k * 128:(k + 1) * 128])
        rows = [[n, [idx(x) for x in comp]] for comp, n in sorted(comps.items(), key=lambda kv: (-kv[1], kv[0]))]
        per.append(dict(rows=rows, distinct=len(comps), mean=c["islands"][k]["mean"], qual=c["islands"][k]["qualifying"],
                        elites=c["islands"][k]["elites"]))
    ISL[str(g)] = per

pool = out["gens"]["control_s1_g1"]
DPPPOOL = sorted(([idx(x) for x in species_of(p)], y) for p, y in zip(pool["battled"], pool["y"]) if y is not None)
DPPPOOL = sorted(DPPPOOL, key=lambda r: -r[1])

names = sorted(vocab, key=vocab.get)
have = {s: "data:image/png;base64," + base64.b64encode((sprites_dir / f"{s}.png").read_bytes()).decode()
        for s in names if (sprites_dir / f"{s}.png").exists()}
data = dict(loop=LOOP, tab=TAB, isl=ISL, dpppool=DPPPOOL, species=names, battled=BATTLED, crowd=CROWD, curve=CURVE, seedpts=SEEDPTS, table=TABLE, peel=peel, island=isl,
            ctrl_top=[idx(s) for s in top_ctrl["species"]], sprites=have, n_missing=len(names) - len(have))
blob = json.dumps(data, separators=(",", ":"))
html = (here / "template.html").read_text().replace("/*__DATA__*/null", blob)
(here / "diversity-methods.html").write_text(html)
print(f"{len(names)} species, {len(have)} sprites, {len(html):,} bytes; control g6 {CURVE['control'][-1]}, taboo g6 {CURVE['taboo'][-1]}")
