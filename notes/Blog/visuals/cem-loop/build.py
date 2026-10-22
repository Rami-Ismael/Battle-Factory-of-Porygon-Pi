"""Build cem-loop.html from template.html + results/cem_v3.json (2026-10-08).

    /tmp/vgc-pilot/.venv/bin/python build.py   -> cem-loop.html (publish that file)
"""
import base64, json, math, sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, "/tmp/vgc-pilot/src")
from corpus import parse_team_text, norm

HERE = Path(__file__).resolve().parent
RES = Path.home() / "Documents/code/vgc-team-generator-pilot/results"
J = json.load(open(RES / "cem_v3.json"))
SPR = HERE.parent / "search-loop/sprites"
ISPR = HERE.parent / "item-sprites"
CUT, ELITE_MIN = J["config"]["cut"], J["config"]["elite_min"]
CORE = tuple(sorted("basculegion chandelure floetteeternal garchomp kingambit whimsicott".split()))
ZARD = tuple(sorted("basculegion charizard floetteeternal garchomp kingambit whimsicott".split()))

display, lineups, lu_ix = {}, [], {}
def key(paste):
    t = parse_team_text(paste)
    for s in t: display[norm(s["species"])] = s["species"]
    k = tuple(sorted(norm(s["species"]) for s in t))
    if k not in lu_ix: lu_ix[k] = len(lineups); lineups.append(list(k))
    return lu_ix[k]
group = lambda i: "core" if tuple(lineups[i]) == CORE else "zard" if tuple(lineups[i]) == ZARD else "other"

anchors = [dict(y=a["y"], lu=key(a["paste"])) for a in J["anchors"]]

gens = []
for g in sorted(J["gens"], key=int):
    v = J["gens"][g]; G = int(g)
    pk = [key(p) for p in v["proposals"]]; bk = [key(p) for p in v["battled"]]
    pc = Counter(group(i) for i in pk); bc = Counter(group(i) for i in bk)
    gm = lambda grp: (lambda ys: sum(ys) / len(ys) if ys else None)([y for i, y in zip(bk, v["y"]) if group(i) == grp])
    # the elites this generation's model was retrained on (same rule as cem_v3.elites_from)
    el = None
    if G > 0:
        rows = [dict(y=a["y"], lu=a["lu"], src="anchor") for a in anchors]
        for h in range(G):
            w = J["gens"][str(h)]
            rows += [dict(y=y, lu=lu_ix[tuple(sorted(norm(s["species"]) for s in parse_team_text(p)))], src="loop", n=len(parse_team_text(p)))
                     for p, y in zip(w["battled"], w["y"]) if y is not None]
        ok = [r for r in rows if r["y"] >= CUT]
        if len(ok) < ELITE_MIN: ok = sorted(rows, key=lambda r: -r["y"])[:ELITE_MIN]
        ok = [r for r in ok if r["src"] == "anchor" or r["n"] == 6]
        loop = [r for r in ok if r["src"] == "loop"]
        el = dict(n=len(ok), anchors=len(ok) - len(loop), loop=len(loop), core=sum(group(r["lu"]) == "core" for r in loop),
                  zard=sum(group(r["lu"]) == "zard" for r in loop), cut=min(r["y"] for r in ok),
                  labels=sum(len(J["gens"][str(h)]["y"]) for h in range(G)))
    gens.append(dict(g=G, props=len(pk), props_sets=len(set(pk)), props_mix=dict(pc), battled_sets=len(set(bk)), battled_mix=dict(bc),
                     battled=[[round(y, 4), i] for i, y in zip(bk, v["y"])], stats=v["stats"], elites=el,
                     group_mean=dict(core=gm("core"), zard=gm("zard"), other=gm("other"))))

def sheet(paste):
    out = []
    for s in parse_team_text(paste):
        sp = {k: v for k, v in s["evs"].items() if v}
        out.append(dict(id=norm(s["species"]), species=s["species"], item=s.get("item"), ability=s.get("ability"),
                        nature=s.get("nature"), moves=s["moves"], sp=" / ".join(f"{v} {k}" for k, v in sp.items())))
    return out

final = [dict(label=f["label"], fresh=f["fresh"], battles=f["battles"], src=f["src"], lu=key(f["paste"]), sheet=sheet(f["paste"]))
         for f in J["final"]]

# comparison: the old gradient loop's best re-battled team (same 192-battle protocol) and the counter grid's best real team
R = json.load(open(RES / "rebattle_top.json"))
ob = max(R["teams"], key=lambda t: t["win_rate"])
old_best = dict(y=ob["win_rate"], se=ob["se"], battles=ob["battles"], lu=key(ob["paste"]))
mb607 = ["sinistcha", "milotic", "incineroar", "raichu", "mimikyu", "ninetalesalola"]
for s, d in zip(mb607, ["Sinistcha", "Milotic", "Incineroar", "Raichu", "Mimikyu", "Ninetales-Alola"]): display.setdefault(s, d)

used = {s for lu in lineups for s in lu} | set(mb607)
sprites = {s: "data:image/png;base64," + base64.b64encode((SPR / f"{s}.png").read_bytes()).decode()
           for s in used if (SPR / f"{s}.png").exists()}                          # no local sprite -> name chip
items = {}
for f in final:
    for s in f["sheet"]:
        if s["item"]:
            iid = norm(s["item"]); p = ISPR / f"item-{iid}.png"
            if p.exists(): items[s["item"]] = "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode()

data = dict(gens=gens, anchors=anchors, lineups=lineups, final=final, stopped=J["stopped"], config=J["config"],
            old_best=old_best, mb607=dict(y=0.691, se=0.018, lu=mb607), sprites=sprites, items=items, names=display,
            core=list(CORE), zard=list(ZARD))
blob = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
html = (HERE / "template.html").read_text().replace("/*DATA*/null", blob)
(HERE / "cem-loop.html").write_text(html)
print(f"cem-loop.html {len(html)/1024:.0f} KB · {len(sprites)}/{len(used)} sprites · {len(items)} items · old best {old_best['y']:.3f} "
      f"{[display[s] for s in lineups[old_best['lu']]]}")
for g in gens: print(g["g"], g["props_sets"], g["props_mix"], g["battled_sets"], g["battled_mix"], g["elites"], {k: v and round(v, 3) for k, v in g["group_mean"].items()})
