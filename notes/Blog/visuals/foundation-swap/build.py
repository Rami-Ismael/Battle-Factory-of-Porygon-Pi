"""Build foundation-swap.html from template.html + results/foundation.json (2026-10-08).

    /tmp/vgc-pilot/.venv/bin/python build.py   -> foundation-swap.html (publish that file)
"""
import base64, json, math, sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, "/tmp/vgc-pilot/src")
from corpus import parse_team_text, norm

HERE = Path(__file__).resolve().parent
RES = Path.home() / "Documents/code/vgc-team-generator-pilot/results"
F = json.load(open(RES / "foundation.json"))
S = F["summary"]
SPR = HERE.parent / "search-loop/sprites"
ARMS = ["base", "cfg", "lowtemp", "protect3", "surrogate", "guide", "repair", "stack"]

rows = []
for a in ARMS:
    o, n = S[f"old:{a}"], S[f"new:{a}"]
    d = n["got"] - o["got"]; sd = math.hypot(n["se"], o["se"])
    go = o["got"] - S["old:base"]["got"]; gn = n["got"] - S["new:base"]["got"]
    si = math.sqrt(n["se"] ** 2 + o["se"] ** 2 + S["new:base"]["se"] ** 2 + S["old:base"]["se"] ** 2)
    rows.append(dict(arm=a, old=dict((k, o[k]) for k in ("got", "se", "valid", "sets")),
                     new=dict((k, n[k]) for k in ("got", "se", "valid", "sets")),
                     diff=d, z=d / sd, gain_old=go, gain_new=gn, inter=gn - go, inter_z=(gn - go) / si))

display = {}
def lineups(tag):
    c = F["cells"][tag]; g = defaultdict(list); freq = Counter(); ys = []
    for p, y in zip(c["pastes"], c["y"]):
        t = parse_team_text(p)
        for s in t: display[norm(s["species"])] = s["species"]
        key = tuple(sorted(norm(s["species"]) for s in t))
        g[key].append(y); freq.update(set(key)); ys.append(y)
    lu = [dict(species=list(k), n=len(v), mean=sum(v) / len(v), best=max(v))
          for k, v in sorted(g.items(), key=lambda kv: (-len(kv[1]), -sum(kv[1]) / len(kv[1])))]
    return dict(lineups=lu, freq=freq.most_common(), y=ys)

stacks = {"old": lineups("old:stack"), "new": lineups("new:stack")}

# ---- version 3 (Codex regmb v3, full-regulation vocabulary): shown only once its report exists
v3 = None
V3P = RES / "foundation_v3.json"
if V3P.exists():
    J = json.load(open(V3P)); S3 = J.get("summary", {})
    if all(f"v3:{a}" in S3 for a in ARMS):
        legacy = set(json.load(open(Path.home() / ".local/share/vgc-pilot-runtime/regmb-v3/data-audit.json"))["legacy_vocabulary"]["species"])
        b3 = S3["v3:base"]; v3rows = []
        for a in ARMS:
            v, n = S3[f"v3:{a}"], S[f"new:{a}"]
            d = v["got"] - n["got"]; sd = math.hypot(v["se"], n["se"])
            v3rows.append(dict(arm=a, old=S[f"old:{a}"]["got"], new=n["got"], new_se=n["se"],
                               v3=dict((k, v[k]) for k in ("got", "se", "valid", "sets", "new_species_teams", "n")),
                               d=d, z=d / sd, gain=v["got"] - b3["got"]))
        newsp = Counter()
        for a in ARMS:
            for p in J["cells"][f"v3:{a}"]["pastes"]:
                for sl in parse_team_text(p):
                    k = norm(sl["species"]); display[k] = sl["species"]
                    if k not in legacy: newsp[k] += 1
        F3 = dict(cells=J["cells"])
        c = J["cells"]["v3:stack"]; g = defaultdict(list); freq = Counter(); ys = []
        for p, y in zip(c["pastes"], c["y"]):
            t = parse_team_text(p)
            for sl in t: display[norm(sl["species"])] = sl["species"]
            key = tuple(sorted(norm(sl["species"]) for sl in t)); g[key].append(y); freq.update(set(key)); ys.append(y)
        stack3 = dict(lineups=[dict(species=list(k), n=len(v), mean=sum(v) / len(v), best=max(v))
                               for k, v in sorted(g.items(), key=lambda kv: (-len(kv[1]), -sum(kv[1]) / len(kv[1])))],
                      freq=freq.most_common(), y=ys)
        read = HERE / "v3_read.html"
        v3 = dict(rows=v3rows, newsp=newsp.most_common(12), legacy_n=len(legacy) - 1, stack=stack3,
                  read=read.read_text() if read.exists() else None)
mu = F["cells"]["new:stack"]["mu"]; yb = F["cells"]["new:stack"]["y"]
mm, my = sum(mu) / len(mu), sum(yb) / len(yb)
r = sum((a - mm) * (b - my) for a, b in zip(mu, yb)) / math.sqrt(sum((a - mm) ** 2 for a in mu) * sum((b - my) ** 2 for b in yb))
species = {s for st in stacks.values() for s, _ in st["freq"]}
if v3: species |= {s for s, _ in v3["stack"]["freq"]} | {s for s, _ in v3["newsp"]}
sprites = {s: "data:image/png;base64," + base64.b64encode((SPR / f"{s}.png").read_bytes()).decode()
           for s in species if (SPR / f"{s}.png").exists()}                       # no local sprite -> name chip

data = dict(rows=rows, stacks=stacks, sprites=sprites, names=display, v3=v3,
            surrogate=dict(pred=mm, got=my, r=r))
blob = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
html = (HERE / "template.html").read_text().replace("/*DATA*/null", blob)
(HERE / "foundation-swap.html").write_text(html)
print(f"foundation-swap.html {len(html)/1024:.0f} KB · {len(sprites)}/{len(species)} sprites · surrogate r {r:+.3f} · v3: {'yes' if v3 else 'not yet'}")
