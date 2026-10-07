"""Build interaction-order.html from the repo's results/interaction_order.json + cube plan.
Sprites are embedded as data URIs from ../search-loop/sprites.
Run: python3 build.py
"""
import base64, json, os, re
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = Path("/Users/ramiismael/Documents/code/vgc-team-generator-pilot")
SRC = Path(os.environ.get("IO_RESULTS", REPO / "results"))          # override for a mock run
R = json.load(open(SRC / "interaction_order.json"))
PLAN = json.load(open(REPO / "results/interaction_order_parts/vgc_plan.json"))
OUT = Path(os.environ.get("IO_OUT", HERE / "interaction-order.html"))
SPR = HERE.parent / "search-loop/sprites"
STONE = {"charizarditey": "charizardmegay", "charizarditex": "charizardmegax"}

def key(entry):
    sp, _, item = entry.partition("|")
    n = re.sub(r"[^a-z0-9]", "", sp.split(" (")[0].lower())
    it = re.sub(r"[^a-z0-9]", "", item.lower())
    if it in STONE: return STONE[it]
    if it.endswith("ite") and it != "eviolite" and (SPR / f"{n}mega.png").exists(): return n + "mega"
    return n

def uri(n):
    p = SPR / f"{n}.png"
    if not p.exists(): p = SPR / f"{n.replace('mega', '')}.png"
    return "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode() if p.exists() else ""

sprites = {}
def team_view(c):
    base = [key(t) for t in c["team"]]
    donors = [key(d + "|" + it if it else d) for d, it in zip(c["in"], c["donor_items"])]
    for n in base + donors: sprites.setdefault(n, uri(n))
    return base, donors

def disp(s): return s.split(" (")[0]

cycles = {"weather": [], "primary": []}
for grp, lst in (("weather", R["cycles"]["weather"]), ("primary", R["cycles"]["primary_selected"])):
    for cy in lst:
        c = next(p for p in PLAN[grp] if p["anchor"] == cy["anchor"])
        base, donors = team_view(c)
        cycles[grp].append({**cy, "base": base, "donors": donors, "slots": c["slots"],
                            "out_names": [disp(x) for x in c["out"]], "in_names": [disp(x) for x in c["in"]]})

data = {"R": {k: R[k] for k in ("design", "vgc", "benchmarks", "vs", "T1", "T2", "signal_ci95", "sanity", "verdict")},
        "cycles": cycles, "sprites": sprites}
html = (HERE / "template.html").read_text().replace("/*__DATA__*/null", json.dumps(data))
OUT.write_text(html)
print("wrote", OUT, f"{len(html)/1024:.0f} KB")
