"""Build team-preview.html: the team-preview section of the blog (Distill article, three figures).
Teams MB522 (you) and MB763 (opponent, 1st place Talon's Fight Club #98) from the vgc-bench Reg M-B pool.
Run: python3 build.py
"""
import base64, json, math, re
from pathlib import Path

HERE = Path(__file__).resolve().parent
POOL = Path("/tmp/vgc-pilot/vgc-bench/teams/reg_mb")
SPR = HERE.parent / "search-loop/sprites"


def parse(tid):
    mons = []
    for blk in (POOL / f"{tid}.txt").read_text().strip().split("\n\n"):
        lines = blk.splitlines()
        sp, _, item = re.sub(r" \([MF]\)", "", lines[0]).partition(" @ ")
        get = lambda pre: next((l.split(": ", 1)[1] for l in lines if l.startswith(pre)), "")
        nature = next((l.split()[0] for l in lines if l.endswith(" Nature")), "")
        key = re.sub(r"[^a-z0-9]", "", sp.lower())
        img = "data:image/png;base64," + base64.b64encode((SPR / f"{key}.png").read_bytes()).decode()
        mons.append({"name": sp.strip(), "item": item.strip(), "ability": get("Ability:"), "sp": get("EVs:"),
                     "nature": nature, "moves": [l[2:] for l in lines if l.startswith("- ")], "img": img})
    return mons


ITEMS = HERE.parent / "item-sprites"                # all 148 Reg M-B items, from smogon/sprites (sources.json)
DOMAIN = json.loads((HERE.parent / "team-search-space/domain.json").read_text())
ROSTER = {f["id"]: f for f in DOMAIN["roster"]}
N_ITEMS = len(DOMAIN["items"])                      # 148 eligible held items (no item is also allowed)
N_ALIGN = len(DOMAIN["alignments"])                 # 21
# Stat Point spreads: six stats, each 0-32, summing to exactly 66 (domain rules)
ways = [1] + [0] * 66
for _ in range(6):
    ways = [sum(ways[t - k] for k in range(0, 33) if t - k >= 0) for t in range(67)]
N_SP = ways[66]


def item_img(name):
    p = ITEMS / f"item-{re.sub(r'[^a-z0-9]', '', name.lower())}.png"
    return "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode() if p.exists() else None


def counts(m):
    """How many legal choices this species had, per field (pinned Reg M-B simulator domain)."""
    f = ROSTER[re.sub(r"[^a-z0-9]", "", m["name"].lower())]
    abil = [p["ability"] for p in f["pairs"]]
    moves = set().union(*[p["moves"] for p in f["pairs"]])
    sets4 = sum(math.comb(len(p["moves"]), 4) for p in f["pairs"])   # ability-move pairs are validated together
    return {"abilities": len(abil), "items": N_ITEMS, "moves": len(moves), "alignments": N_ALIGN, "sp": N_SP,
            "closed": sets4 * (N_ITEMS + 1) * N_ALIGN * N_SP, "open": N_ALIGN * N_SP}


def enrich(mons):
    for m in mons:
        m["itemImg"] = item_img(m["item"]); m["n"] = counts(m)
    return mons


# Figure 1 reels: every legal option per field, from the pinned simulator
SHOWDOWN = Path.home() / ".local/share/vgc-pilot-runtime/unpacked/pokemon-showdown-913da3602a3aa1db79f9fdc5d5222eaf8d39569d"
MOVE_NAMES = {m.group(1): m.group(2) for m in re.finditer(r'^\t([a-z0-9]+): \{.*?\n\t\tname: "([^"]+)"',
                                                         (SHOWDOWN / "data/moves.ts").read_text(), re.S | re.M)}
STATS = ["HP", "Atk", "Def", "SpA", "SpD", "Spe"]


def spreads(seed, k=40):
    """k random legal spreads (six stats, 0-32 each, all 66 points), uniform over compositions by rejection."""
    import random
    rng, out = random.Random(seed), []
    while len(out) < k:
        cuts = sorted(rng.randint(0, 66) for _ in range(5))
        parts = [b - a for a, b in zip([0] + cuts, cuts + [66])]
        if max(parts) <= 32:
            out.append(" · ".join(f"{v} {st}" for v, st in zip(parts, STATS) if v))
    return out


def reel_options(mons):
    for r, m in enumerate(mons):
        f = ROSTER[re.sub(r"[^a-z0-9]", "", m["name"].lower())]
        m["abilities"] = [p["ability"] for p in f["pairs"]]
        m["learn"] = sorted(MOVE_NAMES.get(x, x) for x in set().union(*[p["moves"] for p in f["pairs"]]))
        m["spTrue"] = m["sp"].replace(" / ", " · ")
        m["spOpts"] = spreads(1000 + r) + [m["spTrue"]]
        assert m["ability"] in m["abilities"] and all(x in m["learn"] for x in m["moves"]), m["name"]
    return mons


def plan_matrix(you, them):
    """Measured 90 x 90 plan matrix (scripts/plan_matrix.py), re-indexed into this page's plan order:
    lead pairs in team-file order, then back pairs of the remaining four. Cells: MB522's wins and battles."""
    import itertools
    pm = json.loads((Path("/Users/ramiismael/Documents/code/vgc-team-generator-pilot/results/plan_matrix_MB522_MB763.json")).read_text())
    sid = lambda m: re.sub(r"[^a-z0-9]", "", m["name"].lower())
    def codes(mons):
        ids, out = [sid(m) for m in mons], []
        for l in itertools.combinations(range(6), 2):
            for b in itertools.combinations([i for i in range(6) if i not in l], 2):
                out.append("+".join(sorted(ids[i] for i in l)) + "/" + "+".join(sorted(ids[i] for i in b)))
        return out
    ri, ci = {c: i for i, c in enumerate(pm["row_plans"])}, {c: i for i, c in enumerate(pm["col_plans"])}
    rows, cols = [ri[c] for c in codes(you)], [ci[c] for c in codes(them)]
    return {"wins": [[pm["wins"][r][c] for c in cols] for r in rows], "battles": [[pm["battles"][r][c] for c in cols] for r in rows]}


data = {"you": {"id": "MB522", "mons": enrich(parse("MB522"))}, "them": {"id": "MB763", "mons": reel_options(enrich(parse("MB763")))},
        "items": [{"name": it["name"], "img": item_img(it["name"])} for it in DOMAIN["items"]],
        "natures": [a["name"] for a in DOMAIN["alignments"]]}
data["pm"] = plan_matrix(data["you"]["mons"], data["them"]["mons"])
html = (HERE / "template.html").read_text().replace("/*__DATA__*/null", json.dumps(data))
(HERE / "team-preview.html").write_text(html)
print("wrote team-preview.html", f"{len(html)/1024:.0f} KB")
