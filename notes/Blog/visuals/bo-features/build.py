"""Build bo-features.html: inline the sprites and the measured numbers into template.html.

Every number comes from the pilot repo's results files (no battles are run here):
counter_matrix.json, ruggedness2.json, multimodality.json, plus the Version 2/3 tables of
Teaching/reference/bo-benchmark-derived-vgc-research-comparison.md.
"""
import base64, json, re
from pathlib import Path

HERE = Path(__file__).parent
SPRITES = HERE.parent / "search-loop" / "sprites"
RESULTS = Path("/Users/ramiismael/Documents/code/vgc-team-generator-pilot/results")


def sid(name):
    return re.sub(r"[^a-z0-9]", "", name.lower())


def sprite(name):
    s = sid(name)
    for cand in (s, s.replace("mega", "")):
        p = SPRITES / f"{cand}.png"
        if p.exists():
            return "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode()
    return None


counter = json.loads((RESULTS / "counter_matrix.json").read_text())
teams = {}
for cand, row in zip(counter["candidates"], counter["cells"]):
    w = sum(c["wins"] for c in row)
    n = sum(c["n"] for c in row)
    teams[cand["id"]] = {"id": cand["id"], "kind": cand["kind"], "p": w / n, "n": n, "species": cand["species"]}

rug = json.loads((RESULTS / "ruggedness2.json").read_text())
edits = {}
for start, ops in rug["vgc"].items():
    for op, v in ops.items():
        if isinstance(v, dict):
            edits.setdefault(op, {})[start] = {
                "rho": v["rho"], "ci": v["rho_ci95"], "delta": v["delta_star"],
                "beyond": v["frac_beyond_noise"], "repeat": v["frac_replicates_beyond_noise"],
                "n": v["edits"], "T1": v["T1"], "T2": v["T2"],
            }

multi = json.loads((RESULTS / "multimodality.json").read_text())
climbs = multi["vgc"]["endpoints"]

needed = set()
for t in teams.values():
    needed.update(t["species"])
for c in climbs:
    needed.update(c["species"])
needed.update(["Eelektross", "Clefable"])  # Figure 4 replacement species (ruggedness2 m00_F0, s02_F0)
sprites = {sid(n): sprite(n) for n in sorted(needed)}

data = {"teams": teams, "edits": edits, "climbs": climbs,
        "bench_multi": {k: {"ends": v["distinct_endpoints"], "true": v["true_local_optima_share"]}
                        for k, v in multi["bench"].items()},
        "sprites": sprites}

html = (HERE / "template.html").read_text().replace("/*DATA*/null", json.dumps(data, separators=(",", ":")))
(HERE / "bo-features.html").write_text(html)
missing = [k for k, v in sprites.items() if v is None]
print("wrote bo-features.html", len(html) // 1024, "KB; missing sprites:", missing)
