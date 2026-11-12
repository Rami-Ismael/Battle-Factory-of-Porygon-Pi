"""Build index.html from template.html and data/. Every number on the page is read from data/ or written in the prose.

    python3 notes/Blog/visuals/looped-diffusion-teach/build.py
"""
import base64
import json
import math
from pathlib import Path

HERE = Path(__file__).parent
RAW = json.loads((HERE / "data/field_difficulty.json").read_text())
PAPERS = json.loads((HERE / "data/paper_numbers.json").read_text())
SPRITES = HERE.parent / "item-sprites"
ITEM_NAMES = json.loads((SPRITES / "sources.json").read_text())
DISPLAY = {"incineroar": ("Incineroar", "incineroar"), "milotic": ("Milotic", "milotic"), "mimikyu": ("Mimikyu", "mimikyu"),
           "ninetalesalola": ("Alolan Ninetales", "ninetales-alola"), "raichu": ("Raichu", "raichu"), "sinistcha": ("Sinistcha", "sinistcha")}
STATS = ["HP", "Atk", "Def", "SpA", "SpD", "Spe"]


def group(f):
    return "move" if f.startswith("move") else ("Stat Points" if f.startswith("Stat Points") else f)


g = RAW["lock_grid"]
PASTE = Path.home() / "vgc-data/teams/reg_mb/MB607.txt"
SHOWN = {}
for line in PASTE.read_text().splitlines():                 # display names for the decoded field ids
    text = line.removeprefix("- ").removeprefix("Ability: ").split(" Nature")[0].strip()
    SHOWN["".join(ch for ch in text.lower() if ch.isalnum())] = text
team = []
for slot in g["decoded"]:
    name, sprite = DISPLAY[slot["species"]]
    fields = dict(ability=SHOWN[slot["ability"]], item=slot["item"])
    for j, m in enumerate(slot["moves"]): fields[f"move {j + 1}"] = SHOWN[m]
    fields["Stat Alignment"] = SHOWN[slot["nature"]]
    for s in STATS: fields[f"Stat Points {s}"] = str(slot["evs"][s])
    fields["item"] = ITEM_NAMES[slot["item"]]["name"]
    team.append(dict(name=name, sprite=sprite, item=slot["item"], fields=fields))

grid, mean = {}, {}
for c in g["conditions"]:
    key = "none" if c["lock"] is None else str(c["lock"])
    cells = {i: {} for i in range(6)}
    by = {}
    for x in c["cells"]:
        cells[x["pokemon"]][x["field"]] = round(math.exp(x["entropy"]), 2)
        by.setdefault(group(x["field"]), []).append(x["entropy"])
    grid[key] = cells
    mean[key] = {k: math.exp(sum(v) / len(v)) for k, v in by.items()}

commit = {}
for r in RAW["by_field"]: commit.setdefault(r["field"], []).append(r["entropy"])
commit = {k: math.exp(sum(v) / len(v)) for k, v in commit.items()}

IC = RAW["item_choice"]
keys = {k for v in IC["free"].values() for k, _ in v} | set(IC["true_items"])
for cond, lists in IC["given"].items():
    keys.add(cond.split("|")[1])
    for v in lists.values(): keys |= {k for k, _ in v}
icons = {k: "data:image/png;base64," + base64.b64encode((SPRITES / f"item-{k}.png").read_bytes()).decode() for k in sorted(keys)}
names = {k: ITEM_NAMES[k]["name"] for k in sorted(keys)}

field = dict(team=team, grid=grid, mean=mean, commit=commit, icons=icons, names=names,
             clash=dict(item=RAW["clash_item"]["sampled_together"], species=RAW["clash_species"]["sampled_together"]),
             items=dict(free=IC["free"], given=IC["given"], givable=sorted(IC["given"])))
out = (HERE / "template.html").read_text().replace("__DATA__", json.dumps(dict(field=field, papers=PAPERS), separators=(",", ":")))
(HERE / "index.html").write_text(out)
print("index.html", len(out) // 1024, "KB;", "commit:", {k: round(v, 2) for k, v in commit.items()})
print("clash", field["clash"], "binding", RAW["binding"])
