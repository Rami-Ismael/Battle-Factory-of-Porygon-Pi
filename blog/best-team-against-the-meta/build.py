"""Build index.html for the "Best Team Against the Meta" figure.

Reads teams.json (written from the laptop results files; each team names its source),
embeds the Pokemon sprites and item icons as data URIs, and fills template.html.
Run from the repository root:  python3 blog/best-team-against-the-meta/build.py
"""
import base64, json, re
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SPR = ROOT / "notes/Blog/visuals/search-loop/sprites"
ITM = ROOT / "notes/Blog/visuals/item-sprites"

ident = lambda s: re.sub(r"[^a-z0-9]", "", s.lower())


def uri(path):
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


def parse(paste):
    mons = []
    for block in [b for b in paste.strip().split("\n\n") if b.strip()]:
        lines = [l.strip() for l in block.splitlines() if l.strip()]
        head, _, item = lines[0].partition("@")
        species = re.sub(r"\s*\((M|F)\)\s*$", "", head.strip())
        mon = dict(species=species, item=item.strip(), ability="", alignment="", points="", moves=[])
        for l in lines[1:]:
            if l.startswith("Ability:"): mon["ability"] = l[8:].strip()
            elif l.startswith("EVs:"): mon["points"] = l[4:].strip()
            elif l.endswith("Nature"): mon["alignment"] = l[:-6].strip()
            elif l.startswith("- "): mon["moves"].append(l[2:])
        mons.append(mon)
    return mons


data = json.loads((HERE / "teams.json").read_text())
sprites, items, missing = {}, {}, []
for t in data["teams"]:
    t["mons"] = parse(t.pop("paste"))
    for m in t["mons"]:
        s = ident(m["species"])
        for cand in (s, s.replace("mega", "").rstrip("xy") if "mega" in s else s):
            f = SPR / f"{cand}.png"
            if f.exists():
                sprites[m["species"]] = uri(f); break
        else:
            missing.append(m["species"])
        if m["item"]:
            f = ITM / f"item-{ident(m['item'])}.png"
            if f.exists(): items[m["item"]] = uri(f)
            else: missing.append("item " + m["item"])
data["teams"].sort(key=lambda t: -t["rate"])
data["sprites"], data["items"] = sprites, items
html = (HERE / "template.html").read_text().replace("/*__DATA__*/null", json.dumps(data))
(HERE / "index.html").write_text(html)
print(f"wrote index.html: {len(data['teams'])} teams, {len(sprites)} sprites, {len(items)} item icons; missing: {missing or 'none'}")
