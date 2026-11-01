"""PROTOTYPE (throwaway): Turn One, rewritten from scratch.

Only data is carried over from ../turn-one.html: the two teams (MB522, MB763), their sprites and sets.
Everything else (scenes, field, layouts) lives in turn-one-redo-template.html.
Item sprites come from ../../item-sprites; move targets from code/data/moves.json (Showdown).
Run: python3 build_turn_one_redo.py   ->  turn-one-redo.html
"""
import base64, json, pathlib, re

here = pathlib.Path(__file__).parent
bg = here.parent
visuals = bg.parent
repo = visuals.parent.parent.parent
orig = (bg / "turn-one.html").read_text()
D = json.loads(re.search(r"const D = (\{.*?\});\nconst \$ =", orig, re.S).group(1))

def slug(item):
    return "item-" + re.sub(r"[^a-z0-9]", "", item.lower())

D["items"] = {}
for side in ("you", "them"):
    for mon in D[side]["mons"]:
        f = visuals / "item-sprites" / f"{slug(mon['item'])}.png"
        D["items"][slug(mon["item"])] = "data:image/png;base64," + base64.b64encode(f.read_bytes()).decode()

moves = json.loads((repo / "code" / "data" / "moves.json").read_text())
D["targets"] = {}
for side in ("you", "them"):
    for mon in D[side]["mons"]:
        for mv in mon["moves"]:
            k = re.sub(r"[^a-z0-9]", "", mv.lower())
            D["targets"][k] = moves[k]["target"]

html = (here / "turn-one-redo-template.html").read_text().replace("/*__DATA__*/null", json.dumps(D))
out = here / "turn-one-redo.html"
out.write_text(html)
print(f"{len(html):,} bytes -> {out.name}")
