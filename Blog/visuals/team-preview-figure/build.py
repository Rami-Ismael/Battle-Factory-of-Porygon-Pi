"""Build preview-figure.html: the figure for the team-preview paragraph of the blog (sheets, 90 plans, 8,100 pairs,
game-theory terms). Reuses the team parser of ../team-preview-game/build.py; adds a few real meta teams the
reader can count with.  Run: python3 build.py
"""
import json
from pathlib import Path
HERE = Path(__file__).resolve().parent
src = (HERE.parent / "team-preview-game/build.py").read_text().split("data = {")[0]
g = {"__file__": str(HERE.parent / "team-preview-game/build.py")}; exec(src, g)
data = {k: {"id": v, "mons": g["parse"](v)} for k, v in g["TEAMS"].items()}
PICKS = ["MB522", "MB607", "MB715", "MB689"]          # real Reg M-B meta teams the reader can count with
data["picks"] = [{"id": t, "mons": g["parse"](t)} for t in PICKS]
html = (HERE / "template.html").read_text().replace("/*__DATA__*/null", json.dumps(data))
(HERE / "preview-figure.html").write_text(html)
print("wrote preview-figure.html", f"{len(html)/1024:.0f} KB")
