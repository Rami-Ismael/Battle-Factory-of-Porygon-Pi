"""Build turn-one.html: the battle as a game (sibling of team-preview-game/four-of-six.html).
Same two Reg M-B meta teams; sprites embedded from ../search-loop/sprites.
Run: python3 build.py
"""
import importlib.util, json
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("preview", HERE.parent / "team-preview-game/build.py")
src = (HERE.parent / "team-preview-game/build.py").read_text().split("data = {")[0]   # reuse parse(), TEAMS
g = {"__file__": str(HERE.parent / "team-preview-game/build.py")}; exec(src, g)
data = {k: {"id": v, "mons": g["parse"](v)} for k, v in g["TEAMS"].items()}
html = (HERE / "template.html").read_text().replace("/*__DATA__*/null", json.dumps(data))
(HERE / "turn-one.html").write_text(html)
print("wrote turn-one.html", f"{len(html)/1024:.0f} KB")
