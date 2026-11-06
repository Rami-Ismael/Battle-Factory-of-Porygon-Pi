"""Build the ask-frequencies page: data.json (from build_data.py) + the six core sprites + template.html -> index.html.

    python3 build.py
"""
import base64
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPRITES = HERE.parent / "search-loop" / "sprites"
CORE = ["garchomp", "basculegion", "kingambit", "whimsicott", "charizard", "floetteeternal"]

data = json.loads((HERE / "data.json").read_text())
sprites = {s: "data:image/png;base64," + base64.b64encode((SPRITES / f"{s}.png").read_bytes()).decode()
           for s in CORE if (SPRITES / f"{s}.png").exists()}
html = (HERE / "template.html").read_text()
html = html.replace("/*DATA*/null", json.dumps(data, separators=(",", ":"))).replace("/*SPRITES*/{}", json.dumps(sprites))
(HERE / "index.html").write_text(html)
print(f"index.html: {len(html):,} bytes, {len(sprites)} sprites")
