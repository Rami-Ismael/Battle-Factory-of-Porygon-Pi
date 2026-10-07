"""Build self-consuming-loop.html: a toy 2D simulation of self-consuming retraining and the fixes
from the literature (docs/self-consuming-loop-fixes.md in the code repo). Archetypes are anchored by
real sprites from ../search-loop/sprites.  Run: python3 build.py
"""
import base64, json
from pathlib import Path
HERE = Path(__file__).resolve().parent
SPR = HERE.parent / "search-loop/sprites"
NAMES = ["pelipper","archaludon","charizard","venusaur","farigiraf","sinistcha","whimsicott","aerodactyl","incineroar","garchomp"]
sprites = {n: "data:image/png;base64," + base64.b64encode((SPR / f"{n}.png").read_bytes()).decode() for n in NAMES}
html = (HERE / "template.html").read_text().replace("/*__SPRITES__*/null", json.dumps(sprites))
(HERE / "self-consuming-loop.html").write_text(html)
print("wrote self-consuming-loop.html", f"{len(html)/1024:.0f} KB")
