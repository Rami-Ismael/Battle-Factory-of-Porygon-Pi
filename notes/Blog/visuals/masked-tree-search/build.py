"""Build masked-tree-search.html: an animated explainer of masked diffusion tree search
(Diffusion Large Language Models for Black-Box Optimization, arXiv 2601.14446) applied to
Reg M-B team generation. Sprites embedded from ../search-loop/sprites.  Run: python3 build.py
"""
import base64, json
from pathlib import Path
HERE = Path(__file__).resolve().parent
SPR = HERE.parent / "search-loop/sprites"
NAMES = ["pelipper","archaludon","charizard","venusaur","basculegion","grimmsnarl","incineroar","farigiraf",
         "garchomp","sylveon","aerodactyl","kingambit","whimsicott","sinistcha","gholdengo","sneasler","floetteeternal"]
sprites = {n: "data:image/png;base64," + base64.b64encode((SPR / f"{n}.png").read_bytes()).decode() for n in NAMES}
html = (HERE / "template.html").read_text().replace("/*__SPRITES__*/null", json.dumps(sprites))
(HERE / "masked-tree-search.html").write_text(html)
print("wrote masked-tree-search.html", f"{len(html)/1024:.0f} KB")
