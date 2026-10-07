"""Build seven-jobs.html: the seven roles a diffusion model plays in optimisation, animated as loops
around the battle simulator, plus a do-now / test-next / skip board for the VGC team search.
Sprites embedded from ../search-loop/sprites.  Run: python3 build.py
"""
import base64, json
from pathlib import Path
HERE = Path(__file__).resolve().parent
SPR = HERE.parent / "search-loop/sprites"
NAMES = ["pelipper","charizard","venusaur","archaludon","grimmsnarl","basculegion","incineroar","farigiraf",
         "garchomp","sylveon","aerodactyl","kingambit","whimsicott","sinistcha","gholdengo","sneasler","floetteeternal"]
sprites = {n: "data:image/png;base64," + base64.b64encode((SPR / f"{n}.png").read_bytes()).decode()
           for n in NAMES if (SPR / f"{n}.png").exists()}
html = (HERE / "template.html").read_text().replace("/*__SPRITES__*/null", json.dumps(sprites))
(HERE / "seven-jobs.html").write_text(html)
print("wrote seven-jobs.html", f"{len(html)/1024:.0f} KB", len(sprites), "sprites")
