"""Build the slot-order page: results JSON + sprites -> slot-order.html (2026-10-16).

    python3 build.py
Reads results/slot_order_20261016.json from the code repo (numbers and their sources) and embeds
only the six sprites the page shows, from ../search-loop/sprites. Template: template.html
(placeholders /*DATA*/null and /*SPRITES*/{}).
"""
import base64, json
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = Path("/Users/ramiismael/Documents/code/vgc-team-generator-pilot/results/slot_order_20261016.json")
SPRITES = HERE.parent / "search-loop" / "sprites"
SHOWN = ["sylveon", "chandelure", "garchomp", "basculegion", "kingambit", "floetteeternal"]


def main():
    data = json.load(open(DATA))
    sprites = {s: "data:image/png;base64," + base64.b64encode((SPRITES / f"{s}.png").read_bytes()).decode()
               for s in SHOWN if (SPRITES / f"{s}.png").exists()}
    html = (HERE / "template.html").read_text()
    assert "/*DATA*/null" in html and "/*SPRITES*/{}" in html
    html = html.replace("/*DATA*/null", json.dumps(data, separators=(",", ":"))).replace("/*SPRITES*/{}", json.dumps(sprites))
    (HERE / "slot-order.html").write_text(html)
    print(f"slot-order.html: {len(html) / 1e3:.0f} kB · sprites {len(sprites)}/{len(SHOWN)}")


if __name__ == "__main__":
    main()
