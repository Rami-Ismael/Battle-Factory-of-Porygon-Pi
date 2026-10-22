"""Build index.html from template.html: inline the figure data written by
src/random_battle_graph_report.py (code repo) and the sprites it needs from ../search-loop/sprites."""
import base64, json, re
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIG = Path.home() / "Documents/code/vgc-team-generator-pilot/results/random_battle_graph_figure.json"
SPRITES = HERE.parent / "search-loop" / "sprites"


def names(fig):
    out = set(fig["row_team"]) | set(fig["opp_team"])
    for t in fig["teams"]: out |= set(t["species"])
    def walk(n):
        for side in n.get("detail", []):
            out.update(p["species"] for p in side)
        for side in n.get("path", {}).get("detail", []):
            out.update(p["species"] for p in side)
        for c in n.get("children", []): walk(c)
    walk(fig["tree"]["root"])
    return out


def main():
    fig = json.loads(FIG.read_text())
    key = lambda s: re.sub(r"[^a-z0-9]", "", s.lower())
    sprites, missing = {}, []
    for nm in sorted(names(fig)):
        f = SPRITES / f"{key(nm)}.png"
        if not f.exists(): f = SPRITES / f"{key(nm.split('-')[0])}.png"
        if f.exists(): sprites[nm] = "data:image/png;base64," + base64.b64encode(f.read_bytes()).decode()
        else: missing.append(nm)
    fig["sprites"] = sprites
    page = (HERE / "template.html").read_text().replace("/*DATA*/", "window.RBT = " + json.dumps(fig, separators=(",", ":")) + ";")
    (HERE / "index.html").write_text(page)
    print(f"index.html {len(page) // 1024} KB, {len(sprites)} sprites, missing {missing}")


if __name__ == "__main__":
    main()
