"""Build four-of-six.html: team preview as a simultaneous-move game, with two real Reg M-B meta teams.
Teams are read from the vgc-bench Reg M-B pool; sprites embedded from ../search-loop/sprites.
Run: python3 build.py
"""
import base64, json, re
from pathlib import Path

HERE = Path(__file__).resolve().parent
POOL = Path("/tmp/vgc-pilot/vgc-bench/teams/reg_mb")
SPR = HERE.parent / "search-loop/sprites"
TEAMS = {"you": "MB522", "them": "MB763"}      # MB763: 1st place, Talon's Fight Club #98 (results/top50_evs.json)

def parse(tid):
    mons = []
    for blk in (POOL / f"{tid}.txt").read_text().strip().split("\n\n"):
        lines = blk.splitlines()
        head = re.sub(r" \([MF]\)", "", lines[0])
        sp, _, item = head.partition(" @ ")
        abil = next(l.split(": ", 1)[1] for l in lines if l.startswith("Ability:"))
        moves = [l[2:] for l in lines if l.startswith("- ")]
        key = re.sub(r"[^a-z0-9]", "", sp.lower())
        img = "data:image/png;base64," + base64.b64encode((SPR / f"{key}.png").read_bytes()).decode()
        mons.append({"name": sp.strip(), "item": item.strip(), "ability": abil, "moves": moves, "img": img})
    return mons

data = {k: {"id": v, "mons": parse(v)} for k, v in TEAMS.items()}
html = (HERE / "template.html").read_text().replace("/*__DATA__*/null", json.dumps(data))
(HERE / "four-of-six.html").write_text(html)
print("wrote four-of-six.html", f"{len(html)/1024:.0f} KB")
