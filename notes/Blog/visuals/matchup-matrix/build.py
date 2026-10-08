"""Build matchup-matrix.html from the live SQLite matchup database (~/vgc-data/matchup_regmb.sqlite):
the tested rows against the top-50 columns, every team as its six sprites, plus the measured costs.
Run: python3 build.py
"""
import base64, json, os, re, sqlite3
from pathlib import Path

HERE = Path(__file__).resolve().parent
DB = Path(os.environ.get("MATCHUP_DB", Path.home() / "vgc-data" / "matchup_regmb.sqlite"))
COST = Path("/Users/ramiismael/Documents/code/vgc-team-generator-pilot/results/matchup_cost.json")
SPR = HERE.parent / "search-loop/sprites"

def norm(s): return re.sub(r"[^a-z0-9]", "", s.lower())
def species(paste):
    out = []
    for blk in paste.strip().split("\n\n"):
        head = blk.splitlines()[0]
        sp, _, item = head.partition(" @ ")
        out.append({"name": sp.strip(), "key": norm(sp)})
    return out

con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
pid = con.execute("SELECT policy_id FROM policy ORDER BY policy_id LIMIT 1").fetchone()[0]
cols = con.execute("SELECT ts.team_id, ts.weight, t.paste FROM team_set ts JOIN team t USING(team_id) "
                   "WHERE set_name='top50' ORDER BY ts.team_id").fetchall()
src = dict(con.execute("SELECT team_id, group_concat(source_id, ' = ') FROM team_source GROUP BY team_id"))
rows_ids = [r[0] for r in con.execute(
    "SELECT DISTINCT row_team FROM matchup WHERE policy_id=? AND col_team IN (SELECT team_id FROM team_set WHERE set_name='top50') "
    "AND row_team NOT IN (SELECT team_id FROM team_set WHERE set_name='top50') ORDER BY row_team", (pid,))]
colix = {c[0]: i for i, c in enumerate(cols)}
rows = []
for r in rows_ids:
    paste = con.execute("SELECT paste FROM team WHERE team_id=?", (r,)).fetchone()[0]
    cells = {}
    for c, w, n in con.execute("SELECT col_team, wins, battles FROM matchup WHERE policy_id=? AND row_team=?", (pid, r)):
        if c in colix: cells[colix[c]] = [w, n]
    rows.append({"id": r, "src": src.get(r, ""), "mons": species(paste), "cells": cells})
batches = [dict(zip(("battles", "seconds", "unattributed", "note"), b)) for b in
           con.execute("SELECT battles, seconds, unattributed, note FROM batch ORDER BY batch_id")]
stats = {k: con.execute(q).fetchone()[0] for k, q in {
    "teams": "SELECT COUNT(*) FROM team", "files": "SELECT COUNT(*) FROM team_source",
    "cells": "SELECT COUNT(*) FROM cell", "battles": "SELECT SUM(battles) FROM cell"}.items()}
data = {"cols": [{"id": c, "weight": w, "src": src.get(c, ""), "mons": species(p)} for c, w, p in cols],
        "rows": rows, "batches": batches, "stats": stats, "cost": json.load(open(COST))}
keys = {m["key"] for t in data["cols"] + data["rows"] for m in t["mons"]}
sprites = {}
for k in keys:
    p = SPR / f"{k}.png"
    if not p.exists() and "mega" in k: p = SPR / f"{k.split('mega')[0]}.png"   # no Mega sprite: use the base form
    if p.exists(): sprites[k] = "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode()
data["sprites"] = sprites
missing = sorted(keys - set(sprites))
html = (HERE / "template.html").read_text().replace("/*__DATA__*/null", json.dumps(data))
(HERE / "matchup-matrix.html").write_text(html)
print("wrote matchup-matrix.html", f"{len(html)/1024:.0f} KB", len(rows), "rows ×", len(cols), "cols; missing sprites:", missing)
