"""Build index.html: search the Regulation M-B teams that do NOT bring Kingambit.
Reads the local matchup database (read-only) and the counter grid. Run: python3 build.py
"""
import base64, json, os, re, sqlite3
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPR = HERE.parents[1] / "notes/Blog/visuals/search-loop/sprites"
DB = os.path.expanduser("~/vgc-data/matchup_regmb.sqlite")
GRID = os.path.expanduser("~/Documents/code/vgc-team-generator-pilot/results/counter_matrix.json")
MIN_BATTLES = 96
BANNED = "Kingambit"

c = sqlite3.connect("file:" + DB + "?mode=ro", uri=True)
c.execute("create temp table top as select team_id from team_set where set_name='top50'")
c.execute("create temp table real as select distinct team_id from team_source")
rows = c.execute(f"""
  select t.team_id, t.origin, t.paste, sum(m.wins), sum(m.battles), (t.team_id in (select team_id from real))
  from team t join matchup m on m.row_team = t.team_id and m.policy_id = 1
  join top on top.team_id = m.col_team and top.team_id <> t.team_id
  where t.paste not like '%{BANNED}%' and (t.legal is null or t.legal = 1)
  group by t.team_id having sum(m.battles) >= {MIN_BATTLES}""").fetchall()

def parse(paste):
    mons = []
    for blk in paste.strip().split("\n\n"):
        L = blk.splitlines()
        sp, _, item = re.sub(r" \([MF]\)", "", L[0]).partition(" @ ")
        g = lambda k: next((l.split(": ", 1)[1] for l in L if l.startswith(k + ":")), "")
        mons.append({"n": sp.strip(), "i": item.strip(), "a": g("Ability"), "nat": next((l.split()[0] for l in L if l.endswith(" Nature")), ""),
                     "ev": g("EVs"), "m": [l[2:] for l in L if l.startswith("- ")]})
    return mons

def kind(origin, is_real):
    return "real" if is_real or origin == "vgcpastes" else "search"

teams = []
for tid, origin, paste, w, n, is_real in rows:
    teams.append({"id": f"T{tid}", "tid": tid, "k": kind(origin, is_real), "w": w, "b": n, "opp": 49, "mons": parse(paste),
                  "how": origin.split(":")[0]})

grid = json.load(open(GRID))
for cand, row in zip(grid["candidates"], grid["cells"]):
    if BANNED in cand["species"]:
        continue
    tid, paste = c.execute("select t.team_id, t.paste from team_source s join team t using(team_id) where s.source_id=?", (cand["id"],)).fetchone()
    teams = [t for t in teams if t["id"] != f"T{tid}"]
    teams.append({"id": cand["id"], "tid": tid, "k": "real", "w": sum(x["wins"] for x in row), "b": sum(x["n"] for x in row), "opp": 50,
                  "mons": parse(paste), "how": "counter grid", "_grid": {o["id"]: (x["wins"], x["n"]) for o, x in zip(grid["opponents"], row)}})

# ---- matchup matrix data: the 49 top-placing opponents as columns, one (wins, battles) pair per team per opponent
opp_rows = c.execute("""select s.team_id, min(ts.source_id), min(coalesce(ts.event,'')), min(coalesce(ts.placement,'')), t.paste
    from team_set s join team t using(team_id) join team_source ts using(team_id) where s.set_name='top50' group by s.team_id order by s.team_id""").fetchall()
OPP = [{"id": r[1], "event": r[2], "place": r[3], "tid": r[0], "mons": [m["n"] for m in parse(r[4])]} for r in opp_rows]
oidx = {o["tid"]: i for i, o in enumerate(OPP)}
by_src = {o["id"]: i for i, o in enumerate(OPP)}
for t in teams:
    w, n = [0]*len(OPP), [0]*len(OPP)
    if "_grid" in t:
        for src, (ww, nn) in t.pop("_grid").items():
            if src in by_src: w[by_src[src]], n[by_src[src]] = ww, nn
    else:
        for col, ww, nn in c.execute("select col_team, wins, battles from matchup where policy_id=1 and row_team=?", (t["tid"],)):
            if col in oidx: w[oidx[col]], n[oidx[col]] = ww, nn
    t["cw"], t["cn"] = w, n
    del t["tid"]
print(len(OPP), "opponent columns")

species = sorted({m["n"] for t in teams for m in t["mons"]} | {s for o in OPP for s in o["mons"]})
sprites, missing = {}, []
for s in species:
    key = re.sub(r"[^a-z0-9]", "", s.lower())
    for cand in (key, re.sub(r"[^a-z0-9]", "", s.split("-")[0].lower())):
        f = SPR / f"{cand}.png"
        if f.exists():
            sprites[s] = "data:image/png;base64," + base64.b64encode(f.read_bytes()).decode(); break
    else:
        missing.append(s)
print(len(teams), "teams;", len(species), "species;", "no sprite:", missing)
data = {"teams": teams, "sprites": sprites, "opp": OPP}
html = (HERE / "template.html").read_text().replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":")))
(HERE / "index.html").write_text(html)
print("wrote html", f"{len(html)/1024:.0f} KB")
