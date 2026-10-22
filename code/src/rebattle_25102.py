"""Fresh confirmation of team 25102, the wide-screening loop team (big-pool arm, seed 1), 2026-10-21.
It already has 8 battles per top-50 opponent (4 from the loop, 49 of which picked it, plus 4 from confirm_top).
Raise every cell to 24 and score the NEW battles only: 16 per opponent x 49 distinct teams = 784.
    SUPERVISOR_LOG=/tmp/vgc-pilot/rebattle_25102 python battle_supervisor.py rebattle_25102.py run
"""
import json, math, sys
from pathlib import Path
sys.path.insert(0, "/tmp/vgc-pilot/src"); sys.path.insert(0, str(Path(__file__).resolve().parent))
import lossdown as L, lossdown_battle as LB, matchup_db as MDB
OUT = L.RES / "rebattle_25102.json"
TEAM, PER_CELL = 25102, 24

def run():
    LB._stack(); con = MDB.connect()
    cols = [c for c in MDB.set_members(con, "top50") if c != TEAM]
    weight = dict(con.execute("SELECT team_id, weight FROM team_set WHERE set_name='top50'"))
    before = MDB.row_cells(con, LB.POLICY, TEAM, cols)
    MDB.ensure(con, LB.POLICY, [TEAM], cols, PER_CELL, note="rebattle_25102 2026-10-21")
    after = MDB.row_cells(con, LB.POLICY, TEAM, cols)
    new = {c: (after[c][0] - before.get(c, (0, 0))[0], after[c][1] - before.get(c, (0, 0))[1]) for c in cols}
    w = sum(v[0] for v in new.values()); n = sum(v[1] for v in new.values()); p = w / n
    weighted = sum(weight[c] * a / b for c, (a, b) in new.items() if b) / sum(weight[c] for c, (a, b) in new.items() if b)
    out = dict(team_id=TEAM, opponents=len(cols), new_wins=w, new_battles=n, fresh=p, se=math.sqrt(p * (1 - p) / n),
               fresh_weighted_top50=weighted, beat=sum(a / b > .5 for a, b in new.values() if b),
               per_opponent={str(c): new[c] for c in cols})
    json.dump(out, open(OUT, "w"), indent=1)
    print(f"team {TEAM}: {w}/{n} fresh = {p:.3f} ± {out['se']:.3f}; weighted {weighted:.3f}; won >50% vs {out['beat']} of {len(cols)}", flush=True)

if __name__ == "__main__": run()
