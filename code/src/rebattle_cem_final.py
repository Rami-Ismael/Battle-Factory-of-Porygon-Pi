"""Tie-break for the five best teams of rebattle_cem_top.json, 2026-10-21: 48 NEW battles per top-50 opponent
(49 distinct teams), 2,352 per team, scored on these battles only, so the pick of the five does not bias the score.
    SUPERVISOR_LOG=/tmp/vgc-pilot/rebattle_cem_final python battle_supervisor.py rebattle_cem_final.py run
"""
import json, math, sys
from pathlib import Path
sys.path.insert(0, "/tmp/vgc-pilot/src"); sys.path.insert(0, str(Path(__file__).resolve().parent))
import lossdown as L, lossdown_battle as LB, matchup_db as MDB
OUT = L.RES / "rebattle_cem_final.json"
# The five best fresh scores in rebattle_cem_top.json (0.695, 0.695, 0.690, 0.690, 0.682).
TEAMS = [47383, 75964, 44959, 29158, 34087]
NEW = 48

def run():
    LB._stack(); con = MDB.connect()
    allcols = MDB.set_members(con, "top50")
    weight = dict(con.execute("SELECT team_id, weight FROM team_set WHERE set_name='top50'"))
    out = json.load(open(OUT)) if OUT.exists() else {}
    for t in TEAMS:
        if str(t) in out: continue
        cols = [c for c in allcols if c != t]
        before = MDB.row_cells(con, LB.POLICY, t, cols)
        target = max(n for _, n in before.values()) + NEW
        MDB.ensure(con, LB.POLICY, [t], cols, target, note=f"rebattle_cem_final {t} 2026-10-21")
        after = MDB.row_cells(con, LB.POLICY, t, cols)
        new = {c: (after[c][0] - before.get(c, (0, 0))[0], after[c][1] - before.get(c, (0, 0))[1]) for c in cols}
        w = sum(a for a, _ in new.values()); n = sum(b for _, b in new.values()); p = w / n
        origin, paste = con.execute("SELECT origin, paste FROM team WHERE team_id=?", (t,)).fetchone()
        out[str(t)] = dict(origin=origin, new_wins=w, new_battles=n, fresh=p, se=math.sqrt(p * (1 - p) / n),
                           fresh_weighted_top50=sum(weight[c] * a / b for c, (a, b) in new.items() if b) / sum(weight[c] for c, (a, b) in new.items() if b),
                           beat=sum(a / b > .5 for a, b in new.values() if b), paste=paste,
                           per_opponent={str(c): new[c] for c in cols})
        json.dump(out, open(OUT, "w"), indent=1)
        print(f"team {t} ({origin}): {w}/{n} fresh = {p:.3f} ± {out[str(t)]['se']:.3f}; won >50% vs {out[str(t)]['beat']} of {len(cols)}", flush=True)

if __name__ == "__main__": run()
