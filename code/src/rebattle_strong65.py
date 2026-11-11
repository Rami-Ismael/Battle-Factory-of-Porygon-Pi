"""Re-battle the strong training teams (docs/subset-ceiling-tests.md, round 2).

The 868 training teams whose all-39 label is 0.7 or more were picked by a label from about 39 battles, so part of their
lead may be luck. This tops every cell between them and the 39 training opponents up to 3 battles, and the same for
868 uniformly drawn training teams as the measurement check, then scores each team on the new battles only.

    python src/rebattle_strong65.py prepare     pick the teams and snapshot their stored battles
    python src/rebattle_strong65.py battle      top every cell up to PER_CELL battles, slots shuffled
    python src/rebattle_strong65.py report      win rate on the new battles only, against the label
RS65_SMOKE=1 uses 4 teams per group and 3 opponents, into results/rebattle_strong_smoke.json.
"""

import json
import os
import sqlite3
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import train65 as T65  # puts /tmp/vgc-pilot/src on the path
import lossdata65 as D

SMOKE = bool(os.environ.get("RS65_SMOKE"))
STRONG, SEED, PER_CELL, BATTLE_SEED = 0.7, 8500, 3, 8600
OUT = T65.RES / ("rebattle_strong_smoke.json" if SMOKE else "rebattle_strong_gen65.json")


def save(out):
    tmp = OUT.with_suffix(".tmp")
    json.dump(out, open(tmp, "w"), indent=1)
    tmp.replace(OUT)


def counts(con, policy, team, cols):
    """{opponent: [wins, battles]} for one team from stored cells, either side of the pair."""
    ph = ",".join("?" * len(cols))
    out = {c: [0, 0] for c in cols}
    for opp, wins, b in con.execute(f"SELECT hi, lo_wins, battles FROM cell WHERE policy_id = ? AND lo = ? AND hi IN ({ph}) UNION ALL "
                                    f"SELECT lo, hi_wins, battles FROM cell WHERE policy_id = ? AND hi = ? AND lo IN ({ph})",
                                    [policy, team, *cols, policy, team, *cols]):
        out[opp][0] += wins
        out[opp][1] += b
    return out


def prepare():
    if OUT.exists():
        raise SystemExit(f"{OUT} exists: the snapshot is taken once, before any new battle")
    d, tr, _, sha = T65.load()
    C = d["C"][tr]
    full = T65.full_label(C)
    tid = d["team_id"][tr]
    strong = np.flatnonzero(full >= STRONG)
    rng = np.random.default_rng(SEED)
    random_rows = rng.choice(len(full), len(strong), replace=False)       # the same draw as strong_teams65.py
    if SMOKE:
        strong, random_rows = strong[:4], random_rows[:4]
    con = sqlite3.connect(f"file:{D.DB}?mode=ro", uri=True)
    sides, policy = T65.sides_and_policy(con)
    cols = sides["training"][:3] if SMOKE else sides["training"]
    teams = {}
    for group, rows in (("strong", strong), ("random", random_rows)):
        for r in rows:
            t = int(tid[r])
            if str(t) in teams:                                            # a random draw that is also strong stays strong
                continue
            teams[str(t)] = dict(group=group, label=float(full[r]), before={str(k): v for k, v in counts(con, policy, t, cols).items()})
    save(dict(policy=policy, opponents=cols, per_cell=PER_CELL, data_sha256=sha, strong_threshold=STRONG, teams=teams))
    print(f"{sum(v['group'] == 'strong' for v in teams.values())} strong and {sum(v['group'] == 'random' for v in teams.values())} random teams snapshotted")


def battle():
    import matchup_db as MDB

    out = json.load(open(OUT))
    con = MDB.connect()
    if MDB.SLOT_ORDER != "shuffled":
        raise SystemExit("these battles must shuffle slots")
    n = MDB.ensure(con, out["policy"], sorted(int(t) for t in out["teams"]), out["opponents"], out["per_cell"], seed=BATTLE_SEED, note="rebattle_strong65")
    print(f"ran {n} battles", flush=True)


def report():
    out = json.load(open(OUT))
    con = sqlite3.connect(f"file:{D.DB}?mode=ro", uri=True)
    rows = {}
    for t, v in out["teams"].items():
        after = counts(con, out["policy"], int(t), out["opponents"])
        w = sum(after[o][0] - v["before"][str(o)][0] for o in out["opponents"])
        b = sum(after[o][1] - v["before"][str(o)][1] for o in out["opponents"])
        v["fresh"] = [w, b]
        if b:
            rows.setdefault(v["group"], []).append((v["label"], w / b, b))
    summary = {}
    for g, r in rows.items():
        lab, fresh, b = map(np.array, zip(*r))
        se = float(np.std(fresh, ddof=1) / np.sqrt(len(fresh))) if len(fresh) > 1 else float("nan")
        summary[g] = dict(teams=len(r), fresh_battles=int(b.sum()), label=float(lab.mean()), fresh=float(fresh.mean()), fresh_se=se,
                          drop=float(lab.mean() - fresh.mean()), share_fresh_at_least_strong=float((fresh >= out["strong_threshold"]).mean()))
    s, c = summary.get("strong"), summary.get("random")
    if not s or not c:
        verdict = "incomplete"
    elif abs(c["drop"]) > 0.02:
        verdict = "measurement check failed: random teams moved more than 0.02 from their labels"
    elif s["drop"] >= 0.10:
        verdict = "mostly luck: strong teams lost 0.10 or more on fresh battles"
    elif s["drop"] <= 0.05:
        verdict = "real: strong teams held within 0.05 of their labels"
    else:
        verdict = "partly luck: strong teams lost between 0.05 and 0.10"
    out["result"] = dict(summary=summary, verdict=verdict)
    save(out)
    print(json.dumps(out["result"], indent=1))


if __name__ == "__main__":
    {"prepare": prepare, "battle": battle, "report": report}[sys.argv[1]]()
