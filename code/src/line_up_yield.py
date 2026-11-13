"""Elite line-up yield per 1,000 teams for one model's 23-ask grid (issues #96 and #98).

Candidates: every line-up (six species) with a team at or above the elite cut, in each view of the grid (asks 1/24 to 17/24
without 12/24, the same with 12/24, all 23 asks). Each gets its best-scoring team within that view, ties to the lowest ask, seed
and position, so a view is judged only on teams it contains. Fresh battles: a private database, 4 games
against each of the 49 distinct top-50 meta teams, behaviour-cloning policy on both sides, slots shuffled. A line-up holds if
that team's fresh win rate is at least the cut.

    LY_MODEL=gen84_all_ep21 python line_up_yield.py rebattle | report
    python line_up_yield.py compare

LY_MODEL is the name of a grid's results file after "stat_points_84_", so a grid at another guidance is
LY_MODEL=sched_species_first_g4. LY_TABLE names the table whose training teams are the novelty reference (default: the
older table); LY_LABELLED=1 keeps only its labelled teams, as the labelled-teams trainings do. Ported on 2026-11-12 from
~/vgc-data/line_up_deletion_84/run/ for docs/masking-rate-rows.md; the line-up deletion arms still need that folder.
"""
import json
import math
import os
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, "/tmp/vgc-pilot/src")
sys.path.insert(0, str(HERE))
import asked_vs_got as X
import matchup_db as MDB
import team84 as T
import top50 as T50
from hps_generate import Validator


RES = Path.home() / "vgc-data" / "results"
DB = Path(os.environ.get("LY_DB", Path.home() / "vgc-data" / "line_up_deletion_84" / "fresh.sqlite"))
MODEL = os.environ.get("LY_MODEL", "gen84_all_ep21")
TABLE, LABELLED = os.environ.get("LY_TABLE", "lossdata84.npz"), bool(os.environ.get("LY_LABELLED"))
SPECIES_COLUMNS = [s * 8 for s in range(6)]
SMOKE = bool(os.environ.get("LY_SMOKE"))
CUT, PER_CELL, SEED, DRAW_SEED, DRAW_TEAMS, DRAWS, NEAR = 0.531, 4, 7520, 7521, 1000, 2000, 0.036
ASKS_NO12 = [k for k in range(1, 18) if k != 12]
ASKS_WITH12 = list(range(1, 18))
VIEWS = dict(asks_1_to_17_without_12=ASKS_NO12, asks_1_to_17_with_12=ASKS_WITH12, all_23_asks=list(range(1, 24)))
ARMS = ("dellu_none", "dellu_top1", "dellu_rand", "dellu_top50", "dellu_family")
WIN_MARGIN = 0.03


def norm(text):
    return re.sub(r"[^a-z0-9]", "", str(text).lower())


def line_up(paste):
    return tuple(sorted(norm(b.strip().split("\n")[0].split("@")[0]) for b in re.split(r"\n\s*\n", paste.strip())))


def grid_teams(model):
    cells = json.load(open(RES / f"stat_points_84_{model}.json"))["cells"].values()
    teams = []
    for c in sorted(cells, key=lambda c: (c["numerator"], c["seed"])):
        if "y" not in c:
            raise SystemExit(f"{model}: a cell has no battles")
        teams += [dict(ask=c["numerator"], seed=c["seed"], pos=i, paste=p, y=float(y), line_up=line_up(p))
                  for i, (p, y) in enumerate(zip(c["pastes"], c["y"]))]
    return teams


def best_team(teams):
    return max(teams, key=lambda t: (t["y"], -t["ask"], -t["seed"], -t["pos"]))


def candidates(teams):
    """{line-up: its best team} for line-ups above the cut within these teams."""
    elite = [t for t in teams if t["y"] >= CUT]
    return {lu: best_team([t for t in elite if t["line_up"] == lu]) for lu in {t["line_up"] for t in elite}}


def fresh_db():
    DB.parent.mkdir(parents=True, exist_ok=True)
    con = MDB.connect(DB)
    pid = MDB.get_policy(con, note="elite line-up yield re-battle, slots shuffled, 2026-11-02")
    if "+slots-shuffled" not in con.execute("SELECT row_policy FROM policy WHERE policy_id=?", (pid,)).fetchone()[0]:
        raise SystemExit("these battles must shuffle slots")
    val = Validator()
    try:
        with con:
            con.execute("DELETE FROM team_set WHERE set_name='top50'")
            for t in json.load(open(T50.TOP50)):              # a team listed twice counts twice
                tid = MDB.add_team(con, Path(T50.resolve(t["id"])).read_text(), "vgcpastes", val)[0]
                con.execute("INSERT INTO team_set VALUES ('top50', ?, 1) ON CONFLICT (set_name, team_id) "
                            "DO UPDATE SET weight = weight + 1", (tid,))
    finally:
        val.close()
    return con


def fresh_path(model):
    return RES / f"line_up_yield_{model}{'_smoke' if SMOKE else ''}.json"


def rebattle():
    if MDB.SLOT_ORDER != "shuffled":
        raise SystemExit(f"VGC_SLOT_ORDER={MDB.SLOT_ORDER}: these battles must shuffle slots")
    teams = grid_teams(MODEL)
    cand = candidates(teams)
    pastes = sorted({t["paste"] for asks in VIEWS.values() for t in candidates([t for t in teams if t["ask"] in asks]).values()})
    if SMOKE:
        pastes = pastes[:2]
    print(f"{MODEL}: {len(cand)} elite line-ups, {len(pastes)} teams, {len(pastes) * PER_CELL * 49} battles", flush=True)
    per_cell = 1 if SMOKE else PER_CELL
    scores = MDB.score_pastes(fresh_db(), pastes, per_cell=per_cell, seed=SEED, origin=f"line-up-yield:{MODEL}")
    if not all(s["score"] is not None and s["coverage"] == s["columns"] and s["battles"] >= per_cell * s["columns"] for s in scores):
        raise SystemExit(f"{MODEL}: a team lacks some fresh battles")
    json.dump(dict(model=MODEL, cut=CUT, per_cell=per_cell, seed=SEED,
                   fresh=[dict(paste=p, fresh=s["score"], battles=s["battles"]) for p, s in zip(pastes, scores)]),
              open(fresh_path(MODEL), "w"))
    print("LY_REBATTLE_DONE", flush=True)


def species_names(d, keys):
    """Token id to normalised species name, read through the model's own decoder."""
    V, names = T.regulation_vocab(), {}
    cols = SPECIES_COLUMNS
    for k in keys:
        rows = d[k]
        for i in np.unique(rows[:, cols]):
            if int(i) in names:
                continue
            at = np.argwhere(rows[:, cols] == i)[0]
            names[int(i)] = norm(T.decode(V, rows[at[0]])[at[1]]["species"])
    return names


def named(rows, names):
    return {tuple(sorted(names[int(i)] for i in r)) for r in np.unique(np.sort(np.asarray(rows)[:, SPECIES_COLUMNS], axis=1), axis=0)}


def references(model):
    """Line-ups of the training teams in TABLE, and of the loop and tournament records, which only the older table stores."""
    older = np.load(X.RES / "lossdata84.npz")
    d = {k: older[k] for k in ("all_X", "all_val", "all_W", "loop_X", "records_X")}
    if TABLE != "lossdata84.npz":
        table = np.load(X.RES / TABLE)
        d.update({k: table[k] for k in ("all_X", "all_val", "all_W")})
    names = species_names(d, ("all_X", "loop_X", "records_X"))
    keep = ~d["all_val"]
    if LABELLED:
        keep &= ~np.isnan(d["all_W"])
    if model.startswith("dellu_"):
        import line_up_deletion as LUD                      # only in ~/vgc-data/line_up_deletion_84/run/
        keep &= ~LUD.deleted(d, model[len("dellu_"):])
    return named(d["all_X"][keep], names), named(d["loop_X"], names) | named(d["records_X"], names)


def effective(counts):
    n = sum(counts.values())
    return 1 / sum((v / n) ** 2 for v in counts.values()) if n else 0.0


def draw_yield(pool, held, rng, draws=DRAWS, size=DRAW_TEAMS):
    """Mean count of held elite line-ups in a random `size` teams of `pool`, drawn before the cut is applied."""
    ids = {lu: i for i, lu in enumerate(sorted(held))}
    code = np.array([ids[t["line_up"]] if t["y"] >= CUT and t["line_up"] in ids else -1 for t in pool])
    size = min(size, len(pool))
    counts = [len(set(code[rng.choice(len(pool), size, replace=False)].tolist()) - {-1}) for _ in range(draws)]
    return float(np.mean(counts)), float(np.std(counts)), size


def view(teams, asks, fresh, train_set, common, rng):
    pool = [t for t in teams if t["ask"] in asks]
    cand = picks = candidates(pool)
    scored = {lu: fresh[t["paste"]] for lu, t in picks.items() if t["paste"] in fresh}
    held = {lu for lu, v in scored.items() if v >= CUT}
    mean, sd, size = draw_yield(pool, held, rng)
    bars = [round(b / 50, 2) for b in range(0, 41)]
    return dict(teams=len(pool), mean_win=float(np.mean([t["y"] for t in pool])),
                line_ups=len({t["line_up"] for t in pool}), effective_line_ups=effective(Counter(t["line_up"] for t in pool)),
                elite_teams=sum(t["y"] >= CUT for t in pool), elite_line_ups=len(cand),
                elite_effective_line_ups=effective(Counter(t["line_up"] for t in pool if t["y"] >= CUT)),
                rebattled=len(scored), held=len(held), near_cut=sum(abs(v - CUT) < NEAR for v in scored.values()),
                one_score_mean=float(np.mean([picks[lu]["y"] for lu in scored])) if scored else None,
                fresh_mean=float(np.mean(list(scored.values()))) if scored else None,
                yield_per_1000=mean, yield_draw_sd=sd, draw_teams=size,
                held_new_against_training=len(held - train_set), held_new_against_common=len(held - common),
                elite_new_against_training=len(set(cand) - train_set), elite_new_against_common=len(set(cand) - common),
                curve=[[b, len({t["line_up"] for t in pool if t["y"] >= b})] for b in bars],
                held_line_ups=[list(lu) for lu in sorted(held)])


def report_model(model):
    teams = grid_teams(model)
    data = json.load(open(fresh_path(model)))
    fresh = {r["paste"]: r["fresh"] for r in data["fresh"]}
    train_set, common = references(model)
    rng = np.random.default_rng(DRAW_SEED)
    data["views"] = {name: view(teams, asks, fresh, train_set, common, rng) for name, asks in VIEWS.items()}
    json.dump(data, open(fresh_path(model), "w"))
    for name, v in data["views"].items():
        print(f"{model:16s} {name:26s} win {v['mean_win']:.3f}  line-ups {v['line_ups']:4d}  elite {v['elite_line_ups']:3d}  held {v['held']:3d}"
              f"  near cut {v['near_cut']:3d}  yield/1000 {v['yield_per_1000']:.2f}  new held {v['held_new_against_training']}/{v['held_new_against_common']}", flush=True)
    return data


def compare(boot=400, draws=20):
    """Each arm against the arm with no teams deleted; intervals resample whole teams within each ask."""
    rng = np.random.default_rng(DRAW_SEED + 1)
    runs = {}
    for arm in ARMS:
        teams = [t for t in grid_teams(arm) if t["ask"] in ASKS_NO12]
        data = json.load(open(fresh_path(arm)))
        fresh = {r["paste"]: r["fresh"] for r in data["fresh"]}
        cand = candidates(teams)
        held = {lu for lu, t in cand.items() if fresh[t["paste"]] >= CUT}
        by_ask = {k: [t for t in teams if t["ask"] == k] for k in ASKS_NO12}
        runs[arm] = (by_ask, held, draw_yield(teams, held, rng)[0], float(np.mean([t["y"] for t in teams])))
    samples = {}
    for arm, (by_ask, held, _, _) in runs.items():
        wins, yields = [], []
        for _ in range(boot):
            pool = [g[i] for g in by_ask.values() for i in rng.integers(0, len(g), len(g))]
            wins.append(np.mean([t["y"] for t in pool])); yields.append(draw_yield(pool, held, rng, draws=draws)[0])
        samples[arm] = (np.array(wins), np.array(yields))
    out = {}
    base = "dellu_none"
    for arm in ARMS:
        win_diff, yield_diff = samples[arm][0] - samples[base][0], samples[arm][1] - samples[base][1]
        out[arm] = dict(win=runs[arm][3], yield_per_1000=runs[arm][2], win_diff=runs[arm][3] - runs[base][3],
                        win_diff_interval=[float(x) for x in np.percentile(win_diff, [2.5, 97.5])],
                        yield_diff=runs[arm][2] - runs[base][2],
                        yield_diff_interval=[float(x) for x in np.percentile(yield_diff, [2.5, 97.5])])
        o = out[arm]
        o["passes"] = arm != base and o["yield_diff_interval"][0] > 0 and o["win_diff_interval"][0] > -WIN_MARGIN
        print(f"{arm:14s} win {o['win']:.3f} ({o['win_diff']:+.3f} {o['win_diff_interval'][0]:+.3f} {o['win_diff_interval'][1]:+.3f})"
              f"  yield {o['yield_per_1000']:.2f} ({o['yield_diff']:+.2f} {o['yield_diff_interval'][0]:+.2f} {o['yield_diff_interval'][1]:+.2f})"
              f"  passes {o['passes']}", flush=True)
    json.dump(out, open(RES / "line_up_deletion_compare.json", "w"), indent=1)
    assert math.isfinite(out[base]["win"])


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else ""
    if what == "rebattle":
        rebattle()
    elif what == "report":
        report_model(MODEL)
        print("LY_REPORT_DONE", flush=True)
    elif what == "compare":
        compare()
    else:
        sys.exit(__doc__)
