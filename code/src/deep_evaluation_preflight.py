"""Probes before the deep re-battle of elite line-ups (docs/weakness-pressure-deep-evaluation.md part A, issue #345).

    LY_PER_CELL=16 LY_MODEL=sched_species_first python line_up_yield.py preflight

Each probe battles in a private database under ~/vgc-data/deep_evaluation/ and saves a number and pass or fail under
`preflight` in the deep results file (line_up_yield_<model>_d16.json), which the re-battle keeps. No model runs, so the
generation probes are skipped with the reason written.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

import heldout as HO
import line_up_yield as L
import matchup_db as MDB
import top50 as T50
from hps_generate import Validator

CONTROL_TEAMS, Z = 5, 2.58                                 # repeat: two scores agree within 99% battle noise
TOP50_BAND, UNIFORM_BAND = (0.40, 0.60), (0.005, 0.05)     # "near 0.50" and "near 0.015" (pre-registration, part A)


def private(name):
    path = L.DEEP / f"preflight_d{L.DEPTH}_{name}.sqlite"
    for p in (path, Path(f"{path}-wal"), Path(f"{path}-shm")):
        p.unlink(missing_ok=True)
    return L.fresh_db(path)


def inside(x, band):
    return bool(band[0] < x < band[1])


def controls():
    """Known teams through the real scorer at the deep depth: top-50 teams near 0.50, uniform-legal teams near 0.015."""
    con = private("controls")
    meta = [Path(T50.resolve(t["id"])).read_text() for t in json.load(open(T50.TOP50))[:CONTROL_TEAMS]]
    uniform = [json.loads(line)["team"] for _, line in zip(range(CONTROL_TEAMS), open(Path.home() / "vgc-data/teams/hps_reg_mb_100k.jsonl"))]
    y = lambda ps, o: [s["score"] for s in MDB.score_pastes(con, ps, per_cell=L.DEPTH, seed=L.SEED, origin=o)]
    a, b = y(meta, "deep-preflight:top50"), y(uniform, "deep-preflight:uniform")
    if None in a or None in b:
        return dict(passed=False, note="a control team got no score", top50=a, uniform_legal=b)
    return dict(top50_win=float(np.mean(a)), uniform_legal_win=float(np.mean(b)), top50=a, uniform_legal=b, teams_each=CONTROL_TEAMS,
                battles_per_meta_team=L.DEPTH, top50_band=TOP50_BAND, uniform_legal_band=UNIFORM_BAND,
                passed=inside(np.mean(a), TOP50_BAND) and inside(np.mean(b), UNIFORM_BAND))


class Spy:
    """The fast runner, unchanged, plus the battle requests it logged for each call: opponents, order, slot orders, seeds."""

    def __init__(self):
        self.real = MDB.default_runner(); self.slot_order = self.real.slot_order; self.requests = []

    def __call__(self, con, jobs, seed=None):
        log = Path(self.real.folder) / "requests.jsonl"
        before = log.read_text().splitlines() if log.exists() else []
        out = self.real(con, jobs, seed=seed)
        self.requests += [json.loads(line)["jobs"] for line in log.read_text().splitlines()[len(before):]]
        return out


def repeat(paste):
    """One elite team scored twice at one seed, each time in a fresh database: same logged requests, scores within noise."""
    out = []
    for name in ("repeat_a", "repeat_b"):
        con = private(name); spy = Spy()
        val = Validator()
        try:
            tid = MDB.add_team(con, paste, "deep-preflight:repeat", val)[0]
        finally:
            val.close()
        MDB.ensure(con, MDB.get_policy(con), [tid], MDB.set_members(con, "top50"), L.DEPTH, seed=L.SEED, runner=spy, note="deep-preflight:repeat")
        out.append((spy.requests, MDB.score(con, MDB.get_policy(con), tid)))
    (ra, a), (rb, b) = out
    same = ra == rb and len(ra) > 0
    gap, bound = abs(a["score"] - b["score"]), Z * math.sqrt(a["se"] ** 2 + b["se"] ** 2)
    return dict(score_a=a["score"], score_b=b["score"], battles=a["battles"], gap=gap, bound_99=bound,
                same_schedule_and_slot_orders=same, passed=bool(same and gap <= bound),
                note="compares the requests the fast runner logged; the seed reaches its slot-order generator and each worker's "
                     "seed + job id; Showdown's battle random numbers stay uncontrolled")


def separation():
    """No held-out team (ranks 51-100) among the teams any guidance will battle; score_pastes would also refuse one."""
    found = {}
    for m in L.DEEP_GRIDS:
        pastes = L.teams_to_battle(L.grid_teams(m), L.DEPTH)
        found[m] = dict(teams=len(pastes), heldout=sum(HO.is_heldout_text(p) for p in pastes))
    return dict(**found, passed=all(v["heldout"] == 0 for v in found.values()))


def shallow_arm():
    """The 4-battle re-battle the flips compare against: every all-23-asks line-up, 4 per meta team, seed 7520."""
    out, ok = {}, True
    for m in L.DEEP_GRIDS:
        try:
            out[m] = dict(line_ups=len(L.shallow_rows(m)), per_cell=L.PER_CELL, seed=L.SHALLOW_SEED)
        except SystemExit as e:
            out[m], ok = dict(error=str(e)), False
    return dict(**out, passed=ok)


def independence():
    """Deep evaluation battles in a database of its own: not the 4-battle one, recording depth 16, no cell above 16."""
    con = L.fresh_db()
    L.claim_depth(con, L.DEPTH)
    recorded = con.execute("SELECT value FROM meta WHERE key='line_up_yield_per_cell'").fetchone()[0]
    most = con.execute("SELECT COALESCE(MAX(battles), 0) FROM cell WHERE policy_id=?", (MDB.get_policy(con),)).fetchone()[0]
    own = L.DB.resolve() != L.SHALLOW_DB.resolve()
    return dict(db=str(L.DB), shallow_db=str(L.SHALLOW_DB), recorded_depth=int(recorded), seed=L.SEED, shallow_seed=L.SHALLOW_SEED,
                most_battles_in_a_cell=most, passed=bool(own and int(recorded) == L.DEPTH and most <= L.DEPTH and L.SEED != L.SHALLOW_SEED))


def prediction():
    """Ceiling, read as what the result can still teach: the flip share battle noise alone predicts from the 4-battle scores."""
    out, k = {}, L.effective_columns()
    for m in L.DEEP_GRIDS:
        rows = list(L.shallow_rows(m).values())
        out[m] = L.predicted_flip_share([r["fresh"] for r in rows], np.full(len(rows), L.PER_CELL * k), L.DEPTH * k)
    return dict(**out, effective_columns=k, passed=True,
                note="passes either way: a prediction far above 10% says the verdict on the depth is nearly settled, and the run's use is the held counts at 16")


def main():
    if L.DEPTH == L.PER_CELL:
        sys.exit(f"preflight is for deep evaluation: set LY_PER_CELL={L.DEEP_DEPTH}")
    if MDB.SLOT_ORDER != "shuffled":
        sys.exit(f"VGC_SLOT_ORDER={MDB.SLOT_ORDER}: these battles must shuffle slots")
    cand = L.cohort(L.MODEL)
    first = cand[sorted(cand)[0]]["paste"]
    pf = dict(shallow_arm=shallow_arm(), separation=separation(), independence=independence(), prediction=prediction(),
              controls=controls(), repeat=repeat(first),
              invariance=dict(skipped="every battle shuffles slots, the fast runner draws each battle's order, and rebattle refuses any other order"),
              **{k: dict(skipped="no model runs: the teams are stored grid teams") for k in
                 ("round_trip", "coverage", "read_teams", "training_smoke", "condition", "copies")})
    path = L.fresh_path(L.MODEL)
    data = L.stored(path); data["preflight"] = pf
    json.dump(data, open(path, "w"))
    print(json.dumps({k: {x: y for x, y in v.items() if x not in ("top50", "uniform_legal")} for k, v in pf.items()}, indent=1))
    print("LY_PREFLIGHT_DONE", "PASS" if all(v.get("passed", True) for v in pf.values()) else "FAIL", flush=True)
