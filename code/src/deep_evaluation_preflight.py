"""Probes before the deep re-battle of elite line-ups (docs/weakness-pressure-deep-evaluation.md part A, issue #345).

    LY_PER_CELL=16 LY_MODEL=sched_species_first python line_up_yield.py preflight

Each probe battles in a private database under ~/vgc-data/deep_evaluation/ and saves a number and pass or fail under
`preflight` in the deep results file (line_up_yield_<model>_d16.json), which the re-battle keeps. No model runs, so the
generation probes are skipped with the reason written.
"""
import json
import math
import random
import sys
from pathlib import Path

import numpy as np

import heldout as HO
import line_up_yield as L
import matchup_db as MDB
import top50 as T50
from hps_generate import Validator

MODELS = ("sched_species_first", "sched_species_first_g4", "sched_species_first_g8")
CONTROL_TEAMS, Z = 5, 2.58                                 # repeat: two scores agree within 99% battle noise


def private(name):
    path = L.DEEP / f"preflight_d{L.DEPTH}_{name}.sqlite"
    for p in (path, Path(f"{path}-wal"), Path(f"{path}-shm")):
        p.unlink(missing_ok=True)
    return L.fresh_db(path)


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
                battles_per_meta_team=L.DEPTH, passed=bool(0.40 < np.mean(a) < 0.60 and np.mean(b) < 0.08))


class Spy:
    """The fast runner, recording the battle requests it builds: opponents, their order and each battle's slot orders."""

    def __init__(self):
        self.real = MDB.default_runner(); self.slot_order = self.real.slot_order; self.requests = []

    def __call__(self, con, jobs, seed=None):
        import fast_battle_runner as FBR
        chunk = getattr(self.real, "request_chunk_size", self.real.config["concurrency"])
        order = random.Random(seed) if self.slot_order == "shuffled" else None
        self.requests.append(json.dumps(FBR.build_requests(con, jobs, chunk, order), sort_keys=True, default=str))
        return self.real(con, jobs, seed=seed)


def repeat(paste):
    """One elite team scored twice at one seed, each time in a fresh database: same schedule and slot orders, scores within noise."""
    out = []
    for name in ("repeat_a", "repeat_b"):
        con = private(name); spy = Spy()
        val = Validator()
        try:
            tid = MDB.add_team(con, paste, "deep-preflight:repeat", val)[0]
        finally:
            val.close()
        MDB.ensure(con, MDB.get_policy(con), [tid], MDB.set_members(con, "top50"), L.DEPTH, seed=L.SEED, runner=spy, note="deep-preflight:repeat")
        s = MDB.score(con, MDB.get_policy(con), tid)
        out.append((spy.requests, s))
    (ra, a), (rb, b) = out
    gap, bound = abs(a["score"] - b["score"]), Z * math.sqrt(a["se"] ** 2 + b["se"] ** 2)
    return dict(score_a=a["score"], score_b=b["score"], battles=a["battles"], gap=gap, bound_99=bound,
                same_schedule_and_slot_orders=ra == rb and len(ra) > 0, passed=bool(ra == rb and len(ra) > 0 and gap <= bound),
                note="the seed reaches build_requests' slot-order generator and each worker's seed + job id; Showdown's battle random numbers stay uncontrolled")


def separation():
    """No held-out team (ranks 51-100) among the teams any guidance will battle; score_pastes would also refuse one."""
    found = {}
    for m in MODELS:
        pastes = L.teams_to_battle(L.grid_teams(m), L.DEPTH)
        found[m] = dict(teams=len(pastes), heldout=sum(HO.is_heldout_text(p) for p in pastes))
    return dict(**found, passed=all(v["heldout"] == 0 for v in found.values()))


def shallow_arm():
    """The 4-battle arm the flips compare against: every all-23-asks line-up re-battled, 4 per meta team, on seed 7520."""
    out = {}
    for m in MODELS:
        data = json.load(open(L.fresh_path(m, L.PER_CELL)))
        fresh = {r["paste"]: r for r in data["fresh"]}
        cand = L.candidates(L.grid_teams(m))
        rows = [fresh.get(t["paste"]) for t in cand.values()]
        out[m] = dict(line_ups=len(cand), missing=sum(r is None for r in rows), per_cell=data["per_cell"], seed=data["seed"],
                      battles=sorted({r["battles"] for r in rows if r}))
    return dict(**out, passed=all(v["missing"] == 0 and v["per_cell"] == 4 and v["battles"] == [196] for v in out.values()))


def independence():
    """The deep arm's database is not the 4-battle arm's and records its own depth."""
    con = L.fresh_db()
    L.claim_depth(con, L.DEPTH)
    pid = MDB.get_policy(con)
    cells = con.execute("SELECT COUNT(*), COALESCE(MAX(battles), 0) FROM cell WHERE policy_id=?", (pid,)).fetchone()
    return dict(db=str(L.DB), depth=L.DEPTH, seed=L.SEED, cells_already=cells[0], most_battles_in_a_cell=cells[1],
                passed=bool(L.DB.name != "fresh.sqlite" and cells[1] <= L.DEPTH))


def prediction():
    """Ceiling, read as what the result can still teach: the flip share battle noise alone predicts from the 4-battle scores."""
    out = {}
    for m in MODELS:
        fresh = {r["paste"]: r for r in json.load(open(L.fresh_path(m, L.PER_CELL)))["fresh"]}
        rows = [fresh[t["paste"]] for t in L.candidates(L.grid_teams(m)).values()]
        out[m] = L.predicted_flip_share([r["fresh"] for r in rows], [r["battles"] for r in rows], L.DEPTH * 49)
    return dict(**out, passed=True,
                note="passes either way: a prediction far above 10% says the verdict on the depth is nearly settled, and the run's use is the held counts at the deep depth")


def main():
    if L.DEPTH == L.PER_CELL:
        sys.exit("preflight is for another depth: set LY_PER_CELL, e.g. 16")
    if MDB.SLOT_ORDER != "shuffled":
        sys.exit(f"VGC_SLOT_ORDER={MDB.SLOT_ORDER}: these battles must shuffle slots")
    cand = L.candidates(L.grid_teams(L.MODEL))
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
