"""Controls and shares for the least-likely trade on the all-teams 84-field model (issue on the remasking map, code/docs/least-likely-trade-84-fields.md).

    python trade84.py check | probes | gen | battle | report
    python trade84.py rebattle                  fresh games for line-ups at the elite cut, least-likely 10% and no remasking
    python trade84.py rebattle_report           freezes the final team lists
    python trade84.py heldout                   ranks 51 to 100, once per final team list
"""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import remask84 as R
A = R.A
M = R.M
MDB = R.MDB
from team_diversity import slots_from_paste

ARMS = {
    "remask_ll_05": "Least-likely 5% remasked, 84 passes",
    "remask_ll_20": "Least-likely 20% remasked, 84 passes",
    "cold_all_t08": "Temperature 0.8 on all fields, no remasking",
    "cold_all_t06": "Temperature 0.6 on all fields, no remasking",
    "cold_sets_t06": "Set-field temperature 0.6, species at 1, no remasking",
    "cold_sets_t03": "Set-field temperature 0.3, species at 1, no remasking",
    "spend66": "Stat Points raised to all 66, same no-remasking teams",
}
SHARE = {"remask_ll_05": 0.05, "remask_ll_20": 0.20}
TEMPS = {"cold_all_t08": (0.8, 0.8), "cold_all_t06": (0.6, 0.6), "cold_sets_t06": (1.0, 0.6), "cold_sets_t03": (1.0, 0.3)}
ASKS, SEEDS = R.ASK_NUMERATORS, R.SEEDS
RES = A.RES
OUT = RES / "least_likely_trade.json"
PLAN = RES / "remask84_rebattle_plan.json"
FRESH = RES / "least_likely_trade_fresh.json"
LISTS = RES / "least_likely_trade_final_lists.json"
HELDOUT = RES / "least_likely_trade_heldout.json"
FRESH_DB = Path.home() / "vgc-data/least_likely_trade/fresh.sqlite"
EARLIER = RES / "line_up_yield_gen84_all_ep21.json"
CUT, PER_FRESH, FRESH_SEED, HELD_PER_CELL, HELD_SEED = 0.531, 4, 7520, 4, 7530
CONTROL = RES / "stat_points_84_gen84_all_ep21.json"
STATS = ["HP", "Atk", "Def", "SpA", "SpD", "Spe"]


def load(): return json.load(open(OUT)) if OUT.exists() else {"cells": {}}


def save(out):
    tmp = OUT.with_suffix(".tmp"); json.dump(out, open(tmp, "w")); os.replace(tmp, OUT)


@torch.no_grad()
def sample_cold(w, n, ask, t_species, t_sets):
    """Model84.sample with one temperature for species fields and another for every other field. Equal temperatures match Model84.sample bit for bit."""
    a = torch.full((n,), M.NAN if ask is None else ask, device=M.DEV); an = torch.full((n,), M.NAN, device=M.DEV)
    x = torch.zeros(n, M.COLS, dtype=torch.long, device=M.DEV)
    for step, c in enumerate(M.ORDER):
        t = torch.full((n,), 1.0 - step / M.COLS, device=M.DEV)
        lc = w._column(w._logprobs(x, t, a), c)
        lc = A.GUIDANCE * lc + (1 - A.GUIDANCE) * w._column(w._logprobs(x, t, an), c)
        temp = t_species if c < M.STAT0 and c % 8 == 0 else t_sets
        lc = lc.masked_fill(~w._legal(x, c), -float("inf")) / temp
        v = torch.multinomial(torch.softmax(lc, -1), 1).squeeze(1)
        x[:, c] = v + 1 if c >= M.STAT0 else v
    return x


class Armed:
    def __init__(self, w, arm): self.w, self.arm, self.stats = w, arm, {}

    def sample(self, n, ask=None, guidance=None):
        if self.arm in SHARE:
            R.LEAST_LIKELY = SHARE[self.arm]
            return R.sample(self.w, n, ask, "remask_ll_10", self.stats)
        return sample_cold(self.w, n, ask, *TEMPS[self.arm])

    def to_paste(self, row, repair=False): return self.w.to_paste(row, repair=repair)


def check():
    w = A.load_model()
    torch.manual_seed(1); a = w.sample(16, ask=0.5, guidance=A.GUIDANCE)
    torch.manual_seed(1); b = sample_cold(w, 16, 0.5, 1.0, 1.0)
    if not torch.equal(a, b): raise SystemExit("TRADE84 CHECK FAILED: sample_cold at temperature 1 differs from Model84.sample")
    print("TRADE84_CHECK PASS", flush=True)


def probes():
    w = A.load_model(); out = load(); d = np.load(A.DATA); train = d["all_X"][~d["all_val"]].astype(np.int64); pf = {}
    for arm in ARMS:
        if arm == "spend66": continue
        torch.manual_seed(A.SEED0 + 1)
        rows = Armed(w, arm).sample(96, ask=8 / 24).cpu().numpy()
        pf[arm] = dict(round_trip=A.round_trip(w, rows[:20]), copies=A.copies(rows, train)[1])
        print(arm, json.dumps(pf[arm]), flush=True)
    out["preflight"] = pf; save(out)
    if not all(p["round_trip"].get("passed") for p in pf.values()): raise SystemExit("TRADE84_PROBES FAILED")
    print("TRADE84_PROBES PASS", flush=True)


def raise_to_66(evs):
    """Share the unspent points in proportion to what the Pokemon already spends, largest remainder first, capped at 32 per stat. A hand rule, not a model draw."""
    spent = sum(evs.values())
    if spent == 66: return dict(evs)
    base = {s: evs[s] if spent else 1 for s in STATS}
    weight = sum(base.values()); left = 66 - spent; new = dict(evs)
    room = lambda s: 32 - new[s]
    while left > 0:
        share = {s: left * base[s] / weight if room(s) > 0 else 0.0 for s in STATS}
        if not any(share.values()):
            base = {s: 1 for s in STATS}; weight = 6; continue
        add = {s: min(int(share[s]), room(s)) for s in STATS}
        if not any(add.values()):
            s = max((s for s in STATS if room(s) > 0), key=lambda s: share[s]); add[s] = 1
        for s, v in add.items(): new[s] += v; left -= v
    return new


def ev_line(evs): return "EVs: " + " / ".join(f"{evs[s]} {s}" for s in STATS if evs[s])


def spend66_cell(cell):
    """The saved paste, with only its EVs line changed. Stat Alignment stays as the repair left it."""
    pastes, stats = [], []
    for p, team in zip(cell["pastes"], cell["stat_points"]):
        blocks, row = p.strip().split("\n\n"), []
        if len(blocks) != 6: raise SystemExit("a saved paste lacks six Pokemon")
        out_blocks = []
        for b, (species, evs) in zip(blocks, team):
            new = raise_to_66(evs); lines = b.split("\n"); seen = False
            for i, line in enumerate(lines):
                if line.startswith("EVs:"): lines[i] = ev_line(new); seen = True
            if not seen: lines.insert(next(i for i, l in enumerate(lines) if l.startswith("Level:")) + 1, ev_line(new))
            out_blocks.append("\n".join(lines)); row.append([species, new])
        pastes.append("\n\n".join(out_blocks) + "\n"); stats.append(row)
    return pastes, stats


def gen():
    out = load(); w = A.load_model(); ctl = json.load(open(CONTROL))["cells"]
    out.update(arms=ARMS, asks=ASKS, checkpoint_sha256=hashlib.sha256(A.checkpoint_path().read_bytes()).hexdigest())
    for arm in ARMS:
        for seed in SEEDS:
            for k in ASKS:
                tag = f"{arm}_a{k:02d}of24_s{seed}"
                if tag in out["cells"]: continue
                t0 = time.time()
                if arm == "spend66":
                    kept, stats = spend66_cell(ctl[f"gen84_all_ep21_a{k:02d}of24_s{seed}"])
                    val = A.Validator()
                    try: bad = sum(val(p) is not None for p in kept)
                    finally: val.close()
                    counts = dict(tried=len(kept), rejected=bad, heldout_dropped=0, legal=1 - bad / len(kept), reasons=[])
                else:
                    armed = Armed(w, arm); kept, stats, counts = A.draw(armed, k / 24, seed)
                out["cells"][tag] = dict(arm=arm, ask=k / 24, numerator=k, seed=seed, **counts, pastes=kept, stat_points=stats, seconds=time.time() - t0, measures=R.team_measures(kept, stats) if kept else None)
                save(out); m = out["cells"][tag]["measures"] or {}
                print(f"  {tag:30s} kept {len(kept):3d} legal {counts['legal']:6.1%} line-ups {m.get('lineups', 0):3d} 66 {m.get('spend_66', 0):.0%} {time.time() - t0:5.0f}s", flush=True)


def battle():
    A.require_shuffled(); con = MDB.connect(); out = load()
    for tag, c in out["cells"].items():
        if "y" in c or not c["pastes"]: continue
        sc = MDB.score_pastes(con, c["pastes"], per_cell=A.BATTLES_PER_OPPONENT, seed=A.SEED0 + c["seed"], origin=f"least_likely_trade:{tag}")
        if not A.complete([s for s in sc if not s.get("illegal")]): print(f"  {tag} INCOMPLETE: left for the next run", flush=True); continue
        c["y"] = [s["score"] for s in sc]; save(out); print(f"  {tag:30s} win {np.nanmean([v for v in c['y'] if v is not None]):.3f}", flush=True)


def report():
    out = load(); rows = {}
    for arm in ARMS:
        rows[arm] = {}
        for k in ASKS:
            cs = [out["cells"][f"{arm}_a{k:02d}of24_s{s}"] for s in SEEDS]
            lus = [tuple(sorted(s["species"] for s in slots_from_paste(p))) for c in cs for p in c["pastes"]]
            rows[arm][k] = dict(win=float(np.mean([np.mean([v for v in c["y"] if v is not None]) for c in cs])), by_seed=[float(np.mean([v for v in c["y"] if v is not None])) for c in cs], legal=float(np.mean([c["legal"] for c in cs])), line_ups=len(set(lus)), teams=len(lus))
        print(arm, json.dumps(rows[arm]), flush=True)
    out["summary"] = rows; save(out); print("TRADE84_REPORT_DONE", flush=True)


def fresh_db():
    """A private database the queue copies from the earlier re-battle's, so its fresh games are reused and nothing else writes here."""
    if not FRESH_DB.exists(): raise SystemExit("the queue must copy the fresh database first")
    return MDB.connect(FRESH_DB)


def rebattle():
    A.require_shuffled()
    plan = json.load(open(PLAN))
    pastes = sorted({c["paste"] for a in plan["arms"].values() for c in a["candidates"]})
    con = fresh_db()
    print(f"re-battle: {len(pastes)} teams, up to {len(pastes) * PER_FRESH * 49} battles before reuse", flush=True)
    scores = MDB.score_pastes(con, pastes, per_cell=PER_FRESH, seed=FRESH_SEED, origin="least_likely_trade:rebattle")
    if not all(s["score"] is not None and s["coverage"] == s["columns"] and s["battles"] >= PER_FRESH * s["columns"] for s in scores):
        raise SystemExit("a team lacks some fresh battles")
    json.dump(dict(cut=CUT, per_cell=PER_FRESH, seed=FRESH_SEED, fresh=[dict(paste=p, fresh=s["score"], battles=s["battles"], team_id=s["team_id"]) for p, s in zip(pastes, scores)]), open(FRESH, "w"))
    print("TRADE84_REBATTLE_DONE", flush=True)


def rebattle_report():
    """Freezes one final team list per rule: the best team of each line-up holding the cut on fresh games."""
    plan = json.load(open(PLAN)); fresh = {r["paste"]: r for r in json.load(open(FRESH))["fresh"]}; lists, rows = {}, {}
    for a, v in plan["arms"].items():
        scored = [dict(c, fresh=fresh[c["paste"]]["fresh"], team_id=fresh[c["paste"]]["team_id"]) for c in v["candidates"]]
        held = [c for c in scored if c["fresh"] >= CUT]
        no12 = [c for c in scored if c["ask"] != 12]; held12 = [c for c in held if c["ask"] != 12]
        rows[a] = dict(candidates=len(scored), held=len(held), held_new_vs_training=sum(c["new_vs_training"] for c in held), one_score_mean=float(np.mean([c["one_score"] for c in scored])), fresh_mean=float(np.mean([c["fresh"] for c in scored])),
                       near_cut=sum(abs(c["fresh"] - CUT) < 0.036 for c in scored), candidates_without_12=len(no12), held_without_12=len(held12), held_new_vs_training_without_12=sum(c["new_vs_training"] for c in held12))
        lists[a] = [dict(team_id=c["team_id"], paste=c["paste"], line_up=c["line_up"], fresh=c["fresh"], ask=c["ask"], new_vs_training=c["new_vs_training"]) for c in held]
        print(a, json.dumps(rows[a]), flush=True)
    if LISTS.exists(): raise SystemExit("final team lists already frozen")
    json.dump(dict(summary=rows, lists=lists), open(LISTS, "w"), indent=1)
    print("TRADE84_LISTS_FROZEN", flush=True)


def heldout_prep():
    """seed_heldout adds ranks 51 to 100 without validating them, and the battle runner refuses a team with no legality flag. Plays no battle and reads no score."""
    from hps_generate import Validator
    con = fresh_db(); MDB.seed_heldout(con); val = Validator(); bad = 0
    try:
        rows = con.execute("SELECT team_id, paste FROM team t JOIN team_set s USING (team_id) WHERE set_name=? AND legal IS NULL", (MDB.H.SET_NAME,)).fetchall()
        with con:
            for tid, paste in rows:
                err = val(paste); bad += err is not None
                con.execute("UPDATE team SET legal=?, legal_error=? WHERE team_id=?", (int(err is None), err, tid))
    finally:
        val.close()
    print(f"validated {len(rows)} teams, {bad} illegal", flush=True)
    if bad: raise SystemExit("a rank 51 to 100 team fails Showdown")
    print("TRADE84_HELDOUT_PREP_DONE", flush=True)


def heldout():
    """Ranks 51 to 100, once per frozen list, in the private database."""
    if HELDOUT.exists(): raise SystemExit("this final team list has already had its ranks 51 to 100 check")
    A.require_shuffled()
    lists = json.load(open(LISTS))["lists"]; con = fresh_db(); pid = MDB.get_policy(con); res = {}
    for a, teams in lists.items():
        if not teams: res[a] = dict(teams=0); continue
        ids = [t["team_id"] for t in teams]
        sc = MDB.evaluate_final(con, pid, ids, per_cell=HELD_PER_CELL, seed=HELD_SEED, note=f"least_likely_trade {a}")
        ys = np.array([s["score"] for s in sc], float)
        boot = np.random.default_rng(92).choice(ys, (4000, len(ys))).mean(1)
        res[a] = dict(teams=len(ys), mean=float(ys.mean()), interval=[float(np.quantile(boot, .025)), float(np.quantile(boot, .975))], top50_fresh_mean=float(np.mean([t["fresh"] for t in teams])), battles=int(sum(s["battles"] for s in sc)), at_or_above_cut=int((ys >= CUT).sum()), scores=[float(v) for v in ys])
        print(a, json.dumps({k: v for k, v in res[a].items() if k != "scores"}), flush=True)
    json.dump(res, open(HELDOUT, "w"), indent=1)
    print("TRADE84_HELDOUT_DONE", flush=True)


if __name__ == "__main__":
    what = sys.argv[1]
    {"check": check, "probes": probes, "gen": gen, "battle": battle, "report": report, "rebattle": rebattle, "rebattle_report": rebattle_report, "heldout_prep": heldout_prep, "heldout": heldout}[what]()
