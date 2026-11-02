"""Nucleus sampling on the all-teams 84-field model, and the fresh re-battle of set-field temperature 0.3's line-ups (code/docs/least-likely-trade-84-fields.md, section "Nucleus sampling and the colder-set re-battle").

    python nucleus84.py check | probes | gen | battle | report
    python nucleus84.py rebattle_plan           best team per line-up at the elite cut, set-field temperature 0.3, plays no battle
    python nucleus84.py rebattle                fresh games for that plan, private database
    python nucleus84.py rebattle_report

Sampling and battles only, training: none.
"""
import hashlib
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import trade84 as T
A, M, R, MDB = T.A, T.M, T.R, T.MDB
from team_diversity import slots_from_paste

ARMS = {
    "nucleus_sets_p09": "Set-field nucleus 0.9, species untouched, no remasking",
    "nucleus_sets_p07": "Set-field nucleus 0.7, species untouched, no remasking",
    "nucleus_sets_p05": "Set-field nucleus 0.5, species untouched, no remasking",
    "nucleus_all_p09": "All-field nucleus 0.9, no remasking",
}
TOP_P = {"nucleus_sets_p09": (1.0, 0.9), "nucleus_sets_p07": (1.0, 0.7), "nucleus_sets_p05": (1.0, 0.5), "nucleus_all_p09": (0.9, 0.9)}
ASKS, SEEDS, RES = T.ASKS, T.SEEDS, T.RES
OUT = RES / "least_likely_trade_nucleus.json"
PLAN = RES / "cold_sets_t03_rebattle_plan.json"
FRESH = RES / "cold_sets_t03_fresh.json"
FRESH_REPORT = RES / "cold_sets_t03_fresh_report.json"
COLD_ARM = "cold_sets_t03"
CUT, PER_FRESH, FRESH_SEED = T.CUT, T.PER_FRESH, 7540
FRESH_DB = Path.home() / "vgc-data/least_likely_trade/fresh_nucleus.sqlite"


def load(): return json.load(open(OUT)) if OUT.exists() else {"cells": {}}


def save(out):
    tmp = OUT.with_suffix(".tmp"); json.dump(out, open(tmp, "w")); os.replace(tmp, OUT)


def top_p_keep(probs, p):
    """A mask over the likeliest options, in order, until their odds reach p. The likeliest option always stays."""
    sp, idx = torch.sort(probs, descending=True, dim=-1)
    keep = (sp.cumsum(-1) - sp) < p
    return torch.zeros_like(probs, dtype=torch.bool).scatter(-1, idx, keep)


@torch.no_grad()
def sample_nucleus(w, n, ask, p_species, p_sets):
    """trade84.sample_cold at temperature 1, with nucleus filtering after guidance and the rule masks. p = 1 skips the filter, so it matches Model84.sample bit for bit."""
    a = torch.full((n,), M.NAN if ask is None else ask, device=M.DEV); an = torch.full((n,), M.NAN, device=M.DEV)
    x = torch.zeros(n, M.COLS, dtype=torch.long, device=M.DEV)
    for step, c in enumerate(M.ORDER):
        t = torch.full((n,), 1.0 - step / M.COLS, device=M.DEV)
        lc = w._column(w._logprobs(x, t, a), c)
        lc = A.GUIDANCE * lc + (1 - A.GUIDANCE) * w._column(w._logprobs(x, t, an), c)
        lc = lc.masked_fill(~w._legal(x, c), -float("inf"))
        p = p_species if c < M.STAT0 and c % 8 == 0 else p_sets
        if p < 1: lc = lc.masked_fill(~top_p_keep(torch.softmax(lc, -1), p), -float("inf"))
        v = torch.multinomial(torch.softmax(lc, -1), 1).squeeze(1)
        x[:, c] = v + 1 if c >= M.STAT0 else v
    return x


class Armed:
    def __init__(self, w, arm): self.w, self.arm = w, arm

    def sample(self, n, ask=None, guidance=None): return sample_nucleus(self.w, n, ask, *TOP_P[self.arm])

    def to_paste(self, row, repair=False): return self.w.to_paste(row, repair=repair)


class FixedLogits:
    """A stand-in model with known odds in every field. Odd fields rule out the likeliest option, so the filter must run after the rule masks."""
    ODDS = [.6, .25, .1, .04, .01]

    def _logprobs(self, x, t, a): return None

    def _column(self, lp, c): return torch.tensor(self.ODDS, device=M.DEV).log().expand(self.n, -1)

    def _legal(self, x, c):
        legal = torch.ones(self.n, len(self.ODDS), dtype=torch.bool, device=M.DEV)
        if c % 2: legal[:, 0] = False
        return legal


def support_test():
    """Runs sample_nucleus itself and checks which options each arm can draw, for species and set fields, with and without a rule mask."""
    w = FixedLogits(); w.n = 4000
    want = {1.0: ({0, 1, 2, 3, 4}, {1, 2, 3, 4}), .9: ({0, 1, 2}, {1, 2, 3}), .7: ({0, 1}, {1, 2}), .5: ({0}, {1})}
    for arm, (p_species, p_sets) in {**TOP_P, "no_filter": (1.0, 1.0)}.items():
        torch.manual_seed(1); x = sample_nucleus(w, w.n, 0.5, p_species, p_sets).cpu().numpy()
        for c in M.ORDER:
            species = c < M.STAT0 and c % 8 == 0
            drawn = set((x[:, c] - (1 if c >= M.STAT0 else 0)).tolist())
            if drawn != want[p_species if species else p_sets][c % 2]: raise SystemExit(f"NUCLEUS CHECK FAILED: {arm} field {c} drew {sorted(drawn)}")
    print("NUCLEUS_SUPPORT PASS", flush=True)


def check():
    probs = torch.tensor([[.5, .3, .15, .05, 0.], [.05, .05, .8, .1, 0.]])
    if top_p_keep(probs, .7).tolist() != [[True, True, False, False, False], [False, False, True, False, False]]: raise SystemExit("NUCLEUS CHECK FAILED: wrong options kept")
    if top_p_keep(probs, .01).sum(-1).tolist() != [1, 1]: raise SystemExit("NUCLEUS CHECK FAILED: the likeliest option must stay")
    support_test()
    w = A.load_model()
    torch.manual_seed(1); a = w.sample(16, ask=0.5, guidance=A.GUIDANCE)
    torch.manual_seed(1); b = sample_nucleus(w, 16, 0.5, 1.0, 1.0)
    if not torch.equal(a, b): raise SystemExit("NUCLEUS CHECK FAILED: p = 1 differs from Model84.sample")
    # Same species seed: filtering only the set fields must leave the species draw alone, since species fill first.
    torch.manual_seed(1); c = sample_nucleus(w, 16, 0.5, 1.0, 0.5)
    sp = [i for i in M.ORDER if i < M.STAT0 and i % 8 == 0]
    if not torch.equal(a[:, sp], c[:, sp]): raise SystemExit("NUCLEUS CHECK FAILED: set-field filter changed species")
    if torch.equal(a, c): raise SystemExit("NUCLEUS CHECK FAILED: the set-field filter changed nothing")
    print("NUCLEUS_CHECK PASS", flush=True)


def probes():
    w = A.load_model(); out = load(); d = np.load(A.DATA); train = d["all_X"][~d["all_val"]].astype(np.int64); pf = {}
    for arm in ARMS:
        torch.manual_seed(A.SEED0 + 1)
        rows = Armed(w, arm).sample(96, ask=8 / 24).cpu().numpy()
        pf[arm] = dict(round_trip=A.round_trip(w, rows[:20]), copies=A.copies(rows, train)[1])
        print(arm, json.dumps(pf[arm]), flush=True)
    out["preflight"] = pf; save(out)
    if not all(p["round_trip"].get("passed") for p in pf.values()): raise SystemExit("NUCLEUS_PROBES FAILED")
    print("NUCLEUS_PROBES PASS", flush=True)


def gen():
    out = load(); w = A.load_model()
    out.update(arms=ARMS, asks=ASKS, checkpoint_sha256=hashlib.sha256(A.checkpoint_path().read_bytes()).hexdigest())
    for arm in ARMS:
        for seed in SEEDS:
            for k in ASKS:
                tag = f"{arm}_a{k:02d}of24_s{seed}"
                if tag in out["cells"]: continue
                t0 = time.time()
                kept, stats, counts = A.draw(Armed(w, arm), k / 24, seed)
                out["cells"][tag] = dict(arm=arm, ask=k / 24, numerator=k, seed=seed, **counts, pastes=kept, stat_points=stats, seconds=time.time() - t0, measures=R.team_measures(kept, stats) if kept else None)
                save(out); m = out["cells"][tag]["measures"] or {}
                print(f"  {tag:30s} kept {len(kept):3d} legal {counts['legal']:6.1%} line-ups {m.get('lineups', 0):3d} 66 {m.get('spend_66', 0):.0%} {time.time() - t0:5.0f}s", flush=True)


def battle():
    A.require_shuffled(); con = MDB.connect(); out = load()
    for tag, c in out["cells"].items():
        if "y" in c or not c["pastes"]: continue
        sc = MDB.score_pastes(con, c["pastes"], per_cell=A.BATTLES_PER_OPPONENT, seed=A.SEED0 + c["seed"], origin=f"least_likely_trade_nucleus:{tag}")
        if not A.complete([s for s in sc if not s.get("illegal")]): print(f"  {tag} INCOMPLETE: left for the next run", flush=True); continue
        c["y"] = [s["score"] for s in sc]; save(out); print(f"  {tag:30s} win {np.nanmean([v for v in c['y'] if v is not None]):.3f}", flush=True)


def distinct_sets(pastes):
    """The curve analysis's definition: species, ability, item, nature and moves, ignoring Stat Points."""
    return len({json.dumps({f: s.get(f) for f in ("species", "ability", "item", "nature", "moves")}, sort_keys=True) for p in pastes for s in slots_from_paste(p)})


def report():
    out = load(); rows = {}; cold = json.load(open(T.OUT))["cells"]
    out["comparison"] = {COLD_ARM: dict(by_ask={k: dict(win=float(np.mean([np.mean([v for v in cold[f"{COLD_ARM}_a{k:02d}of24_s{s}"]["y"] if v is not None]) for s in SEEDS])), distinct_sets=distinct_sets([p for s in SEEDS for p in cold[f"{COLD_ARM}_a{k:02d}of24_s{s}"]["pastes"]])) for k in ASKS}, distinct_sets=distinct_sets([p for k in ASKS for s in SEEDS for p in cold[f"{COLD_ARM}_a{k:02d}of24_s{s}"]["pastes"]]))}
    print(COLD_ARM, json.dumps(out["comparison"][COLD_ARM]), flush=True)
    for arm in ARMS:
        rows[arm] = dict(distinct_sets=distinct_sets([p for k in ASKS for s in SEEDS for p in out["cells"][f"{arm}_a{k:02d}of24_s{s}"]["pastes"]]))
        for k in ASKS:
            cs = [out["cells"][f"{arm}_a{k:02d}of24_s{s}"] for s in SEEDS]
            lus = [tuple(sorted(s["species"] for s in slots_from_paste(p))) for c in cs for p in c["pastes"]]
            rows[arm][k] = dict(win=float(np.mean([np.mean([v for v in c["y"] if v is not None]) for c in cs])), by_seed=[float(np.mean([v for v in c["y"] if v is not None])) for c in cs], legal=float(np.mean([c["legal"] for c in cs])), line_ups=len(set(lus)), distinct_sets=distinct_sets([p for c in cs for p in c["pastes"]]), teams=len(lus))
        print(arm, json.dumps(rows[arm]), flush=True)
    out["summary"] = rows; save(out); print("NUCLEUS_REPORT_DONE", flush=True)


def rebattle_plan():
    """Best team per line-up at the cut, ties to lowest ask, seed and position, the rule the earlier plan used. Plays no battle."""
    sys.path.append(str(Path.home() / "vgc-data/line_up_deletion_84/run"))
    import line_up_yield as LY
    cells = json.load(open(T.OUT))["cells"]; train_set, _ = LY.references("gen84_all_ep21"); best, teams_per = {}, Counter()
    for k in ASKS:
        for s in SEEDS:
            c = cells[f"{COLD_ARM}_a{k:02d}of24_s{s}"]
            for pos, (p, y) in enumerate(zip(c["pastes"], c["y"])):
                l = tuple(sorted(x["species"] for x in slots_from_paste(p))); teams_per[l] += 1
                if y is None or y < CUT: continue
                key = (-y, k, s, pos)
                if l not in best or key < best[l][0]: best[l] = (key, p, y, k, s, pos)
    cand = [dict(line_up=list(l), paste=v[1], one_score=v[2], ask=v[3], seed=v[4], position=v[5], new_vs_training=l not in train_set, teams_on_line_up=teams_per[l]) for l, v in sorted(best.items(), key=lambda kv: kv[1][0])]
    json.dump(dict(cut=CUT, games_per_opponent=PER_FRESH, battles_per_team=PER_FRESH * 49, seed=FRESH_SEED, arm=COLD_ARM, line_ups=len(cand), new_vs_training=sum(c["new_vs_training"] for c in cand), battles=len(cand) * PER_FRESH * 49, candidates=cand), open(PLAN, "w"), indent=1)
    print(f"plan: {len(cand)} line-ups, {sum(c['new_vs_training'] for c in cand)} new against training, {len(cand) * PER_FRESH * 49} battles", flush=True)
    print("COLD_PLAN_DONE", flush=True)


def fresh_db():
    if not FRESH_DB.exists(): raise SystemExit("the queue must create the private fresh database first")
    return MDB.connect(FRESH_DB)


def rebattle():
    A.require_shuffled()
    plan = json.load(open(PLAN)); pastes = [c["paste"] for c in plan["candidates"]]; con = fresh_db()
    print(f"re-battle: {len(pastes)} teams, up to {len(pastes) * PER_FRESH * 49} battles", flush=True)
    scores = MDB.score_pastes(con, pastes, per_cell=PER_FRESH, seed=FRESH_SEED, origin="least_likely_trade:cold_sets_t03_rebattle")
    if not all(s["score"] is not None and s["coverage"] == s["columns"] and s["battles"] >= PER_FRESH * s["columns"] for s in scores):
        raise SystemExit("a team lacks some fresh battles")
    json.dump(dict(cut=CUT, per_cell=PER_FRESH, seed=FRESH_SEED, fresh=[dict(paste=p, fresh=s["score"], battles=s["battles"], team_id=s["team_id"]) for p, s in zip(pastes, scores)]), open(FRESH, "w"))
    print("COLD_REBATTLE_DONE", flush=True)


def rebattle_report():
    """Counts only. Freezes no list for ranks 51 to 100: the owner has not asked to spend that check on this list."""
    plan = json.load(open(PLAN)); fresh = {r["paste"]: r for r in json.load(open(FRESH))["fresh"]}
    scored = [dict(c, fresh=fresh[c["paste"]]["fresh"], battles=fresh[c["paste"]]["battles"], team_id=fresh[c["paste"]]["team_id"]) for c in plan["candidates"]]
    held = [c for c in scored if c["fresh"] >= CUT]
    no12 = [c for c in scored if c["ask"] != 12]; held12 = [c for c in held if c["ask"] != 12]
    row = dict(candidates=len(scored), held=len(held), held_new_vs_training=sum(c["new_vs_training"] for c in held), one_score_mean=float(np.mean([c["one_score"] for c in scored])), fresh_mean=float(np.mean([c["fresh"] for c in scored])), near_cut=sum(abs(c["fresh"] - CUT) < 0.036 for c in scored), candidates_without_12=len(no12), held_without_12=len(held12), held_new_vs_training_without_12=sum(c["new_vs_training"] for c in held12), battles=int(sum(c["battles"] for c in scored)))
    json.dump(dict(summary=row, teams=scored), open(FRESH_REPORT, "w"), indent=1)
    print(json.dumps(row), flush=True); print("COLD_REBATTLE_REPORT_DONE", flush=True)


if __name__ == "__main__":
    {"check": check, "probes": probes, "gen": gen, "battle": battle, "report": report, "rebattle_plan": rebattle_plan, "rebattle": rebattle, "rebattle_report": rebattle_report}[sys.argv[1]]()
