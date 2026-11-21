"""Remasking on the all-teams 84-field model (docs/remask-84-fields.md): the least-likely rule, the paper's rules, and random decoding.

The control, "No remasking", is the existing grid in results/stat_points_84_gen84_all_ep21.json, same asks, seeds and scorer.
Every arm keeps species-first order, guidance 2, temperature 1, legal Pokémon construction filter and the Stat Alignment repair.

    python remask84.py check | probes | gen | battle | report      species-first arms
    python remask84.py random_decoding                        the sampling-only legality probe (REMASK84_SMOKE=1: one batch)
    python remask84.py la_probes | la_gen | la_battle | la_report      random decoding with the lookahead, asks 1/24 to 23/24 (issue #92)
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
import ask_grid84 as A                                   # also sets up sys.path for the modules below
import matchup_db as MDB
import sampler84 as S
from team_diversity import slots_from_paste

ARMS = {
    "remask_ll_10": "Least-likely 10% remasked, 84 passes",
    "remask_conf_loop": "Confidence-weighted 0.3 fields remasked, 210 passes",
    "remask_even_loop": "Even-odds 0.5 fields remasked, 210 passes",
}
ASK_NUMERATORS, SEEDS, KEEP = [4, 8, 12, 16], [1, 2, 3], A.KEEP
LOOP_START, LOOP_PASSES, SIGMA_CONF, ETA_CAP, LEAST_LIKELY = S.LOOP_START, S.LOOP_PASSES, S.SIGMA_CONF, S.ETA_CAP, S.LEAST_LIKELY
OUT = A.RES / "remask84.json"
CONTROL = A.RES / "stat_points_84_gen84_all_ep21.json"

sample = S.sample_remask
sample_random = S.sample_random


def load(): return json.load(open(OUT)) if OUT.exists() else {"cells": {}}


def save(out):
    tmp = OUT.with_suffix(".tmp"); json.dump(out, open(tmp, "w")); os.replace(tmp, OUT)


def species_of(paste): return sorted(s["species"] for s in slots_from_paste(paste))


def team_measures(pastes, stat_points):
    """Diversity over the whole group of teams: distinct line-ups, effective species, top species share, and the Stat Points check."""
    lineups = {tuple(species_of(p)) for p in pastes}
    counts = np.array(list(Counter(s for p in pastes for s in species_of(p)).values()), float)
    share = counts / counts.sum()
    totals = [sum(e) if isinstance(e, (list, tuple)) else sum(e.values()) for team in stat_points for _, e in team]
    return dict(teams=len(pastes), lineups=len(lineups), effective_species=float(np.exp(-(share * np.log(share)).sum())),
                top_species_share=float(share.max()), spend_66=float(np.mean([x == 66 for x in totals])), mean_stat_points=float(np.mean(totals)))


def check():
    """Passing no arm must reproduce the deployed sampler bit for bit."""
    w = A.load_model()
    torch.manual_seed(1); a = w.sample(16, ask=0.5, guidance=A.GUIDANCE)
    torch.manual_seed(1); b = sample(w, 16, 0.5, None)
    assert bool((a == b).all()), "plain path differs from Model84.sample"
    print("REMASK84_CHECK PASS: no-remasking path is bit-identical to Model84.sample", flush=True)


def probes():
    """Round trip and copies for each arm; a failed round trip stops the queue."""
    w = A.load_model(); out = load(); d = np.load(A.DATA); train = d["all_X"][~d["all_val"]].astype(np.int64); pf = {}
    for arm in ARMS:
        torch.manual_seed(A.SEED0 + 1)
        rows = sample(w, 96, 8 / 24, arm).cpu().numpy()
        pf[arm] = dict(round_trip=A.round_trip(w, rows[:20]), copies=A.copies(rows, train)[1])
        print(arm, json.dumps(pf[arm]), flush=True)
    out["preflight"] = pf; save(out)
    if not all(p["round_trip"].get("passed") for p in pf.values()): raise SystemExit("REMASK84_PROBES FAILED")
    print("REMASK84_PROBES PASS", flush=True)


def gen(seeds=SEEDS):
    out = load(); w = A.load_model()
    out.update(arms=ARMS, asks=ASK_NUMERATORS, checkpoint_sha256=hashlib.sha256(A.checkpoint_path().read_bytes()).hexdigest())
    for arm in ARMS:
        for seed in seeds:
            for k in ASK_NUMERATORS:
                tag = f"{arm}_a{k:02d}of24_s{seed}"
                if tag in out["cells"]: continue
                armed = S.Sampler84(w, arm, guidance=A.GUIDANCE); t0 = time.time()
                kept, stats, counts = A.draw(armed, k / 24, seed)
                out["cells"][tag] = dict(arm=arm, ask=k / 24, numerator=k, seed=seed, **counts, pastes=kept, stat_points=stats, sampler=armed.stats,
                                         seconds=time.time() - t0, measures=team_measures(kept, stats) if kept else None)
                save(out); m = out["cells"][tag]["measures"] or {}
                print(f"  {tag:34s} kept {len(kept):3d} legal {counts['legal']:6.1%} line-ups {m.get('lineups', 0):3d} eff {m.get('effective_species', 0):5.1f} "
                      f"66 {m.get('spend_66', 0):.0%} {time.time() - t0:5.0f}s", flush=True)


def battle():
    A.require_shuffled(); con = MDB.connect(); out = load()
    for tag, c in out["cells"].items():
        if "y" in c or not c["pastes"]: continue
        sc = MDB.score_pastes(con, c["pastes"], per_cell=A.BATTLES_PER_OPPONENT, seed=A.SEED0 + c["seed"], origin=f"remask84:{tag}")
        if not A.complete(sc): print(f"  {tag} INCOMPLETE: left for the next run", flush=True); continue
        c["y"] = [s["score"] for s in sc]; save(out); print(f"  {tag:34s} win {np.mean(c['y']):.3f}", flush=True)


RANDOM_ARMS = {
    "rand_plain": "No remasking, 84 passes",
    "rand_remask_conf": "Confidence-weighted 0.3 fields remasked, 210 passes",
    "rand_remask_even": "Even-odds 0.5 fields remasked, 210 passes",
    "species_then_random": "Species first, then random order: 6 passes, then 78",
}
SPECIES = S.SPECIES
RANDOM_TRIES = 384


def random_decoding():
    """Sampling only: every try is counted, with no stop at a number of legal teams. Sets the try cap and battle budget later."""
    smoke = bool(os.environ.get("REMASK84_SMOKE"))
    path = A.RES / f"remask84_random_decoding{'_smoke' if smoke else ''}.json"
    out = json.load(open(path)) if path.exists() else {"cells": {}}
    out.update(arms=RANDOM_ARMS, asks=ASK_NUMERATORS, tries_per_cell=RANDOM_TRIES, checkpoint_sha256=hashlib.sha256(A.checkpoint_path().read_bytes()).hexdigest())
    w = A.load_model(); val = A.Validator()
    tries, seeds, asks = (96, [1], [8]) if smoke else (RANDOM_TRIES, SEEDS, ASK_NUMERATORS)
    try:
        for arm in RANDOM_ARMS:
            for seed in seeds:
                for k in asks:
                    tag = f"{arm}_a{k:02d}of24_s{seed}"
                    if tag in out["cells"]: continue
                    torch.manual_seed(A.SEED0 + seed); t0, stats = time.time(), {}
                    legal, rows, stat_points, reasons, rejected, held, tried = [], [], [], Counter(), 0, 0, 0
                    while tried < tries:
                        for r in sample_random(w, 96, k / 24, arm, stats).cpu().numpy():
                            tried += 1
                            p, slots = w.to_paste(r, repair=True); err = val(p)
                            if err is not None: rejected += 1; reasons[A.reason(err)] += 1
                            elif A.HO.is_heldout_text(p): held += 1
                            else: legal.append(p); rows.append(r.tolist()); stat_points.append([(s["species"], s["evs"]) for s in slots])
                    share = (tried - rejected) / tried
                    out["cells"][tag] = dict(arm=arm, ask=k / 24, numerator=k, seed=seed, tried=tried, rejected=rejected, heldout_dropped=held, legal=share, reasons=reasons.most_common(10),
                                             pastes=legal, rows=rows, stat_points=stat_points, sampler=stats, seconds=time.time() - t0, measures=team_measures(legal, stat_points) if legal else None)
                    tmp = path.with_suffix(".tmp"); json.dump(out, open(tmp, "w")); os.replace(tmp, path)
                    print(f"  {tag:36s} legal {share:6.1%} of {tried} · {time.time() - t0:5.0f}s · {reasons.most_common(1)}", flush=True)
    finally:
        val.close()
    picked = [np.array(r) for c in out["cells"].values() for r in c["rows"]][:20]
    out["round_trip"] = A.round_trip(w, picked) if picked else dict(skipped="no legal team")
    out["summary"] = {arm: float(np.mean([c["legal"] for c in out["cells"].values() if c["arm"] == arm])) for arm in RANDOM_ARMS if any(c["arm"] == arm for c in out["cells"].values())}
    tmp = path.with_suffix(".tmp"); json.dump(out, open(tmp, "w")); os.replace(tmp, path)
    print("REMASK84_RANDOM_DONE", json.dumps(out["summary"]), json.dumps(out["round_trip"]), flush=True)


LOOKAHEAD_ARMS = {                                       # code ID -> (random decoding arm, reader name); all with the lookahead on
    "rand_la_plain": ("rand_plain", "No remasking, 84 passes"),
    "rand_la_remask_conf": ("rand_remask_conf", "Confidence-weighted 0.3 fields remasked, 210 passes"),
    "rand_la_remask_even": ("rand_remask_even", "Even-odds 0.5 fields remasked, 210 passes"),
}
LOOKAHEAD_ASKS = list(range(1, 24))
LOOKAHEAD_OUT = A.RES / "remask84_random_lookahead.json"


def la_load(): return json.load(open(LOOKAHEAD_OUT)) if LOOKAHEAD_OUT.exists() else {"cells": {}}


def la_save(out):
    tmp = LOOKAHEAD_OUT.with_suffix(".tmp"); json.dump(out, open(tmp, "w")); os.replace(tmp, LOOKAHEAD_OUT)


def la_probes():
    """Random decoding with the lookahead, per arm at ask 8/24: round trip and copies on 96 teams. A failed round trip stops the queue."""
    w = A.load_model(); out = la_load(); d = np.load(A.DATA); train = d["all_X"][~d["all_val"]].astype(np.int64); pf = {}
    for code, (arm, _) in LOOKAHEAD_ARMS.items():
        torch.manual_seed(A.SEED0 + 1)
        rows = sample_random(w, 96, 8 / 24, arm, {}, lookahead=True).cpu().numpy()
        pf[code] = dict(round_trip=A.round_trip(w, rows[:20]), copies=A.copies(rows, train)[1])
        print(code, json.dumps(pf[code]), flush=True)
    out["preflight"] = pf; la_save(out)
    if not all(p["round_trip"].get("passed") for p in pf.values()): raise SystemExit("REMASK84_LA_PROBES FAILED")
    print("REMASK84_LA_PROBES PASS", flush=True)


def la_gen():
    """Each cell draws through ask_grid84.draw: until 64 legal, non-held-out teams or 1,280 tries, guidance 2."""
    out = la_load(); w = A.load_model()
    out.update(arms={k: v[1] for k, v in LOOKAHEAD_ARMS.items()}, asks=LOOKAHEAD_ASKS, lookahead=True, keep=KEEP, max_tries=A.MAX_TRIES,
               checkpoint_sha256=hashlib.sha256(A.checkpoint_path().read_bytes()).hexdigest())
    for code, (arm, _) in LOOKAHEAD_ARMS.items():
        for seed in SEEDS:
            for k in LOOKAHEAD_ASKS:
                tag = f"{code}_a{k:02d}of24_s{seed}"
                if tag in out["cells"]: continue
                armed = S.Sampler84(w, arm, lookahead=True, guidance=A.GUIDANCE); t0 = time.time()
                kept, stats, counts = A.draw(armed, k / 24, seed)
                out["cells"][tag] = dict(arm=code, ask=k / 24, numerator=k, seed=seed, **counts, pastes=kept, stat_points=stats, sampler=armed.stats,
                                         seconds=time.time() - t0, measures=team_measures(kept, stats) if kept else None)
                la_save(out); m = out["cells"][tag]["measures"] or {}
                print(f"  {tag:38s} kept {len(kept):3d} of {counts['tried']:4d} legal {counts['legal']:6.1%} line-ups {m.get('lineups', 0):3d} "
                      f"{time.time() - t0:5.0f}s", flush=True)


def la_battle():
    A.require_shuffled(); con = MDB.connect(); out = la_load()
    for tag, c in out["cells"].items():
        if "y" in c or not c["pastes"]: continue
        sc = MDB.score_pastes(con, c["pastes"], per_cell=A.BATTLES_PER_OPPONENT, seed=A.SEED0 + c["seed"], origin=f"remask84_la:{tag}")
        if not A.complete(sc): print(f"  {tag} INCOMPLETE: left for the next run", flush=True); continue
        c["y"] = [s["score"] for s in sc]; la_save(out); print(f"  {tag:38s} win {np.mean(c['y']):.3f}", flush=True)


def la_report():
    """Pooled got per ask and the ask-versus-got error of each arm. The paired comparison is remask84_random_error.py."""
    out = la_load(); rows = {}
    for code in LOOKAHEAD_ARMS:
        rows[code] = [dict(numerator=k, win=float(np.mean([np.mean(c["y"]) for c in at])))
                      for k in LOOKAHEAD_ASKS if (at := [c for c in out["cells"].values() if c["arm"] == code and c["numerator"] == k and c.get("y")])]
        line = A.line_measures(rows[code]) if len(rows[code]) == len(LOOKAHEAD_ASKS) else {}
        print(f"{code:22s} {A.headline(line) if line else 'grid not finished'}", flush=True)
    out["summary"] = rows; la_save(out)


def slope(asks, wins):
    return float(np.polyfit(asks, wins, 1)[0])


def accumulation(cells, orders=200, step=16):
    """Species accumulation curve, as in ecology: distinct line-ups, and line-ups at or above 0.531 on these 49 battles, as teams are added in random order."""
    teams = [(tuple(species_of(p)), y) for c in cells for p, y in zip(c["pastes"], c.get("y") or [None] * len(c["pastes"]))]
    rng = np.random.default_rng(0); sizes = list(range(step, len(teams) + 1, step)); seen, strong = np.zeros(len(sizes)), np.zeros(len(sizes))
    for _ in range(orders):
        order = rng.permutation(len(teams))
        for j, n in enumerate(sizes):
            pick = [teams[i] for i in order[:n]]
            seen[j] += len({t for t, _ in pick}); strong[j] += len({t for t, y in pick if y is not None and round(y, 4) >= 0.531})
    return dict(teams=sizes, lineups=(seen / orders).round(2).tolist(), strong_lineups_unconfirmed=(strong / orders).round(2).tolist())


def control_cells():
    cells = json.load(open(CONTROL))["cells"]
    return {f"plain_a{k:02d}of24_s{s}": cells[f"gen84_all_ep21_a{k:02d}of24_s{s}"] for k in ASK_NUMERATORS for s in SEEDS}


def report():
    out = load(); rows = {}
    plain = control_cells()
    for tag, c in plain.items():
        c["arm"] = "plain"; c["measures"] = team_measures(c["pastes"], c["stat_points"])
    cells = list(plain.values()) + list(out["cells"].values())
    for arm in ["plain", *ARMS]:
        rows[arm] = []
        for k in ASK_NUMERATORS:
            at = [c for c in cells if c["arm"] == arm and c["numerator"] == k]
            if not at: continue
            ys = [np.mean(c["y"]) for c in at if c.get("y")]
            meas = lambda f: float(np.mean([c["measures"][f] for c in at if c["measures"]]))
            rows[arm].append(dict(numerator=k, win=float(np.mean(ys)) if ys else None, win_by_seed=[float(y) for y in ys], legal=float(np.mean([c["legal"] for c in at])),
                                  lineups=meas("lineups"), effective_species=meas("effective_species"), spend_66=meas("spend_66")))
        print(arm, json.dumps(rows[arm]), flush=True)
    fits = {}
    for arm, row in rows.items():
        done = [r for r in row if r["win"] is not None]
        if len(done) == len(ASK_NUMERATORS):                    # a partial grid would label a lower ask as the top ask
            keep =[r for r in done if r["numerator"] != 12]    # 12/24 selects a different team population
            fits[arm] = dict(slope=slope([r["numerator"] / 24 for r in done], [r["win"] for r in done]), slope_without_12=slope([r["numerator"] / 24 for r in keep], [r["win"] for r in keep]),
                             top_ask_win=done[-1]["win"], line=A.line_measures(done), accumulation=accumulation([c for c in cells if c["arm"] == arm]))
        print(arm, json.dumps({k: v for k, v in fits.get(arm, {}).items() if k != "accumulation"}), flush=True)
    out["summary"] = rows; out["fits"] = fits; save(out)


if __name__ == "__main__":
    what = sys.argv[1]
    if what == "check": check()
    elif what == "probes": probes()
    elif what == "random_decoding": random_decoding()
    elif what == "gen": gen()
    elif what == "battle": battle()
    elif what == "report": report()
    elif what == "la_probes": la_probes()
    elif what == "la_gen": la_gen()
    elif what == "la_battle": la_battle()
    elif what == "la_report": la_report()
    else: raise SystemExit(__doc__)
