"""Does remasking close the ask gap? (2026-10-30) Pre-registration: docs/remask-ask-gap.md.

The proposer's teams fall short of the win rate they are asked for, and decoding order changes its picks: both
suggest an early unmasking choice can't be undone. Remasking (ReMDM; Wang, Schiff et al., 2025) masks a filled
column again and redraws it under the ask. It changes only the sampler: DiscreteDiffusionModel.sample(remask=, steps=).

Model: DiscreteDiffusionModel.load(), the clean v3 members. Every arm samples as the loop does: classifier-free guidance 2
(cem_dbbo.py), temperature 1, rule masks on, no surrogate tilt, to_paste(repair=True).
Arms: plain masking (the current sampler); remasking 5%, 10% and 20% of the filled columns before each pass, at
the same 48 network passes; remasking 10% at 96 passes (inference scaling). Asks 0.3, 0.5 and 0.531 (elite cut).
Seeds 1-3 at base 7300, shared by every arm, so plain masking at ask 0.5 redraws guidance_legality.py's g2 teams.
Each cell draws until 64 teams pass Showdown's validator and are not held-out teams. Legality = accepted / tried.
Battles: one against each top-50 meta team (49 per team), behaviour-cloning policy on both sides, slots shuffled.
Fresh: per arm and ask, the 4 best distinct teams by that label (seeds pooled), re-battled in a private matchup
database, 4 battles per top-50 meta team (196 per team), so none of the selection battles is reused.

    python remask_ask_gap.py gen
    python battle_supervisor.py /path/to/remask_ask_gap.py battle --then fresh --then report
    python remask_ask_gap.py report
    RA_SMOKE=1 python remask_ask_gap.py gen|battle|fresh|report     # plumbing check, own files
(run with /tmp/vgc-pilot/.venv/bin/python)
"""
import hashlib, json, math, os, sys, time
from collections import Counter
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, "/tmp/vgc-pilot/src")
sys.path.insert(0, str(HERE))
import discrete_diffusion_model
from discrete_diffusion_model import DiscreteDiffusionModel, MEMBERS
sys.path.insert(0, str(HERE))                       # discrete_diffusion_model's imports put /tmp/vgc-pilot/src first again
import matchup_db as MDB
import heldout as HO
import top50 as T50
from hps_generate import Validator
from team_diversity import summary as diversity_summary, slots_from_paste

if Path(discrete_diffusion_model.__file__).resolve().parent != HERE:
    raise SystemExit(f"loaded {discrete_diffusion_model.__file__}, not the discrete diffusion model beside this script")

GUIDANCE, KEEP, BATCH, MAX_TRIES = 2.0, 64, 96, 20 * 64
ASKS = [0.3, 0.5, 0.531]
ARMS = [("plain", 0.0, 48), ("remask05", 0.05, 48), ("remask10", 0.10, 48), ("remask20", 0.20, 48), ("remask10x2", 0.10, 96)]
NAMES = {"plain": "plain masking", "remask05": "remasking 5%", "remask10": "remasking 10%", "remask20": "remasking 20%",
         "remask10x2": "remasking 10%, 96 passes"}
FIXED_COMPUTE = ("remask05", "remask10", "remask20")      # arms that can make remasking pass; the 96-pass arm is a check
SEEDS, SEED0, EXAMPLES = [1, 2, 3], 7300, 5
FRESH_N, FRESH_PER_CELL, FRESH_SEED = 4, 4, 7390
NOISE_K = 2.0                       # "more than seed noise" = more than twice the seed noise
FLOOR = dict(legal=0.01, distinct_lineups=1.0, species_entropy=0.01)   # smallest drop that counts as lowering
SMOKE = bool(os.environ.get("RA_SMOKE"))
if SMOKE: KEEP, BATCH, SEEDS, ASKS, ARMS, FRESH_N, FRESH_PER_CELL = 4, 8, [1], [0.5], [ARMS[0], ARMS[2], ARMS[4]], 1, 1
RES = Path.home() / "vgc-data" / "results"
OUT = RES / ("remask_ask_gap_smoke.json" if SMOKE else "remask_ask_gap.json")
FRESH_DB = Path.home() / "vgc-data" / "remask_ask_gap" / ("fresh_smoke.sqlite" if SMOKE else "fresh.sqlite")
GLEG = RES / "guidance_legality.json"               # its g2 cells are plain masking at ask 0.5, same seeds and batch
ORIGIN = "rmasksmoke" if SMOKE else "rmask"


def save(out):
    tmp = OUT.with_suffix(".tmp"); json.dump(out, open(tmp, "w")); os.replace(tmp, OUT)     # never a half-written file
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def lineup(slots): return tuple(sorted(s["species"] for s in slots))
def fields(slots): return tuple(sorted((s["species"], s["ability"], s["item"], s["nature"], tuple(sorted(s["moves"]))) for s in slots))
def tag_of(arm, ask, seed): return f"{arm}_a{ask}_s{seed}"


def vendi(lineups):
    """Vendi score (Friedman & Dieng, TMLR 2023) with K_ij = shared species / 6, as guidance_legality.py."""
    vocab = {s: i for i, s in enumerate(sorted({s for l in lineups for s in l}))}
    B = np.zeros((len(lineups), len(vocab)))
    for i, l in enumerate(lineups):
        for s in l: B[i, vocab[s]] += 1 / math.sqrt(6)
    lam = np.linalg.eigvalsh(B @ B.T / len(lineups)); lam = lam[lam > 1e-12]
    return float(np.exp(-(lam * np.log(lam)).sum()))


def diversity(pastes):
    """team_diversity.summary (distinct line-ups, effective species, mean slot-matched distance out of 48, ...),
    plus species entropy in nats (log of effective species) and Vendi."""
    teams = [slots_from_paste(p) for p in pastes]
    d = diversity_summary(teams)
    return dict(d, species_entropy=math.log(d["effective_species"]), vendi=vendi([lineup(t) for t in teams if t]))


def draw(w, ask, remask, steps, seed):
    """Sample until KEEP teams pass Showdown and are not held-out teams; counts every try. Plain masking calls the
    current sampler itself; the remasking arms fill `stats` with redraws per field."""
    torch.manual_seed(SEED0 + seed); rng = np.random.default_rng(SEED0 + seed)
    plain = remask == 0 and steps == 48
    stats, sampled, batches = {}, 0, 0
    val = Validator(); kept, tried, rejected, heldout, reasons, examples, t0 = [], 0, 0, 0, Counter(), [], time.perf_counter()
    try:
        while len(kept) < KEEP and tried < MAX_TRIES:
            kw = {} if plain else dict(remask=remask, steps=steps, stats=stats)
            rows = w.sample(BATCH, ask=ask, guidance=GUIDANCE, **kw).cpu().numpy(); sampled += len(rows); batches += 1
            for r in rows:
                tried += 1
                p, _ = w.to_paste(r, rng, repair=True)
                err = val(p)
                if err is not None:
                    rejected += 1; reasons[str(err).split("\n")[0][:120]] += 1
                    if len(examples) < EXAMPLES: examples.append(dict(paste=p, error=str(err)[:400]))
                elif HO.is_heldout_text(p): heldout += 1
                else: kept.append(p)
                if len(kept) >= KEEP or tried >= MAX_TRIES: break
    finally:
        val.close()
    per_team = lambda d: {k: v / sampled for k, v in sorted(d.items())}
    redraws = dict(passes_per_batch=(stats.get("passes", 48 * batches)) / batches, remasked=per_team(stats.get("remasked", {})),
                   changed=per_team(stats.get("changed", {})))
    return dict(kept=kept, tried=tried, rejected=rejected, heldout=heldout, reasons=reasons.most_common(8),
                examples=examples, seconds=time.perf_counter() - t0, redraws=redraws)


def gen():
    out = json.load(open(OUT)) if OUT.exists() else {"config": dict(
        arms=ARMS, asks=ASKS, seeds=SEEDS, seed0=SEED0, keep=KEEP, batch=BATCH, max_tries=MAX_TRIES, guidance=GUIDANCE,
        temperature=1.0, masks="on", repair=True, members=list(MEMBERS), battles_per_opponent=1, slots="shuffled",
        fresh_n=FRESH_N, fresh_per_opponent=FRESH_PER_CELL, team_writer_sha256=sha(discrete_diffusion_model.__file__),
        script_sha256=sha(__file__), script=str(Path(__file__).resolve())), "cells": {}}
    missing = [(seed, ask, arm, remask, steps) for seed in SEEDS for ask in ASKS for arm, remask, steps in ARMS
               if tag_of(arm, ask, seed) not in out["cells"]]
    if not missing: return
    saved = out["config"]; now = dict(team_writer_sha256=sha(discrete_diffusion_model.__file__), script_sha256=sha(__file__),
                                      arms=[list(a) for a in ARMS], asks=ASKS, seeds=SEEDS, keep=KEEP, guidance=GUIDANCE)
    changed = [k for k, v in now.items() if json.loads(json.dumps(saved.get(k))) != json.loads(json.dumps(v))]
    if out["cells"] and changed and not os.environ.get("RA_RESUME_CHANGED"):
        raise SystemExit(f"{OUT} was started with different {changed}; finishing it would mix implementations "
                         "(RA_RESUME_CHANGED=1 to do it anyway)")
    w = DiscreteDiffusionModel.load()
    for seed, ask, arm, remask, steps in missing:
        tag = tag_of(arm, ask, seed)
        r = draw(w, ask, remask, steps, seed)
        legal = (r["tried"] - r["rejected"]) / max(r["tried"], 1)
        out["cells"][tag] = dict(arm=arm, remask=remask, steps=steps, ask=ask, seed=seed, tried=r["tried"],
                                 rejected=r["rejected"], heldout=r["heldout"], legal=legal, kept=len(r["kept"]),
                                 capped=len(r["kept"]) < KEEP, reasons=r["reasons"], examples=r["examples"],
                                 seconds=r["seconds"], redraws=r["redraws"],
                                 div=diversity(r["kept"]) if r["kept"] else None, pastes=r["kept"])
        save(out)
        d = out["cells"][tag]["div"] or {}
        print(f"  {tag:22s} legal {legal:6.1%} ({r['tried'] - r['rejected']}/{r['tried']}) · line-ups "
              f"{d.get('distinct_lineups')} · eff species {d.get('effective_species', 0):.1f} · distance "
              f"{d.get('mean_distance') or 0:.1f} · changed/team {sum(r['redraws']['changed'].values()):.1f} · "
              f"{r['seconds']:.0f}s", flush=True)


def require_shuffled():
    """The grid and the fresh re-battle are defined with slots shuffled every battle; refuse any other order."""
    if MDB.SLOT_ORDER != "shuffled": raise SystemExit(f"VGC_SLOT_ORDER={MDB.SLOT_ORDER}: these battles must shuffle slots")


def complete(scores, per_cell):
    """Every team scored against every column of the set, with at least per_cell battles in each."""
    return all(s["score"] is not None and s["coverage"] == s["columns"] and s["battles"] >= per_cell * s["columns"]
               for s in scores)


def battle():
    require_shuffled(); con = MDB.connect(); out = json.load(open(OUT))
    for tag, c in out["cells"].items():
        if "y" in c or not c["pastes"]: continue
        sc = MDB.score_pastes(con, c["pastes"], per_cell=1, seed=SEED0 + c["seed"], origin=f"{ORIGIN}:{tag}")
        if not complete(sc, 1):
            print(f"  {tag:22s} INCOMPLETE: a team lacks some opponents; left for the next run", flush=True); continue
        y = np.array([s["score"] for s in sc], float)
        c["y"] = [s["score"] for s in sc]
        c["win"] = dict(mean=float(y.mean()), se=float(y.std(ddof=1) / math.sqrt(len(y))) if len(y) > 1 else 0.0, n=len(y))
        save(out)
        print(f"  {tag:22s} battled {len(y)}: win {y.mean():.3f} ± {c['win']['se']:.3f}", flush=True)


def fresh_db():
    """A private matchup database with the top-50 meta teams as its column set (as scripts/jev_rebattle.py), so the
    finalists' battles are all new rather than read back from the selection cells."""
    con = MDB.connect(FRESH_DB)
    pid = MDB.get_policy(con, note="remasking ask-gap finalists, slots shuffled, 2026-10-30")
    if "+slots-shuffled" not in con.execute("SELECT row_policy FROM policy WHERE policy_id=?", (pid,)).fetchone()[0]:
        raise SystemExit("the fresh battles must shuffle slots")
    val = Validator()
    try:
        with con:
            con.execute("DELETE FROM team_set WHERE set_name='top50'")
            for t in json.load(open(T50.TOP50)):              # as matchup_db.seed: a team listed twice counts twice
                tid = MDB.add_team(con, Path(T50.resolve(t["id"])).read_text(), "vgcpastes", val)[0]
                con.execute("INSERT INTO team_set VALUES ('top50', ?, 1) ON CONFLICT (set_name, team_id) "
                            "DO UPDATE SET weight = weight + 1", (tid,))
    finally:
        val.close()
    return con


def fresh():
    """Per arm and ask: the FRESH_N best distinct teams by their 49-battle label, seeds pooled, re-battled fresh."""
    out = json.load(open(OUT)); res = out.setdefault("fresh", {}); con = fresh_db()
    for ask in ASKS:
        for arm, _, _ in ARMS:
            key = f"{arm}_a{ask}"
            if key in res: continue
            cs = [c for c in out["cells"].values() if c["arm"] == arm and c["ask"] == ask]
            if sorted(c["seed"] for c in cs if "y" in c) != SEEDS:          # pool every seed before choosing
                print(f"  {key:18s} skipped: not every seed is battled yet", flush=True); continue
            labelled = sorted(((p, y) for c in cs for p, y in zip(c["pastes"], c["y"]) if y is not None), key=lambda t: -t[1])
            top, seen = [], set()
            for p, y in labelled:
                k = fields(slots_from_paste(p))
                if k in seen: continue
                seen.add(k); top.append((p, y))
                if len(top) == FRESH_N: break
            sc = MDB.score_pastes(con, [p for p, _ in top], per_cell=FRESH_PER_CELL, seed=FRESH_SEED, origin=f"{ORIGIN}-fresh:{key}")
            if not top or not complete(sc, FRESH_PER_CELL):
                print(f"  {key:18s} INCOMPLETE fresh battles: left for the next run", flush=True); continue
            res[key] = [dict(paste=p, label=y, fresh=s["score"], battles=s.get("battles"), se=s.get("se"),
                             species=list(lineup(slots_from_paste(p)))) for (p, y), s in zip(top, sc)]
            save(out)
            best = max((r for r in res[key] if r["fresh"] is not None), key=lambda r: r["fresh"])
            print(f"  {key:18s} fresh {[r['fresh'] for r in res[key]]} · best {best['fresh']:.3f} "
                  f"({best['battles']} battles; label {best['label']:.3f})", flush=True)


def stat(vals):
    v = np.array(vals, float)
    return dict(mean=float(v.mean()), sd=float(v.std(ddof=1)) if len(v) > 1 else 0.0, seeds=v.tolist())


def judge(ask, arm_cells, plain_cells):
    """The pre-registered rule at one ask (docs/remask-ask-gap.md, "Pass rule"). Cells are paired by seed."""
    pw = [c["win"]["mean"] for c in plain_cells]; aw = [c["win"]["mean"] for c in arm_cells]
    noise = max(stat(pw)["sd"], float(np.mean([c["win"]["se"] for c in plain_cells])))
    narrowing = [abs(ask - p) - abs(ask - a) for p, a in zip(pw, aw)]
    has_gap = abs(ask - np.mean(pw)) > NOISE_K * noise
    if has_gap: gap_ok = all(d > 0 for d in narrowing) and np.mean(narrowing) > NOISE_K * noise
    else: gap_ok = np.mean(narrowing) >= -NOISE_K * noise               # no gap to close: must not open one
    lowered = {}
    for m in ("legal", "distinct_lineups", "species_entropy"):
        get = (lambda c: c["legal"]) if m == "legal" else (lambda c, m=m: c["div"][m])
        pv, av = [get(c) for c in plain_cells], [get(c) for c in arm_cells]
        margin = max(NOISE_K * stat(pv)["sd"], FLOOR[m])
        lowered[m] = bool(np.mean(av) < np.mean(pv) - margin)
    return dict(seed_noise=noise, has_gap=bool(has_gap), narrowing=stat(narrowing), all_seeds_narrow=all(d > 0 for d in narrowing),
                gap_ok=bool(gap_ok), lowered=lowered, passes=bool(gap_ok and not any(lowered.values())))


def reproduces_guidance_legality(out):
    """Plain masking at ask 0.5 should redraw guidance_legality.py's g2 teams paste for paste, with the same scores."""
    if SMOKE or not GLEG.exists() or 0.5 not in ASKS: return None
    g = json.load(open(GLEG))["cells"]
    mine = [out["cells"].get(tag_of("plain", 0.5, s)) for s in SEEDS]; theirs = [g.get(f"g2_s{s}", {}) for s in SEEDS]
    if not all(mine): return None                                       # generation not finished: pending
    return dict(pastes=all(a["pastes"] == b.get("pastes") for a, b in zip(mine, theirs)),
                scores=all(a.get("y") == b.get("y") for a, b in zip(mine, theirs)))


def report():
    out = json.load(open(OUT)); cells = out["cells"]; summ, verdict = {}, {}
    get = lambda arm, ask: sorted((c for c in cells.values() if c["arm"] == arm and c["ask"] == ask), key=lambda c: c["seed"])
    for ask in ASKS:
        base = get("plain", ask)
        for arm, remask, steps in ARMS:
            cs = get(arm, ask)
            if [c["seed"] for c in cs if "win" in c] != SEEDS or [c["seed"] for c in base if "win" in c] != SEEDS:
                print(f"{NAMES[arm]} at ask {ask}: incomplete (every seed of it and of plain masking needs a score)"); continue
            fr = [r for r in out.get("fresh", {}).get(f"{arm}_a{ask}") or [] if r["fresh"] is not None]
            best = max(fr, key=lambda r: r["fresh"]) if fr else None
            s = dict(arm=arm, name=NAMES[arm], remask=remask, steps=steps, ask=ask,
                     legal=stat([c["legal"] for c in cs]), win=stat([c["win"]["mean"] for c in cs]),
                     gap=stat([abs(ask - c["win"]["mean"]) for c in cs]), signed_gap=stat([c["win"]["mean"] - ask for c in cs]),
                     **{m: stat([c["div"][m] for c in cs]) for m in ("distinct_lineups", "species_entropy", "effective_species",
                                                                     "distinct_species", "mean_distance", "vendi")},
                     changed_per_team={f: float(np.mean([c["redraws"]["changed"].get(f, 0) for c in cs]))
                                       for f in ("species", "ability", "item", "move", "nature")},
                     remasked_per_team=float(np.mean([sum(c["redraws"]["remasked"].values()) for c in cs])),
                     best_fresh=None if best is None else dict(fresh=best["fresh"], battles=best["battles"], label=best["label"],
                                                               species=best["species"], paste=best["paste"]))
            if arm != "plain": s["rule"] = judge(ask, cs, base)
            summ[f"{arm}_a{ask}"] = s
    for arm, _, _ in ARMS[1:]:
        rules = [summ[f"{arm}_a{a}"]["rule"] for a in ASKS if f"{arm}_a{a}" in summ]
        if len(rules) == len(ASKS): verdict[arm] = all(r["passes"] for r in rules)
    fixed = [verdict[a] for a in FIXED_COMPUTE if a in verdict]
    eligible = [a for a, _, _ in ARMS if a in FIXED_COMPUTE]            # pending until one passes or every one has failed
    passes = True if any(fixed) else False if len(fixed) == len(eligible) else None
    out["summary"], out["verdict"] = summ, dict(arms=verdict, remasking_passes=passes,
                                                 reproduces_guidance_legality=reproduces_guidance_legality(out))
    save(out)
    for k, s in summ.items():
        r = s.get("rule"); b = s["best_fresh"]
        print(f"{s['name']:26s} ask {s['ask']:.3f} · legal {s['legal']['mean']:6.1%} · win {s['win']['mean']:.3f} "
              f"{[round(v, 3) for v in s['win']['seeds']]} · gap {s['gap']['mean']:.3f} · line-ups {s['distinct_lineups']['mean']:4.1f} "
              f"· entropy {s['species_entropy']['mean']:.2f} · distance {s['mean_distance']['mean']:4.1f} · best fresh "
              + (f"{b['fresh']:.3f}" if b and b["fresh"] is not None else "pending")
              + ("" if r is None else f" · narrowing {r['narrowing']['mean']:+.3f} (noise {r['seed_noise']:.3f}) "
                 f"{'PASS' if r['passes'] else 'fail'}{' lowered ' + ','.join(m for m, v in r['lowered'].items() if v) if any(r['lowered'].values()) else ''}"))
    print("verdict:", json.dumps(out["verdict"]))


if __name__ == "__main__":
    {"gen": gen, "battle": battle, "fresh": fresh, "report": report}[sys.argv[1]]()
