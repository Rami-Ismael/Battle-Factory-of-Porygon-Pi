"""Line-up repeats, trained from scratch (2026-10-31). Pre-registration: docs/line-up-repeats.md.

Why: the deployed model's training data holds no duplicate teams (same six Pokémon, any slot order), but species
line-ups repeat: the battle-records model's 23,871 real rows hold 11,118 distinct line-ups, one of them 2,377 times.
This run asks whether those repeats cost win rate, diversity or novelty.

Arms, each the deployed two-member model retrained from scratch with the deployed recipe (lossdown.train_run with the
parents' configs, then retrain_regmb train for the Regulation M-B v3 vocabulary, 150 embedding steps), the same seeds,
the same step caps and the same early stopping as the baseline; only the rows each epoch draws from differ:
  Every row               control            the species quota run's baseline, reused (species_quota_scratch_es.json)
  One team per line-up    one_per_lineup     one randomly picked team per species line-up, the same pick every epoch
  Random rows, same count random_same_count  a random subset the size of one_per_lineup's, repeats kept
The filter applies to every training row in both stages (real, battled and synthetic coverage teams). Each epoch
cycles through the kept rows until it has the baseline's number of rows. Frozen copies of lossdown.py,
retrain_regmb.py, discrete_diffusion_model.py, remask_ask_gap.py and team_diversity.py are byte-identical to the
species quota run's, so the baseline and these arms share their code, including its EARLY hook.

Early stopping, as the species quota run (owner, 2026-11-01): one seeded random 2% of the non-corpus rows (seed
20261101; real-corpus rows, which hold the top-50 meta teams, never enter it; ranks 51-100 and the test teams are
already out) is held out from both members and every arm, before the row filter sees the rows. Stage 1 measures the
common-random-number ELBO (evidence lower bound, the diffusion model's loss) on it after every epoch (lossdown.evaluate, reps 4, seed 12345), stops after 3 epochs
without a new lowest, caps at the parent's epoch count (80 and 60) and restores the best epoch. Stage 2 checks 8 times
over its 2-epoch cap, patience 3 checks; its 150 embedding warm-up steps stay fixed. An epoch here is the baseline's
number of rows, so an arm's epoch cycles its kept rows and the step caps match the baseline's.

Scoring as the species quota run: plain masking, guidance 2, temperature 1, 64 Showdown-legal non-held-out teams per
cell, asks 7/24, 0.3, 8/24, 0.5, 0.531, 13/24, 0.6, 0.7, seeds 1-3, one battle against each of the 49 distinct top-50
meta teams, slots shuffled, behaviour-cloning policy on both sides.

Read-out over asks 12/24, 0.531 and 13/24, each arm against the baseline: win rate change with a 95% interval;
diversity (distinct line-ups, effective species, mean slot-matched distance out of 48); novelty, the share of teams at
zero, one, two, and three or more species swaps from the closest training line-up, against the arm's own training
line-ups and against the baseline's. One training seed per arm, so provisional.

    python line_up_repeats.py train ARM | gen | battle | report     (run with /tmp/vgc-pilot/.venv/bin/python)
    LUR_SMOKE=1 python line_up_repeats.py ...                         # plumbing check, own names and files
"""
import argparse, importlib.util, json, math, os, sys
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, "/tmp/vgc-pilot/src")
sys.path.insert(0, str(HERE))
import lossdown as L
import discrete_diffusion_model
from discrete_diffusion_model import DiscreteDiffusionModel
sys.path.insert(0, str(HERE))                       # discrete_diffusion_model's imports put /tmp/vgc-pilot/src first again
import remask_ask_gap as RA
import matchup_db as MDB
import encode
from checkpoints import load_checkpoint
from encode import NF, NSLOT
from team_diversity import slots_from_paste

for mod in (L, discrete_diffusion_model, RA):
    if Path(mod.__file__).resolve().parent != HERE: raise SystemExit(f"loaded {mod.__file__}, not the frozen copy beside this script")

RETRAIN = HERE / "retrain_regmb.py"
SCRIPTS = Path.home() / "code/vgc-team-generator-pilot/scripts"       # regmb_training_policy lives there
PARENTS = ("F1_final_medium_noheldout", "G2_family_matrix_noprtrain_noheldout")   # loop-teams model, battle-records model
ARMS = ("one_per_lineup", "random_same_count")
NAMES = {"control": "Every row", "one_per_lineup": "One team per line-up", "random_same_count": "Random rows, same count"}
ASKS = [7 / 24, 0.3, 8 / 24, 0.5, 0.531, 13 / 24, 0.6, 0.7]
POOLED_ASKS = (0.5, 0.531, 13 / 24)                   # the read-out pools these asks
PICK_SEED, EPOCH_SEED = 20261101, 20261102
SMOKE = bool(os.environ.get("LUR_SMOKE"))
SEEDS = RA.SEEDS
if SMOKE: RA.KEEP, RA.BATCH, RA.MAX_TRIES, SEEDS, ASKS = 4, 8, 64, [1], [0.5]
PREFIX = "luressmoke_p3" if SMOKE else "lures_p3"     # stopping rule in the name: early stopping, patience 3
RES = Path.home() / "vgc-data" / "results"
OUT = RES / ("line_up_repeats_es_smoke.json" if SMOKE else "line_up_repeats_es.json")
BASELINE = RES / "species_quota_scratch_es.json"       # arm "control": Every row, early-stopped
SPLIT_FRAC, SPLIT_SEED, PATIENCE = 0.02, 20261101, 3   # the species quota run's stopping split and patience


def save(out):
    tmp = OUT.with_suffix(".tmp"); json.dump(out, open(tmp, "w")); os.replace(tmp, OUT)


def load_out(): return json.load(open(OUT)) if OUT.exists() else {"cells": {}, "train": {}}
def members(arm): return tuple(f"{PREFIX}_{arm}_{p}_regmb_v3" for p in PARENTS)
def tag_of(arm, ask, seed): return f"{arm}_a{ask:.4f}_s{seed}"
def lineups_file(arm): return RES / f"line_up_repeats_lineups_{PREFIX}_{arm}.json"


# ---------------------------------------------------------------- the row filter
def keep_rows(lineup_ids, arm, rng):
    """Row indices kept for one training stage. lineup_ids: one integer per row naming its species line-up."""
    first = np.unique(lineup_ids, return_index=True)[1]
    if arm == "one_per_lineup":
        perm = rng.permutation(len(lineup_ids))
        keep = np.sort(perm[np.unique(lineup_ids[perm], return_index=True)[1]])
        assert len(np.unique(lineup_ids[keep])) == len(keep) == len(first)
    else:
        keep = np.sort(rng.choice(len(lineup_ids), len(first), replace=False))
    return keep


def row_filter(arm, log, stage, lineups):
    """lossdown.EPOCH_ORDER hook: each epoch, n rows cycled from the kept rows in a fresh random order."""
    erng = np.random.default_rng(EPOCH_SEED); cache = {}

    def order(x, _w, n):
        key = (stage["name"], len(x))
        if key not in cache:
            sp = np.sort(x[:, [k * NF for k in range(NSLOT)]].cpu().numpy(), 1)
            ids = np.unique(sp, axis=0, return_inverse=True)[1].reshape(-1)
            keep = keep_rows(ids, arm, np.random.default_rng(PICK_SEED)); cache[key] = keep
            itos = stage["itos"]
            names = lambda rows: {tuple(sorted(itos[int(t)] for t in r)) for r in rows}   # tokens match paste names, Megas included
            lineups["all"] |= names(sp); lineups["kept"] |= names(sp[keep])
            log.append(dict(stage=stage["name"], rows=len(x), lineups=int(ids.max() + 1), kept=len(keep)))
            print(f"  {NAMES[arm]} [{stage['name']}]: {len(x)} rows, {ids.max() + 1} line-ups -> {len(keep)} kept", flush=True)
        keep = cache[key]; parts, total = [], 0
        while total < n: e = erng.permutation(keep); parts.append(e); total += len(e)
        return torch.as_tensor(np.concatenate(parts)[:n], device=x.device)
    return order


# ---------------------------------------------------------------- training
def stop_split():
    """species_quota_scratch.stop_split: boolean mask over lossdata rows, the held-out stopping rows shared by both
    members, every arm and the baseline."""
    d = L.load_data(include_heldout=True); pool = np.flatnonzero(L.tag_of(d["S"]) != 0)
    rng = np.random.default_rng(SPLIT_SEED); m = np.zeros(len(d["X"]), bool)
    m[rng.choice(pool, round(SPLIT_FRAC * len(pool)), replace=False)] = True
    return m


def stops_of(arm):
    """Each member's recorded early stop, stage 1 and stage 2, from its checkpoints."""
    out = {}
    for n in members(arm):
        c1 = load_checkpoint(L.CK / f"{n[:-len('_regmb_v3')]}.pt", map_location="cpu")["cfg"].get("early_stopping")
        c2 = load_checkpoint(L.CK / f"{n}.pt", map_location="cpu")["cfg"].get("early_stopping")
        out[n] = dict(stage1=c1, stage2=c2)
    return out


def train(arm):
    # The row filter and early stopping reach training only through these two hooks. Without them the assignments
    # below create unused attributes and every arm trains on every row for the full epoch cap.
    missing = [h for h in ("EPOCH_ORDER", "EARLY") if not hasattr(L, h)]
    if missing:
        raise SystemExit(f"{L.__file__} has no {', '.join(missing)} hook: train from the frozen copy in "
                         "~/vgc-data/line_up_repeats/run/ (checked by SHA256SUMS), not from code/src")
    log, lineups, stage = [], {"all": set(), "kept": set()}, {}
    L.EPOCH_ORDER = row_filter(arm, log, stage, lineups)
    L.OUT = RES / ("line_up_repeats_es_train_smoke.json" if SMOKE else "line_up_repeats_es_train.json")
    L.EARLY = dict(mask=stop_split(), patience=PATIENCE, desc=f"{SPLIT_FRAC:.0%} of non-corpus rows, seed {SPLIT_SEED}")
    stage1 = []
    for p in PARENTS:
        name = f"{PREFIX}_{arm}_{p}"; stage1.append(name)
        if (L.CK / f"{name}.pt").exists(): raise SystemExit(f"{name} exists: the line-up record would be incomplete; remove it to retrain")
        cfg = dict(L.DEFAULT, **load_checkpoint(L.CK / f"{p}.pt", map_location="cpu")["cfg"])
        if SMOKE: cfg["epochs"] = 1
        stage.update(name=f"stage 1 {p}", itos=L.X.setup()[1].itos["species"])
        L.train_run(name, cfg)
    sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location("retrain_regmb", RETRAIN); R = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(R)
    R.OUT = R.OUT.parent / "regmb-v3"; torch.set_num_threads(2)
    stage.update(name="stage 2", itos=encode.Vocab.from_regulation(R.OUT / "training-regulation.json").itos["species"])
    R.train(argparse.Namespace(members=stage1, version="v3", epochs=1 if SMOKE else 2, lr=1e-4, batch=128,
                               embedding_steps=5 if SMOKE else 150, replace=False))
    json.dump({k: sorted(v) for k, v in lineups.items()}, open(lineups_file(arm), "w"))
    stops = stops_of(arm)
    for n, st in stops.items():
        for k, c in st.items():
            assert c, f"{n} {k}: no early-stopping record"
            print(f"  {n} {k}: best {c.get('best_pass', c.get('best_step'))} of cap {c.get('cap', c.get('cap_steps'))}, stopped {c.get('stopped_pass', c.get('stopped_step'))}", flush=True)
    out = load_out(); out["train"][arm] = dict(epochs_first=log, early_stopping=stops); save(out)


# ---------------------------------------------------------------- generation and battles
def gen():
    out = load_out()
    out.setdefault("config", dict(arms={a: list(members(a)) for a in ARMS}, asks=ASKS, seeds=SEEDS, seed0=RA.SEED0,
                                  keep=RA.KEEP, guidance=RA.GUIDANCE, sampler="plain masking", battles_per_opponent=1,
                                  slots="shuffled", script_sha256=RA.sha(__file__)))
    for arm in ARMS:
        todo = [(a, s) for s in SEEDS for a in ASKS if tag_of(arm, a, s) not in out["cells"]]
        if not todo: continue
        w = DiscreteDiffusionModel.load(members=members(arm))
        for ask, seed in todo:
            tag = tag_of(arm, ask, seed); r = RA.draw(w, ask, 0.0, 48, seed); legal = (r["tried"] - r["rejected"]) / max(r["tried"], 1)
            out["cells"][tag] = dict(arm=arm, ask=ask, seed=seed, tried=r["tried"], rejected=r["rejected"], legal=legal,
                                     kept=len(r["kept"]), div=RA.diversity(r["kept"]) if r["kept"] else None, pastes=r["kept"])
            save(out); dv = out["cells"][tag]["div"] or {}
            print(f"  {tag:30s} legal {legal:6.1%} · line-ups {dv.get('distinct_lineups')} · effective species {dv.get('effective_species', 0):.1f}", flush=True)
        del w
        if L.DEV == "mps": torch.mps.empty_cache()


def battle():
    RA.require_shuffled(); con = MDB.connect(); out = load_out()
    for tag, c in out["cells"].items():
        if "y" in c or not c["pastes"]: continue
        sc = MDB.score_pastes(con, c["pastes"], per_cell=1, seed=RA.SEED0 + c["seed"], origin=f"{PREFIX}:{tag}")
        if not RA.complete(sc, 1): print(f"  {tag:30s} INCOMPLETE: left for the next run", flush=True); continue
        c["y"] = [s["score"] for s in sc]; save(out); print(f"  {tag:30s} battled {len(sc)}: win {np.mean(c['y']):.3f}", flush=True)


# ---------------------------------------------------------------- report
def swaps(pastes, reference):
    """Species swaps from each team's line-up to the closest reference line-up (6 minus most species shared)."""
    vocab = {s: i for i, s in enumerate(sorted({s for r in reference for s in r}))}
    train_hot = np.zeros((len(reference), len(vocab) + 1), np.int16)          # one row per training line-up
    for i, r in enumerate(reference): train_hot[i, [vocab[s] for s in r]] = 1
    teams = [[s["species"] for s in t] for t in map(slots_from_paste, pastes) if t]   # unparsed pastes are skipped
    gen_hot = np.zeros((len(teams), len(vocab) + 1), np.int16)                # one row per generated team
    for i, t in enumerate(teams): gen_hot[i, [vocab.get(s, len(vocab)) for s in t]] = 1
    gen_hot[:, len(vocab)] = 0                            # a species no training line-up has shares nothing
    return 6 - (gen_hot @ train_hot.T).max(1)


def novelty(pastes, reference):
    s = swaps(pastes, reference)
    return {"0": float((s == 0).mean()), "1": float((s == 1).mean()), "2": float((s == 2).mean()), "3+": float((s >= 3).mean()),
            "teams": len(s), "unparsed": int(len(pastes) - len(s))}


def mid_cells(cells, arm):
    return [c for c in cells if c["arm"] == arm and any(abs(c["ask"] - a) < 1e-4 for a in POOLED_ASKS) and c.get("y")]


def diff(a, b):
    m = a.mean() - b.mean(); se = math.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return dict(mean=float(m), lo=float(m - 1.96 * se), hi=float(m + 1.96 * se))


def report():
    out = load_out(); cells = list(out["cells"].values())
    bjson = json.load(open(BASELINE)) if BASELINE.exists() else {"cells": {}, "train": {}}
    base = [c for c in bjson["cells"].values() if c["arm"] == "control"]
    stops = {a: list(out["train"][a]["early_stopping"].values()) for a in ARMS}
    if "control" in bjson["train"]: stops["control"] = list(bjson["train"]["control"]["early_stopping"].values())
    for a in stops:   # members in PARENTS order; the stopping split must be the baseline's
        for k in ("stage1", "stage2"):
            got = [(m[k]["split"], m[k]["rows"]) for m in stops[a]]
            want = [(m[k]["split"], m[k]["rows"]) for m in stops.get("control", stops[a])]
            assert got == want, f"{a} {k}: stopping split {got} differs from Every row's {want}"
    need = sum(any(abs(a - m) < 1e-4 for a in ASKS) for m in POOLED_ASKS) * len(SEEDS)   # smoke: one ask, one seed
    ref = {a: json.load(open(lineups_file(a))) for a in ARMS}
    baseline_ref = [tuple(r) for r in ref[ARMS[0]]["all"]]           # every arm's stages see the baseline's rows
    assert all(sorted(map(tuple, ref[a]["all"])) == sorted(baseline_ref) for a in ARMS), "arms saw different full row sets"
    rows = {"control": (mid_cells(base, "control"), baseline_ref, baseline_ref)}
    for a in ARMS: rows[a] = (mid_cells(cells, a), [tuple(r) for r in ref[a]["kept"]], baseline_ref)
    summ = {}
    yc = np.array([v for c in rows["control"][0] for v in c["y"] if v is not None], float)
    for arm, (cs, own, bref) in rows.items():
        if len(cs) < need: raise SystemExit(f"{NAMES[arm]}: incomplete ({len(cs)}/{need} cells), no report")
        y = np.array([v for c in cs for v in c["y"] if v is not None], float)
        pastes = [p for c in cs for p in c["pastes"]]
        dv = {k: float(np.mean([c["div"][k] for c in cs])) for k in ("distinct_lineups", "effective_species", "mean_distance") if k in cs[0]["div"]}
        s = dict(name=NAMES[arm], win=float(y.mean()), n_teams=len(y), diversity=dv,
                 novelty_own=novelty(pastes, own), novelty_vs_baseline=novelty(pastes, bref),
                 training_lineups=len(own), win_vs_baseline=diff(y, yc) if arm != "control" and len(yc) else None,
                 stopped=[{k: dict(best=m[k].get("best_pass", m[k].get("best_step")), cap=m[k].get("cap", m[k].get("cap_steps")))
                           for k in ("stage1", "stage2")} for m in stops.get(arm, [])])
        summ[arm] = s
        w = s["win_vs_baseline"]
        print(f"{NAMES[arm]}: win {s['win']:.3f}" + (f" ({w['mean']:+.3f} [{w['lo']:+.3f}, {w['hi']:+.3f}] vs Every row)" if w else "")
              + " · " + " · ".join(f"{k} {v:.1f}" for k, v in dv.items()))
        fmt = lambda nv: " · ".join(f"{k} swaps {nv[k]:.1%}" for k in ("0", "1", "2", "3+")) + f" ({nv['unparsed']} unparsed)"
        print("  best checkpoint per member (stage 1 epoch, stage 2 step): " + " · ".join(
            f"{m['stage1']['best']}/{m['stage1']['cap']}, {m['stage2']['best']}/{m['stage2']['cap']}" for m in s["stopped"]))
        print(f"  novelty vs own training line-ups ({len(own)}): " + fmt(s["novelty_own"]))
        print("  novelty vs Every row's line-ups:  " + fmt(s["novelty_vs_baseline"]))
    out["summary"] = dict(arms=summ, asks_pooled=list(POOLED_ASKS), note="one training seed per arm: provisional"); save(out)
    print("LUR_REPORT_DONE")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "train": train(sys.argv[2])
    else: {"gen": gen, "battle": battle, "report": report}[cmd]()
