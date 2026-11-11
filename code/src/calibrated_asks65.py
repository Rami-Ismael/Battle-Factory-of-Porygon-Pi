"""Calibrated asks for the 1 to 39 opponents arm (docs/calibrated-asks-subset-labels.md).

Suspect: a label drawn from a few opponents reads high more often because an ordinary team got lucky on its subset,
so the teams the model learned for a high label are weaker than the label says. Test: for each target k/24, ask for
the label value whose training teams average a k/24 win rate against all 39 training opponents, then battle as the
ask grid did and judge got against the target.

    python src/calibrated_asks65.py calibrate     map each target to its label value (no model, no battles)
    python src/calibrated_asks65.py gen           64 legal teams per target and seed, seeds 1 to 3
    python src/calibrated_asks65.py battle        one battle per meta team, slots shuffled
    python src/calibrated_asks65.py split         got against the training opponents; verdict against the original grids
CA65_SMOKE=1 runs one target and seed with 4 teams into results/calibrated_asks_smoke.json.
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("AG84_MODEL", "gen65_subset_1_39")
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import train65 as T65                                    # puts /tmp/vgc-pilot/src on the path
import lossdata65 as D

SMOKE = bool(os.environ.get("CA65_SMOKE"))
ARM, LO, HI = "subset_1_39", 1, 39
TARGETS = [8] if SMOKE else list(range(3, 17))           # 17/24 and up: no label value's teams average that high
SEEDS = [1] if SMOKE else [1, 2, 3]
EPOCHS, STEP, HALF_WINDOW, MIN_ROW_EPOCHS = 80, 192, 2, 200
BAR = 0.001                                              # smallest error difference worth acting on (pre-registration)
OUT = T65.RES / ("calibrated_asks_smoke.json" if SMOKE else f"calibrated_asks_gen65_{ARM}.json")


def load_out():
    return json.load(open(OUT)) if OUT.exists() else dict(arm=ARM, cells={})


def save(out):
    tmp = OUT.with_suffix(".tmp"); json.dump(out, open(tmp, "w"), indent=1); tmp.replace(OUT)


def label_curve():
    """Mean all-39 label of training rows at each subset-label value, over the 80 epochs of draws the arm trained on."""
    d, tr, _, sha = T65.load(); C = d["C"][tr].astype(np.int64); full = T65.full_label(C).astype(np.float64)
    grid = np.arange(STEP + 1) / STEP; s = np.zeros(len(grid)); n = np.zeros(len(grid))
    for p in range(EPOCHS):
        lab = D.subset_labels(C, np.random.default_rng(T65.SUBSET_SEED + p), LO, HI)[1]
        i = np.clip(np.rint(lab * STEP).astype(int), 0, STEP)
        np.add.at(s, i, full); np.add.at(n, i, 1)
    k = np.ones(2 * HALF_WINDOW + 1)
    S, N = np.convolve(s, k, "same"), np.convolve(n, k, "same")
    return grid, np.where(N > 0, S / np.maximum(N, 1), np.nan), N, sha


def calibrate():
    grid, mean, N, sha = label_curve()
    ok = N >= MIN_ROW_EPOCHS
    rising = np.maximum.accumulate(np.where(ok, mean, -1.0))   # smallest label whose teams reach the target
    asks = {}
    for k in range(1, 24):
        hit = np.flatnonzero(ok & (rising >= k / 24))
        asks[str(k)] = None if not len(hit) else dict(label=float(grid[hit[0]]), teams_per_epoch=float(N[hit[0]] / EPOCHS))
    out = load_out()
    if out.get("cells") and out.get("calibration", {}).get("asks") not in (None, asks):
        raise SystemExit("the calibration changed after teams were drawn; move the results file first")
    out["calibration"] = dict(asks=asks, data_sha256=sha, epochs=EPOCHS, step=f"1/{STEP}", window=f"±{HALF_WINDOW}/{STEP}",
                              min_row_epochs=MIN_ROW_EPOCHS, highest_mean=float(np.nanmax(np.where(ok, mean, np.nan))))
    save(out)
    for k in range(1, 24):
        a = asks[str(k)]
        print(f"  target {k:2d}/24 -> " + (f"label {a['label']:.4f} ({a['teams_per_epoch']:.0f} teams per epoch)" if a else "unreachable"))


def gen():
    import ask_grid84 as AG
    out = load_out()
    if "calibration" not in out: raise SystemExit("run calibrate first")
    sha = AG.require_same_checkpoint(out); out.update(checkpoint=AG.checkpoint_path().name, checkpoint_sha256=sha)
    if SMOKE: AG.KEEP, AG.BATCH, AG.MAX_TRIES = 4, 8, 64
    w = AG.load_model()
    for seed in SEEDS:
        for k in TARGETS:
            t = f"k{k}_s{seed}"
            if t in out["cells"]: continue
            label = out["calibration"]["asks"][str(k)]["label"]
            kept, stats, counts = AG.draw(w, label, seed)
            out["cells"][t] = dict(target=k / 24, numerator=k, ask=label, seed=seed, **counts, pastes=kept, stat_points=stats,
                                   div=AG.subset_diversity(kept, AG.SEED0 + seed) if kept else None)
            save(out); print(f"  {t:10s} ask {label:.4f} kept {len(kept):3d} legal {counts['legal']:6.1%}", flush=True)


def battle():
    import ask_grid84 as AG
    import matchup_db as MDB
    AG.require_shuffled(); con = MDB.connect(); out = load_out()
    for t, c in out["cells"].items():
        if "y" in c or not c["pastes"]: continue
        sc = MDB.score_pastes(con, c["pastes"], per_cell=AG.BATTLES_PER_OPPONENT, seed=AG.SEED0 + c["seed"], origin=f"calibrated65:{t}")
        if not AG.complete(sc): print(f"  {t} INCOMPLETE: left for the next run", flush=True); continue
        AG.record_scores(c, sc); save(out); print(f"  {t:10s} win {np.mean(c['y']):.3f}", flush=True)


def error(per_ask, ks):
    e = [(per_ask[k] - k / 24) ** 2 for k in ks if k in per_ask]
    return float(np.mean(e)) if len(e) == len(ks) else None


def split():
    import sqlite3
    out = load_out(); con = sqlite3.connect(f"file:{D.DB}?mode=ro", uri=True); sides, policy = T65.sides_and_policy(con)
    got = {}
    for c in out["cells"].values():
        for t in T65.team_ids(con, c): got.setdefault(c["numerator"], []).append(T65.got_by_side(con, policy, sides, t)["training"])
    calibrated = {k: float(np.mean(v)) for k, v in got.items()}
    ref = {a: {int(k): v["training"]["got"] for k, v in json.load(open(T65.RES / f"subset_size_labels_gen65_{a}.json"))["per_ask"].items()}
           for a in ("subset_39", ARM)}
    err = dict(calibrated=error(calibrated, TARGETS), original=error(ref[ARM], TARGETS), control=error(ref["subset_39"], TARGETS))
    if None in err.values(): verdict = "incomplete"
    elif err["calibrated"] - err["control"] <= BAR: verdict = "explains: calibrated error within 0.001 of the control's"
    elif err["original"] - err["calibrated"] >= BAR: verdict = "partly: calibrated error fell by 0.001 or more but stays above the control's"
    else: verdict = "refuted: calibration did not lower the error by 0.001"
    out["result"] = dict(asks=f"{TARGETS[0]}/24 to {TARGETS[-1]}/24", got_minus_target={k: calibrated[k] - k / 24 for k in sorted(calibrated)},
                         error=err, verdict=verdict, teams=sum(len(v) for v in got.values()))
    save(out); print(json.dumps(out["result"], indent=1))


if __name__ == "__main__":
    {"calibrate": calibrate, "gen": gen, "battle": battle, "split": split}[sys.argv[1]]()
