"""Training-smoke probe for the five position-embedding arms, before their 21-epoch trainings (issues #81, #82; spec #233).

Per arm, on the deployed network size (d 256, 6 layers, 8 heads) and on the CPU:
  start      plain per-field cross-entropy of a fully masked team at initialisation, per field type, against ln K for a
             head with K outputs; passes within 20%
  overfit    32 training teams, 300 epochs of one batch each (warm-up 10 steps): validation loss on those same 32 teams
             (common random numbers, so less noisy than the training loss) falls by at least 40%, and ends within 10% of
             No split's, which this run probes first
  swap       swap two Pokémon in the input and undo the swap on the output: the largest change elsewhere, untrained. Only No
             slot vectors must stay at rounding (under 1e-5); the others are reported
Results land in each arm's ask-grid results file under preflight.training_smoke, beside what `ask_grid84.py preflight`
adds once the arm is trained.

    /tmp/vgc-pilot/.venv/bin/python src/position_split_preflight.py [arm ...]   (from code/; default all five)
"""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

import train84 as TR                                    # puts src/ and /tmp/vgc-pilot/src on sys.path, as when training
import asked_vs_got as X
import diffusion84 as M
import team84 as T

START_TOLERANCE, SWAP_ROUNDING, OVERFIT_EPOCHS, OVERFIT_FALL, NO_SPLIT_MARGIN = 0.20, 1e-5, 300, 0.40, 0.10
# Calibrated 2026-11-07 on the unchanged network after the first run (10% and loss halving in 60 epochs) failed No split
# itself: over five seeds No split starts 5.6% to 12.6% above ln K, and in 300 epochs its validation loss falls 3.33 -> 1.60.
CALIBRATION = ("first run 2026-11-07 used 10% and training loss halving in 60 epochs; No split itself failed both, so the bars were "
               "set from No split: 5 seeds start 5.6% to 12.6% above ln K, 300 epochs take validation loss 3.33 -> 1.60")
RES = Path.home() / "vgc-data/results"                 # as ask_grid84


def start_loss(V, net, x):
    """Mean cross-entropy per field type of the true tokens with every field masked, and ln K for that type."""
    t, w = torch.ones(len(x)), torch.full((len(x),), float("nan"))
    with torch.no_grad(): ce = M.per_column_ce(net.logits(net(torch.zeros_like(x), t, w)), x)
    out = {}
    for k, cols in [*M.L.COLS_BY_KEY.items(), ("stat", list(range(M.STAT0, M.COLS)))]:
        K = V.sizes[k] if k != "stat" else M.SP.NVALUES                # the head's outputs; near-uniform at initialisation
        got = float(ce[:, cols].mean())
        out[k] = dict(cross_entropy=got, ln_k=math.log(K), passed=abs(got - math.log(K)) <= START_TOLERANCE * math.log(K))
    return out


def swap_gap(net, x):
    slot = M.COLUMN_SLOT.clone(); slot[M.COLUMN_SLOT == 1], slot[M.COLUMN_SLOT == 4] = 4, 1
    perm = torch.tensor([int(torch.nonzero((M.COLUMN_TYPE == M.COLUMN_TYPE[c]) & (M.COLUMN_SLOT == slot[c]))[0]) for c in range(M.COLS)])
    t, w = torch.full((len(x),), 0.5), torch.full((len(x),), 0.5)
    with torch.no_grad(): return float((net(x[:, perm], t, w)[:, perm] - net(x, t, w)).abs().max())


def probe(arm, d, V, corpus, no_split_val=None):
    kw = TR.POSITION_ARMS[arm]
    tr = np.flatnonzero(~d["all_val"] & ~np.isnan(d["all_W"]))[:256]
    x, w = torch.as_tensor(d["all_X"][tr]), d["all_W"][tr]
    torch.manual_seed(kw["seed"]); net = M.Net84(V, **TR.NET, pos=kw["pos"]).eval()
    start = start_loss(V, net, x)
    gap = swap_gap(net, x[:16])
    t0 = time.perf_counter()
    m = M.Model84.train_new(V, corpus, d["all_X"][tr[:32]], w[:32], epochs=OVERFIT_EPOCHS, batch=32, warmup=10, val=(d["all_X"][tr[:32]], w[:32]),
                            label_noise=1 / 48, seed=kw["seed"], lr=TR.RECIPE["lr"], sched=TR.RECIPE["sched"], **TR.NET, pos=kw["pos"])
    c = m.info["curves"]
    overfit = dict(epochs=len(c), first_train_loss=c[0]["train_loss"], last_train_loss=c[-1]["train_loss"],
                   first_val_loss=c[0]["val_loss"], last_val_loss=c[-1]["val_loss"], minutes=round((time.perf_counter() - t0) / 60, 2))
    overfit["no_split_last_val_loss"] = no_split_val
    overfit["passed"] = (overfit["last_val_loss"] <= (1 - OVERFIT_FALL) * overfit["first_val_loss"]
                         and (no_split_val is None or overfit["last_val_loss"] <= (1 + NO_SPLIT_MARGIN) * no_split_val))
    swap = dict(largest_change=gap, required="under 1e-5" if kw["pos"] == "no_slot" else "reported only",
                passed=gap < SWAP_ROUNDING if kw["pos"] == "no_slot" else True)
    return dict(position_option=kw["pos"], training_seed=kw["seed"], device="cpu", calibration=CALIBRATION, start=start, overfit=overfit, swap=swap,
                passed=all(v["passed"] for v in start.values()) and overfit["passed"] and swap["passed"])


def save(arm, result):
    path = RES / f"stat_points_84_{arm}.json"
    out = json.load(open(path)) if path.exists() else {"model": arm, "cells": {}}
    out.setdefault("preflight", {})["training_smoke"] = result
    json.dump(out, open(path.with_suffix(".tmp"), "w"), indent=1); path.with_suffix(".tmp").replace(path)


def main(arms):
    d = TR.load(TR.SHUFFLED_ONLY); V = T.regulation_vocab(); corpus = X.setup()[0]
    ok, no_split_val = True, None
    for arm in sorted(arms, key=lambda a: a != TR.CONTROL_ARM):     # No split first: the others are judged against it
        r = probe(arm, d, V, corpus, no_split_val); save(arm, r); ok &= r["passed"]
        if arm == TR.CONTROL_ARM: no_split_val = r["overfit"]["last_val_loss"]
        start = ", ".join("%s %.2f/%.2f" % (k, v["cross_entropy"], v["ln_k"]) for k, v in r["start"].items())
        print(f"{arm}: start {'pass' if all(v['passed'] for v in r['start'].values()) else 'FAIL'} "
              f"({start}) | overfit "
              f"validation {r['overfit']['first_val_loss']:.3f} -> {r['overfit']['last_val_loss']:.3f} {'pass' if r['overfit']['passed'] else 'FAIL'} | "
              f"swap {r['swap']['largest_change']:.2e} {'pass' if r['swap']['passed'] else 'FAIL'}", flush=True)
    print("POSITION_PREFLIGHT_DONE", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main(sys.argv[1:] or list(TR.POSITION_ARMS))
