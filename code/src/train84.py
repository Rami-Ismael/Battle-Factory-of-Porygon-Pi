"""Train the two 84-field models of the Stat Points experiment (issues #59 and #60; docs/stat-points-84-fields.md).

  "Generated, same teams"  two members on the deployed members' own rows, kept teams only, their fixed recipes:
       same loop     the loop-teams member, 80 passes
       same records  the battle-records member, 60 passes, with each team's matchup row as input
     then a two-member mixture, like the deployed 48-field model. A fixed count reproduces a deployed recipe, so
     there is no early stopping; the validation curve is logged anyway on the all-teams validation rows the
     member never trains on.
  "Generated, all teams"   all   one network on every kept team, no matchup input: validation loss after every
     pass on the seeded 2%, stop after 3 passes with no new low, keep the best pass, cap 80 passes.

Labelled teams only, label noise: byepoch   to the same stopping rule, one checkpoint per epoch (docs/slope-by-epoch.md)
                                   ddom_tau01, ddom_tau03   exactly 21 epochs with the DDOM paper's weights (docs/ddom-reweighting-84-fields.md)

Both use the deployed network and recipe: d 256, 6 layers, 8 heads, batch 256, lr 4e-4, exponents 16,4,4,1.5,1 plus
1 for Stat Points, ask dropout 0.15, EMA 0.999. Data: results/lossdata84.npz (src/lossdata84.py).

    python src/train84.py smoke | same loop | same records | same mix | all | all_noise | byepoch | ddom_tau01 | ddom_tau03

--label-noise WIDTH overrides the arm's width, as 1/48 or 0.02; 0 turns noise off. Without it, all_noise, byepoch and the
ddom arms use 1/48 and the others use none. The flag adds the width to the checkpoint and results names, for example
stat84_all_teams_es3_cap80_labelnoise1over48, so no width overwrites another. `same mix --label-noise WIDTH` mixes the
two members trained at that width. To battle one, pass its checkpoint name as AG84_MODEL to ask_grid84.py.
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, "/tmp/vgc-pilot/src")
sys.path.insert(0, str(HERE))
import lossdown as L                                     # before diffusion84: its imports put /tmp/vgc-pilot/src first
sys.path.insert(0, str(HERE))
import stat_points as SP
import team84 as T
from ddom_weights import bin_weights
from label_noise import noise_tag, noise_width
import diffusion84 as M
import asked_vs_got as X

DATA = X.RES / "lossdata84.npz"
CK = L.CK
SMOKE = bool(os.environ.get("S84_SMOKE"))
NET = dict(d=256, nlayer=6, nhead=8)
RECIPE = dict(batch=256, lr=4e-4, sched=M.SCHED, seed=0)
NAMES = dict(loop="stat84_same_loop_p80", records="stat84_same_records_p60", mix="stat84_same_teams",
             all="stat84_all_teams_es3_cap80", all_noise="stat84_all_teams_noise_ep21",
             byepoch="stat84_labelled_noise_byepoch_es3_cap80",
             ddom_tau01="stat84_labelled_noise_ddom_tau01_ep21", ddom_tau03="stat84_labelled_noise_ddom_tau03_ep21")


def name(arm):
    return ("stat84smoke_" if SMOKE else "") + NAMES[arm] + ("" if NOISE is None else noise_tag(NOISE))


def load():
    d = np.load(DATA)
    d = {k: d[k].astype(np.int64) if k.endswith("_X") else d[k] for k in d.files}
    d["sha256"] = hashlib.sha256(DATA.read_bytes()).hexdigest()
    return d


def unseen_val(d, rows):
    """The all-teams validation rows that are not among `rows`."""
    seen = {r.tobytes() for r in rows}
    vx, vw = d["all_X"][d["all_val"]], d["all_W"][d["all_val"]]
    keep = np.array([r.tobytes() not in seen for r in vx], bool)
    return vx[keep], vw[keep]


def small(*arrays, n=512):
    return [a[:n] for a in arrays]


def train(arm, rows, labels, val, passes, patience=None, **net_kw):
    corpus = X.setup()[0]; V = T.regulation_vocab()
    if NOISE is not None: net_kw["label_noise"] = float(NOISE)
    if SMOKE: rows, labels = small(rows, labels); val = tuple(small(*val, n=256)); passes = 1
    print(f"[{name(arm)}] {len(rows)} teams · {len(val[0])} validation teams · cap {passes} passes"
          + (f" · patience {patience}" if patience else " · fixed count"), flush=True)
    t0 = time.perf_counter()
    out = CK / f"{name(arm)}.pt"
    on_best = (lambda m, _pass: m.save(out)) if patience else None
    tau = net_kw.pop("ddom_tau", None)
    if tau: net_kw["row_weights"] = bin_weights(labels, tau)
    if net_kw.pop("every_epoch", False):
        (CK / name(arm)).mkdir(exist_ok=True)
        net_kw["on_epoch"] = lambda m, epoch: m.save(CK / name(arm) / f"{name(arm)}_ep{epoch:02d}.pt")
    m = M.Model84.train_new(V, corpus, rows, labels, passes=passes, val=val, patience=patience, on_best=on_best,
                            **RECIPE, **NET, **net_kw)
    m.info.update(ddom_tau=tau, labelled_only=not np.isnan(np.asarray(labels, float)).any(), arm=arm, data_sha256=DATA_SHA, minutes=round((time.perf_counter() - t0) / 60, 1),
                  val_rows=len(val[0]), val_share=0.02, val_seed=0)
    m.save(out)
    json.dump(m.info, open(X.RES / f"{name(arm)}.json", "w"), indent=1, default=float)
    print(f"[{name(arm)}] done in {m.info['minutes']} min; best pass {m.info['best_pass']}, stopped {m.info['stopped_pass']}"
          f" -> {out}", flush=True)
    return m


def check_rows(d):
    for k in ("all_X", "loop_X", "records_X"):
        v = d[k][:, T.COLS48:] - 1
        assert (v >= 0).all() and (v.reshape(len(v), SP.NSLOT, SP.NSTAT).sum(-1) <= SP.MAX_TOTAL).all(), k


def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("arm", nargs="+")
    p.add_argument("--label-noise", type=noise_width, metavar="WIDTH", help="override the arm's label noise width")
    return p.parse_args(argv)


def main(argv):
    global DATA_SHA, NOISE
    args = parse_args(argv); NOISE = args.label_noise
    for mod in (M, L, T, SP):
        print(f"{mod.__name__:12s} {mod.__file__}", flush=True)
        assert Path(mod.__file__).resolve().parent == HERE, f"{mod.__name__} did not load from {HERE}"
    d = load(); DATA_SHA = d["sha256"]; check_rows(d)
    arm = " ".join(args.arm)
    if arm in ("same loop", "smoke"):
        train("loop", d["loop_X"], d["loop_W"], unseen_val(d, d["loop_X"]), passes=80)
    if arm in ("same records", "smoke"):
        train("records", d["records_X"], d["records_WM"], unseen_val(d, d["records_X"]), passes=60,
              mrow=d["records_WM"].shape[1] - 1)
    if arm in ("same mix", "smoke"):
        a, b = (M.Model84.load(CK / f"{name(k)}.pt") for k in ("loop", "records"))
        M.Model84.mixture([a, b]).save(CK / f"{name('mix')}.pt")
        print(f"[{name('mix')}] saved", flush=True)
    if arm in ("all", "smoke"):
        tr = ~d["all_val"]
        train("all", d["all_X"][tr], d["all_W"][tr], (d["all_X"][d["all_val"]], d["all_W"][d["all_val"]]),
              passes=80, patience=3)
    if arm in ("all_noise", "smoke"):                    # same recipe as the run stopped at epoch 21, plus label noise (docs/label-noise-84-fields.md)
        tr = ~d["all_val"]
        train("all_noise", d["all_X"][tr], d["all_W"][tr], (d["all_X"][d["all_val"]], d["all_W"][d["all_val"]]),
              passes=80, patience=3, label_noise=1 / 48, stop_after=21)
    if arm in ("byepoch", "ddom_tau01", "ddom_tau03"):   # labelled teams only, label noise; docs/slope-by-epoch.md, docs/ddom-reweighting-84-fields.md
        has = ~np.isnan(d["all_W"]); tr = ~d["all_val"] & has; va = d["all_val"] & has
        val = (d["all_X"][va], d["all_W"][va])
        if arm == "byepoch":
            train(arm, d["all_X"][tr], d["all_W"][tr], val, passes=80, patience=3, label_noise=1 / 48, every_epoch=True)
        else:
            train(arm, d["all_X"][tr], d["all_W"][tr], val, passes=80, label_noise=1 / 48, stop_after=21,   # no patience: exactly epoch 21, like its control
                  ddom_tau=dict(ddom_tau01=0.1, ddom_tau03=0.3)[arm])
    if arm not in ("same loop", "same records", "same mix", "all", "all_noise", "byepoch", "ddom_tau01", "ddom_tau03", "smoke"):
        sys.exit(__doc__)


DATA_SHA = None
NOISE = None                                             # set by --label-noise; None keeps each arm's own width and name

if __name__ == "__main__":
    main(sys.argv[1:])
