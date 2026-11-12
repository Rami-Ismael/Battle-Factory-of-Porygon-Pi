"""Train the two 84-field models of the Stat Points experiment (issues #59 and #60; docs/stat-points-84-fields.md).

  "Generated, same teams"  two members on the deployed members' own rows, kept teams only, their fixed recipes:
       same loop     the loop-teams member, 80 epochs
       same records  the battle-records member, 60 epochs, with each team's matchup row as input
     then a two-member mixture, like the deployed 48-field model. A fixed count reproduces a deployed recipe, so
     there is no early stopping; the validation curve is logged anyway on the all-teams validation rows the
     member never trains on.
  "Generated, all teams"   all   one network on every kept team, no matchup input: validation loss after every
     epoch on the seeded 2%, stop after 3 epochs with no new low, keep the best epoch, cap 80 epochs.

Labelled teams only, label noise: byepoch   to the same stopping rule, one checkpoint per epoch (docs/slope-by-epoch.md)
                                   ddom_tau01, ddom_tau03   exactly 21 epochs with the DDOM paper's weights (docs/ddom-reweighting-84-fields.md)

Both use the deployed network and recipe: d 256, 6 layers, 8 heads, batch 256, lr 4e-4, exponents 16,4,4,1.5,1 plus
1 for Stat Points, ask dropout 0.15, EMA 0.999.

Training table. The default is the shuffled-only table, results/lossdata84_shuffled_only.npz: every label comes from
shuffled-slot battles alone (owner, 2026-11-11; src/lossdata84.py --shuffled-only). The arms above this paragraph
reproduce models trained on the older table, results/lossdata84.npz, whose labels pool listed-order battles and run
files. They run only with --older-table, so no training reads those labels by accident.

    python src/train84.py smoke | same loop | same records | same mix | all | all_noise | byepoch | ddom_tau01 | ddom_tau03
                          | pos_per_field_s1 | pos_slot_learned | pos_slot_fixed | pos_no_slot

The five pos_ arms are the position-embedding split (docs/position-embedding-split.md; issues #78, #80 to #83, spec #233):
all teams of the shuffled-only table (results/lossdata84_shuffled_only.npz), label noise 1/48, exactly 21 epochs with the
schedule sized to a cap of 80, patience off so the epoch-21 weights are kept. They differ only in the network's position
option, or for pos_per_field_s1 only in the training seed.

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

OLDER_TABLE = X.RES / "lossdata84.npz"                  # labels pool listed-order battles and run files; only to reproduce its models
SHUFFLED_ONLY = X.RES / "lossdata84_shuffled_only.npz"   # labels from shuffled-slot battles only (issue #229)
DATA = SHUFFLED_ONLY                                     # the default training table (owner, 2026-11-11)
OLDER_TABLE_ARMS = ("same loop", "same records", "same mix", "all", "all_noise", "byepoch", "ddom_tau01", "ddom_tau03", "smoke")
OLDER_TABLE_REFUSAL = ("{arm}: this arm reproduces a model trained on the older table (lossdata84.npz), whose labels include "
                       "listed-order battles and run files. Training reads the shuffled-only table by default. "
                       "Pass --older-table to rerun this arm on the older table.")
CK = L.CK
SMOKE = bool(os.environ.get("S84_SMOKE"))
NET = dict(d=256, nlayer=6, nhead=8)
RECIPE = dict(batch=256, lr=4e-4, sched=M.SCHED, seed=0)
NAMES = dict(loop="stat84_same_loop_p80", records="stat84_same_records_p60", mix="stat84_same_teams",
             all="stat84_all_teams_es3_cap80", all_noise="stat84_all_teams_noise_ep21",
             byepoch="stat84_labelled_noise_byepoch_es3_cap80",
             ddom_tau01="stat84_labelled_noise_ddom_tau01_ep21", ddom_tau03="stat84_labelled_noise_ddom_tau03_ep21")
# Position-embedding split arms: code ID -> the network's position option and training seed
POSITION_ARMS = dict(pos_per_field_s0=dict(pos="per_field", seed=0), pos_per_field_s1=dict(pos="per_field", seed=1),
                     pos_slot_learned=dict(pos="slot_learned", seed=0),
                     pos_slot_fixed=dict(pos="slot_fixed", seed=0), pos_no_slot=dict(pos="no_slot", seed=0))
NAMES.update({arm: f"stat84_{arm}_noise_ep21" for arm in POSITION_ARMS})
CONTROL_ARM = "pos_per_field_s0"                         # No split, training seed 0: the control, with no new code


def name(arm):
    return ("stat84smoke_" if SMOKE else "") + NAMES[arm] + ("" if NOISE is None else noise_tag(NOISE))


def load(path=DATA):
    d = np.load(path)
    d = {k: d[k].astype(np.int64) if k.endswith("_X") else d[k] for k in d.files}
    d["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest(); d["table"] = path.name
    return d


def unseen_val(d, rows):
    """The all-teams validation rows that are not among `rows`."""
    seen = {r.tobytes() for r in rows}
    vx, vw = d["all_X"][d["all_val"]], d["all_W"][d["all_val"]]
    keep = np.array([r.tobytes() not in seen for r in vx], bool)
    return vx[keep], vw[keep]


def small(*arrays, n=512):
    return [a[:n] for a in arrays]


def train(arm, rows, labels, val, epochs, patience=None, **net_kw):
    corpus = X.setup()[0]; V = T.regulation_vocab()
    if NOISE is not None: net_kw["label_noise"] = float(NOISE)
    if SMOKE: rows, labels = small(rows, labels); val = tuple(small(*val, n=256)); epochs = 1
    print(f"[{name(arm)}] {len(rows)} teams · {len(val[0])} validation teams · cap {epochs} epochs"
          + (f" · patience {patience}" if patience else " · fixed count"), flush=True)
    t0 = time.perf_counter()
    out = CK / f"{name(arm)}.pt"
    on_best = (lambda m, _pass: m.save(out)) if patience else None
    tau = net_kw.pop("ddom_tau", None)
    if tau: net_kw["row_weights"] = bin_weights(labels, tau)
    if net_kw.pop("every_epoch", False):
        (CK / name(arm)).mkdir(exist_ok=True)
        net_kw["on_epoch"] = lambda m, epoch: m.save(CK / name(arm) / f"{name(arm)}_ep{epoch:02d}.pt")
    recipe = dict(RECIPE, seed=net_kw.pop("seed", RECIPE["seed"]))
    m = M.Model84.train_new(V, corpus, rows, labels, epochs=epochs, val=val, patience=patience, on_best=on_best,
                            **recipe, **NET, **net_kw)
    m.info.update(table=DATA_TABLE, ddom_tau=tau, labelled_only=not np.isnan(np.asarray(labels, float)).any(), arm=arm, data_sha256=DATA_SHA, minutes=round((time.perf_counter() - t0) / 60, 1),
                  val_rows=len(val[0]), val_share=0.02, val_seed=0)
    m.save(out)
    json.dump(m.info, open(X.RES / f"{name(arm)}.json", "w"), indent=1, default=float)
    print(f"[{name(arm)}] done in {m.info['minutes']} min; best epoch {m.info['best_epoch']}, stopped {m.info['stopped_epoch']}"
          f" -> {out}", flush=True)
    return m


def check_rows(d):
    for k in ("all_X", "loop_X", "records_X") if "loop_X" in d else ("all_X",):
        v = d[k][:, T.COLS48:] - 1
        assert (v >= 0).all() and (v.reshape(len(v), SP.NSLOT, SP.NSTAT).sum(-1) <= SP.MAX_TOTAL).all(), k


def train_position_arms(arm):
    """The position-embedding split arms named by arm (all five for smoke), on the shuffled-only table: label noise 1/48,
    patience off so exactly the epoch-21 weights are kept, and only the position option or the training seed changed."""
    global DATA_SHA, DATA_TABLE
    if arm not in (*POSITION_ARMS, "smoke"): return
    d = load(SHUFFLED_ONLY); check_rows(d); DATA_SHA, DATA_TABLE = d["sha256"], d["table"]
    tr = ~d["all_val"]
    for pos_arm, kw in POSITION_ARMS.items():
        if arm in (pos_arm, "smoke"):
            train(pos_arm, d["all_X"][tr], d["all_W"][tr], (d["all_X"][d["all_val"]], d["all_W"][d["all_val"]]),
                  epochs=80, label_noise=1 / 48, stop_after=21, **kw)


def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("arm", nargs="+")
    p.add_argument("--label-noise", type=noise_width, metavar="WIDTH", help="override the arm's label noise width")
    p.add_argument("--older-table", action="store_true", help="allow an arm that trains on the older table, lossdata84.npz")
    return p.parse_args(argv)


def older_table(arm, allowed):
    """The older table, for an arm that reproduces one of its models; None for every other arm. Such an arm stops here
    unless --older-table was passed, before any table is read or any checkpoint written."""
    global DATA_SHA, DATA_TABLE
    if arm not in OLDER_TABLE_ARMS: return None
    if not allowed: sys.exit(OLDER_TABLE_REFUSAL.format(arm=arm))
    d = load(OLDER_TABLE); DATA_SHA, DATA_TABLE = d["sha256"], d["table"]; check_rows(d)
    return d


def main(argv):
    global DATA_SHA, DATA_TABLE, NOISE
    args = parse_args(argv); NOISE = args.label_noise
    arm = " ".join(args.arm)
    d = older_table(arm, args.older_table)
    for mod in (M, L, T, SP):
        print(f"{mod.__name__:12s} {mod.__file__}", flush=True)
        assert Path(mod.__file__).resolve().parent == HERE, f"{mod.__name__} did not load from {HERE}"
    if arm in ("same loop", "smoke"):
        train("loop", d["loop_X"], d["loop_W"], unseen_val(d, d["loop_X"]), epochs=80)
    if arm in ("same records", "smoke"):
        train("records", d["records_X"], d["records_WM"], unseen_val(d, d["records_X"]), epochs=60,
              mrow=d["records_WM"].shape[1] - 1)
    if arm in ("same mix", "smoke"):
        a, b = (M.Model84.load(CK / f"{name(k)}.pt") for k in ("loop", "records"))
        M.Model84.mixture([a, b]).save(CK / f"{name('mix')}.pt")
        print(f"[{name('mix')}] saved", flush=True)
    if arm in ("all", "smoke"):
        tr = ~d["all_val"]
        train("all", d["all_X"][tr], d["all_W"][tr], (d["all_X"][d["all_val"]], d["all_W"][d["all_val"]]),
              epochs=80, patience=3)
    if arm in ("all_noise", "smoke"):                    # same recipe as the run stopped at epoch 21, plus label noise (docs/label-noise-84-fields.md)
        tr = ~d["all_val"]
        train("all_noise", d["all_X"][tr], d["all_W"][tr], (d["all_X"][d["all_val"]], d["all_W"][d["all_val"]]),
              epochs=80, patience=3, label_noise=1 / 48, stop_after=21)
    if arm in ("byepoch", "ddom_tau01", "ddom_tau03"):   # labelled teams only, label noise; docs/slope-by-epoch.md, docs/ddom-reweighting-84-fields.md
        has = ~np.isnan(d["all_W"]); tr = ~d["all_val"] & has; va = d["all_val"] & has
        val = (d["all_X"][va], d["all_W"][va])
        if arm == "byepoch":
            train(arm, d["all_X"][tr], d["all_W"][tr], val, epochs=80, patience=3, label_noise=1 / 48, every_epoch=True)
        else:
            train(arm, d["all_X"][tr], d["all_W"][tr], val, epochs=80, label_noise=1 / 48, stop_after=21,   # no patience: exactly epoch 21, like its control
                  ddom_tau=dict(ddom_tau01=0.1, ddom_tau03=0.3)[arm])
    train_position_arms(arm)
    if arm not in ("same loop", "same records", "same mix", "all", "all_noise", "byepoch", "ddom_tau01", "ddom_tau03", "smoke",
                   *POSITION_ARMS):
        sys.exit(__doc__)


DATA_SHA = DATA_TABLE = None
NOISE = None                                             # set by --label-noise; None keeps each arm's own width and name

if __name__ == "__main__":
    main(sys.argv[1:])
