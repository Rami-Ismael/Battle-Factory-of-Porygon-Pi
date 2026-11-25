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

Masking schedule. A new arm trains with the linear masking schedule, every field hidden with chance t (owner, 2026-11-14,
after #297 and #335 found the per-field one no better). Every arm listed here trained before that day, and those in
PER_FIELD_ARMS keep the per-field exponents 16,4,4,1.5,1 so a rerun reproduces its model.

Training table. The default is the shuffled-only table, results/lossdata84_shuffled_only.npz: every label comes from
shuffled-slot battles alone (owner, 2026-11-11; src/lossdata84.py --shuffled-only). The arms above this paragraph
reproduce models trained on the older table, results/lossdata84.npz, whose labels pool listed-order battles and run
files. They run only with --older-table, so no training reads those labels by accident.

    python src/train84.py smoke | same loop | same records | same mix | all | all_noise | byepoch | ddom_tau01 | ddom_tau03
                          | pos_per_field_s0 | pos_per_field_s1 | pos_slot_learned | pos_slot_fixed | pos_no_slot
                          | shuffled_per_field | shuffled_linear | shuffled_linear_filtered | shuffled_linear_filtered_gidd_lr
                          | shuffled_linear_no_table_loss
                          | loop_block_equal_compute | loop_block_equal_epochs | carry_state_equal_compute | carry_state_equal_epochs

The four loop arms (docs/looped-84-fields.md) use shuffled_linear's rows and recipe, the linear masking schedule where every field
is hidden with chance t (owner, 2026-11-12); their control is that model. Each
changes the network in one way: the loop block runs layers 1 and 2 (counted from 0) up to 4 times with shared weights
(LoopMDM), or the carried state hands each pass's hidden state to the next (Loopholing). "Equal epochs" trains to the
control's cap of 80. "Equal compute" stops the cap where the arm's training multiply-adds reach the control's
(diffusion84.training_macs), and sizes the learning-rate schedule to that cap.

shuffled_per_field and shuffled_linear are the two trainings of the masking-schedule rows (docs/masking-rate-rows.md; issue #296):
labelled teams of the shuffled-only table, the byepoch recipe (label noise 1/48, validation loss after every epoch, patience 3,
cap 80, one checkpoint per epoch) with early-stopping threshold 0, so any new lowest validation loss resets patience. They
differ only in the masking schedule: the per-field exponents 16,4,4,1.5,1, or the linear rate 1,1,1,1,1 where every field is
hidden with chance t.

shuffled_linear_no_table_loss is "No tables in the loss" (issue #361; docs/no-tables-in-the-loss.md): shuffled_linear's rows
and recipe with the legal Pokémon construction tables and the Stat Point budget left out of the training and validation loss.
The rows the tables reject are still dropped. Its per-epoch legality check draws without the tables, so the illegal share
shows what the network learned. Its control is shuffled_linear's 80 saved epoch checkpoints, checked the same way by
src/checkpoint_legality.py.

The five pos_ arms are the position-embedding split (docs/position-embedding-split.md; issues #78, #80 to #83, spec #233):
all teams of the shuffled-only table (results/lossdata84_shuffled_only.npz), label noise 1/48, exactly 21 epochs with the
schedule sized to a cap of 80, patience off so the epoch-21 weights are kept. They differ only in the network's position
option, or for pos_per_field_s1 only in the training seed.

--label-noise WIDTH overrides the arm's width, as 1/48 or 0.02; 0 turns noise off. Without it, all_noise, byepoch and the
ddom arms use 1/48 and the others use none. The flag adds the width to the checkpoint and results names, for example
stat84_all_teams_es3_cap80_labelnoise1over48, so no width overwrites another. `same mix --label-noise WIDTH` mixes the
two members trained at that width. To battle one, pass its checkpoint name as AG84_MODEL to ask_grid84.py.

First-try legality (issue #248). After every epoch the weights a checkpoint would save draw 64 teams at the elite cut, 0.531,
the way the ask grid draws, and Showdown's validator checks each; the epoch's curve entry in the results JSON gains the
illegal share, its top rejection reasons and the seconds the check took. The draw leaves training's random numbers alone,
so an arm trains the same model with the check on or off. --legality-teams N changes the 64; 0 turns the check off.

Experiment tracking (Weights & Biases). Every training run mirrors its hyperparameters, per-epoch learning curve and summary
to Weights & Biases (src/wandb_tracking.py); the local results JSON stays the authoritative record. Pass --no-wandb to turn
tracking off, or --wandb-mode offline / --wandb-project NAME / --wandb-group GROUP to configure it. Smoke runs disable it
unless --wandb-mode or WANDB_MODE says otherwise.
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
import lr_schedule84 as LS
import asked_vs_got as X
import wandb_tracking as WT

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
PER_FIELD_RECIPE = dict(batch=256, lr=4e-4, sched=M.PER_FIELD_SCHED, seed=0)   # the recipe of every model trained before 2026-11-14
RECIPE = dict(PER_FIELD_RECIPE, sched=M.SCHED)          # new arms: the linear masking schedule (owner, 2026-11-14)
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
LINEAR_MASKING_RATE = M.LINEAR_SCHED                     # every field hidden with chance t (MD4 Appendix J.1; LLaDA Eq. 3)
# Masking-schedule rows: code ID -> the exponents of the masking schedule in training, the only setting that differs
MASKING_RATE_ARMS = dict(shuffled_per_field=dict(sched=M.PER_FIELD_SCHED), shuffled_linear=dict(sched=LINEAR_MASKING_RATE))
NAMES.update(shuffled_per_field="stat84_shuffled_noise_per_field_es3_cap80", shuffled_linear="stat84_shuffled_noise_linear_es3_cap80")
# Issue #385: retrain the default linear checkpoint on the filtered shuffled-only table without wrong-stone teams
FILTERED_DEFAULT_ARMS = dict(shuffled_linear_filtered=dict(sched=LINEAR_MASKING_RATE))
NAMES.update(shuffled_linear_filtered="stat84_shuffled_noise_linear_filtered_es3_cap80")
# Issue #431: shuffled_linear_filtered with one shared GIDD-style learning-rate recipe (peak 5e-4, cosine to the 10% floor 5e-5)
FILTERED_DEFAULT_ARMS.update(shuffled_linear_filtered_gidd_lr=dict(sched=LINEAR_MASKING_RATE, **LS.GIDD_LR_RECIPE))
NAMES.update(shuffled_linear_filtered_gidd_lr="stat84_shuffled_noise_linear_filtered_lr5e4_min5e5_es3_cap80")
# No tables in the loss (issue #361): code ID -> what changes from shuffled_linear, the loss mask and the check's draw
TABLE_LOSS_ARMS = dict(shuffled_linear_no_table_loss=dict(table_loss=False, legality_tables=False, legality_copies=True))
NAMES.update(shuffled_linear_no_table_loss="stat84_shuffled_noise_linear_no_table_loss_es3_cap80")
CONTROL_EPOCHS = 80                                       # the cap of the loop arms' control, shuffled_linear
LOOP_BLOCK = dict(loop_start=1, loop_layers=2, loop_max=4)              # 6 needs 13 GB of GPU memory per step (docs/looped-84-fields.md)
LOOP_NETS = dict(loop_block=LOOP_BLOCK, carry_state=dict(carry=True))


def equal_compute_cap(net_kw):
    """The largest whole number of epochs whose training multiply-adds do not exceed the control's 80 epochs."""
    V = T.regulation_vocab()
    return int(CONTROL_EPOCHS * M.training_macs(V, **NET) / M.training_macs(V, **NET, **net_kw))


LOOP_ARMS = {f"{k}_{match}": dict(net=v, match=match) for k, v in LOOP_NETS.items() for match in ("equal_compute", "equal_epochs")}
NAMES.update({arm: f"stat84_shuffled_noise_{arm}" for arm in LOOP_ARMS})


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


# Arms that reproduce a model trained before the linear default; train() gives them the per-field recipe
PER_FIELD_ARMS = ("loop", "records", "all", "all_noise", "byepoch", "ddom_tau01", "ddom_tau03", *POSITION_ARMS)


def recipe_for(arm, net_kw):
    """The training recipe of arm, with the seed, masking schedule and learning-rate fields (lr_schedule84.GIDD_LR_RECIPE's
    keys) taken out of net_kw when the arm sets them."""
    base = PER_FIELD_RECIPE if arm in PER_FIELD_ARMS else RECIPE
    lr_fields = {k: net_kw.pop(k) for k in LS.GIDD_LR_RECIPE if k in net_kw}
    return dict(base, seed=net_kw.pop("seed", base["seed"]), sched=net_kw.pop("sched", base["sched"]), **lr_fields)


def train(arm, rows, labels, val, epochs, patience=None, **net_kw):
    corpus = X.setup()[0]; V = T.regulation_vocab()
    if NOISE is not None: net_kw["label_noise"] = float(NOISE)
    if SMOKE: rows, labels = small(rows, labels); val = tuple(small(*val, n=256)); epochs = 1
    print(f"[{name(arm)}] {len(rows)} teams · {len(val[0])} validation teams · cap {epochs} epochs"
          + (f" · patience {patience}" if patience else " · fixed count"), flush=True)
    t0 = time.perf_counter()
    out = CK / f"{name(arm)}.pt"
    on_best = (lambda m, _pass: stamped(m, arm).save(out)) if patience else None
    tau = net_kw.pop("ddom_tau", None)
    if tau: net_kw["row_weights"] = bin_weights(labels, tau)
    if net_kw.pop("every_epoch", False):
        (CK / name(arm)).mkdir(exist_ok=True)
        net_kw["on_epoch"] = lambda m, epoch: stamped(m, arm).save(CK / name(arm) / f"{name(arm)}_ep{epoch:02d}.pt")
    tracker = net_kw.pop("tracker", None)
    recipe = recipe_for(arm, net_kw)
    labelled_only = not np.isnan(np.asarray(labels, float)).any()
    if tracker is None:
        run_meta = dict(arm=arm, table=DATA_TABLE, data_sha256=DATA_SHA, ddom_tau=tau, labelled_only=labelled_only,
                        val_rows=len(val[0]), val_share=0.02, val_seed=0)
        tracker = WT.from_cli_args(WANDB_ARGS, name=name(arm), job_type="train", tags=("84-field", arm), config=run_meta, smoke=SMOKE)
    with tracker:
        m = M.Model84.train_new(V, corpus, rows, labels, epochs=epochs, val=val, patience=patience, on_best=on_best,
                                legality_teams=LEGALITY_TEAMS, tracker=tracker, **recipe, **NET, **net_kw)
        m.info.update(table=DATA_TABLE, ddom_tau=tau, labelled_only=labelled_only, arm=arm, data_sha256=DATA_SHA,
                      minutes=round((time.perf_counter() - t0) / 60, 1), val_rows=len(val[0]), val_share=0.02, val_seed=0)
        if meta := tracker.metadata():
            m.info["wandb"] = meta
        tracker.log_summary(m.info, stop_reason=stop_reason(m.info), checkpoint_name=out.name)
    m.save(out)
    with open(X.RES / f"{name(arm)}.json", "w") as f:
        json.dump(m.info, f, indent=1, default=float)
    print(f"[{name(arm)}] done in {m.info['minutes']} min; best epoch {m.info['best_epoch']}, stopped {m.info['stopped_epoch']}"
          f" ({stop_reason(m.info)}) -> {out}", flush=True)
    return m


def stop_reason(info):
    """Why a training ended, from its run info."""
    if info.get("patience") and info["stopped_epoch"] < info["cap"]:
        return f"no new lowest validation loss for {info['patience']} epochs after epoch {info['best_epoch']}"
    if info.get("stop_after") and info["stopped_epoch"] == info["stop_after"] < info["cap"]:
        return f"planned stop at epoch {info['stop_after']}"
    return f"reached the cap of {info['cap']} epochs"


def stamped(model, arm):
    """The model with its training table written into its run info, so a checkpoint saved during training, the best one
    so far or one epoch's, names the table it trained on. The ask grid reads that name to pick the right training teams."""
    model.info.update(table=DATA_TABLE, data_sha256=DATA_SHA, arm=arm)
    return model


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


def train_masking_rate_arms(arm):
    """The masking-schedule training named by arm, on the labelled teams of the shuffled-only table. Both arms get the same rows
    and the default model's recipe with early-stopping threshold 0 and every epoch saved; only sched differs."""
    global DATA_SHA, DATA_TABLE
    if arm not in MASKING_RATE_ARMS: return
    d = load(SHUFFLED_ONLY); check_rows(d); DATA_SHA, DATA_TABLE = d["sha256"], d["table"]
    has = ~np.isnan(d["all_W"]); tr = ~d["all_val"] & has; va = d["all_val"] & has
    train(arm, d["all_X"][tr], d["all_W"][tr], (d["all_X"][va], d["all_W"][va]), epochs=80, patience=3,
          early_stopping_threshold=0.0, label_noise=1 / 48, every_epoch=True, **MASKING_RATE_ARMS[arm])


def assert_no_wrong_stones(V, train_rows, val_rows):
    """Verify with every field visible that neither split holds a wrong-stone (tables-rejected) team (issue #385)."""
    tables = L.Tables(V, [])
    train_bad = int(tables.rejects(np.asarray(train_rows, np.int64)[:, :T.COLS48]).sum())
    val_bad = int(tables.rejects(np.asarray(val_rows, np.int64)[:, :T.COLS48]).sum())
    if train_bad or val_bad:
        raise SystemExit(
            f"wrong-stone check failed (every field visible): {train_bad} training and {val_bad} validation teams rejected"
        )
    return dict(method="Tables.rejects (every field visible)", train_rows=int(len(train_rows)),
                val_rows=int(len(val_rows)), train_rejected=train_bad, val_rejected=val_bad)


def train_filtered_default_arms(arm):
    """Default-checkpoint retrain on the filtered shuffled-only table without wrong-stone teams (issue #385).
    Same recipe as shuffled_linear; refuses to start if Tables.rejects flags any row on either side."""
    global DATA_SHA, DATA_TABLE
    if arm not in FILTERED_DEFAULT_ARMS: return
    d = load(SHUFFLED_ONLY); check_rows(d); DATA_SHA, DATA_TABLE = d["sha256"], d["table"]
    has = ~np.isnan(d["all_W"]); tr = ~d["all_val"] & has; va = d["all_val"] & has
    check = assert_no_wrong_stones(T.regulation_vocab(), d["all_X"][tr], d["all_X"][va])
    print(f"[{name(arm)}] wrong-stone check ({check['method']}): "
          f"{check['train_rejected']}/{check['train_rows']} train, {check['val_rejected']}/{check['val_rows']} val rejected",
          flush=True)
    train(arm, d["all_X"][tr], d["all_W"][tr], (d["all_X"][va], d["all_W"][va]), epochs=80, patience=3,
          early_stopping_threshold=0.0, label_noise=1 / 48, every_epoch=True, **FILTERED_DEFAULT_ARMS[arm])


def train_table_loss_arms(arm):
    """No tables in the loss: shuffled_linear's rows and call with the loss mask off and the legality check drawing without the tables."""
    global DATA_SHA, DATA_TABLE
    if arm not in TABLE_LOSS_ARMS: return
    d = load(SHUFFLED_ONLY); check_rows(d); DATA_SHA, DATA_TABLE = d["sha256"], d["table"]
    has = ~np.isnan(d["all_W"]); tr = ~d["all_val"] & has; va = d["all_val"] & has
    train(arm, d["all_X"][tr], d["all_W"][tr], (d["all_X"][va], d["all_W"][va]), epochs=80, patience=3,
          early_stopping_threshold=0.0, label_noise=1 / 48, every_epoch=True, **MASKING_RATE_ARMS["shuffled_linear"], **TABLE_LOSS_ARMS[arm])


def train_loop_arms(arm):
    """The loop arm named by arm: shuffled_linear's rows and call with one network option added, and for the
    equal-compute arms a cap of equal_compute_cap epochs in place of 80."""
    global DATA_SHA, DATA_TABLE
    if arm not in LOOP_ARMS: return
    d = load(SHUFFLED_ONLY); check_rows(d); DATA_SHA, DATA_TABLE = d["sha256"], d["table"]
    has = ~np.isnan(d["all_W"]); tr = ~d["all_val"] & has; va = d["all_val"] & has
    net = LOOP_ARMS[arm]["net"]
    cap = CONTROL_EPOCHS if LOOP_ARMS[arm]["match"] == "equal_epochs" else equal_compute_cap(net)
    train(arm, d["all_X"][tr], d["all_W"][tr], (d["all_X"][va], d["all_W"][va]), epochs=cap, patience=3,
          early_stopping_threshold=0.0, label_noise=1 / 48, every_epoch=True, **MASKING_RATE_ARMS["shuffled_linear"], **net)


def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("arm", nargs="+")
    p.add_argument("--label-noise", type=noise_width, metavar="WIDTH", help="override the arm's label noise width")
    p.add_argument("--older-table", action="store_true", help="allow an arm that trains on the older table, lossdata84.npz")
    p.add_argument("--legality-teams", type=int, default=64, metavar="N",
                   help="teams the validator checks after every epoch (default 64); 0 turns the check off")
    WT.add_cli_args(p)
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
    global DATA_SHA, DATA_TABLE, NOISE, LEGALITY_TEAMS, WANDB_ARGS
    args = parse_args(argv); NOISE = args.label_noise; LEGALITY_TEAMS = args.legality_teams
    prev_wandb = WANDB_ARGS; WANDB_ARGS = args
    try:
        _run_main(args)
    finally:
        WANDB_ARGS = prev_wandb


def _run_main(args):
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
    train_masking_rate_arms(arm)
    train_filtered_default_arms(arm)
    train_table_loss_arms(arm)
    train_loop_arms(arm)
    if arm not in ("same loop", "same records", "same mix", "all", "all_noise", "byepoch", "ddom_tau01", "ddom_tau03", "smoke",
                   *POSITION_ARMS, *MASKING_RATE_ARMS, *FILTERED_DEFAULT_ARMS, *TABLE_LOSS_ARMS, *LOOP_ARMS):
        sys.exit(__doc__)


DATA_SHA = DATA_TABLE = None
NOISE = None                                             # set by --label-noise; None keeps each arm's own width and name
LEGALITY_TEAMS = 64                                      # set by --legality-teams; teams checked after every epoch (issue #248)
WANDB_ARGS = None                                        # set by main(); None keeps W&B disabled for direct unit-test calls

if __name__ == "__main__":
    main(sys.argv[1:])
