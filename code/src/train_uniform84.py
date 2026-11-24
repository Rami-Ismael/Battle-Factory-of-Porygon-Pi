"""Train the uniform discrete diffusion model (UDLM) twin of train84.py (issues #275 and #276;
docs/uniform-discrete-diffusion.md).

One model trains under the clean two-arm comparison (owner decision 2026-11-24):
  udlm_loglinear   stat84_udlm_loglinear_es3_cap80
                   Log-linear uniform noise alpha(t) = 1 - t on all 84 fields, real vocabularies
                   (347, 200, 149, 497, 26, 33; no [MASK]), continuous-time Eq. 19 loss with no
                   legal Pokémon construction filter, labelled teams of lossdata84_shuffled_only.npz,
                   label noise ±1/48, AdamW lr 4e-4, batch 256, EMA 0.999, patience 3, threshold 0,
                   cap 80 epochs, every epoch saved.

Commands:
    /tmp/vgc-pilot/.venv/bin/python code/src/train_uniform84.py smoke
    /tmp/vgc-pilot/.venv/bin/python code/src/train_uniform84.py udlm_loglinear
"""
import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, "/tmp/vgc-pilot/src")
sys.path.insert(0, str(HERE))
import lossdown as L                                     # before uniform_diffusion84: its imports put /tmp/vgc-pilot/src first
sys.path.insert(0, str(HERE))
import asked_vs_got as X
import stat_points as SP
import team84 as T
import uniform_diffusion84 as UM
import uniform_preflight84 as UP
import wandb_tracking as WT

SHUFFLED_ONLY = X.RES / "lossdata84_shuffled_only.npz"
DATA = SHUFFLED_ONLY
CK = L.CK
SMOKE = bool(os.environ.get("S84_SMOKE"))
NET = dict(d=256, nlayer=6, nhead=8)
RECIPE = dict(batch=256, lr=4e-4, wd=0.01, warmup=500, ema=0.999, seed=0)
NAMES = dict(udlm_loglinear="stat84_udlm_loglinear_es3_cap80")
ARMS = dict(
    udlm_loglinear=dict(
        epochs=80,
        patience=3,
        early_stopping_threshold=0.0,
        label_noise=1 / 48,
        every_epoch=True,
    )
)


def name(arm):
    return ("stat84smoke_" if SMOKE else "") + NAMES[arm]


def load(path=DATA):
    data = np.load(path)
    out = {key: data[key].astype(np.int64) if key.endswith("_X") else data[key] for key in data.files}
    out["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    out["table"] = path.name
    return out


def check_rows(data):
    rows = data["all_X"]
    if bool((rows < 1).any()):
        raise AssertionError("all_X contains [MASK] or negative tokens")
    stat_values = rows[:, T.COLS48:] - 1
    assert (stat_values >= 0).all()
    assert (stat_values.reshape(len(stat_values), SP.NSLOT, SP.NSTAT).sum(-1) <= SP.MAX_TOTAL).all()


def stop_reason(info):
    if info.get("patience") and info["stopped_epoch"] < info["cap"]:
        return f"no new lowest validation loss for {info['patience']} epochs after epoch {info['best_epoch']}"
    if info.get("stop_after") and info["stopped_epoch"] == info["stop_after"] < info["cap"]:
        return f"planned stop at epoch {info['stop_after']}"
    return f"reached the cap of {info['cap']} epochs"


def stamped(model, arm):
    model.info.update(table=DATA_TABLE, data_sha256=DATA_SHA, arm=arm)
    return model


def _verify_posterior_and_cfg(real_vocab):
    """Verify reverse posterior & CFG match verbatim upstream alpha-form and independent Bayes across gamma <= 16."""
    gen = torch.Generator().manual_seed(275)
    max_up_diff = 0.0
    max_bayes_diff = 0.0
    all_ok = True
    for vocab_size in real_vocab.values():
        log_cond = torch.log_softmax(torch.randn(4, 6, vocab_size, generator=gen, dtype=torch.float64), dim=-1)
        log_uncond = torch.log_softmax(torch.randn(4, 6, vocab_size, generator=gen, dtype=torch.float64), dim=-1)
        noisy = torch.randint(0, vocab_size, (4, 6), generator=gen)
        for move_t_val, move_s_val in ((1.0, 1.0 - 1.0 / 128), (0.5, 0.5 - 1.0 / 128), (1.0 / 128, 1e-5)):
            move_t = torch.full((4, 1, 1), move_t_val, dtype=torch.float64)
            move_s = torch.full((4, 1, 1), move_s_val, dtype=torch.float64)
            alpha_t, alpha_s = 1.0 - move_t, 1.0 - move_s
            cand = UM.reverse_posterior(log_cond.exp(), noisy, move_s, move_t, vocab_size)
            ref_up = UP.upstream_reverse_posterior(log_cond.exp(), noisy, alpha_s, alpha_t, vocab_size)
            ref_bayes = UP.bayes_reverse_posterior(log_cond.exp(), noisy, move_s, move_t, vocab_size)
            max_up_diff = max(max_up_diff, float((cand - ref_up).abs().max()))
            max_bayes_diff = max(max_bayes_diff, float((cand - ref_bayes).abs().max()))
            all_ok = all_ok and bool(torch.allclose(cand, ref_up, atol=1e-11, rtol=1e-11))
            all_ok = all_ok and bool(torch.allclose(cand, ref_bayes, atol=1e-11, rtol=1e-11))
            for gamma in (0.0, 1.0, 2.0, 4.0, 8.0, 16.0):
                cfg_cand = UM.cfg_posterior(log_cond, log_uncond, noisy, move_s, move_t, gamma, vocab_size)
                cfg_up = UP.upstream_cfg_posterior(log_cond, log_uncond, noisy, alpha_s, alpha_t, gamma, vocab_size)
                max_up_diff = max(max_up_diff, float((cfg_cand - cfg_up).abs().max()))
                all_ok = all_ok and bool(torch.allclose(cfg_cand, cfg_up, atol=1e-11, rtol=1e-11))
    return dict(passed=all_ok, max_upstream_diff_f64=max_up_diff, max_bayes_diff_f64=max_bayes_diff)


def verify_port_equality(vocab=None):
    """Run the required pre-training port-equality gate (loss in f64/f32 + verbatim upstream/Bayes posterior & CFG)."""
    vocab = T.regulation_vocab() if vocab is None else vocab
    real_vocab = UM.real_vocab_sizes(vocab)
    report_f64 = UP.port_equality_check(UM.port_equality_loss, real_vocab, batch_size=8, dtype=torch.float64)
    report_f32 = UP.port_equality_check(UM.port_equality_loss, real_vocab, batch_size=8, dtype=torch.float32)
    posterior_cfg = _verify_posterior_and_cfg(real_vocab)
    passed = bool(report_f64["passed"] and report_f32["passed"] and posterior_cfg["passed"])
    return dict(passed=passed, float64=report_f64, float32=report_f32, posterior_and_cfg=posterior_cfg)


def _measure_production_batch_step(vocab, sample_rows, sample_labels, seed=0):
    """Verify one full-width (d=256, 6-layer) production batch-256 forward+backward step is finite on DEV (MPS)."""
    torch.manual_seed(seed)
    net = UM.UniformNet84(vocab, **NET).to(UM.DEV).train()
    batch_size = RECIPE["batch"]
    reps = math.ceil(batch_size / len(sample_rows))
    rows_256 = np.tile(sample_rows, (reps, 1))[:batch_size]
    labels_256 = np.tile(sample_labels, reps)[:batch_size]
    clean = UM.table_to_uniform(torch.as_tensor(rows_256, dtype=torch.long, device=UM.DEV))
    labels_t = torch.as_tensor(labels_256, dtype=torch.float32, device=UM.DEV)
    loss = UM.batch_loss(net, clean, labels_t, net.vocab_sizes)
    loss_val = float(loss.detach())
    loss.backward()
    grad_norm = float(torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0))
    ok = math.isfinite(loss_val) and math.isfinite(grad_norm)
    return dict(device=str(UM.DEV), batch_size=batch_size, loss=loss_val, grad_norm=grad_norm, passed=ok)


def run_training_smoke(*, steps=60, seed=0):
    """Pre-registered #275 smoke probe: port-equality, init CE near ln N_k, 32-team overfit >= 40%, batch-256 MPS check, sampler."""
    vocab = T.regulation_vocab()
    corpus = X.setup()[0]
    real_vocab = UM.real_vocab_sizes(vocab)
    port_report = verify_port_equality(vocab)

    table = load(SHUFFLED_ONLY)
    check_rows(table)
    labelled = ~np.isnan(table["all_W"]) & ~table["all_val"]
    sample_rows = table["all_X"][labelled][:32]
    sample_labels = table["all_W"][labelled][:32].astype(np.float32)

    torch.manual_seed(seed)
    net = UM.UniformNet84(vocab, **NET).to(UM.DEV)
    clean_tokens = UM.table_to_uniform(torch.as_tensor(sample_rows, dtype=torch.long, device=UM.DEV))
    labels_tensor = torch.as_tensor(sample_labels, dtype=torch.float32, device=UM.DEV)
    init_ce = _measure_init_ce(net, clean_tokens, labels_tensor, real_vocab)
    overfit = _measure_overfit(net, clean_tokens, labels_tensor, real_vocab, steps=steps)
    prod_batch = _measure_production_batch_step(vocab, sample_rows, sample_labels, seed=seed)

    model = UM.UniformModel84(vocab, corpus, [(net.eval(), dict(NET))], info=dict(smoke=True))
    drawn = model.sample(4, ask=0.5, guidance=2.0, passes=16, seed=7)
    rev_summary = model.last_revisions.summary()
    sampler_ok = bool(
        tuple(drawn.shape) == (4, UM.COLS)
        and (drawn >= 1).all()
        and math.isfinite(rev_summary["mean_fields_changed_per_pass"])
        and math.isfinite(rev_summary["duplicate_species_pass_share"])
        and all(
            math.isfinite(rev_summary["by_field"][key]["changes_of_mind"])
            and math.isfinite(rev_summary["by_field"][key]["value_changes_beyond_first"])
            for key in UM.FIELD_KEYS
        )
    )
    passed = bool(port_report["passed"] and init_ce["passed"] and overfit["passed"] and prod_batch["passed"] and sampler_ok)
    return dict(
        passed=passed,
        port_equality=port_report,
        init_ce=init_ce,
        overfit_32_teams=overfit,
        production_batch_step=prod_batch,
        sampler_smoke=dict(passed=sampler_ok, shape=list(drawn.shape), revisions=rev_summary),
    )


@torch.no_grad()
def _measure_init_ce(net, clean_tokens, labels_tensor, real_vocab):
    """Verify untrained per-column cross-entropy sits near ln(N_k) for every field type."""
    net.eval()
    mid_time = torch.full((len(clean_tokens),), 0.5, device=UM.DEV)
    noisy = UM.corrupt_uniform(clean_tokens, mid_time, real_vocab)
    ce = UM.per_column_ce(net.logits(net(noisy, mid_time, labels_tensor)), clean_tokens)
    per_field = {}
    all_ok = True
    for field_key, cols in UM.FIELD_COLS.items():
        measured = float(ce[:, cols].mean())
        expected = math.log(real_vocab[field_key])
        ok = abs(measured - expected) < 0.35
        all_ok = all_ok and ok
        per_field[field_key] = dict(measured=measured, log_vocab=expected, abs_diff=abs(measured - expected), passed=ok)
    return dict(passed=all_ok, per_field=per_field)


def _measure_overfit(net, clean_tokens, labels_tensor, real_vocab, *, steps=60):
    """Overfit one fixed 32-team batch under fixed time/corruption and check continuous-time loss drops >= 40%."""
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=0.0, betas=(0.9, 0.98))
    fixed_times = torch.linspace(0.1, 0.9, len(clean_tokens), device=UM.DEV)
    torch.manual_seed(123)
    fixed_noisy = UM.corrupt_uniform(clean_tokens, fixed_times, real_vocab)
    net.eval()
    with torch.no_grad():
        start_loss = float(UM.elbo_terms(net, clean_tokens, fixed_noisy, fixed_times, labels_tensor, real_vocab).mean())
    net.train()
    all_finite = math.isfinite(start_loss)
    end_loss = start_loss
    for _ in range(steps):
        loss = UM.elbo_terms(net, clean_tokens, fixed_noisy, fixed_times, labels_tensor, real_vocab).mean()
        all_finite = all_finite and bool(torch.isfinite(loss))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn_grad = float(torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0))
        all_finite = all_finite and math.isfinite(nn_grad)
        optimizer.step()
        end_loss = float(loss.detach())
    net.eval()
    with torch.no_grad():
        eval_end = float(UM.elbo_terms(net, clean_tokens, fixed_noisy, fixed_times, labels_tensor, real_vocab).mean())
    drop_fraction = (start_loss - eval_end) / max(start_loss, 1e-8)
    return dict(
        steps=steps,
        start_loss=start_loss,
        last_train_loss=end_loss,
        end_loss=eval_end,
        drop_fraction=drop_fraction,
        all_finite=all_finite,
        passed=bool(all_finite and drop_fraction >= 0.40),
    )


def train(arm, rows, labels, val, epochs, patience=3, **train_kw):
    """Train one uniform discrete diffusion arm and save best checkpoint, per-epoch checkpoints, and run JSON."""
    port_report = verify_port_equality()
    if not port_report["passed"]:
        raise RuntimeError(f"port-equality gate failed before training: {port_report}")
    corpus = X.setup()[0]
    vocab = T.regulation_vocab()
    if SMOKE:
        rows, labels = rows[:512], labels[:512]
        val = (val[0][:256], val[1][:256])
        epochs = 1
    run_name = name(arm)
    print(
        f"[{run_name}] {len(rows)} teams · {len(val[0])} validation teams · cap {epochs} epochs"
        + (f" · patience {patience}" if patience else " · fixed count"),
        flush=True,
    )
    t0 = time.perf_counter()
    out_path = CK / f"{run_name}.pt"
    on_best = (lambda model, _epoch: stamped(model, arm).save(out_path)) if patience else None
    if train_kw.pop("every_epoch", False):
        (CK / run_name).mkdir(exist_ok=True)
        train_kw["on_epoch"] = lambda model, epoch: stamped(model, arm).save(
            CK / run_name / f"{run_name}_ep{epoch:02d}.pt"
        )
    tracker = train_kw.pop("tracker", None)
    labelled_only = not np.isnan(np.asarray(labels, float)).any()
    if tracker is None:
        run_meta = dict(
            arm=arm,
            noise_process="uniform",
            schedule="log_linear",
            table=DATA_TABLE,
            data_sha256=DATA_SHA,
            labelled_only=labelled_only,
            val_rows=len(val[0]),
            val_share=0.02,
            val_seed=0,
        )
        tracker = WT.from_cli_args(
            WANDB_ARGS,
            name=run_name,
            job_type="train",
            tags=("84-field", "uniform-diffusion", arm),
            config=run_meta,
            smoke=SMOKE,
        )
    with tracker:
        model = UM.UniformModel84.train_new(
            vocab,
            corpus,
            rows,
            labels,
            epochs=epochs,
            val=val,
            patience=patience,
            on_best=on_best,
            legality_teams=LEGALITY_TEAMS,
            tracker=tracker,
            **RECIPE,
            **NET,
            **train_kw,
        )
        model.info.update(
            table=DATA_TABLE,
            labelled_only=labelled_only,
            arm=arm,
            data_sha256=DATA_SHA,
            minutes=round((time.perf_counter() - t0) / 60, 1),
            val_rows=len(val[0]),
            val_share=0.02,
            val_seed=0,
        )
        if meta := tracker.metadata():
            model.info["wandb"] = meta
        tracker.log_summary(model.info, stop_reason=stop_reason(model.info), checkpoint_name=out_path.name)
    model.save(out_path)
    with open(X.RES / f"{run_name}.json", "w") as handle:
        json.dump(model.info, handle, indent=1, default=float)
    print(
        f"[{run_name}] done in {model.info['minutes']} min; best epoch {model.info['best_epoch']}, "
        f"stopped {model.info['stopped_epoch']} ({stop_reason(model.info)}) -> {out_path}",
        flush=True,
    )
    return model


def train_arm(arm):
    """Load labelled rows of the filtered shuffled-only table and train the named uniform arm."""
    global DATA_SHA, DATA_TABLE
    if arm not in ARMS:
        raise ValueError(f"unknown uniform arm {arm!r}; choose from {sorted(ARMS)}")
    table = load(SHUFFLED_ONLY)
    check_rows(table)
    DATA_SHA, DATA_TABLE = table["sha256"], table["table"]
    has_label = ~np.isnan(table["all_W"])
    train_mask = ~table["all_val"] & has_label
    val_mask = table["all_val"] & has_label
    return train(
        arm,
        table["all_X"][train_mask],
        table["all_W"][train_mask],
        (table["all_X"][val_mask], table["all_W"][val_mask]),
        **ARMS[arm],
    )


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("arm", choices=["smoke", *sorted(ARMS)])
    parser.add_argument(
        "--legality-teams",
        type=int,
        default=64,
        metavar="N",
        help="teams checked by Showdown's validator after every epoch (default 64); 0 turns the check off",
    )
    WT.add_cli_args(parser)
    return parser.parse_args(argv)


def main(argv):
    global LEGALITY_TEAMS, WANDB_ARGS
    args = parse_args(argv)
    LEGALITY_TEAMS = args.legality_teams
    prev_wandb = WANDB_ARGS
    WANDB_ARGS = args
    try:
        for mod in (UM, L, T, SP):
            print(f"{mod.__name__:20s} {mod.__file__}", flush=True)
            assert Path(mod.__file__).resolve().parent == HERE, f"{mod.__name__} did not load from {HERE}"
        if args.arm == "smoke":
            report = run_training_smoke()
            print(json.dumps(report, indent=2))
            if not report["passed"]:
                raise SystemExit("TRAIN_UNIFORM84_SMOKE FAILED")
            print("TRAIN_UNIFORM84_SMOKE PASS")
            return
        train_arm(args.arm)
    finally:
        WANDB_ARGS = prev_wandb


DATA_SHA = DATA_TABLE = None
LEGALITY_TEAMS = 64
WANDB_ARGS = None

if __name__ == "__main__":
    main(sys.argv[1:])
