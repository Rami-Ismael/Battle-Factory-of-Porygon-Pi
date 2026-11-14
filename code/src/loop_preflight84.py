"""Battle-free probes for the four loop arms, run before any of their trainings (docs/looped-84-fields.md).

  training_smoke   for the loop block and the carried state: cross-entropy of a fresh network with every field hidden
                   against ln K for each field type, then 32 teams overfit for 300 epochs, beside the plain network on
                   the same teams and seed, all under the linear masking schedule. Pass: every field type within 20% of ln K, validation loss falls by 40% or
                   more, and ends within 10% of the plain network's (position_split_preflight's bars)
  step_time        mean seconds per training step at batch 256 on 2,048 training teams, after 12 untimed steps, for the plain network and both
                   options, against the multiply-add ratio that sets the equal-compute caps. Reported, no pass bar:
                   another session may be training on the same GPU
  sampling_time    seconds to sample 64 teams at ask 0.5, guidance 2, for each fresh network. Reported
  control          the control checkpoint: its run info (table, epochs, best validation loss) and that its network
                   has neither option

Results: ~/vgc-data/results/loop_arms_preflight.json. Every training probe uses the linear masking schedule of the loop arms;
the per-field results of the first run, 2026-11-12, are kept under the keys ending _per_field.

    /tmp/vgc-pilot/.venv/bin/python src/loop_preflight84.py [probe ...]      (from code/; default all four)
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

import train84 as TR                                    # puts src/ and /tmp/vgc-pilot/src on sys.path, as when training
import asked_vs_got as X
import diffusion84 as M
import lossdown as L
import position_split_preflight as PP
import team84 as T
from checkpoints import load_checkpoint

OUT = Path.home() / "vgc-data/results/loop_arms_preflight.json"
CONTROL = L.CK / f"{TR.NAMES['shuffled_linear']}.pt"           # the linear masking schedule (owner, 2026-11-12)
SCHED = TR.LINEAR_MASKING_RATE
OPTIONS = dict(plain={}, **TR.LOOP_NETS)


def rows(d, n):
    labelled = np.flatnonzero(~d["all_val"] & ~np.isnan(d["all_W"]))[:n]
    return d["all_X"][labelled], d["all_W"][labelled]


def training_smoke(d, V, corpus):
    x, w = rows(d, 256)
    out = dict(calibration=PP.CALIBRATION, device=M.DEV)
    for name, kw in OPTIONS.items():
        torch.manual_seed(TR.RECIPE["seed"]); net = M.Net84(V, **TR.NET, **kw).eval()
        start = PP.start_loss(V, net, torch.as_tensor(x))
        t0 = time.perf_counter()
        m = M.Model84.train_new(V, corpus, x[:32], w[:32], epochs=PP.OVERFIT_EPOCHS, batch=32, warmup=10, val=(x[:32], w[:32]),
                                label_noise=1 / 48, seed=TR.RECIPE["seed"], lr=TR.RECIPE["lr"], sched=SCHED, **TR.NET, **kw)
        c = m.info["curves"]
        out[name] = dict(start=start, first_validation_loss=c[0]["val_loss"], last_validation_loss=c[-1]["val_loss"],
                         first_training_loss=c[0]["train_loss"], last_training_loss=c[-1]["train_loss"],
                         minutes=round((time.perf_counter() - t0) / 60, 2))
        out[name]["passed"] = (all(v["passed"] for v in start.values())
                               and c[-1]["val_loss"] <= (1 - PP.OVERFIT_FALL) * c[0]["val_loss"])
    plain = out["plain"]["last_validation_loss"]
    for name in TR.LOOP_NETS:
        out[name]["within_10_percent_of_plain"] = out[name]["last_validation_loss"] <= (1 + PP.NO_SPLIT_MARGIN) * plain
        out[name]["passed"] = out[name]["passed"] and out[name]["within_10_percent_of_plain"]
    out["passed"] = all(out[k]["passed"] for k in OPTIONS)
    return out


def step_time(d, V, corpus, steps=24, warm=12):
    x, w = rows(d, 2048)
    tables = L.Tables(V, corpus)
    xt, wt = torch.as_tensor(x).to(M.DEV), torch.as_tensor(w.astype(np.float32)).to(M.DEV)
    sw = M.sched84(SCHED).to(M.DEV)
    base = M.training_macs(V, **TR.NET)
    out = dict(device=M.DEV, batch=256, steps=steps, warm_up_steps=warm)
    for name, kw in OPTIONS.items():
        torch.manual_seed(0); net = M.Net84(V, **TR.NET, **kw).to(M.DEV).train()
        opt = torch.optim.AdamW(net.parameters(), lr=1e-4)
        times = []
        for s in range(steps + warm):                      # untimed warm-up: the GPU is slow the first time it meets each loop count
            b = torch.randint(0, len(xt), (256,), device=M.DEV)
            if M.DEV == "mps": torch.mps.synchronize()
            t0 = time.perf_counter()
            loss = M.batch_loss(net, tables, xt[b], wt[b], sw); opt.zero_grad(); loss.backward(); opt.step()
            if M.DEV == "mps": torch.mps.synchronize()
            if s >= warm: times.append(time.perf_counter() - t0)
        out[name] = dict(seconds_per_step=float(np.mean(times)), multiply_add_ratio=M.training_macs(V, **TR.NET, **kw) / base)
    for name in OPTIONS:
        out[name]["time_ratio"] = out[name]["seconds_per_step"] / out["plain"]["seconds_per_step"]
    out["passed"] = True
    return out


def sampling_time(V, corpus):
    out = dict(device=M.DEV, teams=64, ask=0.5, guidance=2.0)
    for name, kw in OPTIONS.items():
        torch.manual_seed(0); net = M.Net84(V, **TR.NET, **kw).to(M.DEV).eval()
        m = M.Model84(V, corpus, [(net, dict(TR.NET, **kw))])
        m.sample(8, ask=0.5, guidance=2.0, seed=1)
        t0 = time.perf_counter(); r = m.sample(64, ask=0.5, guidance=2.0, seed=1)
        out[name] = dict(seconds=time.perf_counter() - t0, pass_multiply_add_ratio=M.pass_macs(V, **TR.NET, **kw) / M.pass_macs(V, **TR.NET),
                         every_field_filled=bool((r > 0).all()))
    for name in OPTIONS:
        out[name]["time_ratio"] = out[name]["seconds"] / out["plain"]["seconds"]
    out["passed"] = all(out[k]["every_field_filled"] for k in OPTIONS)
    return out


def control():
    ck = load_checkpoint(CONTROL, map_location="cpu")
    info, kw = ck["info"], ck["members"][0]["net_kw"]
    plain = not kw.get("loop_layers") and not kw.get("carry")
    return dict(checkpoint=CONTROL.name, table=info.get("table"), data_sha256=info.get("data_sha256"), teams_trained=info.get("n_train"),
                best_epoch=info.get("best_epoch"), stopped_epoch=info.get("stopped_epoch"), cap=info.get("cap"),
                best_validation_loss=info.get("best_val_loss"), minutes=info.get("minutes"),
                early_stopping_threshold=info.get("early_stopping_threshold"), label_noise=info.get("label_noise"), net_kw=kw,
                passed=plain and info.get("table") == TR.SHUFFLED_ONLY.name and info.get("cap") == TR.CONTROL_EPOCHS)


def main(names):
    results = json.load(open(OUT)) if OUT.exists() else {}
    V, corpus = T.regulation_vocab(), X.setup()[0]
    d = TR.load(TR.SHUFFLED_ONLY) if {"training_smoke", "step_time"} & set(names) else None
    if d is not None: results["table"] = dict(name=d["table"], sha256=d["sha256"])
    probes = dict(training_smoke=lambda: training_smoke(d, V, corpus), step_time=lambda: step_time(d, V, corpus),
                  sampling_time=lambda: sampling_time(V, corpus), control=control)
    for old in ("training_smoke", "step_time", "control"):            # the first run used the per-field masking schedule
        if old in results and "_per_field_note" not in results.get(old, {}) and results[old].get("masking_rate") is None:
            results[old + "_per_field"] = dict(results.pop(old), _per_field_note="first run, per-field masking schedule")
    for name in names:
        results[name] = probes[name]()
        if isinstance(results[name], dict): results[name]["masking_rate"] = SCHED
        json.dump(results, open(OUT.with_suffix(".tmp"), "w"), indent=1, default=float); OUT.with_suffix(".tmp").replace(OUT)
        print(f"[{name}] {'PASS' if results[name]['passed'] else 'FAIL'}\n{json.dumps(results[name], indent=1, default=float)}", flush=True)
    print("LOOP_PREFLIGHT_DONE", "PASS" if all(results[n]["passed"] for n in names) else "FAIL")


if __name__ == "__main__":
    main(sys.argv[1:] or ["control", "sampling_time", "step_time", "training_smoke"])
