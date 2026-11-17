"""Preflight probes of "No tables in the loss" (issue #361; docs/no-tables-in-the-loss.md), saved under "preflight" in
results/no_tables_in_the_loss.json. No battles: the experiment measures first-try legality only.

    python src/no_tables_in_the_loss_preflight.py
"""
import json
import math
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, "/tmp/vgc-pilot/src")
sys.path.insert(0, str(HERE))
import lossdown as L                                     # before diffusion84: its imports put /tmp/vgc-pilot/src first
sys.path.insert(0, str(HERE))
import numpy as np
import torch

import asked_vs_got as X
import diffusion84 as M
import epoch_legality as E
import team84 as T
from corpus import parse_team_text
from hps_generate import Validator

CONTROL = "stat84_shuffled_noise_linear_es3_cap80"
OUT = X.RES / "no_tables_in_the_loss.json"
SMALLEST_EFFECT = 0.125                                  # two standard errors of the share at 64 teams and p = 0.5


def training_rows():
    d = np.load(X.RES / "lossdata84_shuffled_only.npz")
    has = ~np.isnan(d["all_W"]); tr = ~d["all_val"] & has
    return d["all_X"][tr].astype(np.int64), d["all_W"][tr]


def epoch_model(epoch, corpus):
    return M.Model84.load(L.CK / CONTROL / f"{CONTROL}_ep{epoch:02d}.pt", corpus)


def check(model, tables, known=None, teams=E.TEAMS):
    c = E.EpochLegality(teams, tables=tables, known=known)
    try: return c(model)
    finally: c.close()


def ceiling(model, rows, n=64):
    """The perfect input: training teams through the same paste and validator. Pass: illegal share at least
    SMALLEST_EFFECT below the control's at epoch 80 without the tables."""
    v = Validator()
    try: bad = [v(model.to_paste(r, repair=True)[0]) for r in rows[:n]]
    finally: v.close()
    return dict(teams=n, illegal_share=sum(e is not None for e in bad) / n, errors=[e for e in bad if e][:3])


def round_trip(model, rows, n=20):
    """Encode, paste, parse and encode again: every one of the 84 generated fields comes back, slot and move order aside."""
    same = 0
    for r in rows[:n]:
        back = T.encode(model.V, parse_team_text(model.to_paste(r)[0]))
        same += E.team_key(back) == E.team_key(r)
    return dict(teams=n, identical=same, passed=same == n)


def init_loss(V):
    """Plain per-field cross-entropy at initialization with every field hidden and no tables, against ln K."""
    torch.manual_seed(0)
    net = M.Net84(V, **dict(d=256, nlayer=6, nhead=8)).to(M.DEV).eval()
    x = torch.as_tensor(training_rows()[0][:256], device=M.DEV)
    with torch.no_grad():
        lg = net.logits(net(torch.zeros_like(x), torch.ones(len(x), device=M.DEV), torch.full((len(x),), float("nan"), device=M.DEV)))
        ce = M.per_column_ce(lg, x).mean(0).cpu().numpy()
    out = {}
    for k, cols in L.COLS_BY_KEY.items(): out[k] = dict(ce=float(ce[cols].mean()), ln_k=math.log(V.sizes[k]))
    out["stat"] = dict(ce=float(ce[M.STAT0:].mean()), ln_k=math.log(M.SP.NVALUES))
    return dict(fields=out, passed=all(abs(v["ce"] - v["ln_k"]) < 0.5 for v in out.values()))


def overfit(V, corpus, rows, labels):
    """A small network trained without the loss mask on 32 teams for 300 steps: the loss on those teams, with fixed masks,
    falls below 60% of its first value. (40 steps at lr 4e-4 barely moved it, 4.288 to 4.274; too short to show anything.)"""
    m = M.Model84.train_new(V, corpus, rows[:32], labels[:32], epochs=300, batch=32, d=64, nlayer=2, nhead=2, lr=3e-3,
                            warmup=20, ema=0, table_loss=False, val=(rows[:32], labels[:32]))
    first, last = m.info["curves"][0]["val_loss"], m.info["curves"][-1]["val_loss"]
    return dict(steps=300, first_loss=first, last_loss=last, passed=last < 0.6 * first)


def main():
    t0 = time.perf_counter()
    V, corpus = T.regulation_vocab(), X.setup()[0]
    rows, labels = training_rows(); known = E.team_keys(rows)
    pre = {}
    last = epoch_model(80, corpus)
    pre["control_tables_off"] = {ep: check(epoch_model(ep, corpus) if ep != 80 else last, False, known) for ep in (1, 40, 80)}
    pre["control_tables_on_epoch80"] = check(last, True, known)
    pre["repeat_epoch80_tables_off"] = check(last, False, known)
    pre["repeat_identical"] = pre["repeat_epoch80_tables_off"]["illegal_share"] == pre["control_tables_off"][80]["illegal_share"] \
        and pre["repeat_epoch80_tables_off"]["errors_per_team"] == pre["control_tables_off"][80]["errors_per_team"]
    pre["ceiling"] = ceiling(last, rows)
    pre["ceiling"]["passed"] = pre["ceiling"]["illegal_share"] <= pre["control_tables_off"][80]["illegal_share"] - SMALLEST_EFFECT
    pre["round_trip"] = round_trip(last, rows)
    pre["coverage"] = dict(vocabulary_sizes=V.sizes, training_teams=len(rows), dropped_by_tables=last.info.get("dropped_by_tables"),
                           why_dropped="rows whose true value the tables exclude are dropped before training, in both arms")
    torch.manual_seed(0)
    drawn = last.sample(5, ask=E.ASK, guidance=E.GUIDANCE, seed=E.SEED, tables=False).cpu().numpy()
    pre["read_teams_epoch80_tables_off"] = [last.to_paste(r, repair=True)[0] for r in drawn[:3]]
    pre["init_loss"] = init_loss(V)
    pre["overfit"] = overfit(V, corpus, rows, labels)
    pre["minutes"] = round((time.perf_counter() - t0) / 60, 1)
    result = json.loads(OUT.read_text()) if OUT.exists() else {}
    result["preflight"] = pre
    OUT.write_text(json.dumps(result, indent=1, default=float))
    for k, v in pre.items(): print(k, json.dumps(v, default=float)[:400])
    print("PREFLIGHT_DONE")


if __name__ == "__main__":
    main()
