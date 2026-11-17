"""First-try legality of every saved epoch checkpoint of one training (issue #361; docs/no-tables-in-the-loss.md).

Each checkpoint draws teams the way training's per-epoch check does (epoch_legality.EpochLegality: 64 teams at the elite
cut, 0.531, guidance 2, temperature 1, species-first order, Stat Alignment repaired, Showdown's validator), by default
without the legal Pokémon construction tables and the Stat Point budget, so the illegal share shows what the network
learned. This is the control line of "No tables in the loss": the default model, Tables in the loss, trained before the
check existed, and its 80 epoch checkpoints are reused instead of retrained.

    python src/checkpoint_legality.py stat84_shuffled_noise_linear_es3_cap80            # all 80 epochs, tables off
    python src/checkpoint_legality.py stat84_shuffled_noise_linear_es3_cap80 --tables   # tables on: 0 by construction

Writes results/<training>_legality_<no_tables|tables>.json after every checkpoint; a rerun skips the epochs already
written, so a stopped run resumes where it stopped. Each epoch also logs copied_share against the training teams the
checkpoint's run info names (its table, labelled rows, training split), when that table is on this laptop.
"""
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, "/tmp/vgc-pilot/src")
sys.path.insert(0, str(HERE))
import lossdown as L                                     # before diffusion84: its imports put /tmp/vgc-pilot/src first
sys.path.insert(0, str(HERE))
import asked_vs_got as X
import diffusion84 as M
import epoch_legality as E
import numpy as np
from checkpoints import load_checkpoint

EPOCH = re.compile(r"_ep(\d+)\.pt$")


def checkpoints(training, root=L.CK):
    """{epoch: path} of the training's saved epoch checkpoints."""
    folder = Path(root) / training
    found = {int(EPOCH.search(p.name).group(1)): p for p in folder.glob(f"{training}_ep*.pt") if EPOCH.search(p.name)}
    if not found: raise SystemExit(f"no epoch checkpoints in {folder}")
    return dict(sorted(found.items()))


def training_keys(info):
    """team_keys of the training rows a train84 masking-schedule run used: the labelled teams outside the validation split
    of its table. None when the info names no table or the table is missing."""
    table = info.get("table")
    if not table or not (X.RES / table).exists(): return None
    d = np.load(X.RES / table); tr = ~d["all_val"]
    if info.get("labelled_only"): tr &= ~np.isnan(d["all_W"])
    return E.team_keys(d["all_X"][tr].astype(np.int64))


def result_path(training, tables):
    return X.RES / f"{training}_legality_{'tables' if tables else 'no_tables'}.json"


def run(training, tables=False, teams=E.TEAMS, epochs=None, root=L.CK, out=None):
    """Check each epoch checkpoint not yet in the result file; returns the result dict."""
    out = Path(out or result_path(training, tables))
    result = json.loads(out.read_text()) if out.exists() else dict(
        training=training, tables=tables, teams=teams, ask=E.ASK, guidance=E.GUIDANCE, temperature=E.TEMPERATURE,
        seed=E.SEED, order="species-first", epochs=[])
    if (result["tables"], result["teams"]) != (tables, teams):
        raise SystemExit(f"{out} was written with tables={result['tables']}, teams={result['teams']}; pass the same or remove it")
    done = {e["epoch"] for e in result["epochs"]}
    todo = {k: p for k, p in checkpoints(training, root).items() if k not in done and (epochs is None or k in epochs)}
    if not todo: return result
    check, model = None, None
    try:
        for epoch, path in todo.items():
            if model is None:
                model = M.Model84.load(path)                 # builds the tables and paste lookups once
                kw = model.nets[0][1]
                check = E.EpochLegality(teams, tables=tables, known=training_keys(model.info))
            else:
                ck = load_checkpoint(path, map_location="cpu")
                if len(ck["members"]) != 1 or ck["members"][0]["net_kw"] != kw: raise SystemExit(f"{path.name}: another network")
                model.net.load_state_dict(ck["members"][0]["sd"]); model.info = ck.get("info") or {}
            entry = dict(epoch=epoch, checkpoint=path.name, **check(model))
            result["epochs"] = sorted(result["epochs"] + [entry], key=lambda e: e["epoch"])
            tmp = out.with_suffix(".tmp"); tmp.write_text(json.dumps(result, indent=1)); os.replace(tmp, out)
            print(f"epoch {epoch:2d} · illegal share {entry['illegal_share']:.3f} · errors per team {entry['errors_per_team']:.2f} · {entry['legality_seconds']:.0f} s · "
                  + "; ".join(f"{n}× {r}" for r, n in entry["rejection_reasons"][:2]), flush=True)
    finally:
        if check is not None: check.close()
    return result


def main(argv):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("training", help="checkpoint folder and name stem, such as stat84_shuffled_noise_linear_es3_cap80")
    p.add_argument("--tables", action="store_true", help="draw with the legal Pokémon construction tables (default off)")
    p.add_argument("--teams", type=int, default=E.TEAMS, help=f"teams per checkpoint (default {E.TEAMS})")
    p.add_argument("--epochs", type=lambda s: {int(v) for v in s.split(",")}, help="only these epochs, as 1,40,80")
    a = p.parse_args(argv)
    t0 = time.perf_counter()
    r = run(a.training, a.tables, a.teams, a.epochs)
    print(f"{len(r['epochs'])} epochs in {result_path(a.training, a.tables)} · {(time.perf_counter() - t0) / 60:.1f} min")
    print("CHECKPOINT_LEGALITY_DONE")


if __name__ == "__main__":
    main(sys.argv[1:])
