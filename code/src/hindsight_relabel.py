"""Hindsight-relabeled fine-tune, code ID ft_hindsight_relabel (GitHub issue #106; docs/hindsight-relabel-fine-tune.md).

Fine-tunes the epoch-21 label-noise model (labelled teams only, label noise ±1/48) on every team it generated and
battled in its 23-ask grid, each labelled with the win rate it measured, not the ask it was drawn at. No elite filter:
that filter is what separates the search loop from this. Hindsight Experience Replay and Online Decision Transformer's
"Hindsight Return Relabeling" train on samples labelled with what they achieved; this is that idea on the team model.

Recipe: the search loop's fine-tune learning rate and batch (1e-4, 64; src/cem_dbbo.py), the base training's label
noise, EMA and ask dropout, early stopping on a seeded 20% of the relabelled teams with patience 3 and a cap of 40
passes. Pass 0 is the starting weights, so a fine-tune that never beats them returns the control unchanged. The
labelled validation teams the base model held out are logged every pass to show what the fine-tune forgets.

    /tmp/vgc-pilot/.venv/bin/python src/hindsight_relabel.py train     (from code/; HR_SMOKE=1: plumbing check)
"""
import hashlib
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

CODE_ID = "ft_hindsight_relabel"
BASE_MODEL = "gen84_noise_byepoch_ep21"                  # the control's ask-grid name: its results file holds the battled teams
BASE_CHECKPOINT = "stat84_labelled_noise_byepoch_es3_cap80/stat84_labelled_noise_byepoch_es3_cap80_ep21"   # as ask_grid84.MODELS
VAL_SHARE, SPLIT_SEED = 0.2, 0
RECIPE = dict(passes=40, patience=3, lr=1e-4, batch=64, label_noise=1 / 48, seed=0)
CHECKPOINT = f"stat84_labelled_noise_byepoch_ep21_{CODE_ID}_es3_cap40"
RES = Path.home() / "vgc-data/results"


def relabels(out):
    """(paste, measured win rate, ask numerator, sampling seed) for every battled team in an ask-grid results file.
    A cell whose battles are missing or incomplete is an error: the relabel set is every battled team, no fewer."""
    teams = []
    for tag, c in sorted(out["cells"].items()):
        y = c.get("y")
        if y is None or len(y) != len(c["pastes"]) or any(v is None for v in y):
            raise ValueError(f"{tag}: every team needs its measured win rate")
        teams += [(p, float(v), c["numerator"], c["seed"]) for p, v in zip(c["pastes"], y)]
    return teams


def split(n, share=VAL_SHARE, seed=SPLIT_SEED):
    """Seeded random validation mask over n teams, holding round(share * n) of them."""
    val = np.zeros(n, bool)
    val[np.random.default_rng(seed).permutation(n)[:round(share * n)]] = True
    return val


def per_ask_count(labels):
    """How many relabels sit nearest each k/24, for k = 0..24."""
    c = Counter(int(k) for k in np.clip(np.rint(np.asarray(labels) * 24), 0, 24))
    return {k: c.get(k, 0) for k in range(25)}


def encode_teams(V, pastes):
    """84-field rows for the pastes. Fails on any paste that does not encode or that is a held-out team."""
    import heldout as HO
    import team84 as T
    from corpus import parse_team_text
    rows = []
    for i, p in enumerate(pastes):
        if HO.is_heldout_text(p): raise ValueError(f"team {i} is a held-out team")
        rows.append(T.encode(V, parse_team_text(p)))
    return np.asarray(rows, np.int64)


def train():
    here = str(Path(__file__).resolve().parent)
    sys.path.insert(0, "/tmp/vgc-pilot/src"); sys.path.insert(0, here)
    import lossdown as L                                 # before diffusion84: its imports put /tmp/vgc-pilot/src first
    sys.path.insert(0, here)
    import asked_vs_got as X
    import diffusion84 as M
    smoke = bool(os.environ.get("HR_SMOKE"))
    grid = json.load(open(RES / f"stat_points_84_{BASE_MODEL}.json"))
    base_path = L.CK / f"{BASE_CHECKPOINT}.pt"
    sha = hashlib.sha256(base_path.read_bytes()).hexdigest()
    if sha != grid["checkpoint_sha256"]: raise SystemExit(f"{base_path} is not the checkpoint that drew {BASE_MODEL}'s teams")
    base = M.Model84.load(base_path)
    teams = relabels(grid)
    pastes, labels = [t[0] for t in teams], np.array([t[1] for t in teams], np.float32)
    rows = encode_teams(base.V, pastes)
    val = split(len(rows))
    d = np.load(X.RES / "lossdata84.npz"); va = d["all_val"] & ~np.isnan(d["all_W"])
    watch = dict(labelled=(d["all_X"][va].astype(np.int64), d["all_W"][va]))
    recipe = dict(RECIPE)
    if smoke:
        rows, labels, val = rows[:96], labels[:96], val[:96]
        watch = dict(labelled=tuple(a[:64] for a in watch["labelled"])); recipe.update(passes=1)
    print(f"[{CODE_ID}] {(~val).sum()} relabelled teams to train on, {val.sum()} to stop on; base {base_path.name}", flush=True)
    t0 = time.perf_counter()
    m = base.fine_tuned(rows[~val], labels[~val], val=(rows[val], labels[val]), watch=watch, **recipe)
    name = ("hrsmoke_" if smoke else "") + CHECKPOINT
    m.info.update(code_id=CODE_ID, reader="Hindsight-relabeled fine-tune, epoch 21", base_checkpoint=base_path.name, base_sha256=sha,
                  relabel_source=f"stat_points_84_{BASE_MODEL}.json", relabelled_teams=len(rows), val_share=VAL_SHARE, split_seed=SPLIT_SEED,
                  relabels_per_ask=per_ask_count(labels), minutes=round((time.perf_counter() - t0) / 60, 2), labelled_only=True)
    m.save(L.CK / f"{name}.pt")
    json.dump({k: v for k, v in m.info.items() if k != "base"}, open(RES / f"{name}.json", "w"), indent=1, default=float)
    print(f"[{CODE_ID}] best pass {m.info['best_pass']}, stopped {m.info['stopped_pass']}, {m.info['minutes']} min -> {name}.pt", flush=True)
    print("HR_TRAIN_DONE")


if __name__ == "__main__":
    {"train": train}[sys.argv[1]]() if sys.argv[1:] == ["train"] else sys.exit(__doc__)
