"""Train the no-prompt 84-field model on the shuffled-slot table with three subset-size rules (docs/subset-size-labels.md).

Arms (reader name, code ID), differing only in how many training opponents each label averages:
    10 to 39 opponents per label   subset_10_39   a fresh subset every epoch, size uniform in [10, 39]
    1 to 39 opponents per label    subset_1_39    a fresh subset every epoch, size uniform in [1, 39]
    All 39 opponents per label     subset_39      the control: every row's label over all its training opponents

Data: ~/vgc-data/opponent_prompt_labels/lossdata65.npz (src/lossdata65.py). Rows within EXCLUDE_WITHIN species swaps of a
validation opponent are dropped from training and validation rows (pending the near-copy grill on #118). Validation
rows keep the all-39 label in every arm, so validation loss compares like with like. Recipe as train84 "all": deployed
network, cap 80 epochs, stop after 3 with no new lowest validation loss, keep the best. No label noise beyond the subset.

    /tmp/vgc-pilot/.venv/bin/python src/train65.py smoke          -> preflight training smoke, results/train65_smoke.json
    /tmp/vgc-pilot/.venv/bin/python src/train65.py subset_10_39   -> checkpoint stat65_subset_10_39.pt
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, "/tmp/vgc-pilot/src")
sys.path.insert(0, str(HERE))
import lossdown as L                                     # before diffusion84: its imports put /tmp/vgc-pilot/src first
sys.path.insert(0, str(HERE))
import diffusion84 as M
import lossdata65 as D
import team84 as T
import asked_vs_got as X
from train84 import NET, RECIPE

DATA = D.OUT / "lossdata65.npz"
RES = Path.home() / "vgc-data" / "results"
EXCLUDE_WITHIN = 1              # species swaps; pending the near-copy grill (#118)
SUBSET_SEED = 6500
ARMS = dict(subset_10_39=(10, 39), subset_1_39=(1, 39), subset_39=None)
READER = dict(subset_10_39="10 to 39 opponents per label", subset_1_39="1 to 39 opponents per label",
              subset_39="All 39 opponents per label")


def full_label(C):
    return (C[:, :, 0].sum(1) / np.maximum(C[:, :, 1].sum(1), 1)).astype(np.float32)


def load():
    d = dict(np.load(DATA))
    keep = d["val_dist"] > EXCLUDE_WITHIN
    tr, va = keep & ~d["val_rows"], keep & d["val_rows"]
    return d, tr, va, hashlib.sha256(DATA.read_bytes()).hexdigest()


def label_rule(arm, C):
    """(fixed labels, labels_each_epoch or None) for one arm."""
    if ARMS[arm] is None: return full_label(C), None
    lo, hi = ARMS[arm]
    return full_label(C), lambda p: D.subset_labels(C, np.random.default_rng(SUBSET_SEED + p), lo, hi)[1]


def train(arm):
    d, tr, va, sha = load()
    X84, C = d["X"].astype(np.int64), d["C"]
    labels, each = label_rule(arm, C[tr])
    val = (X84[va], full_label(C[va]))
    corpus, V = X.setup()[0], T.regulation_vocab()
    name = f"stat65_{arm}"; out = L.CK / f"{name}.pt"
    print(f"[{name}] {READER[arm]} · {int(tr.sum())} teams · {int(va.sum())} validation teams · cap 80, patience 3", flush=True)
    t0 = time.perf_counter()
    m = M.Model84.train_new(V, corpus, X84[tr], labels, epochs=80, val=val, patience=3, on_best=lambda m, _p: m.save(out),
                            labels_each_epoch=each, **RECIPE, **NET)
    m.info.update(arm=arm, reader=READER[arm], subset=ARMS[arm], subset_seed=SUBSET_SEED, exclude_within=EXCLUDE_WITHIN,
                  data_sha256=sha, minutes=round((time.perf_counter() - t0) / 60, 1), val_rows=int(va.sum()))
    m.save(out)
    RES.mkdir(parents=True, exist_ok=True)
    json.dump(m.info, open(RES / f"{name}.json", "w"), indent=1, default=float)
    print(f"[{name}] done in {m.info['minutes']} min; best epoch {m.info['best_epoch']}, stopped {m.info['stopped_epoch']} -> {out}")


@torch.no_grad()
def init_ce(V, corpus, rows):
    """Per-field cross-entropy of a fresh network with every field masked, against ln K of the values the tables allow."""
    tables = L.Tables(V, corpus)
    net = M.Net84(V, **NET).to(M.DEV).eval()
    x = torch.as_tensor(rows, device=M.DEV)
    xin = torch.zeros_like(x)
    al = M.allowed(tables, xin)
    lg = {k: v.masked_fill(~al[k], -1e4) for k, v in net.logits(net(xin, torch.ones(len(x), device=M.DEV), torch.full((len(x),), M.NAN, device=M.DEV))).items()}
    ce = M.per_column_ce(lg, x).mean(0)
    lnk = torch.zeros_like(ce)
    for k, cols in L.COLS_BY_KEY.items(): lnk[cols] = al[k].sum(-1).float().clamp(min=1).log().mean(0)
    lnk[M.STAT0:] = al["stat"].sum(-1).float().clamp(min=1).log().mean(0)
    ratio = (ce / lnk.clamp(min=1e-6))
    return dict(mean_ce=round(float(ce.mean()), 3), mean_ln_k=round(float(lnk.mean()), 3),
                ratio_median=round(float(ratio.median()), 3), ratio_min=round(float(ratio.min()), 3), ratio_max=round(float(ratio.max()), 3))


def smoke():
    """Build-experiment training smoke: init cross-entropy near ln K, loss falls on 64 teams, labels redrawn every epoch."""
    d, tr, _va, sha = load()
    idx = np.where(tr)[0][:64]
    X84, C = d["X"][idx].astype(np.int64), d["C"][idx]
    corpus, V = X.setup()[0], T.regulation_vocab()
    calls = []
    def each(p):
        calls.append(p); return D.subset_labels(C, np.random.default_rng(SUBSET_SEED + p), 1, 39)[1]
    ce = init_ce(V, corpus, X84)
    m = M.Model84.train_new(V, corpus, X84, full_label(C), epochs=400, batch=16, lr=1e-3, warmup=10, ema=0, seed=0,
                            sched=M.SCHED, labels_each_epoch=each, **NET)
    curve = [e["train_loss"] for e in m.info["curves"]]
    first, last = float(np.mean(curve[:5])), float(np.mean(curve[-5:]))
    draws = [D.subset_labels(C, np.random.default_rng(SUBSET_SEED + p), 1, 39)[1] for p in (0, 1)]
    out = dict(init_cross_entropy=ce, loss_first5=round(first, 4), loss_last5=round(last, 4), epochs=len(curve),
               labels_redrawn_each_epoch=calls == list(range(len(curve))),
               labels_differ_between_epochs=bool((draws[0] != draws[1]).any()),
               passed=bool(0.7 <= ce["ratio_median"] <= 1.4 and last < 0.5 * first and calls == list(range(len(curve)))),
               data_sha256=sha)
    RES.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(RES / "train65_smoke.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


def sides_and_policy(con):
    import opponent_split as OS
    s = OS.load(); tids = D.meta_team_ids(con, s)
    return dict(training=[tids[i] for i in s["training"]], validation=[tids[i] for i in s["validation"]]), D.shuffled_policy(con)


def got_by_side(con, policy, sides, t):
    """{side: wins / battles of team t against that side's opponents} from stored shuffled-slot battles."""
    out = {}
    for side, cols in sides.items():
        ph = ",".join("?" * len(cols)); w = n = 0
        for wins, b in con.execute(f"SELECT lo_wins, battles FROM cell WHERE policy_id = ? AND lo = ? AND hi IN ({ph}) UNION ALL "
                                   f"SELECT hi_wins, battles FROM cell WHERE policy_id = ? AND hi = ? AND lo IN ({ph})",
                                   [policy, t, *cols, policy, t, *cols]):
            w += wins; n += b
        if n: out[side] = w / n
    return out


def team_ids(con, cell):
    """The cell's database team ids: stored by the battle step, else found by each paste's canonical hash."""
    if "team_ids" in cell: return cell["team_ids"]
    import matchup_db as MDB
    ids = []
    for p in cell.get("pastes", []):
        r = con.execute("SELECT team_id FROM team WHERE format_id = ? AND canon_hash = ?",
                        (MDB.FORMAT, MDB.canon_hash(MDB.canonical_paste(p)))).fetchone()
        if r: ids.append(r[0])
    return ids


def split(model, reader=None):
    """Got per ask against the 39 training opponents (what the ask means) and the 10 validation opponents, from the
    stored battles of an ask grid (ask_grid84.py with AG84_MODEL=model), with ask-versus-got error on each. Works for
    any grid model, so the earlier label-noise model is rescored on the same opponents without new battles."""
    import sqlite3
    grid = json.load(open(RES / f"stat_points_84_{model}.json"))
    con = sqlite3.connect(f"file:{D.DB}?mode=ro", uri=True)
    sides, policy = sides_and_policy(con)
    per_ask = {}
    for c in grid["cells"].values():
        k = round(c["ask"] * 24)
        for t in team_ids(con, c):
            for side, g in got_by_side(con, policy, sides, t).items(): per_ask.setdefault(k, {}).setdefault(side, []).append(g)
    rows = {k: {side: dict(teams=len(v), got=float(np.mean(v))) for side, v in sd.items()} for k, sd in sorted(per_ask.items())}
    def error(side, ks):
        e = [(rows[k][side]["got"] - k / 24) ** 2 for k in ks if k in rows and side in rows[k]]
        return dict(asks=len(e), mse=float(np.mean(e)) if e else None)
    out = dict(model=model, reader=reader or grid.get("reader"), per_ask=rows,
               error={side: dict(judged_1_to_17=error(side, range(1, 18)), apart_18_to_23=error(side, range(18, 24))) for side in sides})
    json.dump(out, open(RES / f"subset_size_labels_{model}.json", "w"), indent=1)
    print(json.dumps(out["error"], indent=1))


def unguided(model="gen65_subset_39"):
    """Unguided baseline at one ask cell's budget: the control sampled with no ask, 64 teams per seed, battled like a grid cell."""
    import os
    import sqlite3
    os.environ["AG84_MODEL"] = model
    import ask_grid84 as AG
    import matchup_db as MDB
    path = RES / f"subset_size_labels_unguided_{model}.json"
    out = json.load(open(path)) if path.exists() else dict(model=model, cells={})
    w = AG.load_model(); con = MDB.connect()
    for seed in AG.SEEDS:
        key = f"s{seed}"
        if "y" in out["cells"].get(key, {}): continue
        kept, _, counts = AG.draw(w, None, seed)
        sc = MDB.score_pastes(con, kept, per_cell=AG.BATTLES_PER_OPPONENT, seed=AG.SEED0 + seed, origin=f"unguided65:{model}:{key}")
        if not AG.complete(sc): print(f"  {key} INCOMPLETE", flush=True); continue
        out["cells"][key] = dict(seed=seed, **counts, pastes=kept, team_ids=[s["team_id"] for s in sc], y=[s["score"] for s in sc],
                                 diversity=AG.subset_diversity(kept, seed))
        json.dump(out, open(path, "w"), indent=1)
    ro = sqlite3.connect(f"file:{D.DB}?mode=ro", uri=True); sides, policy = sides_and_policy(ro)
    got = [got_by_side(ro, policy, sides, t) for c in out["cells"].values() for t in c["team_ids"]]
    out["got"] = {side: float(np.mean([g[side] for g in got if side in g])) for side in sides}
    json.dump(out, open(path, "w"), indent=1)
    print(json.dumps(out["got"], indent=1))


if __name__ == "__main__":
    arm = sys.argv[1]
    if arm == "smoke": smoke()
    elif arm == "split": split(sys.argv[2])
    elif arm == "unguided": unguided(*sys.argv[2:3])
    elif arm in ARMS: train(arm)
    else: raise SystemExit(f"arm must be smoke or one of {sorted(ARMS)}")
