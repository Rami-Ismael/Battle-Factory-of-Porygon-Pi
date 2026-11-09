"""Training rows and per-opponent battle counts for the opponent-prompt models (issue #118, map #115).

Only shuffled-slot battles count (GLOSSARY "Win rate", 2026-11-07): every other battle configuration, listed order
included, supplies no row and no label. Each row is one 84-field team that has shuffled battles against the 39
training opponents of the frozen split (docs/opponent-split.json). Validation-opponent battles are never read into the
table, so no label can carry them.

Stored per row: the 84 tokens, (wins, battles) against each training opponent, and what the trainer needs to apply
the exclusions still being decided: species distance to the nearest validation opponent, whether it has a validation
opponent's 48 fields, and whether it is a training meta team. A label is drawn at training time by subset_labels():
wins / battles over a random subset of the row's training opponents.

Excluded while building: ranks 51-100 by their 48 fields (heldout.py), and teams that do not encode in the Regulation
M-B vocabulary. Identical 84-field teams are pooled into one row.

    /tmp/vgc-pilot/.venv/bin/python src/lossdata65.py   -> ~/vgc-data/opponent_prompt_labels/lossdata65.npz + _counts.json
"""
import hashlib
import json
import os
import sqlite3
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, "/tmp/vgc-pilot/src")
sys.path.insert(0, str(HERE))
import heldout as HO
import lossdata84 as L84
import opponent_split as OS
import team84 as T
import top50 as T50
from corpus import parse_team_text
from temperature_experiment import paste_species

DB = Path(os.environ.get("MATCHUP_DB", Path.home() / "vgc-data" / "matchup_regmb.sqlite"))
OUT = Path.home() / "vgc-data" / "opponent_prompt_labels"
SHUFFLED = "+slots-shuffled"
VAL_SHARE, VAL_SEED = 0.02, 0          # validation rows for early stopping, as in lossdata84
K_MIN, K_MAX = 10, 39                  # proposed subset sizes (issue #118 Q4), not yet confirmed by the owner


def norm(s):
    return "".join(ch for ch in (s or "").lower() if ch.isalnum())


def parse(paste):
    """Six slot dicts; parsed block by block, as heldout.grid_key_text does."""
    team = []
    for block in paste.strip().split("\n\n"): team += parse_team_text(block)
    return team


def shuffled_policy(con):
    ids = [p for (p,) in con.execute("SELECT policy_id FROM policy WHERE row_policy LIKE ? AND col_policy LIKE ?",
                                     (f"%{SHUFFLED}", f"%{SHUFFLED}"))]
    if len(ids) != 1: raise ValueError(f"expected one shuffled-slot configuration, found {ids}")
    return ids[0]


def meta_team_ids(con, split):
    """{MB id: team_id} for the 49 distinct meta teams in the database's top50 set."""
    out = {}
    for tid, src in con.execute("SELECT ts.team_id, s.source_id FROM team_set ts JOIN team_source s ON s.team_id = ts.team_id "
                                "WHERE ts.set_name = 'top50' AND s.source = 'vgcpastes'"):
        if src in split["input_order"]: out[src] = tid
    missing = set(split["input_order"]) - set(out)
    if missing: raise ValueError(f"meta teams missing from the database top50 set: {sorted(missing)}")
    return out


def battle_counts(con, policy, train_tids, val_tids):
    """{team_id: int32 [39, 2] (wins, battles) against each training opponent}, and the validation battles skipped."""
    col = {t: j for j, t in enumerate(train_tids)}
    meta = list(col) + list(val_tids)
    ph = ",".join("?" * len(meta))
    rows, skipped = {}, 0
    for lo, hi, lw, hw, n in con.execute(f"SELECT lo, hi, lo_wins, hi_wins, battles FROM cell WHERE policy_id = ? "
                                         f"AND (lo IN ({ph}) OR hi IN ({ph}))", [policy, *meta, *meta]):
        for me, op, w in ((lo, hi, lw), (hi, lo, hw)):
            if op in col:
                r = rows.get(me)
                if r is None: r = rows[me] = np.zeros((len(col), 2), np.int32)
                r[col[op], 0] += w; r[col[op], 1] += n
            elif op in val_tids: skipped += n
    return rows, skipped


def subset_labels(C, rng, k_min=K_MIN, k_max=K_MAX):
    """A fresh subset per row: k uniform in [k_min, k_max], capped at the row's battled opponents, then wins / battles
    over those opponents. Returns (mask [N, 39] bool, label [N] float32)."""
    battled = C[:, :, 1] > 0
    have = battled.sum(1)
    k = np.minimum(rng.integers(k_min, k_max + 1, len(C)), have)
    score = np.where(battled, rng.random(battled.shape), np.inf)
    rank = score.argsort(1).argsort(1)
    mask = rank < k[:, None]
    wins, games = (C[:, :, 0] * mask).sum(1), (C[:, :, 1] * mask).sum(1)
    return mask, (wins / np.maximum(games, 1)).astype(np.float32)


def build(con, split):
    tids = meta_team_ids(con, split)
    train_tids = [tids[i] for i in split["training"]]
    val_tids = {tids[i] for i in split["validation"]}
    val_species = [split["species"][i] for i in split["validation"]]
    val_grid = {HO.grid_key_text(Path(T50.resolve(i)).read_text()) for i in split["validation"]}
    train_index = {t: j for j, t in enumerate(train_tids)}
    policy = shuffled_policy(con)
    counts, skipped = battle_counts(con, policy, train_tids, val_tids)
    Vreg, held = T.regulation_vocab(), HO.identities()
    drops, pooled, origins = Counter(), {}, Counter()
    for tid, paste, origin in con.execute("SELECT team_id, paste, origin FROM team"):
        c = counts.get(tid)
        if c is None: continue
        team = parse(paste)
        if HO.grid_key_team(team) in held: drops["held_out_51_100"] += 1; continue
        try: x = T.encode(Vreg, team).astype(np.int16)
        except ValueError: drops["not_encodable"] += 1; continue
        k = x.tobytes()
        if k in pooled:
            e = pooled[k]; e["C"] += c; e["merged"] += 1; continue
        sp = paste_species(paste)
        pooled[k] = dict(x=x, C=c.copy(), tid=tid, origin=origin, merged=0,
                         val_dist=min(OS.distance(sp, v) for v in val_species),
                         val_same48=HO.grid_key_team(team) in val_grid, meta=train_index.get(tid, -1))
        origins[origin.split(":")[0]] += 1
    es = list(pooled.values())
    N = len(es)
    val_rows = np.zeros(N, bool)
    val_rows[np.random.default_rng(VAL_SEED).permutation(N)[:round(VAL_SHARE * N)]] = True
    arrays = dict(X=np.stack([e["x"] for e in es]), C=np.stack([e["C"] for e in es]),
                  team_id=np.array([e["tid"] for e in es], np.int64),
                  val_dist=np.array([e["val_dist"] for e in es], np.int8),
                  val_same48=np.array([e["val_same48"] for e in es], bool),
                  meta=np.array([e["meta"] for e in es], np.int8), val_rows=val_rows)
    info = dict(policy_id=policy, battled_teams=len(counts), rows=N, drops=dict(drops),
                pooled_duplicates=sum(e["merged"] for e in es), validation_battles_skipped=int(skipped),
                validation_rows=int(val_rows.sum()), origins=dict(origins.most_common()),
                val_dist=dict(Counter(int(d) for d in arrays["val_dist"])), val_same48=int(arrays["val_same48"].sum()),
                training_meta_rows=int((arrays["meta"] >= 0).sum()),
                training_opponents=split["training"], origin_of_row=[e["origin"] for e in es])
    return arrays, info, Vreg


# ---------------------------------------------------------------- preflight probes (build-experiment step 2)
def round_trip(con, arrays, Vreg, n=20, seed=0):
    """Decode n rows and compare every field with the database paste they came from."""
    bad = []
    for i in np.random.default_rng(seed).choice(len(arrays["X"]), n, replace=False):
        (paste,) = con.execute("SELECT paste FROM team WHERE team_id = ?", (int(arrays["team_id"][i]),)).fetchone()
        want = sorted((norm(s["species"]), norm(s["ability"]), norm(s["item"]), tuple(sorted(norm(m) for m in s["moves"])),
                       norm(s.get("nature")), tuple(int(s["evs"].get(k, 0)) for k in T.SP.STATS)) for s in parse(paste))
        got = sorted((norm(s["species"]), norm(s["ability"]), norm(s["item"]), tuple(sorted(norm(m) for m in s["moves"])),
                      norm(s["nature"]), tuple(int(s["evs"][k]) for k in T.SP.STATS)) for s in T.decode(Vreg, arrays["X"][i]))
        if want != got: bad.append(int(arrays["team_id"][i]))
    return dict(rows=n, mismatched=bad, passed=not bad)


def controls(arrays, info):
    """Training meta teams against the other training opponents near 0.50; uniform-legal teams near 0.015."""
    C = arrays["C"].astype(np.float64)
    wr = C[:, :, 0].sum(1) / np.maximum(C[:, :, 1].sum(1), 1)
    meta = arrays["meta"] >= 0
    hps = np.array([o == "unlabelled84:hps" for o in info["origin_of_row"]])
    m, h = float(wr[meta].mean()), float(wr[hps].mean()) if hps.any() else None
    return dict(meta_rows=int(meta.sum()), meta_mean=round(m, 4), uniform_legal_rows=int(hps.sum()),
                uniform_legal_mean=None if h is None else round(h, 4),
                passed=bool(0.45 <= m <= 0.55 and h is not None and h < 0.05))


def separation(arrays, info, split, Vreg):
    """Re-derive each stored row's 48-field key from its tokens and intersect with ranks 51-100."""
    held = HO.identities()
    held_rows = sum(HO.row_key(x[:T.COLS48], Vreg) in held for x in arrays["X"])
    match = info["training_opponents"] == split["training"]
    return dict(columns=int(arrays["C"].shape[1]), columns_match_split=match, held_out_rows=int(held_rows),
                rows_at_val_dist=info["val_dist"], rows_with_validation_48_fields=info["val_same48"],
                passed=bool(arrays["C"].shape[1] == 39 and match and held_rows == 0))


def noise_floor(arrays, sizes=(5, 10, 20, 30, 39), seed=0):
    """Reliability of a subsampled label: 1 - mean battle-noise variance / variance of labels across rows."""
    C = arrays["C"]
    out = {}
    for k in sizes:
        mask, y = subset_labels(C, np.random.default_rng(seed), k, k)
        n = (C[:, :, 1] * mask).sum(1)
        noise = float((y * (1 - y) / np.maximum(n, 1)).mean())
        out[k] = dict(label_sd=round(float(y.std()), 4), noise_sd=round(noise ** 0.5, 4),
                      reliability=round(1 - noise / float(y.var()), 3))
    return out


def battle_depth(arrays):
    per_opp = arrays["C"][:, :, 1].sum(1) / np.maximum((arrays["C"][:, :, 1] > 0).sum(1), 1)
    return {f">={d}": int((per_opp >= d).sum()) for d in (1, 2, 4, 8, 16)}


def main():
    t0 = time.perf_counter()
    split = OS.load()
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    arrays, info, Vreg = build(con, split)
    assert L84.legal_stat_points(arrays["X"].astype(np.int64)), "Stat Points outside the limits"
    preflight = dict(round_trip=round_trip(con, arrays, Vreg), controls=controls(arrays, info),
                     separation=separation(arrays, info, split, Vreg), noise_floor=noise_floor(arrays),
                     battle_depth=battle_depth(arrays))
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "lossdata65.npz"
    L84.save_npz(path, arrays)
    info.pop("origin_of_row")
    info.update(sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                split_sha256=hashlib.sha256(OS.MANIFEST.read_bytes()).hexdigest(),
                minutes=round((time.perf_counter() - t0) / 60, 1), preflight=preflight)
    json.dump(info, open(OUT / "lossdata65_counts.json", "w"), indent=1)
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
