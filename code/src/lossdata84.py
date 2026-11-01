"""The 84-field training table (issue #58): every training team with its own Stat Points.

It reads the same sources as the 48-field table (`lossdata.gather`: the corpus, the old training pool, every results
file, the matchup database, the 100,000 uniform-legal teams) and keeps a 48-field team only if every appearance of it
carries the same six Stat Points per Pokémon. A team battled with several sets of Stat Points is dropped, and so is
one whose paste no longer encodes in the Regulation M-B vocabulary. Rows are encoded in that vocabulary
(`team84.encode`), so token ids match the deployed 48-field models. Labels are the pooled wins / battles of the team,
which with one set of Stat Points is the label of that exact 84-field team.

Two views:
  same-teams slices  rows the deployed loop-teams and battle-records models trained on (stored `lossdata.npz`,
                     their `lossdown.train_run` row selection), kept teams only, with the labels those models saw;
                     the battle-records slice also carries each team's matchup row (`lossdown.cond_matrix`)
  all-teams set      every kept team today, a seeded 2% marked for validation

The test 500 and the held-out meta teams ranked 51-100 never enter either view (48-field match, as today).

    /tmp/vgc-pilot/.venv/bin/python src/lossdata84.py      -> results/lossdata84.npz + lossdata84_counts.json
"""
import hashlib
import io
import json
import sys
import time
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, "/tmp/vgc-pilot/src")
sys.path.insert(0, str(HERE))                   # this checkout's src/ first: lossdata.gather is new
import lossdata as LD
sys.path.insert(0, str(HERE))                   # lossdata puts /tmp/vgc-pilot/src first again
import asked_vs_got as X
import heldout as HO
import lossdown as L
import stat_points as SP
import team84 as T
from encode import canon

OUT = X.RES / "lossdata84.npz"
COUNTS = X.RES / "lossdata84_counts.json"
VAL_SHARE, VAL_SEED = 0.02, 0


def stat_point_key(team):
    """The team's 36 Stat Points in the 84-field order."""
    return tuple(int(s.get("evs", {}).get(k, 0)) for s in canon(team) for k in SP.STATS)


def single(found):
    """The team's one 84-field row, or None when it has several sets of Stat Points, none, or does not encode.
    found maps each set of Stat Points seen for one 48-field team to its 84-field row (None if it did not encode)."""
    if not found or len(found) != 1: return None
    return next(iter(found.values()))


def legal_stat_points(rows):
    """True when every row's Stat Points are drawn, each at most 32, and each Pokémon's six total at most 66."""
    v = rows[:, T.COLS48:].astype(np.int64) - 1
    return bool((v >= 0).all() and (v <= SP.MAX_STAT).all() and (v.reshape(len(v), SP.NSLOT, SP.NSTAT).sum(-1) <= SP.MAX_TOTAL).all())


def save_npz(path, arrays):
    """np.savez with fixed zip timestamps and key order, so the same arrays give the same bytes."""
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for k in sorted(arrays):
            buf = io.BytesIO(); np.lib.format.write_array(buf, np.ascontiguousarray(arrays[k]), allow_pickle=False)
            info = zipfile.ZipInfo(f"{k}.npy", date_time=(1980, 1, 1, 0, 0, 0)); info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, buf.getvalue())


def collect(Vreg):
    """lossdata.gather() plus, per 48-field key, {Stat Points: 84-field row in the regulation vocabulary or None}."""
    sets = defaultdict(dict)
    def observe(k, team):
        found, sp = sets[k], stat_point_key(team)
        if sp in found: return
        try: found[sp] = T.encode(Vreg, team).astype(np.int16)
        except ValueError: found[sp] = None                  # a field outside the regulation, or illegal Stat Points
    return LD.gather(observe), sets


def same_team_slices(sets, V):
    """The deployed members' training rows (lossdown.train_run with final=1), restricted to kept teams."""
    d = L.load_data(include_heldout=True)
    keep = HO.row_mask(d["X"], V)
    tag = L.tag_of(d["S"])
    family = (tag == L.TEST_TAG) | (tag == 0)
    matrix = d["M"][:, :, 1].sum(1) > 0
    mhold = np.zeros(len(d["X"]), bool); mhold[L.matrix_holdout(d)] = True
    out, counts = {}, {}
    test = {r.tobytes() for r in d["xte"]}
    for name, rows in (("loop", keep & family), ("records", keep & ~mhold & (family | matrix))):
        idx = np.where(rows)[0]
        assert not any(d["X"][i].tobytes() in test for i in idx), f"a test team is in the {name} slice"
        found = [single(sets.get(d["X"][i].tobytes())) for i in idx]
        ok = np.array([r is not None for r in found], bool)
        out[f"{name}_X"] = np.stack([r for r in found if r is not None])
        out[f"{name}_W"] = d["W"][idx[ok]].astype(np.float32)
        if name == "records": out["records_WM"] = L.cond_matrix(d, idx[ok]).astype(np.float32)
        counts[name] = dict(rows=len(idx), kept=int(ok.sum()), dropped=int((~ok).sum()))
    return out, counts


def all_teams(g, sets):
    pool = g["pool"]
    keys = [k for k in pool if single(sets.get(k)) is not None]
    Xa = np.stack([single(sets[k]) for k in keys])
    W = np.array([pool[k]["wins"] / pool[k]["battles"] if pool[k]["battles"] else np.nan for k in keys], np.float32)
    NB = np.array([pool[k]["battles"] for k in keys], np.int32)
    val = np.zeros(len(keys), bool)
    val[np.random.default_rng(VAL_SEED).permutation(len(keys))[:round(VAL_SHARE * len(keys))]] = True
    n_sets = np.array([len(sets.get(k, ())) for k in pool])
    labelled = np.array([pool[k]["battles"] > 0 for k in pool])
    counts = dict(rows=len(pool), kept=len(keys), several_sets=int((n_sets > 1).sum()), no_paste=int((n_sets == 0).sum()),
                  not_encodable=int(((n_sets == 1) & np.array([single(sets.get(k)) is None for k in pool])).sum()),
                  labelled_kept=int(np.isfinite(W).sum()), labelled_several_sets=int((labelled & (n_sets > 1)).sum()),
                  val=int(val.sum()))
    return dict(all_X=Xa, all_W=W, all_NB=NB, all_val=val), counts, keys


def main():
    t0 = time.perf_counter()
    for m in (LD, L, T, SP, X, HO):
        print(f"{m.__name__:14s} {m.__file__}", flush=True)
    _, V, *_ = X.setup()
    g, sets = collect(T.regulation_vocab())
    arrays, counts, keys = all_teams(g, sets)
    slices, slice_counts = same_team_slices(sets, V)
    arrays.update(slices); counts.update(slice_counts)

    held = HO.identities()
    assert not (set(keys) & {r.tobytes() for r in g["xte"]}), "a test team reached the all-teams set"
    assert not any(HO.row_key(g["pool"][k]["x"], V) in held for k in keys), "a held-out team reached the all-teams set"
    for k in ("all_X", "loop_X", "records_X"):
        assert legal_stat_points(arrays[k]), f"{k} has Stat Points outside the limits"
    save_npz(OUT, arrays)
    counts["sha256"] = hashlib.sha256(OUT.read_bytes()).hexdigest()
    counts["minutes"] = round((time.perf_counter() - t0) / 60, 1)
    json.dump(counts, open(COUNTS, "w"), indent=1)
    print(json.dumps(counts, indent=1))


if __name__ == "__main__":
    main()
