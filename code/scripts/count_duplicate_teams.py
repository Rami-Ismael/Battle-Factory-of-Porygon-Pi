"""Count duplicate teams (same six Pokémon, listed or any slot order) in both training stages of the deployed
model's two members. Exit 1 if any.  python scripts/count_duplicate_teams.py src   (/tmp/vgc-pilot/.venv/bin/python)"""
import sys, numpy as np
sys.path.insert(0, sys.argv[1])          # worktree code/src
import lossdown as L
from checkpoints import load_checkpoint
from encode import NF, NSLOT

def report(name, X):
    X = np.ascontiguousarray(X)
    exact = len(X) - len({r.tobytes() for r in X})
    slots = X.reshape(len(X), NSLOT, NF)
    canon = {np.ascontiguousarray(s[np.lexsort(s.T[::-1])]).tobytes() for s in slots}
    unordered = len(X) - len(canon)
    print(f"{name}: rows {len(X)}  exact-duplicate rows {exact}  same-six-Pokemon-any-order duplicates {unordered}")
    return unordered                  # counts every duplicate team, listed order or not (always >= exact)

d = L.load_data(include_heldout=True); corpus, V, *_ = L.X.setup()
keep = L.HO.row_mask(d["X"], V); TAG = L.tag_of(d["S"])
print("lossdata.npz all rows:"); report("  all", d["X"])
bad = 0
for p in ("F1_final_medium_noheldout", "G2_family_matrix_noprtrain_noheldout"):
    cfg = dict(L.DEFAULT, **load_checkpoint(L.CK / f"{p}.pt", map_location="cpu")["cfg"])
    tr, val = L.splits(d)
    if cfg["final"]: tr = np.arange(len(d["X"]))
    if cfg["mrow"]: tr = np.setdiff1d(tr, L.matrix_holdout(d))
    tr = tr[keep[tr]]
    tr = tr[np.isin(d["T"][tr], [["model","real","fill","hps"].index(s) for s in cfg["sources"].split(",")])]
    if cfg["data"] == "family": tr = tr[(TAG[tr] == L.TEST_TAG) | (TAG[tr] == 0)]
    elif cfg["data"] == "family+matrix": tr = tr[(TAG[tr] == L.TEST_TAG) | (TAG[tr] == 0) | (d["M"][tr, :, 1].sum(1) > 0)]
    elif cfg["data"] == "old": tr = tr[(((d["S"][tr] >> L.FAM.index("oldpool")) & 1) == 1) | (TAG[tr] == 0)]
    if cfg["upw"] > 1: tr = np.concatenate([tr] + [tr[TAG[tr] == L.TEST_TAG]] * (cfg["upw"] - 1))
    print(p, "cfg data=%s upw=%s final=%s mrow=%s" % (cfg["data"], cfg["upw"], cfg["final"], cfg["mrow"]))
    bad += report("  stage 1 (old vocabulary)", d["X"][tr])
    # stage 2: retrain_regmb train, v3
    out = L.CK.parent / "regmb-v3"
    m = np.load(out / "remapped.npz"); syn = np.load(out / "synthetic.npy")
    tr2, val2 = L.splits(d); tr2 = tr2[keep[tr2]]
    idx = tr2[(TAG[tr2] == L.TEST_TAG) | (TAG[tr2] == 0) | ((d["M"][tr2, :, 1].sum(1) > 0) if cfg["mrow"] else False)]
    if cfg["mrow"]: idx = np.setdiff1d(idx, L.matrix_holdout(d))
    idx = idx[m["valid"][idx]]
    bad += report("  stage 2 real rows (current vocabulary)", m["X"][idx])
    bad += report("  stage 2 real + synthetic", np.concatenate([m["X"][idx], syn]))
sys.exit(1 if bad else 0)
