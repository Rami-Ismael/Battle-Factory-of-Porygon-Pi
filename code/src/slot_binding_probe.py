"""Slot-binding probe (issue #361): does the 84-field network tie a field to its own Pokémon? Show it 256 real validation
teams; at each Pokémon's ability and first-move step of the species-first draw order, with the true earlier fields shown
and no tables, split its probability into: legal for its own species; forbidden for it but legal for another team
member's species; legal for none of the six. A network that binds fields to slots puts almost nothing in the middle part.

    python src/slot_binding_probe.py <checkpoint .pt> [...]

First read, 2026-11-17, epoch 33 of "No tables in the loss": ability 0.394 / 0.561 / 0.045, first move 0.470 / 0.498 / 0.032.
"""
import sys
sys.path.insert(0, "/tmp/vgc-pilot/src"); sys.path.insert(0, "src")
import lossdown as L
sys.path.insert(0, "src")
import numpy as np, torch
from pathlib import Path
import asked_vs_got as X, diffusion84 as M
d = np.load(X.RES / "lossdata84_shuffled_only.npz"); va = d["all_val"] & ~np.isnan(d["all_W"])
xs, ws = d["all_X"][va][:256].astype(np.int64), d["all_W"][va][:256].astype(np.float32)
corpus = X.setup()[0]
SPECIES = L.COLS_BY_KEY["species"]

@torch.no_grad()
def run(path):
    m = M.Model84.load(path, corpus)
    x0 = torch.as_tensor(xs, device=M.DEV); w = torch.as_tensor(ws, device=M.DEV)
    out = {}
    for field, cols in (("ability", L.COLS_BY_KEY["ability"]), ("first move", [s * 8 + 3 for s in range(6)])):
        own = other = none = 0.0
        for s, c in enumerate(cols):
            step = M.ORDER.index(c); x = x0.clone(); x[:, M.ORDER[step:]] = 0
            t = torch.full((len(x),), 1.0 - step / M.COLS, device=M.DEV)
            p = torch.softmax(m._column(m._logprobs(x, t, w, tables=False), c).masked_fill(
                torch.arange(m.V.sizes[L.KEY_OF_COL[c]], device=M.DEV) == 0, -float("inf")), -1)
            a_own = m._column(M.allowed(m.tables, x), c); a_any = torch.zeros_like(a_own)
            for j in range(6):
                if j == s: continue
                y = x.clone(); y[:, SPECIES[s]] = x[:, SPECIES[j]]; a_any |= m._column(M.allowed(m.tables, y), c)
            own += (p * a_own).sum(-1).mean().item(); other += (p * (~a_own & a_any)).sum(-1).mean().item()
            none += (p * (~a_own & ~a_any)).sum(-1).mean().item()
        out[field] = (own / 6, other / 6, none / 6)
    return out

if __name__ == "__main__":
    for path in sys.argv[1:]:
        for f, (a, b, c) in run(path).items():
            print(f"{Path(path).stem}, {f}: own species {a:.3f} · another team member's {b:.3f} · no team member's {c:.3f}")
