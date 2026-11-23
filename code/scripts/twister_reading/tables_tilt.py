"""How much probability the legal Pokémon construction tables remove from the trained network at each draw, per team,
for the species-first rows of the ask grids (asks 13/24-17/24). Same redraw as tilt_weights.py (each stored cell with
its own seed, kept pastes checked bit for bit against the store); the recorded log_w is now measured against the network
before the tables (base_law.py), so it is the tables' share plus the lookahead's. Species-first rows have no lookahead
removal, so here it is the tables alone. The output has tilt_weights.py's format, for tilt_analysis.py.

Usage: AG84_MODEL=<row> AG84_GUIDANCE=<g> AG84_ASKS=13,14,15,16,17 python tables_tilt.py <out.json>

Runs against the code at commit 787fac31, where the redraws were made: main has since moved the samplers into sampler84.py (#398), and SpeciesFirstArm, RandomDecodingArm and diffusion84.Refusals are gone. To rerun, `git worktree add /tmp/twister 787fac31` and run this file from there."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src")); sys.path.insert(0, str(Path(__file__).parent))
import torch
import ask_grid84 as A
import diffusion84 as M
import remask84 as R
import base_law

KIND = {0: "species", 1: "ability", 2: "item", 3: "move", 4: "move", 5: "move", 6: "move", 7: "Stat Alignment"}
def kind(c): return "Stat Points" if c >= M.STAT0 else KIND[c % 8]
KINDS = ["species", "ability", "item", "move", "Stat Alignment", "Stat Points"]
W = None


class TablesRecorder:
    def __init__(self, n, draws=False):  # noqa: ARG002  (diffusion84.Refusals signature)
        self.by_kind = torch.zeros(n, len(KINDS), dtype=torch.float64)
        self.fallback = torch.zeros(n, dtype=torch.long); self.draws = torch.zeros(n, dtype=torch.long)

    def add(self, scores, plain, kept, fallback, rows=None, column=None, x=None):  # noqa: ARG002  (Refusals.add signature)
        st = W.raw_stash; W.raw_stash = []
        cols = [W._column(r, column) for r in st]
        raw = cols[0] if len(cols) == 1 else A.GUIDANCE * cols[0] + (1 - A.GUIDANCE) * cols[1]
        s = raw.detach().cpu().double(); kept, fallback = kept.cpu(), fallback.cpu()
        allv = torch.ones_like(kept); allv[:, 0] = column >= M.STAT0
        legal = torch.where(fallback.unsqueeze(1), allv, kept)
        support = torch.ones_like(kept); support[:, 0] = column >= M.STAT0     # [MASK] is never drawn
        lse = lambda m: torch.logsumexp(s.masked_fill(~m, -float("inf")), 1)
        lk = (lse(legal) - lse(support)).clamp(max=0.0)
        self.by_kind[:, KINDS.index(kind(column))] += lk
        self.fallback += fallback.long(); self.draws += 1

    def records(self):
        return [dict(log_w=float(b.sum()), log_w_by_kind=dict(zip(KINDS, map(float, b))), log_w_lookahead=0.0,
                     fallback=int(f), draws=int(d)) for b, f, d in zip(self.by_kind, self.fallback, self.draws)]


base_law.patch_stash(); M.Refusals = TablesRecorder
stored = json.load(open(A.OUT)); W = A.load_model()
dec = A.MODELS[A.MODEL].get("decoder", "species_first"); assert dec == "species_first", dec
arm = R.SpeciesFirstArm(W)
out = dict(model=A.MODEL, guidance=A.GUIDANCE, decoder=dec, seed0=A.SEED0, source=str(A.OUT), base="network before tables", cells={})
for seed in A.SEEDS:
    for k in A.ASK_NUMERATORS:
        tag = A.tag_of(k, seed); cell = stored["cells"][tag]; W.raw_stash = []
        kept, _stats, counts = A.draw(arm, A.ask_of(k), seed)
        same = kept == cell["pastes"]
        out["cells"][tag] = dict(numerator=k, seed=seed, same_pastes=same, n=len(kept), tried=counts["tried"],
                                 rejected=counts["rejected"], stored_tried=cell["tried"], team_ids=cell.get("team_ids"),
                                 records=counts["refusals"])
        lw = sorted(r["log_w"] for r in counts["refusals"])
        print(f"{A.MODEL} g{A.GUIDANCE:g} {tag}: same={same} kept={len(kept)} median log_w={lw[len(lw)//2]:.3f} "
              f"range {lw[0]:.2f}..{lw[-1]:.2f}", flush=True)
        json.dump(out, open(sys.argv[1], "w"))
