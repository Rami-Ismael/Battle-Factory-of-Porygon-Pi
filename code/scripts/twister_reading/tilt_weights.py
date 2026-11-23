"""TWISTER's correction weight for every team of the #297, #322 and #335 ask grids at asks 13/24 to 17/24 (sampling only,
no battles). Redraws each stored cell with its own seed through ask_grid84.draw, checks that the kept pastes are the
stored ones bit for bit, and records for each kept team the log of the probability share the sampler removed from the
trained model's distribution at each draw: the values [MASK], the empty first move, the legal Pokemon construction
tables applied inside a pass and the lookahead took away. exp(sum) is the team's probability under the trained model's
per-pass distributions over its probability under the sampler along the same path: TWISTER's incremental weights
multiplied out (arXiv 2609.35609, Proposition re-freezing tilt, problem-trajectory-bias.tex:134-152), with the
lookahead and tables in the place of its automaton.

Usage: AG84_MODEL=<row> AG84_GUIDANCE=<g> AG84_ASKS=13,14,15,16,17 python tilt_weights.py <out.json>

Runs against the code at commit 787fac31, where the redraws were made: main has since moved the samplers into sampler84.py (#398), and SpeciesFirstArm, RandomDecodingArm and diffusion84.Refusals are gone. To rerun, `git worktree add /tmp/twister 787fac31` and run this file from there."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import torch
import ask_grid84 as A
import diffusion84 as M
import remask84 as R

KIND = {0: "species", 1: "ability", 2: "item", 3: "move", 4: "move", 5: "move", 6: "move", 7: "Stat Alignment"}
def kind(c): return "Stat Points" if c >= M.STAT0 else KIND[c % 8]
KINDS = ["species", "ability", "item", "move", "Stat Alignment", "Stat Points"]


class TiltRecorder:
    """Drop-in for diffusion84.Refusals. Per team: log kept mass summed over draws, split by field type, and by rule
    (lookahead alone: plain against kept, as Refusals counts it), plus fallback draws."""
    def __init__(self, n, draws=False):  # noqa: ARG002  (diffusion84.Refusals signature)
        self.n = n
        self.by_kind = torch.zeros(n, len(KINDS), dtype=torch.float64)
        self.look = torch.zeros(n, dtype=torch.float64)
        self.fallback = torch.zeros(n, dtype=torch.long)
        self.draws = torch.zeros(n, dtype=torch.long)

    def add(self, scores, plain, kept, fallback, rows=None, column=None, x=None):  # noqa: ARG002  (Refusals.add signature)
        s = scores.detach().cpu().double(); plain, kept, fallback = plain.cpu(), kept.cpu(), fallback.cpu()
        rows = torch.ones(len(s), dtype=torch.bool) if rows is None else rows.cpu()
        allv = torch.ones_like(kept); allv[:, 0] = False
        legal = torch.where(fallback.unsqueeze(1), allv, kept)          # what the draw used (diffusion84._legal_parts)
        lse = lambda m: torch.logsumexp(s.masked_fill(~m, -float("inf")), 1)
        tot = torch.logsumexp(s, 1)
        lk = (lse(legal) - tot).clamp(max=0.0)
        lk_look = torch.where(fallback, torch.zeros_like(lk), (lse(kept) - lse(plain)).clamp(max=0.0))
        z = torch.zeros_like(lk)
        self.by_kind[:, KINDS.index(kind(column))] += torch.where(rows, lk, z)
        self.look += torch.where(rows, lk_look, z)
        self.fallback += (fallback & rows).long(); self.draws += rows.long()

    def records(self):
        return [dict(log_w=float(b.sum()), log_w_by_kind=dict(zip(KINDS, map(float, b))), log_w_lookahead=float(l),
                     fallback=int(f), draws=int(d))
                for b, l, f, d in zip(self.by_kind, self.look, self.fallback, self.draws)]


M.Refusals = TiltRecorder
stored = json.load(open(A.OUT))
w = A.load_model()
dec = A.MODELS[A.MODEL].get("decoder", "species_first")
arm = R.SpeciesFirstArm(w) if dec == "species_first" else A.load_sampler(w)
out = dict(model=A.MODEL, guidance=A.GUIDANCE, decoder=dec, seed0=A.SEED0, source=str(A.OUT), cells={})
for seed in A.SEEDS:
    for k in A.ASK_NUMERATORS:
        tag = A.tag_of(k, seed); cell = stored["cells"][tag]
        kept, _stats, counts = A.draw(arm, A.ask_of(k), seed)
        same = kept == cell["pastes"]
        out["cells"][tag] = dict(numerator=k, seed=seed, same_pastes=same, n=len(kept), tried=counts["tried"],
                                 rejected=counts["rejected"], stored_tried=cell["tried"], team_ids=cell.get("team_ids"),
                                 records=counts["refusals"])
        lw = [r["log_w"] for r in counts["refusals"]]
        print(f"{A.MODEL} g{A.GUIDANCE:g} {tag}: same={same} kept={len(kept)} tried={counts['tried']}/{cell['tried']} "
              f"median log_w={sorted(lw)[len(lw)//2]:.3f}", flush=True)
        json.dump(out, open(sys.argv[1], "w"))
