"""The trained network's distribution before the legal Pokémon construction tables: Model84._logprobs with the tables
mask left out (only [MASK] is dropped, as every draw drops it). The tables mask is applied in the training loss too
(diffusion84.elbo_terms), so the network never learns what mass to give a value the tables forbid; this measures how
much it gives anyway.

Runs against the code at commit 787fac31, where the redraws were made: main has since moved the samplers into sampler84.py (#398), and SpeciesFirstArm, RandomDecodingArm and diffusion84.Refusals are gone. To rerun, `git worktree add /tmp/twister 787fac31` and run this file from there."""
import math

import torch
import torch.nn.functional as F

import diffusion84 as M
from diffusion84 import allowed


def logprobs_both(w, x, t, cond):
    """(masked, raw) per field type. masked is Model84._logprobs bit for bit; raw skips the tables mask."""
    al = allowed(w.tables, x)
    masked, raw = [], []
    for net, _ in w.nets:
        lg = net.logits(net(x, t, cond))
        masked.append({k: F.log_softmax(v.masked_fill(~al[k], -1e4).float(), -1) for k, v in lg.items()})
        raw.append({k: F.log_softmax(v.float().index_fill(-1, torch.tensor([0], device=v.device), -1e4) if k != "stat" else v.float(), -1)
                    for k, v in lg.items()})
    if len(w.nets) == 1: return masked[0], raw[0]
    mean = lambda outs: {k: torch.logsumexp(torch.stack([o[k] for o in outs]), 0) - math.log(len(outs)) for k in outs[0]}
    return mean(masked), mean(raw)


def patch_stash():
    """Make Model84._logprobs keep its raw twin in w.raw_stash (a list), returning the masked one unchanged."""
    orig = M.Model84._logprobs

    def _logprobs(self, x, t, w, loops=None, state=None, keep=None):
        if loops is not None or state is not None or keep is not None: return orig(self, x, t, w, loops, state, keep)
        masked, raw = logprobs_both(self, x, t, w)
        self.__dict__.setdefault("raw_stash", []).append(raw)
        return masked
    M.Model84._logprobs = _logprobs
