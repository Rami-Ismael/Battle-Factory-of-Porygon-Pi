"""The 84-field discrete diffusion model (issue #57): the deployed recipe of `lossdown.py` with six Stat Points
per Pokémon added as generated fields, built beside the 48-field code, which stays untouched.

    m = Model84.train_new(V, corpus, rows84, labels, passes=2)
    rows = m.sample(64, ask=0.5, guidance=2.0)          # [64, 84] tokens, legal by construction
    paste, slots = m.to_paste(rows[0])                  # Stat Points written from the row, never copied

What is the same as `lossdown.py`: field embeddings and heads per field type, one position per column, the
ask and time conditioning, slot order shuffled every batch, masking with probability t ** exponent, the ELBO
weighting, ask dropout 0.15, AdamW with warm-up and cosine decay, and the structured-output tables. What is new:
36 Stat Point columns (token = value + 1), exponent 1 like Stat Alignment, decoded after Stat Alignment, and a
budget mask that keeps each Pokémon's Stat Points at most 32 each and 66 in total.
"""
import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import alignment as AL
import asked_vs_got as X
import diffusion as D
import lossdown as L
import stat_points as SP
import team84 as T
from encode import NF, NSLOT
from hps_generate import slot_to_text

DEV = L.DEV
COLS = T.COLS                                          # 84
STAT0 = T.COLS48                                       # first Stat Point column
ORDER = list(D.ORDER) + list(range(STAT0, COLS))       # species first, ..., Stat Alignment, then Stat Points
SCHED = "16,4,4,1.5,1"                                 # deployed exponents: species, ability, item, move, Stat Alignment
STAT_EXPONENT = 1.0                                    # Stat Points masked like Stat Alignment
FIRST_MOVE = 3                                         # field offset of a Pokémon's first move
NAN = float("nan")


def sched84(sched=SCHED):
    return torch.cat([L.sched_tensor(sched), torch.full((SP.COLS,), STAT_EXPONENT)])


class Net84(nn.Module):
    """Transformer denoiser over the 84-field grid."""

    def __init__(self, V, d=256, nhead=8, nlayer=6, dropout=0.1, nfreq=8):
        super().__init__()
        self.emb = nn.ModuleDict({k: nn.Embedding(v, d) for k, v in V.sizes.items()})
        self.head = nn.ModuleDict({k: nn.Linear(d, v) for k, v in V.sizes.items()})
        self.stat_emb = nn.Embedding(SP.NVALUES + 1, d)                  # token 0 = [MASK]
        self.stat_head = nn.Linear(d, SP.NVALUES)                         # one output per value 0..32
        self.pos = nn.Parameter(torch.randn(COLS, d) * .02)
        self.tproj = nn.Sequential(nn.Linear(1, d), nn.SiLU(), nn.Linear(d, d))
        self.register_buffer("freqs", (2.0 ** torch.arange(nfreq)) * math.pi)
        self.wproj = nn.Sequential(nn.Linear(2 * nfreq + 1, d), nn.SiLU(), nn.Linear(d, d))
        self.wnull = nn.Parameter(torch.zeros(d))
        layer = nn.TransformerEncoderLayer(d, nhead, 4 * d, dropout, batch_first=True, norm_first=True, activation="gelu")
        self.tr = nn.TransformerEncoder(layer, nlayer, enable_nested_tensor=False)
        self.ln = nn.LayerNorm(d)

    def wvec(self, w):
        null = torch.isnan(w); z = torch.nan_to_num(w, 0.0).view(-1, 1)
        v = self.wproj(torch.cat([z, torch.sin(z * self.freqs), torch.cos(z * self.freqs)], 1))
        return torch.where(null.view(-1, 1), self.wnull.expand_as(v), v)

    def forward(self, x, t, w):
        B, d = x.shape[0], self.pos.shape[1]
        h = torch.empty(B, COLS, d, device=x.device)
        for k, cols in L.COLS_BY_KEY.items():
            h[:, cols] = self.emb[k](x[:, cols])
        h[:, STAT0:] = self.stat_emb(x[:, STAT0:])
        h = h + self.pos.unsqueeze(0) + self.wvec(w).unsqueeze(1) + self.tproj(t.view(-1, 1).float()).unsqueeze(1)
        return self.ln(self.tr(h))

    def logits(self, h):
        out = {k: self.head[k](h[:, cols]) for k, cols in L.COLS_BY_KEY.items()}
        out["stat"] = self.stat_head(h[:, STAT0:])
        return out


def allowed(tables, x):
    """{field type: bool mask over values} for the visible fields; "stat" is indexed by value, not token."""
    al = tables.allowed(x[:, :STAT0])
    al["stat"] = SP.allowed_stat_points(x[:, STAT0:])
    return al


def shuffle_slots(x):
    """The same random Pokémon order applied to the 48-field block and the Stat Point block."""
    B = x.shape[0]
    ps = torch.rand(B, NSLOT, device=x.device).argsort(1)
    a = torch.gather(x[:, :STAT0].view(B, NSLOT, NF), 1, ps.unsqueeze(-1).expand(-1, -1, NF)).reshape(B, STAT0)
    s = torch.gather(x[:, STAT0:].view(B, NSLOT, SP.NSTAT), 1, ps.unsqueeze(-1).expand(-1, -1, SP.NSTAT))
    return torch.cat([a, s.reshape(B, SP.COLS)], 1)


def per_column_ce(lg, x):
    """Cross-entropy [B, 84] of the true tokens under the logits."""
    ce = torch.zeros(x.shape, device=x.device)
    for k, cols in L.COLS_BY_KEY.items():
        ce[:, cols] = F.cross_entropy(lg[k].float().transpose(1, 2), x[:, cols], reduction="none")
    ce[:, STAT0:] = F.cross_entropy(lg["stat"].float().transpose(1, 2), x[:, STAT0:] - 1, reduction="none")
    return ce


def batch_loss(net, tables, x, w, sched, p_uncond=0.15):
    B = x.shape[0]
    x = shuffle_slots(x)
    t = ((torch.rand(1, device=x.device) + torch.arange(B, device=x.device) / B) % 1.0)[torch.randperm(B, device=x.device)]
    t = t.clamp(min=1e-3)
    m = torch.rand(B, COLS, device=x.device) < t.view(-1, 1) ** sched.view(1, -1)
    ww = w.clone(); ww[torch.rand(B, device=x.device) < p_uncond] = NAN
    xin = x.masked_fill(m, 0)
    al = allowed(tables, xin)
    lg = {k: v.masked_fill(~al[k], -1e4) for k, v in net.logits(net(xin, t, ww)).items()}
    ce = per_column_ce(lg, x) * m
    return ((ce * sched.view(1, -1) / t.view(-1, 1)).sum(1) / COLS).mean()


class Model84:
    """A trained 84-field network plus what sampling and pastes need. Build it with Model84.train_new()."""

    def __init__(self, V, corpus, net):
        self.V, self.net = V, net.eval()
        self.tables = L.Tables(V, corpus)
        _, _, _, look, _ = X.setup()
        look.update(V.regulation["names"])
        self.look = look
        _, _, _, self.modes, self.natures = AL.build_tables(corpus)

    @classmethod
    def train_new(cls, V, corpus, rows, labels=None, passes=2, seed=0, batch=256, lr=5e-4, wd=0.01,
                  warmup=500, ema=0.999, sched=SCHED, log=None, **net_kw):
        """Train from scratch on 84-field rows (labels: win rates, None or NaN = unlabelled)."""
        torch.manual_seed(seed)
        tables = L.Tables(V, corpus)
        X84 = np.asarray(rows, dtype=np.int64)
        keep = ~tables.check(X84[:, :STAT0])                   # rows whose true value the tables exclude are dropped
        X84 = X84[keep]
        W = np.full(len(keep), NAN, np.float32) if labels is None else np.asarray(labels, np.float32)
        x, w = torch.as_tensor(X84, device=DEV), torch.as_tensor(W[keep], device=DEV)
        net = Net84(V, **net_kw).to(DEV)
        avg = copy.deepcopy(net).eval() if ema else None
        sw = sched84(sched).to(DEV)
        steps = math.ceil(len(x) / batch) * passes; warm = min(warmup, max(1, steps // 10))
        opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd, betas=(0.9, 0.98))
        lr_at = lambda s: lr * min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps))
        step = 0
        for p in range(passes):
            net.train(); run = 0.0; nb = 0
            for b in torch.randperm(len(x), device=DEV).split(batch):
                loss = batch_loss(net, tables, x[b], w[b], sw)
                for g in opt.param_groups: g["lr"] = lr_at(step)
                opt.zero_grad(set_to_none=True); loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); step += 1
                if avg is not None:
                    with torch.no_grad():
                        dec = min(ema, (1 + step) / (10 + step))
                        for pe, pn in zip(avg.parameters(), net.parameters()): pe.lerp_(pn, 1 - dec)
                run += float(loss.detach()); nb += 1
            if log is not None: log.append(dict(pass_=p + 1, train_loss=run / nb))
        return cls(V, corpus, avg if avg is not None else net)

    def _logprobs(self, x, t, w):
        al = allowed(self.tables, x)
        lg = self.net.logits(self.net(x, t, w))
        return {k: F.log_softmax(v.masked_fill(~al[k], -1e4).float(), -1) for k, v in lg.items()}

    def _column(self, lp, c):
        if c >= STAT0: return lp["stat"][:, c - STAT0]
        k = L.KEY_OF_COL[c]; return lp[k][:, L.COLS_BY_KEY[k].index(c)]

    def _legal(self, x, c):
        """Boolean [B, outputs] for column c; never empty, never [MASK]."""
        if c >= STAT0: return SP.allowed_stat_points(x[:, STAT0:])[:, c - STAT0]
        k = L.KEY_OF_COL[c]; o = self.tables.allowed(x[:, :STAT0])[k][:, L.COLS_BY_KEY[k].index(c)].clone()
        E = self.tables.EMV
        if c % NF == FIRST_MOVE and E is not None:           # Showdown rejects a Pokémon with no moves
            real = o.clone(); real[:, [0, E]] = False; o[real.any(1), E] = False
        o[~o.any(1)] = True; o[:, 0] = False
        return o

    @torch.no_grad()
    def sample(self, n, ask=None, guidance=1.0, temperature=1.0, seed=None):
        """n teams in ORDER; ask is the win-rate condition, guidance the classifier-free weight."""
        if seed is not None: torch.manual_seed(seed)
        w = torch.full((n,), NAN if ask is None else ask, device=DEV); wn = torch.full((n,), NAN, device=DEV)
        x = torch.zeros(n, COLS, dtype=torch.long, device=DEV)
        for step, c in enumerate(ORDER):
            t = torch.full((n,), 1.0 - step / COLS, device=DEV)
            lc = self._column(self._logprobs(x, t, w), c)
            if ask is not None and guidance != 1.0:
                lc = guidance * lc + (1 - guidance) * self._column(self._logprobs(x, t, wn), c)
            lc = lc.masked_fill(~self._legal(x, c), -float("inf")) / temperature
            v = torch.multinomial(torch.softmax(lc, -1), 1).squeeze(1)
            x[:, c] = v + 1 if c >= STAT0 else v                 # a Stat Point output index is its value
        return x

    def to_paste(self, row, repair=False):
        """(Showdown paste, slot dicts) with the row's own Stat Points; repair=True aligns the Stat Alignment."""
        slots = []
        for s in T.decode(self.V, row):
            g = lambda v: self.look.get(v, v)
            slot = dict(species=g(s["species"]), ability=g(s["ability"]), item=g(s["item"]), nature=g(s["nature"]),
                        moves=[g(m) for m in s["moves"]], evs=s["evs"])
            if repair: slot["nature"] = AL.repair_nature(slot, self.modes, self.natures)
            slots.append(slot)
        return "\n\n".join(slot_to_text(s) for s in slots) + "\n", slots
