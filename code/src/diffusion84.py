"""The 84-field discrete diffusion model (issue #57): the deployed recipe of `lossdown.py` with six Stat Points
per Pokémon added as generated fields, built beside the 48-field code, which stays untouched.

    m = Model84.train_new(V, corpus, rows84, labels, passes=2)
    rows = m.sample(64, ask=0.5, guidance=2.0)          # [64, 84] tokens, legal by construction
    paste, slots = m.to_paste(rows[0])                  # Stat Points written from the row, never copied
    m = Model84.train_new(..., passes=80, val=(rows, labels), patience=3)   # early stopping, best pass kept
    both = Model84.mixture([a, b]); both.save(path); Model84.load(path)     # log-mean of the members, like team_writer

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
from label_noise import noisy_labels
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

    def __init__(self, V, d=256, nhead=8, nlayer=6, dropout=0.1, nfreq=8, mrow=0):
        super().__init__()
        self.mrow = mrow
        if mrow:                # the team's matchup row: win rate vs each top-50 meta team, NaN = untested
            self.mproj = nn.Sequential(nn.Linear(2 * mrow, d), nn.SiLU(), nn.Linear(d, d))
            self.mnull = nn.Parameter(torch.zeros(d))
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

    def mvec(self, r):
        has = ~torch.isnan(r)
        v = self.mproj(torch.cat([torch.nan_to_num(r, 0.0), has.float()], 1))
        return torch.where(has.any(1, keepdim=True), v, self.mnull.expand_as(v))

    def forward(self, x, t, w):
        """w: [B] win rates, or [B, 1 + mrow] = [win rate, matchup row] (NaN = no condition)."""
        B, d = x.shape[0], self.pos.shape[1]
        r = None
        if w.dim() == 2:
            r, w = w[:, 1:], w[:, 0]
        if self.mrow and r is None: r = torch.full((B, self.mrow), NAN, device=x.device)
        h = torch.empty(B, COLS, d, device=x.device)
        for k, cols in L.COLS_BY_KEY.items():
            h[:, cols] = self.emb[k](x[:, cols])
        h[:, STAT0:] = self.stat_emb(x[:, STAT0:])
        h = h + self.pos.unsqueeze(0) + self.wvec(w).unsqueeze(1) + self.tproj(t.view(-1, 1).float()).unsqueeze(1)
        if self.mrow: h = h + self.mvec(r).unsqueeze(1)
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


def shuffle_slots(x, ps=None):
    """The same Pokémon order (random unless ps [B, 6] is given) applied to the 48-field block and the Stat Points."""
    B = x.shape[0]
    if ps is None: ps = torch.rand(B, NSLOT, device=x.device).argsort(1)
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


def elbo_terms(net, tables, x, m, t, w, sched):
    """Per-team ELBO [B] of the true tokens x with the fields in m masked, legality-masked as in sampling."""
    xin = x.masked_fill(m, 0)
    al = allowed(tables, xin)
    lg = {k: v.masked_fill(~al[k], -1e4) for k, v in net.logits(net(xin, t, w)).items()}
    ce = per_column_ce(lg, x) * m
    return (ce * sched.view(1, -1) / t.view(-1, 1)).sum(1) / COLS


def batch_loss(net, tables, x, w, sched, p_uncond=0.15):
    B = x.shape[0]
    x = shuffle_slots(x)
    t = ((torch.rand(1, device=x.device) + torch.arange(B, device=x.device) / B) % 1.0)[torch.randperm(B, device=x.device)]
    t = t.clamp(min=1e-3)
    m = torch.rand(B, COLS, device=x.device) < t.view(-1, 1) ** sched.view(1, -1)
    ww = w.clone(); ww[torch.rand(B, device=x.device) < p_uncond] = NAN
    if ww.dim() == 2:                                          # matchup row: dropped whole, or column by column (lossdown)
        ww[torch.rand(B, device=x.device) < 0.3, 1:] = NAN
        ww[:, 1:][torch.rand(B, ww.shape[1] - 1, device=x.device) < 0.2] = NAN
    return elbo_terms(net, tables, x, m, t, ww, sched).mean()


@torch.no_grad()
def val_loss(net, tables, x, w, sched, reps=4, seed=12345, batch=512):
    """Mean ELBO on fixed teams with common random numbers: the same slot orders, times and masks on every call,
    so the curve moves only when the model does. t is stratified per team across reps."""
    g = torch.Generator().manual_seed(seed)
    xs, ws, sc = torch.as_tensor(np.asarray(x, np.int64)), torch.as_tensor(np.asarray(w, np.float32)), sched.cpu()
    N, tot = len(xs), 0.0
    for r in range(reps):
        ps = torch.rand(N, NSLOT, generator=g).argsort(1)
        tt = ((r + torch.rand(N, generator=g)) / reps).clamp(min=1e-3)
        u = torch.rand(N, COLS, generator=g)
        for i in range(0, N, batch):
            sl = slice(i, i + batch)
            m = (u[sl] < tt[sl].view(-1, 1) ** sc.view(1, -1)).to(DEV)
            xb = shuffle_slots(xs[sl], ps[sl]).to(DEV)
            tot += float(elbo_terms(net, tables, xb, m, tt[sl].to(DEV), ws[sl].to(DEV), sched).sum())
    return tot / (N * reps)


class Model84:
    """One or more trained 84-field networks (a mixture samples from the log-mean of its members, like team_writer)
    plus what sampling and pastes need. Build one with Model84.train_new(), Model84.mixture() or Model84.load()."""

    def __init__(self, V, corpus, nets, info=None):
        self.V, self.nets, self.info = V, [(net.eval(), kw) for net, kw in nets], info or {}
        self.tables = L.Tables(V, corpus)
        _, _, _, look, _ = X.setup()
        look.update(V.regulation["names"])
        self.look = look
        _, _, _, self.modes, self.natures = AL.build_tables(corpus)

    @property
    def net(self):
        """The single network of a one-member model."""
        assert len(self.nets) == 1, "a mixture has several networks"
        return self.nets[0][0]

    @classmethod
    def train_new(cls, V, corpus, rows, labels=None, passes=2, seed=0, batch=256, lr=4e-4, wd=0.01,  # noqa: C901
                  warmup=500, ema=0.999, sched=SCHED, log=None, val=None, patience=None, on_best=None, label_noise=0.0, stop_after=None, **net_kw):
        """Train from scratch on 84-field rows. labels: [N] win rates or [N, 1 + mrow] with matchup rows (NaN =
        unlabelled); pass mrow=<columns> in net_kw for the latter. lr 4e-4 is the deployed members' rate.
        val=(rows, labels): validation loss after every pass (common random numbers). patience=P: stop after P
        passes with no new lowest validation loss and keep the best pass's weights; passes is then the cap, and
        the learning-rate schedule is sized to it. on_best(model, pass_number) runs at every new lowest.
        label_noise=W: each batch's labels get fresh uniform noise in [-W, +W]; validation labels stay exact.
        stop_after=N: end after N passes while keeping the schedule sized to passes, so a shorter run matches a stopped one."""
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
        info = dict(n_train=len(x), dropped_by_tables=int((~keep).sum()), cap=passes, patience=patience,
                    lr=lr, batch=batch, seed=seed, sched=sched, net_kw=net_kw)
        info.update(label_noise=label_noise, stop_after=stop_after)
        noise_rng = torch.Generator().manual_seed(seed + 1)
        log = [] if log is None else log
        best, best_pass, best_sd, step = math.inf, 0, None, 0
        for p in range(passes):
            net.train(); run = 0.0; nb = 0
            for b in torch.randperm(len(x), device=DEV).split(batch):
                loss = batch_loss(net, tables, x[b], noisy_labels(w[b], label_noise, noise_rng), sw)
                for g in opt.param_groups: g["lr"] = lr_at(step)
                opt.zero_grad(set_to_none=True); loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); step += 1
                if avg is not None:
                    with torch.no_grad():
                        dec = min(ema, (1 + step) / (10 + step))
                        for pe, pn in zip(avg.parameters(), net.parameters()): pe.lerp_(pn, 1 - dec)
                run += float(loss.detach()); nb += 1
            m = avg if avg is not None else net
            m.eval()
            entry = dict(pass_=p + 1, train_loss=run / nb)
            if val is not None:
                entry["val_loss"] = val_loss(m, tables, val[0], val[1], sw)
                if entry["val_loss"] < best:
                    best, best_pass = entry["val_loss"], p + 1
                    best_sd = {k: v.detach().clone() for k, v in m.state_dict().items()}
                    if on_best is not None: on_best(cls(V, corpus, [(m, net_kw)], dict(info, best_pass=best_pass)), best_pass)
            log.append(entry)
            print("  pass " + " · ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}" for k, v in entry.items()), flush=True)
            if patience and p + 1 - best_pass >= patience: break
            if stop_after and p + 1 >= stop_after: break
        m = avg if avg is not None else net
        if patience and best_sd is not None: m.load_state_dict(best_sd)
        info.update(stopped_pass=len(log), best_pass=best_pass or None, best_val_loss=None if best == math.inf else best,
                    curves=log)
        return cls(V, corpus, [(m, net_kw)], info)

    @classmethod
    def mixture(cls, models, info=None):
        """One model sampling from the log-mean of the members' column distributions (lossdown.ensemble_fn)."""
        m0 = models[0]
        out = cls.__new__(cls)
        out.V, out.tables, out.look, out.modes, out.natures = m0.V, m0.tables, m0.look, m0.modes, m0.natures
        out.nets = [nk for m in models for nk in m.nets]
        out.info = info or dict(members=[m.info for m in models])
        return out

    def save(self, path):
        """Every member's weights and network settings, the regulation snapshot (vocabulary) and the run info."""
        torch.save(dict(members=[dict(sd=net.state_dict(), net_kw=kw) for net, kw in self.nets],
                        regulation=self.V.regulation, info=self.info), path)

    @classmethod
    def load(cls, path, corpus=None):
        from checkpoints import load_checkpoint
        from encode import Vocab
        ck = load_checkpoint(path, map_location="cpu")
        V = Vocab.from_regulation(ck["regulation"])
        corpus = X.setup()[0] if corpus is None else corpus
        nets = []
        for mem in ck["members"]:
            net = Net84(V, **mem["net_kw"]); net.load_state_dict(mem["sd"]); nets.append((net.to(DEV), mem["net_kw"]))
        return cls(V, corpus, nets, ck.get("info"))

    def _logprobs(self, x, t, w):
        al = allowed(self.tables, x)
        outs = []
        for net, _ in self.nets:
            lg = net.logits(net(x, t, w))
            outs.append({k: F.log_softmax(v.masked_fill(~al[k], -1e4).float(), -1) for k, v in lg.items()})
        if len(outs) == 1: return outs[0]
        return {k: torch.logsumexp(torch.stack([o[k] for o in outs]), 0) - math.log(len(outs)) for k in outs[0]}

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
