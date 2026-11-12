"""The 84-field discrete diffusion model (issue #57): the deployed recipe of `lossdown.py` with six Stat Points
per Pokémon added as generated fields, built beside the 48-field code, which stays untouched.

    m = Model84.train_new(V, corpus, rows84, labels, epochs=2)
    rows = m.sample(64, ask=0.5, guidance=2.0)          # [64, 84] tokens, legal by construction
    paste, slots = m.to_paste(rows[0])                  # Stat Points written from the row, never copied
    m = Model84.train_new(..., epochs=80, val=(rows, labels), patience=3)   # early stopping, best epoch kept
    both = Model84.mixture([a, b]); both.save(path); Model84.load(path)     # log-mean of the members, like team_writer

What is the same as `lossdown.py`: field embeddings and heads per field type, one position per column, the
ask and time conditioning, slot order shuffled every batch, masking with probability t ** exponent, the ELBO
weighting, ask dropout 0.15, AdamW with warm-up and cosine decay, and the structured-output tables. What is new:
36 Stat Point columns (token = value + 1), exponent 1 like Stat Alignment, decoded after Stat Alignment, and a
budget mask that keeps each Pokémon's Stat Points at most 32 each and 66 in total.
"""
import copy
import time
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import alignment as AL
import asked_vs_got as X
import diffusion as D
import lossdown as L
from lookahead import lookahead as lookahead_rule
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
CARRY_P = 0.9                                          # share of training steps that run the two-pass carried-state loss "p = 0.9 by default" for Loopholing Discrete Diffusion Models (Loopholing All the Way, Deschenaux et al. 2026, §2.2)


POSITION_OPTIONS = ("per_field", "slot_learned", "slot_fixed", "no_slot")   # issue #81: how a column's position vector is built
_COL = torch.arange(COLS)
_STAT = (_COL - STAT0).clamp(min=0)
COLUMN_TYPE = torch.where(_COL < STAT0, _COL % NF, NF + _STAT % SP.NSTAT)     # 14 field types: 8 fields, then 6 Stat Points
COLUMN_SLOT = torch.where(_COL < STAT0, _COL // NF, _STAT // SP.NSTAT)        # team position 1 to 6, counted from 0
NTYPE = NF + SP.NSTAT


def sched84(sched=SCHED):
    return torch.cat([L.sched_tensor(sched), torch.full((SP.COLS,), STAT_EXPONENT)])


class Net84(nn.Module):
    """Transformer denoiser over the 84-field grid."""

    def __init__(self, V, d=256, nhead=8, nlayer=6, dropout=0.1, nfreq=8, mrow=0, pos="per_field",
                 loop_start=1, loop_layers=0, loop_max=1, carry=False):
        """Two looping options, both off by default (docs/looped-84-fields.md):
        loop_layers=n, loop_start=a, loop_max=S: layers a..a+n-1 form a block run S times per pass with shared weights
            (LoopMDM, Lee et al. 2026, Eq. 4); training draws the count uniformly from 1..S each step (Eq. 5), and a pass
            runs S times unless told otherwise. The parameters are those of the plain network.
        carry=True: the hidden state a pass returns can be handed to the next pass, which adds its layer norm to the
            field embeddings (Loopholing, Jo et al. 2026, Eq. 5 and Algorithm 1). Adds one LayerNorm."""
        super().__init__()
        if pos not in POSITION_OPTIONS: raise ValueError(f"pos={pos!r}; choose one of {POSITION_OPTIONS}")
        if loop_layers and not (0 <= loop_start and loop_start + loop_layers <= nlayer and loop_max >= 1):
            raise ValueError(f"loop block {loop_start}..{loop_start + loop_layers - 1} x {loop_max} does not fit {nlayer} layers")
        self.mrow, self.pos_option = mrow, pos
        self.loop_start, self.loop_layers, self.loop_max, self.carry = loop_start, loop_layers, loop_max, carry
        if mrow:                # the team's matchup row: win rate vs each top-50 meta team, NaN = untested
            self.mproj = nn.Sequential(nn.Linear(2 * mrow, d), nn.SiLU(), nn.Linear(d, d))
            self.mnull = nn.Parameter(torch.zeros(d))
        self.emb = nn.ModuleDict({k: nn.Embedding(v, d) for k, v in V.sizes.items()})
        self.head = nn.ModuleDict({k: nn.Linear(d, v) for k, v in V.sizes.items()})
        self.stat_emb = nn.Embedding(SP.NVALUES + 1, d)                  # token 0 = [MASK]
        self.stat_head = nn.Linear(d, SP.NVALUES)                         # one output per value 0..32
        if pos == "per_field":
            self.pos = nn.Parameter(torch.randn(COLS, d) * .02)
        else:
            self.pos_type = nn.Parameter(torch.randn(NTYPE, d) * .02)
            if pos == "slot_learned": self.pos_slot = nn.Parameter(torch.randn(NSLOT, d) * .02)
            if pos == "slot_fixed": self.register_buffer("pos_slot", torch.randn(NSLOT, d) * .02)   # saved, never trained
            self.register_buffer("column_type", COLUMN_TYPE.clone(), persistent=False)
            self.register_buffer("column_slot", COLUMN_SLOT.clone(), persistent=False)
        self.tproj = nn.Sequential(nn.Linear(1, d), nn.SiLU(), nn.Linear(d, d))
        self.register_buffer("freqs", (2.0 ** torch.arange(nfreq)) * math.pi)
        self.wproj = nn.Sequential(nn.Linear(2 * nfreq + 1, d), nn.SiLU(), nn.Linear(d, d))
        self.wnull = nn.Parameter(torch.zeros(d))
        layer = nn.TransformerEncoderLayer(d, nhead, 4 * d, dropout, batch_first=True, norm_first=True, activation="gelu")
        self.tr = nn.TransformerEncoder(layer, nlayer, enable_nested_tensor=False)
        self.ln = nn.LayerNorm(d)
        if carry: self.carry_ln = nn.LayerNorm(d)

    def wvec(self, w):
        null = torch.isnan(w); z = torch.nan_to_num(w, 0.0).view(-1, 1)
        v = self.wproj(torch.cat([z, torch.sin(z * self.freqs), torch.cos(z * self.freqs)], 1))
        return torch.where(null.view(-1, 1), self.wnull.expand_as(v), v)

    def mvec(self, r):
        has = ~torch.isnan(r)
        v = self.mproj(torch.cat([torch.nan_to_num(r, 0.0), has.float()], 1))
        return torch.where(has.any(1, keepdim=True), v, self.mnull.expand_as(v))

    def position_vectors(self):
        """[84, d]: the vector added to each column, built as the position option says."""
        if self.pos_option == "per_field": return self.pos
        v = self.pos_type[self.column_type]
        return v if self.pos_option == "no_slot" else v + self.pos_slot[self.column_slot]

    def forward(self, x, t, w, loops=None, state=None):
        """w: [B] win rates, or [B, 1 + mrow] = [win rate, matchup row] (NaN = no condition).
        loops: how many times the loop block runs (default loop_max); state: [B, 84, d] carried hidden state (default zeros)."""
        pos = self.position_vectors()
        B, d = x.shape[0], pos.shape[1]
        r = None
        if w.dim() == 2:
            r, w = w[:, 1:], w[:, 0]
        if self.mrow and r is None: r = torch.full((B, self.mrow), NAN, device=x.device)
        h = torch.empty(B, COLS, d, device=x.device)
        for k, cols in L.COLS_BY_KEY.items():
            h[:, cols] = self.emb[k](x[:, cols])
        h[:, STAT0:] = self.stat_emb(x[:, STAT0:])
        h = h + pos.unsqueeze(0) + self.wvec(w).unsqueeze(1) + self.tproj(t.view(-1, 1).float()).unsqueeze(1)
        if self.mrow: h = h + self.mvec(r).unsqueeze(1)
        if self.carry: h = h + self.carry_ln(torch.zeros_like(h) if state is None else state)
        if not self.loop_layers: return self.ln(self.tr(h))
        a, b = self.loop_start, self.loop_start + self.loop_layers
        layers = list(self.tr.layers)
        for layer in layers[:a]: h = layer(h)
        for _ in range(self.loop_max if loops is None else loops):
            for layer in layers[a:b]: h = layer(h)
        for layer in layers[b:]: h = layer(h)
        return self.ln(h)

    def logits(self, h):
        out = {k: self.head[k](h[:, cols]) for k, cols in L.COLS_BY_KEY.items()}
        out["stat"] = self.stat_head(h[:, STAT0:])
        return out


def pass_macs(V, d=256, nlayer=6, loop_layers=0, loop_max=1, loops=None, **_):
    """Multiply-adds of one forward pass for one team: each layer's projections and feed-forward (12 d^2 per field) and
    attention (2 x 84 x d per field), run once, or loops times for the loop block, plus the output heads."""
    layer = COLS * (12 * d * d + 2 * COLS * d)
    heads = d * (sum(V.sizes[L.KEY_OF_COL[c]] for c in range(STAT0)) + SP.NVALUES * (COLS - STAT0))
    runs = nlayer + ((loop_max if loops is None else loops) - 1) * loop_layers
    return runs * layer + heads


def training_macs(V, carry=False, loop_layers=0, loop_max=1, **net_kw):
    """Mean multiply-adds of one training step for one team, with the backward pass counted as twice the forward
    (the 6ND rule LoopMDM Eq. 6 follows). A loop block averages over the uniform count 1..loop_max (Eq. 6); a
    carried-state network adds a forward pass without gradient on CARRY_P of its steps (Loopholing All the Way, Appendix B.1,
    matches compute this way)."""
    mean = sum(pass_macs(V, loop_layers=loop_layers, loop_max=loop_max, loops=s, **net_kw) for s in range(1, loop_max + 1)) / loop_max
    return 3 * mean + (CARRY_P * mean if carry else 0.0)


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


def elbo_terms(net, tables, x, m, t, w, sched, loops=None, self_condition=None):
    """Per-team ELBO [B] of the true tokens x with the fields in m masked, legality-masked as in sampling.
    loops: loop-block count (default the network's loop_max). self_condition (carried-state networks; default True):
    a first pass with a zero state, without gradient, hands its hidden state to the scored second pass (Loopholing §3.2,
    Algorithm 3); False scores one pass with a zero state."""
    xin = x.masked_fill(m, 0)
    al = allowed(tables, xin)
    state = None
    if getattr(net, "carry", False) and (self_condition is None or self_condition):
        with torch.no_grad(): state = net(xin, t, w, loops=loops).detach()
    kw = {} if loops is None and state is None else dict(loops=loops, state=state)
    lg = {k: v.masked_fill(~al[k], -1e4) for k, v in net.logits(net(xin, t, w, **kw)).items()}
    ce = per_column_ce(lg, x) * m
    return (ce * sched.view(1, -1) / t.view(-1, 1)).sum(1) / COLS


def batch_loss(net, tables, x, w, sched, p_uncond=0.15, row_w=None):
    B = x.shape[0]
    x = shuffle_slots(x)
    t = ((torch.rand(1, device=x.device) + torch.arange(B, device=x.device) / B) % 1.0)[torch.randperm(B, device=x.device)]
    t = t.clamp(min=1e-3)
    m = torch.rand(B, COLS, device=x.device) < t.view(-1, 1) ** sched.view(1, -1)
    ww = w.clone(); ww[torch.rand(B, device=x.device) < p_uncond] = NAN
    if ww.dim() == 2:                                          # matchup row: dropped whole, or column by column (lossdown)
        ww[torch.rand(B, device=x.device) < 0.3, 1:] = NAN
        ww[:, 1:][torch.rand(B, ww.shape[1] - 1, device=x.device) < 0.2] = NAN
    kw = {}                                                     # drawn only for looping networks, so the plain recipe's random stream is unchanged
    if getattr(net, "loop_layers", 0): kw["loops"] = int(torch.randint(1, net.loop_max + 1, (1,)))          # LoopMDM Eq. 5
    if getattr(net, "carry", False): kw["self_condition"] = bool(torch.rand(1) < CARRY_P)                 # Loopholing §3.2, probability p
    terms = elbo_terms(net, tables, x, m, t, ww, sched, **kw)
    return (terms * row_w).mean() if row_w is not None else terms.mean()


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


def _training_tensors(tables, rows, labels, row_weights=None):
    """Rows the tables accept, their labels (NaN = unlabelled) and optional weights averaging 1, on the device.
    Rows whose true value the tables exclude are dropped; keep marks the rows kept."""
    X84 = np.asarray(rows, dtype=np.int64)
    keep = ~tables.check(X84[:, :STAT0])
    W = np.full(len(keep), NAN, np.float32) if labels is None else np.asarray(labels, np.float32)
    x, w = torch.as_tensor(X84[keep], device=DEV), torch.as_tensor(W[keep], device=DEV)
    rw = None
    if row_weights is not None:
        kept_w = np.asarray(row_weights, np.float32)[keep]; rw = torch.as_tensor(kept_w / kept_w.mean(), device=DEV)
    return x, w, rw, keep


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
    def train_new(cls, V, corpus, rows, labels=None, epochs=2, seed=0, batch=256, lr=4e-4, wd=0.01,
                  warmup=500, ema=0.999, sched=SCHED, log=None, val=None, patience=None, early_stopping_threshold=0.001, on_best=None, label_noise=0.0, stop_after=None, on_epoch=None, row_weights=None, labels_each_epoch=None, **net_kw):
        """Train from scratch on 84-field rows. labels: [N] win rates or [N, 1 + mrow] with matchup rows (NaN =
        unlabelled); pass mrow=<columns> in net_kw for the latter. lr 4e-4 is the deployed members' rate.
        val=(rows, labels): validation loss after every epoch (common random numbers). patience=P: stop after P
        epochs with no new lowest validation loss and keep the best epoch's weights; epochs is then the cap, and
        the learning-rate schedule is sized to it. on_best(model, epoch) runs at every new lowest.
        early_stopping_threshold=T: as in Hugging Face's EarlyStoppingCallback, only an epoch whose validation loss
        beats the lowest so far by more than T resets the patience count; a smaller gain still becomes the kept best epoch.
        label_noise=W: each batch's labels get fresh uniform noise in [-W, +W]; validation labels stay exact.
        stop_after=N: end after N epochs while keeping the schedule sized to epochs, so a shorter run matches a stopped one.
        on_epoch(model, epoch): runs after every epoch, on the weights a checkpoint would save.
        row_weights: one training weight per row of rows; validation loss stays unweighted.
        labels_each_epoch(e): [N] labels for rows, drawn fresh before epoch e (0-based); replaces labels from then on."""
        torch.manual_seed(seed)
        tables = L.Tables(V, corpus)
        x, w, rw, keep = _training_tensors(tables, rows, labels, row_weights)
        net = Net84(V, **net_kw).to(DEV)
        info = dict(n_train=len(x), dropped_by_tables=int((~keep).sum()), cap=epochs, patience=patience,
                    early_stopping_threshold=early_stopping_threshold, lr=lr, batch=batch, seed=seed, sched=sched, net_kw=net_kw)
        info.update(label_noise=label_noise, stop_after=stop_after, labels_each_epoch=labels_each_epoch is not None)
        relabel = None if labels_each_epoch is None else \
            (lambda e: torch.as_tensor(np.asarray(labels_each_epoch(e), np.float32)[keep], device=DEV))
        return cls._fit(V, corpus, tables, net, x, w, rw, info, epochs=epochs, seed=seed, batch=batch, lr=lr, wd=wd, warmup=warmup,
                        ema=ema, sched=sched, log=log, val=val, patience=patience, threshold=early_stopping_threshold, on_best=on_best,
                        label_noise=label_noise, stop_after=stop_after, on_epoch=on_epoch, net_kw=net_kw, relabel=relabel)

    def fine_tuned(self, rows, labels, epochs, val, patience=3, early_stopping_threshold=0.001, lr=1e-4, label_noise=0.0, seed=0, batch=256, wd=0.01,
                   warmup=500, ema=0.999, sched=SCHED, watch=None, log=None):
        """A new model: this one-member model's weights trained further on 84-field rows; self is unchanged.
        The recipe is train_new's (AdamW, warm-up and cosine decay sized to the cap of `epochs`, EMA, ask dropout,
        label noise on training labels only). Validation loss on val=(rows, labels) is measured before the first epoch
        too, so epoch 0 is the starting weights: if no epoch beats them, early stopping returns them unchanged.
        early_stopping_threshold: the smallest validation-loss gain that resets the patience count, as in train_new.
        watch={name: (rows, labels)}: more validation sets, logged every epoch and never used to stop."""
        net_kw = self.nets[0][1]
        start = copy.deepcopy(self.net).to(DEV)
        torch.manual_seed(seed)
        x, w, rw, keep = _training_tensors(self.tables, rows, labels)
        info = dict(base=self.info, fine_tune=True, n_train=len(x), dropped_by_tables=int((~keep).sum()), cap=epochs,
                    patience=patience, early_stopping_threshold=early_stopping_threshold, lr=lr, batch=batch, seed=seed, sched=sched, net_kw=net_kw, label_noise=label_noise, stop_after=None)
        return type(self)._fit(self.V, None, self.tables, start, x, w, rw, info, epochs=epochs, seed=seed, batch=batch, lr=lr, wd=wd,
                               warmup=warmup, ema=ema, sched=sched, log=log, val=val, patience=patience, threshold=early_stopping_threshold,
                               on_best=None, label_noise=label_noise, stop_after=None, on_epoch=None, net_kw=net_kw, watch=watch, epoch0=True,
                               like=self)

    @classmethod
    def _fit(cls, V, corpus, tables, net, x, w, rw, info, *, epochs, seed, batch, lr, wd, warmup, ema, sched, log, val,  # noqa: C901
             patience, threshold, on_best, label_noise, stop_after, on_epoch, net_kw, watch=None, epoch0=False, like=None, relabel=None):
        """The training loop train_new and fine_tuned share. epoch0=True logs validation loss before any step and
        lets those starting weights be the best epoch. threshold: the validation-loss gain that resets patience. like: a model whose tables and vocabulary the result reuses.
        relabel(e): the kept rows' labels for epoch e (0-based), replacing w before that epoch's steps."""
        make = (lambda m, extra: like._with([(m, net_kw)], dict(info, **extra))) if like is not None else \
               (lambda m, extra: cls(V, corpus, [(m, net_kw)], dict(info, **extra)))
        avg = copy.deepcopy(net).eval() if ema else None
        sw = sched84(sched).to(DEV)
        steps = math.ceil(len(x) / batch) * epochs; warm = min(warmup, max(1, steps // 10))
        opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd, betas=(0.9, 0.98))
        lr_at = lambda s: lr * min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps))
        noise_rng = torch.Generator().manual_seed(seed + 1)
        log = [] if log is None else log
        best, best_epoch, best_sd, step = math.inf, 0, None, 0
        reset_epoch = 0                                   # last epoch that beat the lowest loss by more than threshold
        t_start = time.perf_counter()

        def measure(m, entry):
            if val is not None:
                entry["val_loss"] = val_loss(m, tables, val[0], val[1], sw)
            for name, (vx, vw) in (watch or {}).items():
                entry[f"{name}_val_loss"] = val_loss(m, tables, vx, vw, sw)

        if epoch0:
            m = (avg if avg is not None else net).eval(); entry = dict(epoch=0); measure(m, entry)
            if val is not None:
                best, best_sd = entry["val_loss"], {k: v.detach().clone() for k, v in m.state_dict().items()}
            entry["minutes"] = 0.0; log.append(entry)
            print("  " + " · ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}" for k, v in entry.items()), flush=True)
        for p in range(epochs):
            if relabel is not None:
                w = relabel(p)
            net.train(); run = 0.0; nb = 0
            for b in torch.randperm(len(x), device=DEV).split(batch):
                loss = batch_loss(net, tables, x[b], noisy_labels(w[b], label_noise, noise_rng), sw, row_w=None if rw is None else rw[b])
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
            entry = dict(epoch=p + 1, train_loss=run / nb)
            measure(m, entry)
            if val is not None and entry["val_loss"] < best:
                if best - entry["val_loss"] > threshold: reset_epoch = p + 1
                best, best_epoch = entry["val_loss"], p + 1
                best_sd = {k: v.detach().clone() for k, v in m.state_dict().items()}
                if on_best is not None: on_best(make(m, dict(best_epoch=best_epoch)), best_epoch)
            entry["minutes"] = round((time.perf_counter() - t_start) / 60, 3)
            if on_epoch is not None: on_epoch(make(m, dict(epoch=p + 1)), p + 1)
            log.append(entry)
            print("  " + " · ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}" for k, v in entry.items()), flush=True)
            if patience and p + 1 - reset_epoch >= patience: break
            if stop_after and p + 1 >= stop_after: break
        m = avg if avg is not None else net
        if patience and best_sd is not None: m.load_state_dict(best_sd)
        trained = [e for e in log if e["epoch"] > 0]
        info.update(stopped_epoch=len(trained), best_epoch=best_epoch or (0 if epoch0 and val is not None else None),
                    best_val_loss=None if best == math.inf else best, curves=log)
        return make(m, {})

    def _with(self, nets, info):
        """A model sharing this one's vocabulary, tables and paste lookups, with other networks."""
        out = copy.copy(self); out.nets = [(net.eval(), kw) for net, kw in nets]; out.info = info
        return out

    @classmethod
    def mixture(cls, models, info=None):
        """One model sampling from the log-mean of the members' column distributions (lossdown.ensemble_fn)."""
        m0 = models[0]
        out = cls.__new__(cls)
        out.V, out.tables, out.look, out.modes, out.natures = m0.V, m0.tables, m0.look, m0.modes, m0.natures
        out.nets = [nk for m in models for nk in m.nets]
        if any(getattr(net, "carry", False) for net, _ in out.nets): raise ValueError("a carried-state network cannot join a mixture")
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

    def _logprobs(self, x, t, w, loops=None, state=None, keep=None):
        """Log-probabilities per field type. loops: loop-block count; state: carried hidden state of a one-member model;
        keep: a list that receives the hidden state of each network, for the next pass."""
        al = allowed(self.tables, x)
        outs = []
        for net, _ in self.nets:
            h = net(x, t, w) if loops is None and state is None else net(x, t, w, loops=loops, state=state)
            if keep is not None: keep.append(h)
            lg = net.logits(h)
            outs.append({k: F.log_softmax(v.masked_fill(~al[k], -1e4).float(), -1) for k, v in lg.items()})
        if len(outs) == 1: return outs[0]
        return {k: torch.logsumexp(torch.stack([o[k] for o in outs]), 0) - math.log(len(outs)) for k in outs[0]}

    def _column(self, lp, c):
        if c >= STAT0: return lp["stat"][:, c - STAT0]
        k = L.KEY_OF_COL[c]; return lp[k][:, L.COLS_BY_KEY[k].index(c)]

    def _legal(self, x, c, lookahead=False):
        """Boolean [B, outputs] for column c; never empty, never [MASK]. lookahead=True also drops the values that
        would leave a Pokémon whose species is masked with no legal species (lookahead.py); Stat Points are untouched."""
        if c >= STAT0: return SP.allowed_stat_points(x[:, STAT0:])[:, c - STAT0]
        k = L.KEY_OF_COL[c]; al = self.tables.allowed(x[:, :STAT0]); o = al[k][:, L.COLS_BY_KEY[k].index(c)].clone()
        if lookahead: o &= lookahead_rule(self.tables, x[:, :STAT0], c, al)
        E = self.tables.EMV
        if c % NF == FIRST_MOVE and E is not None:           # Showdown rejects a Pokémon with no moves
            real = o.clone(); real[:, [0, E]] = False; o[real.any(1), E] = False
        o[~o.any(1)] = True; o[:, 0] = False
        return o

    @torch.no_grad()
    def sample(self, n, ask=None, guidance=1.0, temperature=1.0, seed=None, lookahead=False, loops=None):
        """n teams in ORDER; ask is the win-rate condition, guidance the classifier-free weight. lookahead: see _legal;
        species come first in ORDER, so it draws the same teams on or off. loops: loop-block count (default loop_max).
        A carried-state network starts from a zero state and hands each pass's hidden state to the next pass; the
        guided and unguided passes each carry their own (Loopholing Algorithm 2, here one field per pass)."""
        if seed is not None: torch.manual_seed(seed)
        w = torch.full((n,), NAN if ask is None else ask, device=DEV); wn = torch.full((n,), NAN, device=DEV)
        x = torch.zeros(n, COLS, dtype=torch.long, device=DEV)
        carry = any(getattr(net, "carry", False) for net, _ in self.nets)
        if carry and len(self.nets) > 1: raise ValueError("a carried state belongs to one network; sample a carried-state model alone")
        state = dict(ask=None, none=None)

        def logprobs(t, cond, key):
            if not carry: return self._logprobs(x, t, cond, loops=loops)
            keep = []; lp = self._logprobs(x, t, cond, loops=loops, state=state[key], keep=keep); state[key] = keep[0]
            return lp

        for step, c in enumerate(ORDER):
            t = torch.full((n,), 1.0 - step / COLS, device=DEV)
            lc = self._column(logprobs(t, w, "ask"), c)
            if ask is not None and guidance != 1.0:
                lc = guidance * lc + (1 - guidance) * self._column(logprobs(t, wn, "none"), c)
            lc = lc.masked_fill(~self._legal(x, c, lookahead), -float("inf")) / temperature
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
