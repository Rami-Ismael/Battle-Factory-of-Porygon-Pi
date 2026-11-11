"""The discrete diffusion model trained from scratch on valid Pokémon teams: the deployed Regulation M-B model
behind one interface.

    w = DiscreteDiffusionModel.load()               # the loop-teams model + the battle-records model
    rows = w.sample(64, ask=0.5, guidance=2.0)      # [64, 48] token ids, legal by construction
    text, slots = w.to_paste(rows[0], rng)
    w2 = w.fine_tuned(rows, labels, epochs=40)      # a new model; w is unchanged

Everything a caller used to rebuild from the loss-down training module (lossdown.py) lives here: which
member checkpoints, the legal Pokémon construction tables from the checkpoint's regulation snapshot, the two-member mixture,
the matchup-row padding of the battle-records model, the sampler (fixed unmasking order, ask and guidance,
temperature, top-p, surrogate tilt, species guidance, condition-annealed sampling, remasking) and fine-tuning. A fine-tuned model is cached under a key that
hashes every setting that changes its weights, so changing the budget can never reload a stale copy.

It reproduces the foundation-swap sampler (foundation.Gen("v3")) and the cross-entropy loops' fine-tunes
(cem_v3.members, cem_dbbo.members), except that it never samples a Pokémon with no moves, which Showdown
rejects (scripts/test_discrete_diffusion_model.py). Finished experiments keep calling lossdown directly.
"""
import copy, hashlib, json
from pathlib import Path
import numpy as np
import torch

import lossdown as L
import asked_vs_got as X
import alignment as AL
import diffusion as D
import heldout as HO
import hps_generate
from hps_generate import slot_to_text
from checkpoints import load_checkpoint
from encode import NF, NSLOT, Vocab
from lookahead import lookahead as lookahead_rule

MEMBERS = ("F1_final_medium_noheldout_regmb_v3", "G2_family_matrix_noprtrain_noheldout_regmb_v3")
CACHE = L.CK / "team_writer"
V3DIR = Path.home() / ".local/share/vgc-pilot-runtime/regmb-v3"
SPECIES_COLS = [c for c in range(L.COLS) if c % NF == 0]
NAN = float("nan")
FIRST_MOVE = 3                  # field offset of a slot's first move column (species, ability, item, moves, nature)
MATRIX_COLUMNS = 49             # the battle-records model's matchup row: one column per other top-50 meta team


def _digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


POKEMON_ORDER = [s * NF + j for s in range(NSLOT) for j in range(NF)]   # each Pokémon's eight fields before the next
ORDERS = ("species", "pokemon", "random", "confidence")


def _lowest_confidence(conf, fraction):
    """Boolean [B, 48]: for each team, round(fraction x its filled columns), rounded stochastically so the mean is
    exact, of its filled columns with the lowest draw probability (conf; inf = not filled)."""
    filled = conf.isfinite()
    k = torch.floor(fraction * filled.sum(1) + torch.rand(len(conf), device=conf.device)).long()
    columns = torch.arange(conf.shape[1], device=conf.device).expand_as(conf)
    rank = torch.empty_like(columns).scatter_(1, conf.argsort(1), columns)
    return (rank < k[:, None]) & filled


def _count_redraws(stats, redo, old, x):
    """Add, per field, the columns remasked in one pass and those whose redraw changed the value."""
    for c in redo.any(0).nonzero().flatten().tolist():
        f, r = L.KEY_OF_COL[c], redo[:, c]
        stats["remasked"][f] = stats["remasked"].get(f, 0) + int(r.sum())
        stats["changed"][f] = stats["changed"].get(f, 0) + int((x[r, c] != old[r, c]).sum())


class DiscreteDiffusionModel:
    """One discrete diffusion model trained from scratch: an ensemble of member networks plus what sampling and
    fine-tuning need.
    Build it with DiscreteDiffusionModel.load() or model.fine_tuned(); never mutate one in place."""

    @classmethod
    def load(cls, members=MEMBERS, start=None):
        """The members' own checkpoints, or with start=PREFIX the weights saved as PREFIX_<member>.pt
        (start="dial_precise" is the precise-label model of ask_dial.py)."""
        ck = load_checkpoint(L.CK / (members[0] + ".pt"), map_location="cpu")
        if not ck.get("regulation"):
            raise ValueError("the model needs a versioned checkpoint with a regulation snapshot")
        V = Vocab.from_regulation(ck["regulation"])
        corpus, corpus_vocab, _, look, spreads = X.setup()
        look.update(V.regulation["names"])
        tables = L.Tables(V, corpus)
        nets = []
        for name in members:
            net, cfg, _ = L.load_net(name, V)
            if start:
                net.load_state_dict(load_checkpoint(L.CK / f"{start}_{name}.pt", map_location=L.DEV)["sd"]); net.eval()
            nets.append((name, net, cfg))
        _, _, _, modes, natures = AL.build_tables(corpus)
        shared = dict(V=V, tables=tables, corpus=corpus, corpus_vocab=corpus_vocab, look=look, spreads=spreads,
                      modes=modes, natures=natures, original_rows={})
        return cls(shared, nets, dict(members=list(members), start=start))

    def __init__(self, shared, nets, provenance):
        self._shared, self._nets, self.provenance = shared, nets, provenance
        self.key = _digest(provenance)
        self._f = L.ensemble_fn([L.logprobs_fn(net, shared["tables"], tuple(cfg["rules"].split(","))) for _, net, cfg in nets])

    V = property(lambda self: self._shared["V"])
    tables = property(lambda self: self._shared["tables"])
    corpus = property(lambda self: self._shared["corpus"])

    # ---------------------------------------------------------------- scoring
    def logprobs(self, xin, t, w):
        """{field: log-probs [B, columns, values]} of the member mixture, legality-masked."""
        return self._f(xin, t, w)

    def column_logprobs(self, x, t, w, c):
        k = L.KEY_OF_COL[c]; return self._f(x, t, w)[k][:, L.COLS_BY_KEY[k].index(c)]

    def legal_values(self, x, c):
        """Boolean [B, values] for column c given the visible fields; never empty, never [MASK]. Moves are
        stored sorted with the empty move last, so a slot's first move column may not be empty: Showdown
        rejects a Pokémon with no moves, which the shared tables allow."""
        return self._legal(x, c)[0]

    def _legal(self, x, c, lookahead=False):
        """(legal_values(x, c), Boolean [B]: the teams for which no value was legal, so every value was allowed).
        lookahead=True also drops the values that would leave a Pokémon with no legal species (the shared rule in
        lookahead.py, which the 84-field model calls too)."""
        k = L.KEY_OF_COL[c]; al = self.tables.allowed(x); o = al[k][:, L.COLS_BY_KEY[k].index(c)].clone()
        if lookahead: o &= lookahead_rule(self.tables, x, c, al)
        E = self.tables.EMV
        if c % NF == FIRST_MOVE and E is not None:                    # only where a real move is still allowed
            real = o.clone(); real[:, [0, E]] = False; o[real.any(1), E] = False
        empty = ~o.any(1); o[empty] = True; o[:, 0] = False
        return o, empty

    # ---------------------------------------------------------------- sampling
    @torch.no_grad()
    def sample(self, n, ask=None, guidance=1.0, temperature=1.0, top_p=1.0, surrogate=None, lam=0.0,
               species_guidance=None, cads=None, remask=0.0, steps=None, stats=None, order="species", lookahead=False):
        """n teams in diffusion.ORDER. ask: the win-rate condition (None = unconditioned); guidance: the
        classifier-free weight against the unconditioned model; species_guidance: that weight on the six
        species columns only; surrogate/lam: add lam * surrogate.tilt to every column's logits; cads: the noise
        scale of condition-annealed sampling (CADS), noise on the condition embedding annealed away over the
        steps (docs/guidance-timing-cads.md). remask, steps, stats: remasking (_sample_remasking). order: the decoding
        order, "species" (species-first, the default), "pokemon" (Pokémon by Pokémon), "random" (each team its own
        random order) or "confidence" (each team's most confident masked field first); docs/decoding-order.md.
        lookahead: allow only values that leave every Pokémon a legal species (lookahead.py; docs/lookahead-legality.md)."""
        if remask or (steps is not None and steps != L.COLS) or stats is not None or order != "species" or lookahead:
            return self._sample_remasking(n, ask, guidance, temperature, top_p, surrogate, lam, species_guidance, cads,
                                          remask, L.COLS if steps is None else steps, stats, order, lookahead)
        w = torch.full((n,), NAN if ask is None else ask, device=L.DEV); wn = torch.full((n,), NAN, device=L.DEV)
        x = torch.zeros(n, L.COLS, dtype=torch.long, device=L.DEV)
        try:
            for step, c in enumerate(D.ORDER):
                t = torch.full((n,), 1.0 - step / L.COLS, device=L.DEV)
                L.COND_NOISE = None if cads is None else (cads, step / (L.COLS - 1))
                gc = species_guidance if (species_guidance is not None and c in SPECIES_COLS) else guidance
                p = self._probs(self.column_logprobs(x, t, w, c), None if ask is None else (lambda: self.column_logprobs(x, t, wn, c)),
                                x, c, gc, temperature, top_p, surrogate, lam)
                x[:, c] = torch.multinomial(p, 1).squeeze(1)
        finally:
            L.COND_NOISE = None
        return x

    def _probs(self, lc, uncond, x, c, gc, temperature, top_p, surrogate, lam, legal=None):
        """The distribution column c is drawn from: conditioned log-probs lc, guided at weight gc against uncond()
        (None = no ask), tilted, restricted to legal values (legal_values(x, c) unless given), tempered and top-p
        truncated."""
        if uncond is not None and gc != 1.0: lc = gc * lc + (1 - gc) * uncond()
        if surrogate is not None and lam > 0: lc = lc + lam * surrogate.tilt(self.V, x, c)
        lc = lc.masked_fill(~(self.legal_values(x, c) if legal is None else legal), -float("inf")) / temperature
        p = torch.softmax(lc, -1)
        if top_p < 1.0:
            ps, ix = p.sort(-1, descending=True); keep = ps.cumsum(-1) - ps < top_p
            p = torch.zeros_like(p).scatter(1, ix, ps * keep); p = p / p.sum(-1, keepdim=True)
        return p

    @torch.no_grad()
    def _sample_remasking(self, n, ask, guidance, temperature, top_p, surrogate, lam, species_guidance, cads,
                          remask, steps, stats, order="species", lookahead=False):
        """sample() with remasking (ReMDM; Wang, Schiff et al., 2025), docs/remask-ask-gap.md. The 48 columns
        unmask in diffusion.ORDER, spread evenly over `steps` network passes (48 = one new column per pass, as
        sample()). Before each pass after the first, each team masks again round(remask x its filled columns),
        rounded stochastically, choosing the columns drawn with the lowest probability; that same pass redraws
        them under the ask (_fill_pass). A pass with nothing to draw (no new column, nothing remasked) is skipped,
        so stats["passes"] can fall below steps. remask=0 at 48 steps reproduces sample() bit for bit. stats: a
        dict to fill with network passes, teams whose redraws hit a dead end and were undone ("reverted"), and, per
        field, columns remasked and columns whose redraw changed value. order picks which new columns each pass fills
        (sample()); with steps below 48 a pass fills several, in diffusion.ORDER within the pass. lookahead: see sample()."""
        if not 1 <= steps: raise ValueError("steps must be at least 1")
        if order not in ORDERS: raise ValueError(f"order must be one of {ORDERS}")
        if not 0.0 <= remask < 1.0: raise ValueError("remask is the fraction of filled columns masked again, in [0, 1)")
        w = torch.full((n,), NAN if ask is None else ask, device=L.DEV); wn = torch.full((n,), NAN, device=L.DEV)
        x = torch.zeros(n, L.COLS, dtype=torch.long, device=L.DEV)
        conf = torch.full((n, L.COLS), float("inf"), device=L.DEV)     # probability each filled column was drawn with
        quota = [0] * steps                              # new columns per pass: column j of an order is due at pass j*steps//48
        for j in range(L.COLS): quota[j * steps // L.COLS] += 1
        seq = {"species": D.ORDER, "pokemon": POKEMON_ORDER}.get(order)
        perm = torch.rand(n, L.COLS, device=L.DEV).argsort(1) if order == "random" else None
        done = 0                                         # new columns filled so far, per team (the same for every team)
        stats = {} if stats is None else stats
        for k, v in (("passes", 0), ("reverted", 0), ("remasked", {}), ("changed", {})): stats.setdefault(k, v)
        draw = dict(guidance=guidance, species_guidance=species_guidance, temperature=temperature, top_p=top_p,
                    surrogate=surrogate, lam=lam, lookahead=lookahead)
        try:
            for step in range(steps):
                redo = _lowest_confidence(conf, remask) if remask > 0 and step > 0 else torch.zeros_like(x, dtype=torch.bool)
                old, old_conf = x.clone(), conf.clone(); x[redo] = 0; conf[redo] = float("inf")
                L.COND_NOISE = None if cads is None else (cads, step / max(steps - 1, 1))
                due, pre = self._due(x, seq, perm, done, quota[step], w, None if ask is None else wn, draw, redo)
                done += quota[step]
                targets = sorted(((due | redo).any(0)).nonzero().flatten().tolist(), key=D.ORDER.index)
                if not targets: continue
                stats["reverted"] += self._fill_pass(x, conf, targets, due, redo, w, None if ask is None else wn,
                                                     draw, old, old_conf, pre)
                stats["passes"] += 1; _count_redraws(stats, redo, old, x)
        finally:
            L.COND_NOISE = None
        if (x == 0).any(): raise RuntimeError("a column is still masked after the last pass")
        return x

    def _fill_pass(self, x, conf, targets, due, redo, w, wn, draw, old, old_conf, pre=None):
        """One network pass of _sample_remasking. Draws each target column in diffusion.ORDER, for every team if
        the column is due and for the teams that remasked it otherwise, so the legal Pokémon construction tables see each earlier
        draw of the pass. Writes the values into x and their probabilities into conf. wn = None: no ask.
        Redrawing several columns can reach a dead end: two remasked species where an earlier draw takes, under
        Species Clause, the only species the other's visible set allows. A team that hits one gets its pre-pass
        fields back (old, old_conf: a legal partial team) and only the pass's due columns are drawn for it, from the
        network's predictions on that restored team (one extra network pass for the reverted teams).
        due: Boolean [B, 48], the new columns each team fills this pass. pre: (t, logprobs, unconditioned logprobs)
        already computed on this pass's input. Returns how many teams were reverted."""
        if pre is None:
            t = (1.0 - (x != 0).sum(1).cpu().double() / L.COLS).float().to(x.device)   # masked fraction, as sample()'s 1 - step / 48
            lp, lpn = self.logprobs(x, t, w), {}
        else: t, lp, lpn = pre
        xp = x.clone()
        stuck = torch.zeros(len(x), dtype=torch.bool, device=x.device)

        def draw_column(c, rows):
            key = L.KEY_OF_COL[c]; i = L.COLS_BY_KEY[key].index(c)
            gc = draw["species_guidance"] if (draw["species_guidance"] is not None and c in SPECIES_COLS) else draw["guidance"]

            def uncond():
                if not lpn: lpn.update(self.logprobs(xp, t, wn))     # once per pass, on the pass's own input
                return lpn[key][rows, i]
            legal, empty = self._legal(x[rows], c, draw.get("lookahead", False))
            p = self._probs(lp[key][rows, i], None if wn is None else uncond, x[rows], c, gc, draw["temperature"],
                            draw["top_p"], draw["surrogate"], draw["lam"], legal=legal)
            v = torch.multinomial(p, 1).squeeze(1)
            x[rows, c] = v; conf[rows, c] = p.gather(1, v[:, None]).squeeze(1)
            return rows[empty]

        for c in targets:
            stuck[draw_column(c, (due[:, c] | redo[:, c]).nonzero().squeeze(1))] = True
        back = stuck & redo.any(1)                     # a dead end with nothing remasked is sample()'s own fallback
        if back.any():
            x[back] = old[back]; conf[back] = old_conf[back]
            if wn is not None and not lpn: lpn.update(self.logprobs(xp, t, wn))
            lp, lpn = self._refresh(x, back.nonzero().squeeze(1), w, wn, lp, lpn)
            for c in sorted((due & back[:, None]).any(0).nonzero().flatten().tolist(), key=D.ORDER.index):
                draw_column(c, (due[:, c] & back).nonzero().squeeze(1))
        return int(back.sum())

    def _refresh(self, x, rows, w, wn, lp, lpn):
        """Copies of a pass's predictions (lp, and lpn unless wn is None) with these rows recomputed on x as it is now,
        for teams whose fields were restored mid-pass."""
        t = (1.0 - (x[rows] != 0).sum(1).cpu().double() / L.COLS).float().to(x.device)
        lp = {k: v.clone() for k, v in lp.items()}
        for k, v in self.logprobs(x[rows], t, w[rows]).items(): lp[k][rows] = v
        if wn is None: return lp, lpn
        lpn = {k: v.clone() for k, v in lpn.items()}
        for k, v in self.logprobs(x[rows], t, wn[rows]).items(): lpn[k][rows] = v
        return lp, lpn

    def _due(self, x, seq, perm, done, k, w, wn, draw, redo=None):
        """The k new columns each team fills this pass, as a Boolean [B, 48], and the pass's precomputed network
        outputs when choosing them needed one (confidence order). seq: a fixed order shared by every team; perm: one
        random order per team; neither: confidence order. redo: the columns remasked this pass, which are masked in x
        but not new."""
        due = torch.zeros_like(x, dtype=torch.bool)
        if seq is not None: due[:, seq[done:done + k]] = True
        elif perm is not None: due.scatter_(1, perm[:, done:done + k], True)
        elif k: return self._most_confident(x, w, wn, draw, k, redo)
        return due, None

    def _most_confident(self, x, w, wn, draw, k, redo=None):
        """Confidence order: for each team, the k never-filled masked columns whose drawing distribution (as _probs
        would give it) puts the most probability on one value; columns remasked this pass (redo) are not candidates.
        Returns (due mask [B, 48], the pass's (t, logprobs, unconditioned logprobs)) so _fill_pass reuses this pass's
        network outputs."""
        t = (1.0 - (x != 0).sum(1).cpu().double() / L.COLS).float().to(x.device)
        lp = self.logprobs(x, t, w); lpn = self.logprobs(x, t, wn) if wn is not None else {}
        fresh = (x == 0) if redo is None else (x == 0) & ~redo
        score = torch.full(x.shape, -1.0, device=x.device)
        for c in range(L.COLS):
            rows = fresh[:, c].nonzero().squeeze(1)
            if not len(rows): continue
            key = L.KEY_OF_COL[c]; i = L.COLS_BY_KEY[key].index(c)
            gc = draw["species_guidance"] if (draw["species_guidance"] is not None and c in SPECIES_COLS) else draw["guidance"]
            p = self._probs(lp[key][rows, i], None if wn is None else (lambda key=key, i=i, rows=rows: lpn[key][rows, i]), x[rows], c, gc,
                            draw["temperature"], draw["top_p"], draw["surrogate"], draw["lam"],
                            legal=self._legal(x[rows], c, draw.get("lookahead", False))[0])
            score[rows, c] = p.max(1).values
        due = torch.zeros_like(x, dtype=torch.bool).scatter_(1, score.topk(k, dim=1).indices, True) & fresh
        return due, (t, lp, lpn)

    def to_paste(self, row, rng, repair=False):
        """(Showdown paste, slot dicts) for one sampled row. Stat Points come from a same-species corpus
        team, or a random legal spread for a species the corpus lacks; repair=True aligns the nature."""
        sh = self._shared; V, look = sh["V"], sh["look"]
        slots = []
        for s in range(NSLOT):
            g = lambda j: V.decode_field(s * NF + j, int(row[s * NF + j]))
            sp = g(0); pl = sh["spreads"].get(sp) or [hps_generate.sample_spread(rng)]
            slot = dict(species=look.get(sp, sp), item=look.get(g(2), g(2)), ability=look.get(g(1), g(1)),
                        nature=look.get(g(7), g(7)), moves=[look.get(g(3 + j), g(3 + j)) for j in range(4)
                                                            if g(3 + j) not in ("", "[MASK]")],
                        evs=pl[int(rng.integers(0, len(pl)))])
            if repair: slot["nature"] = AL.repair_nature(slot, sh["modes"], sh["natures"])
            slots.append(slot)
        return "\n\n".join(slot_to_text(s) for s in slots) + "\n", slots

    # ---------------------------------------------------------------- fine-tuning
    def fine_tuned(self, rows, labels, *, epochs=None, steps=None, batch=64, lr=1e-4, mix_original=False, seed=0,  # noqa: C901
                   cache_dir=CACHE):
        """A new model: every member trained further from this model's weights on (rows, labels).
        Training starts from this model's weights: fine-tune DiscreteDiffusionModel.load() for the original members,
        or DiscreteDiffusionModel.load(start="dial_precise") for the precise-label ones.
        epochs=E: E shuffled passes over the rows (the cross-entropy loop's re-steer, cem_v3). steps=S: S batches
        drawn with replacement; with mix_original=True half of each batch comes from the member's own training
        rows (the diffusion black-box-optimization arm, cem_dbbo). AdamW, weight decay 0.01, gradients clipped
        at 1. Cached under a hash of this model, every argument above except cache_dir, and, with
        mix_original, the files the members' own rows are read from."""
        if (epochs is None) == (steps is None):
            raise ValueError("give exactly one of epochs or steps")
        if mix_original and steps is None:
            raise ValueError("mix_original needs steps: an epoch over the member's own rows is not defined")
        rows = np.asarray(rows, dtype=np.int64); labels = np.asarray(labels, dtype=np.float32)
        recipe = dict(parent=self.key, rows=hashlib.sha256(rows.tobytes()).hexdigest(),
                      labels=hashlib.sha256(labels.tobytes()).hexdigest(), n=len(rows), epochs=epochs, steps=steps,
                      batch=batch, lr=lr, mix_original=bool(mix_original), seed=seed)
        if mix_original: recipe["original_data"] = self._original_data_digest()
        key = _digest(recipe); cache_dir = Path(cache_dir); cache_dir.mkdir(parents=True, exist_ok=True)
        tables = self.tables; nets = []
        for name, parent, cfg in self._nets:
            net = copy.deepcopy(parent); ck = cache_dir / f"{key[:20]}_{name}.pt"
            if ck.exists():
                net.load_state_dict(load_checkpoint(ck, map_location=L.DEV)["sd"]); net.eval()
                nets.append((name, net, cfg)); continue
            torch.manual_seed(seed)
            xr = torch.as_tensor(rows, device=L.DEV); wr = torch.as_tensor(labels, device=L.DEV)
            if cfg.get("mrow"): wr = torch.cat([wr[:, None], torch.full((len(wr), MATRIX_COLUMNS), NAN, device=L.DEV)], 1)
            sched = L.sched_tensor(cfg["sched"]).to(L.DEV); rules = tuple(cfg["rules"].split(","))
            opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.01)
            net.train()
            def update(bx, bw):
                loss = L.batch_loss(net, tables, rules, bx, bw, None, sched, cfg)
                opt.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
            if epochs is not None:
                for _ in range(epochs):
                    perm = torch.randperm(len(xr), device=L.DEV)
                    for i in range(0, len(xr), batch):
                        b = perm[i:i + batch]; update(xr[b], wr[b])
            elif mix_original:
                xo, wo = self._original_rows(name, cfg); half = batch // 2
                for _ in range(steps):
                    bo = torch.randint(len(xo), (half,), device=L.DEV); bl = torch.randint(len(xr), (half,), device=L.DEV)
                    update(torch.cat([xo[bo], xr[bl]]), torch.cat([wo[bo], wr[bl]]))
            else:
                for _ in range(steps):
                    b = torch.randint(len(xr), (batch,), device=L.DEV); update(xr[b], wr[b])
            net.eval()
            torch.save({"sd": net.state_dict(), "recipe": {**recipe, "member": name}}, ck)
            nets.append((name, net, cfg))
        return DiscreteDiffusionModel(self._shared, nets, dict(self.provenance, fine_tune=recipe))

    def _original_data_digest(self):
        """sha256 of the three files the members' own training rows come from."""
        if "original_digest" not in self._shared:
            h = hashlib.sha256()
            for f in (L.DATA, V3DIR / "remapped.npz", V3DIR / "synthetic.npy"): h.update(Path(f).read_bytes())
            self._shared["original_digest"] = h.hexdigest()
        return self._shared["original_digest"]

    def _original_rows(self, name, cfg):
        """The member's own training rows, rebuilt exactly as scripts/retrain_regmb.py chose them."""
        cache = self._shared["original_rows"]
        if name in cache: return cache[name]
        d = L.load_data(include_heldout=True); mapped = np.load(V3DIR / "remapped.npz"); syn = np.load(V3DIR / "synthetic.npy")
        keep = HO.row_mask(d["X"], self._shared["corpus_vocab"]); tr, _ = L.splits(d); tr = tr[keep[tr]]; tag = L.tag_of(d["S"])
        idx = tr[(tag[tr] == L.TEST_TAG) | (tag[tr] == 0) | ((d["M"][tr, :, 1].sum(1) > 0) if cfg.get("mrow") else False)]
        if cfg.get("mrow"): idx = np.setdiff1d(idx, L.matrix_holdout(d))
        idx = idx[mapped["valid"][idx]]
        w = L.cond_matrix(d, idx) if cfg.get("mrow") else d["W"][idx]
        sw = np.full((len(syn), 1 + MATRIX_COLUMNS) if cfg.get("mrow") else (len(syn),), np.nan, np.float32)
        xx = np.concatenate([mapped["X"][idx], syn]); ww = np.concatenate([w, sw]).astype(np.float32)
        cache[name] = (torch.as_tensor(xx, device=L.DEV), torch.as_tensor(ww, device=L.DEV))
        return cache[name]
