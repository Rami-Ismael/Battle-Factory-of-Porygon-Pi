"""The 84-field discrete diffusion sampler (three parts: decoding order, reshaping the odds, and draw).

Consolidates the 84-field sampling loops and configured sampler objects into one module:
  1. Decoding order:
     - species_first: one field per pass in diffusion84.ORDER (species first, then set fields, then Stat Points)
     - species_first_corrected_t: species-first order with pass k told the expected-share training masking rate t
     - random / rand_plain: LLaDA Algorithm 4 random order (84 passes, each hidden field unmasked with chance 1/(passes left))
     - species_then_random: six species in species-first order (6 passes), then the 78 set fields in random order (78 passes)
     - confidence: top-probability position selection (pick the masked field with highest top probability, then draw)
     - entropy_init: pass 1 picks the masked field with highest entropy and draws at first_temperature (1.5), passes 2..84 follow confidence order
  2. Sampling (reshaping the odds):
     - classifier-free guidance weight (z = z_no_ask + s * (z_ask - z_no_ask))
     - legal Pokémon construction tables and Stat Point budget mask (tables=True), plus one-Pokemon lookahead (lookahead=True)
     - set-only sharpening: separate species and set-field temperatures, and separate species and set-field nucleus top_p shares
     - Loopholing carried hidden state across passes (carry=True)
  3. Draw, remasking, and per-team diagnostics:
     - multinomial draw from each field's reshaped distribution
     - remasking schedules: least-likely fraction (remask_ll_05, remask_ll_10, remask_ll_20), confidence-weighted loop
       (remask_conf_loop, rand_remask_conf), and even-odds loop (remask_even_loop, rand_remask_even)
     - Refusals: per-team lookahead refusal shares, fill_order, non_top_fields, and optional per-draw records
"""
import math

import torch

import diffusion84 as M
import stat_points as SP
from encode import NF, NSLOT

LOOP_START = 63                                          # alpha 0.75: ReMDM's t_on 0.25 on 84 fields
LOOP_PASSES = 126                                        # ReMDM's T = 4L has 3T/8 loop passes; L = 84
SIGMA_CONF = 0.3                                         # ReMDM: 0.9 x min(1, (1 - alpha) / alpha) at alpha 0.75
ETA_CAP = 0.008                                          # ReMDM's best MAUVE setting on text (Table 5)
LEAST_LIKELY = 0.10
FIRST_TEMPERATURE = 1.5                                  # Jeon et al. 2026 §6.2: 0.9 / 0.6 times base temperature 1
SPECIES = list(M.ORDER[:NSLOT])                          # species-first order begins with the six species

REMASK_SHARES = {"remask_ll_05": 0.05, "remask_ll_10": 0.10, "remask_ll_20": 0.20}
REMASK_ARMS = ("remask_ll_05", "remask_ll_10", "remask_ll_20", "remask_conf_loop", "remask_even_loop")
RANDOM_ARMS = ("rand_plain", "rand_remask_conf", "rand_remask_even", "species_then_random")
DECODERS = ("species_first", "species_first_corrected_t", "random", "species_then_random", "confidence", "entropy_init")


def _pair(value):
    """(species value, set-field value) from a scalar or a 2-tuple."""
    return (float(value[0]), float(value[1])) if isinstance(value, (tuple, list)) else (float(value), float(value))


def top_p_keep(probs, p):
    """A mask over the likeliest options, in order, until their odds reach p. The likeliest option always stays."""
    sp, idx = torch.sort(probs, descending=True, dim=-1)
    keep = (sp.cumsum(-1) - sp) < p
    return torch.zeros_like(probs, dtype=torch.bool).scatter(-1, idx, keep)


def least_likely(conf, fraction):
    """[B, COLS]: per team, round(fraction x filled), rounded stochastically, of its filled fields drawn with the lowest probability."""
    filled = conf.isfinite()
    k = torch.floor(fraction * filled.sum(1) + torch.rand(len(conf), device=conf.device)).long()
    columns = torch.arange(conf.shape[1], device=conf.device).expand_as(conf)
    rank = torch.empty_like(columns).scatter_(1, conf.argsort(1), columns)
    return (rank < k[:, None]) & filled


def odds_by_confidence(conf):
    """[B, COLS]: ReMDM-conf: odds sigma x exp(-p) / sum(exp(-p)) over filled fields, drawn independently."""
    filled = conf.isfinite()
    weight = torch.exp(-conf.masked_fill(~filled, 0.0)) * filled
    return torch.rand_like(conf) < SIGMA_CONF * weight / weight.sum(1, keepdim=True).clamp_min(1e-9)


def even_odds(conf):
    """[B, COLS]: ReMDM-cap: every filled field remasked with odds eta, independently."""
    return (torch.rand_like(conf) < ETA_CAP) & conf.isfinite()


class Refusals:
    """What the lookahead refused and how each team was drawn while sampling n teams (issues #207, #321, #332).
    Each drawn field adds its refused share: the probability the lookahead removed from the base distribution, the
    model's guided, temperature-scaled distribution after the legal Pokémon construction tables and before the lookahead.

        rec = Refusals(n); ... rec.add(scores, plain, kept, fallback, rows, column=c, x=x) ...; rec.records()

    Per team: kept_share, the product over drawn fields of (1 - refused share), and log_kept_share; refused_share,
    1 - kept share; refusing_fields, the draws where the lookahead refused any value the plain sampler allows (a
    remasked field counts each time it is drawn); fallback_fields, the draws where no value was legal, so every value
    was allowed and nothing counts as refused; fill_order, the columns drawn for the team in draw order; non_top_fields,
    how many columns finished on a value other than their drawing distribution's most likely value. With draws=True each
    record also holds "draws": parallel lists of column, removed_share, top_removed, and fields_before."""

    def __init__(self, n, draws=False):
        self.log_kept = torch.zeros(n, dtype=torch.float64)
        self.refusing = torch.zeros(n, dtype=torch.long)
        self.fallback = torch.zeros(n, dtype=torch.long)
        self.fill_order = [[] for _ in range(n)]
        self.non_top = torch.zeros(n, M.COLS, dtype=torch.bool)
        self.draws = [dict(column=[], removed_share=[], top_removed=[], fields_before=[]) for _ in range(n)] if draws else None

    def add(self, scores, plain, kept, fallback, rows=None, column=None, x=None):
        """scores [B, outputs]: the base distribution's logits before any legality mask; plain, kept, fallback: from
        Model84._legal_parts; rows [B]: the teams this field was drawn for (default all). column: the field drawn; x
        [B, COLS]: the teams before this draw. Both are needed only when every draw is kept."""
        plain, kept, fallback = plain.cpu(), kept.cpu(), fallback.cpu()
        rows = torch.ones_like(fallback) if rows is None else rows.cpu()
        refused = (plain & ~kept).any(1)
        s = scores.cpu().double()
        lse = lambda m: torch.logsumexp(s.masked_fill(~m, -float("inf")), 1)
        log_kept = torch.where(refused, (lse(kept) - lse(plain)).clamp(max=0.0), torch.zeros_like(self.log_kept))
        self.log_kept += torch.where(rows, log_kept, torch.zeros_like(log_kept))
        self.refusing += (refused & rows).long(); self.fallback += (fallback & rows).long()
        if self.draws is not None: self._keep_draws(s, plain, kept, refused, log_kept, rows, column, x)

    def note_draw(self, column, rows, drawn, probs):
        """Record which column was filled and whether the drawn value differed from the distribution's top value."""
        if rows is None:
            for order in self.fill_order: order.append(column)
            self.non_top[:, column] = (drawn != probs.argmax(1)).cpu()
            return
        rows_cpu = rows.cpu()
        for i in rows_cpu.nonzero().flatten().tolist(): self.fill_order[i].append(column)
        picked = drawn[rows] if drawn.shape[0] == len(rows) and not bool(rows.all()) else drawn
        self.non_top[rows_cpu, column] = (picked != probs[rows].argmax(1)).cpu()

    def _keep_draws(self, s, plain, kept, refused, log_kept, rows, column, x):
        if column is None or x is None: raise ValueError("keeping every draw needs the column and the team before the draw")
        top = s.masked_fill(~plain, -float("inf")).argmax(1)
        top_removed = refused & ~kept.gather(1, top[:, None]).squeeze(1)
        removed = (-torch.expm1(log_kept)).tolist()
        before = [None] * len(rows)
        if column < M.STAT0 and column % NF == 0:              # a species: count its Pokémon's other fields already drawn
            slot, xc = column // NF, x.cpu()
            drawn = (xc[:, slot * NF + 1:(slot + 1) * NF] != 0).sum(1) + (xc[:, M.STAT0 + slot * SP.NSTAT:M.STAT0 + (slot + 1) * SP.NSTAT] != 0).sum(1)
            before = drawn.tolist()
        for i in rows.nonzero().flatten().tolist():
            d = self.draws[i]
            d["column"].append(column); d["removed_share"].append(removed[i]); d["top_removed"].append(bool(top_removed[i]))
            d["fields_before"].append(before[i])

    def records(self):
        """One dict per team. log_kept_share is None where some draw's lookahead refused every value the plain sampler
        allows; a very small kept share can also round to 0.0 while its logarithm stays finite. With draws=True each
        record also holds "draws": parallel lists in the order drawn; fields_before is None at a draw that is not a species."""
        out = [dict(kept_share=math.exp(lk), log_kept_share=lk if lk > -math.inf else None, refused_share=-math.expm1(lk),
                    refusing_fields=r, fallback_fields=f, fill_order=fo, non_top_fields=int(nt))
               for lk, r, f, fo, nt in zip(self.log_kept.tolist(), self.refusing.tolist(), self.fallback.tolist(),
                                           self.fill_order, self.non_top.sum(1).tolist())]
        if self.draws is not None:
            for r, d in zip(out, self.draws): r["draws"] = d
        return out


def _legal_parts_of(w, x, c, lookahead=False, tables=True, al=None):
    if not tables: return w._unmasked_parts(x, c)
    if hasattr(w, "_legal_parts"): return w._legal_parts(x, c, lookahead, al=al)
    legal = w._legal(x, c)
    return legal, legal, legal, torch.zeros(len(x), dtype=torch.bool, device=x.device)


def _field_setting(c, pair):
    return pair[0] if (c < M.STAT0 and c % NF == 0) else pair[1]


def _reshape_column(lc, legal, temp, p_keep):
    logits = lc.masked_fill(~legal, -float("inf")) / temp
    if p_keep < 1.0:
        logits = logits.masked_fill(~top_p_keep(torch.softmax(logits, -1), p_keep), -float("inf"))
    return torch.softmax(logits, -1)


def _record_refusal(refusals, scores, plain, kept, fallback, rows, column, x, drawn, probs):
    if refusals is None: return
    refusals.add(scores, plain, kept, fallback, rows, column=column, x=x)
    if hasattr(refusals, "note_draw"): refusals.note_draw(column, rows, drawn, probs)


def _logprob_runner(w, x, loops, tables):
    nets = getattr(w, "nets", ())
    carry = any(getattr(net, "carry", False) for net, _ in nets)
    if carry and len(nets) > 1: raise ValueError("a carried state belongs to one network; sample a carried-state model alone")
    state = dict(ask=None, none=None)

    def logprobs(t, cond, key):
        if not carry:
            return w._logprobs(x, t, cond) if loops is None and tables else w._logprobs(x, t, cond, loops=loops, tables=tables)
        keep = []
        lp = w._logprobs(x, t, cond, loops=loops, state=state[key], keep=keep, tables=tables)
        state[key] = keep[0]
        return lp

    return logprobs


def _default_t_of_pass(w, corrected_t):
    info = getattr(w, "info", None) or {}
    return M.corrected_t_per_pass(M.trained_sched(info)) if corrected_t else [1.0 - step / M.COLS for step in range(M.COLS)]


@torch.no_grad()
def sample_species_first(w, n, ask=None, guidance=1.0, temperature=1.0, top_p=1.0, seed=None, lookahead=False,
                         loops=None, corrected_t=False, refusals=None, tables=True, order=None,
                         t_of_pass=None, other_rules=None):
    """n teams in fixed column order (default diffusion84.ORDER); one field per pass.
    Supports set-only sharpening via temperature=(t_species, t_sets) and top_p=(p_species, p_sets), carried hidden states
    (carry=True), corrected per-pass t (corrected_t=True), and per-pass t total-variation probes (t_of_pass, other_rules)."""
    if not tables and lookahead: raise ValueError("the lookahead works on the legal Pokémon construction tables; draw with tables=True")
    if seed is not None: torch.manual_seed(seed)
    temps, top_ps = _pair(temperature), _pair(top_p)
    ask_w = torch.full((n,), M.NAN if ask is None else ask, device=M.DEV)
    null_w = torch.full((n,), M.NAN, device=M.DEV)
    x = torch.zeros(n, M.COLS, dtype=torch.long, device=M.DEV)
    logprobs = _logprob_runner(w, x, loops, tables)
    guided = ask is not None and guidance != 1.0
    t_schedule = _default_t_of_pass(w, corrected_t) if t_of_pass is None else t_of_pass
    total_variation = {name: [] for name in other_rules or {}}

    def column_logits(t_val, c):
        t = torch.full((n,), t_val, device=M.DEV)
        lc = w._column(logprobs(t, ask_w, "ask"), c)
        return guidance * lc + (1 - guidance) * w._column(logprobs(t, null_w, "none"), c) if guided else lc

    for step, c in enumerate(M.ORDER if order is None else order):
        temp, p_keep = _field_setting(c, temps), _field_setting(c, top_ps)
        lc = column_logits(t_schedule[step], c)
        legal, plain, kept, fallback = _legal_parts_of(w, x, c, lookahead=lookahead, tables=tables)
        p = _reshape_column(lc, legal, temp, p_keep)
        for name, other_t in (other_rules or {}).items():
            other_p = _reshape_column(column_logits(other_t[step], c), legal, temp, p_keep)
            total_variation[name].append(float(0.5 * (p - other_p).abs().sum(-1).mean()))
        v = torch.multinomial(p, 1).squeeze(1)
        _record_refusal(refusals, lc / temp, plain, kept, fallback, None, c, x, v, p)
        x[:, c] = v + 1 if c >= M.STAT0 else v
    return (x, total_variation) if other_rules is not None else x


@torch.no_grad()
def sample_with_t(model, n, ask, t_of_pass, seed, other_rules=None, guidance=2.0):
    """Species-first order with pass k told t_of_pass[k]. When other_rules ({name: t_per_pass}) is given, also returns
    the mean total variation per rule and pass against t_of_pass on the same partly filled teams."""
    out = sample_species_first(model, n, ask=ask, guidance=guidance, seed=seed, t_of_pass=t_of_pass, other_rules=other_rules or {})
    x, total_variation = out
    return x.cpu().numpy(), total_variation


def fill_pass(w, x, conf, targets, due, redo, ask_w, null_w, temperature=1.0, guidance=2.0, lookahead=False, refusals=None, top_p=1.0):
    """One network pass: draws each target field in ORDER, using one set of logits, so legality, and the lookahead when
    it is on (lookahead.py), see each earlier draw. refusals: a Refusals, told what each draw refused."""
    n = len(x)
    t_species, t_sets = _pair(temperature)
    p_species, p_sets = _pair(top_p)
    t = (1.0 - (x != 0).sum(1).float() / M.COLS).to(x.device)
    lp = w._logprobs(x, t, ask_w); lpn = w._logprobs(x, t, null_w)
    everyone = torch.ones(n, dtype=torch.bool, device=x.device)
    for c in targets:
        rows = everyone if c in due else redo[:, c]
        if not rows.any(): continue
        is_species = c < M.STAT0 and c % NF == 0
        temp = t_species if is_species else t_sets
        p_keep = p_species if is_species else p_sets
        lc = guidance * w._column(lp, c) + (1 - guidance) * w._column(lpn, c)
        legal, plain, kept, fallback = w._legal_parts(x, c, lookahead)
        if refusals is not None: refusals.add(lc / temp, plain, kept, fallback, rows, column=c, x=x)
        lc = lc.masked_fill(~legal, -float("inf")) / temp
        if p_keep < 1.0: lc = lc.masked_fill(~top_p_keep(torch.softmax(lc, -1), p_keep), -float("inf"))
        p = torch.softmax(lc, -1)
        v = torch.multinomial(p, 1).squeeze(1)
        if refusals is not None and hasattr(refusals, "note_draw"): refusals.note_draw(c, rows, v, p)
        conf[rows, c] = p.gather(1, v[:, None]).squeeze(1)[rows]
        x[rows, c] = (v + 1 if c >= M.STAT0 else v)[rows]    # a Stat Point output index is its value


def _run_remask_schedule(arm, step, conf, none, remask_share=None):
    share = REMASK_SHARES.get(arm, LEAST_LIKELY) if remask_share is None else remask_share
    if arm in REMASK_SHARES or arm == "least_likely":
        for i, c in enumerate(M.ORDER):
            step([c], least_likely(conf, share) if i else none)
    elif arm is None:
        for c in M.ORDER: step([c], none)
    elif arm in ("remask_conf_loop", "remask_even_loop"):
        rule = odds_by_confidence if arm == "remask_conf_loop" else even_odds
        for c in M.ORDER[:LOOP_START]: step([c], none)
        for _ in range(LOOP_PASSES): step([], rule(conf))
        for c in M.ORDER[LOOP_START:]: step([c], none)
    else:
        raise ValueError(f"unknown remasking arm {arm!r}")


@torch.no_grad()
def sample_remask(w, n, ask, arm, stats=None, guidance=2.0, temperature=1.0, top_p=1.0, lookahead=False,
                  refusals=None, remask_share=None):
    """Species-first order with remasking. arm None: plain decode, one field per pass. remask_ll_05 / remask_ll_10 /
    remask_ll_20: remask lowest-confidence fraction before every pass. remask_conf_loop / remask_even_loop: fill 63
    fields, loop LOOP_PASSES passes that only redraw, then fill the last 21."""
    ask_w = torch.full((n,), M.NAN if ask is None else ask, device=M.DEV)
    null_w = torch.full((n,), M.NAN, device=M.DEV)
    x = torch.zeros(n, M.COLS, dtype=torch.long, device=M.DEV)
    conf = torch.full((n, M.COLS), float("inf"), device=M.DEV)
    stats = stats if stats is not None else {}
    stats.setdefault("passes", 0); stats.setdefault("redrawn", 0); stats.setdefault("changed", 0); stats.setdefault("redrawn_species", 0)
    none = torch.zeros_like(x, dtype=torch.bool)

    def step(new, redo):
        old = x.clone(); x[redo] = 0; conf[redo] = float("inf")
        targets = sorted(set(new) | set(redo.any(0).nonzero().flatten().tolist()), key=M.ORDER.index)
        if not targets: return
        fill_pass(w, x, conf, targets, set(new), redo, ask_w, null_w, temperature=temperature, guidance=guidance,
                  lookahead=lookahead, refusals=refusals, top_p=top_p)
        stats["passes"] += 1; stats["redrawn"] += int(redo.sum()); stats["changed"] += int((redo & (old != x)).sum())
        stats["redrawn_species"] += int(redo[:, ::8][:, :6].sum())

    _run_remask_schedule(arm, step, conf, none, remask_share=remask_share)
    if (x == 0).any(): raise RuntimeError("a field is still masked after the last pass")
    return x


def _random_remask_loop(arm, step, fill):
    """Fill 63 fields, loop LOOP_PASSES passes that remask and redraw, then fill the last 21."""
    share = LOOP_START / M.COLS
    fill(LOOP_START, 0.0, share)
    if arm == "rand_remask_even":
        for _ in range(LOOP_PASSES): step(ETA_CAP * share / (1 - share), even_odds)
    else:
        for _ in range(LOOP_PASSES): step(lambda m: SIGMA_CONF / m.sum(1, keepdim=True).clamp_min(1), odds_by_confidence)
    fill(M.COLS - LOOP_START, share, 1.0)


def _species_then_random(step, fill, conf):
    """Issue #322: one species per pass in species-first order, then the 78 set fields as rand_plain fills them."""
    for c in SPECIES:
        odds = torch.zeros_like(conf); odds[:, c] = 1.0; step(odds)
    fill(M.COLS - len(SPECIES), len(SPECIES) / M.COLS, 1.0)


@torch.no_grad()
def sample_random(w, n, ask, arm="rand_plain", stats=None, guidance=2.0, temperature=1.0, lookahead=False,
                  refusals=None, top_p=1.0):
    """Random decoding (LLaDA Algorithm 4): each masked field unmasks independently, and a remasked field goes back to
    the pool until the unmasking odds pick it. Within one pass the drawn fields go in ORDER, so legality sees earlier draws."""
    if arm not in RANDOM_ARMS: raise ValueError(f"arm must be one of {sorted(RANDOM_ARMS)}")
    stats = stats if stats is not None else {}
    ask_w = torch.full((n,), M.NAN if ask is None else ask, device=M.DEV)
    null_w = torch.full((n,), M.NAN, device=M.DEV)
    x = torch.zeros(n, M.COLS, dtype=torch.long, device=M.DEV)
    conf = torch.full((n, M.COLS), float("inf"), device=M.DEV)

    def step(unmask_odds, rule=None):
        masked = x == 0
        odds = unmask_odds(masked) if callable(unmask_odds) else unmask_odds
        reveal = masked & (torch.rand_like(conf) < odds)
        remask = rule(conf) if rule is not None else torch.zeros_like(masked)
        targets = sorted(reveal.any(0).nonzero().flatten().tolist(), key=M.ORDER.index)
        if targets:
            fill_pass(w, x, conf, targets, set(), reveal, ask_w, null_w, temperature=temperature, guidance=guidance,
                      lookahead=lookahead, refusals=refusals, top_p=top_p)
            stats["passes"] = stats.get("passes", 0) + 1
        x[remask] = 0; conf[remask] = float("inf")
        stats["remasked"] = stats.get("remasked", 0) + int(remask.sum())

    def fill(passes, start, end):
        for i in range(passes):
            a_t, a_s = start + (end - start) * i / passes, start + (end - start) * (i + 1) / passes
            step(1.0 if i == passes - 1 and end == 1.0 else (a_s - a_t) / (1 - a_t))

    if arm == "rand_plain":
        fill(M.COLS, 0.0, 1.0)
    elif arm == "species_then_random":
        _species_then_random(step, fill, conf)
    else:
        _random_remask_loop(arm, step, fill)
    if (x == 0).any(): raise RuntimeError("a field is still masked after the last pass")
    return x


def _score_confidence_pass(w, x, lp, lpn, al, masked, columns, guidance, guided, temps, top_ps, lookahead, by_entropy, first_temperature):
    score = torch.full((len(x), M.COLS), -1.0, device=M.DEV); parts = {}
    for c in columns:
        base_temp = _field_setting(c, temps)
        draw_temp = first_temperature if by_entropy and first_temperature is not None else base_temp
        p_keep = _field_setting(c, top_ps)
        lc = w._column(lp, c)
        if guided: lc = guidance * lc + (1 - guidance) * w._column(lpn, c)
        legal, plain, kept, fallback = w._legal_parts(x, c, lookahead, al=al)
        logits = lc.masked_fill(~legal, -float("inf"))
        if p_keep < 1.0: logits = logits.masked_fill(~top_p_keep(torch.softmax(logits / base_temp, -1), p_keep), -float("inf"))
        p = torch.softmax(logits / base_temp, -1)
        ranked = torch.special.entr(p).sum(1) if by_entropy else p.max(1).values
        if draw_temp != base_temp: p = torch.softmax(logits / draw_temp, -1)
        parts[c] = (lc, plain, kept, fallback, p, draw_temp)
        score[:, c] = torch.where(masked[:, c], ranked, torch.full_like(score[:, c], -1.0))
    return score, parts


@torch.no_grad()
def sample_confidence(w, n, ask=None, guidance=1.0, temperature=1.0, lookahead=True, refusals=None, order=None,
                      first_pick="confidence", first_temperature=None, passes=None, score_log=None, top_p=1.0):
    """(teams [n, COLS], fill order [n, COLS], non-top [n, COLS]). Each pass, every team fills the masked field whose
    drawing distribution puts the most probability on one value, and draws that field's value from the distribution.
    Ties go to the lower column. first_pick="entropy": pass 1 instead fills the masked field whose drawing distribution
    at `temperature` has the highest entropy, -sum p log p, and draws its value at first_temperature (issue #333)."""
    if first_pick not in ("confidence", "entropy"): raise ValueError(f"unknown first pick {first_pick!r}")
    temps, top_ps = _pair(temperature), _pair(top_p)
    ask_w = torch.full((n,), M.NAN if ask is None else ask, device=M.DEV)
    null_w = torch.full((n,), M.NAN, device=M.DEV)
    x = torch.zeros(n, M.COLS, dtype=torch.long, device=M.DEV)
    fill_order = torch.zeros(n, M.COLS, dtype=torch.long)
    non_top = torch.zeros(n, M.COLS, dtype=torch.bool)
    guided = ask is not None and guidance != 1.0
    for step in range(M.COLS if passes is None else passes):
        by_entropy = step == 0 and first_pick == "entropy" and order is None
        t = torch.full((n,), 1.0 - step / M.COLS, device=M.DEV)
        lp = w._logprobs(x, t, ask_w); lpn = w._logprobs(x, t, null_w) if guided else None
        al, masked = M.allowed(w.tables, x), x == 0
        columns = [order[step]] if order is not None else masked.any(0).nonzero().flatten().tolist()
        score, parts = _score_confidence_pass(w, x, lp, lpn, al, masked, columns, guidance, guided, temps, top_ps,
                                              lookahead, by_entropy, first_temperature)
        if score_log is not None: score_log.append(score.cpu())
        pick = torch.full((n,), order[step], device=M.DEV) if order is not None else score.argmax(1)
        before = x.clone()
        for c in sorted(set(pick.tolist())):
            rows = pick == c
            lc, plain, kept, fallback, p, draw_temp = parts[c]
            v = torch.multinomial(p if bool(rows.all()) else p[rows], 1).squeeze(1)
            _record_refusal(refusals, lc / draw_temp, plain, kept, fallback, rows, c, before, v, p)
            non_top[rows.cpu(), c] = (v != p[rows].argmax(1)).cpu()
            x[rows, c] = v + 1 if c >= M.STAT0 else v
        fill_order[:, step] = pick.cpu()
    if passes is None and (x == 0).any(): raise RuntimeError("a field is still masked after the last pass")
    return x, fill_order, non_top


_MODE_RESOLVER = {
    "species_first": ("species_first", None, False),
    "species_first_corrected_t": ("species_first", None, True),
    "random": ("random", "rand_plain", False),
    "rand_plain": ("random", "rand_plain", False),
    "species_then_random": ("species_then_random", "species_then_random", False),
    "rand_remask_conf": ("random", "rand_remask_conf", False),
    "rand_remask_even": ("random", "rand_remask_even", False),
    "confidence": ("confidence", None, False),
    "entropy_init": ("entropy_init", None, False),
    **{arm: ("species_first", arm, False) for arm in REMASK_ARMS},
}


class Sampler84:
    """Configured 84-field sampler for ask_grid84.draw and experiment runners: sample() and to_paste(),
    recording one Refusals diagnostic record per team in last_refusals."""

    def __init__(self, w, mode="species_first", *, decoder=None, arm=None, lookahead=None, corrected_t=False,
                 order=None, first_pick=None, first_temperature=None, temperature=1.0, top_p=1.0,
                 guidance=1.0, remask_share=None, record_draws=False, loops=None, tables=True):
        chosen = decoder or arm or mode
        if chosen not in _MODE_RESOLVER:
            raise ValueError(f"unknown sampler mode {chosen!r}; choose one of {sorted(_MODE_RESOLVER)}")
        resolved_decoder, resolved_arm, mode_corrected_t = _MODE_RESOLVER[chosen]
        if arm is not None and resolved_arm is None: resolved_arm = arm
        if resolved_decoder == "entropy_init":
            first_pick = "entropy" if first_pick is None else first_pick
            first_temperature = FIRST_TEMPERATURE if first_temperature is None else first_temperature
        self.w = self.model = w
        self.decoder = resolved_decoder
        self.arm = resolved_arm
        self.corrected_t = bool(corrected_t or mode_corrected_t)
        self.lookahead = (resolved_decoder in ("confidence", "entropy_init")) if lookahead is None else bool(lookahead)
        self.order = order
        self.first_pick = "confidence" if first_pick is None else first_pick
        self.first_temperature = first_temperature
        self.temperature = temperature
        self.top_p = top_p
        self.guidance = float(guidance)
        self.remask_share = remask_share
        self.record_draws = bool(record_draws)
        self.loops = loops
        self.tables = bool(tables)
        self.stats = {}
        self.last_refusals = None

    @torch.no_grad()
    def sample(self, n, ask=None, guidance=None, temperature=None, seed=None, refusals=None):
        g = self.guidance if guidance is None else float(guidance)
        temp = self.temperature if temperature is None else temperature
        rec = Refusals(n, draws=self.record_draws) if refusals is None else refusals
        if self.decoder in ("random", "species_then_random"):
            if seed is not None: torch.manual_seed(seed)
            x = sample_random(self.w, n, ask, self.arm, self.stats, guidance=g, temperature=temp,
                              lookahead=self.lookahead, refusals=rec, top_p=self.top_p)
        elif self.decoder in ("confidence", "entropy_init"):
            if seed is not None: torch.manual_seed(seed)
            x, _, _ = sample_confidence(self.w, n, ask=ask, guidance=g, temperature=temp, lookahead=self.lookahead,
                                        refusals=rec, order=self.order, first_pick=self.first_pick,
                                        first_temperature=self.first_temperature, top_p=self.top_p)
        elif self.arm in REMASK_ARMS:
            if seed is not None: torch.manual_seed(seed)
            x = sample_remask(self.w, n, ask, self.arm, self.stats, guidance=g, temperature=temp,
                              top_p=self.top_p, lookahead=self.lookahead, refusals=rec, remask_share=self.remask_share)
        else:
            x = sample_species_first(self.w, n, ask=ask, guidance=g, temperature=temp, top_p=self.top_p,
                                     seed=seed, lookahead=self.lookahead, loops=self.loops,
                                     corrected_t=self.corrected_t, refusals=rec, tables=self.tables, order=self.order)
        self.last_refusals = rec.records() if hasattr(rec, "records") else None
        return x

    def to_paste(self, row, repair=False):
        return self.w.to_paste(row, repair=repair)
