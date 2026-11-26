"""Nats check: the exact held-out log-likelihood of an 84-field masked model in a fixed fill order (docs/nats-check.md).

A masked model that fills one field per pass in a fixed order gives every team an exact probability: the product of
the probabilities its sampler gave each true value when that field's turn came. The check reveals a held-out team's
fields in that order, one pass per field, and adds up -log p, with each field's probabilities renormalised over the
values the sampler may draw, as sampler84.sample_species_first does. Nothing is bounded or sampled, so two models
scored on the same teams differ only by what they learned. Lower nats = the model finds held-out teams more likely.

    python src/nats_check.py selftest                          # the scorer on a toy distribution with known entropy
    python src/nats_check.py score linear_filtered [...]       # a name in MODELS, or a checkpoint name
    python src/nats_check.py report                            # tables, intervals, references, rank agreement, chart

What `score` measures for one checkpoint, on 1,024 held-out teams (validation rows of the default table; the seed and
the teams of the 2026-11-15 and 2026-11-23 probes, so their numbers stay comparable). `score` looks up the table the
checkpoint trained on and refuses to score when any of these teams is one of its training rows:

  fixed fill order   species-first, `diffusion84.ORDER`, the order every ask grid samples in
  tables on / off    the sampler with the legal Pokémon construction tables, and the sampler without them, which
                     also drops the Stat Point budget. The 2026-11-23 convention (no renormalising, budget kept in
                     both modes) is stored beside it as `reference_convention`, to reproduce the old numbers
  per field          nats per team for each field type, and the same per column
  per team           total nats of each team. The per-field terms must sum to it: 8 teams are totalled again by a
                     second route (each team's 84 partly filled states in one batch) and must match
  sampler check      the sampler draws 8 teams and reports the probability it gave each drawn value; the scorer must
                     give those teams the same per-field nats. `score` writes nothing when either check fails
  tables-rejected    teams holding a value the tables-on sampler cannot draw when its turn comes (probability 0).
                     They are listed apart and left out of the fixed-order means of both modes. A team only another
                     order rejects is left out of that order's gap alone
  moves-first        the 2026-11-15 reference order, for the species-first minus moves-first gap
  16-order gap       the first 256 of the teams scored in 16 random fill orders. Gap = mean nats over the 16 orders
                     minus nats in the fixed order, per team. A model that is one joint distribution has gap 0
  first-try legality teams drawn with the tables and the Stat Point budget off that Showdown accepts on the first try

Every interval is a paired bootstrap over teams (2,000 resamples of the per-team difference, 2.5% and 97.5%).
A uniform discrete diffusion model has no hidden fields to fill in an order, so it has no fixed-order likelihood here.
"""
import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
OUT = Path(os.environ.get("NATS_CHECK_OUT", Path.home() / "vgc-data/results/nats_check"))
TEAMS, TEAM_SEED = 1024, 20261114                        # the held-out draw of the 2026-11-15 and 2026-11-23 probes
ORDERS, ORDER_SEED = 16, 20261125
ORDER_TEAMS = 256                                        # the first teams of the draw, scored in every random order (35 s per order per 1,024 teams)
BOOTSTRAPS, BOOTSTRAP_SEED = 2000, 0
LEGALITY_TEAMS = 256                                     # standard error at most 0.031 on the share
FIELD_TYPES = ("species", "ability", "item", "move", "Stat Alignment", "Stat Points")
MODES = ("tables on", "tables off")
SUM_TEAMS, SUM_TOLERANCE = 8, 1e-3                       # teams per scorer check; nats allowed between two routes (float32 passes)

# Code ID -> checkpoint, reader name (the difference first), role, and the stored species-first ask grid at guidance 2.
# References are the checkpoints the 2026-11-15 and 2026-11-23 probes scored; `score` reruns them on the same teams.
MODELS = {
    "per_field": dict(checkpoint="stat84_shuffled_noise_per_field_es3_cap80", role="reference", grid="sched_species_first",
                      reader="Per-field masking schedule", note="tables in the loss, table before the wrong-stone filter"),
    "linear": dict(checkpoint="stat84_shuffled_noise_linear_es3_cap80", role="reference", grid="sched_linear_species_first",
                   reader="Linear masking schedule", note="tables in the loss, table before the wrong-stone filter"),
    "no_table_loss_ep33": dict(checkpoint="stat84_shuffled_noise_linear_no_table_loss_es3_cap80_stopped_ep33", role="reference", grid=None,
                               reader="No tables in the loss, epoch 33", note="linear schedule; stopped early (issue #361)"),
    "linear_filtered": dict(checkpoint="stat84_shuffled_noise_linear_filtered_es3_cap80", role="new", grid="linear_masked_filter",
                            reader="Wrong-stone teams filtered", note="linear schedule, learning rate 4e-4 (issue #385)"),
    "linear_filtered_gidd_lr": dict(checkpoint="stat84_shuffled_noise_linear_filtered_lr5e4_min5e5_es3_cap80", role="new",
                                    grid="shuffled_linear_filtered_gidd_lr", reader="Cosine learning rate, wrong-stone teams filtered",
                                    note="peak 5e-4 falling on a cosine to 5e-5, the GIDD paper's recipe (issue #431)"),
    "no_table_loss": dict(checkpoint="stat84_shuffled_noise_linear_no_table_loss_es3_cap80", role="new", grid=None,
                          reader="No tables in the loss, 80 epochs", note="linear schedule, filtered table (issue #361 rerun)"),
}
# Stored numbers the report sets each model against. `ours` names the measure of this check that is closest;
# `same` says whether it is the same quantity on the same teams (then a rerun must reproduce it) or only a neighbour.
REFERENCES = [
    dict(date="2026-10-04", what="Masked cross-entropy at random masks, 48-field model (loss diagnosis)", value=2.57, unit="nats per field",
         ours="block48_per_field_off", same=False, why="random masks instead of a fill order, a 48-field model, 500 other test teams",
         source="src/loss_diagnosis.py"),
    dict(date="2026-10-07", what="Fixed-order nats, 48-field model before the retrain, no tables", value=2.362, unit="nats per field",
         ours="block48_per_field_off", same=False, why="48-field model, 500 other test teams, 4 slot shuffles", source="docs/loss-down.md"),
    dict(date="2026-10-07", what="Fixed-order nats, 48-field mixture after the retrain, tables on", value=1.418, unit="nats per field",
         ours="block48_per_field", same=False, why="48-field model, 500 other test teams, 4 slot shuffles", source="docs/loss-down.md"),
    dict(date="2026-11-15", what="Species-first minus moves-first gap, per-field schedule", value=18.22, unit="nats per team",
         ours="reference_moves_first_gap", same=False, why="same seed, but the table still held wrong-stone teams (2 fields at the floor)",
         source="mdlm_reading/order_consistency.json", model="per_field"),
    dict(date="2026-11-15", what="Species-first minus moves-first gap, linear schedule", value=12.33, unit="nats per team",
         ours="reference_moves_first_gap", same=False, why="same seed, but the table still held wrong-stone teams (2 fields at the floor)",
         source="mdlm_reading/order_consistency.json", model="linear"),
]
for _model, _on, _off, _gap_on, _gap_off in (("per_field", 124.055, 207.955, 18.150, -7.150), ("linear", 127.152, 199.752, 12.461, -3.461),
                                             ("no_table_loss_ep33", 135.212, 163.478, 8.736, 1.713)):
    for _mode, _nats, _gap in (("tables on", _on, _gap_on), ("tables off", _off, _gap_off)):
        _suffix = "" if _mode == "tables on" else "_off"
        REFERENCES.append(dict(date="2026-11-23", what=f"Fixed-order nats, {_mode}", value=_nats, unit="nats per team", ours=f"reference_nats{_suffix}",
                               same=True, model=_model, source="mdlm_reading/order_gap_tables/order_gap_tables.json"))
        REFERENCES.append(dict(date="2026-11-23", what=f"Species-first minus moves-first gap, {_mode}", value=_gap, unit="nats per team",
                               ours=f"reference_moves_first_gap{_suffix}", same=True, model=_model, source="mdlm_reading/order_gap_tables/order_gap_tables.json"))
REPRODUCE_TOLERANCE = 0.05                               # nats per team; a rerun of a `same` reference must land this close


# ---------------------------------------------------------------- the scorer, free of project imports

def chain_log_likelihood(true_log_prob, rows, order):
    """[B, C] float64: log-probability of each true value of `rows` when the fields are revealed in `order`.
    true_log_prob(x, c) -> [B] is the model's log-probability of rows[:, c] given the revealed fields x (0 = hidden).
    Each row's total is the exact log-likelihood of the team under the sampler that fills fields in this order."""
    order = [int(c) for c in order]
    if sorted(order) != list(range(rows.shape[1])): raise ValueError("a fill order must name every column once")
    x = torch.zeros_like(rows)
    out = torch.zeros(rows.shape, dtype=torch.float64)
    for c in order:
        out[:, c] = true_log_prob(x, c).detach().cpu().double()
        x[:, c] = rows[:, c]
    return out


def random_orders(count, columns, seed=ORDER_SEED):
    """`count` fill orders, each a seeded uniform shuffle of every column; the same for every model and team."""
    rng = np.random.default_rng(seed)
    return [rng.permutation(columns).tolist() for _ in range(count)]


def paired_bootstrap(difference, resamples=BOOTSTRAPS, seed=BOOTSTRAP_SEED):
    """Mean and 95% interval of a per-team difference: teams are resampled with replacement, each keeping its pair."""
    d = np.asarray(difference, np.float64)
    if not len(d): return dict(mean=None, lo=None, hi=None, teams=0)
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), (resamples, len(d)))].mean(1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return dict(mean=float(d.mean()), lo=float(lo), hi=float(hi), teams=len(d))


def sum_check(per_field, totals):
    """Largest absolute difference, in nats, between a team's per-field terms summed and its stored total."""
    return float(np.abs(np.asarray(per_field, np.float64).sum(1) - np.asarray(totals, np.float64)).max())


def order_gap(fixed_nats, order_nats):
    """The 16-order gap from per-team nats: fixed_nats [B], order_nats [orders, B]. Positive = the fixed order scores
    the team better than a random order does. spread: a team's standard deviation across the orders."""
    order_nats = np.asarray(order_nats, np.float64)
    gap = order_nats.mean(0) - np.asarray(fixed_nats, np.float64)
    return dict(paired_bootstrap(gap), orders=len(order_nats), share_fixed_better=float((gap > 0).mean()),
                mean_spread=float(order_nats.std(0).mean()), mean_range=float((order_nats.max(0) - order_nats.min(0)).mean()),
                mean_order_nats=[float(v) for v in order_nats.mean(1)])


def spearman(a, b):
    """Spearman rank correlation with average ranks for ties; None when either side is constant."""
    def ranks(v):
        v = np.asarray(v, np.float64); order = v.argsort(kind="stable"); r = np.empty(len(v)); r[order] = np.arange(len(v))
        for value in np.unique(v): r[v == value] = r[v == value].mean()
        return r
    ra, rb = ranks(a), ranks(b)
    if ra.std() == 0 or rb.std() == 0: return None
    return float(np.corrcoef(ra, rb)[0, 1])


def rank_agreement(nats, other, higher_is_better=True):
    """Does lower nats go with a better `other`? Pairs of models ordered the same way by both, of the pairs neither
    measure ties, plus Spearman of (-nats, other). Both are dicts keyed by model; only shared keys count."""
    keys = [k for k in nats if other.get(k) is not None and nats[k] is not None]
    sign = 1.0 if higher_is_better else -1.0
    good = {k: -nats[k] for k in keys}; value = {k: sign * other[k] for k in keys}
    agree = disagree = tied = 0
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            product = (good[a] - good[b]) * (value[a] - value[b])
            if product > 0: agree += 1
            elif product < 0: disagree += 1
            else: tied += 1
    rho = spearman([good[k] for k in keys], [value[k] for k in keys]) if len(keys) >= 3 else None
    verdict = ("too few models" if len(keys) < 3 else "cannot tell: every pair ties" if agree + disagree == 0 else "agrees" if disagree == 0
               else "disagrees" if agree == 0 else "partly agrees")
    if tied > agree + disagree and agree + disagree: verdict += ", but most pairs tie"
    return dict(models=keys, pairs_agree=agree, pairs_disagree=disagree, pairs_tied=tied, spearman=rho, verdict=verdict)


# ---------------------------------------------------------------- the toy check

class ToyJoint:
    """A small joint distribution with exact conditionals: column c takes tokens 1..sizes[c], 0 = hidden."""

    def __init__(self, sizes=(3, 4, 2, 3), seed=7):
        rng = np.random.default_rng(seed)
        self.sizes = tuple(sizes)
        p = rng.gamma(0.6, size=self.sizes)              # uneven and correlated: the columns are far from independent
        self.p = torch.as_tensor(p / p.sum(), dtype=torch.float64)

    @property
    def entropy(self):
        return float(-(self.p * self.p.log()).sum())

    def outcomes(self):
        """Every outcome as tokens [N, C] and its probability [N]."""
        grid = torch.cartesian_prod(*[torch.arange(1, s + 1) for s in self.sizes])
        return grid, self.p[tuple((grid - 1).T)]

    def marginal_entropy_sum(self):
        """The nats a scorer pays if it ignores the revealed fields: the sum of the columns' own entropies."""
        total = 0.0
        for c in range(len(self.sizes)):
            m = self.p.sum([d for d in range(len(self.sizes)) if d != c])
            total += float(-(m * m.log()).sum())
        return total

    def true_log_prob(self, rows, blind=False):
        """The scorer's callback for `rows`. blind=True ignores the revealed fields: a deliberately broken scorer."""
        def log_prob(x, c):
            out = torch.zeros(len(rows), dtype=torch.float64)
            for i in range(len(rows)):
                joint = self.p
                for d in range(len(self.sizes) - 1, -1, -1):          # highest dimension first, so indices stay put
                    if d == c: continue
                    joint = joint.sum(d, keepdim=True) if blind or int(x[i, d]) == 0 else joint.index_select(d, x[i, d:d + 1] - 1)
                column = joint.flatten()
                out[i] = (column / column.sum()).log()[int(rows[i, c]) - 1]
            return out
        return log_prob


def selftest(orders=ORDERS):
    """The scorer on the toy distribution. Passes when (1) the probability-weighted mean nats equals the entropy in the
    fixed order and in every random order, (2) every team's per-field terms sum to -log of its joint probability,
    (3) the order gap is 0, and (4) a scorer that ignores the revealed fields is caught: it pays the marginal entropies."""
    toy = ToyJoint()
    rows, prob = toy.outcomes()
    columns = rows.shape[1]
    fixed = list(range(columns))
    nats = -chain_log_likelihood(toy.true_log_prob(rows), rows, fixed)
    by_order = np.stack([(-chain_log_likelihood(toy.true_log_prob(rows), rows, o)).sum(1).numpy() for o in random_orders(orders, columns)])
    blind = -chain_log_likelihood(toy.true_log_prob(rows, blind=True), rows, fixed)
    weights = prob.numpy()
    result = dict(outcomes=len(rows), entropy=toy.entropy, fixed_order_nats=float((weights * nats.sum(1).numpy()).sum()),
                  worst_random_order_error=float(np.abs(by_order @ weights - toy.entropy).max()),
                  sum_error=sum_check(nats.numpy(), -prob.log().numpy()),
                  order_gap=float(np.abs(by_order - nats.sum(1).numpy()).max()),
                  blind_scorer_nats=float((weights * blind.sum(1).numpy()).sum()), marginal_entropy_sum=toy.marginal_entropy_sum())
    result["passes"] = bool(abs(result["fixed_order_nats"] - result["entropy"]) < 1e-9 and result["worst_random_order_error"] < 1e-9
                            and result["sum_error"] < 1e-9 and result["order_gap"] < 1e-9
                            and abs(result["blind_scorer_nats"] - result["marginal_entropy_sum"]) < 1e-9
                            and result["blind_scorer_nats"] > result["entropy"] + 0.01)
    return result


# ---------------------------------------------------------------- the 84-field model

def _project():
    """The project modules, imported on first use so `selftest` runs on a machine with no data or checkpoints.
    The modules the score depends on must load from this checkout, not from the copy in /tmp/vgc-pilot/src."""
    sys.path.insert(0, "/tmp/vgc-pilot/src"); sys.path.insert(0, str(HERE))
    import lossdown as L                                 # before diffusion84: its imports put /tmp/vgc-pilot/src first
    sys.path.insert(0, str(HERE))
    import diffusion84 as M
    import sampler84 as S
    import train84 as TR
    for module in (L, M, S, TR):
        if Path(module.__file__).resolve().parent != HERE:
            raise SystemExit(f"{module.__name__} loaded from {module.__file__}, not from this checkout ({HERE})")
    return L, M, S, TR


def module_files():
    """Where the modules behind a score were loaded from, and whether the 48-field module, which older imports take
    from /tmp/vgc-pilot/src, is the same file as this checkout's."""
    import diffusion as D
    loaded = Path(D.__file__).resolve()
    return dict(diffusion=str(loaded), diffusion_matches_checkout=loaded.read_bytes() == (HERE / "diffusion.py").read_bytes())


def column_types(M):
    """The field type of each of the 84 columns."""
    from encode import NF
    names = {0: "species", 1: "ability", 2: "item", 7: "Stat Alignment"}
    return ["Stat Points" if c >= M.STAT0 else names.get(c % NF, "move") for c in range(M.COLS)]


def moves_first(M):
    """The 2026-11-15 reference order: every move, then the rest of the species-first order."""
    types = column_types(M)
    return [c for c in M.ORDER if types[c] == "move"] + [c for c in M.ORDER if types[c] != "move"]


def held_out_teams(TR, count=TEAMS, seed=TEAM_SEED):
    """`count` validation rows of the default training table that are not also training rows, drawn with `seed`."""
    d = TR.load(TR.DATA)
    val = d["all_val"].astype(bool)
    train_rows = {r.tobytes() for r in d["all_X"][~val]}
    rows = np.array([r for r in d["all_X"][val] if r.tobytes() not in train_rows])
    picked = rows[np.random.default_rng(seed).permutation(len(rows))[:count]]
    return picked, dict(table=d["table"], table_sha256=d["sha256"], held_out=len(rows), used=len(picked), seed=seed,
                        teams_sha256=hashlib.sha256(np.ascontiguousarray(picked).tobytes()).hexdigest())


def held_out_check(info, picked, tables_folder):
    """Did this checkpoint train on any scored team? Finds the training table its run info names by SHA-256, counts
    the scored teams among that table's rows outside the validation split, and checks the run's training-row count is
    that split's labelled rows minus the rows the tables dropped (so no validation row was folded in).
    status: "held out", "LEAK", or "unverified" with the reason."""
    info = info or {}
    sha = info.get("data_sha256")
    if not sha: return dict(status="unverified", why="the run info names no training table")
    table = next((p for p in sorted(Path(tables_folder).glob("lossdata84*.npz")) if hashlib.sha256(p.read_bytes()).hexdigest() == sha), None)
    if table is None: return dict(status="unverified", why=f"no table with SHA-256 {sha[:12]} in {tables_folder}")
    d = np.load(table)
    train = ~d["all_val"].astype(bool)
    train_rows = {r.tobytes() for r in d["all_X"][train].astype(np.int64)}
    overlap = sum(r.tobytes() in train_rows for r in picked)
    expected = int((train & ~np.isnan(d["all_W"])).sum()) - int(info.get("dropped_by_tables", 0)) if info.get("labelled_only", True) else int(train.sum())
    out = dict(table=table.name, scored_teams_in_training_rows=int(overlap), n_train=info.get("n_train"), expected_n_train=expected)
    if overlap: return dict(out, status="LEAK", why=f"{overlap} scored teams are training rows of {table.name}")
    if info.get("n_train") != expected:
        return dict(out, status="unverified", why=f"the run trained on {info.get('n_train')} rows, not the {expected} of {table.name}'s training split")
    return dict(out, status="held out")


class Model84Fields:
    """true_log_prob for a Model84 on `rows`: the probability the species-first sampler gives each true value, with no
    ask. The sampler takes the model's log-probabilities for the field and renormalises them over the values it may
    draw (sampler84.sample_species_first): with the tables on, the tables' values without [MASK] and without an empty
    first move; with the tables off, every value but [MASK] and no Stat Point budget.

    Also kept, per field: `reference`, the log-probability before that renormalisation, with the Stat Point budget on
    in both modes (the 2026-11-15 and 2026-11-23 probes' convention, kept to reproduce their numbers), and `rejected`,
    the true values the sampler cannot draw."""

    def __init__(self, M, model, rows, tables=True):
        self.M, self.model, self.rows, self.tables = M, model, rows, tables
        self.ask = torch.full((len(rows),), float("nan"), device=rows.device)
        self.rejected = torch.zeros(rows.shape, dtype=torch.bool)
        self.reference = torch.zeros(rows.shape, dtype=torch.float64)

    @torch.no_grad()
    def forward(self, x, raw=False):
        """(log-probabilities under the reference masks, the same with no mask at all when raw=True, the tables' masks)."""
        M = self.M
        real = M.allowed(self.model.tables, x)
        masks = real if self.tables else {k: (v if k == "stat" else torch.ones_like(v)) for k, v in real.items()}
        hidden_share = 1 - (x != 0).sum(1).float() / M.COLS
        lp = self.model._logprobs(x, hidden_share, self.ask[:len(x)], al=masks)
        return lp, (self.model._logprobs(x, hidden_share, self.ask[:len(x)], tables=False) if raw else None), real

    def needs_raw(self, c):
        """With the tables off the sampler drops the Stat Point budget too, so a Stat Point needs the unmasked pass."""
        return not self.tables and c >= self.M.STAT0

    def index(self, c):
        """Where each row's true value of column c sits in the column's outputs: a Stat Point token is its value + 1."""
        return ((self.rows[:, c] - 1) if c >= self.M.STAT0 else self.rows[:, c]).view(-1, 1)

    def read(self, lp, raw, real, x, c):
        """(sampler log-probability, reference log-probability, drawable) [B] of the true value of column c."""
        index = self.index(c)
        reference = self.model._column(lp, c)
        scores = self.model._column(raw, c) if self.needs_raw(c) else reference
        legal = (self.model._legal_parts(x, c, al=real) if self.tables else self.model._unmasked_parts(x, c))[0]
        sampler = scores.gather(1, index).squeeze(1) - torch.logsumexp(scores.masked_fill(~legal, -float("inf")), 1)
        drawable = legal.gather(1, index).squeeze(1)
        return sampler.masked_fill(~drawable, -float("inf")), reference.gather(1, index).squeeze(1), drawable

    def __call__(self, x, c):
        sampler, reference, drawable = self.read(*self.forward(x, raw=self.needs_raw(c)), x, c)
        self.rejected[:, c] = ~drawable.cpu()
        self.reference[:, c] = reference.cpu().double()
        return sampler


def score_order(M, model, rows, order, tables):
    """(nats [B, 84], reference-convention nats [B, 84], rejected [B, 84]) for one fill order; nats are inf where rejected."""
    fields = Model84Fields(M, model, rows, tables)
    nats = -chain_log_likelihood(fields, rows, order).numpy()
    return nats, -fields.reference.numpy(), fields.rejected.numpy()


@torch.no_grad()
def second_route_totals(M, model, rows, order, tables):
    """Team totals in nats [B] by another route than chain_log_likelihood: one batch holds a single team's 84 partly
    filled states, state k showing the first k fields of the order, and row k is read at the order's field k."""
    totals = []
    for row in rows:
        x = torch.zeros(M.COLS, M.COLS, dtype=rows.dtype, device=rows.device)
        for step, c in enumerate(order): x[step + 1:, c] = row[c]
        fields = Model84Fields(M, model, row.expand(M.COLS, -1), tables)
        lp, raw, real = fields.forward(x, raw=not tables)
        totals.append(-sum(float(fields.read(lp, raw, real, x, c)[0][step]) for step, c in enumerate(order)))
    return np.array(totals)


class DrawLog:
    """Stands in for sampler84.Refusals: keeps the log-probability the sampler gave each value it drew."""

    def __init__(self, teams, columns):
        self.log_p = torch.zeros(teams, columns, dtype=torch.float64)

    def add(self, *_args, **_kwargs):
        return None

    def note_draw(self, column, _rows, drawn, probs):
        self.log_p[:, column] = probs.gather(1, drawn.view(-1, 1)).squeeze(1).log().cpu().double()


def sampler_error(M, S, model, tables, teams=SUM_TEAMS, seed=TEAM_SEED):
    """Largest gap, in nats per field, between the scorer and the sampler itself: the sampler draws `teams` teams with
    no ask and reports the probability it gave each value it drew; the scorer then scores those teams."""
    log = DrawLog(teams, M.COLS)
    rows = S.sample_species_first(model, teams, seed=seed, refusals=log, tables=tables)
    nats, _, _ = score_order(M, model, rows, M.ORDER, tables)
    return float(np.abs(nats + log.log_p.numpy()).max())


def summarise_mode(scored, kept, kept_moves_first, kept_orders, types):
    """The stored summary of one table mode from its scored arrays. kept [B]: teams the fixed order can draw with the
    tables on; kept_moves_first and kept_orders [first teams] also leave out the teams the other orders cannot draw."""
    nats, reference, mf, mf_reference, order_totals = (scored[k] for k in ("nats", "reference", "moves_first", "moves_first_reference", "orders"))
    columns_of = {t: [c for c, name in enumerate(types) if name == t] for t in FIELD_TYPES}
    block48 = [c for c, name in enumerate(types) if name != "Stat Points"]
    n = order_totals.shape[1]
    totals, k = nats.sum(1), nats[kept]
    gap = lambda a, b, keep: dict(paired_bootstrap((a - b)[keep]), share_fixed_better=float(((a - b)[keep] > 0).mean()) if keep.any() else None)
    finite = lambda v: float(v) if math.isfinite(v) else None
    out = dict(teams=int(kept.sum()), nats=float(k.sum(1).mean()), nats_median=float(np.median(k.sum(1))), nats_all_teams=finite(totals.mean()),
               nats_per_field=float(k.sum(1).mean() / nats.shape[1]), block48_per_field=float(k[:, block48].sum(1).mean() / len(block48)),
               per_field={t: dict(nats_per_team=float(k[:, cols].sum(1).mean()), columns=len(cols), nats_per_column=float(k[:, cols].mean()))
                          for t, cols in columns_of.items()},
               moves_first=dict(teams=int(kept_moves_first.sum()), nats=float(mf[kept_moves_first].mean()), gap=gap(mf, totals, kept_moves_first)),
               order_gap=dict(order_gap(totals[:n][kept_orders], order_totals[:, kept_orders]), teams_scored=n),
               reference_convention=dict(nats=float(reference[kept].sum(1).mean()), block48_per_field=float(reference[kept][:, block48].sum(1).mean() / len(block48)),
                                         moves_first_gap=gap(mf_reference, reference.sum(1), kept_moves_first)))
    out["per_field_sum_error"] = abs(sum(v["nats_per_team"] for v in out["per_field"].values()) - out["nats"])
    return out


def rejected_teams(M, model, rows, rejected, types):
    """One entry per team the fixed-order sampler cannot draw with the tables on: the fields, their values, the line-up."""
    import team84 as T
    from encode import NF
    out = []
    for i in np.flatnonzero(rejected.any(1)):
        slots = T.decode(model.V, rows[i])
        fields = []
        for c in np.flatnonzero(rejected[i]):
            c = int(c); slot = (c // NF) if c < M.STAT0 else (c - M.STAT0) // 6
            value = (slots[slot]["evs"] if c >= M.STAT0 else model.V.decode_field(c, int(rows[i, c])))
            fields.append(dict(column=c, field=types[c], pokemon=slot + 1, species=slots[slot]["species"], value=value))
        out.append(dict(team=int(i), line_up=[s["species"] for s in slots], fields=fields))
    return out


def first_try_legality(model, teams=LEGALITY_TEAMS):
    """Share of `teams` drawn with the tables and the Stat Point budget off that Showdown accepts on the first try,
    after the Stat Alignment is repaired to match the moves (epoch_legality's draw: elite cut 0.531, guidance 2,
    temperature 1, species-first order, one seed)."""
    import epoch_legality as E
    check = E.EpochLegality(teams=teams, tables=False)
    try: found = check(model)
    finally: check.close()
    return dict(share=1 - found["illegal_share"], teams=found["legality_teams"], ask=E.ASK, guidance=E.GUIDANCE,
                rule_shares=found["rule_shares"], seconds=found["legality_seconds"])


def resolve(name):
    """(code ID, checkpoint name) for a name in MODELS or a checkpoint name."""
    if name in MODELS: return name, MODELS[name]["checkpoint"]
    for code, row in MODELS.items():
        if row["checkpoint"] == name: return code, name
    return name, name


def score_modes(M, model, rows, fill_orders, order_teams, code):
    """{mode: arrays} plus, from the tables-on pass, the rejected flags of the fixed, moves-first and random orders."""
    scored, flags = {}, {}
    for mode in MODES:
        tables = mode == "tables on"
        nats, reference, rejected = score_order(M, model, rows, M.ORDER, tables)
        mf, mf_reference, mf_rejected = score_order(M, model, rows, moves_first(M), tables)
        random_scored = [score_order(M, model, rows[:order_teams], o, tables) for o in fill_orders]
        scored[mode] = dict(nats=nats, reference=reference, moves_first=mf.sum(1), moves_first_reference=mf_reference.sum(1),
                            orders=np.stack([n.sum(1) for n, _, _ in random_scored]))
        if tables: flags = dict(fixed=rejected, moves_first=mf_rejected.any(1), orders=np.any([r.any(1) for _, _, r in random_scored], 0))
        print(f"[{code}] {mode}: scored {2 + len(fill_orders)} orders", flush=True)
    return scored, flags


def scorer_checks(M, S, model, rows, scored):
    """Per mode: the per-field terms of the first teams against their second-route totals, and the scorer against the
    sampler's own draw probabilities. Raises SystemExit, before anything is written, when either misses its limit."""
    out = {}
    for mode in MODES:
        tables = mode == "tables on"
        first = np.flatnonzero(np.isfinite(scored[mode]["nats"].sum(1)))[:SUM_TEAMS]
        second = second_route_totals(M, model, rows[torch.as_tensor(first, device=rows.device)], M.ORDER, tables)
        out[mode] = dict(sum_error=sum_check(scored[mode]["nats"][first], second), sum_teams=len(first),
                         sampler_error=sampler_error(M, S, model, tables), sampler_teams=SUM_TEAMS)
        worst = max(out[mode]["sum_error"], out[mode]["sampler_error"])
        if not worst <= SUM_TOLERANCE:
            raise SystemExit(f"{mode}: scorer check failed, nothing written. Per-field terms miss the team total by {out[mode]['sum_error']:.2e} nats; "
                             f"the scorer misses the sampler's own probabilities by {out[mode]['sampler_error']:.2e} (limit {SUM_TOLERANCE:.0e})")
    return out


def score(name, out=OUT, teams=TEAMS, orders=ORDERS, legality_teams=LEGALITY_TEAMS, order_teams=ORDER_TEAMS):
    """Score one checkpoint and write <out>/<code ID>.json (summary) and .npz (per-team, per-field nats)."""
    _, M, S, TR = _project()
    import lossdown as L
    from checkpoints import load_checkpoint
    code, checkpoint = resolve(name)
    path = L.CK / f"{checkpoint}.pt"
    if str(load_checkpoint(path, map_location="cpu").get("kind", "")).startswith("uniform"):
        raise SystemExit(f"{checkpoint}: a uniform discrete diffusion model starts from random values and has no fields hidden, "
                         "so it has no fill order and no fixed-order likelihood. The nats check covers masked models.")
    model = M.Model84.load(path)
    for net, _ in model.nets: net.eval()
    info = model.info or {}
    picked, teams_info = held_out_teams(TR, teams)
    held_out = held_out_check(info, picked, TR.DATA.parent)
    if held_out["status"] == "LEAK": raise SystemExit(f"{checkpoint}: {held_out['why']}; nothing scored")
    if held_out["status"] != "held out": print(f"[{code}] WARNING held-out status unverified: {held_out['why']}", flush=True)
    rows = torch.as_tensor(picked, device=M.DEV)
    types = column_types(M)
    order_teams = min(order_teams, len(picked))
    scored, flags = score_modes(M, model, rows, random_orders(orders, M.COLS), order_teams, code)
    checks = scorer_checks(M, S, model, rows, scored)
    kept = ~flags["fixed"].any(1)
    kept_moves_first, kept_orders = kept & ~flags["moves_first"], kept[:order_teams] & ~flags["orders"]
    result = dict(model=code, checkpoint=checkpoint, reader=MODELS.get(code, {}).get("reader", checkpoint), teams=teams_info, held_out=held_out,
                  fixed_order=list(M.ORDER), random_orders=dict(count=orders, seed=ORDER_SEED, teams=order_teams), modules=module_files(),
                  run=dict(table_loss=info.get("table_loss", True), sched=info.get("sched"), best_epoch=info.get("best_epoch"), val_loss=info.get("best_val_loss")),
                  modes={mode: dict(summarise_mode(scored[mode], kept, kept_moves_first, kept_orders, types), **checks[mode]) for mode in MODES},
                  rejected=dict(fixed_order=rejected_teams(M, model, picked, flags["fixed"], types), teams_fixed_order=int((~kept).sum()),
                                moves_first_only=[int(i) for i in np.flatnonzero(kept & flags["moves_first"])],
                                random_orders_only=[int(i) for i in np.flatnonzero(kept[:order_teams] & flags["orders"])]))
    result["first_try_legality"] = first_try_legality(model, legality_teams) if legality_teams else None
    arrays = dict(kept=kept, kept_moves_first=kept_moves_first, kept_orders=kept_orders)
    for mode, key in (("tables on", "on"), ("tables off", "off")):
        for k, v in scored[mode].items(): arrays[f"{k}_{key}"] = v
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f"{code}.npz", **arrays)
    (out / f"{code}.json").write_text(json.dumps(result, indent=1))
    on, off = result["modes"]["tables on"], result["modes"]["tables off"]
    print(f"[{code}] {result['reader']}: {on['nats']:.2f} nats per team tables on, {off['nats']:.2f} off; "
          f"{orders}-order gap {on['order_gap']['mean']:+.2f} on, {off['order_gap']['mean']:+.2f} off; rejected teams {result['rejected']['teams_fixed_order']}; "
          f"sum error {max(on['sum_error'], off['sum_error']):.1e}, sampler error {max(on['sampler_error'], off['sampler_error']):.1e}; "
          f"held-out check: {held_out['status']}", flush=True)
    return result


# ---------------------------------------------------------------- the report

def grid_measures(grid, results=None):
    """Mean win rate over the 23 asks and novel share of a stored species-first ask grid at guidance 2; None if missing."""
    if grid is None: return None
    path = Path(results or Path.home() / "vgc-data/results") / f"stat_points_84_{grid}.json"
    if not path.exists(): return None
    d = json.loads(path.read_text()); summary = d["summary"]
    novelty = summary.get("novelty")
    if novelty is None:
        _project()
        import grid_novelty
        cells = d["cells"]; cells = list(cells.values()) if isinstance(cells, dict) else cells
        novelty = grid_novelty.from_cells(cells, grid_novelty.training_line_ups())
    return dict(grid=grid, win=float(np.mean([a["win"] for a in summary["asks"]])), novel_share=novelty.get("novel_share"),
                line_ups=novelty.get("line_ups"), teams=summary["teams"], seed0=d.get("seed0"), guidance=d.get("guidance"),
                win_rate_definition=summary.get("win_rate_definition"))


def our_measure(result, key):
    """The number of this check a stored reference is set against. A "reference_" key reads the reference convention."""
    mode = result["modes"]["tables off" if key.endswith("_off") else "tables on"]
    key = key.removesuffix("_off")
    if key.startswith("reference_"):
        mode, key = mode["reference_convention"], key.removeprefix("reference_")
        return mode["moves_first_gap"]["mean"] if key == "moves_first_gap" else mode[key]
    return mode["moves_first"]["gap"]["mean"] if key == "moves_first_gap" else mode[key]


def compare(new, reference, arrays):
    """Paired nats difference, new minus reference, on the teams both keep; negative = the new model is better."""
    keep = arrays[new]["kept"] & arrays[reference]["kept"]
    return {mode: paired_bootstrap((arrays[new][f"nats_{key}"].sum(1) - arrays[reference][f"nats_{key}"].sum(1))[keep])
            for mode, key in (("tables on", "on"), ("tables off", "off"))}


def same_teams(results):
    """Refuse to pair files that scored different teams: every file must hold the same team hash and order settings."""
    key = lambda r: (r["teams"]["teams_sha256"], r["random_orders"]["count"], r["random_orders"]["seed"], r["random_orders"]["teams"])
    groups = {}
    for code, r in results.items(): groups.setdefault(key(r), []).append(code)
    if len(groups) > 1:
        raise SystemExit("the scored files do not hold the same teams and orders, so they cannot be paired: "
                         + "; ".join(f"{', '.join(codes)} ({k[0][:8]}, {k[1]} orders on {k[3]} teams)" for k, codes in groups.items()))


def build_report(out=OUT, grids=None):
    """Everything the check concludes, from the scored files in `out`."""
    out = Path(out)
    results = {p.stem: json.loads(p.read_text()) for p in sorted(out.glob("*.json")) if p.stem not in ("report", "selftest")} if out.is_dir() else {}
    if not results: raise SystemExit(f"no scored checkpoint in {out}; run `nats_check.py score <model>` first")
    same_teams(results)
    arrays = {code: dict(np.load(out / f"{code}.npz")) for code in results}
    order = [c for c in MODELS if c in results] + [c for c in results if c not in MODELS]
    role = lambda c: MODELS.get(c, {}).get("role", "new")
    references, new = [c for c in order if role(c) == "reference"], [c for c in order if role(c) == "new"]
    grid = {c: grid_measures(MODELS.get(c, {}).get("grid"), grids) for c in order}
    report = dict(selftest=selftest(), models=order, new=new, references=references, grid=grid,
                  checks={c: {k: max(results[c]["modes"][m][k] for m in MODES) for k in ("sum_error", "sampler_error")} for c in order},
                  held_out={c: results[c]["held_out"] for c in order},
                  comparisons={n: {r: compare(n, r, arrays) for r in references} for n in new}, stored_references=[])
    for ref in REFERENCES:
        measured_on = ref.get("model")
        row = dict(ref, new={c: our_measure(results[c], ref["ours"]) for c in new},
                   rescored=our_measure(results[measured_on], ref["ours"]) if measured_on in results else None)
        if ref["same"] and row["rescored"] is not None: row["reproduced"] = bool(abs(row["rescored"] - ref["value"]) <= REPRODUCE_TOLERANCE)
        report["stored_references"].append(row)
    nats_on = {c: results[c]["modes"]["tables on"]["nats"] for c in order}
    nats_off = {c: results[c]["modes"]["tables off"]["nats"] for c in order}
    legality = {c: (results[c].get("first_try_legality") or {}).get("share") for c in order}
    win = {c: (grid[c] or {}).get("win") for c in order}
    novel = {c: (grid[c] or {}).get("novel_share") for c in order}
    report["measures"] = dict(nats_tables_on=nats_on, nats_tables_off=nats_off, first_try_legality=legality, win_rate=win, novel_share=novel)
    report["rank_agreement"] = {
        "first-try legality, tables off": dict(against="nats, tables off", **rank_agreement(nats_off, legality)),
        "win rate": dict(against="nats, tables on", **rank_agreement(nats_on, win)),
        "novel share": dict(against="nats, tables on", **rank_agreement(nats_on, novel)),
    }
    (out / "report.json").write_text(json.dumps(report, indent=1))
    return report, results


def interval(b, digits=2):
    return "n/a" if b["mean"] is None else f"{b['mean']:+.{digits}f} [{b['lo']:+.{digits}f}, {b['hi']:+.{digits}f}]"


def markdown(report, results):  # noqa: C901
    """The report as Markdown: one table per question, each named by what it answers."""
    name = lambda c: results[c]["reader"]
    share = lambda v: "n/a" if v is None else f"{v:.1%}"
    first = results[report["models"][0]]
    orders, order_teams = first["random_orders"]["count"], first["random_orders"]["teams"]
    lines = ["# Nats check", "", "Exact held-out log-likelihood under the species-first sampler, in nats per team. Lower is better. "
             f"Teams: {first['teams']['used']:,} held-out teams; means leave out tables-rejected teams.", ""]
    st = report["selftest"]
    lines += ["## Is the scorer right?", "",
              f"- Toy distribution, {st['outcomes']} outcomes: entropy {st['entropy']:.6f} nats, scorer {st['fixed_order_nats']:.6f}; "
              f"worst of 16 random orders off by {st['worst_random_order_error']:.1e}; order gap {st['order_gap']:.1e}. "
              f"{'Passes' if st['passes'] else 'FAILS'}.",
              f"- A scorer that ignores revealed fields pays {st['blind_scorer_nats']:.4f} nats, the sum of the columns' own entropies "
              f"({st['marginal_entropy_sum']:.4f}), so the check can fail.",
              f"- Per-field terms against team totals computed by a second route ({SUM_TEAMS} teams per model and table mode): "
              f"largest gap {max(v['sum_error'] for v in report['checks'].values()):.1e} nats (limit {SUM_TOLERANCE:.0e}).",
              f"- Scorer against the sampler's own probability for each value it drew ({SUM_TEAMS} sampled teams per model and table mode): "
              f"largest gap {max(v['sampler_error'] for v in report['checks'].values()):.1e} nats per field (limit {SUM_TOLERANCE:.0e}).",
              "- Did any model train on a scored team? " + "; ".join(
                  f"{name(c)}: {h['status']}" + (f" ({h['why']})" if h.get("why") else "") for c, h in report["held_out"].items()) + ".", ""]
    lines += ["## Nats per team, tables on and off", "",
              "| Model | Tables on | Tables off | Rejected teams | First-try legality, tables off |", "|---|---|---|---|---|"]
    for c in report["models"]:
        r = results[c]
        lines.append(f"| {name(c)} | {r['modes']['tables on']['nats']:.2f} | {r['modes']['tables off']['nats']:.2f} | "
                     f"{r['rejected']['teams_fixed_order']} | {share(report['measures']['first_try_legality'][c])} |")
    for mode in MODES:
        lines += ["", f"## Nats per team by field type, {mode}", "", "| Model | " + " | ".join(FIELD_TYPES) + " | Total |", "|---" * (len(FIELD_TYPES) + 2) + "|"]
        for c in report["models"]:
            m = results[c]["modes"][mode]
            lines.append(f"| {name(c)} | " + " | ".join(f"{m['per_field'][t]['nats_per_team']:.2f}" for t in FIELD_TYPES) + f" | {m['nats']:.2f} |")
    lines += ["", "## Order gaps", "", "Nats in the other order minus nats in the species-first order, per team. Positive = species-first scores the team better. "
              f"A single joint distribution has gap 0. Random orders: {orders}, on the first {order_teams} teams.", "",
              f"| Model | {orders}-order gap, tables on | {orders}-order gap, tables off | Moves-first gap, tables on | Moves-first gap, tables off |", "|---|---|---|---|---|"]
    for c in report["models"]:
        m = results[c]["modes"]
        lines.append(f"| {name(c)} | {interval(m['tables on']['order_gap'])} | {interval(m['tables off']['order_gap'])} | "
                     f"{interval(m['tables on']['moves_first']['gap'])} | {interval(m['tables off']['moves_first']['gap'])} |")
    lines += ["", "## Tables-rejected teams", ""]
    listed = [(c, t) for c in report["models"] for t in results[c]["rejected"]["fixed_order"]]
    if not listed: lines.append("None in the fixed order, for any model.")
    for c, t in listed:
        fields = "; ".join(f"Pokémon {f['pokemon']} ({f['species']}) {f['field']} = {f['value']}" for f in t["fields"])
        lines.append(f"- {name(c)}, team {t['team']} ({', '.join(t['line_up'])}): {fields}")
    for key, label in (("moves_first_only", "the moves-first order"), ("random_orders_only", "a random order")):
        other = {c: results[c]["rejected"][key] for c in report["models"] if results[c]["rejected"][key]}
        if other: lines.append(f"- Rejected only in {label}, so left out of that gap alone: " + "; ".join(f"{name(c)} teams {v}" for c, v in other.items()))
    lines += ["", "## Each new model against the rescored references", "", "New minus reference, nats per team, paired bootstrap 95%. Negative = the new model is better.", "",
              "| New model | Reference | Tables on | Tables off |", "|---|---|---|---|"]
    for n, by_ref in report["comparisons"].items():
        for r, d in by_ref.items():
            lines.append(f"| {name(n)} | {name(r)} | {interval(d['tables on'])} | {interval(d['tables off'])} |")
    new = report["new"]
    lines += ["", "## Against the stored reference numbers", "",
              "Each row is one stored number, the same measure rescored now on the checkpoint it came from, and that measure on each new model. "
              "Rows from 2026-11-15 and 2026-11-23 use those probes' convention: the model's probabilities before the sampler renormalises them.", "",
              "| Date | Stored number | Measured on | Stored | Rescored now | " + " | ".join(name(c) for c in new) + " |", "|---" * (5 + len(new)) + "|"]
    for row in report["stored_references"]:
        on = name(row["model"]) if row.get("model") in results else row.get("model") or "the 48-field model"
        rescored = "n/a" if row["rescored"] is None else f"{row['rescored']:.3f}" + ((" (reproduced)" if row.get("reproduced") else " (NOT reproduced)") if row["same"] else "")
        lines.append(f"| {row['date']} | {row['what']} | {on} | {row['value']:.3f} {row['unit']} | {rescored} | " + " | ".join(f"{row['new'][c]:.3f}" for c in new) + " |")
    notes = sorted({f"{r['date']}: {r['why']}" for r in report["stored_references"] if not r["same"]})
    lines += ["", "Not the same quantity as ours, so read these as neighbours: " + "; ".join(notes) + "."]
    lines += ["", "## Does the nats ranking agree with the other measures?", "",
              "| Model | Nats, tables on | Nats, tables off | First-try legality, tables off | Win rate | Novel share |", "|---|---|---|---|---|---|"]
    ms = report["measures"]
    for c in report["models"]:
        win = "no grid" if ms["win_rate"][c] is None else f"{ms['win_rate'][c]:.3f}"
        lines.append(f"| {name(c)} | {ms['nats_tables_on'][c]:.2f} | {ms['nats_tables_off'][c]:.2f} | {share(ms['first_try_legality'][c])} | {win} | "
                     f"{'no grid' if ms['novel_share'][c] is None else share(ms['novel_share'][c])} |")
    lines += ["", "| Measure | Set against | Models | Pairs agreeing | Pairs disagreeing | Pairs tied | Spearman | Verdict |", "|---|---|---|---|---|---|---|---|"]
    for measure, a in report["rank_agreement"].items():
        rho = "n/a" if a["spearman"] is None else f"{a['spearman']:+.2f}"
        lines.append(f"| {measure} | {a['against']} | {len(a['models'])} | {a['pairs_agree']} | {a['pairs_disagree']} | {a['pairs_tied']} | {rho} | {a['verdict']} |")
    lines += ["", "Win rate: mean over the 23 asks of a stored species-first ask grid at guidance 2, against the 49 distinct top-50 meta teams, "
              "behaviour-cloning policy on both sides, slots shuffled, one battle per opponent. Novel share: teams whose six species match no "
              "labelled training team. First-try legality: teams drawn with the tables and the Stat Point budget off that Showdown accepts, "
              "after the Stat Alignment is repaired to match the moves."]
    return "\n".join(lines) + "\n"


def chart(report, results, path):
    """Nats per team by field type, one stacked bar per model and table mode."""
    sys.path.insert(0, str(HERE))
    from echarts_chart import write_chart
    rows = [(c, mode) for c in report["models"] for mode in MODES]
    labels = [f"{results[c]['reader']} · {mode}" for c, mode in rows]
    colors = ["#4269d0", "#efb118", "#ff725c", "#6cc5b0", "#a463f2", "#9498a0"]
    series = [dict(name=t, type="bar", stack="nats", color=color, emphasis=dict(focus="series"),
                   data=[round(results[c]["modes"][mode]["per_field"][t]["nats_per_team"], 2) for c, mode in rows])
              for t, color in zip(FIELD_TYPES, colors, strict=True)]
    series[-1]["label"] = dict(show=True, position="right", formatter="js:p => (" + json.dumps(
        [round(results[c]["modes"][mode]["nats"], 1) for c, mode in rows]) + ")[p.dataIndex] + ' nats'")
    option = dict(legend=dict(top=0), grid=dict(left=8, right=70, top=36, bottom=40, containLabel=True),
                  tooltip=dict(trigger="axis", axisPointer=dict(type="shadow"), valueFormatter="js:v => v.toFixed(2) + ' nats'"),
                  xAxis=dict(type="value", name="nats per held-out team (lower = more likely)", nameLocation="middle", nameGap=28),
                  yAxis=dict(type="category", data=labels, inverse=True), series=series)
    best = min(report["models"], key=lambda c: report["measures"]["nats_tables_on"][c])
    return write_chart(option, path, f"{results[best]['reader']} gives held-out teams the fewest nats with the tables on",
                       "Exact log-likelihood under the species-first sampler, split by field type. Tables off also drops the Stat Point budget.",
                       f"{results[best]['teams']['used']:,} held-out teams; tables-rejected teams left out.", height=90 + 46 * len(rows))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("selftest")
    s = sub.add_parser("score"); s.add_argument("models", nargs="+")
    s.add_argument("--teams", type=int, default=TEAMS); s.add_argument("--orders", type=int, default=ORDERS)
    s.add_argument("--order-teams", type=int, default=ORDER_TEAMS)
    s.add_argument("--legality-teams", type=int, default=LEGALITY_TEAMS); s.add_argument("--out", type=Path, default=OUT)
    r = sub.add_parser("report"); r.add_argument("--out", type=Path, default=OUT); r.add_argument("--grids", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.command == "selftest":
        result = selftest(); print(json.dumps(result, indent=1))
        if not result["passes"]: raise SystemExit("nats check selftest FAILED")
        return
    if args.command == "score":
        if not selftest()["passes"]: raise SystemExit("nats check selftest FAILED; nothing scored")
        for name in args.models: score(name, args.out, args.teams, args.orders, args.legality_teams, args.order_teams)
        return
    report, results = build_report(args.out, args.grids)
    (args.out / "report.md").write_text(markdown(report, results))
    page = chart(report, results, args.out / "nats-by-field-type.html")
    print(markdown(report, results)); print(f"report: {args.out / 'report.md'}\nchart: {page}")


if __name__ == "__main__":
    main()
