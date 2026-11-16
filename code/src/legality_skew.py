"""How far the sampler's legality rules move each draw away from the trained model (2026-11-15). No battles.

Training masks every value the legal Pokémon construction tables forbid inside the loss (diffusion84.elbo_terms), and
the sampler's log-probabilities carry the same mask (Model84._logprobs). So the trained model's distribution for a field
is the guided, temperature-scaled softmax a draw starts from. Four rules run only at sampling time, and this measures the
probability share each one removes from that distribution, draw by draw:

  mask_token  the [MASK] value, never drawn (Model84._with_fallback)
  no_moves    the empty first move, removed while a real move is legal (Model84._no_moves)
  tables      values the tables, or for Stat Points the Stat Point budget, forbid given fields drawn earlier in the same
              pass: zero in species-first order, where every pass recomputes the log-probabilities; random order draws
              several fields from one pass's log-probabilities (remask84.fill_pass)
  lookahead   values that would leave a Pokémon whose species is masked with no legal species (lookahead.py)

A fallback draw, where no value is legal, allows every value but [MASK]; it is counted, and only its [MASK] share is
removed. For a draw the sampler renormalizes over what is left, so the removed share equals the total variation
distance between the trained model's distribution and the one the draw uses. Over a team, the sum of -log(1 - removed)
is the log of how much more likely the sampler makes that team than the trained model would along the same path.

Rows: the per-field and linear masking-schedule checkpoints (docs/masking-rate-rows.md), species-first order with the
lookahead off as the ask grids run it, and random order (LLaDA Algorithm 4) with the lookahead on as the ask grids run it.

    python legality_skew.py run        (LS_SMOKE=1: one cell of 8 teams, its own results file)
    python legality_skew.py report
"""
import hashlib
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, "/tmp/vgc-pilot/src")
sys.path.insert(0, str(HERE))
import lossdown as L                                     # before diffusion84: its imports put /tmp/vgc-pilot/src first
sys.path.insert(0, str(HERE))
import diffusion84 as M

RULES = ("mask_token", "no_moves", "tables", "lookahead")
CHECKPOINTS = {"per_field": "stat84_shuffled_noise_per_field_es3_cap80", "linear": "stat84_shuffled_noise_linear_es3_cap80"}
ORDERS = ("species_first", "random")
SMOKE = bool(os.environ.get("LS_SMOKE"))
GUIDANCES, ASK_NUMERATORS, BATCHES, BATCH = (2.0, 4.0, 8.0), (4, 12, 20), 2, 96
if SMOKE: GUIDANCES, ASK_NUMERATORS, BATCHES, BATCH = (2.0,), (12,), 1, 8
OUT = Path.home() / "vgc-data/results" / f"legality_skew{'_smoke' if SMOKE else ''}.json"
READER = {"per_field": "Per-field masking schedule", "linear": "Linear masking schedule",
          "species_first": "species-first order", "random": "random order (LLaDA Algorithm 4), lookahead on"}


def field_type(c):
    """species, ability, item, move, nature (Stat Alignment) or stat (Stat Points) for column c."""
    return "stat" if c >= M.STAT0 else L.KEY_OF_COL[c]


class SkewLog:
    """A diffusion84.Refusals stand-in that records, for every draw, the share each sampling-only rule removed.

        log = SkewLog(model, n); model.sample(n, ..., refusals=log)   or   remask84.sample_random(..., refusals=log)
    """
    def __init__(self, w, n):
        self.w = w
        self.team_log_ratio = torch.zeros(n, dtype=torch.float64)
        self.team_fallbacks = torch.zeros(n, dtype=torch.long)
        self.sums = defaultdict(lambda: dict(draws=0, fallback=0, **{r: 0.0 for r in RULES}, **{f"{r}_any": 0 for r in RULES}))

    def shares(self, scores, plain, kept, fallback, column, x):
        """[B] share of the trained model's probability each rule removes at this draw, and the fallback rows."""
        p = torch.softmax(scores.detach().cpu().double(), -1)
        plain, kept, fallback = plain.cpu(), kept.cpu(), fallback.cpu()
        zero = torch.zeros(len(p), dtype=torch.float64)
        out = {r: zero.clone() for r in RULES}
        if column >= M.STAT0:                               # Stat Points: the budget rule, as in training, but on the
            out["tables"] = (p * ~plain).sum(1)             # teams now, so Stat Points drawn earlier in the pass count
        else:
            k = L.KEY_OF_COL[column]
            tab = self.w.tables.allowed(x[:, :M.STAT0])[k][:, L.COLS_BY_KEY[k].index(column)].cpu()
            out["mask_token"] = p[:, 0].clone()
            tab = tab.clone(); tab[:, 0] = False
            E = self.w.tables.EMV
            if column % M.NF == M.FIRST_MOVE and E is not None:
                real = tab.clone(); real[:, [0, E]] = False
                hit = real.any(1) & tab[:, E]
                out["no_moves"] = torch.where(hit, p[:, E], zero)
            notmask = torch.ones_like(tab); notmask[:, 0] = False
            out["tables"] = (p * (~tab & notmask)).sum(1)
        out["lookahead"] = (p * (plain & ~kept)).sum(1)
        for r in ("no_moves", "tables", "lookahead"):    # a fallback draw allows every value but [MASK]
            out[r] = torch.where(fallback, zero, out[r])
        direct = 1.0 - (p * kept).sum(1)                   # outside a fallback the draw uses exactly the kept values
        gap = (sum(out[r] for r in RULES) - direct).abs()
        if (gap > 1e-6)[~fallback].any():
            raise AssertionError(f"column {column}: the four shares do not add up to the removed share (gap {float(gap.max()):.2e})")
        return out, fallback

    def add(self, scores, plain, kept, fallback, rows=None, column=None, x=None):
        if column is None or x is None: raise ValueError("SkewLog needs the column drawn and the teams before the draw")
        out, fallback = self.shares(scores, plain, kept, fallback, column, x)
        rows = torch.ones_like(fallback) if rows is None else rows.cpu()
        removed = sum(out[r] for r in RULES).clamp(max=1.0 - 1e-12)
        self.team_log_ratio += torch.where(rows, -torch.log1p(-removed), torch.zeros_like(removed))
        self.team_fallbacks += (fallback & rows).long()
        s = self.sums[field_type(column)]
        s["draws"] += int(rows.sum()); s["fallback"] += int((fallback & rows).sum())
        for r in RULES:
            v = torch.where(rows, out[r], torch.zeros_like(out[r]))
            s[r] += float(v.sum()); s[f"{r}_any"] += int((v > 1e-9).sum())

    def summary(self):
        return dict(by_field_type={k: dict(v) for k, v in self.sums.items()},
                    team_log_ratio=self.team_log_ratio.tolist(), team_fallbacks=self.team_fallbacks.tolist())


def seed_of(tag): return int(hashlib.sha256(tag.encode()).hexdigest()[:8], 16)


def draw_cell(w, order, ask, guidance, n, seed):
    log = SkewLog(w, n)
    if order == "species_first":
        w.sample(n, ask=ask, guidance=guidance, seed=seed, refusals=log)
    else:
        import remask84
        torch.manual_seed(seed)
        remask84.sample_random(w, n, ask, "rand_plain", {}, guidance=guidance, lookahead=True, refusals=log)
    return log.summary()


def merge(parts):
    """One cell's summaries over its batches."""
    out = dict(by_field_type={}, team_log_ratio=[], team_fallbacks=[])
    for part in parts:
        out["team_log_ratio"] += part["team_log_ratio"]; out["team_fallbacks"] += part["team_fallbacks"]
        for k, v in part["by_field_type"].items():
            acc = out["by_field_type"].setdefault(k, {key: 0 for key in v})
            for key, val in v.items(): acc[key] += val
    return out


def load(): return json.load(open(OUT)) if OUT.exists() else {"cells": {}}


def save(res):
    tmp = OUT.with_suffix(".tmp"); json.dump(res, open(tmp, "w")); os.replace(tmp, OUT)


def run():
    res = load()
    for ck, name in CHECKPOINTS.items():
        w = M.Model84.load(L.CK / f"{name}.pt")
        for order in ORDERS:
            for g in GUIDANCES:
                for a in ASK_NUMERATORS:
                    tag = f"{ck}_{order}_g{g:g}_a{a:02d}of24"
                    if tag in res["cells"]: continue
                    parts = [draw_cell(w, order, a / 24, g, BATCH, seed_of(f"{tag}_b{b}")) for b in range(BATCHES)]
                    res["cells"][tag] = dict(checkpoint=ck, order=order, guidance=g, ask=a / 24, teams=BATCH * BATCHES, **merge(parts))
                    save(res); print(tag, "done", flush=True)


def quantile(xs, q):
    xs = sorted(xs); return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else float("nan")


def report():
    res = load()
    rows = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    teams = defaultdict(lambda: dict(log_ratio=[], fallback=0, n=0))
    for cell in res["cells"].values():
        arm = (cell["checkpoint"], cell["order"], cell["guidance"])
        for k, v in cell["by_field_type"].items():
            for key, val in v.items(): rows[arm][k][key] += val
        teams[arm]["log_ratio"] += cell["team_log_ratio"]; teams[arm]["n"] += len(cell["team_fallbacks"])
        teams[arm]["fallback"] += sum(1 for f in cell["team_fallbacks"] if f)
    for arm in sorted(rows):
        ck, order, g = arm
        t = teams[arm]
        print(f"\n{READER[ck]}, {READER[order]}, guidance {g:g}: {t['n']} teams; teams with a fallback draw {t['fallback']}; "
              f"team log ratio median {quantile(t['log_ratio'], 0.5):.4f}, 90th percentile {quantile(t['log_ratio'], 0.9):.4f}")
        for k, v in sorted(rows[arm].items()):
            d = max(v["draws"], 1)
            print(f"  {k:8s} draws {int(v['draws']):6d}  fallback {int(v['fallback']):4d}  " +
                  "  ".join(f"{r} {v[r] / d:.2e} ({int(v[r + '_any']) / d:.1%})" for r in RULES))


FIELD_ORDER = (("species", "Species"), ("ability", "Ability"), ("item", "Item"), ("move", "Moves"), ("stat", "Stat Points"))
RULE_READER = {"lookahead": "Lookahead", "tables": "Same-pass legality rules", "no_moves": "Empty first move", "mask_token": "[MASK]"}
RULE_COLOR = {"lookahead": "#e0702a", "tables": "#2a78d6", "no_moves": "#1baf7a", "mask_token": "#6f6e69"}
SHORT = {"per_field": "Per-field", "linear": "Linear", "species_first": "species-first", "random": "random order"}
ARM_COLOR = {("linear", "random"): "#e0702a", ("per_field", "random"): "#b0452a",
             ("linear", "species_first"): "#2a78d6", ("per_field", "species_first"): "#1b4f9a"}


def figure(folder=HERE.parent / "docs/figures"):
    """Top: mean share of the trained model's probability each rule removes per draw, by field type, in random order
    (both checkpoints, all guidance values pooled; species-first order removes almost none, said in the subtitle). Bottom: the median team's log ratio
    of sampler to trained model by guidance, one line per checkpoint and order."""
    from echarts_chart import write_chart
    res = load()
    pooled = {o: defaultdict(lambda: defaultdict(float)) for o in ORDERS}
    med = defaultdict(lambda: defaultdict(list))
    for cell in res["cells"].values():
        for k, v in cell["by_field_type"].items():
            for key, val in v.items(): pooled[cell["order"]][k][key] += val
        med[(cell["checkpoint"], cell["order"])][cell["guidance"]] += cell["team_log_ratio"]
    cats = [name.replace(" ", "\n") for _, name in FIELD_ORDER]
    series = []
    for r in RULES:
        data = [pooled["random"][k][r] / max(pooled["random"][k]["draws"], 1) for k, _ in FIELD_ORDER]
        series.append(dict(type="bar", name=RULE_READER[r], stack="removed", color=RULE_COLOR[r], xAxisIndex=0, yAxisIndex=0,
                           emphasis=dict(focus="series"), data=data))
    for (ck, o), byg in sorted(med.items()):
        series.append(dict(type="line", name=f"{SHORT[ck]}, {SHORT[o]}", color=ARM_COLOR[(ck, o)], xAxisIndex=1, yAxisIndex=1,
                           symbolSize=8, emphasis=dict(focus="series"), lineStyle=dict(type="solid" if o == "random" else "dashed"),
                           data=[[g, quantile(byg[g], 0.5)] for g in sorted(byg)]))
    pct = "js:v => (v * 100).toFixed(0) + '%'"
    option = dict(grid=[dict(left=64, right=24, top=96, height=230), dict(left=64, right=24, top=470, height=190)],
                  legend=[dict(top=0, data=[RULE_READER[r] for r in RULES]),
                          dict(top=405, data=[s["name"] for s in series if s["type"] == "line"], textStyle=dict(fontSize=11))],
                  tooltip=dict(trigger="axis"),
                  xAxis=[dict(type="category", gridIndex=0, data=cats, axisLabel=dict(interval=0, fontSize=11)),
                         dict(type="value", gridIndex=1, min=1, max=9, interval=1, name="Guidance weight", nameLocation="middle", nameGap=28,
                              axisLabel=dict(formatter="js:v => [2, 4, 8].includes(v) ? v : ''"))],
                  yAxis=[dict(type="value", gridIndex=0, name="Random order: share removed per draw", axisLabel=dict(formatter=pct)),
                         dict(type="value", gridIndex=1, name="Median team, log ratio (nats)")],
                  series=series, animation=False)
    la = {k: pooled["random"][k]["lookahead"] / max(pooled["random"][k]["draws"], 1) for k, _ in FIELD_ORDER}
    sf_max = max(quantile(v, 0.9) for (ck, o), byg in med.items() if o == "species_first" for v in byg.values())
    return write_chart(option, Path(folder) / "legality-skew.html", height=700,
                       title=f"Random order's lookahead removes up to {max(la.values()):.0%} of a draw's probability; species-first order removes almost none",
                       subtitle=f"Share of the trained model's guided probability each sampling-only rule removes. 6,912 teams, two checkpoints, "
                                f"guidance 2, 4 and 8, asks 4, 12 and 20 of 24. In species-first order the 90th-percentile team is at most "
                                f"{math.exp(sf_max):.4f} times as likely under the sampler as under the trained model.",
                       note="Log ratio: the sum over a team's 84 draws of -log(1 - removed share), the log of how much more likely the "
                            "sampler makes that team than the trained model does along the same fill path. Training already masks what the "
                            "legal Pokémon construction tables forbid, so the tables count here only for values a draw earlier in the same pass "
                            "made illegal.")


if __name__ == "__main__":
    {"run": run, "report": report, "figure": figure}[sys.argv[1]]()
