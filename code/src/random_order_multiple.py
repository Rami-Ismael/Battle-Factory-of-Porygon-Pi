"""Did a weaker model cause the 2.8 multiple? (GitHub issue #323)

Random order's ask-versus-got error was 2.8 times species-first's on the all-teams per-field model after 21 epochs
(2026-11-07), and 4.8 to 7.2 times on the linear model (#297). The suspect: the masking schedule changed nothing about
random order, and the multiple rose only because species-first got better on the stronger models.

The test is random order with the lookahead on the control's checkpoint (`stat84_shuffled_noise_per_field_es3_cap80`)
at guidance 2, over the control's own species-first grid. #335 already battled exactly that grid
(`sched_per_field_random`: same checkpoint, decoding as Linear, random order, 69 cells of 64 teams, slots shuffled),
so this script reads stored battles and runs none.

    python random_order_multiple.py read             the verdict, into results/random_order_multiple.json
    python random_order_multiple.py figure DIR       the multiple for the three models, as an ECharts page in DIR
    python random_order_multiple.py comment URL      the verdict comment; URL holds {theme} for light or dark

A multiple is random order's error over species-first's on the same checkpoint. Its interval is #297's: teams
resampled within each ask and sampling seed, separately in each row, the 49 opponents drawn once and shared by both
rows; the multiple of each draw gives the 95% interval. On the 21-epoch model the stored battles are read the way
remask84_random_error.py read them (equal-weight opponents, bootstrap_levels), which reproduces its 0.0534 and 0.0192.

The rule, from the ticket, at guidance 2:
  confirmed      the interval contains the linear model's multiple and sits above the 21-epoch model's
  cleared        the interval sits above the top of the linear model's interval (the linear schedule helped random
                 order, less than expected) or below the linear model's multiple (it made random order relatively worse)
  not separated  anything else
"""
import datetime
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import masking_rate_rows as MRR

PER_FIELD_RANDOM, LINEAR_RANDOM, LINEAR_SPECIES_FIRST, CONTROL = "sched_per_field_random", "sched_linear_random", "sched_linear_species_first", MRR.CONTROL
STREAM = {**{row: i for i, row in enumerate(MRR.ROWS)}, PER_FIELD_RANDOM: len(MRR.ROWS)}   # #297's streams; the new row takes the next
OLD_RANDOM, OLD_SPECIES_FIRST = "rand_la_plain", "species_first_plain"                       # remask84_random_error.py's arm names
MODELS = {"old": "All-teams per-field model, 21 epochs, older table", "control": "The control's checkpoint: per-field, shuffled-only table, 80 epochs",
          "linear": "Linear checkpoint: shuffled-only table, 80 epochs"}
SHORT = {"old": "21-epoch all-teams model", "control": "Control's checkpoint", "linear": "Linear checkpoint"}
CHECKPOINT = {"old": "gen84_all_ep21", "control": MRR.PER_FIELD_CHECKPOINT, "linear": "stat84_shuffled_noise_linear_es3_cap80"}
OUT = MRR.RES / f"random_order_multiple{'_smoke' if MRR.SMOKE else ''}.json"
ISSUE_URL = "https://github.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/issues/323"


def call(interval, old, linear, linear_top):
    """The ticket's rule. interval: the control's-checkpoint multiple's 95% interval; old: the 21-epoch model's multiple;
    linear and linear_top: the linear model's multiple and the top of its interval, all at guidance 2."""
    lo, hi = interval
    if lo > linear_top: return "cleared: the linear masking schedule helped random order"
    if hi < linear: return "cleared: the linear masking schedule made random order relatively worse"
    if lo <= linear <= hi and lo > old: return "confirmed"
    return "not separated"


def row_draws(row, guidance, counts):
    """MRR.bootstrap with a stream for every row, including the one #297 did not have."""
    data = np.load(MRR.opponents_path(row, guidance)); weights = data["weights"]
    rng = np.random.default_rng([MRR.DRAW_SEED, STREAM[row], int(guidance)]); draws = {}
    for k in MRR.ALL_ASKS:
        for seed in MRR.SEEDS:
            rate = data[f"a{k:02d}_s{seed}"]; seen = ~np.isnan(rate)
            scores = (np.where(seen, rate, 0.0) @ (counts * weights).T) / np.maximum(seen @ (counts * weights).T, 1e-12)
            pick = rng.integers(0, len(rate), (MRR.DRAWS, len(rate)))
            draws[(k, seed)] = scores[pick, np.arange(MRR.DRAWS)[:, None]].mean(1)
    return draws


def grid_errors(row, guidance, counts):
    """(error, {seed: error}, [DRAWS] error draws) of one stored shuffled-only grid."""
    out = MRR.load_grid(row, guidance)
    if out is None: raise SystemExit(f"{MRR.grid_name(row, guidance)}: the grid is missing or has cells without battles")
    cells = out["by_ask_seed"]; sizes = {key: len(c["pastes"]) for key, c in cells.items()}
    mean = {key: float(np.mean(c["y"])) for key, c in cells.items()}
    got = {k: sum(mean[(k, s)] * sizes[(k, s)] for s in MRR.SEEDS) / sum(sizes[(k, s)] for s in MRR.SEEDS) for k in MRR.ALL_ASKS}
    by_seed = {s: MRR.error({k: mean[(k, s)] for k in MRR.ALL_ASKS}) for s in MRR.SEEDS}
    return MRR.error(got), by_seed, MRR.error_draws(MRR.pooled(row_draws(row, guidance, counts), sizes))


def multiple(random_error, species_error, random_by_seed, species_by_seed, random_draws, species_draws):
    ratio = random_draws / species_draws
    return dict(multiple=random_error / species_error, interval=MRR.interval(ratio), random_error=random_error, species_first_error=species_error,
                multiple_by_seed={str(s): random_by_seed[s] / species_by_seed[s] for s in species_by_seed})


def shuffled_only_multiple(random_row, species_row, guidance, counts):
    (re, rs, rd), (se, ss, sd) = grid_errors(random_row, guidance, counts), grid_errors(species_row, guidance, counts)
    return multiple(re, se, rs, ss, rd, sd)


def old_model_multiple():
    """The 21-epoch model's multiple from the stored battles, read as remask84_random_error.py read them."""
    MRR.use_pilot_src()
    import remask84_random_error as RE
    from remask84_error_analysis import battle_matrices, bootstrap_levels
    cells = {key: c for key, c in RE.load_cells().items() if key[0] in (OLD_RANDOM, OLD_SPECIES_FIRST)}
    M, n_opps, contract = battle_matrices(cells, policy_id=RE.battle_policy())
    boot = bootstrap_levels(M, n_opps, arms=[OLD_RANDOM, OLD_SPECIES_FIRST], asks=RE.ASKS, seeds=RE.SEEDS)
    errors = {}
    for arm in (OLD_RANDOM, OLD_SPECIES_FIRST):
        by_seed = {s: RE.ask_error({k: float(np.mean(cells[(arm, k, s)]["y"])) for k in RE.ASKS}, RE.SCORED) for s in RE.SEEDS}
        point = {k: float(np.mean([np.mean(cells[(arm, k, s)]["y"]) for s in RE.SEEDS])) for k in RE.ASKS}
        errors[arm] = (RE.ask_error(point, RE.SCORED), by_seed, np.mean([(boot[arm][k] - k / 24) ** 2 for k in RE.SCORED], 0))
    m = multiple(*[errors[a][i] for i in range(3) for a in (OLD_RANDOM, OLD_SPECIES_FIRST)])
    gaps = np.concatenate([np.abs(np.nanmean(M[key], 1) - np.array(c["y"])) for key, c in cells.items()])
    from_matrices = {arm: RE.ask_error({k: float(np.mean([np.nanmean(M[(arm, k, s)]) for s in RE.SEEDS])) for k in RE.ASKS}, RE.SCORED)
                     for arm in (OLD_RANDOM, OLD_SPECIES_FIRST)}
    m.update(contract=contract, teams=len(gaps), teams_drifted=int((gaps > 1.05 / n_opps).sum()),     # more than one battle's worth off the saved score
             random_error_from_matrices=from_matrices[OLD_RANDOM], species_first_error_from_matrices=from_matrices[OLD_SPECIES_FIRST])
    return m


def stored_remasking_multiples():
    """Every random-decoding arm of 2026-11-07 over species-first: where the map's 2.5 and the documents' 2.8 come from."""
    arms = json.load(open(MRR.RES / "remask84_random_error.json"))["arms"]
    sf = arms[OLD_SPECIES_FIRST]["scored"]["error"]
    return {arm: dict(name=a["name"], error=a["scored"]["error"], multiple=a["scored"]["error"] / sf) for arm, a in arms.items() if arm != OLD_SPECIES_FIRST}


def read():
    counts = MRR.opponent_draws(len(np.load(MRR.opponents_path(CONTROL, 2))["weights"]))
    guidance = {}
    for g in MRR.GUIDANCES:
        guidance[str(g)] = dict(control=shuffled_only_multiple(PER_FIELD_RANDOM, CONTROL, g, counts),
                                linear=shuffled_only_multiple(LINEAR_RANDOM, LINEAR_SPECIES_FIRST, g, counts))
    old = old_model_multiple(); g2 = guidance["2"]
    verdict = call(g2["control"]["interval"], old["multiple"], g2["linear"]["multiple"], g2["linear"]["interval"][1])
    result = dict(question="Did a weaker model cause the 2.8 multiple?", verdict=verdict,
                  rule="confirmed: the control's-checkpoint interval contains the linear multiple and sits above the 21-epoch multiple; "
                       "cleared: it sits above the top of the linear interval, or below the linear multiple; otherwise not separated (guidance 2)",
                  resampled="teams within each ask and sampling seed, separately in each row; the 49 opponents once, shared by both rows of a multiple",
                  draws=MRR.DRAWS, draw_seed=MRR.DRAW_SEED, checkpoints=CHECKPOINT, old_model=old, guidance=guidance,
                  remasking_arms_2026_11_07=stored_remasking_multiples())
    json.dump(result, open(OUT, "w"), indent=1)
    print(f"21-epoch model: {old['multiple']:.2f} {[round(x, 2) for x in old['interval']]}")
    for g, e in guidance.items():
        print(f"guidance {g}: control's checkpoint {e['control']['multiple']:.2f} {[round(x, 2) for x in e['control']['interval']]}, "
              f"linear {e['linear']['multiple']:.2f} {[round(x, 2) for x in e['linear']['interval']]}")
    print("VERDICT", verdict); print("RANDOM_ORDER_MULTIPLE_DONE")


COLOR = {"old": "#8a8f98", "control": "#2a78d6", "linear": "#e0702a"}


def points(result):
    """[(key, multiple, interval)] at guidance 2 in the order the chart draws them."""
    g2 = result["guidance"]["2"]
    return [("old", result["old_model"]["multiple"], result["old_model"]["interval"]),
            ("control", g2["control"]["multiple"], g2["control"]["interval"]), ("linear", g2["linear"]["multiple"], g2["linear"]["interval"])]


def headline(result):
    c = result["guidance"]["2"]["control"]
    return {"confirmed": f"The 2.8 came from the weaker model: on the control's checkpoint random order costs {c['multiple']:.1f} times species-first",
            "not separated": f"Not separated: on the control's checkpoint random order costs {c['multiple']:.1f} times species-first"}.get(
        result["verdict"], f"Suspect cleared: on the control's checkpoint random order costs {c['multiple']:.1f} times species-first")


def figure(folder):
    """The three models' multiples at guidance 2, each with its 95% interval."""
    from echarts_chart import write_chart
    result = json.load(open(OUT)); pts = points(result); cats = [SHORT[k] for k, _, _ in pts]
    whisker = ("js:(params, api) => { const a = api.coord([api.value(0), api.value(1)]), b = api.coord([api.value(0), api.value(2)]); "
               "return {type: 'group', children: [{type: 'line', shape: {x1: a[0], y1: a[1], x2: b[0], y2: b[1]}, style: {stroke: api.visual('color'), lineWidth: 2}}, "
               "{type: 'line', shape: {x1: a[0] - 8, y1: a[1], x2: a[0] + 8, y2: a[1]}, style: {stroke: api.visual('color'), lineWidth: 2}}, "
               "{type: 'line', shape: {x1: b[0] - 8, y1: b[1], x2: b[0] + 8, y2: b[1]}, style: {stroke: api.visual('color'), lineWidth: 2}}]}; }")
    series = []
    for i, (key, m, iv) in enumerate(pts):
        series.append(dict(type="scatter", name=SHORT[key], color=COLOR[key], symbolSize=14, emphasis=dict(focus="series"),
                           label=dict(show=True, position="right", distance=12, offset=[0, 16 if key == "control" else -16],
                                      formatter=f"{m:.1f}×  [{iv[0]:.1f}, {iv[1]:.1f}]"), data=[[i, m]]))
        series.append(dict(type="custom", name=SHORT[key], color=COLOR[key], silent=True, tooltip=dict(show=False), renderItem=whisker, data=[[i, *iv]]))
    old, linear = pts[0][1], pts[2][1]
    series[0]["markLine"] = dict(symbol="none", silent=True, label=dict(show=False), lineStyle=dict(type="dashed", width=1.2),
                                 data=[dict(yAxis=old, name="21-epoch multiple"), dict(yAxis=linear, name="linear multiple")])
    option = dict(grid=dict(left=72, right=40, top=40, bottom=64),
                  tooltip=dict(trigger="item", formatter="js:p => p.seriesName + ': ' + p.value[1].toFixed(2) + '×'"),
                  xAxis=dict(type="category", data=cats, name="Checkpoint, each read against its own species-first grid", nameLocation="middle", nameGap=40),
                  yAxis=dict(type="value", min=0, max=8, name="Random order's error ÷ species-first's", nameLocation="middle", nameGap=44,
                             splitLine=dict(lineStyle=dict(opacity=0.35))),
                  series=series, animation=False)
    c = result["guidance"]["2"]["control"]
    return write_chart(option, Path(folder) / "random-order-multiple.html", height=440, title=headline(result),
                       subtitle=f"Guidance 2. Random order with the lookahead on: {c['random_error']:.4f} on the control's checkpoint against the control's "
                                f"{c['species_first_error']:.4f}. Dashed lines: the 21-epoch model's {old:.1f}× and the linear checkpoint's {linear:.1f}×.",
                       note="Multiple: random order's ask-versus-got error over species-first's on the same checkpoint. Error: mean squared gap between the ask k/24 "
                            "and the mean measured win rate of the teams drawn at it, over asks 1/24 to 17/24. Whisker: 95% bootstrap interval, teams resampled "
                            "within each ask and sampling seed, the 49 opponents shared by both rows of a multiple. Win rate: one battle against each of the "
                            "49 distinct top-50 meta teams, behaviour-cloning policy on both sides, slots shuffled.")


def comment(figure_url):
    """The verdict comment, picture first. figure_url has {theme} where light or dark goes."""
    r = json.load(open(OUT)); old, c, lin = r["old_model"], r["guidance"]["2"]["control"], r["guidance"]["2"]["linear"]
    iv = lambda x: f"[{x[0]:.2f}, {x[1]:.2f}]"
    arms = r["remasking_arms_2026_11_07"]
    lines = [f"**Verdict, {datetime.date.today()}: {r['verdict']}.** On the control's checkpoint random order's error is {c['multiple']:.2f} times "
             f"species-first's {iv(c['interval'])} at guidance 2. The interval holds the linear checkpoint's {lin['multiple']:.2f} and sits above the "
             f"21-epoch model's {old['multiple']:.2f}. So the masking schedule did not change how much random order costs; the 2.8 came from the "
             "weaker model's large species-first error." if r["verdict"] == "confirmed" else
             f"**Verdict, {datetime.date.today()}: {r['verdict']}.** On the control's checkpoint random order's error is {c['multiple']:.2f} times "
             f"species-first's {iv(c['interval'])} at guidance 2, against {lin['multiple']:.2f} on the linear checkpoint and {old['multiple']:.2f} on the 21-epoch model.",
             "",
             f'<picture><source media="(prefers-color-scheme: dark)" srcset="{figure_url.format(theme="dark")}"><img alt="Random order\'s error multiple for three checkpoints at guidance 2" src="{figure_url.format(theme="light")}"></picture>',
             "",
             "Each point is random order's ask-versus-got error over species-first's on the same checkpoint, with its 95% interval. "
             "No new battles: [Battle Per-field, random order](https://github.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/issues/335) already ran this ticket's grid "
             "(`sched_per_field_random`: the control's checkpoint, random order with the lookahead, guidance 2, 69 cells of 64 teams, 216,384 battles, none missing, slots shuffled, preflight passed).",
             "",
             "| Checkpoint | Random order | Species-first | Multiple | 95% interval | Multiple by sampling seed (1, 2, 3) |", "|---|---|---|---|---|---|"]
    for key, src in (("old", old), ("control", c), ("linear", lin)):
        seeds = ", ".join(f"{v:.2f}" for v in src["multiple_by_seed"].values())
        lines.append(f"| {MODELS[key]} | {src['random_error']:.4f} | {src['species_first_error']:.4f} | {src['multiple']:.2f}× | {iv(src['interval'])} | {seeds} |")
    lines += ["", f"Rule, from the ticket: {r['rule']}. Bars used: the 21-epoch multiple {old['multiple']:.2f}, the linear multiple {lin['multiple']:.2f}, "
                  f"the top of its interval {lin['interval'][1]:.2f}. The suspect read: random order's own error moved little "
                  f"({old['random_error']:.4f} to {c['random_error']:.4f} to {lin['random_error']:.4f}) while species-first's fell "
                  f"({old['species_first_error']:.4f} to {c['species_first_error']:.4f} to {lin['species_first_error']:.4f}).", "",
              "The same multiple at higher guidance, for the two shuffled-only checkpoints:", "",
              "| Guidance | Control's checkpoint | Linear checkpoint |", "|---|---|---|"]
    for g, e in r["guidance"].items():
        lines.append(f"| {g} | {e['control']['multiple']:.2f}× {iv(e['control']['interval'])} | {e['linear']['multiple']:.2f}× {iv(e['linear']['interval'])} |")
    rm = "; ".join(f"{a['name']} {a['multiple']:.2f}×" for a in arms.values())
    lines += ["", f"**2.5 or 2.8:** 2.8 is right for the arm that matches random order here. The stored 2026-11-07 grids give {rm}, all over species-first's "
                  f"{old['species_first_error']:.4f} (`remask84_random_error.json`). No remasking, 84 passes is the decoding these rows use; the map's 2.5 matches "
                  "the remasking arms instead. The map now reads 2.8.", "",
              "<details><summary>How it was computed</summary>", "",
              "Interval: #297's bootstrap, 4,000 draws, teams resampled within each ask and sampling seed separately in each row, the 49 opponents drawn once and "
              "shared by both rows of a multiple; the 2.5% and 97.5% quantiles of the per-draw multiple. The linear row reproduces #297's stored "
              f"{lin['multiple']:.2f} {iv(lin['interval'])} exactly. The 21-epoch model is read the way `remask84_random_error.py` read it, equal-weight opponents, "
              f"which reproduces its {old['random_error']:.4f} and {old['species_first_error']:.4f}. The battle database has gained battles for {old['teams_drifted']} of those "
              f"{old['teams']:,} teams since 2026-11-07, so its per-opponent shares, which the bootstrap resamples, give {old['random_error_from_matrices']:.4f} and "
              f"{old['species_first_error_from_matrices']:.4f}: a multiple of {old['random_error_from_matrices'] / old['species_first_error_from_matrices']:.2f}. Script: `code/src/random_order_multiple.py read`. Results: `~/vgc-data/results/random_order_multiple.json`.",
              "</details>"]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    {"read": read, "figure": lambda: print(figure(sys.argv[2])), "comment": lambda: print(comment(sys.argv[2]), end="")}[sys.argv[1]]()
