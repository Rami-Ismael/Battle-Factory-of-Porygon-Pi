"""Does random order work better with the linear masking schedule than with the per-field one? (GitHub issue #335)

Per-field, random order decodes the shuffled-only per-field checkpoint (the control's model) exactly as Linear, random
order decodes the linear one: LLaDA's random order with the lookahead on and t = 1 - filled/84. The masking schedule is
the only difference between the two rows. This script reads their grids at guidance 2, 4 and 8 beside Linear,
species-first and the control, which are there for scale.

    python random_order_schedule.py read             the verdict, into results/random_order_schedule.json
    python random_order_schedule.py figure DIR       error by guidance for the four rows, as an ECharts page in DIR
    python random_order_schedule.py comment URL      the verdict comment; URL holds {theme} for light or dark

The rule was written in the ticket before the run. At one guidance, the gap is Linear, random order's ask-versus-got
error minus Per-field, random order's, averaged over the three sampling seeds. The linear masking schedule wins that
guidance when the gap is at most -0.001 and its 95% interval lies below zero; the per-field schedule wins under the
mirror rule. A schedule that wins two or more of the three guidance values wins the question; otherwise the masking
schedule makes no measured difference to random order. A gap under 0.002 is flagged as within training luck, since
each model has one training seed.

The interval is the #297 bootstrap: teams resampled within each ask and sampling seed, separately in each row, and the
49 opponents once for every row (masking_rate_rows.opponent_draws), so the two rows face the same opponents in a draw.
"""
import datetime
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import masking_rate_rows as MRR
import species_late as SL

PER_FIELD_RANDOM, LINEAR_RANDOM = "sched_per_field_random", "sched_linear_random"
LINEAR_SPECIES_FIRST, CONTROL = "sched_linear_species_first", MRR.CONTROL
ROWS = (PER_FIELD_RANDOM, LINEAR_RANDOM, LINEAR_SPECIES_FIRST, CONTROL)
READER = {PER_FIELD_RANDOM: "Per-field, random order", LINEAR_RANDOM: "Linear, random order",
          LINEAR_SPECIES_FIRST: "Linear, species-first", CONTROL: "Per-field, species-first"}
STREAM = {row: i for i, row in enumerate(ROWS)}            # each row's own bootstrap stream, fixed so a new row moves no other
BAR, TRAINING_LUCK = MRR.BAR, MRR.TRAINING_LUCK            # 0.001 and 0.002, the map's bar
OUT = MRR.RES / f"random_order_schedule{'_smoke' if MRR.SMOKE else ''}.json"
PASS_AT_K_STORED = MRR.RES / "pass_at_k_issue331.json"     # Linear, random order's Pass@k, read for #331
PASS_BAND, PASS_DEFINITION = "asks_13_to_17", "held"
ISSUE_URL = "https://github.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/issues/335"


def guidance_call(gap, interval):
    """The rule at one guidance. gap: Linear, random order's error minus Per-field, random order's; interval: its 95%
    interval. Returns "linear", "per_field" or "neither"."""
    if gap <= -BAR and interval[1] < 0: return "linear"
    if gap >= BAR and interval[0] > 0: return "per_field"
    return "neither"


def verdict(calls):
    """calls: {guidance: "linear" | "per_field" | "neither"}. The schedule that wins two or more guidance values wins."""
    for side in ("linear", "per_field"):
        if sum(c == side for c in calls.values()) >= 2: return side
    return "no measured difference"


def read():
    counts = MRR.opponent_draws(len(np.load(MRR.opponents_path(LINEAR_RANDOM, 2))["weights"]))
    summary = json.load(open(MRR.SUMMARY))["grids"] if MRR.SUMMARY.exists() else {}
    refs = None
    result = dict(question="Does random order work better with the linear masking schedule than with the per-field one?",
                  rule=f"at one guidance, the linear schedule wins when Linear, random order's error is at least {BAR} lower than Per-field, random order's "
                       f"with the paired interval below zero; the per-field schedule wins under the mirror rule; two or more of three guidance values decide; "
                       f"a gap under {TRAINING_LUCK} is within training luck",
                  resampled="teams within each ask and sampling seed, separately in each row; the 49 opponents once for every row",
                  draws=MRR.DRAWS, draw_seed=MRR.DRAW_SEED, guidance={})
    for g in MRR.GUIDANCES:
        rows = {row: SL.read_row(row, counts, g, stream=[MRR.DRAW_SEED, 335, STREAM[row], g]) for row in ROWS}
        gap = SL.paired(rows[LINEAR_RANDOM], rows[PER_FIELD_RANDOM])
        call = guidance_call(gap["difference"], gap["interval"])
        entry = dict(gap_linear_minus_per_field=gap["difference"], gap_interval=gap["interval"], call=call,
                     within_training_luck=abs(gap["difference"]) < TRAINING_LUCK, rows={})
        for row, m in rows.items():
            name = MRR.grid_name(row, g)
            if name not in summary:                         # a grid the masking-schedule summary has not read: its measures here
                refs = refs or MRR.references()
                summary[name] = MRR.measures(MRR.load_grid(row, g), refs)
            s = summary[name]
            entry["rows"][row] = dict(reader=READER[row], error=m["error"], error_interval=MRR.interval(m["error_draws"]),
                                      error_by_seed={str(k): v for k, v in m["error_by_seed"].items()},
                                      error_multiple_of_species_first=m["error"] / rows[LINEAR_SPECIES_FIRST if row == LINEAR_RANDOM else CONTROL]["error"]
                                      if row in (LINEAR_RANDOM, PER_FIELD_RANDOM) else None,
                                      got_by_ask={str(k): v for k, v in m["got_by_ask"].items()},
                                      pooled_win=s["pooled_win"], win_top_asks=s["win_top_asks"], first_try_legality=s["first_try_legality"],
                                      line_ups=s["line_ups"], new_against_training=s["new_against_training"],
                                      distinct_novel_line_ups_against_training=s["distinct_novel_line_ups_against_training"],
                                      effective_species=s["effective_species"])
        entry["pass_at_k"] = pass_at_k(g)
        result["guidance"][str(g)] = entry
    result["calls"] = {g: e["call"] for g, e in result["guidance"].items()}
    result["verdict"] = verdict(result["calls"])
    json.dump(result, open(OUT, "w"), indent=1)
    for g, e in result["guidance"].items():
        r = e["rows"]
        print(f"guidance {g}: per-field random {r[PER_FIELD_RANDOM]['error']:.4f}, linear random {r[LINEAR_RANDOM]['error']:.4f}, "
              f"gap {e['gap_linear_minus_per_field']:+.4f} {e['gap_interval']} -> {e['call']}{' (within training luck)' if e['within_training_luck'] else ''}")
    print("VERDICT", result["verdict"]); print("RANDOM_ORDER_SCHEDULE_DONE")


def pass_at_k(g):
    """Pass@1 and Pass@48 at asks 13/24 to 17/24 for the two random-order rows: the new grid's file from
    pass_at_k.py read, and Linear, random order's from the #331 re-read. None where a file is missing."""
    out = {}
    new = MRR.RES / "pass_at_k" / f"{MRR.grid_name(PER_FIELD_RANDOM, g)}.json"
    sources = {PER_FIELD_RANDOM: json.load(open(new)) if new.exists() else None,
               LINEAR_RANDOM: json.load(open(PASS_AT_K_STORED))["grids"].get(MRR.grid_name(LINEAR_RANDOM, g)) if PASS_AT_K_STORED.exists() else None}
    for row, r in sources.items():
        band = r and r["bands"].get(PASS_DEFINITION, {}).get(PASS_BAND)
        out[row] = dict(pass_at_1=band["pass_at_k"][0], pass_at_48=band["pass_at_k"][-1], pass_at_48_interval=band["pass_at_k_interval"][-1],
                        potential_at_48=band["potential_at_k"][-1]) if band else None
    return out


COLOR = {PER_FIELD_RANDOM: "#8a5cd6", LINEAR_RANDOM: "#e0702a", LINEAR_SPECIES_FIRST: "#1baf7a", CONTROL: "#2a78d6"}   # pass_at_k.py colors


def headline(result):
    return {"linear": "Random order works better with the linear masking schedule",
            "per_field": "Random order works better with the per-field masking schedule",
            "no measured difference": "The masking schedule makes no measured difference to random order"}[result["verdict"]]


def figure(folder):
    """Ask-versus-got error against guidance for the four rows on a log axis, each point with its 95% interval."""
    from echarts_chart import write_chart
    result = json.load(open(OUT)); gs = list(result["guidance"]); series = []
    for row in ROWS:
        pts = [result["guidance"][g]["rows"][row] for g in gs]; random_row = row in (PER_FIELD_RANDOM, LINEAR_RANDOM)
        series.append(dict(type="line", name=READER[row], color=COLOR[row], symbolSize=9 if random_row else 6, emphasis=dict(focus="series"),
                           lineStyle=dict(width=2.6 if random_row else 1.4, type="solid" if random_row else "dashed"),
                           label=dict(show=random_row, position="right", formatter="js:p => p.value[1].toFixed(4)"),
                           data=[[i, p["error"]] for i, p in enumerate(pts)]))
        series.append(dict(type="custom", name=READER[row], color=COLOR[row], silent=True, tooltip=dict(show=False),
                           renderItem="js:(params, api) => { const x = api.coord([api.value(0), api.value(1)]), y = api.coord([api.value(0), api.value(2)]); "
                                      "return {type: 'group', children: [{type: 'line', shape: {x1: x[0], y1: x[1], x2: y[0], y2: y[1]}, style: {stroke: api.visual('color'), lineWidth: 1.5}}, "
                                      "{type: 'line', shape: {x1: x[0] - 5, y1: x[1], x2: x[0] + 5, y2: x[1]}, style: {stroke: api.visual('color'), lineWidth: 1.5}}, "
                                      "{type: 'line', shape: {x1: y[0] - 5, y1: y[1], x2: y[0] + 5, y2: y[1]}, style: {stroke: api.visual('color'), lineWidth: 1.5}}]}; }",
                           data=[[i, *p["error_interval"]] for i, p in enumerate(pts)]))
    option = dict(grid=dict(left=72, right=96, top=56, bottom=56), legend=dict(top=0, data=[READER[r] for r in ROWS]),
                  tooltip=dict(trigger="axis", valueFormatter="js:v => v == null ? '' : v.toFixed(4)"),
                  xAxis=dict(type="category", data=[f"Guidance {g}" for g in gs], name="Guidance: how hard sampling pushes toward the ask",
                             nameLocation="middle", nameGap=34),
                  yAxis=dict(type="log", logBase=10, min=0.001, max=0.2, name="Ask-versus-got error, asks 1/24 to 17/24 (log scale)", nameLocation="middle", nameGap=56,
                             splitLine=dict(lineStyle=dict(opacity=0.35))),
                  series=series, animation=False)
    gaps = ", ".join(f"{e['gap_linear_minus_per_field']:+.4f} at guidance {g}" for g, e in result["guidance"].items())
    return write_chart(option, Path(folder) / "random-order-schedule.html", height=460, title=headline(result),
                       subtitle=f"Solid: the two random-order rows. Linear minus per-field error: {gaps}. Dashed: the species-first rows, for scale.",
                       note="Error: mean squared gap between the ask k/24 and the mean measured win rate of the 192 teams drawn at it, over asks 1/24 to 17/24. "
                            "Whisker: 95% bootstrap interval over teams and the 49 opponents. Win rate: one battle against each of the 49 distinct top-50 meta teams, "
                            "behaviour-cloning policy on both sides, slots shuffled.")


def comment(figure_url):
    """The verdict comment, picture first. figure_url has {theme} where light or dark goes."""
    r = json.load(open(OUT)); gs = list(r["guidance"])
    iv = lambda x, f="+.4f": f"[{x[0]:{f}}, {x[1]:{f}}]"
    name = {"linear": "linear", "per_field": "per-field", "neither": "neither"}
    lines = [f"**Verdict, {datetime.date.today()}: {headline(r)}.** "
             f"Calls by guidance: {', '.join(f'{g}: {name[c]}' for g, c in r['calls'].items())}.", "",
             f'<picture><source media="(prefers-color-scheme: dark)" srcset="{figure_url.format(theme="dark")}"><img alt="Ask-versus-got error by guidance for the four rows" src="{figure_url.format(theme="light")}"></picture>', "",
             "The figure compares the two random-order rows at each guidance; the two species-first rows show the scale. "
             "Per-field, random order decodes the control's checkpoint (`stat84_shuffled_noise_per_field_es3_cap80`) exactly as Linear, random order decodes "
             "`stat84_shuffled_noise_linear_es3_cap80`, so the masking schedule is the only difference.", "",
             "| Guidance | Per-field, random order | Linear, random order | Linear minus per-field, paired by seed | Call |", "|---|---|---|---|---|"]
    for g in gs:
        e = r["guidance"][g]; p, lin = e["rows"][PER_FIELD_RANDOM], e["rows"][LINEAR_RANDOM]
        lines.append(f"| {g} | {p['error']:.4f} {iv(p['error_interval'], '.4f')} | {lin['error']:.4f} {iv(lin['error_interval'], '.4f')} | "
                     f"{e['gap_linear_minus_per_field']:+.4f} {iv(e['gap_interval'])}{' (within training luck)' if e['within_training_luck'] else ''} | {name[e['call']]} |")
    lines += ["", f"Rule, from the ticket: {r['rule']}.", "",
              "Each random-order row read against its own model's species-first error:", "",
              "| Guidance | Per-field, random order ÷ the control | Linear, random order ÷ Linear, species-first |", "|---|---|---|"]
    for g in gs:
        e = r["guidance"][g]["rows"]
        lines.append(f"| {g} | {e[PER_FIELD_RANDOM]['error_multiple_of_species_first']:.1f}× | {e[LINEAR_RANDOM]['error_multiple_of_species_first']:.1f}× |")
    lines += ["", "<details><summary>The other measures</summary>", "",
              "| Guidance | Row | Win rate, all asks | Win rate, asks 15/24 to 17/24 | First-try legality | Distinct line-ups | Novel against training | Pass@1, asks 13/24 to 17/24 | Pass@48 |",
              "|---|---|---|---|---|---|---|---|---|"]
    for g in gs:
        e = r["guidance"][g]
        for row in ROWS:
            m = e["rows"][row]; pk = e["pass_at_k"].get(row)
            pk_cells = f"{pk['pass_at_1']:.3f} | {pk['pass_at_48']:.3f} {iv(pk['pass_at_48_interval'], '.3f')}" if pk else "not read | not read"
            lines.append(f"| {g} | {m['reader']} | {m['pooled_win']:.3f} | {m['win_top_asks']:.3f} | {m['first_try_legality']:.1%} | {m['line_ups']} | "
                         f"{m['new_against_training']:.1%} | {pk_cells} |")
    lines += ["", "Win rate: one battle against each of the 49 distinct top-50 meta teams, behaviour-cloning policy on both sides, slots shuffled. "
              "First-try legality: the share of draws the validator accepted before any redraw. Pass@k: the chance that at least one of k teams is an elite line-up "
              "that held in the re-battle (`pass_at_k.py`, #331). Interval: 95% bootstrap, teams resampled within each ask and sampling seed, the 49 opponents once for every row. "
              "Script: `code/src/random_order_schedule.py read`. Results: `~/vgc-data/results/random_order_schedule.json`.", "</details>"]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    {"read": read, "figure": lambda: print(figure(sys.argv[2])), "comment": lambda: print(comment(sys.argv[2]), end="")}[sys.argv[1]]()
