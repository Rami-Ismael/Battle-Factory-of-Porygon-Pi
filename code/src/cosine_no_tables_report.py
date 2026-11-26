"""Issue #451 figures: the table-free 5e-4 recipe (cosine learning rate, no tables in the loss) against its two parents.

Reads the nats check's report (`~/vgc-data/results/nats_check/report.json`) and the two table-free run JSONs, and writes
three ECharts pages (docs/cosine-lr-no-tables-in-the-loss.md, Storyboard):
  nats-2x2.html          nats per team tables on and off, learning-rate recipe by loss, one cell per trained model
  child-minus-parent.html   table-free 5e-4 minus each parent, paired 95% intervals, against the ±1 nat band
  rules-by-epoch.html    share of the 64 teams drawn without the tables that break each rule, every epoch,
                         table-free 5e-4 against table-free 4e-4

    /tmp/vgc-pilot/.venv/bin/python src/cosine_no_tables_report.py --out docs/figures/cosine_no_tables
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from echarts_chart import write_chart

RESULTS = Path.home() / "Documents/code/vgc-team-generator-pilot/results"
NATS = Path.home() / "vgc-data/results/nats_check/report.json"
CHILD = "no_table_loss_gidd_lr"
# nats_check code ID -> (reader name, learning-rate recipe, loss)
MODELS = {
    "linear_filtered": ("Table-masked 4e-4 recipe", "4e-4 recipe", "table-masked"),
    "linear_filtered_gidd_lr": ("Table-masked 5e-4 recipe", "5e-4 recipe", "table-masked"),
    "no_table_loss": ("Table-free 4e-4 recipe", "4e-4 recipe", "table-free"),
    CHILD: ("Table-free 5e-4 recipe", "5e-4 recipe", "table-free"),
}
RECIPES, LOSSES = ("4e-4 recipe", "5e-4 recipe"), ("table-masked", "table-free")
# Run JSON of each table-free model's training, whose curves hold the per-epoch rule shares
RUNS = {"Table-free 5e-4 recipe": "stat84_shuffled_noise_linear_filtered_lr5e4_min5e5_no_table_loss_es3_cap80",
        "Table-free 4e-4 recipe": "stat84_shuffled_noise_linear_no_table_loss_es3_cap80"}
COLORS = {"Table-free 5e-4 recipe": "#c2410c", "Table-free 4e-4 recipe": "#8a8984"}
DARK = {"#c2410c": "#f08a4b", "#8a8984": "#8f8e88"}
RULES = ("ability the species can't have", "move the species can't learn", "same move twice on one Pokémon",
         "wrong Mega Stone", "other item the species can't hold", "species used more than once",
         "item used more than once", "Stat Points over 66")
MARGIN = 1.0                                   # nats per team, the smallest effect worth acting on (pre-registration)
DEFINE = ("Table-free: the legal Pokémon construction tables left out of the training loss. Table-masked: the loss masks every "
          "forbidden value. 5e-4 recipe: peak 5e-4, 1,000 warm-up steps, cosine to 5e-5. 4e-4 recipe: 4e-4, 500 warm-up steps, "
          "cosine to 0. Same table, seed and network otherwise.")
NATS_NOTE = ("Nats per team: exact held-out log-likelihood under the species-first sampler, −log p summed over 84 fields; lower is "
             "better. 1,024 held-out teams. Tables on / off: the sampler draws with or without the legal Pokémon construction "
             "tables. " + DEFINE)


def nats_table(report):
    """{code: {"tables on": nats, "tables off": nats}} for the four models of the 2-by-2 grid."""
    m = report["measures"]
    return {c: {"tables on": m["nats_tables_on"][c], "tables off": m["nats_tables_off"][c]} for c in MODELS if c in m["nats_tables_on"]}


def two_by_two_figure(report, out):
    nats = nats_table(report)
    cell = {(MODELS[c][1], MODELS[c][2]): v for c, v in nats.items()}
    data = []
    for panel, mode in enumerate(("tables on", "tables off")):
        for i, loss in enumerate(LOSSES):
            for j, recipe in enumerate(RECIPES):
                v = cell.get((recipe, loss), {}).get(mode)
                data.append({"value": [j + 3 * panel, i, None if v is None else round(v, 2)]})
    values = [[d["value"][2] for d in data if d["value"][2] is not None and (d["value"][0] >= 3) == bool(k)] for k in (0, 1)]
    option = {
        "tooltip": {"formatter": "js:p => p.value[2] == null ? 'not trained' : p.value[2].toFixed(2) + ' nats per team'"},
        "grid": {"left": 110, "right": 24, "top": 40, "bottom": 70},
        "xAxis": {"type": "category", "data": ["4e-4 recipe", "5e-4 recipe", "", "4e-4 recipe", "5e-4 recipe"],
                  "axisTick": {"show": False}, "axisLabel": {"interval": 0, "fontSize": 11}, "splitArea": {"show": False}},
        "yAxis": {"type": "category", "data": ["table-masked", "table-free"], "axisTick": {"show": False}},
        "visualMap": [{"seriesIndex": k, "min": min(v), "max": max(v), "show": k == 0, "calculable": False, "orient": "horizontal",
                       "left": "center", "bottom": 0, "inRange": {"color": ["#1b7a5a", "#f2efe6", "#b4442a"]},
                       "text": ["more nats", "fewer nats"]} for k, v in enumerate(values)],
        "title": [{"text": "Tables on", "left": 130, "top": 8, "textStyle": {"fontSize": 13}},
                  {"text": "Tables off", "right": 60, "top": 8, "textStyle": {"fontSize": 13}}],
        "series": [{"type": "heatmap", "data": [d for d in data if (d["value"][0] >= 3) == bool(k)],
                    "label": {"show": True, "fontSize": 15, "formatter": "js:p => p.value[2] == null ? 'not trained' : p.value[2].toFixed(1)"},
                    "itemStyle": {"borderColor": "#fcfcfb", "borderWidth": 3}, "emphasis": {"itemStyle": {"borderColor": "#0b0b0b"}}}
                   for k in (0, 1)],
    }
    child = nats.get(CHILD, {}).get("tables on")
    title = "Nats per team, learning-rate recipe by loss"
    if child is not None and len(nats) == 4:
        on = {MODELS[c][1:]: v["tables on"] for c, v in nats.items()}
        additive = on[("5e-4 recipe", "table-masked")] + on[("4e-4 recipe", "table-free")] - on[("4e-4 recipe", "table-masked")]
        title = (f"With the tables on, the two gains do not add: the table-free 5e-4 recipe scores {child:.1f} nats, "
                 f"not the additive {additive:.1f}")
    return write_chart(option, Path(out) / "nats-2x2.html", title=title,
                       subtitle="Nats per team on 1,024 held-out teams, one cell per trained model. Lower is better. Each panel has its own color scale.",
                       note=NATS_NOTE, height=380)


def child_minus_parent_figure(report, out):
    rows = []
    for parent, by_mode in report.get("parent_comparisons", {}).get(CHILD, {}).items():
        b = by_mode["tables on"]
        rows.append((f"vs {MODELS[parent][0]}", b["mean"], b["lo"], b["hi"]))
    if not rows: raise SystemExit(f"no parent comparison for {CHILD} in the nats report; run nats_check.py report after scoring it")
    cats = [r[0] for r in rows]
    option = {
        "tooltip": {"trigger": "item", "formatter": "js:p => p.name + '<br>' + p.data.label"},
        "grid": {"left": 210, "right": 40, "top": 30, "bottom": 50},
        "yAxis": {"type": "category", "data": cats, "inverse": True, "axisTick": {"show": False}},
        "xAxis": {"type": "value", "name": "child minus parent, nats per team", "nameLocation": "middle", "nameGap": 30},
        "series": [
            {"type": "custom", "name": "interval", "silent": True, "z": 2, "encode": {"x": [1, 2], "y": 0},
             "data": [[i, r[2], r[3]] for i, r in enumerate(rows)],
             "renderItem": "js:(params, api) => { const y = api.coord([0, api.value(0)])[1]; const a = api.coord([api.value(1), api.value(0)])[0]; "
                           "const b = api.coord([api.value(2), api.value(0)])[0]; const s = {stroke: '#c2410c', lineWidth: 2.5}; "
                           "return {type: 'group', children: [{type: 'line', shape: {x1: a, y1: y, x2: b, y2: y}, style: s},"
                           "{type: 'line', shape: {x1: a, y1: y - 7, x2: a, y2: y + 7}, style: s},"
                           "{type: 'line', shape: {x1: b, y1: y - 7, x2: b, y2: y + 7}, style: s}]}; }"},
            {"type": "scatter", "name": "mean", "symbolSize": 11, "color": "#c2410c", "z": 3, "emphasis": {"focus": "series"},
             "data": [{"value": [round(r[1], 3), i], "name": r[0], "label": f"{r[1]:+.2f} [{r[2]:+.2f}, {r[3]:+.2f}] nats per team"}
                      for i, r in enumerate(rows)],
             "label": {"show": True, "position": "top", "fontSize": 11, "formatter": "js:p => p.value[0].toFixed(2)"},
             "markArea": {"silent": True, "itemStyle": {"color": "rgba(138,137,132,0.16)"}, "data": [[{"xAxis": -MARGIN}, {"xAxis": MARGIN}]]},
             "markLine": {"silent": True, "symbol": "none", "lineStyle": {"color": "#52514e", "type": "solid"}, "label": {"show": False},
                          "data": [{"xAxis": 0}]}},
        ],
    }
    worse = [r for r in rows if r[1] > 0]; better = [r for r in rows if r[3] < -MARGIN]
    if len(better) == len(rows): title = "The table-free 5e-4 recipe beats both parents with the tables on"
    elif better and worse:
        title = (f"The table-free 5e-4 recipe beats the {better[0][0][3:].lower()} by {-better[0][1]:.2f} nats per team "
                 f"but trails the {worse[0][0][3:].lower()} by {worse[0][1]:.2f}")
    else: title = "The table-free 5e-4 recipe does not beat both parents by a nat with the tables on"
    return write_chart(option, Path(out) / "child-minus-parent.html", title=title,
                       subtitle="Table-free 5e-4 recipe minus each parent, nats per team with the tables on, paired bootstrap 95% over 1,024 held-out teams. "
                                f"Left of zero = the child fits better. Grey band: within ±{MARGIN:g} nat, too small to act on.",
                       note=NATS_NOTE + " One training seed per model: the interval covers the held-out teams, not training luck.",
                       height=260)


def rule_curves(results=RESULTS):
    return {arm: [e for e in json.loads((Path(results) / f"{name}.json").read_text())["curves"] if "rule_shares" in e]
            for arm, name in RUNS.items() if (Path(results) / f"{name}.json").exists()}


def rules_figure(curves, out):
    """Small multiples, one panel per rule, two lines each: the share of 64 drawn teams breaking that rule, every epoch."""
    cols = 4
    grids, xs, ys, series, titles = [], [], [], [], []
    for k, rule in enumerate(RULES):
        r, c = divmod(k, cols)
        grids.append({"left": f"{4 + c * 24.5}%", "width": "19.5%", "top": f"{14 + r * 42}%", "height": "28%"})
        xs.append({"type": "value", "gridIndex": k, "min": 0, "max": 80, "interval": 20, "axisLabel": {"fontSize": 10}})
        ys.append({"type": "value", "gridIndex": k, "min": 0, "max": 1, "interval": 0.5,
                   "axisLabel": {"fontSize": 10, "formatter": "js:v => Math.round(v * 100) + '%'", "show": c == 0}})
        titles.append({"text": rule, "left": f"{4 + c * 24.5}%", "top": f"{8 + r * 42}%", "textStyle": {"fontSize": 12}})
        for arm, entries in curves.items():
            series.append({"type": "line", "name": arm, "xAxisIndex": k, "yAxisIndex": k, "color": COLORS[arm], "symbol": "none",
                           "lineStyle": {"width": 2.2 if arm.endswith("5e-4 recipe") else 1.6}, "emphasis": {"focus": "series"},
                           "data": [[e["epoch"], round(e["rule_shares"].get(rule, 0), 4)] for e in entries]})
    last = {arm: entries[-1] for arm, entries in curves.items() if entries}
    child = last.get("Table-free 5e-4 recipe")
    title = "Share of drawn teams breaking each rule, every epoch of training"
    if child is not None and len(curves) == 2:
        shift = {arm: next((e["epoch"] for e in entries if e["rule_shares"].get(RULES[0], 1) < 0.5), None) for arm, entries in curves.items()}
        end = {arm: sum(e["illegal_share"] for e in entries[-10:]) / len(entries[-10:]) for arm, entries in curves.items()}
        new, old = "Table-free 5e-4 recipe", "Table-free 4e-4 recipe"
        title = (f"Both recipes learn to bind fields to their own Pokémon at the same time (epoch {shift[new]} and {shift[old]}); "
                 f"over the last 10 epochs the 5e-4 recipe draws {end[new]:.0%} illegal teams, the 4e-4 recipe {end[old]:.0%}")
    option = {"tooltip": {"trigger": "axis", "valueFormatter": "js:v => Math.round(v * 100) + '%'"},
              "legend": {"top": 0, "left": 0, "data": list(curves)},
              "grid": grids, "xAxis": xs, "yAxis": ys, "series": series, "title": titles,
              "axisPointer": {"link": [{"xAxisIndex": "all"}]}}
    return write_chart(option, Path(out) / "rules-by-epoch.html", title=title,
                       subtitle="After every epoch the model draws 64 teams without the tables or the Stat Point budget, at the elite cut "
                                "0.531, guidance 2, species-first, draw seed 12345. Each panel: share of those teams breaking one rule. "
                                "A team can break several. Lower is better.",
                       note=DEFINE + " The table-masked models are not drawn: their training check drew through the tables, so it reads 0.",
                       height=560, dark=DARK)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=NATS)
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--rules-only", action="store_true", help="only the rule curves, before the child is scored")
    args = parser.parse_args(argv)
    pages = [rules_figure(rule_curves(args.results), args.out)]
    if not args.rules_only:
        report = json.loads(args.report.read_text())
        pages += [two_by_two_figure(report, args.out), child_minus_parent_figure(report, args.out)]
    for p in pages: print(p)


if __name__ == "__main__":
    main()
