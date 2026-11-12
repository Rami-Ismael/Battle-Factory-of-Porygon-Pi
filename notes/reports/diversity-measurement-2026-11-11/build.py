"""Build the 2026-11-11 diversity page from ~/vgc-data/results/diversity_vs_training_2026-11-11.json.

    python3 build.py      (any Python 3; writes charts/<chart>.html and index.html beside this file)

Each chart's ECharts option is a dict built here and written as its own page by write_chart (code/src/echarts_chart.py),
which follows light and dark and redraws on resize. write_chart draws one chart per page, so index.html, the prose
and tables from template.html, embeds each chart page in an iframe.
"""
import html
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "code/src"))
from echarts_chart import write_chart

RESULT = Path.home() / "vgc-data/results/diversity_vs_training_2026-11-11.json"
SEEDS = (1, 2, 3)
MODEL, TRAIN, TALL = "Default model", "Win-rate-matched training", "Any-win-rate training"
VALIDATION, COPIED, RANDOM = "Validation teams", "One training team repeated", "Random legal teams"
ROW_NAMES = {
    **{f"model_s{seed}": f"{MODEL}, sampling seed {seed}" for seed in SEEDS},
    **{f"train_s{seed}": f"{TRAIN}, draw {seed}" for seed in SEEDS},
    **{f"train_all_s{seed}": f"{TALL}, draw {seed}" for seed in SEEDS},
    "val": VALIDATION, "copied": COPIED, "random": RANDOM,
}
ROW_ORDER = list(ROW_NAMES)
MODEL_KEYS, TRAIN_KEYS, TALL_KEYS = ([f"{prefix}{seed}" for seed in SEEDS] for prefix in ("model_s", "train_s", "train_all_s"))
# Light-surface color, then its dark-surface partner, from the page palette in template.html.
COLORS = {"model": ("#2a78d6", "#3987e5"), "train": ("#eb6834", "#d95926"), "trainall": ("#8a4fc4", "#a77ae0"),
          "val": ("#1d8a6e", "#3fb592"), "anchor": ("#8a8984", "#8f8e86"), "band": ("#e4e2dc", "#34332f"),
          "rule": ("#7a7974", "#97968d"), "copied_part": ("#c43b3b", "#e05a5a"), "one_part": ("#e39a3b", "#e8a84f"),
          "two_part": ("#8fb4de", "#5f86b8")}
COLOR = {name: light for name, (light, _) in COLORS.items()}
DARK = dict(COLORS.values())
# Figure 1: (level, measure key, reader name)
RATIO_ROWS = [
    ("Line-up", "distinct_lineups", "Distinct line-ups"), ("Line-up", "effective_lineups", "Effective line-ups"),
    ("Line-up", "distinct_species", "Distinct species"), ("Line-up", "effective_species", "Effective species"),
    ("Line-up", "distinct_pairs", "Distinct species pairs"), ("Line-up", "vendi_lineup", "Vendi Score, line-ups"),
    ("Builds", "distinct_sets", "Distinct builds"), ("Builds", "effective_sets", "Effective builds"),
    ("Builds", "sets_per_species", "Builds per species, all uses"),
    ("Builds", "builds_per_species", "Builds per species at 30 uses"),
    ("Builds", "vendi_sets", "Vendi Score, builds"), ("Builds", "distance_sets", "Mean distance, builds"),
    ("Stat Points", "distinct_statpoints", "Distinct Stat Points"),
    ("Stat Points", "effective_statpoints", "Effective Stat Points"),
    ("Stat Points", "vendi_statpoints", "Vendi Score, Stat Points"),
    ("Stat Points", "distance_statpoints", "Mean distance, Stat Points"),
    ("Whole team", "distinct_teams84", "Distinct teams, 84 fields"), ("Whole team", "vendi_team", "Vendi Score, whole team"),
    ("Whole team", "dcscore_team", "DCScore at temperature 0.1"), ("Whole team", "distance_team", "Mean distance, whole team"),
    ("Whole team", "nearest_mean", "Mean nearest training distance"),
]
ARCHETYPES = ["sun", "rain", "sand", "snow", "electric terrain", "grassy terrain", "misty terrain", "psychic terrain",
              "Trick Room", "Tailwind", "none"]
RATIO_MIN, RATIO_MAX = 0.3, 3.5
NUMBER_JS = ("js:v => v == null || Number.isNaN(v) ? 'n/a' : Math.abs(v) >= 100 ? Math.round(v).toLocaleString('en-US')"
             " : Math.abs(v) >= 10 ? v.toFixed(1) : Math.abs(v) >= 1 ? v.toFixed(2) : v.toFixed(3)")
PERCENT_JS = "js:v => (100 * v).toFixed(1) + '%'"
PERCENT_AXIS_JS = "js:v => Math.round(v * 100) + '%'"


def mean(values): return sum(values) / len(values)


def number(value):
    """The page's number format: thousands separators from 100, then 1, 2 or 3 decimals."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    size = abs(value)
    if size >= 100:
        return f"{math.floor(value + 0.5):,}"
    return f"{value:.1f}" if size >= 10 else f"{value:.2f}" if size >= 1 else f"{value:.3f}"


def percent(value): return f"{100 * value:.1f}%"


def row_color(key):
    group = "model" if key.startswith("model") else "trainall" if key.startswith("train_all") else "train" if key.startswith("train") \
        else "val" if key == "val" else "anchor"
    return COLOR[group]


def value_axis(name, **settings):
    return {"type": "value", "name": name, "nameLocation": "middle", "nameGap": 28, **settings}


def category_axis(names, **settings):
    return {"type": "category", "data": names, "inverse": True, "axisTick": {"show": False}, **settings}


def axis_tooltip(**settings):
    return {"trigger": "axis", "axisPointer": {"type": "shadow"}, "confine": True, **settings}


def grid(top=40, bottom=44):
    return {"left": 8, "right": 28, "top": top, "bottom": bottom, "containLabel": True}


def ratio_chart(result):
    """Figure 1: default model value over win-rate-matched training value per measure, with the noise floor band."""
    measures, floor, verdict = result["measures"], result["floor"], result["verdict"]
    clip = lambda ratio: min(max(ratio, RATIO_MIN), RATIO_MAX)
    names, band, seed_dots, train_all_dots, tooltips = [], [], [], [], {}
    for index, (level, key, name) in enumerate(RATIO_ROWS):
        label = f"{level} · {name}"
        model, train, train_all = ([measures[row][key] for row in keys] for keys in (MODEL_KEYS, TRAIN_KEYS, TALL_KEYS))
        train_mean = mean(train)
        names.append(label)
        band.append([index, clip(1 + floor[key]["lo"] / train_mean), clip(1 + floor[key]["hi"] / train_mean)])
        seed_dots += [[clip(model_value / train_value), label] for model_value, train_value in zip(model, train)]
        train_all_dots.append([clip(mean(train_all) / train_mean), label])
        tooltips[label] = (f"<b>{html.escape(label)}</b><br>{MODEL}: {', '.join(map(number, model))}<br>"
                           f"{TRAIN}: {', '.join(map(number, train))}<br>{TALL}: {', '.join(map(number, train_all))}<br>"
                           f"{VALIDATION} {number(measures['val'][key])} · {COPIED} {number(measures['copied'][key])} · "
                           f"{RANDOM} {number(measures['random'][key])}<br>"
                           f"Differs beyond the noise floor: {'yes' if verdict[key]['differs'] else 'no'}")
    band_name, seeds_name, train_all_name = "Noise floor", f"{MODEL}, one dot per sampling seed", f"{TALL}, mean of 3 draws"
    draw_band = ("js:(params, api) => { const start = api.coord([api.value(1), api.value(0)]), end = api.coord([api.value(2), api.value(0)]);"
                 " const height = api.size([0, 1])[1] * 0.55; return {type: 'rect', shape: {x: start[0], y: start[1] - height / 2,"
                 " width: Math.max(2, end[0] - start[0]), height, r: 3}, style: {fill: api.visual('color')}}; }")
    option = {
        "animation": False, "legend": {"top": 0, "left": 0, "data": [seeds_name, train_all_name, band_name]},
        "tooltip": axis_tooltip(formatter=f"js:params => ({json.dumps(tooltips)})[params[0].axisValue]"),
        "grid": grid(), "xAxis": value_axis(f"{MODEL} ÷ {TRAIN.lower()}, log scale", type="log", min=RATIO_MIN, max=RATIO_MAX),
        "yAxis": category_axis(names, axisLabel={"fontSize": 11}),
        "series": [
            {"name": band_name, "type": "custom", "renderItem": draw_band, "encode": {"x": [1, 2], "y": 0}, "data": band, "z": 1,
             "itemStyle": {"color": COLOR["band"]}, "silent": True},
            {"name": seeds_name, "type": "scatter", "data": seed_dots, "symbolSize": 9, "itemStyle": {"color": COLOR["model"]}, "z": 3,
             "emphasis": {"focus": "series"},
             "markLine": {"silent": True, "symbol": "none", "label": {"show": False},
                          "lineStyle": {"color": COLOR["rule"], "type": "solid"}, "data": [{"xAxis": 1}]}},
            {"name": train_all_name, "type": "scatter", "data": train_all_dots, "symbol": "diamond", "symbolSize": 10, "z": 4,
             "itemStyle": {"color": "transparent", "borderColor": COLOR["trainall"], "borderWidth": 2}, "emphasis": {"focus": "series"}},
        ],
    }
    return option, dict(
        title="Builds per species at 30 uses and the nearest training distance join the 11-09 gaps on line-ups and builds",
        subtitle=f"Their model dots sit far outside the grey band. The hollow diamonds, {TALL.lower()}, keep about as many line-ups "
                 f"as {TRAIN.lower()} but fewer species and far fewer builds.",
        note=f"Figure 1. The default model's value divided by the value for {TRAIN.lower()}, both at 1,088 teams. Left of 1 means less "
             "diverse, except for the nearest training distance, where left means closer to training. The grey band is how far two "
             "training samples differ, as a share of the training value. Hover a row for the raw numbers.", height=760)


def bar_rows(rows, series, axis_name, value_format=NUMBER_JS, **axis_settings):
    """Horizontal bars, one category per sample row in page order."""
    return {"animation": False, "legend": {"top": 0, "left": 0}, "tooltip": axis_tooltip(valueFormatter=value_format),
            "grid": grid(top=48 if len(series) > 1 else 16), "xAxis": value_axis(axis_name, **axis_settings),
            "yAxis": category_axis([ROW_NAMES[key] for key in ROW_ORDER], axisLabel={"fontSize": 11}),
            "series": [{"type": "bar", "barMaxWidth": 16, "emphasis": {"focus": "series"}, **part,
                        "data": [part["data"](rows[key], key) for key in ROW_ORDER]} for part in series]}


def builds_chart(result):
    rows, uses = result["measures"], result["equal_uses"]
    option = bar_rows(rows, [{"name": f"Builds per species at {uses} uses",
                              "data": lambda row, key: {"value": row["builds_per_species"], "itemStyle": {"color": row_color(key)}}}],
                      f"distinct builds among {uses} uses of one species, mean over species")
    return option, dict(
        title=f"Default-model bars sit near 20.6 builds per species, {TRAIN.lower()} near 25.7",
        subtitle="Two training samples differ by at most 0.68 builds.",
        note="A build is a set without its Stat Points: item, ability, four moves and Stat Alignment. For each species used at least "
             f"{uses} times in a sample, {uses} of its uses are drawn without replacement and their distinct builds counted, 20 draws, "
             f"averaged over species. {uses} uses keeps 36 to 40 species per model seed.", height=420)


def stacked_share_chart(result, parts, stack):
    rows = result["measures"]
    return bar_rows(rows, [{"name": name, "stack": stack, "itemStyle": {"color": COLOR[color]},
                            "data": lambda row, _key, measure=measure: row[measure]} for measure, name, color in parts],
                    "share of teams", value_format=PERCENT_JS, max=1, axisLabel={"formatter": PERCENT_AXIS_JS})


def novelty_chart(result):
    parts = [("novelty_swaps_0", "0 swaps, a training line-up copied", "copied_part"), ("novelty_swaps_1", "1 swap", "one_part"),
             ("novelty_swaps_2", "2 swaps", "two_part"), ("novelty_swaps_3plus", "3 or more swaps, a new line-up", "model")]
    return stacked_share_chart(result, parts, "swaps"), dict(
        title=f"The default model reuses a training line-up exactly for 59% of its teams, against 71% for {TRAIN.lower()}",
        subtitle="It moves one or two species away more often; three or more swaps stays rare at 0.9% to 3.7%.",
        note="Swaps are six minus the species a team shares with the closest of the 170,785 training line-ups. Rows drawn from the "
             "training teams leave each team out of its own reference once, so a training row at zero swaps means another training "
             "team has the same line-up.", height=460)


def nearest_chart(result):
    parts = [("nearest_share_copied_zero_fields", "0 fields, a training team copied", "copied_part"),
             ("nearest_share_one_to_five_fields", "1 to 5 fields", "one_part"),
             ("nearest_share_six_to_twenty_fields", "6 to 20 fields", "two_part"),
             ("nearest_share_over_twenty_fields", "over 20 fields", "model")]
    return stacked_share_chart(result, parts, "fields"), dict(
        title="No team in any row copies a training team at all 84 fields",
        subtitle="The default model's teams sit a median 34 fields from the closest one, against 24 for training and validation teams.",
        note="Fields out of 84 that differ from the closest training team, slots matched to give the fewest, moves compared as a set "
             "within a slot. Exact search over all 170,785 training teams. Training rows and the repeated team leave the team itself "
             "out once.", height=460)


def archetype_chart(result):
    measures = result["measures"]
    groups = [(f"{MODEL}, mean of 3 seeds", MODEL_KEYS, "model"), (f"{TRAIN}, mean of 3 draws", TRAIN_KEYS, "train"),
              (f"{TALL}, mean of 3 draws", TALL_KEYS, "trainall"), (VALIDATION, ["val"], "val"), (RANDOM, ["random"], "anchor")]
    option = {"animation": False, "legend": {"top": 0, "left": 0}, "tooltip": axis_tooltip(valueFormatter=PERCENT_JS),
              "grid": grid(top=76), "xAxis": value_axis("share of teams in the archetype", axisLabel={"formatter": PERCENT_AXIS_JS}),
              "yAxis": category_axis(ARCHETYPES, axisLabel={"fontSize": 12}),
              "series": [{"name": name, "type": "bar", "barMaxWidth": 8, "emphasis": {"focus": "series"}, "itemStyle": {"color": COLOR[color]},
                          "data": [mean([measures[key][f"archetype_{archetype}"] for key in keys]) for archetype in ARCHETYPES]}
                         for name, keys, color in groups]}
    return option, dict(
        title="Tailwind is the commonest archetype everywhere, on 42% to 50% of teams",
        subtitle=f"The default model puts Trick Room on 28% to 37% of its teams and sun on 32% to 35%, against 25% and 29% for "
                 f"{TRAIN.lower()}.",
        note="A team is in an archetype when any of its sets has the setting ability or move, or holds a mega stone whose mega form has "
             "the setting ability. Read from team fields, not from battle logs, so a team in the Trick Room archetype carries Trick Room "
             "but may never use it. A team can be in several archetypes or none.", height=640)


def anchors_chart(result):
    measures = result["measures"]
    levels = [("Line-up", "vendi_lineup"), ("Builds", "vendi_sets"), ("Stat Points", "vendi_statpoints"), ("Whole team", "vendi_team")]
    points = [(COPIED, ["copied"], "anchor", "circle"), (RANDOM, ["random"], "anchor", "rect"), (VALIDATION, ["val"], "val", "triangle"),
              (TRAIN, TRAIN_KEYS, "train", "circle"), (TALL, TALL_KEYS, "trainall", "diamond"), (MODEL, MODEL_KEYS, "model", "circle")]
    option = {"animation": False, "legend": {"top": 0, "left": 0}, "tooltip": axis_tooltip(valueFormatter=NUMBER_JS),
              "grid": grid(top=56), "xAxis": value_axis("Vendi Score: effective number of different teams, log scale", type="log",
                                                        min=1, max=result["n"]),
              "yAxis": category_axis([level for level, _ in levels], axisLabel={"fontWeight": 600}),
              "series": [{"name": name, "type": "scatter", "symbol": symbol, "symbolSize": 11, "emphasis": {"focus": "series"},
                          "itemStyle": {"color": COLOR[color]},
                          "data": [mean([measures[key][measure] for key in keys]) for _, measure in levels]}
                         for name, keys, color, symbol in points]}
    return option, dict(
        title=f"At every level the training teams sit between {COPIED.lower()} and {RANDOM.lower()}, so the score orders the anchors",
        subtitle=f"The default model sits below {TRAIN.lower()} on builds and the whole team, and level with or above {TALL.lower()}.",
        note="Model and training points are means of three sampling seeds or three training draws.", height=380)


CHARTS = {"ratio": ratio_chart, "builds": builds_chart, "novelty": novelty_chart, "nearest": nearest_chart,
          "archetypes": archetype_chart, "anchors": anchors_chart}


def table(header, body_rows):
    """An HTML table; each body cell is (text, numeric), numeric cells get tabular figures."""
    def cell(text, numeric):
        return f'<td class="n">{text}</td>' if numeric else f"<td>{text}</td>"
    head = "".join(f"<th>{name}</th>" for name in header)
    body = "".join("<tr>" + "".join(cell(*value) for value in row) + "</tr>" for row in body_rows)
    return f"<table><tr>{head}</tr>{body}</table>"


def shared_species_table(result):
    rows = []
    for comparison, value in result["builds_shared_species"].items():
        seed = comparison.split("_vs_")[0].removeprefix("model_s")
        other = TALL if "train_all" in comparison else TRAIN
        rows.append([(other, False), (seed, True), (value["species"], True), (number(value["model"]), True),
                     (number(value["training"]), True), (number(value["diff"]), True)])
    floor = result["floor"]["builds_shared_species"]
    differs = "yes" if result["verdict"]["builds_shared_species"]["differs"] else "no"
    body = table([f"{MODEL} against", "Sampling seed", "Species both use 30 times", "Model builds per species",
                  "Training builds per species", "Difference"], rows)
    return body.replace("</table>", f"<tr><td colspan=\"6\">Noise floor for the difference: {number(floor['lo'])} to "
                                    f"{number(floor['hi'])} builds. Differs from {TRAIN.lower()}: {differs}.</td></tr></table>")


def nearest_table(result):
    measures = result["measures"]
    return table(["Row", "Mean fields", "Median fields", "Share at 0 fields"],
                 [[(ROW_NAMES[key], False), (number(measures[key]["nearest_mean"]), True), (number(measures[key]["nearest_median"]), True),
                   (percent(measures[key]["nearest_share_copied_zero_fields"]), True)] for key in ROW_ORDER])


def any_win_rate_table(result):
    measures, verdict = result["measures"], result["verdict_all_win_rates"]
    shown = [("distinct_lineups", "Distinct line-ups"), ("distinct_species", "Distinct species"), ("distinct_sets", "Distinct builds"),
             ("builds_per_species", "Builds per species at 30 uses"), ("vendi_team", "Vendi Score, whole team"),
             ("novelty_swaps_3plus", "Novelty: share at three or more swaps"), ("nearest_mean", "Mean nearest training distance"),
             ("archetype_Tailwind", "Share in Tailwind"), ("archetype_Trick Room", "Share in Trick Room")]
    rows = []
    for measure, name in shown:
        show = percent if measure.startswith(("novelty", "archetype")) else number
        rows.append([(name, False), *[(show(mean([measures[key][measure] for key in keys])), True) for keys in (MODEL_KEYS, TRAIN_KEYS, TALL_KEYS)],
                     ("yes" if verdict[measure]["differs"] else "no", False)])
    return table(["Measure", MODEL, TRAIN, TALL, f"{MODEL} differs from {TALL.lower()}"], rows)


def numbers_table(result):
    measures = result["measures"]
    columns = [("distinct_lineups", "Distinct line-ups"), ("distinct_species", "Distinct species"), ("distinct_sets", "Distinct builds"),
               ("builds_per_species", "Builds per species at 30 uses"), ("builds_species_counted", "Species used 30 times"),
               ("novelty_swaps_0", "0 swaps"), ("novelty_swaps_1", "1 swap"), ("novelty_swaps_2", "2 swaps"),
               ("novelty_swaps_3plus", "3 or more swaps"), ("nearest_mean", "Mean nearest distance"), ("nearest_median", "Median"),
               ("nearest_share_copied_zero_fields", "At 0 fields"), ("vendi_team", "Vendi Score, whole team")]
    show = lambda measure: percent if measure.startswith(("novelty", "nearest_share")) else number
    return table(["Row", *[name for _, name in columns]],
                 [[(ROW_NAMES[key], False), *[(show(measure)(measures[key][measure]), True) for measure, _ in columns]] for key in ROW_ORDER])


TABLES = {"shared": shared_species_table, "nearest": nearest_table, "any-win-rate": any_win_rate_table, "numbers": numbers_table}


def chart_frame(name, words):
    return (f'<iframe class="chart" src="charts/{name}.html" title="{html.escape(words["title"])}" loading="lazy" '
            f'style="height:{words["height"] + 260}px"></iframe>')


def build():
    result = json.loads(RESULT.read_text())
    page = (HERE / "template.html").read_text()
    for name, make in CHARTS.items():
        option, words = make(result)
        write_chart(option, HERE / "charts" / f"{name}.html", words["title"], words["subtitle"], words["note"],
                    height=words["height"], dark=DARK)
        page = page.replace(f"<!--chart:{name}-->", chart_frame(name, words))
    for name, make in TABLES.items():
        page = page.replace(f"<!--table:{name}-->", make(result))
    if "<!--chart:" in page or "<!--table:" in page:
        raise ValueError("template.html names a chart or table that build.py does not make")
    (HERE / "index.html").write_text(page)
    print(f"wrote {HERE / 'index.html'} and {len(CHARTS)} chart pages in {HERE / 'charts'}")


if __name__ == "__main__":
    build()
