"""Six figures from the finished masking-schedule rows (docs/masking-rate-rows.md, Figures frames 1 to 3 and later checks; issue #297).

  05_error_against_line_ups       ask-versus-got error against distinct line-ups, one three-point curve for each row
  06_got_against_ask              got against ask, the control beside Linear, species-first, one panel for each guidance
  07_random_order_error_multiple  random order's error as a multiple of species-first's on the same checkpoint
  08_line_ups_by_ask              each row's distinct line-ups among one ask's teams, one panel for each row, a line for each guidance
  09_line_ups_shared_between_asks each row's line-ups counted ask by ask, split into the grid total and the repeats between asks
  10_training_and_novel_line_ups    each row's distinct line-ups split into training line-ups and novel line-ups

Each builder takes the loaded summary (masking_rate_rows_summary.json) and returns (option, title); the fourth to sixth also return a subtitle.

    /tmp/vgc-pilot/.venv/bin/python src/masking_rate_rows_figures.py      (from code/)
"""
import json
import math
from pathlib import Path

from echarts_chart import write_chart
from masking_rate_rows import CONTROL, GUIDANCES, ROWS, SUMMARY, grid_name

OUT = Path(__file__).resolve().parent.parent / "docs/figures/masking_rate_rows"
BLUE, AQUA, ORANGE, GREY, PLAIN = "#2a78d6", "#1baf7a", "#e0702a", "#a9a8a2", "#6f6e69"
COLORS = {CONTROL: BLUE, "sched_linear_species_first": AQUA, "sched_linear_random": ORANGE}
LINEAR, RANDOM = "sched_linear_species_first", "sched_linear_random"
# Label side for each (row, guidance) point, chosen so the six close points of the two species-first rows stay apart; others go on top.
LABEL_SIDE = {(CONTROL, 2): "left", (CONTROL, 4): "left", (CONTROL, 8): "bottom", (LINEAR, 2): "right", (LINEAR, 4): "right", (LINEAR, 8): "right"}
MASK = dict(backgroundColor="#fff", padding=[1, 3], borderRadius=2)     # hides error bars behind a label; "#fff" turns to the dark surface in dark mode
DEFAULT_MODEL = dict(line_ups=1437, error=0.0053, name="Today's default model, guidance 2")
PER_FIELD_MULTIPLE = dict(random=0.0534, species_first=0.0192)       # all-teams 21-epoch per-field model, 2026-11-07, guidance 2
WIN_RATE = "Win rate: one battle against each of the 49 distinct top-50 meta teams, behaviour-cloning policy on both sides, slots shuffled."
ERROR = "Error: the mean squared gap between the win rate asked for and the win rate got, on asks 1/24 to 17/24. Lower is better."
# Vertical error bar with caps: data rows are [x, low, high]; works on value, log and category axes.
ERROR_BAR = ("js:(p, api) => { const x = api.value(0), lo = api.coord([x, api.value(1)]), hi = api.coord([x, api.value(2)]);"
             " const c = api.visual('color'), w = 5, s = {stroke: c, lineWidth: 1.4, opacity: 0.75};"
             " const seg = (x1, y1, x2, y2) => ({type: 'line', shape: {x1, y1, x2, y2}, style: s});"
             " return {type: 'group', children: [seg(lo[0], lo[1], hi[0], hi[1]), seg(lo[0] - w, lo[1], lo[0] + w, lo[1]), seg(hi[0] - w, hi[1], hi[0] + w, hi[1])]}; }")


def label(row):
    return ROWS[row] + (", the control" if row == CONTROL else "")


def grids_of(summary, row):
    return [summary["grids"][grid_name(row, g)] for g in GUIDANCES]


def error_bars(name, color, rows, z=1, **axes):
    return dict(type="custom", name=name, renderItem=ERROR_BAR, itemStyle=dict(color=color), silent=True, tooltip=dict(show=False), z=z,
                encode=dict(x=0, y=[1, 2]), data=rows, **axes)


def error_against_line_ups(summary):
    series = []
    for row, color in COLORS.items():
        gs = grids_of(summary, row)
        points = [dict(value=[g["line_ups"], round(g["error"], 5), int(g["guidance"])], label=dict(position=LABEL_SIDE.get((row, int(g["guidance"])), "top")))
                  for g in gs]
        series.append(dict(name=label(row), type="line", color=color, symbolSize=9, lineStyle=dict(width=2.5), emphasis=dict(focus="series"), z=3,
                           label=dict(show=True, formatter="js:p => 'guidance ' + p.value[2]", color=color, **MASK), data=points))
        series.append(error_bars(label(row), color, [[g["line_ups"], *g["error_interval"]] for g in gs]))
    series.append(dict(name=DEFAULT_MODEL["name"], type="scatter", symbol="emptyCircle", symbolSize=11, color=PLAIN,
                       itemStyle=dict(borderWidth=2), emphasis=dict(focus="series"),
                       label=dict(show=True, position="left", formatter=DEFAULT_MODEL["name"].replace(", ", ",\n"), color=PLAIN, **MASK),
                       data=[[DEFAULT_MODEL["line_ups"], DEFAULT_MODEL["error"], 2]]))
    tip = "js:p => p.seriesName + '<br>guidance ' + p.value[2] + ': ' + p.value[0].toLocaleString() + ' line-ups, error ' + p.value[1].toFixed(4)"
    option = dict(legend=dict(top=0), grid=dict(left=72, right=40, top=56, bottom=56), tooltip=dict(trigger="item", formatter=tip),
                  xAxis=dict(type="value", name="Distinct line-ups among 4,416 teams", nameLocation="middle", nameGap=32, min=1000, max=4000),
                  yAxis=dict(type="log", name="Error, log scale", nameLocation="middle", nameGap=52, min=0.001, max=0.05,
                             minorTick=dict(show=True), minorSplitLine=dict(show=True, lineStyle=dict(opacity=0.35))),
                  series=series)
    linear = summary["row_verdicts"][LINEAR]["overall"]
    ratios = [r["error"] / c["error"] for r, c in zip(grids_of(summary, RANDOM), grids_of(summary, CONTROL))]
    more = min(r["line_ups"] - c["line_ups"] for r, c in zip(grids_of(summary, RANDOM), grids_of(summary, CONTROL)))
    random_words = (f"Linear, random order adds at least {more:,} line-ups at {min(ratios):.0f} to {max(ratios):.0f} times the control's error" if more > 0
                    else f"Linear, random order has {min(ratios):.0f} to {max(ratios):.0f} times the control's error")
    linear_words = {"matches": "matches the control at every guidance", "wins": "beats the control", "loses": "loses to the control"}.get(linear, f"reads {linear} against the control")
    return option, f"Linear, species-first {linear_words}. {random_words}"


def got_against_ask(summary):
    asks = list(range(1, 24))
    width, gap, left = 27.0, 5.0, 7.0
    grids, xs, ys, titles, series = [], [], [], [], []
    for i, g in enumerate(GUIDANCES):
        x0 = left + i * (width + gap)
        grids.append(dict(left=f"{x0}%", width=f"{width}%", top=78, bottom=58))
        titles.append(dict(text=f"Guidance {g}", left=f"{x0 + width / 2}%", top=44, textAlign="center", textStyle=dict(fontSize=13, fontWeight="normal")))
        xs.append(dict(type="value", gridIndex=i, min=1, max=23, interval=4, axisLabel=dict(formatter="{value}/24"),
                       name="Win rate asked for" if i == 1 else "", nameLocation="middle", nameGap=30))
        ys.append(dict(type="value", gridIndex=i, min=0, max=0.7, name="Win rate got, share of battles won" if i == 0 else "",
                       nameLocation="middle", nameGap=40, axisLabel=dict(show=i == 0)))
        axes = dict(xAxisIndex=i, yAxisIndex=i)
        series.append(dict(name="Got equals ask", type="line", symbol="none", color=GREY, lineStyle=dict(type="dashed", width=1.5),
                           emphasis=dict(focus="series"), data=[[1, round(1 / 24, 4)], [23, round(23 / 24, 4)]],
                           markLine=dict(symbol="none", silent=True, lineStyle=dict(color=GREY, type="solid", width=1),
                                         label=dict(formatter="scored asks end at 17/24", position="insideStartTop", color=PLAIN, fontSize=11),
                                         data=[dict(xAxis=17)]), **axes))
        for row in (CONTROL, LINEAR):
            got = summary["grids"][grid_name(row, g)]["got_by_ask"]
            series.append(dict(name=label(row), type="line", color=COLORS[row], symbolSize=4, lineStyle=dict(width=2.2), emphasis=dict(focus="series"),
                               data=[[k, round(got[str(k)], 4)] for k in asks], **axes))
    option = dict(title=titles, legend=dict(top=0), grid=grids, xAxis=xs, yAxis=ys, series=series,
                  tooltip=dict(trigger="axis", valueFormatter="js:v => v == null ? '' : v.toFixed(3)"))
    top = {g: [summary["grids"][grid_name(row, g)]["win_top_asks"] for row in (CONTROL, LINEAR)] for g in GUIDANCES}
    lift = max(lin - con for con, lin in top.values())
    spread = max(abs(lin - con) for con, lin in top.values())
    if lift > 0.02:
        return option, f"The linear masking schedule lifts got at the top scored asks, by up to {lift:.3f}"
    return option, f"The linear masking schedule does not lift got at the top scored asks: at 15/24 to 17/24 the two lines differ by at most {spread:.3f}"


def random_order_error_multiple(summary):
    """Frame 3, redrawn 2026-11-14: the typical miss of each order side by side, not the ratio of their errors. A ratio grows
    when the number under it shrinks, so the bars show both misses and the axis labels carry the ratio."""
    measured = summary["random_order_against_species_first"]
    miss = lambda error: math.sqrt(error)
    groups = [("Older per-field model, guidance 2, a different model", PER_FIELD_MULTIPLE["species_first"], PER_FIELD_MULTIPLE["random"], 0.45)]
    groups += [(f"Linear model, guidance {g}", summary["grids"][grid_name(LINEAR, g)]["error"], summary["grids"][grid_name(RANDOM, g)]["error"], 1.0)
               for g in GUIDANCES]
    multiples = [PER_FIELD_MULTIPLE["random"] / PER_FIELD_MULTIPLE["species_first"]] + [measured[f"guidance_{g}"]["error_multiple"] for g in GUIDANCES]
    names = [f"{name}\nrandom order's error: {m:.1f} times" for (name, *_), m in zip(groups, multiples)]
    bar = lambda value, colour, opacity: dict(value=round(value, 3), itemStyle=dict(color=colour, opacity=opacity))
    label = dict(show=True, position="top", formatter="js:p => p.value.toFixed(2)")
    series = [dict(name="Species-first order", type="bar", barGap="12%", itemStyle=dict(color=AQUA), data=[bar(miss(s), AQUA, o) for _, s, _, o in groups], label=label, emphasis=dict(focus="series")),
              dict(name="Random order", type="bar", itemStyle=dict(color=ORANGE), data=[bar(miss(r), ORANGE, o) for _, _, r, o in groups], label=label, emphasis=dict(focus="series"))]
    option = dict(grid=dict(left=64, right=24, top=48, bottom=72), legend=dict(top=0),
                  tooltip=dict(trigger="axis", valueFormatter="js:v => v.toFixed(3)"),
                  xAxis=dict(type="category", data=names, axisLabel=dict(interval=0, width=210, overflow="break")),
                  yAxis=dict(type="value", name="Typical miss of the asked win rate", nameLocation="middle", nameGap=44, min=0),
                  series=series)
    gaps = [miss(r) - miss(s) for _, s, r, _ in groups[1:]]
    lo, hi = min(gaps), max(gaps)
    if hi - lo <= 0.01 and lo > 0:
        title = (f"On the linear model random order misses the asked win rate by about {(lo + hi) / 2:.2f} more than species-first at every guidance. "
                 f"The error ratio grows from {multiples[1]:.1f} to {multiples[-1]:.1f} times only because species-first's own miss shrinks")
    else:
        title = f"On the linear model random order misses the asked win rate by {lo:.2f} to {hi:.2f} more than species-first"
    return option, title


def spoken(values):
    return ", ".join(values[:-1]) + " and " + values[-1]


# Guidance is marked by line style and shade of the panel's own row colour, so it never reads as a change of row.
GUIDANCE_MARKS = {2: dict(type="dotted", opacity=0.55), 4: dict(type="dashed", opacity=0.78), 8: dict(type="solid", opacity=1.0)}
LEGEND_GREY = "#8f8e88"     # reads on both surfaces; an image icon is not swapped in dark mode
CEILING_SHARE = 0.99        # a row whose mean is at least this share of the teams of one ask at every guidance sits at the ceiling


def guidance_icon(guidance):
    dash = {"dotted": "2 3", "dashed": "6 3", "solid": "none"}[GUIDANCE_MARKS[guidance]["type"]]
    svg = (f"<svg xmlns='http://www.w3.org/2000/svg' width='28' height='10'><line x1='1' y1='5' x2='27' y2='5' stroke='{LEGEND_GREY}' "
           f"stroke-width='2.5' stroke-dasharray='{dash}' opacity='{GUIDANCE_MARKS[guidance]['opacity']}'/></svg>")
    return "image://data:image/svg+xml;utf8," + svg.replace("#", "%23")


def line_ups_by_ask(summary):
    """Within one ask guidance narrows the line-ups of the species-first rows; over the grid the asks share fewer, so the total can still rise.
    One panel for each row on one shared y scale from 0 to the teams of one ask; in each, a line for each guidance over the asks."""
    rows = list(COLORS)
    grids = {row: grids_of(summary, row) for row in rows}
    asks = sorted(int(k) for k in grids[CONTROL][0]["line_ups_by_ask"]); per_ask = grids[CONTROL][0]["teams"] // len(asks)
    means = {row: [round(sum(g["line_ups_by_ask"].values()) / len(asks)) for g in grids[row]] for row in rows}
    totals = {row: [g["line_ups"] for g in grids[row]] for row in rows}
    shown = {0, len(asks) // 4, len(asks) // 2, 3 * len(asks) // 4, len(asks) - 1}
    width, gap, left = 28.0, 3.5, 8.0
    grids_, xs, ys, titles, series = [], [], [], [], []
    for i, row in enumerate(rows):
        x0 = left + i * (width + gap)
        grids_.append(dict(left=f"{x0}%", width=f"{width}%", top=74, bottom=58))
        titles.append(dict(text=label(row), left=f"{x0 + width / 2}%", top=40, textAlign="center", textStyle=dict(fontSize=13, fontWeight="normal")))
        xs.append(dict(type="category", gridIndex=i, boundaryGap=False, data=[f"{k}/24" for k in asks],
                       axisLabel=dict(interval=f"js:i => {sorted(shown)}.includes(i)"),
                       name="Win rate asked for" if i == 1 else "", nameLocation="middle", nameGap=30))
        ys.append(dict(type="value", gridIndex=i, min=0, max=per_ask, interval=per_ask // 4, axisLabel=dict(show=i == 0),
                       name="Distinct line-ups among one ask's teams" if i == 0 else "", nameLocation="middle", nameGap=44))
        for guidance, g in zip(GUIDANCES, grids[row]):
            mark = GUIDANCE_MARKS[guidance]
            series.append(dict(name=f"Guidance {guidance:g}", type="line", xAxisIndex=i, yAxisIndex=i, symbolSize=4, emphasis=dict(focus="series"),
                               lineStyle=dict(width=2.2, color=COLORS[row], **mark), itemStyle=dict(color=COLORS[row], opacity=mark["opacity"]),
                               data=[g["line_ups_by_ask"][str(k)] for k in asks]))
    legend = dict(top=0, itemWidth=28, itemHeight=10, data=[dict(name=f"Guidance {g:g}", icon=guidance_icon(g)) for g in GUIDANCES])
    # Phone width: panel titles break onto two lines and the asks axis shows only its two ends.
    narrow = dict(title=[dict(text=label(row).replace(", ", ",\n", 1), textStyle=dict(fontSize=11), top=30) for row in rows],
                  grid=[dict(top=70) for _ in rows], xAxis=[dict(axisLabel=dict(interval=f"js:i => i === 0 || i === {len(asks) - 1}", fontSize=10)) for _ in rows])
    option = dict(title=titles, legend=legend, grid=grids_, xAxis=xs, yAxis=ys, series=series, tooltip=dict(trigger="axis"),
                  media=[dict(query=dict(maxWidth=560), option=narrow)])
    lo, hi = GUIDANCES[0], GUIDANCES[-1]
    ceiling = [row for row in rows if min(means[row]) >= CEILING_SHARE * per_ask]
    moving = [row for row in rows if row not in ceiling]
    who = lambda row: "for the control" if row == CONTROL else f"for {ROWS[row]}"
    move = lambda first, last: "fall" if last < first else "rise" if last > first else "stay"
    if moving and all(means[row][-1] < means[row][0] for row in moving):
        title = f"Within one ask, raising guidance from {lo} to {hi} cuts the mean line-ups " + spoken([f"from {means[r][0]} to {means[r][-1]} {who(r)}" for r in moving])
    elif moving:
        title = f"Within one ask, from guidance {lo} to {hi} the mean line-ups " + spoken(
            [f"{move(means[r][0], means[r][-1])} from {means[r][0]} to {means[r][-1]} {who(r)}" for r in moving])
    else:
        title = "Within one ask, every row stays at the ceiling"
    for row in ceiling:
        span = f"{min(means[row])}" if min(means[row]) == max(means[row]) else f"{min(means[row])} to {max(means[row])}"
        title += f". {ROWS[row]} stays at the ceiling, {span} of {per_ask} at every guidance"
    grid_moves = {move(totals[r][0], totals[r][-1]) for r in rows}
    told = spoken([f"{totals[r][0]:,} to {totals[r][-1]:,}" for r in rows])
    subtitle = (f"Over the whole grid the totals {grid_moves.pop() if len(grid_moves) == 1 else 'go'}: {told} line-ups, panel by panel, "
                f"because the asks share fewer line-ups as guidance rises. One panel for each row, all on the same scale. "
                f"Each ask has {per_ask} teams, so {per_ask} is the most one ask can hold. The darker and more solid the line, the higher the guidance.")
    return option, title, subtitle


def line_ups_shared_between_asks(summary):
    """The grid total is the ask-by-ask counts added up, minus the repeats between asks. One panel for each row, on one shared y scale;
    each bar is the ask-by-ask count, its two parts the grid total and the repeats."""
    rows = list(COLORS)
    counted, total, repeats = {}, {}, {}
    for row in rows:
        grids = grids_of(summary, row)
        counted[row] = [sum(g["line_ups_by_ask"].values()) for g in grids]; total[row] = [g["line_ups"] for g in grids]
        repeats[row] = [c - d for c, d in zip(counted[row], total[row])]
    top = -(-max(max(c) for c in counted.values()) // 500) * 500
    comma = "js:p => p.value.toLocaleString('en-US')"
    stripes = "".join(f"<rect x='{8 * i}' width='8' height='14' fill='{COLORS[row]}'/>" for i, row in enumerate(rows)).replace("#", "%23")
    width, gap, left = 28.0, 3.5, 8.0
    grids_, xs, ys, titles, series = [], [], [], [], []
    for i, row in enumerate(rows):
        x0 = left + i * (width + gap)
        grids_.append(dict(left=f"{x0}%", width=f"{width}%", top=74, bottom=78))
        titles.append(dict(text=label(row), left=f"{x0 + width / 2}%", top=40, textAlign="center", textStyle=dict(fontSize=13, fontWeight="normal")))
        xs.append(dict(type="category", gridIndex=i, axisTick=dict(show=False), axisLabel=dict(interval=0, lineHeight=16, fontSize=11.5),
                       data=[f"Guidance {g}\n{c:,} counted\nask by ask" for g, c in zip(GUIDANCES, counted[row])]))
        ys.append(dict(type="value", gridIndex=i, min=0, max=top, interval=1000, axisLabel=dict(show=i == 0),
                       name="Line-ups, each ask counted separately" if i == 0 else "", nameLocation="middle", nameGap=48))
        axes = dict(xAxisIndex=i, yAxisIndex=i)
        on_color = "#fefefe" if row == CONTROL else "#161615"     # neither is in DARK, so both stay put on the coloured bar in dark mode
        series.append(dict(name="Distinct line-ups over the grid", type="bar", stack=f"counted {i}", barWidth="56%", itemStyle=dict(color=COLORS[row]),
                           emphasis=dict(focus="series"), label=dict(show=True, formatter=comma, color=on_color, fontWeight="bold", fontSize=13),
                           data=total[row], **axes))
        series.append(dict(name="Repeats: a line-up already counted at another ask", type="bar", stack=f"counted {i}", itemStyle=dict(color=GREY, opacity=0.5),
                           emphasis=dict(focus="series"), label=dict(show=True, formatter=comma, color="#0b0b0b", fontSize=12, opacity=1), data=repeats[row], **axes))   # a label takes its bar's opacity unless set
    legend = dict(top=0, data=[dict(name="Distinct line-ups over the grid", icon=f"image://data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' "
                                    f"width='24' height='14'>{stripes}</svg>"),
                               dict(name="Repeats: a line-up already counted at another ask")])
    option = dict(title=titles, legend=legend, grid=grids_, xAxis=xs, yAxis=ys, series=series,
                  tooltip=dict(trigger="axis", axisPointer=dict(type="shadow"), valueFormatter="js:v => v.toLocaleString('en-US')"))
    cut = [repeats[row][0] - repeats[row][-1] for row in rows]; gain = [total[row][-1] - total[row][0] for row in rows]
    span = lambda v: f"{min(v):,}" if min(v) == max(v) else f"{min(v):,} to {max(v):,}"
    lo, hi = GUIDANCES[0], GUIDANCES[-1]
    if min(cut) > 0 and min(gain) > 0:
        title = f"In all three rows, raising guidance from {lo} to {hi} cuts the repeats between asks by {span(cut)}, so the grid total grows by {span(gain)} line-ups"
        if all(counted[row][-1] < counted[row][0] for row in rows):
            title += f" while the ask-by-ask count falls by {span([counted[row][0] - counted[row][-1] for row in rows])}"
    else:
        title = f"From guidance {lo} to {hi} the repeats between asks change by {span([-c for c in cut])} and the grid total by {span(gain)} line-ups"
    subtitle = ("One panel for each row, all on the same scale. The whole bar is the distinct line-ups of each ask, added over the 23 asks. "
                "The grey part is the line-ups already counted at another ask. The coloured part is what is left: the distinct line-ups over the whole grid.")
    return option, title, subtitle


TRAINING_PART, NOVEL_PART = "Training line-ups: the six species appear together in a training team", "Novel line-ups: they do not"


def training_and_novel_line_ups(summary):
    """Each grid's distinct line-ups split into training line-ups and novel line-ups. One panel for each row, on one shared y scale;
    each bar is the grid total, its two parts the training line-ups below and the novel line-ups above."""
    rows = list(COLORS)
    training = {row: [g["distinct_training_line_ups"] for g in grids_of(summary, row)] for row in rows}
    new = {row: [g["distinct_novel_line_ups_against_training"] for g in grids_of(summary, row)] for row in rows}
    total = {row: [t + n for t, n in zip(training[row], new[row])] for row in rows}
    top = -(-max(max(t) for t in total.values()) // 500) * 500
    comma = "js:p => p.value.toLocaleString('en-US')"
    stripes = "".join(f"<rect x='{8 * i}' width='8' height='14' fill='{COLORS[row]}'/>" for i, row in enumerate(rows)).replace("#", "%23")
    width, gap, left = 28.0, 3.5, 8.0
    grids_, xs, ys, titles, series = [], [], [], [], []
    for i, row in enumerate(rows):
        x0 = left + i * (width + gap)
        grids_.append(dict(left=f"{x0}%", width=f"{width}%", top=74, bottom=62))
        titles.append(dict(text=label(row), left=f"{x0 + width / 2}%", top=40, textAlign="center", textStyle=dict(fontSize=13, fontWeight="normal")))
        xs.append(dict(type="category", gridIndex=i, axisTick=dict(show=False), axisLabel=dict(interval=0, lineHeight=16, fontSize=11.5),
                       data=[f"Guidance {g}\n{t:,} in all" for g, t in zip(GUIDANCES, total[row])]))
        ys.append(dict(type="value", gridIndex=i, min=0, max=top, interval=1000, axisLabel=dict(show=i == 0),
                       name="Distinct line-ups among 4,416 teams" if i == 0 else "", nameLocation="middle", nameGap=48))
        axes = dict(xAxisIndex=i, yAxisIndex=i)
        on_color = "#fefefe" if row == CONTROL else "#161615"     # neither is in DARK, so both stay put on the coloured bar in dark mode
        series.append(dict(name=TRAINING_PART, type="bar", stack=f"grid {i}", barWidth="56%", itemStyle=dict(color=GREY, opacity=0.5),
                           emphasis=dict(focus="series"), label=dict(show=True, formatter=comma, color="#0b0b0b", fontSize=12, opacity=1),
                           data=training[row], **axes))
        series.append(dict(name=NOVEL_PART, type="bar", stack=f"grid {i}", itemStyle=dict(color=COLORS[row]),
                           emphasis=dict(focus="series"), label=dict(show=True, formatter=comma, color=on_color, fontWeight="bold", fontSize=13),
                           data=new[row], **axes))
    legend = dict(top=0, data=[dict(name=TRAINING_PART),
                               dict(name=NOVEL_PART, icon=f"image://data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='24' height='14'>{stripes}</svg>")])
    option = dict(title=titles, legend=legend, grid=grids_, xAxis=xs, yAxis=ys, series=series,
                  tooltip=dict(trigger="axis", axisPointer=dict(type="shadow"), valueFormatter="js:v => v.toLocaleString('en-US')"))
    lo, hi = GUIDANCES[0], GUIDANCES[-1]
    new_gain = [new[row][-1] - new[row][0] for row in rows]; total_gain = [total[row][-1] - total[row][0] for row in rows]
    training_move = max(abs(training[row][-1] - training[row][0]) for row in rows)
    span = lambda v: f"{min(v):,}" if min(v) == max(v) else f"{min(v):,} to {max(v):,}"
    pct = lambda v: f"{min(v):.0%}" if round(min(v), 2) == round(max(v), 2) else f"{min(v) * 100:.0f}% to {max(v):.0%}"
    if min(total_gain) > 0 and min(n / t for n, t in zip(new_gain, total_gain)) >= 0.9 and training_move < min(new_gain):
        title = (f"In all three rows, the line-ups that guidance {hi} adds over guidance {lo} are novel line-ups: they rise by {span(new_gain)} "
                 f"while training line-ups move by at most {training_move:,}")
    else:
        title = f"From guidance {lo} to {hi}, novel line-ups change by {span(new_gain)} and training line-ups by at most {training_move:,}"
    random_share = [n / t for n, t in zip(new[RANDOM], total[RANDOM])]
    if min(random_share) > 0.5:
        title += f". {label(RANDOM)} is {pct(random_share)} novel line-ups at every guidance"
    subtitle = ("One panel for each row, all on the same scale. Each bar is the grid's distinct line-ups. The grey part is training line-ups, "
                "the coloured part novel line-ups.")
    return option, title, subtitle


def main(summary_path=SUMMARY, out=OUT):
    summary = json.load(open(summary_path))
    note = f"{ERROR} {WIN_RATE} Source: {Path(summary_path).name}."
    pages = []
    option, title = error_against_line_ups(summary)
    pages.append(write_chart(option, out / "05_error_against_line_ups.html", title, height=560, note=note + " Each point pools 3 sampling seeds, 4,416 teams. Bars are 95% bootstrap intervals.",
                             subtitle="Each row is one curve through guidance 2, 4 and 8. A row is better when its curve sits lower and further right "
                                      "than the control's. Line-ups are distinct sets of six species."))
    option, title = got_against_ask(summary)
    pages.append(write_chart(option, out / "06_got_against_ask.html", title, height=440, note=note + " Each point is one ask: 192 teams over 3 sampling seeds.",
                             subtitle="Mean win rate got at each ask, against the dashed line where got equals ask. Linear, random order is left out."))
    option, title = random_order_error_multiple(summary)
    pages.append(write_chart(
        option, out / "07_random_order_error_multiple.html", title, height=460,
        subtitle="Typical miss: how far a team's win rate usually lands from the win rate it was asked for, the square root of the error. "
                 "Shorter bars are better. Each pair is one model and guidance, so the two orders share everything but the order.",
        note=f"The faded pair is the earlier measurement, errors {PER_FIELD_MULTIPLE['species_first']} and {PER_FIELD_MULTIPLE['random']}, on the all-teams "
             "per-field model after 21 epochs, 2026-11-07. That model differs from the linear model in its training table and its epochs as well as in its "
             f"masking schedule, so it is not a before and after. {ERROR} Source: {Path(summary_path).name}."))
    option, title, subtitle = line_ups_by_ask(summary)
    pages.append(write_chart(option, out / "08_line_ups_by_ask.html", title, subtitle=subtitle, height=460,
                             note="A line-up is a distinct set of six species. Each point pools 3 sampling seeds of 64 teams. A row is one model decoded in one "
                                  "order: the per-field masking schedule or the linear masking schedule, filled species-first or in random order. Uncorrected t is the "
                                  "sampler's usual setting: on each pass it tells the model how many fields are still hidden. "
                                  f"Source: {Path(summary_path).name}."))
    option, title, subtitle = line_ups_shared_between_asks(summary)
    pages.append(write_chart(option, out / "09_line_ups_shared_between_asks.html", title, subtitle=subtitle, height=500,
                             note="A line-up is a distinct set of six species. Each ask has 192 teams, 64 from each of 3 sampling seeds, so one ask holds at most 192 "
                                  "line-ups. A row is one model decoded in one order: the per-field masking schedule or the linear masking schedule, filled species-first "
                                  f"or in random order. Uncorrected t is the sampler's usual setting: on each pass it tells the model how many fields are still hidden. "
                                  f"Source: {Path(summary_path).name}."))
    option, title, subtitle = training_and_novel_line_ups(summary)
    pages.append(write_chart(option, out / "10_training_and_novel_line_ups.html", title, subtitle=subtitle, height=500,
                             note="A line-up is a distinct set of six species. A training line-up is the line-up of one of the labelled training teams "
                                  "of the shuffled-only table, validation teams left out; every other line-up is novel. Each grid has 4,416 teams: 23 asks, "
                                  "3 sampling seeds, 64 teams each. A row is one model decoded in one order: the per-field masking schedule or the linear masking "
                                  "schedule, filled species-first or in random order. Uncorrected t is the sampler's usual setting: on each pass it tells the model "
                                  f"how many fields are still hidden. Source: {Path(summary_path).name}."))
    return pages


if __name__ == "__main__":
    for page in main(): print(page)
