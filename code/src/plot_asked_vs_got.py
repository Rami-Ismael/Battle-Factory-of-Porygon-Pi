"""Asked vs got figure for experiment A -> results/asked_vs_got.html, an ECharts page that follows light and dark mode."""
import json

import numpy as np

from echarts_chart import band, write_chart

RES = "/Users/ramiismael/Documents/code/vgc-team-generator-pilot/results/"
d = json.load(open(RES + "asked_vs_got.json"))
S = d["summary"]
REAL = 0.531          # top-50 meta teams vs the other 49, same matrix, same policy
UNC = S["uncond"]["got"]
TRAIN_MAX = d["training"]["max"]
RAMP = {1.0: "#86b6ef", 2.0: "#2a78d6", 4.0: "#104281"}
DARK_RAMP = {"#86b6ef": "#3f6fa6", "#2a78d6": "#3987e5", "#104281": "#b7d3f6"}  # the old dark PNG's ramp
INK2 = "#5f5e5a"
TICKS = [round(0.1 * i, 1) for i in range(10)]
GRIDS = [{"left": "6%", "width": "52%", "top": 60, "bottom": 56}, {"left": "66%", "width": "31%", "top": 60, "bottom": 56}]


def hline(y, text, g, position):
    return {"type": "line", "data": [], "silent": True, "xAxisIndex": g, "yAxisIndex": g, "markLine": {
        "symbol": "none", "lineStyle": {"type": "dotted", "color": INK2}, "data": [{"yAxis": y}],
        "label": {"formatter": text, "position": position, "color": INK2}}}


def shade(g, label):
    return {"type": "line", "data": [], "silent": True, "xAxisIndex": g, "yAxisIndex": g, "markArea": {
        "itemStyle": {"color": "rgba(127,127,127,0.10)"}, "data": [[{"xAxis": TRAIN_MAX}, {"xAxis": 0.95}]],
        "label": {"show": bool(label), "formatter": label, "position": "insideTopLeft", "color": INK2, "fontSize": 11}}}


series = [shade(0, "above every\ntraining label"), shade(1, ""),
          {"type": "line", "name": "got = asked", "silent": True, "symbol": "none", "tooltip": {"show": False},
           "lineStyle": {"type": "dashed", "color": INK2, "width": 1}, "data": [[0, 0], [0.9, 0.9]],
           "endLabel": {"show": True, "formatter": "got = asked", "color": INK2}},
          hline(REAL, f"top-50 meta team, {REAL:.2f}", 0, "insideStartTop"),
          hline(UNC, f"no win-rate condition, {UNC:.2f}", 0, "insideEndBottom")]
for g in (1.0, 2.0, 4.0):
    cs = sorted([s for s in S.values() if s["asked"] is not None and s["guidance"] == g], key=lambda s: s["asked"])
    x = [s["asked"] for s in cs]; y = np.array([s["got"] for s in cs]); e = np.array([1.96 * s["se"] for s in cs])
    name, c, top = f"guidance {g:g}", RAMP[g], int(np.argmax(y))
    series += band(name, c, x, y - e, y + e, name, 0.18)
    series.append({"type": "line", "name": name, "color": c, "lineStyle": {"width": 2}, "symbolSize": 7, "emphasis": {"focus": "series"},
                   "data": [{"value": [a, b], **({"label": {"show": True, "position": "top", "formatter": f"{b:.2f}"}} if g == 4.0 and i == top else {})}
                            for i, (a, b) in enumerate(zip(x, y.tolist()))]})
    series.append({"type": "line", "name": name, "color": c, "lineStyle": {"width": 2}, "symbolSize": 7, "emphasis": {"focus": "series"},
                   "xAxisIndex": 1, "yAxisIndex": 1, "data": [[a, s["species_sets"]] for a, s in zip(x, cs)]})

titles = [{"text": t, "left": gr["left"], "top": 16, "textStyle": {"fontSize": 13}} for t, gr in zip(
    ("Asking for a higher win rate works up to ~0.5, then stops", "…and the teams it gets there with are the same few"), GRIDS)]
x_axis = {"type": "value", "min": -0.05, "max": 0.95, "name": "win rate asked for", "nameLocation": "middle", "nameGap": 28,
          "axisLabel": {"customValues": TICKS}, "axisTick": {"customValues": TICKS}, "splitLine": {"show": False}}
write_chart({"title": titles, "grid": GRIDS, "tooltip": {"trigger": "axis"},
             "legend": {"data": ["guidance 1", "guidance 2", "guidance 4"], "left": "7%", "top": 44, "orient": "vertical"},
             "xAxis": [{**x_axis, "gridIndex": 0}, {**x_axis, "gridIndex": 1}],
             "yAxis": [{"type": "value", "gridIndex": 0, "min": 0, "max": 0.9, "name": "win rate got (49 battles per team, 64 teams per point)",
                        "nameLocation": "middle", "nameGap": 40},
                       {"type": "value", "gridIndex": 1, "min": 0, "max": 68, "name": "different species sets among 64 teams",
                        "nameLocation": "middle", "nameGap": 36}],
             "series": series},
            RES + "asked_vs_got.html", "Asked vs got, experiment A",
            note="Win rate = wins vs the top-50 meta teams (one battle per opponent), VGC-Bench behaviour-cloning policy both sides. "
                 "Bands: 95% intervals over teams.", height=480, dark=DARK_RAMP)
print("ok")
