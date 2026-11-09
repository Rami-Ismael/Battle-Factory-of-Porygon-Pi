"""Random-fill baseline figure -> results/random_fill_baseline.html (ECharts page)."""
import json

from echarts_chart import band, write_chart

RES = "/Users/ramiismael/Documents/code/vgc-team-generator-pilot/results/"
S = json.load(open(RES + "random_fill_baseline.json"))["summary"]
ORIG = 0.450
C = {"pokemon": "#2a78d6", "item": "#eb6834", "stats": "#1baf7a", "alignment": "#eda100", "ability": "#e87ba4",
     "move": "#2a78d6", "mixed": "#eb6834"}
NAME = {"pokemon": "whole Pokémon", "item": "items", "stats": "Stat Points", "alignment": "Stat Alignment",
        "ability": "abilities", "move": "moves", "mixed": "mixed parts"}
PANELS = ((["pokemon", "item", "stats", "alignment", "ability"], 6, "Random Stat Points, abilities and alignments barely hurt"),
          (["move", "mixed"], 40, "Random moves and mixed parts cost steadily"))
GRIDS = [{"left": "6%", "width": "33%", "top": 60, "bottom": 60}, {"left": "47%", "width": "45%", "top": 60, "bottom": 60}]

series, xs, ys, titles = [], [], [], []
for g, (tasks, kmax, title) in enumerate(PANELS):
    axes = {"xAxisIndex": g, "yAxisIndex": g}
    ticks = list(range(1, 7)) if kmax == 6 else [1, *range(5, 41, 5)]
    titles.append({"text": title, "left": GRIDS[g]["left"], "top": 16, "textStyle": {"fontSize": 13}})
    xs.append({"type": "value", "gridIndex": g, "min": 0.7, "max": kmax + (1.6 if kmax == 6 else 7), "name": "k = parts left blank and filled at random",
               "nameLocation": "middle", "nameGap": 28, "axisLabel": {"customValues": ticks}, "axisTick": {"customValues": ticks},
               "splitLine": {"show": False}})
    ys.append({"type": "value", "gridIndex": g, "min": 0, "max": 0.55, "name": "win rate vs the top-50 meta teams" if g == 0 else "",
               "nameLocation": "middle", "nameGap": 40})
    series.append({"type": "line", "data": [], "silent": True, **axes, "markLine": {
        "symbol": "none", "lineStyle": {"type": "dotted", "color": "#5f5e5a"}, "data": [{"yAxis": ORIG}],
        "label": {"formatter": f"original teams, {ORIG:.2f}", "position": "insideStartTop" if kmax == 6 else "insideEndTop"}}})
    for t in tasks:
        ks = sorted(int(k[len(t):]) for k in S if k.startswith(t) and k[len(t):].isdigit())
        y = [S[f"{t}{k}"]["filled"] for k in ks]
        series += band(NAME[t], C[t], ks, [ORIG + S[f"{t}{k}"]["ci"][0] for k in ks], [ORIG + S[f"{t}{k}"]["ci"][1] for k in ks], t, 0.15, **axes)
        series.append({"type": "line", "name": NAME[t], "color": C[t], "lineStyle": {"width": 2}, "symbolSize": 7 if kmax == 6 else 0,
                       "emphasis": {"focus": "series"}, "endLabel": {"show": True, "formatter": NAME[t], "color": "inherit"},
                       "data": [[k, v] for k, v in zip(ks, y)], **axes})

write_chart({"title": titles, "grid": GRIDS, "xAxis": xs, "yAxis": ys, "series": series, "tooltip": {"trigger": "axis"}},
            RES + "random_fill_baseline.html", "Random-fill baseline: what filling parts at random costs a real team",
            note="50 real VGCPastes teams outside the top 50; one random legal fill per team per k; 49 battles per team "
                 "(one per top-50 opponent), behaviour-cloning policy both sides. Bands: 95% intervals over teams.", height=460)
print("ok")
