"""Storyboard figures for issue #100 and #93, one idea each, from least_likely_trade_storyboard.json. No new battles.

    python trade84_storyboard_figures.py      # writes six ECharts pages to ~/vgc-data/least_likely_trade/storyboard
"""
import json
import math
from pathlib import Path

from echarts_chart import write_chart

D = json.loads((Path.home() / "vgc-data/results/least_likely_trade_storyboard.json").read_text())
A = D["arms"]
OUT = Path.home() / "vgc-data/least_likely_trade/storyboard"
PLAIN, GREY, PALE = "#6f6e69", "#a9a8a2", "rgba(127,127,127,0.18)"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"      # validated all-pairs on the light surface
FAMILY = {"plain": PLAIN, "spend66": GREY, "remask_conf_loop": GREY, "remask_even_loop": GREY}
for _a in A:
    FAMILY.setdefault(_a, ORANGE if _a.startswith("remask") else BLUE if _a.startswith("nucleus") else AQUA)
FAMILY_NAME = {ORANGE: "Least-likely remasking", AQUA: "Temperature", BLUE: "Nucleus", GREY: "Other: Stat Points, paper rules"}
WIN = "Win rate against the top-50 meta teams, behaviour-cloning policy on both sides."
PCT = "js:v => Math.round(v * 100) + '%'"
RULES = (("plain", PLAIN, "No remasking"), ("remask_ll_10", ORANGE, "Least-likely 10% remasked"), ("cold_sets_t03", AQUA, "Temperature 0.3 on sets only"))


def axis(name, **kw):
    return {"type": "value", "name": name, "nameLocation": "middle", "nameGap": 30, "scale": True, **kw}


def dot(name, color, points, size=12, labels=None, **kw):
    """A scatter series; labels (one string per point, or None) are drawn beside the points."""
    data = [{"value": list(p), **({"label": {"show": True, "formatter": t}} if t else {})}
            for p, t in zip(points, labels or [None] * len(points))]
    return {"type": "scatter", "name": name, "color": color, "symbolSize": size, "data": data,
            "label": {"color": "inherit", "fontSize": 12}, "emphasis": {"focus": "series"}, **kw}


def segment(name, color, p, q, grid=0, **kw):
    """A two-point line: an interval, or one leg of a path."""
    return {"type": "line", "name": name, "color": color, "symbol": "none", "silent": True, "data": [list(p), list(q)],
            "lineStyle": {"width": 1.6}, "xAxisIndex": grid, "yAxisIndex": grid, "tooltip": {"show": False}, **kw}


def arrow(p, q):
    return {"symbol": ["none", "arrow"], "symbolSize": 11, "silent": True, "label": {"show": False},
            "lineStyle": {"width": 2.2, "type": "solid", "opacity": 0.9}, "data": [[{"coord": list(p)}, {"coord": list(q)}]]}


def rule(x, label, dashed=False):
    """A vertical reference line at x, as a markLine on an empty series."""
    return {"type": "line", "data": [], "silent": True, "markLine": {
        "symbol": "none", "lineStyle": {"color": GREY if dashed else PLAIN, "type": "dashed" if dashed else "solid"},
        "label": {"show": bool(label), "formatter": label, "position": "insideEndTop", "color": GREY}, "data": [{"xAxis": x}]}}


def grids(n, left=8, right=2, top=14, bottom=12, gap=4, rows=1, vgap=12, widths=None):
    """n x rows grids filling the chart, in percent."""
    widths = widths or [1] * n
    unit = (100 - left - right - gap * (n - 1)) / sum(widths)
    height = (100 - top - bottom - vgap * (rows - 1)) / rows
    out, x = [], left
    starts = []
    for w in widths:
        starts.append(x); x += w * unit + gap
    for r in range(rows):
        out += [{"left": f"{starts[j]:.1f}%", "width": f"{widths[j] * unit:.1f}%", "top": f"{top + r * (height + vgap):.1f}%",
                 "height": f"{height:.1f}%"} for j in range(n)]
    return out


def save(option, name, height, head, sub, foot):
    print(write_chart({"tooltip": {"trigger": "item"}, **option}, OUT / f"{name}.html", head, sub, WIN + " " + foot, height=height))


def teaser():
    P = {a: A[a]["pooled"]["all"] for a in A}
    p0 = (P["plain"]["line_ups"], P["plain"]["win"])
    series = [dot("No remasking", PLAIN, [p0], 16, ["No remasking\n438 line-ups, 2,336 Pokémon sets"], label={"position": "left"})]
    for a, col in (("remask_ll_10", ORANGE), ("cold_sets_t03", AQUA)):
        p = P[a]; kept = p["distinct_sets"] / P["plain"]["distinct_sets"]
        text = f"{A[a]['name']}\n{p['gain']:+.3f} win rate · {p['line_ups']} line-ups · {kept:.0%} of the sets"
        series.append(dot(A[a]["name"], col, [(p["line_ups"], p["win"])], 16, [text], label={"position": "left", "color": "inherit"},
                          markLine=arrow(p0, (p["line_ups"], p["win"]))))
    save({"grid": {"left": 64, "right": 24, "top": 16, "bottom": 52},
          "xAxis": axis("Distinct line-ups among 768 teams (asks 4/24, 8/24, 12/24, 16/24)", min=150, max=470),
          "yAxis": axis("Mean win rate", min=0.34, max=0.44, nameGap=44), "series": series},
         "01_teaser", 480, "Set-only sharpening buys most of least-likely's gain and keeps every line-up",
         "Both routes pay about half the Pokémon sets. Least-likely also pays half the line-ups.",
         "768 teams per rule, 49 battles each. Exploratory.")


def moves_per_ask():
    asks = ["4", "8", "16", "12"]
    keys = (("line_ups", "Distinct line-ups"), ("distinct_sets", "Distinct Pokémon sets"))
    gs = grids(4, left=7, top=12, bottom=10, rows=2, vgap=14)
    xs, ys, series, titles = [], [], [], []
    for i, (key, lab) in enumerate(keys):
        for j, k in enumerate(asks):
            g = i * 4 + j
            q = {a: A[a]["per_ask"][k] for a, _, _ in RULES}
            lo = min(min(v["win"], *v.get("win_ci", [v["win"]])) for v in q.values())
            hi = max(max(v["win"], *v.get("win_ci", [v["win"]])) for v in q.values())
            if k == "12":
                gs[g].update(show=True, backgroundColor=PALE, borderWidth=0)
            xs.append(axis(lab + " among 192 teams", gridIndex=g, min=0, scale=False, nameGap=24, splitLine={"show": False}))
            ys.append(axis("Mean win rate" if j == 0 else "", gridIndex=g, min=math.floor(lo * 50) / 50, max=math.ceil(hi * 50) / 50,
                           interval=0.02, nameGap=42))
            p0 = (q["plain"][key], q["plain"]["win"])
            series.append(dot("No remasking", PLAIN, [p0], xAxisIndex=g, yAxisIndex=g))
            for a, col, name in RULES[1:]:
                p = (q[a][key], q[a]["win"])
                series.append(dot(name, col, [p], xAxisIndex=g, yAxisIndex=g, markLine=arrow(p0, p)))
                series.append(segment(name, col, (p[0], q[a]["win_ci"][0]), (p[0], q[a]["win_ci"][1]), g))
            if i == 0:
                titles.append({"text": f"Ask {k}/24" + ("  (another team population)" if k == "12" else ""), "left": gs[g]["left"],
                               "top": "6%",
                               "textStyle": {"fontSize": 13, "width": 200, "overflow": "break"}})
    save({"title": titles, "legend": {"data": [r[2] for r in RULES], "bottom": 0}, "grid": gs, "xAxis": xs, "yAxis": ys, "series": series},
         "02_moves_per_ask", 680, "Least-likely pays in line-ups and sets; set-only sharpening pays only in sets",
         "Top row counts line-ups, bottom row counts Pokémon sets. The aqua arrow is vertical on top: no line-up lost at any ask.",
         "Each point is 192 teams from 3 sampling seeds; bars are 95% intervals over teams. Ask 12/24 sits apart because it selects a different team population.")


def nucleus_vs_temperature():
    gs = grids(2, left=7, top=12, bottom=16, gap=6)
    paths = (("Temperature on sets only: 1, 0.6, 0.3", ["plain", "cold_sets_t06", "cold_sets_t03"], AQUA, ["0.6", "0.3"]),
             ("Nucleus on sets only: 0.9, 0.7, 0.5", ["plain", "nucleus_sets_p09", "nucleus_sets_p07", "nucleus_sets_p05"], BLUE, ["0.9", "0.7", "0.5"]))
    xs, ys, series, titles = [], [], [], []
    for g, (tag, lab) in enumerate((("all", "all four asks, 768 teams"), ("without_12", "without ask 12/24, 576 teams"))):
        Q = {a: A[a]["pooled"][tag] for a in A}
        sets = [Q[a]["distinct_sets"] for a in ("plain", "remask_ll_10", "cold_sets_t03", "nucleus_sets_p05")]
        xs.append(axis("Distinct Pokémon sets (species, ability, item, nature, moves)", gridIndex=g, inverse=True,
                       min=round(min(sets) * .86), max=round(max(sets) * 1.04), scale=False,
                       axisLabel={"showMinLabel": False, "showMaxLabel": False}))
        ys.append(axis("Mean win rate" if g == 0 else "", gridIndex=g, nameGap=44))
        for name, path, col, txt in paths:
            pts = [(Q[a]["distinct_sets"], Q[a]["win"]) for a in path]
            series.append({"type": "line", "name": name, "color": col, "xAxisIndex": g, "yAxisIndex": g, "symbolSize": 10,
                           "lineStyle": {"width": 2}, "emphasis": {"focus": "series"},
                           "data": [{"value": list(pts[0]), "symbolSize": 0}] +
                                   [{"value": list(p), "label": {"show": True, "formatter": t, "position": "bottom" if col == AQUA else "right"}}
                                    for p, t in zip(pts[1:], txt)]})
        series.append(dot("No remasking", PLAIN, [(Q["plain"]["distinct_sets"], Q["plain"]["win"])], 14, ["No remasking"],
                          xAxisIndex=g, yAxisIndex=g, label={"position": "right"}))
        ll = Q["remask_ll_10"]
        series.append(dot("Least-likely 10% remasked", ORANGE, [(ll["distinct_sets"], ll["win"])], 13, ["Least-likely 10%\n(also loses line-ups)"],
                          xAxisIndex=g, yAxisIndex=g, symbol="rect", label={"position": "top"}))
        titles.append({"text": lab.capitalize(), "left": gs[g]["left"], "top": "4%", "textStyle": {"fontSize": 13}})
    save({"title": titles, "legend": {"data": [p[0] for p in paths], "bottom": 0}, "grid": gs, "xAxis": xs, "yAxis": ys, "series": series},
         "03_nucleus_vs_temperature", 520, "Cutting unlikely set options costs the same sets per win as cooling them",
         "Nucleus 0.5 lands on temperature 0.3: -0.001 win rate (-0.010 to +0.009), 1,223 sets against 1,252. Fewer sets to the right.",
         "Species untouched in both, so every point keeps all line-ups. 95% intervals resample teams within each cell.")


def family_dots(order, value, hollow=False):
    """One scatter series per family, so the legend names families; y is the arm's row."""
    out = []
    for col, name in FAMILY_NAME.items():
        pts = [(value(a), i) for i, a in enumerate(order) if FAMILY[a] == col]
        style = {"itemStyle": {"color": "transparent", "borderColor": col, "borderWidth": 1.6}} if hollow else {}
        if pts:
            out.append(dot(name, col, pts, 9 if hollow else 12, **style))
    return out


def menu():
    order = sorted([a for a in A if a != "plain"], key=lambda a: A[a]["pooled"]["all"]["gain"])
    base = A["plain"]["pooled"]["all"]
    gs = grids(3, left=27, top=6, bottom=14, gap=3, widths=[1.5, 1, 1])
    names = [A[a]["name"] for a in order]
    ys = [{"type": "category", "gridIndex": g, "data": names, "axisLabel": {"show": g == 0, "width": 250, "overflow": "break"},
           "axisTick": {"show": False}} for g in range(3)]
    xs = [axis("Win-rate gain over no remasking (filled: all four asks; hollow: without 12/24)", gridIndex=0),
          axis("Line-ups kept", gridIndex=1, min=0, max=1.18, scale=False, axisLabel={"formatter": PCT}),
          axis("Pokémon sets kept", gridIndex=2, min=0, max=1.18, scale=False, axisLabel={"formatter": PCT})]
    series = [segment(FAMILY_NAME.get(FAMILY[a]), FAMILY[a], (A[a]["pooled"]["all"]["gain_ci"][0], i), (A[a]["pooled"]["all"]["gain_ci"][1], i))
              for i, a in enumerate(order)]
    series += family_dots(order, lambda a: A[a]["pooled"]["all"]["gain"])
    series += family_dots(order, lambda a: A[a]["pooled"]["without_12"]["gain"], hollow=True)
    series += [rule(0, ""), rule(.03, "smallest effect worth acting on", dashed=True)]
    for g, key in ((1, "line_ups"), (2, "distinct_sets")):
        series.append({"type": "bar", "name": xs[g]["name"], "xAxisIndex": g, "yAxisIndex": g, "barWidth": "56%",
                       "data": [{"value": p[key] / base[key], "itemStyle": {"color": FAMILY[a], "borderRadius": [0, 4, 4, 0]},
                                 "label": {"show": True, "position": "right", "formatter": f"{p[key] / base[key]:.0%}"}}
                                for a in order for p in [A[a]["pooled"]["all"]]]})
    save({"legend": {"data": list(FAMILY_NAME.values()), "bottom": 0}, "grid": gs, "xAxis": xs, "yAxis": ys, "series": series},
         "04_menu", 620, "Set-only sharpening alone gains over 0.05 without losing a line-up",
         "Temperature 0.3 and nucleus 0.5 on sets only tie. Higher gains cost line-ups; every gain costs Pokémon sets.",
         "Kept shares are against no remasking's 438 line-ups and 2,336 sets over 768 teams. Lines through the dots are 95% intervals over teams.")


def rebattle():
    R = D["rebattle"]
    gs = grids(2, left=20, top=14, bottom=18, gap=4)
    titles, xs, ys, series = [], [], [], []
    for g, (h, c, lab) in enumerate((("held", "candidates", "All four asks"), ("held_without_12", "candidates_without_12", "Without ask 12/24"))):
        titles.append({"text": lab, "left": gs[g]["left"], "top": "4%", "textStyle": {"fontSize": 13}})
        xs.append(axis("Line-ups at the 0.531 elite cut, best team re-battled 196 times", gridIndex=g, min=0, max=82, scale=False))
        ys.append({"type": "category", "inverse": True, "gridIndex": g, "data": [r[2] for r in RULES], "axisLabel": {"show": g == 0}})
        series.append({"type": "bar", "name": "Line-ups at the cut", "xAxisIndex": g, "yAxisIndex": g, "barWidth": "55%", "color": PALE, "silent": True,
                       "data": [{"value": R[a][c], "label": {"show": True, "position": "right", "formatter": f"{R[a][h]} of {R[a][c]} hold"}} for a, _, _ in RULES]})
        series.append({"type": "bar", "name": "Line-ups that hold", "xAxisIndex": g, "yAxisIndex": g, "barWidth": "55%", "barGap": "-100%",
                       "data": [{"value": R[a][h], "itemStyle": {"color": col}} for a, col, _ in RULES]})
    save({"title": titles, "grid": gs, "xAxis": xs, "yAxis": ys, "series": series},
         "05_rebattle", 380, "The strong line-ups of temperature 0.3 on sets only hold on fresh games as often as the others'",
         "27 of 68 hold, against 18 of 52 and 18 of 43. Without ask 12/24, almost none hold for any rule.",
         "Fresh games: 4 per opponent, private database. 34 of temperature 0.3's 68 sit within one standard error (0.036) of the cut.")


def ask_versus_got_error():
    order = sorted([a for a in A if a != "plain"], key=lambda a: A[a]["error"]["all"]["diff_vs_plain"])
    series = [segment(FAMILY_NAME.get(FAMILY[a]), FAMILY[a], (A[a]["error"]["all"]["diff_vs_plain_ci"][0], i),
                      (A[a]["error"]["all"]["diff_vs_plain_ci"][1], i)) for i, a in enumerate(order)]
    series += family_dots(order, lambda a: A[a]["error"]["all"]["diff_vs_plain"])
    series += [rule(0, ""), rule(-.001, "smallest effect, 0.001 closer", dashed=True)]
    save({"legend": {"data": list(FAMILY_NAME.values()), "bottom": 0},
          "grid": {"left": 230, "right": 24, "top": 16, "bottom": 72},
          "xAxis": axis("Ask-versus-got error minus no remasking's 0.0197 (left is closer to got = ask)"),
          "yAxis": {"type": "category", "data": [A[a]["name"] for a in order], "axisLabel": {"width": 210, "overflow": "break"}},
          "series": series},
         "06_ask_versus_got_error", 560, "No remasking rule or control brings got closer to the ask",
         "Every interval crosses zero except least-likely 20%, which moves got further away. Every arm sits near 0.02; near the line is under 0.001.",
         "Error: mean squared gap between ask and mean got on asks 4/24, 8/24, 12/24, 16/24 only, not the full 1/24 to 17/24.")


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    teaser(); moves_per_ask(); nucleus_vs_temperature(); menu(); rebattle(); ask_versus_got_error()
