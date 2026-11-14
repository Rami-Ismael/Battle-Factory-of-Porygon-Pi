"""Build chart.html, the quality-versus-exploration chart that index.html embeds.

Reads data.json beside this file (the #297 grids: distinct line-ups and elite line-ups that held, per row and
guidance) and writes the page with write_chart from code/src/echarts_chart.py.

    python3 notes/Blog/visuals/quality-exploration/build.py
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[3] / "code/src"))
from echarts_chart import write_chart

data = json.loads((HERE / "data.json").read_text())
series = [{
    "name": row["name"], "type": "line", "color": row["color"], "symbol": "circle", "showSymbol": True, "symbolSize": 10, "lineStyle": {"width": 2},
    "emphasis": {"focus": "series"},
    "label": {"show": True, "position": "top", "formatter": "js:p => 'guidance ' + p.data.guidance"},
    "data": [{"value": [p["line_ups"], p["held"]], "guidance": p["guidance"]} for p in row["points"]],
} for row in data["rows"]]

option = {
    "legend": {"top": 0},
    "grid": {"left": 64, "right": 24, "top": 48, "bottom": 56},
    "tooltip": {"trigger": "axis", "axisPointer": {"type": "cross"},
                "formatter": "js:ps => ps.map(p => p.marker + p.seriesName + ', guidance ' + p.data.guidance + ': '"
                             " + p.value[0].toLocaleString() + ' line-ups, ' + p.value[1] + ' held').join('<br>')"},
    "xAxis": {"type": "value", "min": 1200, "name": "Distinct line-ups over the grid's 4,416 teams (exploration)",
              "nameLocation": "middle", "nameGap": 32},
    "yAxis": {"type": "value", "name": "Elite line-ups that held in the re-battle (quality)", "nameLocation": "middle",
              "nameGap": 44},
    "series": series,
}

write_chart(
    option, HERE / "chart.html",
    title="Random order draws more line-ups and far fewer that hold",
    subtitle="Each line is one row of the masking-schedule grids; its three points are guidance 2, 4 and 8.",
    note="Elite: a team's win rate against the top-50 meta teams is at least 0.531. Held: its line-up still reaches 0.531 "
         "in a fresh re-battle of 4 games per opponent. Source: " + data["source"] + ".",
    height=440, dark={row["color"]: row["dark"] for row in data["rows"]})
print(HERE / "chart.html")
