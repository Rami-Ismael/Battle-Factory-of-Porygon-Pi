"""Chart of the Foul Play for doubles head-to-head matchups (code/docs/foul-play-doubles.md).

Reads summary.json and battles.jsonl from each matchup folder and writes one ECharts page: each
matchup's share of points for the first-named policy, with its 95% interval over team pairs, against
the 0.5 line where the two policies are even.
Run: python src/foul_play_doubles_report.py ~/vgc-data/foul-play-doubles/2026-11-23 reports/foul-play-doubles/index.html
"""
import json
import sys
from pathlib import Path

from echarts_chart import write_chart

LABEL = {"foulplay": "Foul Play for doubles", "bc": "behaviour cloning", "montecarlo": "flat Monte Carlo",
         "random": "random play"}
ORDER = ["foulplay-4x-time-vs-bc", "foulplay-one-deep-tree-vs-bc", "foulplay-vs-bc", "foulplay-4x-vs-montecarlo-4x",
         "foulplay-vs-montecarlo", "montecarlo-vs-bc", "foulplay-vs-random"]


def load(root):
    rows = []
    for name in ORDER:
        path = Path(root) / name / "summary.json"
        if not path.exists():
            continue
        s = json.loads(path.read_text())
        battles = [json.loads(line) for line in (Path(root) / name / "battles.jsonl").read_text().splitlines() if line]
        x_side = [b["agents"].get(b["x_side"], {}) for b in battles]
        decisions = sum(a.get("decisions", 0) for a in x_side)
        c = s["config"]
        # Difference first: the search budget, then the opponent; the one row not about Foul Play says so.
        budget = f"{c['samples']} sample{'s' if c['samples'] > 1 else ''} × {c['ms'] / 1000:g} s"
        label = f"{budget} vs {LABEL[s['y']]}"
        if s["x"] != "foulplay":
            label = f"{LABEL[s['x']]} at {budget} vs {LABEL[s['y']]}"

        rows.append({**s, "name": name, "label": label,
                     "iterations_per_decision": sum(a.get("iterations", 0) for a in x_side) / decisions if decisions else 0,
                     "fallbacks": sum(a.get("fallbacks", 0) for a in x_side)})
    return rows


def option(rows):
    labels = [r["label"] for r in rows][::-1]
    data = [[round(r["paired_mean"], 3), i] for i, r in enumerate(rows[::-1])]
    bars = [[r["paired95"][0], r["paired95"][1], i] for i, r in enumerate(rows[::-1])]
    return {
        "grid": {"left": 330, "right": 40, "top": 30, "bottom": 50},
        "tooltip": {"trigger": "item"},
        "xAxis": {"type": "value", "min": 0, "max": 1, "name": "share of points (win 1, tie 0.5)",
                  "nameLocation": "middle", "nameGap": 30},
        "yAxis": {"type": "category", "data": labels},
        "series": [
            {"type": "custom", "name": "95% interval over team pairs", "data": bars,
             "renderItem": "__interval__", "encode": {"x": [0, 1], "y": 2}, "z": 1},
            {"type": "scatter", "name": "share of points", "data": data, "symbolSize": 12, "z": 3,
             "label": {"show": True, "position": "right", "formatter": "{@[0]}"},
             "emphasis": {"focus": "series"},
             "markLine": {"silent": True, "symbol": "none", "data": [{"xAxis": 0.5}],
                          "label": {"formatter": "even"}, "lineStyle": {"type": "dashed"}}},
        ],
    }


def main():
    root, out = sys.argv[1], Path(sys.argv[2])
    rows = load(root)
    opt = option(rows)
    note = "; ".join(f"{r['label']}: {r['battles']} battles, {r['x_points']:g} points, "
                     f"{r['iterations_per_decision']:.0f} search iterations per decision" for r in rows)
    path = write_chart(opt, out, "Foul Play for doubles, head to head",
                       "Foul Play's share of points per matchup, by search budget per decision (sampled battles × search "
                       "time each). 100 pairs of top-50 meta teams, each played twice with the policies swapped; 0.5 is even.",
                       note, height=440)
    # ECharts custom series need a function; the page writer serialises JSON, so splice one in.
    text = path.read_text().replace('"__interval__"', """function (params, api) {
      var lo = api.coord([api.value(0), api.value(2)]), hi = api.coord([api.value(1), api.value(2)]);
      return {type: 'line', shape: {x1: lo[0], y1: lo[1], x2: hi[0], y2: hi[1]},
              style: {stroke: api.visual('color'), lineWidth: 3, opacity: 0.6}};
    }""")
    path.write_text(text)
    (out.parent / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(path)


if __name__ == "__main__":
    main()
