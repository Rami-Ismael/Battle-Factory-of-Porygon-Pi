"""Figures for remasking under random decoding with the lookahead (GitHub issue #92), from remask84_random_error.json.

    python remask84_random_figures.py [out_dir]

1. Got against ask for the three arms and the species-first reference, with got = ask.
2. Each arm's ask-versus-got error on asks 1/24 to 17/24 minus no remasking's, with 95% intervals.
Each figure is an ECharts page (src/echarts_chart.py). The two paper rules take blue and aqua because they are the subject.
"""
import json
import sys
from pathlib import Path

from echarts_chart import write_chart

RES = Path.home() / "vgc-data/results"
GREY, PLAIN = "#a9a8a2", "#6f6e69"
BLUE, AQUA = "#2a78d6", "#1baf7a"
STYLE = {
    "rand_la_plain": dict(color=PLAIN, label="No remasking, 84 passes"),
    "rand_la_remask_conf": dict(color=BLUE, label="Confidence-weighted 0.3 fields remasked, 210 passes"),
    "rand_la_remask_even": dict(color=AQUA, label="Even-odds 0.5 fields remasked, 210 passes"),
    "species_first_plain": dict(color=GREY, label="Species-first decoding, no remasking (reference)", dash=True),
}
WIN = "Win rate against the 49 distinct top-50 meta teams, behaviour-cloning policy on both sides, slots shuffled. 3 seeds x 64 teams per ask, one battle per opponent."
TICKS = [1, 4, 8, 12, 17, 20, 23]


def got_against_ask(r, out):
    ks = list(range(1, 24))
    lines = [{"type": "line", "name": st["label"], "color": st["color"], "symbolSize": 6,
              "lineStyle": {"width": 2.25, "type": "dashed" if st.get("dash") else "solid"},
              "emphasis": {"focus": "series", "lineStyle": {"width": 4}},
              "data": [[k, r["arms"][arm]["win"][str(k)]] for k in ks]} for arm, st in STYLE.items()]
    guide = {"type": "line", "name": "got = ask", "silent": True, "symbol": "none", "tooltip": {"show": False},
             "lineStyle": {"color": GREY, "width": 1, "type": "dotted"},
             "data": [[0.5, 0.5 / 24], [23.5, 23.5 / 24]],
             "endLabel": {"show": True, "formatter": "got = ask", "color": GREY},
             "markArea": {"silent": True, "itemStyle": {"color": "rgba(127,127,127,0.12)"},
                          "label": {"position": "insideBottom", "formatter": "18/24 to 23/24:\nreported apart"},
                          "data": [[{"xAxis": 17.5}, {"xAxis": 23.5}]]}}
    option = {
        "tooltip": {"trigger": "axis", "valueFormatter": "{value}"},
        "legend": {"data": [st["label"] for st in STYLE.values()], "top": 0, "left": 56, "orient": "vertical"},
        "grid": {"left": 56, "right": 72, "top": 96, "bottom": 84},
        "xAxis": {"type": "value", "min": 0.5, "max": 23.5, "name": "Ask (the win rate the model is conditioned on)",
                  "nameLocation": "middle", "nameGap": 28, "splitLine": {"show": False},
                  "axisLabel": {"customValues": TICKS, "formatter": "{value}/24"}, "axisTick": {"customValues": TICKS}},
        "yAxis": {"type": "value", "min": 0, "max": 1, "name": "Got (mean win rate at that ask)",
                  "nameLocation": "middle", "nameGap": 36},
        "dataZoom": [{"type": "inside", "filterMode": "none"}, {"type": "slider", "bottom": 6, "height": 18, "filterMode": "none"}],
        "series": [*lines, guide],
    }
    save(option, out / "remask84_random_01_got_against_ask", 520,
         "Remasking lifts got by about 0.01 at every ask; it doesn't make got follow the ask",
         "Random decoding with the lookahead. Above 11/24, got stays near 0.25 in every arm, while species-first decoding reaches about 0.40.")


def error_difference(r, out):
    rows = ["rand_la_remask_conf", "rand_la_remask_even", "species_first_plain"]
    series = []
    for i, arm in enumerate(rows):
        d = r["arms"][arm]["scored"]["minus_rand_la_plain"]; st = STYLE[arm]
        e = r["arms"][arm]["scored"]["error"]
        text = f"{d['difference']:+.4f} (error {e:.4f}), closer on {d['seeds_closer']} of 3 seeds"
        series.append({"type": "line", "name": st["label"], "color": st["color"], "symbol": "none", "silent": True,
                       "lineStyle": {"width": 3, "cap": "round"}, "data": [[d["interval"][0], i], [d["interval"][1], i]]})
        series.append({"type": "scatter", "name": st["label"], "color": st["color"], "symbolSize": 13,
                       "itemStyle": {"borderColor": "#fff", "borderWidth": 2}, "emphasis": {"focus": "series"},
                       "data": [{"value": [d["difference"], i],
                                 "label": {"show": True, "position": "bottom", "distance": 10, "formatter": text}}]})
    option = {
        "tooltip": {"trigger": "item", "formatter": "{a}: {c}"},
        "grid": {"left": 230, "right": 24, "top": 16, "bottom": 56},
        "xAxis": {"type": "value", "min": -0.046, "max": 0.008, "name": "Error minus no remasking's (lower is closer to got = ask)",
                  "nameLocation": "middle", "nameGap": 30},
        "yAxis": {"type": "category", "inverse": True, "data": [STYLE[a]["label"] for a in rows],
                  "axisLabel": {"width": 210, "overflow": "break"}},
        "series": [*series, {"type": "line", "silent": True, "data": [], "markLine": {"symbol": "none", "label": {"show": False},
                                                                                       "lineStyle": {"color": PLAIN, "type": "solid"},
                                                                                       "data": [{"xAxis": 0}]}}],
    }
    save(option, out / "remask84_random_02_error_difference", 360,
         "Both paper rules sit closer to the line than no remasking, by 0.004",
         "Ask-versus-got error on asks 1/24 to 17/24 minus no remasking's 0.0534, with 95% intervals. Near the line means an error under 0.001.")


def save(option, stem, height, head, sub):
    path = write_chart(option, f"{stem}.html", head, sub, WIN, height=height)
    print(path)


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else RES
    out.mkdir(parents=True, exist_ok=True)
    r = json.load(open(RES / "remask84_random_error.json"))
    got_against_ask(r, out); error_difference(r, out)
