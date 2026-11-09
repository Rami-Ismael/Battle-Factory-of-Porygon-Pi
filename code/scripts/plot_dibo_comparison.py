"""Plot the frozen matched-budget comparison without changing the running driver."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from echarts_chart import band, write_chart

PCT = "js:v => Math.round(v * 1000) / 10 + '%'"
NAME = dict(cem="CEM", dibo="DiBO")


def pct_axis(name, grid, **kw):
    return {"type": "value", "gridIndex": grid, "min": 0, "max": 1, "name": name, "nameLocation": "middle", "nameGap": 44,
            "axisLabel": {"formatter": PCT}, **kw}


def curve_series(arm, out, color):
    """Left panel: batch mean with a ±1.96 standard-error band; right panel: best new candidate so far."""
    rounds = [r for _, r in sorted(out["rounds"].items(), key=lambda kv: int(kv[0])) if "evaluated" in r]
    battles = np.cumsum([r["physical_battles"] for r in rounds]).tolist()
    means = np.array([r["mean"] for r in rounds])
    errors = np.array([r["se"] for r in rounds]) * 1.96
    best = np.maximum.accumulate([r["best"] for r in rounds]).tolist()
    name = NAME[arm]
    return [
        {"type": "line", "name": name, "color": color, "data": [list(p) for p in zip(battles, means.tolist())], "emphasis": {"focus": "series"}},
        *band(name, color, battles, means - errors, means + errors, arm),
        {"type": "line", "name": name, "color": color, "xAxisIndex": 1, "yAxisIndex": 1, "emphasis": {"focus": "series"},
         "data": [list(p) for p in zip(battles, best)]},
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("directory", type=Path)
    args = ap.parse_args()
    root = args.directory.resolve()
    proto = json.loads((root / "protocol.json").read_text())
    summary = json.loads((root / "summary.json").read_text())
    data = {arm: json.loads((root / f"{arm}.json").read_text()) for arm in ("cem", "dibo")}
    colors = dict(cem="#7c4d94", dibo="#087e8b")
    grids = [{"left": "7%", "width": "40%", "top": 60, "bottom": 60}, {"left": "57%", "width": "40%", "top": 60, "bottom": 60}]
    titles = [{"text": text, "left": g["left"], "top": 20, "textStyle": {"fontSize": 13}}
              for text, g in zip(("Mean of each selected battle batch", "Best new candidate's noisy search label"), grids)]
    write_chart({"title": titles, "grid": grids, "tooltip": {"trigger": "axis"}, "legend": {"data": ["CEM", "DiBO"], "bottom": 0},
                 "xAxis": [{"type": "value", "gridIndex": g, "name": "Fresh search battles spent", "nameLocation": "middle", "nameGap": 28,
                            "scale": True} for g in (0, 1)],
                 "yAxis": [pct_axis("Win rate against the top50 meta", g) for g in (0, 1)],
                 "series": [s for arm, out in data.items() for s in curve_series(arm, out, colors[arm])]},
                root / "learning-curves.html", "DiBO versus CEM: equal starting labels and fresh-battle budgets", height=440)
    lines = ["# Matched-budget comparison: plots and interpretation", "",
             "[Learning curves](learning-curves.html)", "",
             "The left band is ±1.96 standard errors over the selected teams in that round; it does not estimate variability across training seeds. The right plot shows noisy search labels, so selecting maxima can inflate apparent performance. Neither plot uses heldout outcomes.", ""]
    if summary["complete"]:
        stages = ("confirmation", "heldout")
        grids = [{"left": "8%", "width": "39%", "top": 60, "bottom": 40}, {"left": "58%", "width": "39%", "top": 60, "bottom": 40}]
        titles = [{"text": "Fresh top50 confirmation" if s == "confirmation" else "Fresh heldout ranks 51–100", "left": g["left"], "top": 20,
                   "textStyle": {"fontSize": 13}} for s, g in zip(stages, grids)]
        bars = [{"type": "bar", "name": "Win rate of first search-selected finalist", "xAxisIndex": g, "yAxisIndex": g, "barWidth": "55%",
                 "data": [{"value": summary[stage][f"{arm}_first"], "itemStyle": {"color": colors[arm], "borderRadius": [4, 4, 0, 0]},
                           "label": {"show": True, "position": "top", "formatter": f"{summary[stage][f'{arm}_first']:.1%}"}} for arm in ("cem", "dibo")]}
                for g, stage in enumerate(stages)]
        write_chart({"title": titles, "grid": grids,
                     "xAxis": [{"type": "category", "gridIndex": g, "data": ["CEM", "DiBO"]} for g in (0, 1)],
                     "yAxis": [pct_axis("Win rate of first search-selected finalist" if g == 0 else "", g) for g in (0, 1)],
                     "series": bars},
                    root / "final-comparison.html", f"Same preselected finalist rule · {proto['config']['final_battles']} new battles per team per stage",
                    height=400)
        lines += ["[Independent evaluations](final-comparison.html)", "",
                  "These are the highest-search-score finalists, locked before independent evaluations. They are not picked by the highest heldout or confirmation result.", "",
                  f"Descriptive primary winner in this single matched run: **{summary['descriptive_primary_winner']}**.", "",
                  "| Independent stage | CEM finalist-set mean | DiBO finalist-set mean | DiBO − CEM | Team-cluster bootstrap 95% interval |",
                  "| --- | ---: | ---: | ---: | --- |"]
        for stage in ("confirmation", "heldout"):
            row = summary[stage]
            lo, hi = row["team_cluster_bootstrap_95"]
            lines.append(f"| {stage} | {row['cem_mean']:.1%} | {row['dibo_mean']:.1%} | {row['mean_difference']:+.1%} | [{lo:+.1%}, {hi:+.1%}] |")
        times = {arm: sum(r.get("model_and_proposal_seconds", 0) for r in out["rounds"].values()) for arm, out in data.items()}
        lines += ["", f"Recorded model/proposal wall time: CEM {times['cem']/60:.1f} minutes; DiBO {times['dibo']/60:.1f} minutes. These wall timings span an overnight run and omit interrupted training; active compute time was not monitored, so they are not a clean compute-cost comparison. Battle expenditure was matched.", "",
                  "One seed and the project's BC policy limit the conclusion. These results compare this categorical DiBO adaptation with a fixed-budget CEM repeat using the same initial observations. Historical CEM used different initial labels and stopped early, so its archived score is context only."]
        for checkpoint in (3,):
            lines += ["", f"## {checkpoint}-round checkpoint", "",
                      "| Method | Fresh battles | Mean in round 3 | Best new search label so far |", "| --- | ---: | ---: | ---: |"]
            for arm, out in data.items():
                rounds = [r for _, r in sorted(out["rounds"].items(), key=lambda kv: int(kv[0])) if "evaluated" in r][:checkpoint]
                if len(rounds) == checkpoint:
                    lines.append(f"| {arm} | {sum(r['physical_battles'] for r in rounds)} | {rounds[-1]['mean']:.1%} | {max(r['best'] for r in rounds):.1%} |")
    else:
        lines += ["Full run is still in progress. Final independent comparison is not available yet."]
    lines += ["", "[Budget ledger and raw report](report.md) · [Frozen protocol](protocol.json) · [Machine-readable summary](summary.json)"]
    (root / "analysis.md").write_text("\n".join(lines) + "\n")
    print(root / "analysis.md")


if __name__ == "__main__":
    main()
