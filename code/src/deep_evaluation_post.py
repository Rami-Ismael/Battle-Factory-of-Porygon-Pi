"""The figure and issue comment for the deep re-battle of elite line-ups (docs/weakness-pressure-deep-evaluation.md part A, issue #345).

    LY_PER_CELL=16 python deep_evaluation_post.py figure <folder>          # flip-share.html, an ECharts page
    LY_PER_CELL=16 python deep_evaluation_post.py comment [<image url>]    # the comment; the url holds a {theme} placeholder

Both read the `flips` block of every guidance's deep results file written so far. The figure shows each measured flip share
with its 95% interval beside the share battle noise alone predicts and the ticket's rule; the comment opens with it and
follows with the matrix of counts.
"""
import json
import sys
from pathlib import Path

import line_up_yield as L

RULE = f"{L.FLIP_RULE:.2f}"
MEASURED, PREDICTED = "#3a7bd5", "#e08a1e"                 # mid-tone blue and amber: readable on light and dark surfaces
SHALLOW, DEEP = f"{L.PER_CELL} battles per meta team", f"{L.DEEP_DEPTH} battles per meta team"
INTERVAL = ("js:function (params, api) { var x = api.coord([api.value(0), api.value(1)]), y = api.coord([api.value(0), api.value(2)]);"
            " var w = 8, s = {stroke: api.visual('color'), lineWidth: 2};"
            " return {type: 'group', children: [{type: 'line', shape: {x1: x[0], y1: x[1], x2: y[0], y2: y[1]}, style: s},"
            " {type: 'line', shape: {x1: x[0] - w, y1: x[1], x2: x[0] + w, y2: x[1]}, style: s},"
            " {type: 'line', shape: {x1: y[0] - w, y1: y[1], x2: y[0] + w, y2: y[1]}, style: s}]}; }")


def option(results):
    """results: [(guidance, flips block)] in run order."""
    names = [f"guidance {g}" for g, _ in results]
    return {"tooltip": {"trigger": "axis", "valueFormatter": "js:v => (v == null ? '' : Number(v).toFixed(3))"},
            "legend": {"top": 0},
            "grid": {"top": 48, "left": 64, "right": 120, "bottom": 48},
            "xAxis": {"type": "category", "data": names},
            "yAxis": {"type": "value", "min": 0, "max": 0.5, "name": "share of elite line-ups whose verdict flips",
                      "nameLocation": "middle", "nameGap": 44},
            "series": [
                {"name": "95% interval over line-ups", "type": "custom", "renderItem": INTERVAL, "encode": {"x": 0, "y": [1, 2]}, "itemStyle": {"color": MEASURED},
                 "data": [[i, *s["flip_share_interval"]] for i, (_, s) in enumerate(results)], "z": 2},
                {"name": "measured flip share", "type": "scatter", "symbolSize": 12, "z": 3, "itemStyle": {"color": MEASURED}, "emphasis": {"focus": "series"},
                 "data": [s["flip_share"] for _, s in results],
                 "markLine": {"symbol": "none", "silent": True, "lineStyle": {"type": "dashed"},
                              "label": {"formatter": f"rule {RULE}", "position": "end"}, "data": [{"yAxis": L.FLIP_RULE}]}},
                {"name": "predicted from battle noise alone", "type": "scatter", "symbol": "diamond", "symbolSize": 12, "z": 3,
                 "symbolOffset": [18, 0], "itemStyle": {"color": PREDICTED},
                 "emphasis": {"focus": "series"}, "data": [s["predicted_from_4"]["mean"] for _, s in results]}]}


def figure(results, folder):
    from echarts_chart import write_chart
    return write_chart(option(results), Path(folder) / "flip-share.html",
                       title=f"How often {SHALLOW} and {DEEP} disagree on an elite line-up",
                       subtitle=f"A verdict is held when the line-up's best team scores at least the elite cut {L.CUT} in its re-battle. "
                                f"Above the dashed rule at {RULE}, {DEEP} becomes the re-battle depth.", height=440)


def verdict_line(g, s):
    if s["verdict"] == "4":
        return "the run stops here: 4 battles per meta team stays the re-battle depth" if g == 2 else \
            f"below the rule at this guidance: here {SHALLOW} would stay the depth"
    return L.VERDICTS[s["verdict"]]


def comment(results, url=None):
    g, last = results[-1]
    lo, hi = last["flip_share_interval"]
    lines = [f"**Deep evaluation, guidance {g}: {last['flips']} of {last['line_ups']} elite line-ups change verdict between "
             f"{SHALLOW} and {DEEP} (share {last['flip_share']:.3f}, 95% interval {lo:.3f} to {hi:.3f}); "
             f"by the rule fixed on this ticket, {verdict_line(g, last)}.**", ""]
    if url:
        lines += [f'<picture><source media="(prefers-color-scheme: dark)" srcset="{url.format(theme="dark")}">'
                  f'<img alt="Measured flip share with its 95% interval, the prediction from battle noise alone and the {RULE} rule, '
                  f'for each guidance run so far" src="{url.format(theme="light")}"></picture>', ""]
    lines += [f"A verdict is held or not held: whether the line-up's best team scores at least the elite cut {L.CUT} in its re-battle. "
              f"A flip is a line-up held at one depth and not at the other. The prediction draws each team's true win rate from its "
              f"{SHALLOW} score and adds battle noise alone.", "",
              f"| Guidance | Elite line-ups | Flips | Flip share (95% interval) | Predicted flip share | Held at {SHALLOW} | "
              f"Held at {DEEP} | Held, then not | Not held, then held | Correlation of the two scores | Rule's verdict |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for gg, s in results:
        a, b = s["flip_share_interval"]
        lines.append(f"| {gg} | {s['line_ups']} | {s['flips']} | {s['flip_share']:.3f} ({a:.3f} to {b:.3f}) | {s['predicted_from_4']['mean']:.3f} | "
                     f"{s['held_at_4']} | {s['held_at_16']} | {s['held_to_not_held']} | {s['not_held_to_held']} | {s['correlation']:.2f} | "
                     f"{verdict_line(gg, s)} |")
    lines += ["", f"Scorer: behaviour-cloning policy on both sides, slots shuffled, the 49 distinct top-50 meta teams with the duplicated one "
              f"weighted twice. The {SHALLOW} re-battle ran on seed {L.SHALLOW_SEED}; the {DEEP} re-battle on seed {L.DEEP_SEED} in a "
              f"database of its own. Results: `~/vgc-data/results/line_up_yield_<grid>_d{L.DEEP_DEPTH}.json`. "
              "Pre-registration: `code/docs/weakness-pressure-deep-evaluation.md` part A."]
    return "\n".join(lines) + "\n"


def results_so_far():
    out = []
    for model, g in L.DEEP_GRIDS.items():
        p = L.fresh_path(model, L.DEEP_DEPTH)
        if p.exists() and "flips" in (d := json.load(open(p))):
            out.append((g, d["flips"]))
    return out


if __name__ == "__main__":
    r = results_so_far()
    if not r:
        sys.exit("no guidance has a flips block yet")
    what = sys.argv[1] if len(sys.argv) > 1 else ""
    if what == "figure":
        print(figure(r, sys.argv[2]))
    elif what == "comment":
        sys.stdout.write(comment(r, sys.argv[2] if len(sys.argv) > 2 else None))
    else:
        sys.exit(__doc__)
