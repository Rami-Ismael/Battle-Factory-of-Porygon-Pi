"""The figures and issue comment for the deep re-battle of elite line-ups (docs/weakness-pressure-deep-evaluation.md part A, issue #345).

    LY_PER_CELL=16 python deep_evaluation_post.py figure <folder>          # scores.html, flip-share.html, held.html
    LY_PER_CELL=16 python deep_evaluation_post.py comment [<image url>]    # the url holds {name} and {theme} placeholders
    LY_POLICY=foulplay LY_PER_CELL=16 ... figure | comment                 # the Foul Play re-battles (issue #460)
    LY_POLICY=foulplay LY_PER_CELL=4 LY_MODEL=<grid> python deep_evaluation_post.py prediction   # before the 16-battle arm

All read the `flips` block of every guidance's deep results file written so far. The storyboard is in the pre-registration:
  scores      each elite line-up's score at 4 and at 16 battles per meta team, by what happened to its verdict (the teaser)
  flip-share  the share of verdicts that flip, its 95% interval, the prediction from battle noise alone and the 0.10 rule
  held        line-ups held at each depth, split into held at both and held at one only
"""
import json
import math
import sys
from pathlib import Path

import line_up_yield as L

RULE = f"{L.FLIP_RULE:.2f}"
CUT = L.VERDICT_CUT                                        # the cut a re-battled team must reach to hold, under this run's policy
CUT_TEXT = f"{CUT:.3f}"
SCORER = "Foul Play deterministic argmax" if L.FOUL_PLAY else "behaviour-cloning policy"
SHALLOW, DEEP = f"{L.PER_CELL} battles per meta team", f"{L.DEEP_DEPTH} battles per meta team"
# One meaning per color across the three figures (categorical slots 1-4 and 7, validated light and dark by the dataviz validator).
BOTH, LOST, GAINED, NEITHER = "#2a78d6", "#eb6834", "#1baf7a", "#a3a29d"
PREDICTED, FLIPS = "#eda100", "#6250d6"
DARK = {BOTH: "#3987e5", LOST: "#d95926", GAINED: "#199e70", NEITHER: "#6f6e69", PREDICTED: "#c98500", FLIPS: "#9085e9"}
GROUPS = (("held at both depths", BOTH), (f"held at {L.PER_CELL} only, lost at {L.DEEP_DEPTH}", LOST),
          (f"held at {L.DEEP_DEPTH} only, gained", GAINED), ("held at neither", NEITHER))
INTERVAL = ("js:function (params, api) { var x = api.coord([api.value(0), api.value(1)]), y = api.coord([api.value(0), api.value(2)]);"
            " var w = 8, s = {stroke: api.visual('color'), lineWidth: 2};"
            " return {type: 'group', children: [{type: 'line', shape: {x1: x[0], y1: x[1], x2: y[0], y2: y[1]}, style: s},"
            " {type: 'line', shape: {x1: x[0] - w, y1: x[1], x2: x[0] + w, y2: x[1]}, style: s},"
            " {type: 'line', shape: {x1: y[0] - w, y1: y[1], x2: y[0] + w, y2: y[1]}, style: s}]}; }")
FIGURES = ("scores", "flip-share", "held")


def group_of(a, b):
    held4, held16 = a >= CUT, b >= CUT
    return 0 if held4 and held16 else 1 if held4 else 2 if held16 else 3


def scores_option(results):
    """Small multiples, one per guidance: 4-battle score across, 16-battle score up, the cut on both axes."""
    n = len(results)
    vals = [v for _, s in results for pair in s["scores"] for v in pair]
    lo, hi = math.floor((min(vals) - 0.01) / 0.05) * 0.05, math.ceil((max(vals) + 0.01) / 0.05) * 0.05
    width = 88 / n
    opt = {"tooltip": {"trigger": "item", "formatter": f"js:p => p.seriesName + '<br>{SHALLOW}: ' + p.value[0].toFixed(3) + "
                                                        f"'<br>{DEEP}: ' + p.value[1].toFixed(3)"},
           "legend": {"top": 0, "data": [name for name, _ in GROUPS]}, "title": [], "grid": [], "xAxis": [], "yAxis": [], "series": []}
    for i, (g, s) in enumerate(results):
        counts = [0, 0, 0, 0]
        for a, b in s["scores"]:
            counts[group_of(a, b)] += 1
        left = 6 + i * width
        opt["title"].append({"text": f"guidance {g}", "subtext": f"{counts[1]} lost, {counts[2]} gained of {s['line_ups']}",
                             "left": f"{left + width / 2 - 3}%", "top": 34, "textAlign": "center",
                             "textStyle": {"fontSize": 13}, "subtextStyle": {"fontSize": 12}})
        opt["grid"].append({"left": f"{left}%", "width": f"{width - 6}%", "top": 96, "bottom": 56})
        ticks = {"interval": 0.05, "axisLabel": {"formatter": "js:v => v.toFixed(2)"}}
        opt["xAxis"].append({"type": "value", "gridIndex": i, "min": lo, "max": hi, **ticks, "name": SHALLOW if i == n // 2 else "",
                             "nameLocation": "middle", "nameGap": 30})
        opt["yAxis"].append({"type": "value", "gridIndex": i, "min": lo, "max": hi, **ticks, "name": DEEP if i == 0 else "",
                             "nameLocation": "middle", "nameGap": 40})
        opt["series"].append({"type": "line", "xAxisIndex": i, "yAxisIndex": i, "silent": True, "symbol": "none", "tooltip": {"show": False},
                              "lineStyle": {"color": NEITHER, "width": 1, "type": "dotted"}, "data": [[lo, lo], [hi, hi]],
                              "markLine": {"symbol": "none", "silent": True, "label": {"show": i == n - 1, "formatter": f"cut {CUT_TEXT}", "position": "end"},
                                           "lineStyle": {"color": NEITHER, "type": "dashed", "width": 1},
                                           "data": [{"xAxis": CUT, "label": {"show": False}}, {"yAxis": CUT}]}})
        for k, (name, color) in enumerate(GROUPS):
            opt["series"].append({"name": name, "type": "scatter", "xAxisIndex": i, "yAxisIndex": i, "symbolSize": 8,
                                  # the flips carry the idea; the two kept verdicts recede
                                  "itemStyle": {"color": color, "borderColor": "#ffffff", "borderWidth": 1, "opacity": 1 if k in (1, 2) else 0.55},
                                  "emphasis": {"focus": "series"},
                                  "data": [[a, b] for a, b in s["scores"] if group_of(a, b) == k]})
    return opt


def flip_share_option(results):
    """results: [(guidance, flips block)] in run order."""
    return {"tooltip": {"trigger": "axis", "valueFormatter": "js:v => (v == null ? '' : Number(v).toFixed(3))"},
            "legend": {"top": 0},
            "grid": {"top": 48, "left": 64, "right": 120, "bottom": 48},
            "xAxis": {"type": "category", "data": [f"guidance {g}" for g, _ in results]},
            "yAxis": {"type": "value", "min": 0, "max": 0.5, "name": "share of elite line-ups whose verdict flips",
                      "nameLocation": "middle", "nameGap": 44},
            "series": [
                {"name": "95% interval over line-ups", "type": "custom", "renderItem": INTERVAL, "encode": {"x": 0, "y": [1, 2]},
                 "itemStyle": {"color": FLIPS}, "data": [[i, *s["flip_share_interval"]] for i, (_, s) in enumerate(results)], "z": 2},
                {"name": "measured flip share", "type": "scatter", "symbolSize": 12, "z": 3, "itemStyle": {"color": FLIPS},
                 "emphasis": {"focus": "series"}, "data": [s["flip_share"] for _, s in results],
                 "label": {"show": True, "position": "left", "formatter": "js:p => p.value.toFixed(2)"},
                 "markLine": {"symbol": "none", "silent": True, "lineStyle": {"type": "dashed", "color": NEITHER},
                              "label": {"formatter": f"rule {RULE}", "position": "end"}, "data": [{"yAxis": L.FLIP_RULE}]}},
                {"name": "predicted from battle noise alone", "type": "scatter", "symbol": "diamond", "symbolSize": 12, "z": 3,
                 "symbolOffset": [18, 0], "itemStyle": {"color": PREDICTED}, "emphasis": {"focus": "series"},
                 "data": [s["predicted_from_4"]["mean"] for _, s in results]}]}


def held_option(results):
    """Two bars per guidance, one per depth: the line-ups held at both depths, then those held at this depth only."""
    rows, both, only4, only16 = [], [], [], []
    for g, s in results:
        b = s["held_at_4"] - s["held_to_not_held"]
        rows += [f"guidance {g}, {SHALLOW}", f"guidance {g}, {DEEP}"]
        both += [b, b]; only4 += [s["held_to_not_held"], None]; only16 += [None, s["not_held_to_held"]]
    seg = lambda name, color, data: {"name": name, "type": "bar", "stack": "held", "barWidth": 18, "data": data,
                                     "itemStyle": {"color": color, "borderColor": "#ffffff", "borderWidth": 2, "borderRadius": 0},
                                     "emphasis": {"focus": "series"}, "label": {"show": True, "position": "inside", "color": "#ffffff"}}
    return {"tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"}},
            "legend": {"top": 0, "data": [GROUPS[0][0], GROUPS[1][0], GROUPS[2][0]]},
            "grid": {"top": 48, "left": 250, "right": 40, "bottom": 48},
            "xAxis": {"type": "value", "name": "elite line-ups held", "nameLocation": "middle", "nameGap": 30},
            "yAxis": {"type": "category", "data": rows, "inverse": True},
            "series": [seg(GROUPS[0][0], BOTH, both), seg(GROUPS[1][0], LOST, only4), seg(GROUPS[2][0], GAINED, only16)]}


def near_cut_share(results, width=0.07):
    """Share of flipped line-ups whose 4-battle score sat within `width` (two standard errors at 196 battles) of the cut."""
    flips = [a for _, s in results for a, b in s["scores"] if group_of(a, b) in (1, 2)]
    return sum(abs(a - CUT) <= width for a in flips) / max(len(flips), 1)


def scores_title(results):
    share = near_cut_share(results)
    if share >= 0.8:
        return f"Flipped line-ups sat near the cut: {share:.0%} within 0.07 of it at {SHALLOW}"
    return f"Flipped line-ups were not confined to the cut: only {share:.0%} within 0.07 of it at {SHALLOW}"


def held_title(results):
    moves = [abs(s["held_at_16"] - s["held_at_4"]) / max(s["held_at_4"], 1) for _, s in results]
    lost = [s["held_to_not_held"] / max(s["held_at_4"], 1) for _, s in results]
    span = f"{lost[0]:.0%}" if len(lost) == 1 else f"{min(lost):.0%} to {max(lost):.0%}"
    if max(moves) <= 0.1:
        return f"The held count moves by at most {max(moves):.0%}, but {span} of the line-ups held at {L.PER_CELL} are not held at {L.DEEP_DEPTH}"
    return f"The held count moves by up to {max(moves):.0%} between the depths"


def figure(results, folder):
    from echarts_chart import write_chart
    folder = Path(folder)
    write_chart(scores_option(results), folder / "scores.html", height=480, dark=DARK, title=scores_title(results),
                subtitle=f"Each dot is an elite line-up's best team, scored in a {SHALLOW} re-battle (across) and a {DEEP} re-battle (up). "
                         f"Dashed lines: the elite cut {CUT_TEXT} under {SCORER}. Dots off the dotted diagonal moved; dots across a dashed line changed verdict.")
    write_chart(flip_share_option(results), folder / "flip-share.html", height=440, dark=DARK,
                title=f"How often {SHALLOW} and {DEEP} disagree on an elite line-up",
                subtitle=f"A verdict is held when the line-up's best team scores at least the elite cut {CUT_TEXT} in its {SCORER} re-battle. "
                         f"Above the dashed rule at {RULE}, {DEEP} becomes the re-battle depth.")
    write_chart(held_option(results), folder / "held.html", height=120 + 70 * len(results), dark=DARK,
                title=held_title(results),
                subtitle="Elite line-ups held at each depth, split into those held at both depths and those held at one only. "
                         "Counts are per guidance; the orange and green tails are the line-ups whose verdict flipped.")
    return [folder / f"{name}.html" for name in FIGURES]


def verdict_line(g, s):
    if s["verdict"] == "4":
        return "the run stops here: 4 battles per meta team stays the re-battle depth" if g == 2 else \
            f"below the rule at this guidance: here {SHALLOW} would stay the depth"
    return L.VERDICTS[s["verdict"]]


def picture(url, name, alt):
    return (f'<picture><source media="(prefers-color-scheme: dark)" srcset="{url.format(name=name, theme="dark")}">'
            f'<img alt="{alt}" src="{url.format(name=name, theme="light")}"></picture>')


def comment(results, url=None):
    g, last = results[-1]
    lo, hi = last["flip_share_interval"]
    lines = [f"**{'Foul Play deep evaluation' if L.FOUL_PLAY else 'Deep evaluation'}, guidance {g}: {last['flips']} of {last['line_ups']} elite line-ups change verdict between "
             f"{SHALLOW} and {DEEP} (share {last['flip_share']:.3f}, 95% interval {lo:.3f} to {hi:.3f}); "
             f"by the rule fixed on this ticket, {verdict_line(g, last)}.**", ""]
    if url:
        lines += [picture(url, "scores", f"Each elite line-up's score at {SHALLOW} against {DEEP}, by what happened to its verdict"), "",
                  picture(url, "flip-share", f"Measured flip share with its 95% interval, the prediction from battle noise alone and the {RULE} rule"), "",
                  picture(url, "held", "Line-ups held at each depth: at both, at 4 only, at 16 only"), ""]
    lines += [f"A verdict is held or not held: whether the line-up's best team scores at least the elite cut {CUT_TEXT} in its re-battle. "
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
    if L.FOUL_PLAY:
        lines += ["", "Read at the behaviour-cloning cut 0.531 instead (exploratory, same battles):", "",
                  "| Guidance | Flips | Flip share (95% interval) | Predicted | Held at 4 | Held at 16 |", "| --- | --- | --- | --- | --- | --- |"]
        for gg, s in results:
            b = s["at_bc_cut"]; lo2, hi2 = b["flip_share_interval"]
            lines.append(f"| {gg} | {b['flips']} | {b['flip_share']:.3f} ({lo2:.3f} to {hi2:.3f}) | {b['predicted']:.3f} | {b['held_at_4']} | {b['held_at_16']} |")
        lines += ["", f"Scorer: Foul Play deterministic argmax (30,000 iterations, 1 sample) on both sides, the same policy row as policy "
                  f"{L.FP_REFERENCE_POLICY} of #447; slots shuffled; the 49 distinct top-50 meta teams with the duplicated one weighted twice. "
                  f"Elite cut {CUT_TEXT}: a top-50 meta team's mean win rate against the other 49 from #447's 1,176 meta x meta pairs. "
                  f"Both depths battled afresh: {SHALLOW} on seed {L.FP_SEEDS[L.PER_CELL]}, {DEEP} on seed {L.FP_SEEDS[L.DEEP_DEPTH]}, each in a "
                  f"database of its own. Teams: #345's, the stored best team per elite line-up. Results: "
                  f"`~/vgc-data/results/line_up_yield_<grid>_fp_d4.json` and `_fp_d16.json`. "
                  "Pre-registration: `code/docs/weakness-pressure-deep-evaluation.md`, part A under Foul Play."]
    else:
        lines += ["", f"Scorer: behaviour-cloning policy on both sides, slots shuffled, the 49 distinct top-50 meta teams with the duplicated one "
                  f"weighted twice. The {SHALLOW} re-battle ran on seed {L.SHALLOW_SEED}; the {DEEP} re-battle on seed {L.DEEP_SEED} in a "
                  f"database of its own. Results: `~/vgc-data/results/line_up_yield_<grid>_d{L.DEEP_DEPTH}.json`. "
                  "Pre-registration: `code/docs/weakness-pressure-deep-evaluation.md` part A."]
    return "\n".join(lines) + "\n"


def prediction_comment(model):
    """Foul Play: the flip share battle noise predicts, posted after the 4-battle re-battle and before any 16-battle result."""
    pr = json.load(open(L.fresh_path(model, L.PER_CELL)))["prediction"]
    a, b = pr["predicted"]["interval_90"]
    g = L.DEEP_GRIDS.get(model, model)
    return "\n".join([
        f"**Prediction before the {DEEP} re-battle, guidance {g}: battle noise alone predicts {pr['predicted']['mean']:.3f} of "
        f"{pr['line_ups']} elite line-ups flip verdict (90% of draws {a:.3f} to {b:.3f}).**", "",
        f"Made {pr['made']} from the {SHALLOW} Foul Play re-battle, before any {DEEP} battle. {pr['held_at_4']} line-ups hold at the "
        f"Foul Play elite cut {pr['cut']:.3f}; {pr['near_cut_at_4']} score within 0.036 of it. Each team's true win rate is drawn from a "
        f"normal empirical-Bayes posterior fitted to the spread of the {SHALLOW} scores (mean {pr['mean_at_4']:.3f}, SD {pr['sd_at_4']:.3f}), "
        f"then given a fresh {DEEP} score. A flat prior gives {pr['predicted_flat_prior']['mean']:.3f}; at the behaviour-cloning cut 0.531 "
        f"the prediction is {pr['predicted_at_bc_cut']:.3f}. The rule fixed on this ticket: {DEEP} stays the depth when the measured "
        f"share's 95% interval lies above {RULE}."]) + "\n"


def results_so_far():
    out = []
    for model, g in L.DEEP_GRIDS.items():
        p = L.fresh_path(model, L.DEEP_DEPTH)
        if p.exists() and "flips" in (d := json.load(open(p))):
            out.append((g, d["flips"]))
    return out


if __name__ == "__main__":
    if sys.argv[1:2] == ["prediction"]:
        sys.stdout.write(prediction_comment(L.MODEL)); sys.exit(0)
    r = results_so_far()
    if not r:
        sys.exit("no guidance has a flips block yet")
    what = sys.argv[1] if len(sys.argv) > 1 else ""
    if what == "figure":
        print("\n".join(map(str, figure(r, sys.argv[2]))))
    elif what == "comment":
        sys.stdout.write(comment(r, sys.argv[2] if len(sys.argv) > 2 else None))
    else:
        sys.exit(__doc__)
