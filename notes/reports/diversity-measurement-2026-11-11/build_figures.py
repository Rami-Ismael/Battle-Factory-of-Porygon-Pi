"""Build the static figures of the 2026-11-11 diversity comment (map #263, issue #264) as ECharts options.

    python3 build_figures.py OUT_DIR      writes OUT_DIR/<figure>-<theme>.json, one ECharts 6.1.0 option per file
    node render_svg.mjs OUT_DIR figures   renders each option to figures/<figure>-<theme>.svg (server-side SVG)

Numbers come only from ~/vgc-data/results/diversity_vs_training_2026-11-11.json. storyboard.md holds each
figure's sentence. The comment's figures are fixed images, so every label is drawn on the mark.
"""
import json
import sys
from pathlib import Path

RESULT = Path.home() / "vgc-data/results/diversity_vs_training_2026-11-11.json"
SEEDS = (1, 2, 3)
W = 800
FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"
THEMES = {
    "light": dict(ink="#0b0b0b", ink2="#52514e", ink3="#6f6e69", rule="#cfcdc6", grid="#e9e7e2", band="#e4e2dc",
                  model="#2a78d6", train="#eb6834", trainall="#8a4fc4", anchor="#8a8984",
                  s0="#d6d4cd", s1="#9fd3c3", s2="#3fa88a", s3="#0f5e4b", on=["#0b0b0b", "#0b0b0b", "#ffffff", "#ffffff"]),
    "dark": dict(ink="#f0efec", ink2="#c3c2b7", ink3="#a9a8a2", rule="#4a4a46", grid="#2c2c2a", band="#34332f",
                 model="#3987e5", train="#d95926", trainall="#a77ae0", anchor="#8f8e86",
                 s0="#4a4945", s1="#2f6e5c", s2="#3fb592", s3="#a8ead5", on=["#f0efec", "#f0efec", "#0b0b0b", "#0b0b0b"]),
}
MODEL, TRAIN, TALL = "Default model", "Training, matched to asks", "Training, all win rates"
COPIED, RANDOM = "One team repeated", "Random legal teams"


def mean(xs): return sum(xs) / len(xs)


def load():
    r = json.loads(RESULT.read_text())
    M = r["measures"]
    seeds = lambda prefix, m: [M[f"{prefix}{s}"][m] for s in SEEDS]
    return r, M, seeds


def base(t, height, title):
    return {"backgroundColor": "transparent", "animation": False, "width": W, "height": height,
            "textStyle": {"fontFamily": FONT, "color": t["ink2"], "fontSize": 13},
            "title": [{"text": title, "left": 16, "top": 12,
                       "textStyle": {"fontSize": 17, "fontWeight": 600, "color": t["ink"], "lineHeight": 23, "fontFamily": FONT}}]}


def cat_axis(t, data, **kw):
    return {"type": "category", "data": data, "inverse": True, "axisTick": {"show": False},
            "axisLine": {"lineStyle": {"color": t["rule"]}},
            "axisLabel": {"color": t["ink"], "fontSize": 13, "fontFamily": FONT}, **kw}


def val_axis(t, name, **kw):
    return {"type": "value", "name": name, "nameLocation": "middle", "nameGap": 26,
            "nameTextStyle": {"color": t["ink2"], "fontSize": 13}, "splitLine": {"lineStyle": {"color": t["grid"]}},
            "axisLine": {"show": False}, "axisLabel": {"color": t["ink2"], "fontSize": 12}, **kw}


def teaser(t, _r, M, seeds):
    """Two bar panels on the scale from one team repeated to random legal teams."""
    rows = [MODEL, TRAIN, TALL, COPIED, RANDOM]
    colors = [t["model"], t["train"], t["trainall"], t["anchor"], t["anchor"]]
    b = [mean(seeds("model_s", "builds_per_species")), mean(seeds("train_s", "builds_per_species")),
         mean(seeds("train_all_s", "builds_per_species")), M["copied"]["builds_per_species"], M["random"]["builds_per_species"]]
    n = [mean(seeds("model_s", "nearest_mean")), mean(seeds("train_s", "nearest_mean")),
         mean(seeds("train_all_s", "nearest_mean")), M["copied"]["nearest_mean"], M["random"]["nearest_mean"]]
    o = base(t, 416, f"The default model writes {b[1] - b[0]:.1f} fewer builds per species than training,\n"
                     f"yet its teams sit {n[0] - n[1]:.1f} fields further from the closest training team")
    head = lambda left, a, bline: [
        {"text": a, "left": left, "top": 78, "textStyle": {"fontSize": 14, "fontWeight": 600, "color": t["ink"], "fontFamily": FONT}},
        {"text": bline, "left": left, "top": 98, "textStyle": {"fontSize": 13, "fontWeight": 400, "color": t["model"], "fontFamily": FONT}}]
    o["title"] += head(196, "Builds per species at 30 uses", f"model: {b[1] - b[0]:.1f} fewer than matched training")
    o["title"] += head(512, "Nearest training distance, fields of 84", f"model: {n[0] - n[1]:.1f} more than matched training")
    o["title"].append({"text": "Model and training bars are means of three sampling seeds or three training draws. One team repeated is a\n"
                               "training team, so its distance is to the next closest training team.",
                       "left": 16, "bottom": 8, "textStyle": {"fontSize": 12, "fontWeight": 400, "color": t["ink2"], "lineHeight": 16, "fontFamily": FONT}})
    o["grid"] = [{"left": 196, "width": 252, "top": 128, "bottom": 88}, {"left": 512, "width": 230, "top": 128, "bottom": 88}]
    o["xAxis"] = [val_axis(t, "distinct builds among 30 uses", gridIndex=0, min=0, max=30, interval=10),
                  val_axis(t, "fields that differ, of 84", gridIndex=1, min=0, max=70, interval=20)]
    o["yAxis"] = [cat_axis(t, rows, gridIndex=0),
                  cat_axis(t, rows, gridIndex=1, axisLabel={"show": False})]
    for g, vals in ((0, b), (1, n)):
        o.setdefault("series", []).append({
            "type": "bar", "xAxisIndex": g, "yAxisIndex": g, "barWidth": 18,
            "data": [{"value": round(v, 4), "itemStyle": {"color": c}} for v, c in zip(vals, colors)],
            "label": {"show": True, "position": "right", "formatter": "{c}", "color": t["ink"], "fontSize": 13},
        })
    for s, vals in zip(o["series"], (b, n)):
        for d, v in zip(s["data"], vals):
            d["label"] = {"formatter": f"{v:.1f}" if v != round(v) else f"{v:.0f}"}
    return o, dict(builds=b, nearest=n)


def builds(t, r, _M, seeds):
    """Builds per species per seed, against the noise floor around matched training."""
    F = r["floor"]["builds_per_species"]
    mod, tr, ta = seeds("model_s", "builds_per_species"), seeds("train_s", "builds_per_species"), seeds("train_all_s", "builds_per_species")
    gap = mean(tr) - mean(mod)
    lo, hi = mean(tr) + F["lo"], mean(tr) + F["hi"]
    rows = [MODEL, TRAIN, TALL]
    o = base(t, 356, f"All three seeds of the default model sit {gap:.1f} builds per species below matched\n"
                     f"training, far outside the band two training draws differ by")
    o["grid"] = {"left": 196, "right": 40, "top": 100, "bottom": 92}
    o["xAxis"] = val_axis(t, "distinct builds among 30 uses of one species, mean over species", min=19, max=27, interval=1)
    o["yAxis"] = cat_axis(t, rows)
    rng = lambda xs: f"{min(xs):.1f} to {max(xs):.1f}" if f"{min(xs):.1f}" != f"{max(xs):.1f}" else f"{min(xs):.1f}"
    series = []
    for name, xs, c, pos in ((MODEL, mod, t["model"], "left"), (TRAIN, tr, t["train"], "right"), (TALL, ta, t["trainall"], "left")):
        pts = [{"value": [round(x, 4), name]} for x in xs]
        edge = min(xs) if pos == "left" else max(xs)
        pts.append({"value": [round(edge, 4), name], "symbolSize": 0,
                    "label": {"show": True, "position": pos, "distance": 10, "formatter": rng(xs), "color": t["ink"], "fontSize": 13}})
        series.append({"type": "scatter", "symbolSize": 13, "itemStyle": {"color": c, "opacity": 0.85}, "data": pts, "z": 3})
    series[1]["markArea"] = {"silent": True, "itemStyle": {"color": t["band"]},
                             "label": {"show": True, "position": "insideTop", "color": t["ink2"], "fontSize": 12, "distance": -18},
                             "data": [[{"name": "noise floor", "xAxis": round(lo, 4)}, {"xAxis": round(hi, 4)}]]}
    series[0]["markLine"] = {"silent": True, "symbol": ["none", "arrow"], "symbolSize": 9,
                             "lineStyle": {"color": t["model"], "type": "dashed", "width": 1.5},
                             "label": {"show": True, "position": "middle", "formatter": f"{gap:.1f} fewer", "color": t["model"],
                                       "fontSize": 13, "distance": 6},
                             "data": [[{"coord": [round(mean(tr), 4), MODEL]}, {"coord": [round(max(mod) + 0.25, 4), MODEL]}]]}
    o["series"] = series
    o["title"].append({"text": "One dot per sampling seed or training draw. The band is the range of the difference between two "
                               "training draws, placed\naround matched training; a gap counts when all three seeds land outside it.",
                       "left": 16, "bottom": 6, "textStyle": {"fontSize": 12, "fontWeight": 400, "color": t["ink2"],
                                                              "lineHeight": 16, "fontFamily": FONT}})
    return o, dict(model=mod, train=tr, train_all=ta, gap=gap, band=(lo, hi))


def novelty(t, _r, _M, seeds):
    """Stacked share of 0, 1, 2 and 3+ species swaps from the closest training line-up."""
    parts = [("novelty_swaps_0", "0 swaps, a training line-up copied"), ("novelty_swaps_1", "1 swap"),
             ("novelty_swaps_2", "2 swaps"), ("novelty_swaps_3plus", "3 or more swaps")]
    rows = [(MODEL, "model_s", t["model"]), (TRAIN, "train_s", t["train"]), (TALL, "train_all_s", t["trainall"])]
    vals = {m: [mean(seeds(p, m)) for _, p, _ in rows] for m, _ in parts}
    o = base(t, 290, f"The default model copies a training line-up {100 * vals['novelty_swaps_0'][0]:.0f}% of the time against "
                     f"{100 * vals['novelty_swaps_0'][1]:.0f}% for matched\ntraining; the rest goes to 1 and 2 swaps, "
                     f"not to new line-ups")
    o["legend"] = {"top": 66, "left": 196, "itemWidth": 14, "itemHeight": 12, "itemGap": 18,
                   "textStyle": {"color": t["ink"], "fontSize": 13}}
    o["grid"] = {"left": 196, "right": 64, "top": 104, "bottom": 56}
    o["xAxis"] = val_axis(t, "", min=0, max=1, interval=0.25, axisLabel={"show": False})
    o["yAxis"] = cat_axis(t, [{"value": n, "textStyle": {"color": c, "fontWeight": 600}} for n, _, c in rows])
    o["series"] = []
    for i, (m, name) in enumerate(parts):
        last = i == len(parts) - 1
        data = []
        for v in vals[m]:
            show = last or v >= 0.045
            data.append({"value": round(v, 4), "label": {"show": show, "formatter": f"{100 * v:.1f}%"}})
        o["series"].append({"type": "bar", "name": name, "stack": "n", "barWidth": 30,
                            "itemStyle": {"color": t[f"s{i}"], "borderColor": "transparent"},
                            "label": {"position": "right" if last else "inside", "fontSize": 13,
                                      "color": t["ink"] if last else t["on"][i]},
                            "data": data})
    o["title"].append({"text": "Share of 1,088 teams per row, mean of three sampling seeds or three training draws. Swaps count the species to replace\n"
                               "to reach the closest training line-up; 3 or more swaps is a new line-up.",
                       "left": 16, "bottom": 6, "textStyle": {"fontSize": 12, "fontWeight": 400, "color": t["ink2"],
                                                              "lineHeight": 16, "fontFamily": FONT}})
    return o, vals


def archetypes(t, r, _M, seeds):
    """Model minus matched training per seed, in percentage points, against each archetype's noise floor."""
    names = [("Trick Room", "Trick Room"), ("sun", "Sun"), ("Tailwind", "Tailwind"), ("rain", "Rain")]
    V, F = r["verdict"], r["floor"]
    xmin, xmax, left, width = -10, 14, 110, 440
    ppu = width / (xmax - xmin)
    right_labels, info = [], {}
    for key, label in names:
        m = f"archetype_{key}"
        mm, tm = mean(seeds("model_s", m)), mean(seeds("train_s", m))
        diffs = [100 * d for d in V[m]["diffs"]]
        lo, hi = 100 * F[m]["lo"], 100 * F[m]["hi"]
        outside = [d < lo or d > hi for d in diffs]
        verdict = "beyond the floor, every seed" if V[m]["differs"] else f"beyond the floor on {sum(outside)} of 3 seeds"
        right_labels.append(f"{100 * mm:.1f}% vs {100 * tm:.1f}%\n{verdict}")
        info[label] = dict(model=mm, train=tm, diffs=diffs, floor=(lo, hi), outside=outside, differs=V[m]["differs"])
    o = base(t, 386, "Trick Room and sun teams are more common than in matched training on every seed,\n"
                     "beyond the noise floor; Tailwind and rain are not")
    o["grid"] = {"left": left, "width": width, "top": 86, "bottom": 96}
    o["xAxis"] = val_axis(t, "default model minus matched training, percentage points", min=xmin, max=xmax, interval=2)
    o["yAxis"] = [cat_axis(t, [l for _, l in names], axisLabel={"color": t["ink"], "fontSize": 14, "fontWeight": 600}),
                  cat_axis(t, right_labels, position="right", axisLine={"show": False},
                           axisLabel={"color": t["ink2"], "fontSize": 13, "lineHeight": 18, "margin": 16})]
    band = [{"value": [round((i["floor"][0] + i["floor"][1]) / 2, 4), l],
             "symbolSize": [round((i["floor"][1] - i["floor"][0]) * ppu, 2), 30]} for l, i in info.items()]
    dots = lambda filled: [{"value": [round(d, 4), l]} for l, i in info.items() for d, out in zip(i["diffs"], i["outside"])
                           if out == filled]
    o["series"] = [
        {"type": "scatter", "symbol": "rect", "data": band, "itemStyle": {"color": t["band"]}, "z": 1, "silent": True,
         "markLine": {"silent": True, "symbol": "none", "label": {"show": False},
                      "lineStyle": {"color": t["ink3"], "type": "solid", "width": 1}, "data": [{"xAxis": 0}]}},
        {"type": "scatter", "symbolSize": 13, "data": dots(True), "itemStyle": {"color": t["model"]}, "z": 3},
        {"type": "scatter", "symbolSize": 12, "data": dots(False),
         "itemStyle": {"color": "transparent", "borderColor": t["model"], "borderWidth": 2}, "z": 3},
    ]
    o["title"].append({"text": "One dot per sampling seed, filled when it lands outside the grey noise floor. Right: share of teams, "
                               "default model\nagainst matched training, means of three. Setters only: a team counts when any set "
                               "carries the setting move, ability or mega stone.",
                       "left": 16, "bottom": 6, "textStyle": {"fontSize": 12, "fontWeight": 400, "color": t["ink2"],
                                                              "lineHeight": 16, "fontFamily": FONT}})
    return o, info


FIGURES = {"teaser": teaser, "builds": builds, "novelty": novelty, "archetypes": archetypes}


def main(out):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    r, M, seeds = load()
    shown = {}
    for name, make in FIGURES.items():
        for theme, t in THEMES.items():
            option, numbers = make(t, r, M, seeds)
            (out / f"{name}-{theme}.json").write_text(json.dumps(option))
        shown[name] = numbers
    (out / "shown_numbers.json").write_text(json.dumps(shown, indent=1, default=list))
    print(json.dumps(shown, default=list))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "options")
