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
FIGURE_WIDTH = 800
FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"
THEMES = {
    "light": dict(ink="#0b0b0b", ink2="#52514e", ink3="#6f6e69", rule="#cfcdc6", grid="#e9e7e2", band="#e4e2dc",
                  model="#2a78d6", train="#eb6834", trainall="#8a4fc4", anchor="#8a8984",
                  s0="#d6d4cd", s1="#9fd3c3", s2="#3fa88a", s3="#0f5e4b", on=["#0b0b0b", "#0b0b0b", "#ffffff", "#ffffff"]),
    "dark": dict(ink="#f0efec", ink2="#c3c2b7", ink3="#a9a8a2", rule="#4a4a46", grid="#2c2c2a", band="#34332f",
                 model="#3987e5", train="#d95926", trainall="#a77ae0", anchor="#8f8e86",
                 s0="#4a4945", s1="#2f6e5c", s2="#3fb592", s3="#a8ead5", on=["#f0efec", "#f0efec", "#0b0b0b", "#0b0b0b"]),
}
MODEL, TRAIN, TALL = "Default model", "Win-rate-matched training", "Any-win-rate training"
COPIED, RANDOM = "One training team repeated", "Random legal teams"


def mean(values): return sum(values) / len(values)


def load():
    result = json.loads(RESULT.read_text())
    measures = result["measures"]

    def seed_values(prefix, measure):
        return [measures[f"{prefix}{seed}"][measure] for seed in SEEDS]
    return result, measures, seed_values


def base(theme, height, title):
    return {"backgroundColor": "transparent", "animation": False, "width": FIGURE_WIDTH, "height": height,
            "textStyle": {"fontFamily": FONT, "color": theme["ink2"], "fontSize": 13},
            "title": [{"text": title, "left": 16, "top": 12,
                       "textStyle": {"fontSize": 17, "fontWeight": 600, "color": theme["ink"], "lineHeight": 23, "fontFamily": FONT}}]}


def cat_axis(theme, data, **settings):
    return {"type": "category", "data": data, "inverse": True, "axisTick": {"show": False},
            "axisLine": {"lineStyle": {"color": theme["rule"]}},
            "axisLabel": {"color": theme["ink"], "fontSize": 13, "fontFamily": FONT}, **settings}


def val_axis(theme, name, **settings):
    return {"type": "value", "name": name, "nameLocation": "middle", "nameGap": 26,
            "nameTextStyle": {"color": theme["ink2"], "fontSize": 13, "lineHeight": 17}, "splitLine": {"lineStyle": {"color": theme["grid"]}},
            "axisLine": {"show": False}, "axisLabel": {"color": theme["ink2"], "fontSize": 12}, **settings}


def noise_floor_series(theme, rows, pixels_per_unit, dot_size, dot_opacity=None):
    """The grey noise-floor band per row, the zero rule, and one dot per sampling seed: filled when the seed's
    difference lands outside the floor, hollow when inside. rows: dicts with label, diffs and floor (low, high)."""
    band = [{"value": [round((low + high) / 2, 4), row["label"]], "symbolSize": [max(round((high - low) * pixels_per_unit, 2), 1), 30]}
            for row in rows for low, high in [row["floor"]]]

    def dots(filled):
        return [{"value": [round(diff, 4), row["label"]]} for row in rows for diff in row["diffs"]
                if (diff < row["floor"][0] or diff > row["floor"][1]) == filled]
    filled_style = {"color": theme["model"]} if dot_opacity is None else {"color": theme["model"], "opacity": dot_opacity}
    return [
        {"type": "scatter", "symbol": "rect", "data": band, "itemStyle": {"color": theme["band"]}, "z": 1, "silent": True,
         "markLine": {"silent": True, "symbol": "none", "label": {"show": False},
                      "lineStyle": {"color": theme["ink3"], "type": "solid", "width": 1}, "data": [{"xAxis": 0}]}},
        {"type": "scatter", "symbolSize": dot_size, "data": dots(True), "itemStyle": filled_style, "z": 3},
        {"type": "scatter", "symbolSize": dot_size - 1, "data": dots(False),
         "itemStyle": {"color": "transparent", "borderColor": theme["model"], "borderWidth": 2}, "z": 3},
    ]


def teaser(theme, _result, measures, seed_values):
    """Two bar panels on the scale from one team repeated to random legal teams."""
    rows = [MODEL, TRAIN, TALL, COPIED, RANDOM]
    colors = [theme["model"], theme["train"], theme["trainall"], theme["anchor"], theme["anchor"]]
    builds_per_species = [mean(seed_values("model_s", "builds_per_species")), mean(seed_values("train_s", "builds_per_species")),
                          mean(seed_values("train_all_s", "builds_per_species")), measures["copied"]["builds_per_species"],
                          measures["random"]["builds_per_species"]]
    nearest = [mean(seed_values("model_s", "nearest_mean")), mean(seed_values("train_s", "nearest_mean")),
               mean(seed_values("train_all_s", "nearest_mean")), measures["copied"]["nearest_mean"], measures["random"]["nearest_mean"]]
    option = base(theme, 436, f"The default model writes {builds_per_species[1] - builds_per_species[0]:.1f} fewer builds per species than win-rate-matched\n"
                              f"training, yet its teams sit {nearest[0] - nearest[1]:.1f} fields further from the closest training team")

    def panel_heading(left, heading, subheading):
        return [{"text": heading, "left": left, "top": 78,
                 "textStyle": {"fontSize": 14, "fontWeight": 600, "color": theme["ink"], "fontFamily": FONT}},
                {"text": subheading, "left": left, "top": 98,
                 "textStyle": {"fontSize": 13, "fontWeight": 400, "color": theme["model"], "fontFamily": FONT}}]
    option["title"] += panel_heading(196, "Builds per species at 30 uses",
                                     f"{builds_per_species[1] - builds_per_species[0]:.1f} fewer than win-rate-matched training")
    option["title"] += panel_heading(512, "Nearest training distance, fields of 84", f"{nearest[0] - nearest[1]:.1f} more than win-rate-matched training")
    option["title"].append({"text": "Model and training bars are means of three sampling seeds or three training draws. One training team repeated\n"
                                    "is left out of its own reference, so its distance is the 17 fields to the next closest training team.",
                            "left": 16, "bottom": 8, "textStyle": {"fontSize": 12, "fontWeight": 400, "color": theme["ink2"], "lineHeight": 16, "fontFamily": FONT}})
    option["grid"] = [{"left": 196, "width": 252, "top": 128, "bottom": 108}, {"left": 512, "width": 218, "top": 128, "bottom": 108}]
    option["xAxis"] = [val_axis(theme, "distinct builds among 30 uses\nhigher means more varied", gridIndex=0, min=0, max=30, interval=10),
                       val_axis(theme, "fields of 84 that differ\nhigher means further from training", gridIndex=1, min=0, max=70, interval=20)]
    option["yAxis"] = [cat_axis(theme, rows, gridIndex=0),
                       cat_axis(theme, rows, gridIndex=1, axisLabel={"show": False})]
    for grid_index, values in ((0, builds_per_species), (1, nearest)):
        option.setdefault("series", []).append({
            "type": "bar", "xAxisIndex": grid_index, "yAxisIndex": grid_index, "barWidth": 18,
            "data": [{"value": round(value, 4), "itemStyle": {"color": color}} for value, color in zip(values, colors)],
            "label": {"show": True, "position": "right", "formatter": "{c}", "color": theme["ink"], "fontSize": 13},
        })
    for series, values in zip(option["series"], (builds_per_species, nearest)):
        for bar, value in zip(series["data"], values):
            bar["label"] = {"formatter": f"{value:.1f}" if value != round(value) else f"{value:.0f}"}
    return option, dict(builds=builds_per_species, nearest=nearest)


def builds(theme, result, _measures, seed_values):
    """Builds per species per seed, against the noise floor around matched training."""
    floor = result["floor"]["builds_per_species"]
    model, train, train_all = (seed_values("model_s", "builds_per_species"), seed_values("train_s", "builds_per_species"),
                               seed_values("train_all_s", "builds_per_species"))
    gap = mean(train) - mean(model)
    low, high = mean(train) + floor["lo"], mean(train) + floor["hi"]
    rows = [MODEL, TRAIN, TALL]
    option = base(theme, 376, f"All three seeds of the default model sit {gap:.1f} builds per species below\n"
                              f"win-rate-matched training, far outside the band two training draws differ by")
    option["grid"] = {"left": 196, "right": 40, "top": 100, "bottom": 112}
    option["xAxis"] = val_axis(theme, "distinct builds among 30 uses of one species, mean over species\nhigher means more varied",
                               min=19, max=27, interval=1)
    option["yAxis"] = cat_axis(theme, rows)

    def value_range(values):
        return f"{min(values):.1f} to {max(values):.1f}" if f"{min(values):.1f}" != f"{max(values):.1f}" else f"{min(values):.1f}"
    series = []
    for name, values, color, side in ((MODEL, model, theme["model"], "left"), (TRAIN, train, theme["train"], "right"),
                                      (TALL, train_all, theme["trainall"], "left")):
        points = [{"value": [round(value, 4), name]} for value in values]
        edge = min(values) if side == "left" else max(values)
        points.append({"value": [round(edge, 4), name], "symbolSize": 0,
                       "label": {"show": True, "position": side, "distance": 10, "formatter": value_range(values),
                                 "color": theme["ink"], "fontSize": 13}})
        series.append({"type": "scatter", "symbolSize": 13, "itemStyle": {"color": color, "opacity": 0.85}, "data": points, "z": 3})
    series[1]["markArea"] = {"silent": True, "itemStyle": {"color": theme["band"]},
                             "label": {"show": True, "position": "insideTop", "color": theme["ink2"], "fontSize": 12, "distance": -18},
                             "data": [[{"name": "noise floor", "xAxis": round(low, 4)}, {"xAxis": round(high, 4)}]]}
    series[0]["markLine"] = {"silent": True, "symbol": ["none", "arrow"], "symbolSize": 9,
                             "lineStyle": {"color": theme["model"], "type": "dashed", "width": 1.5},
                             "label": {"show": True, "position": "middle", "formatter": f"{gap:.1f} fewer", "color": theme["model"],
                                       "fontSize": 13, "distance": 6},
                             "data": [[{"coord": [round(mean(train), 4), MODEL]}, {"coord": [round(max(model) + 0.25, 4), MODEL]}]]}
    option["series"] = series
    option["title"].append({"text": "One dot per sampling seed or training draw. The band is the range of the difference between two "
                                    "training draws, placed\naround win-rate-matched training; a gap counts when all three seeds land outside it.",
                            "left": 16, "bottom": 6, "textStyle": {"fontSize": 12, "fontWeight": 400, "color": theme["ink2"],
                                                                   "lineHeight": 16, "fontFamily": FONT}})
    return option, dict(model=model, train=train, train_all=train_all, gap=gap, band=(low, high))


def novelty(theme, _result, _measures, seed_values):
    """Stacked share of 0, 1, 2 and 3+ species swaps from the closest training line-up."""
    parts = [("novelty_swaps_0", "0 swaps, a training line-up copied"), ("novelty_swaps_1", "1 swap"),
             ("novelty_swaps_2", "2 swaps"), ("novelty_swaps_3plus", "3 or more swaps")]
    rows = [(MODEL, "model_s", theme["model"]), (TRAIN, "train_s", theme["train"]), (TALL, "train_all_s", theme["trainall"])]
    shares = {measure: [mean(seed_values(prefix, measure)) for _, prefix, _ in rows] for measure, _ in parts}
    option = base(theme, 290, f"The default model copies a training line-up {100 * shares['novelty_swaps_0'][0]:.0f}% of the time against "
                              f"{100 * shares['novelty_swaps_0'][1]:.0f}% for\nwin-rate-matched training; the rest goes to 1 and 2 swaps, "
                              f"not to new line-ups")
    option["legend"] = {"top": 66, "left": 196, "itemWidth": 14, "itemHeight": 12, "itemGap": 18,
                        "textStyle": {"color": theme["ink"], "fontSize": 13}}
    option["grid"] = {"left": 196, "right": 76, "top": 104, "bottom": 56}
    option["xAxis"] = val_axis(theme, "", min=0, max=1, interval=0.25, axisLabel={"show": False})
    option["yAxis"] = cat_axis(theme, [{"value": name, "textStyle": {"color": color, "fontWeight": 600}} for name, _, color in rows])
    option["series"] = []
    for part_index, (measure, name) in enumerate(parts):
        last = part_index == len(parts) - 1
        data = []
        for share in shares[measure]:
            show = last or share >= 0.045
            data.append({"value": round(share, 4), "label": {"show": show, "formatter": f"{100 * share:.1f}%"}})
        option["series"].append({"type": "bar", "name": name, "stack": "n", "barWidth": 30,
                                 "itemStyle": {"color": theme[f"s{part_index}"], "borderColor": "transparent"},
                                 "label": {"position": "right" if last else "inside", "fontSize": 13,
                                           "color": theme["ink"] if last else theme["on"][part_index]},
                                 "data": data})
    option["title"].append({"text": "Share of 1,088 teams per row, mean of three sampling seeds or three training draws. Swaps count the species to replace\n"
                                    "to reach the closest training line-up; 3 or more swaps is a new line-up.",
                            "left": 16, "bottom": 6, "textStyle": {"fontSize": 12, "fontWeight": 400, "color": theme["ink2"],
                                                                   "lineHeight": 16, "fontFamily": FONT}})
    return option, shares


def archetypes(theme, result, _measures, seed_values):
    """Model minus matched training per seed, in percentage points, against each archetype's noise floor."""
    names = [("Trick Room", "Trick Room"), ("sun", "Sun"), ("Tailwind", "Tailwind"), ("rain", "Rain")]
    verdict, floor = result["verdict"], result["floor"]
    xmin, xmax, left, width = -10, 14, 110, 440
    right_labels, rows, info = [], [], {}
    for key, label in names:
        measure = f"archetype_{key}"
        model_share, train_share = mean(seed_values("model_s", measure)), mean(seed_values("train_s", measure))
        diffs = [100 * diff for diff in verdict[measure]["diffs"]]
        low, high = 100 * floor[measure]["lo"], 100 * floor[measure]["hi"]
        outside = [diff < low or diff > high for diff in diffs]
        verdict_text = "beyond the floor, every seed" if verdict[measure]["differs"] else f"beyond the floor on {sum(outside)} of 3 seeds"
        right_labels.append(f"{100 * model_share:.1f}% vs {100 * train_share:.1f}%\n{verdict_text}")
        rows.append(dict(label=label, diffs=diffs, floor=(low, high)))
        info[label] = dict(model=model_share, train=train_share, diffs=diffs, floor=(low, high), outside=outside,
                           differs=verdict[measure]["differs"])
    option = base(theme, 402, "Trick Room and sun teams are more common than in win-rate-matched training\n"
                              "on every seed, beyond the noise floor; Tailwind and rain are not")
    option["grid"] = {"left": left, "width": width, "top": 86, "bottom": 112}
    option["xAxis"] = val_axis(theme, "default model minus win-rate-matched training, percentage points", min=xmin, max=xmax, interval=2)
    option["yAxis"] = [cat_axis(theme, [label for _, label in names], axisLabel={"color": theme["ink"], "fontSize": 14, "fontWeight": 600}),
                       cat_axis(theme, right_labels, position="right", axisLine={"show": False},
                                axisLabel={"color": theme["ink2"], "fontSize": 13, "lineHeight": 18, "margin": 16})]
    option["series"] = noise_floor_series(theme, rows, width / (xmax - xmin), dot_size=13)
    option["title"].append({"text": "One dot per sampling seed, filled when it lands outside the grey noise floor. Right: share of teams, "
                                    "default\nmodel against win-rate-matched training, means of three. Setters only: a team counts when any set carries\n"
                                    "the setting move, ability or mega stone.",
                            "left": 16, "bottom": 6, "textStyle": {"fontSize": 12, "fontWeight": 400, "color": theme["ink2"],
                                                                   "lineHeight": 16, "fontFamily": FONT}})
    return option, info


FIGURES = {"teaser": teaser, "builds": builds, "novelty": novelty, "archetypes": archetypes}


def main(out):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    result, measures, seed_values = load()
    shown = {}
    for name, make in FIGURES.items():
        for theme_name, theme in THEMES.items():
            option, numbers = make(theme, result, measures, seed_values)
            (out / f"{name}-{theme_name}.json").write_text(json.dumps(option))
        shown[name] = numbers
    (out / "shown_numbers.json").write_text(json.dumps(shown, indent=1, default=list))
    print(json.dumps(shown, default=list))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "options")
