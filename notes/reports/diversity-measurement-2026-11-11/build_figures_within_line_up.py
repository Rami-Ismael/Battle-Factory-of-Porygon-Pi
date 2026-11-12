"""Build the figures of the within-line-up comment on map #263 (comment-within-line-up.md) as ECharts options.

    python3 build_figures_within_line_up.py OUT_DIR    writes OUT_DIR/<figure>-<theme>.json, one ECharts 6.1.0 option per file
    node render_svg.mjs OUT_DIR figures                renders each option to figures/<figure>-<theme>.svg

Numbers come only from ~/vgc-data/results/novelty_within_lineup_2026-11-11.json (code/scripts/novelty_within_lineup.py)
and ~/vgc-data/results/stat_points_spent_2026-11-11.json (code/scripts/stat_points_spent.py). Palette, fonts and axes
are those of build_figures.py, so the default model keeps its blue. Every figure is model minus win-rate-matched
training per sampling seed, against the noise floor.
"""
import json
import sys
from pathlib import Path

from build_figures import FONT, THEMES, base, cat_axis, noise_floor_series, val_axis

RESULTS = Path.home() / "vgc-data/results"
NOVELTY = RESULTS / "novelty_within_lineup_2026-11-11.json"
SPENT = RESULTS / "stat_points_spent_2026-11-11.json"
SEEDS = (1, 2, 3)
LEFT, WIDTH = 150, 340


def mean(values): return sum(values) / len(values)


def span(values, digits, unit=""):
    low, high = f"{min(values):.{digits}f}", f"{max(values):.{digits}f}"
    return f"{low}{unit}" if low == high else f"{low}{unit} to {high}{unit}"


def note(theme, text):
    return {"text": text, "left": 16, "bottom": 6,
            "textStyle": {"fontSize": 12, "fontWeight": 400, "color": theme["ink2"], "lineHeight": 16, "fontFamily": FONT}}


def difference_figure(theme, title, rows, axis_name, axis, height, footer):
    """rows: dicts with label, diffs (three seeds), floor (low, high) and right (text). One dot per seed, filled when
    outside the floor; the grey band is the floor; the vertical rule is no difference."""
    xmin, xmax, interval = axis
    option = base(theme, height, title)
    option["grid"] = {"left": LEFT, "width": WIDTH, "top": 86, "bottom": 118}
    labels = [row["label"] for row in rows]
    option["xAxis"] = val_axis(theme, axis_name, min=xmin, max=xmax, interval=interval)
    option["yAxis"] = [cat_axis(theme, labels, axisLabel={"color": theme["ink"], "fontSize": 13, "fontWeight": 600, "lineHeight": 17}),
                       cat_axis(theme, [row["right"] for row in rows], position="right", axisLine={"show": False},
                                axisLabel={"color": theme["ink2"], "fontSize": 13, "lineHeight": 18, "margin": 14})]
    option["series"] = noise_floor_series(theme, rows, WIDTH / (xmax - xmin), dot_size=12, dot_opacity=0.9)
    option["title"].append(note(theme, footer))
    return option


def within_line_up(theme, novelty_result, _spent_result):
    """Builds new to the line-up, the 42 other fields and the 36 Stat Point fields, as a percentage of what each counts."""
    summaries, verdict = novelty_result["summaries"], novelty_result["headline_verdict"]
    spec = [("Builds new to\nthe line-up", "mean_within_line_up_new_builds", 6, "of 6 Pokémon", 2),
            ("Item, ability,\nStat Alignment\nand move fields", "mean_non_stat_point_distance_42", 42, "of 42 fields", 1),
            ("Stat Point fields", "mean_stat_point_distance_36", 36, "of 36 fields", 1)]
    rows, shown = [], {}
    for label, key, count, unit, digits in spec:
        model, train = [summaries[f"model_s{seed}"][key] for seed in SEEDS], [summaries[f"train_s{seed}"][key] for seed in SEEDS]
        high_anchor = summaries["anchor_random_sets_on_training_line_ups"][key]
        scale = 100 / count
        rows.append(dict(label=label, diffs=[diff * scale for diff in verdict[key]["diffs"]],
                         floor=(verdict[key]["floor_lo"] * scale, verdict[key]["floor_hi"] * scale),
                         right=f"{span(model, digits)} vs {span(train, digits)}\n{unit}; random legal sets {high_anchor:.{digits}f}"))
        shown[key] = dict(model=model, train=train, random_sets=high_anchor, diffs=verdict[key]["diffs"],
                          floor=(verdict[key]["floor_lo"], verdict[key]["floor_hi"]), differs=verdict[key]["differs"])
    option = difference_figure(
        theme, "Inside a copied line-up, only Stat Points differ from win-rate-matched\ntraining beyond the noise floor on all three seeds", rows,
        "default model minus win-rate-matched training,\npercent of the Pokémon or fields counted", (-5, 30, 5), 420,
        "One dot per sampling seed, filled when it lands outside the grey noise floor. Right: default model against\n"
        "win-rate-matched training, then training line-ups refilled with random legal sets, the high anchor.\n"
        "Only teams whose line-up copies a training line-up, each compared with the training teams that share it.")
    return option, shown


def totals(theme, _novelty_result, spent_result):
    """Share of Pokémon per Stat Point total, model minus training, percentage points."""
    summaries, verdict = spent_result["summaries"], spent_result["verdict"]
    spec = [("All 66 spent", "share_total_66"), ("60 to 65", "share_total_60_to_65"),
            ("40 to 59", "share_total_40_to_59"), ("0 to 39", "share_total_0_to_39")]
    rows, shown = [], {}
    for label, key in spec:
        model = [100 * summaries[f"model_s{seed}"][key] for seed in SEEDS]
        train = [100 * summaries[f"train_s{seed}"][key] for seed in SEEDS]
        rows.append(dict(label=label, diffs=[100 * diff for diff in verdict[key]["diffs"]],
                         floor=(100 * verdict[key]["floor_lo"], 100 * verdict[key]["floor_hi"]),
                         right=f"{span(model, 1, '%')} vs {span(train, 1, '%')}"))
        shown[key] = dict(model=model, train=train, diffs=verdict[key]["diffs"], floor=(verdict[key]["floor_lo"], verdict[key]["floor_hi"]))
    model_all_spent = shown["share_total_66"]["model"]
    option = difference_figure(
        theme, f"{min(model_all_spent):.0f}% to {max(model_all_spent):.0f}% of the default model's Pokémon spend all 66 Stat Points,\n"
               f"against {mean(shown['share_total_66']['train']):.0f}% in win-rate-matched training", rows,
        "default model minus win-rate-matched training,\npercentage points of Pokémon", (-80, 40, 20), 380,
        "Each Pokémon's six Stat Points summed; 66 is the most Showdown allows. One dot per sampling seed. The noise floor\n"
        "is at most 0.3 points either side, narrower than a dot. Right: share of Pokémon, default model against\n"
        "win-rate-matched training. Every Pokémon of all 1,088 teams per seed, copied line-up or not.")
    return option, shown


def by_stat(theme, _novelty_result, spent_result):
    """Mean Stat Points per stat, model minus training."""
    summaries, verdict = spent_result["summaries"], spent_result["verdict"]
    names = [("HP", "HP"), ("Atk", "Attack"), ("Def", "Defense"), ("SpA", "Sp. Attack"), ("SpD", "Sp. Defense"), ("Spe", "Speed")]
    rows, shown = [], {}
    for stat, label in names:
        key = f"mean_{stat}"
        model, train = [summaries[f"model_s{seed}"][key] for seed in SEEDS], [summaries[f"train_s{seed}"][key] for seed in SEEDS]
        rows.append(dict(label=label, diffs=verdict[key]["diffs"], floor=(verdict[key]["floor_lo"], verdict[key]["floor_hi"]),
                         right=f"{span(model, 1)} vs {span(train, 1)}"))
        shown[key] = dict(model=model, train=train, diffs=verdict[key]["diffs"], floor=(verdict[key]["floor_lo"], verdict[key]["floor_hi"]))
    gap = -mean(shown["mean_Spe"]["diffs"])
    option = difference_figure(
        theme, f"Speed carries the gap: the default model gives it {gap:.1f} fewer Stat Points\n"
               "than win-rate-matched training, and Speed is the last stat it draws", rows,
        "default model minus win-rate-matched training, mean Stat Points", (-14, 8, 2), 440,
        "One dot per sampling seed, filled when it lands outside the grey noise floor. Right: mean Stat Points per\n"
        "Pokémon, default model against win-rate-matched training. The model draws each Pokémon's Stat Points\n"
        "in the order HP, Attack, Defense, Sp. Attack, Sp. Defense, Speed.")
    return option, shown


FIGURES = {"within-line-up": within_line_up, "stat-point-totals": totals, "stat-point-by-stat": by_stat}


def main(out):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    novelty_result, spent_result = json.loads(NOVELTY.read_text()), json.loads(SPENT.read_text())
    shown = {}
    for name, make in FIGURES.items():
        for theme_name, theme in THEMES.items():
            option, numbers = make(theme, novelty_result, spent_result)
            (out / f"{name}-{theme_name}.json").write_text(json.dumps(option))
        shown[name] = numbers
    (out / "shown_numbers.json").write_text(json.dumps(shown, indent=1))
    print(json.dumps(shown))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "options")
