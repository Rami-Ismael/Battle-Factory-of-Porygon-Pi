"""Issue #438: report shared-recipe ask grids and compare both models at equal error."""
import argparse
import json
from pathlib import Path

import numpy as np

import retrain_default_report as R
from echarts_chart import band, write_chart

UNIFORM_ROW = "udlm_loglinear_gidd_lr"
MASKED_ROW = "shuffled_linear_filtered_gidd_lr"
UNIFORM = "Uniform, no filter"
MASKED = "Masked, full filter"
GUIDANCES = (1, 2, 4, 8)
COLORS = {UNIFORM: "#0f766e", MASKED: "#365d93"}
DARK = {"#0f766e": "#5eead4", "#365d93": "#93c5fd"}
METRICS = {
    "novel_share": ("Novel line-up share", "novel-share", True),
    "distinct_line_ups": ("Distinct line-ups", "distinct-line-ups", False),
    "pooled_win": ("Pooled win rate", "pooled-win", True),
}


def uniform_row(passes):
    return f"{UNIFORM_ROW}_k{passes}"


def read_grid(row, guidance, results_dir, *, passes=None):
    name = R.grid_name(row, guidance)
    R.check_grid(name, len(R.ALL_ASKS) * len(R.SEEDS), results_dir)
    grid = json.loads((results_dir / f"stat_points_84_{name}.json").read_text())
    cells = list(grid["cells"].values())
    expected = {(ask, seed) for ask in R.ALL_ASKS for seed in R.SEEDS}
    if {(c["numerator"], c["seed"]) for c in cells} != expected or any(len(c["pastes"]) != 64 for c in cells):
        raise ValueError(f"{name}: expected asks 1 to 23, seeds 1 to 3, and exactly 64 teams in each cell")
    if grid["guidance"] != guidance or grid.get("unguided") or (passes is not None and grid.get("passes") != passes):
        raise ValueError(f"{name}: unexpected guidance or pass count in the stored grid")
    return name, grid


def snapshot(results_dir, name, grid):
    from masking_rate_rows import team_scores

    path = results_dir / f"masking_rate_rows_opponents_{name}.npz"
    with np.load(path) as data:
        columns, weights = data["columns"], data["weights"]
        if len(set(columns)) != 49 or weights.shape != (49,):
            raise ValueError(f"{name}: expected 49 distinct top-50 opponents")
        for cell in grid["cells"].values():
            rates = data[f"a{cell['numerator']:02d}_s{cell['seed']}"]
            if rates.shape != (64, 49) or not np.allclose(team_scores(rates, weights), cell["y"], atol=1e-9, rtol=0):
                raise ValueError(f"{name}: opponent snapshot differs from stored scores")
    return path, columns, weights


def lineup_intervals(grid, train_lineups, stream_id):
    """Resample teams within each of the fixed ask-by-seed cells."""
    from line_up_yield import line_up

    rng = np.random.default_rng([R.DRAW_SEED, 438, stream_id])
    lineup_ids = {}
    novel = []
    sampled_cells = []
    novel_draws = np.zeros(R.DRAWS)
    for cell in grid["cells"].values():
        lineups = [line_up(paste) for paste in cell["pastes"]]
        ids = [lineup_ids.setdefault(lineup, len(lineup_ids)) for lineup in lineups]
        is_novel = np.array([lineup not in train_lineups for lineup in lineups])
        picks = rng.integers(0, len(lineups), (R.DRAWS, len(lineups)))
        sampled_cells.append(np.asarray(ids, dtype=np.int32)[picks])
        novel.extend(is_novel)
        novel_draws += is_novel[picks].sum(axis=1)

    sampled = np.concatenate(sampled_cells, axis=1)
    sampled.sort(axis=1)
    distinct = 1 + np.count_nonzero(sampled[:, 1:] != sampled[:, :-1], axis=1)
    # Resampling repeats teams and depresses the raw distinct count. Use its centered
    # variation around the observed count to show uncertainty at this sample size.
    distinct_iv = np.clip(len(lineup_ids) + np.asarray(R._interval(distinct - distinct.mean())), 1, len(novel))
    return {
        "novel_share_interval": R._interval(novel_draws / len(novel)),
        "distinct_line_ups_interval": distinct_iv.tolist(),
    }


def measure_grid(row, guidance, results_dir, train_lineups, *, passes=None, opponent_set=None):
    name, grid = read_grid(row, guidance, results_dir, passes=passes)
    path, columns, weights = snapshot(results_dir, name, grid)
    if opponent_set is not None and not all(np.array_equal(a, b) for a, b in zip(opponent_set, (columns, weights))):
        raise ValueError(f"{name}: top-50 opponent IDs or weights differ across grids")
    stream_id = 10 * guidance + int(row == MASKED_ROW)
    counts = R._opponent_draws(len(weights))
    metrics, _ = R.grid_metrics(grid, path, name, guidance, counts, stream_id, train_lineups, results_dir)
    metrics.update(lineup_intervals(grid, train_lineups, stream_id), passes=grid.get("passes", 84))
    return metrics, (columns, weights)


def curve_at(points, metric, error):
    """Interpolate at a supported error. At tied errors, span all of the tied intervals."""
    errors = sorted({p["error"] for p in points})
    if not errors[0] <= error <= errors[-1]:
        return None
    groups = [[p for p in points if p["error"] == x] for x in errors]
    centers = [np.mean([p[metric] for p in group]) for group in groups]
    # Include the observed point when a percentile interval misses its own estimate.
    lower = [min(min(p[metric], p[f"{metric}_interval"][0]) for p in group) for group in groups]
    upper = [max(max(p[metric], p[f"{metric}_interval"][1]) for p in group) for group in groups]
    return (
        float(np.interp(error, errors, centers)),
        float(np.interp(error, errors, lower)),
        float(np.interp(error, errors, upper)),
    )


def compare_point(point, other_points, metric):
    """Require the point's interval to clear the other band across its full error interval."""
    x = point["error"]
    left, right = point["error_interval"]
    left, right = min(left, x), max(right, x)
    knots = [left, right, *(p["error"] for p in other_points if left < p["error"] < right)]
    comparisons = [curve_at(other_points, metric, at) for at in knots]
    if any(value is None for value in comparisons):
        return None
    low, high = point[f"{metric}_interval"]
    low, high = min(low, point[metric]), max(high, point[metric])
    lower_gap = low - max(value[2] for value in comparisons)
    upper_gap = min(value[1] for value in comparisons) - high
    if max(lower_gap, upper_gap) <= 0:
        return None
    estimate = curve_at(other_points, metric, x)[0]
    return dict(
        difference=point[metric] - estimate,
        clearance=max(lower_gap, upper_gap),
        direction="above" if lower_gap > 0 else "below",
    )


def build_summary(passes, results_dir):
    train_lineups = R._train_lineups()
    arms = {}
    opponent_set = None
    for label, row, guidances in ((UNIFORM, uniform_row(passes), GUIDANCES), (MASKED, MASKED_ROW, R.GUIDANCES)):
        arms[label] = []
        for guidance in guidances:
            point, opponent_set = measure_grid(
                row, guidance, results_dir, train_lineups,
                passes=passes if label == UNIFORM else None, opponent_set=opponent_set,
            )
            arms[label].append(point)
    return {"uniform_passes": passes, "arms": arms}


def line_series(name, color, data, **options):
    return {
        "name": name, "type": "line", "color": color, "data": data, "symbolSize": 7,
        "lineStyle": {"width": 2.5}, "emphasis": {"focus": "series"}, **options,
    }


def axis(name, **options):
    return {"type": "value", "name": name, "nameLocation": "middle", "nameGap": 36, **options}


def build_grid_figure(metrics, out_dir):
    guidance = metrics["guidance"]
    xs = [ask / 24 for ask in R.ALL_ASKS]
    got = metrics["got_by_ask"]
    intervals = [metrics["got_by_ask_interval"][str(ask)] for ask in R.ALL_ASKS]
    color = COLORS[UNIFORM]
    series = band(UNIFORM, color, xs, [v[0] for v in intervals], [v[1] for v in intervals], "uniform")
    series += [
        line_series(UNIFORM, color, [[x, got[str(ask)]] for x, ask in zip(xs, R.ALL_ASKS)]),
        line_series("Got = ask", "#6f6e69", [[0, 0], [1, 1]], symbol="none", silent=True,
                    lineStyle={"type": "dotted", "width": 1.5}),
    ]
    option = {
        "animation": False,
        "grid": {"left": 78, "right": 30, "top": 42, "bottom": 92},
        "legend": {"data": [UNIFORM, "Got = ask"]},
        "tooltip": {"trigger": "axis", "valueFormatter": R.PERCENT_TOOLTIP},
        "xAxis": axis("Requested win rate", min=0, max=1, axisLabel={"formatter": R.PERCENT_AXIS}),
        "yAxis": axis("Measured win rate", min=0, max=1, nameGap=52, axisLabel={"formatter": R.PERCENT_AXIS}),
        "dataZoom": [{"type": "inside"}, {"type": "slider", "bottom": 12, "height": 16}],
        "series": series,
    }
    return write_chart(
        option, out_dir / f"grid-g{guidance}.html", title=f"Uniform, no filter at guidance {guidance}",
        subtitle=f"Shared recipe, {metrics['passes']} passes. Shading shows 95% bootstrap intervals.",
        note="The error averages the squared gaps on asks 1/24 to 17/24.", dark=DARK,
    )


def whiskers(points, metric, color):
    segments = []
    for point in points:
        x, y = point["error"], point[metric]
        x_low, x_high = point["error_interval"]
        y_low, y_high = point[f"{metric}_interval"]
        segments.extend([
            [{"coord": [x_low, y]}, {"coord": [x_high, y]}],
            [{"coord": [x, y_low]}, {"coord": [x, y_high]}],
        ])
    return {
        "symbol": ["none", "none"], "silent": True, "animation": False,
        "label": {"show": False}, "tooltip": {"show": False},
        "lineStyle": {"color": color, "opacity": 0.65, "width": 1.4, "type": "solid"}, "data": segments,
    }


def build_frontier(summary, metric, out_dir):
    label, stem, percent = METRICS[metric]
    series = []
    points = [p for arm in summary["arms"].values() for p in arm]
    x_min = min(min(p["error"], *p["error_interval"]) for p in points)
    x_max = max(max(p["error"], *p["error_interval"]) for p in points)
    margin = max((x_max - x_min) * 0.09, 0.00005)
    for arm, arm_points in summary["arms"].items():
        ordered = sorted(arm_points, key=lambda p: p["error"])
        color = COLORS[arm]
        series.append(line_series(
            arm, color, [[p["error"], p[metric], p["guidance"]] for p in ordered],
            symbol="circle" if arm == UNIFORM else "diamond", markLine=whiskers(ordered, metric, color),
            dimensions=["Ask-versus-got error", label, "Guidance"],
            encode={"x": 0, "y": 1, "tooltip": [1, 2]},
            label={"show": True, "position": "top" if arm == UNIFORM else "bottom",
                   "formatter": "js:p => 'g=' + p.value[2]", "color": color},
        ))
    option = {
        "animation": False,
        "grid": {"left": 83, "right": 34, "top": 45, "bottom": 61},
        "legend": {"top": 4},
        "tooltip": {"trigger": "axis"},
        "xAxis": axis("Ask-versus-got error (mean squared gap)", min=max(0, x_min - margin), max=x_max + margin,
                      axisLabel={"formatter": "js:v => Number(v).toFixed(4)"}),
        "yAxis": axis(label, min=0, max=1 if percent else 4600, nameGap=54,
                      axisLabel={"formatter": R.PERCENT_AXIS} if percent else {}),
        "series": series,
    }
    return write_chart(
        option, out_dir / f"{stem}.html", title=f"{label} at equal ask-versus-got error",
        subtitle="Both models use the shared recipe. Point labels give each model's guidance weight.",
        note="Whiskers show 95% bootstrap intervals. Error uses asks 1/24 to 17/24. The vertical measure pools all 23 asks.",
        dark=DARK,
    )


def picture(base, stem, alt):
    return R.picture(f"{base}/{stem}-light.png", f"{base}/{stem}-dark.png", alt)


def fmt_metric(value, percent):
    return f"{value:.1%}" if percent else f"{value:,.0f}"


def format_grid_comment(metrics, base):
    guidance = metrics["guidance"]
    return "\n".join([
        picture(base, f"grid-g{guidance}", f"Uniform, no filter at guidance {guidance}, measured against requested win rate"),
        "",
        f"Uniform, no filter used {metrics['passes']} passes with the shared recipe at guidance {guidance}.",
        f"The grid scored {metrics['teams']:,} teams across {metrics['cells']} ask-by-seed cells in {metrics['battles']:,} battles.",
        "",
        "| Measure | Result |",
        "| --- | --- |",
        f"| Ask-versus-got error (asks 1/24 to 17/24) | {metrics['error']:.4f} {R.fmt_iv(metrics['error_interval'])} |",
        f"| Novel line-up share | {metrics['novel_share']:.1%} {R.fmt_iv(metrics['novel_share_interval'], '.1%')} |",
        f"| Distinct line-ups | {metrics['distinct_line_ups']:,} {R.fmt_iv(metrics['distinct_line_ups_interval'], ',.0f')} |",
        f"| First-try legality | {metrics['first_try_legality']:.1%} |",
        f"| Pooled win rate | {metrics['pooled_win']:.1%} {R.fmt_iv(metrics['pooled_win_interval'], '.1%')} |",
        "",
        "Line-up measures and pooled win rate use all 23 asks. Brackets show 95% bootstrap intervals.",
        "",
    ])


def format_verdict(arms, metric):
    label, _, percent = METRICS[metric]
    ranges = [sorted(p["error"] for p in points) for points in arms.values()]
    if max(values[0] for values in ranges) > min(values[-1] for values in ranges):
        return ["- The measured error ranges are disjoint."]
    findings = []
    for arm, other in ((UNIFORM, MASKED), (MASKED, UNIFORM)):
        for point in arms[arm]:
            verdict = compare_point(point, arms[other], metric)
            if verdict is None:
                continue
            gap = abs(verdict["difference"])
            amount = f"{100 * gap:.1f} percentage points" if percent else f"{gap:,.0f} line-ups"
            findings.append(
                f"- {arm} at guidance {point['guidance']} sits **{amount} {verdict['direction']}** "
                f"the {other} curve at error {point['error']:.4f}."
            )
    return findings or [f"- The intervals establish no point outside the other curve for {label.lower()}."]


def format_final_comment(summary, base):
    lines = [picture(base, stem, f"{label} at equal ask-versus-got error") for label, stem, _ in METRICS.values()]
    lines += ["", f"Uniform, no filter used {summary['uniform_passes']} passes. Both models used the shared learning-rate recipe."]
    for metric, (label, _, _) in METRICS.items():
        lines += ["", f"**{label}**", "", *format_verdict(summary["arms"], metric)]
    lines += [
        "",
        "Comparisons use only the shared range of measured errors. A point clears the other curve when its "
        "vertical 95% interval clears the interpolated interval band over the point's entire horizontal interval. "
        "Reported gaps use the point estimates at equal error. Straight segments join measured points.",
        "",
        "| Model | Guidance | Error [95% interval] | Novel share | Distinct line-ups | Pooled win rate | Battles |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for arm, points in summary["arms"].items():
        for p in points:
            values = " | ".join(fmt_metric(p[key], percent) for key, (_, _, percent) in METRICS.items())
            lines.append(f"| {arm} | {p['guidance']} | {p['error']:.4f} {R.fmt_iv(p['error_interval'])} | {values} | {p['battles']:,} |")
    lines += [
        "",
        "Each checkpoint has one training seed, and both reached the 80-epoch cap while still improving. "
        "The guidance weights act on different model outputs and have different scales. "
        "The intervals describe resampling within the three observed sampling seeds. Error and win rate also resample the 49 "
        "distinct top-50 opponents. Distinct-count intervals use centered bootstrap variation around the observed count.",
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=R.RES)
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("check-grid")
    check.add_argument("name")
    check.add_argument("cells", type=int)
    grid = commands.add_parser("grid")
    grid.add_argument("passes", type=int)
    grid.add_argument("guidance", type=int)
    grid.add_argument("figure_dir", type=Path)
    summary = commands.add_parser("summary")
    summary.add_argument("passes", type=int)
    summary.add_argument("output", type=Path)
    final = commands.add_parser("final-figures")
    final.add_argument("summary", type=Path)
    final.add_argument("figure_dir", type=Path)
    for command in ("grid-comment", "final-comment"):
        sub = commands.add_parser(command)
        sub.add_argument("data", type=Path)
        sub.add_argument("figure_base")
    args = parser.parse_args()
    if args.command == "check-grid":
        R.check_grid(args.name, args.cells, args.results_dir)
    elif args.command == "grid":
        metrics, _ = measure_grid(
            uniform_row(args.passes), args.guidance, args.results_dir, R._train_lineups(), passes=args.passes,
        )
        build_grid_figure(metrics, args.figure_dir)
        (args.figure_dir / f"grid-g{args.guidance}.json").write_text(json.dumps(metrics, indent=2))
    elif args.command == "summary":
        args.output.write_text(json.dumps(build_summary(args.passes, args.results_dir), indent=2))
    elif args.command == "final-figures":
        data = json.loads(args.summary.read_text())
        for metric in METRICS:
            print(build_frontier(data, metric, args.figure_dir))
    else:
        data = json.loads(args.data.read_text())
        formatter = format_grid_comment if args.command == "grid-comment" else format_final_comment
        print(formatter(data, args.figure_base), end="")


if __name__ == "__main__":
    main()
