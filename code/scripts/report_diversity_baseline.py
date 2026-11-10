"""Verify and compare the no-fine-tuning baseline with saved campaign metrics."""
import argparse
import json
import math
from pathlib import Path
import statistics
import sys

from capture_diversity_baseline import verify_baseline
from temperature_experiment import REPO, digest


def compare(baseline, history):  # noqa: C901
    from corpus import parse_team_text
    if baseline.get("status") != "complete":
        raise ValueError("baseline is not complete")
    verify_baseline(baseline, lambda pastes: [parse_team_text(p) for p in pastes])
    manifest = baseline["manifest"]
    if history["source_sha256"] != manifest["source_campaign_sha256"]:
        raise ValueError("historical campaign identity mismatch")
    if digest(history["source"]) != history["source_sha256"]:
        raise ValueError("historical campaign file changed")
    if digest(REPO / "src/diversity_metrics.py") != history["metric_source_sha256"]:
        raise ValueError("historical report uses a different metric implementation")
    if history["reference"]["content_sha256"] != manifest["reference"]["metadata"]["content_sha256"]:
        raise ValueError("baseline and history use different novelty references")
    seeds = manifest["config"]["seeds"]
    original = json.loads(Path(history["source"]).read_text())
    arms = sorted(original["manifest"]["arms"])
    generations = manifest["config"]["generations"]
    expected = {(seed, arm, g, "generation") for seed in seeds for arm in arms for g in range(1, generations + 1)}
    expected |= {(seed, arm, generations, "holdout") for seed in seeds for arm in arms}
    keys = [(r["seed"], r["arm"], r["generation"], r["phase"]) for r in history["rows"]]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError("historical cells incomplete or duplicated")
    for row in history["rows"]:
        expected_n = manifest["config"]["propose" if row["phase"] == "generation" else "final_sample"]
        if row["metrics"]["sampling"]["accepted"] != expected_n:
            raise ValueError("historical and baseline sample sizes differ")
    rows = []
    for seed in seeds:
        row = baseline["runs"][str(seed)]
        scores = row["scores"]
        rows.append(dict(arm="before fine-tuning", seed=seed,
            proposal=row["proposal"]["diversity"]["team_diversity"],
            holdout=row["holdout"]["diversity"]["team_diversity"],
            win_rate=sum(s["wins"] for s in scores) / sum(s["battles"] for s in scores)))
        for arm in arms:
            proposal = next(r for r in history["rows"] if (r["seed"], r["arm"], r["generation"], r["phase"]) == (seed, arm, generations, "generation"))
            holdout = next(r for r in history["rows"] if (r["seed"], r["arm"], r["phase"]) == (seed, arm, "holdout"))
            rows.append(dict(arm=arm, seed=seed, proposal=proposal["metrics"],
                             holdout=holdout["metrics"], win_rate=holdout["win_rate"]["mean"]))
    return rows


def plot(baseline, history, path):
    """Composition counts and unique-team share over CEM generations, as one ECharts page beside path (.html)."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from echarts_chart import write_chart
    config = baseline["manifest"]["config"]
    seeds = config["seeds"]
    arms = sorted({r["arm"] for r in history["rows"]})
    palette = dict(adaptive="#A05A00", annealed="#007A78", fixed_010="#A72D59",
                   fixed_020="#7250A1", fixed_030="#2965AF")
    labels = dict(adaptive="Adaptive", annealed="Annealed", fixed_010="Fixed 0.10",
                  fixed_020="Fixed 0.20", fixed_030="Fixed 0.30")
    ticks = [0, 1, 3, 5, 7, 9, 11]
    panels = (("composition", "unique_count", 1, "Distinct six-species/form compositions", "Count per 512 accepted draws", 530),
              ("with_spreads", "unique_fraction", 100, "Unique teams including stat spreads", "Percent of 512 accepted draws", 105))
    grids = [{"left": "7%", "width": "40%", "top": 50, "bottom": 70}, {"left": "57%", "width": "40%", "top": 50, "bottom": 70}]
    titles, xs, ys, series = [], [], [], []
    for g, (key, metric, scale, title, ylabel, ymax) in enumerate(panels):
        on = {"xAxisIndex": g, "yAxisIndex": g}
        titles.append({"text": title, "left": grids[g]["left"], "top": 12, "textStyle": {"fontSize": 13}})
        xs.append({"type": "value", "gridIndex": g, "min": 0, "max": config["generations"], "name": "CEM generation (0 = before fine-tuning)",
                   "nameLocation": "middle", "nameGap": 26, "axisLabel": {"customValues": ticks}, "axisTick": {"customValues": ticks}})
        ys.append({"type": "value", "gridIndex": g, "min": 0, "max": ymax, "name": ylabel, "nameLocation": "middle", "nameGap": 40})
        start = statistics.mean(baseline["runs"][str(seed)]["proposal"]["diversity"]["team_diversity"][key][metric] for seed in seeds)
        for arm in arms:
            values = [start]
            for generation in range(1, config["generations"] + 1):
                values.append(statistics.mean(r["metrics"][key][metric] for r in history["rows"]
                    if r["arm"] == arm and r["generation"] == generation and r["phase"] == "generation"))
            series.append({"type": "line", "name": labels[arm], "color": palette[arm], "lineStyle": {"width": 2}, "symbolSize": 5,
                           "emphasis": {"focus": "series"}, "data": [[i, v * scale] for i, v in enumerate(values)], **on})
        series.append({"type": "scatter", "name": "Before fine-tuning", "color": "#222222", "symbolSize": 10, "data": [[0, start * scale]], **on})
    if all(r["metrics"]["with_spreads"]["unique_fraction"] == 1 for r in history["rows"] if r["phase"] == "generation"):
        series.append({"type": "scatter", "silent": True, "symbolSize": 0, "xAxisIndex": 1, "yAxisIndex": 1, "tooltip": {"show": False},
                       "data": [[config["generations"] / 2, 65]],
                       "label": {"show": True, "formatter": "All five arms remain at 100%", "color": "#40516a", "fontSize": 13}})
    write_chart({"title": titles, "grid": grids, "xAxis": xs, "yAxis": ys, "series": series, "tooltip": {"trigger": "axis"},
                 "legend": {"data": [labels[a] for a in arms] + ["Before fine-tuning"], "bottom": 0}},
                path.with_suffix(".html"), "Diversity before and across fine-tuning",
                note="Means across three seeds; repeated draws retained. No baseline generator fine-tuning.", height=440,
                dark={"#222222": "#f2f1ee", "#40516a": "#a9b4be"})


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--baseline", type=Path, required=True)
    cli.add_argument("--history", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    baseline = json.loads(args.baseline.read_text())
    history = json.loads(args.history.read_text())
    rows = compare(baseline, history)
    config = baseline["manifest"]["config"]
    image = args.output.with_suffix(".html")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plot(baseline, history, image)
    lines = ["# Diversity before and across fine-tuning", "",
        "The original checkpoint was evaluated without generator fine-tuning, using the completed campaign's "
        "decoding/guidance settings, initial surrogate-label pool, opponents, battle policy and package versions. "
        "The baseline was captured retrospectively and is saved separately; its seeds and budgets are matched.", "",
        f"Each of {len(config['seeds'])} seeds supplies {config['propose']} accepted proposals and an independent "
        f"{config['final_sample']}-team unranked sample evaluated for {config['final_battles']} battles per team. "
        f"All {baseline['manifest']['expected_battles']:,} baseline battles completed. "
        "The initial training reference contains 692 teams verified against the checkpoint and source files.", "",
        f"[Diversity trajectories]({image.resolve()})", "",
        "| Model | Distinct compositions / 512 | Unique teams / 512 | Novel teams / 512 | Generator win rate |",
        "| --- | ---: | ---: | ---: | ---: |"]
    order = ["before fine-tuning", *sorted({r["arm"] for r in rows if r["arm"] != "before fine-tuning"})]
    for arm in order:
        selected = [r for r in rows if r["arm"] == arm]
        comp = statistics.mean(r["proposal"]["composition"]["unique_count"] for r in selected)
        unique = statistics.mean(r["proposal"]["with_spreads"]["unique_fraction"] for r in selected)
        novel = statistics.mean(r["proposal"]["with_spreads"]["novel_draw_fraction"] for r in selected)
        win = statistics.mean(r["win_rate"] for r in selected)
        lines.append(f"| {arm} | {comp:.1f} | {unique:.1%} | {novel:.1%} | {win:.2%} |")
    lines += ["", "## Where concentration occurs", "",
        "These additional measurements use the same 512-proposal streams. The effective composition count "
        "is exp(−Σ p ln p), where p is each composition's observed fraction of draws. It decreases when draws "
        "concentrate on a few rosters even if many rare rosters remain.", "",
        "| Model | Draws with unseen compositions | Largest roster share | Effective compositions |",
        "| --- | ---: | ---: | ---: |"]
    for arm in order:
        group = [r["proposal"]["composition"] for r in rows if r["arm"] == arm]
        novelty = statistics.mean(m["novel_draw_fraction"] for m in group)
        largest = statistics.mean(m["largest_share"] for m in group)
        effective = []
        for m in group:
            probabilities = [entry["count"] / m["n"] for entry in m["frequencies"]]
            effective.append(math.exp(-sum(p * math.log(p) for p in probabilities)))
        lines.append(f"| {arm} | {novelty:.1%} | {largest:.1%} | {statistics.mean(effective):.1f} |")
    lines += ["", "Values are means over the same three seeds. Post-fine-tuning diversity is measured at generation 11; "
        "win rates come from independent, unranked 128-team samples before and after fine-tuning. "
        "CEM also refits the guidance surrogate, so changes describe the whole proposal procedure rather than isolating generator training. "
        "These descriptive means do not establish statistical significance or new strategic behavior. "
        "The plot tracks composition counts because full-team uniqueness can remain high while roster variety declines.", "",
        f"[Baseline draws and battle counts]({args.baseline.resolve()}) · "
        f"[All historical metric records]({args.history.resolve()})", ""]
    args.output.write_text("\n".join(lines))
    args.output.with_suffix(".json").write_text(json.dumps(dict(
        baseline_sha256=digest(args.baseline), history_sha256=digest(args.history),
        report_script_sha256=digest(__file__), rows=rows), indent=2, allow_nan=False) + "\n")
    print(f"Verified baseline and wrote {args.output}")


if __name__ == "__main__":
    main()
