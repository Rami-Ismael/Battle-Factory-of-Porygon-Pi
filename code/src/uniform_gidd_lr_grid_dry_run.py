"""Replay the #438 report chain on stored #277 and #385 results, with no sampling or battles."""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

import uniform_gidd_lr_grid_report as G
import udlm_pass_count84 as U

HERE = Path(__file__).resolve().parent
REFERENCE = "linear_masked_filter"
FIGURE_BASE = "https://example.invalid/figures"


def stage_grid(source, results, name, guidance, passes, *, sizes=None):
    reference = G.R.grid_name(REFERENCE, guidance if guidance != 1 else 2)
    grid = json.loads((source / f"stat_points_84_{reference}.json").read_text())
    grid.update(guidance=guidance, passes=passes)
    snapshot = "masking_rate_rows_opponents_{}.npz"
    if sizes:
        with np.load(source / snapshot.format(reference)) as old:
            arrays = dict(old)
        for cell in grid["cells"].values():
            key = cell["numerator"], cell["seed"]
            if key in sizes:
                size = sizes[key]
                for field in ("pastes", "y", "team_ids", "stat_points"):
                    cell[field] = cell[field][:size]
                cell["battles"] = 49 * size
                arrays[f"a{key[0]:02d}_s{key[1]}"] = arrays[f"a{key[0]:02d}_s{key[1]}"][:size]
        np.savez_compressed(results / snapshot.format(name), **arrays)
    else:
        shutil.copyfile(source / snapshot.format(reference), results / snapshot.format(name))
    (results / f"stat_points_84_{name}.json").write_text(json.dumps(grid))


def replay_existing_reports(source, results):
    import line_up_yield as LY
    import masking_rate_rows as MRR
    import pass_at_k as PAK

    # The read uses stored one-battle opponent scores. The held result below
    # comes from the stored #385 report so the dry run never opens a battle DB.
    LY.RES, LY.TABLE, LY.LABELLED = results, MRR.TABLE, True
    PAK.RES, PAK.OUT_DIR = results, results / "pass_at_k"
    stage_grid(source, results, REFERENCE, 2, 84, sizes={(13, 1): 0})
    subprocess.run(
        [sys.executable, str(HERE / "ask_grid84.py"), "report"],
        env=dict(os.environ, AG84_MODEL=REFERENCE, AG84_GUIDANCE="2", AG84_RESULTS_DIR=str(results)),
        check=True,
    )
    grid = json.loads((results / f"stat_points_84_{REFERENCE}.json").read_text())
    assert grid["summary"]["teams"] == (23 * 3 - 1) * 64
    G.snapshot(results, REFERENCE, grid)
    PAK.read([REFERENCE], allow_short=True)
    pak = json.loads((PAK.OUT_DIR / f"{REFERENCE}.json").read_text())
    assert pak["definition"] == PAK.ONE_SCORE and list(pak["skipped_bands"]) == ["asks_13_to_17"]

    shutil.copyfile(source / f"line_up_yield_{REFERENCE}.json", LY.fresh_path(REFERENCE))
    assert LY.report_model(REFERENCE)["views"]["all_23_asks"]["rebattled"] > 0
    shutil.copyfile(source / "pass_at_k" / f"{REFERENCE}.json", PAK.OUT_DIR / f"{REFERENCE}.json")
    assert G.R._pass_at_k_summary(REFERENCE, results)["pass_at_48"] >= 0


def check_interval_rule():
    points = [
        {"error": x, "novel_share": y, "novel_share_interval": [y - 0.02, y + 0.02]}
        for x, y in ((0.01, 0.4), (0.03, 0.6))
    ]
    candidate = {"error": 0.02, "error_interval": [0.018, 0.022], "novel_share": 0.6, "novel_share_interval": [0.56, 0.64]}
    result = G.compare_point(candidate, points, "novel_share")
    assert result["direction"] == "above" and np.isclose(result["difference"], 0.1)
    assert G.compare_point({**candidate, "novel_share_interval": [0.53, 0.64]}, points, "novel_share") is None
    assert G.compare_point({**candidate, "error_interval": [0.005, 0.022]}, points, "novel_share") is None
    below = {**candidate, "novel_share": 0.4, "novel_share_interval": [0.36, 0.44]}
    assert G.compare_point(below, points, "novel_share")["direction"] == "below"


def replay(source, work, out):
    results = work / "results"
    results.mkdir()
    out.mkdir(parents=True, exist_ok=True)
    for passes in U.PASS_COUNTS:
        shutil.copyfile(U.grid_path(passes, source), U.grid_path(passes, results, G.UNIFORM_ROW))
    pass_report = U.evaluate_pass_counts(results, out / "pass-count.json", model=G.UNIFORM_ROW, issue=438)
    U.build_figures(pass_report, out)
    (out / "pass-count.md").write_text(U.format_markdown_tables(pass_report))
    assert pass_report["decision"]["chosen_passes"] in U.PASS_COUNTS
    replay_existing_reports(source, results)

    # Deliberately exercise a non-default fixed-pass row independently of the old K selection.
    passes = 128
    for row, guidances, count in ((G.uniform_row(passes), G.GUIDANCES, passes), (G.MASKED_ROW, G.R.GUIDANCES, 84)):
        for guidance in guidances:
            sizes = {(1, 1): 0, (1, 2): 17} if guidance == 1 else None
            stage_grid(source, results, G.R.grid_name(row, guidance), guidance, count, sizes=sizes)
    summary = G.build_summary(passes, results)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    for uniform, masked in zip(summary["arms"][G.UNIFORM][1:], summary["arms"][G.MASKED]):
        for metric in (*G.METRICS, "error"):
            assert uniform[metric] == masked[metric], (metric, uniform[metric], masked[metric])
    for point in summary["arms"][G.UNIFORM]:
        G.build_grid_figure(point, out)
        comment = G.format_grid_comment(point, FIGURE_BASE)
        assert comment.startswith("<picture>")
        if point["guidance"] == 1:
            assert "2 cells were short by 111 teams" in comment
            assert point["battles"] == (23 * 3 * 64 - 111) * 49
        else:
            assert "216,384 battles" in comment
        (out / f"grid-g{point['guidance']}.md").write_text(comment)
    for metric in G.METRICS:
        G.build_frontier(summary, metric, out)
    comment = G.format_final_comment(summary, FIGURE_BASE)
    assert comment.startswith("<picture>") and comment.count("<picture>") == len(G.METRICS)
    assert comment.count("establish no point outside") == 1
    (out / "final-comment.md").write_text(comment)
    check_interval_rule()
    print(f"Stored pass-count selection: {pass_report['decision']['chosen_passes']} passes")
    print("Replayed 4 uniform and 3 masked grid points on stored #385 battles")
    print("UNIFORM_GIDD_LR_GRID_DRY_RUN_PASS")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=G.R.RES)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="uniform-gidd-lr-dry-") as tmp:
        work = Path(tmp)
        replay(args.source, work, args.output_dir or work / "report")


if __name__ == "__main__":
    main()
