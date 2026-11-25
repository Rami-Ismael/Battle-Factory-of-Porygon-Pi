"""Dry run of every #431 grid report step before any battle: pairs the old-recipe grids (#385) with copies of
themselves under the shared-recipe row's names, in a temporary results folder, and expects zero differences.

    python gidd_lr_grid_dry_run.py      # prints GIDD_LR_GRID_DRY_RUN_PASS
"""
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gidd_lr_grid_report as G
import retrain_default_report as R


def copy_reference_grids(source, results):
    (results / "pass_at_k").mkdir(parents=True, exist_ok=True)
    for guidance in R.GUIDANCES:
        reference = R.grid_name(R.REF_ROW, guidance)
        for row in (R.NEW_ROW, R.REF_ROW):
            name = R.grid_name(row, guidance)
            for pattern in ("stat_points_84_{}.json", "masking_rate_rows_opponents_{}.npz", "line_up_yield_{}.json", "pass_at_k/{}.json"):
                path = source / pattern.format(reference)
                if path.exists(): shutil.copy(path, results / pattern.format(name))


def main():
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp); results = out / "results"
        copy_reference_grids(R.RES, results)
        R.check_grid(R.grid_name(R.NEW_ROW, 2), 69, results_dir=results)
        for guidance in R.GUIDANCES: G.check_opponents(guidance, results)
        summary = R.build_summary(out_json=out / "summary.json", results_dir=results)
        for guidance in R.GUIDANCES:
            pair = summary["by_guidance"][str(guidance)]
            assert abs(pair["paired"]["error"]["difference"]) < 1e-12, pair["paired"]["error"]
            assert "shared recipe" in G.format_grid_comment(pair, "light.png", "dark.png")
            R.build_grid_figure(pair, out)
        for build in (G.build_error_gap_figure, G.build_tradeoff_figure, G.build_final_figure): build(summary, out)
        assert "within training luck" in G.format_final_comment(summary, "https://example.invalid")
        assert "every measure within training luck" in G.decision_line(summary)
    print("GIDD_LR_GRID_DRY_RUN_PASS")


if __name__ == "__main__":
    main()
