"""Characterization tests for moving charts from matplotlib to ECharts, one script at a time.

Before a script is ported, capture what its matplotlib figures say (words and data points, see src/chart_record.py):

    /tmp/vgc-pilot/.venv/bin/python scripts/test_chart_ports.py --capture remask84_random_figures

That writes scripts/chart_goldens/<script>.json and refuses to overwrite one. Then port the script and run

    /tmp/vgc-pilot/.venv/bin/python scripts/test_chart_ports.py

Each recipe runs the script's plotting code on its saved results. Whatever the script draws, matplotlib figure or
write_chart page, must carry every golden word and point, figure by figure (matched by output file stem).
Recipes read this laptop's results (~/vgc-data, the pilot repo), so the file stays off scripts/ci_tests.txt.
Nothing is written outside a temporary folder: figures headed elsewhere are recorded, not saved.
"""
import argparse
import json
import runpy
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(CODE / "src"), str(CODE / "scripts")]
import chart_record
import echarts_chart

GOLDEN = CODE / "scripts/chart_goldens"
DATA = Path.home() / "vgc-data/results"
PILOT = Path.home() / "Documents/code/vgc-team-generator-pilot/results"


def _load(path):
    return json.loads(Path(path).read_text())


def _remask84_random_figures(tmp):
    import remask84_random_figures as m
    r = _load(DATA / "remask84_random_error.json")
    m.got_against_ask(r, tmp)
    m.error_difference(r, tmp)


def _trade84_storyboard_figures(tmp):
    runpy.run_path(str(CODE / "src/trade84_storyboard_figures.py"), run_name="__main__")


def _plot_random_fill(tmp):
    runpy.run_path(str(CODE / "src/plot_random_fill.py"))


def _plot_asked_vs_got(tmp):
    runpy.run_path(str(CODE / "src/plot_asked_vs_got.py"))


def _plot_dibo_comparison(tmp):
    src = PILOT / "dibo_vs_cem_full_20261012"
    for name in ("protocol.json", "summary.json", "cem.json", "dibo.json"):
        shutil.copy(src / name, tmp / name)  # the script also writes analysis.md beside its inputs
    sys.argv = ["plot_dibo_comparison.py", str(tmp)]
    runpy.run_path(str(CODE / "scripts/plot_dibo_comparison.py"), run_name="__main__")


def _dibo_diversity(script, run):
    def recipe(tmp):
        for name in ("protocol.json", "summary.json", "cem.json", "dibo.json"):
            shutil.copy(PILOT / run / name, tmp / name)  # main() writes diversity/ beside its inputs
        sys.argv = [script, str(tmp)]
        runpy.run_path(str(CODE / "scripts" / script), run_name="__main__")
    return recipe


def _plot_ask_vs_got_loop(tmp):
    real_open = open

    def safe_open(path, mode="r", *args, **kwargs):  # main() rewrites results/ask_vs_got_loop.json; send writes to tmp
        if any(c in mode for c in "wax+"):
            path = tmp / Path(path).name
        return real_open(path, mode, *args, **kwargs)
    runpy.run_path(str(CODE / "src/plot_ask_vs_got_loop.py"), init_globals={"open": safe_open}, run_name="__main__")


def _analyze_temperature_comparison(tmp):
    import analyze_temperature_comparison as m
    m.plot(_load(DATA / "temperature_comparison_parallel.json"), tmp / "temperature_comparison_parallel.png")


def _render_checkpoint_report(tmp):
    import render_checkpoint_report as m
    shutil.copy(CODE / "reports/training-checkpoints-2026-11-05/metrics.json", tmp / "metrics.json")
    m.export_figures(tmp / "metrics.json")


def _report_cem_score_direction(tmp):
    import report_cem_score_direction as m
    m.plot(tmp / "cem_score_direction.json", _load(DATA / "cem_score_direction.json"), _load(DATA / "cem_score_direction.analysis.json"))


def _report_diversity_baseline(tmp):
    import report_diversity_baseline as m
    m.plot(_load(PILOT / "diversity_baseline.json"), _load(PILOT / "temperature_diversity_metrics.json"), tmp / "diversity_before_after.png")


def _reverse_score_diversity(tmp):
    import reverse_score_diversity as m
    shutil.copy(DATA / "reverse_score_diversity.json", tmp / "reverse_score_diversity.json")  # the report hashes it
    m.write_report(tmp / "reverse_score_diversity.json", _load(tmp / "reverse_score_diversity.json"))


def _smoothness_experiment(tmp):
    import smoothness_experiment as m
    src = PILOT / "smoothness_top50_50x50"
    shutil.copy(src / "benchmarks.json", tmp / "benchmarks.json")  # plot() reads it from the output folder for the lower panels
    m.plot(tmp, _load(src / "manifest.json"), _load(src / "analysis.json"))


# script -> (recipe, inputs it reads). Scripts with no recipe:
#   analyze_dibo_diversity_v2.py: no saved run passes its own checks (the full v4 run never chose finalists, the harness run
#     has too few teams). Its plot() calls analyze_dibo_diversity.plot, which the first recipe covers.
RECIPES = {
    "plot_ask_vs_got_loop": (_plot_ask_vs_got_loop, [DATA / "sampler_qd_run.json"]),
    "analyze_dibo_diversity": (_dibo_diversity("analyze_dibo_diversity.py", "dibo_vs_cem_full_20261012"),
                               [PILOT / "dibo_vs_cem_full_20261012/cem.json", Path("/tmp/vgc-pilot/data/pokedex.json")]),
    "remask84_random_figures": (_remask84_random_figures, [DATA / "remask84_random_error.json"]),
    "trade84_storyboard_figures": (_trade84_storyboard_figures, [DATA / "least_likely_trade_storyboard.json"]),
    "plot_random_fill": (_plot_random_fill, [PILOT / "random_fill_baseline.json"]),
    "plot_asked_vs_got": (_plot_asked_vs_got, [PILOT / "asked_vs_got.json"]),
    "plot_dibo_comparison": (_plot_dibo_comparison, [PILOT / "dibo_vs_cem_full_20261012/cem.json"]),
    "analyze_temperature_comparison": (_analyze_temperature_comparison, [DATA / "temperature_comparison_parallel.json"]),
    "render_checkpoint_report": (_render_checkpoint_report, [CODE / "reports/training-checkpoints-2026-11-05/metrics.json"]),
    "report_cem_score_direction": (_report_cem_score_direction, [DATA / "cem_score_direction.analysis.json"]),
    "report_diversity_baseline": (_report_diversity_baseline, [PILOT / "diversity_baseline.json"]),
    "reverse_score_diversity": (_reverse_score_diversity, [DATA / "reverse_score_diversity.json"]),
    "smoothness_experiment": (_smoothness_experiment, [PILOT / "smoothness_top50_50x50/analysis.json"]),
}


def run(name):
    """Run one recipe; return {output stem: record} for every figure or page it produced."""
    recipe, _ = RECIPES[name]
    records, pages = {}, []
    echarts_chart.CAPTURE = pages
    with tempfile.TemporaryDirectory() as folder:
        tmp = Path(folder)
        restore = _record_matplotlib(records, tmp)
        argv = sys.argv[:]
        try:
            recipe(tmp)
        finally:
            sys.argv = argv
            restore()
            echarts_chart.CAPTURE = None
    for path, option, texts, _ in pages:
        records[Path(path).stem] = chart_record.record_option(option, texts)
    return records


def _record_matplotlib(records, tmp):
    """Make Figure.savefig record each figure; save it only when it is headed into tmp."""
    try:
        from matplotlib.figure import Figure  # noqa: TID251 -- reads the old figures this test guards
    except ImportError:
        return lambda: None
    original = Figure.savefig

    def savefig(fig, fname, *args, **kwargs):
        records[Path(str(fname)).stem] = chart_record.record_figure(fig)
        if Path(str(fname)).resolve().is_relative_to(tmp.resolve()):
            original(fig, fname, *args, **kwargs)

    Figure.savefig = savefig
    return lambda: setattr(Figure, "savefig", original)


class ChartPorts(unittest.TestCase):
    def test_every_golden_figure_still_says_the_same(self):
        goldens = sorted(GOLDEN.glob("*.json"))
        self.assertTrue(goldens, "no goldens yet: run --capture first")
        for path in goldens:
            name = path.stem
            with self.subTest(script=name):
                if not all(p.exists() for p in RECIPES[name][1]):
                    self.skipTest(f"{name}: inputs not on this machine")
                golden, now = _load(path), run(name)
                for stem, record in golden.items():
                    self.assertIn(stem, now, f"{name} no longer draws {stem}")
                    words, points = chart_record.missing(record, now[stem])
                    self.assertEqual(words, [], f"{name}/{stem} lost words")
                    self.assertEqual(points[:10], [], f"{name}/{stem} lost {len(points)} data points")


def capture(names):
    GOLDEN.mkdir(exist_ok=True)
    for name in names:
        out = GOLDEN / f"{name}.json"
        if out.exists():
            print(f"{name}: golden exists, left as is"); continue
        missing_inputs = [str(p) for p in RECIPES[name][1] if not p.exists()]
        if missing_inputs:
            print(f"{name}: skipped, missing {missing_inputs}"); continue
        try:
            records = run(name)
        except Exception as e:  # report and go on to the next script
            print(f"{name}: FAILED {type(e).__name__}: {e}"); continue
        out.write_text(json.dumps(records, indent=1, ensure_ascii=False) + "\n")
        print(f"{name}: {len(records)} figures, " + ", ".join(f"{s} ({len(r['texts'])} words, {len(r['points'])} points)" for s, r in records.items()))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    cli.add_argument("--capture", nargs="*", metavar="SCRIPT", help="record goldens for these scripts (none named: all)")
    args, rest = cli.parse_known_args()
    if args.capture is not None:
        capture(args.capture or sorted(RECIPES))
    else:
        unittest.main(argv=[sys.argv[0], *rest])
