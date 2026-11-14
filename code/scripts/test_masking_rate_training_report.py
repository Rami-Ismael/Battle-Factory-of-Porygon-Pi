"""The report run after each masking-schedule training (src/masking_rate_training_report.py; docs/masking-rate-rows.md; issue #296).

No test calls a training entry point: train84.main runs only with train, load and check_rows replaced. The gap and
cross-rate tests score a stored checkpoint, stat84_pos_per_field_s0_noise_ep21, on small slices.

Needs this laptop's /tmp/vgc-pilot data; skipped without it.

    /tmp/vgc-pilot/.venv/bin/python scripts/test_masking_rate_training_report.py   (from code/)
"""
import ast
import hashlib
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

import repo_paths  # noqa: F401  (src/ first on sys.path)

HERE = Path("/tmp/vgc-pilot/data/pokedex.json").exists()
if HERE:
    sys.path.insert(1, "/tmp/vgc-pilot/src")
    import numpy as np
    import torch

    import masking_rate_training_report as R
    import train84 as TR

STORED = "stat84_pos_per_field_s0_noise_ep21"
PER_FIELD, LINEAR = "stat84_shuffled_noise_per_field_es3_cap80", "stat84_shuffled_noise_linear_es3_cap80"


def fake_table():
    labels = np.array([0.5, np.nan, 0.25, 0.75, np.nan, 0.1, 0.6, 0.3])
    return dict(all_X=np.arange(8 * 84).reshape(8, 84), all_W=labels,
                all_val=np.array([False, False, False, True, True, False, True, False]),
                sha256="abc", table="lossdata84_shuffled_only.npz")


def source_functions():
    """terms, summary, sizes and gap from the source file, compiled on their own with this repo's diffusion84 as M.
    Importing the source by path is unsafe: at import it reads sys.argv[1], puts the frozen run folder of another queue
    first on sys.path and imports that folder's diffusion84. So only the four function definitions are taken from its
    text, after its sha256 is checked, and run beside the port with the same M, so the test compares the functions."""
    text = R.SOURCE.read_bytes()
    assert hashlib.sha256(text).hexdigest() == R.SOURCE_SHA256, "the source file changed since the port"
    tree = ast.parse(text)
    keep = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in ("terms", "summary", "sizes", "gap")]
    assert len(keep) == 4
    namespace = dict(np=np, torch=torch, M=R.M, DEV=R.M.DEV, DRY=True)
    exec(compile(ast.Module(body=keep, type_ignores=[]), str(R.SOURCE), "exec"), namespace)
    return namespace


@unittest.skipUnless(HERE, "needs /tmp/vgc-pilot data")
class RowSplit(unittest.TestCase):
    def test_the_split_is_the_one_the_masking_rate_trainings_use(self):
        table = fake_table()
        with (mock.patch.object(TR, "train") as train, mock.patch.object(TR, "load", return_value=table),
              mock.patch.object(TR, "check_rows"), redirect_stdout(io.StringIO())):
            TR.main(["shuffled_linear"])
        _arm, rows, labels, (val_rows, val_labels) = train.call_args.args
        training, validation = R.labelled_split(table)
        np.testing.assert_array_equal(table["all_X"][training], rows)
        np.testing.assert_array_equal(table["all_W"][training], labels)
        np.testing.assert_array_equal(table["all_X"][validation], val_rows)
        np.testing.assert_array_equal(table["all_W"][validation], val_labels)

    def test_the_real_table_has_4380_labelled_validation_teams(self):
        table = R.load_table()
        training, validation = R.labelled_split(table)
        self.assertEqual(int(validation.sum()), 4380)
        self.assertFalse((training & validation).any())
        self.assertFalse(np.isnan(table["all_W"][training | validation]).any())

    def test_sizes_takes_as_many_training_teams_as_validation_teams_like_the_source(self):
        source = source_functions()
        table = R.load_table()
        training, validation = R.labelled_split(table)
        source_training, source_validation = source["sizes"](training, validation)       # DRY=True: the first 64 of each
        ported_training, ported_validation = R.sizes(training, validation)
        np.testing.assert_array_equal(ported_training[:64], source_training)
        np.testing.assert_array_equal(ported_validation[:64], source_validation)
        self.assertEqual(len(ported_training), len(ported_validation))


@unittest.skipUnless(HERE, "needs /tmp/vgc-pilot data")
class OnTheStoredCheckpoint(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = R.load_model(STORED)
        cls.table = R.load_table()
        cls.training, cls.validation = R.labelled_split(cls.table)

    def test_the_ported_gap_matches_the_source_function(self):
        source = source_functions()
        training_index, validation_index = (index[:64] for index in R.sizes(self.training, self.validation))
        exponents = R.masking_rate_tensor(self.model)
        draws = 4
        theirs = source["gap"](self.model, self.model.net, exponents, self.table["all_X"], self.table["all_W"],
                               training_index, validation_index, draws)
        mine = R.gap(self.model, self.model.net, exponents, self.table["all_X"], self.table["all_W"],
                     training_index, validation_index, draws)
        self.assertEqual(mine.keys(), theirs.keys())
        for key in theirs:
            for side in ("training", "validation"):
                for field in ("mean", "se"):
                    self.assertAlmostEqual(mine[key][side][field], theirs[key][side][field], delta=1e-6)
            self.assertAlmostEqual(mine[key]["gap"], theirs[key]["gap"], delta=1e-6)
            self.assertAlmostEqual(mine[key]["gap_se"], theirs[key]["gap_se"], delta=1e-6)
            self.assertEqual(mine[key]["validation_teams"], theirs[key]["validation_teams"])

    def test_cross_rate_also_scores_without_the_teams_the_tables_exclude(self):
        import math
        result = R.cross_rate(self.model, self.table, self.validation, limit=96)
        accepted, teams = R.comparable_losses(result)
        self.assertEqual(teams, result["validation_teams_tables_accepted"])
        self.assertLessEqual(teams, result["validation_teams"])
        self.assertTrue(all(math.isfinite(accepted[key]) for key in R.RATES))
        self.assertEqual(R.comparable_losses(dict(validation_loss=dict(per_field=1.0, linear=2.0), validation_teams=7)),
                         (dict(per_field=1.0, linear=2.0), 7))

    def test_cross_rate_gives_two_different_finite_losses(self):
        result = R.cross_rate(self.model, self.table, self.validation, limit=96)
        per_field, linear = result["validation_loss"]["per_field"], result["validation_loss"]["linear"]
        self.assertTrue(np.isfinite(per_field) and np.isfinite(linear))
        self.assertNotAlmostEqual(per_field, linear, delta=1e-3)
        self.assertEqual(result["validation_teams"], 96)

    def test_the_own_rate_check_passes_and_fails_correctly(self):
        result = R.cross_rate(self.model, self.table, self.validation, limit=64)
        own = result["validation_loss"]["per_field"]
        self.assertEqual(result["own_masking_rate"], "per_field")
        self.assertTrue(R.own_rate_check(result, dict(best_val_loss=own + 5e-5))["passed"])
        failed = R.own_rate_check(result, dict(best_val_loss=own + 0.01))
        self.assertFalse(failed["passed"])
        self.assertAlmostEqual(failed["difference"], -0.01, delta=1e-9)
        self.assertFalse(R.own_rate_check(result, {})["passed"])

    def test_a_mismatch_is_written_and_exits_non_zero(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            with mock.patch.object(R, "load_model", return_value=self.model), self.assertRaises(SystemExit) as stop:
                R.main(["cross_rate", STORED, "--limit", "64", "--results-dir", tmp])
            self.assertNotEqual(stop.exception.code, 0)
            written = json.loads((Path(tmp) / f"masking_rate_cross_rate_{STORED}.json").read_text())
        self.assertFalse(written["own_rate_check"]["passed"])
        self.assertEqual(written["validation_teams"], 64)


def write_fakes(folder, with_linear=True):
    folder = Path(folder)
    runs = {PER_FIELD: dict(best=4, stopped=7, minutes=401.5, val=[1.31, 1.27, 1.2601, 1.2543, 1.2551, 1.2560, 1.2577], sched="16,4,4,1.5,1"),
            LINEAR: dict(best=5, stopped=8, minutes=388.2, val=[1.52, 1.47, 1.4512, 1.4433, 1.4391, 1.4402, 1.4410, 1.4415], sched="1,1,1,1,1")}
    cross = {PER_FIELD: dict(per_field=1.2543, linear=1.4471), LINEAR: dict(per_field=1.2987, linear=1.4391)}
    gaps = {PER_FIELD: (0.0123, 0.0061), LINEAR: (0.0047, 0.0058)}
    for name in runs if with_linear else [PER_FIELD]:
        run = runs[name]
        curves = [dict(epoch=e + 1, train_loss=v + 0.1, val_loss=v, minutes=50.0 * (e + 1)) for e, v in enumerate(run["val"])]
        (folder / f"{name}.json").write_text(json.dumps(dict(curves=curves, best_epoch=run["best"], stopped_epoch=run["stopped"],
                                                             best_val_loss=run["val"][run["best"] - 1], sched=run["sched"], minutes=run["minutes"])))
        (folder / f"masking_rate_cross_rate_{name}.json").write_text(json.dumps(dict(validation_loss=cross[name], validation_teams=4380)))
        gap, se = gaps[name]
        side = dict(training=dict(mean=1.2, se=0.004), validation=dict(mean=1.2 + gap, se=0.004), gap=gap, gap_se=se, validation_teams=4380)
        (folder / f"masking_rate_gap_{name}.json").write_text(json.dumps(dict(rule_table_teams_removed=side, all_teams=side, teams_each_side=4380)))


@unittest.skipUnless(HERE, "needs /tmp/vgc-pilot data")
class FiguresAndSummary(unittest.TestCase):
    def test_figures_hold_the_reader_names_and_the_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_fakes(tmp)
            out = Path(tmp) / "figures"
            with redirect_stdout(io.StringIO()):
                pages = R.figures(run_info_dir=Path(tmp), results_dir=Path(tmp), out_dir=out)
            self.assertEqual(len(pages), 3)
            texts = {page.name: page.read_text() for page in pages}
        everything = "\n".join(texts.values())
        for words in ("Per-field masking schedule", "Linear masking schedule", "not comparable"):
            self.assertIn(words, everything)
        self.assertNotIn("noise schedule", everything.lower()); self.assertNotIn("masking rate", everything.lower())
        self.assertNotIn("spread", everything.lower())
        by_epoch, two_by_two, gap_page = (texts[name] for name in sorted(texts))
        for value in ("1.2543", "1.4391", "1.2577"):
            self.assertIn(value, by_epoch)
        for value in ("1.2543", "1.4471", "1.2987", "1.4391"):
            self.assertIn(value, two_by_two)
        for value in ("0.0123", "0.0047"):
            self.assertIn(value, gap_page)
        self.assertIn('"focus": "series"', by_epoch)
        self.assertIn('"trigger": "axis"', by_epoch)

    def test_a_chart_whose_inputs_are_missing_is_skipped_by_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_fakes(tmp, with_linear=False)
            for name in (f"masking_rate_cross_rate_{PER_FIELD}.json", f"masking_rate_gap_{PER_FIELD}.json"):
                (Path(tmp) / name).unlink()
            printed = io.StringIO()
            with redirect_stdout(printed):
                pages = R.figures(run_info_dir=Path(tmp), results_dir=Path(tmp), out_dir=Path(tmp) / "figures")
        self.assertEqual(len(pages), 1)
        self.assertIn(f"{LINEAR}.json", printed.getvalue())
        self.assertIn(f"masking_rate_gap_{PER_FIELD}.json", printed.getvalue())

    def test_summary_prints_one_row_per_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_fakes(tmp)
            text = R.summary(run_info_dir=Path(tmp), results_dir=Path(tmp))
        lines = [line for line in text.splitlines() if line.startswith("| ")]
        self.assertEqual(len(lines), 4)                  # header, rule, two models
        self.assertIn("Per-field masking schedule", lines[2])
        self.assertIn("Linear masking schedule", lines[3])
        for value in ("| 4 |", "| 7 |", "401.5", "1.2543", "1.4471", "+0.0123"):
            self.assertIn(value, lines[2])
        self.assertNotIn("noise schedule", text.lower()); self.assertNotIn("masking rate", text.lower())

    def test_summary_skips_a_model_without_run_info(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_fakes(tmp, with_linear=False)
            text = R.summary(run_info_dir=Path(tmp), results_dir=Path(tmp))
        self.assertIn("Per-field masking schedule", text)
        self.assertNotIn("| Linear masking schedule", text)
        self.assertIn(f"{LINEAR}.json", text)


if __name__ == "__main__":
    unittest.main()
