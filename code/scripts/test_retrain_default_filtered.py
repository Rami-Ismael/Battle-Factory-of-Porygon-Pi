"""Tests for issue #385: retraining the default linear checkpoint on the filtered shuffled-only table.

Covers:
- Pure unit tests for retrain_default_report.py (strict 64-team cell checks, training-curve and trade-off
  ECharts figures, beyond-training-luck detection, and comments) — clean-machine safe.
- Laptop-only tests (skipped without /tmp/vgc-pilot) verifying train84.py's shuffled_linear_filtered arm
  keeps exact recipe parity with shuffled_linear, refuses wrong-stone teams with every field visible,
  and ask_grid84.py's linear_masked_filter row draws on fresh non-7300 seeds.

    /tmp/vgc-pilot/.venv/bin/python scripts/test_retrain_default_filtered.py   (from code/)
"""
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

import numpy as np

import repo_paths  # noqa: F401  (src/ first on sys.path)
import grid_seeds
import retrain_default_report as RDR

HERE = Path("/tmp/vgc-pilot/data/pokedex.json").exists()
if HERE:
    sys.path.insert(1, "/tmp/vgc-pilot/src")
    import ask_grid84 as A
    import train84 as TR


class ReportAndChecksTests(unittest.TestCase):
    def _fake_grid_file(self, tmp, name, *, pastes_per_cell=64, battles_per_team=49):
        cells = {}
        for k in (1, 2):
            for s in (1, 2):
                pastes = [f"team_{i}" for i in range(pastes_per_cell)]
                cells[f"{name}_a{k:02d}of24_s{s}"] = dict(
                    numerator=k,
                    seed=s,
                    pastes=pastes,
                    y=[0.5] * len(pastes),
                    battles=battles_per_team * len(pastes),
                )
        path = Path(tmp) / f"stat_points_84_{name}.json"
        path.write_text(json.dumps(dict(keep=64, cells=cells)))
        return path

    def test_check_grid_passes_when_every_cell_has_64_teams_and_full_battles(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            self._fake_grid_file(tmp, "linear_masked_filter", pastes_per_cell=64, battles_per_team=49)
            res = RDR.check_grid("linear_masked_filter", 4, results_dir=tmp)
        self.assertEqual(res["short_teams"], 0)
        self.assertEqual(res["short_battles"], 0)

    def test_check_grid_rejects_a_cell_short_of_64_teams(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            self._fake_grid_file(tmp, "linear_masked_filter", pastes_per_cell=63, battles_per_team=49)
            with self.assertRaises(SystemExit):
                RDR.check_grid("linear_masked_filter", 4, results_dir=tmp)

    def test_evaluate_beyond_luck_flags_error_win_novelty_and_pass_at_k(self):
        new_m = dict(
            novel_share=0.45,
            pass_at_k=dict(pass_at_8=0.60, pass_at_8_interval=[0.55, 0.65], pass_at_48=0.99, pass_at_48_interval=[0.98, 1.0]),
        )
        ref_m = dict(
            unbiased_breadth=dict(novel_share=0.33),
            pass_at_k=dict(pass_at_8=0.81, pass_at_8_interval=[0.75, 0.86], pass_at_48=1.0, pass_at_48_interval=[0.99, 1.0]),
        )
        flags = RDR.evaluate_beyond_luck(
            new_m,
            ref_m,
            dict(difference=0.0025, interval=[0.0010, 0.0040]),
            dict(difference=-0.030, interval=[-0.045, -0.015]),
        )
        self.assertEqual(len(flags), 4)
        self.assertTrue(any("error" in f for f in flags))
        self.assertTrue(any("pooled win rate" in f for f in flags))
        self.assertTrue(any("novel share" in f for f in flags))
        self.assertTrue(any("Pass@8" in f for f in flags))

    def test_evaluate_beyond_luck_empty_when_within_luck(self):
        new_m = dict(
            novel_share=0.34,
            pass_at_k=dict(pass_at_8=0.80, pass_at_8_interval=[0.74, 0.85], pass_at_48=1.0, pass_at_48_interval=[0.99, 1.0]),
        )
        ref_m = dict(
            unbiased_breadth=dict(novel_share=0.33),
            pass_at_k=dict(pass_at_8=0.81, pass_at_8_interval=[0.75, 0.86], pass_at_48=1.0, pass_at_48_interval=[0.99, 1.0]),
        )
        flags = RDR.evaluate_beyond_luck(
            new_m,
            ref_m,
            dict(difference=0.0005, interval=[-0.0004, 0.0014]),
            dict(difference=-0.004, interval=[-0.012, 0.004]),
        )
        self.assertEqual(flags, [])

    def test_train_figure_and_comment_include_early_stopping_and_reference(self):
        info = dict(
            n_train=214055,
            val_rows=4368,
            dropped_by_tables=0,
            cap=80,
            patience=3,
            best_epoch=77,
            stopped_epoch=80,
            best_val_loss=2.0110,
            minutes=430.5,
            seed=0,
            amp_dtype="float32",
            data_sha256="e58af7689e62a873",
            curves=[
                dict(epoch=1, train_loss=2.12, val_loss=2.28),
                dict(epoch=77, train_loss=1.60, val_loss=2.0110),
                dict(epoch=80, train_loss=1.59, val_loss=2.0115),
            ],
        )
        ref = dict(best_epoch=80, stopped_epoch=80, best_val_loss=2.0142, minutes=398.0, curves=[dict(epoch=1, train_loss=2.13, val_loss=2.29), dict(epoch=80, train_loss=1.60, val_loss=2.0142)])
        with tempfile.TemporaryDirectory() as tmp:
            html_path = RDR.build_train_figure(info, tmp, ref)
            html_text = html_path.read_text()
            self.assertIn("echarts@6.1.0", html_text)
            self.assertIn("kept ep 77", html_text)
            self.assertIn("dataZoom", html_text)
        md = RDR.format_train_comment(info, "abcdef1234567890", "https://ex/l.png", "https://ex/d.png", ref)
        self.assertIn("kept epoch 77 of 80", md)
        self.assertIn("early stopping fired: **no**", md)
        self.assertIn("0 train / 0 val", md)
        self.assertIn(RDR.NEW_CK, md)

    def test_final_figures_and_handoff_comment_render_cleanly(self):
        def fake_m(err, win, lu, nov, held, nov_held, p8):
            return dict(
                name="x",
                error=err,
                error_interval=[err - 0.001, err + 0.001],
                pooled_win=win,
                win_top_asks=win + 0.15,
                novel_share=nov,
                distinct_line_ups=lu,
                within_ask_line_ups_mean_23=160.0,
                elite_line_up_yield=dict(held=held, held_new_against_training=nov_held),
                pass_at_k=dict(
                    pass_at_8=p8,
                    pass_at_8_interval=[p8 - 0.04, min(1.0, p8 + 0.04)],
                    pass_at_48=1.0,
                    pass_at_48_interval=[0.99, 1.0],
                    pass_at_k=[min(1.0, 0.2 + 0.02 * k) for k in range(48)],
                    pass_at_k_interval=[[max(0.0, 0.15 + 0.02 * k), min(1.0, 0.25 + 0.02 * k)] for k in range(48)],
                ),
            )

        by_g = {}
        for g, e in ((2, 0.0074), (4, 0.0048), (8, 0.0039)):
            by_g[str(g)] = dict(
                guidance=g,
                filtered=fake_m(e, 0.355, 1610, 0.33, 98, 20, 0.80),
                unfiltered=dict(fake_m(e + 0.0001, 0.353, 1578, 0.183, 105, 18, 0.81), unbiased_breadth=dict(novel_share=0.33, distinct_line_ups=1603, within_ask_line_ups_mean_23=162.0)),
                paired=dict(error=dict(difference=-0.0001, interval=[-0.0006, 0.0004]), pooled_win=dict(difference=0.002, interval=[-0.005, 0.009])),
                flags_beyond_luck=[],
            )
        summary = dict(by_guidance=by_g, findings_beyond_luck=[], all_within_training_luck=True)
        run_info = dict(best_epoch=80, stopped_epoch=80, cap=80, patience=3, best_val_loss=2.0120, data_sha256="e58af7689e62")
        with tempfile.TemporaryDirectory() as tmp:
            p1, p2 = RDR.build_final_figures(summary, tmp)
            self.assertTrue(p1.exists() and p2.exists())
            self.assertIn("echarts@6.1.0", p1.read_text())
            self.assertIn("echarts@6.1.0", p2.read_text())
        final_md = RDR.format_final_comment(summary, run_info, "deadbeef1234", "https://ex/1l.png", "https://ex/1d.png", "https://ex/2l.png", "https://ex/2d.png")
        handoff_md = RDR.format_handoff_comment(summary, run_info)
        self.assertIn(RDR.NEW_CK, final_md)
        self.assertIn("#64", final_md)
        self.assertIn(RDR.NEW_CK, handoff_md)
        self.assertIn("linear_masked_filter", handoff_md)


@unittest.skipUnless(HERE, "needs /tmp/vgc-pilot data")
class TrainAndGridWiringTests(unittest.TestCase):
    def fake_table(self):
        labels = np.array([0.5, np.nan, 0.25, 0.75, np.nan, 0.1])
        return dict(
            all_X=np.arange(6 * 84).reshape(6, 84),
            all_W=labels,
            all_val=np.array([False, False, False, True, True, False]),
            sha256="e58af7689e62",
            table="lossdata84_shuffled_only.npz",
        )

    def test_checkpoint_name_does_not_overwrite_unfiltered_linear(self):
        self.assertEqual(TR.NAMES["shuffled_linear_filtered"], "stat84_shuffled_noise_linear_filtered_es3_cap80")
        self.assertNotEqual(TR.NAMES["shuffled_linear_filtered"], TR.NAMES["shuffled_linear"])

    def test_filtered_arm_matches_unfiltered_linear_recipe_exactly(self):
        calls = {}
        for arm in ("shuffled_linear", "shuffled_linear_filtered"):
            with (
                mock.patch.object(TR, "train") as train,
                mock.patch.object(TR, "load", return_value=self.fake_table()) as load,
                mock.patch.object(TR, "check_rows"),
                mock.patch.object(TR, "assert_no_wrong_stones", return_value=dict(method="mock", train_rows=3, val_rows=1, train_rejected=0, val_rejected=0)),
                redirect_stdout(io.StringIO()),
            ):
                TR.main([arm])
            load.assert_called_once_with(TR.SHUFFLED_ONLY)
            train.assert_called_once()
            calls[arm] = train.call_args
        self.assertEqual(calls["shuffled_linear"].kwargs, calls["shuffled_linear_filtered"].kwargs)
        for a, b in zip(calls["shuffled_linear"].args[1:], calls["shuffled_linear_filtered"].args[1:], strict=True):
            self.assertEqual(repr(a), repr(b))

    def test_assert_no_wrong_stones_refuses_rejected_rows_on_either_side(self):
        fake_tables = mock.MagicMock()
        fake_tables.rejects.side_effect = [np.array([False, True, False]), np.array([False])]
        with mock.patch.object(TR.L, "Tables", return_value=fake_tables):
            with self.assertRaises(SystemExit) as ctx:
                TR.assert_no_wrong_stones(None, np.zeros((3, 84), np.int64), np.zeros((1, 84), np.int64))
        self.assertIn("1 training and 0 validation", str(ctx.exception))

    def test_ask_grid_row_uses_filtered_checkpoint_and_fresh_seeds(self):
        row = A.MODELS["linear_masked_filter"]
        self.assertEqual(row["checkpoint"], "stat84_shuffled_noise_linear_filtered_es3_cap80")
        self.assertEqual(row["reader"], "Linear masked, full filter")
        self.assertEqual(row["decoder"], "species_first")
        self.assertEqual(row["table"], "lossdata84_shuffled_only.npz")
        self.assertTrue(row["labelled"])
        self.assertNotIn("seed0", row)
        with tempfile.TemporaryDirectory() as tmp:
            bases = {g: grid_seeds.seed0(RDR.grid_name("linear_masked_filter", g), Path(tmp) / f"missing_{g}.json") for g in (2, 4, 8)}
        self.assertEqual(len(set(bases.values())), 3)
        for b in bases.values():
            self.assertGreaterEqual(b, 100_000)
            self.assertNotEqual(b, grid_seeds.LEGACY)


if __name__ == "__main__":
    unittest.main()
