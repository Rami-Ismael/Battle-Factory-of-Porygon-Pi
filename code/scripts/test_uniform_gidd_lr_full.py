import tempfile
import unittest
from pathlib import Path

import repo_paths  # noqa: F401
import uniform_gidd_lr_part_a_full as PA
import uniform_gidd_lr_part_b_battled as PB


def _fake_record(guidance, error, uniq, novel, srd, top_win, complete=True):
    by_seed = {}
    for s in PA.SEEDS:
        by_seed[str(s)] = {
            "error": error + 0.0001 * (s - 2),
            "top_ask_win": top_win,
            "legality": 0.45,
            "within_ask": {
                "species_replacement_distance": srd,
                "composition": {
                    "unique_fraction": uniq,
                    "largest_share": 0.15,
                    "novel_draw_fraction": novel,
                },
                "categorical_48": {"unique_fraction": 0.99},
                "with_spreads": {"unique_fraction": 0.99},
            },
            "pooled": {"composition": {"novel_draw_fraction": novel, "unique_fraction": uniq}, "species_replacement_distance": srd},
        }
    return {
        "guidance": float(guidance),
        "passes": 84,
        "cells_complete_scored": complete,
        "by_seed": by_seed,
        "error": {"mean": error, "by_seed": [by_seed[str(s)]["error"] for s in PA.SEEDS]},
        "top_ask_win": {"mean": top_win, "by_seed": [top_win] * 3},
        "legality": {"mean": 0.45, "by_seed": [0.45] * 3},
        "within_ask_unique": {"mean": uniq, "by_seed": [uniq] * 3},
        "within_ask_largest_share": {"mean": 0.15, "by_seed": [0.15] * 3},
        "within_ask_novel": {"mean": novel, "by_seed": [novel] * 3},
        "within_ask_srd": {"mean": srd, "by_seed": [srd] * 3},
        "within_ask_cat48_unique": {"mean": 0.99, "by_seed": [0.99] * 3},
        "within_ask_spreads_unique": {"mean": 0.99, "by_seed": [0.99] * 3},
        "revisions": None,
    }


class UniformGiddLrFullTest(unittest.TestCase):
    def test_usability_bend_points_and_frontier_verdict(self):
        u_recs = [
            _fake_record(1, 0.015, 0.92, 0.60, 4.1, 0.50),
            _fake_record(2, 0.008, 0.90, 0.55, 3.9, 0.55),
            _fake_record(4, 0.004, 0.85, 0.50, 3.7, 0.58),
            _fake_record(8, 0.002, 0.80, 0.48, 3.5, 0.60),
        ]
        m_recs = [
            _fake_record(2, 0.003, 0.88, 0.25, 3.8, 0.57),
            _fake_record(4, 0.0015, 0.86, 0.28, 3.6, 0.59),
            _fake_record(8, 0.001, 0.82, 0.35, 3.4, 0.60),
        ]
        meta = PA.evaluate_usability_and_bend_points(u_recs)
        self.assertEqual(meta["g_max"], 8.0)
        self.assertLessEqual(len(meta["bend_points"]), 2)
        verdict = PA.compute_frontier_verdict(u_recs, m_recs)
        self.assertEqual(verdict["uniform_g_max"], 8.0)
        self.assertEqual(verdict["frontiers"]["within_ask_novel"]["verdict"], "beats")
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            report = {"passes": 84, "uniform_records": u_recs, "masked_records": m_recs, "verdict": verdict}
            figs = PA.build_frontier_and_revision_figures(report, out)
            self.assertEqual(len(figs), 3)
            for p in figs:
                self.assertTrue(p.exists())
            comment = PA.format_frontier_verdict_comment(report, "https://example.test/raw")
            self.assertIn("frontier-wa-novelty-dark.png", comment)

    def test_high_asks_and_swaps_and_part_b_comment(self):
        paste_a = "\n\n".join(f"Sp{i} @ Item{i}\nAbility: Ab\nLevel: 50\nEVs: 4 HP\nSerious Nature\n- Move1" for i in range(6))
        paste_b = "\n\n".join(f"Sp{i if i < 5 else 99} @ Item{i}\nAbility: Ab\nLevel: 50\nEVs: 4 HP\nSerious Nature\n- Move1" for i in range(6))
        grid = {
            "guidance": 8.0,
            "cells": {
                f"a{k}_s1": {"numerator": k, "seed": 1, "pastes": [paste_a, paste_b], "y": [0.6, 0.5]}
                for k in range(1, 24)
            },
        }
        train_lu = {frozenset(f"sp{i}" for i in range(6))}
        res = PB.analyze_high_asks_and_swaps(grid, train_lu)
        self.assertAlmostEqual(res["high_asks_swaps"]["0_swaps_exact"], 0.5)
        self.assertAlmostEqual(res["high_asks_swaps"]["1_swap"], 0.5)
        self.assertAlmostEqual(res["high_asks_win"]["mean"], 0.55)

    def test_deep_summary_reads_line_up_yield_schema(self):
        import json
        with tempfile.TemporaryDirectory() as tmp:
            res_dir = Path(tmp)
            (res_dir / "line_up_yield_demo.json").write_text(
                json.dumps({"views": {"all_23_asks": {"elite_line_ups": 20, "held": 12, "held_new_against_training": 5}}})
            )
            (res_dir / "line_up_yield_demo_d16.json").write_text(
                json.dumps({
                    "views": {"all_23_asks": {"elite_line_ups": 20, "held": 11, "held_new_against_training": 4}},
                    "flips": {"flip_share": 0.25, "flip_share_interval": [0.15, 0.35]},
                    "fresh": [],
                })
            )
            out = PB.summarize_deep_and_weakness(["demo"], res_dir / "missing.sqlite", results_dir=res_dir)
            row = out["grids"]["demo"]
            self.assertEqual(row["elite_candidates"], 20)
            self.assertEqual(row["held_at_4"], 12)
            self.assertEqual(row["novel_held_at_4"], 5)
            self.assertEqual(row["held_at_16"], 11)
            self.assertEqual(row["novel_held_at_16"], 4)
            self.assertAlmostEqual(row["flip_share"], 0.25)


if __name__ == "__main__":
    unittest.main()
