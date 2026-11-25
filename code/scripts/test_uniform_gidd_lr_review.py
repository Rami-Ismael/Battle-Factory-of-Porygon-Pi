import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import repo_paths  # noqa: F401
import line_up_yield as LY
import udlm_pass_count84 as U
import uniform_gidd_lr_grid_report as G


def slots(team):
    return [
        {"species": f"{team}{i}", "item": "", "ability": "ability", "nature": "", "moves": ["move"], "evs": {}}
        for i in range(6)
    ]


class UniformGridReview(unittest.TestCase):
    def test_pass_counts_match_teams_per_cell_and_drop_empty_cells(self):
        def grid(pastes):
            return {"cells": {
                str(i): {"seed": 1, "numerator": i, "pastes": teams}
                for i, teams in enumerate(pastes, 1)
            }}

        grids = {
            84: grid([["a", "b"], ["c"]]),
            128: grid([["a", "a", "a"], []]),
            256: grid([["a", "b", "c", "d"], ["d"]]),
        }
        with patch.object(U, "SCORED_ASKS", (1, 2)), patch.object(U, "SEEDS", (1,)):
            stats, sizes = U.matched_lineup_uniqueness(grids, slots)
            self.assertEqual(U.matched_lineup_uniqueness(grids, slots), (stats, sizes))
        self.assertEqual(sizes, {"a01_s1": 2, "a02_s1": 0})
        self.assertEqual([stats[k]["mean"] for k in U.PASS_COUNTS], [1.0, 0.5, 1.0])

    def test_selection_uses_matched_uniqueness(self):
        summaries = {
            k: {
                "seed0": 123, "legality": U._seed_stats([0.2] * 3),
                "within_ask_species_replacement_distance": U._seed_stats([4.0] * 3),
                "matched_lineup_uniqueness": U._seed_stats([0.5 if k == 84 else 1.0] * 3),
                "within_ask_diversity": {"composition": {"unique_fraction": U._seed_stats([1.0] * 3)}},
            }
            for k in U.PASS_COUNTS
        }
        self.assertEqual(U.select_pass_count(summaries)["chosen_passes"], 128)

    def test_pass_selection_requires_finite_measurements_for_all_seeds(self):
        distance = "within_ask_species_replacement_distance"
        summaries = {
            k: {
                "seed0": 123,
                "legality": U._seed_stats([0.2] * 3),
                distance: U._seed_stats([4.0] * 3),
                "matched_lineup_uniqueness": U._seed_stats([1.0] * 3),
            }
            for k in U.PASS_COUNTS
        }
        for missing in (float("nan"), float("inf")):
            with self.subTest(missing=missing):
                summaries[84][distance]["by_seed"] = [4.0, missing, 4.0]
                decision = U.select_pass_count(summaries)
                self.assertEqual(decision["chosen_passes"], 128)
                self.assertFalse(decision["comparisons_vs_256"]["84"][distance]["within_seed_noise"])

        summaries[256][distance]["by_seed"] = [4.0, float("nan"), 4.0]
        self.assertEqual(U.select_pass_count(summaries)["chosen_passes"], 256)

        summaries[256][distance]["by_seed"] = [4.0] * 3
        summaries[84][distance]["by_seed"] = [4.0] * 3
        summaries[84]["matched_lineup_uniqueness"]["by_seed"] = [1.0] * 2
        self.assertEqual(U.select_pass_count(summaries)["chosen_passes"], 128)

    def test_short_cells_pass_only_when_every_kept_team_has_battles(self):
        with tempfile.TemporaryDirectory() as directory:
            results = Path(directory)
            file = results / "stat_points_84_test.json"
            cells = {
                f"{ask}-{seed}": {"numerator": ask, "seed": seed, "pastes": [], "y": [], "battles": 0}
                for ask in G.R.ALL_ASKS for seed in G.R.SEEDS
            }
            reference = G.R.grid_name(G.MASKED_ROW, G.R.GUIDANCES[0])
            np.savez(results / f"masking_rate_rows_opponents_{reference}.npz", columns=np.arange(49))
            cells["1-1"].update(pastes=["one", "two"], team_ids=[0, 50], y=[0.5, 0.7], battles=97)
            file.write_text(json.dumps({"cells": cells}))
            G.check_uniform_grid("test", results)
            cells["1-1"]["team_ids"] = [49, 50]
            file.write_text(json.dumps({"cells": cells}))
            with self.assertRaisesRegex(ValueError, "missing battles for ask 1/24 seed 1"):
                G.check_uniform_grid("test", results)
            cells["1-1"]["y"].pop()
            file.write_text(json.dumps({"cells": cells}))
            G.check_uniform_grid("test", results, sampled_only=True)
            with self.assertRaisesRegex(ValueError, "missing battles for ask 1/24 seed 1"):
                G.check_uniform_grid("test", results)

    def test_snapshot_allows_only_the_self_play_gap(self):
        cell = {"numerator": 1, "seed": 1, "pastes": ["meta team", "other team"],
                "team_ids": [0, 50], "y": [0.5, 0.5], "battles": 97}
        grid = {"cells": {"1-1": cell}}
        with tempfile.TemporaryDirectory() as directory:
            results = Path(directory)
            file = results / "masking_rate_rows_opponents_test.npz"
            rates = np.full((2, 49), 0.5)
            rates[0, 0] = np.nan
            np.savez(file, columns=np.arange(49), weights=np.ones(49), a01_s1=rates)
            G.snapshot(results, "test", grid)

            for gaps in ([(0, 1)], [(0, 0), (0, 1)], [(0, 0), (1, 0)], []):
                with self.subTest(gaps=gaps):
                    rates.fill(0.5)
                    for row, column in gaps:
                        rates[row, column] = np.nan
                    np.savez(file, columns=np.arange(49), weights=np.ones(49), a01_s1=rates)
                    with self.assertRaisesRegex(ValueError, "opponent snapshot differs"):
                        G.snapshot(results, "test", grid)

    def test_missing_ask_has_no_error_or_win_rate_draw(self):
        sizes = {(k, s): int(k != 1) for k in G.R.ALL_ASKS for s in G.R.SEEDS}
        draws = {key: np.full(3, 0.5 if size else 0.0) for key, size in sizes.items()}
        got = G.R._pooled_got_draws(draws, sizes)
        self.assertIsNone(got[1])
        self.assertIsNone(G.R._grid_error(got, G.R.SCORED_ASKS))
        np.testing.assert_array_equal(G.R._pooled_draws(draws, sizes, G.R.ALL_ASKS), [0.5] * 3)

    def test_failed_yield_write_preserves_previous_report(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "yield.json"
            LY.save_yield(path, {"kept": 4})
            with self.assertRaises(TypeError):
                LY.save_yield(path, {"kept": object()})
            self.assertEqual(json.loads(path.read_text()), {"kept": 4})
            self.assertEqual(list(path.parent.iterdir()), [path])
