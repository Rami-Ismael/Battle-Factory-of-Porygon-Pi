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

    def test_short_cells_pass_only_when_every_kept_team_has_battles(self):
        with tempfile.TemporaryDirectory() as directory:
            results = Path(directory)
            file = results / "stat_points_84_test.json"
            cells = {
                f"{ask}-{seed}": {"numerator": ask, "seed": seed, "pastes": [], "y": [], "battles": 0}
                for ask in G.R.ALL_ASKS for seed in G.R.SEEDS
            }
            cells["1-1"].update(pastes=["one", "two"], y=[0.5, 0.7], battles=98)
            file.write_text(json.dumps({"cells": cells}))
            G.check_uniform_grid("test", results)
            cells["1-1"]["y"].pop()
            file.write_text(json.dumps({"cells": cells}))
            G.check_uniform_grid("test", results, sampled_only=True)
            with self.assertRaisesRegex(ValueError, "missing battles for ask 1/24 seed 1"):
                G.check_uniform_grid("test", results)

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
