"""Each ask grid's own sampling seeds (issue on the seed replay, 2026-11-15). No model, no data.

    /tmp/vgc-pilot/.venv/bin/python scripts/test_grid_seeds.py   (from code/)
"""
import json
import tempfile
import unittest
from pathlib import Path

import repo_paths  # noqa: F401  (src/ first on sys.path)
import grid_seeds as G


class GridSeeds(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())

    def test_a_stored_grid_keeps_the_seed_base_its_file_records(self):
        path = self.dir / "old.json"; path.write_text(json.dumps(dict(seed0=7300, cells={})))
        self.assertEqual(G.seed0("sched_linear_species_first", path), 7300)

    def test_a_stored_grid_that_never_wrote_its_seed_base_drew_with_7300(self):
        path = self.dir / "older.json"; path.write_text(json.dumps(dict(cells={})))
        self.assertEqual(G.seed0("gen84_all_ep21", path), 7300)

    def test_a_new_grid_gets_a_base_of_its_own_that_repeats_from_run_to_run(self):
        a = G.seed0("sched_linear_species_first_g4", self.dir / "absent.json")
        self.assertEqual(a, G.seed0("sched_linear_species_first_g4", self.dir / "absent.json"))
        self.assertNotEqual(a, G.LEGACY)

    def test_two_new_grids_never_share_a_sampling_seed_with_each_other_or_with_the_stored_grids(self):
        names = [f"{row}{suffix}" for row in ("sched_species_first", "sched_linear_species_first", "sched_linear_random", "a", "b", "c")
                 for suffix in ("", "_g4", "_g8", "_unguided")]
        seeds = [G.seed0(n, self.dir / "absent.json") + s for n in names for s in (1, 2, 3)]
        self.assertEqual(len(seeds), len(set(seeds)))
        self.assertFalse({G.LEGACY + s for s in (1, 2, 3)} & set(seeds))


if __name__ == "__main__":
    unittest.main()
