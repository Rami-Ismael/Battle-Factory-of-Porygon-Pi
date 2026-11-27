import tempfile
import unittest
from pathlib import Path

import numpy as np

import repo_paths  # noqa: F401
import uniform_gidd_lr_probes as P


class UniformGiddLrProbesTest(unittest.TestCase):
    def test_species_swap_perm_is_involution_covering_all_14_fields(self):
        perm = P.species_swap_perm(1, 4)
        self.assertEqual(len(perm), 84)
        self.assertEqual(sorted(perm), list(range(84)))
        # Involution: applying perm twice recovers identity
        self.assertEqual([perm[i] for i in perm], list(range(84)))
        # Slot 1 base fields (8..15) map to slot 4 (32..39)
        self.assertEqual(perm[8:16], list(range(32, 40)))
        self.assertEqual(perm[32:40], list(range(8, 16)))
        # Slot 1 stat points (54..59) map to slot 4 (72..77)
        self.assertEqual(perm[54:60], list(range(72, 78)))
        self.assertEqual(perm[72:78], list(range(54, 60)))
        # Untouched slot 0 stays identity
        self.assertEqual(perm[0:8], list(range(0, 8)))
        self.assertEqual(perm[48:54], list(range(48, 54)))

    def test_tv_distance_and_bootstrap_mean(self):
        p = np.array([[0.7, 0.3], [0.5, 0.5]])
        q = np.array([[0.7, 0.3], [0.2, 0.8]])
        np.testing.assert_allclose(P.tv_distance(p, q), [0.0, 0.3])
        boot = P.bootstrap_mean([0.2, 0.4, 0.6, 0.8], draws=500, seed=1)
        self.assertAlmostEqual(boot["mean"], 0.5)
        self.assertLessEqual(boot["interval"][0], boot["mean"])
        self.assertGreaterEqual(boot["interval"][1], boot["mean"])

    def test_figures_and_comment_render_from_synthetic_payloads(self):
        sb = {
            "epochs": [45, 49, 80],
            "arms": {
                arm["id"]: {
                    "reader": arm["reader"],
                    "by_epoch": {
                        str(ep): {
                            "species_first_prefix": {
                                field: {
                                    "own": {"mean": 0.9 if ep == 80 else 0.4, "interval": [0.85, 0.95]},
                                    "other": {"mean": 0.05, "interval": [0.02, 0.08]},
                                    "none": {"mean": 0.05, "interval": [0.02, 0.08]},
                                }
                                for field in ("ability", "first_move")
                            }
                        }
                        for ep in (45, 49, 80)
                    },
                }
                for arm in P.ARMS
            },
        }
        sw = {
            "arms": {
                arm["id"]: {
                    "reader": arm["reader"],
                    "species_swap_tracking": {
                        "ability": {
                            "similarity_1_minus_tv": {"mean": 0.95, "interval": [0.94, 0.96]},
                            "top1_match": {"mean": 0.96, "interval": [0.95, 0.97]},
                            "new_species_legal": {"mean": 0.98, "interval": [0.97, 0.99]},
                            "old_species_only_legal": {"mean": 0.01, "interval": [0.005, 0.015]},
                        },
                        "first_move": {
                            "similarity_1_minus_tv": {"mean": 0.93, "interval": [0.92, 0.94]},
                            "top1_match": {"mean": 0.92, "interval": [0.90, 0.94]},
                            "new_species_legal": {"mean": 0.95, "interval": [0.94, 0.96]},
                            "old_species_only_legal": {"mean": 0.02, "interval": [0.01, 0.03]},
                        },
                        "stat_points": {
                            "similarity_1_minus_tv": {"mean": 0.91, "interval": [0.90, 0.92]},
                            "top1_match": {"mean": 0.89, "interval": [0.88, 0.90]},
                        },
                    },
                    "full_slot_equivariance_tv_shift": {
                        "ability": {"mean": 0.012, "interval": [0.010, 0.014]},
                        "moves": {"mean": 0.015, "interval": [0.013, 0.017]},
                        "stat_points": {"mean": 0.011, "interval": [0.009, 0.013]},
                    },
                }
                for arm in P.ARMS
            }
        }
        sm = {
            "rows": [
                {
                    "arm_id": "uniform_gidd_lr",
                    "reader": "Uniform, no filter (shared recipe)",
                    "passes": 84,
                    "guidance": 1.0,
                    "first_try_legality": 0.44,
                    "first_try_legality_interval": [0.37, 0.51],
                    "legal_teams": 84,
                    "tries": 192,
                    "seconds_per_try": 0.05,
                    "seconds_per_valid_team": 0.114,
                    "seconds_per_valid_team_interval": [0.098, 0.135],
                    "peak_gib": 0.35,
                    "ceiling_gib": 17.76,
                    "fits_ceiling": True,
                }
            ]
        }
        vl = {
            "arms": {
                arm["id"]: {
                    "reader": arm["reader"],
                    "val_loss": 1.3153,
                    "stored_best_val_loss": 1.3153,
                    "note": "test loss",
                }
                for arm in P.ARMS
            }
        }
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            p1 = P.build_slot_binding_figure(sb, out)
            p2 = P.build_slot_swap_figure(sw, out)
            p3 = P.build_speed_memory_figure(sm, out)
            for p in (p1, p2, p3):
                self.assertTrue(p.exists())
                self.assertIn("echarts", p.read_text())
        comment = P.format_probes_comment(sb, sw, sm, vl, "https://example.test/raw")
        self.assertIn("slot-binding-dark.png", comment)
        self.assertIn("slot-swap-light.png", comment)
        self.assertIn("speed-memory-light.png", comment)
        self.assertIn("1.3153", comment)


if __name__ == "__main__":
    unittest.main()
