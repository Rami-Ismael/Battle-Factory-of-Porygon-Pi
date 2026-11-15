"""New 84-field trainings use the linear masking schedule by default (owner, 2026-11-14, after #297 and #335).

Every model trained before that day used the per-field masking schedule, so the arms that reproduce one keep it, and a
checkpoint that never wrote its schedule down is read as per-field. No training runs here.

Needs this laptop's /tmp/vgc-pilot data; skipped without it.

    /tmp/vgc-pilot/.venv/bin/python scripts/test_default_masking_schedule.py   (from code/)
"""
import inspect
import sys
import unittest
from pathlib import Path

import repo_paths  # noqa: F401  (src/ first on sys.path)

HERE = Path("/tmp/vgc-pilot/data/pokedex.json").exists()
if HERE:
    sys.path.insert(1, "/tmp/vgc-pilot/src")
    import diffusion84 as M
    import train84 as TR
    import train65

LINEAR, PER_FIELD = "1,1,1,1,1", "16,4,4,1.5,1"


@unittest.skipUnless(HERE, "needs /tmp/vgc-pilot data")
class DefaultMaskingSchedule(unittest.TestCase):
    def test_the_default_is_linear_and_the_per_field_exponents_keep_a_name(self):
        self.assertEqual(M.SCHED, LINEAR); self.assertEqual(M.LINEAR_SCHED, LINEAR); self.assertEqual(M.PER_FIELD_SCHED, PER_FIELD)

    def test_a_training_call_without_sched_trains_linear(self):
        self.assertEqual(inspect.signature(M.Model84.train_new).parameters["sched"].default, LINEAR)
        self.assertEqual(TR.RECIPE["sched"], LINEAR)

    def test_a_checkpoint_without_its_schedule_is_read_as_per_field(self):
        self.assertEqual(M.trained_sched({}), PER_FIELD)
        self.assertEqual(M.trained_sched({"sched": LINEAR}), LINEAR)


@unittest.skipUnless(HERE, "needs /tmp/vgc-pilot data")
class ArmsKeepTheirSchedule(unittest.TestCase):
    PAST = ("loop", "records", "all", "all_noise", "byepoch", "ddom_tau01", "ddom_tau03", *("pos_per_field_s0", "pos_per_field_s1",
            "pos_slot_learned", "pos_slot_fixed", "pos_no_slot"))

    def test_arms_that_reproduce_a_model_trained_before_the_change_stay_per_field(self):
        for arm in self.PAST:
            with self.subTest(arm=arm): self.assertEqual(TR.recipe_for(arm, {})["sched"], PER_FIELD)

    def test_the_masking_schedule_arms_keep_their_own_schedule(self):
        self.assertEqual(TR.recipe_for("shuffled_per_field", dict(TR.MASKING_RATE_ARMS["shuffled_per_field"]))["sched"], PER_FIELD)
        self.assertEqual(TR.recipe_for("shuffled_linear", dict(TR.MASKING_RATE_ARMS["shuffled_linear"]))["sched"], LINEAR)

    def test_a_new_arm_trains_linear(self):
        self.assertEqual(TR.recipe_for("a_new_arm", {})["sched"], LINEAR)

    def test_the_recipe_takes_the_seed_and_schedule_out_of_the_network_settings(self):
        kw = dict(seed=1, sched=PER_FIELD, pos="slot_fixed")
        self.assertEqual(TR.recipe_for("a_new_arm", kw), dict(batch=256, lr=4e-4, sched=PER_FIELD, seed=1))
        self.assertEqual(kw, dict(pos="slot_fixed"))

    def test_the_65_field_trainings_stay_per_field(self):
        self.assertEqual(train65.RECIPE["sched"], PER_FIELD)


if __name__ == "__main__":
    unittest.main()
