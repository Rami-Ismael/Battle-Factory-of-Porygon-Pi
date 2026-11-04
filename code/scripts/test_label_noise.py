"""Label noise for the 84-field model (docs/label-noise-84-fields.md). Needs only torch, so CI can run it.

    .venv/bin/python scripts/test_label_noise.py   (from code/)
"""
import unittest

import torch

import repo_paths  # noqa: F401  (src/ first on sys.path)
from label_noise import flagged_checkpoint, flagged_row, noise_tag, noise_width, noisy_labels


class NoisyLabels(unittest.TestCase):
    def setUp(self):
        self.rng = torch.Generator().manual_seed(0)

    def test_zero_width_returns_the_labels_and_draws_nothing(self):
        w = torch.tensor([0.5, float("nan")])
        state = self.rng.get_state()
        self.assertIs(noisy_labels(w, 0.0, self.rng), w)
        self.assertTrue(torch.equal(state, self.rng.get_state()))

    def test_noise_stays_within_its_width(self):
        w = torch.full((10000,), 0.5)
        got = noisy_labels(w, 1 / 48, self.rng)
        self.assertLessEqual(float((got - w).abs().max()), 1 / 48 + 1e-6)
        self.assertGreater(float(got.std()), 0.005)

    def test_a_team_with_no_label_keeps_no_label(self):
        got = noisy_labels(torch.tensor([float("nan"), 0.5]), 1 / 48, self.rng)
        self.assertTrue(torch.isnan(got[0]))
        self.assertFalse(torch.isnan(got[1]))

    def test_labels_stay_between_zero_and_one(self):
        got = noisy_labels(torch.tensor([0.0, 1.0] * 500), 1 / 48, self.rng)
        self.assertGreaterEqual(float(got.min()), 0.0)
        self.assertLessEqual(float(got.max()), 1.0)

    def test_each_call_draws_fresh_noise(self):
        w = torch.full((100,), 0.5)
        self.assertFalse(torch.equal(noisy_labels(w, 0.02, self.rng), noisy_labels(w, 0.02, self.rng)))


class NoiseWidthFlag(unittest.TestCase):
    def test_a_fraction_and_its_decimal_give_the_same_width(self):
        self.assertEqual(noise_width("1/50"), noise_width("0.02"))

    def test_widths_outside_zero_to_one_and_bad_text_are_refused(self):
        for text in ("-0.1", "1.5", "1/0", "nan", "wide"):
            with self.subTest(text), self.assertRaises(ValueError):
                noise_width(text)

    def test_the_tag_names_the_width(self):
        self.assertEqual(noise_tag(noise_width("1/48")), "_labelnoise1over48")
        self.assertEqual(noise_tag(noise_width("0")), "_labelnoise0")

    def test_close_widths_get_different_tags(self):
        self.assertNotEqual(noise_tag(noise_width("1/48")), noise_tag(noise_width("0.0208333")))


class FlaggedRow(unittest.TestCase):
    ALL = "stat84_all_teams_es3_cap80_labelnoise1over100"

    def test_a_name_without_the_flag_has_no_row(self):
        self.assertIsNone(flagged_checkpoint("gen84_all_ep21"))
        self.assertIsNone(flagged_checkpoint("stat84_all_teams_es3_cap80"))

    def test_with_patience_the_epochs_count_the_best_epoch_the_file_holds(self):
        info = dict(label_noise=0.01, patience=3, best_pass=18, stopped_pass=21)
        row = flagged_row(self.ALL, info)
        self.assertEqual(row["checkpoint"], self.ALL)
        self.assertEqual(row["reader"], "Label noise ±1/100, all teams, 18 epochs")
        self.assertEqual(row["epochs"], dict(all_teams=18))
        self.assertIn("stopped at epoch 21", row["stopped"])
        self.assertFalse(row["labelled"])

    def test_with_a_fixed_stop_the_epochs_count_the_last_epoch(self):
        info = dict(label_noise=0.01, patience=None, best_pass=18, stopped_pass=21, ddom_tau=0.1)
        row = flagged_row("stat84_labelled_noise_ddom_tau01_ep21_labelnoise1over100", info)
        self.assertEqual(row["reader"], "Label noise ±1/100, labelled teams, tau 0.1 weights, 21 epochs")
        self.assertEqual(row["epochs"], dict(labelled_teams=21))
        self.assertTrue(row["labelled"])

    def test_a_per_epoch_checkpoint_sits_in_its_training_folder(self):
        base = "stat84_labelled_noise_byepoch_es3_cap80_labelnoise1over100"
        row = flagged_row(f"{base}_ep03", dict(label_noise=0.01, epoch=3))
        self.assertEqual(row["checkpoint"], f"{base}/{base}_ep03")
        self.assertEqual(row["epochs"], dict(labelled_teams=3))

    def test_width_zero_reads_as_no_noise(self):
        row = flagged_row("stat84_all_teams_es3_cap80_labelnoise0", dict(label_noise=0.0, patience=3, best_pass=5, stopped_pass=8))
        self.assertEqual(row["reader"], "No label noise, all teams, 5 epochs")

    def test_a_mixture_reads_each_members_epochs(self):
        member = lambda n: dict(label_noise=0.01, stopped_pass=n)
        row = flagged_row("stat84_same_teams_labelnoise1over100", dict(members=[member(80), member(60)]))
        self.assertEqual(row["epochs"], dict(loop_teams=80, battle_records=60))

    def test_a_width_that_disagrees_with_the_name_is_refused(self):
        with self.assertRaises(ValueError):
            flagged_row(self.ALL, dict(label_noise=1 / 48, stopped_pass=21))

    def test_a_training_still_running_is_refused(self):
        with self.assertRaises(ValueError):
            flagged_row(self.ALL, dict(label_noise=0.01, patience=3, best_pass=4))


if __name__ == "__main__":
    unittest.main()
