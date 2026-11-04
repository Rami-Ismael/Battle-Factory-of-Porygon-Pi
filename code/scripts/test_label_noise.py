"""Label noise for the 84-field model (docs/label-noise-84-fields.md). Needs only torch, so CI can run it.

    .venv/bin/python scripts/test_label_noise.py   (from code/)
"""
import unittest

import torch

import repo_paths  # noqa: F401  (src/ first on sys.path)
from label_noise import noise_tag, noise_width, noisy_labels


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


if __name__ == "__main__":
    unittest.main()
