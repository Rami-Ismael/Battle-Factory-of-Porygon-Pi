"""The verdict of issue #335: does random order work better with the linear masking schedule than with the per-field one?
Hand-made numbers, so no grid or checkpoint is needed. The row itself is tested in test_masking_rate_rows.py.

    /tmp/vgc-pilot/.venv/bin/python scripts/test_random_order_schedule.py   (from code/)
"""
import json
import tempfile
import unittest
from pathlib import Path

import repo_paths  # noqa: F401  (src/ first on sys.path)
import echarts_chart as E
import random_order_schedule as R


class GuidanceCall(unittest.TestCase):
    """The gap is Linear, random order's error minus Per-field, random order's: below zero, linear is lower."""
    def test_linear_wins_a_guidance_when_the_gap_passes_the_bar_and_the_interval_is_below_zero(self):
        self.assertEqual(R.guidance_call(-0.004, [-0.006, -0.002]), "linear")

    def test_per_field_wins_under_the_mirror_rule(self):
        self.assertEqual(R.guidance_call(0.004, [0.002, 0.006]), "per_field")

    def test_a_gap_under_the_bar_calls_neither_even_with_a_tight_interval(self):
        self.assertEqual(R.guidance_call(-0.0008, [-0.0009, -0.0007]), "neither")

    def test_an_interval_that_crosses_zero_calls_neither(self):
        self.assertEqual(R.guidance_call(-0.004, [-0.009, 0.001]), "neither")
        self.assertEqual(R.guidance_call(0.004, [-0.001, 0.009]), "neither")

    def test_a_gap_of_exactly_the_bar_counts(self):
        self.assertEqual(R.guidance_call(-R.BAR, [-0.002, -0.0001]), "linear")


class Verdict(unittest.TestCase):
    def test_two_of_three_guidance_values_decide(self):
        self.assertEqual(R.verdict({"2": "linear", "4": "neither", "8": "linear"}), "linear")
        self.assertEqual(R.verdict({"2": "per_field", "4": "per_field", "8": "linear"}), "per_field")

    def test_one_win_each_or_one_win_alone_is_no_measured_difference(self):
        self.assertEqual(R.verdict({"2": "linear", "4": "per_field", "8": "neither"}), "no measured difference")
        self.assertEqual(R.verdict({"2": "linear", "4": "neither", "8": "neither"}), "no measured difference")


def row(reader, error, pooled_win=0.3):
    return dict(reader=reader, error=error, error_interval=[error - 0.001, error + 0.001], error_by_seed={"1": error, "2": error, "3": error},
                error_multiple_of_species_first=4.0, got_by_ask={"1": 0.1}, pooled_win=pooled_win, win_top_asks=0.4, first_try_legality=0.9,
                line_ups=3000, new_against_training=0.8, distinct_novel_line_ups_against_training=2500, effective_species=40.0)


def result():
    guidance = {}
    for g, (per_field, linear) in {"2": (0.050, 0.036), "4": (0.045, 0.030), "8": (0.040, 0.039)}.items():
        gap = linear - per_field
        guidance[g] = dict(gap_linear_minus_per_field=gap, gap_interval=[gap - 0.002, gap + 0.002], call=R.guidance_call(gap, [gap - 0.002, gap + 0.002]),
                           within_training_luck=abs(gap) < R.TRAINING_LUCK,
                           rows={R.PER_FIELD_RANDOM: row("Per-field, random order", per_field), R.LINEAR_RANDOM: row("Linear, random order", linear),
                                 R.LINEAR_SPECIES_FIRST: row("Linear, species-first", 0.006), R.CONTROL: row("Per-field, species-first (the control)", 0.006)},
                           pass_at_k={R.PER_FIELD_RANDOM: None, R.LINEAR_RANDOM: dict(pass_at_1=0.01, pass_at_48=0.43, pass_at_48_interval=[0.4, 0.46], potential_at_48=0.4)})
    calls = {g: e["call"] for g, e in guidance.items()}
    return dict(rule="the rule", guidance=guidance, calls=calls, verdict=R.verdict(calls))


class Posting(unittest.TestCase):
    def setUp(self):
        self.folder = Path(tempfile.mkdtemp()); self.saved = R.OUT
        R.OUT = self.folder / "random_order_schedule.json"; json.dump(result(), open(R.OUT, "w"))

    def tearDown(self): R.OUT = self.saved

    def test_the_hand_made_gaps_call_linear_at_two_guidance_values(self):
        r = json.load(open(R.OUT))
        self.assertEqual(r["calls"], {"2": "linear", "4": "linear", "8": "neither"}); self.assertEqual(r["verdict"], "linear")

    def test_the_comment_opens_with_the_verdict_then_the_picture(self):
        text = R.comment("https://example.org/fig-{theme}.png").splitlines()
        self.assertTrue(text[0].startswith("**Verdict, ") and "works better with the linear masking schedule" in text[0])
        self.assertIn("fig-dark.png", text[2]); self.assertIn("fig-light.png", text[2])
        self.assertIn("| 8 | 0.0400", "\n".join(text)); self.assertIn("within training luck", "\n".join(text))
        self.assertIn("not read | not read", "\n".join(text))      # a missing Pass@k file says so instead of failing

    def test_the_figure_draws_both_random_rows_and_the_two_species_first_rows(self):
        E.CAPTURE = []
        try: R.figure(self.folder)
        finally: captured, E.CAPTURE = E.CAPTURE, None
        _, option, words, _ = captured[0]
        lines = [s for s in option["series"] if s["type"] == "line"]
        self.assertEqual([s["name"] for s in lines], [R.READER[r] for r in R.ROWS])
        self.assertTrue(all(s["emphasis"] == {"focus": "series"} for s in lines)); self.assertEqual(option["tooltip"]["trigger"], "axis")
        self.assertEqual(lines[0]["data"], [[0, 0.050], [1, 0.045], [2, 0.040]])
        self.assertEqual(words[0], "Random order works better with the linear masking schedule")


if __name__ == "__main__":
    unittest.main()
