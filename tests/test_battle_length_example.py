"""Regression tests for the executable aggregation example in the design note.

Run from the repository root:
    python3 -m unittest discover -s tests -v

These tests exercise recorded-result aggregation, not the live battle simulator.
"""

import ast
from pathlib import Path
import re
import unittest


DOCUMENT = Path(__file__).resolve().parents[1] / "Battle length per generation.md"


def load_summarizer():
    """Find the documented function by name, not by fence position."""
    examples = re.findall(r"^```python\n(.*?)^```\s*$", DOCUMENT.read_text(), re.M | re.S)
    matches = []
    for example in examples:
        tree = ast.parse(example)
        if any(isinstance(node, ast.FunctionDef) and node.name == "summarize_generations"
               for node in tree.body):
            matches.append(tree)
    if len(matches) != 1:
        raise AssertionError("Expected exactly one documented summarize_generations example")
    namespace = {}
    exec(compile(matches[0], str(DOCUMENT), "exec"), namespace)
    return namespace["summarize_generations"]


def result(turns, **overrides):
    record = dict(run_id="run01", method="cem", generation=1, phase="search",
                  status="completed", outcome="win", turns=turns)
    record.update(overrides)
    return record


class BattleLengthExampleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.summarize = staticmethod(load_summarizer())

    def test_known_average_and_median(self):
        summary, = self.summarize([result(10), result(15), result(20)])
        self.assertEqual(summary["mean_turns"], 15)
        self.assertEqual(summary["median_turns"], 15)
        self.assertEqual(summary["n_evaluations"], 3)
        self.assertEqual(summary["n_completed"], 3)
        self.assertEqual(summary["n_measured"], 3)

    def test_unequal_batches_are_weighted_by_battle_count(self):
        first_batch = [result(10)]
        second_batch = [result(20), result(30), result(40)]
        summary, = self.summarize(first_batch + second_batch)
        self.assertEqual(summary["mean_turns"], 25)

    def test_wins_losses_and_draws_have_distinct_denominators(self):
        summary, = self.summarize([
            result(10), result(20), result(40, outcome="loss"),
            result(30, outcome="draw"),
        ])
        self.assertEqual(summary["mean_turns"], 25)
        self.assertEqual(summary["mean_turns_win"], 15)
        self.assertEqual(summary["mean_turns_loss"], 40)
        self.assertEqual(summary["win_rate"], 0.5)

    def test_unfinished_battles_are_counted_but_excluded(self):
        summary, = self.summarize([
            result(10), result(300, status="turn_cap", outcome=None),
            result(80, status="timeout", outcome=None),
            result(None, status="error", outcome=None),
        ])
        self.assertEqual(summary["mean_turns"], 10)
        self.assertEqual(summary["win_rate"], 1)
        self.assertEqual(summary["n_evaluations"], 4)
        self.assertEqual(summary["n_completed"], 1)
        for key in ("n_turn_cap", "n_timeout", "n_error"):
            self.assertEqual(summary[key], 1)

    def test_missing_turn_counts_do_not_become_zero(self):
        summary, = self.summarize([result(20), result(None, outcome="loss")])
        self.assertEqual(summary["mean_turns"], 20)
        self.assertEqual(summary["n_missing_turns"], 1)
        self.assertEqual(summary["n_measured"], 1)
        self.assertEqual(summary["n_completed"], 2)
        self.assertEqual(summary["win_rate"], 0.5)
        self.assertIsNone(summary["mean_turns_loss"])

    def test_all_missing_turns_preserve_known_outcomes(self):
        summary, = self.summarize([result(None)])
        self.assertIsNone(summary["mean_turns"])
        self.assertIsNone(summary["median_turns"])
        self.assertIsNone(summary["mean_turns_win"])
        self.assertEqual(summary["win_rate"], 1)

    def test_no_completed_battles_have_no_mean_or_win_rate(self):
        summary, = self.summarize([result(5, status="timeout", outcome=None)])
        for key in ("mean_turns", "median_turns", "mean_turns_win",
                    "mean_turns_loss", "win_rate"):
            self.assertIsNone(summary[key])
        self.assertEqual(summary["n_completed"], 0)

    def test_empty_input_has_no_groups(self):
        self.assertEqual(self.summarize([]), [])

    def test_zero_turns_are_not_treated_as_missing(self):
        summary, = self.summarize([result(0), result(10)])
        self.assertEqual(summary["mean_turns"], 5)
        self.assertEqual(summary["n_missing_turns"], 0)

    def test_each_group_dimension_is_isolated(self):
        for field, alternative in [("run_id", "run02"), ("method", "map_elites"),
                                   ("method", "dibo"), ("generation", 2),
                                   ("phase", "checkpoint_eval")]:
            with self.subTest(field=field, alternative=alternative):
                summaries = self.summarize([result(10), result(90, **{field: alternative})])
                self.assertEqual(len(summaries), 2)
                by_value = {r[field]: r for r in summaries}
                self.assertEqual(by_value[result(10)[field]]["mean_turns"], 10)
                self.assertEqual(by_value[alternative]["mean_turns"], 90)

    def test_completion_order_does_not_change_summaries(self):
        records = [result(10), result(30, generation=2), result(20),
                   result(50, method="dibo")]
        self.assertEqual(self.summarize(records), self.summarize(list(reversed(records))))

    def test_cached_evaluation_slot_uses_its_recorded_turn_count(self):
        summary, = self.summarize([result(10, cache_hit=False), result(30, cache_hit=True)])
        self.assertEqual(summary["mean_turns"], 20)
        self.assertEqual(summary["n_evaluations"], 2)


if __name__ == "__main__":
    unittest.main()
