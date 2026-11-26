"""Unit tests for src/nats_check.py: the fixed-order scorer on a toy distribution with known entropy, the sum of
per-field terms, the order gap, the paired bootstrap interval, the rank agreement, the held-out check and the report's
refusals. No data or checkpoints. The 84-field path is checked inside every `score` run instead: against a second
route to each team's total and against the sampler's own draw probabilities."""
import hashlib
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import nats_check as NC


class ToyDistribution(unittest.TestCase):
    def setUp(self):
        self.toy = NC.ToyJoint()
        self.rows, self.prob = self.toy.outcomes()
        self.order = list(range(self.rows.shape[1]))

    def nats(self, order, blind=False):
        return -NC.chain_log_likelihood(self.toy.true_log_prob(self.rows, blind=blind), self.rows, order)

    def test_selftest_passes(self):
        self.assertTrue(NC.selftest()["passes"])

    def test_mean_nats_equal_the_entropy(self):
        mean = float((self.prob * self.nats(self.order).sum(1)).sum())
        self.assertAlmostEqual(mean, self.toy.entropy, places=12)

    def test_per_field_terms_sum_to_the_team_total(self):
        self.assertLess(NC.sum_check(self.nats(self.order).numpy(), -self.prob.log().numpy()), 1e-12)

    def test_every_fill_order_gives_the_same_team_total(self):
        fixed = self.nats(self.order).sum(1)
        for order in NC.random_orders(16, len(self.order)):
            self.assertLess(float((self.nats(order).sum(1) - fixed).abs().max()), 1e-12)

    def test_a_scorer_that_ignores_revealed_fields_is_caught(self):
        mean = float((self.prob * self.nats(self.order, blind=True).sum(1)).sum())
        self.assertAlmostEqual(mean, self.toy.marginal_entropy_sum(), places=12)
        self.assertGreater(mean, self.toy.entropy + 0.01)

    def test_sum_check_reports_a_dropped_field(self):
        nats = self.nats(self.order).numpy()
        self.assertGreater(NC.sum_check(nats[:, 1:], -self.prob.log().numpy()), 0.01)

    def test_an_order_must_name_every_column_once(self):
        with self.assertRaises(ValueError):
            NC.chain_log_likelihood(self.toy.true_log_prob(self.rows), self.rows, [0, 0, 1, 2])

    def test_fields_are_revealed_in_the_order_given(self):
        seen = []
        def log_prob(x, c):
            seen.append((c, [int(v) for v in (x[0] != 0)])); return torch.zeros(len(x))
        NC.chain_log_likelihood(log_prob, self.rows[:1], [2, 0, 3, 1])
        self.assertEqual(seen, [(2, [0, 0, 0, 0]), (0, [0, 0, 1, 0]), (3, [1, 0, 1, 0]), (1, [1, 0, 1, 1])])


class Intervals(unittest.TestCase):
    def test_order_gap_is_random_order_mean_minus_fixed_order(self):
        gap = NC.order_gap([10.0, 20.0], [[12.0, 20.0], [14.0, 24.0]])
        self.assertAlmostEqual(gap["mean"], 2.5)                # teams: (13 - 10) and (22 - 20)
        self.assertEqual(gap["share_fixed_better"], 1.0)
        self.assertAlmostEqual(gap["mean_range"], 3.0)

    def test_paired_bootstrap_brackets_the_mean(self):
        d = np.random.default_rng(1).normal(0.5, 1.0, 400)
        b = NC.paired_bootstrap(d)
        self.assertLess(b["lo"], b["mean"]); self.assertLess(b["mean"], b["hi"])
        self.assertAlmostEqual(b["hi"] - b["lo"], 4 * d.std() / 20, delta=0.03)

    def test_paired_bootstrap_uses_the_pairing(self):
        shared = np.random.default_rng(2).normal(0, 50, 300)      # team difficulty, far larger than the model difference
        b = NC.paired_bootstrap((shared + 1.0) - shared)
        self.assertAlmostEqual(b["lo"], 1.0); self.assertAlmostEqual(b["hi"], 1.0)

    def test_same_seed_gives_the_same_orders(self):
        self.assertEqual(NC.random_orders(16, 84), NC.random_orders(16, 84))
        self.assertTrue(all(sorted(o) == list(range(84)) for o in NC.random_orders(16, 84)))


class RankAgreement(unittest.TestCase):
    def test_lower_nats_with_higher_win_rate_agrees(self):
        a = NC.rank_agreement(dict(a=100.0, b=110.0, c=120.0), dict(a=0.5, b=0.4, c=0.3))
        self.assertEqual((a["pairs_agree"], a["pairs_disagree"], a["verdict"]), (3, 0, "agrees"))
        self.assertAlmostEqual(a["spearman"], 1.0)

    def test_lower_nats_with_lower_novelty_disagrees(self):
        a = NC.rank_agreement(dict(a=100.0, b=110.0, c=120.0), dict(a=0.1, b=0.2, c=0.3))
        self.assertEqual((a["pairs_agree"], a["pairs_disagree"], a["verdict"]), (0, 3, "disagrees"))

    def test_models_without_the_measure_are_left_out(self):
        a = NC.rank_agreement(dict(a=100.0, b=110.0, c=120.0), dict(a=0.5, b=None))
        self.assertEqual(a["models"], ["a"]); self.assertEqual(a["verdict"], "too few models")

    def test_mixed_order_partly_agrees(self):
        a = NC.rank_agreement(dict(a=100.0, b=110.0, c=120.0), dict(a=0.5, b=0.2, c=0.3))
        self.assertEqual((a["pairs_agree"], a["pairs_disagree"], a["verdict"]), (2, 1, "partly agrees"))


class References(unittest.TestCase):
    def test_every_reference_names_a_measure_and_a_date(self):
        dates = {r["date"] for r in NC.REFERENCES}
        self.assertEqual(dates, {"2026-10-04", "2026-10-07", "2026-11-15", "2026-11-23"})
        for r in NC.REFERENCES:
            self.assertIn(r["ours"].removesuffix("_off").removeprefix("reference_"), ("nats", "block48_per_field", "moves_first_gap"))
            self.assertTrue(r["same"] or r["why"])

    def test_every_model_row_has_a_reader_name_and_a_role(self):
        for row in NC.MODELS.values():
            self.assertIn(row["role"], ("reference", "new")); self.assertTrue(row["reader"])

    def test_reader_names_differ_in_their_first_two_words(self):
        starts = [" ".join(row["reader"].split()[:2]) for row in NC.MODELS.values() if row["role"] == "new"]
        self.assertEqual(len(starts), len(set(starts)))


class HeldOutCheck(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory(); self.addCleanup(self.folder.cleanup)
        rows = np.arange(40, dtype=np.int64).reshape(10, 4)
        val = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1], bool)
        labels = np.array([0.5] * 6 + [np.nan] + [0.5] * 3, np.float32)        # one unlabelled training row
        path = Path(self.folder.name) / "lossdata84_toy.npz"
        np.savez(path, all_X=rows, all_val=val, all_W=labels)
        self.rows, self.sha = rows, hashlib.sha256(path.read_bytes()).hexdigest()

    def check(self, picked, **info):
        return NC.held_out_check({**dict(data_sha256=self.sha, n_train=6, labelled_only=True), **info}, picked, self.folder.name)

    def test_validation_rows_of_the_runs_table_are_held_out(self):
        self.assertEqual(self.check(self.rows[7:])["status"], "held out")

    def test_a_scored_training_row_is_a_leak(self):
        found = self.check(self.rows[5:8])
        self.assertEqual((found["status"], found["scored_teams_in_training_rows"]), ("LEAK", 2))

    def test_a_run_that_trained_on_more_rows_than_the_split_is_unverified(self):
        self.assertEqual(self.check(self.rows[7:], n_train=9)["status"], "unverified")

    def test_rows_the_tables_dropped_are_allowed_for(self):
        self.assertEqual(self.check(self.rows[7:], n_train=5, dropped_by_tables=1)["status"], "held out")

    def test_an_unknown_table_is_unverified(self):
        self.assertEqual(NC.held_out_check(dict(data_sha256="0" * 64), self.rows[7:], self.folder.name)["status"], "unverified")
        self.assertEqual(NC.held_out_check({}, self.rows[7:], self.folder.name)["status"], "unverified")


class ReportRefusals(unittest.TestCase):
    def test_a_folder_with_no_scores_is_refused_before_anything_is_written(self):
        with tempfile.TemporaryDirectory() as folder:
            for out in (Path(folder), Path(folder) / "missing"):
                with self.assertRaises(SystemExit): NC.build_report(out)
                self.assertFalse((out / "report.json").exists())

    def test_files_that_scored_different_teams_are_not_paired(self):
        result = lambda sha, teams: dict(teams=dict(teams_sha256=sha), random_orders=dict(count=16, seed=1, teams=teams))
        NC.same_teams(dict(a=result("x" * 64, 256), b=result("x" * 64, 256)))
        with self.assertRaises(SystemExit): NC.same_teams(dict(a=result("x" * 64, 256), b=result("y" * 64, 256)))
        with self.assertRaises(SystemExit): NC.same_teams(dict(a=result("x" * 64, 256), b=result("x" * 64, 32)))


if __name__ == "__main__":
    unittest.main()
