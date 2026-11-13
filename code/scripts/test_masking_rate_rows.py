"""The masking-rate rows in the ask grid, and the rules that judge them (docs/masking-rate-rows.md, issues #295 and #297).

The ask-grid settings are read from the environment at import, so each case runs ask_grid84 in its own process.
Those cases need this laptop's /tmp/vgc-pilot data and the two checkpoints; they are skipped without them.

    /tmp/vgc-pilot/.venv/bin/python scripts/test_masking_rate_rows.py   (from code/)
"""
import json
import os
import subprocess
import unittest
import xml.dom.minidom
from pathlib import Path

import numpy as np

import repo_paths  # noqa: F401  (src/ first on sys.path)
import masking_rate_rows as R

PY, SRC = "/tmp/vgc-pilot/.venv/bin/python", Path(__file__).resolve().parents[1] / "src"
CHECKPOINTS = Path.home() / ".local/share/vgc-pilot-runtime/lossdown"
HERE = Path("/tmp/vgc-pilot/data/pokedex.json").exists() and (CHECKPOINTS / "stat84_shuffled_noise_linear_es3_cap80.pt").exists()
PROBE = ("import json, ask_grid84 as A; s = A.load_sampler(); "
         "print(json.dumps(dict(out=A.OUT.name, tag=A.tag_of(A.ASK_NUMERATORS[0], 1), guidance=A.GUIDANCE, asks=len(A.ASK_NUMERATORS), ask=A.ask_of(A.ASK_NUMERATORS[0]), "
         "checkpoint=A.checkpoint_path().name, exists=A.checkpoint_path().exists(), table=A.table_of(A.MODELS[A.MODEL]), sampler=type(s).__name__, "
         "lookahead=getattr(s, 'lookahead', None), sched=A.load_model().info.get('sched'))))")


def ask_grid(model, **env):
    done = subprocess.run([PY, "-c", PROBE], cwd=SRC, env={**os.environ, "AG84_MODEL": model, **env}, capture_output=True, text=True, check=True)
    return json.loads(done.stdout.strip().splitlines()[-1])


@unittest.skipUnless(HERE, "needs /tmp/vgc-pilot data and the two masking-rate checkpoints")
class AskGridRows(unittest.TestCase):
    def test_each_code_id_loads_its_checkpoint_and_decoder(self):
        want = {"sched_species_first": ("stat84_shuffled_noise_per_field_es3_cap80.pt", "Model84", None, "16,4,4,1.5,1"),
                "sched_linear_species_first": ("stat84_shuffled_noise_linear_es3_cap80.pt", "Model84", None, "1,1,1,1,1"),
                "sched_linear_random": ("stat84_shuffled_noise_linear_es3_cap80.pt", "RandomDecodingArm", True, "1,1,1,1,1")}
        for row, (checkpoint, sampler, lookahead, sched) in want.items():
            got = ask_grid(row)
            self.assertEqual((got["checkpoint"], got["sampler"], got["lookahead"], got["sched"]), (checkpoint, sampler, lookahead, sched), row)
            self.assertTrue(got["exists"]); self.assertEqual(got["table"], "lossdata84_shuffled_only.npz")

    def test_guidance_2_keeps_the_stored_names(self):
        for env in ({}, {"AG84_GUIDANCE": "2"}):
            got = ask_grid("sched_species_first", **env)
            self.assertEqual((got["out"], got["tag"], got["guidance"]), ("stat_points_84_sched_species_first.json", "sched_species_first_a01of24_s1", 2.0))

    def test_another_guidance_gets_its_own_results_file_and_tags(self):
        got = ask_grid("sched_linear_random", AG84_GUIDANCE="8")
        self.assertEqual((got["out"], got["tag"], got["guidance"], got["asks"]), ("stat_points_84_sched_linear_random_g8.json", "sched_linear_random_g8_a01of24_s1", 8.0, 23))

    def test_unguided_cells_carry_no_ask_and_no_push(self):
        got = ask_grid("sched_linear_species_first", AG84_UNGUIDED="1", AG84_GUIDANCE="8")
        self.assertEqual((got["out"], got["tag"], got["guidance"], got["asks"], got["ask"]),
                         ("stat_points_84_sched_linear_species_first_unguided.json", "sched_linear_species_first_unguided_a00of24_s1", 1.0, 1, None))


def grid(row, guidance, error, line_ups, pooled_win=0.37):
    return dict(row=row, reader=R.ROWS[row], guidance=float(guidance), error=error, error_interval=[error - 0.0004, error + 0.0004],
                error_by_seed={"1": error + 0.0003, "2": error, "3": error - 0.0003}, error_asks_18_to_23=0.14, win_top_asks=0.51, win_top_asks_interval=[0.5, 0.52],
                pooled_win=pooled_win, pooled_win_interval=[pooled_win - 0.003, pooled_win + 0.003], first_try_legality=1.0, line_ups=line_ups,
                line_ups_by_seed={"1": line_ups // 3 - 5, "2": line_ups // 3, "3": line_ups // 3 + 5}, new_against_training=0.27, new_against_search_and_tournaments=0.73,
                effective_species=14.9, top_species_share=0.135, teams=4416, battles=216384, cells=69, cells_short_of_64=0)


def draws(center, spread=0.0002):
    return dict(error=center + spread * np.linspace(-2, 2, R.DRAWS))


class VerdictRule(unittest.TestCase):
    control = grid(R.CONTROL, 2, 0.0060, 1400)

    def judge(self, error, **changes):
        arm = dict(grid("sched_linear_species_first", 2, error, 1400), **changes)
        return R.verdict(arm, self.control, draws(error), draws(0.0060), linear=True)

    def test_a_win_needs_the_bar_and_an_interval_below_zero(self):
        self.assertEqual(self.judge(0.0045)["verdict"], "wins")
        self.assertEqual(self.judge(0.0052)["verdict"], "matches")       # 0.0008 lower: under the 0.001 bar
        self.assertEqual(self.judge(0.0057)["verdict"], "matches")       # the interval spans zero

    def test_a_loss_needs_only_the_interval_above_zero(self):
        self.assertEqual(self.judge(0.0069)["verdict"], "loses")

    def test_a_small_gap_between_two_trainings_is_marked_as_training_luck(self):
        self.assertTrue(self.judge(0.0045)["inside_training_luck"])       # 0.0015 lower
        self.assertFalse(self.judge(0.0035)["inside_training_luck"])      # 0.0025 lower

    def test_a_win_that_costs_win_rate_or_line_ups_is_a_trade(self):
        self.assertFalse(self.judge(0.0035)["trade"])
        self.assertTrue(self.judge(0.0035, pooled_win=0.34)["trade"])
        self.assertTrue(self.judge(0.0035, line_ups=1300)["trade"])

    def test_a_row_wins_with_two_wins_and_no_loss(self):
        cases = (["wins", "wins", "matches"], ["wins", "matches", "matches"], ["wins", "wins", "loses"], ["matches"] * 3, ["loses", "matches", "matches"])
        self.assertEqual([R.row_verdict(words) for words in cases], ["wins", "weak win", "mixed", "matches", "loses"])

    def test_seed_by_seed_differences_are_reported(self):
        self.assertEqual(self.judge(0.0045)["seeds_lower"], 3)


class Measures(unittest.TestCase):
    def test_error_is_the_mean_squared_gap_on_the_asks_given(self):
        got = {k: k / 24 for k in range(1, 24)}; got[17] += 0.17
        self.assertAlmostEqual(R.error(got), 0.17 ** 2 / 17); self.assertAlmostEqual(R.error(got, range(18, 24)), 0.0)
        self.assertIsNone(R.error({12: 0.5}, range(18, 24)))

    def test_a_team_score_weights_opponents_and_skips_the_unbattled(self):
        rate = np.array([[1.0, 0.0, np.nan], [0.5, 0.5, 1.0]])
        np.testing.assert_allclose(R.team_scores(rate, np.array([1.0, 2.0, 1.0])), [1 / 3, 2.5 / 4])

    def test_pooling_weights_each_cell_by_its_teams(self):
        cells = {(k, s): np.full(R.DRAWS, float(s)) for k in R.ALL_ASKS for s in R.SEEDS}
        sizes = {key: 64 if key[1] < 3 else 32 for key in cells}
        self.assertAlmostEqual(float(R.pooled(cells, sizes)[R.ALL_ASKS[0]][0]), (64 + 128 + 96) / 160)

    def test_a_point_is_outside_the_frontier_only_when_its_interval_clears(self):
        frontier = [grid(R.CONTROL, g, e, n) for g, e, n in ((2, 0.0060, 1500), (4, 0.0050, 1200), (8, 0.0040, 900))]
        inside = R.outside_frontier(grid("sched_linear_species_first", 2, 0.0056, 1500), frontier)
        below = R.outside_frontier(grid("sched_linear_species_first", 2, 0.0040, 1500), frontier)
        wider = R.outside_frontier(grid("sched_linear_random", 2, 0.0060, 2400), frontier)
        self.assertFalse(inside["outside"]); self.assertTrue(below["lower_error_than_every_control_point_with_as_many_line_ups"])
        self.assertTrue(wider["more_line_ups_than_every_control_point_with_error_as_low"])


class Post(unittest.TestCase):
    def result(self):
        grids = {R.grid_name(R.CONTROL, 2): grid(R.CONTROL, 2, 0.0060, 1400), R.grid_name("sched_linear_species_first", 2): grid("sched_linear_species_first", 2, 0.0045, 1380)}
        verdicts = {"sched_linear_species_first": R.verdict(grids["sched_linear_species_first"], grids[R.CONTROL], draws(0.0045), draws(0.0060), linear=True)}
        return dict(grids=grids, verdicts_against_control=verdicts, corrected_t_probe=dict(largest_total_variation=0.0078, bar=0.01))

    def test_the_figure_is_well_formed_in_both_themes_and_marks_grids_not_run(self):
        result = self.result()
        for theme in R.THEMES:
            svg = R.figure(result, "sched_linear_species_first", theme, R.headline_of(result, "sched_linear_species_first"))
            xml.dom.minidom.parseString(svg)
            self.assertEqual(svg.count("not run yet"), 7); self.assertIn("wins, -0.0015", svg); self.assertIn("0.0078", svg)

    def test_the_comment_opens_with_the_finding_and_the_picture(self):
        result = self.result(); text = R.comment(result, "sched_linear_species_first", "<picture>figure</picture>").split("\n")
        self.assertIn("Linear, species-first at guidance 2 beats the control: error 0.0045 against 0.0060", text[0])
        self.assertEqual(text[2], "<picture>figure</picture>"); self.assertIn("inside the luck of one training seed", "\n".join(text))

    def test_a_control_grid_is_named_as_the_control(self):
        self.assertIn("The control, Per-field, uncorrected t at guidance 2, reads", R.headline_of(self.result(), R.CONTROL))


if __name__ == "__main__":
    unittest.main()
