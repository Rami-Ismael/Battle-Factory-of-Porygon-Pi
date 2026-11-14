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
import masking_rate_rows_figures as F

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
        best = R.outside_frontier(grid("sched_linear_random", 2, 0.0030, 2400), frontier)     # more line-ups and a lower error than every control point
        self.assertTrue(best["outside"]); self.assertTrue(best["outside_by_point_estimates_alone"])


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


class Figures(unittest.TestCase):
    """Each option builder carries the fake summary's numbers into its series; nothing is read from ~/vgc-data or written."""

    def summary(self):
        numbers = {R.CONTROL: (0.0080, 0.0050, 0.0030, 1500), "sched_linear_species_first": (0.0075, 0.0049, 0.0037, 1580),
                   "sched_linear_random": (0.0350, 0.0290, 0.0270, 2600)}
        grids = {}
        for row, (*errors, line_ups) in numbers.items():
            for i, (g, e) in enumerate(zip(R.GUIDANCES, errors)):
                grids[R.grid_name(row, g)] = dict(grid(row, g, e, line_ups + 400 * i), got_by_ask={str(k): k / 24 - 0.01 * (row != R.CONTROL) for k in range(1, 24)},
                                                  win_top_asks=0.50 - 0.01 * (row != R.CONTROL))
        multiples = {f"guidance_{g}": dict(error_multiple=m, error_multiple_interval=[m - 1, m + 2]) for g, m in zip(R.GUIDANCES, (4.5, 5.5, 6.5))}
        return dict(grids=grids, random_order_against_species_first=multiples,
                    row_verdicts={"sched_linear_species_first": dict(overall="matches"), "sched_linear_random": dict(overall="loses")})

    def test_error_against_line_ups_draws_a_curve_and_bars_for_each_row_and_the_default_model(self):
        option, title = F.error_against_line_ups(self.summary())
        lines = {s["name"]: s for s in option["series"] if s["type"] == "line"}
        self.assertEqual(list(lines), [F.label(row) for row in R.ROWS])
        self.assertEqual([p["value"] for p in lines[F.label(R.CONTROL)]["data"]], [[1500, 0.008, 2], [1900, 0.005, 4], [2300, 0.003, 8]])
        bars = [s for s in option["series"] if s["type"] == "custom"]
        self.assertEqual(bars[2]["data"][0], [2600, 0.035 - 0.0004, 0.035 + 0.0004])
        default = next(s for s in option["series"] if s["type"] == "scatter")
        self.assertEqual((default["name"], default["data"]), ("Today's default model, guidance 2", [[1437, 0.0053, 2]]))
        self.assertEqual(option["yAxis"]["type"], "log")
        self.assertIn("matches the control at every guidance", title); self.assertIn("adds at least 1,100 line-ups at 4 to 9 times", title)

    def test_got_against_ask_has_three_panels_without_random_order(self):
        option, title = F.got_against_ask(self.summary())
        self.assertEqual(len(option["grid"]), 3)
        names = {s["name"] for s in option["series"]}
        self.assertEqual(names, {"Got equals ask", F.label(R.CONTROL), F.label("sched_linear_species_first")})
        linear_g8 = next(s for s in option["series"] if s["name"] == F.label("sched_linear_species_first") and s["xAxisIndex"] == 2)
        self.assertEqual(linear_g8["data"][11], [12, round(12 / 24 - 0.01, 4)])
        self.assertEqual(option["series"][0]["markLine"]["data"], [dict(xAxis=17)])
        self.assertIn("does not lift got", title); self.assertIn("at most 0.010", title)

    def test_random_order_multiple_puts_the_earlier_bar_first_then_the_three_guidance_values(self):
        option, title = F.random_order_error_multiple(self.summary())
        bars, intervals = option["series"]
        self.assertEqual([b["value"] for b in bars["data"]], [round(0.0534 / 0.0192, 2), 4.5, 5.5, 6.5])
        self.assertEqual(bars["data"][0]["itemStyle"]["color"], F.GREY)
        self.assertEqual(intervals["data"], [[1, 3.5, 6.5], [2, 4.5, 7.5], [3, 5.5, 8.5]])
        self.assertEqual(bars["markLine"]["data"], [dict(yAxis=1)])
        self.assertEqual(title, "On the linear model, random order's error is 4.5 to 6.5 times species-first's, higher than the 2.8 measured on a per-field model")

    def test_line_ups_by_ask_draws_a_line_for_each_guidance_and_names_both_directions(self):
        summary = self.summary()
        for i, g in enumerate(R.GUIDANCES):
            summary["grids"][R.grid_name(R.CONTROL, g)].update(line_ups_by_ask={str(k): 180 - 30 * i for k in range(1, 24)}, asks_sharing_a_line_up=2.5 - 0.5 * i)
        option, title, subtitle = F.line_ups_by_ask(summary)
        self.assertEqual([s["name"] for s in option["series"]], ["Guidance 2", "Guidance 4", "Guidance 8"])
        self.assertEqual(option["series"][2]["data"], [120] * 23)
        self.assertIn("The mean line-ups of one ask fall from 180 to 120", title); self.assertIn("they rise from 1,500 to 2,300", title)
        self.assertIn("2.50, 2.00 and 1.50 asks on average at guidance 2, 4 and 8", subtitle)

    def test_line_ups_shared_between_asks_splits_the_ask_by_ask_count_into_the_grid_total_and_the_repeats(self):
        summary = self.summary()
        per_ask = {R.CONTROL: (180, 150, 120), "sched_linear_species_first": (180, 150, 120), "sched_linear_random": (191, 190, 189)}
        for row, counts in per_ask.items():
            for g, n in zip(R.GUIDANCES, counts):
                summary["grids"][R.grid_name(row, g)].update(line_ups_by_ask={str(k): n for k in range(1, 24)})
        option, title, _ = F.line_ups_shared_between_asks(summary)
        self.assertEqual(len(option["grid"]), 3); self.assertEqual(len(option["series"]), 6)
        self.assertEqual([t["text"] for t in option["title"]], [F.label(row) for row in R.ROWS])
        self.assertEqual(len({(y["min"], y["max"]) for y in option["yAxis"]}), 1)
        self.assertEqual([d["name"] for d in option["legend"]["data"]], ["Distinct line-ups over the grid", "Repeats: a line-up already counted at another ask"])
        for r, row in enumerate(R.ROWS):
            total, repeats = [s for s in option["series"] if s["xAxisIndex"] == r]
            grid_total = [summary["grids"][R.grid_name(row, g)]["line_ups"] for g in R.GUIDANCES]
            counted = [n * 23 for n in per_ask[row]]
            self.assertEqual(total["data"], grid_total); self.assertEqual(repeats["data"], [c - t for c, t in zip(counted, grid_total)])
            self.assertEqual(total["stack"], repeats["stack"]); self.assertEqual(total["itemStyle"]["color"], F.COLORS[row])
            self.assertEqual(repeats["itemStyle"]["color"], F.GREY)
            self.assertEqual(option["xAxis"][r]["data"], [f"Guidance {g}\n{c:,} counted\nask by ask" for g, c in zip(R.GUIDANCES, counted)])
        self.assertEqual(title, "In all three rows, raising guidance from 2 to 8 cuts the repeats between asks by 846 to 2,180, "
                                "so the grid total grows by 800 line-ups while the ask-by-ask count falls by 46 to 1,380")


if __name__ == "__main__":
    unittest.main()
