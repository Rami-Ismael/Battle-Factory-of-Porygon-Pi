"""The t check figure for the new per-field model (src/masking_rate_preflight_figures.py), built from a made-up probe.

Reads no stored results and samples nothing.

    /tmp/vgc-pilot/.venv/bin/python scripts/test_masking_rate_preflight_figures.py   (from code/)
"""
import tempfile
import unittest
from pathlib import Path

import repo_paths  # noqa: F401  (src/ first on sys.path)
import echarts_chart
import masking_rate_preflight_figures as F

FIELDS = ("species", "ability_and_item", "moves", "stat_alignment", "stat_points")


def probe(scale):
    """Corrected-t readings scale * (ask index + 1) * (field index + 1) / 1000, at three asks."""
    by_ask = {ask: dict(total_variation_by_rule_and_field_type=dict(corrected={f: scale * (i + 1) * (j + 1) / 1000 for j, f in enumerate(FIELDS)}))
              for i, ask in enumerate(("6/24", "12/24", "17/24"))}
    return dict(bar=0.01, sampling_seed=7301, by_ask=by_ask)


class NewPerFieldModelFigure(unittest.TestCase):
    def setUp(self):
        results = {"t_sensitivity": probe(0.1), f"t_sensitivity_{F.NEW_PER_FIELD_MODEL}": probe(0.5)}
        echarts_chart.CAPTURE = []
        self.addCleanup(setattr, echarts_chart, "CAPTURE", None)
        with tempfile.TemporaryDirectory() as tmp:
            self.path = F.t_sensitivity_new_per_field_model(results, Path(tmp))
        (_, self.option, self.texts, _), = echarts_chart.CAPTURE

    def test_one_bar_series_per_model_and_ask(self):
        series = self.option["series"]
        self.assertEqual([s["name"] for s in series],
                         [f"{m}, ask {a}" for m in ("New per-field model", "Default model") for a in ("6/24", "12/24", "17/24")])
        self.assertEqual(series[0]["data"], [0.0005, 0.001, 0.0015, 0.002, 0.0025])   # new model, ask 6/24
        self.assertEqual(series[5]["data"], [0.0003, 0.0006, 0.0009, 0.0012, 0.0015])  # default model, ask 17/24
        self.assertTrue(all(s["emphasis"] == dict(focus="series") for s in series))
        self.assertEqual(self.option["tooltip"]["trigger"], "axis")

    def test_bar_line_at_the_stored_bar(self):
        self.assertEqual(self.option["series"][0]["markLine"]["data"], [dict(yAxis=0.01)])

    def test_title_states_the_new_models_largest_reading(self):
        title, subtitle, note = self.texts
        self.assertIn("0.0075", title)  # 0.5 * 3 * 5 / 1000, Stat Points at ask 17/24
        self.assertIn("Stat Points at ask 17/24", note)
        self.assertIn("0.0015", note)    # the default model's largest
        for text in (title, subtitle):
            self.assertNotIn("—", text); self.assertNotIn("(", text)
        self.assertEqual(self.path.name, "11_t_sensitivity_new_per_field_model.html")


if __name__ == "__main__":
    unittest.main()
