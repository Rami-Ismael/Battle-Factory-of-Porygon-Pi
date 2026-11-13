"""The chart recorder behind scripts/test_chart_ports.py: src/chart_record.py.

    python scripts/test_chart_record.py   (from code/)
"""
import unittest

import repo_paths  # noqa: F401  (src/ first on sys.path)
import chart_record


class RecordOption(unittest.TestCase):
    def test_a_series_on_the_second_y_axis_uses_that_axis(self):
        option = dict(xAxis=dict(type="value"), yAxis=[dict(type="value"), dict(type="category", data=["a", "b"])],
                      series=[dict(yAxisIndex=1, data=[5, 7])])
        self.assertEqual(chart_record.record_option(option)["points"], [[5.0, 0.0], [7.0, 1.0]])

    def test_a_y_axis_index_past_the_axes_is_named_in_the_error(self):
        option = dict(xAxis=dict(type="category"), yAxis=dict(type="value"), series=[dict(name="Win rate", yAxisIndex=1, data=[1, 2])])
        with self.assertRaises(ValueError) as stop:
            chart_record.record_option(option)
        self.assertIn("Win rate", str(stop.exception))
        self.assertIn("yAxisIndex 1", str(stop.exception))

    def test_an_option_without_a_y_axis_still_records(self):
        self.assertEqual(chart_record.record_option(dict(series=[dict(data=[3])]))["points"], [[0.0, 3.0]])


class Missing(unittest.TestCase):
    def record(self, *points):
        return dict(texts=[], points=[list(point) for point in points])

    def test_values_on_either_side_of_a_rounding_boundary_still_match(self):
        """0.123456 and 0.123455 round to 0.12346 and 0.12345 at 5 decimals, and they differ by one part in a million."""
        self.assertNotEqual(round(0.123456, 5), round(0.123455, 5))
        words, lost = chart_record.missing(self.record((1.0, 0.123456)), self.record((1.0, 0.123455)))
        self.assertEqual((words, lost), ([], []))

    def test_a_point_further_off_than_the_tolerance_is_reported(self):
        _words, lost = chart_record.missing(self.record((1.0, 0.5)), self.record((1.0, 0.50002)))
        self.assertEqual(lost, [[1.0, 0.5]])

    def test_both_coordinates_must_match(self):
        _words, lost = chart_record.missing(self.record((1.0, 0.5)), self.record((2.0, 0.5), (1.0, 0.6)))
        self.assertEqual(lost, [[1.0, 0.5]])

    def test_missing_words_are_listed(self):
        words, _lost = chart_record.missing(dict(texts=["Win rate", "Ask"], points=[]), dict(texts=["Ask"], points=[]))
        self.assertEqual(words, ["Win rate"])


if __name__ == "__main__":
    unittest.main()
