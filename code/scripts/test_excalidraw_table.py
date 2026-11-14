"""Tests for the Markdown-table-to-Excalidraw plan. They never start a browser.

Run:  python scripts/test_excalidraw_table.py
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import excalidraw_table as drawn

COMMENT = """Lead sentence.

| Row | Why its total differs | Evidence |
| --- | --- | --- |
| Per-field, uncorrected t, the control | Neighbouring asks draw the same line-ups | 2,216 of 2,329 repeats. The top 20 take 239 slots. Novel line-ups go from 502 to 1,236. Training line-ups fall to 890 |
| Linear, species-first | The **same** thing | The rows share 776 line-ups. The gap is `noise` |
| Linear, random order | It rarely draws a training line-up | 12% of teams. Teams win 0.236, against 0.323 |

Closing sentence.

| A | B |
| - | - |
| 1 | 2 |
"""


class Parsing(unittest.TestCase):
    def test_finds_every_table_with_its_lines(self):
        first, second = drawn.tables_in(COMMENT)
        self.assertEqual((first[0], first[1]), (2, 6))
        self.assertEqual(first[2], ["Row", "Why its total differs", "Evidence"])
        self.assertEqual(len(first[3]), 3)
        self.assertEqual((second[2], second[3]), (["A", "B"], [["1", "2"]]))

    def test_a_line_of_pipes_without_a_rule_is_not_a_table(self):
        self.assertEqual(drawn.tables_in("| a | b |\n| c | d |\n| e | f |\n"), [])

    def test_plain_drops_the_marks_excalidraw_cannot_draw(self):
        self.assertEqual(drawn.plain("The **same** `thing`, see [the map](https://example.com/1)"), "The same thing, see the map")

    def test_a_decimal_point_does_not_end_a_sentence(self):
        self.assertEqual(drawn.sentences("Teams win 0.236, against 0.323. The rest lose"), ["Teams win 0.236, against 0.323", "The rest lose"])


class Plan(unittest.TestCase):
    def setUp(self):
        _, _, header, rows = drawn.tables_in(COMMENT)[0]
        self.spec = drawn.spec_of(header, rows, widths=[220, 430, 630])

    def test_a_column_with_a_four_sentence_cell_turns_into_bullets(self):
        for row, count in zip(self.spec["rows"], (4, 2, 2), strict=True):
            self.assertEqual([block["bullet"] for block in row["cells"][2]], [True] * count)
            self.assertEqual([block["bullet"] for block in row["cells"][1]], [False])
            self.assertEqual([block["bullet"] for block in row["cells"][0]], [False])

    def test_the_three_rows_keep_the_colours_of_their_figures(self):
        self.assertEqual([(row["label"], row["tint"], row["strong"]) for row in self.spec["rows"]], [drawn.BLUE, drawn.TEAL, drawn.ORANGE])

    def test_no_word_of_the_table_is_lost(self):
        text = " ".join(block["text"] for row in self.spec["rows"] for cell in row["cells"] for block in cell)
        for words in ("2,216 of 2,329 repeats", "Training line-ups fall to 890", "The same thing", "The gap is noise", "Teams win 0.236, against 0.323"):
            self.assertIn(words, text)

    def test_widths_default_to_a_share_of_1280(self):
        _, _, header, rows = drawn.tables_in(COMMENT)[0]
        widths = drawn.spec_of(header, rows)["widths"]
        self.assertEqual(len(widths), 3)
        self.assertLess(widths[0], widths[2])
        self.assertAlmostEqual(sum(widths), 1280, delta=3)

    def test_unknown_rows_cycle_through_the_palette(self):
        spec = drawn.spec_of(["A", "B"], [["one", "x"], ["two", "y"]])
        self.assertEqual([row["label"] for row in spec["rows"]], [drawn.BLUE[0], drawn.TEAL[0]])


class Scene(unittest.TestCase):
    def scene(self, texts):
        elements = [dict(type="rectangle"), *(dict(type="text", text=wrapped, originalText=whole) for wrapped, whole in texts)]
        folder = tempfile.TemporaryDirectory(); self.addCleanup(folder.cleanup)
        path = Path(folder.name) / "scene.excalidraw"
        path.write_text(json.dumps(dict(elements=elements)), encoding="utf-8")
        return path

    def test_texts_are_read_unwrapped(self):
        self.assertEqual(drawn.texts_of(self.scene([("Novel line-\nups", "Novel line-ups"), ("1,236", None)])), ["Novel line-ups", "1,236"])

    def test_a_dropped_number_or_word_is_reported_and_a_full_scene_is_not(self):
        header, rows = ["Row", "Guidance", "Line-ups"], [["Linear, **random** order", "8", "3,820"]]
        self.assertEqual(drawn.missing_from_scene(header, rows, self.scene([("Row Guidance Line-ups", None), ("Linear, random order at 8: 3,820", None)])), [])
        self.assertEqual(drawn.missing_from_scene(header, rows, self.scene([("Row Guidance Line-ups", None), ("Linear order 8 and 3,82", None)])), ["3,820", "random"])


class Retry(unittest.TestCase):
    def test_a_failed_fetch_is_tried_again_and_another_error_is_not(self):
        calls = []
        def once(*arguments):
            calls.append(arguments)
            if len(calls) == 1: raise RuntimeError("Failed to fetch dynamically imported module")
            return ["scene"]
        with mock.patch.object(drawn, "render_once", once), mock.patch.object(drawn.time, "sleep"):
            self.assertEqual(drawn.render(dict(skeleton=[]), "folder", "name"), ["scene"])
        self.assertEqual(len(calls), 2)
        with mock.patch.object(drawn, "render_once", side_effect=RuntimeError("the page says 'working'")) as broken, self.assertRaises(RuntimeError):
            drawn.render(dict(skeleton=[]), "folder", "name")
        self.assertEqual(broken.call_count, 1)


if __name__ == "__main__":
    unittest.main()
