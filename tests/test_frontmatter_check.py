"""Tests for tools/check_frontmatter.py, the pre-commit check on notes/.

Run from the repository root:
    python3 -m unittest discover -s tests -v

Each test writes notes into a temporary folder and runs the check on them as
pre-commit would: one process, the staged paths as arguments.
"""

from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


CHECK = Path(__file__).resolve().parents[1] / "tools" / "check_frontmatter.py"


class FrontmatterCheck(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)

    def note(self, name, text):
        path = self.root / name
        path.write_bytes(text.encode())
        return path

    def check(self, *paths):
        return subprocess.run([sys.executable, str(CHECK), *map(str, paths)], capture_output=True, text=True)

    def test_valid_frontmatter_passes(self):
        note = self.note("dated.md", "---\ncreated_at: 2026-10-23\ntags:\n  - vocabulary\n---\n# Body\n")
        self.assertEqual(self.check(note).returncode, 0)

    def test_note_without_frontmatter_passes(self):
        note = self.note("plain.md", "# A plain note\n\nNo properties here.\n")
        self.assertEqual(self.check(note).returncode, 0)

    def test_empty_frontmatter_passes(self):
        note = self.note("empty.md", "---\n---\nBody\n")
        self.assertEqual(self.check(note).returncode, 0)

    def test_unclosed_opening_rule_is_not_frontmatter(self):
        note = self.note("rule.md", "---\nA note that opens with a horizontal rule.\n")
        self.assertEqual(self.check(note).returncode, 0)

    def test_windows_line_endings_pass(self):
        note = self.note("crlf.md", "---\r\ncreated_at: 2026-10-23\r\n---\r\nBody\r\n")
        self.assertEqual(self.check(note).returncode, 0)

    def test_invalid_yaml_fails_and_names_the_note(self):
        note = self.note("broken.md", "---\ntags: [vocabulary, glossary\n---\nBody\n")
        result = self.check(note)
        self.assertEqual(result.returncode, 1)
        self.assertIn("broken.md", result.stdout)

    def test_only_the_broken_note_is_listed(self):
        good = self.note("good.md", "---\ncreated_at: 2026-10-23\n---\n")
        bad = self.note("bad.md", "---\ntitle: a: b: c\n---\n")
        result = self.check(good, bad)
        self.assertEqual(result.returncode, 1)
        self.assertIn("bad.md", result.stdout)
        self.assertNotIn("good.md", result.stdout)

    def test_no_paths_passes(self):
        self.assertEqual(self.check().returncode, 0)


if __name__ == "__main__":
    unittest.main()
