"""Regression tests for the Kingambit-free team finder (blog/kingambit-finder, served by GitHub Pages).

Run from the repository root:
    python3 -m unittest discover -s tests -v

Two layers:
  * data contract: the numbers embedded in the built page are internally consistent and match the sources
    they were drawn from (win rate = wins / battles, matchup cells add up to the team totals, no Kingambit).
  * search logic: the page's own script is executed in Node against a stub DOM, and the filters
    (Must bring, Leave out, minimum battles, ranking basis) are checked against an independent recount.
These do not drive a real browser; layout and motion were checked by hand in the browser pane.
"""

import json
import math
import re
import shutil
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "blog/kingambit-finder/index.html"
HARNESS = Path(__file__).resolve().parent / "no_kingambit_finder_harness.js"


def load_data():
    text = PAGE.read_text()
    match = re.search(r"const D = (\{.*?\});\nconst T = D\.teams", text, re.S)
    if not match:
        raise AssertionError("embedded data block not found in the built page")
    return json.loads(match.group(1)), text


def wilson_low(wins, battles, z=1.96):
    p = wins / battles
    d = 1 + z * z / battles
    c = p + z * z / (2 * battles)
    m = z * math.sqrt(p * (1 - p) / battles + z * z / (4 * battles * battles))
    return (c - m) / d


DATA, PAGE_TEXT = load_data()
TEAMS = DATA["teams"]


class DataContract(unittest.TestCase):
    def test_no_team_brings_kingambit(self):
        offenders = [t["id"] for t in TEAMS if any(m["n"] == "Kingambit" for m in t["mons"])]
        self.assertEqual(offenders, [])

    def test_every_team_is_six_sets_with_enough_battles(self):
        for t in TEAMS:
            self.assertEqual(len(t["mons"]), 6, t["id"])
            self.assertGreaterEqual(t["b"], 96, t["id"])
            self.assertTrue(0 <= t["w"] <= t["b"], t["id"])
            self.assertEqual(len({m["n"] for m in t["mons"]}), 6, f"{t['id']} repeats a species")

    def test_team_ids_are_unique(self):
        ids = [t["id"] for t in TEAMS]
        self.assertEqual(len(ids), len(set(ids)))

    def test_opponents_are_49_distinct_placed_teams(self):
        opp = DATA["opp"]
        self.assertEqual(len(opp), 49)
        self.assertEqual(len({o["id"] for o in opp}), 49)
        for o in opp:
            self.assertEqual(len(o["mons"]), 6, o["id"])
            self.assertTrue(o["place"] and o["event"], f"{o['id']} lacks a placement or event")

    def test_matchup_cells_add_up_to_team_totals(self):
        for t in TEAMS:
            self.assertEqual(len(t["cw"]), 49, t["id"])
            self.assertEqual(len(t["cn"]), 49, t["id"])
            for w, n in zip(t["cw"], t["cn"]):
                self.assertTrue(0 <= w <= n, t["id"])
            if t["opp"] == 49:      # matrix database teams: cells are the whole record
                self.assertEqual((sum(t["cw"]), sum(t["cn"])), (t["w"], t["b"]), t["id"])
            else:                   # counter-grid teams faced 50 opponents; one is not a column here
                self.assertLessEqual(sum(t["cn"]), t["b"], t["id"])
                self.assertGreaterEqual(sum(t["cn"]), t["b"] - 48, t["id"])

    def test_every_species_has_a_sprite(self):
        names = {m["n"] for t in TEAMS for m in t["mons"]} | {n for o in DATA["opp"] for n in o["mons"]}
        missing = sorted(n for n in names if not DATA["sprites"].get(n, "").startswith("data:image/png;base64,"))
        self.assertEqual(missing, [])

    def test_best_real_team_is_mb607_at_0_691(self):
        real = [t for t in TEAMS if t["k"] == "real" and t["b"] == 2400]
        self.assertEqual(len(real), 4)
        best = max(real, key=lambda t: t["w"] / t["b"])
        self.assertEqual(best["id"], "MB607")
        self.assertEqual(round(best["w"] / best["b"], 3), 0.691)
        self.assertEqual(sorted(m["n"] for m in best["mons"]),
                         sorted(["Sinistcha", "Milotic", "Incineroar", "Raichu", "Mimikyu", "Ninetales-Alola"]))

    def test_mb607_has_the_highest_cautious_rating(self):
        top = max(TEAMS, key=lambda t: wilson_low(t["w"], t["b"]))
        self.assertEqual(top["id"], "MB607")
        self.assertAlmostEqual(wilson_low(top["w"], top["b"]), 0.672, places=3)

    def test_raw_ranking_would_prefer_a_team_with_few_battles(self):
        # The page's reason for a cautious rating: the best raw win rate belongs to a small-sample team.
        raw_top = max(TEAMS, key=lambda t: t["w"] / t["b"])
        self.assertLess(raw_top["b"], 2400)
        self.assertGreater(raw_top["w"] / raw_top["b"], 0.691)


class PagesPage(unittest.TestCase):
    """What GitHub Pages and a shared link need: a real title, a description, one self-contained file."""

    def test_title_and_social_metadata(self):
        self.assertIn("<title>Kingambit-Free Team Finder</title>", PAGE_TEXT)
        self.assertRegex(PAGE_TEXT, r'<meta name="description" content="[^"]{40,}')
        self.assertIn('<meta property="og:title" content="Kingambit-Free Team Finder">', PAGE_TEXT)

    def test_page_is_self_contained(self):
        # Pages serves the file as is; nothing may load from another host.
        external = re.findall(r'(?:src|href)="(https?://[^"]+)"', PAGE_TEXT.split("<script>")[0])
        self.assertEqual(external, [])

    def test_dark_theme_sets_color_scheme(self):
        style = PAGE_TEXT.split("</style>")[0]
        media = style.split("@media (prefers-color-scheme:dark)")[1].split("}}")[0]
        explicit = style.split(':root[data-theme="dark"]')[1].split("}")[0]
        self.assertIn("color-scheme:dark", media)
        self.assertIn("color-scheme:dark", explicit)

    def test_readme_in_folder_names_the_rebuild_command(self):
        readme = (PAGE.parent / "README.md").read_text()
        self.assertIn("python3 build.py", readme)
        self.assertTrue((PAGE.parent / "template.html").exists())


@unittest.skipUnless(shutil.which("node"), "node is required for the search-logic tests")
class SearchLogic(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        def run(exprs):
            proc = subprocess.run(["node", str(HARNESS), str(PAGE), json.dumps(exprs)],
                                  capture_output=True, text=True, timeout=60)
            if proc.returncode != 0:
                raise AssertionError(proc.stdout + proc.stderr)
            return json.loads(proc.stdout)
        cls.run_js = staticmethod(run)

    def evaluate(self, *exprs):
        result = self.run_js(list(exprs))
        self.assertTrue(result.get("ok"), result)
        for x in result["out"]:
            self.assertFalse(isinstance(x, str) and x.startswith("ERR "), x)
        return result["out"]

    def test_script_runs_and_loads_all_teams(self):
        n, opp = self.evaluate("T.length", "D.opp.length")
        self.assertEqual((n, opp), (len(TEAMS), 49))

    def test_default_filter_returns_every_team_best_first(self):
        count, first, order_ok = self.evaluate(
            "filtered().length", "filtered()[0].id",
            "filtered().every((t, i, a) => i === 0 || score(a[i-1]) >= score(t))")
        self.assertEqual(count, len(TEAMS))
        self.assertEqual(first, "MB607")
        self.assertTrue(order_ok)

    def test_must_bring_matches_an_independent_recount(self):
        (count,) = self.evaluate("S.need = new Set(['Sinistcha']); filtered().length")
        expected = sum(1 for t in TEAMS if any(m["n"] == "Sinistcha" for m in t["mons"]))
        self.assertEqual(count, expected)
        self.assertGreater(expected, 0)

    def test_leave_out_and_must_bring_combine(self):
        (count,) = self.evaluate("S.need = new Set(['Sinistcha']); S.out = new Set(['Garchomp']); filtered().length")
        expected = sum(1 for t in TEAMS
                       if any(m["n"] == "Sinistcha" for m in t["mons"])
                       and not any(m["n"] == "Garchomp" for m in t["mons"]))
        self.assertEqual(count, expected)

    def test_minimum_battles_filter(self):
        (count,) = self.evaluate("S.need = new Set(); S.out = new Set(); S.min = 784; filtered().length")
        self.assertEqual(count, sum(1 for t in TEAMS if t["b"] >= 784))

    def test_source_filter_real_teams_only(self):
        (count,) = self.evaluate("S.min = 96; S.src = 'real'; filtered().length")
        self.assertEqual(count, sum(1 for t in TEAMS if t["k"] == "real"))

    def test_raw_basis_reorders_toward_small_samples(self):
        first_cautious, first_raw = self.evaluate(
            "S.src = 'all'; S.basis = 'cautious'; filtered()[0].id",
            "S.basis = 'raw'; filtered()[0].id")
        self.assertEqual(first_cautious, "MB607")
        self.assertNotEqual(first_raw, "MB607")

    def test_wilson_interval_matches_reference(self):
        lo, hi = self.evaluate("(() => { const w = wilson(1658, 2400); return [w.lo, w.hi]; })()")[0]
        self.assertAlmostEqual(lo, wilson_low(1658, 2400), places=9)
        self.assertGreater(hi, lo)

    def test_opponent_names_lead_with_the_placement_not_the_identifier(self):
        (name,) = self.evaluate("oppName({id: 'MB155', place: 'Runner Up', event: 'Example Cup'})")
        self.assertEqual(name, "Runner Up at Example Cup (MB155)")

    def test_matrix_tooltip_counts_match_the_data(self):
        text, = self.evaluate("(() => { const t = byId['MB607'], i = 0; return mxTipText(1, t, D.opp[i], i); })()")
        t = next(x for x in TEAMS if x["id"] == "MB607")
        self.assertIn(f"{t['cw'][0]} of {t['cn'][0]} battles", text)


if __name__ == "__main__":
    unittest.main()
