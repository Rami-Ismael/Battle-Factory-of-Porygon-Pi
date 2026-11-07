"""The training table keys a neutral or missing nature as Serious, as the matchup database stores it.

Otherwise a team battled through the matchup database gets two rows: the label on its Serious copy, none on the
original (2,271 teams on 2026-11-06). Needs this laptop's /tmp/vgc-pilot runtime; skipped without it.

    /tmp/vgc-pilot/.venv/bin/python scripts/test_lossdata_nature.py   (from code/)
"""
import unittest
from pathlib import Path

import repo_paths  # noqa: F401  (src/ first on sys.path)

LD = None
if Path("/tmp/vgc-pilot/data/pokedex.json").exists():
    import lossdata as LD


@unittest.skipUnless(LD is not None, "the /tmp/vgc-pilot runtime is not available")
class NeutralNatureAsSerious(unittest.TestCase):
    def natures(self, *given):
        return [s["nature"] for s in LD.neutral_as_serious([{"nature": n} for n in given])]

    def test_neutral_and_missing_natures_become_serious(self):
        self.assertEqual(self.natures("Bashful", "docile", "Hardy", "Quirky", "", None),
                         ["Serious"] * 6)

    def test_natures_that_change_stats_stay(self):
        self.assertEqual(self.natures("Adamant", "Timid", "Serious"), ["Adamant", "Timid", "Serious"])

    def test_a_paste_and_its_matchup_database_copy_get_one_key(self):
        from asked_vs_got import setup
        _, V, *_ = setup()
        con = LD.MDB.connect()
        paste = con.execute("SELECT paste FROM team WHERE legal=1 AND paste LIKE '%Serious Nature%' LIMIT 1").fetchone()[0]
        written = paste.replace("Serious Nature\n", "Bashful Nature\n", 1)
        enc = lambda p: V.encode(LD.neutral_as_serious(LD.parse_team_text(p))).tobytes()
        self.assertNotEqual(written, paste)
        self.assertEqual(enc(written), enc(paste))


if __name__ == "__main__":
    unittest.main()
