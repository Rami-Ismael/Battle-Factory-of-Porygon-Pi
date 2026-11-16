"""Random order without the lookahead, redrawn until Showdown accepts (issue #358): the ask-grid row and its reader.

The ask-grid settings are read from the environment at import, so the row cases run ask_grid84 in their own process.
The sampler case trains a tiny model for two epochs on 256 corpus teams, as test_species_then_random.py does. The
reader case reads the stored species-then-random grid in place of the new row. All need this laptop's /tmp/vgc-pilot
data and results; they are skipped without them.

    /tmp/vgc-pilot/.venv/bin/python scripts/test_plain_reject.py   (from code/)
"""
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

import repo_paths  # noqa: F401  (src/ first on sys.path)

PY, SRC = "/tmp/vgc-pilot/.venv/bin/python", Path(__file__).resolve().parents[1] / "src"
CHECKPOINTS = Path.home() / ".local/share/vgc-pilot-runtime/lossdown"
RES = Path.home() / "vgc-data/results"
HERE = Path("/tmp/vgc-pilot/data/pokedex.json").exists() and (CHECKPOINTS / "stat84_shuffled_noise_linear_es3_cap80.pt").exists()
STORED = HERE and all((RES / f"masking_rate_rows_opponents_{r}.npz").exists() for r in
                      ("sched_linear_species_then_random", "sched_linear_species_first", "sched_linear_random"))

PROBE = ("import json, ask_grid84 as A; s = A.load_sampler(); "
         "print(json.dumps(dict(out=A.OUT.name, checkpoint=A.checkpoint_path().name, sampler=type(s).__name__, lookahead=getattr(s, 'lookahead', None), "
         "arm=getattr(s, 'arm', None), max_tries=A.MAX_TRIES, seed0=A.SEED0, keep=A.KEEP)))")

# A sampler whose every team Showdown refuses: draw() must stop only at the row's try cap.
DRAW_PROBE = """
import json, torch, ask_grid84 as A
class Refuse:
    def __call__(self, paste): return "Pokemon has an illegal ability"
    def close(self): pass
class Fake:
    def sample(self, n, ask=None, guidance=1.0): return torch.zeros(n, 84, dtype=torch.long)
    def to_paste(self, row, repair=False): return "paste", []
A.Validator = Refuse
kept, stats, counts = A.draw(Fake(), 15 / 24, 1)
print(json.dumps(dict(kept=len(kept), tried=counts["tried"], rejected=counts["rejected"])))
"""


def run(code, model, **env):
    done = subprocess.run([PY, "-c", code], cwd=SRC, env={**os.environ, "AG84_MODEL": model, **env}, capture_output=True, text=True, check=True)
    return json.loads(done.stdout.strip().splitlines()[-1])


@unittest.skipUnless(HERE, "needs /tmp/vgc-pilot data and the linear masking-schedule checkpoint")
class Row(unittest.TestCase):
    def test_the_row_samples_random_order_with_the_lookahead_off_on_the_linear_checkpoint(self):
        got = run(PROBE, "sched_linear_random_reject")
        self.assertEqual((got["checkpoint"], got["sampler"], got["lookahead"], got["arm"]),
                         ("stat84_shuffled_noise_linear_es3_cap80.pt", "RandomDecodingArm", False, "rand_plain"))
        self.assertEqual((got["out"], got["seed0"], got["keep"]), ("stat_points_84_sched_linear_random_reject.json", 7300, 64))

    def test_only_this_row_raises_the_try_cap(self):
        self.assertEqual(run(PROBE, "sched_linear_random_reject")["max_tries"], 20_000)
        self.assertEqual(run(PROBE, "sched_linear_random")["max_tries"], 1280)

    def test_draw_keeps_counting_tries_past_1280_for_this_row_only(self):
        self.assertEqual(run(DRAW_PROBE, "sched_linear_random_reject"), dict(kept=0, tried=20_000, rejected=20_000))
        self.assertEqual(run(DRAW_PROBE, "sched_linear_random"), dict(kept=0, tried=1280, rejected=1280))


@unittest.skipUnless(HERE, "the /tmp/vgc-pilot runtime is not available")
class Sampler(unittest.TestCase):
    def test_the_arm_draws_the_same_teams_as_sample_random_with_the_lookahead_off(self):
        import torch
        import asked_vs_got as X
        import diffusion84 as M
        import remask84 as R
        import team84 as T
        vocab, corpus, rows = T.regulation_vocab(), X.setup()[0], []
        for team in corpus:
            try: rows.append(T.encode(vocab, team))
            except ValueError: continue
            if len(rows) == 256: break
        model = M.Model84.train_new(vocab, corpus, rows, labels=None, epochs=2, seed=0)
        for seed in (1, 2):
            torch.manual_seed(seed); arm = R.RandomDecodingArm(model, "rand_plain", lookahead=False).sample(48, ask=15 / 24, guidance=2.0)
            torch.manual_seed(seed); direct = R.sample_random(model, 48, 15 / 24, "rand_plain", {}, guidance=2.0, lookahead=False)
            self.assertTrue(bool((arm == direct).all()), f"seed {seed}")
            torch.manual_seed(seed); on = R.sample_random(model, 48, 15 / 24, "rand_plain", {}, guidance=2.0, lookahead=True)
            self.assertFalse(bool((on == direct).all()), f"seed {seed}: the lookahead changed nothing, so the test cannot tell the arms apart")


class Verdict(unittest.TestCase):
    def setUp(self):
        import plain_reject as P
        self.P = P

    def test_a_gap_no_worse_than_minus_003_confirms(self):
        self.assertEqual(self.P.verdict(-0.03, [-0.06, 0.0]), "confirmed")
        self.assertEqual(self.P.verdict(0.01, [-0.02, 0.04]), "confirmed")

    def test_clearing_needs_the_gap_and_the_interval_below_minus_003(self):
        self.assertEqual(self.P.verdict(-0.05, [-0.08, -0.031]), "cleared")
        self.assertEqual(self.P.verdict(-0.06, [-0.09, -0.02]), "partial")

    def test_a_gap_between_the_two_bars_is_partial(self):
        self.assertEqual(self.P.verdict(-0.04, [-0.07, -0.035]), "partial")


@unittest.skipUnless(STORED, "needs the stored masking-schedule grids and their opponent snapshots")
class Reader(unittest.TestCase):
    """The reader on real grids, with the stored species-then-random row standing in for the new one."""
    @classmethod
    def setUpClass(cls):
        import plain_reject as P
        cls.P, cls.tmp = P, Path(tempfile.mkdtemp())
        cls.saved = (P.ROW, P.RESULT)
        stand_in = "sched_linear_species_then_random"
        P.READER[stand_in], P.STREAM[stand_in], P.ROW_COLOR[stand_in] = P.READER[P.ROW], 0, P.ROW_COLOR[P.ROW]
        P.ROW, P.RESULT = stand_in, cls.tmp / "plain_reject_grid.json"
        P.read(); cls.result = json.load(open(P.RESULT))

    @classmethod
    def tearDownClass(cls): cls.P.ROW, cls.P.RESULT = cls.saved

    def test_each_row_reads_15_cells_of_64_teams_with_49_battles_each(self):
        for row in self.result["rows"].values():
            self.assertEqual((row["cells"], row["teams"], row["battles"]), (15, 960, 47_040), row["reader"])

    def test_the_stored_rows_match_their_known_means(self):
        m = self.result["rows"]
        self.assertAlmostEqual(m["sched_linear_species_first"]["mean_win"], 0.4896, places=3)
        self.assertAlmostEqual(m["sched_linear_random"]["mean_win"], 0.3336, places=3)
        self.assertAlmostEqual(m["sched_linear_random"]["best_of_64"], 0.5947, places=3)

    def test_the_gap_is_the_difference_of_the_means_and_sits_inside_its_interval(self):
        g, m = self.result["mean_win_against_species_first"], self.result["rows"]
        self.assertAlmostEqual(g["difference"], m[self.P.ROW]["mean_win"] - m["sched_linear_species_first"]["mean_win"])
        self.assertLessEqual(g["interval"][0], g["difference"]); self.assertLessEqual(g["difference"], g["interval"][1])
        self.assertEqual(self.result["verdict"], self.P.verdict(g["difference"], g["interval"]))

    def test_the_figures_and_comment_carry_the_verdict_and_both_pictures(self):
        self.P.figure(self.tmp)
        for page in ("plain-reject-gap.html", "plain-reject-got.html"): self.assertTrue((self.tmp / page).exists(), page)
        import contextlib, io
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf): self.P.comment("https://example.invalid/{name}-{theme}.png")
        text = buf.getvalue()
        self.assertTrue(text.startswith(f"**"), text[:80]); self.assertIn(f"Verdict {self.result['verdict']}", text.splitlines()[0])
        for name in ("plain-reject-gap", "plain-reject-got"):
            for theme in ("light", "dark"): self.assertIn(f"{name}-{theme}.png", text)


if __name__ == "__main__":
    unittest.main()
