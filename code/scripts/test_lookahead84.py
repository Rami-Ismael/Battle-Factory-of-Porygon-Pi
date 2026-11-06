"""The lookahead on the 84-field model (issue #205), through the arm objects the ask grid takes: sample() and to_paste().
A tiny model trained for two epochs on 256 corpus teams, and the real Showdown validator, as in test_team84.py.
Needs this laptop's /tmp/vgc-pilot data and the v3 loop-teams checkpoint (for the vocabulary); skipped without them.

    /tmp/vgc-pilot/.venv/bin/python scripts/test_lookahead84.py   (from code/)
"""
import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import repo_paths                # src/ first on sys.path; ROOT finds the repository

BEFORE = "3115c395"                  # main before the lookahead: random decoding as it drew then
ASK, GUIDANCE, SEEDS, N = 0.5, 2.0, (1, 2), 48

VOCAB = None
if Path("/tmp/vgc-pilot/data/pokedex.json").exists():
    import team84 as T
    import asked_vs_got as X
    VOCAB = T.regulation_vocab()
    CORPUS = X.setup()[0]


def module_at(commit, path, name):
    """The module at path as it was at commit, loaded under name; None without git or the commit."""
    root = repo_paths.ROOT
    try:
        text = subprocess.run(["git", "show", f"{commit}:code/{path}"], cwd=root, capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    tmp = Path(tempfile.mkdtemp()) / f"{name}.py"; tmp.write_text(text)
    spec = importlib.util.spec_from_file_location(name, tmp); mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod; spec.loader.exec_module(mod)
    return mod


@unittest.skipUnless(VOCAB is not None, "the /tmp/vgc-pilot runtime or the v3 checkpoint is not available")
class LookaheadArms(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        import diffusion84 as M
        import remask84 as R
        from hps_generate import Validator
        cls.torch, cls.R = torch, R
        rows = []
        for team in CORPUS:
            try: rows.append(T.encode(VOCAB, team))
            except ValueError: continue
            if len(rows) == 256: break
        cls.model = M.Model84.train_new(VOCAB, CORPUS, rows, labels=None, passes=2, seed=0)
        cls.drawn = {la: [cls.draw(R.RandomDecodingArm(cls.model, lookahead=la), s) for s in SEEDS] for la in (False, True)}
        val = Validator()
        try:
            cls.accepted = {la: sum(val(cls.model.to_paste(r, repair=True)[0]) is None for rows in cls.drawn[la] for r in rows)
                            for la in (False, True)}
        finally:
            val.close()

    @classmethod
    def draw(cls, arm, seed, guidance=GUIDANCE, n=N):
        cls.torch.manual_seed(seed)
        return arm.sample(n, ask=ASK, guidance=guidance).cpu().numpy()

    def test_species_first_draws_the_same_rows_with_the_lookahead_on_and_off(self):
        for seed in SEEDS:
            off = self.draw(self.R.SpeciesFirstArm(self.model, lookahead=False), seed)
            on = self.draw(self.R.SpeciesFirstArm(self.model, lookahead=True), seed)
            self.assertTrue((off == on).all())

    def test_random_decoding_with_the_switch_off_draws_the_rows_it_drew_before_the_lookahead(self):
        before = module_at(BEFORE, "src/remask84.py", "remask84_before")
        if before is None: self.skipTest(f"git or commit {BEFORE} not available")
        for seed, now in zip(SEEDS, self.drawn[False]):
            self.torch.manual_seed(seed)
            then = before.sample_random(self.model, N, ASK, "rand_plain", {}).cpu().numpy()
            self.assertTrue((now == then).all())

    def test_the_switch_is_off_by_default(self):
        default = self.draw(self.R.RandomDecodingArm(self.model), SEEDS[0])
        self.assertTrue((default == self.drawn[False][0]).all())

    def test_the_lookahead_gives_more_teams_showdown_accepts(self):
        print(f"\n  Showdown accepts {self.accepted[False]} of {N * len(SEEDS)} without the lookahead, "
              f"{self.accepted[True]} with it", flush=True)
        self.assertGreater(self.accepted[True], self.accepted[False])

    def test_the_arm_uses_the_guidance_weight_it_is_given_and_refuses_settings_it_does_not_use(self):
        arm = self.R.RandomDecodingArm(self.model, lookahead=True)
        self.assertFalse((self.draw(arm, SEEDS[0], guidance=1.0) == self.drawn[True][0]).all())
        with self.assertRaises(TypeError): arm.sample(4, ask=ASK, guidance=GUIDANCE, top_p=0.9)

    def test_every_team_is_complete_and_keeps_its_stat_points_in_budget_and_in_its_paste(self):
        for rows in self.drawn[True]:
            self.assertTrue((rows != 0).all())
            for r in rows:
                want = sorted(tuple(s["evs"][k] for k in T.SP.STATS) for s in T.decode(VOCAB, r))
                for evs in want:
                    self.assertTrue(all(0 <= v <= 32 for v in evs)); self.assertLessEqual(sum(evs), 66)
                _, slots = self.model.to_paste(r, repair=True)
                self.assertEqual(sorted(tuple(s["evs"].get(k, 0) for k in T.SP.STATS) for s in slots), want)

    def test_the_ask_grid_draw_loop_takes_a_lookahead_arm_unchanged(self):
        import ask_grid84 as A
        with mock.patch.multiple(A, KEEP=4, BATCH=8, MAX_TRIES=32):
            kept, stat_points, counts = A.draw(self.R.RandomDecodingArm(self.model, lookahead=True), ASK, SEEDS[0])
        self.assertLessEqual(counts["tried"], 32)
        self.assertEqual(counts["tried"], counts["rejected"] + counts["heldout_dropped"] + len(kept))
        self.assertAlmostEqual(counts["legal"], (counts["tried"] - counts["rejected"]) / counts["tried"])
        self.assertEqual(sum(n for _, n in counts["reasons"]), counts["rejected"])
        self.assertEqual(len(stat_points), len(kept))
        self.assertTrue(len(kept) == 4 or counts["tried"] == 32)


if __name__ == "__main__":
    unittest.main()
