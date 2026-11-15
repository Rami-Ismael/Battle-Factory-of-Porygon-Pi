"""The lookahead on the 84-field model (issue #205), through the arm objects the ask grid takes: sample() and to_paste().
A tiny model trained for two epochs on 256 corpus teams, and the real Showdown validator, as in test_team84.py.
Needs this laptop's /tmp/vgc-pilot data and the v3 loop-teams checkpoint (for the vocabulary); skipped without them.

    /tmp/vgc-pilot/.venv/bin/python scripts/test_lookahead84.py   (from code/)
"""
import importlib.util
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import repo_paths                # src/ first on sys.path; ROOT finds the repository

BEFORE = "3115c395"                  # main before the lookahead: random decoding as it drew then
ASK, GUIDANCE, SEEDS, N = 0.5, 2.0, (1, 2), 48
RECORD = {"kept_share", "log_kept_share", "refused_share", "refusing_fields", "fallback_fields"}   # issue #207
DRAW_KEYS = {"column", "removed_share", "top_removed", "fields_before"}                                 # issue #321, one entry per draw

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
        cls.model = M.Model84.train_new(VOCAB, CORPUS, rows, labels=None, epochs=2, seed=0)
        cls.drawn, cls.records = {}, {}
        for la in (False, True):
            arm = R.RandomDecodingArm(cls.model, lookahead=la)
            cls.drawn[la], cls.records[la] = zip(*[(cls.draw(arm, s), arm.last_refusals) for s in SEEDS])
        arm = R.RandomDecodingArm(cls.model, lookahead=True, record_draws=True)
        cls.drawn_with_draws, cls.records_with_draws = zip(*[(cls.draw(arm, s), arm.last_refusals) for s in SEEDS])
        cls.species_first = {}
        for la in (False, True):
            arm = R.SpeciesFirstArm(cls.model, lookahead=la)
            cls.species_first[la] = [(cls.draw(arm, s), arm.last_refusals) for s in SEEDS]
        arm = R.SpeciesFirstArm(cls.model, lookahead=True, record_draws=True)
        cls.species_first_draws = []
        for s in SEEDS:
            cls.draw(arm, s); cls.species_first_draws += arm.last_refusals
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
        for (off, _), (on, _) in zip(self.species_first[False], self.species_first[True]):
            self.assertTrue((off == on).all())

    def assert_records(self, records, n):
        self.assertEqual(len(records), n)
        for r in records:
            self.assertEqual(set(r), RECORD)
            self.assertIsInstance(r["refusing_fields"], int); self.assertIsInstance(r["fallback_fields"], int)
            self.assertGreaterEqual(r["fallback_fields"], 0)
            self.assertTrue(0.0 <= r["refused_share"] <= 1.0)
            self.assertAlmostEqual(r["kept_share"] + r["refused_share"], 1.0, places=12)
            if r["log_kept_share"] is None: self.assertEqual(r["kept_share"], 0.0)
            else: self.assertAlmostEqual(math.exp(r["log_kept_share"]), r["kept_share"], places=12)
            if r["refusing_fields"] == 0: self.assertEqual(r["refused_share"], 0.0)

    def test_each_lookahead_team_carries_a_record_of_what_was_refused(self):
        for records in self.records[True]: self.assert_records(records, N)
        refusing = [r["refusing_fields"] for records in self.records[True] for r in records]
        self.assertGreater(max(refusing), 0, "random decoding with the lookahead should refuse something on some team")
        steered = [r for records in self.records[True] for r in records if r["fallback_fields"] == 0 and r["refusing_fields"]]
        self.assertTrue(all(0.0 < r["kept_share"] < 1.0 for r in steered))

    def test_a_fallback_field_refuses_nothing(self):
        # A lookahead that refuses every value makes each of the 48 fields before the Stat Points fall back to allowing
        # every value; each is drawn once without remasking, so each team has 48 fallback fields and nothing refused.
        import diffusion84 as M
        refuse_all = lambda tables, x, c, al=None: self.torch.zeros_like(al[M.L.KEY_OF_COL[c]][:, 0])
        arm = self.R.RandomDecodingArm(self.model, lookahead=True)
        with mock.patch.object(M, "lookahead_rule", refuse_all):
            self.draw(arm, SEEDS[0], n=8)
        self.assert_records(arm.last_refusals, 8)
        for r in arm.last_refusals:
            self.assertEqual((r["fallback_fields"], r["refusing_fields"], r["kept_share"]), (M.STAT0, 0, 1.0))

    def test_the_draw_loop_refuses_records_that_do_not_pair_with_the_teams(self):
        import ask_grid84 as A
        class Short(self.R.RandomDecodingArm):
            def sample(self, n, **kw):
                x = super().sample(n, **kw); self.last_refusals = self.last_refusals[1:]; return x
        with mock.patch.multiple(A, KEEP=4, BATCH=8, MAX_TRIES=32), self.assertRaises(ValueError):
            A.draw(Short(self.model, lookahead=True), ASK, SEEDS[0])

    def test_a_remasking_arm_with_the_lookahead_carries_the_record_too(self):
        arm = self.R.RandomDecodingArm(self.model, "rand_remask_even", lookahead=True)
        self.draw(arm, SEEDS[0], n=8)
        self.assert_records(arm.last_refusals, 8)

    def test_nothing_is_refused_in_species_first_order_or_with_the_switch_off(self):
        records = [r for la in (False, True) for _, rs in self.species_first[la] for r in rs]
        records += [r for rs in self.records[False] for r in rs]
        self.assert_records(records, N * len(SEEDS) * 3)
        self.assertEqual(sum(r["refusing_fields"] for r in records), 0)
        self.assertTrue(all(r["refused_share"] == 0.0 and r["kept_share"] == 1.0 and r["log_kept_share"] == 0.0 for r in records))

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
        self.assert_records(counts["refusals"], len(kept))         # saved beside the pastes in each cell
        self.assertTrue(len(kept) == 4 or counts["tried"] == 32)

    def test_recording_every_draw_leaves_the_random_draws_untouched(self):
        for off, on in zip(self.drawn[True], self.drawn_with_draws):
            self.assertTrue((off == on).all())

    def test_each_team_records_every_draw_in_the_order_it_was_drawn(self):
        import diffusion84 as M
        for records in self.records_with_draws:
            for r in records:
                self.assertEqual(set(r), RECORD | {"draws"}); d = r["draws"]
                self.assertEqual(set(d), DRAW_KEYS)
                self.assertEqual(sorted(d["column"]), list(range(M.COLS)))          # no remasking: each field drawn once
                self.assertEqual(len({len(v) for v in d.values()}), 1)
                self.assertTrue(all(0.0 <= v <= 1.0 for v in d["removed_share"]))
                self.assertTrue(all(v > 0 for v, top in zip(d["removed_share"], d["top_removed"]) if top))
                # the per-team kept share is the product of the per-draw kept shares, where none was refused outright (a fallback)
                if max(d["removed_share"]) < 1.0:
                    self.assertAlmostEqual(sum(math.log1p(-v) for v in d["removed_share"]), r["log_kept_share"], places=6)

    def test_a_species_draw_counts_the_fields_of_its_pokemon_drawn_before_it(self):
        import diffusion84 as M
        slot_of = lambda c: c // M.NF if c < M.STAT0 else (c - M.STAT0) // M.SP.NSTAT
        late = 0
        for records in self.records_with_draws:
            for r in records:
                cols, before = r["draws"]["column"], r["draws"]["fields_before"]
                for i, c in enumerate(cols):
                    if c < M.STAT0 and c % M.NF == 0:
                        want = sum(slot_of(e) == slot_of(c) for e in cols[:i])
                        self.assertEqual(before[i], want); late += want > 0
                    else:
                        self.assertIsNone(before[i])
        self.assertGreater(late, 0, "random order should draw some species after one of its Pokémon's fields")

    def test_species_first_order_draws_every_species_before_its_fields_and_removes_nothing(self):
        import diffusion84 as M
        for r in self.species_first_draws:
            d = r["draws"]; self.assertEqual(d["column"], list(M.ORDER))
            self.assertEqual([b for b in d["fields_before"] if b is not None], [0] * M.NSLOT)
            self.assertEqual(set(d["removed_share"]), {0.0}); self.assertFalse(any(d["top_removed"]))


if __name__ == "__main__":
    unittest.main()
