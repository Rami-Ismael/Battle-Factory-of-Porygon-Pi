"""Run: python scripts/test_foul_play_stat_points.py   (no battles, no laptop data; CI runs it).

Tests the issue #448 Stat-Point sensitivity experiment (src/foul_play_stat_points.py) at its public seams:
base-team selection, Stat-Point arm transforms, opponent schedule isolation, paired battle/report statistics,
and ECharts storyboard generation. Each test fails on one concrete planted bug."""
import json
import random
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import echarts_chart as EC
import foul_play_stat_points as SP
import heldout as H
import matchup_db as M
import rust_matchup_runner as R

PROV = {k: k * 32 for k in ("engine", "dex", "spreads", "agent", "runtime")}
STATS = {"decisions": 8, "iterations": 240000, "fallbacks": 0, "failedSamples": 0}


def make_slots(prefix, ev_val=11, nature="Adamant"):
    return [
        {
            "species": f"{prefix}Mon{i}",
            "item": f"Item{i}",
            "ability": f"Ability{i}",
            "nature": nature,
            "evs": {s: ev_val for s in SP.STATS},
            "moves": [f"Move{i}A", f"Move{i}B", f"Move{i}C", f"Move{i}D"],
        }
        for i in range(6)
    ]


def make_spreads(prefixes):
    out = {"_source": "synthetic test spreads"}
    for prefix in prefixes:
        for i in range(6):
            key = M._norm(f"{prefix}Mon{i}")
            out[key] = [
                {"nature": "Jolly", "item": f"Item{i}", "evs": {"hp": 11, "atk": 11, "def": 11, "spa": 11, "spd": 11, "spe": 11}, "count": 5},
                {"nature": "Modest", "item": f"Item{i}", "evs": {"hp": 32, "atk": 0, "def": 2, "spa": 32, "spd": 0, "spe": 0}, "count": 3},
                {"nature": "Timid", "item": f"Item{i}", "evs": {"hp": 2, "atk": 0, "def": 0, "spa": 32, "spd": 0, "spe": 32}, "count": 2},
            ]
    return out


class StatPointsStrengthPlay:
    """Fake battle engine where a team's strength scales linearly with its total Stat Points by `sensitivity`."""

    def __init__(self, team_totals, sensitivity, base_bonus=None):
        self.team_totals = team_totals
        self.sensitivity = sensitivity
        self.base_bonus = base_bonus or {}

    def _p1_win_prob(self, p1, p2):
        # Difference in total Stat Points (out of 396) scaled by sensitivity, centered at 0.5
        s1 = self.base_bonus.get(p1, 0.0) + self.sensitivity * (self.team_totals[p1] / 396.0)
        s2 = self.base_bonus.get(p2, 0.0) + self.sensitivity * (self.team_totals[p2] / 396.0)
        return max(0.05, min(0.95, 0.5 + (s1 - s2)))

    def rust_play(self, games, spec, on_done):
        for g in games:
            p = self._p1_win_prob(g["p1_team"], g["p2_team"])
            # Deterministic quasi-exact outcome across game index g=0..3
            p1_wins = ((g["game"] + 0.5) / 4.0) < p
            on_done(g, {"winner": "p1" if p1_wins else "p2", "turns": 6, "ms": 100, "stats": {"p1": STATS, "p2": STATS}}, None)

    def bc_runner(self, con, jobs, seed=None):
        results = []
        for j in jobs:
            r = j["row"]
            counts = {}
            for c in j["schedule"]:
                counts[c] = counts.get(c, 0) + 1
            for c, n in counts.items():
                p = self._p1_win_prob(r, c)
                w = round(p * n)
                results.append((r, c, w, n - w, 0))
        return results, 0, 0.1


class StatPointsSensitivityTests(unittest.TestCase):
    def test_random_legal_spread_respects_budget_and_per_stat_cap(self):
        rng = random.Random(448)
        seen = set()
        for _ in range(200):
            evs = SP.random_legal_spread(rng)
            self.assertEqual(set(evs.keys()), set(SP.STATS))
            self.assertEqual(sum(evs.values()), SP.MAX_TOTAL)
            for v in evs.values():
                self.assertGreaterEqual(v, 0)
                self.assertLessEqual(v, SP.MAX_STAT)
            seen.add(tuple(evs[s] for s in SP.STATS))
        self.assertGreater(len(seen), 150)

    def test_apply_arm_preserves_48_non_stat_point_fields_and_alters_only_stat_points(self):
        base = make_slots("Alpha", ev_val=11, nature="Brave")
        spreads = make_spreads(["Alpha"])
        for arm in SP.ARMS:
            edited = SP.apply_arm(base, arm, spreads, random.Random(448))
            self.assertEqual(len(edited), 6)
            for orig, mod in zip(base, edited):
                # All 48 non-Stat-Point fields (including Stat Alignment!) stay identical
                for field in ("species", "item", "ability", "nature", "moves"):
                    self.assertEqual(orig[field], mod[field], f"{arm} altered {field}")
                vals = [mod["evs"][s] for s in SP.STATS]
                self.assertTrue(all(0 <= v <= SP.MAX_STAT for v in vals))
                self.assertLessEqual(sum(vals), SP.MAX_TOTAL)
            dist = SP.l1_stat_distance(base, edited)
            if arm == "stored":
                self.assertEqual(dist, 0)
            elif arm == "zeroed":
                self.assertEqual(dist, 396)
                self.assertEqual(sum(sum(s["evs"].values()) for s in edited), 0)
            else:
                self.assertGreater(dist, 0)
                self.assertEqual(sum(sum(s["evs"].values()) for s in edited), 396)

    def test_redraw_spread_prefers_different_spread_when_available(self):
        spreads = make_spreads(["Beta"])
        own = {s: 11 for s in SP.STATS}
        for seed in range(30):
            got = SP.redraw_spread(M._norm("BetaMon0"), own, spreads, random.Random(seed))
            self.assertNotEqual(got, own)
            self.assertEqual(sum(got.values()), 66)

    def test_eligible_and_draw_base_teams_exclude_missing_species_and_held_out(self):
        spreads = make_spreads(["M0", "M1", "M2"])  # M3 missing from spreads
        meta_entries = [
            {"id": "MB1", "team_id": 1, "slots": make_slots("M0"), "grid_hash": "g0"},
            {"id": "MB2", "team_id": 2, "slots": make_slots("M1"), "grid_hash": "g1"},
            {"id": "MB3", "team_id": 3, "slots": make_slots("M2"), "grid_hash": "g2"},
            {"id": "MB4", "team_id": 4, "slots": make_slots("M3"), "grid_hash": "g3"},
            {"id": "MB5", "team_id": 5, "slots": make_slots("M1"), "grid_hash": "g-held"},
        ]
        elig = SP.eligible_meta_entries(meta_entries, spreads, held_grids={"g-held"})
        self.assertEqual([e["team_id"] for e in elig], [1, 2, 3])
        d1 = SP.draw_base_teams(elig, n=2, seed=448)
        d2 = SP.draw_base_teams(list(reversed(elig)), n=2, seed=448)
        self.assertEqual([e["team_id"] for e in d1], [e["team_id"] for e in d2])
        self.assertEqual(len(d1), 2)

    def test_opponents_for_base_excludes_self_and_preserves_meta_weights(self):
        weights = {10: 1.0, 20: 2.0, 30: 1.0}
        opp = SP.opponents_for_base(10, weights)
        self.assertEqual(opp, {20: 2.0, 30: 1.0})
        opp2 = SP.opponents_for_base(20, weights)
        self.assertEqual(opp2, {10: 1.0, 30: 1.0})

    def test_weighted_win_rate_and_battle_noise_se(self):
        # Cell (1, 10): 4 wins of 4 (var=0), weight=1
        # Cell (1, 20): 2 wins of 4 (sample var=4*(0.25)/3 = 1/3, var of mean = 1/12), weight=2
        cells = {(1, 10): (4, 0, 0), (1, 20): (2, 2, 0)}
        wr, se = SP.weighted_win_rate(cells, 1, {10: 1.0, 20: 2.0})
        self.assertAlmostEqual(wr, (1.0 * 1.0 + 2.0 * 0.5) / 3.0)
        # Var(W) = (2^2 * (1/12)) / 3^2 = (4/12) / 9 = 1/27
        self.assertAlmostEqual(se, (1.0 / 27.0) ** 0.5)

    def test_end_to_end_battles_report_and_storyboard(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            con = M.connect(td / "battles.sqlite")
            bc_pid = M.get_policy(con, "bc_100.zip:test+slots-shuffled", "bc_100.zip:test+slots-shuffled", "showdown:test")
            prefixes = [f"T{i}" for i in range(6)]
            spreads = make_spreads(prefixes)
            meta_ids, meta_weights, team_totals = [], {}, {}
            spec_bases = []

            def insert_slots(slots, origin, grid_tag):
                paste = SP.slots_to_paste(slots)
                h = M.canon_hash(paste)
                row = con.execute("SELECT team_id FROM team WHERE format_id=? AND canon_hash=?", (M.FORMAT, h)).fetchone()
                if row:
                    tid = row[0]
                else:
                    skey = ",".join(sorted(M._norm(s["species"]) for s in slots))
                    with con:
                        tid = con.execute(
                            "INSERT INTO team (format_id, canon_hash, paste, species_key, base_key, legal, origin, grid_hash) "
                            "VALUES (?, ?, ?, ?, ?, 1, ?, ?)",
                            (M.FORMAT, h, paste, skey, skey, origin, grid_tag),
                        ).lastrowid
                team_totals[tid] = sum(sum(s["evs"].values()) for s in slots)
                return tid

            for i, pfx in enumerate(prefixes):
                slots = make_slots(pfx, ev_val=11)
                tid = insert_slots(slots, "vgcpastes", f"grid-{pfx}")
                meta_ids.append(tid)
                meta_weights[tid] = 2.0 if i == 5 else 1.0
                if i < 4:  # 4 base teams
                    arms = {}
                    for a_idx, arm in enumerate(SP.ARMS):
                        mod = SP.apply_arm(slots, arm, spreads, random.Random(448 + 100 * i + a_idx))
                        vid = insert_slots(mod, f"sp448:{arm}", f"grid-{pfx}")
                        arms[arm] = {"team_id": vid, "l1_distance": SP.l1_stat_distance(slots, mod),
                                     "total_sp": sum(sum(s["evs"].values()) for s in mod)}
                    spec_bases.append({"mb_id": f"MB{i}", "base_team_id": tid, "arms": arms})

            spec = {"issue": 448, "seed": 448, "per_cell": 4, "meta_weights": {str(k): v for k, v in meta_weights.items()},
                    "bases": spec_bases}

            fp_engine = StatPointsStrengthPlay(team_totals, sensitivity=0.40)
            bc_engine = StatPointsStrengthPlay(team_totals, sensitivity=0.04)
            rust_runner = R.RustMatchupRunner(play=fp_engine.rust_play, prov=PROV, evidence=td / "ev")
            fp_pid = rust_runner.register(con)

            with mock.patch.object(H, "identities", lambda: frozenset()):
                SP.battle_spec(con, spec, fp_pid, runner=rust_runner, label="fp", log=lambda m: None)
                SP.battle_spec(con, spec, bc_pid, runner=bc_engine.bc_runner, label="bc", log=lambda m: None)
                rep = SP.build_report(con, spec, fp_pid, bc_pid, boot=200)

            self.assertEqual(rep["held_out_cells"], 0)
            self.assertEqual(rep["incomplete_cells"]["foul_play"], 0)
            self.assertEqual(rep["incomplete_cells"]["behaviour_cloning"], 0)
            # Foul Play responds strongly to zeroed Stat Points; behaviour cloning responds weakly
            fp_zero = rep["arms"]["foul_play"]["zeroed"]["delta_vs_stored"]["mean"]
            bc_zero = rep["arms"]["behaviour_cloning"]["zeroed"]["delta_vs_stored"]["mean"]
            self.assertLess(fp_zero, -0.20)
            self.assertGreater(bc_zero, -0.10)
            self.assertLess(rep["labeler_gap"]["zeroed"]["mean"], -0.15)
            # Markdown contains reader-facing names and battle-noise SEs
            md = SP.report_markdown(rep)
            self.assertIn("| Stored |", md)
            self.assertIn("| Zeroed |", md)
            self.assertIn("Foul Play deterministic argmax", md)
            self.assertIn("battle-noise contrast SE", md)
            # bc_policy_id rejects non-bc or listed-order policies
            bad_pid = M.get_policy(con, "bc_100.zip:listed", "bc_100.zip:listed", "showdown:test")
            with mock.patch.object(M, "get_policy", return_value=bad_pid):
                with self.assertRaises(ValueError):
                    SP.bc_policy_id(con)
            # Storyboard writes all 4 ECharts pages
            captured = []
            prev = EC.CAPTURE
            try:
                EC.CAPTURE = captured
                SP.build_storyboard_figures(rep, td / "storyboard")
            finally:
                EC.CAPTURE = prev
            self.assertEqual([p.name for p, _, _, _ in captured],
                             ["0_teaser.html", "1_arm_ladder.html", "2_labeler_gap.html", "3_team_drops.html"])
            con.close()


if __name__ == "__main__":
    unittest.main()
