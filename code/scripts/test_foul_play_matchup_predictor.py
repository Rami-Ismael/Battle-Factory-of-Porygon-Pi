"""Run: /tmp/vgc-pilot/.venv/bin/python scripts/test_foul_play_matchup_predictor.py  (no battles, no laptop DB; CI runs it).

Tests the issue #419 matchup predictor (src/foul_play_matchup_predictor.py) at its public seams:
  1. make_folds / make_grouped_folds: deterministic, disjoint, complete, balanced folds; grouped folds keep lineups together.
  2. FeatureMap: slot-order invariant; built strictly from training teams (ignores test-only tokens and test stats).
  3. solve_ridge_with_loocv / fit_ridge_arm / RidgePredictor.predict: closed-form LOOCV matches brute-force leave-one-out
     refits; intercept and BC slope stay unpenalized; recovers exact linear multi-fidelity signal; team-only ignores BC.
  4. evaluate_arms / paired_bootstrap_ci: shared five folds across all arms; gap sign convention (arm - guess_bc_label < 0
     is better); verdict logic; deployment-noise 1-battle draw helper.
  5. End-to-end report + ECharts figure builder on synthetic results (no laptop DB needed).
"""
import random
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import foul_play_matchup_predictor as MP


def make_team(idx, rng, *, species_pool=None, sp_boost=0):
    """Build a synthetic 6-Pokemon parsed team dict matching corpus.parse_team_text's schema."""
    pool = species_pool or [f"Sp{i}" for i in range(24)]
    chosen = rng.sample(pool, 6)
    abilities = ["Intimidate", "Drought", "Prankster", "Unseen Fist", "Good as Gold", "Protosynthesis"]
    items = ["Focus Sash", "Choice Specs", "Leftovers", "Life Orb", "Sitrus Berry", "Assault Vest"]
    moves = ["Protect", "Fake Out", "Heat Wave", "Shadow Ball", "Tailwind", "Trick Room", "Moonblast", "Earth Power"]
    natures = ["Adamant", "Modest", "Timid", "Jolly", "Bold", "Calm"]
    slots = []
    for j, sp in enumerate(chosen):
        spe = min(32, (idx * 3 + j * 5) % 33 + sp_boost)
        hp = min(32, max(0, 66 - spe - 20))
        slots.append(
            {
                "species": sp,
                "ability": abilities[(idx + j) % len(abilities)],
                "item": items[(idx * 2 + j) % len(items)],
                "nature": natures[(idx + j * 2) % len(natures)],
                "moves": rng.sample(moves, 4),
                "evs": {"HP": hp, "Atk": 10, "Def": 5, "SpA": 5, "SpD": 0, "Spe": spe},
            }
        )
    return slots


def brute_force_loocv_mse(X_unpen, X_pen, y, lam_vec):
    """Independent brute-force leave-one-out MSE refit for every row (source of truth for closed-form LOOCV)."""
    n = len(y)
    X = np.hstack([X_unpen, X_pen])
    pen = np.concatenate([np.zeros(X_unpen.shape[1]), lam_vec])
    errs = []
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        Xtr, ytr = X[mask], y[mask]
        w = np.linalg.solve(Xtr.T @ Xtr + np.diag(pen), Xtr.T @ ytr)
        errs.append(float(y[i] - X[i] @ w))
    return float(np.mean(np.square(errs)))


class TestFolds(unittest.TestCase):
    def test_make_folds_is_disjoint_complete_balanced_and_order_independent(self):
        ids = list(range(100, 600))
        folds_a = MP.make_folds(ids, k=5, seed=419)
        folds_b = MP.make_folds(list(reversed(ids)), k=5, seed=419)
        self.assertEqual(folds_a, folds_b)
        self.assertEqual([len(f) for f in folds_a], [100] * 5)
        flat = [t for f in folds_a for t in f]
        self.assertEqual(sorted(flat), ids)
        self.assertEqual(len(set(flat)), 500)
        self.assertNotEqual(folds_a, MP.make_folds(ids, k=5, seed=420))

    def test_make_grouped_folds_keeps_identical_lineups_in_same_fold(self):
        ids = list(range(50))
        # Teams 0..9 share lineup "L0", teams 10..14 share "L1", rest unique
        keys = {i: ("L0" if i < 10 else "L1" if i < 15 else f"U{i}") for i in ids}
        folds = MP.make_grouped_folds(ids, keys, k=5, seed=419)
        self.assertEqual(sorted(t for f in folds for t in f), ids)
        fold_of = {t: fi for fi, f in enumerate(folds) for t in f}
        self.assertEqual(len({fold_of[i] for i in range(10)}), 1)
        self.assertEqual(len({fold_of[i] for i in range(10, 15)}), 1)


class TestFeatureMap(unittest.TestCase):
    def test_features_are_slot_order_invariant_and_ignore_test_only_tokens(self):
        rng = random.Random(7)
        train_teams = [make_team(i, rng) for i in range(30)]
        fmap = MP.FeatureMap.fit(train_teams, min_pair=3)

        # Permuting the 6 slots of a team must yield identical feature vector
        t0 = train_teams[0]
        v0 = fmap.transform([t0])[0]
        for _ in range(5):
            perm = list(t0)
            rng.shuffle(perm)
            vp = fmap.transform([perm])[0]
            np.testing.assert_allclose(v0, vp, atol=0.0, rtol=0.0)

        # A test team with completely unseen species/move/item must not raise and must have zero for unseen tokens
        unseen = [
            {
                "species": f"BrandNewMon{j}",
                "ability": "BrandNewAbility",
                "item": "BrandNewItem",
                "nature": "Quirky",
                "moves": ["Splash", "Celebrate", "Hold Hands", "Happy Hour"],
                "evs": {"HP": 11, "Atk": 11, "Def": 11, "SpA": 11, "SpD": 11, "Spe": 11},
            }
            for j in range(6)
        ]
        vu = fmap.transform([unseen])[0]
        self.assertEqual(vu.shape, (fmap.dim,))
        # Categorical single + pair indicators should all be 0 for unseen tokens
        self.assertEqual(float(np.abs(vu[: fmap.n_single + fmap.n_pair]).sum()), 0.0)


class TestRidgeLoocv(unittest.TestCase):
    def test_closed_form_loocv_matches_brute_force_with_unpenalized_columns(self):
        rng = np.random.default_rng(12)
        n, p_pen = 40, 25
        X_unpen = np.column_stack([np.ones(n), rng.uniform(0, 1, n)])
        X_pen = rng.normal(0, 1, (n, p_pen))
        y = 0.05 + 0.8 * X_unpen[:, 1] + X_pen[:, 0] * 0.04 + rng.normal(0, 0.02, n)
        lam_vec = np.concatenate([np.full(15, 10.0), np.full(10, 30.0)])

        expected = brute_force_loocv_mse(X_unpen, X_pen, y, lam_vec)
        got, w_unpen, w_pen = MP.solve_ridge_with_loocv(X_unpen, X_pen, y, lam_vec)
        self.assertAlmostEqual(got, expected, places=9)

        # Verify solution matches direct normal equation
        X_full = np.hstack([X_unpen, X_pen])
        pen_full = np.concatenate([np.zeros(2), lam_vec])
        w_direct = np.linalg.solve(X_full.T @ X_full + np.diag(pen_full), X_full.T @ y)
        np.testing.assert_allclose(np.concatenate([w_unpen, w_pen]), w_direct, atol=1e-9)

    def test_multi_fidelity_ridge_recovers_bc_shift_and_team_residual(self):
        rng = random.Random(99)
        np_rng = np.random.default_rng(99)
        teams = [make_team(i, rng, species_pool=[f"Mon{k}" for k in range(12)]) for i in range(120)]
        bc = np_rng.uniform(0.1, 0.7, len(teams))
        has_mon0 = np.array([any(s["species"] == "Mon0" for s in t) for t in teams], dtype=float)
        # True FP win rate = 0.02 + 0.85 * bc + 0.12 * I(Mon0) (shifted & scaled from bc, plus species signal)
        fp = np.clip(0.02 + 0.85 * bc + 0.12 * has_mon0 + np_rng.normal(0, 0.005, len(teams)), 0.0, 1.0)

        model = MP.fit_ridge_arm(teams[:90], fp[:90], bc[:90], use_bc=True, min_pair=3)
        pred = model.predict(teams[90:], bc[90:])
        rmse = float(np.sqrt(np.mean((pred - fp[90:]) ** 2)))
        self.assertLess(rmse, 0.02)

        # Team-only model must ignore bc even if bc is poisoned with NaNs/huge values
        model_only = MP.fit_ridge_arm(teams[:90], fp[:90], None, use_bc=False, min_pair=3)
        pred_only_a = model_only.predict(teams[90:], None)
        pred_only_b = model_only.predict(teams[90:], np.full(30, 999.0))
        np.testing.assert_allclose(pred_only_a, pred_only_b, atol=0.0)


class TestEvaluateArms(unittest.TestCase):
    def test_evaluate_arms_uses_shared_folds_and_computes_paired_gaps_and_verdict(self):
        rng = random.Random(419)
        np_rng = np.random.default_rng(419)
        n = 100
        team_ids = list(range(1000, 1000 + n))
        teams = {tid: make_team(i, rng, species_pool=[f"Mon{k}" for k in range(12)]) for i, tid in enumerate(team_ids)}
        bc = {tid: float(np_rng.uniform(0.15, 0.75)) for tid in team_ids}
        has_mon0 = {tid: float(any(s["species"] == "Mon0" for s in teams[tid])) for tid in team_ids}
        fp = {
            tid: float(np.clip(0.05 + 0.70 * bc[tid] + 0.15 * has_mon0[tid] + np_rng.normal(0, 0.01), 0.0, 1.0))
            for tid in team_ids
        }
        folds = MP.make_folds(team_ids, k=5, seed=419)
        res = MP.evaluate_arms(team_ids, teams, fp, bc, folds, min_pair=3, boot=400, seed=419)

        self.assertEqual(res["folds"], folds)
        for arm in ("predictor_team_bc", "predictor_team_only", "guess_bc_label", "calibrate_bc_only", "guess_train_mean"):
            self.assertIn(arm, res["arms"])
            self.assertEqual(len(res["arms"][arm]["oof_pred"]), n)
            self.assertEqual(len(res["arms"][arm]["per_fold_rmse"]), 5)

        # On this synthetic data, predictor_team_bc must clearly beat guess_bc_label and calibrate_bc_only
        gap_vs_guess = res["comparisons"]["predictor_team_bc_minus_guess_bc_label"]["rmse"]
        self.assertLess(gap_vs_guess["diff"], -0.05)
        self.assertLess(gap_vs_guess["ci95"][1], 0.0)
        self.assertEqual(gap_vs_guess["verdict"], "better")

        gap_vs_cal = res["comparisons"]["predictor_team_bc_minus_calibrate_bc_only"]["rmse"]
        self.assertLess(gap_vs_cal["ci95"][1], 0.0)

        # guess_bc_label predictions must equal the raw bc input exactly
        np.testing.assert_allclose(res["arms"]["guess_bc_label"]["oof_pred"], [bc[t] for t in team_ids])

    def test_draw_one_battle_bc_returns_valid_rates_deterministically(self):
        # 2 teams x 3 opponents, each cell (w, l, t) with 2 battles
        cells = {
            (1, 10): (2, 0, 0),
            (1, 11): (1, 1, 0),
            (1, 12): (0, 2, 0),
            (2, 10): (1, 1, 0),
            (2, 11): (0, 1, 1),
            (2, 12): (2, 0, 0),
        }
        a = MP.draw_one_battle_labels(cells, [1, 2], [10, 11, 12], seed=419)
        b = MP.draw_one_battle_labels(cells, [1, 2], [10, 11, 12], seed=419)
        self.assertEqual(a, b)
        for v in a.values():
            self.assertGreaterEqual(v, 0.0)
            self.assertLessEqual(v, 1.0)
        # Team 1 has deterministic outcomes against opps 10 (1.0) and 12 (0.0), so its 1-battle rate is 1/3 or 2/3
        self.assertIn(round(a[1], 4), (round(1 / 3, 4), round(2 / 3, 4)))


class TestFiguresAndReport(unittest.TestCase):
    def test_figures_and_markdown_render_from_synthetic_result(self):
        rng = random.Random(419)
        np_rng = np.random.default_rng(419)
        team_ids = list(range(50))
        teams = {tid: make_team(i, rng, species_pool=[f"Mon{k}" for k in range(10)]) for i, tid in enumerate(team_ids)}
        bc = {tid: float(np_rng.uniform(0.1, 0.7)) for tid in team_ids}
        fp = {tid: float(np.clip(0.8 * bc[tid] + np_rng.normal(0, 0.04), 0.0, 1.0)) for tid in team_ids}
        folds = MP.make_folds(team_ids, k=5, seed=419)
        primary = MP.evaluate_arms(team_ids, teams, fp, bc, folds, min_pair=3, boot=100, seed=419)
        payload = {
            "issue": 419,
            "n_teams": len(team_ids),
            "battle_noise_floor": {"fp_rmse_floor": 0.0423, "bc_rmse_floor": 0.0472},
            "primary": primary,
            "sensitivity": {
                "one_battle_bc": primary,
                "validation_opponents": primary,
                "grouped_lineup_folds": primary,
            },
            "exploratory_mlp": {
                "predictor_team_bc_mlp": {"rmse": 0.06, "rmse_ci95": [0.05, 0.07]},
                "predictor_team_only_mlp": {"rmse": 0.11, "rmse_ci95": [0.10, 0.12]},
            },
            "top_residual_features": {"positive": [("species:mon0", 0.02)], "negative": [("species:mon1", -0.02)]},
        }
        with tempfile.TemporaryDirectory() as td:
            pages = MP.write_storyboard_figures(payload, Path(td))
            self.assertEqual(len(pages), 4)
            for p in pages:
                self.assertTrue(p.exists())
                text = p.read_text()
                self.assertIn("echarts", text)
            md = MP.format_markdown_report(payload)
            self.assertIn("Team and behaviour-cloning win rate", md)
            self.assertIn("Team only, no behaviour-cloning win rate", md)
            self.assertIn("No training, behaviour-cloning win rate as the guess", md)


class TestUncertaintyAndCorpusStorage(unittest.TestCase):
    def test_predictive_std_matches_exact_normal_inverse_and_responds_to_bc_depth(self):
        rng = random.Random(31)
        np_rng = np.random.default_rng(31)
        teams = [make_team(i, rng, species_pool=[f"Mon{k}" for k in range(12)]) for i in range(80)]
        bc = np_rng.uniform(0.1, 0.7, len(teams))
        y = np.clip(0.03 + 0.7 * bc + np_rng.normal(0, 0.04, len(teams)), 0.0, 1.0)
        noise_var = 0.02**2
        model = MP.fit_ridge_with_uncertainty(teams[:60], y[:60], bc[:60], use_bc=True, min_pair=3, target_noise_var=noise_var)

        # Verify exact quadratic form x^T (X^T X + P)^-1 x against explicit inverse
        Xtr = np.hstack([MP._unpenalized_matrix(60, bc[:60], True), model.predictor.fmap.transform(teams[:60])])
        pen = np.concatenate([np.zeros(2), model.predictor.fmap.penalty_vector(model.predictor.lam_s, model.predictor.lam_p)])
        A_inv = np.linalg.inv(Xtr.T @ Xtr + np.diag(pen))
        Xte = np.hstack([MP._unpenalized_matrix(20, bc[60:], True), model.predictor.fmap.transform(teams[60:])])
        want_ep = model.sigma_resid * np.sqrt(np.einsum("ij,jk,ik->i", Xte, A_inv, Xte))

        pred_2b, sd_2b, ep_2b = model.predict_with_std(
            teams[60:], bc[60:], bc_battles=np.full(20, 78), bc_opponents=np.full(20, 39)
        )
        pred_1b, sd_1b, ep_1b = model.predict_with_std(
            teams[60:], bc[60:], bc_battles=np.full(20, 39), bc_opponents=np.full(20, 39)
        )
        np.testing.assert_allclose(pred_2b, model.predictor.predict(teams[60:], bc[60:]), atol=1e-12)
        np.testing.assert_allclose(pred_1b, pred_2b, atol=1e-12)
        np.testing.assert_allclose(ep_2b, want_ep, atol=1e-10)
        np.testing.assert_allclose(ep_1b, want_ep, atol=1e-10)
        self.assertTrue(np.all(sd_1b > sd_2b))
        self.assertTrue(np.all(sd_2b >= ep_2b))

        # Slot-order permutation invariance for pred, pred_std, epistemic_std
        shuf = []
        for t in teams[60:]:
            cp = list(t)
            rng.shuffle(cp)
            shuf.append(cp)
        p_s, sd_s, ep_s = model.predict_with_std(shuf, bc[60:], bc_battles=np.full(20, 39), bc_opponents=np.full(20, 39))
        np.testing.assert_allclose(p_s, pred_1b, atol=1e-12)
        np.testing.assert_allclose(sd_s, sd_1b, atol=1e-12)
        np.testing.assert_allclose(ep_s, ep_1b, atol=1e-12)

    def test_write_corpus_predictions_registers_new_predictor_and_guards_heldout(self):
        con = sqlite3.connect(":memory:")
        con.executescript(
            """
            CREATE TABLE team (team_id INTEGER PRIMARY KEY, legal INTEGER, origin TEXT, paste TEXT);
            CREATE TABLE policy (policy_id INTEGER PRIMARY KEY, row_policy TEXT);
            CREATE TABLE cell (policy_id INTEGER, lo INTEGER, hi INTEGER, lo_wins INTEGER, hi_wins INTEGER, ties INTEGER, battles INTEGER);
            INSERT INTO policy VALUES (8, 'bc_100.zip+slots-shuffled'), (10, 'rust_foulplay_det+slots-shuffled');
            INSERT INTO team VALUES (1, 1, 'vgcpastes', 'p1'), (2, 1, 'vgcpastes', 'p2'),
                                    (10, 1, 'gen', 'p10'), (11, 1, 'gen', 'p11'), (99, 1, 'held', 'p99');
            INSERT INTO cell VALUES (8, 1, 2, 4, 4, 0, 8),
                                    (8, 1, 10, 1, 1, 0, 2), (8, 2, 10, 0, 2, 0, 2),
                                    (8, 1, 11, 1, 0, 0, 1), (8, 2, 11, 0, 0, 1, 1),
                                    (8, 1, 99, 1, 0, 0, 1), (8, 2, 99, 1, 0, 0, 1);
            """
        )
        entries = MP.load_corpus_bc_stats(con, 8, [1, 2], excluded_ids={99})
        by_id = {e[0]: e for e in entries}
        self.assertEqual(set(by_id), {1, 2, 10, 11})
        self.assertNotIn(99, by_id)
        # Team 11 had (0 wins, 0 ties, 1 battle vs 1) + (0 wins, 1 tie, 1 battle vs 2) -> 0.5 / 2 = 0.25
        self.assertAlmostEqual(by_id[11][3], 0.25)
        self.assertEqual((by_id[1][4], by_id[10][4]), (1, 2))

        meta = {
            "predictor_name": "foul_play_det_ridge_team_bc_v1",
            "target_policy_id": 10,
            "input_policy_id": 8,
            "opponent_set": "train39_of_top50",
            "training_n": 500,
            "training_source": "issue_447_pilot_seed447",
            "oof_rmse": 0.0818,
            "deploy_1b_rmse": 0.0846,
            "sigma_resid": 0.0759,
            "sigma_true_resid": 0.0630,
            "fp_2b_noise_sd": 0.0423,
            "lam_single": 100.0,
            "lam_pair": 300.0,
            "intercept": 0.0206,
            "bc_slope": 0.5658,
            "n_features": 961,
            "git_commit": "abc123",
            "metadata": {"test": True},
        }
        rows = [(10, 0.55, 0.074, 0.032, 0.75, 2, 4, 1), (11, 0.22, 0.081, 0.041, 0.25, 2, 2, 0)]
        pid = MP.write_corpus_predictions(con, meta, rows, held_ids={99})
        self.assertEqual(pid, 1)
        # Battle tables are untouched; summary view marks kind='predicted_score'
        self.assertEqual(con.execute("SELECT COUNT(*) FROM cell").fetchone()[0], 7)
        summary = con.execute(
            "SELECT predictor_name, kind, team_id, pred_win_rate, pred_std, epistemic_std, in_pilot_train "
            "FROM team_predicted_score_summary ORDER BY team_id"
        ).fetchall()
        self.assertEqual(len(summary), 2)
        self.assertEqual(summary[0][:3], ("foul_play_det_ridge_team_bc_v1", "predicted_score", 10))
        self.assertAlmostEqual(summary[0][4], 0.074)
        self.assertEqual(summary[1][6], 0)

        # Duplicate predictor_name refuses silent overwrite unless overwrite=True
        with self.assertRaises(ValueError):
            MP.write_corpus_predictions(con, meta, rows, held_ids={99}, overwrite=False)
        pid2 = MP.write_corpus_predictions(con, meta, rows, held_ids={99}, overwrite=True)
        self.assertEqual(con.execute("SELECT COUNT(*) FROM team_predicted_score WHERE predictor_id=?", (pid2,)).fetchone()[0], 2)

        # Held-out team raises HeldoutLeak
        import heldout as HO

        with self.assertRaises(HO.HeldoutLeak):
            MP.write_corpus_predictions(con, {**meta, "predictor_name": "bad_v2"}, [(99, 0.3, 0.07, 0.03, 0.0, 2, 2, 0)], held_ids={99})


if __name__ == "__main__":
    unittest.main()
