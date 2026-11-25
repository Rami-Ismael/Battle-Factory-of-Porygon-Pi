"""Dry run of every retrain_default_report.py step, on synthetic grids, in a few seconds.

The #385 queue stopped after eight hours of training and battles because the guidance 2 figure step crashed on a
duplicate keyword. This dry run writes small fake grids for both checkpoints at guidance 2, 4, and 8 (69 cells of
8 teams each, with opponent tables, Pass@k reads, and elite line-up yields), then runs the training figure and
comment, each guidance's comparison, figure, and comment, the summary, the final figures, and the final and
handoff comments. The queue runs it before training, so a crash stops the queue at minute one.

    /tmp/vgc-pilot/.venv/bin/python src/retrain_default_dry_run.py [OUT_DIR]   (from code/; prints DRY_RUN_PASS)
"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import retrain_default_report as RDR

SPECIES = ["Arbok", "Weavile", "Garchomp", "Charizard", "Incineroar", "Rillaboom", "Amoonguss", "Kingambit", "Gholdengo", "Dragonite"]
TEAMS_PER_CELL = 8
OPPONENTS = 49


def fake_paste(rng):
    """A Showdown paste of six distinct species; only the species line matters to the line-up measures."""
    six = rng.choice(len(SPECIES), 6, replace=False)
    return "\n\n".join(f"{SPECIES[i]} @ Item\nAbility: X\n- Move" for i in six)


def fake_grid(rng, name):
    """One grid's results file and its opponent table, shaped like ask_grid84 and masking_rate_rows write them."""
    cells, opponents = {}, {"weights": np.full(OPPONENTS, 1 / OPPONENTS)}
    for k in RDR.ALL_ASKS:
        for s in RDR.SEEDS:
            rate = rng.uniform(0, 1, (TEAMS_PER_CELL, OPPONENTS))
            rate[rng.uniform(size=rate.shape) < 0.05] = np.nan            # a few missing battles
            opponents[f"a{k:02d}_s{s}"] = rate
            cells[f"{name}_a{k:02d}_s{s}"] = dict(
                numerator=k, seed=s, tried=TEAMS_PER_CELL + 1, rejected=1, battles=OPPONENTS * TEAMS_PER_CELL,
                pastes=[fake_paste(rng) for _ in range(TEAMS_PER_CELL)],
                y=list(rng.uniform(0, 1, TEAMS_PER_CELL)),
                div=dict(effective_species=float(rng.uniform(10, 20))),
            )
    grid = dict(keep=TEAMS_PER_CELL, seed0=123, checkpoint="ck", checkpoint_sha256="abc", cells=cells)
    return grid, opponents


def write_fixture(results_dir, rng):
    """Every file the report reads after the battles, for both checkpoints at every guidance."""
    results_dir = Path(results_dir)
    (results_dir / "pass_at_k").mkdir(parents=True, exist_ok=True)
    bases = RDR._load_redraw_bases()
    for g in RDR.GUIDANCES:
        for row in (RDR.NEW_ROW, RDR.REF_ROW):
            name = RDR.grid_name(row, g)
            grid, opponents = fake_grid(rng, name)
            (results_dir / f"stat_points_84_{name}.json").write_text(json.dumps(grid))
            np.savez(results_dir / f"masking_rate_rows_opponents_{name}.npz", **opponents)
            if row == RDR.REF_ROW:
                (results_dir / f"stat_points_84_{name}_seed{bases[g]}.json").write_text(json.dumps(grid))

            pak = sorted(rng.uniform(0, 1, 48))
            held = dict(pass_at_k=pak, pass_at_k_interval=[[v - 0.05, v + 0.05] for v in pak], potential_at_k=pak)
            (results_dir / "pass_at_k" / f"{name}.json").write_text(json.dumps(dict(definition="d", bands=dict(held=dict(asks_13_to_17=held)))))
            view = dict(elite_line_ups=40, rebattled=30, held=20, held_new_against_training=5, yield_per_1000=9.5, one_score_mean=0.6, fresh_mean=None)
            (results_dir / f"line_up_yield_{name}.json").write_text(json.dumps(dict(views=dict(all_23_asks=view))))


def fake_run_info(best_epoch=78, stopped_epoch=80):
    curves = [dict(epoch=e, train_loss=2.0 - 0.008 * e, val_loss=2.1 - 0.009 * e) for e in range(1, stopped_epoch + 1)]
    return dict(best_epoch=best_epoch, stopped_epoch=stopped_epoch, cap=80, patience=3, best_val_loss=curves[best_epoch - 1]["val_loss"],
                minutes=420.7, seed=0, data_sha256=RDR.EXPECTED_TABLE_SHA, dropped_by_tables=0, curves=curves)


def run(out_dir):
    """Run every report step on a fresh fixture under out_dir; return the files it wrote."""
    import line_up_yield as LY

    out_dir = Path(out_dir)
    results_dir, figures = out_dir / "results", out_dir / "figures"
    rng = np.random.default_rng(0)
    write_fixture(results_dir, rng)
    train_lineups = {LY.line_up(fake_paste(np.random.default_rng(i))) for i in range(30)}
    written = []

    run_info, ref_info = fake_run_info(), fake_run_info(best_epoch=80)
    written.append(RDR.build_train_figure(run_info, figures, ref_info))
    for ref in (ref_info, None):
        assert RDR.NEW_CK in RDR.format_train_comment(run_info, "deadbeef1234", "L", "D", ref)

    for g in RDR.GUIDANCES:
        RDR.check_grid(RDR.grid_name(RDR.NEW_ROW, g), len(RDR.ALL_ASKS) * len(RDR.SEEDS), results_dir=results_dir)
        pair = RDR.read_guidance_pair(g, train_lineups=train_lineups, results_dir=results_dir)
        written.append(RDR.build_grid_figure(pair, figures))
        assert f"Guidance {g} grid landed" in RDR.format_grid_comment(pair, "L", "D")

    summary = RDR.build_summary(out_dir / "summary.json", results_dir=results_dir, train_lineups=train_lineups)
    summary = json.loads((out_dir / "summary.json").read_text())       # the queue reads the summary back from disk
    written += RDR.build_final_figures(summary, figures)
    assert "Full trade-off table" in RDR.format_final_comment(summary, run_info, "deadbeef1234", "a", "b", "c", "d")
    assert RDR.NEW_CK in RDR.format_handoff_comment(summary, run_info)
    return written


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    if args:
        written = run(args[0])
    else:
        with tempfile.TemporaryDirectory() as tmp:
            written = run(tmp)
    print(f"DRY_RUN_PASS {len(written)} figures")


if __name__ == "__main__":
    main()
