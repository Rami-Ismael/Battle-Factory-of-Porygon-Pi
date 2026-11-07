"""Re-judge the remasking arms and their controls on the all-teams 84-field model by the ask-versus-got error (GitHub issue #93).

The remasking ticket (#90) and the least-likely trade ticket (#100) were judged by slope, which the owner replaced with the
ask-versus-got error on 2026-11-05. This reads the saved battles only: no sampling and no new battles. Each arm's error is the
mean over asks of (mean got - ask)^2, with and without ask 12/24, against no remasking with a bootstrap over whole teams within
each cell and opponents shared across arms, the resampling the slope ticket used.

    python remask84_error_analysis.py [output.json]
"""
import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path

import numpy as np

import ask_grid84  # noqa: F401  (sets up sys.path for the modules below)
import heldout as H
import matchup_db as MDB
from ask_error_compare import ask_error, paired_difference
from ask_line import NEAR_LINE_MSE
from team_diversity import slots_from_paste

RES = Path.home() / "vgc-data/results"
ASKS, NO12, SEEDS = [4, 8, 12, 16], [4, 8, 16], [1, 2, 3]
B = 4000
SOURCES = {                                              # arm -> (results file, cell key prefix)
    "plain": ("stat_points_84_gen84_all_ep21.json", "gen84_all_ep21"),
    **{a: ("remask84.json", a) for a in ("remask_ll_10", "remask_conf_loop", "remask_even_loop")},
    **{a: ("least_likely_trade.json", a) for a in ("remask_ll_05", "remask_ll_20", "cold_all_t08", "cold_all_t06", "cold_sets_t06", "cold_sets_t03", "spend66")},
    **{a: ("least_likely_trade_nucleus.json", a) for a in ("nucleus_sets_p09", "nucleus_sets_p07", "nucleus_sets_p05", "nucleus_all_p09")},
}
NAMES = {
    "plain": "No remasking", "remask_ll_10": "Least-likely 10% remasked", "remask_conf_loop": "Confidence-weighted, 0.3 of fields remasked",
    "remask_even_loop": "Even odds, 0.5 of fields remasked", "remask_ll_05": "Least-likely 5% remasked", "remask_ll_20": "Least-likely 20% remasked",
    "cold_all_t08": "All-field temperature 0.8", "cold_all_t06": "All-field temperature 0.6", "cold_sets_t06": "Set-field temperature 0.6",
    "cold_sets_t03": "Set-field temperature 0.3", "spend66": "Stat Points raised to 66", "nucleus_sets_p09": "Set-field nucleus 0.9",
    "nucleus_sets_p07": "Set-field nucleus 0.7", "nucleus_sets_p05": "Set-field nucleus 0.5", "nucleus_all_p09": "All-field nucleus 0.9",
}


def load_cells():
    files = {f: json.load(open(RES / f))["cells"] for f, _ in SOURCES.values()}
    return {(arm, k, s): files[f][f"{prefix}_a{k:02d}of24_s{s}"] for arm, (f, prefix) in SOURCES.items() for k in ASKS for s in SEEDS}


def battle_matrices(cells, policy_id=None):
    """Per cell, one row per team and one column per top-50 opponent: the saved battles' win share. Plus the contract check.
    policy_id: the battle policy every team was scored under. None looks it up per team, a scan of every cell row (about 100 minutes)."""
    con = sqlite3.connect(f"file:{MDB.DEFAULT_DB}?mode=ro", uri=True)
    held = H.team_ids(con)
    opps = sorted(c for (c,) in con.execute("SELECT team_id FROM team_set WHERE set_name='top50'") if c not in held)
    M, contract = {}, {}
    for key, c in cells.items():
        rows, policies = [], Counter()
        for p in c["pastes"]:
            h = MDB.canon_hash(MDB.canonical_paste(p))
            t = con.execute("SELECT team_id FROM team WHERE format_id=? AND canon_hash=?", (MDB.FORMAT, h)).fetchone()[0]
            policy = policy_id or con.execute("SELECT policy_id FROM cell WHERE lo=? OR hi=? GROUP BY policy_id ORDER BY COUNT(*) DESC LIMIT 1", (t, t)).fetchone()[0]
            policies[policy] += 1
            r = MDB.row_cells(con, policy, t, {o: 1 for o in opps})
            rows.append([r[o][0] / r[o][1] if o in r else np.nan for o in opps])
        M[key] = m = np.array(rows)
        a = contract.setdefault(key[0], dict(cells=0, teams=0, battles=0, missing=0, largest_gap_to_saved_score=0.0, policies=Counter()))
        a["cells"] += 1; a["teams"] += len(m); a["battles"] += int(np.isfinite(m).sum()); a["missing"] += int(np.isnan(m).sum())
        a["largest_gap_to_saved_score"] = max(a["largest_gap_to_saved_score"], float(np.abs(np.nanmean(m, 1) - np.array(c["y"])).max()))
        a["policies"].update(policies)
    return M, len(opps), {arm: {**v, "policies": dict(v["policies"])} for arm, v in contract.items()}


def bootstrap_levels(M, n_opps, arms=SOURCES, asks=ASKS, seeds=SEEDS):
    """arm -> ask -> B draws of the mean got over seeds: teams resampled within each cell, opponents shared across arms."""
    rng = np.random.default_rng(93)
    boot = {arm: {k: np.zeros(B) for k in asks} for arm in arms}
    for b in range(B):
        ow = rng.multinomial(n_opps, np.full(n_opps, 1 / n_opps)).astype(float)
        for arm in arms:
            for k in asks:
                level = 0.0
                for s in seeds:
                    m = M[(arm, k, s)]
                    tw = rng.multinomial(len(m), np.full(len(m), 1 / len(m)))
                    level += np.nansum(tw[:, None] * m * ow[None, :]) / (len(m) * ow.sum())
                boot[arm][k][b] = level / len(seeds)
    return boot


def judge(cells, boot):
    """Per arm: win per ask, the error with and without 12/24, and against no remasking its paired difference and seeds closer."""
    by_seed = {arm: {s: {k: float(np.mean(cells[(arm, k, s)]["y"])) for k in ASKS} for s in SEEDS} for arm in SOURCES}
    point = {arm: {k: float(np.mean([by_seed[arm][s][k] for s in SEEDS])) for k in ASKS} for arm in SOURCES}
    arms = {}
    for arm in SOURCES:
        lineups = {tuple(sorted(x["species"] for x in slots_from_paste(p))) for (a, _, _), c in cells.items() if a == arm for p in c["pastes"]}
        row = dict(name=NAMES[arm], win=point[arm], squared_gap_by_ask={k: (point[arm][k] - k / 24) ** 2 for k in ASKS}, line_ups=len(lineups))
        for label, asks in (("all_four", ASKS), ("without_12", NO12)):
            e = dict(error=ask_error(point[arm], asks), by_seed=[ask_error(by_seed[arm][s], asks) for s in SEEDS])
            e["near_line"] = e["error"] < NEAR_LINE_MSE
            if arm != "plain":
                e["difference"] = e["error"] - ask_error(point["plain"], asks)
                e.update(paired_difference(boot[arm], boot["plain"], asks))
                e["seeds_closer"] = sum(x < y for x, y in zip(e["by_seed"], [ask_error(by_seed["plain"][s], asks) for s in SEEDS]))
            row[label] = e
        arms[arm] = row
    return arms


def show(r, label):
    e = r[label]
    tail = f" {e['difference']:+.4f} [{e['interval'][0]:+.4f}, {e['interval'][1]:+.4f}] seeds {e['seeds_closer']}/3" if "interval" in e else ""
    return f"{e['error']:.4f}{tail}"


def main(path):
    cells = load_cells()
    M, n_opps, contract = battle_matrices(cells)
    arms = judge(cells, bootstrap_levels(M, n_opps))
    out = dict(opponents=n_opps, bootstrap_draws=B, contract=contract, arms=arms)
    json.dump(out, open(path, "w"), indent=1)
    for r in arms.values():
        print(f"{r['name']:<44} line-ups {r['line_ups']:>3} | all four {show(r, 'all_four')} | without 12/24 {show(r, 'without_12')}")
    print(json.dumps(contract))
    print("ERROR_ANALYSIS_DONE", path)

if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else RES / "remask84_error_analysis.json")
