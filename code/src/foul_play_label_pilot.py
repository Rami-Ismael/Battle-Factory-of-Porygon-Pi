"""Foul Play deterministic argmax labels for 500 training teams, and the meta x meta baseline (issue #447, map #416).

Two battle sets under the Rust labeler's own policy_id (src/rust_matchup_runner.py, `rust_foulplay_det`, 30,000
iterations, 1 sample, both sides, Pokémon order shuffled every battle):

  meta    all 1,176 pairs of the 49 distinct meta teams, 8 battles each: the Matchup ICC beside behaviour
          cloning's on the same pairs (policy_id 8, 0.063 when #416 was charted).
  pilot   500 training teams, 2 battles against each meta team: the split-half reliability of the label, first
          battle per cell against second, beside the same measure under behaviour cloning.

The pilot teams are a fixed-seed draw from the generated training teams that already have a behaviour-cloning
label (policy_id 8 cells against all 49 meta teams). Held-out teams (ranks 51-100) and VGCPastes pool teams are
left out: the pool teams' own Stat Points are in the Foul Play spreads file, so their opponent would guess their
spreads better than a generated team's. A team's label is its Foul Play deterministic argmax win rate over the 39
training opponents (docs/opponent-split.json), as lossdata65.py computes behaviour-cloning labels; the 10
validation opponents are battled too and reported apart.

Behaviour cloning's split-half needs 2 battles per cell, and most training cells hold 1, so `bc-topup` battles the
500 teams' behaviour-cloning cells up to 2. Behaviour-cloning cells store only totals, so its two halves are 2
battles drawn per cell without replacement (fixed seed); the Rust halves are game 0 and game 1 of each cell,
read from the runner's evidence files.

    PY=/tmp/vgc-pilot/.venv/bin/python
    $PY src/foul_play_label_pilot.py sample          # writes pilot_teams.json once; later calls check it
    $PY src/foul_play_label_pilot.py battle-meta
    $PY src/foul_play_label_pilot.py bc-topup
    $PY src/foul_play_label_pilot.py battle-pilot
    $PY src/foul_play_label_pilot.py report meta     # report.json + post_meta.md (+ figure page)
    $PY src/foul_play_label_pilot.py report pilot
"""
import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

OUT = Path(os.environ.get("FP_PILOT_OUT", Path.home() / "vgc-data" / "foul_play_label_pilot_447"))
BC_POLICY = 8                   # behaviour cloning, both sides, slots shuffled: the training labels
BC_ICC_AT_CHARTING = 0.063      # map #416: policy 8 over the same 1,176 meta x meta cells
N_TEAMS, SEED = 500, 447
PER_CELL_META, PER_CELL_PILOT = 8, 2
BOOTSTRAP = 1000
LABELER = "Foul Play deterministic argmax"


# ---------------------------------------------------------------- the pilot teams
def eligible(con, meta, held, policy=BC_POLICY):
    """Generated training teams with behaviour-cloning cells against every meta team, sorted. Left out: the meta
    teams, held-out teams (ranks 51-100) and VGCPastes pool teams (their spreads are in the Foul Play spreads file)."""
    meta = sorted(meta)
    ph = ",".join("?" * len(meta))
    q = (f"SELECT t FROM (SELECT hi AS t, lo AS m FROM cell WHERE policy_id=? AND lo IN ({ph}) "
         f"UNION ALL SELECT lo, hi FROM cell WHERE policy_id=? AND hi IN ({ph})) "
         f"GROUP BY t HAVING COUNT(DISTINCT m) = ?")
    covered = {t for (t,) in con.execute(q, [policy, *meta, policy, *meta, len(meta)])}
    pool = {t for (t,) in con.execute("SELECT team_id FROM team WHERE origin = 'vgcpastes'")}
    return sorted(covered - set(meta) - set(held) - pool)


def draw(pool, n=N_TEAMS, seed=SEED):
    """n team ids drawn without replacement from the sorted pool with random.Random(seed), returned sorted."""
    return sorted(random.Random(seed).sample(sorted(pool), n))


def meta_split(con):
    """(training team ids [39], validation team ids [10]) in the frozen opponent split's order."""
    import lossdata65 as L65
    import opponent_split as OS
    split = OS.load()
    tids = L65.meta_team_ids(con, split)
    return [tids[i] for i in split["training"]], [tids[i] for i in split["validation"]]


def sample(con, out=OUT, n=N_TEAMS, seed=SEED):
    """Draw the pilot teams once and save them; a saved draw is checked and reused, never redrawn."""
    import heldout as H
    path = out / "pilot_teams.json"
    train, val = meta_split(con)
    if path.exists():
        saved = json.loads(path.read_text())
        if (saved["seed"], saved["n"]) != (seed, n): raise ValueError(f"{path} holds seed {saved['seed']}, n {saved['n']}")
        return saved
    held = H.team_ids(con)
    pool = eligible(con, train + val, held)
    teams = draw(pool, n, seed)
    saved = {"issue": 447, "seed": seed, "n": n, "population": len(pool), "behaviour_cloning_policy_id": BC_POLICY,
             "rule": "generated teams with policy 8 cells against all 49 meta teams; not a meta, held-out or VGCPastes "
                     "pool team; random.Random(seed).sample over the sorted ids",
             "drawn_at": time.strftime("%Y-%m-%d %H:%M:%S"), "training_opponents": train,
             "validation_opponents": val, "team_ids": teams}
    out.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(saved, indent=1) + "\n")
    return saved


# ---------------------------------------------------------------- battles
def rust_runner(workers=8):
    import rust_matchup_runner as R
    return R.RustMatchupRunner("rust_foulplay_det", samples=1, iterations=30000, workers=workers)


def check_bc_policy(con):
    row = con.execute("SELECT row_policy, col_policy FROM policy WHERE policy_id=?", (BC_POLICY,)).fetchone()
    if not row or not row[0].startswith("bc_100.zip:") or not row[0].endswith("+slots-shuffled") or row[0] != row[1]:
        raise ValueError(f"policy {BC_POLICY} is {row}, not behaviour cloning with shuffled slots")


def battle_meta(con, runner, meta, policy_id):
    import matchup_db as M
    return M.ensure(con, policy_id, meta, meta, PER_CELL_META, seed=SEED, runner=runner, note="issue 447 meta x meta")


def battle_pilot(con, runner, teams, meta, policy_id, chunk=50, log=print):
    import matchup_db as M
    ran = 0
    for i in range(0, len(teams), chunk):
        part = teams[i:i + chunk]
        t0 = time.perf_counter()
        n = M.ensure(con, policy_id, part, meta, PER_CELL_PILOT, seed=SEED, runner=runner, note="issue 447 pilot teams")
        ran += n
        log(f"pilot teams {i + len(part)}/{len(teams)}: {n} battles in {time.perf_counter() - t0:.0f} s")
    return ran


def smoke(con, meta, folder, workers=2):
    """A few real battles in a private database before the run: 3 meta teams, 2 battles per pair. Raises on any
    failed battle or fallback, so the queue stops before it writes to the shared database."""
    import matchup_db as M
    import rust_matchup_runner as R
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    priv = M.connect(folder / "smoke.sqlite")
    ids = [M.add_team(priv, con.execute("SELECT paste FROM team WHERE team_id=?", (t,)).fetchone()[0], "smoke")[0]
           for t in meta[:3]]
    runner = R.RustMatchupRunner("rust_foulplay_det", samples=1, iterations=30000, workers=workers,
                                 evidence=folder / "evidence")
    pid = runner.register(priv)
    n = M.ensure(priv, pid, ids, ids, 2, seed=SEED, runner=runner, note="issue 447 smoke")
    if n != 6 or runner.last["errors"] or runner.last["fallbacks"]:
        raise RuntimeError(f"smoke: {n} battles, {runner.last}")
    return runner.last


def bc_topup(con, teams, meta):
    import matchup_db as M
    check_bc_policy(con)
    return M.ensure(con, BC_POLICY, teams, meta, PER_CELL_PILOT, seed=SEED, note="issue 447 split-half top-up")


# ---------------------------------------------------------------- reading cells and games
def directed_cells(con, policy_id, rows, cols):
    """{(row, col): (wins, losses, ties)} from the row team's side; untested cells are absent."""
    out = {}
    for r in rows:
        for c in cols:
            if r == c: continue
            lo, hi = min(r, c), max(r, c)
            got = con.execute("SELECT lo_wins, hi_wins, ties FROM cell WHERE policy_id=? AND lo=? AND hi=?",
                              (policy_id, lo, hi)).fetchone()
            if got and sum(got): out[(r, c)] = (got[0], got[1], got[2]) if r == lo else (got[1], got[0], got[2])
    return out


def games_of(w, l, t):
    return [1.0] * w + [0.0] * l + [0.5] * t


def evidence_folders(con, policy_id):
    folders = []
    for (note,) in con.execute("SELECT note FROM batch WHERE policy_id=? ORDER BY batch_id", (policy_id,)):
        for word in (note or "").split():
            if word.startswith("evidence=") and word[9:] not in folders: folders.append(word[9:])
    return folders


def rust_games(folders):
    """{(lo, hi): {game index: lo's score}} from the runner's battles.jsonl files; failed games are skipped."""
    out = {}
    for f in folders:
        path = Path(f) / "battles.jsonl"
        if not path.exists(): continue
        for line in path.read_text().splitlines():
            g = json.loads(line)
            if "winner" not in g: continue
            winner = g["winner"]
            score = 0.5 if winner is None else float((g["p1_team"] if winner == "p1" else g["p2_team"]) == g["lo"])
            out.setdefault((g["lo"], g["hi"]), {})[g["game"]] = score
    return out


def first_two(games, row, col):
    """(first, second) score of `row` against `col`: the two lowest game indices with a result."""
    lo, hi = min(row, col), max(row, col)
    played = games.get((lo, hi), {})
    if len(played) < 2: return None
    a, b = (played[g] for g in sorted(played)[:2])
    return (a, b) if row == lo else (1 - a, 1 - b)


def drawn_two(cell, rng):
    """Two of a cell's battles drawn without replacement: the halves when only totals are stored."""
    pool = games_of(*cell)
    return tuple(rng.sample(pool, 2)) if len(pool) >= 2 else None


# ---------------------------------------------------------------- measures
def matchup_icc(cells):
    """Battle noise against true matchup spread over pairs: within-pair variance, true matchup SD, ICC, with a
    bootstrap 95% interval over pairs (foul_play_doubles_battles.noise_metrics, as #416 used)."""
    import foul_play_doubles_battles as F
    pairs = {k: games_of(*v) for k, v in cells.items()}
    return {"pairs": len(pairs), "battles": sum(len(v) for v in pairs.values()),
            "ties": sum(v[2] for v in cells.values()), **F.noise_metrics(pairs), "ci95": F._bootstrap_noise_ci(pairs)}


def pearson(xs, ys):
    n = len(xs)
    if n < 3: return None
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx, syy = sum((x - mx) ** 2 for x in xs), sum((y - my) ** 2 for y in ys)
    return sxy / math.sqrt(sxx * syy) if sxx > 0 and syy > 0 else None


def split_half(halves, seed=SEED, boot=BOOTSTRAP):
    """halves: {team: (first-half label, second-half label)}. Pearson r across teams with a bootstrap 95% interval
    over teams, and the Spearman-Brown reliability of the full two-battle label, 2r / (1 + r)."""
    teams = sorted(halves)
    xs, ys = [halves[t][0] for t in teams], [halves[t][1] for t in teams]
    r = pearson(xs, ys)
    rng, rs = random.Random(seed), []
    for _ in range(boot):
        pick = [rng.randrange(len(teams)) for _ in teams]
        b = pearson([xs[i] for i in pick], [ys[i] for i in pick])
        if b is not None: rs.append(b)
    rs.sort()
    ci = [rs[int(0.025 * len(rs))], rs[int(0.975 * len(rs)) - 1]] if rs else None
    return {"teams": len(teams), "r": r, "ci95": ci, "spearman_brown": None if r is None or r <= -1 else 2 * r / (1 + r)}


def team_halves(pairs_of_team, opponents):
    """{team: (mean first score, mean second score)} over `opponents`; a team missing a cell is left out."""
    out = {}
    for team, by_opp in pairs_of_team.items():
        got = [by_opp.get(o) for o in opponents if o != team]
        if not got or any(g is None for g in got): continue
        out[team] = (sum(g[0] for g in got) / len(got), sum(g[1] for g in got) / len(got))
    return out


def labels(cells, teams, opponents):
    """{team: win rate over `opponents`} pooled over battles, ties as half a win; teams missing a cell are left out."""
    out = {}
    for t in teams:
        got = [cells.get((t, o)) for o in opponents]
        if any(g is None for g in got): continue
        w = sum(g[0] + 0.5 * g[2] for g in got); n = sum(sum(g) for g in got)
        out[t] = w / n
    return out


def summary(values):
    v = list(values)
    m = sum(v) / len(v)
    return {"n": len(v), "mean": m, "sd": math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1)) if len(v) > 1 else None}


# ---------------------------------------------------------------- reports
def held_out_battled(con, policy_id):
    import heldout as H
    held = sorted(H.team_ids(con))
    if not held: return 0
    ph = ",".join("?" * len(held))
    return con.execute(f"SELECT COUNT(*) FROM cell WHERE policy_id=? AND (lo IN ({ph}) OR hi IN ({ph}))",
                       [policy_id, *held, *held]).fetchone()[0]


def report_meta(con, policy_id, meta):
    pairs = [(a, b) for i, a in enumerate(meta) for b in meta[i + 1:]]
    fp, bc = {}, {}
    for a, b in pairs:
        fp.update(directed_cells(con, policy_id, [a], [b]))
        bc.update(directed_cells(con, BC_POLICY, [a], [b]))
    short = sum(1 for a, b in pairs if sum(fp.get((a, b), (0, 0, 0))) < PER_CELL_META)
    return {"part": "meta", "policy_id": policy_id, "pairs_expected": len(pairs), "pairs_short_of_8": short,
            "held_out_cells": held_out_battled(con, policy_id),
            "foul_play": matchup_icc(fp), "behaviour_cloning": matchup_icc(bc),
            "behaviour_cloning_icc_at_charting": BC_ICC_AT_CHARTING,
            "pair_win_rates": {"foul_play": [round(sum(games_of(*v)) / sum(v), 4) for v in fp.values()],
                               "behaviour_cloning": [round(sum(games_of(*v)) / sum(v), 4) for v in bc.values()]}}


def report_pilot(con, policy_id, saved):
    teams, train, val = saved["team_ids"], saved["training_opponents"], saved["validation_opponents"]
    meta = train + val
    fp_cells = directed_cells(con, policy_id, teams, meta)
    bc_cells = directed_cells(con, BC_POLICY, teams, meta)
    games = rust_games(evidence_folders(con, policy_id))
    rng = random.Random(SEED)
    fp_pairs = {t: {o: first_two(games, t, o) for o in meta} for t in teams}
    bc_pairs = {t: {o: drawn_two(bc_cells[(t, o)], rng) if (t, o) in bc_cells else None for o in meta} for t in teams}
    complete = [t for t in teams if all(sum(fp_cells.get((t, o), (0, 0, 0))) >= PER_CELL_PILOT for o in meta)]
    fp_label, bc_label = labels(fp_cells, teams, train), labels(bc_cells, teams, train)
    both = sorted(set(fp_label) & set(bc_label))
    evidence_vs_db = sum(1 for t in teams for o in meta
                         if (t, o) in fp_cells and len(games.get((min(t, o), max(t, o)), {})) != sum(fp_cells[(t, o)]))
    return {"part": "pilot", "policy_id": policy_id, "teams": len(teams), "teams_with_2_in_all_49": len(complete),
            "held_out_cells": held_out_battled(con, policy_id), "cells_where_evidence_differs_from_db": evidence_vs_db,
            "split_half": {"foul_play": split_half(team_halves(fp_pairs, train)),
                           "behaviour_cloning": split_half(team_halves(bc_pairs, train)),
                           "foul_play_all_49": split_half(team_halves(fp_pairs, meta)),
                           "behaviour_cloning_all_49": split_half(team_halves(bc_pairs, meta))},
            "label": {"foul_play": summary(fp_label.values()), "behaviour_cloning": summary(bc_label.values()),
                      "r_between_labelers": pearson([fp_label[t] for t in both], [bc_label[t] for t in both]),
                      "foul_play_validation_opponents": summary(labels(fp_cells, teams, val).values())},
            "halves": {"foul_play": team_halves(fp_pairs, train), "behaviour_cloning": team_halves(bc_pairs, train)}}


def fmt(x, d=3):
    return "n/a" if x is None else f"{x:.{d}f}"


def ci(c, d=3):
    return "" if not c else f" [{c[0]:.{d}f}, {c[1]:.{d}f}]"


def meta_markdown(r, figure=None):
    fp, bc = r["foul_play"], r["behaviour_cloning"]
    lines = [f"### Meta × meta baseline: Matchup ICC under {LABELER}", ""]
    if figure: lines += [f"![Pair win rates on the 1,176 meta × meta pairs]({figure})", ""]
    lines += ["Matchup ICC is the share of a single battle's outcome variance that belongs to the pair of teams "
              "rather than to battle noise (GLOSSARY.md). Same 1,176 pairs of the 49 meta teams, 8 battles per pair.", "",
              f"| | Behaviour cloning (policy {BC_POLICY}) | {LABELER} (policy {r['policy_id']}) |",
              "|---|---|---|",
              f"| Pairs / battles | {bc['pairs']:,} / {bc['battles']:,} | {fp['pairs']:,} / {fp['battles']:,} |",
              f"| Matchup ICC [95% CI] | {fmt(bc.get('icc'))}{ci(bc['ci95'].get('icc'))} | "
              f"**{fmt(fp.get('icc'))}**{ci(fp['ci95'].get('icc'))} |",
              f"| 1-game SD | {fmt(bc.get('single_game_sd'))} | {fmt(fp.get('single_game_sd'))} |",
              f"| True matchup SD | {fmt(bc.get('true_matchup_sd'))} | {fmt(fp.get('true_matchup_sd'))} |",
              f"| 49-game label SE | {fmt(bc.get('label_se_49'), 4)} | {fmt(fp.get('label_se_49'), 4)} |",
              f"| Ties | {bc['ties']:,} | {fp['ties']:,} |", "",
              f"Behaviour cloning's ICC was {BC_ICC_AT_CHARTING} when map #416 was charted; the column above recomputes "
              f"it from the same cells today. Pairs short of 8 battles: {r['pairs_short_of_8']}. "
              f"Cells with a held-out team (ranks 51–100) under policy {r['policy_id']}: {r['held_out_cells']}."]
    return "\n".join(lines) + "\n"


def pilot_markdown(r, figure=None):
    s, lab = r["split_half"], r["label"]
    lines = [f"### 500 training teams: split-half reliability of the {LABELER} win rate", ""]
    if figure: lines += [f"![First-battle label against second-battle label, 500 teams]({figure})", ""]
    lines += ["Each team's label is its win rate over the 39 training opponents. Split-half: the label from the "
              "first battle per cell against the label from the second, correlated across teams; Spearman–Brown "
              "gives the reliability of the full two-battle label.", "",
              f"| | Behaviour cloning (policy {BC_POLICY}) | {LABELER} (policy {r['policy_id']}) |",
              "|---|---|---|",
              f"| Teams | {s['behaviour_cloning']['teams']} | {s['foul_play']['teams']} |",
              f"| Split-half r [95% CI] | {fmt(s['behaviour_cloning']['r'])}{ci(s['behaviour_cloning']['ci95'])} | "
              f"**{fmt(s['foul_play']['r'])}**{ci(s['foul_play']['ci95'])} |",
              f"| Two-battle label reliability (Spearman–Brown) | {fmt(s['behaviour_cloning']['spearman_brown'])} | "
              f"{fmt(s['foul_play']['spearman_brown'])} |",
              f"| Split-half r over all 49 opponents | {fmt(s['behaviour_cloning_all_49']['r'])} | "
              f"{fmt(s['foul_play_all_49']['r'])} |",
              f"| Label mean (SD) | {fmt(lab['behaviour_cloning']['mean'])} ({fmt(lab['behaviour_cloning']['sd'])}) | "
              f"{fmt(lab['foul_play']['mean'])} ({fmt(lab['foul_play']['sd'])}) |", "",
              f"Correlation between the two labelers' win rates on the same teams: {fmt(lab['r_between_labelers'])}. "
              f"{LABELER} win rate against the 10 validation opponents (kept out of training labels): mean "
              f"{fmt(lab['foul_play_validation_opponents']['mean'])}.", "",
              f"Behaviour-cloning cells store totals only, so its halves are 2 battles drawn per cell without "
              f"replacement (seed {SEED}); {LABELER}'s halves are game 0 and game 1 of each cell. Teams with 2 battles "
              f"in all 49 cells: {r['teams_with_2_in_all_49']}/{r['teams']}. Cells with a held-out team: "
              f"{r['held_out_cells']}. Cells whose evidence count differs from the database: "
              f"{r['cells_where_evidence_differs_from_db']}."]
    return "\n".join(lines) + "\n"


def meta_figure(r, path):
    from echarts_chart import write_chart
    bins = [i / 8 for i in range(9)]

    def hist(vals):                     # a tie makes a half step; it rounds to the nearest eighth
        return [sum(1 for v in vals if math.floor(v * 8 + 0.5) == k) for k in range(9)]
    fp, bc = r["pair_win_rates"]["foul_play"], r["pair_win_rates"]["behaviour_cloning"]
    option = {"tooltip": {"trigger": "axis"}, "legend": {"top": 0},
              "xAxis": {"type": "category", "name": "win rate of the lower-id team over 8 battles",
                        "nameLocation": "middle", "nameGap": 30, "data": [f"{b:.3f}" for b in bins]},
              "yAxis": {"type": "value", "name": "pairs"},
              "series": [{"name": f"Behaviour cloning (ICC {fmt(r['behaviour_cloning'].get('icc'))})", "type": "bar",
                          "data": hist(bc), "emphasis": {"focus": "series"}, "itemStyle": {"color": "#8a8984"}},
                         {"name": f"{LABELER} (ICC {fmt(r['foul_play'].get('icc'))})", "type": "bar",
                          "data": hist(fp), "emphasis": {"focus": "series"}, "itemStyle": {"color": "#2a6fdb"}}]}
    return write_chart(option, path, title="How far pair win rates spread from 0.5",
                       subtitle="1,176 meta × meta pairs, 8 battles each. A wider spread at the ends means more of "
                                "the outcome belongs to the pair, less to battle noise.")


def pilot_figure(r, path):
    from echarts_chart import write_chart
    h = r["halves"]
    s = r["split_half"]
    option = {"tooltip": {"trigger": "item", "formatter": "js:p => p.seriesName + ': ' + p.value[0].toFixed(3) + ' vs ' + p.value[1].toFixed(3)"},
              "legend": {"top": 0},
              "xAxis": {"type": "value", "name": "label from the first battle per cell", "nameLocation": "middle",
                        "nameGap": 30, "min": 0, "max": 1},
              "yAxis": {"type": "value", "name": "label from the second battle", "min": 0, "max": 1},
              "series": [{"name": f"Behaviour cloning (r {fmt(s['behaviour_cloning']['r'])})", "type": "scatter",
                          "symbolSize": 5, "data": [list(v) for v in h["behaviour_cloning"].values()],
                          "emphasis": {"focus": "series"}, "itemStyle": {"color": "#8a8984", "opacity": 0.6}},
                         {"name": f"{LABELER} (r {fmt(s['foul_play']['r'])})", "type": "scatter", "symbolSize": 5,
                          "data": [list(v) for v in h["foul_play"].values()], "emphasis": {"focus": "series"},
                          "itemStyle": {"color": "#2a6fdb", "opacity": 0.6}},
                         {"name": "same label twice", "type": "line", "data": [[0, 0], [1, 1]], "symbol": "none",
                          "lineStyle": {"type": "dashed", "color": "#6f6e69"}, "silent": True}]}
    return write_chart(option, path, title="Does a team get the same label twice?",
                       subtitle="500 training teams, win rate over the 39 training opponents from one battle per cell. "
                                "Points near the diagonal mean a repeatable label.", height=560)


# ---------------------------------------------------------------- command line
def main(argv=None):
    import matchup_db as M
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--db", default=str(M.DEFAULT_DB))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--workers", type=int, default=8)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("sample", "smoke", "battle-meta", "bc-topup", "battle-pilot", "policy"): sub.add_parser(name)
    rp = sub.add_parser("report"); rp.add_argument("part", choices=("meta", "pilot"))
    a = ap.parse_args(argv)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    con = M.connect(a.db)
    saved = sample(con, out)
    meta = saved["training_opponents"] + saved["validation_opponents"]
    say = lambda m: print(f"{time.strftime('%m-%d %H:%M')} {m}", flush=True)
    if a.cmd == "sample":
        say(f"{saved['n']} pilot teams from {saved['population']:,}, seed {saved['seed']}: {out / 'pilot_teams.json'}")
        return
    if a.cmd == "smoke":
        say(f"smoke passed: {json.dumps(smoke(con, sorted(meta), out / 'smoke'))}")
        return
    runner = rust_runner(a.workers)
    pid = runner.register(con)
    pinned = out / "policy_id"
    if pinned.exists() and int(pinned.read_text()) != pid:
        raise SystemExit(f"the Rust inputs changed: policy {pid} now, {pinned.read_text().strip()} when this run began")
    pinned.write_text(f"{pid}\n")
    if a.cmd == "policy":
        say(f"{LABELER} policy_id {pid}")
    elif a.cmd == "battle-meta":
        say(f"meta x meta: {battle_meta(con, runner, sorted(meta), pid)} battles run")
    elif a.cmd == "bc-topup":
        say(f"behaviour-cloning top-up: {bc_topup(con, saved['team_ids'], meta)} battles run")
    elif a.cmd == "battle-pilot":
        say(f"pilot teams: {battle_pilot(con, runner, saved['team_ids'], meta, pid, log=say)} battles run")
    elif a.cmd == "report":
        r = report_meta(con, pid, sorted(meta)) if a.part == "meta" else report_pilot(con, pid, saved)
        (out / f"report_{a.part}.json").write_text(json.dumps(r, indent=1) + "\n")
        page = out / "figure" / f"{a.part}.html"
        (meta_figure if a.part == "meta" else pilot_figure)(r, page)
        url = os.environ.get("FP_FIGURE_URL")
        md = (meta_markdown if a.part == "meta" else pilot_markdown)(r, url)
        (out / f"post_{a.part}.md").write_text(md)
        print(md)


if __name__ == "__main__":
    main()
