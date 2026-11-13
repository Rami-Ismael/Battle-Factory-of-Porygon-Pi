"""Judge the masking-rate rows and write each grid's post (docs/masking-rate-rows.md, GitHub issue #297).

Three rows are battled, each at guidance 2, 4 and 8: nine grids of ask_grid84.py. This script reads their results files.

    python masking_rate_rows.py legality_probe            random order with the lookahead off: sampling only, no battles
    python masking_rate_rows.py check NAME CELLS          exit non-zero unless results file NAME has CELLS cells, all battled
    python masking_rate_rows.py opponents ROW GUIDANCE    save each team's win rate against each opponent beside the grid
    python masking_rate_rows.py summary                   every measure and verdict for the grids that have finished
    python masking_rate_rows.py post ROW GUIDANCE DIR     that grid's figure, light and dark, and its comment text, into DIR

ROW is a code ID in ROWS. MRR_SMOKE=1 reads the smoke files of ask_grid84.py and draws 200 bootstrap samples.

The verdict of a row at one guidance value compares its ask-versus-got error with the control's at the same guidance.
The interval of the difference is a bootstrap that resamples whole teams within each ask and sampling seed, separately
in each row, and the 49 opponents once for both rows, because battles against one opponent are correlated.
"""
import html
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RES = Path.home() / "vgc-data/results"
SMOKE = bool(os.environ.get("MRR_SMOKE"))
ROWS = {"sched_species_first": "Per-field, uncorrected t", "sched_linear_species_first": "Linear, species-first",
        "sched_linear_random": "Linear, random order"}
ROW_NOTES = {"sched_species_first": "the control: our own settings", "sched_linear_species_first": "the published masking rate, our decoding order",
             "sched_linear_random": "the published masking rate and decoding order"}
CONTROL, CORRECTED_T = "sched_species_first", "Per-field, corrected t"
GUIDANCES = (2, 4, 8)
BAR, TRAINING_LUCK, WIN_RATE_LUCK = 0.001, 0.002, 0.026   # the plan's verdict rule, docs/masking-rate-rows.md
SCORED_ASKS, TOP_ASKS = range(1, 18), (15, 16, 17)
ALL_ASKS = [12] if SMOKE else list(range(1, 24))
SEEDS = [1] if SMOKE else [1, 2, 3]
DRAWS, DRAW_SEED = 200 if SMOKE else 4000, 2971
TABLE = "lossdata84_shuffled_only.npz"
PER_FIELD_CHECKPOINT = "stat84_shuffled_noise_per_field_es3_cap80"
PROBE_TRIES, PROBE_ASKS = 384, (4, 8, 12, 16)
SUMMARY = RES / f"masking_rate_rows_summary{'_smoke' if SMOKE else ''}.json"
LEGALITY = RES / f"masking_rate_rows_lookahead_off{'_smoke' if SMOKE else ''}.json"
ISSUE_URL = "https://github.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/issues/297"


def grid_name(row, guidance):
    return row + ("" if float(guidance) == 2 else f"_g{float(guidance):g}")


def grid_path(row, guidance):
    return RES / f"stat_points_84_{grid_name(row, guidance)}{'_smoke' if SMOKE else ''}.json"


def opponents_path(row, guidance):
    return RES / f"masking_rate_rows_opponents_{grid_name(row, guidance)}{'_smoke' if SMOKE else ''}.npz"


def load_grid(row, guidance):
    """The grid's results, or None until every cell has drawn its teams and battled them."""
    path = grid_path(row, guidance)
    if not path.exists(): return None
    out = json.load(open(path))
    cells = {(c["numerator"], c["seed"]): c for c in out["cells"].values()}
    if any((k, s) not in cells or "y" not in cells[(k, s)] for k in ALL_ASKS for s in SEEDS): return None
    out["by_ask_seed"] = cells
    return out


def error(got_by_ask, asks=SCORED_ASKS):
    """Mean squared gap between each ask k/24 and the mean got at that ask. None when none of the asks was run."""
    gaps = [(got_by_ask[k] - k / 24) ** 2 for k in asks if k in got_by_ask]
    return float(np.mean(gaps)) if gaps else None


# ---------------------------------------------------------------- per-opponent win rates, saved beside each grid
def opponents(row, guidance):
    """Each team's win rate against each of the 49 opponents, read from the battle database the moment the grid ends.
    The bootstrap resamples opponents from these, and the weighted mean of a row must equal the score the grid stored."""
    sys.path.insert(0, "/tmp/vgc-pilot/src")
    import matchup_db as MDB
    out = load_grid(row, guidance)
    if out is None: raise SystemExit(f"{grid_name(row, guidance)}: the grid has cells without battles")
    con = MDB.connect(); policy = MDB.get_policy(con); held = MDB.H.team_ids(con)
    members = dict(con.execute("SELECT team_id, weight FROM team_set WHERE set_name='top50'").fetchall())
    cols = sorted(c for c in members if c not in held)
    arrays, worst = dict(columns=np.array(cols), weights=np.array([members[c] for c in cols], float)), 0.0
    for (k, seed), cell in sorted(out["by_ask_seed"].items()):
        rate = np.full((len(cell["team_ids"]), len(cols)), np.nan)
        for i, team in enumerate(cell["team_ids"]):
            for c, (wins, battles) in MDB.row_cells(con, policy, team, cols).items(): rate[i, cols.index(c)] = wins / battles
        arrays[f"a{k:02d}_s{seed}"] = rate
        worst = max(worst, float(np.abs(team_scores(rate, arrays["weights"]) - np.array(cell["y"], float)).max()))
    print(f"{grid_name(row, guidance)}: {len(out['by_ask_seed'])} cells, {len(cols)} opponents, largest gap from the stored scores {worst:.2e}")
    if worst > 1e-9: raise SystemExit("a team's per-opponent win rates no longer give the score its grid stored; nothing saved")
    np.savez_compressed(opponents_path(row, guidance), **arrays)


def check(name, cells_wanted):
    """Exit non-zero unless the results file `name` holds every cell, each with teams and one finished battle per opponent."""
    path = RES / f"stat_points_84_{name}.json"
    cells = list(json.load(open(path))["cells"].values()) if path.exists() else []
    short = [f"a{c['numerator']:02d}_s{c['seed']}" for c in cells if not c["pastes"] or len(c.get("y", [])) != len(c["pastes"]) or None in c["y"] or c.get("battles", 0) < 49 * len(c["pastes"])]
    print(f"{name}: {len(cells)} of {cells_wanted} cells, {len(short)} without teams or with battles missing {short[:5]}")
    if len(cells) != cells_wanted or short: raise SystemExit(1)


def team_scores(rate, weights):
    """[teams]: each team's weighted mean win rate over the opponents it has battled. rate is [teams, opponents]."""
    seen = ~np.isnan(rate)
    return (np.where(seen, rate, 0.0) @ weights) / (seen @ weights)


def opponent_draws(n_opponents):
    """[DRAWS, opponents]: how often each opponent appears in each bootstrap sample. Shared by every grid."""
    return np.random.default_rng(DRAW_SEED).multinomial(n_opponents, np.full(n_opponents, 1 / n_opponents), DRAWS).astype(float)


def bootstrap(row, guidance, counts):
    """{(ask numerator, seed): [DRAWS] mean got} with teams resampled within the cell and opponents by `counts`."""
    data = np.load(opponents_path(row, guidance)); weights = data["weights"]
    rng = np.random.default_rng([DRAW_SEED, list(ROWS).index(row), int(guidance)]); draws = {}
    for k in ALL_ASKS:
        for seed in SEEDS:
            rate = data[f"a{k:02d}_s{seed}"]; seen = ~np.isnan(rate)
            scores = (np.where(seen, rate, 0.0) @ (counts * weights).T) / np.maximum(seen @ (counts * weights).T, 1e-12)   # [teams, DRAWS]
            pick = rng.integers(0, len(rate), (DRAWS, len(rate)))
            draws[(k, seed)] = scores[pick, np.arange(DRAWS)[:, None]].mean(1)
    return draws


def pooled(draws, sizes, seeds=None):
    """{ask numerator: [DRAWS] mean got over the named seeds}, each cell weighted by its number of teams."""
    seeds = seeds or SEEDS
    return {k: sum(draws[(k, s)] * sizes[(k, s)] for s in seeds) / sum(sizes[(k, s)] for s in seeds) for k in ALL_ASKS}


def error_draws(got):
    asks = [k for k in SCORED_ASKS if k in got]
    return np.mean([(got[k] - k / 24) ** 2 for k in asks], 0) if asks else np.full(DRAWS, np.nan)


def interval(values):
    return [float(np.quantile(values, .025)), float(np.quantile(values, .975))]


# ---------------------------------------------------------------- measures of one grid
def references():
    """Line-ups of the labelled training teams of the shuffled-only table, and of earlier search runs and tournament records."""
    import line_up_yield as LY
    LY.TABLE, LY.LABELLED = TABLE, True
    return LY.references("")


def measures(out, refs):
    import line_up_yield as LY
    cells = out["by_ask_seed"]; teams = [(k, s, p, float(y)) for (k, s), c in cells.items() for p, y in zip(c["pastes"], c["y"])]
    line_ups = [LY.line_up(p) for _, _, p, _ in teams]
    got = {k: float(np.mean([y for a, _, _, y in teams if a == k])) for k in ALL_ASKS}
    by_seed = {s: {k: float(np.mean(cells[(k, s)]["y"])) for k in ALL_ASKS} for s in SEEDS}
    tried, rejected = sum(c["tried"] for c in cells.values()), sum(c["rejected"] for c in cells.values())
    top = [y for a, _, _, y in teams if a in TOP_ASKS]
    return dict(reader=out["reader"], guidance=out["guidance"], decoder=out.get("decoder"), checkpoint=out["checkpoint"], checkpoint_sha256=out["checkpoint_sha256"],
                teams=len(teams), battles=sum(c["battles"] for c in cells.values()), cells=len(cells),
                cells_short_of_64=sum(len(c["pastes"]) < out["keep"] for c in cells.values()),
                error=error(got), error_by_seed={str(s): error(g) for s, g in by_seed.items()}, error_asks_18_to_23=error(got, range(18, 24)),
                got_by_ask={str(k): v for k, v in got.items()},
                win_top_asks=float(np.mean(top)) if top else None, pooled_win=float(np.mean([y for *_, y in teams])),
                first_try_legality=(tried - rejected) / max(tried, 1), tries=tried,
                line_ups=len(set(line_ups)), line_ups_by_seed={str(s): len({lu for (_, t, _, _), lu in zip(teams, line_ups) if t == s}) for s in SEEDS},
                new_against_training=float(np.mean([lu not in refs[0] for lu in line_ups])),
                new_against_search_and_tournaments=float(np.mean([lu not in refs[1] for lu in line_ups])),
                effective_species=float(np.mean([c["div"]["effective_species"] for c in cells.values() if c["div"]])),
                top_species_share=float(np.mean([c["div"]["top_species_share"] for c in cells.values() if c["div"]])),
                rejections=sum((Counter(dict(c["reasons"])) for c in cells.values()), Counter()).most_common(3))


def verdict(arm, control, arm_draws, control_draws, linear):
    """The plan's rule at one guidance value, plus the two flags it names: training luck and a trade."""
    diff = arm["error"] - control["error"]; lo, hi = interval(arm_draws["error"] - control_draws["error"])
    word = "wins" if diff <= -BAR and hi < 0 else "loses" if lo > 0 else "matches"
    by_seed = {s: arm["error_by_seed"][s] - control["error_by_seed"][s] for s in arm["error_by_seed"]}
    return dict(verdict=word, difference=diff, interval=[lo, hi], difference_by_seed=by_seed, seeds_lower=sum(v < 0 for v in by_seed.values()),
                inside_training_luck=bool(linear and word != "matches" and abs(diff) < TRAINING_LUCK),
                pooled_win_change=arm["pooled_win"] - control["pooled_win"], line_ups_change=arm["line_ups"] - control["line_ups"],
                trade=bool(word == "wins" and (arm["pooled_win"] - control["pooled_win"] < -WIN_RATE_LUCK or arm["line_ups"] < control["line_ups"])))


def outside_frontier(point, frontier):
    """The plan's Pareto rule, counted only when the intervals clear. A point is outside the control's curve when it has
    a lower error than every control point with at least as many line-ups, or more line-ups than every control point
    with an error at least as low. "At least as many" and "at least as low" include every control point the intervals
    cannot rule out, and "lower" and "more" need the intervals apart. Error has its 95% bootstrap interval; line-ups have
    the range over the three sampling seeds, because a bootstrap of teams undercounts distinct line-ups."""
    seeds = lambda g: g["line_ups_by_seed"].values()
    as_many = [c for c in frontier if max(seeds(c)) >= min(seeds(point))]      # none: the point has more line-ups than every control point
    by_error = all(point["error_interval"][1] < c["error_interval"][0] for c in as_many)
    as_low = [c for c in frontier if c["error_interval"][0] <= point["error_interval"][1]]   # none: its error is below every control point's
    by_line_ups = all(min(seeds(point)) > max(seeds(c)) for c in as_low)
    plain = all(point["error"] < c["error"] for c in frontier if c["line_ups"] >= point["line_ups"]) or \
            all(point["line_ups"] > c["line_ups"] for c in frontier if c["error"] <= point["error"])
    return dict(lower_error_than_every_control_point_with_as_many_line_ups=by_error,
                more_line_ups_than_every_control_point_with_error_as_low=by_line_ups, outside=by_error or by_line_ups,
                outside_by_point_estimates_alone=bool(plain))


def probe_numbers():
    path = RES / "masking_rate_rows_preflight.json"
    probe = json.load(open(path)).get(f"t_sensitivity_{PER_FIELD_CHECKPOINT}") if path.exists() else None
    if not probe: return None
    return dict(largest_total_variation=probe["largest_total_variation_corrected"], bar=probe["bar"],
                largest_share_of_fields_changed=max(v["share_of_fields_changed"] for v in probe["by_ask"].values()),
                largest_share_of_line_ups_changed=max(v["share_of_line_ups_changed"] for v in probe["by_ask"].values()))


def unguided(row):
    path = RES / f"stat_points_84_{row}_unguided{'_smoke' if SMOKE else ''}.json"
    if not path.exists(): return None
    import line_up_yield as LY
    cells = list(json.load(open(path))["cells"].values())
    if any("y" not in c for c in cells) or len(cells) < len(SEEDS): return None
    y = [v for c in cells for v in c["y"]]; tried = sum(c["tried"] for c in cells)
    return dict(teams=len(y), battles=sum(c["battles"] for c in cells), win=float(np.mean(y)), win_by_seed={str(c["seed"]): float(np.mean(c["y"])) for c in cells},
                line_ups=len({LY.line_up(p) for c in cells for p in c["pastes"]}), first_try_legality=(tried - sum(c["rejected"] for c in cells)) / max(tried, 1))


def elite_yield(name):
    path = RES / f"line_up_yield_{name}.json"
    views = json.load(open(path)).get("views") if path.exists() else None
    if not views: return None
    return {v: {k: views[v][k] for k in ("elite_line_ups", "rebattled", "held", "yield_per_1000", "one_score_mean", "fresh_mean", "held_new_against_training")} for v in views}


def finished_grids(refs):
    """({grid name: measures}, {grid name: bootstrap draws}) for the grids that have finished."""
    grids, draws, counts = {}, {}, None
    for row in ROWS:
        for g in GUIDANCES:
            out = load_grid(row, g)
            if out is None: continue
            name = grid_name(row, g); m = grids[name] = dict(row=row, **measures(out, refs)); m["elite_line_up_yield"] = elite_yield(name)
            if not opponents_path(row, g).exists(): continue
            if counts is None: counts = opponent_draws(len(np.load(opponents_path(row, g))["weights"]))
            sizes = {key: len(c["pastes"]) for key, c in out["by_ask_seed"].items()}; got = pooled(bootstrap(row, g, counts), sizes)
            draws[name] = dict(error=error_draws(got), pooled_win=np.mean([got[k] for k in ALL_ASKS], 0),
                               win_top=np.mean([got[k] for k in TOP_ASKS if k in ALL_ASKS] or [np.full(DRAWS, np.nan)], 0))
            m.update(error_interval=interval(draws[name]["error"]), win_top_asks_interval=interval(draws[name]["win_top"]), pooled_win_interval=interval(draws[name]["pooled_win"]))
    return grids, draws


def row_verdict(words):
    """A row wins when it wins at two or more guidance values and loses at none; one win among three is weak."""
    wins, loses = words.count("wins"), words.count("loses")
    return "wins" if wins >= 2 and not loses else "weak win" if wins == 1 and not loses else "loses" if loses and not wins else "mixed" if wins and loses else "matches"


def comparisons(grids, draws):
    """Each new row against the control at the same guidance, random order against species-first on the linear model, and each row's verdict."""
    verdicts, order, rows = {}, {}, {}
    for g in GUIDANCES:
        control = grid_name(CONTROL, g)
        for name in (grid_name(row, g) for row in ROWS if row != CONTROL):
            if name in draws and control in draws: verdicts[name] = verdict(grids[name], grids[control], draws[name], draws[control], linear=True)
        a, b = grid_name("sched_linear_random", g), grid_name("sched_linear_species_first", g)
        if a in draws and b in draws:
            order[f"guidance_{g}"] = dict(verdict(grids[a], grids[b], draws[a], draws[b], linear=False), error_multiple=grids[a]["error"] / grids[b]["error"],
                                          error_multiple_interval=interval(draws[a]["error"] / draws[b]["error"]))
    for row in ROWS:
        words = [verdicts[grid_name(row, g)]["verdict"] for g in GUIDANCES if grid_name(row, g) in verdicts]
        if len(words) == len(GUIDANCES): rows[row] = dict(verdicts=words, overall=row_verdict(words))
    return verdicts, order, rows


def summary():
    """Every measure for every finished grid, the verdicts against the control, and the Pareto reading."""
    sys.path.insert(0, "/tmp/vgc-pilot/src"); sys.path.insert(0, str(HERE))
    grids, draws = finished_grids(references()); verdicts, order, rows = comparisons(grids, draws)
    frontier = [grids[n] for n in (grid_name(CONTROL, g) for g in GUIDANCES) if n in draws]
    pareto = {n: outside_frontier(m, frontier) for n, m in grids.items() if m["row"] != CONTROL and n in draws and len(frontier) == len(GUIDANCES)}
    result = dict(grids=grids, verdicts_against_control=verdicts, random_order_against_species_first=order, pareto=pareto, row_verdicts=rows,
                  corrected_t_probe=probe_numbers(), unguided={row: unguided(row) for row in ROWS},
                  lookahead_off=json.load(open(LEGALITY)).get("summary") if LEGALITY.exists() else None,
                  reference_default_model_yield=elite_yield("gen84_noise_byepoch_ep80"),
                  bootstrap=dict(draws=DRAWS, seed=DRAW_SEED, resampled="teams within each ask and seed, separately in each row; the 49 opponents once for all rows"),
                  win_rate_definition="one battle against each of the 49 distinct top-50 meta teams, behaviour-cloning policy on both sides, slots shuffled")
    json.dump(result, open(SUMMARY, "w"), indent=1)
    return result


# ---------------------------------------------------------------- random order with the lookahead off
def legality_probe():
    """First-try legality of random order on the linear model with the lookahead off, as on 2026-11-07: sampling only."""
    os.environ.update(AG84_MODEL="sched_linear_random")
    sys.path.insert(0, str(HERE))
    import torch
    import ask_grid84 as A
    import remask84
    tries, asks, seeds = (48, (8,), [1]) if SMOKE else (PROBE_TRIES, PROBE_ASKS, SEEDS)
    out = json.load(open(LEGALITY)) if LEGALITY.exists() else dict(cells={})
    out.update(row="sched_linear_random", lookahead=False, guidance=A.GUIDANCE, tries_per_cell=tries, checkpoint=A.checkpoint_path().name)
    w = A.load_model(); arm = remask84.RandomDecodingArm(w, "rand_plain", lookahead=False); val = A.Validator()
    try:
        for seed in seeds:
            for k in asks:
                tag = f"a{k:02d}of24_s{seed}"
                if tag in out["cells"]: continue
                torch.manual_seed(A.SEED0 + seed); rejected, reasons, tried = 0, Counter(), 0
                while tried < tries:
                    for r in arm.sample(min(96, tries - tried), ask=k / 24, guidance=A.GUIDANCE).cpu().numpy():
                        tried += 1; err = val(w.to_paste(r, repair=True)[0])
                        if err is not None: rejected += 1; reasons[A.reason(err)] += 1
                out["cells"][tag] = dict(numerator=k, seed=seed, tried=tried, rejected=rejected, legal=(tried - rejected) / tried, reasons=reasons.most_common(5))
                json.dump(out, open(LEGALITY, "w")); print(f"  {tag}: legal {out['cells'][tag]['legal']:.1%} of {tried}", flush=True)
    finally:
        val.close()
    cells = out["cells"].values(); tried = sum(c["tried"] for c in cells)
    out["summary"] = dict(tried=tried, first_try_legality=(tried - sum(c["rejected"] for c in cells)) / tried,
                          rejections=sum((Counter(dict(c["reasons"])) for c in cells), Counter()).most_common(3))
    json.dump(out, open(LEGALITY, "w")); print("MRR_LEGALITY_DONE", json.dumps(out["summary"]))


# ---------------------------------------------------------------- the figure and comment of one grid
THEMES = {
    "light": dict(bg="#ffffff", ink="#0b0b0b", soft="#52514e", box="#f3f2ee", edge="#d5d4cd", c1="#2a78d6", ok="#16803c", bad="#c0392b", plain="#52514e"),
    "dark": dict(bg="#0d1117", ink="#f2f1ee", soft="#a9a8a2", box="#1f1f1e", edge="#3a3a35", c1="#5b9bf0", ok="#4ade80", bad="#f87171", plain="#a9a8a2"),
}
FONT = '-apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif'
TONE = dict(wins="ok", loses="bad", matches="plain")
WIDTH, PAD, LABEL_W, ROW_H = 800, 20, 204, 92
READING = [
    "Big number: the ask-versus-got error. The ask is the win rate the model is told to aim for, and got is the win rate its teams then score. "
    "The error is the mean squared gap between the two on asks 1/24 to 17/24. Lower is better.",
    "Row: one way to train and sample the model. Per-field and linear name the masking rate, the chance a field is hidden during training: "
    "set by field type, or the same for every field. Species-first and random order name the order the sampler fills fields in.",
    "Guidance: how hard the sampler pushes toward the ask. 1 is no push.",
    "Line-ups: distinct sets of six species among the grid's teams. Win: mean win rate at asks 15/24 to 17/24, one battle against each of "
    "the 49 distinct top-50 meta teams, slots shuffled.",
    "Wins, matches, loses: against the control at the same guidance. A win needs an error at least 0.001 lower with the 95% interval of the "
    "difference below zero. A loss needs that interval above zero.",
]


def wrap(text, width, size, bold=False):
    per_char = size * (0.58 if bold else 0.53); limit, lines, line = max(8, int(width / per_char)), [], ""
    for word in str(text).split():
        if line and len(line) + 1 + len(word) > limit: lines.append(line); line = word
        else: line = f"{line} {word}".strip()
    return [*lines, line] if line else lines or [""]


def figure(result, landed, theme, headline):
    """The results so far as a matrix: one line per row, one column per guidance value, each cell colored by its verdict."""
    t, parts = THEMES[theme], []
    text = lambda x, y, s, size=12, tone="soft", bold=False, anchor="start": parts.append(
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{t[tone]}" text-anchor="{anchor}"{" font-weight=" + chr(34) + "600" + chr(34) if bold else ""}>{html.escape(str(s))}</text>')
    rect = lambda x, y, w, h, fill, stroke, dash=False, width=1: parts.append(
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="6" fill="{fill}" stroke="{stroke}" stroke-width="{width}"{" stroke-dasharray=" + chr(34) + "5 4" + chr(34) if dash else ""}/>')
    y = PAD + 18
    for line in wrap(headline, WIDTH - 2 * PAD, 18, True): text(PAD, y, line, 18, "ink", True); y += 24
    text(PAD, y, "Results so far for the masking-rate rows. The thick border marks the grid that just finished.", 12); y += 22
    cell_w = (WIDTH - 2 * PAD - LABEL_W) / len(GUIDANCES)
    for i, g in enumerate(GUIDANCES): text(PAD + LABEL_W + cell_w * (i + .5), y, f"guidance {g}", 12, "soft", True, "middle")
    y += 10
    for row, reader in ROWS.items():
        text(PAD, y + 30, reader, 13, "ink", True)
        for j, line in enumerate(wrap(ROW_NOTES[row], LABEL_W - 12, 11)): text(PAD, y + 48 + 14 * j, line, 11)
        for i, g in enumerate(GUIDANCES):
            name, x = grid_name(row, g), PAD + LABEL_W + cell_w * i + 4
            m, v, mid = result["grids"].get(name), result["verdicts_against_control"].get(name), x + (cell_w - 8) / 2
            if m is None:
                rect(x, y + 4, cell_w - 8, ROW_H - 8, "none", t["edge"], dash=True); text(mid, y + ROW_H / 2 + 4, "not run yet", 12, "soft", False, "middle"); continue
            tone = "c1" if row == CONTROL else TONE[v["verdict"]] if v else "plain"
            rect(x, y + 4, cell_w - 8, ROW_H - 8, t["box"], t[tone], width=3 if name == landed else 1.5)
            text(mid, y + 36, f"{m['error']:.4f}", 22, "ink", True, "middle")
            word = "the control" if row == CONTROL else f"{v['verdict']}, {v['difference']:+.4f}" if v else "control not run yet"
            text(mid, y + 55, word, 12, tone, True, "middle")
            text(mid, y + 73, f"{m['line_ups']:,} line-ups · win {m['win_top_asks']:.3f}" if m["win_top_asks"] is not None else f"{m['line_ups']:,} line-ups", 11, "soft", False, "middle")
        y += ROW_H
    probe = result.get("corrected_t_probe")
    text(PAD, y + 30, CORRECTED_T, 13, "ink", True); text(PAD, y + 48, "the control's model, told another t", 11)
    rect(PAD + LABEL_W + 4, y + 4, WIDTH - 2 * PAD - LABEL_W - 8, ROW_H - 30, "none", t["edge"], dash=True)
    note = (f"Not battled. Time t says how much of the team is hidden. Telling the model the t its training used moved its drawing odds by "
            f"{probe['largest_total_variation']:.4f} at most, under the {probe['bar']} bar set beforehand.") if probe else "Not battled: its probe stayed under the bar set beforehand."
    for j, line in enumerate(wrap(note, WIDTH - 2 * PAD - LABEL_W - 40, 12)): text(PAD + LABEL_W + 18, y + 26 + 16 * j, line, 12)
    y += ROW_H - 8
    text(PAD, y + 4, "How to read this", 12, "ink", True); y += 22
    for item in READING:
        for line in wrap(item, WIDTH - 2 * PAD, 11.5): text(PAD, y, line, 11.5); y += 15.5
        y += 5
    body = "".join(parts)
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {y + PAD - 8:.0f}" width="{WIDTH}" font-family=\'{FONT}\' role="img">'
            f'<title>{html.escape(headline)}</title><rect width="100%" height="100%" fill="{t["bg"]}"/>{body}</svg>')


def headline_of(result, name):
    m, v = result["grids"][name], result["verdicts_against_control"].get(name)
    who = f"{ROWS[m['row']]} at guidance {m['guidance']:g}"
    if m["row"] == CONTROL: return f"The control, {who}, reads an ask-versus-got error of {m['error']:.4f}"
    if v is None: return f"{who} reads an ask-versus-got error of {m['error']:.4f}; the control at this guidance has not run"
    control = result["grids"][grid_name(CONTROL, m["guidance"])]
    verb = dict(wins="beats", loses="loses to", matches="matches")[v["verdict"]]
    return f"{who} {verb} the control: error {m['error']:.4f} against {control['error']:.4f}"


def comment(result, name, picture):
    m, v = result["grids"][name], result["verdicts_against_control"].get(name)
    control = result["grids"].get(grid_name(CONTROL, m["guidance"])) if m["row"] != CONTROL else None
    who = f"{ROWS[m['row']]}, guidance {m['guidance']:g}"
    pct, num = (lambda x: f"{x:.1%}"), (lambda x, d=4: "none run" if x is None else f"{x:.{d}f}")
    with_interval = lambda g, key, d=4: f"{num(g[key], d)} ({g[key + '_interval'][0]:.{d}f} to {g[key + '_interval'][1]:.{d}f})" if g.get(key + "_interval") else num(g[key], d)
    lines = [("Ask-versus-got error, asks 1/24 to 17/24, with its 95% interval", lambda g: with_interval(g, "error")),
             ("The same error for sampling seeds 1, 2 and 3", lambda g: ", ".join(num(e) for e in g["error_by_seed"].values())),
             ("Error on asks 18/24 to 23/24", lambda g: num(g["error_asks_18_to_23"])),
             ("Win rate at asks 15/24 to 17/24", lambda g: with_interval(g, "win_top_asks", 3)),
             ("Win rate over all 23 asks", lambda g: with_interval(g, "pooled_win", 3)),
             ("First-try legality", lambda g: pct(g["first_try_legality"])),
             ("Distinct line-ups", lambda g: f"{g['line_ups']:,}"),
             ("Teams with a line-up absent from the training teams", lambda g: pct(g["new_against_training"])),
             ("Teams with a line-up absent from earlier search runs and tournament records", lambda g: pct(g["new_against_search_and_tournaments"])),
             ("Effective species and top-species share, in 5-team subsets", lambda g: f"{g['effective_species']:.1f} and {g['top_species_share']:.3f}"),
             ("Teams and battles", lambda g: f"{g['teams']:,} and {g['battles']:,}")]
    head = f"| Measure | {who} |" + (f" The control at guidance {m['guidance']:g} |" if control else "")
    table = [head, "|---|---:|" + ("---:|" if control else "")] + [f"| {label} | {show(m)} |" + (f" {show(control)} |" if control else "") for label, show in lines]
    out = [f"**Grid finished: {who}.** {headline_of(result, name)}.", "", picture, "", *table, ""]
    if v:
        signs = ", ".join(f"{d:+.4f}" for d in v["difference_by_seed"].values())
        out.append(f"Against the control the error differs by {v['difference']:+.4f}, 95% interval {v['interval'][0]:+.4f} to {v['interval'][1]:+.4f}, "
                   f"and seed by seed by {signs}. The verdict at this guidance is **{v['verdict']}**."
                   + (" The gap is under 0.002, so it is inside the luck of one training seed." if v["inside_training_luck"] else "")
                   + (" The win comes with a lower win rate or fewer line-ups, so it is a trade." if v["trade"] else ""))
    if m["cells_short_of_64"]: out.append(f"{m['cells_short_of_64']} of {m['cells']} cells ended with fewer than 64 legal teams after 1,280 tries.")
    out.append("Win rate means one battle against each of the 49 distinct top-50 meta teams, with the behaviour-cloning policy on both sides and slots shuffled. "
               "The model generates all 84 fields. The interval resamples teams within each ask and seed, and the 49 opponents.")
    out.append("The queue posted this the moment the grid ended, and nobody has read these numbers yet. The elite line-up yield, the unguided baselines "
               "and the reading across all nine grids follow in the final table.")
    return "\n".join(out)


def post(row, guidance, folder):
    """Write the grid's figure, light and dark, and its comment. FIGURE_URL holds the address the two files will have, with {theme} in it."""
    result, name = summary(), grid_name(row, guidance)
    if name not in result["grids"]: raise SystemExit(f"{name}: the grid has not finished")
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True); headline = headline_of(result, name)
    for theme in THEMES: (folder / f"grid-{name}-{theme}.svg").write_text(figure(result, name, theme, headline))
    url = os.environ.get("FIGURE_URL", f"grid-{name}-{{theme}}.svg")
    alt = html.escape(headline + ".", quote=True)
    picture = (f'<picture><source media="(prefers-color-scheme: dark)" srcset="{url.format(theme="dark", name=name)}">'
               f'<img width="800" alt="{alt}" src="{url.format(theme="light", name=name)}"></picture>')
    (folder / f"comment-{name}.md").write_text(comment(result, name, picture) + "\n")
    print(folder / f"comment-{name}.md")


if __name__ == "__main__":
    step, args = (sys.argv[1] if len(sys.argv) > 1 else ""), sys.argv[2:]
    if step == "legality_probe": legality_probe()
    elif step == "opponents": opponents(args[0], float(args[1]))
    elif step == "check": check(args[0], int(args[1]))
    elif step == "summary": print(json.dumps({k: v for k, v in summary().items() if k in ("verdicts_against_control", "row_verdicts", "pareto")}, indent=1)); print("MRR_SUMMARY_DONE")
    elif step == "post": post(args[0], float(args[1]), args[2])
    else: sys.exit(__doc__)
