"""Does Linear, random order draw weaker teams because it draws species late? (GitHub issue #322)

Step one reads the stored grids of Linear, random order and Linear, species-first (masking_rate_rows.py) and compares
teams that share a line-up, the same six species, at the same ask. If random order's teams lose even with the same six
species, the loss is in the sets. If they win about as well, the loss is in which species random order picks.

    python species_late.py stored        the comparison at guidance 2, 4 and 8, into results/species_late_stored.json
    python species_late.py stored_figure DIR    its chart as an ECharts page in DIR
    python species_late.py grid          step two: Linear, species then random order at guidance 2 against the two rows, into results/species_late_grid.json
    python species_late.py grid_figure DIR      got against ask for the three rows, as an ECharts page in DIR
    python species_late.py grid_comment URL     the verdict comment; URL holds {theme} for light or dark

The interval is a bootstrap that resamples teams within each line-up at each ask, separately in each row, and the 49
opponents once for both rows, as masking_rate_rows.py does for its grids.
"""
import datetime
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import masking_rate_rows as MRR

RANDOM, SPECIES_FIRST, SPECIES_THEN_RANDOM = "sched_linear_random", "sched_linear_species_first", "sched_linear_species_then_random"
READER = {SPECIES_THEN_RANDOM: "Linear, species then random order", SPECIES_FIRST: "Linear, species-first", RANDOM: "Linear, random order"}
STREAM = {SPECIES_THEN_RANDOM: 0, SPECIES_FIRST: 1, RANDOM: 2}                # each row's own bootstrap stream, fixed so a new row moves no other
HIGH_ASKS = range(13, 24)                                # where got stops following the ask on Linear, random order
VERDICT_ASKS = range(13, 18)                             # the ticket reads got at asks 13/24 to 17/24
GOT_MARGIN = 0.03                                        # within 0.03 of a row's got, as the ticket states
STORED, GRID = MRR.RES / "species_late_stored.json", MRR.RES / "species_late_grid.json"


def scores(rate, weights, counts=None):
    """[teams] each team's weighted win rate, or [teams, DRAWS] with the opponents counted `counts` [DRAWS, opponents] times."""
    seen = ~np.isnan(rate); won = np.where(seen, rate, 0.0)
    if counts is None: return (won @ weights) / (seen @ weights)
    w = (counts * weights).T
    return (won @ w) / np.maximum(seen @ w, 1e-12)


def group_mean(teams, weights, counts, rng):
    """Mean score of a list of [opponents] rate rows: a number, or [DRAWS] with the teams resampled when counts is given."""
    s = scores(np.stack(teams), weights, counts)
    if counts is None: return float(s.mean())
    pick = rng.integers(0, len(teams), (len(counts), len(teams)))
    return s[pick, np.arange(len(counts))[:, None]].mean(1)


def gaps(random_teams, first_teams, weights, counts=None, rng=None, asks=None):
    """random_teams, first_teams: {ask numerator: [(line-up, rate [opponents])]}. Returns the same-line-up gap (random
    order's matched teams minus species-first's teams with the same line-up at the same ask, weighted by random order's
    team count), the two matched means, the total gap (each ask's mean over all its teams, averaged over asks), and the
    matched counts. Numbers without counts; [DRAWS] arrays with counts and rng."""
    asks = sorted(set(random_teams) & set(first_teams) if asks is None else set(asks) & set(random_teams) & set(first_teams))
    matched_r, matched_f, n_total, groups, totals = 0.0, 0.0, 0, 0, []
    for k in asks:
        by_r, by_f = defaultdict(list), defaultdict(list)
        for lineup, rate in random_teams[k]: by_r[tuple(sorted(lineup))].append(rate)
        for lineup, rate in first_teams[k]: by_f[tuple(sorted(lineup))].append(rate)
        for lineup in sorted(by_r.keys() & by_f.keys()):        # a fixed order, so the bootstrap draws repeat from run to run
            n = len(by_r[lineup])
            matched_r = matched_r + n * group_mean(by_r[lineup], weights, counts, rng)
            matched_f = matched_f + n * group_mean(by_f[lineup], weights, counts, rng)
            n_total += n; groups += 1
        totals.append(group_mean([r for _, r in random_teams[k]], weights, counts, rng) - group_mean([r for _, r in first_teams[k]], weights, counts, rng))
    nan = np.nan if counts is None else np.full(len(counts), np.nan)
    r, f = (matched_r / n_total, matched_f / n_total) if n_total else (nan, nan)
    return dict(same_line_up=r - f, random_matched=r, species_first_matched=f, total=np.mean(totals, 0) if totals else nan,
                matched_random_teams=n_total, matched_line_ups=groups)


def stored_teams(row, guidance):
    """{ask numerator: [(line-up, rate [opponents])]} pooled over sampling seeds, and the opponents' weights."""
    out = MRR.load_grid(row, guidance)
    if out is None: raise SystemExit(f"{MRR.grid_name(row, guidance)}: the grid is missing or has cells without battles")
    data = np.load(MRR.opponents_path(row, guidance)); teams = defaultdict(list)
    for (k, seed), cell in sorted(out["by_ask_seed"].items()):
        rate = data[f"a{k:02d}_s{seed}"]
        if len(rate) != len(cell["stat_points"]): raise SystemExit(f"{MRR.grid_name(row, guidance)} a{k:02d}_s{seed}: opponent rows and teams differ")
        teams[k] += [([s for s, _ in slots], rate[i]) for i, slots in enumerate(cell["stat_points"])]
    return dict(teams), data["weights"]


def summarize(random_teams, first_teams, weights, counts, rng, asks=None):
    point, draws = gaps(random_teams, first_teams, weights, asks=asks), gaps(random_teams, first_teams, weights, counts, rng, asks)
    out = {k: v for k, v in point.items() if k.startswith("matched")}
    for k in ("same_line_up", "random_matched", "species_first_matched", "total"):
        out[k] = float(point[k]); out[f"{k}_interval"] = MRR.interval(draws[k]) if point["matched_random_teams"] else None
    out["share_of_total_with_same_line_up"] = float(point["same_line_up"] / point["total"]) if point["total"] else None
    return out


def stored():
    result = dict(rows=dict(random=RANDOM, species_first=SPECIES_FIRST), draws=MRR.DRAWS, draw_seed=MRR.DRAW_SEED,
                  resampled="teams within each line-up at each ask, separately in each row; the 49 opponents once for both rows",
                  win_rate_definition="one battle against each of the 49 distinct top-50 meta teams, behaviour-cloning policy on both sides, slots shuffled",
                  guidance={})
    for g in MRR.GUIDANCES:
        (r, weights), (f, _) = stored_teams(RANDOM, g), stored_teams(SPECIES_FIRST, g)
        counts = MRR.opponent_draws(len(weights)); rng = np.random.default_rng([MRR.DRAW_SEED, 322, g])
        result["guidance"][str(g)] = dict(
            all_asks=summarize(r, f, weights, counts, rng),
            high_asks=summarize(r, f, weights, counts, rng, asks=HIGH_ASKS),
            asks_pooled=summarize({0: [t for ts in r.values() for t in ts]}, {0: [t for ts in f.values() for t in ts]}, weights, counts, rng))   # the first look: line-ups matched across asks
    json.dump(result, open(STORED, "w"), indent=1)
    print("| Guidance | Asks | Random-order teams matched | Random order | Species-first, same line-ups | Same-line-up gap | Total gap |")
    print("|---|---|---:|---:|---:|---|---|")
    fmt = lambda m, k: f"{m[k]:+.3f} [{m[k + '_interval'][0]:+.3f}, {m[k + '_interval'][1]:+.3f}]" if m[k + "_interval"] else "none matched"
    for g, by in result["guidance"].items():
        for name, m in by.items():
            print(f"| {g} | {name} | {m['matched_random_teams']} | {m['random_matched']:.3f} | {m['species_first_matched']:.3f} | {fmt(m, 'same_line_up')} | {fmt(m, 'total')} |")
    print("SPECIES_LATE_STORED_DONE")


def verdict(got, got_first, got_random, error, error_first):
    """The ticket's rule. got: mean win rate at asks 13/24 to 17/24; error: ask-versus-got error on asks 1/24 to 17/24.
    Confirmed: got within 0.03 of Linear, species-first's and the error within training luck of its error. Cleared: got
    within 0.03 of Linear, random order's. Anything between is partial."""
    if abs(got - got_first) <= GOT_MARGIN and abs(error - error_first) <= MRR.TRAINING_LUCK: return "confirmed"
    if abs(got - got_random) <= GOT_MARGIN: return "cleared"
    return "partial"


def read_row(row, counts, guidance=2):
    """Point estimates and [DRAWS] bootstrap draws of one grid: got by ask, error, error by seed, got at asks 13 to 17."""
    out = MRR.load_grid(row, guidance)
    if out is None: raise SystemExit(f"{MRR.grid_name(row, guidance)}: the grid is missing or has cells without battles")
    data = np.load(MRR.opponents_path(row, guidance)); weights = data["weights"]
    rng = np.random.default_rng([MRR.DRAW_SEED, 322, STREAM[row]])
    point, draws = {}, {}
    for k in MRR.ALL_ASKS:
        for seed in MRR.SEEDS:
            rate = data[f"a{k:02d}_s{seed}"]
            if abs(scores(rate, weights).mean() - np.mean(out["by_ask_seed"][(k, seed)]["y"])) > 1e-9: raise SystemExit(f"{row} a{k:02d}_s{seed}: snapshot and stored scores differ")
            point[(k, seed)] = (float(scores(rate, weights).mean()), len(rate))
            s = scores(rate, weights, counts); pick = rng.integers(0, len(rate), (len(counts), len(rate)))
            draws[(k, seed)] = s[pick, np.arange(len(counts))[:, None]].mean(1)
    sizes = {c: n for c, (_, n) in point.items()}
    got_p = {k: sum(point[(k, s)][0] * sizes[(k, s)] for s in MRR.SEEDS) / sum(sizes[(k, s)] for s in MRR.SEEDS) for k in MRR.ALL_ASKS}
    got_d = MRR.pooled(draws, sizes)
    by_seed_p = {s: MRR.error({k: point[(k, s)][0] for k in MRR.ALL_ASKS}) for s in MRR.SEEDS}
    by_seed_d = {s: MRR.error_draws({k: draws[(k, s)] for k in MRR.ALL_ASKS}) for s in MRR.SEEDS}
    return dict(got_by_ask=got_p, got_by_ask_draws=got_d, error=MRR.error(got_p), error_draws=MRR.error_draws(got_d),
                error_by_seed=by_seed_p, error_by_seed_draws=by_seed_d,
                got_verdict_asks=float(np.mean([got_p[k] for k in VERDICT_ASKS])), got_verdict_asks_draws=np.mean([got_d[k] for k in VERDICT_ASKS], 0))


def paired(a, b):
    """Mean over sampling seeds of a's error minus b's at the same seed, with its interval."""
    diff = np.mean([a["error_by_seed"][s] - b["error_by_seed"][s] for s in MRR.SEEDS])
    draws = np.mean([a["error_by_seed_draws"][s] - b["error_by_seed_draws"][s] for s in MRR.SEEDS], 0)
    return dict(difference=float(diff), interval=MRR.interval(draws))


def grid():
    counts = MRR.opponent_draws(len(np.load(MRR.opponents_path(RANDOM, 2))["weights"]))
    rows = {row: read_row(row, counts) for row in (SPECIES_THEN_RANDOM, SPECIES_FIRST, RANDOM)}
    new, first, rand = rows[SPECIES_THEN_RANDOM], rows[SPECIES_FIRST], rows[RANDOM]
    result = dict(guidance=2, verdict=verdict(new["got_verdict_asks"], first["got_verdict_asks"], rand["got_verdict_asks"], new["error"], first["error"]),
                  rule=f"confirmed: got at asks 13/24 to 17/24 within {GOT_MARGIN} of {READER[SPECIES_FIRST]} and error within {MRR.TRAINING_LUCK} of its error; "
                       f"cleared: got within {GOT_MARGIN} of {READER[RANDOM]}; otherwise partial",
                  error_against_species_first=paired(new, first), error_against_random=paired(new, rand),
                  got_against_species_first=dict(difference=new["got_verdict_asks"] - first["got_verdict_asks"],
                                                 interval=MRR.interval(new["got_verdict_asks_draws"] - first["got_verdict_asks_draws"])),
                  got_against_random=dict(difference=new["got_verdict_asks"] - rand["got_verdict_asks"],
                                          interval=MRR.interval(new["got_verdict_asks_draws"] - rand["got_verdict_asks_draws"])),
                  error_multiple_of_species_first=new["error"] / first["error"],
                  resampled="teams within each ask and sampling seed, separately in each row; the 49 opponents once for all three rows",
                  draws=MRR.DRAWS, draw_seed=MRR.DRAW_SEED, rows={})
    for row, m in rows.items():
        result["rows"][row] = dict(reader=READER[row], error=m["error"], error_interval=MRR.interval(m["error_draws"]),
                                   error_by_seed={str(s): v for s, v in m["error_by_seed"].items()},
                                   got_asks_13_to_17=m["got_verdict_asks"], got_asks_13_to_17_interval=MRR.interval(m["got_verdict_asks_draws"]),
                                   got_by_ask={str(k): v for k, v in m["got_by_ask"].items()},
                                   got_by_ask_interval={str(k): MRR.interval(v) for k, v in m["got_by_ask_draws"].items()})
    json.dump(result, open(GRID, "w"), indent=1)
    for row, m in result["rows"].items():
        print(f"{m['reader']:38s} error {m['error']:.4f} {m['error_interval']}  got 13-17 {m['got_asks_13_to_17']:.3f} {m['got_asks_13_to_17_interval']}")
    print("against species-first:", result["error_against_species_first"], result["got_against_species_first"])
    print("against random order: ", result["error_against_random"], result["got_against_random"])
    print("VERDICT", result["verdict"]); print("SPECIES_LATE_GRID_DONE")


BLUE, ORANGE = "#2a78d6", "#e0702a"                        # masking_rate_rows_figures.py colors


def stored_figure(folder):
    """Two panels of the same y axis: random order minus species-first at asks 13/24 to 23/24, for teams with the same
    six species and for all teams. A box is the 95% interval, a dot the estimate."""
    from echarts_chart import write_chart
    result = json.load(open(STORED)); gs = list(result["guidance"])
    panels = [("same_line_up", "Same six species at the same ask", BLUE), ("total", "All teams", ORANGE)]
    grids, xs, ys, series = [], [], [], []
    for i, (key, name, color) in enumerate(panels):
        grids.append(dict(left=f"{8 + 47 * i}%", width="40%", top=24, bottom=60))
        xs.append(dict(type="category", gridIndex=i, data=[f"Guidance {g}" for g in gs], name=name, nameLocation="middle", nameGap=34,
                       nameTextStyle=dict(fontSize=14, fontWeight="bold", color=color)))
        ys.append(dict(type="value", gridIndex=i, min=-0.2, max=0.05, interval=0.05, axisLabel=dict(show=i == 0, formatter="js:v => (v > 0 ? '+' : '') + v.toFixed(2)"),
                       name="Win rate, random order minus species-first" if i == 0 else "", nameLocation="middle", nameGap=52, splitLine=dict(lineStyle=dict(opacity=0.35))))
        m = [result["guidance"][g]["high_asks"] for g in gs]
        series.append(dict(type="candlestick", name=name, xAxisIndex=i, yAxisIndex=i, barWidth=26,
                           itemStyle=dict(color=color, color0=color, borderColor=color, borderColor0=color, opacity=0.35),
                           data=[[v[f"{key}_interval"][0], v[f"{key}_interval"][1], v[f"{key}_interval"][0], v[f"{key}_interval"][1]] for v in m],
                           markLine=dict(silent=True, symbol="none", lineStyle=dict(color="#6f6e69", type="solid", width=1), label=dict(show=False), data=[dict(yAxis=0)])))
        series.append(dict(type="scatter", name=name, xAxisIndex=i, yAxisIndex=i, symbolSize=11, itemStyle=dict(color=color), emphasis=dict(focus="series"),
                           label=dict(show=True, position="right", distance=16, fontSize=13, formatter="js:p => (p.value[1] > 0 ? '+' : '') + p.value[1].toFixed(3)"),
                           data=[[j, v[key]] for j, v in enumerate(m)]))
    n = [v["high_asks"]["matched_random_teams"] for v in result["guidance"].values()]
    option = dict(grid=grids, xAxis=xs, yAxis=ys, series=series, tooltip=dict(trigger="item"), animation=False)
    return write_chart(option, Path(folder) / "species-late-stored.html", height=400,
                       title="With the same six species, random order's teams win about as well as species-first's",
                       subtitle="Linear masking rate, asks 13/24 to 23/24. Below zero: random order's teams win less.",
                       note=f"Same six species: random-order teams whose line-up species-first also drew at the same ask ({min(n)} to {max(n)} teams per guidance). "
                            "Box: 95% bootstrap interval over teams and the 49 opponents. Win rate: one battle against each of the 49 distinct top-50 meta teams, "
                            "behaviour-cloning policy on both sides, slots shuffled.")


AQUA = "#1baf7a"
ROW_COLOR = {SPECIES_THEN_RANDOM: AQUA, SPECIES_FIRST: BLUE, RANDOM: ORANGE}


def grid_figure(folder):
    """Got against ask at guidance 2 for the three rows, each with its 95% band, beside the line got = ask."""
    from echarts_chart import band, write_chart
    result = json.load(open(GRID)); asks = [int(k) for k in result["rows"][RANDOM]["got_by_ask"]]
    xs = [k / 24 for k in asks]; series = []
    for i, (row, m) in enumerate(result["rows"].items()):
        lo = [m["got_by_ask_interval"][str(k)][0] for k in asks]; hi = [m["got_by_ask_interval"][str(k)][1] for k in asks]
        series += band(m["reader"], ROW_COLOR[row], xs, lo, hi, stack=f"band{i}")
        series.append(dict(type="line", name=m["reader"], color=ROW_COLOR[row], symbolSize=6, lineStyle=dict(width=2.5 if row == SPECIES_THEN_RANDOM else 1.8),
                           emphasis=dict(focus="series"), data=[[x, m["got_by_ask"][str(k)]] for x, k in zip(xs, asks)]))
    series.append(dict(type="line", name="Got = ask", color="#6f6e69", symbol="none", silent=True, lineStyle=dict(type="dashed", width=1), data=[[0, 0], [1, 1]]))
    pct = "js:v => Math.round(v * 100) + '%'"
    option = dict(grid=dict(left=64, right=24, top=56, bottom=56), legend=dict(top=0, data=[m["reader"] for m in result["rows"].values()] + ["Got = ask"]),
                  tooltip=dict(trigger="axis", valueFormatter="js:v => v == null ? '' : (v * 100).toFixed(1) + '%'"),
                  xAxis=dict(type="value", min=0, max=1, interval=1 / 6, name="Ask: the win rate the model is asked for", nameLocation="middle", nameGap=34,
                             axisLabel=dict(formatter=pct)),
                  yAxis=dict(type="value", min=0, max=0.8, name="Got: measured win rate", nameLocation="middle", nameGap=46, axisLabel=dict(formatter=pct)),
                  series=series, animation=False)
    new = result["rows"][SPECIES_THEN_RANDOM]
    title = {"confirmed": "Confirmed: drawing the species first brings random order's teams back to species-first's",
             "cleared": "Cleared: drawing the species first leaves random order's teams as weak as before",
             "partial": "Partial: drawing the species first brings random order's teams part of the way back"}[result["verdict"]]
    return write_chart(option, Path(folder) / "species-late-grid.html", height=460, title=title,
                       subtitle=f"Guidance 2, linear masking rate. Ask-versus-got error: {new['error']:.4f} with species first then random order, "
                                f"{result['rows'][SPECIES_FIRST]['error']:.4f} species-first, {result['rows'][RANDOM]['error']:.4f} random order.",
                       note="Band: 95% bootstrap interval over teams and the 49 opponents. Got: one battle against each of the 49 distinct top-50 meta teams, "
                            "behaviour-cloning policy on both sides, slots shuffled; 192 teams per ask over 3 sampling seeds.")


def grid_comment(figure_url):
    """The verdict comment, picture first. figure_url has {theme} where light or dark goes."""
    r = json.load(open(GRID)); m = r["rows"]; new, first, rand = m[SPECIES_THEN_RANDOM], m[SPECIES_FIRST], m[RANDOM]
    iv = lambda x: f"[{x[0]:+.4f}, {x[1]:+.4f}]"
    lines = [f"**Step two, {datetime.date.today()}: verdict {r['verdict']}.** {READER[SPECIES_THEN_RANDOM]} draws the six species first, in species-first order, "
             "then the other 78 fields as Linear, random order fills them. Same linear checkpoint, guidance 2.", "",
             f'<picture><source media="(prefers-color-scheme: dark)" srcset="{figure_url.format(theme="dark")}"><img alt="Got against ask for the three rows at guidance 2" src="{figure_url.format(theme="light")}"></picture>', "",
             f"- Got at asks 13/24 to 17/24: {new['got_asks_13_to_17']:.3f}, against {first['got_asks_13_to_17']:.3f} for Linear, species-first "
             f"(gap {r['got_against_species_first']['difference']:+.3f} {iv(r['got_against_species_first']['interval'])}) and {rand['got_asks_13_to_17']:.3f} for Linear, random order.",
             f"- Ask-versus-got error on asks 1/24 to 17/24: {new['error']:.4f}, against {first['error']:.4f} for Linear, species-first "
             f"(paired by sampling seed: {r['error_against_species_first']['difference']:+.4f} {iv(r['error_against_species_first']['interval'])}) "
             f"and {rand['error']:.4f} for Linear, random order ({r['error_against_random']['difference']:+.4f} {iv(r['error_against_random']['interval'])}).",
             f"- Rule, from the ticket: {r['rule']}. Training luck is {MRR.TRAINING_LUCK}.", "",
             "This removes two things at once: late species and the lookahead's pull on species. Read it beside [How often does the lookahead override the model's draw in Linear, random order?](https://github.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/issues/321), which separates the two.", "",
             "<details><summary>Every row</summary>", "",
             "| Row | Error, asks 1/24 to 17/24 | Error by seed 1, 2, 3 | Got, asks 13/24 to 17/24 |", "|---|---|---|---|"]
    for row in m.values():
        lines.append(f"| {row['reader']} | {row['error']:.4f} {iv(row['error_interval'])} | {', '.join(f'{v:.4f}' for v in row['error_by_seed'].values())} | "
                     f"{row['got_asks_13_to_17']:.3f} [{row['got_asks_13_to_17_interval'][0]:.3f}, {row['got_asks_13_to_17_interval'][1]:.3f}] |")
    lines += ["", "Interval: 95% bootstrap, teams resampled within each ask and sampling seed, the 49 opponents resampled once for all three rows. "
              "Script: `code/src/species_late.py grid`. Results: `~/vgc-data/results/species_late_grid.json`.", "</details>"]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    {"stored": stored, "stored_figure": lambda: print(stored_figure(sys.argv[2])), "grid": grid,
     "grid_figure": lambda: print(grid_figure(sys.argv[2])), "grid_comment": lambda: print(grid_comment(sys.argv[2]), end="")}[sys.argv[1]]()
