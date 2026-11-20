"""Random order without the lookahead, redrawn until Showdown accepts: does exact random order match species-first? (GitHub issue #358)

Linear, random order with the lookahead off draws from the model's own random-order distribution; keeping only the teams
Showdown's validator accepts gives that distribution among legal teams, with no tilt and no weights (TWISTER, Thimmaiah
et al. 2026, sections 3.1 and 3.4). The row is ask_grid84's sched_linear_random_reject: the linear checkpoint, guidance 2,
asks 13/24 to 17/24, sampling seeds 1 to 3 on seed base 7300, 64 teams a cell, at most 20,000 tries a cell.

    python plain_reject.py smoke SECONDS       acceptance of the first cell (ask 15/24, seed 1) and the projected sampling time;
                                               exits 3 if acceptance is under 0.3%, the ticket's stop
    python plain_reject.py opponents           per-opponent win rates of the new grid, saved beside it (masking_rate_rows.opponents)
    python plain_reject.py read                the verdict and diagnostics against the two stored rows, into results/plain_reject_grid.json
    python plain_reject.py figure DIR          the two charts as ECharts pages in DIR
    python plain_reject.py headline            the verdict line
    python plain_reject.py comment URL         the verdict comment; URL holds {name} and {theme}

Mean win rate: the mean over asks 13/24 to 17/24 of each ask's mean over its 192 teams, each team scored by one battle
against each of the 49 distinct top-50 meta teams, behaviour-cloning policy on both sides, slots shuffled. Best of 64:
the mean over the 15 cells of each cell's best team. The interval is a bootstrap that resamples teams within each ask
and sampling seed, separately in each row, and the 49 opponents once for all three rows (masking_rate_rows.py).
"""
import datetime
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import masking_rate_rows as MRR

ROW, FIRST, RANDOM = "sched_linear_random_reject", "sched_linear_species_first", "sched_linear_random"
READER = {ROW: "Random order, no lookahead, redrawn until legal", FIRST: "Linear, species-first", RANDOM: "Linear, random order (lookahead on)"}
STREAM = {ROW: 0, FIRST: 1, RANDOM: 2}                    # each row's own bootstrap stream
ASKS, GUIDANCE, KEEP = list(range(13, 18)), 2, 64
CONFIRM, CLEAR = -0.03, -0.05                             # the ticket's bar on the mean-win-rate gap to Linear, species-first
STOP_ACCEPTANCE, SMOKE_CELL = 0.003, (15, 1)
PREDICTION = dict(acceptance=(0.008, 0.011), mean_win=(0.44, 0.47), best_of_64=(0.56, 0.62))
RESULT = MRR.RES / "plain_reject_grid.json"
ISSUE = "https://github.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/issues"
WIN_RATE = "one battle against each of the 49 distinct top-50 meta teams, behaviour-cloning policy on both sides, slots shuffled"


def grid_file(row): return MRR.RES / f"stat_points_84_{row}.json"


def acceptance(cells):
    """Share of tries Showdown's validator accepted, over the cells given."""
    tried = sum(c["tried"] for c in cells)
    return (tried - sum(c["rejected"] for c in cells)) / max(tried, 1)


def verdict(gap, interval):
    """The ticket's bar. Confirmed: the mean-win-rate gap to Linear, species-first is no worse than -0.03. Cleared: it is
    -0.05 or worse and the interval's upper end is below -0.03. Anything between is partial."""
    if gap >= CONFIRM: return "confirmed"
    if gap <= CLEAR and interval[1] < CONFIRM: return "cleared"
    return "partial"


def smoke(seconds):
    """The first cell's acceptance, its tries and time, and the sampling time the other 14 cells project to."""
    cell = next((c for c in json.load(open(grid_file(ROW)))["cells"].values() if (c["numerator"], c["seed"]) == SMOKE_CELL), None)
    if cell is None: raise SystemExit("the smoke cell a15_s1 is not in the results file")
    a, minutes = acceptance([cell]), float(seconds) / 60
    per_try = minutes / cell["tried"]; per_cell = KEEP / max(a, 1e-9) * per_try
    out = dict(acceptance=a, tried=cell["tried"], kept=len(cell["pastes"]), minutes=minutes, tries_per_minute=cell["tried"] / minutes,
               projected_hours_other_cells=per_cell * (len(ASKS) * 3 - 1) / 60, reasons=cell["reasons"][:5], stop=a < STOP_ACCEPTANCE)
    json.dump(out, open(MRR.RES / "plain_reject_smoke.json", "w"), indent=1)
    lo, hi = PREDICTION["acceptance"]
    print(f"**Smoke cell, {datetime.date.today()}: Showdown accepted {a:.2%} of tries** at ask 15/24, seed 1 "
          f"({len(cell['pastes'])} teams kept from {cell['tried']:,} tries in {minutes:.0f} min, {out['tries_per_minute']:.0f} tries a minute). "
          f"The prediction was {lo:.1%} to {hi:.1%}. At this rate the other 14 cells take about {out['projected_hours_other_cells']:.1f} h of sampling.")
    print()
    print("Most common rejections: " + "; ".join(f"{r} ({n})" for r, n in cell["reasons"][:5]) + ".")
    print()
    print(f"Acceptance is under the ticket's stop of {STOP_ACCEPTANCE:.1%}, so the grid stops here." if out["stop"]
          else f"Acceptance clears the ticket's stop of {STOP_ACCEPTANCE:.1%}, so the full grid of 15 cells runs next.")
    if out["stop"]: raise SystemExit(3)


def opponents(): MRR.opponents(ROW, GUIDANCE, ASKS)


def read_row(row, counts):
    """Point estimates and [DRAWS] bootstrap draws of one row at asks 13/24 to 17/24."""
    out = MRR.load_grid(row, GUIDANCE, ASKS)
    if out is None: raise SystemExit(f"{row}: the grid is missing or has cells without battles at asks 13/24 to 17/24")
    data = np.load(MRR.opponents_path(row, GUIDANCE)); weights = data["weights"]
    rng = np.random.default_rng([MRR.DRAW_SEED, 358, STREAM[row]])
    point, draws, best, best_draws, cells = {}, {}, {}, {}, {}
    for k in ASKS:
        for seed in MRR.SEEDS:
            cell = out["by_ask_seed"][(k, seed)]; rate = data[f"a{k:02d}_s{seed}"]
            s = MRR.team_scores(rate, weights)
            if len(s) != len(cell["y"]) or abs(s.mean() - np.mean(cell["y"])) > 1e-9: raise SystemExit(f"{row} a{k:02d}_s{seed}: snapshot and stored scores differ")
            seen = ~np.isnan(rate); w = (counts * weights).T
            sd = (np.where(seen, rate, 0.0) @ w) / np.maximum(seen @ w, 1e-12)           # [teams, DRAWS]
            pick = rng.integers(0, len(rate), (len(counts), len(rate))); resampled = sd[pick, np.arange(len(counts))[:, None]]
            point[(k, seed)], draws[(k, seed)] = float(s.mean()), resampled.mean(1)
            best[(k, seed)], best_draws[(k, seed)] = float(s.max()), resampled.max(1)
            cells[(k, seed)] = cell
    got = {k: float(np.mean([point[(k, s)] for s in MRR.SEEDS])) for k in ASKS}            # every cell holds 64 teams
    got_d = {k: np.mean([draws[(k, s)] for s in MRR.SEEDS], 0) for k in ASKS}
    err = lambda g: float(np.mean([(g[k] - k / 24) ** 2 for k in ASKS]))
    err_d = lambda g: np.mean([(g[k] - k / 24) ** 2 for k in ASKS], 0)
    lineups = [len({tuple(sorted(sp for sp, _ in team)) for team in c["stat_points"]}) for c in cells.values()]
    reasons = Counter()
    for c in cells.values(): reasons.update(dict(c["reasons"]))
    return dict(got_by_ask=got, got_by_ask_draws=got_d, mean_win=float(np.mean(list(got.values()))), mean_win_draws=np.mean(list(got_d.values()), 0),
                best_of_64=float(np.mean(list(best.values()))), best_of_64_draws=np.mean(list(best_draws.values()), 0),
                error=err(got), error_draws=err_d(got_d),
                error_by_seed={s: err({k: point[(k, s)] for k in ASKS}) for s in MRR.SEEDS},
                error_by_seed_draws={s: err_d({k: draws[(k, s)] for k in ASKS}) for s in MRR.SEEDS},
                acceptance=acceptance(list(cells.values())), acceptance_by_ask={k: acceptance([cells[(k, s)] for s in MRR.SEEDS]) for k in ASKS},
                tries=sum(c["tried"] for c in cells.values()), teams=sum(len(c["pastes"]) for c in cells.values()),
                battles=sum(c.get("battles", 0) for c in cells.values()), cells=len(cells),
                distinct_line_ups_per_cell=float(np.mean(lineups)),
                distinct_line_ups=len({tuple(sorted(sp for sp, _ in team)) for c in cells.values() for team in c["stat_points"]}),
                rejections=reasons.most_common(8), slot_order=out.get("slot_order", "shuffled"))


def paired_error(a, b):
    diff = np.mean([a["error_by_seed"][s] - b["error_by_seed"][s] for s in MRR.SEEDS])
    d = np.mean([a["error_by_seed_draws"][s] - b["error_by_seed_draws"][s] for s in MRR.SEEDS], 0)
    return dict(difference=float(diff), interval=MRR.interval(d), by_seed={str(s): a["error_by_seed"][s] - b["error_by_seed"][s] for s in MRR.SEEDS})


def difference(a, b, key):
    return dict(difference=a[key] - b[key], interval=MRR.interval(a[f"{key}_draws"] - b[f"{key}_draws"]))


def read():
    counts = MRR.opponent_draws(len(np.load(MRR.opponents_path(FIRST, GUIDANCE))["weights"]))
    rows = {row: read_row(row, counts) for row in (ROW, FIRST, RANDOM)}
    new, first, rand = rows[ROW], rows[FIRST], rows[RANDOM]
    gap = difference(new, first, "mean_win")
    result = dict(guidance=GUIDANCE, asks=ASKS, verdict=verdict(gap["difference"], gap["interval"]),
                  rule=f"confirmed: mean-win-rate gap to {READER[FIRST]} no worse than {CONFIRM}; cleared: {CLEAR} or worse with the interval's upper end below {CONFIRM}; otherwise partial",
                  mean_win_against_species_first=gap, mean_win_against_random=difference(new, rand, "mean_win"),
                  random_order_against_species_first=difference(rand, first, "mean_win"),
                  best_of_64_against_species_first=difference(new, first, "best_of_64"), best_of_64_against_random=difference(new, rand, "best_of_64"),
                  error_against_species_first=paired_error(new, first), error_against_random=paired_error(new, rand),
                  share_of_random_order_gap_won_back=(new["mean_win"] - rand["mean_win"]) / (first["mean_win"] - rand["mean_win"]),
                  prediction=PREDICTION, win_rate_definition=WIN_RATE, draws=MRR.DRAWS, draw_seed=MRR.DRAW_SEED,
                  resampled="teams within each ask and sampling seed, separately in each row; the 49 opponents once for all three rows", rows={})
    for row, m in rows.items():
        result["rows"][row] = dict(reader=READER[row], **{k: v for k, v in m.items() if not k.endswith("_draws") and k not in ("got_by_ask", "error_by_seed", "acceptance_by_ask")},
                                   mean_win_interval=MRR.interval(m["mean_win_draws"]), best_of_64_interval=MRR.interval(m["best_of_64_draws"]),
                                   error_interval=MRR.interval(m["error_draws"]),
                                   got_by_ask={str(k): v for k, v in m["got_by_ask"].items()},
                                   got_by_ask_interval={str(k): MRR.interval(v) for k, v in m["got_by_ask_draws"].items()},
                                   error_by_seed={str(s): v for s, v in m["error_by_seed"].items()},
                                   acceptance_by_ask={str(k): v for k, v in m["acceptance_by_ask"].items()})
    json.dump(result, open(RESULT, "w"), indent=1)
    for m in result["rows"].values():
        print(f"{m['reader']:48s} mean win {m['mean_win']:.3f} {m['mean_win_interval']}  best of 64 {m['best_of_64']:.3f}  error 13-17 {m['error']:.4f}  "
              f"accepted {m['acceptance']:.2%}  line-ups/cell {m['distinct_line_ups_per_cell']:.1f}")
    print("gap to species-first:", gap, " won back:", f"{result['share_of_random_order_gap_won_back']:.0%}")
    print("VERDICT", result["verdict"]); print("PLAIN_REJECT_READ_DONE")


ROW_COLOR = {ROW: "#8a5cd6", FIRST: "#2a78d6", RANDOM: "#e0702a"}
TITLE = {"confirmed": "Confirmed: without the lookahead's tilt, random order wins as often as species-first",
         "cleared": "Cleared: random order stays weaker than species-first even without the lookahead's tilt",
         "partial": "Partial: dropping the lookahead's tilt wins back part of random order's gap to species-first"}


def headline(r=None):
    r = r or json.load(open(RESULT)); g = r["mean_win_against_species_first"]; m = r["rows"]
    return (f"Verdict {r['verdict']}: random order without the lookahead, redrawn until legal, wins {m[ROW]['mean_win']:.3f} at asks 13/24 to 17/24, "
            f"{g['difference']:+.3f} [{g['interval'][0]:+.3f}, {g['interval'][1]:+.3f}] against Linear, species-first's {m[FIRST]['mean_win']:.3f}, "
            f"and wins back {r['share_of_random_order_gap_won_back']:.0%} of Linear, random order's gap")


def gap_figure(folder, r):
    """Each random-order row's mean-win-rate gap to Linear, species-first, with its interval, over the ticket's three zones."""
    from echarts_chart import write_chart
    rows = [(ROW, r["mean_win_against_species_first"]), (RANDOM, r["random_order_against_species_first"])]
    names = [READER[row] for row, _ in rows][::-1]
    lo = min(-0.2, min(g["interval"][0] for _, g in rows) - 0.02)
    zones = [dict(name="Confirmed: no worse than -0.03", x0=CONFIRM, x1=0.05, color="rgba(42,120,214,0.10)"),
             dict(name="Partial", x0=CLEAR, x1=CONFIRM, color="rgba(160,160,150,0.12)"),
             dict(name="Cleared: -0.05 or worse", x0=lo, x1=CLEAR, color="rgba(224,112,42,0.10)")]
    series = [dict(type="custom", name="95% interval", renderItem="js:(p, api) => { const y = api.coord([0, api.value(2)])[1], a = api.coord([api.value(0), 0])[0], "
                   "b = api.coord([api.value(1), 0])[0]; return {type: 'line', shape: {x1: a, y1: y, x2: b, y2: y}, style: {stroke: api.visual('color'), lineWidth: 3}}; }",
                   color="#6f6e69", data=[[g["interval"][0], g["interval"][1], names.index(READER[row])] for row, g in rows], z=2, tooltip=dict(show=False)),
              dict(type="scatter", name="Gap", symbolSize=14, z=3, emphasis=dict(focus="series"),
                   data=[dict(value=[g["difference"], names.index(READER[row])], itemStyle=dict(color=ROW_COLOR[row])) for row, g in rows],
                   label=dict(show=True, position="top", formatter="js:p => (p.value[0] >= 0 ? '+' : '') + p.value[0].toFixed(3)", color="#52514e"),
                   markArea=dict(silent=True, data=[[dict(xAxis=z["x0"], itemStyle=dict(color=z["color"]), label=dict(show=True, position="insideTop", formatter=z["name"], color="#52514e", fontSize=11)),
                                                    dict(xAxis=z["x1"])] for z in zones]),
                   markLine=dict(silent=True, symbol="none", lineStyle=dict(color="#6f6e69", type="dashed"), label=dict(show=False), data=[dict(xAxis=0)]))]
    option = dict(grid=dict(left=260, right=32, top=40, bottom=56), tooltip=dict(trigger="item", formatter="js:p => p.value ? names[p.value[1]] + ': ' + p.value[0].toFixed(3) : ''"),
                  xAxis=dict(type="value", min=round(lo, 2), max=0.05, name="Mean win rate minus Linear, species-first's, asks 13/24 to 17/24",
                             nameLocation="middle", nameGap=34),
                  yAxis=dict(type="category", data=names, axisTick=dict(show=False)), series=series, animation=False)
    option["tooltip"]["formatter"] = option["tooltip"]["formatter"].replace("names[", f"{json.dumps(names)}[")
    return write_chart(option, Path(folder) / "plain-reject-gap.html", height=300, title=TITLE[r["verdict"]],
                       subtitle=f"Gap in mean win rate to Linear, species-first ({r['rows'][FIRST]['mean_win']:.3f}), linear checkpoint, guidance 2. "
                                "The lookahead is the rule that refuses any value leaving a Pokémon whose species is still hidden with no legal species.",
                       note="Line: 95% bootstrap interval over teams and the 49 opponents. Mean win rate: " + WIN_RATE + "; 192 teams per ask over 3 sampling seeds.")


def got_figure(folder, r):
    """Got against ask at asks 13/24 to 17/24 for the three rows, each with its 95% band, beside got = ask."""
    from echarts_chart import band, write_chart
    xs = [k / 24 for k in ASKS]; series = []
    for i, (row, m) in enumerate(r["rows"].items()):
        series += band(m["reader"], ROW_COLOR[row], xs, [m["got_by_ask_interval"][str(k)][0] for k in ASKS], [m["got_by_ask_interval"][str(k)][1] for k in ASKS], stack=f"band{i}")
        series.append(dict(type="line", name=m["reader"], color=ROW_COLOR[row], symbolSize=7, lineStyle=dict(width=2.6 if row == ROW else 1.8),
                           emphasis=dict(focus="series"), data=[[x, m["got_by_ask"][str(k)]] for x, k in zip(xs, ASKS)]))
    series.append(dict(type="line", name="Got = ask", color="#6f6e69", symbol="none", silent=True, lineStyle=dict(type="dashed", width=1), data=[[xs[0], xs[0]], [xs[-1], xs[-1]]]))
    pct = "js:v => Math.round(v * 100) + '%'"
    option = dict(grid=dict(left=64, right=24, top=56, bottom=56), legend=dict(top=0, data=[m["reader"] for m in r["rows"].values()] + ["Got = ask"]),
                  tooltip=dict(trigger="axis", valueFormatter="js:v => v == null ? '' : (v * 100).toFixed(1) + '%'"),
                  xAxis=dict(type="value", min=12 / 24, max=18 / 24, interval=1 / 24, name="Ask: the win rate the model is asked for", nameLocation="middle", nameGap=34,
                             axisLabel=dict(formatter="js:v => Math.round(v * 24) + '/24'")),
                  yAxis=dict(type="value", min=0.2, max=0.8, interval=0.1, name="Got: measured win rate", nameLocation="middle", nameGap=46, axisLabel=dict(formatter=pct)),
                  series=series, animation=False)
    m = r["rows"]
    return write_chart(option, Path(folder) / "plain-reject-got.html", height=440,
                       title=f"Ask-versus-got error at asks 13/24 to 17/24: {m[ROW]['error']:.4f} without the lookahead, {m[FIRST]['error']:.4f} species-first, {m[RANDOM]['error']:.4f} with the lookahead",
                       subtitle="Got at each ask for the three rows, linear checkpoint, guidance 2. Error: the mean squared gap between ask and got over these five asks.",
                       note="Band: 95% bootstrap interval over teams and the 49 opponents. Got: " + WIN_RATE + "; 192 teams per ask over 3 sampling seeds.")


def figure(folder):
    r = json.load(open(RESULT)); Path(folder).mkdir(parents=True, exist_ok=True)
    print(gap_figure(folder, r)); print(got_figure(folder, r))


SPECIES_THEN_RANDOM = "sched_linear_species_then_random"
TWISTER_ESTIMATE = (0.84, 0.89)       # #357: reweighting stored random-order teams to TWISTER's target, guidance 2, both checkpoints


def share_figure(folder):
    """The masking-schedule map's frame for #358: how much of Linear, random order's win-rate gap to species-first each fix wins back.

    Storyboard. Sentence: turning the lookahead off wins back half of random order's gap; drawing species first wins back
    nearly all of it. Expected: the exact draw lands near #357's reweighting estimate, 84% to 89%. Surprise: it lands at half.
    Share = (row - random order) / (species-first - random order); the interval comes from paired draws that resample
    teams within each row and the 49 opponents once for all four rows."""
    from echarts_chart import write_chart
    READER.setdefault(SPECIES_THEN_RANDOM, "Linear, species then random order"); STREAM.setdefault(SPECIES_THEN_RANDOM, 3)
    counts = MRR.opponent_draws(len(np.load(MRR.opponents_path(FIRST, GUIDANCE))["weights"]))
    rows = {row: read_row(row, counts) for row in (ROW, SPECIES_THEN_RANDOM, FIRST, RANDOM)}
    first, rand = rows[FIRST], rows[RANDOM]
    share = lambda m: (m["mean_win"] - rand["mean_win"]) / (first["mean_win"] - rand["mean_win"])
    share_draws = lambda m: (m["mean_win_draws"] - rand["mean_win_draws"]) / (first["mean_win_draws"] - rand["mean_win_draws"])
    names = ["Reweighting stored teams to the same target (#357)", "Lookahead off, redrawn until legal (#358)", "Species first, then random order (#322)"]
    measured = [(names[2], SPECIES_THEN_RANDOM, "#1baf7a"), (names[1], ROW, ROW_COLOR[ROW])]
    points = {name: (share(rows[row]), MRR.interval(share_draws(rows[row])), color) for name, row, color in measured}
    line = "js:(p, api) => { const y = api.coord([0, api.value(2)])[1], a = api.coord([api.value(0), 0])[0], b = api.coord([api.value(1), 0])[0]; " \
           "return {type: 'line', shape: {x1: a, y1: y, x2: b, y2: y}, style: {stroke: api.value(3), lineWidth: api.value(4)}}; }"
    pct = "js:v => Math.round(v * 100) + '%'"
    series = [dict(type="custom", name="Estimate, not battled", renderItem=line, z=2, tooltip=dict(show=False),
                   data=[[TWISTER_ESTIMATE[0], TWISTER_ESTIMATE[1], names.index(names[0]), "rgba(138,92,214,0.45)", 14]]),
              dict(type="custom", name="95% interval", renderItem=line, z=2, tooltip=dict(show=False),
                   data=[[iv[0], iv[1], names.index(n), "#8f8d86", 3] for n, (_, iv, _) in points.items()]),
              dict(type="scatter", name="Battled", symbolSize=15, z=3, emphasis=dict(focus="series"),
                   data=[dict(value=[s, names.index(n)], itemStyle=dict(color=c)) for n, (s, _, c) in points.items()],
                   label=dict(show=True, position="top", distance=8, formatter="js:p => Math.round(p.value[0] * 100) + '%'", fontSize=13, fontWeight="bold")),
              dict(type="scatter", name="Estimate", symbolSize=0, data=[dict(value=[sum(TWISTER_ESTIMATE) / 2, names.index(names[0])])], tooltip=dict(show=False),
                   label=dict(show=True, position="top", distance=12, formatter=f"{TWISTER_ESTIMATE[0]:.0%} to {TWISTER_ESTIMATE[1]:.0%}, estimate", fontSize=12)),
              dict(type="line", name="ends", data=[], markLine=dict(silent=True, symbol="none", data=[
                  dict(xAxis=0, lineStyle=dict(color=ROW_COLOR[RANDOM], type="dashed", width=1.5),
                       label=dict(formatter=f"Random order, lookahead on\n{rand['mean_win']:.3f}", position="start", color=ROW_COLOR[RANDOM])),
                  dict(xAxis=1, lineStyle=dict(color=ROW_COLOR[FIRST], type="dashed", width=1.5),
                       label=dict(formatter=f"Species-first\n{first['mean_win']:.3f}", position="start", color=ROW_COLOR[FIRST]))]))]
    option = dict(grid=dict(left=310, right=48, top=72, bottom=56),
                  tooltip=dict(trigger="item", formatter="js:p => p.value ? Math.round(p.value[0] * 100) + '% of the gap' : ''"),
                  xAxis=dict(type="value", min=0, max=1.1, interval=0.1, name="Share of random order's win-rate gap to species-first won back, asks 13/24 to 17/24",
                             nameLocation="middle", nameGap=34, axisLabel=dict(formatter=pct), splitLine=dict(show=False)),
                  yAxis=dict(type="category", data=names, axisTick=dict(show=False), inverse=True), series=series, animation=False)
    s = points[names[1]]
    print({n: (round(v[0], 3), [round(x, 3) for x in v[1]]) for n, v in points.items()})
    return write_chart(option, Path(folder) / "plain-reject-share.html", height=350,
                       title=f"Turning the lookahead off wins back {s[0]:.0%} of random order's gap; drawing species first wins back {points[names[2]][0]:.0%}",
                       subtitle="Linear checkpoint, guidance 2. 0% is random order with the lookahead (mean win rate "
                                f"{rand['mean_win']:.3f}), 100% is species-first ({first['mean_win']:.3f}). The reweighting estimate predicted most of the gap; battled, the exact draw got half.",
                       note="Line: 95% bootstrap interval, teams resampled in each row, the 49 opponents once for all four rows. Faded bar: #357's estimate from stored teams, "
                            "the range over the linear and per-field checkpoints, no new battles. Mean win rate: " + WIN_RATE + "; 960 teams per row.")


def within(x, band): return "inside" if band[0] <= x <= band[1] else "outside"


def comment(url):
    """The verdict comment, picture first. url has {name} (plain-reject-gap or plain-reject-got) and {theme} (light or dark)."""
    r = json.load(open(RESULT)); m = r["rows"]; new, first, rand = m[ROW], m[FIRST], m[RANDOM]
    iv = lambda d, p=3: f"[{d['interval'][0]:+.{p}f}, {d['interval'][1]:+.{p}f}]"
    pic = lambda name, alt: (f'<picture><source media="(prefers-color-scheme: dark)" srcset="{url.format(name=name, theme="dark")}">'
                             f'<img width="760" alt="{alt}" src="{url.format(name=name, theme="light")}"></picture>')
    P = r["prediction"]; g = r["mean_win_against_species_first"]
    lines = [f"**{datetime.date.today()}: {headline(r)}.**", "",
             pic("plain-reject-gap", TITLE[r["verdict"]]), "",
             f"Random order without the lookahead draws every field from the model's own random-order distribution; keeping only the teams Showdown accepts gives that distribution among legal teams, "
             f"the target TWISTER corrects towards, with no weights. Same linear checkpoint, guidance 2, asks 13/24 to 17/24, seeds 1 to 3 on seed base 7300.", "",
             f"- Mean win rate at asks 13/24 to 17/24: {new['mean_win']:.3f}, against {first['mean_win']:.3f} for Linear, species-first (gap {g['difference']:+.3f} {iv(g)}) "
             f"and {rand['mean_win']:.3f} for Linear, random order with the lookahead ({r['mean_win_against_random']['difference']:+.3f} {iv(r['mean_win_against_random'])}). "
             f"That wins back {r['share_of_random_order_gap_won_back']:.0%} of random order's gap.",
             f"- Best of 64 (mean over the 15 cells of each cell's best team): {new['best_of_64']:.3f}, against {rand['best_of_64']:.3f} with the lookahead "
             f"({r['best_of_64_against_random']['difference']:+.3f} {iv(r['best_of_64_against_random'])}) and {first['best_of_64']:.3f} species-first.",
             f"- Showdown accepted {new['acceptance']:.2%} of {new['tries']:,} tries.",
             f"- Rule, from the ticket: {r['rule']}.", "",
             pic("plain-reject-got", "Got against ask at asks 13/24 to 17/24 for the three rows"), "",
             f"Ask-versus-got error at asks 13/24 to 17/24 (paired by sampling seed): {new['error']:.4f}, against {first['error']:.4f} species-first "
             f"({r['error_against_species_first']['difference']:+.4f} {iv(r['error_against_species_first'], 4)}) and {rand['error']:.4f} with the lookahead "
             f"({r['error_against_random']['difference']:+.4f} {iv(r['error_against_random'], 4)}).", "",
             "**Prediction, written before the run:**", "",
             "| Measure | Predicted | Measured | |", "|---|---|---|---|",
             f"| Acceptance per try | {P['acceptance'][0]:.1%} to {P['acceptance'][1]:.1%} | {new['acceptance']:.2%} | {within(new['acceptance'], P['acceptance'])} |",
             f"| Mean win rate, asks 13/24 to 17/24 | {P['mean_win'][0]:.2f} to {P['mean_win'][1]:.2f} | {new['mean_win']:.3f} | {within(new['mean_win'], P['mean_win'])} |",
             f"| Best of 64 | {P['best_of_64'][0]:.2f} to {P['best_of_64'][1]:.2f}, no rise over random order | {new['best_of_64']:.3f} | {within(new['best_of_64'], P['best_of_64'])} |", "",
             "<details><summary>Every row, acceptance by ask, rejections</summary>", "",
             "| Row | Mean win rate | Best of 64 | Error, asks 13/24 to 17/24 | Accepted | Distinct line-ups per cell of 64 | Distinct line-ups, all 960 teams | Battles |", "|---|---|---|---|---|---|---|---|"]
    for row in m.values():
        lines.append(f"| {row['reader']} | {row['mean_win']:.3f} [{row['mean_win_interval'][0]:.3f}, {row['mean_win_interval'][1]:.3f}] | {row['best_of_64']:.3f} | "
                     f"{row['error']:.4f} | {row['acceptance']:.2%} | {row['distinct_line_ups_per_cell']:.1f} | {row['distinct_line_ups']} | {row['battles']:,} |")
    lines += ["", "Acceptance by ask: " + ", ".join(f"{k}/24 {v:.2%}" for k, v in new["acceptance_by_ask"].items()) + ".", "",
              "Most common Showdown rejections: " + "; ".join(f"{reason} ({n:,})" for reason, n in new["rejections"][:6]) + ".", "",
              "The ticket quoted Linear, random order's best of 64 as 0.587; under the definition here it reads "
              f"{rand['best_of_64']:.3f}, so the prediction is read against that.", "",
              "Interval: 95% bootstrap, teams resampled within each ask and sampling seed, the 49 opponents resampled once for all three rows. "
              "Script: `code/src/plain_reject.py read`. Results: `~/vgc-data/results/plain_reject_grid.json`.", "</details>"]
    print("\n".join(lines))


if __name__ == "__main__":
    {"smoke": lambda: smoke(sys.argv[2]), "opponents": opponents, "read": read, "figure": lambda: figure(sys.argv[2]), "share": lambda: print(share_figure(sys.argv[2])),
     "headline": lambda: print(headline()), "comment": lambda: comment(sys.argv[2])}[sys.argv[1]]()
