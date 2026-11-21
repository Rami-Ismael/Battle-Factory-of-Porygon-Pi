"""Entropy-guided initialization, then confidence order, on the 84-field model (GitHub issue #333).

Entropy-guided initialization (Jeon et al. 2026, §6.2, https://arxiv.org/abs/2609.34509): pass 1 fills the masked field
whose drawing distribution (guidance, legal Pokémon construction filter, lookahead, temperature 1) has the highest
entropy, H = -sum p log p, and draws its value at temperature 1.5; passes 2 to 84 follow confidence order
(confidence_order84.py) at temperature 1. 1.5 keeps Jeon et al.'s ratio of first-pass to base temperature, 0.9 over 0.6.

Every team at one ask starts from the same empty team, so pass 1 lands on the same field for all of them: the probe
needs one pass per ask, not the grid. The grid read checks it against every kept team's recorded fill order.

    python entropy_init84.py probe                 the field each ask's pass 1 lands on, into results/entropy_init_probe.json
    python entropy_init84.py probe_figure DIR      the probe as an ECharts page in DIR
    python entropy_init84.py probe_comment [URL]   the probe as an issue comment; exit 3 if species fields take >= 95% of teams
    python entropy_init84.py read                  the verdict against confidence order, species-first order and
                                                   random order, into results/entropy_init.json
    python entropy_init84.py figure DIR            its charts as ECharts pages in DIR
    python entropy_init84.py comment URL           the verdict comment; URL holds {name} and {theme}

The ask grid row is sched_linear_entropy_init (ask_grid84.py), decoder "entropy_init".
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import confidence_order84 as C
import diffusion84 as M
import sampler84 as S

FIRST_TEMPERATURE = S.FIRST_TEMPERATURE                    # Jeon et al.'s 0.9 / 0.6, times our base temperature 1
SPECIES_RULE = 0.95                                        # the ticket's stop: species fields for >= 95% of teams
ROW, CONFIDENCE, SPECIES_FIRST, RANDOM = "sched_linear_entropy_init", C.ROW, C.SPECIES_FIRST, C.RANDOM
READER = {ROW: "Entropy-guided initialization, then confidence order", CONFIDENCE: "Confidence order",
          SPECIES_FIRST: "Species-first order", RANDOM: "Random order"}   # every row: linear schedule, guidance 2
STREAM = {ROW: 0, CONFIDENCE: 1, SPECIES_FIRST: 2, RANDOM: 3}   # each row's own bootstrap stream, fixed so a new row moves no other
COLOR = {ROW: "#1f9e74", CONFIDENCE: "#8a5cd6", SPECIES_FIRST: "#2a78d6", RANDOM: "#e0702a"}
SPECIES_FIRST_REDRAWN = "sched_linear_species_first_seed940135810"   # Linear, species-first on seeds of its own (#353): novelty reference
ISSUE_URL = "https://github.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/issues/333"
GUIDANCE = 2.0


def arm(w):
    """The ask grid's arm: confidence order whose first pass picks by entropy and draws at FIRST_TEMPERATURE."""
    return S.Sampler84(w, "entropy_init", first_temperature=FIRST_TEMPERATURE)


def field_type(col):
    """species, item, ability, move, tera_type, nature, ... or stat_points, from the column index."""
    import lossdown as L
    return "stat_points" if col >= M.STAT0 else L.KEY_OF_COL[col]


def results_dir(): return C.results_dir()


def probe_path(): return results_dir() / "entropy_init_probe.json"


def out_path(): return results_dir() / "entropy_init.json"


def first_pass(w, ask, guidance=GUIDANCE, n=4):
    """Pass 1 at one ask: the column picked, its field type, and the highest entropy of each field type."""
    log = []
    _, order, _ = S.sample_confidence(w, n, ask, guidance, 1.0, True, None, None, "entropy", FIRST_TEMPERATURE, passes=1, score_log=log)
    picks = set(order[:, 0].tolist())
    if len(picks) != 1: raise RuntimeError(f"ask {ask}: teams with the same empty start picked different first fields {picks}")
    col = picks.pop(); h = log[0][0]
    by_type = {}
    for c in range(M.COLS): by_type[field_type(c)] = max(by_type.get(field_type(c), -1.0), float(h[c]))
    return dict(column=col, field_type=field_type(col), entropy=float(h[col]),
                entropy_by_field_type=dict(sorted(by_type.items(), key=lambda kv: -kv[1])))


def probe():
    """The field pass 1 lands on at each of the grid's 23 asks, and the share of the grid's teams whose pass 1 is a species."""
    import ask_grid84 as A
    w = M.Model84.load(A.L.CK / f"{A.MODELS[ROW]['checkpoint']}.pt")
    asks = {}
    for k in range(1, 24):
        torch.manual_seed(k); asks[k] = first_pass(w, k / 24)
        print(f"  ask {k:2d}/24: pass 1 fills column {asks[k]['column']:2d} ({asks[k]['field_type']}), entropy {asks[k]['entropy']:.3f} nats", flush=True)
    types = [a["field_type"] for a in asks.values()]
    out = dict(row=ROW, checkpoint=A.MODELS[ROW]["checkpoint"], guidance=GUIDANCE, first_temperature=FIRST_TEMPERATURE, asks=asks,
               teams_per_ask=A.KEEP * 3, share_by_field_type={t: types.count(t) / len(types) for t in sorted(set(types))},
               species_share=types.count("species") / len(types), rule=SPECIES_RULE)
    out["stop_for_owner"] = out["species_share"] >= SPECIES_RULE
    json.dump(out, open(probe_path(), "w"), indent=1)
    print(f"species share {out['species_share']:.1%}; stop for the owner: {out['stop_for_owner']}"); print("ENTROPY_PROBE_DONE")


PROBE_COLORS = {"species": "#2a78d6", "move": "#1f9e74", "stat_points": "#e0702a", "ability": "#8a5cd6", "item": "#c2417a"}


def probe_figure(folder):
    """One page: at each ask, the highest pass-1 entropy of each field type; the top line at an ask is the field pass 1 fills."""
    from echarts_chart import write_chart
    r = json.load(open(probe_path())); asks = sorted(r["asks"], key=int)
    series = [dict(type="line", name=t.replace("_", " ").capitalize(), color=c, symbolSize=6, emphasis=dict(focus="series"),
                   data=[[int(k) / 24, r["asks"][k]["entropy_by_field_type"].get(t)] for k in asks]) for t, c in PROBE_COLORS.items()]
    option = dict(grid=dict(left=64, right=24, top=56, bottom=56), legend=dict(top=0),
                  tooltip=dict(trigger="axis", valueFormatter="js:v => v == null ? '' : v.toFixed(2) + ' nats'"),
                  xAxis=dict(type="value", min=0, max=1, interval=1 / 6, name="Ask", nameLocation="middle", nameGap=34,
                             axisLabel=dict(formatter="js:v => Math.round(v * 24) + '/24'")),
                  yAxis=dict(type="value", min=2, name="Highest entropy of the field type (nats)", nameLocation="middle", nameGap=46),
                  series=series, animation=False)
    share = r["share_by_field_type"]
    write_chart(option, Path(folder) / "entropy-probe.html", height=440,
                title=f"Pass 1 starts at a move for {share.get('move', 0):.0%} of teams, Stat Points {share.get('stat_points', 0):.0%}, a species {share.get('species', 0):.0%}",
                subtitle="At each ask the top line is the field type pass 1 fills: species at asks 1–3 and 23, a move at 4–14 and 22, Stat Points at 15–21.",
                note="Entropy of the drawing distribution on the empty team: guidance 2, legal Pokémon construction filter, lookahead, temperature 1. "
                     "Every team at one ask starts from the same empty team, so all 192 teams of an ask start at the same field.")


def probe_comment(figure_url=None):
    """The probe, posted before any battle, picture first when figure_url ({theme} for light or dark) is given. Exit 3
    when the 95% rule stops the grid for the owner."""
    import datetime
    r = json.load(open(probe_path()))
    asks = r["asks"]; types = sorted(r["share_by_field_type"], key=lambda t: -r["share_by_field_type"][t])
    lead = ", ".join(f"{r['share_by_field_type'][t]:.0%} {t.replace('_', ' ')}" for t in types)
    head = (f"**Probe, {datetime.date.today()}: pass 1 lands on a species field for {r['species_share']:.0%} of the grid's "
            f"{r['teams_per_ask'] * len(asks):,} teams ({lead}).**")
    if r["stop_for_owner"]:
        verdict = ("That is at least 95%, so this arm would differ from confidence order only in the first species's "
                   "temperature. No battle runs until the owner decides.")
    else:
        verdict = "That is under 95%, so the ticket's stop rule does not fire."
    if figure_url:
        head += ("\n\n" f'<picture><source media="(prefers-color-scheme: dark)" srcset="{figure_url.format(theme="dark")}">'
                 f'<img alt="Highest pass-1 entropy of each field type at each ask" src="{figure_url.format(theme="light")}"></picture>')
    rows = ["| Ask | Pass 1 fills | Its entropy (nats) | Highest entropy by field type |", "|---|---|---|---|"]
    for k, a in sorted(asks.items(), key=lambda kv: int(kv[0])):
        top = ", ".join(f"{t.replace('_', ' ')} {h:.2f}" for t, h in list(a["entropy_by_field_type"].items())[:4])
        rows.append(f"| {k}/24 | {a['field_type'].replace('_', ' ')} (column {a['column']}) | {a['entropy']:.3f} | {top} |")
    lines = [head, "", verdict, "",
             "Every team at one ask starts from the same empty team, so the drawing distributions of pass 1, and the field "
             "with the highest entropy, are the same for all 192 teams of that ask; one pass per ask gives the share for all "
             "4,416. The grid read checks this against each kept team's recorded fill order.", "",
             "<details><summary>Pass 1 at each ask</summary>", "", *rows, "",
             f"Entropy H = −Σ p log p of the drawing distribution: guidance 2, legal Pokémon construction filter, lookahead, "
             f"temperature 1. The chosen field's value is then drawn at temperature {FIRST_TEMPERATURE}. Checkpoint `{r['checkpoint']}`. "
             "Script: `code/src/entropy_init84.py probe`.", "</details>"]
    print("\n".join(lines))
    return 3 if r["stop_for_owner"] else 0


# ---------------------------------------------------------------- the fill-order diagnostic
def edit_distances(orders):
    """Levenshtein distance between every pair of fill orders (rows of an [n, L] int array), as a flat array of n(n-1)/2."""
    orders = np.asarray(orders); n, L = orders.shape
    i, j = np.triu_indices(n, 1); a, b = orders[i], orders[j]
    prev = np.tile(np.arange(L + 1), (len(i), 1))
    for r in range(1, L + 1):
        cur = np.empty_like(prev); cur[:, 0] = r
        sub = prev[:, :-1] + (a[:, r - 1:r] != b)
        best = np.minimum(prev[:, 1:] + 1, sub)
        for c in range(1, L + 1): cur[:, c] = np.minimum(best[:, c - 1], cur[:, c - 1] + 1)
        prev = cur
    return prev[:, L]


def fill_order_edit_distance(cells):
    """Mean pairwise normalized edit distance between the fill orders of the teams at one ask (Jeon et al., App. C.4):
    Levenshtein distance over 84, pooled over the ask's sampling seeds; by ask and the mean over asks."""
    by_ask = {}
    for c in cells: by_ask.setdefault(c["numerator"], []).extend(r["fill_order"] for r in c["refusals"])
    per = {k: float(edit_distances(v).mean() / M.COLS) for k, v in sorted(by_ask.items()) if len(v) > 1}
    return dict(mean=float(np.mean(list(per.values()))), by_ask={str(k): v for k, v in per.items()})


def first_field_types(cells):
    """Share of kept teams whose pass 1 filled each field type, from the recorded fill orders."""
    types = [field_type(r["fill_order"][0]) for c in cells for r in c["refusals"]]
    return {t: types.count(t) / len(types) for t in sorted(set(types))}


# ---------------------------------------------------------------- the verdict, read after the grid
def prediction(new, conf, k=C.K_PREDICTED):
    """The ticket's prediction, part by part. Pass@48 above: the new row's point lies above the upper end of confidence
    order's interval. Win rate within: the new row's mean win rate lies inside confidence order's interval."""
    i = k - 1
    parts = dict(pass_at_48_above_confidence_order=new["pass_at_k"]["pass_at_k"][i] > conf["pass_at_k"]["pass_at_k_interval"][i][1],
                 win_rate_within_confidence_order_interval=conf["pooled_win_interval"][0] <= new["pooled_win"] <= conf["pooled_win_interval"][1])
    held = sum(parts.values())
    return dict(parts=parts, pass_at_48_point_above=new["pass_at_k"]["pass_at_k"][i] > conf["pass_at_k"]["pass_at_k"][i],
                verdict="confirmed" if held == 2 else "not confirmed" if held == 0 else "partly confirmed")


LINE_UP_MARGIN, WIN_MARGIN = 100, 0.007                    # owner, 2026-11-17: the primary test (docs/entropy-first-then-confidence.md)


def frontier_points():
    """(row, guidance) -> {line_ups, win} from the diversity front of #263 (diversity_novelty_frontier.py)."""
    f = json.load(open(results_dir() / "diversity_novelty_frontier.json"))
    return {(p["row"], int(p["guidance"])): dict(line_ups=p["line_ups"], win=p["win"], arm=p["arm"]) for p in f["points"]}


def confidence_curve_win(curve, line_ups):
    """Confidence order's win rate at this many line-ups: straight lines between its guidance points (ordered by
    line-ups); the end point's win rate beyond either end."""
    pts = sorted((c["line_ups"], c["win"]) for c in curve)
    if line_ups <= pts[0][0]: return pts[0][1]
    if line_ups >= pts[-1][0]: return pts[-1][1]
    return float(np.interp(line_ups, [x for x, _ in pts], [y for _, y in pts]))


def pareto(new, curve):
    """The primary test. new and each curve point: {line_ups, win}. Outside: at least LINE_UP_MARGIN more line-ups than
    the curve's widest point at a win rate no more than WIN_MARGIN below it, or a win rate more than WIN_MARGIN above
    the curve at the same line-up count. Inside: some curve point has at least LINE_UP_MARGIN more line-ups and at least
    WIN_MARGIN more win rate. Otherwise on the curve, within the margins."""
    widest = max(curve, key=lambda c: c["line_ups"])
    at = confidence_curve_win(curve, new["line_ups"])
    wider = new["line_ups"] >= widest["line_ups"] + LINE_UP_MARGIN and new["win"] >= widest["win"] - WIN_MARGIN
    higher = new["win"] > at + WIN_MARGIN
    inside = any(c["line_ups"] >= new["line_ups"] + LINE_UP_MARGIN and c["win"] >= new["win"] + WIN_MARGIN for c in curve)
    verdict = "outside" if wider or higher else "inside" if inside else "on the curve"
    return dict(verdict=verdict, wider=bool(wider), higher=bool(higher), dominated=bool(inside), curve_win_at_line_ups=at,
                line_up_margin=LINE_UP_MARGIN, win_margin=WIN_MARGIN, point=new, curve=curve)


def read():
    import grid_novelty as GN
    import masking_rate_rows as MRR
    import species_late as SL
    MRR.use_pilot_src()
    counts = MRR.opponent_draws(len(np.load(MRR.opponents_path(SPECIES_FIRST, 2))["weights"]))
    rows = {row: SL.read_row(row, counts, 2, stream=[MRR.DRAW_SEED, 333, STREAM[row]]) for row in READER}
    summary = json.load(open(MRR.SUMMARY))["grids"] if MRR.SUMMARY.exists() else {}
    refs = MRR.references(); train = GN.training_line_ups()
    stored_pass = json.load(open(MRR.RES / "pass_at_k_issue331.json"))["grids"]
    win = {row: dict(point=float(np.mean(list(m["got_by_ask"].values()))), draws=np.mean(list(m["got_by_ask_draws"].values()), 0)) for row, m in rows.items()}
    cells = {row: list(json.load(open(MRR.grid_path(row, 2)))["cells"].values()) for row in READER}
    result = dict(question="Does starting each team at its most uncertain field, then confidence order, beat confidence order on Pass@48?",
                  prediction_written="Pass@k at k = 48 above confidence order's; mean win rate within confidence order's interval",
                  guidance=2, first_temperature=FIRST_TEMPERATURE,
                  resampled="teams within each ask and sampling seed, separately in each row; the 49 opponents once for all four rows",
                  draws=MRR.DRAWS, draw_seed=MRR.DRAW_SEED, rows={}, against={})
    for row, m in rows.items():
        s = MRR.measures(MRR.load_grid(row, 2), refs) if row in (ROW, CONFIDENCE) or row not in summary else summary[row]
        novelty_cells = cells[row] if row != SPECIES_FIRST else list(json.load(open(MRR.RES / f"stat_points_84_{SPECIES_FIRST_REDRAWN}.json"))["cells"].values())
        result["rows"][row] = dict(reader=READER[row], error=m["error"], error_interval=MRR.interval(m["error_draws"]),
                                   error_by_seed={str(k): v for k, v in m["error_by_seed"].items()},
                                   pooled_win=win[row]["point"], pooled_win_interval=MRR.interval(win[row]["draws"]),
                                   got_by_ask={str(k): v for k, v in m["got_by_ask"].items()},
                                   got_by_ask_interval={str(k): MRR.interval(v) for k, v in m["got_by_ask_draws"].items()},
                                   first_try_legality=s["first_try_legality"], line_ups=s["line_ups"], line_ups_by_seed=s["line_ups_by_seed"],
                                   effective_species=s["effective_species"], elite_line_up_yield=MRR.elite_yield(row),
                                   novelty=GN.from_cells(novelty_cells, train),
                                   novelty_source="redrawn on seeds of its own (#353)" if row == SPECIES_FIRST else "this grid",
                                   pass_at_k=C.pass_at_k_of(row, stored_pass))
    for other in (CONFIDENCE, SPECIES_FIRST, RANDOM):
        result["against"][other] = dict(error=SL.paired(rows[ROW], rows[other]),
                                        pooled_win=dict(difference=win[ROW]["point"] - win[other]["point"], interval=MRR.interval(win[ROW]["draws"] - win[other]["draws"])))
    result["non_top_draw_share"] = {row: float(np.mean([v for vs in C.team_non_top(cells[row]).values() for v in vs])) for row in (ROW, CONFIDENCE)}
    result["fill_order"] = {row: C.fill_order_measures(cells[row]) for row in (ROW, CONFIDENCE)}
    result["edit_distance"] = {row: fill_order_edit_distance(cells[row]) for row in (ROW, CONFIDENCE)}
    result["first_field_types"] = {row: first_field_types(cells[row]) for row in (ROW, CONFIDENCE)}
    new, conf = result["rows"][ROW], result["rows"][CONFIDENCE]
    result["prediction"] = prediction(new, conf) if new["pass_at_k"] and conf["pass_at_k"] else None
    points = frontier_points()
    curve = [dict(guidance=g, line_ups=points[(CONFIDENCE, g)]["line_ups"], win=points[(CONFIDENCE, g)]["win"]) for g in (2, 4, 8)]
    result["pareto"] = pareto(dict(line_ups=new["line_ups"], win=new["pooled_win"]), curve)
    result["front_points"] = [dict(row=r_, guidance=g, **v) for (r_, g), v in sorted(points.items())]
    json.dump(result, open(out_path(), "w"), indent=1)
    for row, m in result["rows"].items():
        print(f"{m['reader']:45s} error {m['error']:.4f} win {m['pooled_win']:.3f} legal {m['first_try_legality']:.1%} line-ups {m['line_ups']} "
              f"novel {m['novelty'].get('novel_share', float('nan')):.1%} Pass@48 {m['pass_at_k']['pass_at_k'][-1] if m['pass_at_k'] else None}")
    print("against confidence order:", result["against"][CONFIDENCE])
    print("edit distance:", {r: round(v["mean"], 3) for r, v in result["edit_distance"].items()})
    print("pass 1 field types:", result["first_field_types"][ROW])
    print("ticket prediction:", (result["prediction"] or {}).get("verdict"))
    print("VERDICT", result["pareto"]["verdict"]); print("ENTROPY_INIT_READ_DONE")


def headline(r):
    p = r["pareto"]; n = p["point"]
    word = {"outside": "lands outside", "inside": "lands inside", "on the curve": "lands on"}[p["verdict"]]
    return (f"Entropy-guided initialization {word} confidence order's win rate and line-up curve "
            f"({n['line_ups']:,} line-ups at {n['win']:.3f}, against {p['curve_win_at_line_ups']:.3f} on the curve)")


FRONT_ROWS = {CONFIDENCE: "Confidence order", SPECIES_FIRST: "Species-first order", RANDOM: "Random order"}


def pareto_figure(folder, r):
    """Win rate against distinct line-ups: each order's guidance curve (labelled by guidance) and the new point."""
    from echarts_chart import write_chart
    series = []
    for row, name in FRONT_ROWS.items():
        pts = sorted((p for p in r["front_points"] if p["row"] == row), key=lambda p: p["guidance"])
        series.append(dict(type="line", name=name, color=COLOR[row], symbolSize=8, emphasis=dict(focus="series"),
                           label=dict(show=True, position="top", formatter="js:p => 'g' + p.data[2]"),
                           data=[[p["line_ups"], p["win"], p["guidance"]] for p in pts]))
    n = r["pareto"]["point"]
    series.append(dict(type="scatter", name=READER[ROW], color=COLOR[ROW], symbolSize=18, symbol="diamond",
                       label=dict(show=True, position="right", formatter="g2, new"), data=[[n["line_ups"], n["win"], 2]]))
    option = dict(grid=dict(left=64, right=40, top=72, bottom=56), legend=dict(top=0),
                  tooltip=dict(trigger="item", formatter="js:p => p.seriesName + ', guidance ' + p.data[2] + ': ' + p.data[0].toLocaleString() + ' line-ups, win ' + (p.data[1] * 100).toFixed(1) + '%'"),
                  xAxis=dict(type="value", scale=True, name="Distinct line-ups of 4,416 teams (more is more diverse)", nameLocation="middle", nameGap=34),
                  yAxis=dict(type="value", scale=True, name="Mean win rate, 23 asks", nameLocation="middle", nameGap=50,
                             axisLabel=dict(formatter="js:v => (v * 100).toFixed(0) + '%'")),
                  series=series, animation=False)
    write_chart(option, Path(folder) / "entropy-pareto.html", height=480, title=headline(r),
                subtitle=f"Outside means at least {LINE_UP_MARGIN} more line-ups than confidence order's widest point at a win rate within {WIN_MARGIN}, "
                         f"or a win rate more than {WIN_MARGIN} above its curve at the same line-up count. Linear schedule; g: guidance.",
                note="Win rate: one battle against each of the 49 distinct top-50 meta teams, behaviour-cloning policy on both sides, slots shuffled. "
                     "Species-first line-ups from its grids redrawn on seeds of their own (#353).")


def figure(folder):
    """Four pages: the win rate against line-ups front, got against ask for the four rows, Pass@k at asks 13/24 to
    17/24, and fill-order edit distance by ask."""
    from echarts_chart import band, write_chart
    r = json.load(open(out_path())); rows = r["rows"]; pct = "js:v => Math.round(v * 100) + '%'"
    pareto_figure(folder, r)
    asks = [int(k) for k in rows[ROW]["got_by_ask"]]; xs = [k / 24 for k in asks]; series = []
    for i, (row, m) in enumerate(rows.items()):
        series += band(m["reader"], COLOR[row], xs, [m["got_by_ask_interval"][str(k)][0] for k in asks], [m["got_by_ask_interval"][str(k)][1] for k in asks], stack=f"b{i}")
        series.append(dict(type="line", name=m["reader"], color=COLOR[row], symbolSize=6, lineStyle=dict(width=2.6 if row == ROW else 1.6),
                           emphasis=dict(focus="series"), data=[[x, m["got_by_ask"][str(k)]] for x, k in zip(xs, asks)]))
    series.append(dict(type="line", name="Got = ask", color="#6f6e69", symbol="none", silent=True, lineStyle=dict(type="dashed", width=1), data=[[0, 0], [1, 1]]))
    option = dict(grid=dict(left=64, right=24, top=72, bottom=56), legend=dict(top=0, data=[m["reader"] for m in rows.values()] + ["Got = ask"]),
                  tooltip=dict(trigger="axis", valueFormatter="js:v => v == null ? '' : (v * 100).toFixed(1) + '%'"),
                  xAxis=dict(type="value", min=0, max=1, interval=1 / 6, name="Ask: the win rate the model is asked for", nameLocation="middle", nameGap=34, axisLabel=dict(formatter=pct)),
                  yAxis=dict(type="value", min=0, max=0.8, name="Got: measured win rate", nameLocation="middle", nameGap=46, axisLabel=dict(formatter=pct)),
                  series=series, animation=False)
    gap = r["against"][CONFIDENCE]
    write_chart(option, Path(folder) / "entropy-got.html", height=480,
                title=f"Mean win rate {rows[ROW]['pooled_win']:.3f} with entropy-guided initialization against {rows[CONFIDENCE]['pooled_win']:.3f} confidence order",
                subtitle=f"Guidance 2, linear checkpoint. Ask-versus-got error {rows[ROW]['error']:.4f} with entropy-guided initialization, {rows[CONFIDENCE]['error']:.4f} confidence order "
                         f"(paired gap {gap['error']['difference']:+.4f}), {rows[SPECIES_FIRST]['error']:.4f} species-first, {rows[RANDOM]['error']:.4f} random order.",
                note="Band: 95% bootstrap interval over teams and the 49 opponents. Got: one battle against each of the 49 distinct top-50 meta teams, "
                     "behaviour-cloning policy on both sides, slots shuffled; 192 teams per ask over 3 sampling seeds.")
    ks = list(range(1, C.K_PREDICTED + 1)); series = []
    for i, (row, m) in enumerate(rows.items()):
        p = m["pass_at_k"]
        if not p: continue
        series += band(m["reader"], COLOR[row], ks, [v[0] for v in p["pass_at_k_interval"]], [v[1] for v in p["pass_at_k_interval"]], stack=f"p{i}")
        series.append(dict(type="line", name=m["reader"], color=COLOR[row], symbol="none", lineStyle=dict(width=2.6 if row == ROW else 1.6),
                           emphasis=dict(focus="series"), data=[[k, v] for k, v in zip(ks, p["pass_at_k"])]))
    option = dict(grid=dict(left=64, right=24, top=72, bottom=56), legend=dict(top=0, data=[m["reader"] for m in rows.values()]),
                  tooltip=dict(trigger="axis", valueFormatter="js:v => v == null ? '' : (v * 100).toFixed(1) + '%'"),
                  xAxis=dict(type="value", min=1, max=C.K_PREDICTED, name="k: teams drawn at one ask", nameLocation="middle", nameGap=34),
                  yAxis=dict(type="value", min=0, max=1, name="Pass@k", nameLocation="middle", nameGap=46, axisLabel=dict(formatter=pct)),
                  series=series, animation=False)
    write_chart(option, Path(folder) / "entropy-pass-at-k.html", height=440, title=headline(r),
                subtitle="Pass@k: the chance that at least one of k teams at asks 13/24 to 17/24 has an elite line-up that held. Elite: win rate at least 0.531 "
                         "against the top-50 meta teams, and the line-up's best team still reaches 0.531 in a re-battle of 4 games per opponent.",
                note="Band: 95% bootstrap interval over sampling seeds, teams and the 49 opponents (pass_at_k.py).")
    ed = r["edit_distance"]; series = []
    for row in (ROW, CONFIDENCE):
        series.append(dict(type="line", name=READER[row], color=COLOR[row], symbolSize=6, emphasis=dict(focus="series"),
                           data=[[int(k) / 24, v] for k, v in ed[row]["by_ask"].items()]))
    option = dict(grid=dict(left=64, right=24, top=56, bottom=56), legend=dict(top=0),
                  tooltip=dict(trigger="axis", valueFormatter="js:v => v == null ? '' : v.toFixed(3)"),
                  xAxis=dict(type="value", min=0, max=1, interval=1 / 6, name="Ask", nameLocation="middle", nameGap=34, axisLabel=dict(formatter=pct)),
                  yAxis=dict(type="value", min=0, max=1, name="Mean pairwise edit distance / 84", nameLocation="middle", nameGap=46),
                  series=series, animation=False)
    write_chart(option, Path(folder) / "entropy-edit-distance.html", height=400,
                title=f"Fill orders differ by {ed[ROW]['mean']:.3f} of 84 places with entropy-guided initialization, {ed[CONFIDENCE]['mean']:.3f} confidence order",
                subtitle="Mean Levenshtein distance between the fill orders of two teams at the same ask, divided by 84 (Jeon et al. 2026, App. C.4). "
                         "0: every team fills its fields in the same order.",
                note="All 192 teams of an ask, pooled over its 3 sampling seeds.")


def _iv(x, f="+.4f"): return f"[{x[0]:{f}}, {x[1]:{f}}]"


def _pass(r, row, key):
    p = r["rows"][row]["pass_at_k"]
    if not p: return "n/a"
    return f"{p['pass_at_k'][-1]:.3f} {_iv(p['pass_at_k_interval'][-1], '.3f')}" if key == "pass" else f"{p['potential_at_k'][-1]:.3f}"


def _novelty(r, row, key):
    nv = r["rows"][row].get("novelty") or {}
    if not nv.get("teams"): return "n/a"
    return f"{nv['novel_share']:.1%}" if key == "novel_share" else f"{nv['novel_line_ups']:,}"


def _diagnostic(r, row, key, f):
    return format(r[key][row] if key != "fill_order" else r[key][row]["mean_pass_of_last_species"], f) if row in r[key] else "n/a"


def _edit(r, row): return f"{r['edit_distance'][row]['mean']:.3f}" if row in r["edit_distance"] else "n/a"


# The verdict table: (label, how one row's cell reads from the result of read()).
MEASURES = (("Pass@48, asks 13/24 to 17/24", lambda r, row: _pass(r, row, "pass")),
            ("Potential@48, asks 13/24 to 17/24", lambda r, row: _pass(r, row, "potential")),
            ("Mean win rate, 23 asks", lambda r, row: f"{r['rows'][row]['pooled_win']:.3f} {_iv(r['rows'][row]['pooled_win_interval'], '.3f')}"),
            ("Ask-versus-got error, asks 1/24 to 17/24", lambda r, row: f"{r['rows'][row]['error']:.4f} {_iv(r['rows'][row]['error_interval'], '.4f')}"),
            ("First-try legality", lambda r, row: f"{r['rows'][row]['first_try_legality']:.1%}"),
            ("Distinct line-ups", lambda r, row: f"{r['rows'][row]['line_ups']:,}"),
            ("Teams with a novel line-up", lambda r, row: _novelty(r, row, "novel_share")),
            ("Distinct novel line-ups", lambda r, row: _novelty(r, row, "novel_line_ups")),
            ("Effective species, 5-team subsets", lambda r, row: f"{r['rows'][row]['effective_species']:.1f}"),
            ("Elite line-ups that held, all asks", lambda r, row: str((r["rows"][row]["elite_line_up_yield"] or {}).get("all_23_asks", {}).get("held", "n/a"))),
            ("Fill-order edit distance / 84", _edit),
            ("Pass of the last species", lambda r, row: _diagnostic(r, row, "fill_order", ".1f")),
            ("Non-top draw share", lambda r, row: _diagnostic(r, row, "non_top_draw_share", ".3f")))


def comment(figure_url):
    """The verdict comment, picture first. figure_url holds {name} (pareto, got, pass-at-k or edit-distance) and {theme}."""
    import datetime
    r = json.load(open(out_path())); rows = r["rows"]; new, conf = rows[ROW], rows[CONFIDENCE]
    iv = lambda x, f="+.4f": f"[{x[0]:{f}}, {x[1]:{f}}]"
    pic = lambda name, alt: (f'<picture><source media="(prefers-color-scheme: dark)" srcset="{figure_url.format(name=name, theme="dark")}">'
                             f'<img alt="{alt}" src="{figure_url.format(name=name, theme="light")}"></picture>')
    pred = r["prediction"] or {"verdict": "not read", "parts": {}}
    word = lambda ok: "held" if ok else "did not hold"
    g = r["against"][CONFIDENCE]; ed = r["edit_distance"]; ft = r["first_field_types"][ROW]
    pa = r["pareto"]
    lines = [f"**Result, {datetime.date.today()}: {headline(r)}. Primary test: {pa['verdict']}.**", "",
             pic("pareto", "Mean win rate against distinct line-ups for each decoding order's guidance curve, and the new arm at guidance 2"), "",
             f"Primary test (owner, 2026-11-17): outside confidence order's curve means at least {LINE_UP_MARGIN} more line-ups than its widest point "
             f"({max(c['line_ups'] for c in pa['curve']):,}) at a win rate within {WIN_MARGIN}, or more than {WIN_MARGIN} win rate above the curve at the same "
             f"line-up count. More line-ups: {'yes' if pa['wider'] else 'no'}. Higher win rate: {'yes' if pa['higher'] else 'no'}.", "",
             pic("pass-at-k", "Pass@k at asks 13/24 to 17/24 for entropy-guided initialization, confidence order, species-first order and random order"), "",
             "The ticket's Pass@48 prediction, reported but not the test (its ceiling probe failed: confidence order's interval reaches 0.998), part by part: "
             + "; ".join(f"{k.replace('_', ' ')}: {word(v)}" for k, v in pred["parts"].items()) + f" (overall {pred['verdict']}).", "",
             pic("got", "Got against ask for the four rows at guidance 2"), "",
             f"- Mean win rate over the 23 asks: {new['pooled_win']:.3f}, against {conf['pooled_win']:.3f} {iv(conf['pooled_win_interval'], '.3f')} for confidence order "
             f"(gap {g['pooled_win']['difference']:+.3f} {iv(g['pooled_win']['interval'], '+.3f')}), {rows[SPECIES_FIRST]['pooled_win']:.3f} species-first, "
             f"{rows[RANDOM]['pooled_win']:.3f} random order.",
             f"- Ask-versus-got error on asks 1/24 to 17/24: {new['error']:.4f}, against {conf['error']:.4f} for confidence order "
             f"(paired by sampling seed: {g['error']['difference']:+.4f} {iv(g['error']['interval'])}).",
             "- Pass 1 field type over the kept teams: " + ", ".join(f"{v:.0%} {k.replace('_', ' ')}" for k, v in sorted(ft.items(), key=lambda kv: -kv[1])) + ".", "",
             pic("edit-distance", "Mean pairwise fill-order edit distance by ask, with entropy-guided initialization against confidence order"), "",
             f"Fill orders at one ask differ by {ed[ROW]['mean']:.3f} of 84 places on average with entropy-guided initialization, {ed[CONFIDENCE]['mean']:.3f} with confidence order "
             "(Jeon et al. report 0.45 to 0.52 on GSM8K).", "",
             "| Measure | " + " | ".join(m["reader"] for m in rows.values()) + " |", "|---|" + "---|" * len(rows)]
    lines += [f"| {label} | " + " | ".join(show(r, row) for row in rows) + " |" for label, show in MEASURES]
    lines += ["", "<details><summary>Method</summary>", "",
              f"Entropy-guided initialization: pass 1 fills the masked field whose drawing distribution (guidance 2, legal Pokémon construction filter, lookahead, temperature 1) "
              f"has the highest entropy, and draws its value at temperature {FIRST_TEMPERATURE}; passes 2 to 84 follow confidence order at temperature 1 "
              "(Jeon et al. 2026, §6.2). Checkpoint `stat84_shuffled_noise_linear_es3_cap80`, 23 asks × 3 sampling seeds × 64 teams, one battle per team against "
              "each of the 49 distinct top-50 meta teams, behaviour-cloning policy on both sides, slots shuffled. Interval: 95% bootstrap, teams resampled within "
              "each ask and sampling seed, the 49 opponents once for all four rows. Pass@48 above means the point lies above the upper end of confidence order's "
              f"interval; the point alone {'is' if pred.get('pass_at_48_point_above') else 'is not'} above confidence order's point. Species-first's novelty comes "
              "from its grid redrawn on seeds of its own (#353).", "",
              "Scripts: `code/src/entropy_init84.py read`, `figure`, `comment`. Results: `~/vgc-data/results/entropy_init.json`.", "</details>"]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "comment": print(comment(sys.argv[2]), end="")
    elif cmd == "probe_comment": sys.exit(probe_comment(*sys.argv[2:]))
    else: {"probe": probe, "probe_figure": probe_figure, "read": read, "figure": figure}[cmd](*sys.argv[2:])
