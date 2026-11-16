"""How near the line is the model with no guidance? (docs/no-guidance-line.md, GitHub issue #73)

Reads the four grids of Linear, species-first on the linear masking-schedule checkpoint: guidance 1, new, and guidance
2, 4 and 8, stored by masking_rate_rows.py. Each guided arm is judged against guidance 1 by the pre-registered rule.

    python no_guidance_line.py read              every measure and verdict, into results/no_guidance_line.json
    python no_guidance_line.py figure DIR        got against ask for the four arms, as an ECharts page in DIR
    python no_guidance_line.py comment URL       the result comment for #73; URL holds {theme} for light or dark

The interval is the bootstrap of species_late.read_row: teams resampled within each ask and sampling seed, separately
in each arm, and the 49 opponents once for all four. The guided arms keep masking_rate_rows.py's team-resampling
streams, so their intervals reproduce the ones #297 posted; read() checks that for guidance 2.
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
import species_late as SL

ROW = "sched_linear_species_first"
GUIDANCES = (1, 2, 4, 8)
READER = {1: "Guidance 1, no push", 2: "Guidance 2", 4: "Guidance 4", 8: "Guidance 8"}
COLOR = {1: "#d6452a", 2: "#2a78d6", 4: "#7a5bd0", 8: "#1f9e74"}
BAR = MRR.BAR                                            # near the line, and the smallest change the rule acts on
DATA_LIMIT = 0.0038                                      # re-battled training teams, stored Stat Points (issue #72)
PREDICTED = (0.008, 0.03)                                # guidance 1's error, docs/no-guidance-line.md, Prediction
RESULT = MRR.RES / "no_guidance_line.json"
ISSUE_URL = "https://github.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/issues/73"


def judge(difference, interval):
    """A guided arm's error minus guidance 1's, paired by sampling seed, with its 95% interval."""
    if difference <= -BAR and interval[1] < 0: return "nearer"
    if interval[0] > 0: return "farther"
    return "no measurable change"


def near_line(error): return error < BAR


def line_ups_per_cell(cells, line_up=None):
    """cells: {(ask numerator, seed): [paste]}. Distinct line-ups in each 64-team cell, averaged over the cells of the
    scored asks, and the mean by ask."""
    if line_up is None:
        import line_up_yield as LY
        line_up = LY.line_up
    count = {key: len({line_up(p) for p in pastes}) for key, pastes in cells.items()}
    by_ask = defaultdict(list)
    for (k, _), n in sorted(count.items()): by_ask[k].append(n)
    scored = [n for (k, _), n in count.items() if k in MRR.SCORED_ASKS]
    return dict(scored_asks=float(np.mean(scored)), by_ask={k: float(np.mean(v)) for k, v in by_ask.items()})


def split(no_guidance, guidance_2):
    """The map's three parts of the miss: the data limit, the fit above it, and what guidance 2 takes away."""
    return dict(data_limit=DATA_LIMIT, fit=no_guidance - DATA_LIMIT, guidance_2_removes=no_guidance - guidance_2)


def stream(guidance):
    """masking_rate_rows.bootstrap's team-resampling stream for this row, so stored intervals reproduce."""
    return [MRR.DRAW_SEED, list(MRR.ROWS).index(ROW), int(guidance)]


def read():
    MRR.use_pilot_src()
    counts = MRR.opponent_draws(len(np.load(MRR.opponents_path(ROW, 1))["weights"]))
    grids = {g: MRR.load_grid(ROW, g) for g in GUIDANCES}
    missing = [g for g, out in grids.items() if out is None]
    if missing: raise SystemExit(f"grids missing or short of battles at guidance {missing}")
    arms = {g: SL.read_row(ROW, counts, guidance=g, stream=stream(g)) for g in GUIDANCES}
    stored = json.load(open(MRR.SUMMARY))["grids"][MRR.grid_name(ROW, 2)]
    reproduced = MRR.interval(arms[2]["error_draws"])
    if abs(arms[2]["error"] - stored["error"]) > 1e-12 or max(abs(a - b) for a, b in zip(reproduced, stored["error_interval"])) > 1e-12:
        raise SystemExit(f"guidance 2 no longer reproduces #297: error {arms[2]['error']} {reproduced} against {stored['error']} {stored['error_interval']}")
    result = dict(model=ROW, checkpoint=grids[1]["checkpoint"], checkpoint_sha256=grids[1]["checkpoint_sha256"], decoder=grids[1].get("decoder"),
                  rule=f"near the line: error under {BAR}. Against guidance 1: nearer if the paired difference is at most -{BAR} with its interval below zero; "
                       "farther if the interval is above zero; otherwise no measurable change",
                  resampled="teams within each ask and sampling seed, separately in each arm; the 49 opponents once for all four arms",
                  draws=MRR.DRAWS, draw_seed=MRR.DRAW_SEED, arms={})
    if any(out["checkpoint_sha256"] != result["checkpoint_sha256"] for out in grids.values()): raise SystemExit("the four grids do not share one checkpoint")
    for g, m in arms.items():
        cells = grids[g]["by_ask_seed"]
        lu = line_ups_per_cell({key: c["pastes"] for key, c in cells.items()})
        tried = sum(c["tried"] for c in cells.values()); rejected = sum(c["rejected"] for c in cells.values())
        arm = dict(reader=READER[g], code_id=MRR.grid_name(ROW, g), error=m["error"], error_interval=MRR.interval(m["error_draws"]),
                   error_by_seed={str(s): v for s, v in m["error_by_seed"].items()},
                   error_asks_18_to_23=MRR.error(m["got_by_ask"], range(18, 24)),
                   got_by_ask={str(k): v for k, v in m["got_by_ask"].items()},
                   got_by_ask_interval={str(k): MRR.interval(v) for k, v in m["got_by_ask_draws"].items()},
                   line_ups_per_64=lu["scored_asks"], line_ups_per_64_by_ask={str(k): v for k, v in lu["by_ask"].items()},
                   first_try_legality=(tried - rejected) / max(tried, 1), teams=sum(len(c["pastes"]) for c in cells.values()),
                   battles=sum(c["battles"] for c in cells.values()))
        if g != 1:
            p = SL.paired(m, arms[1]); arm["against_no_guidance"] = dict(p, verdict=judge(p["difference"], p["interval"]))
        result["arms"][str(g)] = arm
    no, two = result["arms"]["1"], result["arms"]["2"]
    result["near_line"] = near_line(no["error"])
    result["split"] = split(no["error"], two["error"])
    result["prediction_held"] = bool(two["against_no_guidance"]["verdict"] == "nearer" and PREDICTED[0] <= no["error"] <= PREDICTED[1]
                                     and no["line_ups_per_64"] >= two["line_ups_per_64"])
    json.dump(result, open(RESULT, "w"), indent=1)
    for g, a in result["arms"].items():
        v = a.get("against_no_guidance", {})
        print(f"{a['reader']:22s} error {a['error']:.4f} [{a['error_interval'][0]:.4f}, {a['error_interval'][1]:.4f}]  line-ups/64 {a['line_ups_per_64']:.1f}  {v.get('verdict', '')} {v.get('difference', '')}")
    print("NEAR_LINE", result["near_line"], "SPLIT", result["split"], "PREDICTION_HELD", result["prediction_held"])
    print("NO_GUIDANCE_LINE_DONE")


def figure(folder):
    """Got against ask for the four arms, each with its 95% band, beside the line got = ask."""
    from echarts_chart import band, write_chart
    r = json.load(open(RESULT)); asks = [int(k) for k in r["arms"]["1"]["got_by_ask"]]; xs = [k / 24 for k in asks]; series = []
    for i, (g, a) in enumerate(r["arms"].items()):
        g = int(g)
        series += band(a["reader"], COLOR[g], xs, [a["got_by_ask_interval"][str(k)][0] for k in asks], [a["got_by_ask_interval"][str(k)][1] for k in asks], stack=f"band{i}")
        series.append(dict(type="line", name=a["reader"], color=COLOR[g], symbolSize=6, lineStyle=dict(width=2.6 if g == 1 else 1.6),
                           emphasis=dict(focus="series"), data=[[x, a["got_by_ask"][str(k)]] for x, k in zip(xs, asks)]))
    series.append(dict(type="line", name="Got = ask", color="#6f6e69", symbol="none", silent=True, lineStyle=dict(type="dashed", width=1), data=[[0, 0], [1, 1]]))
    pct = "js:v => Math.round(v * 100) + '%'"
    option = dict(grid=dict(left=64, right=24, top=56, bottom=56), legend=dict(top=0, data=[a["reader"] for a in r["arms"].values()] + ["Got = ask"]),
                  tooltip=dict(trigger="axis", valueFormatter="js:v => v == null ? '' : (v * 100).toFixed(1) + '%'"),
                  xAxis=dict(type="value", min=0, max=1, interval=1 / 6, name="Ask: the win rate the model is asked for", nameLocation="middle", nameGap=34,
                             axisLabel=dict(formatter=pct)),
                  yAxis=dict(type="value", min=0, max=0.8, name="Got: measured win rate", nameLocation="middle", nameGap=46, axisLabel=dict(formatter=pct)),
                  series=series, animation=False)
    no, two = r["arms"]["1"], r["arms"]["2"]
    title = (f"With no guidance the model {'lands near' if r['near_line'] else 'misses'} the line: error {no['error']:.4f} against the {BAR} bar")
    return write_chart(option, Path(folder) / "no-guidance-line.html", height=460, title=title,
                       subtitle=f"Linear masking schedule, species-first. Guidance 2 is {two['against_no_guidance']['verdict']}, at {two['error']:.4f}. "
                                f"Line-ups per 64 teams: {no['line_ups_per_64']:.1f} with no guidance, {two['line_ups_per_64']:.1f} at guidance 2.",
                       note="Band: 95% bootstrap interval over teams and the 49 opponents. Got: one battle against each of the 49 distinct top-50 meta teams, "
                            "behaviour-cloning policy on both sides, slots shuffled; 192 teams per ask over 3 sampling seeds.")


def comment(r, figure_url):
    """The result comment, result first. figure_url has {theme} where light or dark goes."""
    a = r["arms"]; no, two = a["1"], a["2"]; iv = lambda x, f="+.4f": f"[{x[0]:{f}}, {x[1]:{f}}]"
    s = r["split"]
    lines = [f"**Result, {datetime.date.today()}: with no guidance the model's ask-versus-got error is {no['error']:.4f} {iv(no['error_interval'], '.4f')}, "
             f"{'under' if r['near_line'] else 'above'} the {BAR} bar. Guidance 2 is {two['against_no_guidance']['verdict']}, at {two['error']:.4f}.**", "",
             f'<picture><source media="(prefers-color-scheme: dark)" srcset="{figure_url.format(theme="dark")}"><img alt="Got against ask at guidance 1, 2, 4 and 8" src="{figure_url.format(theme="light")}"></picture>', "",
             "Each arm is the linear masking-schedule checkpoint `stat84_shuffled_noise_linear_es3_cap80`, decoded species-first, 192 teams per ask over 3 sampling seeds. "
             "Guidance 1 is new; guidance 2, 4 and 8 are the stored grids of the masking-schedule rows. Each guided arm is compared with guidance 1, paired by sampling seed.", "",
             "| Arm | Error, asks 1/24 to 17/24 | Against guidance 1 | Error, asks 18/24 to 23/24 | Line-ups per 64 teams |", "|---|---|---|---|---:|"]
    for g, arm in a.items():
        v = arm.get("against_no_guidance")
        lines.append(f"| {arm['reader']} | {arm['error']:.4f} {iv(arm['error_interval'], '.4f')} | "
                     f"{'baseline' if v is None else v['verdict'] + ', ' + format(v['difference'], '+.4f') + ' ' + iv(v['interval'])} | "
                     f"{arm['error_asks_18_to_23']:.4f} | {arm['line_ups_per_64']:.1f} |")
    lines += ["", "## For the map", "",
              f"The miss with no guidance splits into the data limit, {s['data_limit']:.4f} (re-battled training teams, #72), and the fit above it, {s['fit']:.4f}. "
              f"Guidance 2 then takes away {s['guidance_2_removes']:.4f}. #74 reads these side by side. The data limit came from the all-teams set on the older table, so the split is approximate.", "",
              f"Prediction (`code/docs/no-guidance-line.md`): guidance 2 nearer, guidance 1's error between {PREDICTED[0]} and {PREDICTED[1]}, and at least as many line-ups per 64 teams with no guidance. "
              f"**{'Held' if r['prediction_held'] else 'Did not hold'}.**", "",
              "<details><summary>Method</summary>", "",
              "Win rate: one battle against each of the 49 distinct top-50 meta teams, behaviour-cloning policy on both sides, slots shuffled. "
              "Line-ups per 64 teams: distinct six-species sets in one cell (one ask, one sampling seed), averaged over asks 1/24 to 17/24. "
              "Interval: 95% bootstrap, teams resampled within each ask and sampling seed, the 49 opponents once for all four arms. "
              "Script: `code/src/no_guidance_line.py read`. Results: `~/vgc-data/results/no_guidance_line.json`.", "</details>"]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    {"read": read, "figure": lambda: print(figure(sys.argv[2])),
     "comment": lambda: print(comment(json.load(open(RESULT)), sys.argv[2]), end="")}[sys.argv[1]]()
