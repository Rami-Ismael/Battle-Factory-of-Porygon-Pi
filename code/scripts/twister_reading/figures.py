"""Charts for the reading of Thimmaiah et al. 2026, "Twist, Don't Tilt" (TWISTER, arXiv 2609.35609), against the #297,
#322 and #335 ask grids. Writes one chart page per figure with write_chart and returns the options, so build_page.py
shows the same numbers. Usage: figures.py <data dir> <out dir>
Data dir: tilt_analysis.json (tilt_analysis.py), trace_summary.json and order_consistency.json (Two Temperatures reading,
2026-11-14, reused unchanged)."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from echarts_chart import write_chart

BLUE, AQUA, ORANGE, VIOLET, PINK, GREY = "#2a78d6", "#1baf7a", "#e0702a", "#6250d6", "#d6457a", "#6f6e69"
ROWS = {  # code ID -> (reader name: decoding order first, then masking schedule; colour)
    "sched_species_first": ("Species-first, per-field schedule (control)", BLUE),
    "sched_linear_species_first": ("Species-first, linear schedule", AQUA),
    "sched_linear_species_then_random": ("Species-then-random, linear schedule", VIOLET),
    "sched_linear_random": ("Random-order, linear schedule", ORANGE),
    "sched_per_field_random": ("Random-order, per-field schedule", PINK)}
KINDS = ["species", "ability", "item", "move", "Stat Alignment", "Stat Points"]
TIP = dict(trigger="axis", axisPointer=dict(type="shadow"))
end = lambda fmt: dict(show=True, formatter=fmt, color="inherit", fontSize=12)
CHARTS = {}


def chart(key, option, title, sub="", note="", height=420):
    CHARTS[key] = dict(option=option, title=title, sub=sub, note=note, height=height)
    if OUT: write_chart(option, OUT / f"{key}.html", title=title, subtitle=sub, note=note, height=height)


# 1. The paper's own evidence, Figure 1, re-plotted from the rendered PDFs ------------------------------------------
STEPS = [1, 2, 4, 8, 16]
FIG1 = {  # read off figures/intro/bias-*.pdf rendered at 110 dpi; accurate to about +-0.005
    "OWT-130M": dict(dang=[.069, .090, .131, .200, .260], smc4=[.069, .076, .084, .106, .114], smc8=[.069, .075, .080, .091, .095], floor=[.089, .092, .096, .100, .104]),
    "Dream-Coder-Instruct 7B": dict(dang=[.111, .235, .281, .349, .409], smc4=[.126, .138, .133, .140, .180], smc8=[.137, .119, .103, .113, .138], floor=[.107, .067, .073, .080, .081]),
    "Dream base 7B": dict(dang=[.161, .161, .173, .190, .223], smc4=[.159, .161, .170, .174, .186], smc8=[.160, .159, .169, .172, .175], floor=[.156, .159, .167, .169, .166])}
SER = [("dang", "Step-exact decoder, uncorrected (Dang & Ermon, reimplemented)", ORANGE, "solid"),
       ("smc4", "TWISTER, 4 particles", BLUE, "solid"), ("smc8", "TWISTER, 8 particles", AQUA, "solid"),
       ("floor", "Noise floor: two halves of the rejection samples", GREY, "dashed")]


def fig1():
    grids, xs, ys, series = [], [], [], []
    for i, (m, d) in enumerate(FIG1.items()):
        grids.append(dict(left=f"{4 + i * 33}%", width="27%", top=70, bottom=60, containLabel=True))
        xs.append(dict(type="category", data=[str(s) for s in STEPS], gridIndex=i, name=m, nameLocation="middle", nameGap=30))
        ys.append(dict(type="value", gridIndex=i, min=0, max=0.45, name="distance" if i == 0 else "",
                       axisLabel=dict(formatter="{value}")))
        for key, name, color, dash in SER:
            series.append(dict(name=name, type="line", xAxisIndex=i, yAxisIndex=i, data=d[key], color=color, symbolSize=6,
                               lineStyle=dict(type=dash), emphasis=dict(focus="series")))
    return dict(grid=grids, xAxis=xs, yAxis=ys, series=series, legend=dict(top=0), tooltip=dict(trigger="axis"))


OUT = None
if __name__ == "__main__":
    DATA, OUT = Path(sys.argv[1]), Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
else:
    DATA = Path(__import__("os").environ["TWISTER_DATA"])

chart("01_paper_figure1", fig1(), "The paper's only bias evidence: uncorrected drift grows with denoising steps; TWISTER pulls it back",
      "Total variation distance from rejection sampling against the number of denoising steps, three of the six models",
      "Re-plotted from the paper's Figure 1 (figures/intro/bias-owt.pdf, bias-dreamcoder-instruct.pdf, bias-dream-base.pdf), "
      "values read off the rendered plots, accurate to about ±0.005. Constraint: a toy regular language over token classes "
      "('a.*'); 40,000 rejection samples and 10,000 constrained samples per point, random remasking, temperature 1 "
      "(method-eval.tex:10). No run count or seeds are given; the grey line is the paper's split-half noise floor.", height=440)

# 2. Tilt weights on our grid teams ---------------------------------------------------------------------------------
TA = DATA / "tilt_analysis.json"
T = json.load(open(TA)) if TA.exists() else {}
order_rows = ["sched_species_first", "sched_linear_species_first", "sched_linear_species_then_random", "sched_linear_random", "sched_per_field_random"]
if T:
    import numpy as np
    keys = [f"{r}|g2" for r in order_rows if f"{r}|g2" in T and "error" not in T[f"{r}|g2"]]
    # 2a. distribution of the removed log mass per team, guidance 2
    box = []
    for k in keys:
        lw = -np.array(T[k]["log_w"])
        box.append([round(float(q), 3) for q in np.quantile(lw, [0.05, 0.25, 0.5, 0.75, 0.95])])
    chart("02_tilt_per_team", dict(
        grid=dict(left=10, right=40, top=20, bottom=40, containLabel=True),
        yAxis=dict(type="category", data=[ROWS[k.split("|")[0]][0] for k in keys], inverse=True),
        xAxis=dict(type="value", name="nats the sampler added to the team", nameLocation="middle", nameGap=28),
        tooltip=dict(trigger="item", formatter="js:p => p.name + '<br>5th ' + p.data[1] + ' · median ' + p.data[3] + ' · 95th ' + p.data[5] + ' nats'"),
        series=[dict(type="boxplot", data=[dict(value=b, itemStyle=dict(color="transparent", borderColor=ROWS[k.split("|")[0]][1], borderWidth=2)) for b, k in zip(box, keys)])]),
        "Species-first rows carry no tilt; random-order rows carry about 14 nats per team",
        "Per team: how much more likely the sampler made it than the trained model's own per-pass distributions, guidance 2, asks 13/24 to 17/24",
        "For each drawn field, the sampler removes probability the trained model gave (the [MASK] value, the empty first move, "
        "values the legal Pokémon construction tables forbid given fields drawn earlier in the same pass, and the lookahead) "
        "and renormalises. Summed -log(kept share) over the 84 draws = the log of TWISTER's correction weight for that team. "
        "Box: 25th to 75th percentile; whiskers: 5th and 95th. 192 teams per ask, the grid's own teams redrawn with their seeds "
        "and checked identical.", height=300)
    # 2b. where the tilt sits, by field type (random-order rows only: species-first rows are all zero)
    rkeys = [k for k in keys if T[k]["row"] in ("sched_linear_random", "sched_per_field_random")]
    series = []
    for i, kd in enumerate(KINDS):
        series.append(dict(name=kd, type="bar", stack="t", emphasis=dict(focus="series"),
                           data=[round(-T[k]["mean_log_w_by_kind"][kd], 3) for k in rkeys]))
    chart("03_tilt_by_field", dict(
        grid=dict(left=10, right=30, top=40, bottom=40, containLabel=True), legend=dict(top=0), tooltip=TIP,
        yAxis=dict(type="category", data=[ROWS[k.split("|")[0]][0] for k in rkeys], inverse=True),
        xAxis=dict(type="value", name="mean nats per team", nameLocation="middle", nameGap=28), series=series,
        color=[BLUE, AQUA, VIOLET, ORANGE, PINK, GREY]),
        "Random order's tilt sits at moves and abilities, the fields drawn before their species",
        "Mean removed log mass per team, split by the type of the field being drawn, guidance 2; species-first rows remove nothing and are left out", height=240)
    # 2c. win rate by tilt decile
    SF = {2: 0.490, 4: 0.516, 8: 0.530}          # species-first mean win rate, linear checkpoint, asks 13-17 (bootstrap.json, Two Temperatures reading)
    SF64 = {2: 0.682, 4: 0.702, 8: 0.711}
    rk = [k for k in T if "error" not in T[k] and T[k]["row"] in ("sched_linear_random", "sched_per_field_random", "sched_linear_species_then_random")]
    rk.sort(key=lambda k: (order_rows.index(T[k]["row"]), T[k]["guidance"]))
    dash = {2: "solid", 4: "dashed", 8: "dotted"}
    series = [dict(name=f"{ROWS[T[k]['row']][0]}, guidance {T[k]['guidance']}", type="line", color=ROWS[T[k]["row"]][1],
                   lineStyle=dict(type=dash[T[k]["guidance"]]), symbolSize=6, emphasis=dict(focus="series"),
                   data=[round(b["win"], 3) for b in T[k]["deciles"]]) for k in rk if T[k]["row"] != "sched_linear_species_then_random"]
    series.append(dict(name="Species-first, linear schedule, guidance 2 (all teams)", type="line", color=AQUA, symbol="none",
                       lineStyle=dict(type="dashed", width=2), data=[SF[2]] * 10))
    chart("04_win_by_tilt_decile", dict(
        grid=dict(left=10, right=20, top=70, bottom=50, containLabel=True), legend=dict(top=0, type="scroll"),
        tooltip=dict(trigger="axis", valueFormatter="js:v => (v * 100).toFixed(1) + '%'"),
        xAxis=dict(type="category", name="tilt decile within the ask: 1 = most probability removed, 10 = least", nameLocation="middle", nameGap=30,
                   data=[str(i + 1) for i in range(10)]),
        yAxis=dict(type="value", min=0.2, max=0.55, axisLabel=dict(formatter="js:v => Math.round(v * 100) + '%'"), name="mean win rate"),
        series=series),
        "The more the sampler had to bend a random-order team, the more that team loses",
        "Mean win rate of random-order teams, grouped by how much probability the sampler removed while drawing them, asks 13/24 to 17/24",
        "Deciles of the team's log correction weight, taken within each ask so the asks mix evenly; about 96 teams per point. "
        "Dashed teal line: species-first order on the linear checkpoint, all teams, guidance 2. Win rate: one battle against each "
        "of the 49 distinct top-50 meta teams, behaviour-cloning policy, slots shuffled.", height=420)
    # 2d. tempered correction: dose curve
    series = []
    for k in rk:
        r = T[k]; nm = f"{ROWS[r['row']][0]}, guidance {r['guidance']}"
        for metric, idx in (("mean", 0), ("best@64", 1)):
            series.append(dict(name=nm, type="line", xAxisIndex=idx, yAxisIndex=idx, color=ROWS[r["row"]][1], lineStyle=dict(type=dash[r["guidance"]]),
                               symbolSize=6, emphasis=dict(focus="series"),
                               data=[[c["beta"], round(c[metric]["point"], 3), round(c["ess_share"], 3)] for c in r["tempered"]]))
    for idx, ref in ((0, SF), (1, SF64)):
        series.append(dict(name="Species-first, linear schedule, guidance 2 (all teams)", type="line", xAxisIndex=idx, yAxisIndex=idx, color=AQUA,
                           symbol="none", lineStyle=dict(type="dashed", width=2), data=[[0, ref[2]], [1, ref[2]]]))
    chart("05_tempered_correction", dict(
        grid=[dict(left="6%", width="40%", top=80, bottom=50, containLabel=True), dict(left="56%", width="40%", top=80, bottom=50, containLabel=True)],
        legend=dict(top=0, type="scroll"),
        tooltip=dict(trigger="item", formatter="js:p => p.seriesName + '<br>strength ' + p.value[0] + ': ' + (p.value[1] * 100).toFixed(1) + '%' + (p.value[2] !== undefined ? ' · effective teams ' + Math.round(p.value[2] * 100) + '%' : '')"),
        xAxis=[dict(type="value", min=0, max=1, gridIndex=i, name=n, nameLocation="middle", nameGap=28) for i, n in ((0, "correction strength (0 = as drawn, 1 = TWISTER)"), (1, "correction strength"))],
        yAxis=[dict(type="value", gridIndex=0, min=0.3, max=0.55, name="mean win rate", axisLabel=dict(formatter="js:v => Math.round(v * 100) + '%'")),
               dict(type="value", gridIndex=1, min=0.5, max=0.75, name="best of 64", axisLabel=dict(formatter="js:v => Math.round(v * 100) + '%'"))],
        series=series),
        "Correcting the tilt lifts the average random-order team toward species-first, but not the best one",
        "Teams reweighted by their correction weight raised to a strength between 0 and 1; first panel: mean win rate, second: expected best of 64",
        "Strength 1 is TWISTER's target: the model's own random-order conditionals restricted to legal teams. With 192 teams per ask "
        "the full weights leave about 2% effective teams (hover a point), so strengths near 1 rest on a handful of teams; read the "
        "curve's direction, not its end point. Best of 64 here draws with replacement under the weights. Dashed teal: species-first "
        "order on the linear checkpoint, guidance 2.", height=440)

# 2e. the mechanism, exact on one Pokémon (tilt_exact_toy.py) -------------------------------------------------------
TOY = DATA / "tilt_exact_toy.json"
if TOY.exists():
    d = json.load(open(TOY))
    SPN = {"incineroar": "Incineroar", "meowsticfmega": "Mega Meowstic (F)", "meowsticf": "Meowstic (F)", "raichu": "Raichu", "grimmsnarl": "Grimmsnarl"}
    ABN = {"intimidate": "Intimidate", "competitive": "Competitive", "lightningrod": "Lightning Rod", "blaze": "Blaze"}
    TGT, SMP = "Model's own odds among legal teams", "Lookahead sampler"
    sp = [r for r in d["species"] if r["name"] in SPN][::-1]; ab = [r for r in d["abilities"] if r["name"] in ABN][::-1]
    pct = lambda v: round(100 * v, 1)
    lab = dict(show=True, position="right", formatter="{c}%", fontSize=11, color="inherit")
    series = [dict(name=nm, type="bar", xAxisIndex=i, yAxisIndex=i, color=col, label=lab, barGap="10%", emphasis=dict(focus="series"),
                   data=[pct(r[key]) for r in rows])
              for i, rows in enumerate([sp, ab]) for nm, key, col in [(TGT, "target", GREY), (SMP, "sampler", ORANGE)]]
    chart("00_exact_toy", dict(
        tooltip=dict(trigger="axis", axisPointer=dict(type="shadow"), valueFormatter="js:v => v + '%'"), legend=dict(top=0, data=[TGT, SMP]),
        title=[dict(text="Which species you get", top=34, left=0, textStyle=dict(fontSize=13)),
               dict(text="Which ability (share of move 2 kept legal)", top=300, left=0, textStyle=dict(fontSize=13))],
        grid=[dict(top=58, height=210, left=10, right=60, containLabel=True), dict(top=324, bottom=10, left=10, right=60, containLabel=True)],
        xAxis=[dict(gridIndex=i, type="value", max=60, splitNumber=3, axisLabel=dict(formatter="{value}%")) for i in (0, 1)],
        yAxis=[dict(gridIndex=0, type="category", data=[SPN[r["name"]] for r in sp]),
               dict(gridIndex=1, type="category", data=[f"{ABN[r['name']]} (keeps {round(100 * r['keep_share'])}%)" for r in ab])],
        series=series),
        "Every draw is legal, and the team still comes out tilted",
        f"One Pokémon shows Fake Out; ability, then move 2, then species are drawn. Random-order model, linear schedule, ask {d['ask']:.3f}, "
        f"guidance {d['guidance']:g}; all {d['paths']:,} legal paths enumerated exactly",
        f"The lookahead sampler moves {pct(d['tv_joint'])}% of the probability to other outcomes (total variation distance). Why: an ability "
        "is kept if any Fake Out user has it, but not weighted by how much of the model's move-2 probability then stays legal (Intimidate "
        "keeps 40%, Competitive 20%). Draw-then-reject loses most Competitive paths at move 2; the lookahead saves every one. "
        f"{d['n_draws']:,} real draws each way match the exact numbers within 0.6 points; draw-then-reject keeps "
        f"{pct(d['p_valid_draw_then_reject'])}% of tries. Grey bars: draw, then throw out dead ends. Script: tilt_exact_toy.py.", height=560)

# 2f. the tables against the lookahead (tables_tilt.py + tilt_analysis.py on tables_base/) ----------------------------
TB = DATA / "tables_analysis.json"
if T and TB.exists():
    TBA = json.load(open(TB))
    cmp_rows = [("Tables", "sched_species_first", TBA), ("Tables", "sched_linear_species_first", TBA),
                ("Lookahead", "sched_per_field_random", T), ("Lookahead", "sched_linear_random", T)][::-1]
    names = [f"{w}: {ROWS[r][0]}" for w, r, _ in cmp_rows]
    lab = lambda fmt: dict(show=True, position="right", formatter=fmt, fontSize=11, color="inherit")
    chart("08_tables_vs_lookahead", dict(
        tooltip=TIP,
        title=[dict(text="Probability removed per team (median, nats)", top=0, left=0, textStyle=dict(fontSize=13)),
               dict(text="How well a team's removal tracks its win rate (Spearman, within ask)", top="52%", left=0, textStyle=dict(fontSize=13))],
        grid=[dict(top=28, height="36%", left=10, right=50, containLabel=True), dict(top="60%", bottom=20, left=10, right=50, containLabel=True)],
        xAxis=[dict(gridIndex=0, type="value"), dict(gridIndex=1, type="value", min=0, max=0.5)],
        yAxis=[dict(gridIndex=i, type="category", data=names) for i in (0, 1)],
        series=[dict(name="Removed (nats)", type="bar", xAxisIndex=0, yAxisIndex=0, label=lab("{c}"), emphasis=dict(focus="series"),
                     data=[dict(value=round(-src[f"{r}|g2"]["median_log_w"], 1), itemStyle=dict(color=ROWS[r][1])) for _, r, src in cmp_rows]),
                dict(name="Spearman", type="bar", xAxisIndex=1, yAxisIndex=1, label=lab("+{c}"), emphasis=dict(focus="series"),
                     data=[dict(value=round(src[f"{r}|g2"]["spearman_within_ask"], 2), itemStyle=dict(color=ROWS[r][1])) for _, r, src in cmp_rows])]),
        "The tables remove eight to ten times more than the lookahead, and their cut tells you nothing",
        "Grid teams at asks 13/24 to 17/24, guidance 2. Tables: the network before the legal Pokémon construction tables against after, "
        "on the species-first rows. Lookahead: the tables-masked model against the sampler, on the random-order rows",
        "The training loss masks the same values the tables forbid (diffusion84.elbo_terms), so those values get no gradient and the mass "
        "the bare network puts on them is untrained. Removing it bends nothing the model learned. Reweighting species-first teams by the "
        "tables' share leaves 1% effective teams and moves the mean win rate in opposite directions on the two rows (+0.08 linear, -0.07 "
        "per-field): noise. All 1,920 species-first teams were redrawn with their seeds and matched the stored teams bit for bit. "
        "Scripts: base_law.py, tables_tilt.py.", height=560)

# 2g. the fix, run: rejection without the lookahead (#358) against the reweighting estimate ---------------------------
REJ = dict(point=0.410, lo=0.396, hi=0.426, share=49, share_lo=41, share_hi=57)   # #358 comment 6966763730, 2026-11-17; plain_reject.py
if T:
    est = next(c["mean"]["point"] for c in T["sched_linear_random|g2"]["tempered"] if c["beta"] == 1)
    drawn = next(c["mean"]["point"] for c in T["sched_linear_random|g2"]["tempered"] if c["beta"] == 0)
    gap = lambda v: round(100 * (v - drawn) / (SF[2] - drawn))
    pts = [("Random order, as drawn (lookahead)", drawn, ORANGE, "circle", "bottom"),
           (f"Reweighting estimate: {gap(est)}% of gap", est, GREY, "emptyCircle", "top"),
           (f"Exact target, run (#358): {REJ['share']}% of gap", REJ["point"], ORANGE, "diamond", "top"),
           ("Species-first order", SF[2], AQUA, "circle", "bottom")]
    series = [dict(name=nm, type="scatter", symbol=sym, symbolSize=16, color=col, emphasis=dict(focus="series"),
                   label=dict(show=True, position=pos, formatter="js:p => (p.value[0] * 100).toFixed(1) + '%'", fontSize=12, color="inherit"),
                   data=[[round(v, 3), 0]]) for nm, v, col, sym, pos in pts]
    series.append(dict(name="95% interval of the run", type="line", color=ORANGE, symbol="none", lineStyle=dict(width=3),
                       data=[[REJ["lo"], 0], [REJ["hi"], 0]], tooltip=dict(show=False)))
    chart("09_fix_measured", dict(
        legend=dict(top=0, orient="vertical", left=0, data=[p[0] for p in pts]), tooltip=dict(trigger="item", valueFormatter="js:v => v"),
        grid=dict(left=20, right=30, top=118, bottom=40, containLabel=True),
        xAxis=dict(type="value", min=0.30, max=0.55, interval=0.05, name="mean win rate, asks 13/24 to 17/24", nameLocation="middle", nameGap=28,
                   axisLabel=dict(formatter="js:v => Math.round(v * 100) + '%'")),
        yAxis=dict(type="category", data=["Linear schedule, guidance 2"], axisLabel=dict(show=False)), series=series),
        f"Drawing exactly from the target won back half of random order's gap, not the {gap(est)}% reweighting estimated",
        "Mean win rate of random-order teams: as drawn, reweighted to TWISTER's target, and drawn exactly from that target by rejection (#358)",
        f"The run's share of the gap: {REJ['share']}% [{REJ['share_lo']}%, {REJ['share_hi']}%]. Reweighting reuses the 192 grid teams per ask under weights that leave 1-2% effective teams, so a few heavy teams set the estimate. "
        "Rejection draws fresh teams with random order and no lookahead until Showdown's validator accepts (1.52% of 63,295 tries): exactly "
        "the model's own random-order distribution among legal teams. Interval: paired bootstrap against species-first, -0.079 "
        "[-0.094, -0.064]. Source: issue #358, plain_reject.py, 2026-11-17.", height=300)

# 3. Does the model behave like one joint distribution? (reused, Two Temperatures reading) ---------------------------
OC = json.load(open(DATA / "order_consistency.json"))["results"]
labels = {"per-field masking schedule (control)": "Per-field schedule", "linear masking schedule": "Linear schedule"}
series = []
for ck, nm in labels.items():
    pt = OC[ck]["per_type_terms"]
    series.append(dict(name=nm, type="bar", emphasis=dict(focus="series"), color=BLUE if "Per" in nm else AQUA,
                       data=[round(pt[k]["sf"] - pt[k]["mf"], 2) for k in KINDS],
                       label=dict(show=True, position="right", formatter="js:p => (p.value > 0 ? '+' : '') + p.value.toFixed(1)", fontSize=11)))
chart("06_order_gap_by_field", dict(
    grid=dict(left=10, right=50, top=40, bottom=40, containLabel=True), legend=dict(top=0), tooltip=TIP,
    yAxis=dict(type="category", data=KINDS, inverse=True), xAxis=dict(type="value", name="species-first minus moves-first, nats per team", nameLocation="middle", nameGap=28),
    series=series),
    "The model is not one joint distribution: the same held-out team scores 18.2 nats higher species-first",
    "Log-likelihood of 1,024 held-out teams in species-first order minus moves-first order, by field type",
    "Reused from the Two Temperatures reading (2026-11-14), order_consistency.py. One joint distribution would give 0 in every "
    "row. Species-first total minus moves-first total: 18.2 nats per team [17.7, 18.8] on the per-field checkpoint and 12.3 "
    "[11.9, 12.7] on the linear one. Species score better when drawn after their moves; moves score far worse when drawn before "
    "their species, and the second effect is larger.", height=330)

# 4. Non-top share by field type and guidance (reused) ---------------------------------------------------------------
TS = json.load(open(DATA / "trace_summary.json"))
series = []
for row in order_rows:
    for g, dash in ((2, "solid"), (4, "dashed"), (8, "dotted")):
        k = f"{row}_g{g}"
        if k not in TS: continue
        series.append(dict(name=f"{ROWS[row][0]}, guidance {g}", type="line", color=ROWS[row][1], lineStyle=dict(type=dash),
                           symbolSize=6, emphasis=dict(focus="series"),
                           data=[round(TS[k]["by_type"][kd]["non_top"], 3) for kd in KINDS]))
chart("07_non_top_share", dict(
    grid=dict(left=10, right=20, top=70, bottom=40, containLabel=True), legend=dict(top=0, type="scroll"),
    tooltip=dict(trigger="axis", valueFormatter="js:v => (v * 100).toFixed(1) + '%'"),
    xAxis=dict(type="category", data=KINDS), yAxis=dict(type="value", min=0, max=1, axisLabel=dict(formatter="js:v => Math.round(v * 100) + '%'")),
    series=series),
    "About two fields in five are filled with a value that was not the model's top choice",
    "Share of filled fields whose drawn value was not the top value of the distribution it was drawn from, by field type",
    "Reused from the Two Temperatures reading (trace_probe.py), 480 teams per row and guidance (2,880 species draws) at asks 13/24 to 17/24. "
    "LLaDA's released default (low-confidence remasking at temperature 0) is 0% by construction: it takes the argmax.", height=420)

if __name__ == "__main__":
    json.dump(CHARTS, open(OUT / "charts.json", "w"))
    print("charts:", sorted(CHARTS))
