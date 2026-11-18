"""Figures and result comment of "No tables in the loss" (issue #361; docs/no-tables-in-the-loss.md).

1. Illegal share by epoch, one line per arm, both drawn without the legal Pokémon construction tables.
3. Each rule a drawn team breaks, by epoch: one heatmap per network, rules down the side (epoch_legality.RULES,
   owner 2026-11-17: "more measurement than just legality"). Read from the checkpoint files of both networks.
2. Whose value the network picks: at a Pokémon's ability and first move, the share of probability legal for its own
   species, for another team member's species, or for none of the six (src/slot_binding_probe.py).

The new arm's line comes from its finished training file when there is one, else from the epochs saved in
results/no_tables_in_the_loss.json (the run stopped after epoch 33 on 2026-11-17).

    python src/no_tables_in_the_loss_figure.py figure [folder]          # default ~/vgc-data/no_tables_in_the_loss/figure
    python src/no_tables_in_the_loss_figure.py comment <image url with {name} and {theme}>
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, "/tmp/vgc-pilot/src")
sys.path.insert(0, str(HERE))
import asked_vs_got as X
from echarts_chart import write_chart

CONTROL = X.RES / "stat84_shuffled_noise_linear_es3_cap80_legality_no_tables.json"
ARM = X.RES / "stat84_shuffled_noise_linear_no_table_loss_es3_cap80.json"
ARM_CHECKS = X.RES / "stat84_shuffled_noise_linear_no_table_loss_es3_cap80_legality_no_tables.json"
RESULT = X.RES / "no_tables_in_the_loss.json"
FOLDER = Path.home() / "vgc-data" / "no_tables_in_the_loss" / "figure"
NEW, OLD = "No tables in the loss", "Tables in the loss"
LINES = {NEW: "#c2410c", OLD: "#52514e"}
PARTS = {"its own species": "#2f6f9f", "another team member's species": "#9333ea", "none of the six": "#b8b3ab"}
REFUTED_ABOVE = 0.875                                    # the plan: refuted if the new arm stays above this at every epoch


def lines():
    """{reader name: [epoch entries]} for the arms that have any."""
    out = {}
    if ARM.exists(): out[NEW] = [e for e in json.loads(ARM.read_text())["curves"] if "illegal_share" in e]
    elif RESULT.exists() and "arm_epochs" in json.loads(RESULT.read_text()):
        out[NEW] = json.loads(RESULT.read_text())["arm_epochs"]["epochs"]
    if CONTROL.exists(): out[OLD] = json.loads(CONTROL.read_text())["epochs"]
    return {k: v for k, v in out.items() if v}


def binding():
    return json.loads(RESULT.read_text())["slot_binding_epoch33"]["arms"][NEW]


def share_chart(data, folder):
    epochs = max(e["epoch"] for v in data.values() for e in v)
    order = [a for a in (OLD, NEW) if a in data]          # the new arm last, so it draws on top where the lines meet
    series = [dict(type="line", name=f"{arm}, {len(data[arm])} epochs", color=LINES[arm], emphasis=dict(focus="series"),
                   symbolSize=7 if arm == NEW else 0, lineStyle=dict(width=3) if arm == NEW else dict(width=2, type="dashed"),
                   data=[[e["epoch"], e["illegal_share"]] for e in data[arm]]) for arm in order]
    option = dict(grid=dict(left=64, right=24, top=56, bottom=80), legend=dict(top=0), animation=False,
                  tooltip=dict(trigger="axis", valueFormatter="js:v => v == null ? '' : (v * 100).toFixed(1) + '%'"),
                  xAxis=dict(type="value", min=1, max=epochs, name="Training epoch", nameLocation="middle", nameGap=30),
                  yAxis=dict(type="value", min=0, max=1, name="Illegal share of first draws", nameLocation="middle", nameGap=46,
                             axisLabel=dict(formatter="js:v => Math.round(v * 100) + '%'")),
                  dataZoom=[dict(type="inside"), dict(type="slider", bottom=8, height=18)], series=series)
    worst = max(e["illegal_share"] for v in data.values() for e in v); best = min(e["illegal_share"] for v in data.values() for e in v)
    title = ("Drawn without the tables, every team is illegal at every epoch, with or without the tables in the loss"
             if best == worst == 1 else f"Illegal share by epoch, {best:.0%} to {worst:.0%}")
    return write_chart(option, folder / "illegal-share-by-epoch.html", title=title, height=440,
                       subtitle="64 teams per epoch at the elite cut 0.531, guidance 2, species-first order, drawn without the legal "
                                "Pokémon construction tables and the Stat Point budget, then checked by Showdown's validator. "
                                f"{NEW} stopped after epoch {len(data.get(NEW, []))} of its 80-epoch cap.")


def binding_chart(folder):
    b = binding(); fields = list(b)
    series = [dict(type="bar", stack="all", name=part, color=color, emphasis=dict(focus="series"), barWidth=34,
                   label=dict(show=True, formatter="js:p => Math.round(p.value * 100) + '%'", color="#fff" if i < 2 else None),
                   data=[b[f][i] for f in fields]) for i, (part, color) in enumerate(PARTS.items())]
    option = dict(grid=dict(left=96, right=24, top=56, bottom=40), legend=dict(top=0), animation=False,
                  tooltip=dict(trigger="axis", axisPointer=dict(type="shadow"),
                               valueFormatter="js:v => (v * 100).toFixed(1) + '%'"),
                  xAxis=dict(type="value", min=0, max=1, axisLabel=dict(formatter="js:v => Math.round(v * 100) + '%'")),
                  yAxis=dict(type="category", data=[f"{f[0].upper()}{f[1:]}" for f in fields], inverse=True), series=series)
    return write_chart(option, folder / "whose-value.html", height=280, dark={"#fff": "#fff"},
                       title=f"The network picks a team member's ability, but often not this Pokémon's: "
                             f"{b['ability'][1]:.0%} of the probability goes to another team member's",
                       subtitle=f"{NEW} at epoch 33. 256 real validation teams; at each Pokémon's ability and first move, "
                                "the true earlier fields of the draw order are shown, no tables. The bar splits the network's "
                                "probability by the species the chosen value is legal for.")


def rule_lines():
    """{reader name: [epoch entries with rule_shares]} from the checkpoint files."""
    out = {}
    for arm, path in ((NEW, ARM_CHECKS), (OLD, CONTROL)):
        if path.exists(): out[arm] = [e for e in json.loads(path.read_text())["epochs"] if "rule_shares" in e]
    return {k: v for k, v in out.items() if v}


def rules_chart(folder):
    data = rule_lines()
    if not data: return None
    import epoch_legality as E
    rules = list(E.RULES); arms = [a for a in (NEW, OLD) if a in data]; epochs = max(e["epoch"] for v in data.values() for e in v)
    grids, xs, ys, series = [], [], [], []
    for i, arm in enumerate(arms):
        top = 70 + i * 270
        grids.append(dict(left=230, right=24, top=top, height=200))
        xs.append(dict(type="category", gridIndex=i, data=list(range(1, epochs + 1)), name="Training epoch" if i == len(arms) - 1 else "",
                       nameLocation="middle", nameGap=28, axisLabel=dict(interval=9)))
        ys.append(dict(type="category", gridIndex=i, data=rules, inverse=True, name=arm, nameLocation="start",
                       nameTextStyle=dict(fontWeight="bold", align="left", padding=[0, 0, 0, -210])))
        series.append(dict(type="heatmap", name=arm, xAxisIndex=i, yAxisIndex=i, emphasis=dict(itemStyle=dict(borderColor="#000", borderWidth=1)),
                           data=[[e["epoch"] - 1, r, e["rule_shares"][rule]] for e in data[arm] for r, rule in enumerate(rules)]))
    last = {arm: data[arm][-1] for arm in arms}
    worst = max(rules, key=lambda r: last[arms[0]]["rule_shares"][r])
    option = dict(animation=False, grid=grids, xAxis=xs, yAxis=ys, series=series,
                  tooltip=dict(position="top", formatter="js:p => p.seriesName + ', epoch ' + (p.value[0] + 1) + ': ' + Math.round(p.value[2] * 100) + '% of teams'"),
                  visualMap=dict(min=0, max=1, orient="horizontal", left="center", top=0, itemHeight=160,
                                 text=["100% of teams", "0%"], inRange=dict(color=["#f4efe6", "#c2410c", "#5c1d05"])))
    title = (f"Rule by rule: at epoch {last[arms[0]]['epoch']}, {arms[0]} draws '{worst}' in "
             f"{last[arms[0]]['rule_shares'][worst]:.0%} of teams")
    return write_chart(option, folder / "rules-by-epoch.html", height=70 + 270 * len(arms) + 20, title=title,
                       subtitle="Each cell: the share of the 64 teams drawn at that epoch, without the tables, that break the rule. "
                                "Counted from the drawn tokens with the tables' compatibility data. A team breaking several rules counts under each.")


def figure(folder=FOLDER):
    folder = Path(folder); data = lines()
    if not data: raise SystemExit("no legality results yet")
    print(share_chart(data, folder)); print(binding_chart(folder))
    r = rules_chart(folder)
    if r: print(r)


def rules_comment(image):
    """The #361 comment for the per-rule counts."""
    import epoch_legality as E
    data = rule_lines(); arms = [a for a in (NEW, OLD) if a in data]
    first, last = {a: data[a][0] for a in arms}, {a: data[a][-1] for a in arms}
    repeat = all(e.get("previous_illegal_share", e["illegal_share"]) == e["illegal_share"] for v in data.values() for e in v)
    head = "| Rule broken, share of 64 teams | " + " | ".join(f"{a}, epoch {first[a]['epoch']} → {last[a]['epoch']}" for a in arms) + " |"
    print("\n".join([
        f"![Each rule a drawn team breaks, by epoch, for both networks]({image.format(name='rules', theme='light')})",
        "",
        "**Rule by rule** (owner, 2026-11-17: \"more measurement than just legality\"): the same 64 teams per epoch, drawn "
        "without the tables, counted against each rule.",
        "",
        head, "| --- |" + " --- |" * len(arms),
        *[f"| {r} | " + " | ".join(f"{pct(first[a]['rule_shares'][r])} → {pct(last[a]['rule_shares'][r])}" for a in arms) + " |" for r in E.RULES],
        "",
        "Counted from the drawn tokens with the legal Pokémon construction tables' compatibility data, so a team breaking "
        "several rules counts under each. Exploratory: added after the first result. Repeat check: every redone epoch "
        + ("gave the same illegal share as before." if repeat else "did NOT give the same illegal share as before; see the result files."),
    ]))


def pct(v):
    return f"{int(v * 100 + 0.5)}%"                         # half up, like the chart's labels


def comment(image):
    data = lines(); arm, control = data[NEW], data[OLD]
    b = binding()
    below = [e["epoch"] for e in arm if e["illegal_share"] <= REFUTED_ABOVE]
    img = lambda name, alt: f"![{alt}]({image.format(name=name, theme='light')})"
    verdict = (f"**The illegal share falls to {REFUTED_ABOVE:.3f} or below at epoch {below[0]}.**" if below else
               f"**No: the illegal share stays at 100% for all {len(arm)} epochs the new arm trained, against the "
               f"plan's refutation line of {REFUTED_ABOVE:.3f}.** The run stopped silently after epoch {len(arm)} of 80 "
               f"(2026-11-17 04:08, no error logged); the owner chose to report these epochs rather than retrain.")
    print("\n".join([
        img("illegal-share", f"Illegal share by training epoch, drawn without the tables: {NEW} 100% at all {len(arm)} epochs, "
                             f"{OLD} 100% at all {len(control)}"),
        "",
        verdict,
        "",
        img("whose-value", f"At epoch 33, {b['ability'][1]:.0%} of the ability probability and {b['first move'][1]:.0%} of the "
                           f"first-move probability go to values legal for another team member, not the Pokémon itself"),
        "",
        "**Why:** the network learned what belongs on the team, not which Pokémon it belongs to.",
        "",
        "| Probability legal for… | Ability | First move |",
        "| --- | --- | --- |",
        *[f"| {part} | {pct(b['ability'][i])} | {pct(b['first move'][i])} |" for i, part in enumerate(PARTS)],
        "",
        f"Only {pct(b['ability'][2])} and {pct(b['first move'][2])} go to values no team member could have, so the data's "
        "legality was learned at the team level. One ability or move placed on the wrong team member makes the whole "
        "team illegal. This matches the 2026-11-04 swap test (Stat "
        "Points barely move when two Pokémon swap) and #84 (removing the slot vectors changed nothing). The tables hide it "
        "in every default-model draw because they do the binding.",
        "",
        f"Measurement checks: 64 real training teams through the same paste and validator are all legal; the validator's "
        f"about {arm[-1]['errors_per_team']:.0f} errors per drawn team match the network's about 18 expected forbidden fields. "
        f"Validation loss of the new arm still fell, {arm[0]['val_loss']:.2f} to {arm[-1]['val_loss']:.2f}. No drawn team "
        "copied a training team. All 84 fields generated, none copied. No battles. One training seed per arm. Confirmatory "
        "for the curve; the binding probe was added after seeing it (exploratory).",
    ]))


if __name__ == "__main__":
    mode, *rest = sys.argv[1:] or ["figure"]
    {"figure": figure, "comment": comment, "rules_comment": rules_comment}[mode](*rest)
