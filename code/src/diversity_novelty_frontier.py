"""Diversity against novelty as a Pareto trade-off, over every decoding order the masking-schedule map (#291) battled,
each traced across guidance 1, 2, 4 and 8, with win rate as the third axis.

    diversity  = distinct line-ups among the grid's 4,416 teams (1,472 per sampling seed)
    novelty    = novel share: teams whose six species appear together in no labelled training team of the shuffled-only table
    quality    = pooled win rate against the 49 top-50 meta teams, slots shuffled, all 23 asks

Species-first grids drawn on seeds 7301-7303 replay earlier grids' teams, which are now training teams (#353), so their
diversity and novelty are read from the redraws on seeds of their own (`..._seed<base>.json`, no battles); their win
rate is read from the battled grid, which the replay does not bias. Every other decoder's stored grid escapes the replay.

One point dominates another when it is at least as high on every axis on each of the three sampling seeds and higher on
at least one, seed for seed: the diversity map's rule that a difference counts when all three seeds agree (#263).

    python diversity_novelty_frontier.py read          # ~/vgc-data/results/diversity_novelty_frontier.json
    python diversity_novelty_frontier.py figure DIR    # DIR/frontier-diversity-novelty.html, DIR/frontier-win.html
    python diversity_novelty_frontier.py comment URL HEADLINE   # issue comment; URL has {name} and {theme} placeholders
    python diversity_novelty_frontier.py confidence_headline     # the headline of #363's post
"""
import json
import sys
from pathlib import Path

import numpy as np

import masking_rate_rows as MRR

OUT = MRR.RES / "diversity_novelty_frontier.json"
BASES = Path.home() / "vgc-data/seed_replay/bases.txt"
GUIDANCES = (1, 2, 4, 8)
# reader name (difference first: the decoding order, then the masking schedule the checkpoint trained on) -> code row
ARMS = {"Species-first, linear schedule": "sched_linear_species_first",
        "Species then random order, linear schedule": "sched_linear_species_then_random",
        "Confidence order, linear schedule": "sched_linear_confidence",
        "Random order, linear schedule": "sched_linear_random",
        "Species-first, per-field schedule": "sched_species_first",
        "Random order, per-field schedule": "sched_per_field_random"}
COLOR = {"Species-first, linear schedule": "#2a78d6", "Species then random order, linear schedule": "#1d9e75",
         "Confidence order, linear schedule": "#8a5cd6", "Random order, linear schedule": "#e0702a",
         "Species-first, per-field schedule": "#7fb2ea", "Random order, per-field schedule": "#f0b07a"}
REPLAYED = {"sched_linear_species_first", "sched_species_first"}      # decoders whose stored seeds replay training teams
AXES = ("line_ups", "novel_share", "win")


def redraw_bases():
    return dict(line.split() for line in BASES.read_text().splitlines() if line.strip()) if BASES.exists() else {}


def cells_of(path):
    return list(json.load(open(path))["cells"].values())


def breadth(cells, training):
    """Distinct line-ups and novel share, pooled and per seed."""
    import line_up_yield as LY
    teams = [(c["seed"], LY.line_up(p)) for c in cells for p in c["pastes"]]
    def of(sel):
        lu = [x for s, x in teams if sel(s)]
        return len(set(lu)), float(np.mean([x not in training for x in lu])), len({x for x in lu if x not in training})
    pooled, by = of(lambda _: True), {s: of(lambda t, s=s: t == s) for s in MRR.SEEDS}
    return dict(teams=len(teams), line_ups=pooled[0], novel_share=pooled[1], novel_line_ups=pooled[2],
                line_ups_by_seed={str(s): v[0] for s, v in by.items()}, novel_share_by_seed={str(s): v[1] for s, v in by.items()},
                novel_line_ups_by_seed={str(s): v[2] for s, v in by.items()})


def win(cells):
    ys = lambda sel: [float(y) for c in cells if sel(c["seed"]) for y in c["y"]]
    return dict(win=float(np.mean(ys(lambda _: True))), win_by_seed={str(s): float(np.mean(ys(lambda t, s=s: t == s))) for s in MRR.SEEDS})


def point(arm, row, guidance, training, bases):
    name = MRR.grid_name(row, guidance); stored = MRR.grid_path(row, guidance)
    if not stored.exists() or len(cells_of(stored)) < len(MRR.ALL_ASKS) * len(MRR.SEEDS): return None
    source = stored
    if row in REPLAYED:
        if name not in bases: return dict(arm=arm, row=row, guidance=guidance, missing=f"no redraw of {name} on seeds of its own")
        source = MRR.RES / f"stat_points_84_{name}_seed{bases[name]}.json"
        if not source.exists() or len(cells_of(source)) < len(MRR.ALL_ASKS) * len(MRR.SEEDS):
            return dict(arm=arm, row=row, guidance=guidance, missing=f"{source.name} is not finished")
    held = (MRR.elite_yield(name) or {}).get("all_23_asks", {}).get("held")
    return dict(arm=arm, row=row, guidance=guidance, breadth_from=source.name, win_from=stored.name,
                **breadth(cells_of(source), training), **win(cells_of(stored)), elite_held=held)


def by_seed(p, axis):
    return np.array([p[f"{axis}_by_seed"][str(s)] for s in MRR.SEEDS], float)


def dominates(a, b, axes):
    """a dominates b: at least as high on every axis on each seed, seed for seed, and higher somewhere."""
    ge = all((by_seed(a, x) >= by_seed(b, x)).all() for x in axes)
    gt = any((by_seed(a, x) > by_seed(b, x)).any() for x in axes)
    return ge and gt


def front(points, axes):
    """Labels of the points no other point dominates on these axes, and for each dominated point who dominates it."""
    beaten = {label(p): [label(q) for q in points if q is not p and dominates(q, p, axes)] for p in points}
    return [k for k, v in beaten.items() if not v], {k: v for k, v in beaten.items() if v}


def label(p): return f"{p['arm']}, guidance {p['guidance']:g}"


def read():
    MRR.use_pilot_src()
    training = MRR.references()[0]; bases = redraw_bases()
    pts = [point(arm, row, g, training, bases) for arm, row in ARMS.items() for g in GUIDANCES]
    missing = [p for p in pts if p and "missing" in p]; pts = [p for p in pts if p and "missing" not in p]
    result = dict(question="Diversity (distinct line-ups) against novelty (novel share), with win rate as the third axis",
                  reference=f"line-ups of the labelled training teams of {MRR.TABLE} ({len(training):,})",
                  dominance="seed for seed on all three sampling seeds", points=pts, missing=missing, fronts={})
    for key, axes in dict(diversity_novelty=("line_ups", "novel_share"), diversity_win=("line_ups", "win"),
                          novelty_win=("novel_share", "win"), all_three=AXES).items():
        on, beaten = front(pts, axes); result["fronts"][key] = dict(axes=list(axes), on_front=on, dominated_by=beaten)
    OUT.write_text(json.dumps(result, indent=1))
    for p in pts:
        print(f"{label(p):58s} line-ups {p['line_ups']:5d}  novel {p['novel_share']:.3f}  win {p['win']:.3f}  held {p['elite_held']}")
    for k, f in result["fronts"].items(): print(f"\n{k}: on the front\n  " + "\n  ".join(f["on_front"]))
    for p in missing: print("missing:", p["missing"])
    print(f"\nwrote {OUT}")


# ---------------------------------------------------------------- figures
WHISKER = """js:(params, api) => {
  const [x, y, x0, x1, y0, y1] = [0, 1, 2, 3, 4, 5].map(i => api.value(i));
  const p = api.coord([x, y]), a = api.coord([x0, y]), b = api.coord([x1, y]), c = api.coord([x, y0]), d = api.coord([x, y1]);
  const style = {stroke: api.visual('color'), lineWidth: 1.2, opacity: 0.55};
  return {type: 'group', children: [{type: 'line', shape: {x1: a[0], y1: a[1], x2: b[0], y2: b[1]}, style},
                                     {type: 'line', shape: {x1: c[0], y1: c[1], x2: d[0], y2: d[1]}, style}]};
}"""
TIP = "js:p => p.data && p.data.tip ? p.data.tip : ''"


def tip(p):
    held = "not re-battled" if p["elite_held"] is None else f"{p['elite_held']}"
    seeds = lambda axis, f: " / ".join(f.format(v) for v in by_seed(p, axis))
    return (f"<b>{p['arm']}</b><br>guidance {p['guidance']:g}<br>"
            f"distinct line-ups {p['line_ups']:,} (seeds {seeds('line_ups', '{:.0f}')})<br>"
            f"novel share {p['novel_share']:.1%} (seeds {seeds('novel_share', '{:.1%}')})<br>"
            f"win rate {p['win']:.3f} (seeds {seeds('win', '{:.3f}')})<br>elite line-ups that held {held}")


def series_for(points, xa, ya, grid=0):
    out = []
    for arm in ARMS:
        mine = sorted((p for p in points if p["arm"] == arm), key=lambda p: p["guidance"])
        if not mine: continue
        common = dict(name=arm, color=COLOR[arm], xAxisIndex=grid, yAxisIndex=grid)
        # seed range; per-seed line-up counts (of 1,472 teams) are scaled so their mean is the pooled count (of 4,416)
        xs = lambda p, a: by_seed(p, a) * (100 if a == "novel_share" else p[a] / by_seed(p, a).mean() if a == "line_ups" else 1)
        val = lambda p, a: p[a] * (1 if a != "novel_share" else 100)
        out.append({**common, "type": "line", "symbolSize": 10, "emphasis": {"focus": "series"},
                    "label": {"show": True, "formatter": "js:p => 'g' + p.data.g", "position": "right", "fontSize": 10, "color": COLOR[arm]},
                    "data": [{"value": [val(p, xa), val(p, ya)], "g": f"{p['guidance']:g}", "tip": tip(p)} for p in mine]})
        out.append({**common, "type": "custom", "renderItem": WHISKER, "silent": True, "tooltip": {"show": False}, "z": 1,
                    "data": [[val(p, xa), val(p, ya), xs(p, xa).min(), xs(p, xa).max(), xs(p, ya).min(), xs(p, ya).max()] for p in mine]})
    return out


def front_rings(points, on, xa, ya, grid=0):
    """A ring around each point on the front. Seed-for-seed dominance does not follow the pooled point estimates, so the
    front is marked point by point instead of drawn as a staircase."""
    scale = lambda a: 100 if a == "novel_share" else 1
    return {"type": "scatter", "name": "On the Pareto front", "symbol": "circle", "symbolSize": 22, "z": 0, "silent": True,
            "itemStyle": {"color": "transparent", "borderColor": "#6f6e69", "borderWidth": 1.6, "borderType": "dashed"},
            "tooltip": {"show": False}, "xAxisIndex": grid, "yAxisIndex": grid,
            "data": [[p[xa] * scale(xa), p[ya] * scale(ya)] for p in points if label(p) in on]}


AXIS = {"line_ups": "Distinct line-ups (of 4,416 teams)", "novel_share": "Novel share (% of teams)", "win": "Win rate against the top-50 meta teams"}


def axis(a, grid=0, **kw):
    out = {"type": "value", "name": AXIS[a], "nameLocation": "middle", "nameGap": 34, "scale": True, "gridIndex": grid, **kw}
    if a == "novel_share": out["axisLabel"] = {"formatter": "{value}%"}
    return out


def short(p):
    """'confidence order g2', 'random order g8 (per-field)': the order, its guidance, and the schedule only when per-field."""
    order, schedule = p["arm"].split(", ")
    return f"{order[0].lower() + order[1:]} g{p['guidance']:g}" + (" (per-field)" if schedule.startswith("per-field") else "")


def short_list(points, on):
    names = [short(p) for p in sorted(points, key=lambda p: (list(ARMS).index(p["arm"]), p["guidance"])) if label(p) in on]
    return ", ".join(names[:-1]) + (" and " if len(names) > 1 else "") + names[-1]


def ends(points, on):
    """The front's highest-win point and its broadest point."""
    mine = [p for p in points if label(p) in on]
    return f"{short(max(mine, key=lambda p: p['win']))}, with the most wins, to {short(max(mine, key=lambda p: p['line_ups']))}, with the most line-ups"


def figure(folder):
    from echarts_chart import write_chart
    r = json.load(open(OUT)); pts = r["points"]; folder = Path(folder)
    legend = {"top": 0, "left": 0, "itemGap": 14, "data": [*[a for a in ARMS if any(p["arm"] == a for p in pts)], "On the Pareto front"]}
    tooltip = {"trigger": "item", "formatter": TIP}
    on = r["fronts"]["diversity_novelty"]["on_front"]
    write_chart({"legend": legend, "tooltip": tooltip, "grid": {"top": 96, "left": 64, "right": 40, "bottom": 56},
                 "xAxis": axis("line_ups"), "yAxis": axis("novel_share", nameGap=44),
                 "series": [front_rings(pts, on, "line_ups", "novel_share"), *series_for(pts, "line_ups", "novel_share")]},
                folder / "frontier-diversity-novelty.html", height=560,
                title="On the diversity–novelty front: " + short_list(pts, on),
                subtitle="Each line is one decoding order traced across guidance (g1 to g8). Whiskers span the three sampling seeds "
                         "(line-ups: each seed's count, scaled to the pooled count). "
                         "Up and right is broader on both axes. Ringed: on the front.",
                note=f"Novel share: teams whose six species appear together in no labelled training team ({r['reference']}). "
                     "Species-first breadth is read from redraws on seeds of their own (#353); every other grid as stored. "
                     "On the front: no other point is at least as high on both axes on each of the three seeds, seed for seed.")
    on3 = r["fronts"]["all_three"]["on_front"]
    grids = [{"top": 96, "left": 64, "right": "53%", "bottom": 56}, {"top": 96, "left": "55%", "right": 24, "bottom": 56}]
    write_chart({"legend": legend, "tooltip": tooltip, "grid": grids,
                 "xAxis": [axis("line_ups", 0), axis("novel_share", 1)], "yAxis": [axis("win", 0, nameGap=44), axis("win", 1, nameGap=44)],
                 "series": [front_rings(pts, r["fronts"]["diversity_win"]["on_front"], "line_ups", "win", 0), *series_for(pts, "line_ups", "win", 0),
                            front_rings(pts, r["fronts"]["novelty_win"]["on_front"], "novel_share", "win", 1), *series_for(pts, "novel_share", "win", 1)]},
                folder / "frontier-win.html", height=520,
                title="Win rate against breadth: the front of all three axes runs from " + ends(pts, on3),
                subtitle="Left: win rate against distinct line-ups. Right: win rate against novel share. Ringed: on that plane's Pareto front.",
                note="On the front of all three axes: " + short_list(pts, on3) + ".")
    print(f"wrote {folder}/frontier-diversity-novelty.html and frontier-win.html")


def comment(url, headline):
    """The issue comment: headline, both pictures (url has {name} and {theme} placeholders), definitions, numbers folded."""
    r = json.load(open(OUT)); pts = r["points"]
    pic = lambda name, alt: (f'<picture><source media="(prefers-color-scheme: dark)" srcset="{url.format(name=name, theme="dark")}">'
                             f'<img alt="{alt}" src="{url.format(name=name, theme="light")}"></picture>')
    rows = "\n".join(f"| {p['arm']} | {p['guidance']:g} | {p['line_ups']:,} | {p['novel_share']:.1%} | {p['win']:.3f} | "
                      f"{'not re-battled' if p['elite_held'] is None else p['elite_held']} | "
                      f"{'yes' if label(p) in r['fronts']['diversity_novelty']['on_front'] else ''} | "
                      f"{'yes' if label(p) in r['fronts']['all_three']['on_front'] else ''} |" for p in pts)
    missing = "".join(f"\n- Not on the charts yet: {m['missing']}." for m in r["missing"])
    print(f"""{headline}

{pic("frontier-diversity-novelty", "Distinct line-ups against novel share, one line per decoding order across guidance")}

{pic("frontier-win", "Win rate against distinct line-ups and against novel share")}

- Diversity is the number of distinct line-ups among a grid's 4,416 teams (23 asks × 3 sampling seeds × 64 teams).
- Novelty is the novel share: teams whose six species appear together in no labelled training team of the shuffled-only table.
- Win rate is pooled over all 23 asks, from one battle against each of the 49 top-50 meta teams, slots shuffled.
- A point is on the front when no other point is at least as high on every axis on each of the three sampling seeds, seed for seed. The training-split noise floor is not applied yet.
- Species-first grids give diversity and novelty from redraws on seeds of their own (#353) and win rate from the battled grid.{missing}

<details><summary>Every point</summary>

| Decoding order, schedule | Guidance | Distinct line-ups | Novel share | Win rate | Elite line-ups that held | Diversity–novelty front | Front of all three |
|---|---|---|---|---|---|---|---|
{rows}

</details>

Script: `code/src/diversity_novelty_frontier.py`. Results: `~/vgc-data/results/diversity_novelty_frontier.json`.""")


def confidence_headline():
    """The headline of the guidance 4 and 8 post (#363), with its prediction checked: guidance 8 on the front of all
    three axes and a win rate above guidance 2's."""
    r = json.load(open(OUT)); on3 = r["fronts"]["all_three"]["on_front"]
    conf = {p["guidance"]: p for p in r["points"] if p["row"] == "sched_linear_confidence"}
    if not {2, 4, 8} <= set(conf): raise SystemExit(f"confidence order has guidance {sorted(conf)} only")
    g8 = conf[8]; held = label(g8) in on3 and g8["win"] > conf[2]["win"]
    parts = "; ".join(f"guidance {g:g}: {p['line_ups']:,} line-ups, {p['novel_share']:.0%} novel, win rate {p['win']:.3f}"
                      f"{', on the front of all three axes' if label(p) in on3 else ''}" for g, p in sorted(conf.items()))
    print(f"**Confidence order across guidance, {__import__('datetime').date.today()}: the prediction {'held' if held else 'missed'}.** "
          f"{parts}. Prediction, written before the run: guidance 8 lands on the front of all three axes and wins more than guidance 2's {conf[2]['win']:.3f}.")


if __name__ == "__main__":
    {"read": read, "figure": lambda: figure(sys.argv[2]), "comment": lambda: comment(sys.argv[2], sys.argv[3]),
     "confidence_headline": confidence_headline}[sys.argv[1]]()
