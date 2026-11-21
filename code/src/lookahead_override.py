"""How often does the lookahead override the model's draw in Linear, random order? (GitHub issue #321)

Redraws every cell of the three stored Linear, random order grids (masking_rate_rows.py, guidance 2, 4 and 8) from its
stored sampling seed with every draw recorded (diffusion84.Refusals, draws=True), checks the kept teams are the stored
ones, and reads the record against each team's stored win rate. Sampling only: no battles.

    python lookahead_override.py redraw G        guidance G, one file per cell in results/lookahead_override/ (resumes)
    python lookahead_override.py report          the readings and the verdict, into results/lookahead_override.json
    python lookahead_override.py figure DIR      the chart as an ECharts page in DIR
    python lookahead_override.py comment URL     the verdict comment; URL holds {theme} for light or dark

LO_SMOKE=1 redraws only the first cell of the grid. The bars were posted on the ticket before any count:
https://github.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/issues/321#issuecomment-6887486171
"""
import datetime
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import masking_rate_rows as MRR

ROW = "sched_linear_random"
SMOKE = bool(os.environ.get("LO_SMOKE"))
FOLDER = MRR.RES / f"lookahead_override{'_smoke' if SMOKE else ''}"
RESULT = MRR.RES / f"lookahead_override{'_smoke' if SMOKE else ''}.json"
CONFIRM_SHARE, CLEAR_SHARE, TOP_MAJORITY = 0.5, 0.1, 0.5   # the bars posted on the ticket
DRAWS, DRAW_SEED = MRR.DRAWS, 321
NOT_SPECIES = -1                                         # fields_before at a draw that is not a species


def cell_path(guidance, k, seed): return FOLDER / f"g{guidance:g}_a{k:02d}_s{seed}.npz"


def match(drawn, stored):
    """Positions in drawn of the stored pastes, in order (each the first match after the last), or None where one is missing."""
    out, i = [], 0
    for p in stored:
        while i < len(drawn) and drawn[i] != p: i += 1
        if i == len(drawn): return None
        out.append(i); i += 1
    return out


def arrays(records):
    """The per-draw records of a list of teams as [teams, draws] arrays, plus the per-team record."""
    d = [r["draws"] for r in records]
    return dict(column=np.array([x["column"] for x in d], np.int16),
                removed_share=np.array([x["removed_share"] for x in d], np.float32),
                top_removed=np.array([x["top_removed"] for x in d], bool),
                fields_before=np.array([[NOT_SPECIES if b is None else b for b in x["fields_before"]] for x in d], np.int8),
                log_kept_share=np.array([-np.inf if r["log_kept_share"] is None else r["log_kept_share"] for r in records]),
                fallback_fields=np.array([r["fallback_fields"] for r in records], np.int16))


def redraw(guidance):
    guidance = float(guidance)
    os.environ["AG84_MODEL"], os.environ["AG84_GUIDANCE"] = ROW, str(guidance)
    import torch
    import ask_grid84 as A
    import sampler84 as S
    stored = json.load(open(MRR.grid_path(ROW, guidance)))
    if stored["checkpoint_sha256"] != A.require_same_checkpoint({}): raise SystemExit("the checkpoint is not the one the stored grid drew with")
    if stored.get("decoder") != "random" or float(stored["guidance"]) != guidance: raise SystemExit("the stored grid is not Linear, random order at this guidance")
    FOLDER.mkdir(parents=True, exist_ok=True)
    arm = S.Sampler84(A.load_model(), "rand_plain", lookahead=True, record_draws=True)
    cells = sorted(stored["cells"].values(), key=lambda c: (c["seed"], c["numerator"]))   # the order the grid drew them
    for c in cells[:1] if SMOKE else cells:
        k, seed, path = c["numerator"], c["seed"], cell_path(guidance, c["numerator"], c["seed"])
        if path.exists(): continue
        torch.manual_seed(A.SEED0 + seed)                 # as ask_grid84.draw does, batch by batch, until the stored tries are covered
        drawn, records = [], []
        while len(drawn) < c["tried"]:
            rows = arm.sample(A.BATCH, ask=k / 24, guidance=guidance).cpu().numpy()
            drawn += [arm.to_paste(r, repair=True)[0] for r in rows]; records += arm.last_refusals
        at = match(drawn[:c["tried"]], c["pastes"])
        matched = at is not None and len(at) == len(c["pastes"]) and (len(at) < A.KEEP or at[-1] == c["tried"] - 1)
        keep = at if matched else list(range(c["tried"]))     # a mismatch keeps every try, read without win rates
        tmp = path.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, matched=matched, tried=c["tried"], drawn_index=np.array(keep), **arrays([records[i] for i in keep]))
        os.replace(tmp, path)
        print(f"  g{guidance:g} a{k:02d} s{seed}: {'matched' if matched else 'MISMATCH'} {len(c['pastes'])} kept of {c['tried']} tried", flush=True)
    print("LOOKAHEAD_OVERRIDE_REDRAW_DONE")


def load(guidance):
    """{(ask numerator, seed): cell arrays with the stored win rate y}, or a SystemExit naming the cells not redrawn."""
    stored = MRR.load_grid(ROW, guidance)
    if stored is None: raise SystemExit(f"{MRR.grid_name(ROW, guidance)}: the stored grid is missing or not battled")
    cells, missing = {}, []
    for (k, seed), c in sorted(stored["by_ask_seed"].items()):
        path = cell_path(guidance, k, seed)
        if not path.exists(): missing.append(f"a{k:02d}_s{seed}"); continue
        z = dict(np.load(path)); z["matched"] = bool(z["matched"])
        z["y"] = np.array(c["y"], float) if z["matched"] else None
        cells[(k, seed)] = z
    if missing and not SMOKE: raise SystemExit(f"guidance {guidance:g}: cells not redrawn: {', '.join(missing)}")
    return cells


def draw_kinds(z):
    """Boolean [teams, draws] masks: late species, early species and set draws."""
    species = z["fields_before"] != NOT_SPECIES
    return dict(late_species=z["fields_before"] >= 1, early_species=z["fields_before"] == 0, set=~species)


def share(z, mask, key):
    """Per-team sums and counts of a per-draw value over the draws in mask."""
    v = z[key].astype(float)
    return (v * mask).sum(1), mask.sum(1)


def ratio_interval(cells, kind, key, rng):
    """Mean of key over the draws of a kind, pooled over teams, with a 95% interval: teams resampled within each cell."""
    sums, counts = np.zeros(DRAWS), np.zeros(DRAWS); s_all = n_all = 0.0
    for z in cells.values():
        s, n = share(z, draw_kinds(z)[kind], key); s_all += s.sum(); n_all += n.sum()
        pick = rng.integers(0, len(s), (DRAWS, len(s))); sums += s[pick].sum(1); counts += n[pick].sum(1)
    return dict(mean=float(s_all / n_all) if n_all else None, interval=MRR.interval(sums / np.maximum(counts, 1)) if n_all else None, draws=int(n_all))


def team_removed(z):
    """[teams] each team's mean removed share over its six species draws."""
    s, n = share(z, z["fields_before"] != NOT_SPECIES, "removed_share")
    return s / n


def slope(cells, rng):
    """Within-cell slope of win rate on team's removed share, both centred on the cell mean, and its bootstrap interval."""
    sxy = sxx = 0.0; bxy, bxx = np.zeros(DRAWS), np.zeros(DRAWS)
    for z in cells.values():
        if z["y"] is None: continue
        x, y = team_removed(z), z["y"]
        xc, yc = x - x.mean(), y - y.mean(); sxy += (xc * yc).sum(); sxx += (xc * xc).sum()
        pick = rng.integers(0, len(x), (DRAWS, len(x))); xb, yb = x[pick], y[pick]
        xb = xb - xb.mean(1, keepdims=True); yb = yb - yb.mean(1, keepdims=True)
        bxy += (xb * yb).sum(1); bxx += (xb * xb).sum(1)
    return dict(slope=float(sxy / sxx), interval=MRR.interval(bxy / bxx))


def verdict(late_share, late_top, fall):
    """The bars on the ticket. fall: the slope's interval sits below zero."""
    if (late_share >= CONFIRM_SHARE or late_top > TOP_MAJORITY) and fall: return "confirmed"
    if late_share < CLEAR_SHARE and not fall: return "cleared"
    return "partial"


BANDS = [("None", 0.0, 0.0), ("Under 1%", 0.0, 0.01), ("1% to 5%", 0.01, 0.05), ("5% or more", 0.05, 1.0)]


def by_bands(cells, rng):
    """Teams in fixed bands of their removed share: count, and mean win rate minus their cell's mean with a 95% interval
    (teams resampled within the band). A band is (name, above, up to and including); the first holds exactly 0."""
    x, y = [], []
    for z in cells.values():
        if z["y"] is None: continue
        x.append(team_removed(z)); y.append(z["y"] - z["y"].mean())
    x, y = np.concatenate(x), np.concatenate(y); out = []
    for name, lo, hi in BANDS:
        m = x == 0 if hi == 0 else (x > lo) & (x <= hi); yb = y[m]
        draws = yb[rng.integers(0, len(yb), (DRAWS, len(yb)))].mean(1) if len(yb) else None
        out.append(dict(band=name, teams=int(m.sum()), win_rate_against_cell=float(yb.mean()) if len(yb) else None,
                        interval=MRR.interval(draws) if len(yb) else None))
    return out


def read(guidance):
    """The readings at one guidance value over the cells that reproduce the stored teams. A mismatched cell holds every
    try rather than the kept teams, so it is listed and read apart, without win rates."""
    every = load(guidance); rng = np.random.default_rng([DRAW_SEED, int(guidance)])
    cells = {key: z for key, z in every.items() if z["matched"]}
    apart = {key: z for key, z in every.items() if not z["matched"]}
    out = dict(cells=len(every), matched_cells=len(cells), mismatched=[f"a{k:02d}_s{s}" for (k, s) in apart],
               mismatched_late_species=ratio_interval(apart, "late_species", "removed_share", rng) if apart else None,
               teams=int(sum(len(z["removed_share"]) for z in cells.values())),
               fallback_draws=int(sum(z["fallback_fields"].sum() for z in cells.values())))
    for kind in ("late_species", "early_species", "set"):
        out[kind] = dict(removed_share=ratio_interval(cells, kind, "removed_share", rng), top_removed=ratio_interval(cells, kind, "top_removed", rng))
    species = int(sum((z["fields_before"] != NOT_SPECIES).sum() for z in cells.values()))
    out["late_share_of_species_draws"] = out["late_species"]["removed_share"]["draws"] / float(species)
    before = defaultdict(lambda: [0.0, 0.0, 0])
    asks, set_asks = defaultdict(lambda: [0.0, 0]), defaultdict(lambda: [0.0, 0])
    for (k, _), z in cells.items():
        fb, rs, top = z["fields_before"], z["removed_share"], z["top_removed"]
        for b in range(0, 14):
            m = fb == b; before[b][0] += float(rs[m].sum()); before[b][1] += int(top[m].sum()); before[b][2] += int(m.sum())
        m = fb >= 1; asks[k][0] += float(rs[m].sum()); asks[k][1] += int(m.sum())
        m = fb == NOT_SPECIES; set_asks[k][0] += float(rs[m].sum()); set_asks[k][1] += int(m.sum())
    if sum(v[2] for v in before.values()) != species: raise SystemExit("some species draws fall outside 0 to 13 fields before")
    out["by_fields_before"] = {str(b): dict(removed_share=v[0] / v[2], top_removed=v[1] / v[2], draws=v[2]) for b, v in sorted(before.items()) if v[2]}
    out["late_species_by_ask"] = {str(k): v[0] / v[1] for k, v in sorted(asks.items()) if v[1]}
    out["set_by_ask"] = {str(k): v[0] / v[1] for k, v in sorted(set_asks.items()) if v[1]}
    out["win_rate_slope"] = slope(cells, rng)
    out["win_rate_by_removed_share"] = by_bands(cells, rng)
    fall = out["win_rate_slope"]["interval"][1] < 0
    out["verdict"] = verdict(out["late_species"]["removed_share"]["mean"], out["late_species"]["top_removed"]["mean"], fall)
    return out


def report():
    result = dict(row=ROW, reader="Linear, random order", draws=DRAWS, draw_seed=DRAW_SEED,
                  bars=dict(confirmed=f"mean removed share at late species draws >= {CONFIRM_SHARE}, or the top species removed at more than {TOP_MAJORITY:.0%} of them; and the win-rate slope's interval below zero",
                            cleared=f"mean removed share at late species draws < {CLEAR_SHARE}, and the win-rate slope's interval not below zero",
                            partial="anything else"),
                  win_rate_definition="one battle against each of the 49 distinct top-50 meta teams, behaviour-cloning policy on both sides, slots shuffled",
                  guidance={})
    for g in MRR.GUIDANCES: result["guidance"][str(g)] = read(g)
    words = {v["verdict"] for v in result["guidance"].values()}
    result["verdict"] = words.pop() if len(words) == 1 else "partial"
    json.dump(result, open(RESULT, "w"), indent=1)
    for g, v in result["guidance"].items():
        ls, lt = v["late_species"]["removed_share"], v["late_species"]["top_removed"]
        print(f"guidance {g}: {v['matched_cells']}/{v['cells']} cells matched, {v['teams']} teams; late species draws {v['late_share_of_species_draws']:.1%} of species draws; "
              f"removed {ls['mean']:.3f} {ls['interval']}, top removed {lt['mean']:.3f}; early species removed {v['early_species']['removed_share']['mean']:.3f}; "
              f"set removed {v['set']['removed_share']['mean']:.3f}, top {v['set']['top_removed']['mean']:.3f}; slope {v['win_rate_slope']}; {v['verdict']}")
    print("VERDICT", result["verdict"]); print("LOOKAHEAD_OVERRIDE_REPORT_DONE")


COLORS = {"2": "#2a78d6", "4": "#e0702a", "8": "#1baf7a"}


def figure(folder):
    """Top: removed share at late species draws by ask, set draws dotted, with the two bars. Bottom: win rate of teams
    in each band of their removed share, a dot per guidance with its 95% interval. Stacked, so it reads at phone width."""
    from echarts_chart import write_chart
    r = json.load(open(RESULT)); gs = list(r["guidance"]); series = []
    pct = "js:v => Math.round(v * 100) + '%'"
    offset = {g: (i - (len(gs) - 1) / 2) * 0.22 for i, g in enumerate(gs)}
    for g in gs:
        v = r["guidance"][g]; bands = v["win_rate_by_removed_share"]
        series.append(dict(type="line", name=f"Guidance {g}", color=COLORS[g], xAxisIndex=0, yAxisIndex=0, symbolSize=6, emphasis=dict(focus="series"),
                           data=[[int(k) / 24, s] for k, s in v["late_species_by_ask"].items()],
                           markLine=None if g != gs[0] else dict(silent=True, symbol="none", lineStyle=dict(color="#6f6e69", type="dashed", width=1),
                                                                 label=dict(position="insideEndTop", formatter="{b}"),
                                                                 data=[dict(yAxis=CONFIRM_SHARE, name="Confirm bar"), dict(yAxis=CLEAR_SHARE, name="Clear bar")])))
        series.append(dict(type="line", name=f"Guidance {g}", color=COLORS[g], xAxisIndex=0, yAxisIndex=0, symbol="none", emphasis=dict(focus="series"),
                           lineStyle=dict(type="dotted", width=1.5), data=[[int(k) / 24, s] for k, s in v["set_by_ask"].items()],
                           endLabel=None if g != gs[-1] else dict(show=True, formatter="Set draws", fontSize=12)))
        whiskers = []
        for i, band in enumerate(bands): whiskers += [[i + offset[g], band["interval"][0]], [i + offset[g], band["interval"][1]], "-"] if band["interval"] else []
        series.append(dict(type="line", name=f"Guidance {g}", color=COLORS[g], xAxisIndex=1, yAxisIndex=1, symbol="none", silent=True,
                           tooltip=dict(show=False), lineStyle=dict(width=2), data=whiskers))
        series.append(dict(type="scatter", name=f"Guidance {g}", color=COLORS[g], xAxisIndex=1, yAxisIndex=1, symbolSize=11, emphasis=dict(focus="series"),
                           data=[[i + offset[g], band["win_rate_against_cell"], band["teams"]] for i, band in enumerate(bands)]))
    series = [{k: v for k, v in s.items() if v is not None} for s in series]
    names = json.dumps([b[0] for b in BANDS])
    option = dict(grid=[dict(left=64, right=72, top=72, height=250), dict(left=64, right=24, top=430, height=250)],
                  legend=dict(top=0, data=[f"Guidance {g}" for g in gs]),
                  tooltip=dict(trigger="item", formatter="js:p => p.seriesType === 'scatter' ? `${p.seriesName}: ${(p.value[1] >= 0 ? '+' : '') + p.value[1].toFixed(3)}, ${p.value[2]} teams` "
                                                         ": `${p.seriesName}: ${(p.value[1] * 100).toFixed(1)}% at ask ${Math.round(p.value[0] * 24)}/24`"),
                  xAxis=[dict(type="value", gridIndex=0, min=0, max=1, interval=1 / 6, name="Ask", nameLocation="middle", nameGap=30, axisLabel=dict(formatter=pct)),
                         dict(type="value", gridIndex=1, min=-0.5, max=len(BANDS) - 0.5, splitLine=dict(show=False), axisTick=dict(customValues=list(range(len(BANDS)))),
                              name="Team's removed share, mean over its six species draws", nameLocation="middle", nameGap=30,
                              axisLabel=dict(customValues=list(range(len(BANDS))), formatter=f"js:v => ({names})[Math.round(v)] || ''"))],
                  yAxis=[dict(type="value", gridIndex=0, min=0, max=0.6, name="Removed share",
                              nameLocation="end", nameTextStyle=dict(align="left"), axisLabel=dict(formatter=pct)),
                         dict(type="value", gridIndex=1, name="Win rate against its cell", nameLocation="end", nameTextStyle=dict(align="left"), axisLine=dict(onZero=False),
                              axisLabel=dict(formatter="js:v => (v > 0 ? '+' : '') + v.toFixed(2)"))],
                  series=series, animation=False)
    g2 = r["guidance"]["2"]["late_species"]
    shares = [r["guidance"][g]["late_species"]["removed_share"]["mean"] for g in gs]
    falls = [g for g in gs if r["guidance"][g]["win_rate_slope"]["interval"][1] < 0]
    title = {"confirmed": "Confirmed: the lookahead overrides most of the model's late species choices, and those teams win less",
             "cleared": f"Cleared: the lookahead removes {min(shares):.1%} to {max(shares):.1%} of the model's late species choice",
             "partial": f"Partial: the lookahead removes only {min(shares):.1%} to {max(shares):.1%} of the model's late species choice, "
                        f"yet at guidance {', '.join(falls) or 'none'} the teams it steered more win less"}[r["verdict"]]
    return write_chart(option, Path(folder) / "lookahead-override.html", height=740, title=title,
                       subtitle=f"Linear, random order. A late species draw comes after one of its Pokémon's other fields. At guidance 2 the lookahead removed "
                                f"{g2['removed_share']['mean']:.1%} of the model's probability there, and the model's top species at {g2['top_removed']['mean']:.1%} of them.",
                       note="Removed share: the model's probability, after the legal Pokémon construction tables, on values the lookahead refuses. "
                            "Top: solid lines, late species draws; dotted, every other field. Bottom: win rate minus the mean of the teams at the same ask and "
                            "sampling seed; dot, mean over the teams in the band; line, its 95% bootstrap interval. Win rate is one battle against each of the 49 distinct "
                            "top-50 meta teams, behaviour-cloning policy on both sides, slots shuffled.")


def comment(figure_url):
    """The verdict comment, picture first. figure_url has {theme} where light or dark goes."""
    r = json.load(open(RESULT)); gv = r["guidance"]; gs = list(gv)
    late = {g: v["late_species"] for g, v in gv.items()}; slopes = {g: v["win_rate_slope"] for g, v in gv.items()}
    band = {g: v["win_rate_by_removed_share"][-1] for g, v in gv.items()}
    iv = lambda x: f"[{x[0]:+.2f}, {x[1]:+.2f}]"
    signed = lambda x: "none (no teams)" if x is None else format(x, "+.3f")
    teams, matched, cells = sum(v["teams"] for v in gv.values()), sum(v["matched_cells"] for v in gv.values()), sum(v["cells"] for v in gv.values())
    words = ", ".join(f"guidance {g} {v['verdict']}" for g, v in gv.items())
    falls = [g for g in gs if slopes[g]["interval"][1] < 0]
    pick = lambda f: ", ".join(f(g) for g in gs)
    removed = pick(lambda g: format(late[g]["removed_share"]["mean"], ".1%"))
    few = pick(lambda g: str(band[g]["teams"]) + " of " + format(gv[g]["teams"], ","))
    slope_text = pick(lambda g: format(slopes[g]["slope"], "+.2f") + " " + iv(slopes[g]["interval"]))
    lines = [f"**Verdict {datetime.date.today()}: {r['verdict']} ({words}). The lookahead almost never overrides the model's species: at species drawn after "
             f"their Pokémon's other fields it removed {min(late[g]['removed_share']['mean'] for g in gs):.1%} to {max(late[g]['removed_share']['mean'] for g in gs):.1%} "
             "of the model's probability.**", "",
             f'<picture><source media="(prefers-color-scheme: dark)" srcset="{figure_url.format(theme="dark")}"><img alt="Removed share by ask, and win rate by removed share" src="{figure_url.format(theme="light")}"></picture>', "",
             f"- **Reproduced:** {matched} of {cells} cells redrawn from their stored sampling seeds gave back their stored teams, {teams:,} teams in all. Zero new battles.",
             f"- **Late species draws**, a species drawn after at least one of its Pokémon's other 13 fields, are {gv[gs[0]]['late_share_of_species_draws']:.0%} of species draws. "
             f"There the lookahead removed {removed} "
             f"of the model's probability at guidance {', '.join(gs)}, and the model's top species at {pick(lambda g: format(late[g]['top_removed']['mean'], '.1%'))} of them. "
             f"At every other field it removed {pick(lambda g: format(gv[g]['set']['removed_share']['mean'], '.1%'))}.",
             f"- **Win rate** falls with the team's removed share at guidance {', '.join(falls) or 'none'}: slope {slope_text} "
             f"at guidance {', '.join(gs)}. The fall comes from the few teams with 5% or more removed ({few}), which win "
             f"{pick(lambda g: signed(band[g]['win_rate_against_cell']))} against the other teams at their ask and sampling seed.",
             "- **Why so little:** the legal Pokémon construction tables, which the model trained under, already give zero odds to every species that cannot have the moves, "
             "item or ability already drawn, before the lookahead runs. At a species draw the lookahead itself removes only a species that another hidden Pokémon needs as "
             "its last option. The species-removing step this ticket pictured is the tables' work, inside the model's own distribution.", "",
             "So the lookahead does not explain why random order's ask-versus-got error was 4.8 to 7.2 times species-first's. "
             "[Does Linear, random order draw weaker teams because it draws species late?](https://github.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/issues/322) "
             "found that drawing the species first restores species-first's results. With the lookahead's pull on species near 1%, that gain comes from when the species "
             "are drawn, not from the lookahead.", "",
             "Bars, [posted before the count](https://github.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/issues/321#issuecomment-6887486171): confirmed needs a removed "
             "share of 0.5 or more (or the top species removed at most late species draws) and a falling win rate; cleared needs a removed share under 0.1 and no fall.", "",
             "<details><summary>Every reading</summary>", "",
             "| Guidance | Cells reproduced | Teams | Late species: removed, top removed | Early species: removed, top removed | Other fields: removed, top removed | Draws with no legal value |",
             "|---|---|---|---|---|---|---|"]
    for g, v in gv.items():
        f = lambda kind: f"{v[kind]['removed_share']['mean']:.3f}, {v[kind]['top_removed']['mean']:.1%}"
        lines.append(f"| {g} | {v['matched_cells']} of {v['cells']} | {v['teams']} | {f('late_species')} | {f('early_species')} | {f('set')} | {v['fallback_draws']} |")
    lines += ["", "Win rate against the team's cell, by the team's removed share (mean over its six species draws), with 95% intervals:", "",
              "| Guidance | " + " | ".join(b[0] for b in BANDS) + " |", "|---|" + "---|" * len(BANDS)]
    for g, v in gv.items():
        lines.append(f"| {g} | " + " | ".join("no teams" if b["interval"] is None else f"{b['win_rate_against_cell']:+.3f} [{b['interval'][0]:+.3f}, {b['interval'][1]:+.3f}], {b['teams']} teams"
                                               for b in v["win_rate_by_removed_share"]) + " |")
    fb = gv[gs[0]]["by_fields_before"]
    lines += ["", f"Removed share by how many of the Pokémon's other fields were drawn before its species, guidance {gs[0]}:", "",
              "| Fields before | " + " | ".join(fb) + " |", "|---|" + "---|" * len(fb),
              "| Removed share | " + " | ".join(f"{b['removed_share']:.3f}" for b in fb.values()) + " |",
              "| Species draws | " + " | ".join(str(b["draws"]) for b in fb.values()) + " |", "",
              "Removed share: the model's probability (guided, temperature 1, after the legal Pokémon construction tables) on values the lookahead refuses. "
              "Intervals: 95% bootstrap, teams resampled within each ask and sampling seed, 4,000 draws. Script: `code/src/lookahead_override.py`. "
              "Results: `~/vgc-data/results/lookahead_override.json`; per-draw records in `~/vgc-data/results/lookahead_override/`.", "</details>"]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    {"redraw": lambda: redraw(sys.argv[2]), "report": report, "figure": lambda: print(figure(sys.argv[2])),
     "comment": lambda: print(comment(sys.argv[2]), end="")}[sys.argv[1]]()
