"""Issue #431 plan step 4: ask grids for Masked, full filter under the shared GIDD-style learning-rate recipe.

Pairs the shared-recipe checkpoint (`stat84_shuffled_noise_linear_filtered_lr5e4_min5e5_es3_cap80`, ask-grid row
`shuffled_linear_filtered_gidd_lr`) with the same model trained on the old 4e-4 cosine-to-0 recipe
(`stat84_shuffled_noise_linear_filtered_es3_cap80`, row `linear_masked_filter`, battled in #385) at guidance 2, 4 and 8.
Both rows draw on seed base 7300, so the bootstrap and the training-luck bars are retrain_default_report's (#385).
Uniform, no filter's grids belong to #438, which picks its pass count first.

    python gidd_lr_grid_report.py check-grid NAME CELLS
    python gidd_lr_grid_report.py grid-figure GUIDANCE OUT_DIR
    python gidd_lr_grid_report.py grid-comment GUIDANCE FIG_LIGHT FIG_DARK
    python gidd_lr_grid_report.py summary OUT_JSON
    python gidd_lr_grid_report.py final-figure SUMMARY_JSON OUT_DIR
    python gidd_lr_grid_report.py final-comment SUMMARY_JSON FIG_LIGHT FIG_DARK
    python gidd_lr_grid_report.py update-map SUMMARY_JSON
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import retrain_default_report as R
from echarts_chart import write_chart

COMPARISON = R.GridComparison(
    issue="431",
    new_row="shuffled_linear_filtered_gidd_lr",
    ref_row="linear_masked_filter",
    new_checkpoint="stat84_shuffled_noise_linear_filtered_lr5e4_min5e5_es3_cap80",
    ref_checkpoint="stat84_shuffled_noise_linear_filtered_es3_cap80",
    new_label="Masked, full filter, shared recipe",
    ref_label="Masked, full filter, old recipe",
    new_short="Shared recipe",
    ref_short="old recipe",
    ref_redraw_bases={},  # Both rows draw on 7300.
)
NOVEL_LUCK = 0.05                                        # the novel-share gap retrain_default_report flags
SUMMARY_PATH = R.RES / "gidd_lr_grid_summary.json"

BETTER, WORSE, WITHIN, NO_BAR = 0, 1, 2, 3
CELL_COLORS = {BETTER: "#0f766e", WORSE: "#c2410c", WITHIN: "#8a8780", NO_BAR: "#57534e"}
CELL_NAMES = {BETTER: "Shared recipe better, beyond luck", WORSE: "Shared recipe worse, beyond luck",
              WITHIN: "Within training luck", NO_BAR: "No luck bar: count only"}


def measures(pair):
    """One row per measure: (name, shared-recipe value, old value, difference text, verdict code)."""
    nm, rm, pe, pw = pair["filtered"], pair["unfiltered"], pair["paired"]["error"], pair["paired"]["pooled_win"]

    def by_bar(diff, bar, higher_is_better=True):
        if abs(diff) < bar: return WITHIN
        return BETTER if (diff > 0) == higher_is_better else WORSE

    def by_interval(new_iv, ref_iv, diff):
        if new_iv[1] < ref_iv[0] or new_iv[0] > ref_iv[1]: return BETTER if diff > 0 else WORSE
        return WITHIN

    rows = [("Ask-versus-got error", f"{nm['error']:.4f}", f"{rm['error']:.4f}", f"{pe['difference']:+.4f}",
             by_bar(pe["difference"], R.ERROR_LUCK, higher_is_better=False)),
            ("Pooled win rate", f"{nm['pooled_win']:.3f}", f"{rm['pooled_win']:.3f}", f"{pw['difference']:+.3f}",
             by_bar(pw["difference"], R.WIN_LUCK)),
            ("Novel share", f"{nm['novel_share']:.1%}", f"{rm['novel_share']:.1%}", f"{nm['novel_share'] - rm['novel_share']:+.1%}",
             by_bar(nm["novel_share"] - rm["novel_share"], NOVEL_LUCK)),
            ("Distinct line-ups", f"{nm['distinct_line_ups']:,}", f"{rm['distinct_line_ups']:,}",
             f"{nm['distinct_line_ups'] - rm['distinct_line_ups']:+,}", NO_BAR)]
    ny, ry = nm.get("elite_line_up_yield") or {}, rm.get("elite_line_up_yield") or {}
    if "held" in ny and "held" in ry:
        rows.append(("Elite line-ups that held", f"{ny['held']}", f"{ry['held']}", f"{ny['held'] - ry['held']:+d}", NO_BAR))
    npk, rpk = nm.get("pass_at_k") or {}, rm.get("pass_at_k") or {}
    if npk and rpk:
        d = npk["pass_at_8"] - rpk["pass_at_8"]
        rows.append(("Pass@8 (asks 13/24–17/24)", f"{npk['pass_at_8']:.3f}", f"{rpk['pass_at_8']:.3f}", f"{d:+.3f}",
                     by_interval(npk["pass_at_8_interval"], rpk["pass_at_8_interval"], d)))
        d48 = npk["pass_at_48"] - rpk["pass_at_48"]
        rows.append(("Pass@48 (asks 13/24–17/24)", f"{npk['pass_at_48']:.3f}", f"{rpk['pass_at_48']:.3f}", f"{d48:+.3f}",
                     by_interval(npk["pass_at_48_interval"], rpk["pass_at_48_interval"], d48)))
    return rows


def check_opponents(guidance, results_dir=None):
    """Both grids' opponent snapshots list the same 49 meta teams, in the same order, with the same weights: the paired
    bootstrap shares opponent draws by column position."""
    import numpy as np
    results_dir = Path(results_dir or R.RES)
    new_name = R.grid_name(COMPARISON.new_row, guidance)
    ref_name = R.grid_name(COMPARISON.ref_row, guidance)
    with (np.load(results_dir / f"masking_rate_rows_opponents_{new_name}.npz") as new,
          np.load(results_dir / f"masking_rate_rows_opponents_{ref_name}.npz") as ref):
        for key in ("columns", "weights"):
            if not np.array_equal(new[key], ref[key]):
                raise SystemExit(f"guidance {guidance}: opponent {key} differ between {new_name} and {ref_name}")


def pair_at(guidance):
    check_opponents(guidance)
    return R.read_guidance_pair(int(guidance), comparison=COMPARISON)


def build_summary(out_json, results_dir=R.RES):
    """Include novel share in the verdict shared by the summary, figures, and posts."""
    for g in R.GUIDANCES:
        check_opponents(g, results_dir)
    return R.build_summary(out_json, results_dir, comparison=COMPARISON, flags_for_pair=flags)


def flags(pair):
    out = list(pair["flags_beyond_luck"])
    nm, rm = pair["filtered"], pair["unfiltered"]
    gap = nm["novel_share"] - rm["novel_share"]
    if abs(gap) >= NOVEL_LUCK:
        out.append(f"novel share differs by {gap:+.1%} ({nm['novel_share']:.1%} vs {rm['novel_share']:.1%}), past the {NOVEL_LUCK:.0%} bar")
    return out


def format_grid_comment(pair, fig_light, fig_dark):
    g, nm, rm = pair["guidance"], pair["filtered"], pair["unfiltered"]
    pe, pw = pair["paired"]["error"], pair["paired"]["pooled_win"]
    found = flags(pair)
    status = "differs from the old recipe beyond training luck" if found else "matches the old recipe within training luck"
    alt = (f"Got against ask at guidance {g} for {COMPARISON.new_label} versus {COMPARISON.ref_label}: ask-versus-got error "
           f"{nm['error']:.4f} vs {rm['error']:.4f} (paired difference {pe['difference']:+.4f})")
    ny, ry = nm.get("elite_line_up_yield") or {}, rm.get("elite_line_up_yield") or {}
    npk, rpk = nm.get("pass_at_k") or {}, rm.get("pass_at_k") or {}
    pak = lambda p: f"{p['pass_at_8']:.3f} / {p['pass_at_48']:.3f}" if p else "n/a"
    lines = [
        f"**Guidance {g} grid landed for {COMPARISON.new_label} (`{nm['name']}`): it {status}.**",
        "",
        R.picture(fig_light, fig_dark, alt),
        "",
        f"Completeness: `{nm['cells']}/69` cells, **`{nm['cells_short_of_64']}` cells short of 64 teams**, `{nm['teams']:,}` teams, "
        f"`{nm['battles']:,}` battles. Both recipes draw on sampling seed base `{nm['seed0']}` / `{rm['seed0']}`, so each cell meets the same random numbers.",
        "",
        f"| Measure (guidance {g}) | {COMPARISON.new_label} | {COMPARISON.ref_label} (#385) | Difference |",
        "| --- | --- | --- | --- |",
        f"| Ask-versus-got error (asks 1/24–17/24) | **{nm['error']:.4f}** {R.fmt_iv(nm['error_interval'])} | {rm['error']:.4f} {R.fmt_iv(rm['error_interval'])} | **{pe['difference']:+.4f}** {R.fmt_iv(pe['interval'], '+.4f')} (luck bar `{R.ERROR_LUCK}`) |",
        f"| Error on asks 18/24–23/24 | `{nm['error_asks_18_to_23']:.4f}` | `{rm['error_asks_18_to_23']:.4f}` | `{nm['error_asks_18_to_23'] - rm['error_asks_18_to_23']:+.4f}` |",
        f"| Pooled win rate (23 asks) | **{nm['pooled_win']:.3f}** {R.fmt_iv(nm['pooled_win_interval'], '.3f')} | {rm['pooled_win']:.3f} {R.fmt_iv(rm['pooled_win_interval'], '.3f')} | **{pw['difference']:+.3f}** {R.fmt_iv(pw['interval'], '+.3f')} (luck bar `{R.WIN_LUCK}`) |",
        f"| Win rate at asks 15/24–17/24 | {nm['win_top_asks']:.3f} | {rm['win_top_asks']:.3f} | {pair['paired']['win_top_asks']['difference']:+.3f} |",
        f"| Novel share (line-ups unseen in labelled training) | **{nm['novel_share']:.1%}** | {rm['novel_share']:.1%} | {nm['novel_share'] - rm['novel_share']:+.1%} |",
        f"| Distinct line-ups | **{nm['distinct_line_ups']:,}** pooled / {nm['within_ask_line_ups_mean_23']:.1f} per ask | {rm['distinct_line_ups']:,} / {rm['within_ask_line_ups_mean_23']:.1f} | {nm['distinct_line_ups'] - rm['distinct_line_ups']:+,} |",
        f"| First-try legality | {nm['first_try_legality']:.1%} | {rm['first_try_legality']:.1%} | {nm['first_try_legality'] - rm['first_try_legality']:+.1%} |",
        f"| Effective species | {nm['effective_species']:.1f} | {rm['effective_species']:.1f} | {nm['effective_species'] - rm['effective_species']:+.1f} |",
        f"| Elite line-ups that held a 4-game re-battle | {ny.get('held', 'n/a')} | {ry.get('held', 'n/a')} | — |",
        f"| Novel elite line-ups that held | **{ny.get('held_new_against_training', 'n/a')}** | {ry.get('held_new_against_training', 'n/a')} | — |",
        f"| Pass@8 / Pass@48 (asks 13/24–17/24) | {pak(npk)} | {pak(rpk)} | — |",
        "",
    ]
    lines += (["**Beyond training luck:**"] + [f"- {f}" for f in found]) if found else ["No measure at this guidance differs beyond training luck."]
    lines += ["", "Intervals: 95% bootstrap over teams within each cell and the 49 distinct top-50 meta teams, shared across both "
              "checkpoints (slots shuffled, behaviour-cloning policy, one battle per meta team). One training seed per recipe."]
    return "\n".join(lines) + "\n"


def build_final_figure(summary, out_dir):
    """Matrix: one row per measure, one column per guidance; each cell is the shared recipe minus the old one."""
    gs = [str(g) for g in summary["guidances"]]
    per_g = {g: measures(summary["by_guidance"][g]) for g in gs}
    names = [m[0] for m in per_g[gs[0]]]
    data = []
    for x, g in enumerate(gs):
        for row in per_g[g]:
            y = names.index(row[0])
            data.append({"value": [x, len(names) - 1 - y, row[4]], "itemStyle": {"color": CELL_COLORS[row[4]]},
                         "label": {"formatter": f"{row[3]}\n{row[1]} vs {row[2]}"}, "name": CELL_NAMES[row[4]]})
    legend_series = [{"type": "scatter", "name": CELL_NAMES[c], "color": CELL_COLORS[c], "data": [], "symbol": "rect"}
                     for c in (BETTER, WORSE, WITHIN, NO_BAR)]
    option = {
        "animation": False,
        "grid": {"left": 190, "right": 16, "top": 64, "bottom": 40},
        "legend": {"top": 0, "data": [CELL_NAMES[c] for c in (BETTER, WORSE, WITHIN, NO_BAR)]},
        "tooltip": {"trigger": "item", "formatter": "js:p => p.name"},
        "xAxis": {"type": "category", "data": [f"Guidance {g}" for g in gs], "position": "top", "splitArea": {"show": False}},
        "yAxis": {"type": "category", "data": list(reversed(names))},
        "series": [{"type": "heatmap", "data": data, "label": {"show": True, "color": "#ffffff", "fontSize": 12, "lineHeight": 16},
                    "itemStyle": {"borderColor": "transparent", "borderWidth": 3}, "emphasis": {"itemStyle": {"borderColor": "#ffffff"}}},
                   *legend_series],
    }
    better = sum(r[4] == BETTER for g in gs for r in per_g[g])
    worse = sum(r[4] == WORSE for g in gs for r in per_g[g])
    title = f"Shared recipe against the old recipe, Masked, full filter: {better} cells better and {worse} worse beyond training luck"
    subtitle = "Each cell: shared recipe minus old recipe, then both values. 4,416 teams and 216,384 battles per grid, seed base 7300 for both."
    note = ("Luck bars: error 0.002, pooled win rate 0.026, novel share 5 points; Pass@8 counts when the two 95% intervals do not overlap. "
            "Line-up counts carry no luck bar. One training seed per recipe.")
    return write_chart(option, Path(out_dir) / "recipe-matrix.html", title=title, subtitle=subtitle, note=note,
                       height=120 + 70 * len(names))


NEW_COLOR, REF_COLOR = R.NEW_COLOR, R.REF_COLOR    # the grid figure's colours: shared recipe teal, old recipe grey


def build_error_gap_figure(summary, out_dir):
    """Frame 2: the paired ask-versus-got error difference at each guidance, with its 95% interval, against the
    training-luck band."""
    gs = [str(g) for g in summary["guidances"]]
    rows = [summary["by_guidance"][g]["paired"]["error"] for g in gs]
    series = [{"type": "line", "name": "95% interval", "color": NEW_COLOR, "symbol": "none", "lineStyle": {"width": 3},
               "data": [[r["interval"][0], i], [r["interval"][1], i]], "silent": True} for i, r in enumerate(rows)]
    series.append({"type": "scatter", "name": "Shared minus old", "color": NEW_COLOR, "symbolSize": 13,
                   "data": [[r["difference"], i] for i, r in enumerate(rows)],
                   "markArea": {"silent": True, "itemStyle": {"color": "rgba(138,135,128,0.16)"},
                                "data": [[{"xAxis": -R.ERROR_LUCK}, {"xAxis": R.ERROR_LUCK}]]},
                   "markLine": {"silent": True, "symbol": "none", "lineStyle": {"color": "#8a8780", "type": "dashed"},
                                "data": [{"xAxis": 0}], "label": {"show": False}}})
    inside = sum(-R.ERROR_LUCK < r["interval"][0] and r["interval"][1] < R.ERROR_LUCK for r in rows)
    span = max(R.ERROR_LUCK * 1.6, *(abs(v) * 1.15 for r in rows for v in r["interval"]))
    option = {"animation": False, "grid": {"left": 96, "right": 28, "top": 40, "bottom": 56},
              "tooltip": {"trigger": "axis", "valueFormatter": "js:v => (v >= 0 ? '+' : '') + v.toFixed(4)"},
              "xAxis": {"type": "value", "min": -span, "max": span, "name": "Ask-versus-got error, shared recipe minus old recipe",
                        "nameLocation": "middle", "nameGap": 34, "axisLabel": {"formatter": "js:v => v.toFixed(3)"}},
              "yAxis": {"type": "category", "data": [f"Guidance {g}" for g in gs], "inverse": True},
              "series": series}
    title = (f"The shared recipe hits the asked win rate as closely as the old one at {inside} of {len(gs)} guidance weights"
             if inside else "The shared recipe changes how closely teams hit the asked win rate")
    subtitle = "Dot: paired difference; bar: 95% bootstrap interval; grey band: the 0.002 training-luck bar. Asks 1/24 to 17/24."
    note = "Masked, full filter, both recipes on sampling seed base 7300; 216,384 battles per grid, 49 top-50 meta teams, slots shuffled."
    return write_chart(option, Path(out_dir) / "error-gap.html", title=title, subtitle=subtitle, note=note, height=300)


def build_tradeoff_figure(summary, out_dir):
    """Frame 3: ask-versus-got error against distinct line-ups, one line per recipe through guidance 2, 4 and 8."""
    gs = [str(g) for g in summary["guidances"]]
    series = []
    for side, label, color, dash in (("filtered", COMPARISON.new_label, NEW_COLOR, "solid"), ("unfiltered", COMPARISON.ref_label, REF_COLOR, "dashed")):
        pts = [summary["by_guidance"][g][side] for g in gs]
        series.append({"type": "line", "name": label, "color": color, "symbolSize": 9, "lineStyle": {"type": dash, "width": 2.4},
                       "emphasis": {"focus": "series"},
                       "label": {"show": True, "position": "top", "formatter": "js:p => 'g' + p.data[2]"},
                       "data": [[m["distinct_line_ups"], m["error"], g] for m, g in zip(pts, gs)]})
    option = {"animation": False, "grid": {"left": 72, "right": 28, "top": 56, "bottom": 56},
              "legend": {"top": 0, "data": [COMPARISON.new_label, COMPARISON.ref_label]}, "tooltip": {"trigger": "item"},
              "xAxis": {"type": "value", "scale": True, "name": "Distinct line-ups (pooled, 4,416 teams)", "nameLocation": "middle", "nameGap": 34},
              "yAxis": {"type": "value", "scale": True, "name": "Ask-versus-got error", "nameLocation": "middle", "nameGap": 54,
                        "axisLabel": {"formatter": "js:v => v.toFixed(4)"}},
              "series": series}
    new = [summary["by_guidance"][g]["filtered"]["distinct_line_ups"] for g in gs]
    old = [summary["by_guidance"][g]["unfiltered"]["distinct_line_ups"] for g in gs]
    gap = sum(n - o for n, o in zip(new, old)) / len(gs)
    title = f"The shared recipe draws {gap:+,.0f} distinct line-ups per grid on average at guidance 2 to 8"
    subtitle = "Lower and further right is better: closer to the ask with more line-up variety. Labels: guidance weight."
    note = "Line-up: the six species of a team. Error: mean squared gap between measured and asked win rate, asks 1/24 to 17/24."
    return write_chart(option, Path(out_dir) / "tradeoff.html", title=title, subtitle=subtitle, note=note, height=420)


CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
FINAL_FIGURES = (("error-gap", 380), ("tradeoff", 520), ("recipe-matrix", 760))   # stem, window height for the PNG


def render_final_pngs(out_dir, gist_dir):
    """Light and dark PNGs of every final figure into the gist folder, so the queue's push sends all three."""
    for stem, height in FINAL_FIGURES:
        for theme, scheme in (("light", 1), ("dark", 0)):
            subprocess.run([CHROME, "--headless=new", "--hide-scrollbars", f"--blink-settings=preferredColorScheme={scheme}",
                            "--virtual-time-budget=8000", f"--window-size=1000,{height}", "--force-device-scale-factor=2",
                            f"--screenshot={Path(gist_dir) / f'{stem}-{theme}.png'}", f"file://{Path(out_dir) / f'{stem}.html'}"],
                           check=True, capture_output=True, timeout=120)


def format_final_comment(summary, base_url):
    """Frames 2, 3 and 4 of docs/gidd-lr-recipe-grids.md, each as light and dark PNGs at BASE_URL/<stem>-<theme>.png."""
    gs = [str(g) for g in summary["guidances"]]
    all_flags = [f"guidance {g}: {f}" for g in gs for f in flags(summary["by_guidance"][g])]
    errs = {g: summary["by_guidance"][g]["paired"]["error"] for g in gs}
    by = summary["by_guidance"]
    headline = ("the shared GIDD-style recipe changes Masked, full filter's ask grids beyond training luck"
                if all_flags else "the shared GIDD-style recipe leaves Masked, full filter's ask grids within training luck of the old recipe")
    fig = lambda stem, alt: R.picture(f"{base_url}/{stem}-light.png", f"{base_url}/{stem}-dark.png", alt)
    alt_gap = ("Paired ask-versus-got error difference, shared recipe minus old, at guidance 2, 4 and 8, against the 0.002 luck band: "
               + "; ".join(f"guidance {g} {errs[g]['difference']:+.4f} {R.fmt_iv(errs[g]['interval'], '+.4f')}" for g in gs))
    alt_trade = ("Ask-versus-got error against distinct line-ups for both recipes at guidance 2, 4 and 8: "
                 + "; ".join(f"guidance {g} {by[g]['filtered']['distinct_line_ups']:,} vs {by[g]['unfiltered']['distinct_line_ups']:,} line-ups" for g in gs))
    alt_matrix = "Matrix of shared-recipe minus old-recipe differences for every measure at guidance 2, 4 and 8"
    lines = [f"**Ask grids done: {headline}.**", "", fig("error-gap", alt_gap), "", fig("tradeoff", alt_trade), "",
             fig("recipe-matrix", alt_matrix), "",
             "| Guidance | Error, shared recipe | Error, old recipe | Paired difference (95% interval) | Per-seed differences |",
             "| --- | --- | --- | --- | --- |"]
    for g in gs:
        p = summary["by_guidance"][g]
        e = errs[g]
        seeds = ", ".join(f"{v:+.4f}" for v in e["difference_by_seed"].values())
        lines.append(f"| {g} | {p['filtered']['error']:.4f} | {p['unfiltered']['error']:.4f} | {e['difference']:+.4f} {R.fmt_iv(e['interval'], '+.4f')} | {seeds} |")
    lines += [""] + ((["**Beyond training luck:**"] + [f"- {f}" for f in all_flags]) if all_flags else ["No measure differs beyond training luck at any guidance."])
    lines += ["", "**Caveats.** One training seed per recipe, and both checkpoints stopped at the 80-epoch cap still improving. "
              "The old recipe trained on the laptop's MPS GPU and the shared one on a Modal A10; their validation curves agree within 0.007 through epoch 40. "
              "Uniform, no filter's grids run in [Run every masked-model evaluation on the shared-recipe Uniform, no filter checkpoint]"
              "(https://github.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/issues/438), which picks its pass count first. "
              "Battles moved to Modal CPU containers at 11:30 (20 of 69 guidance 2 cells had run on the laptop); a 768-team re-battle "
              "matched the laptop (-0.0035 [-0.0096, +0.0027], ask-versus-got error +0.0002). Storyboard: `code/docs/gidd-lr-recipe-grids.md`."]
    return "\n".join(lines) + "\n"


MAP_ISSUE = "270"
TASK_TITLE = ("Test one shared GIDD-style learning-rate recipe (lr 5e-4, cosine to 10% floor 5e-5) on both Uniform, no filter "
              "and Masked, full filter")


def decision_line(summary):
    gs = [str(g) for g in summary["guidances"]]
    pairs = [summary["by_guidance"][g] for g in gs]
    errors = " / ".join(f"`{p['filtered']['error']:.4f}` vs `{p['unfiltered']['error']:.4f}`" for p in pairs)
    found = sum(len(flags(p)) for p in pairs)
    verdict = "every measure within training luck" if not found else f"{found} difference(s) beyond training luck"
    return (f"- [{TASK_TITLE}](https://github.com/{R.REPO}/issues/431): the shared recipe (peak `5e-4`, cosine to `5e-5`, "
            f"trained on a Modal A10) lowers best validation loss on both arms (Uniform, no filter `1.4384` -> `1.3153`, "
            f"Masked, full filter `1.3143` -> `1.2138`) by moving the late cliff earlier, and lifts Uniform, no filter's "
            f"first-try legality `15.6%` -> `43.8%`. Masked, full filter's ask-versus-got error at guidance "
            f"{', '.join(gs)}: {errors} against the old recipe ({verdict}). Uniform, no filter's grids moved to #438.")


def update_map(summary):
    """Add #431's result under "Decisions so far" on map #270, once, reading the map body fresh."""
    body = subprocess.run(["gh", "issue", "view", MAP_ISSUE, "--repo", R.REPO, "--json", "body", "-q", ".body"],
                          check=True, capture_output=True, text=True).stdout
    if "## Decisions so far" not in body or "\n## Not yet specified" not in body:
        raise RuntimeError(f"map #{MAP_ISSUE} is missing its Decisions so far / Not yet specified sections")
    decisions = body.split("## Decisions so far", 1)[1].split("## Not yet specified", 1)[0]
    if "/issues/431)" in decisions:
        print(f"map #{MAP_ISSUE} already records #431", file=sys.stderr)
        return
    before, after = body.split("\n## Not yet specified", 1)
    updated = before.rstrip() + "\n" + decision_line(summary) + "\n\n## Not yet specified" + after
    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as handle:
        handle.write(updated)
    try:
        subprocess.run(["gh", "issue", "edit", MAP_ISSUE, "--repo", R.REPO, "--body-file", handle.name], check=True,
                       capture_output=True, text=True)
    finally:
        Path(handle.name).unlink(missing_ok=True)
    print(f"updated map #{MAP_ISSUE} Decisions so far", file=sys.stderr)


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    cmd = args[0]
    if cmd == "check-grid":
        R.check_grid(args[1], int(args[2]))
    elif cmd == "grid-figure":
        print(R.build_grid_figure(pair_at(args[1]), args[2], comparison=COMPARISON))
    elif cmd == "grid-comment":
        sys.stdout.write(format_grid_comment(pair_at(args[1]), args[2], args[3]))
    elif cmd == "summary":
        build_summary(Path(args[1]) if len(args) > 1 else SUMMARY_PATH)
    elif cmd == "final-figure":
        summary = json.loads(Path(args[1]).read_text())
        for build in (build_error_gap_figure, build_tradeoff_figure, build_final_figure): print(build(summary, args[2]))
        render_final_pngs(args[2], Path(args[2]).parent / "gist")     # the queue folder holds figure/ and gist/ side by side
    elif cmd == "final-comment":
        base = args[2].rsplit("/", 1)[0] if args[2].endswith(".png") else args[2]   # a PNG URL from the queue, or the base
        sys.stdout.write(format_final_comment(json.loads(Path(args[1]).read_text()), base))
    elif cmd == "update-map":
        update_map(json.loads(Path(args[1]).read_text()))
    else:
        raise SystemExit(f"unknown command {cmd}")


if __name__ == "__main__":
    main()
