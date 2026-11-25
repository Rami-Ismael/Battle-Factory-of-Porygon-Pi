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
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import retrain_default_report as R
from echarts_chart import write_chart

R.ISSUE = "431"
R.NEW_ROW, R.REF_ROW = "shuffled_linear_filtered_gidd_lr", "linear_masked_filter"
R.NEW_CK = "stat84_shuffled_noise_linear_filtered_lr5e4_min5e5_es3_cap80"
R.REF_CK = "stat84_shuffled_noise_linear_filtered_es3_cap80"
R.NEW_LABEL, R.REF_LABEL = "Masked, full filter, shared recipe", "Masked, full filter, old recipe"
R.NEW_SHORT, R.REF_SHORT = "Shared recipe", "old recipe"
R.REF_REDRAW_BASES = {}                                  # both rows draw on 7300; no non-replayed redraw to substitute
R.BASES_FILE = Path("/nonexistent/gidd-lr-grid-bases.txt")
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
    return rows


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
    alt = (f"Got against ask at guidance {g} for {R.NEW_LABEL} versus {R.REF_LABEL}: ask-versus-got error "
           f"{nm['error']:.4f} vs {rm['error']:.4f} (paired difference {pe['difference']:+.4f})")
    ny, ry = nm.get("elite_line_up_yield") or {}, rm.get("elite_line_up_yield") or {}
    npk, rpk = nm.get("pass_at_k") or {}, rm.get("pass_at_k") or {}
    pak = lambda p: f"{p['pass_at_8']:.3f} / {p['pass_at_48']:.3f}" if p else "n/a"
    lines = [
        f"**Guidance {g} grid landed for {R.NEW_LABEL} (`{nm['name']}`): it {status}.**",
        "",
        R.picture(fig_light, fig_dark, alt),
        "",
        f"Completeness: `{nm['cells']}/69` cells, **`{nm['cells_short_of_64']}` cells short of 64 teams**, `{nm['teams']:,}` teams, "
        f"`{nm['battles']:,}` battles. Both recipes draw on sampling seed base `{nm['seed0']}` / `{rm['seed0']}`, so each cell meets the same random numbers.",
        "",
        f"| Measure (guidance {g}) | {R.NEW_LABEL} | {R.REF_LABEL} (#385) | Difference |",
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
                    "itemStyle": {"borderColor": "transparent", "borderWidth": 3}, "emphasis": {"itemStyle": {"borderColor": "#ffffff"}}}]
                  + legend_series,
    }
    better = sum(r[4] == BETTER for g in gs for r in per_g[g])
    worse = sum(r[4] == WORSE for g in gs for r in per_g[g])
    title = f"Shared recipe against the old recipe, Masked, full filter: {better} cells better and {worse} worse beyond training luck"
    subtitle = "Each cell: shared recipe minus old recipe, then both values. 4,416 teams and 216,384 battles per grid, seed base 7300 for both."
    note = ("Luck bars: error 0.002, pooled win rate 0.026, novel share 5 points; Pass@8 counts when the two 95% intervals do not overlap. "
            "Line-up counts carry no luck bar. One training seed per recipe.")
    return write_chart(option, Path(out_dir) / "recipe-matrix.html", title=title, subtitle=subtitle, note=note,
                       height=120 + 70 * len(names))


def format_final_comment(summary, fig_light, fig_dark):
    gs = [str(g) for g in summary["guidances"]]
    all_flags = [f"guidance {g}: {f}" for g in gs for f in flags(summary["by_guidance"][g])]
    errs = {g: summary["by_guidance"][g]["paired"]["error"] for g in gs}
    headline = ("the shared GIDD-style recipe changes Masked, full filter's ask grids beyond training luck"
                if all_flags else "the shared GIDD-style recipe leaves Masked, full filter's ask grids within training luck of the old recipe")
    alt = ("Matrix of shared-recipe minus old-recipe differences for Masked, full filter at guidance 2, 4 and 8: "
           + "; ".join(f"guidance {g} error {errs[g]['difference']:+.4f}" for g in gs))
    lines = [f"**Ask grids done: {headline}.**", "", R.picture(fig_light, fig_dark, alt), "",
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
              "(https://github.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/issues/438), which picks its pass count first."]
    return "\n".join(lines) + "\n"


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    cmd = args[0]
    if cmd == "check-grid":
        R.check_grid(args[1], int(args[2]))
    elif cmd == "grid-figure":
        print(R.build_grid_figure(R.read_guidance_pair(int(args[1])), args[2]))
    elif cmd == "grid-comment":
        sys.stdout.write(format_grid_comment(R.read_guidance_pair(int(args[1])), args[2], args[3]))
    elif cmd == "summary":
        R.build_summary(out_json=Path(args[1]) if len(args) > 1 else SUMMARY_PATH)
    elif cmd == "final-figure":
        print(build_final_figure(json.loads(Path(args[1]).read_text()), args[2]))
    elif cmd == "final-comment":
        sys.stdout.write(format_final_comment(json.loads(Path(args[1]).read_text()), args[2], args[3]))
    else:
        raise SystemExit(f"unknown command {cmd}")


if __name__ == "__main__":
    main()
