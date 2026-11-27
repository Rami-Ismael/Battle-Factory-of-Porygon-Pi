"""Full Part A evaluation and reporting for Issue #438 (items 1-4: pass count post, unguided
comparison, bend-point selection, diversity/novelty/revision frontier & dominated-area verdict).

Complements uniform_gidd_lr_grid_report.py and udlm_pass_count84.py so every pre-registered
measure from map #270 (#277-#280) is computed, plotted with ECharts, and posted to Issue #438:
  - pass-count-comment: format Item 1 comment with storyboard figures + 3-col comparison vs 4e-4 Uniform.
  - unguided-report: combine val_loss, unguided (gamma=0), and conditional (gamma=1) baselines (Item 2).
  - bend-points: apply uniform_preflight84.bend_points_to_add to the coarse {1, 2, 4, 8} grids (Item 3).
  - frontier-verdict: full diversity_metrics.py, species replacement distance, dominated area HV,
    equal-error advantage, highest usable guidance g_max, 5-axis combined Pareto front, and
    pre-registered revision numbers + exploratory diagnostics across all guidance weights (Items 3-4).

Pure analysis, Pareto, and ECharts helpers load no /tmp/vgc-pilot modules so unit tests run on CPU.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import diversity_metrics as DM
from echarts_chart import write_chart
import uniform_math84 as UMath
import uniform_preflight84 as UP

RES_DIR = Path.home() / "vgc-data/results"
UNIFORM_ROW = "udlm_loglinear_gidd_lr"
MASKED_ROW = "shuffled_linear_filtered_gidd_lr"
OLD_MASKED_ROW = "linear_masked_filter"
SCORED_ASKS = tuple(range(1, 18))
ALL_ASKS = tuple(range(1, 24))
SEEDS = (1, 2, 3)

COLORS = {
    "uniform_gidd_lr": "#0f766e",
    "uniform_4e4": "#c2410c",
    "masked_gidd_lr": "#365d93",
    "masked_4e4": "#6f6e69",
}
DARK = {
    "#0f766e": "#5eead4",
    "#c2410c": "#fdba74",
    "#365d93": "#93c5fd",
    "#6f6e69": "#a9a8a2",
}


def grid_name(row: str, guidance: float | None, *, unguided: bool = False) -> str:
    if unguided:
        return f"{row}_unguided"
    if guidance is None or float(guidance) == 2.0:
        return row
    return f"{row}_g{float(guidance):g}"


def _picture(base_url: str, stem: str, alt: str) -> str:
    return (
        f'<picture><source media="(prefers-color-scheme: dark)" srcset="{base_url}/{stem}-dark.png">'
        f'<img src="{base_url}/{stem}-light.png" alt="{alt}" width="900"></picture>'
    )


def _normalize_paste(paste: str):
    sys.path.insert(0, "/tmp/vgc-pilot/src")
    sys.path.insert(0, str(HERE))
    from team_diversity import slots_from_paste

    slots = []
    for slot in slots_from_paste(paste):
        evs_in = slot.get("evs") or {}
        slots.append(
            dict(
                species=str(slot["species"]),
                ability=str(slot.get("ability") or ""),
                item=str(slot.get("item") or ""),
                nature=str(slot.get("nature") or ""),
                moves=[str(m) for m in slot.get("moves", []) if m],
                evs={s: int(evs_in.get(s, 0)) for s in DM.STATS},
            )
        )
    return slots


def _build_reference():
    import udlm_pass_count84 as UPC

    reference, _, _ = UPC._build_training_reference()
    return reference


def _cell_diversity(pastes, reference, parser):
    if not pastes:
        return None
    teams = [parser(p) for p in pastes]
    measured = DM.measure(teams, reference=reference)
    comps = [DM.team_keys(t)[2] for t in teams]
    srd = UP._mean_species_replacement_distance(comps)
    out = dict(species_replacement_distance=srd)
    for rep in ("categorical_48", "with_spreads", "composition"):
        out[rep] = {
            k: measured[rep].get(k)
            for k in ("unique_fraction", "repeat_fraction", "novel_draw_fraction", "novel_unique_fraction", "largest_share")
            if k in measured[rep]
        }
    return out


def _summarize_seed_scored(grid: dict, seed: int, reference, parser):
    wa_rows, pooled_pastes, ask_gaps, top_wins = [], [], [], []
    seed_tried, seed_rejected, complete = 0, 0, True
    for ask in SCORED_ASKS:
        cell = next((c for c in grid["cells"].values() if c["numerator"] == ask and c["seed"] == seed), None)
        if cell is None:
            complete = False
            continue
        pastes = cell.get("pastes", [])
        if len(pastes) < 64:
            complete = False
        seed_tried += int(cell.get("tried", len(pastes)))
        seed_rejected += int(cell.get("rejected", 0))
        pooled_pastes.extend(pastes)
        cd = _cell_diversity(pastes, reference, parser)
        if cd is not None:
            wa_rows.append(cd)
        if cell.get("y"):
            ask_gaps.append((float(np.mean(cell["y"])) - ask / 24.0) ** 2)
            if ask in (15, 16, 17):
                top_wins.extend(cell["y"])

    wa_agg = {}
    if wa_rows:
        srds = [r["species_replacement_distance"] for r in wa_rows if math.isfinite(r["species_replacement_distance"])]
        wa_agg["species_replacement_distance"] = float(np.mean(srds)) if srds else float("nan")
        for rep in ("categorical_48", "with_spreads", "composition"):
            wa_agg[rep] = {
                k: float(np.mean([r[rep][k] for r in wa_rows if r[rep][k] is not None]))
                if any(r[rep][k] is not None for r in wa_rows)
                else float("nan")
                for k in wa_rows[0][rep]
            }

    return complete, dict(
        error=float(np.mean(ask_gaps)) if len(ask_gaps) == len(SCORED_ASKS) else None,
        top_ask_win=float(np.mean(top_wins)) if top_wins else None,
        legality=(seed_tried - seed_rejected) / max(seed_tried, 1),
        within_ask=wa_agg,
        pooled=_cell_diversity(pooled_pastes, reference, parser) if pooled_pastes else None,
    )


def summarize_grid_diversity_and_revisions(grid: dict, reference, *, parser=_normalize_paste):
    """Compute per-seed within-ask and pooled diversity, species replacement distance, usability, and revisions on scored asks."""
    by_seed = {}
    cells_complete = True
    for seed in SEEDS:
        ok, seed_rec = _summarize_seed_scored(grid, seed, reference, parser)
        cells_complete = cells_complete and ok
        by_seed[str(seed)] = seed_rec

    def _agg(getter):
        vals = [getter(by_seed[str(s)]) for s in SEEDS]
        finite = [v for v in vals if v is not None and math.isfinite(v)]
        return dict(
            by_seed=vals,
            mean=float(np.mean(finite)) if len(finite) == len(SEEDS) else None,
            min=float(np.min(finite)) if finite else None,
            max=float(np.max(finite)) if finite else None,
        )

    # Aggregate uniform revision summaries across scored cells (if present)
    rev_summary = aggregate_grid_revisions(grid)

    return dict(
        guidance=float(grid.get("guidance", 2.0)),
        passes=int(grid.get("passes", 84)),
        cells_complete_scored=cells_complete,
        by_seed=by_seed,
        error=_agg(lambda s: s["error"]),
        top_ask_win=_agg(lambda s: s["top_ask_win"]),
        legality=_agg(lambda s: s["legality"]),
        within_ask_unique=_agg(lambda s: s["within_ask"].get("composition", {}).get("unique_fraction")),
        within_ask_largest_share=_agg(lambda s: s["within_ask"].get("composition", {}).get("largest_share")),
        within_ask_novel=_agg(lambda s: s["within_ask"].get("composition", {}).get("novel_draw_fraction")),
        within_ask_srd=_agg(lambda s: s["within_ask"].get("species_replacement_distance")),
        within_ask_cat48_unique=_agg(lambda s: s["within_ask"].get("categorical_48", {}).get("unique_fraction")),
        within_ask_spreads_unique=_agg(lambda s: s["within_ask"].get("with_spreads", {}).get("unique_fraction")),
        pooled_novel=_agg(lambda s: (s["pooled"] or {}).get("composition", {}).get("novel_draw_fraction")),
        pooled_unique=_agg(lambda s: (s["pooled"] or {}).get("composition", {}).get("unique_fraction")),
        pooled_srd=_agg(lambda s: (s["pooled"] or {}).get("species_replacement_distance")),
        revisions=rev_summary,
    )


def aggregate_grid_revisions(grid: dict):
    """Average uniform revision summaries over scored cells (asks 1..17, all tries)."""
    cells = [
        c for c in grid["cells"].values()
        if c["numerator"] in SCORED_ASKS and isinstance(c.get("revisions"), dict) and c["revisions"].get("all_tries")
    ]
    if not cells:
        return None
    weights = [max(int(c["revisions"]["all_tries"].get("teams", c.get("tried", 1))), 1) for c in cells]
    total_w = float(sum(weights))

    def wmean(getter):
        vals = [(getter(c["revisions"]["all_tries"]), w) for c, w in zip(cells, weights)]
        valid = [(v, w) for v, w in vals if v is not None and math.isfinite(v)]
        if not valid:
            return None
        return float(sum(v * w for v, w in valid) / sum(w for _, w in valid))

    by_field = {}
    for fk in UMath.FIELD_KEYS:
        sample_f = cells[0]["revisions"]["all_tries"]["by_field"][fk]
        by_field[fk] = dict(
            vocab_size=sample_f.get("vocab_size"),
            noise_floor=sample_f.get("noise_floor"),
            value_changes_beyond_first=wmean(lambda r, k=fk: r["by_field"][k].get("value_changes_beyond_first")),
            changes_of_mind=wmean(lambda r, k=fk: r["by_field"][k].get("changes_of_mind")),
            guided_changes_of_mind=wmean(lambda r, k=fk: r["by_field"][k].get("guided_changes_of_mind")),
            confident_changes_of_mind=wmean(lambda r, k=fk: r["by_field"][k].get("confident_changes_of_mind")),
        )
    return dict(
        teams_total=int(total_w),
        mean_fields_changed_per_pass=wmean(lambda r: r.get("mean_fields_changed_per_pass")),
        duplicate_species_pass_share=wmean(lambda r: r.get("duplicate_species_pass_share")),
        final_duplicate_species_share=wmean(lambda r: r.get("final_duplicate_species_share")),
        by_field=by_field,
    )


def evaluate_usability_and_bend_points(records: list[dict]):
    """Apply uniform_preflight84.is_usable_strength and bend_points_to_add to sorted guidance summaries."""
    ordered = sorted(records, key=lambda r: r["guidance"])
    best_lower = None
    coarse = []
    for rec in ordered:
        err = rec["error"]["mean"]
        uniq = rec["within_ask_unique"]["mean"]
        largest = rec["within_ask_largest_share"]["mean"]
        usable = bool(
            err is not None
            and uniq is not None
            and largest is not None
            and UP.is_usable_strength(rec["cells_complete_scored"], uniq, largest, err, best_lower)
        )
        rec["usable"] = usable
        if err is not None:
            best_lower = err if best_lower is None else min(best_lower, err)
            coarse.append(
                dict(
                    weight=float(rec["guidance"]),
                    error=float(err),
                    within_ask_unique_fraction=float(uniq or 0.0),
                    usable=usable,
                )
            )
    added = UP.bend_points_to_add(coarse, max_added=2) if len(coarse) >= 2 else []
    usable_weights = [r["guidance"] for r in ordered if r.get("usable")]
    g_max = max(usable_weights) if usable_weights else None
    return dict(g_max=g_max, bend_points=added, coarse=coarse)


def five_axis_pareto_front(arm_points: dict[str, list[dict]]):
    """5-axis seed-for-seed Pareto dominance over (-error, wa_novel, wa_unique, wa_srd, top_ask_win)."""
    all_pts = []
    for arm, pts in arm_points.items():
        for p in pts:
            if not p.get("usable"):
                continue
            vecs = []
            ok = True
            for s in SEEDS:
                sd = p["by_seed"][str(s)]
                vals = (
                    -(sd["error"] if sd["error"] is not None else math.inf),
                    sd["within_ask"].get("composition", {}).get("novel_draw_fraction"),
                    sd["within_ask"].get("composition", {}).get("unique_fraction"),
                    sd["within_ask"].get("species_replacement_distance"),
                    sd["top_ask_win"],
                )
                if any(v is None or not math.isfinite(v) for v in vals):
                    ok = False
                    break
                vecs.append(vals)
            if ok:
                all_pts.append(dict(arm=arm, guidance=p["guidance"], vecs=vecs))

    def dominates(a, b):
        return all(
            all(x >= y for x, y in zip(va, vb)) and any(x > y for x, y in zip(va, vb))
            for va, vb in zip(a["vecs"], b["vecs"])
        )

    front = []
    for pt in all_pts:
        if not any(dominates(other, pt) for other in all_pts if other is not pt):
            front.append(dict(arm=pt["arm"], guidance=pt["guidance"]))
    return front


def compute_frontier_verdict(uniform_records: list[dict], masked_records: list[dict]):
    """Compute per-seed dominated area HV, equal-error advantage, and verdict for each frontier."""
    u_meta = evaluate_usability_and_bend_points(uniform_records)
    m_meta = evaluate_usability_and_bend_points(masked_records)
    # Note: if a uniform grid has slightly short cells on ask 1/24 despite 20,000 tries, also report all-points HV beside strict-usable HV.
    frontiers = (
        ("within_ask_novel", "Within-ask novelty", lambda sd: sd["within_ask"]["composition"]["novel_draw_fraction"], False),
        ("within_ask_unique", "Within-ask line-up uniqueness", lambda sd: sd["within_ask"]["composition"]["unique_fraction"], False),
        ("within_ask_srd", "Within-ask species replacement distance", lambda sd: sd["within_ask"]["species_replacement_distance"], False),
        ("within_ask_1_minus_largest", "Within-ask 1 − largest line-up share", lambda sd: 1.0 - sd["within_ask"]["composition"]["largest_share"], False),
    )

    out = dict(
        uniform_g_max=u_meta["g_max"],
        masked_g_max=m_meta["g_max"],
        frontiers={},
        combined_pareto_front=five_axis_pareto_front(dict(uniform=uniform_records, masked=masked_records)),
    )

    for fkey, label, getter, _ in frontiers:
        u_hv_strict, u_hv_all, m_hv = [], [], []
        eq_adv = []
        for s in SEEDS:
            sk = str(s)
            u_pts_strict = [
                (r["by_seed"][sk]["error"], getter(r["by_seed"][sk]))
                for r in uniform_records
                if r.get("usable") and r["by_seed"][sk]["error"] is not None
            ]
            u_pts_all = [
                (r["by_seed"][sk]["error"], getter(r["by_seed"][sk]))
                for r in uniform_records
                if r["by_seed"][sk]["error"] is not None
            ]
            m_pts = [
                (r["by_seed"][sk]["error"], getter(r["by_seed"][sk]))
                for r in masked_records
                if r["by_seed"][sk]["error"] is not None
            ]
            u_hv_strict.append(UP.dominated_area(u_pts_strict))
            u_hv_all.append(UP.dominated_area(u_pts_all))
            m_hv.append(UP.dominated_area(m_pts))
            adv = UP.equal_error_advantage(u_pts_strict or u_pts_all, m_pts)
            eq_adv.append(adv["mean_advantage"] if adv is not None else None)

        diffs = [u - m for u, m in zip(u_hv_strict, m_hv)]
        diffs_all = [u - m for u, m in zip(u_hv_all, m_hv)]
        valid_eq = [a for a in eq_adv if a is not None]
        if all(d > 0 for d in diffs) and (not valid_eq or all(a > 0 for a in valid_eq)):
            verdict = "beats"
        elif all(d < 0 for d in diffs):
            verdict = "loses"
        else:
            verdict = "inconclusive"

        out["frontiers"][fkey] = dict(
            label=label,
            uniform_hv_usable=dict(by_seed=u_hv_strict, mean=float(np.mean(u_hv_strict))),
            uniform_hv_all=dict(by_seed=u_hv_all, mean=float(np.mean(u_hv_all))),
            masked_hv=dict(by_seed=m_hv, mean=float(np.mean(m_hv))),
            diff_usable=dict(by_seed=diffs, mean=float(np.mean(diffs))),
            diff_all=dict(by_seed=diffs_all, mean=float(np.mean(diffs_all))),
            equal_error_advantage=dict(by_seed=eq_adv, mean=float(np.mean(valid_eq)) if valid_eq else None),
            verdict=verdict,
        )
    return out


# ---------------------------------------------------------------- Item 1 pass-count comment
def format_pass_count_comment(shared_report: dict, old_report: dict | None, base_url: str) -> str:
    """Format Item 1 comment: storyboard figures first + 3-col / 2-recipe pass-count comparison."""
    dec = shared_report["decision"]
    k_star = dec["chosen_passes"]
    by_p = shared_report["by_passes"]
    lines = [
        f"**Pass-count selection done (Item 1, #277): shared-recipe Uniform, no filter chooses {k_star} passes at guidance 1 ({by_p[str(k_star)]['legality']['mean']:.1%} first-try legality vs {old_report['by_passes'][str(k_star)]['legality']['mean']:.1%} on the 4e-4 checkpoint).**"
        if old_report
        else f"**Pass-count selection done (Item 1, #277): shared-recipe Uniform, no filter chooses {k_star} passes at guidance 1 ({by_p[str(k_star)]['legality']['mean']:.1%} first-try legality).**",
        "",
        "Two storyboard figures compare first-try legality, matched within-ask diversity, simultaneous edits, and changes of mind across 84, 128, and 256 passes.",
        "",
        _picture(base_url, "01_pass_count_selection", f"Pass-count selection across 84, 128, and 256 passes at guidance 1: chosen K*={k_star}"),
        "",
        _picture(base_url, "02_simultaneous_updates_and_revisions", "Simultaneous updates, duplicate-species collisions, and changes of mind across 84, 128, and 256 passes"),
        "",
        "| Pass count | Shared-recipe Uniform legality (complete cells) | 4e-4 Uniform legality (complete cells) | Shared-recipe matched line-up uniq. | Shared-recipe species repl. dist. | Shared-recipe final dup. species |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for k in (84, 128, 256):
        r = by_p[str(k)]
        o = old_report["by_passes"][str(k)] if old_report else None
        old_txt = f"{o['legality']['mean']:.1%} ({o['cells_complete']}/{o['cells_expected']})" if o else "—"
        star = " (chosen `K*`)" if k == k_star else ""
        lines.append(
            f"| **{k} passes{star}** | {r['legality']['mean']:.1%} ± {r['legality']['se']:.1%} ({r['cells_complete']}/{r['cells_expected']}) | "
            f"{old_txt} | {r['matched_lineup_uniqueness']['mean']:.3f} | "
            f"{r['within_ask_species_replacement_distance']['mean']:.3f} | "
            f"{r['revisions_all_tries']['final_duplicate_species_share']['mean']:.1%} |"
        )
    lines += [
        "",
        "Battles: 0 (asks 1/24 to 17/24, 3 seeds, target 64 teams/cell, cap 20,000 tries; 4e-4 reference used cap 1,280 in #277).",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------- Item 2 unguided/conditional report
def build_unguided_figure(summary: dict, out_dir: Path) -> Path:
    """Grouped bar chart of unguided (gamma=0) win rate, legality, and conditional (gamma=1) error."""
    arms = summary["arms"]
    names = [a["reader"] for a in arms]
    g0_leg = [round(a["g0_legality"], 4) if a.get("g0_legality") is not None else 0.0 for a in arms]
    g0_win = [round(a["g0_win"], 4) if a.get("g0_win") is not None else 0.0 for a in arms]
    g1_leg = [round(a["g1_legality"], 4) if a.get("g1_legality") is not None else 0.0 for a in arms]
    option = {
        "animation": False,
        "legend": {"top": 4},
        "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"}, "valueFormatter": "js:v => (Number(v) * 100).toFixed(1) + '%'"},
        "grid": {"left": 70, "right": 28, "top": 52, "bottom": 52},
        "xAxis": {"type": "category", "data": names},
        "yAxis": {"type": "value", "min": 0, "max": 1.05, "axisLabel": {"formatter": "js:v => Math.round(v * 100) + '%'"}},
        "series": [
            {"name": "Unguided (γ=0) legality", "type": "bar", "color": "#0f766e", "data": g0_leg, "emphasis": {"focus": "series"}},
            {"name": "Conditional (γ=1) legality", "type": "bar", "color": "#365d93", "data": g1_leg, "emphasis": {"focus": "series"}},
            {"name": "Unguided (γ=0) win rate", "type": "bar", "color": "#c2410c", "data": g0_win, "emphasis": {"focus": "series"}},
        ],
    }
    return write_chart(
        option,
        out_dir / "unguided-baseline.html",
        title="Unguided (γ=0) and conditional (γ=1) baselines on the shared-recipe checkpoints",
        subtitle="192 unguided teams per arm (64 × 3 seeds × 49 top-50 meta teams = 9,408 battles) and conditional γ=1 grids.",
        note="Slots shuffled, behaviour-cloning policy on both sides.",
        height=400,
        dark=DARK,
    )


def format_unguided_comment(summary: dict, base_url: str) -> str:
    lines = [
        "**Unguided and conditional baselines done (Item 2, #278): validation loss, first-try legality, unguided win rate, and γ=1 ask-versus-got error.**",
        "",
        _picture(base_url, "unguided-baseline", "Unguided (gamma=0) legality and win rate plus conditional (gamma=1) legality across shared-recipe Uniform and Masked checkpoints"),
        "",
        "| Model | `val_loss` (4,368 rows) | γ=0 legality | γ=0 win rate [95% CI] | γ=0 battles | γ=1 legality | γ=1 ask-versus-got error |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for a in summary["arms"]:
        vl = f"{a['val_loss']:.4f}" if a.get("val_loss") is not None else "—"
        g0_l = f"{a['g0_legality']:.1%}" if a.get("g0_legality") is not None else "—"
        if a.get("g0_win") is not None and a.get("g0_win_interval"):
            lo, hi = a["g0_win_interval"]
            g0_w = f"{a['g0_win']:.3f} [{lo:.3f}, {hi:.3f}]"
        else:
            g0_w = "—"
        g0_b = f"{a['g0_battles']:,}" if a.get("g0_battles") else "—"
        g1_l = f"{a['g1_legality']:.1%}" if a.get("g1_legality") is not None else "—"
        g1_e = f"{a['g1_error']:.4f}" if a.get("g1_error") is not None else "—"
        lines.append(f"| {a['reader']} | {vl} | {g0_l} | {g0_w} | {g0_b} | {g1_l} | {g1_e} |")
    lines += [
        "",
        "Battles: 49 distinct top-50 meta teams, slots shuffled, behaviour-cloning policy. One training seed per checkpoint.",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------- Items 3-4 frontier & revision figures
def build_frontier_and_revision_figures(report: dict, out_dir: Path) -> list[Path]:
    u_recs = sorted(report["uniform_records"], key=lambda r: r["guidance"])
    m_recs = sorted(report["masked_records"], key=lambda r: r["guidance"])
    paths = []

    for key, label, stem, pct in (
        ("within_ask_novel", "Within-ask novel line-up share", "frontier-wa-novelty", True),
        ("within_ask_unique", "Within-ask line-up uniqueness", "frontier-wa-uniqueness", False),
        ("within_ask_srd", "Within-ask mean species replacement distance", "frontier-wa-srd", False),
    ):
        series = []
        for name, recs, color, sym in (
            ("Uniform, no filter (shared recipe)", u_recs, COLORS["uniform_gidd_lr"], "circle"),
            ("Masked, full filter (shared recipe)", m_recs, COLORS["masked_gidd_lr"], "diamond"),
        ):
            pts = sorted(
                [[r["error"]["mean"], r[key]["mean"], r["guidance"]] for r in recs if r["error"]["mean"] is not None and r[key]["mean"] is not None],
                key=lambda row: row[0],
            )
            series.append({
                "name": name,
                "type": "line",
                "color": color,
                "symbol": sym,
                "symbolSize": 8,
                "lineStyle": {"width": 2.5},
                "emphasis": {"focus": "series"},
                "data": pts,
                "label": {"show": True, "position": "top", "formatter": "js:p => 'g=' + p.value[2]", "color": color},
            })
        option = {
            "animation": False,
            "legend": {"top": 4},
            "tooltip": {"trigger": "axis"},
            "grid": {"left": 80, "right": 34, "top": 48, "bottom": 58},
            "xAxis": {"type": "value", "name": "Ask-versus-got error (asks 1/24 to 17/24)", "nameLocation": "middle", "nameGap": 34, "axisLabel": {"formatter": "js:v => Number(v).toFixed(4)"}},
            "yAxis": {
                "type": "value",
                "name": label,
                "nameLocation": "middle",
                "nameGap": 52,
                "axisLabel": {"formatter": "js:v => Math.round(v * 100) + '%'"} if pct else {},
            },
            "series": series,
        }
        verdict = report["verdict"]["frontiers"].get(key, {})
        sub = f"Verdict: {verdict.get('verdict', 'n/a')} · HV (usable) Uniform {verdict.get('uniform_hv_usable', {}).get('mean', 0):.5f} vs Masked {verdict.get('masked_hv', {}).get('mean', 0):.5f}."
        paths.append(write_chart(option, out_dir / f"{stem}.html", title=f"{label} against ask-versus-got error", subtitle=sub, dark=DARK))

    # Revision mechanism figure across guidance strengths
    rev_recs = [r for r in u_recs if r.get("revisions")]
    if rev_recs:
        gs = [f"g={r['guidance']:g}" for r in rev_recs]
        series = []
        field_palette = {
            "species": "#0f766e",
            "ability": "#2563eb",
            "item": "#d97706",
            "move": "#dc2626",
            "nature": "#7c3aed",
            "stat": "#4b5563",
        }
        field_names = {"species": "Species", "ability": "Ability", "item": "Item", "move": "Move", "nature": "Stat Alignment", "stat": "Stat Points"}
        for fk in UMath.FIELD_KEYS:
            series.append({
                "name": field_names[fk],
                "type": "line",
                "color": field_palette[fk],
                "symbolSize": 7,
                "lineStyle": {"width": 2.2},
                "emphasis": {"focus": "series"},
                "data": [round(r["revisions"]["by_field"][fk]["changes_of_mind"], 3) for r in rev_recs],
            })
        option = {
            "animation": False,
            "legend": {"top": 4},
            "tooltip": {"trigger": "axis"},
            "grid": {"left": 70, "right": 28, "top": 48, "bottom": 48},
            "xAxis": {"type": "category", "data": gs, "name": "Guidance strength", "nameLocation": "middle", "nameGap": 28},
            "yAxis": {"type": "value", "name": "Changes of mind per field", "nameLocation": "middle", "nameGap": 46},
            "series": series,
        }
        paths.append(write_chart(option, out_dir / "revisions-by-guidance.html", title="Changes of mind across guidance strengths by field type (Uniform, no filter)", subtitle="Mean top-prediction flips per column across reverse passes z^(0)..z^(K) before validator redraws.", dark=DARK))
    return paths


def format_frontier_verdict_comment(report: dict, base_url: str) -> str:
    v = report["verdict"]
    lines = [
        "**Full Part A frontier & revision verdict done (Items 3 & 4, #279/#280): within-ask diversity, novelty, dominated area, highest usable guidance, and revision mechanism.**",
        "",
        _picture(base_url, "frontier-wa-novelty", "Within-ask novel line-up share against ask-versus-got error for shared-recipe Uniform vs Masked"),
        "",
        _picture(base_url, "frontier-wa-uniqueness", "Within-ask line-up uniqueness against ask-versus-got error for shared-recipe Uniform vs Masked"),
        "",
        _picture(base_url, "frontier-wa-srd", "Within-ask mean species replacement distance against ask-versus-got error"),
        "",
        _picture(base_url, "revisions-by-guidance", "Changes of mind per field type across guidance weights for shared-recipe Uniform"),
        "",
        f"Highest usable guidance `g_max`: Uniform, no filter = **{v['uniform_g_max']}**, Masked, full filter = **{v['masked_g_max']}**.",
        "",
        "### Dominated area (`HV`, reference box `error_ref = 0.025`) and equal-error advantage",
        "",
        "| Frontier | Uniform HV (usable / all) | Masked HV | Paired diff (usable) | Equal-error advantage `Δm_eq` | Verdict |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for f in v["frontiers"].values():
        adv = f["equal_error_advantage"]["mean"]
        adv_txt = f"{adv:+.4f}" if adv is not None else "no overlap"
        lines.append(
            f"| {f['label']} | {f['uniform_hv_usable']['mean']:.5f} / {f['uniform_hv_all']['mean']:.5f} | "
            f"{f['masked_hv']['mean']:.5f} | {f['diff_usable']['mean']:+.5f} | {adv_txt} | **{f['verdict']}** |"
        )

    lines += [
        "",
        "### Guidance-by-guidance within-ask diversity, legality, and revision counts (Uniform, no filter)",
        "",
        "| Guidance | Usable? | Error (1–17) | First-try legality | WA novel | WA uniq. | WA species repl. | Cat-48 uniq. | Spreads uniq. | Species `R_c` / floor / `M_c` |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in sorted(report["uniform_records"], key=lambda x: x["guidance"]):
        rev = r.get("revisions")
        if rev:
            sp = rev["by_field"]["species"]
            rev_txt = f"{sp['value_changes_beyond_first']:.2f} / {sp['noise_floor']:.2f} / {sp['changes_of_mind']:.2f}"
        else:
            rev_txt = "—"
        err_txt = f"{r['error']['mean']:.4f}" if r["error"]["mean"] is not None else "—"
        lines.append(
            f"| {r['guidance']:g} | {'Yes' if r.get('usable') else 'No'} | {err_txt} | {r['legality']['mean']:.1%} | "
            f"{r['within_ask_novel']['mean']:.1%} | {r['within_ask_unique']['mean']:.3f} | {r['within_ask_srd']['mean']:.3f} | "
            f"{r['within_ask_cat48_unique']['mean']:.3f} | {r['within_ask_spreads_unique']['mean']:.3f} | {rev_txt} |"
        )
    front_txt = ", ".join(f"{p['arm']} g={p['guidance']:g}" for p in v["combined_pareto_front"]) or "none"
    lines += [
        "",
        f"Combined 5-axis Pareto front (non-dominated seed-for-seed): {front_txt}.",
        "",
    ]
    return "\n".join(lines)


def _load_grid(results_dir: Path, name: str) -> dict:
    return json.loads((results_dir / f"stat_points_84_{name}.json").read_text())


def _summarize_unguided_cell_file(path: Path):
    if not path.exists():
        return None
    grid = json.loads(path.read_text())
    ys = [y for c in grid["cells"].values() for y in c.get("y", [])]
    tried = sum(int(c.get("tried", len(c.get("pastes", [])))) for c in grid["cells"].values())
    rejected = sum(int(c.get("rejected", 0)) for c in grid["cells"].values())
    battles = sum(int(c.get("battles", 0)) for c in grid["cells"].values())
    if not ys:
        return dict(legality=(tried - rejected) / max(tried, 1), win=None, win_interval=None, battles=battles)
    rng = np.random.default_rng(43802)
    arr = np.asarray(ys, dtype=np.float64)
    idx = rng.integers(0, len(arr), size=(2000, len(arr)))
    lo, hi = np.percentile(arr[idx].mean(axis=1), [2.5, 97.5])
    return dict(
        legality=(tried - rejected) / max(tried, 1),
        win=float(arr.mean()),
        win_interval=[float(lo), float(hi)],
        battles=battles,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=RES_DIR)
    sub = parser.add_subparsers(dest="cmd", required=True)

    pc = sub.add_parser("pass-count-comment")
    pc.add_argument("pass_count_json", type=Path)
    pc.add_argument("base_url")
    pc.add_argument("--old-json", type=Path, default=RES_DIR / "udlm_pass_count_g1.json")

    ug = sub.add_parser("unguided-report")
    ug.add_argument("passes", type=int)
    ug.add_argument("val_loss_json", type=Path)
    ug.add_argument("figure_dir", type=Path)
    ug.add_argument("output_json", type=Path)

    ugc = sub.add_parser("unguided-comment")
    ugc.add_argument("summary_json", type=Path)
    ugc.add_argument("base_url")

    bp = sub.add_parser("bend-points")
    bp.add_argument("passes", type=int)

    fv = sub.add_parser("frontier-verdict")
    fv.add_argument("passes", type=int)
    fv.add_argument("--guidances", required=True, help="Comma-separated uniform guidance weights (e.g. 1,2,4,8,5.7)")
    fv.add_argument("--figure-dir", type=Path, required=True)
    fv.add_argument("--output-json", type=Path, required=True)

    fvc = sub.add_parser("frontier-comment")
    fvc.add_argument("report_json", type=Path)
    fvc.add_argument("base_url")

    args = parser.parse_args()

    if args.cmd == "pass-count-comment":
        shared = json.loads(args.pass_count_json.read_text())
        old = json.loads(args.old_json.read_text()) if args.old_json.exists() else None
        print(format_pass_count_comment(shared, old, args.base_url), end="")
    elif args.cmd == "unguided-report":
        vl = json.loads(args.val_loss_json.read_text())["arms"] if args.val_loss_json.exists() else {}
        u_row = f"{UNIFORM_ROW}_k{args.passes}"
        u_g0 = _summarize_unguided_cell_file(args.results_dir / f"stat_points_84_{grid_name(u_row, None, unguided=True)}.json")
        m_g0 = _summarize_unguided_cell_file(args.results_dir / f"stat_points_84_{grid_name(MASKED_ROW, None, unguided=True)}.json")
        u_g1_path = args.results_dir / f"stat_points_84_{grid_name(u_row, 1)}.json"
        u_g1_err, u_g1_leg = None, None
        if u_g1_path.exists():
            g1 = json.loads(u_g1_path.read_text())
            u_g1_err = (g1.get("summary") or {}).get("line", {}).get("mse_scored_asks")
            cells_17 = [c for c in g1["cells"].values() if c["numerator"] in SCORED_ASKS]
            tr = sum(int(c.get("tried", 0)) for c in cells_17)
            rj = sum(int(c.get("rejected", 0)) for c in cells_17)
            u_g1_leg = (tr - rj) / max(tr, 1)
        m_g1_path = args.results_dir / f"stat_points_84_{grid_name(MASKED_ROW, 1)}.json"
        m_g1_err, m_g1_leg = None, None
        if m_g1_path.exists():
            mg1 = json.loads(m_g1_path.read_text())
            m_g1_err = (mg1.get("summary") or {}).get("line", {}).get("mse_scored_asks")
            m_g1_leg = 1.0
        summary = dict(
            issue=438,
            passes=args.passes,
            arms=[
                dict(
                    id="uniform_gidd_lr",
                    reader="Uniform, no filter (shared recipe)",
                    val_loss=(vl.get("uniform_gidd_lr") or {}).get("val_loss"),
                    g0_legality=(u_g0 or {}).get("legality"),
                    g0_win=(u_g0 or {}).get("win"),
                    g0_win_interval=(u_g0 or {}).get("win_interval"),
                    g0_battles=(u_g0 or {}).get("battles"),
                    g1_legality=u_g1_leg,
                    g1_error=u_g1_err,
                ),
                dict(
                    id="masked_gidd_lr",
                    reader="Masked, full filter (shared recipe)",
                    val_loss=(vl.get("masked_gidd_lr") or {}).get("val_loss"),
                    g0_legality=(m_g0 or {}).get("legality"),
                    g0_win=(m_g0 or {}).get("win"),
                    g0_win_interval=(m_g0 or {}).get("win_interval"),
                    g0_battles=(m_g0 or {}).get("battles"),
                    g1_legality=m_g1_leg,
                    g1_error=m_g1_err,
                ),
            ],
        )
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, indent=2))
        build_unguided_figure(summary, args.figure_dir)
    elif args.cmd == "unguided-comment":
        print(format_unguided_comment(json.loads(args.summary_json.read_text()), args.base_url), end="")
    elif args.cmd == "bend-points":
        ref = _build_reference()
        u_row = f"{UNIFORM_ROW}_k{args.passes}"
        recs = [
            summarize_grid_diversity_and_revisions(_load_grid(args.results_dir, grid_name(u_row, g)), ref)
            for g in (1, 2, 4, 8)
        ]
        decision = evaluate_usability_and_bend_points(recs)
        for w in decision["bend_points"]:
            print(f"{w:g}")
    elif args.cmd == "frontier-verdict":
        ref = _build_reference()
        u_row = f"{UNIFORM_ROW}_k{args.passes}"
        u_gs = [float(x) for x in args.guidances.split(",") if x.strip()]
        u_recs = [
            summarize_grid_diversity_and_revisions(_load_grid(args.results_dir, grid_name(u_row, g)), ref)
            for g in u_gs
        ]
        m_gs = [g for g in (1, 2, 4, 8) if (args.results_dir / f"stat_points_84_{grid_name(MASKED_ROW, g)}.json").exists()]
        m_recs = [
            summarize_grid_diversity_and_revisions(_load_grid(args.results_dir, grid_name(MASKED_ROW, g)), ref)
            for g in m_gs
        ]
        verdict = compute_frontier_verdict(u_recs, m_recs)
        report = dict(issue=438, passes=args.passes, uniform_records=u_recs, masked_records=m_recs, verdict=verdict)
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2))
        build_frontier_and_revision_figures(report, args.figure_dir)
    elif args.cmd == "frontier-comment":
        print(format_frontier_verdict_comment(json.loads(args.report_json.read_text()), args.base_url), end="")


if __name__ == "__main__":
    main()
