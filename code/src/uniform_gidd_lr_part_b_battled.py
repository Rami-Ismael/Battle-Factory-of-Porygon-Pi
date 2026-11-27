"""Part B battled evaluations (Items 8-12) for Issue #438, reusing Part A's ask grids.

Covers:
  - Item 8 & 11: Deep 16-battle re-battle & weakness pressure (worst-5-matchup win rate on
    fresh 16-battle-per-opponent scores, flip share vs 4-battle re-battle, held & novel-held
    elite line-ups).
  - Item 9: Pass@k for k = 1..64 (unbiased estimator 1 - C(n-c, k)/C(n, k), noting Fang et al.
    App. E's n >= 4k mark at k = 48) and Pass@NFE accounting for validator redraws.
  - Item 10: Held-out meta teams, ranks 51-100 (heldout.py) scored ONCE on the final selected
    top elite line-ups (strictly after all top-50 selection; zero held-out leakage).
  - Item 12: High asks (18/24 to 23/24) and strong training teams — win rate, MSE on 18-23,
    exact-copy rate, and nearest-training-team swap distribution (0, 1, 2, 3+ swaps).

Pure analysis and ECharts helpers load no /tmp/vgc-pilot modules so unit tests run cleanly on CPU.
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from echarts_chart import write_chart
import pass_at_k as PAK

RES_DIR = Path.home() / "vgc-data/results"
UNIFORM_ROW = "udlm_loglinear_gidd_lr"
MASKED_ROW = "shuffled_linear_filtered_gidd_lr"
OLD_MASKED_ROW = "linear_masked_filter"
HIGH_ASKS = tuple(range(18, 24))
COLORS = {"Uniform (shared recipe)": "#0f766e", "Masked, full filter (shared recipe)": "#365d93", "Masked, full filter (4e-4 recipe)": "#6f6e69"}
DARK = {"#0f766e": "#5eead4", "#365d93": "#93c5fd", "#6f6e69": "#a9a8a2"}


def _grid_name(row: str, guidance: float) -> str:
    return row if float(guidance) == 2.0 else f"{row}_g{float(guidance):g}"


def _picture(base_url: str, stem: str, alt: str) -> str:
    return (
        f'<picture><source media="(prefers-color-scheme: dark)" srcset="{base_url}/{stem}-dark.png">'
        f'<img src="{base_url}/{stem}-light.png" alt="{alt}" width="900"></picture>'
    )


def bootstrap_mean_ci(values, *, draws=2000, seed=43812):
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return dict(mean=None, interval=[None, None], n=0)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(draws, arr.size))
    lo, hi = np.percentile(arr[idx].mean(axis=1), [2.5, 97.5])
    return dict(mean=float(arr.mean()), interval=[float(lo), float(hi)], n=int(arr.size))


# ---------------------------------------------------------------- Item 12: High asks & training-team swaps
def analyze_high_asks_and_swaps(grid: dict, train_lineups: set[frozenset[str]]):
    """Compute asks 18/24-23/24 win rates, MSE, exact-copy rate, and nearest-training-team swap counts."""
    from line_up_yield import line_up

    train_sets = [frozenset(lu) for lu in train_lineups]
    cache: dict[tuple[str, ...], int] = {}

    def min_swaps(lu_tuple):
        if lu_tuple in cache:
            return cache[lu_tuple]
        s = frozenset(lu_tuple)
        if s in train_lineups:
            cache[lu_tuple] = 0
            return 0
        best = 6
        for ts in train_sets:
            d = 6 - len(s & ts)
            if d < best:
                best = d
                if best == 1:
                    break
        cache[lu_tuple] = best
        return best

    by_ask = {}
    high_ys = []
    high_swaps = []
    all_swaps = []
    for ask in range(1, 24):
        cells = [c for c in grid["cells"].values() if c["numerator"] == ask]
        ys = [float(y) for c in cells for y in c.get("y", [])]
        pastes = [p for c in cells for p in c.get("pastes", [])]
        swaps = [min_swaps(line_up(p)) for p in pastes]
        all_swaps.extend(swaps)
        if ask in HIGH_ASKS:
            high_ys.extend(ys)
            high_swaps.extend(swaps)
            by_ask[str(ask)] = dict(
                teams=len(pastes),
                win=bootstrap_mean_ci(ys, seed=43812 + ask),
                exact_copy_rate=float(np.mean([s == 0 for s in swaps])) if swaps else None,
            )

    mse_18_23 = (
        float(np.mean([(by_ask[str(k)]["win"]["mean"] - k / 24.0) ** 2 for k in HIGH_ASKS if by_ask[str(k)]["win"]["mean"] is not None]))
        if any(by_ask[str(k)]["win"]["mean"] is not None for k in HIGH_ASKS)
        else None
    )

    def swap_hist(sw):
        n = max(len(sw), 1)
        return {
            "0_swaps_exact": sum(s == 0 for s in sw) / n,
            "1_swap": sum(s == 1 for s in sw) / n,
            "2_swaps": sum(s == 2 for s in sw) / n,
            "3_plus_swaps": sum(s >= 3 for s in sw) / n,
            "n": len(sw),
        }

    return dict(
        guidance=float(grid.get("guidance", 2.0)),
        mse_asks_18_to_23=mse_18_23,
        high_asks_win=bootstrap_mean_ci(high_ys, seed=43813),
        high_asks_swaps=swap_hist(high_swaps),
        all_asks_swaps=swap_hist(all_swaps),
        by_ask=by_ask,
    )


def build_high_asks_figure(report: dict, out_dir: Path) -> Path:
    """Two-panel chart: asks 18/24-23/24 measured win rate and nearest-training-team swap distribution."""
    asks = [f"{k}/24" for k in HIGH_ASKS]
    swap_cats = ["0 (exact copy)", "1 swap", "2 swaps", "3+ swaps"]
    swap_keys = ("0_swaps_exact", "1_swap", "2_swaps", "3_plus_swaps")
    series = []
    for arm_label, arm_data in report["arms"].items():
        color = COLORS.get(arm_label, "#0f766e")
        # Pick guidance 8 (or highest guidance present) for the clearest high-ask comparison
        best_g = max(arm_data.keys(), key=lambda x: float(x))
        rec = arm_data[best_g]
        series.append({
            "name": f"{arm_label} (g={best_g})",
            "type": "line",
            "xAxisIndex": 0,
            "yAxisIndex": 0,
            "color": color,
            "symbolSize": 7,
            "lineStyle": {"width": 2.4},
            "emphasis": {"focus": "series"},
            "data": [round(rec["by_ask"][str(k)]["win"]["mean"], 4) if rec["by_ask"][str(k)]["win"]["mean"] is not None else None for k in HIGH_ASKS],
        })
        series.append({
            "name": f"{arm_label} (g={best_g})",
            "type": "bar",
            "xAxisIndex": 1,
            "yAxisIndex": 1,
            "color": color,
            "emphasis": {"focus": "series"},
            "data": [round(rec["high_asks_swaps"][sk], 4) for sk in swap_keys],
        })
    option = {
        "animation": False,
        "legend": {"top": 4},
        "tooltip": {"trigger": "axis", "valueFormatter": "js:v => v == null ? '' : (Number(v) * 100).toFixed(1) + '%'"},
        "title": [
            {"text": "Win rate on high asks (18/24 to 23/24)", "left": "25%", "top": 36, "textAlign": "center", "textStyle": {"fontSize": 13}},
            {"text": "Nearest training line-up distance (asks 18/24–23/24)", "left": "75%", "top": 36, "textAlign": "center", "textStyle": {"fontSize": 13}},
        ],
        "grid": [
            {"left": 64, "width": "40%", "top": 74, "bottom": 48},
            {"left": "55%", "width": "40%", "top": 74, "bottom": 48},
        ],
        "xAxis": [
            {"type": "category", "gridIndex": 0, "data": asks},
            {"type": "category", "gridIndex": 1, "data": swap_cats},
        ],
        "yAxis": [
            {"type": "value", "gridIndex": 0, "min": 0, "max": 1.0, "axisLabel": {"formatter": "js:v => Math.round(v * 100) + '%'"}},
            {"type": "value", "gridIndex": 1, "min": 0, "max": 1.0, "axisLabel": {"formatter": "js:v => Math.round(v * 100) + '%'"}},
        ],
        "series": series,
    }
    return write_chart(
        option,
        out_dir / "high-asks-training-swaps.html",
        title="High asks (18/24 to 23/24) win rates and nearest-training-team swap distances",
        subtitle="Measured against the 214,055 labelled training line-ups of lossdata84_shuffled_only.npz.",
        height=430,
        dark=DARK,
    )


# ---------------------------------------------------------------- Item 9: Pass@k (k = 1..64)
def compute_pass_at_k_64(grid_names: list[str], *, results_dir: Path = RES_DIR):
    """Compute unbiased Pass@k for k = 1..64 on asks 13..17 and 18..23 (noting Fang et al. App. E n>=4k at k=48)."""
    out = dict(issue=438, k_values=list(range(1, 65)), fang_bound_k=48, grids={})
    for name in grid_names:
        gpath = results_dir / f"stat_points_84_{name}.json"
        if not gpath.exists():
            continue
        grid = json.loads(gpath.read_text())
        ypath = results_dir / f"line_up_yield_{name}.json"
        held_lineups = None
        if ypath.exists():
            yd = json.loads(ypath.read_text())
            held_lineups = {tuple(lu) for lu in yd["views"]["all_23_asks"]["held_line_ups"]}
        from line_up_yield import line_up

        by_band = {}
        for band_name, asks in (("asks_13_to_17", range(13, 18)), ("asks_18_to_23", range(18, 24))):
            curves = []
            tries_per_kept = []
            for ask in asks:
                cells = [c for c in grid["cells"].values() if c["numerator"] == ask]
                pastes = [p for c in cells for p in c.get("pastes", [])]
                ys = [float(y) for c in cells for y in c.get("y", [])]
                n = len(ys)
                if n < 64:
                    continue
                if held_lineups is not None:
                    c_succ = sum(1 for p, y in zip(pastes, ys) if y >= PAK.CUT and line_up(p) in held_lineups)
                else:
                    c_succ = sum(1 for y in ys if y >= PAK.CUT)
                curves.append([PAK.pass_at_k(n, c_succ, k) for k in range(1, 65)])
                tried = sum(int(c.get("tried", len(c.get("pastes", [])))) for c in cells)
                tries_per_kept.append(tried / max(n, 1))
            mean_curve = np.mean(curves, axis=0).tolist() if curves else None
            passes_per_try = int(grid.get("passes", 84))
            mult = 2 if (not grid.get("unguided") and float(grid.get("guidance", 2.0)) != 1.0) else 1
            nfe_per_team = float(np.mean(tries_per_kept) * passes_per_try * mult) if tries_per_kept else None
            by_band[band_name] = dict(
                asks_used=len(curves),
                definition="held" if held_lineups is not None else "one_score",
                pass_at_k=mean_curve,
                pass_at_1=mean_curve[0] if mean_curve else None,
                pass_at_8=mean_curve[7] if mean_curve else None,
                pass_at_48=mean_curve[47] if mean_curve else None,
                pass_at_64=mean_curve[63] if mean_curve else None,
                nfe_per_legal_team=nfe_per_team,
            )
        out["grids"][name] = dict(reader=grid.get("reader", name), guidance=float(grid.get("guidance", 2.0)), passes=int(grid.get("passes", 84)), bands=by_band)
    return out


def build_pass_at_k_64_figure(report: dict, out_dir: Path) -> Path:
    ks = report["k_values"]
    series = []
    palette = ["#0f766e", "#0d9488", "#14b8a6", "#2dd4bf", "#365d93", "#2563eb", "#60a5fa"]
    for idx, (name, rec) in enumerate(report["grids"].items()):
        curve = rec["bands"]["asks_13_to_17"].get("pass_at_k")
        if not curve:
            continue
        series.append({
            "name": f"{name} (g={rec['guidance']:g})",
            "type": "line",
            "color": palette[idx % len(palette)],
            "symbol": "none",
            "lineStyle": {"width": 2.3, "type": "solid" if "udlm" in name else "dashed"},
            "emphasis": {"focus": "series"},
            "data": [round(v, 4) for v in curve],
        })
    option = {
        "animation": False,
        "legend": {"top": 4, "type": "scroll"},
        "tooltip": {"trigger": "axis", "valueFormatter": "js:v => (Number(v) * 100).toFixed(1) + '%'"},
        "grid": {"left": 70, "right": 28, "top": 52, "bottom": 52},
        "xAxis": {"type": "category", "data": [str(k) for k in ks], "name": "Sample budget k (Fang et al. n >= 4k bound at k=48)", "nameLocation": "middle", "nameGap": 30},
        "yAxis": {"type": "value", "min": 0, "max": 1.05, "name": "Pass@k (asks 13/24 to 17/24)", "nameLocation": "middle", "nameGap": 48, "axisLabel": {"formatter": "js:v => Math.round(v * 100) + '%'"}},
        "series": series,
    }
    return write_chart(
        option,
        out_dir / "pass-at-k-64.html",
        title="Pass@k from k = 1 to 64 on asks 13/24 to 17/24 (held elite line-ups)",
        subtitle="Unbiased estimator 1 − C(n−c, k)/C(n, k) across 192 teams per ask; solid = Uniform, dashed = Masked.",
        height=420,
        dark=DARK,
    )


# ---------------------------------------------------------------- Item 8 & 11: Deep 16-battle + weakness pressure
def _worst5_from_deep_db(con, policy, cols, fresh_rows, MDB):
    if con is None or policy is None or not cols:
        return None
    worst5_scores = []
    for f in fresh_rows:
        if f["fresh"] < PAK.CUT:
            continue
        trow = con.execute("SELECT team_id FROM team WHERE canon_hash=?", (MDB.canon_hash(MDB.canonical_paste(f["paste"])),)).fetchone()
        if trow is None:
            continue
        cells = MDB.row_cells(con, policy, trow[0], cols)
        rates = sorted(w / n for (w, n) in cells.values() if n > 0)
        if len(rates) >= 5:
            worst5_scores.append(float(np.mean(rates[:5])))
    return bootstrap_mean_ci(worst5_scores, seed=43811) if worst5_scores else None


def summarize_deep_and_weakness(grid_names: list[str], deep_db: Path, *, results_dir: Path = RES_DIR):
    """Read 4-battle and 16-battle line_up_yield results plus worst-5-matchup weakness pressure from deep_db."""
    sys.path.insert(0, "/tmp/vgc-pilot/src")
    sys.path.insert(0, str(HERE))
    import matchup_db as MDB

    con = sqlite3.connect(f"file:{deep_db}?mode=ro", uri=True) if deep_db.exists() else None
    policy, cols = None, []
    if con is not None:
        row = con.execute("SELECT policy_id FROM policy WHERE row_policy LIKE '%+slots-shuffled' AND row_policy NOT LIKE 'rust%'").fetchone()
        policy = row[0] if row else None
        cols = [r[0] for r in con.execute("SELECT team_id FROM team_set WHERE set_name='top50' ORDER BY team_id")]

    rows = {}
    for name in grid_names:
        y4_path = results_dir / f"line_up_yield_{name}.json"
        if not y4_path.exists():
            continue
        v4 = json.loads(y4_path.read_text())["views"]["all_23_asks"]
        rec = dict(
            name=name,
            elite_candidates=v4["elite_line_ups"],
            held_at_4=v4["held"],
            novel_held_at_4=v4.get("held_new_against_training"),
            held_at_16=None,
            novel_held_at_16=None,
            flip_share=None,
            flip_share_interval=None,
            worst5_mean_d16=None,
            worst5_interval_d16=None,
        )
        y16_path = results_dir / f"line_up_yield_{name}_d16.json"
        if y16_path.exists():
            y16 = json.loads(y16_path.read_text())
            v16 = y16["views"]["all_23_asks"]
            flips = y16.get("flips") or {}
            boot = _worst5_from_deep_db(con, policy, cols, y16.get("fresh", []), MDB)
            rec.update(
                held_at_16=v16["held"],
                novel_held_at_16=v16.get("held_new_against_training"),
                flip_share=flips.get("flip_share"),
                flip_share_interval=flips.get("flip_share_interval"),
                worst5_mean_d16=boot["mean"] if boot else None,
                worst5_interval_d16=boot["interval"] if boot else None,
            )
        rows[name] = rec
    if con is not None:
        con.close()
    return dict(issue=438, grids=rows)


# ---------------------------------------------------------------- Item 10: Held-out meta teams ranks 51-100
def run_heldout_once(grid_names: list[str], out_path: Path, db_path: Path, *, top_n: int = 16, per_cell: int = 4):
    """Score the final selected top held elite line-ups ONCE against the 50 held-out meta teams (ranks 51-100)."""
    sys.path.insert(0, "/tmp/vgc-pilot/src")
    sys.path.insert(0, str(HERE))
    import heldout as HO
    import matchup_db as MDB
    from hps_generate import Validator

    if MDB.SLOT_ORDER != "shuffled":
        raise SystemExit(f"VGC_SLOT_ORDER={MDB.SLOT_ORDER}: held-out battles must shuffle slots")

    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = MDB.connect(db_path)
    val = Validator()
    try:
        with con:
            con.execute("DELETE FROM team_set WHERE set_name=?", (HO.SET_NAME,))
            for fpath in HO.files():
                tid = MDB.add_team(con, Path(fpath).read_text(), "vgcpastes", val)[0]
                con.execute(
                    "INSERT INTO team_set VALUES (?, ?, 1) ON CONFLICT (set_name, team_id) DO UPDATE SET weight = weight + 1",
                    (HO.SET_NAME, tid),
                )
    finally:
        val.close()

    report = dict(issue=438, opponent_set=HO.SET_NAME, opponents=HO.N, per_cell=per_cell, top_n=top_n, arms={})
    for name in grid_names:
        # Prefer 16-battle re-battle ranking if available, else 4-battle re-battle ranking
        ypath = RES_DIR / f"line_up_yield_{name}_d16.json"
        if not ypath.exists():
            ypath = RES_DIR / f"line_up_yield_{name}.json"
        if not ypath.exists():
            continue
        yd = json.loads(ypath.read_text())
        ranked = sorted(yd.get("fresh", []), key=lambda r: -r["fresh"])[:top_n]
        pastes = [r["paste"] for r in ranked]
        if any(HO.is_heldout_text(p) for p in pastes):
            raise HO.HeldoutLeak(f"{name}: held-out team found in final candidate list")
        scores = MDB.score_pastes(con, pastes, set_name=HO.SET_NAME, per_cell=per_cell, seed=43851, origin=f"heldout438:{name}")
        ys = [float(s["score"]) for s in scores]
        top50_ys = [float(r["fresh"]) for r in ranked]
        report["arms"][name] = dict(
            n_teams=len(pastes),
            battles=sum(int(s["battles"]) for s in scores),
            top50_rebattle_mean=float(np.mean(top50_ys)) if top50_ys else None,
            heldout_win=bootstrap_mean_ci(ys, seed=43851),
            teams=[dict(top50_fresh=r["fresh"], heldout_win=s["score"], battles=s["battles"]) for r, s in zip(ranked, scores)],
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(f"HELDOUT_438_DONE {out_path}", flush=True)
    return report


def build_part_b_comment(high_asks: dict, pak: dict, deep: dict, heldout: dict | None, base_url: str) -> str:
    lines = [
        "**Part B battled evaluations done (Items 8, 9, 10, 11, and 12): elite yield (4 & 16 battles), weakness pressure, Pass@k (k=1..64), held-out ranks 51–100, and high asks 18/24–23/24.**",
        "",
        _picture(base_url, "high-asks-training-swaps", "High asks 18/24 to 23/24 win rates and nearest-training-team swap distances"),
        "",
        _picture(base_url, "pass-at-k-64", "Pass@k from k=1 to 64 on asks 13/24 to 17/24 for held elite line-ups"),
        "",
        "### Items 8 & 11 — Elite line-ups that hold (4 vs 16 battles/meta team) and weakness pressure",
        "",
        "| Grid | Elite candidates | Held at 4 (novel) | Held at 16 (novel) | Flip share [95% CI] | Worst-5-matchup win rate (16 battles) |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, r in deep["grids"].items():
        h4 = f"{r['held_at_4']} ({r['novel_held_at_4']})"
        h16 = f"{r['held_at_16']} ({r['novel_held_at_16']})" if r["held_at_16"] is not None else "—"
        fl = f"{r['flip_share']:.1%} [{r['flip_share_interval'][0]:.1%}, {r['flip_share_interval'][1]:.1%}]" if r["flip_share"] is not None and r["flip_share_interval"] else "—"
        w5 = f"{r['worst5_mean_d16']:.3f} [{r['worst5_interval_d16'][0]:.3f}, {r['worst5_interval_d16'][1]:.3f}]" if r["worst5_mean_d16"] is not None else "—"
        lines.append(f"| `{name}` | {r['elite_candidates']} | {h4} | {h16} | {fl} | {w5} |")

    lines += [
        "",
        "### Item 9 — Pass@k (`k = 1, 8, 48, 64`) and NFE per legal team (asks 13/24 to 17/24)",
        "",
        "| Grid | Definition | Pass@1 | Pass@8 | Pass@48 (Fang bound) | Pass@64 | NFE / legal team |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, r in pak["grids"].items():
        b = r["bands"]["asks_13_to_17"]
        fmt = lambda v: f"{v:.3f}" if v is not None else "—"
        nfe = f"{b['nfe_per_legal_team']:,.0f}" if b["nfe_per_legal_team"] is not None else "—"
        lines.append(f"| `{name}` | {b['definition']} | {fmt(b['pass_at_1'])} | {fmt(b['pass_at_8'])} | {fmt(b['pass_at_48'])} | {fmt(b['pass_at_64'])} | {nfe} |")

    if heldout and heldout.get("arms"):
        lines += [
            "",
            "### Item 10 — Held-out meta teams, ranks 51–100 (scored once on final top elite line-ups)",
            "",
            "| Grid | Top teams scored | Top-50 re-battle win rate | Held-out 51–100 win rate [95% CI] | Held-out battles |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        for name, r in heldout["arms"].items():
            hw = r["heldout_win"]
            hw_txt = f"{hw['mean']:.3f} [{hw['interval'][0]:.3f}, {hw['interval'][1]:.3f}]" if hw["mean"] is not None else "—"
            lines.append(f"| `{name}` | {r['n_teams']} | {r['top50_rebattle_mean']:.3f} | {hw_txt} | {r['battles']:,} |")

    lines += [
        "",
        "### Item 12 — High asks (18/24 to 23/24) and nearest-training-team swap distances",
        "",
        "| Model & guidance | Mean win (18–23) [95% CI] | MSE (18–23) | Exact copy (0 swaps) | 1 swap | 2 swaps | 3+ swaps |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for arm_label, by_g in high_asks["arms"].items():
        for g, rec in sorted(by_g.items(), key=lambda kv: float(kv[0])):
            w = rec["high_asks_win"]
            w_txt = f"{w['mean']:.3f} [{w['interval'][0]:.3f}, {w['interval'][1]:.3f}]" if w["mean"] is not None else "—"
            mse = f"{rec['mse_asks_18_to_23']:.4f}" if rec["mse_asks_18_to_23"] is not None else "—"
            sw = rec["high_asks_swaps"]
            lines.append(
                f"| {arm_label} (`g={g}`) | {w_txt} | {mse} | {sw['0_swaps_exact']:.1%} | {sw['1_swap']:.1%} | {sw['2_swaps']:.1%} | {sw['3_plus_swaps']:.1%} |"
            )

    lines += [
        "",
        "Items 13 (in-place infilling) and 14 (temperature and species repulsion) stay excluded pending separate pre-registration, as #438 specifies.",
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=RES_DIR)
    sub = parser.add_subparsers(dest="cmd", required=True)

    ha = sub.add_parser("high-asks")
    ha.add_argument("passes", type=int)
    ha.add_argument("--guidances", required=True)
    ha.add_argument("--output-json", type=Path, required=True)
    ha.add_argument("--figure-dir", type=Path, required=True)

    pk = sub.add_parser("pass-at-k-64")
    pk.add_argument("passes", type=int)
    pk.add_argument("--guidances", required=True)
    pk.add_argument("--output-json", type=Path, required=True)
    pk.add_argument("--figure-dir", type=Path, required=True)

    dp = sub.add_parser("deep-summary")
    dp.add_argument("passes", type=int)
    dp.add_argument("--guidances", required=True)
    dp.add_argument("--deep-db", type=Path, required=True)
    dp.add_argument("--output-json", type=Path, required=True)

    ho = sub.add_parser("heldout-final")
    ho.add_argument("passes", type=int, nargs="?", default=int(os.environ.get("HELDOUT438_PASSES", "84")))
    ho.add_argument("--guidances", default=os.environ.get("HELDOUT438_GUIDANCES", "1,2,4,8"))
    ho.add_argument("--db", type=Path, default=Path(os.environ.get("HELDOUT438_DB", str(Path.home() / "vgc-data/uniform_gidd_lr_438/heldout.sqlite"))))
    ho.add_argument("--output-json", type=Path, default=Path(os.environ.get("HELDOUT438_OUT", str(Path.home() / "vgc-data/uniform_gidd_lr_438/part_b_heldout.json"))))
    ho.add_argument("--top-n", type=int, default=16)
    ho.add_argument("--per-cell", type=int, default=4)

    cm = sub.add_parser("comment")
    cm.add_argument("--high-asks", type=Path, required=True)
    cm.add_argument("--pass-at-k", type=Path, required=True)
    cm.add_argument("--deep", type=Path, required=True)
    cm.add_argument("--heldout", type=Path, default=None)
    cm.add_argument("--base-url", required=True)

    args = parser.parse_args()

    if args.cmd in ("high-asks", "pass-at-k-64", "deep-summary", "heldout-final"):
        u_row = f"{UNIFORM_ROW}_k{args.passes}"
        u_gs = [float(x) for x in args.guidances.split(",") if x.strip()]
        u_names = [_grid_name(u_row, g) for g in u_gs]
        m_names = [_grid_name(MASKED_ROW, g) for g in (2, 4, 8)]

    if args.cmd == "high-asks":
        import retrain_default_report as R

        train_lu = {frozenset(lu) for lu in R._train_lineups()}
        arms_out = {}
        for label, row, gs in (
            ("Uniform (shared recipe)", u_row, u_gs),
            ("Masked, full filter (shared recipe)", MASKED_ROW, (2, 4, 8)),
            ("Masked, full filter (4e-4 recipe)", OLD_MASKED_ROW, (2, 4, 8)),
        ):
            arms_out[label] = {}
            for g in gs:
                p = args.results_dir / f"stat_points_84_{_grid_name(row, g)}.json"
                if p.exists():
                    arms_out[label][f"{g:g}"] = analyze_high_asks_and_swaps(json.loads(p.read_text()), train_lu)
        rep = dict(issue=438, passes=args.passes, arms=arms_out)
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(rep, indent=2))
        build_high_asks_figure(rep, args.figure_dir)
    elif args.cmd == "pass-at-k-64":
        rep = compute_pass_at_k_64(u_names + m_names, results_dir=args.results_dir)
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(rep, indent=2))
        build_pass_at_k_64_figure(rep, args.figure_dir)
    elif args.cmd == "deep-summary":
        rep = summarize_deep_and_weakness(u_names + m_names, args.deep_db, results_dir=args.results_dir)
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(rep, indent=2))
    elif args.cmd == "heldout-final":
        # Score top uniform grid (best by held elite count) + masked g=8 once against ranks 51-100
        run_heldout_once([*u_names, m_names[-1]], args.output_json, args.db, top_n=args.top_n, per_cell=args.per_cell)
    elif args.cmd == "comment":
        ho_data = json.loads(args.heldout.read_text()) if (args.heldout and args.heldout.exists()) else None
        print(
            build_part_b_comment(
                json.loads(args.high_asks.read_text()),
                json.loads(args.pass_at_k.read_text()),
                json.loads(args.deep.read_text()),
                ho_data,
                args.base_url,
            ),
            end="",
        )


if __name__ == "__main__":
    main()
