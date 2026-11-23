"""ECharts reports for VGC doubles battle-noise and policy trade-off sweeps.

Builds two standalone ECharts pages:
  1. Battle noise & signal-to-noise comparison (within-pair 1-game SD and Matchup ICC)
  2. Head-to-head playing strength (paired win share with 95% confidence interval over team pairs)

Run:
  /tmp/vgc-pilot/.venv/bin/python src/foul_play_doubles_noise_report.py \
      ~/vgc-data/foul-play-doubles/2026-11-23-noise \
      ~/vgc-data/foul-play-doubles/2026-11-23-h2h \
      reports/foul-play-doubles/
"""
import json
import sys
from pathlib import Path

from echarts_chart import write_chart

POLICY_NAMES = {
    "bc": "Behaviour cloning, stochastic (current baseline)",
    "bc_det": "Behaviour cloning, deterministic argmax",
    "bc_det_noshuffle": "Behaviour cloning, deterministic, fixed slot order",
    "foulplay": "Showdown JS Foul Play, 75% stochastic vote",
    "foulplay_det": "Showdown JS Foul Play, deterministic argmax",
    "foulplay_doubles": "Showdown JS Foul Play, doubles evaluator",
    "foulplay_joint": "Showdown JS Foul Play, joint-action PUCT",
    "foulplay_hybrid": "Showdown JS Foul Play, 2-turn rollout + doubles eval",
    "montecarlo": "Showdown JS Flat Monte Carlo",
    "uct_rollout": "Showdown JS UCT with random playouts",
    "rust_foulplay": "Rust Reg M-B Foul Play, 75% stochastic vote (30k iters)",
    "rust_foulplay_det": "Rust Reg M-B Foul Play, deterministic argmax (30k iters)",
    "rust_foulplay_doubles": "Rust Reg M-B Foul Play, doubles evaluator (30k iters)",
    "rust_foulplay_joint": "Rust Reg M-B Foul Play, joint-action PUCT (30k iters)",
    "rust_foulplay_hybrid": "Rust Reg M-B Foul Play, 2-turn rollout + doubles eval (30k iters)",
    "rust_uct_rollout": "Rust Reg M-B UCT with random playouts (30k iters)",
    "rust_uct_rollout_100k": "Rust Reg M-B UCT with random playouts (100k iters, 1 sample)",
    "rust_uct_rollout_4x25k": "Rust Reg M-B UCT with random playouts (4 samples × 25k iters)",
    "rust_montecarlo": "Rust Reg M-B Flat Monte Carlo (30k iters)",
}


def load_noise(root: Path):
    rows = []
    if not root.exists():
        return rows
    for p in sorted(root.glob("*/summary.json")):
        name = p.parent.name
        d = json.loads(p.read_text())
        if "within_pair_var" not in d:
            continue
        rows.append({
            "key": name,
            "label": POLICY_NAMES.get(name, name),
            **d,
        })
    # Sort best signal-to-noise (highest ICC, lowest within-pair variance) first
    rows.sort(key=lambda r: (-r.get("icc", 0.0), r.get("within_pair_var", 1.0)))
    return rows


def build_noise_chart(rows, out_path: Path):
    ordered = rows[::-1]
    labels = [r["label"] for r in ordered]
    icc_data = [round(r.get("icc", 0.0), 3) for r in ordered]
    sd_data = [round(r.get("single_game_sd", 0.5), 3) for r in ordered]
    opt = {
        "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"}},
        "legend": {"top": 4},
        "grid": {"left": 380, "right": 50, "top": 40, "bottom": 45},
        "xAxis": {"type": "value", "min": 0, "max": 0.65,
                  "name": "Matchup ICC (higher = cleaner signal) / Single-game SD (lower = less noise)",
                  "nameLocation": "middle", "nameGap": 28},
        "yAxis": {"type": "category", "data": labels},
        "series": [
            {"name": "Matchup ICC (true signal share)", "type": "bar", "data": icc_data,
             "emphasis": {"focus": "series"}, "label": {"show": True, "position": "right"}},
            {"name": "Single-game noise SD (lower is better)", "type": "bar", "data": sd_data,
             "emphasis": {"focus": "series"}, "label": {"show": True, "position": "right"}},
        ],
    }
    note = " | ".join(
        f"{r['label']}: ICC {r.get('icc', 0):.3f}, 1-game SD {r.get('single_game_sd', 0):.3f}, "
        f"49-game label SE {r.get('label_se_49', 0):.4f}, split-half r {r.get('split_half_r', 0) or 0:.2f}"
        for r in rows[:6]
    )
    return write_chart(
        opt, out_path,
        "VGC Regulation M-B battle noise by battle policy",
        "Self-play on 24 fixed top-50 meta team pairs (8 games per pair, 192 battles each). "
        "Higher Matchup ICC and lower single-game SD yield cleaner offline win-rate labels for the surrogate.",
        note, height=max(420, 34 * len(rows) + 100),
    )


SHORT_NAMES = {
    "bc": "BC (stochastic)",
    "bc_det": "BC (deterministic)",
    "rust_foulplay_det": "Foul Play argmax",
    "rust_foulplay_doubles": "Foul Play doubles eval",
    "rust_foulplay_joint": "Foul Play joint PUCT",
    "rust_foulplay_hybrid": "Foul Play 2-turn hybrid",
    "rust_uct_rollout": "UCT random playouts",
    "rust_montecarlo": "Flat Monte Carlo",
}


def load_h2h(root: Path):
    rows = []
    if not root.exists():
        return rows
    for p in sorted(root.glob("*/summary.json")):
        d = json.loads(p.read_text())
        if "paired_mean" not in d:
            continue
        key = p.parent.name
        cfg = d.get("config", {})
        budget = f"{cfg['iterations']//1000}k" if cfg.get("iterations") else f"{cfg.get('ms', 0)} ms"
        x_name = SHORT_NAMES.get(d["x"], POLICY_NAMES.get(d["x"], d["x"]))
        y_name = SHORT_NAMES.get(d["y"], POLICY_NAMES.get(d["y"], d["y"]))
        rows.append({
            "key": key,
            "label": f"[{budget}] {x_name} vs {y_name}",
            **d,
        })
    rows.sort(key=lambda r: -r.get("paired_mean", 0.5))
    return rows


def build_h2h_chart(rows, out_path: Path):
    ordered = rows[::-1]
    labels = [r["label"] for r in ordered]
    pts = [[round(r["paired_mean"], 3), i] for i, r in enumerate(ordered)]
    bars = [[r["paired95"][0], r["paired95"][1], i] for i, r in enumerate(ordered)]
    opt = {
        "grid": {"left": 310, "right": 48, "top": 28, "bottom": 48},
        "tooltip": {"trigger": "item"},
        "xAxis": {"type": "value", "min": 0.2, "max": 1.0, "name": "First policy share of points (0.5 = even)",
                  "nameLocation": "middle", "nameGap": 28},
        "yAxis": {"type": "category", "data": labels, "axisLabel": {"fontSize": 11}},
        "series": [
            {"type": "custom", "name": "95% paired interval", "data": bars,
             "renderItem": "js:(params, api) => { const lo = api.coord([api.value(0), api.value(2)]), hi = api.coord([api.value(1), api.value(2)]); return {type: 'line', shape: {x1: lo[0], y1: lo[1], x2: hi[0], y2: hi[1]}, style: {stroke: api.visual('color'), lineWidth: 3, opacity: 0.7}}; }",
             "encode": {"x": [0, 1], "y": 2}, "z": 1},
            {"type": "scatter", "name": "Point share", "data": pts, "symbolSize": 10, "z": 3,
             "label": {"show": True, "position": "right", "distance": 6, "formatter": "{@[0]}"},
             "emphasis": {"focus": "series"},
             "markLine": {"silent": True, "symbol": "none", "data": [{"xAxis": 0.5}],
                          "label": {"formatter": "even"}, "lineStyle": {"type": "dashed"}}},
        ],
    }
    note = " | ".join(
        f"{r['label']}: {r['paired_mean']:.3f} ({r['paired95'][0]:.2f}–{r['paired95'][1]:.2f}, n={r['battles']})"
        for r in rows[:6]
    )
    return write_chart(
        opt, out_path,
        "Rust Regulation M-B policy trade-offs, head to head",
        "Mirrored paired matchups on 100 top-50 meta team pairs (200 battles per row, sides swapped on identical slot shuffles). All search policies use the Rust Reg M-B engine; BC = behaviour cloning.",
        note, height=max(460, 34 * len(rows) + 90),
    )


def main():
    noise_dir, h2h_dir, out_dir = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    out_dir.mkdir(parents=True, exist_ok=True)
    n_rows = load_noise(noise_dir)
    if n_rows:
        p1 = build_noise_chart(n_rows, out_dir / "noise.html")
        (out_dir / "noise_summary.json").write_text(json.dumps(n_rows, indent=2) + "\n")
        print(f"Wrote {p1} ({len(n_rows)} policies)")
    h_rows = load_h2h(h2h_dir)
    if h_rows:
        p2 = build_h2h_chart(h_rows, out_dir / "tradeoffs.html")
        (out_dir / "h2h_summary.json").write_text(json.dumps(h_rows, indent=2) + "\n")
        print(f"Wrote {p2} ({len(h_rows)} matchups)")


if __name__ == "__main__":
    main()
