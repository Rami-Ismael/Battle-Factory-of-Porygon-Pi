"""Asked vs got: one-shot decoding against inside the cross-entropy loop (2026-10-16).

One-shot, earlier model: results/sampler_qd_run.json (the older-vocabulary clean members F1/G2 noheldout, no loop,
64 teams x 49 battles per ask), arm 'settemp05' (species at 1, other fields at 0.5 = the loop's sampling).
One-shot, v3: results/ask_sweep_dbbo.json - the original v3 members and the same members after cem_dbbo's
generation-3 fine-tune (all labelled teams, shrunk labels), same decoder, no ridge guidance.
Inside the loop (right panel): one line per cem_dbbo-family loop, joining its low-ask teams (y - 0.2) and high-ask
teams (y), pooled over generations 1-3; x = the mean ask of each group. Win rate = wins / battles vs the top-50 meta teams,
behaviour-cloned policy on both sides.

    python plot_ask_vs_got_loop.py        # writes results/ask_vs_got_loop.html (ECharts page) and .json
"""
import json, sys
import numpy as np

sys.path.insert(0, "/tmp/vgc-pilot/src")
import lossdown as L
from echarts_chart import write_chart

LOOP_ARMS = ["cem_fixed_elite", "cem_fixed06_elite", "cem_fixed07_elite", "cem_fixed08_elite", "cem_moving_elite",
             "cem_fixed_elite_notilt", "cem_fixed_elite_random", "cem_precise_notilt", "cem_dbbo", "cem_scratch", "cem_scratch_fixed"]
# Two families. Cross-entropy: retrain on the elites only (cem_v3's rule). Diffusion-BBO: retrain on every labelled team.
LOOP_NAMES = {"cem_fixed_elite": "Cross-entropy, ask 0.5", "cem_fixed06_elite": "Cross-entropy, ask 0.6",
              "cem_fixed07_elite": "Cross-entropy, ask 0.7", "cem_fixed08_elite": "Cross-entropy, ask 0.8",
              "cem_moving_elite": "Cross-entropy, moving ask", "cem_fixed_elite_notilt": "Cross-entropy, ask 0.5, ridge push off",
              "cem_fixed_elite_random": "Cross-entropy, ask 0.5, ridge off, random pick",
              "cem_precise_notilt": "Cross-entropy from precise labels, ridge push off",
              "cem_dbbo": "Diffusion-BBO, fine-tuned v3", "cem_scratch": "Diffusion-BBO, trained from scratch",
              "cem_scratch_fixed": "Trained from scratch, fixed ask 0.5"}
LOOP_COLORS = {"cem_fixed_elite": "#9fe1cb", "cem_fixed06_elite": "#5dcaa5", "cem_fixed07_elite": "#1d9e75",
               "cem_fixed08_elite": "#085041", "cem_moving_elite": "#52514e", "cem_fixed_elite_notilt": "#eda100",
               "cem_fixed_elite_random": "#b07a00", "cem_precise_notilt": "#2a78d6",
               "cem_dbbo": "#e87ba4", "cem_scratch": "#e34948", "cem_scratch_fixed": "#6250d6"}   # cross-entropy: teal / neutral / amber; Diffusion-BBO: magenta, red
TRAIN_MAX = 0.792
INK, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"


def mse(y):
    y = np.asarray([v for v in y if v is not None], float)
    return float(y.mean()), float(y.std(ddof=1) / np.sqrt(len(y))), len(y)


def _with_bars(name, col, asks, means, ses, g):
    """A line with markers, plus its ±1.96 standard-error bars as one broken line under the same legend name."""
    bars = []
    for a, m, s in zip(asks, means, ses):
        bars += [[a, m - 1.96 * s], [a, m + 1.96 * s], "-"]
    on = {"xAxisIndex": g, "yAxisIndex": g}
    return [{"type": "line", "name": name, "color": col, "lineStyle": {"width": 2}, "symbolSize": 7, "emphasis": {"focus": "series"},
             "data": [[a, m] for a, m in zip(asks, means)], **on},
            {"type": "line", "name": name, "color": col, "symbol": "none", "silent": True, "connectNulls": False, "tooltip": {"show": False},
             "lineStyle": {"width": 1.2}, "data": bars, **on}]


def _option(one, sweep, pooled):
    series = []
    for g in (0, 1):
        series.append({"type": "line", "name": "got = asked", "silent": True, "symbol": "none", "tooltip": {"show": False},
                       "lineStyle": {"type": "dashed", "width": 1, "color": MUTED}, "data": [[0, 0], [1, 1]],
                       "xAxisIndex": g, "yAxisIndex": g})
    series.append({"type": "scatter", "silent": True, "symbolSize": 0, "tooltip": {"show": False}, "data": [[0.66, 0.70]],
                   "label": {"show": True, "formatter": "got = asked", "color": MUTED, "rotate": 38}})
    series[0]["markLine"] = {"symbol": "none", "silent": True, "lineStyle": {"color": GRID, "width": 2, "type": "solid"},
                             "label": {"formatter": "highest training\nlabel 0.792", "position": "insideEndBottom", "color": MUTED},
                             "data": [{"xAxis": TRAIN_MAX}]}
    lines = [(one["settemp05"], C2, "Earlier model (2026-10-14)")]
    if "v3_original" in sweep: lines.append((sweep["v3_original"], C1, "v3 model as trained"))
    if "dbbo_g3" in sweep: lines.append((sweep["dbbo_g3"], C4, "v3 after all-data training, shrunk labels"))
    if "v3_precise" in sweep: lines.append((sweep["v3_precise"], "#008300", "v3 retrained on 1,105 precise labels"))
    for pts, col, name in lines:
        a, m, s_, _ = zip(*pts)
        series += _with_bars(name, col, a, m, s_, 0)
    for p in pooled:
        series += _with_bars(p["name"], LOOP_COLORS[p["arm"]], [p["ask_lo"], p["ask_hi"]], [p["got_lo"][0], p["got_hi"][0]],
                             [p["got_lo"][1], p["got_hi"][1]], 1)
    grids = [{"left": "6%", "width": "42%", "top": 80, "bottom": 230}, {"left": "54%", "width": "43%", "top": 80, "bottom": 230}]
    heads = ["The model alone, no search loop: 64 teams per ask",
             "Inside the search loop: cross-entropy vs Diffusion-BBO\n(low-ask vs high-ask teams, generations pooled)"]
    axis = {"type": "value", "name": "Asked win rate", "nameLocation": "middle", "nameGap": 26, "splitLine": {"show": False}}
    return {"title": [{"text": h, "left": gr["left"], "top": 8, "textStyle": {"fontSize": 13}} for h, gr in zip(heads, grids)],
            "grid": grids, "tooltip": {"trigger": "item"},
            "legend": [{"data": [n for _, _, n in lines], "left": "6%", "bottom": 0, "orient": "vertical"},
                       {"data": [p["name"] for p in pooled], "left": "54%", "bottom": 0, "width": "44%"}],
            "xAxis": [{**axis, "gridIndex": 0, "min": 0, "max": 1}, {**axis, "gridIndex": 1, "min": 0.1, "max": 0.9}],
            "yAxis": [{"type": "value", "gridIndex": g, "min": 0, "max": 0.85, "name": "Got: mean win rate of the teams (±95%)" if g == 0 else "",
                       "nameLocation": "middle", "nameGap": 40, "axisLabel": {"showMaxLabel": False}} for g in (0, 1)],
            "series": series}


def main():  # noqa: C901
    q = json.load(open(L.RES / "sampler_qd_run.json"))["cells"]
    one = {}
    for arm in ("settemp05",):
        pts = sorted((c["asked"], *mse(c["y"])) for c in q.values() if c["arm"] == arm and "y" in c)
        one[arm] = pts
    loop = []
    for tag in LOOP_ARMS:
        f = L.RES / f"{tag}.json"
        if not f.exists(): continue
        o = json.load(open(f))
        for g, c in o["gens"].items():
            if g == "0" or "got_vs_ask" not in c: continue
            hi = [v for a, v in zip(c["asks_battled"], c["y"]) if v is not None and np.isclose(a, c["ask"])]
            lo = [v for a, v in zip(c["asks_battled"], c["y"]) if v is not None and np.isclose(a, c["ask_contrast"])]
            loop.append(dict(arm=tag, gen=int(g), ask=c["ask"], got=mse(hi), ask_lo=c["ask_contrast"], got_lo=mse(lo)))
    sweep = {}
    if (L.RES / "ask_sweep_dbbo.json").exists():
        for c in json.load(open(L.RES / "ask_sweep_dbbo.json"))["cells"].values():
            if c.get("y"): sweep.setdefault(c["model"], []).append((c["asked"], *mse(c["y"])))
        if (L.RES / "ask_sweep_precise.json").exists():
            for c in json.load(open(L.RES / "ask_sweep_precise.json"))["cells"].values():
                if c.get("y"): sweep.setdefault(c["model"], []).append((c["asked"], *mse(c["y"])))
        sweep = {k: sorted(v) for k, v in sweep.items()}
    json.dump(dict(one_shot=one, sweep=sweep, loop=loop), open(L.RES / "ask_vs_got_loop.json", "w"), indent=1)

    pooled = []                                                      # one line per loop: low-ask vs high-ask teams, all generations
    for tag in LOOP_ARMS:
        f = L.RES / f"{tag}.json"
        if not f.exists(): continue
        o = json.load(open(f)); lo_a, lo_y, hi_a, hi_y = [], [], [], []
        for g, c in o["gens"].items():
            if (g == "0" and not tag.startswith(("cem_scratch", "cem_precise"))) or "got_vs_ask" not in c: continue
            for a, v in zip(c["asks_battled"], c["y"]):
                if v is None: continue
                if np.isclose(a, c["ask"]): hi_a.append(a); hi_y.append(v)
                else: lo_a.append(a); lo_y.append(v)
        if hi_y and lo_y:
            pooled.append(dict(arm=tag, name=LOOP_NAMES[tag], ask_lo=float(np.mean(lo_a)), got_lo=mse(lo_y),
                               ask_hi=float(np.mean(hi_a)), got_hi=mse(hi_y)))
    json.dump(dict(one_shot=one, sweep=sweep, loop=loop, pooled=pooled), open(L.RES / "ask_vs_got_loop.json", "w"), indent=1)

    write_chart(_option(one, sweep, pooled), L.RES / "ask_vs_got_loop.html",
                "Asked vs got: one-shot decoding against inside the cross-entropy loop",
                note="Win rate = wins / battles vs the top-50 meta teams, behaviour-cloned policy both sides; 49 battles per team. "
                     "Both panels use the loop's sampling. Inside each loop generation, 96 teams ask y and 32 ask y − 0.2.\n"
                     "A working ask would make a line climb parallel to the dashed line; a flat line means the ask changes nothing.",
                height=680, dark={INK: "#f2f1ee", MUTED: "#c3c2b7", "#085041": "#5dcaa5"})
    print("one-shot:", {k: [(a, round(m, 3)) for a, m, _, _ in v] for k, v in one.items()})
    for p in pooled:
        print(f"{p['name']:40s} asked {p['ask_lo']:.2f} -> {p['got_lo'][0]:.3f} ± {p['got_lo'][1]:.3f} · "
              f"asked {p['ask_hi']:.2f} -> {p['got_hi'][0]:.3f} ± {p['got_hi'][1]:.3f}")


if __name__ == "__main__":
    main()
