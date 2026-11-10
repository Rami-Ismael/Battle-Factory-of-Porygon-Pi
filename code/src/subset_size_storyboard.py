"""Storyboard figures for the subset-size experiment (docs/subset-size-labels.md, issue #118), from stored battles only.

    python src/subset_size_storyboard.py data      -> ~/vgc-data/results/subset_size_storyboard.json
    python src/subset_size_storyboard.py figures   -> ~/vgc-data/subset_size_labels/storyboard/*.html (ECharts)

Every arm whose ask grid exists is drawn; the rest are listed as pending. Got is against the 39 training opponents. Intervals
resample whole teams within each ask and the 39 opponents, shared across arms in each replicate (run-experiment step 3).
"""
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, "/tmp/vgc-pilot/src")
sys.path.insert(0, str(HERE))
import train65 as S
from echarts_chart import write_chart

RES = Path.home() / "vgc-data" / "results"
OUT = Path.home() / "vgc-data" / "subset_size_labels" / "storyboard"
DATA = RES / "subset_size_storyboard.json"
ARMS = {                                                  # grid model -> (reader name, role)
    "gen84_all_ep21": ("No label noise, 21 epochs", "reference"),
    "gen84_all_noise_ep21": ("±1/48 label noise, 21 epochs (frontier)", "reference"),
    "gen65_subset_39": ("All 39 opponents per label (control)", "arm"),
    "gen65_subset_10_39": ("10 to 39 opponents per label", "arm"),
    "gen65_subset_1_39": ("1 to 39 opponents per label", "arm"),
}
CONTROL, B, JUDGED = "gen65_subset_39", 2000, range(1, 18)


def per_opponent(con, policy, cols, t):
    """[wins, battles] of team t against each column opponent, from stored shuffled-slot battles."""
    idx = {c: i for i, c in enumerate(cols)}; w = np.zeros(len(cols)); n = np.zeros(len(cols))
    ph = ",".join("?" * len(cols))
    for opp, wins, b in con.execute(f"SELECT hi, lo_wins, battles FROM cell WHERE policy_id = ? AND lo = ? AND hi IN ({ph}) UNION ALL "
                                    f"SELECT lo, hi_wins, battles FROM cell WHERE policy_id = ? AND hi = ? AND lo IN ({ph})",
                                    [policy, t, *cols, policy, t, *cols]):
        w[idx[opp]] += wins; n[idx[opp]] += b
    return w, n


def line_up(paste):
    from temperature_experiment import paste_species
    return tuple(sorted(paste_species(paste)))


def load_arm(con, policy, cols, model):
    grid = json.load(open(RES / f"stat_points_84_{model}.json"))
    asks = {}
    for c in grid["cells"].values():
        k = round(c["ask"] * 24); a = asks.setdefault(k, dict(W=[], N=[], lineups=set()))
        for t in S.team_ids(con, c):
            w, n = per_opponent(con, policy, cols, t)
            if n.sum(): a["W"].append(w); a["N"].append(n)
        a["lineups"].update(line_up(p) for p in c.get("pastes", []))
    return {k: dict(W=np.array(v["W"]), N=np.array(v["N"]), lineups=len(v["lineups"])) for k, v in sorted(asks.items())}


def got(a, opp=None, teams=None):
    W, N = a["W"], a["N"]
    if teams is not None: W, N = W[teams], N[teams]
    if opp is not None: W, N = W[:, opp], N[:, opp]
    return float(np.mean(W.sum(1) / np.maximum(N.sum(1), 1)))


def error(g): return float(np.mean([(g[k] - k / 24) ** 2 for k in JUDGED if k in g]))


def data():
    con = sqlite3.connect(f"file:{S.D.DB}?mode=ro", uri=True)
    sides, policy = S.sides_and_policy(con); cols = sides["training"]
    arms = {m: load_arm(con, policy, cols, m) for m in ARMS if (RES / f"stat_points_84_{m}.json").exists()}
    rng = np.random.default_rng(6501); boot = {m: dict(err=[], got={k: [] for k in a}) for m, a in arms.items()}
    for _ in range(B):
        opp = rng.integers(0, len(cols), len(cols))
        for m, a in arms.items():
            g = {k: got(v, opp, rng.integers(0, len(v["W"]), len(v["W"]))) for k, v in a.items()}
            boot[m]["err"].append(error(g))
            for k in g: boot[m]["got"][k].append(g[k])
    ci = lambda x: [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]
    out = dict(B=B, opponents=len(cols), arms={})
    for m, a in arms.items():
        g = {k: got(v) for k, v in a.items()}
        r = dict(reader=ARMS[m][0], role=ARMS[m][1], error=error(g), error_ci=ci(boot[m]["err"]),
                 per_ask={k: dict(got=g[k], ci=ci(boot[m]["got"][k]), teams=len(a[k]["W"]), lineups=a[k]["lineups"]) for k in g})
        if CONTROL in arms and m != CONTROL:
            d = np.array(boot[m]["err"]) - np.array(boot[CONTROL]["err"])
            r.update(minus_control=r["error"] - error({k: got(v) for k, v in arms[CONTROL].items()}), minus_control_ci=ci(d))
        out["arms"][m] = r
    out["pending"] = [m for m in ARMS if m not in arms]
    json.dump(out, open(DATA, "w"), indent=1)
    for m, r in out["arms"].items():
        print(f"{r['reader']:45s} error {r['error']:.4f} [{r['error_ci'][0]:.4f}, {r['error_ci'][1]:.4f}]"
              + (f"  minus control {r['minus_control']:+.4f} [{r['minus_control_ci'][0]:+.4f}, {r['minus_control_ci'][1]:+.4f}]" if "minus_control" in r else ""))



INK, INK2, GREY, PALE, BAND = "#0b0b0b", "#52514e", "#a9a8a2", "#cfcec9", "rgba(127,127,127,0.18)"
BLUE, ORANGE = "#2a78d6", "#eb6834"
COL = {"gen84_all_ep21": PALE, "gen84_all_noise_ep21": GREY, "gen65_subset_39": INK, "gen65_subset_10_39": BLUE, "gen65_subset_1_39": ORANGE}


def axis(name, **kw):
    return {"type": "value", "name": name, "nameLocation": "middle", "nameGap": 30, **kw}


def reported_apart(label=""):
    """Shade asks 18/24 and above, which the error leaves out and the report shows apart."""
    return {"type": "line", "data": [], "silent": True, "markArea": {
        "itemStyle": {"color": BAND}, "label": {"color": INK2, "position": "insideBottom"},
        "data": [[{"xAxis": 17.5 / 24, **({"name": label} if label else {})}, {"xAxis": 1}]]}}


def curve(m, r, key):
    """One arm's line over the asks; this experiment's arms solid, the earlier models dashed."""
    pa = r["per_ask"]; arm = r["role"] == "arm"
    return {"type": "line", "name": r["reader"], "color": COL[m], "symbolSize": 5, "emphasis": {"focus": "series"},
            "lineStyle": {"width": 2.4 if arm else 1.6, "type": "solid" if arm else "dashed"},
            "data": [[int(k) / 24, pa[k][key]] for k in sorted(pa, key=int)]}


def band(m, r):
    """95% interval of got over the asks, drawn as one quadrilateral between each pair of neighbouring asks."""
    pa = r["per_ask"]
    quad = ("js:(params, api) => { const i = params.dataIndexInside; if (i === 0) return;"
            " const p = (d, j) => api.coord([api.value(0, j), api.value(d, j)]);"
            " return {type: 'polygon', shape: {points: [p(1, i - 1), p(1, i), p(2, i), p(2, i - 1)]},"
            " style: {fill: api.visual('color'), opacity: 0.15}}; }")
    return {"type": "custom", "name": f"{r['reader']}, 95% interval", "color": COL[m], "renderItem": quad, "silent": True, "tooltip": {"show": False}, "z": 1,
            "encode": {"x": 0, "y": [1, 2]}, "data": [[int(k) / 24, *pa[k]["ci"]] for k in sorted(pa, key=int)]}


def save(option, name, D, head, sub, height, intervals=True):
    pend = [ARMS[m][0] for m in D["pending"]]
    note = "\n".join([f"Got = win rate against the {D['opponents']} training opponents, behaviour-cloning policy on both sides, slots shuffled; "
                      "192 teams per ask, one battle per opponent."]
                     + (["95% intervals resample teams and opponents."] if intervals else [])
                     + ([f"Pending: {', '.join(pend)}."] if pend else []))
    print(write_chart(option, OUT / f"{name}.html", head, sub, note, height=height))


def fig_error(D):
    """1. Headline: each rule's error minus the control's, paired over the same opponent draws."""
    A = D["arms"]; rows = [m for m in ARMS if m != CONTROL and (m in A or ARMS[m][1] == "arm")]
    lim = max([abs(v) for m in A if "minus_control_ci" in A[m] for v in A[m]["minus_control_ci"]] + [.005]) + .003
    series = [{"type": "line", "data": [], "silent": True, "markLine": {
        "symbol": "none", "lineStyle": {"color": INK, "type": "solid", "width": 1.4},
        "label": {"formatter": f"control, error {A[CONTROL]['error']:.4f}", "position": "end", "color": INK}, "data": [{"xAxis": 0}]}}]
    for i, m in enumerate(rows):
        y = len(rows) - 1 - i
        if m not in A:
            series.append({"type": "scatter", "name": ARMS[m][0], "symbolSize": 0, "silent": True, "tooltip": {"show": False},
                           "data": [{"value": [0, y], "label": {"show": True, "formatter": "pending", "position": "right", "color": GREY}}]})
            continue
        r = A[m]
        series.append({"type": "line", "name": r["reader"], "color": COL[m], "symbol": "none", "silent": True, "tooltip": {"show": False},
                       "lineStyle": {"width": 3, "cap": "round"}, "data": [[v, y] for v in r["minus_control_ci"]]})
        series.append({"type": "scatter", "name": r["reader"], "color": COL[m], "symbolSize": 12, "emphasis": {"focus": "series"},
                       "itemStyle": {"borderColor": "#fff", "borderWidth": 1.2},
                       "data": [{"value": [r["minus_control"], y], "label": {"show": True, "formatter": f"{r['minus_control']:+.4f}",
                                                                           "position": "top", "color": INK2}}]})
    option = {"tooltip": {"trigger": "item", "valueFormatter": "js:v => typeof v === 'number' ? v.toFixed(4) : v"},
              "grid": {"left": 250, "right": 40, "top": 36, "bottom": 56},
              "xAxis": axis("Ask-versus-got error minus the control's (left of zero = closer to the ask)", min=-lim, max=lim),
              "yAxis": {"type": "category", "data": [ARMS[m][0] for m in reversed(rows)], "axisTick": {"show": False},
                        "axisLabel": {"width": 230, "overflow": "break"}},
              "series": series}
    save(option, "01_error_by_arm", D, "Does a noisier label bring got closer to the ask?",
         "Error on asks 1/24 to 17/24, each label rule minus the all-39 control, with 95% paired intervals.", 380)


def fig_got(D):
    """2. Mechanism: got against ask."""
    A = D["arms"]; shown = [m for m in ARMS if m in A]
    series = [reported_apart("reported apart"),
              {"type": "line", "name": "got = ask", "color": INK2, "symbol": "none", "silent": True, "tooltip": {"show": False},
               "lineStyle": {"type": "dotted", "width": 1}, "endLabel": {"show": True, "formatter": "got = ask", "color": INK2},
               "data": [[0, 0], [.78, .78]]}]
    series += [band(m, A[m]) for m in shown if A[m]["role"] == "arm"]
    series += [curve(m, A[m], "got") for m in shown]
    option = {"tooltip": {"trigger": "axis", "valueFormatter": "js:v => typeof v === 'number' ? v.toFixed(3) : v"},
              "legend": {"data": [A[m]["reader"] for m in shown], "top": 0},
              "grid": {"left": 60, "right": 70, "top": 48, "bottom": 52},
              "xAxis": axis("Ask", min=0, max=1), "yAxis": axis("Got", min=0, max=.78, nameGap=40), "series": series}
    save(option, "02_got_against_ask", D, "Where along the ask does each label rule miss?",
         "Got for each of the 23 asks; solid lines are this experiment's arms, dashed the earlier models.", 520)


def fig_variety(D):
    """3. Variety at each ask."""
    A = D["arms"]; shown = [m for m in ARMS if m in A]
    option = {"tooltip": {"trigger": "axis"}, "legend": {"data": [A[m]["reader"] for m in shown], "top": 0},
              "grid": {"left": 60, "right": 24, "top": 48, "bottom": 52},
              "xAxis": axis("Ask", min=0, max=1), "yAxis": axis("Distinct line-ups among the 192 teams", min=0, max=200, nameGap=40),
              "series": [reported_apart()] + [curve(m, A[m], "lineups") for m in shown]}
    save(option, "03_line_ups_by_ask", D, "Does the label rule change variety?",
         "Distinct six-species line-ups among the 192 teams of each ask.", 480, intervals=False)


def figures():
    D = json.load(open(DATA)); OUT.mkdir(parents=True, exist_ok=True)
    fig_error(D); fig_got(D); fig_variety(D)


if __name__ == "__main__":
    {"data": data, "figures": figures}[sys.argv[1]]()
