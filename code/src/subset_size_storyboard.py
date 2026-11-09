"""Storyboard figures for the subset-size experiment (docs/subset-size-labels.md, issue #118), from stored battles only.

    python src/subset_size_storyboard.py data      -> ~/vgc-data/results/subset_size_storyboard.json
    python src/subset_size_storyboard.py figures   -> ~/vgc-data/subset_size_labels/storyboard/*.png|svg

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


SURF, INK, INK2, GRID, GREY, PALE = "#fcfcfb", "#0b0b0b", "#52514e", "#e7e6e2", "#a9a8a2", "#cfcec9"
BLUE, ORANGE = "#2a78d6", "#eb6834"
COL = {"gen84_all_ep21": PALE, "gen84_all_noise_ep21": GREY, "gen65_subset_39": INK, "gen65_subset_10_39": BLUE, "gen65_subset_1_39": ORANGE}


def fig(w, h):
    import matplotlib.pyplot as plt
    f, ax = plt.subplots(figsize=(w, h), dpi=180, facecolor=SURF); ax.set_facecolor(SURF); ax.grid(True, color=GRID, lw=.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"): ax.spines[side].set_visible(False)
    return f, ax


def words(f, D, head, sub, intervals=True):
    import textwrap
    foot = (f"Got = win rate against the {D['opponents']} training opponents, behaviour-cloning policy on both sides, slots shuffled; "
            "192 teams per ask, one battle per opponent." + (" 95% intervals resample teams and opponents." if intervals else ""))
    pend = [ARMS[m][0] for m in D["pending"]]
    f.text(.012, .975, head, fontsize=14, fontweight="bold", color=INK, va="top")
    f.text(.012, .915, sub, fontsize=10, color=INK2, va="top")
    f.text(.012, .012, textwrap.fill(foot, 150) + (f"\nPending: {', '.join(pend)}." if pend else ""), fontsize=8, color=INK2, va="bottom")


def save(f, name):
    import matplotlib.pyplot as plt
    for ext in ("png", "svg"): f.savefig(OUT / f"{name}.{ext}", facecolor=SURF)
    plt.close(f)


def fig_error(D):
    """1. Headline: each rule's error minus the control's, paired over the same opponent draws."""
    A = D["arms"]; rows = [m for m in ARMS if m != CONTROL and (m in A or ARMS[m][1] == "arm")]
    f, ax = fig(9, 4.6); f.subplots_adjust(left=.34, right=.95, top=.8, bottom=.24)
    lim = max([abs(v) for m in A if "minus_control_ci" in A[m] for v in A[m]["minus_control_ci"]] + [.005]) + .003
    ax.axvline(0, color=INK, lw=1.4)
    ax.text(0, len(rows) - .45, f" control, error {A[CONTROL]['error']:.4f}", fontsize=9, color=INK)
    for i, m in enumerate(rows):
        y = len(rows) - 1 - i
        if m not in A:
            ax.text(0, y, "  pending", va="center", fontsize=9.5, color=GREY); continue
        r = A[m]
        ax.plot(r["minus_control_ci"], [y, y], color=COL[m], lw=3, solid_capstyle="round")
        ax.scatter([r["minus_control"]], [y], s=70, color=COL[m], zorder=3, edgecolor=SURF, lw=1.2)
        ax.text(r["minus_control"], y + .25, f"{r['minus_control']:+.4f}", ha="center", fontsize=9.5, color=INK2)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([ARMS[m][0] for m in reversed(rows)])
    ax.set_xlim(-lim, lim); ax.set_ylim(-.6, len(rows) - .2)
    ax.set_xlabel("Ask-versus-got error minus the control's (left of zero = closer to the ask)")
    words(f, D, "Does a noisier label bring got closer to the ask?",
          "Error on asks 1/24 to 17/24, each label rule minus the all-39 control, with 95% paired intervals.")
    save(f, "01_error_by_arm")


def fig_got(D):
    """2. Mechanism: got against ask."""
    A = D["arms"]; f, ax = fig(9, 5.6); f.subplots_adjust(left=.08, right=.97, top=.84, bottom=.17)
    ax.axvspan(17.5 / 24, 1, color=GRID, alpha=.6, lw=0); ax.text(20.5 / 24, .03, "reported apart", ha="center", fontsize=9, color=INK2)
    ax.plot([0, .78], [0, .78], color=INK2, ls=":", lw=1); ax.text(.66, .69, "got = ask", fontsize=9, color=INK2, rotation=33)
    for m in (m for m in ARMS if m in A):
        pa = A[m]["per_ask"]; ks = sorted(pa, key=int); x = [int(k) / 24 for k in ks]; arm = A[m]["role"] == "arm"
        if arm: ax.fill_between(x, [pa[k]["ci"][0] for k in ks], [pa[k]["ci"][1] for k in ks], color=COL[m], alpha=.15, lw=0)
        ax.plot(x, [pa[k]["got"] for k in ks], color=COL[m], lw=2.4 if arm else 1.6, ls="-" if arm else "--", label=A[m]["reader"])
    ax.set_xlim(0, 1); ax.set_ylim(0, .78); ax.set_xlabel("Ask"); ax.set_ylabel("Got")
    ax.legend(frameon=False, fontsize=9, loc="upper left", labelcolor=INK2)
    words(f, D, "Where along the ask does each label rule miss?",
          "Got for each of the 23 asks; solid lines are this experiment's arms, dashed the earlier models.")
    save(f, "02_got_against_ask")


def fig_variety(D):
    """3. Variety at each ask."""
    A = D["arms"]; f, ax = fig(9, 5.2); f.subplots_adjust(left=.09, right=.97, top=.84, bottom=.17)
    ax.axvspan(17.5 / 24, 1, color=GRID, alpha=.6, lw=0)
    for m in (m for m in ARMS if m in A):
        pa = A[m]["per_ask"]; ks = sorted(pa, key=int); arm = A[m]["role"] == "arm"
        ax.plot([int(k) / 24 for k in ks], [pa[k]["lineups"] for k in ks], color=COL[m], marker="o", ms=3.5,
                lw=2.4 if arm else 1.6, ls="-" if arm else "--", label=A[m]["reader"])
    ax.set_xlim(0, 1); ax.set_ylim(0, 200); ax.set_xlabel("Ask"); ax.set_ylabel("Distinct line-ups among the 192 teams")
    ax.legend(frameon=False, fontsize=9, loc="lower left", labelcolor=INK2)
    words(f, D, "Does the label rule change variety?", "Distinct six-species line-ups among the 192 teams of each ask.", intervals=False)
    save(f, "03_line_ups_by_ask")


def figures():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "Helvetica", "font.size": 10.5, "axes.edgecolor": GREY, "axes.linewidth": .8, "xtick.color": INK2,
                         "ytick.color": INK2, "axes.labelcolor": INK2, "svg.fonttype": "none"})
    D = json.load(open(DATA)); OUT.mkdir(parents=True, exist_ok=True)
    fig_error(D); fig_got(D); fig_variety(D)
    print("figures ->", OUT)


if __name__ == "__main__":
    {"data": data, "figures": figures}[sys.argv[1]]()
