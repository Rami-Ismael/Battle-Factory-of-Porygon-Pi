"""Storyboard figures for issue #93: what the remasking failure section can claim. Reads saved results only, no new battles.

Frames, each with its one sentence, the expected result and the result that would surprise:

1. Teaser. Remasking missed the bar where it was tested; the paper's own setting, random decoding, has never been battled.
   Expected: every species-first cell says "no closer to the line"; every random-decoding cell says "never battled".
   Surprise: a species-first cell whose interval sits below zero.
2. Verdict. No sampler brings got closer to the ask than no remasking does.
   Expected: every interval spans zero. Surprise: an interval wholly left of zero.
3. Mechanism. A lift at every ask cannot close the gap: low asks already sit above the line, and the 16/24 shortfall barely shrinks.
   Expected: arrows point up at every ask. Surprise: a sampler that lifts 16/24 more than 4/24.
4. Never tested. Battling the paper's rules under random decoding needs 4 to 9 times today's try cap.
   Expected: remasking lowers the legal share. Surprise: a rule that raises it.

The trade frame is #100's teaser, reused so each rule keeps one colour across both issues.

    python remask84_storyboard.py [output_dir]
"""
import json
import sys
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

RES = Path.home() / "vgc-data/results"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "reports/remask84-storyboard-2026-11-06"
ERR = json.load(open(RES / "remask84_error_analysis.json"))["arms"]
NOV = json.load(open(RES / "remask84_slope_analysis.json"))["novelty"]
RAND = json.load(open(RES / "remask84_random_decoding.json"))["summary"]

# Colours and type from #100's storyboard (least_likely_trade/storyboard_figures.py), so a rule keeps its colour across issues.
SURF, INK, INK2, GRID, GREY, PALE = "#fcfcfb", "#0b0b0b", "#52514e", "#e7e6e2", "#a9a8a2", "#ecebe7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
plt.rcParams.update({"font.family": "Helvetica", "font.size": 10.5, "axes.edgecolor": GREY, "axes.linewidth": .8, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.labelcolor": INK2, "svg.fonttype": "none"})
WIN = "Win rate against the top-50 meta teams, behaviour-cloning policy on both sides, 3 seeds × 64 teams per ask."
NAMES = {
    "remask_ll_05": "Least-likely 5% remasked, 84 steps", "remask_ll_10": "Least-likely 10% remasked, 84 steps",
    "remask_ll_20": "Least-likely 20% remasked, 84 steps", "remask_conf_loop": "Confidence-weighted 0.3 remasked, 210 steps",
    "remask_even_loop": "Even-odds 0.5 remasked, 210 steps", "cold_sets_t03": "Temperature 0.3 split-variance",
    "cold_sets_t06": "Temperature 0.6 split-variance", "nucleus_sets_p05": "Nucleus 0.5 split-variance",
    "nucleus_sets_p07": "Nucleus 0.7 split-variance", "nucleus_sets_p09": "Nucleus 0.9 split-variance",
    "cold_all_t06": "Temperature 0.6 on all fields", "cold_all_t08": "Temperature 0.8 on all fields",
    "nucleus_all_p09": "Nucleus 0.9 on all fields", "spend66": "Stat Points raised to 66",
}


def colour(arm):
    if arm == "plain":
        return INK
    if arm in ("spend66", "remask_conf_loop", "remask_even_loop"):
        return GREY
    return ORANGE if arm.startswith("remask") else BLUE if arm.startswith("nucleus") else AQUA


def figure(w, h, **kw):
    f, ax = plt.subplots(figsize=(w, h), dpi=180, facecolor=SURF, **kw)
    ax.set_facecolor(SURF); ax.grid(True, color=GRID, lw=.8); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    return f, ax


def heading(f, head, sub, y=.975):
    f.text(.012, y, head, fontsize=14, fontweight="bold", color=INK, va="top")
    f.text(.012, y - .055, sub, fontsize=10, color=INK2, va="top")


def foot(f, text, width=150):
    f.text(.012, .012, "\n".join(textwrap.wrap(text, width)), fontsize=8, color=INK2, va="bottom", linespacing=1.4)


def save(f, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        f.savefig(OUT / f"{name}.{ext}", facecolor=SURF)
    plt.close(f)


def err(arm, key="without_12"):
    return ERR[arm][key]


def teaser():
    f = plt.figure(figsize=(12, 6.6), dpi=180, facecolor=SURF)
    ax = f.add_axes((0, .1, 1, .74)); ax.set_xlim(0, 12.3); ax.set_ylim(0, 6.2); ax.axis("off")
    cols = [(3.55, "Species-first decoding", "species fill before any remasking · tested"), (8.0, "Random decoding", "the paper's setting · never battled")]
    for x, head, sub in cols:
        ax.text(x + 2.1, 6.0, head, ha="center", fontsize=12, fontweight="bold", color=INK)
        ax.text(x + 2.1, 5.68, sub, ha="center", fontsize=9.5, color=INK2)
    legal = {"remask_conf_loop": RAND["rand_remask_conf"], "remask_even_loop": RAND["rand_remask_even"]}
    rows = [("remask_ll_10", "Least-likely 10%\nremasked"), ("remask_conf_loop", "Confidence-weighted\n0.3 remasked"), ("remask_even_loop", "Even-odds 0.5\nremasked")]
    for i, (arm, label) in enumerate(rows):
        y = 4.0 - i * 1.75
        ax.text(3.3, y + .72, label, ha="right", va="center", fontsize=11, color=INK, fontweight="bold")
        ax.add_patch(FancyBboxPatch((3.33, y + .2), .08, 1.05, boxstyle="round,pad=0.01", color=colour(arm)))
        e = err(arm); n = NOV[arm]
        reach = (f"reaches species: {n['line_ups'] - n['shared_with_no_remasking']} of {n['line_ups']} line-ups are new"
                 if arm == "remask_ll_10" else f"barely reaches species: {n['shared_with_no_remasking']} of {n['line_ups']} line-ups also in no remasking")
        tested = f"Error vs no remasking {e['difference']:+.4f} ({e['interval'][0]:+.4f} to {e['interval'][1]:+.4f})\nno closer to the line\n{reach}"
        ax.add_patch(FancyBboxPatch((3.55, y + .05), 4.2, 1.35, boxstyle="round,pad=0.04,rounding_size=0.12", fc=PALE, ec=GREY, lw=.8))
        ax.text(3.75, y + .72, tested, va="center", fontsize=9.5, color=INK, linespacing=1.45)
        untested = ("Not planned: least-likely is not the paper's rule" if arm == "remask_ll_10"
                    else f"Never battled\n{legal[arm]:.2%} of tries pass Showdown (#89)\nwaits on a try budget (#91, #92)")
        ax.add_patch(FancyBboxPatch((8.0, y + .05), 4.2 - .25, 1.35, boxstyle="round,pad=0.04,rounding_size=0.12", fc=SURF, ec=GREY, lw=1, ls=(0, (4, 3))))
        ax.text(8.2, y + .72, untested, va="center", fontsize=9.5, color=INK2, linespacing=1.45)
    heading(f, "Remasking missed the bar where it was tested; the paper's own setting is untested",
            "The paper's rules fill species first here, so they mostly resample set fields. Their real test is random decoding, which has not been battled.")
    foot(f, f"Error = mean over asks 4/24, 8/24, 16/24 of (mean got − ask)², minus no remasking's {err('plain')['error']:.4f}; 95% intervals. Near the line is under 0.001. "
            + WIN)
    save(f, "01_teaser")


def verdict():
    groups = [("Remasking rules", ["remask_ll_05", "remask_ll_10", "remask_ll_20", "remask_conf_loop", "remask_even_loop"]),
              ("No remasking, other samplers", ["cold_sets_t03", "cold_sets_t06", "nucleus_sets_p05", "nucleus_sets_p07", "nucleus_sets_p09",
                                                 "cold_all_t06", "cold_all_t08", "nucleus_all_p09", "spend66"])]
    f, ax = figure(10, 7.4)
    y, ticks, labels = 0, [], []
    for group, arms in groups:
        y -= .3; ax.text(-.0118, y, group, fontsize=10, fontweight="bold", color=INK, va="center"); y -= 1
        for arm in arms:
            e = err(arm); c = colour(arm); worse = e["interval"][0] > 0
            ax.plot(e["interval"], [y, y], color=c, lw=3 if worse else 2, solid_capstyle="round", zorder=2)
            ax.scatter([e["difference"]], [y], s=55, color=c, edgecolor=SURF, lw=1.3, zorder=3)
            if worse:
                ax.annotate("further from the line", (e["interval"][1], y), xytext=(6, 0), textcoords="offset points", va="center", fontsize=9, color=INK)
            ticks.append(y); labels.append(NAMES[arm]); y -= 1
    ax.axvline(0, color=INK, lw=1, zorder=1)
    ax.set_yticks(ticks, labels, fontsize=9.5, color=INK); ax.tick_params(axis="y", length=0)
    ax.set_xlim(-.012, .019); ax.set_ylim(y + .4, .4); ax.grid(axis="y", visible=False)
    ax.set_xlabel(f"Ask-versus-got error minus no remasking's {err('plain')['error']:.4f}   (left is closer to got = ask)")
    heading(f, "No sampler brings got closer to the ask than no remasking does",
            "Every 95% interval crosses zero except least-likely 20%, which moves got further away. All sit 23 to 32 times above the 0.001 near-the-line bar.")
    foot(f, "Asks 4/24, 8/24, 16/24; 12/24 left out because it selects a different team population. Intervals resample teams within each cell and opponents across arms. " + WIN[:-1] + ".")
    f.tight_layout(rect=(0, .07, 1, .87)); save(f, "02_verdict")


def mechanism():
    f, ax = figure(9.6, 6.6)
    asks = [4, 8, 12, 16]
    ax.plot([0, 1], [0, 1], color=INK2, lw=1, ls=(0, (5, 4)), zorder=1)
    ax.text(.70, .715, "got = ask", rotation=36, color=INK2, fontsize=9.5)
    ax.axvspan(11.3 / 24, 12.7 / 24, color=PALE, zorder=0)
    ax.text(12 / 24, .135, "12/24 left out\nof the error", ha="center", fontsize=8.5, color=INK2)
    base = {k: ERR["plain"]["win"][str(k)] for k in asks}
    for arm, dx in (("remask_ll_10", -.006), ("cold_sets_t03", .006)):
        for k in asks:
            p, q = (k / 24 + dx, base[k]), (k / 24 + dx, ERR[arm]["win"][str(k)])
            ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=11, color=colour(arm), lw=1.8, shrinkA=3, shrinkB=0, zorder=3))
            ax.scatter(*q, s=40, color=colour(arm), edgecolor=SURF, lw=1, zorder=4)
    ax.scatter([k / 24 for k in asks], [base[k] for k in asks], s=70, color=INK, edgecolor=SURF, lw=1.3, zorder=5)
    for k in (4, 16):
        ax.plot([k / 24, k / 24], [k / 24, base[k]], color=INK, lw=.8, ls=":", zorder=2)
    g4, g16 = base[4] - 4 / 24, base[16] - 16 / 24
    ax.annotate(f"already {g4:+.3f} above the ask;\nlifting it moves away from the line", (4 / 24, base[4]), xytext=(4 / 24 + .03, .56), fontsize=9.5, color=INK,
                arrowprops=dict(arrowstyle="-", color=INK2, lw=.7))
    ax.annotate(f"{g16:+.3f} below the ask;\nthe lift closes only {ERR['remask_ll_10']['win']['16'] - base[16]:.3f} of it", (16 / 24, base[16]), xytext=(.70, .30), fontsize=9.5, color=INK,
                arrowprops=dict(arrowstyle="-", color=INK2, lw=.7))
    hs = [plt.Line2D([], [], marker="o", ls="", ms=7, color=c, label=t) for c, t in
          ((INK, "No remasking"), (ORANGE, NAMES["remask_ll_10"]), (AQUA, NAMES["cold_sets_t03"]))]
    ax.legend(handles=hs, loc="upper left", frameon=False, fontsize=9.5, labelcolor=INK2)
    ax.set_xticks([k / 24 for k in asks], [f"{k}/24" for k in asks]); ax.set_xlim(.1, .74); ax.set_ylim(.1, .74)
    ax.set_xlabel("Ask: the win rate requested"); ax.set_ylabel("Got: mean win rate of the teams")
    heading(f, "A lift at every ask cannot bring got to the ask",
            "Low asks already sit above the line, so in squared error the lift costs at 4/24 and 8/24 about what it saves at 16/24.")
    foot(f, WIN)
    f.tight_layout(rect=(0, .03, 1, .87)); save(f, "03_mechanism")


def never_tested():
    f, ax = figure(9.6, 4.6)
    cap, target = 1280, 64
    rows = [("rand_plain", "No remasking, 84 steps", INK), ("rand_remask_conf", NAMES["remask_conf_loop"], GREY), ("rand_remask_even", NAMES["remask_even_loop"], GREY)]
    for i, (key, label, c) in enumerate(rows):
        need = target / RAND[key]
        ax.barh(i, need, height=.5, color=c, zorder=2)
        ax.text(need + 150, i, f"{need:,.0f} tries · {RAND[key]:.2%} legal · {need / cap:.1f}× the cap", va="center", fontsize=9.5, color=INK)
    ax.axvline(cap, color=INK, lw=1.2, zorder=3)
    ax.text(cap + 100, 2.55, f"today's cap: {cap:,} tries per cell", fontsize=9.5, color=INK)
    ax.set_yticks(range(len(rows)), [r[1] for r in rows], fontsize=9.5, color=INK); ax.tick_params(axis="y", length=0); ax.invert_yaxis()
    ax.set_xlim(0, 19000); ax.set_ylim(2.8, -.6); ax.grid(axis="y", visible=False)
    ax.set_xlabel(f"Tries per cell to reach {target} legal teams under random decoding")
    heading(f, "Battling random decoding needs 4 to 9 times today's try cap",
            "Each masked field unmasks independently, and both paper rules lower the share of tries that pass Showdown.")
    foot(f, "Legal shares from #89: 384 tries per cell, 4 asks × 3 seeds per arm, sampling only. Every recorded rejection is a move the species can't learn. Budget: #91.")
    f.tight_layout(rect=(0, .05, 1, .80)); save(f, "04_never_tested")


if __name__ == "__main__":
    teaser(); verdict(); mechanism(); never_tested()
    print("STORYBOARD_DONE", OUT)
