"""Storyboard figures for issue #100 and #93, one idea each, from least_likely_trade_storyboard.json. No new battles."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

D = json.load(open(Path.home() / "vgc-data/results/least_likely_trade_storyboard.json"))
A = D["arms"]
OUT = Path.home() / "vgc-data/least_likely_trade/storyboard"; OUT.mkdir(exist_ok=True)
SURF, INK, INK2, GRID, GREY, PALE = "#fcfcfb", "#0b0b0b", "#52514e", "#e7e6e2", "#a9a8a2", "#ecebe7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"      # validated all-pairs on the light surface; no remasking is ink
FAMILY = {"plain": INK, "spend66": GREY, "remask_conf_loop": GREY, "remask_even_loop": GREY}
for a in A:
    FAMILY.setdefault(a, ORANGE if a.startswith("remask") else BLUE if a.startswith("nucleus") else AQUA)
WIN = "Win rate against the top-50 meta teams, behaviour-cloning policy on both sides."
plt.rcParams.update({"font.family": "Helvetica", "font.size": 10.5, "axes.edgecolor": GREY, "axes.linewidth": .8, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.labelcolor": INK2, "axes.titlecolor": INK, "svg.fonttype": "none"})


def frame(ax):
    ax.set_facecolor(SURF); ax.grid(True, color=GRID, lw=.8); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)


def fig(w, h, rows=1, cols=1, **kw):
    f, axs = plt.subplots(rows, cols, figsize=(w, h), dpi=180, facecolor=SURF, **kw)
    for ax in (axs.flat if hasattr(axs, "flat") else [axs]): frame(ax)
    return f, axs


def title(f, head, sub, y=.975):
    f.text(.012, y, head, fontsize=14, fontweight="bold", color=INK, va="top")
    f.text(.012, y - .055, sub, fontsize=10, color=INK2, va="top")


def foot(f, text): f.text(.012, .012, text, fontsize=8, color=INK2, va="bottom")


def save(f, name):
    for ext in ("png", "svg"): f.savefig(OUT / f"{name}.{ext}", facecolor=SURF)
    plt.close(f)


def family_legend(f, **where):
    hs = [plt.Line2D([], [], marker="o", ls="", ms=7, color=c, label=t) for c, t in ((ORANGE, "Least-likely remasking"), (AQUA, "Temperature"), (BLUE, "Nucleus"), (GREY, "Other: Stat Points, paper rules"))]
    f.legend(handles=hs, **(where or dict(loc="upper right", bbox_to_anchor=(.99, .985), ncol=2)), frameon=False, fontsize=9.5, labelcolor=INK2)


def arrow(ax, p, q, col, lw=2.2):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=14, color=col, lw=lw, shrinkA=6, shrinkB=6, zorder=4))


# 1. Teaser: two routes from the same start.
f, ax = fig(9.2, 6.4)
P = {a: A[a]["pooled"]["all"] for a in A}
x0, y0 = P["plain"]["line_ups"], P["plain"]["win"]
ax.scatter([x0], [y0], s=90, color=INK, zorder=5, edgecolor=SURF, lw=1.5)
ax.annotate("No remasking\n438 line-ups, 2,336 Pokémon sets", (x0, y0), xytext=(-12, -26), textcoords="offset points", ha="right", va="top", fontsize=10, color=INK)
for a, col, dx, dy, ha in (("remask_ll_10", ORANGE, -14, 6, "right"), ("cold_sets_t03", AQUA, -14, 14, "right")):
    p = P[a]; arrow(ax, (x0, y0), (p["line_ups"], p["win"]), col)
    ax.scatter([p["line_ups"]], [p["win"]], s=90, color=col, zorder=5, edgecolor=SURF, lw=1.5)
    kept = p["distinct_sets"] / P["plain"]["distinct_sets"]
    ax.annotate(f"{A[a]['name']}\n{p['gain']:+.3f} win rate · {p['line_ups']} line-ups · {kept:.0%} of the sets", (p["line_ups"], p["win"]), xytext=(dx, dy), textcoords="offset points", ha=ha, va="bottom", fontsize=10, color=INK)
ax.set_xlim(150, 470); ax.set_ylim(.34, .44)
ax.set_xlabel("Distinct line-ups among 768 teams (asks 4/24, 8/24, 12/24, 16/24)"); ax.set_ylabel("Mean win rate")
title(f, "Set-only sharpening buys most of least-likely's gain and keeps every line-up", "Both routes pay about half the Pokémon sets. Least-likely also pays half the line-ups.")
foot(f, WIN + " 768 teams per rule, 49 battles each. Exploratory.")
f.tight_layout(rect=(0, .03, 1, .87)); save(f, "01_teaser")

# 2. The same two moves at each ask, counted in line-ups and in Pokémon sets.
f, axs = fig(13, 7.2, 2, 4, sharey="col")
for j, k in enumerate(["4", "8", "16", "12"]):
    for i, (key, lab) in enumerate((("line_ups", "Distinct line-ups"), ("distinct_sets", "Distinct Pokémon sets"))):
        ax = axs[i, j]; q = {a: A[a]["per_ask"][k] for a in ("plain", "remask_ll_10", "cold_sets_t03")}
        if k == "12": ax.set_facecolor(PALE)
        p0 = (q["plain"][key], q["plain"]["win"])
        ax.scatter(*p0, s=60, color=INK, zorder=5, edgecolor=SURF, lw=1.2)
        for a, col in (("remask_ll_10", ORANGE), ("cold_sets_t03", AQUA)):
            p = (q[a][key], q[a]["win"]); arrow(ax, p0, p, col, 1.8)
            ax.scatter(*p, s=60, color=col, zorder=5, edgecolor=SURF, lw=1.2)
            ax.plot([p[0]] * 2, q[a]["win_ci"], color=col, lw=1.2, zorder=3)
        ax.set_xlim(0, max(v[key] for v in q.values()) * 1.12)
        if i == 0: ax.set_title(f"Ask {k}/24" + ("  (another team population)" if k == "12" else ""), loc="left", fontsize=11, fontweight="bold")
        if i == 1: ax.set_xlabel(lab + " among 192 teams")
        else: ax.set_xlabel(lab + " among 192 teams")
        if j == 0: ax.set_ylabel("Mean win rate")
axs[0, 0].scatter([], [], s=50, color=INK, label="No remasking")
axs[0, 0].scatter([], [], s=50, color=ORANGE, label="Least-likely 10% remasked")
axs[0, 0].scatter([], [], s=50, color=AQUA, label="Temperature 0.3 on sets only")
axs[0, 0].legend(loc="lower left", frameon=False, fontsize=9, labelcolor=INK2, handletextpad=.3)
title(f, "Least-likely pays in line-ups and sets; set-only sharpening pays only in sets", "Top row counts line-ups, bottom row counts Pokémon sets. The aqua arrow is vertical on top: no line-up lost at any ask.")
foot(f, WIN + " Each point is 192 teams from 3 sampling seeds; bars are 95% intervals over teams. Ask 12/24 sits apart because it selects a different team population.")
f.tight_layout(rect=(0, .03, 1, .88)); save(f, "02_moves_per_ask")

# 3. Nucleus against temperature on the set fields.
f, axs = fig(12.5, 6, 1, 2)
for ax, tag, lab in ((axs[0], "all", "all four asks, 768 teams"), (axs[1], "without_12", "without ask 12/24, 576 teams")):
    Q = {a: A[a]["pooled"][tag] for a in A}
    for path, col, txt in ((["plain", "cold_sets_t06", "cold_sets_t03"], AQUA, ["1", "0.6", "0.3"]), (["plain", "nucleus_sets_p09", "nucleus_sets_p07", "nucleus_sets_p05"], BLUE, ["", "0.9", "0.7", "0.5"])):
        xs, ys = [Q[a]["distinct_sets"] for a in path], [Q[a]["win"] for a in path]
        ax.plot(xs, ys, color=col, lw=2, zorder=3); ax.scatter(xs[1:], ys[1:], s=55, color=col, zorder=5, edgecolor=SURF, lw=1.2)
        for x, y, t in zip(xs[1:], ys[1:], txt[1:]):
            ax.annotate(t, (x, y), xytext=(0, -17) if col == AQUA else (9, -4), textcoords="offset points", ha="center" if col == AQUA else "left", va="center", fontsize=9.5, color=INK2)
    ax.scatter(Q["plain"]["distinct_sets"], Q["plain"]["win"], s=70, color=INK, zorder=6, edgecolor=SURF, lw=1.2)
    ax.annotate("No remasking", (Q["plain"]["distinct_sets"], Q["plain"]["win"]), xytext=(8, -2), textcoords="offset points", ha="left", va="top", fontsize=9.5, color=INK)
    ll = Q["remask_ll_10"]; ax.scatter(ll["distinct_sets"], ll["win"], s=60, marker="s", color=ORANGE, zorder=5, edgecolor=SURF, lw=1.2)
    ax.annotate("Least-likely 10%\n(also loses line-ups)", (ll["distinct_sets"], ll["win"]), xytext=(0, 10), textcoords="offset points", ha="center", va="bottom", fontsize=9.5, color=INK)
    ax.set_title(lab.capitalize(), loc="left", fontsize=11, fontweight="bold"); ax.set_xlabel("Distinct Pokémon sets (species, ability, item, nature, moves)")
    xs_all = [Q[a]["distinct_sets"] for a in ("plain", "remask_ll_10", "cold_sets_t03", "nucleus_sets_p05")]
    ax.set_xlim(max(xs_all) * 1.04, min(xs_all) * .86)
axs[0].set_ylabel("Mean win rate")
axs[0].plot([], [], color=AQUA, lw=2, label="Temperature on sets only: 1, 0.6, 0.3"); axs[0].plot([], [], color=BLUE, lw=2, label="Nucleus on sets only: 0.9, 0.7, 0.5")
axs[1].legend(*axs[0].get_legend_handles_labels(), loc="lower right", frameon=False, fontsize=10, labelcolor=INK2)
title(f, "Cutting unlikely set options costs the same sets per win as cooling them", "Nucleus 0.5 lands on temperature 0.3: -0.001 win rate (-0.010 to +0.009), 1,223 sets against 1,252. Fewer sets to the right.")
foot(f, WIN + " Species untouched in both, so every point keeps all line-ups. 95% intervals resample teams within each cell.")
f.tight_layout(rect=(0, .03, 1, .86)); save(f, "03_nucleus_vs_temperature")

# 4. The menu: every option's gain and what it keeps.
order = sorted([a for a in A if a != "plain"], key=lambda a: A[a]["pooled"]["all"]["gain"])
f, axs = fig(13, 7.4, 1, 3, sharey=True, gridspec_kw=dict(width_ratios=[1.5, 1, 1]))
base = A["plain"]["pooled"]["all"]
for i, a in enumerate(order):
    col = FAMILY[a]; p, q = A[a]["pooled"]["all"], A[a]["pooled"]["without_12"]
    axs[0].plot(p["gain_ci"], [i, i], color=col, lw=1.4); axs[0].scatter(p["gain"], i, s=55, color=col, zorder=5, edgecolor=SURF, lw=1)
    axs[0].scatter(q["gain"], i, s=40, facecolor=SURF, edgecolor=col, lw=1.4, zorder=4)
    for ax, key in ((axs[1], "line_ups"), (axs[2], "distinct_sets")):
        v = p[key] / base[key]
        ax.barh(i, v, height=.56, color=col, alpha=.9 if col not in (GREY,) else .7)
        ax.text(min(v, 1.0) + .015, i, f"{v:.0%}", va="center", fontsize=9, color=INK2)
axs[0].set_yticks(range(len(order))); axs[0].set_yticklabels([A[a]["name"] for a in order], color=INK)
axs[0].axvline(0, color=INK2, lw=.9); axs[0].axvline(.03, color=GREY, lw=.9, ls=(0, (3, 3)))
axs[0].text(.031, len(order) - .4, "smallest effect worth acting on", fontsize=8.5, color=INK2, va="bottom")
axs[0].set_xlabel("Win-rate gain over no remasking (filled: all four asks; hollow: without 12/24)")
axs[1].set_xlabel("Line-ups kept"); axs[2].set_xlabel("Pokémon sets kept")
for ax in axs[1:]: ax.set_xlim(0, 1.18); ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
title(f, "Set-only sharpening alone gains over 0.05 without losing a line-up", "Temperature 0.3 and nucleus 0.5 on sets only tie. Higher gains cost line-ups; every gain costs Pokémon sets.")
foot(f, WIN + " Kept shares are against no remasking's 438 line-ups and 2,336 sets over 768 teams. Lines through the dots are 95% intervals over teams.")
family_legend(f)
f.tight_layout(rect=(0, .03, 1, .87)); save(f, "04_menu")

# 5. Do strong line-ups hold on fresh games?
R = D["rebattle"]; rules = [("plain", INK, "No remasking"), ("remask_ll_10", ORANGE, "Least-likely 10% remasked"), ("cold_sets_t03", AQUA, "Temperature 0.3 on sets only")]
f, axs = fig(12, 4.6, 1, 2, sharey=True)
for ax, (h, c, lab) in zip(axs, (("held", "candidates", "All four asks"), ("held_without_12", "candidates_without_12", "Without ask 12/24"))):
    for i, (a, col, _) in enumerate(rules):
        held, n = R[a][h], R[a][c]
        ax.barh(i, n, height=.55, color=PALE); ax.barh(i, held, height=.55, color=col)
        ax.text(n + 1, i, f"{held} of {n} hold", va="center", fontsize=10, color=INK)
    ax.set_title(lab, loc="left", fontsize=11, fontweight="bold"); ax.set_xlabel("Line-ups at the 0.531 elite cut, best team re-battled 196 times")
    ax.set_xlim(0, 82)
axs[0].set_yticks(range(3)); axs[0].set_yticklabels([r[2] for r in rules], color=INK); axs[0].invert_yaxis()
title(f, "The strong line-ups of temperature 0.3 on sets only hold on fresh games as often as the others'", "27 of 68 hold, against 18 of 52 and 18 of 43. Without ask 12/24, almost none hold for any rule.", y=.96)
foot(f, WIN + " Fresh games: 4 per opponent, private database. 34 of temperature 0.3's 68 sit within one standard error (0.036) of the cut.")
f.tight_layout(rect=(0, .05, 1, .8)); save(f, "05_rebattle")

# 6. Ask-versus-got error: does any rule bring got closer to the ask?
order = sorted([a for a in A if a != "plain"], key=lambda a: A[a]["error"]["all"]["diff_vs_plain"])
f, ax = fig(10.5, 7)
for i, a in enumerate(order):
    e = A[a]["error"]["all"]; col = FAMILY[a]
    ax.plot(e["diff_vs_plain_ci"], [i, i], color=col, lw=1.4); ax.scatter(e["diff_vs_plain"], i, s=55, color=col, zorder=5, edgecolor=SURF, lw=1)
ax.axvline(0, color=INK2, lw=.9); ax.axvline(-.001, color=GREY, lw=.9, ls=(0, (3, 3)))
ax.text(-.00105, len(order) - .4, "smallest effect, 0.001 closer", fontsize=8.5, color=INK2, ha="right", va="bottom")
ax.set_yticks(range(len(order))); ax.set_yticklabels([A[a]["name"] for a in order], color=INK)
ax.set_xlabel("Ask-versus-got error minus no remasking's 0.0197 (left is closer to got = ask)")
title(f, "No remasking rule or control brings got closer to the ask", "Every interval crosses zero except least-likely 20%, which moves got further away. Every arm sits near 0.02; near the line is under 0.001.")
foot(f, WIN + " Error: mean squared gap between ask and mean got on asks 4/24, 8/24, 12/24, 16/24 only, not the full 1/24 to 17/24.")
family_legend(f, loc="lower right", bbox_to_anchor=(.97, .12), ncol=1)
f.tight_layout(rect=(0, .03, 1, .87)); save(f, "06_ask_versus_got_error")
print("wrote", sorted(p.name for p in OUT.iterdir()))
