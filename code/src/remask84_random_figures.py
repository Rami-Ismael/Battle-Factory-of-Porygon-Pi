"""Figures for remasking under random decoding with the lookahead (GitHub issue #92), from remask84_random_error.json.

    python remask84_random_figures.py [out_dir]

1. Got against ask for the three arms and the species-first reference, with got = ask.
2. Each arm's ask-versus-got error on asks 1/24 to 17/24 minus no remasking's, with 95% intervals.
Colours and type follow the earlier remasking storyboards; the two paper rules take blue and aqua here because they are the subject.
"""
import json
import sys
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = Path.home() / "vgc-data/results"
SURF, INK, INK2, GRID, GREY = "#fcfcfb", "#0b0b0b", "#52514e", "#e7e6e2", "#a9a8a2"
BLUE, AQUA = "#2a78d6", "#1baf7a"
plt.rcParams.update({"font.family": "Helvetica", "font.size": 10.5, "axes.edgecolor": GREY, "axes.linewidth": .8, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.labelcolor": INK2})
STYLE = {
    "rand_la_plain": dict(color=INK, label="No remasking, 84 passes"),
    "rand_la_remask_conf": dict(color=BLUE, label="Confidence-weighted 0.3 fields remasked, 210 passes"),
    "rand_la_remask_even": dict(color=AQUA, label="Even-odds 0.5 fields remasked, 210 passes"),
    "species_first_plain": dict(color=GREY, label="Species-first decoding, no remasking (reference)", ls="--"),
}
WIN = "Win rate against the 49 distinct top-50 meta teams, behaviour-cloning policy on both sides, slots shuffled. 3 seeds x 64 teams per ask, one battle per opponent."


def frame(w, h, head, sub, foot):
    f, ax = plt.subplots(figsize=(w, h), dpi=180, facecolor=SURF)
    ax.set_facecolor(SURF); ax.grid(True, color=GRID, lw=.8); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    f.text(.012, .975, head, fontsize=14, fontweight="bold", color=INK, va="top")
    f.text(.012, .915, sub, fontsize=10, color=INK2, va="top")
    f.text(.012, .015, foot, fontsize=8, color=INK2, va="bottom")
    return f, ax


def got_against_ask(r, out):
    f, ax = frame(10, 6.2, "Remasking lifts got by about 0.01 at every ask; it doesn't make got follow the ask",
                  "Random decoding with the lookahead. Above 11/24, got stays near 0.25 in every arm, while species-first decoding reaches about 0.40.", WIN)
    f.subplots_adjust(left=.08, right=.97, top=.84, bottom=.15)
    ks = list(range(1, 24))
    ax.axvspan(17.5, 23.5, color=GRID, alpha=.6, lw=0)
    ax.text(20.5, .03, "18/24 to 23/24:\nreported apart", ha="center", fontsize=8.5, color=INK2)
    ax.plot([0, 1], [0, 1], color=GREY, lw=1, ls=":", transform=ax.transAxes)
    ax.text(10.6, 10.6 / 24 + .02, "got = ask", rotation=36, fontsize=8.5, color=INK2)
    for arm, st in STYLE.items():
        ys = [r["arms"][arm]["win"][str(k)] for k in ks]
        ax.plot(ks, ys, color=st["color"], lw=2, ls=st.get("ls", "-"), marker="o", ms=3.5, label=st["label"])
    ax.set_xlim(.5, 23.5); ax.set_ylim(0, 1)
    ax.set_xticks([1, 4, 8, 12, 17, 20, 23], ["1/24", "4/24", "8/24", "12/24", "17/24", "20/24", "23/24"])
    ax.set_xlabel("Ask (the win rate the model is conditioned on)"); ax.set_ylabel("Got (mean win rate at that ask)")
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    save(f, out / "remask84_random_01_got_against_ask")


def error_difference(r, out):
    f, ax = frame(10, 4.6, "Both paper rules sit closer to the line than no remasking, by 0.004",
                  "Ask-versus-got error on asks 1/24 to 17/24 minus no remasking's 0.0534, with 95% intervals. Near the line means an error under 0.001.", WIN)
    f.subplots_adjust(left=.27, right=.97, top=.76, bottom=.2)
    rows = ["rand_la_remask_conf", "rand_la_remask_even", "species_first_plain"]
    for i, arm in enumerate(rows):
        d = r["arms"][arm]["scored"]["minus_rand_la_plain"]; st = STYLE[arm]
        ax.plot(d["interval"], [i, i], color=st["color"], lw=2.5, solid_capstyle="round")
        ax.plot([d["difference"]], [i], "o", color=st["color"], ms=8, mec=SURF, mew=2)
        e = r["arms"][arm]["scored"]["error"]
        x, ha = (-.0008, "right") if d["difference"] > -.01 else (d["difference"], "center")
        ax.text(x, i - .22, f"{d['difference']:+.4f} (error {e:.4f}), closer on {d['seeds_closer']} of 3 seeds", ha=ha, va="bottom", fontsize=9, color=INK)
    ax.axvline(0, color=INK2, lw=1)
    ax.set_yticks(range(len(rows)), [textwrap.fill(STYLE[a]["label"], 30) for a in rows]); ax.set_ylim(len(rows) - .5, -.7)
    ax.set_xlim(-.046, .008); ax.set_xlabel("Error minus no remasking's (lower is closer to got = ask)")
    save(f, out / "remask84_random_02_error_difference")


def save(f, stem):
    f.savefig(f"{stem}.png", facecolor=SURF); plt.close(f); print(f"{stem}.png")


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else RES
    out.mkdir(parents=True, exist_ok=True)
    r = json.load(open(RES / "remask84_random_error.json"))
    got_against_ask(r, out); error_difference(r, out)
