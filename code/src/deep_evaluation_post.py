"""The issue comment for the deep re-battle of elite line-ups (docs/weakness-pressure-deep-evaluation.md part A, issue #345).

    LY_PER_CELL=16 python deep_evaluation_post.py > comment.md

Reads the `flips` block of every guidance's deep results file written so far and prints one comment: a picture of each
measured flip share and its interval beside the prediction and the ticket's 0.10 rule, then the matrix of counts.
"""
import json
import sys

import line_up_yield as L

GRIDS = (("sched_species_first", 2), ("sched_species_first_g4", 4), ("sched_species_first_g8", 8))
WIDTH, TOP = 50, 0.5                                       # the picture's flip-share axis runs 0 to 0.5


def col(x):
    return max(0, min(WIDTH, round(x / TOP * WIDTH)))


def strip(s):
    """One text row: [ ] the 95% interval, ● the measured share, p the prediction, | the 0.10 rule."""
    row = [" "] * (WIDTH + 1)
    row[col(L.FLIP_RULE)] = "|"
    lo, hi = s["flip_share_interval"]
    for i in range(col(lo), col(hi) + 1):
        row[i] = "="
    row[col(lo)], row[col(hi)] = "[", "]"
    row[col(s["predicted_from_4"]["mean"])] = "p"
    row[col(s["flip_share"])] = "●"
    return "".join(row)


def comment(results):
    """results: [(guidance, flips block)] in run order."""
    g, last = results[-1]
    verdict = {"16": "16 battles per meta team becomes the re-battle depth",
               "4": "the run stops here: 4 battles per meta team stays the depth",
               "undecided": "the interval straddles 0.10, so the rule does not decide"}[last["verdict"]]
    lo, hi = last["flip_share_interval"]
    lines = [f"**Deep evaluation, guidance {g}: {last['flips']} of {last['line_ups']} elite line-ups change verdict between 4 and "
             f"{last['depth']} battles per meta team (share {last['flip_share']:.3f}, 95% interval {lo:.3f} to {hi:.3f}); "
             f"by the rule fixed on this ticket, {verdict}.**", "",
             "A verdict is held / not held: the line-up's best team at or above the elite cut 0.531 in its fresh re-battle.", "",
             "```text", f"flip share  {'0'.ljust(col(0.1))}{'0.1'.ljust(col(0.2) - col(0.1))}{'0.2'.ljust(col(0.3) - col(0.2))}"
             f"{'0.3'.ljust(col(0.4) - col(0.3))}0.4"]
    lines += [f"guidance {gg}  {strip(s)}" for gg, s in results]
    lines += ["● measured   [===] 95% interval over line-ups   p predicted from battle noise alone   | the 0.10 rule", "```", "",
              "| Guidance | Line-ups | Flips | Share (95%) | Predicted | Held at 4 | Held at 16 | Held → not | Not → held | Correlation 4 vs 16 | Verdict |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for gg, s in results:
        a, b = s["flip_share_interval"]
        lines.append(f"| {gg} | {s['line_ups']} | {s['flips']} | {s['flip_share']:.3f} ({a:.3f}–{b:.3f}) | {s['predicted_from_4']['mean']:.3f} | "
                     f"{s['held_at_4']} | {s['held_at_16']} | {s['held_to_not_held']} | {s['not_held_to_held']} | {s['correlation']:.2f} | {s['verdict']} |")
    lines += ["", f"Scorer: behaviour-cloning policy on both sides, slots shuffled, the 49 distinct top-50 meta teams; 4-battle arm seed 7520, "
              f"{last['depth']}-battle arm seed {L.SEED} in its own database. Results: `~/vgc-data/results/line_up_yield_<grid>_d{last['depth']}.json`. "
              "Pre-registration: `code/docs/weakness-pressure-deep-evaluation.md` part A."]
    return "\n".join(lines) + "\n"


def results_so_far():
    out = []
    for model, g in GRIDS:
        p = L.fresh_path(model)
        if p.exists() and "flips" in (d := json.load(open(p))):
            out.append((g, d["flips"]))
    return out


if __name__ == "__main__":
    r = results_so_far()
    if not r:
        sys.exit("no guidance has a flips block yet")
    sys.stdout.write(comment(r))
