"""Every headline Jev and Mercury win rate in one place: docs/jev-measurements.md and ~/vgc-data/jev/measurements.json.

Each number says what it averages over, because two correct numbers can differ only in their mask set: Jev's top pick
wins 48.1% over all 270 held-out masks but 48.6% over the 183 masks Mercury (high) also filled legally. Comparisons
between two models always use the masks both filled legally.

Reads each grid's panels.json (the battle step's per-team scores) and eval2.tasks.json: listed slot order, one battle
per top-50 opponent (49 per team). The held-out grid's shuffled re-battle (scripts/jev_rebattle.py, 2026-10-29) comes
from ~/vgc-data/jev/rebattle-heldout.json: six Pokémon shuffled every battle, 8 battles per opponent, against the
top-50 teams and against the held-out teams. Intervals are 95%, from resampling masks and from resampling the five
unmasked teams (jev_rebattle.interval). The re-battle also carries Mercury with its best DSPy prompt
(scripts/mercury_dspy_grid.py, 2026-10-29), and the diversity table comes from that script's diversity.json.

Every model row names the prompt its teams came from (PROMPTS), in the tables and in measurements.json, so a result is
never quoted against the wrong prompt. The tuned-prompt rows are checked against the program hash each reasoning
setting's manifest recorded when it was sent.

    python3 scripts/jev_numbers.py
"""
import json
import statistics
from pathlib import Path

from jev_rebattle import interval

PIPELINE = Path("/Users/ramiismael/Documents/yakumsi-vault/Personal Project/Using Advance method search in the whole search of "
                "possible vgc pokemon format to find the right counter to a pokemon team/team_completion_pipeline/runs")
GRIDS = {"held-out": PIPELINE / "heldout-starters", "meta": PIPELINE / "meta-starters", "random": PIPELINE / "prompt-ab"}
REBATTLE = Path.home() / "vgc-data/jev/rebattle-heldout.json"
DOC = Path(__file__).resolve().parents[1] / "docs/jev-measurements.md"
OUT = Path.home() / "vgc-data/jev/measurements.json"
ARMS = {"Jev, top pick": "jev:top:", "Mercury, off": "mercury-off:regmb@16384:", "Mercury, low": "mercury-low:regmb@16384:",
        "Mercury, medium": "mercury-medium:regmb@16384:", "Mercury, high": "mercury:regmb@16384:", "random legal fill": "random:"}
REBATTLE_ARMS = {**dict(zip(ARMS, ["jev", "mercury-off", "mercury-low", "mercury-medium", "mercury", "random"])),
                 **{f"Mercury DSPy, {level}": f"mercury-dspy-{level}" for level in ("off", "low", "medium", "high")}}
PROMPTS = {"Jev, top pick": "none: Jev picks from legal option lists (jev_fill)",
           **{arm: "Mask Lab prompt, regmb version (ling_grid.py)" for arm in ARMS if arm.startswith("Mercury")},
           **{arm: "GEPA-tuned prompt on legal-choice types (dspy-optimize-luna4-20261016)"
              for arm in REBATTLE_ARMS if arm.startswith("Mercury DSPy")},
           "random legal fill": "none"}
DSPY_RUN = Path.home() / "vgc-data/mercury-dspy/heldout"   # mercury_dspy_grid.OUT
DSPY_PROGRAM_SHA256 = "b16ce050756f05fb985a0b551b319713b9858e6399578ecdccf8a9774e3305f6"   # dspy-optimize-luna4-20261016
DIVERSITY = Path.home() / "vgc-data/mercury-dspy/heldout/diversity.json"   # mercury_dspy_grid.DIVERSITY
WRITTEN = {"species": "Species (whole Pokémon)", "item": "Items", "ability": "Abilities", "nature": "Stat Alignment",
           "moves": "Moves", "statPoints": "Stat Points spreads"}
REBATTLE_SETS = {"top50": "held-out grid, re-battled against the top-50 teams",
                 "heldout": "held-out grid, re-battled against the held-out teams"}


def grid_numbers(folder):
    panels = json.loads((folder / "panels.json").read_text())["panels"]
    tasks = {t["id"]: t for t in json.loads((folder / "eval2.tasks.json").read_text())["tasks"]}
    score = lambda label: (panels.get(label) or {}).get("score")
    win = {(arm, i): score(prefix + i) for arm, prefix in ARMS.items() for i in tasks}
    unmasked = [p["score"] for label, p in panels.items() if label.startswith("orig:") and p.get("score") is not None]
    return numbers(tasks, win, unmasked)


def check_dspy_prompt():
    """Refuse to label the tuned-prompt rows unless every reasoning setting sent the program PROMPTS names."""
    for level in ("off", "low", "medium", "high"):
        sent = json.loads((DSPY_RUN / level / "manifest.json").read_text())["program_sha256"]
        if sent != DSPY_PROGRAM_SHA256:
            raise SystemExit(f"Mercury DSPy, {level} sent program {sent[:12]}, not dspy-optimize-luna4-20261016; update PROMPTS")


def rebattle_numbers():
    """The re-battle's per-mask scores, one entry per opponent set, in grid_numbers' shape."""
    sets = json.loads(REBATTLE.read_text())["sets"]
    tasks = {t["id"]: t for t in json.loads((GRIDS["held-out"] / "eval2.tasks.json").read_text())["tasks"]}
    return {name: dict(numbers(tasks, {(arm, i): s["scores"].get(key, {}).get(i) for arm, key in REBATTLE_ARMS.items() for i in tasks},
                               [v for v in s["unmasked"].values() if v is not None]), opponents=s["opponents"])
            for name, s in sets.items()}


def numbers(tasks, win, unmasked):
    """win: {(arm, task id): score, or None when the arm has no legal team}; unmasked: the unmasked teams' scores."""
    present = [arm for arm in dict.fromkeys(arm for arm, _ in win) if any(win[(arm, i)] is not None for i in tasks)]
    out = {"masks": len(tasks), "unmasked_teams": {"mean": round(statistics.mean(unmasked), 4), "teams": len(unmasked)}, "arms": {}, "pairs": {}}
    for arm in present:
        legal = [i for i in tasks if win[(arm, i)] is not None]
        out["arms"][arm] = {"prompt": PROMPTS[arm], "legal": len(legal),
                            "usable": interval([(tasks[i]["source"], win[(arm, i)] or 0) for i in tasks], "mean"),
                            "when_legal": interval([(tasks[i]["source"], win[(arm, i)]) for i in legal], "mean")}
    for arm in present:
        if arm.startswith("Mercury"):
            shared = [i for i in tasks if win[("Jev, top pick", i)] is not None and win[(arm, i)] is not None]
            out["pairs"][arm] = {"prompt": PROMPTS[arm], "jev": round(statistics.mean(win[("Jev, top pick", i)] for i in shared), 4),
                                 "other": round(statistics.mean(win[(arm, i)] for i in shared), 4),
                                 **interval([(tasks[i]["source"], win[("Jev, top pick", i)] - win[(arm, i)]) for i in shared], "diff")}
    return out


def pct(x):
    return f"{x * 100:.1f}%"


def ci(r):
    lo, hi = r["ci_teams"]
    return f"[{lo * 100:.1f}, {hi * 100:.1f}]"


def section(title, d):
    lines = [f"## {title}", "", f"Unmasked teams: {pct(d['unmasked_teams']['mean'])} ({d['unmasked_teams']['teams']} teams).", "",
             "| Model | Prompt | Legal | Usable win rate | When legal |", "|---|---|---|---|---|"]
    for arm, r in d["arms"].items():
        lines.append(f"| {arm} | {r['prompt']} | {r['legal']}/{d['masks']} | {pct(r['usable']['mean'])} {ci(r['usable'])} | {pct(r['when_legal']['mean'])} over {r['legal']} |")
    if d["pairs"]:
        lines += ["", "Same masks, Jev against each Mercury setting:", "",
                  "| Against | Its prompt | Masks | Jev | Other | Jev minus other, points |", "|---|---|---|---|---|---|"]
        for arm, r in d["pairs"].items():
            lines.append(f"| {arm} | {r['prompt']} | {r['masks']} | {pct(r['jev'])} | {pct(r['other'])} | {r['diff'] * 100:+.1f} {ci(r)} |")
    return [*lines, ""]


def diversity_section():
    """What each arm wrote into the masks: written / distinct / effective values per field type."""
    if not DIVERSITY.exists(): return []
    d = json.loads(DIVERSITY.read_text())
    names = {key: arm for arm, key in REBATTLE_ARMS.items()}
    lines = ["# Diversity of the filled values (held-out grid)", "",
             f"**Diversity:** {d['definition']}. Effective values grow with the number written, so compare arms with "
             "similar counts. Each cell is written / distinct / effective.", "",
             "| Model | Prompt | " + " | ".join(WRITTEN.values()) + " |", "|---" * (len(WRITTEN) + 2) + "|"]
    for key, fields in d["arms"].items():
        cells = [f"{f['written']} / {f['distinct']} / {f['effective']}" if (f := fields.get(k)) else "—" for k in WRITTEN]
        lines.append(f"| {names[key]} | {PROMPTS[names[key]]} | " + " | ".join(cells) + " |")
    lines += ["", "Same masks: effective values written by each Mercury setting, and by Jev's top pick on the masks that "
              "setting filled legally (Mercury / Jev). This is the fair diversity comparison.", "",
              "| Model | Prompt | " + " | ".join(WRITTEN.values()) + " |", "|---" * (len(WRITTEN) + 2) + "|"]
    for key, fields in d.get("jev_same_masks", {}).items():
        if not key.startswith("mercury"): continue
        cells = [f"{d['arms'][key][k]['effective']} / {f['effective']}" if (f := fields.get(k)) and k in d["arms"][key] else "—"
                 for k in WRITTEN]
        lines.append(f"| {names[key]} | {PROMPTS[names[key]]} | " + " | ".join(cells) + " |")
    return [*lines, ""]


def main():
    check_dspy_prompt()
    data = {"rebattle_definition": "share of battles won, behaviour-cloning policy on both sides, six Pokémon shuffled every "
                                   "battle, 8 battles per opponent; against the 49 top-50 meta teams, or against the 45 "
                                   "held-out teams (ranks 51-100 minus the five unmasked teams)",
            "definition": "share of battles won against the 49 top-50 meta teams, behaviour-cloning policy on both sides; "
                          "listed slot order, one battle per opponent",
            "rebattle": rebattle_numbers(), "grids": {g: grid_numbers(f) for g, f in GRIDS.items()}}
    OUT.write_text(json.dumps(data, indent=1) + "\n")
    lines = ["# Jev and Mercury measurements", "",
             "Generated by `scripts/jev_numbers.py` from `~/vgc-data/jev/rebattle-heldout.json` and each grid's `panels.json`; "
             "rerun it after any battle step. Do not edit by hand.", "",
             "Intervals are 95%, from resampling the five unmasked teams.", "",
             "- **Usable win rate:** over all 270 masks; a mask with no legal team scores 0.",
             "- **When legal:** over the masks this model filled legally. Two models' when-legal numbers sit on different masks, so never compare them.",
             "- **Same masks:** over the masks both models filled legally. This is the only fair head-to-head.", "",
             "# Shuffled re-battle (2026-10-29)", "", f"**Win rate:** {data['rebattle_definition']}.", ""]
    for name, d in data["rebattle"].items():
        lines += section(REBATTLE_SETS[name], d)
    lines += diversity_section()
    lines += ["# First battles, listed slot order", "", f"**Win rate:** {data['definition']}. The behaviour-cloning policy "
              "picks its four by slot number, so these scores carry a slot-order bias; prefer the re-battle above.", ""]
    for g, d in data["grids"].items():
        lines += section(f"{g} grid", d)
    DOC.write_text("\n".join(lines))
    print(DOC.read_text())


if __name__ == "__main__":
    main()
