"""Post 2 figure: a tuned prompt closed most of Mercury's gap to Jev (variant C of the prototype, promoted).

Writes post2-mercury-tuned.html from post2-mercury-tuned.template.html, with every number read from the
re-battle measurements. render_post2_video.py turns the page into the mp4 and the still.

    python3 notes/Blog/visuals/mercury-tuned/build_post2_figure.py
"""
import base64
import itertools
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
VISUALS = HERE.parent
DATA = Path.home() / "vgc-data"
TASKS = Path("/Users/ramiismael/Documents/yakumsi-vault/Personal Project/Using Advance method search in the whole "
             "search of possible vgc pokemon format to find the right counter to a pokemon team/"
             "team_completion_pipeline/runs/heldout-starters/eval2.tasks.json")
EXAMPLE_TASK = "9ef4e0a9d46ca625"          # unmasked team MB574, led by Charizard, with 9 moves hidden: the opening's task
LEVELS = ["off", "low", "medium", "high"]


def png_uri(path):
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


def key(name):
    return "".join(ch for ch in name.lower() if ch.isalnum())


def gap_range(scores, team_of):
    """Share of Jev's lead that tuning closed, (tuned - plain) / (Jev - plain), with illegal masks scoring 0.

    The 95% range comes from resampling the five unmasked teams together for all three arms, over all 5^5 draws."""
    teams = sorted(set(team_of.values()))
    masks = {t: [m for m, s in team_of.items() if s == t] for t in teams}
    def share(draw):
        ids = [m for t in draw for m in masks[t]]
        jev, plain, tuned = (sum(scores[a].get(m) or 0 for m in ids) / len(ids) for a in ("jev", "mercury", "mercury-dspy-high"))
        return (tuned - plain) / (jev - plain)
    draws = sorted(share(d) for d in itertools.product(teams, repeat=len(teams)))
    return share(teams), {"lo": draws[round(.025 * (len(draws) - 1))], "hi": draws[round(.975 * (len(draws) - 1))]}


def main():
    top50 = json.loads((DATA / "jev/measurements.json").read_text())["rebattle"]["top50"]
    arms, shared = top50["arms"], top50["pairs"]["Mercury DSPy, high"]
    plain = [arms[f"Mercury, {lv}"] for lv in LEVELS]
    tuned = [arms[f"Mercury DSPy, {lv}"] for lv in LEVELS]
    jev = arms["Jev, top pick"]
    task = next(t for t in json.loads(TASKS.read_text())["tasks"] if t["id"] == EXAMPLE_TASK)
    sprites = json.loads((VISUALS / "diversity-methods/sprites.json").read_text())
    scores = json.loads((DATA / "jev/rebattle-heldout.json").read_text())["sets"]["top50"]["scores"]
    team_of = {r["task_id"]: r["source"] for r in map(json.loads, (TASKS.parent / "mercury.jsonl").read_text().splitlines())}
    point, gap = gap_range(scores, team_of)
    # The per-mask scores must reproduce the headline numbers, so the range belongs to the same arms.
    headline = (tuned[3]["usable"]["mean"] - plain[3]["usable"]["mean"]) / (jev["usable"]["mean"] - plain[3]["usable"]["mean"])
    assert abs(point - headline) < .002, (point, headline)
    assert task["masked_team"][0]["species"] == "Charizard", "the opening's Fly-not-Surf line is about Charizard"

    data = {
        "levels": LEVELS,
        "plain": {"legal": [a["legal"] for a in plain], "usable": [a["usable"]["mean"] for a in plain]},
        "tuned": {"legal": [a["legal"] for a in tuned], "usable": [a["usable"]["mean"] for a in tuned]},
        "jev": {"legal": jev["legal"], "usable": jev["usable"]["mean"]},
        "shared": {"masks": shared["masks"], "jev": shared["jev"], "mercury": shared["other"]},
        "tasks": jev["usable"]["masks"],
        "example": {"lead": task["masked_team"][0]["species"], "hidden": task["k"], "team": [
            {"sprite": sprites[key(p["species"])], "itemIcon": png_uri(VISUALS / f"item-sprites/item-{key(p['item'])}.png")}
            for p in task["masked_team"]]},
        "gap": gap,
        "files": "jev-measurements.md in the repo",
    }
    html = (HERE / "post2-mercury-tuned.template.html").read_text().replace("__DATA__", json.dumps(data))
    (HERE / "post2-mercury-tuned.html").write_text(html)
    print("wrote post2-mercury-tuned.html")


if __name__ == "__main__":
    main()
