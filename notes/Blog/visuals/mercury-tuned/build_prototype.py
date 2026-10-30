"""PROTOTYPE: three video styles for post 2 (tuned Mercury closes most of the gap to Jev).

Writes post2-variants.prototype.html from post2-variants.prototype.template.html, with every
number read from the files below. Throwaway: the winning variant gets rebuilt properly.

    python3 notes/Blog/visuals/mercury-tuned/build_prototype.py
"""
import base64
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
VISUALS = HERE.parent
DATA = Path.home() / "vgc-data"
PIPELINE = Path("/Users/ramiismael/Documents/yakumsi-vault/Personal Project/Using Advance method search in the whole "
                "search of possible vgc pokemon format to find the right counter to a pokemon team/"
                "team_completion_pipeline/runs/heldout-starters")
EXAMPLE_TASK = "1c26d075be57f269"          # unmasked team MB401, 9 moves hidden, one of them Kleavor's second move
LEVELS = ["off", "low", "medium", "high"]


def png_uri(path):
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


def main():
    top50 = json.loads((DATA / "jev/measurements.json").read_text())["rebattle"]["top50"]
    arms, pairs = top50["arms"], top50["pairs"]
    plain = [arms[f"Mercury, {lv}"] for lv in LEVELS]
    tuned = [arms[f"Mercury DSPy, {lv}"] for lv in LEVELS]
    jev = arms["Jev, top pick"]
    shared = pairs["Mercury DSPy, high"]

    tasks = [json.loads(line) for line in (PIPELINE / "mercury.jsonl").read_text().splitlines()]
    candidates = json.loads((DATA / "mercury-dspy/heldout/candidates.json").read_text())["candidates"]
    tuned_legal = {c["task_id"] for c in candidates if c["kind"] == "mercury-dspy-high"}
    plain_row = next(r for r in tasks if r["task_id"] == EXAMPLE_TASK)
    masked = next(t for t in json.loads((PIPELINE / "eval2.tasks.json").read_text())["tasks"]
                  if t["id"] == EXAMPLE_TASK)
    tuned_team = json.loads((DATA / f"mercury-dspy/heldout/high/completions/{EXAMPLE_TASK}.json").read_text())["team"]

    sprites = json.loads((VISUALS / "diversity-methods/sprites.json").read_text())
    team = []
    for i, (given, filled) in enumerate(zip(masked["masked_team"], tuned_team)):
        item_key = "".join(ch for ch in given["item"].lower() if ch.isalnum())
        team.append({
            "species": given["species"], "item": given["item"],
            "sprite": sprites["".join(ch for ch in given["species"].lower() if ch.isalnum())],
            "itemIcon": png_uri(VISUALS / f"item-sprites/item-{item_key}.png"),
            "moves": given["moves"], "filled": filled["moves"],
            "hidden": [j for j in range(4) if 4 * i + j in masked["slots"]],
        })

    data = {
        "levels": LEVELS,
        "plain": {"legal": [a["legal"] for a in plain], "usable": [a["usable"]["mean"] for a in plain]},
        "tuned": {"legal": [a["legal"] for a in tuned], "usable": [a["usable"]["mean"] for a in tuned]},
        "jev": {"legal": jev["legal"], "usable": jev["usable"]["mean"]},
        "shared": {"masks": shared["masks"], "jev": shared["jev"], "mercury": shared["other"]},
        "tasks": len(tasks),
        "grid": [{"plain": r["status"] == "legal", "tuned": r["task_id"] in tuned_legal} for r in tasks],
        "exampleIndex": next(i for i, r in enumerate(tasks) if r["task_id"] == EXAMPLE_TASK),
        "example": {"source": masked["source"], "hidden": masked["k"], "team": team,
                    "plainError": plain_row["error"], "tunedLegal": EXAMPLE_TASK in tuned_legal},
        "files": "~/vgc-data/jev/measurements.json (re-battle, top-50)",
    }
    assert sum(g["plain"] for g in data["grid"]) == data["plain"]["legal"][-1]
    assert sum(g["tuned"] for g in data["grid"]) == data["tuned"]["legal"][-1]
    html = (HERE / "post2-variants.prototype.template.html").read_text().replace("__DATA__", json.dumps(data))
    (HERE / "post2-variants.prototype.html").write_text(html)
    print("wrote post2-variants.prototype.html")


if __name__ == "__main__":
    main()
