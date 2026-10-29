"""Build shuffle-figures.html, the figures that explain experiment 2 (ticket 10, shuffle the list).

Reads the run's records in ~/vgc-data/jev/ (shuffle-<grid>.jsonl, <grid>.jsonl) and each grid's panels,
picks two real held-out questions to show (one where shuffling picks a different move, one where it only
moves a move to another slot), and writes the data into shuffle-figures.template.html.

    /tmp/vgc-pilot/.venv/bin/python notes/Blog/visuals/jev-picks-a-team/build_shuffle_figures.py
"""
import collections
import json
import random
import re
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[3] / "code/scripts"))
import jev_fill  # noqa: E402
import jev_shuffle as S  # noqa: E402

DATA = Path.home() / "vgc-data/jev"
SPRITES = json.loads((HERE.parent / "diversity-methods/sprites.json").read_text())


def norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def top_teams(grid: str) -> dict:
    return {r["task_id"]: r["team"] for r in map(json.loads, open(DATA / f"{grid}.jsonl")) if r["run"] == "top"}


def boot(groups: dict, stat, n: int = 2000) -> list:
    keys, rng, vals = list(groups), random.Random(0), []
    for _ in range(n):
        vals.append(stat([x for k in (rng.choice(keys) for _ in keys) for x in groups[k]]))
    vals.sort()
    return [round(vals[int(0.025 * n)], 4), round(vals[int(0.975 * n)], 4)]


def grid_numbers(grid: str) -> dict:
    teams, rows = top_teams(grid), S.rows(grid)
    panels = {k: p.get("score") for k, p in json.loads((jev_fill.GRIDS[grid] / "panels.json").read_text())["panels"].items()}
    by = collections.defaultdict(list)
    for r in rows:
        mon = teams[r["task_id"]][jev_fill.parse_path(r["path"])[0]]
        a, s = r["alphabetical"]["choice"], r["shuffled"]["choice"]
        sw, tp = panels.get(f"{S.KIND}:{r['path']}:{r['task_id']}"), panels.get(f"jev:top:{r['task_id']}")
        by[r["source"]].append({
            "noise_move": r["main"] != a and a not in mon["moves"], "noise_slot": r["main"] != a and a in mon["moves"],
            "shuf_move": a != s and s not in mon["moves"], "shuf_slot": a != s and s in mon["moves"],
            "rel_s": r["shuffled"]["position"] / max(1, r["options"] - 1), "rel_a": r["alphabetical"]["position"] / max(1, r["options"] - 1),
            "diff": None if sw is None or tp is None else round(sw - tp, 3)})
    allr = [x for v in by.values() for x in v]
    out = {"questions": len(allr)}
    for k in ("noise_move", "noise_slot", "shuf_move", "shuf_slot"):
        out[k] = round(sum(x[k] for x in allr) / len(allr), 4)
    out["noise_ci"] = boot(by, lambda xs: sum(x["noise_move"] or x["noise_slot"] for x in xs) / len(xs))
    out["shuf_ci"] = boot(by, lambda xs: sum(x["shuf_move"] or x["shuf_slot"] for x in xs) / len(xs))
    out["hist_s"] = [sum(min(9, int(x["rel_s"] * 10)) == b for x in allr) for b in range(10)]
    out["hist_a"] = [sum(min(9, int(x["rel_a"] * 10)) == b for x in allr) for b in range(10)]
    diffs = {k: [x for x in v if x["diff"] is not None] for k, v in by.items()}
    diffs = {k: v for k, v in diffs.items() if v}
    ds = [x["diff"] for v in diffs.values() for x in v]
    out["win"] = {"n": len(ds), "mean": round(statistics.mean(ds), 4), "ci": boot(diffs, lambda xs: statistics.mean(x["diff"] for x in xs)),
                  "points": ds}
    return out


def example(kind: str) -> dict:
    """A real held-out question: kind 'move' (shuffled pick is a new move) or 'slot' (it is one of the Pokémon's moves)."""
    teams = top_teams("heldout")
    best = None
    for r in S.rows("heldout"):
        team = teams[r["task_id"]]
        i = jev_fill.parse_path(r["path"])[0]
        a, s = r["alphabetical"]["choice"], r["shuffled"]["choice"]
        if a == s or not 25 <= r["options"] <= 60 or not all(norm(m["species"]) in SPRITES for m in team if m):
            continue
        if (s in team[i]["moves"]) != (kind == "slot") or r["main"] != a:
            continue
        gap = abs(r["alphabetical"]["probabilities"].get(a, 0) - r["alphabetical"]["probabilities"].get(s, 0))
        if best is None or gap < best[0]:
            best = (gap, r, team, i)
    _, r, team, i = best
    meta = jev_fill.meta_pastes()
    q = next(q for q in S.replay("heldout", meta) if q["id"] == r["id"])
    _, qs = S.request(q)
    return {"path": r["path"], "slot": i, "move_index": jev_fill.parse_path(r["path"])[2], "options": r["options"],
            "team": [{"species": m["species"], "moves": m["moves"], "item": m.get("item")} for m in team],
            "unmasked_team": r["source"], "kind": kind,
            "alphabetical": {"order": list(qs["alphabetical"]["criteria"]), **{k: r["alphabetical"][k] for k in ("choice", "position", "probabilities")}},
            "shuffled": {"order": list(qs["shuffled"]["criteria"]), **{k: r["shuffled"][k] for k in ("choice", "position", "probabilities")}}}


def main() -> None:
    grids = {g: grid_numbers(g) for g in S.GRIDS}
    ex = {"move": example("move"), "slot": example("slot")}
    species = {m["species"] for e in ex.values() for m in e["team"] if m}
    data = {"grids": grids, "examples": ex, "sprites": {s: SPRITES[norm(s)] for s in species},
            "fakes": {"first_option": 0.98, "by_name": 0.0}, "cost": 6.52, "requests": 5542}
    page = (HERE / "shuffle-figures.template.html").read_text().replace("__DATA__", json.dumps(data, separators=(",", ":")).replace("</", "<\\/"))
    (HERE / "shuffle-figures.html").write_text(page)
    for k, e in ex.items():
        print(k, e["unmasked_team"], e["path"], e["team"][e["slot"]]["species"], e["alphabetical"]["choice"], "->", e["shuffled"]["choice"], e["options"], "options")
    print({g: {k: v[k] for k in ("questions", "noise_move", "noise_slot", "shuf_move", "shuf_slot")} for g, v in grids.items()})


if __name__ == "__main__":
    main()
