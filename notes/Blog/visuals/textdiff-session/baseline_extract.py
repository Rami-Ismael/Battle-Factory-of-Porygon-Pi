"""Copies the Mask Lab baseline results for the page into sources/baseline_results.json.

    python3 baseline_extract.py

Reads each grid's run folder in team_completion_pipeline/runs (eval2.tasks.json: the 270 masks; panels.json: every
legal filled team's 49 battles against the top-50 meta teams, behaviour-cloning policy on both sides). Net per mask =
the fill's win rate minus its starting team's, with a rejected fill keeping the starting team (change 0). Each
starting team weighs equally, 95% bootstrap over the five starting teams, the same draws and seed as
code/scripts/text_vs_ling.py. The random grid was re-battled in full on 2026-10-24, so its numbers differ from the
2026-10-21 text_vs_ling.json by up to 1 point.
"""
import json, random, re
from pathlib import Path

RUNS = Path.home() / "Documents/yakumsi-vault/Personal Project/Using Advance method search in the whole search of possible vgc pokemon format to find the right counter to a pokemon team/team_completion_pipeline/runs"
GRIDS = {"random": "prompt-ab", "heldout": "heldout-starters", "meta": "meta-starters"}
METHODS = {"textdiff": "textdiff", "textdiff3b": "textdiff3b", "diffusion": "diffusion", "random": "random",
           "llm_regmb": "llm_regmb", "llm_baseline": "llm_baseline", "llm_mb-original-v1": "llm_mb-original-v1",
           "ling": "ling:regmb@16384", "ling_baseline16k": "ling:baseline@16384", "ling_baseline4k": "ling:baseline@4096"}
# the prompt-version writers' random-grid fills were battled in their own run (same opponents and policy), kept beside it
EXTRA = {"random": Path.home() / "vgc-data/textdiff_llmformat/random-run/panels.json"}
SEED = 20261008


def bootstrap(pairs):
    by_start = {}
    for start, value in pairs: by_start.setdefault(start, []).append(value)
    means = [sum(v) / len(v) for v in by_start.values()]
    rng = random.Random(SEED); draws = sorted(sum(rng.choice(means) for _ in means) / len(means) for _ in range(4000))
    return dict(mean=round(sum(means) / len(means) * 100, 2), lo=round(draws[100] * 100, 2), hi=round(draws[-100] * 100, 2),
                n=sum(len(v) for v in by_start.values()))


def example(task="0dd56a1a37aceb04", start="HPS003641", slot="Mawile-Mega"):
    """One random-grid mask, as every writer filled it: the starting team, the hidden move, each fill and its win rate."""
    run = RUNS / "prompt-ab"
    panels = json.load(open(run / "panels.json"))["panels"]
    cands = {c["label"]: c for c in json.load(open(run / "candidates.json"))["candidates"] if "paste" in c}
    manifest = json.load(open(Path.home() / "vgc-data/mask_lab_diffusion/baseline/manifest.json"))
    team = next(t for t in manifest["starts"] if any(s["species"] == slot for s in t))
    moves = lambda label: [l[2:] for b in cands[label]["paste"].split("\n\n") if b.startswith(slot) for l in b.splitlines() if l.startswith("- ")]
    hidden = next(s for s in team if s["species"] == slot)["moves"][0]
    fills = [dict(writer=w, move=next(m for m in moves(f"{prefix}:{task}") if m not in next(s for s in team if s["species"] == slot)["moves"][1:]),
                  win=panels[f"{prefix}:{task}"]["score"]) for w, prefix in (("textdiff3b", "textdiff3b"), ("textdiff", "textdiff"),
                  ("ling", "ling:regmb@16384"), ("diffusion", "diffusion"), ("random", "random"))]
    # the answer text the writer was given (mask_lab_textdiff.py: earlier Pokémon in full, then this one up to the gap;
    # moves in training order, known ones alphabetical and the hidden one last) and what the 3B writer wrote after it
    key = lambda m: re.sub(r"[^a-z0-9]", "", m.lower())
    block = lambda t, moves: "\n".join([f"{t['species']} @ {t['item']}", f"Ability: {t['ability']}", f"{t['nature']} Nature"] + [f"- {m}" for m in moves])
    i = next(j for j, t in enumerate(team) if t["species"] == slot)
    prefix = "\n\n".join(block(t, sorted([m for m in t["moves"] if m], key=key)) for t in team[:i])
    prefix += "\n\n" + block(team[i], sorted([m for m in team[i]["moves"][1:] if m], key=key)) + "\n"
    raw = json.load(open(Path.home() / f"vgc-data/mask_lab_diffusion/textdiff3b/raw/{task}.json"))["raw"][0]["text"]
    return dict(task=task, start=start, slot=slot, hidden=hidden, original=panels[f"orig:{start}"]["score"],
                team=[dict(species=s["species"], item=s["item"]) for s in team], fills=fills, prefix=prefix, written_3b=raw)


def main():
    out = {}
    for grid, folder in GRIDS.items():
        run = RUNS / folder
        panels = json.load(open(run / "panels.json"))["panels"]
        if grid in EXTRA and EXTRA[grid].exists(): panels.update(json.load(open(EXTRA[grid]))["panels"])
        masks = json.load(open(run / "eval2.tasks.json"))["tasks"]
        score = lambda label: (panels.get(label) or {}).get("score")
        rows = []
        for m in masks:
            r = dict(start=m["source"], family=m["task"], original=score(f"orig:{m['source']}"))
            for name, prefix in METHODS.items(): r[name] = score(f"{prefix}:{m['id']}")
            rows.append(r)
        net = lambda r, k: (r[k] if r[k] is not None else r["original"]) - r["original"]
        present = [k for k in METHODS if any(r[k] is not None for r in rows)]
        g = dict(masks=len(rows), originals={s: round(score(f"orig:{s}"), 3) for s in sorted({r["start"] for r in rows})}, methods={})
        for k in present:
            g["methods"][k] = dict(legal=sum(r[k] is not None for r in rows),
                                   net=bootstrap([(r["start"], net(r, k)) for r in rows]),
                                   families={f: bootstrap([(r["start"], net(r, k)) for r in rows if r["family"] == f])
                                             for f in sorted({r["family"] for r in rows})})
        g["net_3b_minus_15b"] = bootstrap([(r["start"], net(r, "textdiff3b") - net(r, "textdiff")) for r in rows])
        out[grid] = g
    out["example"] = example()
    out["source"] = "team_completion_pipeline/runs/{prompt-ab,heldout-starters,meta-starters}/panels.json, read 2026-10-24"
    Path(__file__).with_name("sources").joinpath("baseline_results.json").write_text(json.dumps(out, indent=1))
    for grid in GRIDS:
        print(grid, {k: (v["legal"], v["net"]["mean"], v["net"]["lo"], v["net"]["hi"]) for k, v in out[grid]["methods"].items()},
              "3B-1.5B", out[grid]["net_3b_minus_15b"])


if __name__ == "__main__":
    main()
