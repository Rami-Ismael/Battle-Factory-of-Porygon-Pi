"""Mercury 2.5 with its best DSPy prompt on the held-out Mask Lab grid's own masks (2026-10-29).

The best prompt is GEPA's, written by GPT-6 Luna with each task's answer shape shown, on top of the legal-choice
answer types (docs/dspy-structure-experiment.md, run dspy-optimize-luna4-20261016: 64.6% pooled legality on the
reasoning study's 20 held-out starters). Plain Mercury on this grid got the Mask Lab's regmb prompt (ling_grid.py).
This sends the DSPy prompt to inception/mercury-2.5 on the same 270 masks at the same four reasoning settings, so its
fills meet the same re-battle and measurements as Jev and plain Mercury (jev_rebattle.py, jev_numbers.py).

    PY=~/.local/share/vgc-pilot-runtime/venv/bin/python
    $PY scripts/mercury_dspy_grid.py prepare --program <program.json>   # builds and checks every request, sends nothing
    $PY scripts/mercury_dspy_grid.py run --level off --shards 4 --shard 0   # paid; a rerun resumes
    $PY scripts/mercury_dspy_grid.py collect                            # legal fills -> CANDIDATES
    $PY scripts/mercury_dspy_grid.py diversity                          # what each arm wrote into the masks

Each level is an llm_completion manifest in OUT/<level>: starts are the grid's five unmasked teams, trials its 270
masks, opponents the top-50 meta teams plain Mercury's grid prompt got (jev_fill.meta_pastes), and the structure the
frozen program's instructions with legal-choice types and the chat adapter, as the program was optimized and tested.
llm_completion.complete sends, saves and validates each request. collect checks every legal team again with the grid's
own checks (jev_fill.check_legal) and writes it to CANDIDATES as mercury-dspy-<level>:<task id>, which
jev_rebattle.py reads beside the grid's candidates.json; this script never writes into the grid's folder.

diversity measures only the filled values, because the unmasked fields come from five fixed teams: the species
written into whole-Pokémon masks, and the values written into each field type, as distinct and effective counts
(exp of the Shannon entropy). It covers every re-battled arm and writes DIVERSITY, which jev_numbers.py renders.
"""
import argparse
import json
import math
import os
import shutil
import subprocess
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import repo_paths  # noqa: F401 -- puts src/ first on sys.path
import jev_fill
import llm_completion
import results
import showdown_validator
from team_masks import masked

OUT = Path.home() / "vgc-data/mercury-dspy/heldout"
CANDIDATES = OUT / "candidates.json"
DIVERSITY = OUT / "diversity.json"
GRID = jev_fill.GRIDS["heldout"]
# The 2026-10-13 reasoning study's settings, which plain Mercury on this grid also used (ling_grid.MODELS).
LEVELS = {"off": {"enabled": False}, "low": {"effort": "low"}, "medium": {"effort": "medium"}, "high": {"effort": "high"}}
SETTINGS = dict(model="inception/mercury-2.5", format=showdown_validator.FORMAT, temperature=0.7, max_tokens=16384,
                request_timeout=600.0,
                provider={"order": ["inception"], "allow_fallbacks": False, "require_parameters": True})


def to_path(text):
    """'team[3].statPoints' -> [3, 'evs']: a Mask Lab path as an llm_completion path."""
    path = jev_fill.parse_path(text)
    return [path[0], "evs"] if path[1:] == ["statPoints"] else path


def to_llm(team):
    """Mask Lab slots as llm_completion slots {species, item, ability, nature, moves, evs}; None stays None."""
    return [None if s is None else dict(species=s["species"], item=s["item"], ability=s["ability"], nature=s["nature"],
                                         moves=list(s["moves"]), evs=s["statPoints"]) for s in team]


def to_mask_lab(team, original):
    """A filled llm_completion team back in the original's Mask Lab slots (slot number, ivs, level kept)."""
    return [dict(o, species=s["species"], item=s["item"], ability=s["ability"], nature=s["nature"],
                 moves=list(s["moves"]), statPoints=dict(s["evs"])) for s, o in zip(team, original)]


def tasks():
    return jev_fill.tasks("heldout")


def manifest(program, level):
    import dspy_structure as S
    starts, index, trials = [], {}, []
    for task in tasks():
        if task["source"] not in index:
            index[task["source"]] = len(starts)
            starts.append(to_llm(task["original"]))
        trial = dict(id=task["id"], start=index[task["source"]], task=task["task"], k=task["k"],
                     paths=[to_path(p) for p in task["masked_paths"]])
        if masked(starts[trial["start"]], trial["paths"]) != to_llm(task["masked_team"]):
            raise ValueError(f"{task['id']}: converted paths hide different fields than the grid's mask")
        trials.append(trial)
    opponents = [dict(team_id=i, paste=p) for i, p in enumerate(jev_fill.meta_pastes())]
    structure = S.config("chat", instructions=program["instructions"], demos=program["demos"], constraints="legal-choices")
    return dict(SETTINGS, schema=1, seed=None, reasoning=LEVELS[level], starts=starts, trials=trials, opponents=opponents,
                structure=structure, grid=str(GRID), program_sha256=results.digest(program))


def prepare(a):
    import dspy_structure as S
    program = json.loads(a.program.read_text())
    if program["optimizer"].get("constraints") != "legal-choices":
        raise SystemExit("the program was not optimized with legal-choice types")
    OUT.mkdir(parents=True, exist_ok=True)
    shutil.copy(a.program, OUT / "program.json")
    for level in LEVELS:
        folder = OUT / level
        m = manifest(program, level)
        old = folder / "manifest.json"
        if old.exists() and json.loads(old.read_text()) != m:
            raise SystemExit(f"{old} exists with different requests; move it aside first")
        for trial in m["trials"]:
            S.check_information(m, trial)              # every grid input reaches the structured request unchanged
        results.save(old, m)
        sizes = [len(json.dumps(llm_completion.completion_payload(m, t))) for t in m["trials"]]
        print(f"{level}: {len(m['trials'])} requests, {len(m['starts'])} starts, {len(m['opponents'])} opponents, "
              f"{sum(sizes) / len(sizes) / 1000:.0f}k characters on average")


def api_key():
    if os.environ.get("OPENROUTER_API_KEY"): return
    r = subprocess.run(["security", "find-generic-password", "-s", "openrouter", "-w"], capture_output=True, text=True)
    if r.returncode or not r.stdout.strip():
        raise SystemExit("no OpenRouter key: set OPENROUTER_API_KEY or add the Keychain item 'openrouter'")
    os.environ["OPENROUTER_API_KEY"] = r.stdout.strip()


def run(a):
    api_key()
    llm_completion.complete(SimpleNamespace(output=OUT / a.level, shards=a.shards, shard=a.shard))


def collect(_a):
    pipeline, by_id = jev_fill.load_pipeline(), {t["id"]: t for t in tasks()}
    cands, summary = [], {}
    for level in LEVELS:
        m = json.loads((OUT / level / "manifest.json").read_text())
        status, cost = Counter(), 0.0
        for trial in m["trials"]:
            path = OUT / level / "completions" / f"{trial['id']}.json"
            if not path.exists():
                status["not sent"] += 1; continue
            record = json.loads(path.read_text())
            cost += ((record.get("response") or {}).get("usage") or {}).get("cost") or 0
            if not record["valid"]:
                status["illegal"] += 1; continue
            task = by_id[trial["id"]]
            team = to_mask_lab(record["team"], task["original"])
            try:
                jev_fill.check_legal(pipeline, task, team)
            except ValueError as error:
                status["illegal"] += 1; print(f"{level} {trial['id']}: passed llm_completion, failed the grid: {error}"); continue
            status["legal"] += 1
            cands.append(dict(label=f"mercury-dspy-{level}:{trial['id']}", kind=f"mercury-dspy-{level}", task_id=trial["id"],
                              source=task["source"], team=team, paste=jev_fill.team_paste(team)))
        summary[level] = dict(status, cost_usd=round(cost, 4))
        print(level, dict(status), f"${cost:.3f}")
    results.save(CANDIDATES, {"candidates": cands, "summary": summary})
    print(f"wrote {len(cands)} candidates to {CANDIDATES}")


def effective(counts):
    total = sum(counts.values())
    return round(math.exp(-sum(c / total * math.log(c / total) for c in counts.values())), 2) if total else None


def filled(task, team):
    """{field kind: [values written into the mask]}; a whole Pokémon contributes its species."""
    out = {}
    for text in task["masked_paths"]:
        path = jev_fill.parse_path(text)
        slot = team[path[0]]
        if len(path) == 1:
            out.setdefault("species", []).append(slot["species"])
        elif path[1] == "moves":
            out.setdefault("moves", []).append(slot["moves"][path[2]])
        elif path[1] == "statPoints":
            out.setdefault("statPoints", []).append(json.dumps(slot["statPoints"], sort_keys=True))
        else:
            out.setdefault(path[1], []).append(slot[path[1]])
    return out


def written_by_arm():
    """({arm: {field: Counter}}, {arm: {field: Counter}} of Jev's values on that arm's legal masks)."""
    from jev_rebattle import candidates as rebattled
    by_id = {t["id"]: t for t in tasks()}
    values, masks, jev = {}, {}, {}
    for c in rebattled():
        if c["arm"] == "unmasked" or not c.get("team"): continue
        written = filled(by_id[c["task_id"]], c["team"])
        masks.setdefault(c["arm"], set()).add(c["task_id"])
        if c["arm"] == "jev": jev[c["task_id"]] = written
        for kind, vals in written.items():
            values.setdefault(c["arm"], {}).setdefault(kind, Counter()).update(vals)
    same = {}
    for arm, ids in masks.items():
        for i in sorted(ids & set(jev)) if arm != "jev" else ():
            for kind, vals in jev[i].items():
                same.setdefault(arm, {}).setdefault(kind, Counter()).update(vals)
    return values, same


def table(title, arms):
    kinds = ["species", "item", "ability", "nature", "moves", "statPoints"]
    print(f"{title:22}" + "".join(f"{k:>24}" for k in kinds))
    for arm, fields in arms.items():
        print(f"{arm:22}" + "".join(f"{'{written} / {distinct} / {effective}'.format(**fields[k]) if k in fields else '-':>24}" for k in kinds))


def diversity(_a):
    values, same = written_by_arm()
    count = lambda c: dict(written=sum(c.values()), distinct=len(c), effective=effective(c))
    out = {"definition": "values each arm wrote into the held-out grid's masks, over its legal fills only: distinct "
                         "values, and effective values = exp(Shannon entropy) of their frequencies",
           "arms": {arm: {kind: count(c) for kind, c in sorted(fields.items())} for arm, fields in values.items()},
           "jev_same_masks": {arm: {kind: count(c) for kind, c in sorted(fields.items())} for arm, fields in same.items()}}
    results.save(DIVERSITY, out)
    table("written/distinct/eff.", out["arms"])
    table("Jev, same masks", out["jev_same_masks"])
    print("wrote", DIVERSITY)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare").add_argument("--program", type=Path, required=True)
    r = sub.add_parser("run")
    r.add_argument("--level", choices=list(LEVELS), required=True)
    r.add_argument("--shards", type=int, default=1); r.add_argument("--shard", type=int, default=0)
    sub.add_parser("collect"); sub.add_parser("diversity")
    a = p.parse_args()
    globals()[a.command](a)
