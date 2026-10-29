"""An LLM baseline (Ling, or Mercury 2.5 with --model mercury) on a Mask Lab grid's own masks: the random, held-out or
meta starting teams.

    /tmp/vgc-pilot/.venv/bin/python scripts/ling_grid.py run --grid heldout --dry-run      # counts calls, sends nothing
    /tmp/vgc-pilot/.venv/bin/python scripts/ling_grid.py run --grid heldout
    /tmp/vgc-pilot/.venv/bin/python scripts/ling_grid.py collect --grid heldout

run sends every (prompt version, mask) pair once, as prompt_ab.py sent the random grid's: benchmark.ling.json's method
(inclusionai/ling-3.0-flash-vl on Novita, temperature 0.7) with max_tokens 16384, through prompt_ab.attempt, so the
prompt, parsing and Showdown validation are the random grid's. Answers go to <grid>/ling.jsonl; a rerun skips finished
pairs and retries transport errors. The key comes from OPENROUTER_API_KEY or the macOS Keychain item "openrouter" and is
never printed. collect appends the legal answers to <grid>/candidates.json as ling:<version>@16384:<task> (the labels
of the random grid's Ling arm) after a timestamped backup; it never rebuilds the file, which holds the other arms'
fills. The grid's battle step and textdiff_llmformat.py report then score them.

--model mercury (2026-10-24; mercury-off, -low and -medium added 2026-10-28) sends the same prompt and masks to inception/mercury-2.5 on Inception, reasoning effort
high (the best of the 2026-10-13 reasoning study, docs/reasoning-experiment.md), require_parameters so a provider that
would drop the reasoning setting is refused. pipeline.py builds no reasoning field, so run() adds it to the request body.
Answers go to <grid>/mercury.jsonl and fills are labelled mercury:<version>@16384:<task>.
"""
import argparse, concurrent.futures as cf, json, os, shutil, subprocess, sys, threading, time
from pathlib import Path

PIPELINE = Path.home() / ("Documents/yakumsi-vault/Personal Project/Using Advance method search in the whole search of "
                          "possible vgc pokemon format to find the right counter to a pokemon team/team_completion_pipeline")
GRIDS = {"random": PIPELINE / "runs/prompt-ab", "heldout": PIPELINE / "runs/heldout-starters", "meta": PIPELINE / "runs/meta-starters"}
TOKENS = 16384
MODELS = {"ling": {},                                 # benchmark.ling.json's method as it is
          "mercury": dict(name="mercury-2.5", model="inception/mercury-2.5",
                          provider={"order": ["inception"], "allow_fallbacks": False, "require_parameters": True},
                          reasoning={"effort": "high"})}
# Mercury at the other reasoning settings of the 2026-10-13 study, requested exactly as that study did (off = enabled
# false, else an effort level), each to its own <grid>/mercury-<level>.jsonl and mercury-<level>:… labels (2026-10-28).
for _level, _reasoning in (("off", {"enabled": False}), ("low", {"effort": "low"}), ("medium", {"effort": "medium"})):
    MODELS[f"mercury-{_level}"] = dict(MODELS["mercury"], name=f"mercury-2.5-{_level}", reasoning=_reasoning)
# Novita stopped serving Ling by 2026-10-28 (pinned calls return HTTP 404); DeepInfra is its one OpenRouter provider
# left, so this is Ling at the same settings on DeepInfra, kept in its own <grid>/ling-deepinfra.jsonl.
MODELS["ling-deepinfra"] = dict(name="ling-3.0-flash-vl-deepinfra",
                                provider={"order": ["DeepInfra"], "allow_fallbacks": False})


def method(config, model):
    return dict(config["methods"][0], **{k: v for k, v in MODELS[model].items() if k != "reasoning"})


def add_reasoning(P, model):
    """pipeline.py sends no reasoning field: wrap its canonical() so the chat request body carries this model's."""
    reasoning = MODELS[model].get("reasoning")
    if not reasoning or getattr(P.canonical, "wrapped", False): return
    original = P.canonical
    def canonical(obj):
        if isinstance(obj, dict) and "messages" in obj and "model" in obj: obj = dict(obj, reasoning=reasoning)
        return original(obj)
    canonical.wrapped = True; P.canonical = canonical


def modules():
    if str(PIPELINE) not in sys.path: sys.path.insert(0, str(PIPELINE))
    import pipeline as P, prompt_ab as A, mask_lab_prepare as M
    return P, A, M


def api_key():
    if os.environ.get("OPENROUTER_API_KEY"): return
    r = subprocess.run(["security", "find-generic-password", "-s", "openrouter", "-w"], capture_output=True, text=True)
    if r.returncode or not r.stdout.strip():
        raise SystemExit("no OpenRouter key: set OPENROUTER_API_KEY or add the Keychain item 'openrouter'")
    os.environ["OPENROUTER_API_KEY"] = r.stdout.strip()


def run(a):  # noqa: C901
    P, A, _M = modules()
    config = A.load_config(str(PIPELINE / "benchmark.ling.json"))
    versions = a.versions.split(",")
    unknown = [v for v in versions if v not in P.PROMPTS]
    if unknown: raise SystemExit(f"unknown prompt {unknown}; known: {sorted(P.PROMPTS)}")
    arms = {f"{v}@{TOKENS}": dict(method(config, a.model), prompt=v, max_tokens=TOKENS, arm=f"{v}@{TOKENS}") for v in versions}
    meta = [row["team"] for row in P.load_teams(config["meta_teams"])]
    tasks = json.loads((GRIDS[a.grid] / "eval2.tasks.json").read_text())["tasks"]
    out = GRIDS[a.grid] / f"{a.model}.jsonl"
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            r = json.loads(line)
            if r["status"] != "transport": done.add((r["arm"], r["task_id"]))
    todo = [(arm, t) for t in tasks for arm in arms if (arm, t["id"]) not in done]
    if a.limit: todo = todo[:a.limit]                  # a smoke test: the first N pairs, kept in ling.jsonl like any other
    print(f"{a.grid} ({a.model}): {len(tasks)} masks x {len(arms)} versions; {len(done)} done; {len(todo)} to send", flush=True)
    if a.dry_run: return
    if len(todo) > a.max_calls: raise SystemExit(f"{len(todo)} calls exceeds --max-calls {a.max_calls}")
    api_key(); add_reasoning(P, a.model)
    lock, counts = threading.Lock(), {}

    def work(item):
        arm, task = item
        try:
            record = A.attempt(config, arms[arm], meta, task)
        except AttributeError as error:               # pipeline.py strips message.content; Mercury can return none (all reasoning)
            record = dict(arm=arm, prompt=arms[arm]["prompt"], max_tokens=arms[arm]["max_tokens"], task_id=task["id"],
                          task=task["task"], k=task.get("k"), source=task["source"], stage="generation", status="illegal",
                          error=f"no answer text (content was empty): {error}"[:200], seconds=None, usage=None)
        with lock:
            with out.open("a") as f: f.write(json.dumps(record, sort_keys=True) + "\n")
            counts[record["status"]] = counts.get(record["status"], 0) + 1
            if sum(counts.values()) % 25 == 0: print("sent", dict(counts), flush=True)

    with cf.ThreadPoolExecutor(a.workers) as pool:
        list(pool.map(work, todo))
    print("finished", dict(counts), flush=True)


def collect(a):
    P, A, M = modules()
    config = A.load_config(str(PIPELINE / "benchmark.ling.json"))
    run_dir = GRIDS[a.grid]
    rows = [json.loads(line) for line in (run_dir / f"{a.model}.jsonl").read_text().splitlines()]
    path = run_dir / "candidates.json"
    cands = json.loads(path.read_text())["candidates"]
    shutil.copy(path, run_dir / f"candidates.before-{a.model}-{time.strftime('%Y%m%d-%H%M%S')}.json")
    have = {c["label"] for c in cands}
    new = []
    for r in rows:                                    # mask_lab_prepare.main's Ling block, appended instead of rebuilt
        label = f"{a.model}:{r['arm']}:{r['task_id']}"
        if r["status"] != "legal" or label in have: continue
        c = dict(label=label, kind=a.model, arm=r["arm"], task_id=r["task_id"], source=r["source"], team=r["team"],
                 paste=M.to_paste(r["team"]))
        try:
            P.validate_team(config, c["team"])
        except ValueError as error:
            c.update(team=None, paste=None, invalid=str(error)[:200])
        new.append(c); have.add(label)
    path.write_text(json.dumps({"candidates": cands + new}, indent=1) + "\n")
    by_arm = {}
    for r in rows: by_arm.setdefault(r["arm"], {}).setdefault(r["status"], 0); by_arm[r["arm"]][r["status"]] += 1
    print(f"{a.grid}: {len(new)} {a.model} candidates appended ({sum(c['paste'] is not None for c in new)} valid); "
          f"answers by status {by_arm}; candidates now {len(cands) + len(new)}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run"); r.add_argument("--versions", default="regmb,baseline"); r.add_argument("--workers", type=int, default=12)
    r.add_argument("--max-calls", type=int, default=600); r.add_argument("--dry-run", action="store_true")
    r.add_argument("--limit", type=int, default=0, help="send only the first N pairs (a smoke test)")
    c = sub.add_parser("collect")
    for q in (r, c):
        q.add_argument("--grid", choices=list(GRIDS), required=True); q.add_argument("--model", choices=list(MODELS), default="ling")
    a = p.parse_args()
    globals()[a.command](a)
