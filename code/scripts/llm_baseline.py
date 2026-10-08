"""Frozen, resumable Ling-only masked-completion pilot. See docs/llm-baseline.md."""
import argparse
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import sqlite3
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
MODEL = "inclusionai/ling-3.0-flash-vl"
FORMAT = "gen9championsvgc2026regmb"
BC_SHA256 = "57f5edcab415cf6ccc1b6231923c8b66d3b1b7249b6b2562b97531471e4ca60b"
GRID = {"stats": [1, 3, 6], "item": [1, 3, 6], "move": [1, 12, 24],
        "pokemon": [1, 3, 6], "ability": [1, 3, 6], "alignment": [1, 3, 6],
        "mixed": [1, 20, 40]}
FIELDS = {"stats": "evs", "item": "item", "ability": "ability", "alignment": "nature"}
STATS = ["HP", "Atk", "Def", "SpA", "SpD", "Spe"]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open('w') as handle:
        handle.write(json.dumps(value, indent=2) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    tmp.replace(path)
    directory = os.open(str(path.parent), os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def paths_for(task):
    if task == "pokemon":
        return [[s] for s in range(6)]
    if task == "move":
        return [[s, "moves", m] for s in range(6) for m in range(4)]
    if task == "mixed":
        return [p for t in FIELDS for p in paths_for(t)] + paths_for("move")
    return [[s, FIELDS[task]] for s in range(6)]


def set_path(team, path, value):
    obj = team
    for key in path[:-1]:
        obj = obj[key]
    obj[path[-1]] = value


def masked(team, paths):
    result = copy.deepcopy(team)
    for path in paths:
        set_path(result, path, None)
    return result


def apply_completion(team, paths, response):
    """Patches only: no reordering, missing fills, duplicate paths, or extra edits."""
    if not isinstance(response, dict) or set(response) != {"fills"}:
        raise ValueError("expected exactly {fills: [{path, value}, ...]}")
    fills = response["fills"]
    if not isinstance(fills, list) or len(fills) != len(paths):
        raise ValueError("wrong number of fills")
    remaining = [list(p) for p in paths]
    result = copy.deepcopy(team)
    for fill in fills:
        if not isinstance(fill, dict) or set(fill) != {"path", "value"}:
            raise ValueError("each fill must contain path and value")
        path = fill["path"]
        if path not in remaining or any(type(k) not in (str, int) for k in path):
            raise ValueError("duplicate or unmasked path")
        remaining.remove(path)
        if len(path)==3 and path[1]=='moves' and not fill['value']:
            raise ValueError('masked moves require a nonempty completion')
        set_path(result, path, fill["value"])
    validate_shape(result)
    for path in paths:
        if len(path)==1 and any(not move for move in result[path[0]]['moves']):
            raise ValueError('whole-slot completions require four nonempty moves')
    if masked(result, paths) != masked(team, paths):
        raise ValueError("unmasked fields changed")
    return result


def validate_shape(team):
    if not isinstance(team, list) or len(team) != 6:
        raise ValueError("team must have six slots")
    for slot in team:
        if not isinstance(slot, dict) or set(slot) != {"species", "item", "ability", "nature", "moves", "evs"}:
            raise ValueError("invalid slot fields")
        if any(not isinstance(slot[k], str) for k in ("species", "item", "ability", "nature")):
            raise ValueError("slot names must be strings")
        if (not isinstance(slot["moves"], list) or len(slot["moves"]) != 4
                or any(not isinstance(m, str) for m in slot["moves"])):
            raise ValueError("four move strings required")
        evs = slot["evs"]
        if (not isinstance(evs, dict) or set(evs) != set(STATS)
                or any(type(v) is not int or not 0 <= v <= 32 for v in evs.values())
                or sum(evs.values()) > 66):
            raise ValueError("Stat Points require 0..32 per stat, total <=66")


def runtime():
    import hps_generate as h
    import matchup_db as db
    h.SHOWDOWN = Path.home() / ".local/share/vgc-pilot-runtime/validator-913da36"
    db.SHOWDOWN = h.SHOWDOWN
    return h, db


def paste(team, h):
    return "\n\n".join(h.slot_to_text(dict(s,moves=[m for m in s['moves'] if m])) for s in team) + "\n"


def random_fill(team, paths, tables, rng, h, validator):
    import numpy as np
    from corpus import norm, NATURES
    for _ in range(1000):
        result = copy.deepcopy(team)
        # Whole slots are sampled against all fixed slots, including later slots.
        whole = {p[0] for p in paths if len(p) == 1}
        bases = {tables["base_of"][norm(s["species"])] for i, s in enumerate(team) if i not in whole}
        items = {norm(s["item"]) for i, s in enumerate(team) if i not in whole}
        for path in paths:
            s = path[0]
            if len(path) == 1:
                value = h.sample_slot(tables, rng, bases, items)
                if value is None:
                    break
            else:
                sp = norm(team[s]["species"])
                field = path[1]
                if field == "evs": value = h.sample_spread(rng)
                elif field == "nature": value = str(rng.choice(NATURES))
                elif field == "moves": value = str(rng.choice(h.TRUE[sp]))
                elif field == "ability": value = str(rng.choice(tables["abil"][sp]))
                else: value = str(rng.choice(tables["items"]))
            set_path(result, path, value)
        else:
            validate_shape(result)
            text = paste(result, h)
            if validator(text) is None:
                assert masked(result, paths) == masked(team, paths)
                return result
    raise RuntimeError("random fill failed after 1000 attempts; no fallback to original")


def opponents_for(manifest, start):
    excluded = manifest.get('exclude_starter_team_ids')
    return [o for o in manifest['opponents'] if not excluded or o['team_id'] != excluded[start]]


def schedule_for(manifest, start):
    ids = [o['team_id'] for o in opponents_for(manifest, start)]
    if not ids: raise ValueError('empty opponent panel')
    random.Random(manifest['seed']).shuffle(ids)
    return (ids * (50 // len(ids) + 1))[:50]


def candidate_start(label):
    if label.startswith('start-'): return int(label.split('-')[1])
    return int(label.split('-')[1][1:])


def prompt(manifest, trial):
    team = manifest["starts"][trial["start"]]
    instructions = (
        "Complete only the null fields of this Pokemon Champions Regulation M-B team to maximize "
        "win rate against the supplied opponent pool. These are Champions Stat Points, NOT Scarlet/Violet EVs: "
        "integer 0..32 each, total <=66; Stat Alignment means nature. Species Clause and Item Clause apply. "
        "Four distinct legal moves per Pokemon; legal species, ability and item combinations. No Terastallization. "
        "Return only JSON: {\"fills\":[{\"path\":[0,\"item\"],\"value\":\"...\"}]}. "
        "Return exactly one fill per requested path, using the identical path. For a whole slot return "
        "{species,item,ability,nature,moves,evs}; evs keys are HP,Atk,Def,SpA,SpD,Spe. "
        "Never change, reorder, or return unmasked fields. No prose or Markdown."
    )
    if manifest.get('preserve_empty_move_slots'):
        instructions += ' Empty-string move slots are intentionally unused: preserve them when unmasked. Fill each masked move position with a nonempty legal move.'
    messages = [{"role": "system", "content": instructions}, {"role": "user", "content": json.dumps({
        "format": FORMAT, "opponents": [o["paste"] for o in opponents_for(manifest, trial['start'])],
        "masked_team": masked(team, trial["paths"]), "requested_paths": trial["paths"]})}]
    if manifest.get('prompt_version') == 'mb-grounded-v2':
        from mb_prompt import strengthen
        messages = strengthen(messages,manifest,trial)
    return messages


def prepare(args):
    import numpy as np
    h, db = runtime()
    destination = args.output / "manifest.json"
    if destination.exists():
        raise ValueError("manifest already exists; use another directory or resume")
    tables = h.build_tables()
    con = sqlite3.connect(f"file:{db.DEFAULT_DB}?mode=ro", uri=True)
    opponents = [dict(team_id=r[0], paste=r[1]) for r in con.execute(
        "SELECT t.team_id,t.paste FROM team t JOIN team_set s USING(team_id) "
        "WHERE s.set_name='top50' AND t.format_id=? AND t.legal=1 ORDER BY t.team_id", (FORMAT,))]
    con.close()
    if not opponents:
        raise ValueError("no legal meta opponents")
    rng = np.random.default_rng(args.seed)
    validator = h.Validator()
    starts, trials, seen = [], [], {db.canon_hash(db.canonical_paste(o["paste"])) for o in opponents}
    try:
        for o in opponents:
            error = validator(o["paste"])
            if error: raise ValueError(f"invalid opponent {o['team_id']}: {error}")
        for _ in range(10000):
            team = h.one_team(tables, rng)
            if team is None: continue
            text = paste(team, h)
            key = db.canon_hash(db.canonical_paste(text))
            if key in seen or validator(text) is not None: continue
            validate_shape(team)
            seen.add(key)
            starts.append(team)
            if len(starts) == args.teams: break
        if len(starts) != args.teams: raise ValueError("could not sample starting teams")
        for i, team in enumerate(starts):
            for task, sizes in GRID.items():
                # Nested masks make k comparisons interpretable within a task.
                order = paths_for(task)
                random.Random(f"{args.seed}:{i}:{task}").shuffle(order)
                for k in sizes:
                    paths = order[:k]
                    control = random_fill(team, paths, tables, rng, h, validator)
                    trials.append(dict(id=f"s{i}-{task}-{k}", start=i, task=task, k=k,
                                       paths=paths, random_team=control))
    finally:
        validator.close()
    manifest = dict(schema=1, model=MODEL, format=FORMAT, seed=args.seed, grid=GRID,
                    starts=starts, trials=trials, opponents=opponents, battles=50,
                    temperature=0.7, max_tokens=4096, reasoning={"enabled": False}, replicates=1,
                    sampler="hierarchical product, legal tables; items limited to corpus vocabulary; not uniform over teams",
                    seed_limit="shared policy RNG seeds; Showdown RNG is not seeded by existing runner",
                    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    save(destination, manifest)
    save(args.output / "example-request.json", {"model": MODEL, "messages": prompt(manifest, trials[0])})
    print(json.dumps(dict(prepared=str(destination), teams=len(starts), trials=len(trials),
                          opponents=len(opponents), planned_battles=(len(starts)+2*len(trials))*50)))


def load_manifest(args):
    m = json.loads((args.output / "manifest.json").read_text())
    if m["model"] != MODEL or m["format"] != FORMAT:
        raise ValueError("model/format mismatch")
    return m


def complete(args):
    m = load_manifest(args)
    key = os.environ.get("OPENROUTER_API_KEY")
    h, _ = runtime()
    validator = h.Validator()
    try:
        for trial_index, trial in enumerate(m["trials"]):
            if trial_index % getattr(args, 'shards', 1) != getattr(args, 'shard', 0): continue
            out = args.output / "completions" / (trial["id"] + ".json")
            payload = dict(model=MODEL, messages=prompt(m, trial), temperature=m["temperature"],
                           max_tokens=m["max_tokens"],
                           reasoning=m.get("reasoning", {"enabled": True}),
                           provider={"order": ["novita"], "allow_fallbacks": False})
            request_hash = digest(payload)
            if out.exists():
                record = json.loads(out.read_text())
                if record["request_hash"] != request_hash:
                    raise ValueError("saved completion request mismatch")
                if not record['valid'] and 'error' not in record:
                    validate_record(record, m, trial, h, validator)
                    save(out, record)
                continue
            pending = out.with_suffix(".pending")
            if pending.exists():
                raise RuntimeError(f"uncertain previous API request: {pending}; inspect before retrying")
            if not key: raise RuntimeError("OPENROUTER_API_KEY is not configured; no requests sent")
            save(pending, dict(request_hash=request_hash, payload=payload, started_at=time.time()))
            req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions",
                  data=json.dumps(payload).encode(), headers={"Authorization": f"Bearer {key}",
                  "Content-Type": "application/json"})
            for attempt in range(3):
                try:
                    with urllib.request.urlopen(req, timeout=180) as res:
                        response = json.load(res)
                    break
                except urllib.error.HTTPError as exc:
                    detail = exc.read().decode()[:2000]
                    save(out.with_suffix(".http-error.json"), dict(status=exc.code, detail=detail))
                    if exc.code == 429 and attempt < 2:
                        print("OpenRouter rate limit; retrying in 20 seconds", flush=True)
                        time.sleep(20)
                        continue
                    pending.unlink()
                    raise RuntimeError(f"OpenRouter HTTP {exc.code}: {detail}") from None
            record = dict(request_hash=request_hash, response=response, valid=False)
            # Persist the paid response before parsing or validating it.
            save(out, record)
            pending.unlink()
            validate_record(record, m, trial, h, validator)
            save(out, record)
            print(trial["id"], "legal" if record["valid"] else "invalid", flush=True)
    finally:
        validator.close()


def validate_record(record, manifest, trial, h, validator):
    """Recover validation after a crash without resending the paid request."""
    try:
        choice = record['response']['choices'][0]
        if choice['finish_reason'] != 'stop': raise ValueError('truncated/non-stop completion')
        team = apply_completion(manifest['starts'][trial['start']], trial['paths'],
                                json.loads(choice['message']['content']))
        error = validator(paste(team, h))
        if error: raise ValueError(error)
        record.update(valid=True, team=team)
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        record['error'] = str(exc)


def report(args):
    m = load_manifest(args)
    records = [json.loads(p.read_text()) for p in (args.output / "completions").glob("*.json")
               if not p.name.endswith(".http-error.json")]
    status = dict(model=MODEL, format=FORMAT, starting_teams=len(m["starts"]),
                  planned_calls=len(m["trials"]), completed_calls=len(records),
                  legal_completions=sum(r["valid"] for r in records),
                  actual_opponents=len(m["opponents"]),
                  reported_cost_usd=sum(r["response"].get("usage", {}).get("cost", 0) or 0 for r in records),
                  interpretation="pilot only; no individual team claim without separate 192-battle confirmation")
    labels = {p.stem: json.loads(p.read_text()) for p in (args.output / "labels").glob("*.json")}
    comparisons = []
    for trial in m["trials"]:
        keys = [f"start-{trial['start']}", "random-" + trial["id"], "llm-" + trial["id"]]
        if not all(k in labels for k in keys): continue
        start, control, llm = [labels[k]["wins"] / labels[k]["battles"] for k in keys]
        comparisons.append(dict(trial=trial["id"], start=trial["start"], task=trial["task"], k=trial["k"],
                                original=start, random=control, llm=llm,
                                delta_original=llm-start, delta_random=llm-control))
    summaries = []
    for task, sizes in m['grid'].items():
        for k in sizes:
            rows = [r for r in comparisons if r["task"] == task and r["k"] == k]
            if not rows:
                summaries.append(dict(task=task, k=k, n=0))
                continue
            summaries.append(dict(task=task, k=k, n=len(rows), **{
                field: sum(r[field] for r in rows)/len(rows)
                for field in ("original", "random", "llm", "delta_original", "delta_random")}))
    status.update(scored_panels=len(labels), paired_comparisons=len(comparisons), summaries=summaries,
                  comparisons=comparisons, complete=(len(records) == len(m["trials"])
                  and len(comparisons) == sum(r["valid"] for r in records)
                  and all(f"start-{i}" in labels for i in range(len(m["starts"])))
                  and all("random-" + t["id"] in labels for t in m["trials"])))
    unique_panels = {r["cache_key"]: r for r in labels.values()}
    status["unique_cached_panels"] = len(unique_panels)
    status["unique_battles"] = sum(r["battles"] for r in unique_panels.values())
    save(args.output / "status.json", status)
    lines = ["# Ling masked-completion pilot", "",
             f"Model: `{MODEL}`. Format: Champions M-B. Behaviour-cloning policy on both sides.", "",
             f"{len(m['starts'])} random legal starting teams; {len(m['trials'])} planned completions; "
             f"{len(records)} returned; {status['legal_completions']} legal; {len(comparisons)} scored comparisons.",
             f"{status['unique_battles']} battles in {len(unique_panels)} distinct cached panels "
             f"({len(labels)} candidate labels; identical candidates reuse their panel).",
             f"Reported API cost: ${status['reported_cost_usd']:.6f} (excludes separate setup smoke).", "",
             "Each candidate plays a frozen 50-battle panel, with repeated opponents as needed. "
             "The schedule matches across methods for each starter. Policy RNG seeds match; Showdown RNG is not seeded. "
             f"This experiment has {len(m['starts'])} starting teams and does not establish held-out generalization. "
             "The supplied meta pool is visible in the prompt, so this measures performance against that pool, "
             "not generalization to unseen threats. Invalid completions are failures, never repaired or silently replaced. "
             "Win-rate comparisons below are conditional on legal completions with complete battle panels.", "",
             "| Task | k | Legal/scored pairs | Original | Random fill | Ling | Δ original (pp) | Δ random (pp) |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in summaries:
        if not row["n"]:
            lines.append(f"| {row['task']} | {row['k']} | 0 | — | — | — | — | — |")
            continue
        lines.append(f"| {row['task']} | {row['k']} | {row['n']} | {row['original']:.1%} | "
                     f"{row['random']:.1%} | {row['llm']:.1%} | {100*row['delta_original']:+.1f} | "
                     f"{100*row['delta_random']:+.1f} |")
    if comparisons:
        lines += ["", "Aggregate paired differences (each starting team weighted equally):"]
        import numpy as np
        for field in ("delta_original", "delta_random"):
            groups = sorted({r["start"] for r in comparisons})
            values = np.array([np.mean([r[field] for r in comparisons if r["start"] == s]) for s in groups])
            draws = np.random.default_rng(m["seed"]).choice(values, (4000, len(values))).mean(axis=1)
            lo, hi = np.quantile(draws, [.025, .975])
            lines += [f"- {field}: {100*values.mean():+.1f} percentage points; exploratory 95% "
                      f"starting-team bootstrap interval [{100*lo:+.1f}, {100*hi:+.1f}]."]
        lines += ["", f"There are {len(m['starts'])} starter clusters; the intervals are exploratory. "
                  "No single candidate is presented as a validated result; that requires a fresh 192-battle panel."]
    errors = [r.get("error", "unknown") for r in records if not r["valid"]]
    if errors:
        from collections import Counter
        lines += ["", "Validation failures:"] + [f"- {n} × {error}" for error, n in Counter(errors).items()]
    (args.output / "report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({k: v for k, v in status.items() if k not in ("summaries", "comparisons")}))


def battle(args, candidates_override=None):
    """Separate exact-panel cache; never reuse aggregate cells with unknown seeds."""
    import asyncio
    import importlib.util
    import torch
    from stable_baselines3 import PPO
    h, db = runtime()
    m = load_manifest(args)
    bench = Path.home() / ".local/share/vgc-pilot-runtime/unpacked/vgc-bench-d79f9532947ac114dce1dda2456a590afcd375b2"
    os.chdir(bench)  # vgc_bench loads data/*.json relative to its checkout.
    sys.path.insert(0, str(bench))
    # Load the project's attribution-aware worker, using the local shard module.
    spec = importlib.util.spec_from_file_location("llm_pair_worker", ROOT / "src/pair_shard.py")
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    torch.set_num_threads(1)
    if hashlib.sha256(db.POLICY_FILE.read_bytes()).hexdigest() != BC_SHA256:
        raise ValueError("checkpoint differs from the verified behaviour-cloning model")
    policy = PPO.load(str(db.POLICY_FILE), device=torch.device("cpu")).policy
    if not policy.choose_on_teampreview:
        raise ValueError("checkpoint does not use learned team preview")
    runtime_files = [db.POLICY_FILE, ROOT / "src/pair_shard.py", ROOT / "src/shard.py"]
    runtime_files += sorted((bench / "vgc_bench").rglob("*.py"))
    runtime_files += sorted((bench / "data").rglob("*.json"))
    runtime_files += sorted((h.SHOWDOWN / "dist").rglob("*.js"))
    fingerprints = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in runtime_files}
    config = dict(format=FORMAT, policy="behaviour-cloning both sides", fingerprints=fingerprints,
                  concurrency=1, seed=m["seed"], battles=50, opponents=m["opponents"],
                  seed_limit=m["seed_limit"])
    if m.get('exclude_starter_team_ids'):
        config['exclude_starter_team_ids'] = m['exclude_starter_team_ids']
    con = db.connect()
    con.execute("CREATE TABLE IF NOT EXISTS llm_completion_panel "
                "(cache_key TEXT PRIMARY KEY, config_json TEXT NOT NULL, result_json TEXT NOT NULL)")
    con.commit()
    pid = db.get_policy(con, simulator="showdown-content:" + digest(fingerprints),
                        note="Ling baseline; BC both sides; exact-panel cache stored separately")
    candidates = [(f"start-{i}", t) for i, t in enumerate(m["starts"])]
    for trial in m["trials"]:
        candidates.append(("random-" + trial["id"], trial["random_team"]))
        path = args.output / "completions" / (trial["id"] + ".json")
        if path.exists():
            row = json.loads(path.read_text())
            if row["valid"]: candidates.append(("llm-" + trial["id"], row["team"]))
    if candidates_override is not None:
        candidates = candidates_override
    # Own an isolated server built from the same checkout used for validation.
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    log = (args.output / "showdown.log").open("a")
    server = subprocess.Popen(["node", "pokemon-showdown", "start", str(port), "--no-security"],
                              cwd=h.SHOWDOWN, stdout=log, stderr=subprocess.STDOUT)
    validator = h.Validator()
    try:
        for _ in range(100):
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=.2): break
            except OSError:
                if server.poll() is not None: raise RuntimeError("Showdown failed; see showdown.log")
                time.sleep(.1)
        else: raise RuntimeError("Showdown startup timeout")
        opp_ids = [o["team_id"] for o in m["opponents"]]
        for o in m["opponents"]:
            stored = con.execute("SELECT paste FROM team WHERE team_id=?", (o["team_id"],)).fetchone()
            if not stored or stored[0] != o["paste"]: raise ValueError("opponent database changed")
        schedule = schedule_for(m, 0)
        panels = {'schedules_by_start': {str(i):schedule_for(m,i) for i in range(len(m['starts']))}} if m.get('exclude_starter_team_ids') else {'schedule':schedule}
        save(args.output / "battle-config.json", dict(config, policy_id=pid, **panels))
        for label, team in candidates:
            schedule = schedule_for(m, candidate_start(label))
            text = paste(team, h)
            error = validator(text)
            if error: raise ValueError(error)
            tid = db.add_team(con, text, "llm-completion:" + args.output.name, validator)[0]
            key = digest(dict(config=config, candidate=db.canonical_paste(text), schedule=schedule))
            cached = con.execute("SELECT result_json FROM llm_completion_panel WHERE cache_key=?", (key,)).fetchone()
            if cached:
                result = json.loads(cached[0])
            else:
                results, unattributed = [], 0
                # jobs_for splits preview collisions while preserving common panels.
                from collections import Counter
                jobs = db.jobs_for(con, [(tid, oid, n) for oid, n in Counter(schedule).items()])
                for job in jobs:
                    cols = sorted(set(job["schedule"]))
                    info = {c: con.execute("SELECT species_key,base_key FROM team WHERE team_id=?", (c,)).fetchone() for c in cols}
                    job_spec = dict(row=tid, row_file=db.paste_file(con, tid),
                                    schedule=[db.paste_file(con, c) for c in job["schedule"]],
                                    key2col={info[c][0]: c for c in cols}, base2col={info[c][1]: c for c in cols})
                    raw = asyncio.run(asyncio.wait_for(worker.run_job(policy, job_spec, 1, port, m["seed"]), timeout=600))
                    unattributed += raw["unattributed"]
                    results += [(tid, int(c), *counts) for c, counts in raw["by_col"].items()]
                n = sum(w+l+t for _, _, w,l,t in results)
                if unattributed or n != 50: raise RuntimeError(f"incomplete panel: {n}, unattributed {unattributed}")
                result = dict(team_id=tid, wins=sum(r[2] for r in results), battles=n,
                              results=results, cache_key=key)
                # The aggregate matrix excludes diagonal cells. A completion may
                # equal another pool team; retain its mirror outcomes in the exact
                # panel without changing the shared opponent schedule.
                off_diagonal = [r for r in results if r[0] != r[1]]
                if off_diagonal:
                    db.record(con, pid, off_diagonal, seed=m["seed"], note="llm-completion:" + args.output.name)
                with con:
                    con.execute("INSERT INTO llm_completion_panel VALUES (?,?,?)",
                                (key, json.dumps(config), json.dumps(result)))
            save(args.output / "labels" / (label + ".json"), result)
            print(label, f"{result['wins']}/50", flush=True)
    finally:
        validator.close()
        server.terminate()
        try: server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill(); server.wait()
        log.close()
        con.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "complete", "battle", "report"])
    parser.add_argument("--output", type=Path, default=ROOT / "results/llm-baseline-instant-pilot")
    parser.add_argument("--teams", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20261008)
    parser.add_argument('--shards', type=int, default=1)
    parser.add_argument('--shard', type=int, default=0)
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.teams < 1: parser.error("--teams must be positive")
    if not 0 <= args.shard < args.shards: parser.error('invalid shard')
    args.output.mkdir(parents=True, exist_ok=True)
    lock_name = '.' + args.command + (f'-{args.shard}-of-{args.shards}' if args.command=='complete' else '') + '.lock'
    with (args.output / lock_name).open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        globals()[args.command](args)


if __name__ == "__main__":
    main()
