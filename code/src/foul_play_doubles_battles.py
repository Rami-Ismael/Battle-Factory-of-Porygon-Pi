"""Head-to-head battles between battle policies in VGC Reg M-B doubles (code/docs/foul-play-doubles.md).

Policies:
  foulplay    Foul Play rebuilt for doubles (src/foul_play_doubles.cjs): sampled opponent sets, then
              decoupled-UCT search with a heuristic leaf score on each sample, visit-weighted vote
  montecarlo  flat Monte Carlo: random first choices, random play to the end, best mean per slot
  random      a uniformly random legal choice per slot
  bc          the behaviour-cloning policy (/tmp/bc_100.zip), the project's battle policy, played
              through poke-env exactly as the fast evaluator plays it (stochastic, actor only)

A matchup X vs Y plays each ordered pair of top-50 meta teams (A, B) twice: X with A against Y
with B, then Y with A against X with B, with both teams' slots shuffled the same way in both games.
Team strength and side cancel, so X's share of points (win 1, tie 0.5) above 0.5 is policy strength.

Run with the pilot environment, which has poke-env and the BC checkpoint:
  /tmp/vgc-pilot/.venv/bin/python src/foul_play_doubles_battles.py spreads
  /tmp/vgc-pilot/.venv/bin/python src/foul_play_doubles_battles.py match foulplay bc --pairs 100 --out DIR
"""
import argparse
import asyncio
import hashlib
import json
import math
import os
import random
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
RUNNER = HERE / "foul_play_doubles_runner.cjs"
SPREADS = REPO / "data" / "foul-play-doubles-spreads.json"
DB = Path.home() / "vgc-data" / "matchup_regmb.sqlite"
POOL = Path("/tmp/vgc-pilot/vgc-bench/teams/reg_mb")
RUNTIME = Path(os.environ.get("SHOWDOWN_RUNTIME", Path.home() / "vgc-data/battle-evaluations/runtime"))
LAB = (Path.home() / "Documents/yakumsi-vault/Personal Project"
       / "Using Advance method search in the whole search of possible vgc pokemon format to find the right counter to a pokemon team"
       / "battle_speed_lab")
BENCH = Path.home() / ".local/share/vgc-pilot-runtime/unpacked/vgc-bench-d79f9532947ac114dce1dda2456a590afcd375b2"
FORMAT = "gen9championsvgc2026regmb"
POLICIES = ("foulplay", "foulplay_det", "foulplay_doubles", "foulplay_joint", "foulplay_hybrid",
            "uct_rollout", "montecarlo", "random", "bc", "bc_det",
            "rust_foulplay", "rust_foulplay_det", "rust_foulplay_doubles", "rust_foulplay_joint",
            "rust_foulplay_hybrid", "rust_uct_rollout", "rust_montecarlo")
BC_POLICIES = ("bc", "bc_det")
NAMES = {"p1": "Candidate", "p2": "Opponent"}


# ---------------------------------------------------------------- data
def top50():
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    rows = con.execute("SELECT t.team_id, t.paste FROM team t JOIN team_set s ON s.team_id = t.team_id "
                       "WHERE s.set_name = 'top50' ORDER BY t.team_id").fetchall()
    con.close()
    return rows


def species_key(sets):
    return ",".join(sorted(s["species"] for s in sets))


def import_pastes(pastes):
    """Showdown's own paste parser, so species, natures and Stat Points read as the simulator reads them."""
    script = ("const {Teams}=require(require('path').join(process.argv[1],'dist/sim'));"
              "let d='';process.stdin.on('data',c=>d+=c).on('end',()=>{"
              "process.stdout.write(JSON.stringify(JSON.parse(d).map(p=>Teams.import(p)||[])))})")
    out = subprocess.run(["node", "-e", script, str(RUNTIME)], input=json.dumps(pastes), capture_output=True,
                         text=True, check=True)
    return json.loads(out.stdout)


def build_spreads(out=SPREADS):
    """Foul Play's team dataset for this format: the Stat Points each species carries on public
    VGCPastes Reg M-B teams, by nature and item, with counts. The 49 top-50 meta teams, the
    opponents of every matchup here, are left out so a foe's exact spread is never looked up."""
    pool = [p.read_text() for p in sorted(POOL.glob("*.txt"))]
    held_out = {species_key(t) for t in import_pastes([paste for _, paste in top50()])}
    table, kept = {}, 0
    for team in import_pastes(pool):
        if len(team) != 6 or species_key(team) in held_out:
            continue
        kept += 1
        for s in team:
            key = "".join(c for c in s["species"].lower() if c.isalnum())
            evs = {k: s.get("evs", {}).get(k, 0) for k in ("hp", "atk", "def", "spa", "spd", "spe")}
            row = {"nature": s.get("nature") or "Serious", "item": s.get("item", ""), "evs": evs}
            rows = table.setdefault(key, [])
            match = next((r for r in rows if r["nature"] == row["nature"] and r["item"] == row["item"] and r["evs"] == evs), None)
            if match:
                match["count"] += 1
            else:
                rows.append({**row, "count": 1})
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"_source": f"{kept} VGCPastes Reg M-B teams from {POOL}, top-50 meta teams held out",
                               **dict(sorted(table.items()))}, indent=0) + "\n")
    print(f"{out}: {len(table)} species from {kept} teams ({len(pool) - kept} skipped)")
    return out


def shuffle_paste(paste, rng):
    blocks = [b.strip() for b in paste.strip().split("\n\n") if b.strip()]
    rng.shuffle(blocks)
    return "\n\n".join(blocks) + "\n"


def schedule(x, y, pairs, seed):
    """2 * pairs battles: each ordered pair of distinct top-50 teams with the policies swapped."""
    teams = top50()
    rng = random.Random(seed)
    chosen = set()
    while len(chosen) < pairs:
        a, b = rng.sample(range(len(teams)), 2)
        chosen.add((a, b))
    ordered = sorted(chosen)
    rng.shuffle(ordered)
    jobs = []
    for k, (a, b) in enumerate(ordered):
        order = random.Random(seed * 1000003 + k)
        pa, pb = shuffle_paste(teams[a][1], order), shuffle_paste(teams[b][1], order)
        for swap in (0, 1):
            p1, p2 = (x, y) if swap == 0 else (y, x)
            jobs.append(dict(id=f"{k}-{swap}", pair=k, a_id=teams[a][0], b_id=teams[b][0], a=pa, b=pb,
                             seed=seed * 7919 + 2 * k + swap, p1=p1, p2=p2, x_side="p1" if swap == 0 else "p2"))
    return jobs


# ---------------------------------------------------------------- the BC side
def load_bc():
    """The production evaluator's BC policy and poke-env player (battle_speed_lab/worker.py)."""
    sys.path.insert(0, str(LAB))
    cwd = os.getcwd()
    os.chdir(BENCH)
    import worker as w  # the vault lab module, importable only from its own folder
    os.chdir(cwd)
    w.torch.set_num_threads(1)
    policy = w.PPO.load("/tmp/bc_100.zip", device=w.torch.device("cpu")).policy
    policy.set_training_mode(False)
    return w, policy


class Transport:
    """poke-env's websocket client, rerouted to a runner process (as in battle_speed_lab/direct_bc.py)."""
    def __init__(self, original, send, identity, side):
        self.original, self.send, self.identity, self.side = original, send, identity, side

    def __getattr__(self, name):
        return getattr(self.original, name)

    async def send_message(self, message, room=None):  # noqa: ARG002  (poke-env's signature)
        await self.send({"kind": "choice", "id": self.identity, "side": self.side, "choice": message})


async def play(jobs, args, out_dir, w=None, policy=None):  # noqa: C901
    """Every job on `args.workers` runner processes; BC sides are played here through poke-env."""
    base_cfg = {"batcher": "original", "threads": 1, "concurrency": args.concurrency * args.workers, "shared": True,
                "encoder": "fast", "actor_only": True, "inference_mode": True, "policy_seed": args.seed}
    metrics = {"embedding_seconds": 0., "mask_seconds": 0., "forward_seconds": 0., "stack_seconds": 0.,
               "queue_seconds": [], "batch_sizes": []}
    services = {}
    if w:
        needed = {j[s]["policy"] for j in jobs for s in ("p1", "p2") if j[s]["policy"] in BC_POLICIES}
        if "bc" in needed:
            services["bc"] = w.Inference(policy, {**base_cfg, "deterministic": False, "per_battle_rng": True}, metrics)
        if "bc_det" in needed:
            services["bc_det"] = w.Inference(policy, {**base_cfg, "deterministic": True, "per_battle_rng": False}, metrics)
    env = {**os.environ, "FOUL_PLAY_SPREADS": str(SPREADS), "SHOWDOWN_RUNTIME": str(RUNTIME)}
    log = (out_dir / "runner.log").open("a")
    procs, channels, results = [], {}, {}
    done = {j["id"]: asyncio.get_running_loop().create_future() for j in jobs}
    for _ in range(args.workers):
        proc = await asyncio.create_subprocess_exec("node", str(RUNNER), stdin=asyncio.subprocess.PIPE,
                                                    stdout=asyncio.subprocess.PIPE, stderr=log, env=env,
                                                    limit=2 ** 26)
        ready = json.loads(await proc.stdout.readline())
        if ready.get("kind") != "ready":
            raise RuntimeError("runner did not start")
        procs.append(proc)

    def sender(proc):
        async def send(message):
            proc.stdin.write((json.dumps(message) + "\n").encode())
            await proc.stdin.drain()
        return send

    async def read(proc):
        while True:
            line = await proc.stdout.readline()
            if not line:
                for f in done.values():
                    if not f.done():
                        f.set_exception(RuntimeError("runner exited"))
                return
            item = json.loads(line)
            if item["kind"] == "error":
                if not done[item["id"]].done():
                    done[item["id"]].set_exception(RuntimeError(item["error"][:2000]))
            elif item["kind"] == "result":
                results[item["id"]] = item
                if not done[item["id"]].done():
                    done[item["id"]].set_result(item)
            else:
                await channels[(item["id"], item["side"])].put(item)

    readers = [asyncio.create_task(read(p)) for p in procs]
    per_proc = [asyncio.Semaphore(args.concurrency) for _ in procs]
    sink = (out_dir / "battles.jsonl").open("a")

    async def battle(job, index):
        k = index % len(procs)
        async with per_proc[k]:
            send = sender(procs[k])
            consumers, players = [], {}
            for side in ("p1", "p2"):
                pol = job[side]["policy"]
                if pol not in BC_POLICIES:
                    continue
                q = channels[(job["id"], side)] = asyncio.Queue()
                paste = job["a"] if side == "p1" else job["b"]
                player = w.InstrumentedPlayer(services[pol], metrics, policy=policy,
                    account_configuration=w.AccountConfiguration(NAMES[side], None), battle_format=FORMAT,
                    max_concurrent_battles=1, start_listening=False, accept_open_team_sheet=True,
                    deterministic=(pol == "bc_det"), team=w.SequenceTeams([paste]), log_level=50)
                player.ps_client = Transport(player.ps_client, send, job["id"], side)
                players[side] = player
                tag = f"battle-{FORMAT}-{job['id']}"

                async def consume(player=player, q=q, tag=tag):
                    while True:
                        item = await q.get()
                        if item["kind"] == "end":
                            break
                        lines = [[f">{tag}"]] + [line.split("|") for line in item["chunk"].split("\n")]
                        await player._handle_battle_message(lines)
                consumers.append(asyncio.create_task(consume()))
            message = {k2: job[k2] for k2 in ("id", "a", "b", "seed", "p1", "p2")}
            await send({"kind": "start", **message})
            try:
                result = await asyncio.wait_for(done[job["id"]], args.timeout)
                await asyncio.wait_for(asyncio.gather(*consumers), 30)
            finally:
                for c in consumers:
                    c.cancel()
            row = {**{k2: job[k2] for k2 in ("id", "pair", "a_id", "b_id", "seed", "x_side")},
                   "p1": job["p1"]["policy"], "p2": job["p2"]["policy"], "winner": result["winner"],
                   "turns": result["turns"], "seconds": result["ms"] / 1000, "agents": result["stats"]}
            sink.write(json.dumps(row) + "\n")
            sink.flush()
            return row

    start = time.perf_counter()
    rows = []
    try:
        tasks = [asyncio.create_task(battle(j, i)) for i, j in enumerate(jobs)]
        for t in asyncio.as_completed(tasks):
            rows.append(await t)
            if len(rows) % 10 == 0 or len(rows) == len(jobs):
                print(f"{len(rows)}/{len(jobs)} battles, {time.perf_counter() - start:.0f} s", flush=True)
    finally:
        for r in readers:
            r.cancel()
        for p in procs:
            p.stdin.close()
            try:
                await asyncio.wait_for(p.wait(), 10)
            except asyncio.TimeoutError:
                p.kill()
        for svc in services.values():
            await svc.close()
        sink.close()
        log.close()
    return rows, time.perf_counter() - start


# ---------------------------------------------------------------- scoring
def wilson(points, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    p = points / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def summarize(rows, x, y):
    """X's share of points, and the same share with each pair's two games as one unit (paired)."""
    def score(r):
        if r["winner"] is None:
            return 0.5
        return 1.0 if r["winner"] == r["x_side"] else 0.0
    points = sum(score(r) for r in rows)
    n = len(rows)
    pairs = {}
    for r in rows:
        pairs.setdefault(r["pair"], []).append(score(r))
    pair_means = [sum(v) / len(v) for v in pairs.values() if len(v) == 2]
    m = sum(pair_means) / len(pair_means) if pair_means else float("nan")
    sd = math.sqrt(sum((v - m) ** 2 for v in pair_means) / (len(pair_means) - 1)) if len(pair_means) > 1 else float("nan")
    se = sd / math.sqrt(len(pair_means)) if pair_means else float("nan")
    lo, hi = wilson(points, n)
    return {"x": x, "y": y, "battles": n, "x_points": points, "x_share": points / n if n else None,
            "wilson95": [lo, hi], "complete_pairs": len(pair_means), "paired_mean": m,
            "paired95": [m - 1.96 * se, m + 1.96 * se],
            "ties": sum(r["winner"] is None for r in rows),
            "mean_turns": sum(r["turns"] for r in rows) / n if n else None}


def git_head():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    except OSError:
        return None


def noise_schedule(policy_spec, pairs, repeats, seed, shuffle=True):
    """Self-play schedule: `pairs` distinct top-50 team pairs, `repeats` games each (sides alternating)."""
    teams = top50()
    rng = random.Random(seed)
    chosen = set()
    while len(chosen) < pairs:
        a, b = rng.sample(range(len(teams)), 2)
        if a > b:
            a, b = b, a
        chosen.add((a, b))
    ordered = sorted(chosen)
    jobs = []
    for k, (a, b) in enumerate(ordered):
        for r in range(repeats):
            pa, pb = teams[a][1], teams[b][1]
            if shuffle:
                sh = random.Random(seed * 1000003 + k * 97 + r)
                pa, pb = shuffle_paste(pa, sh), shuffle_paste(pb, sh)
            swap = r % 2
            a_paste, b_paste = (pa, pb) if swap == 0 else (pb, pa)
            jobs.append(dict(id=f"n{k}-r{r}", pair=k, a_id=teams[a][0], b_id=teams[b][0], a=a_paste, b=b_paste,
                             seed=seed * 7919 + k * 997 + r, p1=policy_spec, p2=policy_spec,
                             x_side="p1" if swap == 0 else "p2"))
    return jobs


def noise_metrics(pair_scores):
    """Unbiased within-pair battle variance E[p(1-p)], true matchup SD, ICC, and 49-game label SE."""
    s2s, ps, ns = [], [], []
    for vals in pair_scores.values():
        g = len(vals)
        if g < 2:
            continue
        p = sum(vals) / g
        s2 = (g / (g - 1)) * p * (1 - p)
        s2s.append(s2)
        ps.append(p)
        ns.append(g)
    if len(ps) < 2:
        return {}
    mean_s2 = sum(s2s) / len(s2s)
    mean_p = sum(ps) / len(ps)
    var_phat = sum((p - mean_p) ** 2 for p in ps) / (len(ps) - 1)
    mean_samp_var = sum(s2 / n for s2, n in zip(s2s, ns)) / len(ns)
    true_var = max(0.0, var_phat - mean_samp_var)
    icc = true_var / (true_var + mean_s2) if (true_var + mean_s2) > 0 else 0.0
    decisive = sum(1 for p in ps if p <= 0.125 or p >= 0.875) / len(ps)
    return {"within_pair_var": mean_s2, "single_game_sd": math.sqrt(mean_s2),
            "true_matchup_sd": math.sqrt(true_var), "icc": icc,
            "decisive_share": decisive, "label_se_49": math.sqrt(mean_s2 / 49),
            "label_se_24": math.sqrt(mean_s2 / 24)}


def _split_half_corr(pairs):
    h1, h2 = [], []
    for vals in pairs.values():
        if len(vals) >= 4:
            mid = len(vals) // 2
            h1.append(sum(vals[:mid]) / mid)
            h2.append(sum(vals[mid:2 * mid]) / mid)
    if len(h1) < 3:
        return None
    m1, m2 = sum(h1) / len(h1), sum(h2) / len(h2)
    num = sum((a - m1) * (b - m2) for a, b in zip(h1, h2))
    den = math.sqrt(sum((a - m1) ** 2 for a in h1) * sum((b - m2) ** 2 for b in h2))
    return num / den if den > 0 else 0.0


def _bootstrap_noise_ci(pairs):
    keys = sorted(pairs.keys())
    brng = random.Random(42)
    boots = {k: [] for k in ("within_pair_var", "single_game_sd", "true_matchup_sd", "icc", "label_se_49")}
    for _ in range(1000):
        bm = noise_metrics({i: pairs[brng.choice(keys)] for i in range(len(keys))})
        for k in boots:
            if k in bm:
                boots[k].append(bm[k])
    return {k: [sorted(v)[int(0.025 * len(v))], sorted(v)[int(0.975 * len(v))]] for k, v in boots.items() if v}


def summarize_noise(rows, policy_name):
    pairs = {}
    for r in rows:
        sc = 0.5 if r["winner"] is None else (1.0 if r["winner"] == r["x_side"] else 0.0)
        pairs.setdefault(r["pair"], []).append(sc)
    all_agents = [a for r in rows for a in (r.get("agents") or {}).values()]
    decs = sum(a.get("decisions", 0) for a in all_agents)
    n = len(rows)
    return {"policy": policy_name, "battles": n, "pairs": len(pairs),
            "repeats_per_pair": n / len(pairs) if pairs else 0,
            **noise_metrics(pairs), "split_half_r": _split_half_corr(pairs), "ci95": _bootstrap_noise_ci(pairs),
            "ties": sum(r["winner"] is None for r in rows),
            "mean_turns": sum(r["turns"] for r in rows) / n if n else None,
            "mean_seconds": sum(r["seconds"] for r in rows) / n if n else None,
            "iterations_per_decision": sum(a.get("iterations", 0) for a in all_agents) / decs if decs else 0.0,
            "fallbacks": sum(a.get("fallbacks", 0) for a in all_agents)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("spreads", help="build the Stat Points dataset from the VGCPastes pool")
    m = sub.add_parser("match", help="X vs Y on paired top-50 team schedules")
    m.add_argument("x", choices=POLICIES)
    m.add_argument("y", choices=POLICIES)
    m.add_argument("--pairs", type=int, default=100)
    m.add_argument("--seed", type=int, default=20261123)
    m.add_argument("--workers", type=int, default=8)
    m.add_argument("--concurrency", type=int, default=2, help="battles at once per runner process")
    m.add_argument("--samples", type=int, default=4, help="sampled battles per decision (Foul Play's parallelism)")
    m.add_argument("--ms", type=int, default=250, help="search time per sampled battle, ms")
    m.add_argument("--preview-ms", type=int, default=1000, help="search time per sample at team preview, ms")
    m.add_argument("--iterations", type=int, default=0, help="fixed iterations per sample (0 = time budget)")
    m.add_argument("--timeout", type=int, default=1800)
    m.add_argument("--out", type=Path, required=True)
    n = sub.add_parser("noise", help="self-play battle noise measurement on fixed top-50 team pairs")
    n.add_argument("policy", choices=POLICIES)
    n.add_argument("--pairs", type=int, default=24)
    n.add_argument("--repeats", type=int, default=8)
    n.add_argument("--seed", type=int, default=20261123)
    n.add_argument("--no-shuffle", action="store_true", help="keep listed team slot order")
    n.add_argument("--workers", type=int, default=8)
    n.add_argument("--concurrency", type=int, default=2)
    n.add_argument("--samples", type=int, default=1)
    n.add_argument("--ms", type=int, default=1000)
    n.add_argument("--preview-ms", type=int, default=1000)
    n.add_argument("--iterations", type=int, default=0)
    n.add_argument("--timeout", type=int, default=1800)
    n.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.command == "spreads":
        build_spreads()
        return
    if not SPREADS.exists():
        build_spreads()
    spec = {"samples": args.samples, "ms": args.ms, "previewMs": args.preview_ms, "iterations": args.iterations}
    if args.command == "noise":
        jobs = noise_schedule({"policy": args.policy, **spec}, args.pairs, args.repeats, args.seed,
                              shuffle=not args.no_shuffle)
        args.out.mkdir(parents=True, exist_ok=True)
        done_ids = set()
        if (args.out / "battles.jsonl").exists():
            done_ids = {json.loads(line)["id"] for line in (args.out / "battles.jsonl").read_text().splitlines() if line}
        todo = [j for j in jobs if j["id"] not in done_ids]
        config = {"command": "noise", "policy": args.policy, "pairs": args.pairs, "repeats": args.repeats,
                  "seed": args.seed, "shuffle": not args.no_shuffle, **spec, "workers": args.workers,
                  "concurrency": args.concurrency, "format": FORMAT, "git_head": git_head()}
        (args.out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        print(f"{len(todo)} of {len(jobs)} noise battles to play: {args.policy} self-play", flush=True)
        w = policy = None
        if args.policy in BC_POLICIES:
            w, policy = load_bc()
            _, seconds = asyncio.run_coroutine_threadsafe(play(todo, args, args.out, w, policy), w.POKE_LOOP).result()
        else:
            _, seconds = asyncio.run(play(todo, args, args.out))
        rows = [json.loads(line) for line in (args.out / "battles.jsonl").read_text().splitlines() if line]
        summary = {**summarize_noise(rows, args.policy), "seconds_this_run": seconds, "config": config}
        (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps({k: summary[k] for k in ("policy", "battles", "within_pair_var", "single_game_sd",
                                                  "true_matchup_sd", "icc", "label_se_49", "split_half_r", "fallbacks")}),
              flush=True)
        os._exit(0)
    jobs = schedule({"policy": args.x, **spec}, {"policy": args.y, **spec}, args.pairs, args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    done_ids = set()
    if (args.out / "battles.jsonl").exists():   # resume: keep finished battles, play the rest
        done_ids = {json.loads(line)["id"] for line in (args.out / "battles.jsonl").read_text().splitlines() if line}
    todo = [j for j in jobs if j["id"] not in done_ids]
    config = {"x": args.x, "y": args.y, "pairs": args.pairs, "seed": args.seed, "samples": args.samples,
              "ms": args.ms, "preview_ms": args.preview_ms, "iterations": args.iterations,
              "workers": args.workers, "concurrency": args.concurrency,
              "format": FORMAT, "git_head": git_head(), "foul_play_source": "pmariglia/foul-play 8f541f8, poke-engine f7b147f",
              "bc_checkpoint": "/tmp/bc_100.zip", "opponents": "top-50 meta teams, slots shuffled per pair",
              "source_sha256": {f.name: hashlib.sha256(f.read_bytes()).hexdigest()
                                for f in (HERE / "foul_play_doubles.cjs", RUNNER, Path(__file__), SPREADS)}}
    (args.out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(f"{len(todo)} of {len(jobs)} battles to play: {args.x} vs {args.y}", flush=True)
    w = policy = None
    if args.x in BC_POLICIES or args.y in BC_POLICIES:
        w, policy = load_bc()
        _, seconds = asyncio.run_coroutine_threadsafe(play(todo, args, args.out, w, policy), w.POKE_LOOP).result()
    else:
        _, seconds = asyncio.run(play(todo, args, args.out))
    rows = [json.loads(line) for line in (args.out / "battles.jsonl").read_text().splitlines() if line]
    summary = {**summarize(rows, args.x, args.y), "seconds_this_run": seconds, "config": config}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in ("x", "y", "battles", "x_share", "wilson95", "paired95", "ties")}), flush=True)
    os._exit(0)   # poke-env's background loop thread does not exit on its own


if __name__ == "__main__":
    main()
