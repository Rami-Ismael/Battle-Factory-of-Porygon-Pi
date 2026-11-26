"""Label matchup-database cells with our Rust Regulation M-B doubles search engine (issue #418).

A drop-in battle runner for matchup_db.ensure and matchup_db.score_pastes. Both sides play the
same Rust search policy (code/rust/poke-engine-doubles, driven by src/foul_play_doubles_runner.cjs)
on the pinned Showdown runtime, so a cell's win rate is a property of the two teams, not of a
behaviour-cloning checkpoint. The labeler (#417) is Foul Play deterministic argmax
(`rust_foulplay_det`, 30,000 iterations, 1 sample); UCT with random playouts (`rust_uct_rollout`)
is the other finalist from PR #415 and stays available.

Cells land under their own policy row (get_rust_policy): the policy name, iterations and samples,
and SHA-256 hashes of the engine binary, regmb-dex.json, the Stat Points dataset and the agent
JavaScript, with the Showdown runtime's content hash as the simulator. The behaviour-cloning rows
(policy_id 1-8) never receive these battles, and matchup_db refuses to give a Rust policy to any
other runner.

Game g of a cell (lo, hi) is fixed by (seed, lo, hi, g): lo plays p1 on even g and p2 on odd g,
both teams' slots are shuffled with a generator seeded from that tuple, and the Showdown and search
seeds come from the same hash. g is reserved per cell in table rust_next_game before battles start,
so a top-up plays new games and an index is never reused, even after a failed or killed game.

Finished battles reach the database every `flush_every` battles (matchup_db.record), so a killed
job loses at most that many; rerunning ensure plays only the deficit.

    python src/matchup_db.py --labeler rust_foulplay_det ensure --rows 12,13 --cols top50 --per-cell 2
"""
import asyncio, hashlib, json, os, random, shutil, sys, time, uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
CODE = HERE.parent
sys.path.insert(0, str(HERE))
from slot_order import MODE as SLOT_ORDER, POLICY_TAG, shuffle_paste

POLICIES = ("rust_foulplay_det", "rust_uct_rollout")
DEFAULT_POLICY = "rust_foulplay_det"            # #417: the labeler
BIN = Path(os.environ.get("RUST_ENGINE_BIN", CODE / "rust/poke-engine-doubles/target/release/poke-engine-doubles"))
DEX = Path(os.environ.get("REGMB_DEX", CODE / "data/regmb-dex.json"))
SPREADS = CODE / "data/foul-play-doubles-spreads.json"
RUNNER_JS = HERE / "foul_play_doubles_runner.cjs"
AGENT_JS = (HERE / "foul_play_doubles.cjs", RUNNER_JS, HERE / "rust_engine_client.cjs")
RUNTIME = Path(os.environ.get("SHOWDOWN_RUNTIME", Path.home() / "vgc-data/battle-evaluations/runtime"))
EVIDENCE = Path(os.environ.get("RUST_MATCHUP_EVIDENCE", Path.home() / "vgc-data/rust-matchup"))


class RustRunnerError(RuntimeError):
    """A battle failed (engine missing or crashed, runner exited, timeout). Finished battles are recorded first."""


# ---------------------------------------------------------------- provenance
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""): h.update(chunk)
    return h.hexdigest()


def sha256_files(paths, root=None):
    h = hashlib.sha256()
    for p in paths:
        h.update(str(Path(p).relative_to(root) if root else Path(p).name).encode() + b"\0")
        h.update(sha256_file(p).encode())
    return h.hexdigest()


RUNTIME_PARTS = ("dist/sim", "dist/data", "dist/config", "dist/lib")   # dist/config/formats.js holds the Reg M-B rules


def runtime_hash(runtime=None):
    """Content hash of the Showdown runtime that referees: package.json and every file under RUNTIME_PARTS."""
    runtime = Path(runtime or RUNTIME)
    dist = sorted(p for sub in RUNTIME_PARTS for p in (runtime / sub).rglob("*") if p.is_file())
    return sha256_files([runtime / "package.json", *dist], root=runtime)


def provenance(bin=None, dex=None, spreads=None, runtime=None):
    bin, dex, spreads, runtime = bin or BIN, dex or DEX, spreads or SPREADS, runtime or RUNTIME
    for p, what in ((bin, "Rust engine binary (cargo build --release in code/rust/poke-engine-doubles)"),
                    (dex, "regmb-dex.json"), (spreads, "Stat Points dataset"), (runtime, "Showdown runtime")):
        if not Path(p).exists(): raise FileNotFoundError(f"{what} not found: {p}")
    return {"engine": sha256_file(bin), "dex": sha256_file(dex), "spreads": sha256_file(spreads),
            "agent": sha256_files(AGENT_JS), "runtime": runtime_hash(runtime)}


def policy_strings(policy_name, iterations, samples, prov):
    row = (f"{policy_name}:iterations={iterations}:samples={samples}:engine={prov['engine'][:16]}"
           f":dex={prov['dex'][:16]}:spreads={prov['spreads'][:16]}:agent={prov['agent'][:16]}{POLICY_TAG}")
    return row, f"showdown-runtime:{prov['runtime']}"


def get_rust_policy(con, policy_name=DEFAULT_POLICY, iterations=30000, samples=1, slot_order=SLOT_ORDER, prov=None):
    """The policy_id of one Rust search configuration, registered on first use. Distinct from every
    behaviour-cloning row: row_policy names the search and hashes its inputs. Raises on a slot order
    other than shuffled, the only order matchup cells take (src/slot_order.py)."""
    if policy_name not in POLICIES: raise ValueError(f"policy must be one of {POLICIES}, not {policy_name!r}")
    if slot_order != "shuffled":
        raise ValueError(f"Rust policies are battled with shuffled slots only, not {slot_order!r}")
    if iterations <= 0 or samples <= 0: raise ValueError("iterations and samples must be positive")
    import matchup_db as M
    prov = prov or provenance()
    row, sim = policy_strings(policy_name, iterations, samples, prov)
    note = json.dumps({"labeler": policy_name, "iterations": iterations, "samples": samples, "sha256": prov,
                       "issue": 418})
    pid = M.get_policy(con, row, row, sim, note)
    got = con.execute("SELECT row_policy, col_policy FROM policy WHERE policy_id=?", (pid,)).fetchone()
    if got != (row, row): raise RuntimeError(f"policy {pid} is {got}, not the Rust policy {row}")
    return pid


# ---------------------------------------------------------------- games
def game_seed(seed, lo, hi, g):
    """(battle seed, shuffle seed) for game g of cell (lo, hi)."""
    h = hashlib.sha256(f"{seed}:{lo}:{hi}:{g}".encode()).hexdigest()
    return int(h[:8], 16) & 0x7FFFFFFF, int(h[8:24], 16)


NEXT_GAME = """CREATE TABLE IF NOT EXISTS rust_next_game (  -- first unplayed game index per Rust cell (rust_matchup_runner)
  policy_id INTEGER NOT NULL, lo INTEGER NOT NULL, hi INTEGER NOT NULL, next_game INTEGER NOT NULL,
  PRIMARY KEY (policy_id, lo, hi)) WITHOUT ROWID"""


def cell_counts(jobs):
    """{(lo, hi): games asked for} over a job list."""
    counts = {}
    for j in jobs:
        for c in j["schedule"]:
            key = tuple(sorted((j["row"], c))); counts[key] = counts.get(key, 0) + 1
    return counts


def claim_games(con, policy_id, counts):
    """Reserve each cell's game indices before any battle starts; returns {(lo, hi): first index}.
    A cell's next index never falls back, so a game that failed, or was still running when a job died,
    is skipped rather than its index handed to a later game whose twin is already recorded."""
    import matchup_db as M
    with con: con.execute(NEXT_GAME)
    starts = {}
    con.execute("BEGIN IMMEDIATE")          # write lock before the read: two jobs never claim the same index
    try:
        for (lo, hi), n in counts.items():
            r = con.execute("SELECT next_game FROM rust_next_game WHERE policy_id=? AND lo=? AND hi=?", (policy_id, lo, hi)).fetchone()
            starts[(lo, hi)] = start = max(r[0] if r else 0, M.battles_done(con, policy_id, lo, hi))
            con.execute("INSERT INTO rust_next_game VALUES (?,?,?,?) ON CONFLICT (policy_id, lo, hi) "
                        "DO UPDATE SET next_game=excluded.next_game", (policy_id, lo, hi, start + n))
        con.commit()
    except BaseException:
        con.rollback(); raise
    return starts


def make_games(jobs, pastes, starts, seed):
    """One game per schedule entry. jobs: matchup_db.jobs_for output; pastes: {team_id: paste};
    starts: {(lo, hi): first game index} (claim_games)."""
    games, counts = [], dict(starts)
    for j in jobs:
        for c in j["schedule"]:
            lo, hi = sorted((j["row"], c))
            g = counts[(lo, hi)]; counts[(lo, hi)] += 1
            bseed, sseed = game_seed(seed, lo, hi, g)
            rng = random.Random(sseed)
            p1, p2 = (lo, hi) if g % 2 == 0 else (hi, lo)
            games.append({"id": f"{lo}-{hi}-{g}", "lo": lo, "hi": hi, "game": g, "p1_team": p1, "p2_team": p2,
                          "a": shuffle_paste(pastes[p1], rng), "b": shuffle_paste(pastes[p2], rng), "seed": bseed})
    return games


def outcome(game, winner):
    """(lo, hi, lo_wins, hi_wins, ties) from Showdown's winner side ('p1', 'p2' or None for a tie)."""
    if winner is None: return (game["lo"], game["hi"], 0, 0, 1)
    won = game["p1_team"] if winner == "p1" else game["p2_team"]
    return (game["lo"], game["hi"], int(won == game["lo"]), int(won == game["hi"]), 0)


def agent_counts(stats):
    """Fallbacks, rejected choices and failed samples summed over both sides of one battle."""
    sides = (stats or {}).values()
    return {"fallbacks": sum(s.get("fallbacks", 0) for s in sides),
            "rejected": sum(len(s.get("rejected") or []) for s in sides),
            "failed_samples": sum(s.get("failedSamples", 0) for s in sides),
            "decisions": sum(s.get("decisions", 0) for s in sides),
            "iterations": sum(s.get("iterations", 0) for s in sides)}


# ---------------------------------------------------------------- the node processes
async def play_node(games, spec, on_done, workers=8, concurrency=1, timeout=1800, log=None, env=None, command=None):  # noqa: C901
    """Play every game on `workers` foul_play_doubles_runner.cjs processes (each with its own Rust engine
    daemon), `concurrency` battles at a time per process, pulling from one shared queue.
    on_done(game, result_or_None, error_or_None). A process that exits fails only its own battles; one
    whose battle times out is killed and retired, and the queue goes to the live processes."""
    procs, owned, dead, readers = [], {}, set(), []      # owned[k]: {game id: future} on process k

    def retire(k, why):
        dead.add(k)
        if procs[k].returncode is None:
            try: procs[k].kill()
            except ProcessLookupError: pass
        for f in owned[k].values():
            if not f.done(): f.set_exception(RustRunnerError(why))

    async def read(proc, k):
        try:
            while True:
                line = await proc.stdout.readline()
                if not line: break
                try: item = json.loads(line)
                except ValueError: continue          # a stray non-protocol line on stdout
                f = owned[k].get(item.get("id"))
                if f is None or f.done(): continue
                if item.get("kind") == "error": f.set_exception(RustRunnerError(str(item.get("error"))[:2000]))
                elif item.get("kind") == "result": f.set_result(item)
        finally:
            retire(k, f"battle runner {k} stopped (exit code {proc.returncode})")

    async def send_and_wait(k, game, f):
        msg = {"kind": "start", "id": game["id"], "a": game["a"], "b": game["b"], "seed": game["seed"], "p1": spec, "p2": spec}
        procs[k].stdin.write((json.dumps(msg) + "\n").encode()); await procs[k].stdin.drain()
        return await f

    queue = asyncio.Queue()
    for g in games: queue.put_nowait(g)

    async def consume(k):
        while k not in dead:
            try: game = queue.get_nowait()
            except asyncio.QueueEmpty: return
            f = owned[k][game["id"]] = asyncio.get_running_loop().create_future()
            try:
                result = await asyncio.wait_for(send_and_wait(k, game, f), timeout)
            except asyncio.TimeoutError:
                on_done(game, None, f"TimeoutError: no result in {timeout} s"); retire(k, "battle runner retired after a timeout")
                continue
            except (RustRunnerError, OSError) as e:
                on_done(game, None, f"{type(e).__name__}: {e}"[:2000]); continue
            finally:
                owned[k].pop(game["id"], None)
            on_done(game, result, None)

    try:
        for k in range(workers):
            owned[k] = {}
            p = await asyncio.create_subprocess_exec(*(command or ("node", str(RUNNER_JS))), stdin=asyncio.subprocess.PIPE,
                                                     stdout=asyncio.subprocess.PIPE, stderr=log, env=env, limit=2 ** 26)
            procs.append(p)
            try: ready = json.loads(await asyncio.wait_for(p.stdout.readline(), 120) or b"{}")
            except (asyncio.TimeoutError, ValueError): ready = {}
            if ready.get("kind") != "ready": raise RustRunnerError("foul_play_doubles_runner.cjs did not start")
        readers = [asyncio.create_task(read(p, k)) for k, p in enumerate(procs)]
        await asyncio.gather(*(consume(k) for k in range(len(procs)) for _ in range(concurrency)))
        while not queue.empty():                       # every process retired before the queue emptied
            on_done(queue.get_nowait(), None, "RustRunnerError: no live battle runner left")
    finally:
        for p in procs:
            try: p.stdin.close()
            except OSError: pass
        for p in procs:
            try: await asyncio.wait_for(p.wait(), 10)
            except asyncio.TimeoutError: p.kill()
        for r in readers: r.cancel()


# ---------------------------------------------------------------- the runner
class RustMatchupRunner:
    """runner(con, jobs, seed=None) -> (results, unattributed, seconds), as matchup_db.ensure expects.

    matchup_db.ensure calls it with policy_id and sink: battles then reach the database in batches of
    flush_every as they finish, and policy_id must be this runner's own Rust policy."""
    slot_order = "shuffled"
    streams = True

    def __init__(self, policy=DEFAULT_POLICY, samples=1, iterations=30000, workers=8, concurrency=1,
                 ms=1000, preview_ms=1000, flush_every=100, timeout=1800, evidence=None, allow_errors=False,
                 check_spreads=True, play=None, prov=None):
        if policy not in POLICIES: raise ValueError(f"policy must be one of {POLICIES}, not {policy!r}")
        self.policy, self.samples, self.iterations = policy, samples, iterations
        self.workers, self.concurrency, self.ms, self.preview_ms = workers, concurrency, ms, preview_ms
        self.flush_every, self.timeout, self.allow_errors = flush_every, timeout, allow_errors
        self.evidence, self.check_spreads, self._play = evidence, check_spreads, play
        self._prov = prov                    # tests pass fixed hashes; otherwise hashed afresh on every call
        self._spreads_checked = set()

    def spec(self):
        return {"policy": self.policy, "samples": self.samples, "ms": self.ms, "previewMs": self.preview_ms,
                "iterations": self.iterations}

    def register(self, con, inputs=None):
        """This runner's policy_id for the inputs as they are now (created on first use). inputs: the
        snapshot {bin, dex, spreads} a call battles with; default the live files."""
        return get_rust_policy(con, self.policy, self.iterations, self.samples, self.slot_order, self.preflight(inputs))

    def preflight(self, inputs=None):
        """Hash the inputs and fail before the first battle: engine, dex, runtime and node present; the
        spreads leave out the evaluation teams (checked once per spreads content)."""
        if self._prov is not None: return self._prov
        prov = provenance(**(inputs or {}))
        if self._play is None and shutil.which("node") is None: raise FileNotFoundError("node is not on PATH")
        if self.check_spreads and self._play is None and prov["spreads"] not in self._spreads_checked:
            import foul_play_doubles_battles as F
            F.check_spreads((inputs or {}).get("spreads") or SPREADS)
            self._spreads_checked.add(prov["spreads"])
        return prov

    def snapshot(self, folder):
        """Copies of the engine, dex and spreads in the run folder: the workers run the files that were hashed."""
        d = Path(folder) / "inputs"; d.mkdir(parents=True, exist_ok=True)
        out = {}
        for key, src in (("bin", BIN), ("dex", DEX), ("spreads", SPREADS)):
            if not Path(src).exists(): provenance()        # raises the named FileNotFoundError
            out[key] = d / Path(src).name; shutil.copy2(src, out[key])
        return out

    def __call__(self, con, jobs, seed=None, policy_id=None, sink=None):  # noqa: C901
        folder = Path(self.evidence or EVIDENCE / f"{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}-{uuid.uuid4().hex[:6]}")
        folder.mkdir(parents=True, exist_ok=True)
        inputs = None if self._play else self.snapshot(folder)
        prov = self.preflight(inputs)
        own = get_rust_policy(con, self.policy, self.iterations, self.samples, self.slot_order, prov)
        if policy_id is not None and policy_id != own:
            raise ValueError(f"this runner battles policy {own} ({self.policy}) with its current inputs; "
                             f"it cannot add battles to policy {policy_id}")
        ids = sorted({j["row"] for j in jobs} | {c for j in jobs for c in j["schedule"]})
        pastes = dict(con.execute(f"SELECT team_id, paste FROM team WHERE team_id IN ({','.join('?' * len(ids))})", ids))
        games = make_games(jobs, pastes, claim_games(con, own, cell_counts(jobs)), seed or 0)   # seed None plays as 0
        (folder / "config.json").write_text(json.dumps({**self.spec(), "workers": self.workers, "concurrency": self.concurrency,
                                                        "policy_id": own, "seed": seed, "games": len(games),
                                                        "sha256": prov}, indent=1) + "\n")
        out = (folder / "battles.jsonl").open("a")
        results, part, errors, totals = [], [], [], {"fallbacks": 0, "rejected": 0, "failed_samples": 0, "decisions": 0, "iterations": 0}
        window = {"t": time.perf_counter(), "unattributed": 0, **{k: 0 for k in totals}}
        t0 = time.perf_counter()

        def flush():
            if not part and not window["unattributed"]: return
            n = len(part) + window["unattributed"]
            note = (f"{self.policy} iterations={self.iterations} samples={self.samples} battles={n} "
                    f"fallbacks={window['fallbacks']} rejected={window['rejected']} failed_samples={window['failed_samples']} "
                    f"errors={window['unattributed']} evidence={folder}")
            if sink: sink(list(part), window["unattributed"], time.perf_counter() - window["t"], note)
            part.clear()
            window.update({"t": time.perf_counter(), "unattributed": 0, **{k: 0 for k in totals}})

        def on_done(game, result, error):
            if error is None and "winner" not in result: error = "battle ended without a winner line"
            row = {k: game[k] for k in ("id", "lo", "hi", "game", "p1_team", "p2_team", "seed")}
            if error is not None:
                errors.append(error); window["unattributed"] += 1
                out.write(json.dumps({**row, "error": error}) + "\n")
            else:
                r = outcome(game, result["winner"]); c = agent_counts(result.get("stats"))
                results.append(r); part.append(r)
                for k in totals: totals[k] += c[k]; window[k] += c[k]
                out.write(json.dumps({**row, "winner": result["winner"], "turns": result.get("turns"),
                                      "seconds": (result.get("ms") or 0) / 1000, **c}) + "\n")
            out.flush()
            if len(part) + window["unattributed"] >= self.flush_every: flush()

        env = {**os.environ, "SHOWDOWN_RUNTIME": str(RUNTIME)}
        if inputs: env.update(FOUL_PLAY_SPREADS=str(inputs["spreads"]), RUST_ENGINE_BIN=str(inputs["bin"]), REGMB_DEX=str(inputs["dex"]))
        try:
            if self._play: self._play(games, self.spec(), on_done)
            else:
                with (folder / "runner.log").open("a") as log:
                    asyncio.run(play_node(games, self.spec(), on_done, self.workers, self.concurrency,
                                          self.timeout, log, env))
        finally:
            flush(); out.close()
        seconds = time.perf_counter() - t0
        (folder / "summary.json").write_text(json.dumps({"battles": len(results), "errors": len(errors), **totals,
                                                         "seconds": seconds, "first_errors": errors[:3]}, indent=1) + "\n")
        self.last = {"battles": len(results), "errors": len(errors), **totals, "seconds": seconds, "evidence": str(folder)}
        if errors and not self.allow_errors:
            raise RustRunnerError(f"{len(errors)} of {len(games)} battles failed ({len(results)} recorded); "
                                  f"first: {errors[0][:500]}; evidence {folder}")
        return results, len(errors), seconds
