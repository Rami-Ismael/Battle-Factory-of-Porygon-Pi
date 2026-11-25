"""Shuffled-slot battles on Modal CPU containers, for matchup_db's runner contract (owner, 2026-11-25, #431 grids).

The owner chose Modal for the #431 ask-grid battles. Only the battles move: the matchup database, the request
building (fast_battle_runner.build_requests, same chunking, same slot-shuffle stream and per-request seeds) and the
attribution check (fast_battle_runner.attribute) stay on the laptop in modal_battle_runner.py. Each container runs
the laptop's own battle worker (battle_speed_lab/worker.py and run.py's Server and Worker) against a private
Showdown server built from the pinned validator checkout, with the same behaviour-cloning checkpoint (sha256 checked),
the same vgc-bench checkout and the laptop venv's package versions. Inputs sit at their laptop paths, HOME included.

    modal deploy code/src/battle_modal.py                 # once, and after any change to this file
    <modal python> code/src/battle_modal.py call IN OUT   # modal_battle_runner.py runs this; IN/OUT are zlib JSON

Before any launch: state the estimated cost and the credit left (`modal billing summary`).
"""
from __future__ import annotations

import json
import sys
import zlib
from pathlib import Path

from showdown_validator import SHOWDOWN as VALIDATOR, VGC_BENCH as BENCH

HOME = Path.home()
RUNTIME = HOME / ".local/share/vgc-pilot-runtime"
VAULT = HOME / ("Documents/yakumsi-vault/Personal Project/Using Advance method search in the whole search of possible vgc "
                "pokemon format to find the right counter to a pokemon team")
LAB = VAULT / "battle_speed_lab"                                     # worker.py, run.py, encoder.py, inference.py
FAST_RUNNER = VAULT / "team_completion_pipeline/fast_battle_runner.py"
CORPUS = HOME / "Documents/code/vgc-team-generator-pilot/src/corpus.py"         # worker.py imports dex_entry from here
POLICY = RUNTIME / "files/bc_100.zip"                                # /tmp/bc_100.zip links here
POLICY_SHA256 = "57f5edcab415cf6ccc1b6231923c8b66d3b1b7249b6b2562b97531471e4ca60b"   # fast_battle_runner.CHECKPOINT_SHA256
APP_NAME, FUNCTION_NAME = "vgc-battles", "run_shard"
CPU, MEMORY_MB = 8.0, 16_384
WORKERS_PER_CONTAINER = 4               # the laptop runs 6 worker/server pairs on 12 cores: one pair per 2 cores
MAX_CONTAINERS = 24
REQUEST_TIMEOUT = 180                   # seconds per 24-battle request, as fast_battle_runner sends
PACKAGES = ("numpy==2.4.4", "stable-baselines3==2.8.0", "gymnasium==1.2.3", "cloudpickle==3.1.2", "pandas==3.0.5",
            "matplotlib==3.11.1", "websockets==16.0", "orjson==3.12.0", "tabulate==0.10.0", "requests==2.34.2",
            "pettingzoo==1.27.0", "farama-notifications==0.0.6",
            "poke-env @ git+https://github.com/cameronangliss/poke-env.git@379a04628e42b2a2c94790c0788204b652f7ae1b")
TORCH = "torch==2.12.0"                 # the laptop venv's version, CPU build


def manifest() -> str:
    """Digest of everything that decides a battle outside the request: this file, the battle worker's sources, the
    Showdown build and formats, vgc-bench source and data, the policy checkpoint and the package pins. The laptop
    compares it with the deployed app's on every call, so different code cannot slip results into the same cache."""
    import hashlib
    sys.path.insert(0, str(LAB))
    import run as lab
    sys.path.insert(0, str(FAST_RUNNER.parent))
    from fast_battle_runner import python_source_digest

    digest = hashlib.sha256()
    for path in [Path(__file__), *(LAB / n for n in ("worker.py", "run.py", "encoder.py", "inference.py")),
                 VALIDATOR / "config/formats.ts", POLICY]:
        digest.update(path.name.encode() + b"\0" + Path(path).read_bytes())
    digest.update(lab.tree_digest(VALIDATOR / "dist").encode())
    digest.update(python_source_digest(BENCH / "vgc_bench/src").encode())
    digest.update(lab.tree_digest(BENCH / "data").encode())
    digest.update("|".join((TORCH, *PACKAGES)).encode())
    return digest.hexdigest()


def pack(value) -> bytes:
    return zlib.compress(json.dumps(value).encode(), 6)


def unpack(blob: bytes):
    return json.loads(zlib.decompress(blob))


# ---------------------------------------------------------------- container side

_pool = None


class Pool:
    """WORKERS_PER_CONTAINER worker/server pairs, started once per container and reused across calls."""

    def __init__(self, config):
        import tempfile
        sys.path.insert(0, str(LAB))
        import run as lab
        self.lab, self.config = lab, config
        self.folder = Path(tempfile.mkdtemp(prefix="battles-"))
        self.runtime = Path("/tmp/showdown-runtime")
        lab.prepare_runtime(self.runtime)
        self.servers = [lab.Server(self.runtime, self.folder) for _ in range(WORKERS_PER_CONTAINER)]
        self.workers = [lab.Worker(config, self.folder, str(i)) for i in range(WORKERS_PER_CONTAINER)]

    def restart(self, i):
        for process in (self.workers[i], self.servers[i]):
            try: process.close()
            except Exception: pass
        self.servers[i] = self.lab.Server(self.runtime, self.folder)
        self.workers[i] = self.lab.Worker(self.config, self.folder, f"{i}r")

    def run(self, requests):
        import queue
        import threading
        todo = queue.Queue()
        for request in requests: todo.put(request)
        out, lock = [], threading.Lock()

        def consume(i):
            while True:
                try: request = todo.get_nowait()
                except queue.Empty: return
                for attempt in (1, 2):                      # one restart of this pair on a failed request
                    try:
                        result = self.workers[i].trial({**request, "port": self.servers[i].port, "timeout": REQUEST_TIMEOUT})
                        with lock: out.append({"job_id": request["job_id"], "result": result})
                        break
                    except Exception as exc:
                        if attempt == 2:
                            with lock: out.append({"job_id": request["job_id"], "error": f"{type(exc).__name__}: {exc}"})
                        else:
                            self.restart(i)

        threads = [threading.Thread(target=consume, args=(i,)) for i in range(WORKERS_PER_CONTAINER)]
        for t in threads: t.start()
        for t in threads: t.join()
        return out


_manifest = None


def run_in_container(payload: bytes) -> bytes:
    global _pool, _manifest
    import hashlib
    shard = unpack(payload)
    if hashlib.sha256(Path("/tmp/bc_100.zip").read_bytes()).hexdigest() != POLICY_SHA256:
        raise ValueError("behaviour-cloning checkpoint differs from the verified policy")
    if _pool is None or _pool.config != shard["config"]:
        _pool = Pool(shard["config"])
    _manifest = _manifest or manifest()
    return pack({"manifest": _manifest, "outcomes": _pool.run(shard["requests"])})


# ---------------------------------------------------------------- laptop side: called by modal_battle_runner.py

def call(in_path, out_path):
    """Run every shard in IN on the deployed app and write their outcomes, in any order, to OUT."""
    import modal
    shards = unpack(Path(in_path).read_bytes())
    function = modal.Function.from_name(APP_NAME, FUNCTION_NAME)
    manifests, outcomes = set(), []
    for blob in function.map([pack(s) for s in shards], order_outputs=False):
        part = unpack(blob)
        manifests.add(part["manifest"]); outcomes.extend(part["outcomes"])
    Path(out_path).write_bytes(pack({"manifests": sorted(manifests), "outcomes": outcomes}))


try:
    import modal
except ImportError:
    modal = None

if modal is not None:
    image = (modal.Image.from_registry("node:22-bookworm-slim", add_python="3.11")
             .apt_install("git", "python3", "make", "g++")
             .pip_install(TORCH, index_url="https://download.pytorch.org/whl/cpu")
             .pip_install(*PACKAGES)
             .env({"HOME": str(HOME), "PYTHONUNBUFFERED": "1"})
             .add_local_dir(VALIDATOR, str(VALIDATOR), copy=True,
                            ignore=["**/.git/**", "logs/**", "databases/**", "node_modules/sqlite3/build/**",
                                    "node_modules/better-sqlite3/build/**"])
             .run_commands(f"cd '{VALIDATOR}' && (npm rebuild sqlite3 better-sqlite3 || true)",
                           "mkdir -p /tmp/vgc-pilot/vgc-bench",
                           f"ln -sfn '{POLICY}' /tmp/bc_100.zip",
                           f"ln -sfn '{RUNTIME / 'data'}' /tmp/vgc-pilot/data",
                           f"ln -sfn '{RUNTIME / 'teams'}' /tmp/vgc-pilot/vgc-bench/teams")
             .add_local_dir(LAB, str(LAB), ignore=["**/__pycache__/**", "runs/**"])
             .add_local_dir(BENCH, str(BENCH), ignore=["**/__pycache__/**", "**/.git/**"])
             .add_local_file(FAST_RUNNER, str(FAST_RUNNER))
             .add_local_dir(RUNTIME / "data", str(RUNTIME / "data"))
             .add_local_dir(RUNTIME / "teams", str(RUNTIME / "teams"))
             .add_local_file(CORPUS, str(CORPUS))
             .add_local_file(POLICY, str(POLICY))
             .add_local_python_source("showdown_validator"))
    app = modal.App(APP_NAME)

    @app.function(image=image, cpu=CPU, memory=MEMORY_MB, timeout=3600, scaledown_window=60, max_containers=MAX_CONTAINERS)
    def run_shard(payload: bytes) -> bytes:
        return run_in_container(payload)


if __name__ == "__main__":
    if sys.argv[1:2] == ["call"]:
        call(sys.argv[2], sys.argv[3])
    else:
        raise SystemExit(__doc__)
