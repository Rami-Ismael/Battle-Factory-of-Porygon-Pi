"""matchup_db runner that battles on Modal (src/battle_modal.py) instead of the laptop's fast evaluator.

Same contract as fast_battle_runner.FastBattleRunner: runner(con, jobs, seed) -> ([(row, col, wins, losses, ties)],
unattributed, seconds), returning only once every requested battle is finished and attributed. The requests are
fast_battle_runner's own (build_requests: 24-battle chunks, the shuffled-slot stream from random.Random(seed), seed
+ job_id per request, the same worker config), and fast_battle_runner.attribute checks every outcome here, so a
battle counts the same whichever machine ran it. The database never leaves the laptop.

Select it with MATCHUP_BACKEND=modal (matchup_db.default_runner).
"""
import os
import random
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import battle_modal as BM

MODAL_PYTHON = os.environ.get("MODAL_PYTHON", str(Path.home() / ".local/share/uv/tools/modal/bin/python"))
REQUESTS_PER_SHARD = int(os.environ.get("MODAL_REQUESTS_PER_SHARD", 8))
CALL_TIMEOUT = int(os.environ.get("MODAL_CALL_TIMEOUT", 1200))        # seconds; one cell takes under a minute
FAILURE_FILE = os.environ.get("MODAL_FAILURE_FILE")                   # consecutive failed calls, kept across restarts
MAX_FAILURES = int(os.environ.get("MODAL_MAX_FAILURES", 3))           # then refuse before spending more credit
CONFIG = {"batcher": "original", "threads": 1, "concurrency": 24, "shared": True, "encoder": "fast",
          "actor_only": True, "inference_mode": True}                 # FastBattleRunner.config with matchup_db's defaults


def shards_of(requests, per_shard=REQUESTS_PER_SHARD, max_shards=BM.MAX_CONTAINERS * 4):
    """Largest requests first, dealt round-robin so every shard carries about the same number of battles."""
    count = min(max_shards, max(1, -(-len(requests) // per_shard)))
    shards = [[] for _ in range(count)]
    for i, request in enumerate(sorted(requests, key=lambda r: (-len(r["pairs"]), r["job_id"]))):
        shards[i % count].append(request)
    return shards


class ModalBattleRunner:
    def __init__(self, slot_order, fast_runner_dir, call=None):
        if slot_order not in ("listed", "shuffled"):
            raise ValueError("slot_order must be listed or shuffled")
        self.slot_order = slot_order
        if str(fast_runner_dir) not in sys.path: sys.path.append(str(fast_runner_dir))
        import fast_battle_runner as FBR
        self.FBR = FBR
        self.call = call or self.call_modal
        self.calls, self.battles, self.seconds = 0, 0, 0.0
        self.manifest = BM.manifest() if call is None else None

    def failures(self):
        try: return int(Path(FAILURE_FILE).read_text()) if FAILURE_FILE else 0
        except (OSError, ValueError): return 0

    def note_failure(self, failed):
        if FAILURE_FILE: Path(FAILURE_FILE).write_text(str(self.failures() + 1 if failed else 0))

    def call_modal(self, shards):
        with tempfile.TemporaryDirectory() as tmp:
            in_path, out_path = Path(tmp) / "in.z", Path(tmp) / "out.z"
            in_path.write_bytes(BM.pack(shards))
            done = subprocess.run([MODAL_PYTHON, str(HERE / "battle_modal.py"), "call", str(in_path), str(out_path)],
                                  capture_output=True, text=True, timeout=CALL_TIMEOUT)
            if done.returncode != 0:
                raise RuntimeError(f"Modal battle call failed: {done.stderr[-800:]}")
            reply = BM.unpack(out_path.read_bytes())
        if reply["manifests"] != [self.manifest]:
            raise RuntimeError(f"deployed battle app {reply['manifests']} differs from this checkout's {self.manifest}; "
                               "redeploy src/battle_modal.py")
        return reply["outcomes"]

    def __call__(self, con, jobs, seed=None):
        seed = 20261013 if seed is None else seed
        order_rng = random.Random(seed) if self.slot_order == "shuffled" else None
        requests = self.FBR.build_requests(con, jobs, CONFIG["concurrency"], order_rng)
        if not requests:
            return [], 0, 0.0
        if not isinstance(seed, int) or seed < 0 or seed + len(requests) >= 2**32:
            raise ValueError("Seed range is outside NumPy limits")
        for request in requests:
            request.update(seed=seed + request["job_id"], config=CONFIG)
        if self.failures() >= MAX_FAILURES:
            raise RuntimeError(f"{self.failures()} Modal battle calls failed in a row ({FAILURE_FILE}); refusing to spend more")
        start = time.perf_counter()
        try:
            outcomes = self.call([{"config": CONFIG, "requests": shard} for shard in shards_of(requests)])
            by_job = {o["job_id"]: o for o in outcomes}
            failed = [o["error"] for o in outcomes if "error" in o]
            if failed:
                raise RuntimeError(f"{len(failed)} Modal battle request(s) failed; nothing committed: {failed[0][:300]}")
            missing = [r["job_id"] for r in requests if r["job_id"] not in by_job]
            if missing:
                raise RuntimeError(f"{len(missing)} Modal battle request(s) returned nothing; nothing committed")
            results = [entry for r in requests for entry in self.FBR.attribute(r, by_job[r["job_id"]]["result"])]
        except BaseException:
            self.note_failure(True)
            raise
        self.note_failure(False)
        seconds = time.perf_counter() - start
        self.calls += 1; self.battles += sum(sum(r[2:]) for r in results); self.seconds += seconds
        print(f"  modal: {sum(sum(r[2:]) for r in results)} battles in {seconds:.1f}s "
              f"({len(requests)} requests, {len(shards_of(requests))} shards)", flush=True)
        return results, 0, seconds
