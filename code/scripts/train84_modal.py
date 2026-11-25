"""Modal GPU launcher for the 84-field trainers, src/train84.py and src/train_uniform84.py (issue #431).

The owner, 2026-11-25: "run the code on modal for not on my laptop there will be resource contention". The
laptop runs the battle queues; this trains on a Modal GPU with the trainers' own CLI and the laptop's exact inputs.

How the container matches the laptop:
  - HOME is the laptop's home, and every input sits at its laptop path: the trainers' code (src/), the pilot
    checkout /tmp/vgc-pilot/src points at, the runtime data, the Reg M-B team pool, the pinned Showdown validator,
    the training table and the /tmp/vgc-pilot symlinks. RUNTIME_INPUTS lists them.
  - The trainers pick their device in diffusion.py ("mps" or "cpu"). BOOT sets that module's DEV to "cuda" the moment
    it loads, so every module that copies DEV (lossdown, diffusion84, uniform_diffusion84) trains on the GPU.
    bf16 autocast stays on, as on the laptop's MPS GPU, so the GPU list leaves out the T4, which lacks bf16.
  - Outputs go to the Modal volume vgc-train84: the checkpoint folder (best and every epoch), the run JSON, the
    training log and the offline Weights & Biases run. Fetch them with fetch_commands(), then `wandb sync`.

Before any launch: state the estimated cost and the credit left (`modal billing summary`), and wait for the owner.

    modal run code/scripts/train84_modal.py::main --arm udlm_loglinear_gidd_lr --smoke          # 1 epoch on 512 teams
    modal run --detach code/scripts/train84_modal.py::main --arm udlm_loglinear_gidd_lr
    modal run --detach code/scripts/train84_modal.py::main --arm shuffled_linear_filtered_gidd_lr

First-try legality of a training's saved epoch checkpoints with the legal Pokémon construction filter off
(src/checkpoint_legality.py), for an arm whose training check drew through the filter and so read 0 every epoch:

    modal volume put vgc-train84 <checkpoint folder> lossdown/<training>      # unless the training ran here
    modal run code/scripts/train84_modal.py::legality_main --trainings <training>,<training>
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1]
REMOTE_CODE = "/root/repo/code"
HOME = Path.home()                       # the container gets the same HOME, so Path.home() paths match
RUNTIME = HOME / ".local/share/vgc-pilot-runtime"
CHECKPOINTS = RUNTIME / "lossdown"       # lossdown.CK
RESULTS = HOME / "Documents/code/vgc-team-generator-pilot/results"   # asked_vs_got.RES, written as a literal there
PILOT_SRC = Path("/tmp/vgc-pilot/src")
VALIDATOR = RUNTIME / "validator-913da36"                            # showdown_validator.SHOWDOWN
TABLE = "lossdata84_shuffled_only.npz"
VOLUME_NAME = "vgc-train84"
VOLUME = "/vol"
GPU = ["L4", "A10G"]                     # bf16 on both; L4 is the cheaper
CPU, MEMORY_MB = 4.0, 16_384
TIMEOUT_SECONDS = 12 * 3600

# Code ID -> trainer script and the checkpoint name the trainer saves under (its NAMES entry)
ARMS = {
    "udlm_loglinear_gidd_lr": dict(script="src/train_uniform84.py", checkpoint="stat84_udlm_loglinear_lr5e4_min5e5_es3_cap80"),
    "shuffled_linear_filtered_gidd_lr": dict(script="src/train84.py",
                                             checkpoint="stat84_shuffled_noise_linear_filtered_lr5e4_min5e5_es3_cap80"),
}
SMOKE_PREFIX = "stat84smoke_"            # the trainers' S84_SMOKE name prefix
SEED_CHECKPOINTS = ("F1_final_medium_noheldout_regmb_v3.pt",)        # read from lossdown.CK at import
PACKAGES = ("torch==2.12.0", "numpy==2.4.4", "beartype==0.23.1", "jaxtyping==0.3.11", "tqdm==4.70.1",
            "typing_extensions==4.16.0", "wandb==0.31.2")            # the laptop venv's versions
DEVICE_MODULES = ("diffusion", "propose")                            # modules that set DEV to "mps" or "cpu"


def runtime_inputs():
    """(laptop path, container path) for every input outside src/, found by tracing the files training opens."""
    validator = [VALIDATOR / p for p in ("dist", "config", "node_modules", "validate-teams-batch.js", "package.json")]
    paths = [PILOT_SRC.resolve(), RUNTIME / "data", RUNTIME / "files/learnset_true.json", RUNTIME / "files/top50_evs.json",
             RUNTIME / "teams/reg_mb", *validator, RESULTS / TABLE, RESULTS / "heldout_next50_evs.json"]
    pairs = [(p, p) for p in paths]
    pairs += [(CHECKPOINTS / name, Path("/opt/vgc/seed") / name) for name in SEED_CHECKPOINTS]
    return pairs


def symlinks():
    """(link, target) pairs that recreate the laptop's /tmp/vgc-pilot layout, and point lossdown.CK at the volume."""
    return [(PILOT_SRC, PILOT_SRC.resolve()), (Path("/tmp/vgc-pilot/data"), RUNTIME / "data"),
            (Path("/tmp/vgc-pilot/learnset_true.json"), RUNTIME / "files/learnset_true.json"),
            (Path("/tmp/vgc-pilot/top50_evs.json"), RUNTIME / "files/top50_evs.json"),
            (Path("/tmp/vgc-pilot/vgc-bench/teams"), RUNTIME / "teams"),
            (Path("/tmp/vgc-pilot/vgc-bench/pokemon-showdown"), VALIDATOR),
            (CHECKPOINTS, Path(VOLUME) / "lossdown")]


def run_name(arm, smoke=False):
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; choose from {sorted(ARMS)}")
    return (SMOKE_PREFIX if smoke else "") + ARMS[arm]["checkpoint"]


def trainer_command(arm, *, legality_teams=64, wandb_mode="offline"):
    """The trainer's own CLI, run through BOOT so its device is CUDA."""
    script = f"{REMOTE_CODE}/{ARMS[arm]['script']}"
    run_name(arm)                                                     # validates arm
    return [sys.executable, "-u", "-c", BOOT, script, arm,
            "--legality-teams", str(legality_teams), "--wandb-mode", wandb_mode]


def trainer_env(*, smoke=False, base=None):
    env = dict(os.environ if base is None else base, HOME=str(HOME), PYTHONUNBUFFERED="1", WANDB_DIR=f"{VOLUME}/wandb")
    if smoke:
        env["S84_SMOKE"] = "1"
    else:
        env.pop("S84_SMOKE", None)
    return env


def fetch_commands(arm, smoke=False):
    """Shell commands that copy one run's outputs from the volume to the laptop's checkpoint and results folders."""
    name = run_name(arm, smoke)
    return [f"modal volume get {VOLUME_NAME} lossdown/{name}.pt '{CHECKPOINTS}/'",
            f"modal volume get {VOLUME_NAME} lossdown/{name} '{CHECKPOINTS}/'",
            f"modal volume get {VOLUME_NAME} results/{name}.json '{RESULTS}/'",
            f"modal volume get {VOLUME_NAME} logs/{name}.log '{RESULTS}/'"]


# Runs inside the container: python -c BOOT <script> <args...>. A finder wraps each module in DEVICE_MODULES so its
# DEV is "cuda" before any other module copies it, whichever checkout the trainer's sys.path finds it in.
BOOT = r'''
import importlib.abc, os, runpy, sys, torch
if os.environ.get("TRAIN84_MODAL_REQUIRE_CUDA", "1") == "1":
    assert torch.cuda.is_available(), "no CUDA device in the container"
GPU_NAME = torch.cuda.get_device_name() if torch.cuda.is_available() else "no CUDA device"
DEVICE_MODULES = %r
class CudaDevice(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name not in DEVICE_MODULES: return None
        for finder in sys.meta_path:
            if finder is self or not hasattr(finder, "find_spec"): continue
            spec = finder.find_spec(name, path, target)
            if spec is not None: break
        else: return None
        run = spec.loader.exec_module
        def exec_module(module):
            run(module); module.DEV = "cuda"
            print(f"[modal] {name}.DEV -> cuda ({module.__file__}; {GPU_NAME})", flush=True)
        spec.loader.exec_module = exec_module
        return spec
sys.meta_path.insert(0, CudaDevice())
script = sys.argv[1]; sys.argv = sys.argv[1:]
runpy.run_path(script, run_name="__main__")
''' % (DEVICE_MODULES,)


def legality_command(training, guidance=None):
    """src/checkpoint_legality.py on one training's epoch checkpoints, no legal Pokémon construction filter, through BOOT.
    guidance None keeps the script's default, the training check's guidance 2."""
    extra = [] if guidance is None else ["--guidance", f"{guidance:g}"]
    return [sys.executable, "-u", "-c", BOOT, f"{REMOTE_CODE}/src/checkpoint_legality.py", training, *extra]


def legality_result_name(training, guidance=None):
    """checkpoint_legality.result_path with tables off."""
    suffix = "" if guidance in (None, 2.0) else f"_g{guidance:g}"
    return f"{training}_legality_no_tables{suffix}.json"


def prepare_volume():
    """Make the volume's folders and seed its checkpoint folder with the checkpoints lossdown reads at import."""
    volume_dir = Path(VOLUME)
    for sub in ("lossdown", "results", "logs", "wandb"):
        (volume_dir / sub).mkdir(parents=True, exist_ok=True)
    for seed_name in SEED_CHECKPOINTS:
        if not (volume_dir / "lossdown" / seed_name).exists():
            shutil.copy2(Path("/opt/vgc/seed") / seed_name, volume_dir / "lossdown" / seed_name)
    return volume_dir


def stream(command, log_path, env, commit):
    """Run command with its output teed to log_path, committing the volume every 10 minutes; returns the exit code."""
    with open(log_path, "w") as log:
        process = subprocess.Popen(command, env=env, cwd=REMOTE_CODE, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1)
        last_commit = time.monotonic()
        for line in process.stdout:
            sys.stdout.write(line); log.write(line); log.flush()
            if time.monotonic() - last_commit > 600:
                commit(); last_commit = time.monotonic()
        return process.wait()


def legality_in_container(trainings, guidance=None, commit=lambda: None):
    """Container body: no-filter first-try legality of every epoch checkpoint of each training, which must already be
    in the volume's lossdown folder. Copies each result JSON to the volume's results folder."""
    volume_dir = prepare_volume()
    out = {}
    for training in trainings:
        if not (volume_dir / "lossdown" / training).is_dir():
            raise FileNotFoundError(f"{VOLUME_NAME}:/lossdown/{training} is missing; upload it with modal volume put")
        start_time = time.perf_counter()
        result = RESULTS / legality_result_name(training, guidance)
        exit_code = stream(legality_command(training, guidance), volume_dir / "logs" / result.with_suffix(".log").name,
                           trainer_env(), commit)
        if result.exists():
            shutil.copy2(result, volume_dir / "results" / result.name)
        commit()
        if exit_code:
            raise RuntimeError(f"checkpoint_legality.py {training} exited with code {exit_code}")
        out[training] = round((time.perf_counter() - start_time) / 60, 1)
    return out


def run_in_container(arm, *, smoke, legality_teams, commit=lambda: None):
    """Container body: seed the volume's checkpoint folder, run the trainer with its log teed to the volume,
    then copy the run JSON to the volume."""
    volume_dir = prepare_volume()
    name = run_name(arm, smoke)
    log_path = volume_dir / "logs" / f"{name}.log"
    start_time = time.perf_counter()
    exit_code = stream(trainer_command(arm, legality_teams=legality_teams), log_path, trainer_env(smoke=smoke), commit)
    run_json = RESULTS / f"{name}.json"
    if run_json.exists():
        shutil.copy2(run_json, volume_dir / "results" / run_json.name)
    commit()
    if exit_code:
        raise RuntimeError(f"{ARMS[arm]['script']} {arm} exited with code {exit_code}; log: {VOLUME_NAME}:/logs/{log_path.name}")
    return dict(arm=arm, run=name, minutes=round((time.perf_counter() - start_time) / 60, 1),
                run_json=run_json.read_text() if run_json.exists() else None)


try:
    import modal
except ImportError:
    modal = None

if modal is not None:
    image = (modal.Image.from_registry("node:22-bookworm-slim", add_python="3.11")
             .pip_install(*PACKAGES)
             .env({"HOME": str(HOME), "PYTHONUNBUFFERED": "1"}))
    image = image.run_commands(*[f"mkdir -p '{link.parent}' && ln -sfn '{target}' '{link}'" for link, target in symlinks()])
    image = image.add_local_dir(CODE_DIR / "src", f"{REMOTE_CODE}/src", ignore=["**/__pycache__/**", "*.bak*"])
    for local, remote in runtime_inputs():
        image = (image.add_local_dir(local, str(remote), ignore=["**/__pycache__/**", "**/.git/**"]) if local.is_dir()
                 else image.add_local_file(local, str(remote)))

    volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
    app = modal.App("train84-gidd-lr")

    @app.function(image=image, gpu=GPU, cpu=CPU, memory=MEMORY_MB, timeout=TIMEOUT_SECONDS, volumes={VOLUME: volume})
    def train(arm: str, smoke: bool = False, legality_teams: int = 64) -> dict:
        return run_in_container(arm, smoke=smoke, legality_teams=legality_teams, commit=volume.commit)

    @app.local_entrypoint()
    def main(arm: str, smoke: bool = False, legality_teams: int = 64):
        print(f"[train84_modal] {arm} -> {run_name(arm, smoke)} on {GPU}, timeout {TIMEOUT_SECONDS // 3600} h", flush=True)
        out = train.remote(arm, smoke, legality_teams)
        print(f"[train84_modal] done in {out['minutes']} min. Fetch with:")
        for cmd in fetch_commands(arm, smoke):
            print("  " + cmd)

    @app.function(image=image, gpu=GPU, cpu=CPU, memory=MEMORY_MB, timeout=2 * 3600, volumes={VOLUME: volume})
    def check_legality(trainings: list[str], guidance: float | None = None) -> dict:
        return legality_in_container(trainings, guidance, commit=volume.commit)

    @app.local_entrypoint()
    def legality_main(trainings: str, guidance: float = 2.0):
        """trainings: comma-separated checkpoint folder names already on the volume."""
        names = [t for t in trainings.split(",") if t]
        print(f"[train84_modal] no-filter legality of {names} at guidance {guidance:g} on {GPU}", flush=True)
        minutes = check_legality.remote(names, guidance)
        print(f"[train84_modal] done: {minutes} min. Fetch with:")
        for name in names:
            print(f"  modal volume get {VOLUME_NAME} results/{legality_result_name(name, guidance)} '{RESULTS}/'")
