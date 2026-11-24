"""Minimal Modal GPU launcher for the masked Low-Rank Adaptation (LoRA) trainer tracer bullet (#408).

Packages only the masked trainer and its direct dependencies (no block diffusion files and no
Mask Lab pipeline files), runs `masked_lora_trainer.py` with its unchanged CLI inside one
Modal GPU container, copies the `--out` results back to the laptop, and stops the container
as soon as the command exits. #411 extends this launcher for full training runs.

Before any Modal launch: state its estimated cost and credit left (`modal billing summary`),
then wait for the owner's go-ahead.

    modal run code/scripts/masked_lora_modal.py \
        --argv "tracer --data /tmp/rows.npz --out /tmp/out" \
        --data-file /tmp/diffugpt_tracer_256.npz \
        --local-out artifacts/diffugpt-tracer-408
"""
from __future__ import annotations

import shlex
import subprocess
import time
from pathlib import Path
from typing import Any

CODE_DIR = Path(__file__).resolve().parents[1]
REMOTE_CODE = '/root/code'
REMOTE_TRAINER = f'{REMOTE_CODE}/scripts/masked_lora_trainer.py'
REMOTE_DATA = '/tmp/rows.npz'
REMOTE_OUT = '/tmp/out'
DEFAULT_GPU: list[str] = ['L4', 'T4', 'A10G']
TIMEOUT_SECONDS = 20 * 60            # 20-minute cap (~$0.20-$0.38 max across T4/L4/A10G)
HF_CACHE_VOLUME = 'vgc-hf-cache'

IMAGE_PACKAGES: tuple[str, ...] = (
    'torch==2.14.1',
    'transformers==4.57.3',
    'peft==0.18.1',
    'accelerate==1.15.0',
    'safetensors==0.8.0',
    'numpy==2.4.6',
    'huggingface_hub==0.36.2',
)

# Relative paths under code/ packaged into the Modal image: only the masked trainer and its deps.
PACKAGED_FILES: tuple[str, ...] = (
    'scripts/masked_lora_trainer.py',
    'scripts/masked_diffusion_text.py',
    'scripts/masked_diffusion_memory_bench.py',
    'scripts/masked_diffusion_sampler.py',
    'scripts/mask_paste.py',
    'data/regmb-vocabulary.json',
)


def build_remote_command(argv: str) -> list[str]:
    """Build the trainer subprocess command line, keeping the trainer's CLI unchanged."""
    tokens = shlex.split(argv)
    if not tokens:
        raise ValueError('argv must contain a masked_lora_trainer.py subcommand and flags')
    return ['python', REMOTE_TRAINER, *tokens]


def extract_out_dir(argv: str, default: Path = Path(REMOTE_OUT)) -> Path:
    """Read `--out <dir>` or `--out=<dir>` from `argv` so results copy from the requested path."""
    tokens = shlex.split(argv)
    for idx, token in enumerate(tokens):
        if token == '--out' and idx + 1 < len(tokens):
            return Path(tokens[idx + 1])
        if token.startswith('--out='):
            return Path(token.removeprefix('--out='))
    return default


def collect_output_files(out_dir: Path) -> dict[str, bytes]:
    """Read all files written under `out_dir` as relative-path -> raw bytes for return to the laptop."""
    if not out_dir.exists():
        return {}
    return {
        str(p.relative_to(out_dir)): p.read_bytes()
        for p in sorted(out_dir.rglob('*'))
        if p.is_file()
    }


def write_output_files(local_out: Path, files: dict[str, bytes]) -> list[Path]:
    """Write returned remote output files into `local_out`, refusing path traversal."""
    root = local_out.resolve()
    written: list[Path] = []
    for rel, content in files.items():
        dest = (root / rel).resolve()
        if not dest.is_relative_to(root):
            raise ValueError(f'refusing to write outside {root}: {rel}')
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(content)
        written.append(dest)
    return written


def run_trainer_subprocess(
    argv: str,
    *,
    data_bytes: bytes | None = None,
    data_dest: Path = Path(REMOTE_DATA),
    out_dir: Path | None = None,
) -> dict[str, Any]:
    """Container-side body: stage data bytes, execute trainer CLI, collect outputs."""
    if data_bytes is not None:
        data_dest.parent.mkdir(parents=True, exist_ok=True)
        data_dest.write_bytes(data_bytes)
    target_out = out_dir if out_dir is not None else extract_out_dir(argv)
    target_out.mkdir(parents=True, exist_ok=True)
    cmd = build_remote_command(argv)
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=f'{REMOTE_CODE}/scripts', check=False)
    elapsed = time.perf_counter() - t0
    files = collect_output_files(target_out)
    if proc.returncode != 0:
        raise RuntimeError(f'masked_lora_trainer.py exited with code {proc.returncode}')
    return {'returncode': proc.returncode, 'elapsed_seconds': elapsed, 'files': files}


try:
    import modal
except ImportError:
    modal = None

if modal is not None:
    image = (
        modal.Image.debian_slim(python_version='3.11')
        .pip_install(*IMAGE_PACKAGES)
        .env({'PYTHONUNBUFFERED': '1'})
    )
    for _rel in PACKAGED_FILES:
        image = image.add_local_file(CODE_DIR / _rel, f'{REMOTE_CODE}/{_rel}')

    hf_cache = modal.Volume.from_name(HF_CACHE_VOLUME, create_if_missing=True)
    app = modal.App('masked-lora-trainer')

    @app.function(
        image=image,
        gpu=DEFAULT_GPU,
        timeout=TIMEOUT_SECONDS,
        volumes={'/root/.cache/huggingface': hf_cache},
    )
    def run_trainer(argv: str, data_bytes: bytes | None = None) -> dict[str, Any]:
        try:
            return run_trainer_subprocess(argv, data_bytes=data_bytes)
        finally:
            hf_cache.commit()

    @app.local_entrypoint()
    def main(argv: str, data_file: str = '', local_out: str = '') -> None:
        payload = Path(data_file).read_bytes() if data_file else None
        result = run_trainer.remote(argv, payload)
        print(f"Remote trainer finished in {result['elapsed_seconds']:.1f}s", flush=True)
        if local_out:
            written = write_output_files(Path(local_out), result['files'])
            print(f'Copied {len(written)} output file(s) to {local_out}', flush=True)
