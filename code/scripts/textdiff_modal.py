"""src/textdiff.py on Modal's NVIDIA GPUs instead of the M4 Pro: any textdiff.py command line, unchanged.

    modal setup                                                     # once: signs this laptop in to Modal
    modal run scripts/textdiff_modal.py::upload                     # train.jsonl / val.jsonl -> the vgc-textdiff volume
    modal run scripts/textdiff_modal.py::upload --folder llmformat/regmb       # an LLM-format version's folder
    modal run --detach scripts/textdiff_modal.py::run --argv "train --name qwen3b_r32 --base Qwen/Qwen2.5-3B-Instruct --limit 8000"
    modal volume get vgc-textdiff qwen3b_r32 ~/.local/share/vgc-pilot-runtime/textdiff/

The volume is mounted where textdiff.py keeps its runtime folder (HOME is /root in the container), so OUT, --name and
--data mean what they mean on the laptop; an LLM-format run passes --data /root/.local/share/vgc-pilot-runtime/textdiff/
llmformat/regmb. --detach keeps the run going after the laptop disconnects; `modal app logs textdiff` follows it.
Package pins are venv-dlm's (transformers 5 breaks the released remote code). H100: $3.95/h on the Starter plan's
$30 a month of credit (modal.com/pricing, 2026-10-23); the timeout caps one call at $16.
"""
import shlex, subprocess, time
from pathlib import Path

import modal

SRC = Path(__file__).resolve().parents[1] / "src"
LOCAL = Path.home() / ".local/share/vgc-pilot-runtime/textdiff"
REMOTE = "/root/.local/share/vgc-pilot-runtime/textdiff"

app = modal.App("textdiff")
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch==2.14.1", "transformers==4.57.3", "peft==0.18.1", "accelerate==1.15.0", "safetensors==0.8.0",
                      "einops==0.8.2", "numpy==2.4.6", "huggingface_hub==0.36.2")
         .env({"PYTHONUNBUFFERED": "1"})
         .add_local_file(SRC / "textdiff.py", "/root/src/textdiff.py")
         .add_local_file(SRC / "textdiff_port.py", "/root/src/textdiff_port.py"))
data = modal.Volume.from_name("vgc-textdiff", create_if_missing=True)
hf = modal.Volume.from_name("vgc-hf-cache", create_if_missing=True)          # model downloads, kept between runs


@app.function(image=image, gpu="H100", timeout=4 * 3600, volumes={REMOTE: data, "/root/.cache/huggingface": hf})
def textdiff(argv: str):
    p = subprocess.Popen(["python", "textdiff.py", *shlex.split(argv)], cwd="/root/src")
    while p.poll() is None:
        time.sleep(120)
        data.commit()                           # the step adapters (every --eval-every) survive a crash or a timeout
    data.commit(); hf.commit()
    if p.returncode: raise RuntimeError(f"textdiff.py exited with {p.returncode}")


@app.local_entrypoint()
def run(argv: str):
    textdiff.remote(argv)


@app.local_entrypoint()
def upload(folder: str = ""):
    """folder: '' = the first team writer's examples; 'llmformat/<version>' = one LLM-format version's."""
    with data.batch_upload(force=True) as b:
        for name in ("train.jsonl", "val.jsonl"):
            b.put_file(LOCAL / folder / name, f"/{folder}/{name}".replace("//", "/"))
    print(f"uploaded {LOCAL / folder} -> vgc-textdiff:/{folder}")
