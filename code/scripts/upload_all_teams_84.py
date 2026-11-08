"""Publish the all-teams 84-field model (21 epochs) to the Hugging Face Hub after its battles, and prove the copy is intact.

    PYTHONPATH=~/vgc-data/pydeps-hfupload python upload_all_teams_84.py prepare [--draft] | upload | verify

prepare builds the repo folder: the checkpoint, its weights as safetensors, JSON tables, sample.py, a reference model card
and a how-to guide (Diataxis keeps those two apart). --draft allows missing battle results and cannot be uploaded.
upload makes one commit to a public repo. verify downloads every file again, compares SHA-256 and tensors, and checks that
sample.py from the download writes the same teams as the project's own model class.
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, "/tmp/vgc-pilot/src")
sys.path.insert(0, str(HERE))

REPO = "Rami/vgc-regmb-discrete-diffusion-84-fields-all-teams-ep21"
READER = "Generated, all teams, 21 epochs"
CHECKPOINT = "stat84_all_teams_ep21.pt"
CK = Path.home() / ".local/share/vgc-pilot-runtime/lossdown" / CHECKPOINT
RESULTS = Path.home() / "vgc-data/results/stat_points_84_gen84_all_ep21.json"
TRAIN_LOG = Path.home() / "vgc-data/stat_points_84/queue.log"
TABLE = Path.home() / "Documents/code/vgc-team-generator-pilot/results/lossdata84.npz"
STAGE = Path.home() / "vgc-data/stat_points_84/hub" / REPO.split("/")[1]
RECEIPT = Path.home() / "vgc-data/stat_points_84/hub/receipt.json"
PROJECT = "https://github.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi"
LOSS_EPOCHS = (1, 5, 10, 15, 20, 21)
CHECKED = ("stat84_all_teams_ep21.pt", "model.safetensors", "regulation.json", "sampling_tables.json", "config.json", "sample.py")
PRIVATE = re.compile(r"/Users/|/home/|hf_[A-Za-z0-9]{20,}|[\w.+-]+@[\w-]+\.[\w.]+")

SAMPLE_PY = '''"""Write sampled teams as Showdown pastes. Needs code/src of Battle-Factory-of-Porygon-Pi on PYTHONPATH.

    python sample.py --ask 0.5 --n 8 --seed 0 --out teams.txt
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import torch

import diffusion84 as M
import lossdown as L
from checkpoints import load_checkpoint
from encode import Vocab

HERE = Path(__file__).resolve().parent


def load():
    """The model, built from this folder: the project's own constructor also reads its private team files."""
    ck = load_checkpoint(HERE / "stat84_all_teams_ep21.pt", map_location="cpu")
    tables = json.loads((HERE / "sampling_tables.json").read_text())
    V = Vocab.from_regulation(ck["regulation"])
    model = M.Model84.__new__(M.Model84)
    nets = []
    for member in ck["members"]:
        net = M.Net84(V, **member["net_kw"])
        net.load_state_dict(member["sd"])
        nets.append((net.to(M.DEV).eval(), member["net_kw"]))
    model.V, model.nets, model.info, model.tables = V, nets, ck.get("info"), L.Tables(V, [])
    model.look = dict(V.regulation["names"])
    model.modes = {species: Counter(counts) for species, counts in tables["stat_alignment_counts"].items()}
    model.natures = set(tables["stat_alignments"])
    return model


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ask", type=float, default=0.5, help="win rate to ask for, 0 to 1; left out with --no-ask")
    p.add_argument("--no-ask", action="store_true", help="sample with no win-rate condition")
    p.add_argument("--n", type=int, default=8, help="number of teams")
    p.add_argument("--guidance", type=float, default=2.0, help="classifier-free guidance weight")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-repair", action="store_true", help="keep the generated Stat Alignment")
    p.add_argument("--out", type=Path, help="file for the pastes; standard output if left out")
    a = p.parse_args()
    model = load()
    ask = None if a.no_ask else a.ask
    rows = model.sample(a.n, ask=ask, guidance=a.guidance, temperature=a.temperature, seed=a.seed).cpu().numpy()
    text = "\\n".join(model.to_paste(r, repair=not a.no_repair)[0] for r in rows)
    if a.out: a.out.write_text(text)
    else: print(text)


if __name__ == "__main__":
    main()
'''

HOW_TO = """# How to sample teams from this model

This guide shows you how to download the model, check its files, and write sampled teams as Showdown pastes. It assumes you can use Python and read a Showdown paste. For settings, file formats and evaluation results, see the [model reference](README.md).

## 1. Download the model

```
hf download {repo} --local-dir vgc-model
```

## 2. Check the files

On macOS:

```
cd vgc-model && shasum -a 256 -c SHA256SUMS && cd ..
```

On Linux, use `sha256sum -c SHA256SUMS`. If any line says `FAILED`, download the model again.

## 3. Get the model's code

```
git clone --depth 1 {project}
python -m venv vgc-env
vgc-env/bin/pip install -r Battle-Factory-of-Porygon-Pi/code/requirements.txt
```

The code reads its game data from `/tmp/vgc-pilot/data`. To point that path at the clone, run:

```
mkdir -p /tmp/vgc-pilot && ln -s "$PWD/Battle-Factory-of-Porygon-Pi/code/data" /tmp/vgc-pilot/data
```

## 4. Sample teams

```
PYTHONPATH=Battle-Factory-of-Porygon-Pi/code/src vgc-env/bin/python vgc-model/sample.py --ask 0.5 --n 8 --seed 0 --out teams.txt
```

`teams.txt` now holds eight teams, each as six Showdown paste blocks.

## 5. Change what you sample

- To ask for a different win rate, change `--ask`. The evaluation table in the [model reference](README.md) shows what each ask measured.
- To sample with no win-rate condition, replace `--ask 0.5` with `--no-ask`.
- To keep the Stat Alignment the model generated, add `--no-repair`. The evaluation used the repaired Stat Alignment.
- To repeat a sample, keep `--seed` and your PyTorch version the same. Another PyTorch version can give different teams for the same seed.

## 6. Check that the teams are legal

The model does not guarantee a legal team. To use a team, validate it first with the Pokémon Showdown validator for the format `gen9championsvgc2026regmb`.

## Use the weights without the project's code

To read only the weights, load `model.safetensors` with `safetensors.torch.load_file`. To read the vocabulary and rule tables, load `regulation.json`.
"""


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""): h.update(block)
    return h.hexdigest()


def training_losses():
    """{epoch: (training loss, validation loss)} from the all-teams section of the training log."""
    text = TRAIN_LOG.read_text()
    section = text[text.rindex("[stat84_all_teams_es3_cap80] 298488 teams"):]
    return {int(e): (float(t), float(v)) for e, t, v in
            re.findall(r"(?:pass pass_|epoch) (\d+) · train_loss ([\d.]+) · val_loss ([\d.]+)", section)}


def labelled_training_teams():
    import numpy as np
    z = np.load(TABLE)
    return int((np.isfinite(z["all_W"]) & ~z["all_val"]).sum()), len(z["all_W"]), int(z["all_val"].sum())


def evaluation_section(ev):
    if ev is None: return "Pending: the battles have not finished. A draft of this card cannot be uploaded.\n"
    s = ev["summary"]
    from team_diversity import slots_from_paste
    short = {}
    for c in ev["cells"].values():
        short.setdefault(c["numerator"], []).extend(len(p["moves"]) < 4 for paste in c["pastes"] for p in slots_from_paste(paste))
    rows = ["| Ask | Win rate | 95% interval | Teams | Legal on first check | Pokémon under 4 moves | Effective species | Line-ups in 5 teams | Top species share |",
            "|---|---|---|---|---|---|---|---|---|"]
    for r in s["asks"]:
        rows.append(f"| {r['numerator']}/24 | {r['win']:.3f} | {r['lo']:.3f} to {r['hi']:.3f} | {r['teams']} | {r['legal']:.1%} | "
                    f"{sum(short[r['numerator']]) / len(short[r['numerator']]):.1%} | {r['effective_species']:.1f} | {r['distinct_lineups']:.2f} | {r['top_species_share']:.3f} |")
    line, sp = s["line"], s["stat_points"]
    return f"""Win rate is the share of battles won against the 49 distinct top-50 meta teams, one battle per opponent, behaviour-cloning policy on both sides, with the six Pokémon shuffled every battle. An ask is the win rate requested from the model. The diversity columns are means over 200 random subsets of 5 teams.

{chr(10).join(rows)}

| Measure | Value |
|---|---|
| Teams | {s['teams']:,} |
| Battles | {s['battles']:,} |
| Tries | {s['tries']:,} |
| Share of tries legal | {s['legal']:.1%} |
| Mean squared error between ask and win rate, asks 1/24 to 17/24 | {line['mse_scored_asks']:.4f} |
| Mean squared error between ask and win rate, asks 18/24 to 23/24 | {line['mse_asks_18_to_23']:.4f} |
| Slope of win rate on ask, all asks | {line['slope']:.3f} |
| Share of Pokémon spending exactly 66 Stat Points | {sp['share_spending_66']:.1%} |
| Mean Stat Points spent per Pokémon | {sp['mean_stat_points_spent']:.1f} |
| Share of Pokémon with fewer than four moves | {sp['share_under_four_moves']:.1%} |
| Share of Pokémon whose Stat Points exactly match a corpus set for the species | {sp['share_matching_corpus_set']:.1%} |
"""


def model_card(ck, ev, losses):
    info, reg = ck["info"], ck["regulation"]
    labelled, rows, validation = labelled_training_teams()
    params = sum(v.numel() for v in ck["members"][0]["sd"].values())
    field_names = dict(species="Species", ability="Ability", item="Item", move="Move", nature="Stat Alignment")
    sizes = "\n".join(f"| {field_names[k]} | {len(reg['values'][k]) - 1} |" for k in field_names)
    loss_rows = "\n".join(f"| {e} | {losses[e][0]:.4f} | {losses[e][1]:.4f} |" for e in LOSS_EPOCHS)
    files = "\n".join(f"| `{n}` | {d} |" for n, d in (
        (CHECKPOINT, "The checkpoint the project's code loads. PyTorch format. Load it only with `weights_only=True`."),
        ("model.safetensors", "The same weights in safetensors format."),
        ("regulation.json", "The vocabulary and the per-species rule tables, as stored in the checkpoint."),
        ("sampling_tables.json", "Stat Alignment counts per species, read by the Stat Alignment repair."),
        ("config.json", "Network size, training settings and the training and validation loss per epoch."),
        ("evaluation.json", "The evaluation results below, as data."),
        ("sample.py", "Writes sampled teams as Showdown pastes."),
        ("SHA256SUMS", "SHA-256 hashes of the model and table files."),
        ("HOW-TO.md", "How to sample teams from this model.")))
    return f"""---
license: mit
library_name: pytorch
language:
- en
tags:
- pytorch
- safetensors
- discrete-diffusion
- pokemon
- vgc
- pokemon-showdown
- stat-points
- epochs-21
---

# Discrete diffusion model for Pokémon Champions VGC Regulation M-B teams: all 84 fields, all teams, 21 epochs

Model reference. To sample teams, see [How to sample teams from this model](HOW-TO.md). Project: [Battle Factory of Porygon-Pi]({PROJECT}).

| Property | Value |
|---|---|
| Reader name | {READER} |
| Code ID | `gen84_all_ep21` |
| Format | `{reg['format']}` |
| Epochs trained | 21 |
| Fields generated | 84 |
| Fields copied | None |
| Field changed after sampling | Stat Alignment, repaired to match the moves |
| Trained from | Scratch |

## Fields

A team is six Pokémon. Each Pokémon has 14 fields: species, ability, item, four moves, Stat Alignment and six Stat Points (HP, Atk, Def, SpA, SpD, Spe). Sampling decodes species first and Stat Points last.

| Field type | Values |
|---|---|
{sizes}
| Stat Point | 33 (0 to 32) |

Sampling allows a Stat Point only if it keeps that stat at 32 or less and the Pokémon's total at 66 or less. It does not require a total of 66.

## Files

| File | Contents |
|---|---|
{files}

## Network

| Property | Value |
|---|---|
| Type | Transformer encoder over the 84 fields |
| Width | {info['net_kw']['d']} |
| Layers | {info['net_kw']['nlayer']} |
| Attention heads | {info['net_kw']['nhead']} |
| Parameters | {params:,} |
| Conditions | Noise level; win-rate ask, or no ask |

## Training teams

| Count | Teams |
|---|---|
| All-teams set | {rows:,} |
| Validation split, seeded 2% | {validation:,} |
| Trained on, after the rule tables dropped {info['dropped_by_tables']} | {info['n_train']:,} |
| With a win-rate label, before that drop | {labelled:,} |

The set holds real corpus teams, earlier search teams, teams from the project's battle records and 100,000 uniform-legal teams, keeping a team only if it has one set of Stat Points. A label is the team's pooled wins divided by its pooled battles. The test 500 teams and the held-out meta teams ranked 51 to 100 are excluded.

## Training

| Setting | Value |
|---|---|
| Batch size | {info['batch']} |
| Learning rate | {info['lr']} |
| Seed | {info['seed']} |
| Masking exponents: species, ability, item, move, Stat Alignment | {info['sched']} |
| Masking exponent: Stat Points | 1 |
| Planned stopping rule | 3 epochs with no new lowest validation loss |
| Planned cap | {info['cap']} epochs |
| Actual stop | Stopped by hand at epoch 21. The stopping rule had not fired. |
| Kept checkpoint | Epoch {info['best_epoch']}, the lowest validation loss |

| Epoch | Training loss | Validation loss |
|---|---|---|
{loss_rows}

## Sampling settings used in the evaluation

| Setting | Value |
|---|---|
| Asks | 1/24 to 23/24 |
| Sampling seeds | 3 |
| Guidance | 2 |
| Temperature | 1 |
| Teams per ask and seed | 64 that pass the Showdown validator and are not held-out teams |
| Tries per ask and seed | At most 1,280 |

## Evaluation

{evaluation_section(ev)}
## Limitations

- These win rates hold only against these 49 teams under this behaviour-cloning policy.
- The battles that produced these win rates are the only battles these teams have played. They have not been re-battled or scored against the held-out teams.
- Generated Stat Points differ from the training teams: 99.3% of Pokémon in the training teams spend exactly 66.
- Training stopped before its stopping rule, while validation loss was still falling.
- The model does not guarantee a legal team.
- Output depends on the exact ask, not only its size. In the training teams with labels from 0.25 to 0.7, 0.04% of Pokémon have fewer than four moves when the label is a multiple of 1/24 only, and 3.71% when it is a multiple of 1/50 only. In samples of 256 teams at guidance 2, 0.8% of Pokémon have fewer than four moves at ask 11/24 and 14.4% at ask 11/24 + 0.002. Ask 12/24 is the only ask in the evaluation that is a multiple of both.
- `sample.py` needs the project's code, its game data at `/tmp/vgc-pilot/data` and PyTorch. Sampled teams for one seed depend on the PyTorch version.
"""


def build_sampling_tables():
    import alignment as AL
    import asked_vs_got as X
    corpus = X.setup()[0]
    _, _, _, modes, natures = AL.build_tables(corpus)
    return dict(stat_alignment_counts={s: dict(c) for s, c in modes.items()}, stat_alignments=sorted(natures))


def battle_results(draft):
    """The finished evaluation of this checkpoint, or None for a draft that has none yet."""
    ev = json.loads(RESULTS.read_text()) if RESULTS.exists() else None
    if ev is not None and "summary" not in ev: ev = None
    if ev is None and not draft: raise SystemExit("no battle results yet: only --draft can run")
    if ev is not None and ev["summary"]["battles"] != 216384 and not draft:
        raise SystemExit(f"expected 216,384 battles, found {ev['summary']['battles']:,}")
    if ev is not None and ev["checkpoint_sha256"] != sha256(CK): raise SystemExit("the battled checkpoint is not this file")
    return ev


def refuse_private_text():
    for f in STAGE.iterdir():
        if f.suffix in (".md", ".json", ".py") or f.name == "SHA256SUMS":
            hit = PRIVATE.search(f.read_text())
            if hit: raise SystemExit(f"{f.name} holds something private: {hit.group(0)}")


def prepare(a):
    from checkpoints import load_checkpoint
    from safetensors.torch import save_file

    ev = battle_results(a.draft)
    ck = load_checkpoint(CK, map_location="cpu")
    if len(ck["members"]) != 1: raise SystemExit("expected one network")
    if ck["info"]["best_epoch"] != 21: raise SystemExit(f"expected epoch 21, found {ck['info']['best_epoch']}")
    losses = training_losses()
    if max(losses) != 21: raise SystemExit(f"the log ends at epoch {max(losses)}")

    shutil.rmtree(STAGE, ignore_errors=True)
    STAGE.mkdir(parents=True)
    shutil.copy2(CK, STAGE / CHECKPOINT)
    save_file({k: v.clone().contiguous() for k, v in ck["members"][0]["sd"].items()}, STAGE / "model.safetensors",
              metadata=dict(format="pt", code_id="gen84_all_ep21", epochs_trained="21"))
    write = lambda name, text: (STAGE / name).write_text(text)
    write("regulation.json", json.dumps(ck["regulation"], ensure_ascii=False))
    write("sampling_tables.json", json.dumps(build_sampling_tables()))
    write("config.json", json.dumps(dict(code_id="gen84_all_ep21", reader_name=READER, epochs_trained=21, net=ck["info"]["net_kw"],
                                         training=ck["info"], loss_per_epoch={e: dict(training=t, validation=v) for e, (t, v) in losses.items()}), indent=1))
    write("sample.py", SAMPLE_PY)
    write("README.md", model_card(ck, ev, losses))
    write("HOW-TO.md", HOW_TO.format(repo=REPO, project=PROJECT))
    if ev is not None:
        keep = {k: ev[k] for k in ("model", "reader", "epochs_trained", "stopped", "checkpoint", "checkpoint_sha256", "guidance", "keep", "max_tries", "seed0", "summary")}
        write("evaluation.json", json.dumps(keep, indent=1))
    write("SHA256SUMS", "".join(f"{sha256(STAGE / n)}  {n}\n" for n in CHECKED))
    refuse_private_text()
    if ev is None: (STAGE / ".draft").touch()
    print("prepared", STAGE, sorted(p.name for p in STAGE.iterdir()))


def upload(_a):
    from huggingface_hub import HfApi
    if not STAGE.exists() or (STAGE / ".draft").exists(): raise SystemExit("run prepare without --draft first")
    if RECEIPT.exists(): raise SystemExit(f"already uploaded: {RECEIPT}")
    api = HfApi()
    api.create_repo(REPO, repo_type="model", private=False, exist_ok=True)
    commit = api.upload_folder(repo_id=REPO, folder_path=STAGE, commit_message="All-teams 84-field discrete diffusion model, 21 epochs, with evaluation")
    print("uploaded", commit.commit_url)
    (STAGE.parent / "commit.txt").write_text(commit.oid + "\n")


def same_teams_as_project_model(folder, python=sys.executable):
    """sample.py from folder and the project's Model84 write the same pastes for the same seed."""
    import diffusion84 as M
    env = dict(os.environ, PYTHONPATH=str(HERE))
    out = Path(tempfile.mkdtemp()) / "teams.txt"
    subprocess.run([python, str(Path(folder) / "sample.py"), "--ask", "0.5", "--n", "4", "--seed", "7", "--out", str(out)], check=True, env=env)
    m = M.Model84.load(CK)
    rows = m.sample(4, ask=0.5, guidance=2.0, seed=7).cpu().numpy()
    return out.read_text() == "\n".join(m.to_paste(r, repair=True)[0] for r in rows)


def verify(_a):
    import torch
    from checkpoints import load_checkpoint
    from huggingface_hub import HfApi, snapshot_download
    from safetensors.torch import load_file

    api = HfApi()
    expected = (STAGE.parent / "commit.txt").read_text().strip()
    info = api.repo_info(REPO, revision=expected, files_metadata=True)
    tmp = Path(tempfile.mkdtemp(prefix="hub-verify-"))
    folder = Path(snapshot_download(REPO, revision=expected, local_dir=tmp / "repo", force_download=True))
    hub = {s.rfilename: s for s in info.siblings}
    local = {p.name: sha256(p) for p in STAGE.iterdir()}
    missing = sorted(set(local) - set(hub))
    extra = sorted(set(hub) - set(local) - {".gitattributes"})      # an older file left in the public repository stays public
    wrong = sorted(n for n, h in local.items() if n in hub and sha256(folder / n) != h)
    lfs_wrong = sorted(n for n, s in hub.items() if s.lfs and n in local and s.lfs.sha256 != local[n])
    if sha256(folder / CHECKPOINT) != sha256(CK): wrong.append("checkpoint differs from the training output")
    a = load_checkpoint(folder / CHECKPOINT, map_location="cpu")["members"][0]["sd"]
    b = load_checkpoint(CK, map_location="cpu")["members"][0]["sd"]
    c = load_file(folder / "model.safetensors")
    tensors = set(a) == set(b) == set(c) and all(torch.equal(a[k], b[k]) and torch.equal(a[k], c[k]) for k in a)
    teams = same_teams_as_project_model(folder)
    ok = not (missing or extra or wrong or lfs_wrong) and tensors and teams
    receipt = dict(repo=REPO, url=f"https://huggingface.co/{REPO}", commit=expected, private=info.private, checkpoint_sha256=sha256(CK),
                   files=local, missing=missing, extra_on_hub=extra, wrong_hash=wrong, wrong_lfs_hash=lfs_wrong, tensors_equal=tensors, sample_py_matches_project_model=teams, passed=ok)
    RECEIPT.write_text(json.dumps(receipt, indent=1))
    shutil.rmtree(tmp, ignore_errors=True)
    print(json.dumps({k: v for k, v in receipt.items() if k != "files"}, indent=1))
    if not ok: raise SystemExit("VERIFY FAILED")
    print("HUB_VERIFIED")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare").add_argument("--draft", action="store_true")
    sub.add_parser("upload"); sub.add_parser("verify")
    args = p.parse_args()
    globals()[args.command](args)
