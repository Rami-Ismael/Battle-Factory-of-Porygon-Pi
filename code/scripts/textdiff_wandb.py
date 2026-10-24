"""Weights & Biases record of the text writers: training curves and inference benchmarks, logged from the laptop.

    uvx --with wandb python scripts/textdiff_wandb.py train qwen3b_r32 qwen15_r32 --device "H100 (Modal)"
    uvx --with wandb python scripts/textdiff_wandb.py train lora_r32 --device "M4 Pro (MPS)"
    uvx --with wandb python scripts/textdiff_wandb.py bench --model qwen3b_r32 --device h100 \
        --speed bench_h100.log --legality legality_h100.jsonl

train: one run per adapter folder (OUT/<name>/log.json from textdiff.py train): step loss, learning rate, tokens/s and
peak memory every 25 steps, validation loss every --eval-every steps, and the run's arguments as config.
battle: the win-rate table of every scored cell in results/textdiff.json (textdiff_battle.py report).
bench: one run per (model, device) with a table of sampler settings: seconds per team, forward passes, tokens, and the
raw legality and species-set count of the same texts. --speed takes the JSON lines textdiff.py bench prints (any log
holding them), --legality the lines textdiff_battle.py bench prints.

Each run carries a sync_key in its config (train-<name>, bench-<model>-<device>[-<tag>]); a second sync skips keys already
logged, and --force deletes that run and logs a new one (W&B never reuses a deleted run's id). The key stays on the laptop
(wandb login); nothing here runs on Modal.
"""
import argparse, json
from pathlib import Path

import wandb

OUT = Path.home() / ".local/share/vgc-pilot-runtime/textdiff"
PROJECT, ENTITY = "vgc-textdiff", "compress_rl"      # the account's workspace (the key cannot write to the personal entity)


def json_lines(path, key):
    out = []
    for line in open(path):
        line = line.strip()
        if line.startswith("{") and f'"{key}"' in line:
            out.append(json.loads(line))
    return out


def fresh(key, force):
    """True when the run should be logged now: no run carries this sync_key yet, or --force deleted the one that did."""
    try:
        runs = list(wandb.Api().runs(f"{ENTITY}/{PROJECT}", filters={"config.sync_key": key}))
    except (ValueError, wandb.errors.CommError):            # no project yet
        runs = []
    if runs and not force:
        print(f"{key}: already logged, skipped (--force to replace)"); return False
    for r in runs: r.delete()
    return True


def train(args):
    for name in args.names:
        d = json.loads((OUT / name / "log.json").read_text()); a = d["args"]
        run_id = f"train-{name}"
        if not fresh(run_id, args.force): continue
        run = wandb.init(entity=ENTITY, project=PROJECT, name=name, job_type="train", group="bigger-model-2026-10-23",
                         config={**a, "sync_key": run_id, "device": args.device, "backbone": a.get("base") or "Efficient-Large-Model/Fast_dLLM_v2_1.5B",
                                 "minutes": d.get("minutes")})
        for e in d["log"]:
            row = {k: e[k] for k in ("loss", "lr", "tok_per_s", "peak_gb") if k in e}
            if "val" in e: row["val_loss"] = e["val"]
            run.log(row, step=e["step"])
        run.summary["final_val_loss"] = [e["val"] for e in d["log"] if "val" in e][-1]
        run.summary["minutes"] = d.get("minutes")
        run.finish(); print(f"{run_id}: {len(d['log'])} log entries")


def bench(args):
    run_id = f"bench-{args.model}-{args.device}" + (f"-{args.tag}" if args.tag else "")
    if not fresh(run_id, args.force): return
    speed = {r["setting"]: r for path in args.speed for r in json_lines(path, "seconds_per_team") if "forwards_per_team" in r}
    legal = {r["setting"]: r for path in args.legality for r in json_lines(path, "species_sets")}
    cols = ["setting", "batch", "block_cache", "threshold", "temperature", "top_p", "quant", "seconds_per_team", "teams_per_hour",
            "forwards_per_team", "tokens_per_team", "legal", "legal_rate", "species_sets", "memory_gb"]
    table = wandb.Table(columns=cols)
    for s, r in speed.items():
        batch = 1 if s.startswith("released") else int(s.removeprefix("batch").split("_")[0])
        v = legal.get(s, {}).get("valid", "")
        rate = int(v.split("/")[0]) / int(v.split("/")[1]) if v else None
        table.add_data(s, batch, "_cache" in s, r.get("threshold", 0.9), r.get("temperature", 1.0), r.get("top_p", 1.0),
                       r.get("quant", "none"), r["seconds_per_team"], 3600 / r["seconds_per_team"],
                       r["forwards_per_team"], r["tokens_per_team"], v, rate, legal.get(s, {}).get("species_sets"), r.get("memory_gb"))
    run = wandb.init(entity=ENTITY, project=PROJECT, name=run_id, job_type="inference", group="bigger-model-2026-10-23",
                     config=dict(sync_key=run_id, model=args.model, device=args.device, tag=args.tag, teams_per_setting=64))
    run.log({"settings": table,
             "seconds_per_team": wandb.plot.bar(table, "setting", "seconds_per_team", title=f"{args.device}: seconds per team"),
             "legal_rate": wandb.plot.bar(table, "setting", "legal_rate", title=f"{args.device}: raw legality")})
    best = min(speed.values(), key=lambda r: r["seconds_per_team"])
    run.summary.update(dict(fastest_setting=best["setting"], fastest_seconds_per_team=best["seconds_per_team"],
                            released_seconds_per_team=speed.get("released", {}).get("seconds_per_team")))
    run.finish(); print(f"{run_id}: {len(speed)} settings")


def battle(args):
    """Win rate against the top-50 meta of every scored cell in results/textdiff.json (textdiff_battle.py report): the
    scratch model, the 1.5B writer (text:) and the 3B writer (text3b:), one table."""
    run_id = f"battle-{args.tag}"
    if not fresh(run_id, args.force): return
    summ = json.loads(Path(args.results).read_text())["summary"]
    cols = ["cell", "writer", "arm", "win_rate", "se", "teams", "p90", "best", "above_0531", "raw_legal", "species_sets"]
    table = wandb.Table(columns=cols)
    for cell, v in summ.items():
        table.add_data(cell, cell.split(":")[0], v["arm"], v["got"], v["se"], v["n"], v["p90"], v["best"], v["above_531"],
                       v["valid"], v["sets"])
    run = wandb.init(entity=ENTITY, project=PROJECT, name=run_id, job_type="battle", group="bigger-model-2026-10-23",
                     config=dict(sync_key=run_id, protocol="64 valid teams x top-50 meta, one battle per opponent, BC policy both sides"))
    run.log({"win_rates": table, "win_rate": wandb.plot.bar(table, "cell", "win_rate", title="win rate vs top-50 meta")})
    run.finish(); print(f"{run_id}: {len(summ)} cells")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    t = sub.add_parser("train"); t.add_argument("names", nargs="+"); t.add_argument("--device", required=True)
    b = sub.add_parser("bench"); b.add_argument("--model", required=True); b.add_argument("--device", required=True)
    b.add_argument("--tag", default="", help="e.g. t0.7: a run per sampler threshold, so setting names do not collide")
    b.add_argument("--speed", nargs="+", required=True); b.add_argument("--legality", nargs="*", default=[])
    w = sub.add_parser("battle"); w.add_argument("--results", required=True); w.add_argument("--tag", default="textdiff-3b-2026-10-23")
    for q in (t, b, w): q.add_argument("--force", action="store_true")
    a = p.parse_args()
    globals()[a.command](a)
