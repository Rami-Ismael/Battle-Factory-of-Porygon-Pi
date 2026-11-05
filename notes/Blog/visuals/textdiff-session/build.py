"""Builds index.html for the text-diffusion session explainer (PROTOTYPE: three variants behind ?variant=A|B|C).

    python3 build.py

Every number on the page comes from a file read here: sources/ (result lines copied from the 2026-10-23..24 run logs,
results/textdiff.json, the random grid's text_vs_ling.json) and each run's log.json under
~/.local/share/vgc-pilot-runtime/textdiff/. Each value carries its file name into the page's captions. Sprites come from
../search-loop/sprites (Showdown gen5) and ../item-sprites, embedded as data URIs.
"""
import base64, json, re
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "sources"
RUNS = Path.home() / ".local/share/vgc-pilot-runtime/textdiff"
SPRITES, ITEMS = HERE.parent / "search-loop/sprites", HERE.parent / "item-sprites"
norm = lambda s: re.sub(r"[^a-z0-9]", "", (s or "").lower())


def jsonl(name):
    return [json.loads(l) for l in open(SRC / name)]


def log(name):
    d = json.loads((RUNS / name / "log.json").read_text())
    steps = {e["step"]: e for e in d["log"] if "epoch" in e}
    vals = {e["step"]: e["val"] for e in d["log"] if "val" in e}
    ordered = sorted(steps); t = n = 0.0
    for a, b in zip(ordered, ordered[1:]):          # steady seconds per step: 25-step windows with no compile or validation
        if a >= 50 and a % 500 != 0: t += steps[b]["minutes"] - steps[a]["minutes"]; n += b - a
    return dict(file=f"~/.local/share/vgc-pilot-runtime/textdiff/{name}/log.json", val=[[k, round(v, 4)] for k, v in sorted(vals.items())],
                steady=round(t * 60 / n, 3) if n else None, peak=round(max(e["peak_gb"] for e in steps.values() if "peak_gb" in e), 1) if any("peak_gb" in e for e in steps.values()) else None,
                minutes=round(d.get("minutes") or 0, 1))


def data_uri(path):
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode() if path.exists() else None


def parse_team(paste):
    team = []
    for block in paste.strip().split("\n\n"):
        head = block.splitlines()[0]
        species, _, item = head.partition(" @ ")
        species = re.sub(r"\s*\(.*?\)\s*", "", species).strip()
        moves = [l[2:].strip() for l in block.splitlines() if l.startswith("- ")]
        team.append(dict(species=species, item=item.strip(), moves=moves,
                         sprite=data_uri(SPRITES / f"{norm(species)}.png"), item_sprite=data_uri(ITEMS / f"item-{norm(item)}.png")))
    return team


def main():
    results = json.loads((SRC / "textdiff_results.json").read_text())
    summ, cells = results["summary"], results["cells"]
    arms = {"base": "target unknown", "ask50": "target 0.50", "ask60": "target 0.60", "ask50_g2": "target 0.50, guidance 2"}
    win = []
    for writer, label in (("scratch", "scratch diffusion"), ("text", "1.5B text writer"), ("text3b", "3B text writer")):
        for arm in ("base", "cfg", "ask50", "ask60", "ask50_g2"):
            v = summ.get(f"{writer}:{arm}")
            if v: win.append(dict(writer=writer, label=label, arm=arms.get(arm, "guidance 2 (scratch's guided arm)"), key=f"{writer}:{arm}",
                                  win=round(v["got"], 3), se=round(v["se"], 3), legal=round(v["valid"], 3), sets=v["sets"]))
    teams = {}
    for writer in ("scratch", "text", "text3b"):
        best = max(((y, c["pastes"][i], k) for k, c in cells.items() if k.startswith(writer + ":") and c.get("y")
                    for i, y in enumerate(c["y"]) if y is not None), key=lambda x: x[0])
        teams[writer] = dict(win=round(best[0], 3), cell=best[2], team=parse_team(best[1]), paste=best[1])
    ceiling = [r for r in jsonl("h100_batch_ceiling.jsonl") if r.get("dtype", "bf16") == "bf16"]
    h100 = {r["setting"]: r for r in jsonl("h100_samplers.jsonl")}
    mac = {r["setting"]: r for r in jsonl("m4pro_samplers.jsonl") if "forwards_per_team" in r}
    sweep = {r["setting"]: r for r in jsonl("h100_sampler_sweep.jsonl")}
    legal = {}
    for r in jsonl("legality_h100.jsonl"): legal[r["setting"]] = r["valid"]
    checks = jsonl("checks.jsonl")
    ling = json.loads((SRC / "text_vs_ling_random_grid.json").read_text())["groups"]["all"]
    pick = lambda d: dict(mean=round(d["mean"] * 100, 1), lo=round(d["lo"] * 100, 1), hi=round(d["hi"] * 100, 1))
    example = json.loads((SRC / "winrate_example.json").read_text())   # one team's 49 cells, copied from the matchup database
    arm = cells[example["cell"]]
    winrate = dict(games=[dict(won=g["won"], battles=g["battles"], weight=g["weight"], species=g["species"]) for g in example["games"]],
                   cell=example["cell"], arm_scores=[round(y, 3) for y in arm["y"] if y is not None],
                   arm_mean=round(summ[example["cell"]]["got"], 3), arm_se=round(summ[example["cell"]]["se"], 3))
    baseline = json.loads((SRC / "baseline_results.json").read_text())       # written by baseline_extract.py
    for slot in baseline["example"]["team"]:
        slot.update(sprite=data_uri(SPRITES / f"{norm(slot['species'])}.png"), item_sprite=data_uri(ITEMS / f"item-{norm(slot['item'])}.png"))
    prompt = json.loads((SRC / "prompt_example.json").read_text())          # the training prompt and one real training row
    ling_calls = json.loads((SRC / "ling_calls.json").read_text())         # Ling's 810 random-grid calls, summarised per arm
    targets = json.loads((SRC / "train_targets.json").read_text())        # the target win rates in the 8,000 training rows
    dspy = json.loads((SRC / "dspy_results.json").read_text())             # DSPy on Mercury, from docs/dspy-structure-experiment.md
    earlier = json.loads((SRC / "earlier_studies.json").read_text())      # the 2026-10-04..10 baseline studies, from their docs
    data = dict(winrate=winrate, baseline=baseline, prompt=prompt, ling_calls=ling_calls, targets=targets, dspy=dspy, earlier=earlier,
        win=win, meta_mean=0.531, teams=teams,
        curves={k: log(k) for k in ("lora_r32", "qwen3b_r32", "qwen15_r32")},
        train={k: log(k) for k in ("qwen3b_r32", "qwen3b_r32_tf32", "qwen3b_r32_amp", "qwen3b_r32_amp_ckpt")},
        ceiling=[dict(batch=int(r["setting"].removeprefix("batch")) if "batch" not in r else r["batch"], oom=bool(r.get("oom")),
                      s=r.get("seconds_per_team"), gb=r.get("memory_gb"), error=r.get("error")) for r in ceiling],
        h100=[dict(setting=s, s=h100[s]["seconds_per_team"]) for s in ("released", "batch1", "batch8", "batch32", "batch64") if s in h100],
        mac=[dict(setting=s, s=mac[s]["seconds_per_team"], legal=None) for s in ("released", "batch1", "batch1_cache", "batch8_cache", "batch32", "batch64")
             if s in mac],
        samplers=[dict(setting=s, T=sweep[s]["temperature"], p=sweep[s]["top_p"], s=sweep[s]["seconds_per_team"], legal=legal.get(s))
                  for s in ("batch128_T0_p1", "batch128_T1_p1", "batch128_T1_p0.9", "batch128_T0.7_p0.9", "batch128_T0.2_p0.95") if s in sweep],
        checks=checks,
        ling=dict(legal_text=ling["legal"]["text"], legal_ling=ling["legal"]["regmb@16384"], n=ling["n"],
                  net_text=pick(ling["net_text"]), net_ling=pick(ling["net_regmb@16384"]), diff=pick(ling["net_text_minus_regmb@16384"])),
    )
    html = (HERE / "template.html").read_text().replace("/*DATA*/null", json.dumps(data))
    (HERE / "index.html").write_text(html)
    missing = [m["species"] for t in teams.values() for m in t["team"] if not m["sprite"]]
    print(f"index.html {len(html) / 1e6:.2f} MB; teams {[(w, t['win']) for w, t in teams.items()]}; sprites missing: {missing or 'none'}")


if __name__ == "__main__":
    main()
