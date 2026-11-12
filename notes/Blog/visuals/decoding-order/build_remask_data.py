"""Data for the remasking section of the decoding-order page (2026-10-31). Writes remask.json beside this file;
build.py merges it into the page.

    /tmp/vgc-pilot/.venv/bin/python build_remask_data.py

Sources:
- ~/vgc-data/results/remask_random_order.json: per cell legality, kept teams and battled win rates for four remasking
  arms on random decoding order, and its summary (control = the decoding-order experiment's random-order cells, one
  field per pass; reference = species-first order). Pre-registration: code/docs/remask-random-order.md.
- Remasking boards: fresh teams at ask 0.5 with the run's settings for random 20% remasked (random decoding order,
  48 passes, guidance 2, temperature 1), recording each pass's new field, its remasked fields with their old values,
  whether the pass was undone, and which filled fields the legal Pokémon construction tables say clash with the rest of the team.
  Two boards: the first draw in which a clash appeared and was gone by the last pass with Showdown accepting the team,
  and the first draw Showdown rejected. They illustrate remasking; no number on the page comes from them.
"""
import base64
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
SRC = HERE.parents[3] / "code" / "src"
sys.path.insert(0, "/tmp/vgc-pilot/src")
sys.path.insert(0, str(SRC))
import discrete_diffusion_model as M

sys.path.insert(0, str(SRC))                        # discrete_diffusion_model's imports put /tmp/vgc-pilot/src first again
import lossdown as L
import remask_random_order as RRO
from hps_generate import Validator

from build_data import add_names, team

RES = Path.home() / "vgc-data" / "results"
DEX = Path.home() / ".local/share/vgc-pilot-runtime/validator-913da36/dist/data"   # Showdown's data files
SPRITES, ITEMS = HERE.parent / "search-loop" / "sprites", HERE.parent / "item-sprites"
ARMS = [("control", "No remasking", None), ("lowest10", "Least-likely 10% remasked, 48 passes", "lowest10"),
        ("random10", "Random 10% remasked, 48 passes", "random10"), ("random20", "Random 20% remasked, 48 passes", "random20"),
        ("random10_96", "Random 10% remasked, 96 passes", "random10_96")]
MEASURES = ("legal", "win", "ask_slope", "gap", "effective_species", "distinct_lineups", "top_species_share",
            "mean_distance", "copies", "new_lineups", "lineup_distance")
BOARD_ARM, BOARD_REMASK, BOARD_SEED, BOARD_ASK, BOARD_TRIES = "random20", 0.20, 7311, 0.5, 64


def showdown_names(ids):
    """Display names from Showdown's data files for ids no paste named (values a remask redraw replaced)."""
    js = ("const D=process.argv[1],ids=JSON.parse(process.argv[2]),o={};"
          "const T=['pokedex','moves','items','abilities','natures'].map(f=>Object.values(require(D+'/'+f+'.js'))[0]);"
          "for(const i of ids){const e=T.map(t=>t[i]).find(Boolean);if(e)o[i]=e.name;}console.log(JSON.stringify(o));")
    return json.loads(subprocess.run(["node", "-e", js, str(DEX), json.dumps(sorted(ids))], capture_output=True, text=True,
                                     check=True).stdout)


def stat(s, k=6):
    return {"mean": round(s["mean"], k), "sd": round(s["sd"], k), "seeds": [round(x, k) for x in s["seeds"]]}


def vs(s, k=6):
    return {"diff": round(s["diff"], k), "noise": round(s["noise"], k), "seeds": [round(x, k) for x in s["seeds"]]}


def clashes(w, x):
    """Filled columns of the one team in x whose value the legal Pokémon construction tables rule out given every other filled field."""
    out = []
    for c in (x[0] != 0).nonzero().flatten().tolist():
        y = x.clone(); y[0, c] = 0
        legal, empty = w._legal(y, c)
        if bool(empty[0]) or not bool(legal[0, int(x[0, c])]): out.append(c)
    return out


def board(w, val, seed):
    """One team at BOARD_ASK with random 20% remasked: per pass, its new and remasked fields, old values of the
    remasked fields, the 48 values after the pass, whether the pass was undone, and the clashing fields."""
    word = lambda c, v: "" if int(v) == 0 else w.V.itos[L.KEY_OF_COL[c]][int(v)]
    passes, real_fill = [], w._fill_pass

    def fill(x, conf, targets, due, redo, *a, **k):
        before = a[3]                                    # _fill_pass's `old`: the team before this pass masked anything
        undone = real_fill(x, conf, targets, due, redo, *a, **k)
        passes.append({"new": due[0].nonzero().flatten().tolist(), "redo": redo[0].nonzero().flatten().tolist(),
                       "was": {c: word(c, before[0, c]) for c in redo[0].nonzero().flatten().tolist()},
                       "values": [word(c, x[0, c]) for c in range(L.COLS)], "undone": bool(undone),
                       "clash": clashes(w, x)})
        return undone

    M._lowest_confidence, w._fill_pass = RRO.random_pick, fill
    try:
        torch.manual_seed(seed)
        row = w.sample(1, ask=BOARD_ASK, guidance=2.0, order="random", remask=BOARD_REMASK, steps=48, stats={})[0]
    finally:
        M._lowest_confidence = RRO.LOWEST
        del w._fill_pass
    row = row.cpu().numpy()
    paste, _ = w.to_paste(row, np.random.default_rng(seed), repair=True)
    err = val(paste)
    return {"passes": passes, "legal": err is None, "paste": paste, "seed": seed,
            "verdict": None if err is None else str(err).split("\n")[0][:400],
            "ever_clash": any(p["clash"] for p in passes), "end_clash": passes[-1]["clash"]}


def main():
    run = json.loads((RES / "remask_random_order.json").read_text())
    cells, summ, cfg = run["cells"], run["summary"], run["config"]
    out = {"asks": cfg["asks"], "seeds": cfg["seeds"], "keep": cfg["keep"], "max_tries": cfg["max_tries"],
           "rescue_floor": 0.5, "species": {m: stat(summ["species"][m]) for m in MEASURES}, "arms": []}
    species, items, names = set(), set(), {}
    for key, name, code in ARMS:
        s = summ[key]
        o = {"key": key, "name": name, "code": code, "kept": s["kept"], "tried": s["tried"], "capped": s["capped"],
             "undone": s["undone"], "rescues": s["rescues"], **{m: stat(s[m]) for m in MEASURES}}
        o["vs_species"] = {m: vs(v) for m, v in s.get("vs_species", {}).items()}
        o["vs_control"] = {m: vs(v) for m, v in s.get("vs_control", {}).items()}
        cs = [c for c in cells.values() if c["arm"] == key]
        if cs:
            o["legal_teams"] = sum(c["tried"] - c["rejected"] for c in cs)
            o["seconds"] = round(sum(c["seconds"] for c in cs), 1)
            pastes = [p for c in sorted(cs, key=lambda c: (c["seed"], abs(c["ask"] - 0.5))) if c["seed"] == 1
                      for p in c["pastes"] if team(p)][:6]
            o["examples"] = [team(p) for p in pastes]
            for p in pastes: add_names(p, names)
            for t in o["examples"]:
                species |= {x["species"] for x in t}; items |= {x["item"] for x in t if x["item"]}
        out["arms"].append(o)
    w, val = M.DiscreteDiffusionModel.load(), Validator()
    boards = {}
    try:
        for i in range(BOARD_TRIES):
            b = board(w, val, BOARD_SEED + i); b["draw"] = i + 1
            print(f"draw {i + 1}: legal {b['legal']}, clash ever {b['ever_clash']}, at end {b['end_clash']}, "
                  f"undone passes {sum(p['undone'] for p in b['passes'])}", flush=True)
            if "repaired" not in boards and b["legal"] and b["ever_clash"] and not b["end_clash"]: boards["repaired"] = b
            if "rejected" not in boards and not b["legal"]: boards["rejected"] = b
            if len(boards) == 2: break
    finally:
        val.close()
    for b in boards.values():
        add_names(b.pop("paste"), names)
        v = b["passes"][-1]["values"]
        species |= {v[s * 8] for s in range(6)} | {p["values"][s * 8] for p in b["passes"] for s in range(6) if p["values"][s * 8]}
        items |= {p["values"][s * 8 + 2] for p in b["passes"] for s in range(6) if p["values"][s * 8 + 2]}
        items |= {x for p in b["passes"] for c, x in p["was"].items() if int(c) % 8 == 2 and x}
        species |= {x for p in b["passes"] for c, x in p["was"].items() if int(c) % 8 == 0 and x}
    out["boards"] = boards
    shown = {v for b in boards.values() for p in b["passes"] for v in [*p["values"], *p["was"].values()] if v}
    names.update(showdown_names(shown - names.keys()))
    norm = lambda s: "".join(ch for ch in s.lower() if ch.isalnum())

    def sprite(s):                                   # a Mega form without its own sprite falls back to the base species
        for nm in (norm(s), norm(s).removesuffix("megax").removesuffix("megay").removesuffix("mega")):
            if (SPRITES / f"{nm}.png").exists(): return SPRITES / f"{nm}.png"
        return None
    out["sprites"] = {s: "data:image/png;base64," + base64.b64encode(sprite(s).read_bytes()).decode()
                      for s in sorted(species) if sprite(s)}
    out["items"] = {i: "data:image/png;base64," + base64.b64encode((ITEMS / f"item-{norm(i)}.png").read_bytes()).decode()
                    for i in sorted(items) if (ITEMS / f"item-{norm(i)}.png").exists()}
    out["names"] = names
    out["missing"] = {"species": sorted(s for s in species if s not in out["sprites"]),
                      "items": sorted(i for i in items if i not in out["items"])}
    (HERE / "remask.json").write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")))
    print("boards", {k: b["draw"] for k, b in boards.items()}, "missing", out["missing"],
          "remask.json bytes", (HERE / "remask.json").stat().st_size)


if __name__ == "__main__":
    main()
