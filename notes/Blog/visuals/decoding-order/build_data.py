"""Data for the decoding-order results page (2026-10-31). Writes data.json beside this file.

    /tmp/vgc-pilot/.venv/bin/python build_data.py

Sources:
- ~/vgc-data/results/decoding_order.json: per cell legality, kept teams, battled win rates, Showdown's rejection reasons;
  its summary (per order: legality, win rate, win rate by ask, ask slope, gap; each against species-first order).
- ~/vgc-data/results/decoding_order_diversity.json: size-matched diversity (5-team subsets) and novelty.
- Fill traces: one fresh team per order at ask 0.5, drawn with the experiment's settings (guidance 2, temperature 1)
  from the discrete diffusion model trained from scratch on valid Pokémon teams, recording which fields each pass
  filled and which fields had no legal value left. They illustrate the orders; no number on the page comes from them.
- Sprites: ../search-loop/sprites (species), ../item-sprites (items).
"""
import base64
import json
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
from hps_generate import Validator
from team_diversity import slots_from_paste

RES = Path.home() / "vgc-data" / "results"
SPRITES, ITEMS = HERE.parent / "search-loop" / "sprites", HERE.parent / "item-sprites"
ORDERS = [("species", "Species-first", "species", 48), ("pokemon", "Pokémon-by-Pokémon", "pokemon", 48),
          ("random1", "Random, one field per pass", "random", 48), ("random2", "Random, two fields per pass", "random", 24),
          ("confidence", "Confidence", "confidence", 48)]
FIELDS = ["species", "ability", "item", "move", "move", "move", "move", "nature"]
TRACE_SEED, TRACE_ASK, TRACE_TRIES = 7311, 0.5, 16


def stat(s, k=4):
    return {"mean": round(s["mean"], k), "sd": round(s["sd"], k), "seeds": [round(x, k) for x in s["seeds"]]}


def vs(s, k=4):
    return {"diff": round(s["diff"], k), "noise": round(s["noise"], k), "seeds": [round(x, k) for x in s["seeds"]]}


def add_names(paste, names):
    """Display names keyed by id, from a Showdown paste ("Garchomp @ Sitrus Berry", "Ability: Rough Skin", "- Protect")."""
    norm = lambda t: "".join(ch for ch in t.lower() if ch.isalnum())
    for line in paste.splitlines():
        line = line.strip()
        if " @ " in line: parts = [x.strip() for x in line.split(" @ ")]
        elif line.startswith("Ability:"): parts = [line[8:].strip()]
        elif line.startswith("- "): parts = [line[2:].strip()]
        elif line.endswith(" Nature"): parts = [line[:-7].strip()]
        else: continue
        for x in parts:
            x = x.split(" (")[0]
            if x: names.setdefault(norm(x), x)


def team(paste):
    s = slots_from_paste(paste)
    return [{"species": x["species"], "item": x["item"]} for x in s] if s else None


def trace(w, order, steps, val):
    """Draw up to TRACE_TRIES teams one at a time until one is legal (random orders: keep the first, which is not);
    return its fill sequence, values, dead ends and Showdown's verdict."""
    real_fill, real_legal = w._fill_pass, w._legal
    for i in range(TRACE_TRIES):
        passes, dead, inside = [], set(), [False]
        def fill(x, conf, targets, due, redo, *a, **k):
            passes.append(due[0].nonzero().flatten().tolist()); inside[0] = True
            try: return real_fill(x, conf, targets, due, redo, *a, **k)
            finally: inside[0] = False
        def legal(x, c):
            ok, empty = real_legal(x, c)
            if inside[0] and bool(empty.any()): dead.add(c)
            return ok, empty
        w._fill_pass, w._legal = fill, legal
        try:
            torch.manual_seed(TRACE_SEED + i)
            row = w.sample(1, ask=TRACE_ASK, guidance=2.0, order=order, steps=steps, stats={})[0].cpu().numpy()
        finally:
            del w._fill_pass, w._legal
        paste, _ = w.to_paste(row, np.random.default_rng(TRACE_SEED + i), repair=True)
        err = val(paste)
        if err is None or order == "random" or i == TRACE_TRIES - 1:
            values = [w.V.itos[L.KEY_OF_COL[c]][int(row[c])] for c in range(L.COLS)]
            return {"passes": passes, "values": values, "dead": sorted(dead), "legal": err is None, "paste": paste,
                    "verdict": None if err is None else str(err).split("\n")[0][:400], "draw": i + 1}
    raise AssertionError("unreachable")


def main():
    run = json.loads((RES / "decoding_order.json").read_text())
    div = json.loads((RES / "decoding_order_diversity.json").read_text())
    cells, summ, dsum = run["cells"], run["summary"], div["summary"]
    cfg = run["config"]
    out = {"asks": cfg["asks"], "seeds": cfg["seeds"], "keep": cfg["keep"], "max_tries": cfg["max_tries"],
           "match": div["match"], "draws": div["draws"], "fields": FIELDS, "orders": []}
    species, items, names = set(), set(), {}
    for key, name, _, steps in ORDERS:
        s, d = summ[key], dsum[key]
        cs = [c for c in cells.values() if c["arm"] == key]
        pastes = [p for c in sorted(cs, key=lambda c: (c["seed"], abs(c["ask"] - 0.5))) if c["seed"] == 1
                  for p in c["pastes"] if team(p)][:6]
        examples = [team(p) for p in pastes]
        for p in pastes: add_names(p, names)
        reasons = sorted(((r, n) for c in cs if c["seed"] == 1 for r, n in c["reasons"]), key=lambda t: len(t[0]))
        o = {"key": key, "name": name, "passes": steps,
             "tried": sum(c["tried"] for c in cs), "legal_teams": sum(c["tried"] - c["rejected"] for c in cs),
             "kept": sum(c["kept"] for c in cs), "capped": sum(c["capped"] for c in cs),
             "seconds": round(sum(c["seconds"] for c in cs), 1),
             "legal": stat(s["legal"]), "win": stat(s["win"]), "ask_slope": stat(s["ask_slope"], 3), "gap": stat(s["gap"]),
             "win_by_ask": {a: stat(v) for a, v in s["win_by_ask"].items()},
             "diversity": {m: stat(d[m], 3) for m in ("effective_species", "distinct_lineups", "top_species_share",
                                                      "mean_distance", "new_lineups", "lineup_distance", "copies")},
             "examples": examples, "reason": reasons[0][0] if reasons else None}
        if "vs_species" in s:
            o["vs"] = {m: vs(s["vs_species"][m]) for m in ("legal", "win", "ask_slope", "gap")}
            o["vs"].update({m: vs(d["vs_species"][m], 3) for m in ("effective_species", "mean_distance", "new_lineups")})
        out["orders"].append(o)
        for t in examples:
            species |= {x["species"] for x in t}; items |= {x["item"] for x in t if x["item"]}
    w, val = M.DiscreteDiffusionModel.load(), Validator()
    try:
        for o, (_, _, order, steps) in zip(out["orders"], ORDERS):
            o["trace"] = trace(w, order, steps, val)
            v = o["trace"]["values"]; add_names(o["trace"].pop("paste"), names)
            species |= {v[s * 8] for s in range(6)}; items |= {v[s * 8 + 2] for s in range(6)}
            print(f"{o['name']:28s} trace: draw {o['trace']['draw']}, legal {o['trace']['legal']}, passes "
                  f"{len(o['trace']['passes'])}, dead ends at {o['trace']['dead']}", flush=True)
    finally:
        val.close()
    norm = lambda s: "".join(ch for ch in s.lower() if ch.isalnum())
    def sprite(s):                                   # a Mega form without its own sprite falls back to the base species
        for name in (norm(s), norm(s).removesuffix("megax").removesuffix("megay").removesuffix("mega")):
            if (SPRITES / f"{name}.png").exists(): return SPRITES / f"{name}.png"
        return None
    out["sprites"] = {s: "data:image/png;base64," + base64.b64encode(sprite(s).read_bytes()).decode()
                      for s in sorted(species) if sprite(s)}
    out["items"] = {i: "data:image/png;base64," + base64.b64encode((ITEMS / f"item-{norm(i)}.png").read_bytes()).decode()
                    for i in sorted(items) if (ITEMS / f"item-{norm(i)}.png").exists()}
    out["names"] = names
    out["cellKept"] = {k: sorted(c["kept"] for c in cells.values() if c["arm"] == k) for k, *_ in ORDERS}
    out["missing"] = {"species": sorted(s for s in species if s not in out["sprites"]),
                      "items": sorted(i for i in items if i not in out["items"])}
    (HERE / "data.json").write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")))
    print("missing sprites", out["missing"], "data.json bytes", (HERE / "data.json").stat().st_size)


if __name__ == "__main__":
    main()
