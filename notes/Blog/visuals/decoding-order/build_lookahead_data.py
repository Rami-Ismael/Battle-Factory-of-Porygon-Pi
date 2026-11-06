"""Data for the lookahead section of the decoding-order page (2026-11-01). Writes lookahead.json beside this file;
build.py merges it into the page.

    /tmp/vgc-pilot/.venv/bin/python build_lookahead_data.py

Sources:
- ~/vgc-data/results/lookahead_legality.json: per cell legality, kept teams, battled win rates and Showdown's most
  common rejection reasons (the top 8 per cell) for three orders with the lookahead, and its summary (controls = the
  same orders without it, from the decoding-order run; reference = species-first order). Pre-registration:
  code/docs/lookahead-legality.md.
- Lookahead boards: one seed sampled twice at ask 0.5 with random order, one field per pass, 48 passes, guidance 2,
  temperature 1: once without the lookahead and once with it. The random field order is drawn first, so both runs
  fill the fields in the same order. Per pass: the new field, the 48 values after the pass, each masked species's
  candidate species (the species that still fit its Pokémon's visible fields and Species Clause), the share of the
  model's probability for the new field that the lookahead refuses (with it on) or would refuse (with it off), and the
  fields the legality tables say clash with the rest of the team. The pair shown is the first seed whose run without
  the lookahead Showdown rejected, with a value the lookahead would have refused drawn before that Pokémon's species,
  and whose run with the lookahead Showdown accepted. It illustrates the lookahead; no number on the page comes from it.
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

from build_data import add_names, team
from build_remask_data import clashes, showdown_names

RES = Path.home() / "vgc-data" / "results"
SPRITES, ITEMS = HERE.parent / "search-loop" / "sprites", HERE.parent / "item-sprites"
ORDERS = [("random1", "Random order, one field per pass"), ("random2", "Random order, two fields per pass"),
          ("confidence", "Confidence order")]
MEASURES = ("legal", "win", "ask_slope", "gap", "effective_species", "distinct_lineups", "top_species_share",
            "mean_distance", "copies", "new_lineups", "lineup_distance")
FEW = 4                                             # list the candidate species when this many or fewer are left
BOARD_SEED, BOARD_ASK, BOARD_TRIES = 7351, 0.5, 64


def stat(s, k=6):
    return {"mean": round(s["mean"], k), "sd": round(s["sd"], k), "seeds": [round(x, k) for x in s["seeds"]]}


def vs(s, k=6):
    return {"diff": round(s["diff"], k), "noise": round(s["noise"], k), "seeds": [round(x, k) for x in s["seeds"]]}


def reason_kind(r):
    """The kind of Showdown's first complaint about a rejected team."""
    first = r.split(";")[0]
    if "can't learn" in first: return "move"
    if "can't have" in first: return "ability"
    if "transforms in-battle" in first: return "mega_item"
    return "other"


def board(w, val, seed, lookahead):
    """One team at BOARD_ASK, random order, one field per pass, with or without the lookahead; see the module doc."""
    word = lambda c, v: "" if int(v) == 0 else w.V.itos[L.KEY_OF_COL[c]][int(v)]
    passes, real_fill, real_probs, buf = [], w._fill_pass, type(w)._probs, {}

    def probs(lc, uncond, x, c, gc, temperature, top_p, surrogate, lam, legal=None):
        p = real_probs(w, lc, uncond, x, c, gc, temperature, top_p, surrogate, lam, legal=legal)
        base, la = w._legal(x, c, False)[0], w._legal(x, c, True)[0]
        pb = real_probs(w, lc, uncond, x, c, gc, temperature, top_p, surrogate, lam, legal=base)   # without the lookahead
        ref = (base & ~la)[0]
        share = float(pb[0][ref].sum())
        top = int((pb[0] * ref).argmax()) if share > 0 else 0
        buf[c] = {"share": share, "top": word(c, top) if share > 0 else "", "top_p": float(pb[0, top]) if share > 0 else 0.0,
                  "ref": ref}
        return p

    def fill(x, conf, targets, due, redo, *a, **k):
        buf.clear()
        undone = real_fill(x, conf, targets, due, redo, *a, **k)
        new, al = due[0].nonzero().flatten().tolist(), w.tables.allowed(x)["species"][0]
        cand = []
        for s in range(M.NSLOT):
            if int(x[0, s * M.NF]): cand.append(None); continue
            ix = al[s].nonzero().flatten().tolist()
            cand.append({"n": len(ix), "few": [word(0, i) for i in ix] if len(ix) <= FEW else []})
        refused = {c: {"share": round(buf[c]["share"], 6), "top": buf[c]["top"], "top_p": round(buf[c]["top_p"], 6),
                       "drawn": bool(buf[c]["ref"][int(x[0, c])])} for c in new if c in buf and buf[c]["share"] > 0}
        passes.append({"new": new, "values": [word(c, x[0, c]) for c in range(L.COLS)], "cand": cand, "refused": refused,
                       "clash": clashes(w, x), "undone": bool(undone)})
        return undone

    w._fill_pass, w._probs = fill, probs
    try:
        torch.manual_seed(seed)
        row = w.sample(1, ask=BOARD_ASK, guidance=2.0, order="random", steps=48, lookahead=lookahead)[0]
    finally:
        del w._fill_pass, w._probs
    paste, _ = w.to_paste(row.cpu().numpy(), np.random.default_rng(seed), repair=True)
    err = val(paste)
    return {"passes": passes, "legal": err is None, "paste": paste, "seed": seed,
            "verdict": None if err is None else str(err).split("\n")[0][:400]}


def foreseen(b):
    """First pass (0-based) that drew a value the lookahead would refuse, before that Pokémon's species; else None."""
    for i, p in enumerate(b["passes"]):
        for c, r in p["refused"].items():
            if r["drawn"] and c % M.NF and p["values"][(c // M.NF) * M.NF] == "": return i
    return None


def main():
    run = json.loads((RES / "lookahead_legality.json").read_text())
    cells, summ, cfg = run["cells"], run["summary"], run["config"]
    out = {"asks": cfg["asks"], "seeds": cfg["seeds"], "keep": cfg["keep"], "max_tries": cfg["max_tries"], "fix_floor": 0.9,
           "species": {m: stat(summ["species"][m]) for m in MEASURES} | {"seconds_per_team": summ["species"]["seconds_per_team"]},
           "orders": []}
    species, items, names = set(), set(), {}
    for key, name in ORDERS:
        o = {"key": key, "name": name}
        for side, k in (("off", key), ("on", key + "_la")):
            s = summ[k]
            o[side] = {"kept": s["kept"], "tried": s["tried"], "capped": s["capped"], "seconds_per_team": round(s["seconds_per_team"], 4),
                       **{m: stat(s[m]) for m in MEASURES}}
        s = summ[key + "_la"]
        o["vs_without"] = {m: vs(v) for m, v in s["vs_without"].items()}
        o["vs_species"] = {m: vs(v) for m, v in s["vs_species"].items()}
        o["fixes_legality"], o["matches_species_first"] = s["fixes_legality"], s["matches_species_first"]
        cs = [c for c in cells.values() if c["arm"] == key + "_la"]
        kinds = {}
        for c in cs:
            for r, n in c["reasons"]: kinds[reason_kind(r)] = kinds.get(reason_kind(r), 0) + n
        o["rejected"], o["reasons_logged"], o["reason_kinds"] = sum(c["rejected"] for c in cs), sum(kinds.values()), kinds
        pastes = [p for c in sorted(cs, key=lambda c: abs(c["ask"] - 0.5)) if c["seed"] == 1 for p in c["pastes"] if team(p)][:6]
        o["examples"] = [team(p) for p in pastes]
        for p in pastes: add_names(p, names)
        for t in o["examples"]:
            species |= {x["species"] for x in t}; items |= {x["item"] for x in t if x["item"]}
        out["orders"].append(o)

    w, val = M.DiscreteDiffusionModel.load(), Validator()
    pair = None
    try:
        for i in range(BOARD_TRIES):
            off = board(w, val, BOARD_SEED + i, False)
            f = foreseen(off)
            print(f"draw {i + 1}: without lookahead legal {off['legal']}, foreseen dead end at pass "
                  f"{None if f is None else f + 1}", flush=True)
            if off["legal"] or f is None: continue
            on = board(w, val, BOARD_SEED + i, True)
            print(f"         with lookahead legal {on['legal']}", flush=True)
            if on["legal"]: pair = {"draw": i + 1, "seed": BOARD_SEED + i, "off": off, "on": on}; break
    finally:
        val.close()
    if pair is None: raise SystemExit(f"no seed in {BOARD_TRIES} tries gave a pair to show")
    same = [all(a == b for a, b in zip(p["values"], q["values"])) for p, q in zip(pair["off"]["passes"], pair["on"]["passes"])]
    if not all(p["new"] == q["new"] for p, q in zip(pair["off"]["passes"], pair["on"]["passes"])):
        raise SystemExit("the two runs filled fields in different orders")
    pair["same_until"] = same.index(False) if False in same else len(same)        # passes with identical values
    for side in ("off", "on"):
        b = pair[side]
        add_names(b.pop("paste"), names)
        for p in b["passes"]:
            species |= {p["values"][s * M.NF] for s in range(M.NSLOT) if p["values"][s * M.NF]}
            species |= {x for cd in p["cand"] if cd for x in cd["few"]}
            items |= {p["values"][s * M.NF + 2] for s in range(M.NSLOT) if p["values"][s * M.NF + 2]}
    out["pair"] = pair
    shown = {v for s in ("off", "on") for p in pair[s]["passes"] for v in
             [*p["values"], *(r["top"] for r in p["refused"].values()), *(x for cd in p["cand"] if cd for x in cd["few"])] if v}
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
    (HERE / "lookahead.json").write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")))
    print("pair draw", pair["draw"], "same until pass", pair["same_until"], "missing", out["missing"],
          "lookahead.json bytes", (HERE / "lookahead.json").stat().st_size)


if __name__ == "__main__":
    main()
