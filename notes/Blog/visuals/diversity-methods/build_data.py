#!/usr/bin/env python3
"""Extract data.json for the diversity-methods web figure.

Inputs (read-only) live in RESULTS:
  diversity_methods_all.json, cem_v3.json, confirm_top.json
Output: data.json, written into the folder containing this script.
"""
import json
import os
import re
from collections import Counter, defaultdict

RESULTS = os.path.expanduser("~/Documents/code/vgc-team-generator-pilot/results/")
HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PATH = os.path.join(HERE, "data.json")

CUT = 0.531
ARMS = ["control", "taboo", "peel", "island", "big", "dpp"]
SEEDS = [1, 2, 3]
GENS = [1, 2, 3, 4, 5, 6]
CORE = sorted(["basculegion", "chandelure", "floetteeternal", "garchomp", "kingambit", "whimsicott"])
CORE_T = tuple(CORE)

norm = lambda s: re.sub(r"[^a-z0-9]", "", s.lower())


def species_list(paste):          # in paste order
    names = []
    for block in paste.strip().split("\n\n"):
        first = block.strip().split("\n")[0].split(" @ ")[0].strip()
        m = re.search(r"\(([^()]{3,})\)\s*(\([MF]\))?$", first)
        if m and m.group(1) not in ("M", "F"):
            first = m.group(1)
        first = re.sub(r"\s*\([MF]\)$", "", first)
        names.append(first)
    return names                  # display names, e.g. "Floette-Eternal"


def comp_of_paste(p):
    return tuple(sorted(norm(n) for n in species_list(p)))


# ---- vocabulary -----------------------------------------------------------
species = []
_sidx = {}


def idx(sid):
    if sid not in _sidx:
        _sidx[sid] = len(species)
        species.append(sid)
    return _sidx[sid]


def cidx(comp):
    """Composition (sorted id tuple) -> 6 indices, in sorted-id order."""
    return [idx(s) for s in comp]


names = {}


def note_names(paste):
    for n in species_list(paste):
        names[norm(n)] = n


def load(name):
    with open(os.path.join(RESULTS, name), encoding="utf-8") as f:
        return json.load(f)


def avg(xs):
    return sum(xs) / len(xs)


def make_gen(stream, distinct, battled, ys):
    """One G block. stream: list of sorted-id tuples in sampling order."""
    cnt = Counter(stream)
    top_items = sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0]))[:6]
    box_comps = stream[:30]
    box_distinct = list(dict.fromkeys(box_comps))       # first-appearance order
    bcomps = [comp_of_paste(p) for p in battled]
    bcnt = Counter(bcomps)
    won_comps = [c for c, y in zip(bcomps, ys) if y is not None and y >= CUT]
    bwon = Counter(won_comps)
    return {
        "distinct": distinct,
        "distinct_parsed": len(cnt),
        "n": len(stream),
        "top": [[c, cidx(t)] for t, c in top_items],
        "core_proposed": cnt.get(CORE_T, 0),
        "box": [cidx(t) for t in box_comps],
        "boxinfo": [[cidx(t), cnt[t], bcnt.get(t, 0), bwon.get(t, 0)] for t in box_distinct],
        "battled_distinct": len(bcnt),
        "won": len(won_comps),
        "core_battled": bcnt.get(CORE_T, 0),
        "core_won": bwon.get(CORE_T, 0),
    }


def main():
    out = load("diversity_methods_all.json")
    c3 = load("cem_v3.json")
    confirm = load("confirm_top.json")
    G = out["gens"]

    def entry(arm, seed, g):
        return G[f"{arm}_s{seed}_g{g}"]

    def stream_of(e):
        return [tuple(sorted(x)) for x in e["stream_comps"]]

    # display names: every species seen in any paste
    for e in G.values():
        for p in e["battled"]:
            note_names(p)
    for lst in out["rebattle"].values():
        for d in lst:
            note_names(d["paste"])
    for p in c3["gens"]["0"]["proposals"] + c3["gens"]["0"]["battled"]:
        note_names(p)
    for d in confirm:
        note_names(d["paste"])
    for x in G["island_s1_g1"]["_isl"]:
        note_names(x["seed"]["paste"])

    core_idx = cidx(CORE_T)

    # ---- gen: seed 1 only; G0 shared by every arm, from c3 -----------------
    c3_g0 = c3["gens"]["0"]
    g0_stream = [comp_of_paste(p) for p in c3_g0["proposals"]]
    g0_block = make_gen(g0_stream, out["g0"]["sets"], c3_g0["battled"], c3_g0["y"])
    gen = {}
    for arm in ARMS:
        rows = [g0_block]
        for g in GENS:
            e = entry(arm, 1, g)
            rows.append(make_gen(stream_of(e), e["div"]["distinct_compositions"],
                                 e["battled"], e["y"]))
        gen[arm] = rows

    # ---- mean: 3-seed means per generation ----------------------------------
    mean = {}
    for arm in ARMS:
        rows = [[out["g0"]["sets"], g0_block["top"][0][0] / 512, out["g0"]["stats"]["mean"]]]
        for g in GENS:
            es = [entry(arm, s, g) for s in SEEDS]
            rows.append([avg([e["div"]["distinct_compositions"] for e in es]),
                         avg([e["div"]["largest_share"] for e in es]),
                         avg([e["stats"]["mean"] for e in es])])
        mean[arm] = rows

    # ---- seeds6: generation 6, per seed --------------------------------------
    seeds6 = {}
    for arm in ARMS:
        seeds6[arm] = [[entry(arm, s, 6)["div"]["distinct_compositions"],
                        entry(arm, s, 6)["div"]["largest_share"],
                        entry(arm, s, 6)["stats"]["mean"]] for s in SEEDS]

    # ---- verdict ------------------------------------------------------------
    def control_qual(seed):
        comps = set()
        for g in GENS:
            e = entry("control", seed, g)
            for p, y in zip(e["battled"], e["y"]):
                if y is not None and y >= CUT:
                    comps.add(comp_of_paste(p))
        return len(comps)

    def best_score(arm, seed):
        return max(d["score"] for d in out["rebattle"][f"{arm}_s{seed}"])

    verdict = {}
    for arm in ARMS:
        best = [best_score(arm, s) for s in SEEDS]
        if arm == "control":
            verdict[arm] = {"diff": 0, "ci": None, "passes": None,
                            "qual": [control_qual(s) for s in SEEDS],
                            "off": None, "best": best}
        else:
            sm = out["summary"][arm]
            verdict[arm] = {"diff": sm["win_diff"], "ci": sm["ci"], "passes": sm["passes"],
                            "qual": [sm["qualifying_cores"][str(s)] for s in SEEDS],
                            "off": [sm["off_core_cores"][str(s)] for s in SEEDS],
                            "best": best}

    # ---- taboo: 3-seed means per generation ---------------------------------
    taboo = []
    for g in GENS:
        es = [entry("taboo", s, g) for s in SEEDS]
        taboo.append({
            "raw": avg([e["div_raw"]["distinct_compositions"] for e in es]),
            "raw_share": avg([e["div_raw"]["largest_share"] for e in es]),
            "kept": avg([e["div"]["distinct_compositions"] for e in es]),
            "attempts": avg([e["div"]["attempts"] for e in es]),
            "rejected": avg([e["taboo_rejected"] for e in es]),
        })

    # ---- stream: taboo rule replayed on control seed-1 gen-1 proposals ------
    gen0_set = set(g0_stream)
    accepted_count = Counter()
    comps_table, comp_pos, items = [], {}, []
    acc = rej_seen = rej_twice = 0
    for c in stream_of(entry("control", 1, 1)):
        if c not in comp_pos:
            comp_pos[c] = len(comps_table)
            comps_table.append(cidx(c))
        if c in gen0_set:
            code = 1
            rej_seen += 1
        elif accepted_count[c] >= 2:
            code = 2
            rej_twice += 1
        else:
            code = 0
            accepted_count[c] += 1
            acc += 1
        items.append([comp_pos[c], code])
    stream_block = {"comps": comps_table, "items": items, "accepted": acc,
                    "rej_seen": rej_seen, "rej_twice": rej_twice}

    # ---- winners: seed 1, generations 1..6 pooled ---------------------------
    winners = {}
    for arm in ARMS:
        groups = defaultdict(list)
        for g in GENS:
            e = entry(arm, 1, g)
            for p, y in zip(e["battled"], e["y"]):
                if y is not None and y >= CUT:
                    groups[comp_of_paste(p)].append(y)
        rows = [(len(ys), round(max(ys), 3), c) for c, ys in groups.items()]
        rows.sort(key=lambda r: (-r[0], -r[1], r[2]))
        winners[arm] = [[cnt, cidx(c), by] for cnt, by, c in rows]

    # ---- peel_banned / island_seeds -----------------------------------------
    peel_banned = {}
    for g in GENS:
        ids = G[f"peel_s1_g{g}"]["banned"]
        for s in ids:
            idx(s)
        peel_banned[str(g)] = list(ids)
    island_seeds = [cidx(comp_of_paste(x["seed"]["paste"]))
                    for x in G["island_s1_g1"]["_isl"]]

    # ---- best: confirmed top teams, file order ------------------------------
    best = []
    for d in confirm:
        paste = d["paste"]
        sp = species_list(paste)
        order_ids = [norm(n) for n in sp]
        sheet = []
        for block, name in zip(paste.strip().split("\n\n"), sp):
            lines = block.strip().split("\n")
            item = lines[0].split(" @ ", 1)[1].strip() if " @ " in lines[0] else None
            ability, moves = None, []
            for ln in lines[1:]:
                s = ln.strip()
                if s.startswith("Ability: "):
                    ability = s[len("Ability: "):].strip()
                elif s.startswith("- "):
                    moves.append(s[2:].strip())
            sheet.append({"species": name, "item": item, "ability": ability, "moves": moves})
        best.append({
            "arm": d["arm"],
            "label": float(d["label"]),
            "rebattle": float(d["rebattle"]),
            "fresh": round(float(d["fresh"]), 3),
            "new_battles": int(d["new_battles"]),
            "comp": cidx(tuple(sorted(order_ids))),
            "order": [idx(i) for i in order_ids],
            "sheet": sheet,
        })

    data = {
        "species": species,
        "names": {s: names.get(s, s) for s in species},
        "core": core_idx,
        "arms": ARMS,
        "gen": gen,
        "mean": mean,
        "seeds6": seeds6,
        "verdict": verdict,
        "taboo": taboo,
        "stream": stream_block,
        "winners": winners,
        "peel_banned": peel_banned,
        "island_seeds": island_seeds,
        "best": best,
    }

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, separators=(",", ":"))

    # ---- checks -------------------------------------------------------------
    size = os.path.getsize(OUT_PATH)
    print(f"wrote {OUT_PATH}")
    print(f"len(species) = {len(species)}")
    print(f"data.json size = {size / 1024:.1f} KB ({size} bytes)")
    print("control distinct g0..g6 =", [R["distinct"] for R in gen["control"]])
    print("control core_proposed g0..g6 =", [R["core_proposed"] for R in gen["control"]])
    print("taboo distinct g1..g6 =", [R["distinct"] for R in gen["taboo"][1:]])
    taboo_core_hits = sum(1 for s in SEEDS for g in GENS
                          for c in stream_of(entry("taboo", s, g)) if c == CORE_T)
    print("CORE copies in taboo stream_comps (seeds 1-3, gens 1-6) =", taboo_core_hits)
    print("mean distinct at g6 per arm =",
          {arm: round(mean[arm][6][0], 2) for arm in ARMS})
    print(f"stream accepted={acc} rej_seen={rej_seen} rej_twice={rej_twice}")
    print("winners compositions per arm =", {arm: len(winners[arm]) for arm in ARMS})
    print("control box distinct per gen g0..g6 =",
          [len(set(map(tuple, R["box"]))) for R in gen["control"]])
    print("taboo box distinct per gen g0..g6 =",
          [len(set(map(tuple, R["box"]))) for R in gen["taboo"]])


if __name__ == "__main__":
    main()
