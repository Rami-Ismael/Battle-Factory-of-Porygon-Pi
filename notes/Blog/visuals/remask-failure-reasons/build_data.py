"""Data for the remasking-failure page (2026-10-30): why remasking did not close the gap between the asked and the
achieved win rate. Writes data.json beside this file. Reads only results files and the sprite folder.

    /tmp/vgc-pilot/.venv/bin/python build_data.py

Sources: ~/vgc-data/results/remask_ask_gap.json (plain decoding and remasking 10%, least likely fields) and
~/vgc-data/results/remask_why.json (remasking 10%, random fields; plain decoding at the on-grid asks).
"""
import base64
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[3] / "code" / "src"))
from team_diversity import slots_from_paste

RES = Path.home() / "vgc-data" / "results"
SPRITES = HERE.parent / "search-loop" / "sprites"
CORE = ("basculegion", "charizard", "floetteeternal", "garchomp", "kingambit", "whimsicott")
ASKS = (0.3, 0.5, 0.531)
FIELDS = {"species": 6, "ability": 6, "item": 6, "move": 24, "nature": 6}


def cells(src, arm, ask):
    cs = [c for c in src.values() if c["arm"] == arm and abs(c["ask"] - ask) < 1e-9]
    if sorted(c["seed"] for c in cs) != [1, 2, 3] or any("win" not in c or len(c["pastes"]) != 64 for c in cs):
        raise SystemExit(f"{arm} at ask {ask}: expected seeds 1-3, each 64 scored teams")
    return sorted(cs, key=lambda c: c["seed"])


def lineup(p):
    s = slots_from_paste(p)
    return tuple(sorted(x["species"] for x in s)) if s else None


def main():
    gap = json.loads((RES / "remask_ask_gap.json").read_text())["cells"]
    why = json.loads((RES / "remask_why.json").read_text())["cells"]
    arms = {"plain": gap, "remask10": gap, "random10": why}
    out = {"asks": list(ASKS), "core": list(CORE), "teams": {}, "summary": {}, "grid_asks": {}, "redraws": {}}
    species = set(CORE)
    for ask in ASKS:
        for arm, src in arms.items():
            cs = cells(src, arm, ask)
            s1 = [lineup(p) for p in cs[0]["pastes"]]
            groups = Counter(t for t in s1 if t)
            species |= {x for t in groups for x in t}
            pooled = [lineup(p) for c in cs for p in c["pastes"]]
            ranked = [t for t, _ in groups.most_common()]
            out["teams"][f"{arm}@{ask}"] = [{"lineup": list(t), "n": n} for t, n in groups.most_common()]
            out["order"] = out.get("order", {})                        # seed 1 in the order the model wrote them
            out["order"][f"{arm}@{ask}"] = [ranked.index(t) for t in s1 if t]
            out["summary"][f"{arm}@{ask}"] = {
                "win_seeds": [round(c["win"]["mean"], 4) for c in cs],
                "lineups_seeds": [len({lineup(p) for p in c["pastes"]} - {None}) for c in cs],
                "core_share": round(float(np.mean([t is not None and set(CORE) <= set(t) for t in pooled])), 4),
                "seed1_teams": len(s1), "seed1_unparsed": s1.count(None)}
    for ask in (7 / 24, 8 / 24, 13 / 24):
        out["grid_asks"][f"{ask:.4f}"] = [round(c["win"]["mean"], 4) for c in cells(why, "plain", ask)]
    for arm in ("remask10",):
        cs = cells(gap, arm, 0.5)
        rem = {f: float(np.mean([c["redraws"]["remasked"].get(f, 0) for c in cs])) for f in FIELDS}
        chg = {f: float(np.mean([c["redraws"]["changed"].get(f, 0) for c in cs])) for f in FIELDS}
        out["redraws"] = {"ask": 0.5, "arm": arm, "fields": FIELDS, "remasked_per_team": rem, "changed_per_team": chg}
    out["sprites"] = {s: "data:image/png;base64," + base64.b64encode((SPRITES / f"{s}.png").read_bytes()).decode()
                      for s in sorted(species) if (SPRITES / f"{s}.png").exists()}
    out["missing_sprites"] = sorted(species - set(out["sprites"]))
    (HERE / "data.json").write_text(json.dumps(out, separators=(",", ":")))
    for k, v in out["summary"].items(): print(k, v["lineups_seeds"], v["core_share"], v["win_seeds"])
    print("redraws at ask 0.5, remasking 10%:", {f: round(x, 2) for f, x in out["redraws"]["remasked_per_team"].items()},
          "changed:", {f: round(x, 2) for f, x in out["redraws"]["changed_per_team"].items()})
    print("species", len(species), "missing sprites", out["missing_sprites"], "data.json bytes", (HERE / "data.json").stat().st_size)


if __name__ == "__main__":
    main()
