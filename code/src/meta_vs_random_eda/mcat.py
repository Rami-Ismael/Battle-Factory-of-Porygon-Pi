import json, sys, numpy as np
sys.path.insert(0, sys.argv[1]); from feats import parse, effective_species, DEX, MOVES, nid, PROTECT, SPEED_CTRL, SUPPORT, SPREAD
o = json.load(open('/Users/ramiismael/Documents/code/vgc-team-generator-pilot/results/random_fill_baseline.json'))
orig = {int(k): v for k, v in o['original'].items()}
tasks = {t["start_team"]: t["team_paste"] for t in json.load(open('/Users/ramiismael/Documents/code/vgc-team-generator-pilot/results/random_fill_tasks.json'))["tasks"]}
CATS = ["Protect", "STAB attack", "Coverage attack", "Spread attack", "Speed control", "Support move", "Other status"]
def cat(move, mon):
    i = nid(move); m = MOVES.get(i, {})
    sid, _ = effective_species(mon); d = DEX.get(sid, {}); bs = d.get("baseStats", {}); types = set(d.get("types", []))
    if i in PROTECT: return "Protect"
    if i in SPEED_CTRL: return "Speed control"
    if i in SUPPORT: return "Support move"
    if m.get("category") in ("Physical", "Special"):
        if m.get("target") in SPREAD: return "Spread attack"
        if m.get("type") in types: return "STAB attack"
        return "Coverage attack"
    return "Other status"
