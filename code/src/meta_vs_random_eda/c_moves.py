import json, sys, numpy as np
sys.path.insert(0, sys.argv[1]); from feats import parse, effective_species, DEX, MOVES, nid, PROTECT, SPEED_CTRL, SUPPORT, SPREAD
o = json.load(open('/Users/ramiismael/Documents/code/vgc-team-generator-pilot/results/random_fill_baseline.json'))
orig = {int(k): v for k, v in o['original'].items()}
tasks = {t["start_team"]: t["team_paste"] for t in json.load(open('/Users/ramiismael/Documents/code/vgc-team-generator-pilot/results/random_fill_tasks.json'))["tasks"]}
CATS = ["Protect", "STAB attack", "Coverage attack", "Spread attack", "Speed control", "Support move", "Other status"]
def cat(move, mon):
    i = nid(move); m = MOVES.get(i, {})
    sid, _ = effective_species(mon); d = DEX.get(sid, {}); types = set(d.get("types", []))
    if i in PROTECT: return "Protect"
    if i in SPEED_CTRL: return "Speed control"
    if i in SUPPORT: return "Support move"
    if m.get("category") in ("Physical", "Special"):
        if m.get("target") in SPREAD: return "Spread attack"
        if m.get("type") in types: return "STAB attack"
        return "Coverage attack"
    return "Other status"
rows, y, st = [], [], []
lost_c = {c: 0 for c in CATS}; gain_c = {c: 0 for c in CATS}
for f in o["fills"]:
    if f["task"] != "move" or f.get("score") is None or f["start_team"] not in orig: continue
    a, b = parse(tasks[f["start_team"]]), parse(f["paste"])
    lost = {c: 0 for c in CATS}; gain = {c: 0 for c in CATS}
    for ma, mb in zip(a, b):
        for mv in set(ma["moves"]) - set(mb["moves"]): lost[cat(mv, ma)] += 1
        for mv in set(mb["moves"]) - set(ma["moves"]): gain[cat(mv, mb)] += 1
    for c in CATS: lost_c[c] += lost[c]; gain_c[c] += gain[c]
    rows.append([lost[c] for c in CATS] + [gain[c] for c in CATS]); y.append(f["score"] - orig[f["start_team"]]); st.append(f["start_team"])
R = np.array(rows, float); X = (R[:, 7:] - R[:, :7])[:, :6]; y = np.array(y); st = np.array(st)  # net change per category; 'Other status' = 0
# Δ = Σ cost(lost category) − Σ value(gained category); no intercept: zero changes = zero effect
beta = np.linalg.lstsq(X, y, rcond=None)[0]
us = np.unique(st); rng = np.random.default_rng(1); B = []
for _ in range(1000):
    idx = np.concatenate([np.where(st == u)[0] for u in rng.choice(us, len(us))]); B.append(np.linalg.lstsq(X[idx], y[idx], rcond=None)[0])
lo, hi = np.quantile(B, [.025, .975], axis=0)
print("n fills", len(y)); out = {}
r2 = 1 - ((y - X @ beta) ** 2).sum() / ((y - y.mean()) ** 2).sum(); print("R2", round(r2, 3))
for j, c in enumerate(CATS[:6]):
    print(f"value of {c:16s} vs other status: {beta[j]:+.4f} [{lo[j]:+.4f},{hi[j]:+.4f}]  lost={lost_c[c]} gained={gain_c[c]}")
    out[c] = dict(value=[beta[j], lo[j], hi[j]], n_lost=lost_c[c], n_gain=gain_c[c])
# what share of real moves vs random moves fall in each category
real = {c: 0 for c in CATS}; rand = {c: 0 for c in CATS}
for p in tasks.values():
    for m in parse(p):
        for mv in m["moves"]: real[cat(mv, m)] += 1
import sqlite3
con = sqlite3.connect("/Users/ramiismael/vgc-data/matchup_regmb.sqlite")
for (p,) in con.execute("select paste from team where origin='randomfill:pokemon6'"):
    for m in parse(p):
        for mv in m["moves"]: rand[cat(mv, m)] += 1
tr, tn = sum(real.values()), sum(rand.values())
print({c: (round(real[c] / tr, 3), round(rand[c] / tn, 3)) for c in CATS})
json.dump(dict(cost=out, mix={c: [real[c] / tr, rand[c] / tn] for c in CATS}), open(sys.argv[1] + "/c_moves.json", "w"), default=float)
