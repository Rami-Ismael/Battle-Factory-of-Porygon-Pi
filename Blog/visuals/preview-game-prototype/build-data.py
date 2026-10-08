"""PROTOTYPE — throwaway. Build data.js for the team-preview game prototype (index.html).

Sources
- results/plan_matrix_MB522_MB763.json in the code repo: 90 x 90 plans, MB522's wins out of 24 per cell
  (team preview forced to each plan, then the behaviour-cloning policy plays both sides).
- ~/vgc-data/matchup_regmb.sqlite: plan_contribution batches 310/311/312 (three pours of 8 battles per cell),
  the two teams' pastes (team 436 = MB522, team 606 = MB763), and the 681 legal VGCPastes teams.

Every equilibrium here is the row player's maximin linear programme (HiGHS); MB522 is the row player.
Run: python3 build-data.py
"""
import json
import re
import sqlite3
from itertools import combinations, product
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

HERE = Path(__file__).resolve().parent
REPO = Path.home() / "Documents/code/vgc-team-generator-pilot"
DB = Path.home() / "vgc-data/matchup_regmb.sqlite"
SPRITES = HERE.parent / "search-loop/sprites"
ITEMS = HERE.parent / "item-sprites"
key = lambda s: re.sub(r"[^a-z0-9]", "", s.lower())

src = json.loads((REPO / "results/plan_matrix_MB522_MB763.json").read_text())
R, C = src["row_plans"], src["col_plans"]
W = np.array(src["wins"])
assert (np.array(src["battles"]) == 24).all()
M = W / 24

db = sqlite3.connect(DB)
ri, ci = {p: i for i, p in enumerate(R)}, {p: j for j, p in enumerate(C)}
POUR = {}
for b, lp, hp, lw, n in db.execute(
        "select batch_id, lo_plan, hi_plan, lo_wins, battles from plan_contribution where lo = 436 and hi = 606"):
    assert n == 8
    POUR.setdefault(b, np.zeros((90, 90), int))[ri[lp], ci[hp]] += lw
pours = sorted(POUR)
assert len(pours) == 3 and (sum(POUR[b] for b in pours) == W).all()


def parse_paste(text):
    mons = []
    for blk in text.strip().split("\n\n"):
        lines = blk.strip().splitlines()
        head = re.sub(r" \([MF]\)", "", lines[0])
        sp, _, item = head.partition(" @ ")
        sp = re.sub(r".*\((.*)\).*", r"\1", sp) if "(" in sp else sp
        get = lambda pre: next((l.split(": ", 1)[1].strip() for l in lines if l.startswith(pre)), "")
        mons.append({"id": key(sp), "name": sp.strip(), "item": item.strip(), "ability": get("Ability:"),
                     "moves": [l[2:].strip() for l in lines if l.startswith("- ")]})
    return mons


def team(tid):
    mons = parse_paste(db.execute("select paste from team where team_id = ?", (tid,)).fetchone()[0])
    for m in mons:
        assert (SPRITES / f"{m['id']}.png").exists(), m["id"]
        m["sprite"] = f"../search-loop/sprites/{m['id']}.png"
        it = ITEMS / f"item-{key(m['item'])}.png"
        m["itemSprite"] = f"../item-sprites/{it.name}" if it.exists() else None
    return mons


YOU, THEM = team(436), team(606)


def plans(codes, mons):
    idx = {m["id"]: k for k, m in enumerate(mons)}
    out = []
    for c in codes:
        lead, back = c.split("/")
        out.append({"lead": [idx[s] for s in lead.split("+")], "back": [idx[s] for s in back.split("+")]})
    return out


ROWS, COLS = plans(R, YOU), plans(C, THEM)


def nash(A):
    """Value and both equilibrium mixes of the zero-sum game A (row player maximises)."""
    m, n = A.shape
    r = linprog(np.r_[np.zeros(m), -1], A_ub=np.c_[-A.T, np.ones(n)], b_ub=np.zeros(n),
                A_eq=np.r_[np.ones(m), 0][None], b_eq=[1], bounds=[(0, None)] * m + [(None, None)], method="highs")
    c = linprog(np.r_[np.zeros(n), 1], A_ub=np.c_[A, -np.ones(m)], b_ub=np.zeros(m),
                A_eq=np.r_[np.ones(n), 0][None], b_eq=[1], bounds=[(0, None)] * n + [(None, None)], method="highs")
    clean = lambda v: (lambda w: w / w.sum())(np.where(v > 1e-6, v, 0))
    return float(r.x[m]), clean(r.x[:m]), clean(c.x[:n])


def sparse(w, keep=None):
    keep = range(len(w)) if keep is None else keep
    return {int(k): round(float(v), 4) for k, v in zip(keep, w) if v > 0}


v, x, y = nash(M)
EQ = {"value": round(v, 4), "x": sparse(x), "y": sparse(y)}

# Pull one Pokemon (or none) from each side: the game shrinks to the plans that never bring it.
DROP = {}
for a, b in product(range(-1, 6), range(-1, 6)):
    rk = [i for i, p in enumerate(ROWS) if a not in p["lead"] + p["back"]]
    ck = [j for j, p in enumerate(COLS) if b not in p["lead"] + p["back"]]
    dv, dx, dy = nash(M[np.ix_(rk, ck)])
    DROP[f"{a}|{b}"] = {"value": round(dv, 4), "x": sparse(dx, rk), "y": sparse(dy, ck)}

# Jenga with up to two blocks out per side (owner 2026-10-08): two out = the two left at home at team preview.
# Every combination of 0, 1 or 2 pulled per side: 22 × 22 = 484 games; both sides at two leaves the 6 × 6 "who leads" game.
SUBSETS = [()] + [(k,) for k in range(6)] + list(combinations(range(6), 2))
DROP2 = {}
for A_, B_ in product(SUBSETS, SUBSETS):
    rk = [i for i, p in enumerate(ROWS) if not set(A_) & set(p["lead"] + p["back"])]
    ck = [j for j, p in enumerate(COLS) if not set(B_) & set(p["lead"] + p["back"])]
    dv, dx, dy = nash(M[np.ix_(rk, ck)])
    DROP2["-".join(map(str, A_)) + "|" + "-".join(map(str, B_))] = {"value": round(dv, 4), "x": sparse(dx, rk), "y": sparse(dy, ck)}

# The sand timer: the matrix after one, two and three pours of 8 battles per cell.
SAND = []
for k in range(1, 4):
    Wk = sum(POUR[b] for b in pours[:k])
    Mk = Wk / (8 * k)
    sv, sx, sy = nash(Mk)
    noise = float((Mk * (1 - Mk) / (8 * k - 1)).mean() / Mk.var())
    single = nash(POUR[pours[k - 1]] / 8)[0]
    SAND.append({"battles": 8 * k, "value": round(sv, 4), "x": sparse(sx), "y": sparse(sy),
                 "noiseShare": round(noise, 3), "singlePourValue": round(single, 4)})

# Peeking, scored on fresh battles: pick on two pours (16 battles a cell), score on the third (8).
folds = []
for k, b in enumerate(pours):
    tr = [p for p in pours if p != b]
    Wt = POUR[tr[0]] + POUR[tr[1]]
    Mt, H = Wt / 16, POUR[b] / 8
    tv, tx, ty = nash(Mt)
    bestRow, bestCol = Mt.argmax(0), Mt.argmin(1)
    folds.append({
        "youReadThem": float(H[bestRow, np.arange(90)].mean()), "theyReadYou": float(H[np.arange(90), bestCol].mean()),
        "youReadThemPromised": float(Mt.max(0).mean()), "theyReadYouPromised": float(Mt.min(1).mean()),
        "nobody": float(tx @ H @ ty), "youThinkTheyRandom": float((tx @ H).mean()),
        "youRandomTheyThink": float((H @ ty).mean()),
        "examples": {"you": [[int(bestRow[j]), int(Wt[bestRow[j], j]), int(POUR[b][bestRow[j], j])] for j in range(90)],
                     "them": [[int(bestCol[i]), int(Wt[i, bestCol[i]]), int(POUR[b][i, bestCol[i]])] for i in range(90)]}})
mean = lambda f: round(float(np.mean([fo[f] for fo in folds])), 4)
PEEK = {f: mean(f) for f in ("youReadThem", "theyReadYou", "youReadThemPromised", "theyReadYouPromised", "nobody",
                             "youThinkTheyRandom", "youRandomTheyThink")}
PEEK["examples"] = folds[2]["examples"]  # picked on pours 1-2, scored on pour 3: [plan, wins of 16, wins of 8]

THINK = {"randomRandom": round(float(M.mean()), 4), "youThinkTheyRandom": round(float(x @ M.mean(1)), 4),
         "youRandomTheyThink": round(float(M.mean(0) @ y), 4), "bothThink": round(v, 4),
         "heldOut": {k: PEEK[k] for k in ("youThinkTheyRandom", "youRandomTheyThink", "nobody")}}

# Guess Who: the item + ability + move combinations the meta has played on each of MB763's six.
want = {m["id"] for m in THEM}
combos = {s: {} for s in want}
teams_with = {s: 0 for s in want}
n_teams = 0
for (paste,) in db.execute("select paste from team where origin = 'vgcpastes' and legal = 1"):
    n_teams += 1
    for m in parse_paste(paste):
        if m["id"] in want:
            k = (m["item"], m["ability"], tuple(sorted(m["moves"])))
            combos[m["id"]][k] = combos[m["id"]].get(k, 0) + 1
            teams_with[m["id"]] += 1
GUESS = {"teams": n_teams, "boards": []}
for m in THEM:
    own = (m["item"], m["ability"], tuple(sorted(m["moves"])))
    rows = sorted(combos[m["id"]].items(), key=lambda kv: (-kv[1], kv[0]))
    cards = []
    for (item, ability, moves), cnt in rows:
        it = ITEMS / f"item-{key(item)}.png"
        cards.append({"item": item, "ability": ability, "moves": list(moves), "teams": cnt,
                      "itemSprite": f"../item-sprites/{it.name}" if it.exists() else None, "own": (item, ability, moves) == own})
    assert sum(c["own"] for c in cards) == 1, m["id"]
    GUESS["boards"].append({"id": m["id"], "teamsWith": teams_with[m["id"]], "cards": cards})

# The PC box for variant A3: every species legal in Reg M-B (pinned simulator domain), in dex order.
DOMAIN = json.loads((HERE.parent / "team-search-space/domain.json").read_text())
BOX = [{"id": f["id"], "name": f["name"],
        "sprite": f"../search-loop/sprites/{f['id']}.png" if (SPRITES / f"{f['id']}.png").exists() else None}
       for f in sorted(DOMAIN["roster"], key=lambda f: (f["num"], f["id"]))]
assert all(m["id"] in {b["id"] for b in BOX} for m in YOU + THEM)

out = {
    "box": BOX,
    "teams": {"you": YOU, "them": THEM}, "names": {"you": "MB522", "them": "MB763"},
    "rows": ROWS, "cols": COLS, "wins": W.tolist(), "battles": 24,
    "pours": [POUR[b].tolist() for b in pours],
    "eq": EQ, "drop": DROP, "drop2": DROP2, "sand": SAND, "peek": PEEK, "think": THINK, "guess": GUESS,
}
(HERE / "data.js").write_text("// PROTOTYPE — built by build-data.py; do not edit\nwindow.PREVIEW = "
                              + json.dumps(out, separators=(",", ":")) + ";\n")
print(f"value {v:.3f}  supports {len(EQ['x'])}/{len(EQ['y'])}  think {THINK}")
print("peek", {k: PEEK[k] for k in PEEK if k != 'examples'})
print("sand", [(s["battles"], s["value"], len(s["x"]), s["noiseShare"], s["singlePourValue"]) for s in SAND])
print("drop singles you", [(YOU[a]["name"], DROP[f'{a}|-1']['value']) for a in range(6)])
print("drop singles them", [(THEM[b]["name"], DROP[f'-1|{b}']['value']) for b in range(6)])
print("guess", [(b["id"], len(b["cards"])) for b in GUESS["boards"]], "teams", n_teams)
best_home = sorted(((DROP2["-".join(map(str, c)) + "|"]["value"], c) for c in combinations(range(6), 2)), reverse=True)
print("MB522 best two to leave home", [(round(v, 3), [YOU[k]["name"] for k in c]) for v, c in best_home[:3]], "worst", [(round(v, 3), [YOU[k]["name"] for k in c]) for v, c in best_home[-2:]])
print("data.js", round((HERE / "data.js").stat().st_size / 1024), "KB")
