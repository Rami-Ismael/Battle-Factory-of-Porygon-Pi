"""Random-play battles as a random walk: run the battles (2026-10-21).

Question: does a battle where both sides play at random, and the tree of such battles grown from
one start, carry features of a team that the behaviour-cloning win rate also sees?

Teams: 150 rows of the matchup database with a behaviour-cloning label against every top-50 team
(policy 1, at least 4 battles per column), stratified over that win rate in 0.05-wide bins.
Battles (src/random_battle_tree.cjs, random play both sides):
  walk  8 battles against each of the 49 top-50 teams per row (392 per row), per-turn trajectories
  tree  16 spines per row against 16 top-50 teams drawn by weight; 32 rollouts at the root and at
        every turn boundary of a spine
Output: ~/vgc-data/random-battle-graph/{teams.json, walk.jsonl, tree.jsonl}.
Analysis: src/random_battle_graph_report.py.
"""
import json, os, random, sqlite3, subprocess, sys, time
from pathlib import Path

DB = Path.home() / "vgc-data" / "matchup_regmb.sqlite"
OUT = Path.home() / "vgc-data" / "random-battle-graph"
NODE_SCRIPT = Path(__file__).resolve().parent / "random_battle_tree.cjs"
POLICY = 1                      # behaviour-cloning policy both sides (bc_100.zip), the fixed evaluator
PER_BIN, BIN = 10, 0.05
WALK_N, TREE_SPINES, TREE_K = 8, 16, 32
WORKERS = 10
SEED = 20261021


def load():
    db = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    top = dict(db.execute("select team_id, weight from team_set where set_name='top50'"))
    q = """with m as (
             select lo as r, hi as c, lo_wins as w, ties as t, battles as n from cell where policy_id=?
             union all
             select hi, lo, hi_wins, ties, battles from cell where policy_id=?)
           select r, c, w, t, n from m"""
    per = {}
    for r, c, w, t, n in db.execute(q, (POLICY, POLICY)):
        if c in top and r not in top and n > 0:
            per.setdefault(r, {})[c] = (w, t, n)
    rows = []
    for r, cols in per.items():
        if len(cols) == len(top) and min(v[2] for v in cols.values()) >= 4:
            wr = sum(top[c] * (w + 0.5 * t) / n for c, (w, t, n) in cols.items()) / sum(top.values())
            rows.append((r, wr, sum(v[2] for v in cols.values())))
    pastes = {}
    for tid, paste, origin in db.execute("select team_id, paste, origin from team"):
        pastes[tid] = (paste, origin)
    return top, rows, pastes


def select(rows):
    rng = random.Random(SEED)
    bins = {}
    for r, wr, nb in rows:
        bins.setdefault(min(int(wr / BIN), 14), []).append((r, wr, nb))
    chosen = []
    for b in sorted(bins):
        pool = sorted(bins[b])
        chosen += rng.sample(pool, min(PER_BIN, len(pool)))
    return chosen


def run(mode, pairs, teams, extra):
    chunks = [pairs[i::WORKERS] for i in range(WORKERS)]
    out = OUT / f"{mode}.jsonl"
    procs = []
    for i, ch in enumerate(chunks):
        req = {"mode": mode, "teams": teams, "pairs": ch, "seed": f"{SEED}-{mode}", **extra}
        f_in = OUT / f"_{mode}_{i}.json"
        f_in.write_text(json.dumps(req))
        procs.append(subprocess.Popen(["node", str(NODE_SCRIPT)], stdin=open(f_in),
                                      stdout=open(OUT / f"_{mode}_{i}.jsonl", "w"),
                                      stderr=open(OUT / f"_{mode}_{i}.err", "w")))
    t0 = time.time()
    codes = [p.wait() for p in procs]
    with open(out, "w") as g:
        for i in range(WORKERS):
            g.write((OUT / f"_{mode}_{i}.jsonl").read_text())
    n = sum(1 for _ in open(out))
    print(f"{mode}: {n} records in {time.time() - t0:.0f}s, exit codes {codes}", flush=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    top, rows, pastes = load()
    chosen = select(rows)
    print(f"eligible rows {len(rows)}, chosen {len(chosen)}", flush=True)
    meta = {"policy": POLICY, "top50": {str(k): v for k, v in top.items()},
            "rows": [{"team_id": r, "bc_win_rate": wr, "bc_battles": nb, "origin": pastes[r][1]}
                     for r, wr, nb in chosen],
            "walk_n": WALK_N, "tree_spines": TREE_SPINES, "tree_k": TREE_K, "seed": SEED}
    (OUT / "teams.json").write_text(json.dumps(meta, indent=1))
    ids = [r for r, _, _ in chosen] + list(top)
    teams = {str(t): pastes[t][0] for t in ids}
    mode = sys.argv[1] if len(sys.argv) > 1 else "both"
    if mode in ("walk", "both"):
        pairs = [[str(r), str(c), WALK_N] for r, _, _ in chosen for c in sorted(top)]
        run("walk", pairs, teams, {})
    if mode in ("tree", "both"):
        rng = random.Random(SEED + 1)
        cols, w = sorted(top), [top[c] for c in sorted(top)]
        pairs = []
        for r, _, _ in chosen:
            pick = []
            while len(pick) < TREE_SPINES:          # weighted draw without replacement
                c = rng.choices(cols, w)[0]
                if c not in pick: pick.append(c)
            pairs += [[str(r), str(c), 1] for c in pick]
        run("tree", pairs, teams, {"k": TREE_K})


if __name__ == "__main__":
    main()
