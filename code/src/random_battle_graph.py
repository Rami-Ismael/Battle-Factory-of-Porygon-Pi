"""Random-play battles as a random walk: run the battles (2026-10-21).

Question: does a battle where both sides play at random, and the tree of such battles grown from
one start, carry features of a team that the behaviour-cloning win rate also sees?

Teams: 150 rows of the matchup database with a behaviour-cloning label against every top-50 team
(policy 1, at least 4 battles per column): up to 10 rows per 0.05-wide band of that win rate from
0 to 0.70, and one band for 0.70 and above.
Battles (src/random_battle_tree.cjs, random play both sides). A spine is one random battle from team
preview to the end; a rollout is a random battle played to the end from a saved copy of its state.
  walk  8 battles against every team in the top-50 set per row (49 distinct teams here, so 392 per
        row), with Pokemon left and HP every turn
  tree  16 spines per row against 16 top-50 teams drawn by weight; 32 rollouts at the start and at
        every turn of a spine
  branch  the page's Figure 1: one matchup, 3 random continuations per turn for 3 turns, 48 rollouts
        per node, each leaf played to the end with 32 rollouts per turn
Output: ~/vgc-data/random-battle-graph/{teams.json, walk.jsonl, tree.jsonl, branch_<row>_<col>.json},
and run_<mode>.json per run (records, errors, seconds, decisions, default-choice fallbacks).
Run: python src/random_battle_graph.py [walk|tree|branch|all]. Analysis: src/random_battle_graph_report.py.
"""
import json, random, sqlite3, subprocess, sys, time
from pathlib import Path

DB = Path.home() / "vgc-data" / "matchup_regmb.sqlite"
OUT = Path.home() / "vgc-data" / "random-battle-graph"
NODE_SCRIPT = Path(__file__).resolve().parent / "random_battle_tree.cjs"
POLICY = 1                      # behaviour-cloning policy both sides (bc_100.zip), the fixed evaluator
PER_BIN, BIN = 10, 0.05
WALK_N, TREE_SPINES, TREE_K = 8, 16, 32
WORKERS = 10
SEED = 20261021
FIG_ROW, FIG_COL = 46151, 585   # Figure 1: the sample's best team against a top-50 Incineroar team
BRANCH = {"branch": 3, "depth": 3, "k": 48, "k_path": 32, "seed": "figure-20261021"}
DESCRIBE = {"walk": "independent random battles", "tree": "spines with rollouts at every turn",
            "branch": "Figure 1 tree"}


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
    paste_of, origin_of = {}, {}
    for tid, paste, origin in db.execute("select team_id, paste, origin from team"):
        paste_of[tid], origin_of[tid] = paste, origin
    return top, rows, paste_of, origin_of


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
    seconds = time.time() - t0
    with open(out, "w") as g:
        for i in range(WORKERS):
            g.write((OUT / f"_{mode}_{i}.jsonl").read_text())
    write_run_log(mode, out, seconds, codes, [OUT / f"_{mode}_{i}.err" for i in range(WORKERS)])


def write_run_log(mode, out, seconds, codes, err_files):
    """run_<mode>.json: what the report needs to say how the run went."""
    recs = [json.loads(l) for l in open(out)]
    tally = {"decisions": 0, "fallback": 0}
    for f in err_files:
        for line in f.read_text().splitlines():
            if line.startswith('{"done"'):
                d = json.loads(line)
                for k in tally: tally[k] += d.get(k, 0)
    log = {"mode": mode, "records": len(recs), "errors": sum("error" in r for r in recs),
           "seconds": round(seconds, 1), "exit_codes": codes, **tally}
    (OUT / f"run_{mode}.json").write_text(json.dumps(log))
    print(f"{DESCRIBE[mode]}: {log}", flush=True)


def run_branch(paste_of):
    """Figure 1's tree: one Node process, deterministic given BRANCH['seed']."""
    req = {"mode": "branch", "teams": {str(t): paste_of[t] for t in (FIG_ROW, FIG_COL)},
           "pairs": [[str(FIG_ROW), str(FIG_COL), 1]], **BRANCH}
    f_in, out, err = OUT / "_branch.json", OUT / f"branch_{FIG_ROW}_{FIG_COL}.json", OUT / "_branch.err"
    f_in.write_text(json.dumps(req))
    t0 = time.time()
    code = subprocess.call(["node", str(NODE_SCRIPT)], stdin=open(f_in), stdout=open(out, "w"), stderr=open(err, "w"))
    write_run_log("branch", out, time.time() - t0, [code], [err])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    top, rows, paste_of, origin_of = load()
    chosen = select(rows)
    print(f"eligible rows {len(rows)}, chosen {len(chosen)}", flush=True)
    meta = {"policy": POLICY, "top50": {str(k): v for k, v in top.items()},
            "rows": [{"team_id": r, "bc_win_rate": wr, "bc_battles": nb, "origin": origin_of[r]}
                     for r, wr, nb in chosen],
            "walk_n": WALK_N, "tree_spines": TREE_SPINES, "tree_k": TREE_K, "seed": SEED}
    (OUT / "teams.json").write_text(json.dumps(meta, indent=1))
    ids = [r for r, _, _ in chosen] + list(top)
    teams = {str(t): paste_of[t] for t in ids}
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode in ("walk", "all"):
        pairs = [[str(r), str(c), WALK_N] for r, _, _ in chosen for c in sorted(top)]
        run("walk", pairs, teams, {})
    if mode in ("branch", "all"):
        run_branch(paste_of)
    if mode in ("tree", "all"):
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
