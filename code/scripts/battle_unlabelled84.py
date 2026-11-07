"""Battle every team in the 84-field table that has no win rate, so the next lossdata84 build labels it (2026-11-05).

list    lossdata.gather() as lossdata84 runs it, then one line per kept team with no battles: its source family and
        the first paste text it appeared as -> OUT/teams.jsonl
battle  matchup_db.score_pastes in chunks of one family: Showdown validation, then one battle against each top-50
        meta team, behaviour-cloning policy on both sides, slots shuffled. Battled cells are cached in the matchup
        database, so a restart resumes where it stopped -> OUT/scores.jsonl, one line per team

    /tmp/vgc-pilot/.venv/bin/python scripts/battle_unlabelled84.py list|battle   (from code/)
"""
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import repo_paths  # noqa: F401  (src/ first on sys.path)

OUT = Path.home() / "vgc-data" / "unlabelled_battles_84"
TEAMS, SCORES = OUT / "teams.jsonl", OUT / "scores.jsonl"
CHUNK, SEED0, BATTLES_PER_OPPONENT = 500, 9100, 1


def listing():
    import lossdata as LD
    import lossdata84 as L84
    import team84 as T
    seen = {}                                    # id(parsed team) -> (paste text, parsed team kept alive)
    parse = LD.parse_team_text
    def recording_parse(text, *a, **kw):
        t = parse(text, *a, **kw); seen[id(t)] = (text, t); return t
    LD.parse_team_text = recording_parse
    paste_of, sets, Vreg = {}, defaultdict(dict), T.regulation_vocab()
    def observe(k, team):                         # lossdata84.collect's observer, plus the paste text
        if k not in paste_of and id(team) in seen: paste_of[k] = seen[id(team)][0]
        found, sp = sets[k], L84.stat_point_key(team)
        if sp in found: return
        try: found[sp] = T.encode(Vreg, team).astype("int16")
        except ValueError: found[sp] = None
    g = LD.gather(observe)
    _, counts, keys = L84.all_teams(g, sets)
    OUT.mkdir(parents=True, exist_ok=True)
    n = missing = 0
    with open(TEAMS, "w") as f:
        for k in keys:
            e = g["pool"][k]
            if e["battles"]: continue
            if k not in paste_of: missing += 1; continue          # a corpus team: parsed from its file, no paste text kept
            fam = "+".join(x for j, x in enumerate(LD.FAM) if (e["fam"] >> j) & 1)
            f.write(json.dumps(dict(key=hashlib.sha1(k).hexdigest(), family=fam, paste=paste_of[k])) + "\n"); n += 1
    print(json.dumps(dict(table=counts, unlabelled_listed=n, unlabelled_without_paste=missing)), flush=True)


def battle():
    import matchup_db as MDB
    if MDB.SLOT_ORDER != "shuffled": raise SystemExit(f"VGC_SLOT_ORDER={MDB.SLOT_ORDER}: these battles must shuffle slots")
    teams = [json.loads(line) for line in open(TEAMS)]
    done = set()
    if SCORES.exists():
        lines = open(SCORES).read().splitlines()
        for j, line in enumerate(lines):
            try: done.add(json.loads(line)["key"])
            except json.JSONDecodeError:
                if j < len(lines) - 1: raise
                SCORES.write_text("".join(x + "\n" for x in lines[:-1]))   # the next append must start on its own line
                print("dropped a cut-off last line from scores.jsonl; that team is battled again", flush=True)
    todo = sorted((t for t in teams if t["key"] not in done), key=lambda t: t["family"])
    chunks = [todo[i:i + CHUNK] for i in range(0, len(todo), CHUNK)]
    chunks = [[t for t in c if t["family"] == fam] for c in chunks for fam in dict.fromkeys(t["family"] for t in c)]
    con = MDB.connect()
    print(f"{len(teams)} teams, {len(done)} already scored, {len(todo)} to battle in {len(chunks)} chunks", flush=True)
    t0, n = time.time(), 0
    for i, c in enumerate(chunks):
        sc = MDB.score_pastes(con, [t["paste"] for t in c], per_cell=BATTLES_PER_OPPONENT,
                              seed=SEED0 + len(done) + n, origin=f"unlabelled84:{c[0]['family']}")
        with open(SCORES, "a") as f:
            for t, s in zip(c, sc): f.write(json.dumps(dict(key=t["key"], family=t["family"], **s)) + "\n")
        n += len(c)
        legal = sum(s.get("score") is not None for s in sc)
        h = (time.time() - t0) / 3600
        print(f"{time.strftime('%H:%M:%S')} chunk {i + 1}/{len(chunks)} {c[0]['family']}: {legal}/{len(c)} legal · "
              f"{n}/{len(todo)} teams in {h:.2f} h · about {h / n * (len(todo) - n):.1f} h left", flush=True)


if __name__ == "__main__":
    {"list": listing, "battle": battle}[sys.argv[1]]()
