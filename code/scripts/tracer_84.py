"""Tracer bullet for issue #57: the 84-field model end to end at toy size, against the real Showdown validator and
one real battle. Model quality does not matter here; every stage has to run.

    /tmp/vgc-pilot/.venv/bin/python scripts/tracer_84.py   (from code/)

Data: the training corpus (unlabelled) plus labelled teams from the old asked-vs-got pool, about 2,000 in all,
each with the Stat Points it was battled with. Writes ~/vgc-data/results/tracer_84.json.
"""
import json
import tempfile
import time
from pathlib import Path

import numpy as np

import repo_paths  # noqa: F401  (src/ first on sys.path)
import asked_vs_got as X
import diffusion84 as M
import heldout as HO
import matchup_db as MDB
import pool
import team84 as T
from corpus import parse_team_text
from hps_generate import Validator

N_TEAMS, PASSES, N_SAMPLE, ASK, GUIDANCE, SEED = 2000, 5, 64, 0.5, 2.0, 0
OUT = Path.home() / "vgc-data/results/tracer_84.json"


def training_rows(V, corpus):
    rows, labels = [], []
    for team in corpus:
        try: rows.append(T.encode(V, team)); labels.append(np.nan)
        except ValueError: pass
    for r in json.load(open(X.DATA))["rows"]:
        if len(rows) >= N_TEAMS: break
        team = parse_team_text(r["paste"])
        if len(team) != 6 or HO.is_heldout_team(team): continue
        try: rows.append(T.encode(V, team)); labels.append(r["y"])
        except ValueError: pass
    return np.stack(rows), np.array(labels, np.float32)


def one_meta_opponent():
    con = MDB.connect()
    tid = MDB.set_members(con, "top50")[0]
    (paste,) = con.execute("SELECT paste FROM team WHERE team_id=?", (tid,)).fetchone()
    return tid, paste


def main():
    t0 = time.perf_counter()
    V = T.regulation_vocab(); corpus = X.setup()[0]
    rows, labels = training_rows(V, corpus)
    log = []
    model = M.Model84.train_new(V, corpus, rows, labels, passes=PASSES, seed=SEED, log=log)
    print(f"trained on {len(rows)} teams ({int(np.isfinite(labels).sum())} labelled): "
          + ", ".join(f"pass {e['pass_']} loss {e['train_loss']:.3f}" for e in log), flush=True)

    sampled = model.sample(N_SAMPLE, ask=ASK, guidance=GUIDANCE, seed=SEED).cpu().numpy()
    pastes = [model.to_paste(r, repair=True)[0] for r in sampled]
    val = Validator()
    try: errors = [val(p) for p in pastes]
    finally: val.close()
    valid = [p for p, e in zip(pastes, errors) if e is None]
    totals = [sum(s["evs"].values()) for r in sampled for s in T.decode(V, r)]
    print(f"Showdown accepted {len(valid)} of {N_SAMPLE}; Stat Point totals {min(totals)}–{max(totals)}, "
          f"{np.mean(np.array(totals) == 66):.0%} of Pokémon spend exactly 66", flush=True)

    battle = None
    if valid:
        tid, opp = one_meta_opponent()
        with tempfile.TemporaryDirectory() as d:
            mine, theirs = Path(d) / "tracer.txt", Path(d) / "opponent.txt"
            mine.write_text(valid[0]); theirs.write_text(opp)
            res = pool.score([(str(mine), [str(theirs)])], battles=1, quiet=True)
        battle = dict(opponent_team_id=tid, **res[str(mine)])
        print(f"one battle against top-50 meta team {tid}: {battle}", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(dict(n_train=len(rows), n_labelled=int(np.isfinite(labels).sum()), passes=PASSES, log=log,
                   ask=ASK, guidance=GUIDANCE, sampled=N_SAMPLE, accepted=len(valid),
                   rejections=[str(e).split("\n")[0][:160] for e in errors if e is not None][:10],
                   share_spending_66=float(np.mean(np.array(totals) == 66)), battle=battle,
                   example_paste=valid[0] if valid else None, minutes=(time.perf_counter() - t0) / 60),
              open(OUT, "w"), indent=1)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
