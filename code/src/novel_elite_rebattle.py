"""Do the novel elite teams hold up when battled again? (2026-10-31)

Two decoding schemes tied on novel teams at or above the elite cut (0.531) per 1,000 tries: species-first order
(12 teams, decoding_order.json) and confidence order with the lookahead (11 teams, lookahead_legality.json). Novel:
the team's six species never appear together in either network's training rows. Each kept team had one score, its
win rate over one battle against each top-50 meta team (49 distinct, one counted twice), so some teams over the cut
got lucky, and more so in the scheme with the lower average (0.313 against 0.396).

Rule, fixed before any battle: every one of those teams is re-battled in a private matchup database (no battle read
back from the original runs), 4 battles against each of the 49 distinct teams, behaviour-cloning policy on both sides,
slots shuffled; the team listed twice counts twice. A team **holds** if its fresh win rate is at least 0.531. Report per
scheme: teams that hold, mean fresh win rate, and the difference in means between the schemes with a 95% interval
(Welch). Confidence order with the lookahead **matches** species-first on novel strong teams if at least as many of
its teams hold and the interval's lower end is above -0.03.

    python novel_elite_rebattle.py         (run with /tmp/vgc-pilot/.venv/bin/python)
"""
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, "/tmp/vgc-pilot/src")
sys.path.insert(0, str(HERE))
from discrete_diffusion_model import DiscreteDiffusionModel

sys.path.insert(0, str(HERE))                       # discrete_diffusion_model's imports put /tmp/vgc-pilot/src first again
import decoding_order as DO
import matchup_db as MDB
import top50 as T50
from hps_generate import Validator

RES = Path.home() / "vgc-data" / "results"
OUT = RES / "novel_elite_rebattle.json"
DB = Path.home() / "vgc-data" / "novel_elite_rebattle" / "fresh.sqlite"
CUT, PER_CELL, SEED, BOUND = 0.531, 4, 7420, -0.03
SCHEMES = [("species", "Species-first order", "decoding_order.json"),
           ("confidence_la", "Confidence order + lookahead", "lookahead_legality.json")]


def novel_elites(lineups):
    """Per scheme, (paste, original score) for each kept team at or above the cut whose line-up is not in training."""
    out = {}
    for arm, _, fname in SCHEMES:
        cells = json.loads((RES / fname).read_text())["cells"].values()
        out[arm] = [(p, y) for c in cells if c["arm"] == arm for p, r, y in zip(c["pastes"], c["rows"], c["y"])
                    if y is not None and y >= CUT and tuple(sorted(r[s * DO.NF] for s in range(DO.NSLOT))) not in lineups]
    return out


def fresh_db():
    """A private matchup database whose column set is the top-50 meta teams, so every battle here is new."""
    DB.parent.mkdir(parents=True, exist_ok=True)
    con = MDB.connect(DB)
    pid = MDB.get_policy(con, note="novel elite re-battle, slots shuffled, 2026-10-31")
    if "+slots-shuffled" not in con.execute("SELECT row_policy FROM policy WHERE policy_id=?", (pid,)).fetchone()[0]:
        raise SystemExit("these battles must shuffle slots")
    val = Validator()
    try:
        with con:
            con.execute("DELETE FROM team_set WHERE set_name='top50'")
            for t in json.load(open(T50.TOP50)):              # a team listed twice counts twice
                tid = MDB.add_team(con, Path(T50.resolve(t["id"])).read_text(), "vgcpastes", val)[0]
                con.execute("INSERT INTO team_set VALUES ('top50', ?, 1) ON CONFLICT (set_name, team_id) "
                            "DO UPDATE SET weight = weight + 1", (tid,))
    finally:
        val.close()
    return con


def mean_sd(v):
    m = sum(v) / len(v)
    return m, (math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1)) if len(v) > 1 else 0.0)


def welch(a, b):
    """Difference in means b - a with a 95% interval (normal approximation; Welch standard error)."""
    (ma, sa), (mb, sb) = mean_sd(a), mean_sd(b)
    se = math.sqrt(sa * sa / len(a) + sb * sb / len(b))
    return mb - ma, mb - ma - 1.96 * se, mb - ma + 1.96 * se


def main():
    if MDB.SLOT_ORDER != "shuffled": raise SystemExit(f"VGC_SLOT_ORDER={MDB.SLOT_ORDER}: these battles must shuffle slots")
    _, lineups, _ = DO.training_sets(DiscreteDiffusionModel.load())
    teams, con = novel_elites(lineups), fresh_db()
    out = {"cut": CUT, "per_cell": PER_CELL, "seed": SEED, "schemes": {}}
    for arm, name, _ in SCHEMES:
        sc = MDB.score_pastes(con, [p for p, _ in teams[arm]], per_cell=PER_CELL, seed=SEED, origin=f"novel-elite:{arm}")
        if not all(s["score"] is not None and s["coverage"] == s["columns"] and s["battles"] >= PER_CELL * s["columns"] for s in sc):
            raise SystemExit(f"{arm}: a team lacks some fresh battles")
        rows = [dict(paste=p, label=y, fresh=s["score"], battles=s["battles"]) for (p, y), s in zip(teams[arm], sc)]
        f = [r["fresh"] for r in rows]; m, sd = mean_sd(f)
        out["schemes"][arm] = dict(name=name, teams=rows, hold=sum(x >= CUT for x in f), n=len(f), fresh_mean=m, fresh_sd=sd,
                                   label_mean=sum(r["label"] for r in rows) / len(rows))
        print(f"{name:30s} {len(f)} teams · label mean {out['schemes'][arm]['label_mean']:.3f} → fresh mean {m:.3f} "
              f"(sd {sd:.3f}) · hold {out['schemes'][arm]['hold']} · fresh {sorted(round(x, 3) for x in f)}", flush=True)
    s, c = out["schemes"]["species"], out["schemes"]["confidence_la"]
    d, lo, hi = welch([r["fresh"] for r in s["teams"]], [r["fresh"] for r in c["teams"]])
    out["difference"] = dict(diff=d, lo=lo, hi=hi, matches=c["hold"] >= s["hold"] and lo > BOUND)
    print(f"confidence + lookahead minus species-first: {d:+.3f} [{lo:+.3f}, {hi:+.3f}] · matches: {out['difference']['matches']}")
    OUT.write_text(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
