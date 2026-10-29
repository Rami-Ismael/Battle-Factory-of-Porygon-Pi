"""Build post1-jev-vs-mercury.html, its scroll-first version 2 (-v2), the animated versions 3 and 4, and version 5 (-v5, with 95% ranges), the first build-in-public figure: Jev against Mercury on the 270 held-out tasks.

Reads ~/vgc-data/jev/heldout-report.json (win rates per mask) and the held-out grid's mercury.jsonl (why each
Mercury fill was rejected), and writes the numbers into post1-jev-vs-mercury.template.html.

    python3 notes/Blog/visuals/jev-picks-a-team/build_post1_figure.py
"""
import json
import re
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
VAULT_RUNS = Path("/Users/ramiismael/Documents/yakumsi-vault/Personal Project/Using Advance method search in the whole search of possible vgc pokemon format to find the right counter to a pokemon team/team_completion_pipeline/runs/heldout-starters")
SPRITES = json.loads((HERE.parent / "diversity-methods/sprites.json").read_text())


def reason(error: str) -> str:
    """Why a Mercury fill was rejected, in one of three plain kinds."""
    if "unmasked field changed" in error:
        return "changed"
    if "JSONDecodeError" in error or "no answer text" in error:
        return "unreadable"
    return "rule"


def main() -> None:
    masks = json.loads((Path.home() / "vgc-data/jev/heldout-report.json").read_text())["masks"]
    rows = {json.loads(line)["task_id"]: json.loads(line) for line in open(VAULT_RUNS / "mercury.jsonl")}
    mercury = []
    for m in masks:
        r = rows[m["task_id"]]
        mercury.append("legal" if m.get("mercury") is not None else reason(str(r.get("error"))))
    both = [m for m in masks if m.get("mercury") is not None]
    data = {
        "tasks": len(masks),
        "mercury": mercury,
        "counts": {k: mercury.count(k) for k in ("legal", "changed", "rule", "unreadable")},
        "shared": {"n": len(both), "jev": round(statistics.mean(m["jev_top"] for m in both), 3),
                   "mercury": round(statistics.mean(m["mercury"] for m in both), 3)},
        "usable": {"jev": round(sum(m["jev_top"] for m in masks) / len(masks), 3),
                   "mercury": round(sum(m["mercury"] or 0 for m in masks) / len(masks), 3)},
        "examples": ["Kleavor can't learn Avalanche", "Archaludon given 68 Stat Points (the limit is 66)",
                     "two Pokémon holding Choice Scarf"],
        "sprite": SPRITES["kleavor"],
    }
    data["team"] = [{"species": s, "sprite": SPRITES[re.sub(r"[^a-z0-9]", "", s.lower())]}
                    for s in ("Armarouge", "Aerodactyl", "Kingambit", "Garchomp", "Metagross", "Grimmsnarl")]   # unmasked team MB514
    rows: dict[str, list[str]] = {}
    for m, status in zip(masks, mercury):           # one row of 54 tasks per unmasked team, in task order
        rows.setdefault(m["source"], []).append(status)
    data["rows"] = list(rows.values())
    numbers = json.loads((Path.home() / "vgc-data/jev/measurements.json").read_text())    # jev_numbers.py
    held = numbers["grids"]["held-out"]["arms"]
    data["ci"] = {"jev": held["Jev, top pick"]["usable"]["ci_teams"], "mercury": held["Mercury, high"]["usable"]["ci_teams"]}
    for name in ("post1-jev-vs-mercury", "post1-jev-vs-mercury-v2", "post1-jev-vs-mercury-v3", "post1-jev-vs-mercury-v4"):
        page = (HERE / f"{name}.template.html").read_text().replace("__DATA__", json.dumps(data))
        (HERE / f"{name}.html").write_text(page)
    # Version 5 shows the shuffled re-battle against the top-50 teams (8 battles per opponent); legality is unchanged.
    re50 = numbers["rebattle"]["top50"]
    arms, pair = re50["arms"], re50["pairs"]["Mercury, high"]
    v5 = dict(data, shared={"n": pair["masks"], "jev": round(pair["jev"], 3), "mercury": round(pair["other"], 3),
                            "diff": pair["diff"], "ci": pair["ci_teams"]},
              usable={"jev": round(arms["Jev, top pick"]["usable"]["mean"], 3), "mercury": round(arms["Mercury, high"]["usable"]["mean"], 3)},
              ci={"jev": arms["Jev, top pick"]["usable"]["ci_teams"], "mercury": arms["Mercury, high"]["usable"]["ci_teams"]})
    # The fair check's clause, rounded as docs/jev-measurements.md rounds: name the leader only when the interval clears zero.
    p = lambda x: f"{x * 100:.1f}"
    lo, hi = pair["ci_teams"]
    if lo > 0:
        v5["fair"] = (f"Jev's teams won slightly more often: {p(pair['jev'])}% against {p(pair['other'])}%, "
                      f"{p(pair['diff'])} points ahead (95% interval {p(lo)} to {p(hi)})")
    elif hi < 0:
        v5["fair"] = (f"Mercury's teams won slightly more often: {p(pair['other'])}% against {p(pair['jev'])}%, "
                      f"{p(-pair['diff'])} points ahead (95% interval {p(-hi)} to {p(-lo)})")
    else:
        v5["fair"] = (f"they won about equally often: {p(pair['jev'])}% vs {p(pair['other'])}% "
                      f"(95% interval for the gap {p(lo)} to {p(hi)} points)")
    page = (HERE / "post1-jev-vs-mercury-v5.template.html").read_text().replace("__DATA__", json.dumps(v5))
    (HERE / "post1-jev-vs-mercury-v5.html").write_text(page)
    print(data["counts"], data["shared"], data["usable"], "v5:", v5["shared"], v5["usable"])


if __name__ == "__main__":
    main()
