"""Build post1-jev-vs-mercury.html, its scroll-first version 2 (-v2), the animated versions 3 and 4, version 5 (-v5, with 95% ranges), and version 6 (-v6, Mercury with its tuned prompt), the first build-in-public figure: Jev against Mercury on the 270 held-out tasks.

Reads ~/vgc-data/jev/heldout-report.json (win rates per mask) and the held-out grid's mercury.jsonl (why each
Mercury fill was rejected), and writes the numbers into post1-jev-vs-mercury.template.html.

    python3 notes/Blog/visuals/jev-picks-a-team/build_post1_figure.py
"""
import json
import re
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
VAULT_RUNS = Path("/Users/ramiismael/Documents/yakumsi-vault/Personal Project/Using Advance method search in the whole search of possible vgc pokemon format to find the right counter to a pokemon team/team_completion_pipeline/runs/heldout-starters")
REBATTLE = Path.home() / "vgc-data/jev/rebattle-heldout.json"                     # jev_rebattle.py
TUNED = Path.home() / "vgc-data/mercury-dspy/heldout/high/completions"            # mercury_dspy_grid.py
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
    v6 = tuned(masks, data)
    page = (HERE / "post1-jev-vs-mercury-v6.template.html").read_text().replace("__DATA__", json.dumps(v6))
    (HERE / "post1-jev-vs-mercury-v6.html").write_text(page)
    print(data["counts"], data["shared"], data["usable"], "v5:", v5["shared"], v5["usable"],
          "v6:", v6["counts"], v6["shared"], v6["usable"], v6["ci"])


def tuned_reason(record: dict) -> str:
    """Why a tuned-Mercury answer was rejected: a field it wasn't asked for counts as changing the team it was told to keep."""
    error = str(record.get("error"))
    if "extra_forbidden" in error:
        return "changed"
    return reason(error)


def tuned(masks: list[dict], data: dict) -> dict:
    """Version 6: Mercury with its tuned prompt (GEPA + GPT-6 Luna, legal-choice types, high reasoning), from the shuffled
    re-battle against the top-50 teams. Every number is rounded once, from the per-mask scores."""
    sys.path.insert(0, str(HERE.parents[3] / "code/scripts"))
    from jev_rebattle import interval
    scores = json.loads(REBATTLE.read_text())["sets"]["top50"]["scores"]
    jev, merc = scores["jev"], scores["mercury-dspy-high"]
    status = []
    for m in masks:
        record = json.loads((TUNED / f"{m['task_id']}.json").read_text())
        legal = merc.get(m["task_id"]) is not None
        status.append("legal" if legal else tuned_reason(record) if not record["valid"] else "rule")   # valid but failed the grid's check
    exact = lambda pairs, key: {k: [x / 1000 for x in v] if isinstance(v, list) else v / 1000   # interval rounds to 4 places;
                                for k, v in interval([(s, d * 1000) for s, d in pairs], key).items() if k != "masks"}  # scale first
    usable = {name: exact([(m["source"], s.get(m["task_id"]) or 0) for m in masks], "mean") for name, s in (("jev", jev), ("mercury", merc))}
    both = [m for m in masks if jev.get(m["task_id"]) is not None and merc.get(m["task_id"]) is not None]
    gap = exact([(m["source"], jev[m["task_id"]] - merc[m["task_id"]]) for m in both], "diff")
    shared = {"n": len(both), "jev": statistics.mean(jev[m["task_id"]] for m in both),
              "mercury": statistics.mean(merc[m["task_id"]] for m in both), "diff": gap["diff"], "ci": gap["ci_teams"]}
    counts = {k: status.count(k) for k in ("legal", "changed", "rule", "unreadable")}
    p = lambda x: f"{x * 100:.1f}"
    n = round(shared["diff"] * 100)
    points = f"{n} point{'' if n == 1 else 's'}"
    lo, hi = shared["ci"]
    assert lo > 0, "the fair-check wording below says Jev is ahead"
    fair = (f"Jev's teams won slightly more often: {p(shared['jev'])}% against {p(shared['mercury'])}%, about {points} ahead (95% interval for the gap {p(lo)} to {p(hi)} points)")
    (jlo, jhi), (mlo, mhi) = usable["jev"]["ci_teams"], usable["mercury"]["ci_teams"]
    overlap = mhi >= jlo and jhi >= mlo
    gap_all = usable["jev"]["mean"] - usable["mercury"]["mean"]
    lost = sum(jev[m["task_id"]] for m in masks if merc.get(m["task_id"]) is None) / len(masks)   # Jev's wins where Mercury's team is illegal
    plural = lambda n, one, many: f"{n} {one if n == 1 else many}"
    caps = {"changed": plural(counts["changed"], "changed a part of the team it was told to keep", "changed parts of the team it was told to keep"),
            "rule": plural(counts["rule"], "broke a game rule", "broke a game rule"),
            "unreadable": plural(counts["unreadable"], "answer couldn't be read", "answers couldn't be read"),
            "both": f"Both legal ({len(both)} tasks): {p(shared['jev'])}% vs {p(shared['mercury'])}%, Jev about {points} ahead.",
            "ranges": (f"The ranges overlap: tuned Mercury is close. About {lost * 100:.0f} of the {gap_all * 100:.1f} points "
                       f"between them come from its {len(masks) - counts['legal']} illegal teams." if overlap else
                       "The two ranges don't overlap, so the gap is unlikely to be chance.")}
    rows: dict[str, list[str]] = {}
    for m, s in zip(masks, status):                # one row of 54 tasks per unmasked team, as in version 1
        rows.setdefault(m["source"], []).append(s)
    return dict(data, mercury=status, counts=counts, rows=list(rows.values()),
                shared=shared, usable={k: v["mean"] for k, v in usable.items()},
                ci={k: v["ci_teams"] for k, v in usable.items()}, fair=fair, caps=caps,
                plain_legal=sum(m.get("mercury") is not None for m in masks),
                examples=["Whimsicott can't learn Icy Wind", "Whimsicott given 50 Stat Points in Speed (the limit is 32)",
                          "two Pokémon holding Choice Scarf"], sprite=SPRITES["whimsicott"])


if __name__ == "__main__":
    main()
