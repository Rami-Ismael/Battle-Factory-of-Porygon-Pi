"""Build mercury-dial.html: Mercury 2.5 at four reasoning settings on the held-out grid, one row of charts per question.

Win rates, intervals and costs are experiment 12's numbers, read from the COST12B constant in jev-picks-a-team.html so
the two pages cannot disagree; same-mask comparisons with Jev come from ~/vgc-data/jev/measurements.json
(scripts/jev_numbers.py). Why each fill failed, call latency and prompt-cache hits come from the four held-out
Mercury logs. Every Mercury call was billed at the 80%-off launch price ($0.04 / $0.15 per million tokens; list price
$0.20 / $0.75, Inception's Mercury 2.5 launch post), so the list-price cost is the billed cost times five.

    python3 notes/Blog/visuals/jev-picks-a-team/build_mercury_dial.py
"""
import json
import re
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = Path("/Users/ramiismael/Documents/yakumsi-vault/Personal Project/Using Advance method search in the whole search of possible vgc pokemon format to find the right counter to a pokemon team/team_completion_pipeline/runs/heldout-starters")
LOGS = {"off": "mercury-off", "low": "mercury-low", "medium": "mercury-medium", "high": "mercury"}
LIST_OVER_BILLED = 5                                  # $0.20 / $0.04 input and $0.75 / $0.15 output


def reason(row: dict) -> str:
    """Why a fill ended as it did, in the four kinds the figure shows (build_post1_figure.py's split, plus legal)."""
    error = str(row.get("error"))
    if row["status"] == "legal":
        return "legal"
    if "unmasked field changed" in error:
        return "changed"
    if "JSONDecodeError" in error or "no answer text" in error:
        return "unreadable"
    return "rule"


def main() -> None:
    page = (HERE / "jev-picks-a-team.html").read_text()
    cost12b = json.loads(re.search(r"const COST12B *= *(\{.*?\});\n", page).group(1))
    levels = {}
    for level, name in LOGS.items():
        rows = [json.loads(line) for line in (RUNS / f"{name}.jsonl").read_text().splitlines()]
        last = {r["task_id"]: r for r in rows}          # a rerun's row replaces the earlier one for its task
        final = [r for r in last.values() if r["status"] != "transport"]
        assert len(final) == 270, (level, len(final))
        kinds = {k: sum(reason(r) == k for r in final) for k in ("legal", "changed", "rule", "unreadable")}
        assert kinds["legal"] == cost12b[level]["legal"], (level, kinds)
        usage = [r["usage"] for r in rows if r.get("usage")]
        levels[level] = dict(cost12b[level], kinds=kinds,
                             seconds=round(statistics.median(r["seconds"] for r in rows if r.get("seconds")), 1),
                             cached=sum(bool((u.get("prompt_tokens_details") or {}).get("cached_tokens")) for u in usage),
                             list_cost_per_legal=round(cost12b[level]["cost_per_legal"] * LIST_OVER_BILLED, 6))
    pairs = json.loads((Path.home() / "vgc-data/jev/measurements.json").read_text())["grids"]["held-out"]["pairs"]
    for level in levels:                                 # Jev and Mercury on the masks both filled legally (jev_numbers.py)
        levels[level]["pair"] = pairs[f"Mercury, {level}"]
    data = {"levels": levels, "jev": cost12b["jev"], "unmasked": cost12b["unmasked"], "random_fill": cost12b["random_fill"]}
    out = (HERE / "mercury-dial.template.html").read_text().replace("__DATA__", json.dumps(data))
    (HERE / "mercury-dial.html").write_text(out)
    for level, r in levels.items():
        print(level, r["kinds"], r["seconds"], "s", r["cached"], "cached", f"{r['list_cost_per_legal'] * 100:.2f}¢ at list")


if __name__ == "__main__":
    main()
