"""Build jev-tickets.html, the Jev baseline ticket board, from the ticket files in .scratch/jev-baseline/issues/.

Rerun after a ticket changes:  python3 notes/Blog/visuals/jev-picks-a-team/build_tickets.py
The ticket files are the source of truth; STAGE below is the one judgement the files do not record
(which tickets are done, which is current), and the checkboxes stay as the owner ticks them.
"""
import html
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
ISSUES = HERE.parents[3] / ".scratch/jev-baseline/issues"
OUT = HERE / "jev-tickets.html"
JEV_PAGE = "https://claude.ai/artifact/KMaYSMi5xQkxyGjNN8DTri"
UPDATED = "2026-10-27"
# done = the work is finished; current = work finished, the owner's ticks outstanding; next; waiting = blocked
STAGE = {**{n: "done" for n in range(1, 10)}, 10: "current", 11: "next", **{n: "waiting" for n in range(12, 20)}}


def inline(text: str) -> str:
    t = html.escape(text)
    t = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", t)
    t = re.sub(r"`([^`]+)`", r"<code>\1</code>", t)
    return re.sub(r"\((https://[^)\s]+)\)", r'(<a href="\1">link</a>)', t)


def parse(path: Path) -> dict:
    lines = path.read_text().splitlines()
    num, title = re.match(r"# (\d+): (.+)", lines[0]).groups()
    t = {"n": int(num), "title": title, "build": "", "blocked": "", "checks": [], "notes": [], "experiment": None}
    m = re.match(r"Experiment (\d+): (.+)", title)
    if m:
        t["experiment"], t["title"] = int(m.group(1)), m.group(2)
    for line in lines[1:]:
        if line.startswith("**What to build:**"):
            t["build"] = inline(line.split(":**", 1)[1].strip())
        elif line.startswith("**Blocked by:**"):
            t["blocked"] = line.split(":**", 1)[1].strip()
        elif re.match(r"- \[[ x]\] ", line):
            t["checks"].append({"done": line[3] == "x", "text": inline(line[6:])})
        elif line.startswith(("**Status:**", "Source:")) or not line.strip():
            continue
        else:
            t["notes"].append(inline(line.lstrip("- ")))
    t["stage"] = STAGE[t["n"]]
    return t


tickets = [parse(p) for p in sorted(ISSUES.glob("[0-9][0-9]-*.md"))]
data = json.dumps({"tickets": tickets, "jev": JEV_PAGE, "updated": UPDATED}, ensure_ascii=False)
page = (HERE / "jev-tickets.template.html").read_text().replace("__DATA__", data.replace("</", "<\\/"))
OUT.write_text(page)
print(f"{OUT.name}: {len(tickets)} tickets, {sum(t['stage'] == 'done' for t in tickets)} done")
