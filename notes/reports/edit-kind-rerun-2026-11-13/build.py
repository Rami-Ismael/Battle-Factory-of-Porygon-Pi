"""Storyboard figures and issue comments for the edit-kind rerun (GitHub issue 313), one folder for each issue.

Copied from notes/reports/masking-rate-map-2026-11-11/build.py. Differences: the plan path, the date, the folder name,
a grid may have up to eight rows, a new frame kind `examples`, and one frame to a row: the sketch on the left and
its sentences on the right in larger type, so each frame has room.

Each folder holds a spec.json written for a reader new to the project. This script draws the spec as a static SVG, light
and dark, with text kept as text, and assembles the comment that embeds it. It follows the diversity result comment on
issue 263: a plain setup paragraph, one figure with its finding as alt text, and a "How to read this" table.

    python3 notes/reports/edit-kind-rerun-2026-11-13/build.py [issue-313 ...] [check] [commit=SHA]   (from the repo root; no dependencies)

Naming folders builds only those. `check` also lists every number in a spec's words that neither the plan nor the
folder's source.txt (the saved issue text) contains. `commit=SHA` sets the commit the comment's image links point at.
A folder whose spec fails is reported and skipped, so one broken spec never blocks the others.

spec.json:
  issue, title, url           the GitHub issue
  headline                    the figure's title: the one sentence a reader should leave with
  setup                       two or three plain sentences that define what the reader needs
  alt                         the figure's finding, for readers who cannot see it
  terms                       [[term, meaning], ...] for the "How to read this" table
  frames                      [{title, kind, data, takes, expected, surprise} or {..., takes, measured}]
  kinds and their data:
    flow       {"boxes": [{"label", "sub", "tone"}]}                      left to right, at most 4
    bars       {"bars": [{"label", "height", "style", "note"}], "line": {"at", "label"}}   height and at are 0 to 1
    lines      {"x", "y", "diagonal", "series": [{"name", "tone", "points": [[x, y], ...]}]}   points are 0 to 1
    grid       {"cols": [...], "rows": [...], "cells": [[{"text", "tone"}, ...], ...]}
    compare    {"left": {"title", "lines": [...]}, "right": {"title", "lines": [...]}}
    squares    {"groups": [{"label", "count", "filled"}], "per_row": 42}
    checklist  {"items": [{"text", "state"}]}                            state is done, next, todo or hold
    examples   {"head": ["", "Before", "After"], "rows": [{"label", "before", "after"}]}   at most 6 rows
  tone is plain, ok, warn, bad, c1, c2 or c3; style is solid, dashed or muted.
"""
import html
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = "Rami-Ismael/Battle-Factory-of-Porygon-Pi"
PLAN = HERE.parents[2] / "code/docs/ruggedness2.md"
THEMES = {
    "light": dict(ink="#0b0b0b", soft="#52514e", box="#f3f2ee", edge="#d5d4cd", c1="#2a78d6", c2="#14946a", c3="#b07800", ok="#16803c", warn="#a16207", bad="#c0392b", plain="#52514e"),
    "dark": dict(ink="#f2f1ee", soft="#a9a8a2", box="#1f1f1e", edge="#3a3a35", c1="#5b9bf0", c2="#2dd4a4", c3="#f5c04a", ok="#4ade80", warn="#fbbf24", bad="#f87171", plain="#a9a8a2"),
}
FONT = '-apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif'
WIDTH, PAD, GAP, CARD_PAD, SKETCH_W, SKETCH_H = 800, 16, 18, 22, 348, 150
CARD_W, COL_GAP = WIDTH - 2 * PAD, 30
TEXT_W = CARD_W - 2 * CARD_PAD - SKETCH_W - COL_GAP


def wrap(text, width, size, bold=False):
    """Greedy word wrap by an average character width, which is close enough for sans-serif labels."""
    per_char = size * (0.58 if bold else 0.53)
    limit, lines, line = max(8, int(width / per_char)), [], ""
    for word in str(text).split():
        if line and len(line) + 1 + len(word) > limit: lines.append(line); line = word
        else: line = f"{line} {word}".strip()
    return [*lines, line] if line else lines or [""]


class Svg:
    def __init__(self, theme): self.t, self.out = THEMES[theme], []

    def color(self, tone): return self.t.get(tone, self.t["soft"])

    def text(self, x, y, s, size=12, tone="soft", bold=False, anchor="start"):
        weight = ' font-weight="650"' if bold else ""
        self.out.append(f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{self.color(tone)}" text-anchor="{anchor}"{weight}>{html.escape(str(s))}</text>')

    def block(self, x, y, s, width, size=12, tone="soft", bold=False, leading=1.4, anchor="start", limit=None):
        """Wrapped text from baseline y; returns the y just under the last line."""
        lines = wrap(s, width, size, bold)[:limit]
        for i, line in enumerate(lines): self.text(x, y + i * size * leading, line, size, tone, bold, anchor)
        return y + len(lines) * size * leading

    def rect(self, x, y, w, h, fill="none", stroke=None, r=6, dash=False, opacity=1.0, width=1):
        dashes = ' stroke-dasharray="4 3"' if dash else ""
        self.out.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{r}" fill="{fill}" fill-opacity="{opacity}"'
                        f' stroke="{stroke or "none"}" stroke-width="{width}"{dashes}/>')

    def path(self, d, stroke, width=1.5, dash=False):
        dashes = ' stroke-dasharray="4 3"' if dash else ""
        self.out.append(f'<path d="{d}" fill="none" stroke="{stroke}" stroke-width="{width}" stroke-linejoin="round"{dashes}/>')


def flow(g, x, y, d, h=SKETCH_H):
    boxes = d["boxes"][:4]; n = len(boxes); arrow = 14
    w = (SKETCH_W - arrow * (n - 1)) / n
    for i, b in enumerate(boxes):
        bx = x + i * (w + arrow); tone = b.get("tone", "plain")
        g.rect(bx, y + 14, w, h - 28, fill=g.t["box"], stroke=g.color(tone) if tone != "plain" else g.t["edge"], width=1.5 if tone != "plain" else 1)
        end = g.block(bx + 8, y + 36, b["label"], w - 16, 12, "ink", True, limit=3)
        g.block(bx + 8, end + 4, b.get("sub", ""), w - 16, 10.5, tone if tone != "plain" else "soft", limit=6)
        if i < n - 1: g.text(bx + w + arrow / 2, y + h / 2 + 4, "→", 14, "soft", anchor="middle")


def bars(g, x, y, d, h=SKETCH_H):
    items = d["bars"][:6]; n = len(items); base, top, slot = y + SKETCH_H - 38, y + 18, SKETCH_W / max(n, 1)
    g.path(f"M{x} {base}H{x + SKETCH_W}", g.t["soft"], 1)
    for i, b in enumerate(items):
        h = max(2.0, float(b["height"]) * (base - top)); bw = min(48, slot * 0.55); bx = x + i * slot + (slot - bw) / 2
        style, tone = b.get("style", "solid"), b.get("tone", "c1")
        if style == "dashed": g.rect(bx, base - h, bw, h, stroke=g.color(tone), r=2, dash=True, width=1.5)
        else: g.rect(bx, base - h, bw, h, fill=g.t["soft"] if style == "muted" else g.color(tone), r=2)
        if b.get("note"): g.text(bx + bw / 2, base - h - 5, b["note"], 11, "ink", True, "middle")
        g.block(x + i * slot + slot / 2, base + 12, b["label"], slot - 4, 10, leading=1.2, anchor="middle", limit=3)
    line = d.get("line")
    if line:
        ly = base - float(line["at"]) * (base - top)
        g.path(f"M{x} {ly}H{x + SKETCH_W}", g.t["soft"], 1, True)
        width = len(line["label"]) * 5.7 + 8                 # a patch behind the label keeps it readable over a bar
        g.rect(x + SKETCH_W - width, ly - 15, width, 13, fill=g.t["box"], r=3, opacity=0.9); g.text(x + SKETCH_W - 4, ly - 5, line["label"], 10.5, anchor="end")


def lines(g, x, y, d, h=SKETCH_H):
    """Curves under a legend, so two lines that overlap still each carry a name."""
    series = d["series"][:4]
    lx, ly = x + 26, y + 10
    for s in series:
        tone = s.get("tone", "c1"); width = 18 + len(s["name"]) * 6.1 + 12
        if lx + width > x + SKETCH_W + 8 and lx > x + 26: lx, ly = x + 26, ly + 14      # the legend wraps onto a new row
        g.path(f"M{lx} {ly}H{lx + 14}", g.color(tone), 2.2, s.get("style") == "dashed")
        g.text(lx + 18, ly + 3.5, s["name"], 10.5, tone, True); lx += width
    left, base, top, right = x + 26, y + SKETCH_H - 24, ly + 16, x + SKETCH_W - 6
    g.path(f"M{left} {top}V{base}H{right}", g.t["soft"], 1)
    g.text((left + right) / 2, base + 17, d.get("x", ""), 10.5, anchor="middle")
    g.out.append(f'<text x="{x + 9}" y="{(top + base) / 2}" font-size="10.5" fill="{g.t["soft"]}" text-anchor="middle" transform="rotate(-90 {x + 9} {(top + base) / 2})">{html.escape(d.get("y", ""))}</text>')
    px, py = lambda v: left + v * (right - left), lambda v: base - v * (base - top)
    if d.get("diagonal"): g.path(f"M{px(0)} {py(0)}L{px(1)} {py(1)}", g.t["soft"], 1, True)
    for s in series:
        g.path("M" + "L".join(f"{px(a):.1f} {py(b):.1f}" for a, b in s["points"]), g.color(s.get("tone", "c1")), 2.2, s.get("style") == "dashed")


def grid(g, x, y, d, h=SKETCH_H):
    cols, rows = d["cols"][:3], d["rows"][:8]; head_w, head_h = 78, 34
    cw, ch = (SKETCH_W - head_w) / len(cols), (h - head_h - 4) / len(rows)
    for j, c in enumerate(cols): g.block(x + head_w + j * cw + cw / 2, y + 12, c, cw - 6, 10.5, leading=1.2, anchor="middle", limit=2)
    for i, r in enumerate(rows):
        g.block(x, y + head_h + i * ch + ch / 2 - 2, r, head_w - 6, 10.5, limit=2)
        for j, cell in enumerate(d["cells"][i][:len(cols)]):
            tone = cell.get("tone", "plain"); cx, cy = x + head_w + j * cw + 2, y + head_h + i * ch + 2
            g.rect(cx, cy, cw - 4, ch - 4, fill=g.color(tone) if tone != "plain" else g.t["box"], opacity=0.2 if tone != "plain" else 1, stroke=g.t["edge"])
            n = len(wrap(cell["text"], cw - 12, 11, True)[:3])
            g.block(cx + (cw - 4) / 2, cy + (ch - 4) / 2 - (n - 1) * 7.5 + 4, cell["text"], cw - 12, 11, "ink", True, 1.35, "middle", 3)


def compare(g, x, y, d, h=SKETCH_H):
    w = (SKETCH_W - 12) / 2
    for i, side in enumerate((d["left"], d["right"])):
        bx = x + i * (w + 12); tone = side.get("tone", "plain")
        g.rect(bx, y + 4, w, h - 8, fill=g.t["box"], stroke=g.color(tone) if tone != "plain" else g.t["edge"], width=1.5 if tone != "plain" else 1)
        end = g.block(bx + 8, y + 24, side["title"], w - 16, 12, "ink", True, limit=2)
        for item in side["lines"][:6]: end = g.block(bx + 8, end + 3, item, w - 16, 10.5, limit=3)


def squares(g, x, y, d, h=SKETCH_H):
    per_row = d.get("per_row", 28); size = (SKETCH_W - (per_row - 1) * 2) / per_row; n = 0
    for group in d["groups"]:
        for _ in range(int(group["count"])):
            sx, sy = x + (n % per_row) * (size + 2), y + 12 + (n // per_row) * (size + 3)
            if group.get("filled"): g.rect(sx, sy, size, size, fill=g.color(group.get("tone", "c1")), r=1.5)
            else: g.rect(sx, sy, size, size, stroke=g.t["soft"], r=1.5, dash=True)
            n += 1
    end = y + 12 + ((n - 1) // per_row + 1) * (size + 3) + 14
    for filled, word in ((True, "Filled"), (False, "Hidden")):
        names = ", ".join(f'{grp["label"]} {grp["count"]}' for grp in d["groups"] if bool(grp.get("filled")) == filled)
        if names: end = g.block(x, end, f"{word}: {names}", SKETCH_W, 10.5, "ink" if filled else "soft") + 3


def checklist(g, x, y, d, h=SKETCH_H):
    marks = dict(done=("✓", "ok"), next=("→", "c1"), todo=("○", "soft"), hold=("‖", "warn"))
    end = y + 18
    for item in d["items"][:8]:
        mark, tone = marks.get(item.get("state", "todo"), marks["todo"])
        g.text(x + 4, end, mark, 13, tone, True); end = g.block(x + 24, end, item["text"], SKETCH_W - 28, 11.5, "ink" if item.get("state") != "todo" else "soft", limit=3) + 5


def examples(g, x, y, d, h=SKETCH_H):
    """Rows of one real change each: a label, what was there, an arrow, what replaced it."""
    head = d.get("head", ["", "Before", "After"]); before, after, top = x + 98, x + 232, y + 16
    g.text(before, top, head[1], 10.5, "soft", True); g.text(after, top, head[2], 10.5, "soft", True)
    for i, row in enumerate(d["rows"][:6]):
        ry = top + 10 + i * EXAMPLE_ROW
        g.path(f"M{x} {ry}H{x + SKETCH_W}", g.t["edge"], 1)
        g.block(x, ry + 19, row["label"], 90, 11.5, "ink", True, 1.25, limit=2)
        g.block(before, ry + 19, row["before"], 112, 10.5, "soft", False, 1.3, limit=3)
        g.text(after - 18, ry + 19, "→", 12, "c1", True)
        g.block(after, ry + 19, row["after"], 114, 10.5, "ink", False, 1.3, limit=3)


EXAMPLE_ROW = 48
KINDS = dict(flow=flow, bars=bars, lines=lines, grid=grid, compare=compare, squares=squares, checklist=checklist, examples=examples)
LABELS = (("takes", "The point", "ink"), ("measured", "What we measured", "c1"), ("expected", "Expected", "ok"), ("surprise", "Surprise", "warn"))


def sketch_height(frame):
    """The room a frame's sketch needs: the usual height, or more for a long checklist, comparison or flow."""
    d, kind, need = frame["data"], frame["kind"], SKETCH_H
    if kind == "checklist":
        need = 22 + sum(len(wrap(i["text"], SKETCH_W - 28, 11.5)[:3]) * 11.5 * 1.4 + 5 for i in d["items"][:8])
    elif kind == "compare":
        w = (SKETCH_W - 12) / 2 - 16
        side = lambda s: 30 + len(wrap(s["title"], w, 12, True)[:2]) * 16.8 + sum(len(wrap(i, w, 10.5)[:3]) * 14.7 + 3 for i in s["lines"][:6])
        need = max(side(d["left"]), side(d["right"])) + 8
    elif kind == "flow":
        n = len(d["boxes"][:4]); w = (SKETCH_W - 14 * (n - 1)) / n - 16
        need = max(60 + len(wrap(b["label"], w, 12, True)[:3]) * 16.8 + len(wrap(b.get("sub", ""), w, 10.5)[:6]) * 14.7 for b in d["boxes"][:4])
    elif kind == "examples":
        need = 38 + EXAMPLE_ROW * len(d["rows"][:6])
    elif kind == "grid":
        need = 38 + 34 * len(d["rows"][:8])      # the original drew three rows; a longer grid gets 34 px a row
    return max(SKETCH_H, need)


def notes_height(frame):
    """The room the sentences beside a sketch need: a small label, then the sentence in 13 px type."""
    return sum(18 + len(wrap(frame[key], TEXT_W, 13)) * 19.5 + 14 for key, _, _ in LABELS if frame.get(key))


def card_height(frame):
    title = len(wrap(frame["title"], CARD_W - 2 * CARD_PAD, 14.5, True)) * 20
    return CARD_PAD + title + 12 + max(sketch_height(frame), notes_height(frame)) + CARD_PAD


def draw(spec, theme):
    """One frame to a row: its sketch on the left, its sentences on the right."""
    g = Svg(theme); y = PAD + 20
    y = g.block(PAD, y, spec["headline"], WIDTH - 2 * PAD, 18, "ink", True, 1.35) + 10
    for f in spec["frames"]:
        height, x = card_height(f), PAD
        g.rect(x, y, CARD_W, height, fill=g.t["box"], stroke=g.t["edge"], r=10, opacity=0.55)
        ty = g.block(x + CARD_PAD, y + CARD_PAD + 12, f["title"], CARD_W - 2 * CARD_PAD, 14.5, "ink", True, 20 / 14.5)
        KINDS[f["kind"]](g, x + CARD_PAD, ty - 2, f["data"], sketch_height(f))
        nx, ny = x + CARD_PAD + SKETCH_W + COL_GAP, ty + 16
        for key, label, tone in LABELS:
            if not f.get(key): continue
            g.text(nx, ny, label, 11, tone if tone != "ink" else "soft", True)
            ny = g.block(nx, ny + 21, f[key], TEXT_W, 13, "ink", False, 1.5) + 12
        y += height + GAP
    total = y + PAD - GAP
    return (f'<svg width="{WIDTH}" height="{total:.0f}" viewBox="0 0 {WIDTH} {total:.0f}" xmlns="http://www.w3.org/2000/svg" font-family=\'{FONT}\' role="img">'
            f'<title>{html.escape(spec["alt"])}</title>{"".join(g.out)}</svg>')


def comment(spec, commit):
    raw = f"https://raw.githubusercontent.com/{REPO}/{commit}/notes/reports/{HERE.name}/issue-{spec['issue']}/figures/storyboard"
    rows = "\n".join(f"| {a} | {b} |" for a, b in spec["terms"])
    return (f"**Storyboard for [{spec['title']}]({spec['url']}), 2026-11-13.**\n\n{spec['setup']}\n\n"
            f'<picture><source media="(prefers-color-scheme: dark)" srcset="{raw}-dark.svg"><img width="800" alt="{html.escape(spec["alt"], quote=True)}" src="{raw}-light.svg"></picture>\n\n'
            f"### How to read this\n\n| Term | What it means here |\n|---|---|\n{rows}\n\n"
            f"The frames and their sentences are in `notes/reports/{HERE.name}/issue-{spec['issue']}/`. The plan is `code/docs/ruggedness2.md`.\n")


def storyboard(spec, old):
    """storyboard.md from the spec, so the sentences beside the figure never drift from it. The maker's own Cut list,
    what was left out and why, is kept from the file already there."""
    out = [f"# Storyboard: {spec['title']}", "", f"Figure sentence: {spec['headline']}.", ""]
    for f in spec["frames"]:
        out += [f"## {f['title']}", "", f"- Sentence: {f['takes']}"]
        out += [f"- {label}: {f[key]}" for key, label, _ in LABELS[1:] if f.get(key)] + [""]
    cut = old[old.index("## Cut"):].strip() if "## Cut" in old else "## Cut\n\n- Nothing recorded."
    return "\n".join(out) + cut + "\n"


def numbers(text): return set(re.findall(r"\d[\d,]*\.?\d*", text))


def strings(node):
    """Every piece of text in a spec, without the numbers that only place a drawing."""
    if isinstance(node, str): return [node]
    if isinstance(node, dict): return [t for v in node.values() for t in strings(v)]
    if isinstance(node, list): return [t for v in node for t in strings(v)]
    return []


def main(argv):
    commit = next((a.split("=", 1)[1] for a in argv if a.startswith("commit=")), "COMMIT")
    plan = PLAN.read_text() if PLAN.exists() else ""
    only = [a for a in argv if a.startswith("issue-")]
    for folder in sorted(HERE.glob("issue-*")):
        if only and folder.name not in only: continue
        try:
            spec = json.loads((folder / "spec.json").read_text())
            pictures = {theme: draw(spec, theme) for theme in THEMES}
            text = comment(spec, commit)
        except Exception as error:      # noqa: BLE001  one broken spec is reported, the rest still build
            print(f"{folder.name}: FAILED, {type(error).__name__}: {error}"); continue
        (folder / "figures").mkdir(exist_ok=True)
        for theme, picture in pictures.items(): (folder / "figures" / f"storyboard-{theme}.svg").write_text(picture)
        (folder / "comment.md").write_text(text)
        notes = folder / "storyboard.md"
        notes.write_text(storyboard(spec, notes.read_text() if notes.exists() else ""))
        print(f"{folder.name}: OK, {len(spec['frames'])} frames, {len(spec['terms'])} terms")
        if "check" in argv:
            source = plan + ((folder / "source.txt").read_text() if (folder / "source.txt").exists() else "")
            said = " ".join(strings(spec))
            unknown = sorted(n for n in numbers(said) if len(n.strip(".,")) > 1 and n.strip(".,") not in {m.strip(".,") for m in numbers(source)} and n.strip(".,") != str(spec["issue"]))
            if unknown: print("   numbers not in the plan or the issue text:", ", ".join(unknown))


if __name__ == "__main__":
    main(sys.argv[1:])
