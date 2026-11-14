"""Draw the guidance 4 control grid's table as a shop till receipt, and export it as light and dark PNGs.

Run:  python build_grid-control-g4-receipt.py [output folder]
"""
import sys
import textwrap
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[4] / "code" / "src"))
from excalidraw_table import missing_from_scene, plain, render_scene  # noqa: E402

HEADER = ["Measure", "Per-field, uncorrected t, guidance 4"]
ROWS = [
    ["Ask-versus-got error, asks 1/24 to 17/24, with its 95% interval", "0.0049 (0.0022 to 0.0091)"],
    ["The same error for sampling seeds 1, 2 and 3", "0.0057, 0.0047, 0.0045"],
    ["Error on asks 18/24 to 23/24", "0.1079"],
    ["Win rate at asks 15/24 to 17/24", "0.532 (0.492 to 0.574)"],
    ["Win rate over all 23 asks", "0.375 (0.349 to 0.403)"],
    ["First-try legality", "100.0%"],
    ["Distinct line-ups", "1,918"],
    ["Teams with a line-up absent from the training teams", "24.4%"],
    ["Teams with a line-up absent from earlier search runs and tournament records", "84.4%"],
    ["Effective species and top-species share, in 5-team subsets", "14.2 and 0.143"],
    ["Teams and battles", "4,416 and 216,384"],
]
GROUPS = [[0, 1, 2], [3, 4], [5, 6, 7, 8], [9]]                    # item lines between rows of dashes; row 10 is the bottom line

INK, PAPER, MONO = "#1864ab", "#f3f9ff", 8                         # dark blue ink of the control row, Comic Shanns
WIDTH, LEFT, RIGHT = 760, 44, 716
ADVANCE = 0.55                                                     # Comic Shanns is monospaced: every character is 0.55 of the font size wide
ITEM, BIG, TITLE = 20, 26, 34
WRAP = 30                                                          # characters per line of an item name


def text(id, x, y, words, size, align="left"):
    width = max(len(line) for line in words.split("\n")) * ADVANCE * size
    x = {"left": x, "right": x - width, "center": x - width / 2}[align]
    return {"type": "text", "id": id, "x": x, "y": y, "text": words, "fontSize": size, "fontFamily": MONO,
            "strokeColor": INK}


def dashes(id, y):
    count = int((RIGHT - LEFT) / (ADVANCE * ITEM))
    return text(id, WIDTH / 2, y, "-" * count, ITEM, "center")


def leader(id, x0, x1, y):
    return {"type": "line", "id": id, "x": x0, "y": y, "points": [[0, 0], [x1 - x0, 0]], "strokeColor": INK,
            "strokeWidth": 2, "strokeStyle": "dotted", "roughness": 0}


def item(r, top, size):
    """One item line: the name at the left, wrapped, the value at the right on the name's last line, a dotted leader between."""
    name, value = plain(ROWS[r][0]), plain(ROWS[r][1])
    lines = textwrap.wrap(name, WRAP, break_on_hyphens=False) if size == ITEM else [name]
    step = 1.25 * size
    parts = [text(f"name-{r}", LEFT, top, "\n".join(lines), size),
             text(f"value-{r}", RIGHT, top + step * (len(lines) - 1), value, size, "right")]
    start = LEFT + len(lines[-1]) * ADVANCE * size + 8
    end = RIGHT - len(value) * ADVANCE * size - 16
    if end - start > 12:
        parts.append(leader(f"leader-{r}", start, end, top + step * (len(lines) - 1) + size * 0.95))
    return parts, top + step * len(lines)


def zigzag(top, bottom, tooth=20, depth=14):
    """The torn paper outline: zigzag along the top and the bottom, straight sides, as one closed line."""
    teeth = WIDTH // tooth
    points = [[i * tooth / 2, 0 if i % 2 else depth] for i in range(2 * teeth + 1)]
    points += [[WIDTH, bottom - top - depth]]
    points += [[WIDTH - i * tooth / 2, bottom - top - (0 if i % 2 else depth)] for i in range(1, 2 * teeth + 1)]
    points += [[0, depth]]
    return {"type": "line", "id": "paper", "x": 0, "y": top, "points": points, "strokeColor": INK, "strokeWidth": 1,
            "backgroundColor": PAPER, "fillStyle": "solid", "roughness": 1}


def skeleton():
    body, y = [], 44
    body.append(text("title", WIDTH / 2, y, "GRID FINISHED", TITLE, "center")); y += 1.25 * TITLE + 10
    body.append(dashes("rule-top", y)); y += 1.25 * ITEM + 2
    body.append(text("header-0", LEFT, y, plain(HEADER[0]), ITEM))
    body.append(text("header-1", RIGHT, y, plain(HEADER[1]), ITEM, "right"))
    y += 1.25 * ITEM + 2
    for g, group in enumerate(GROUPS):
        body.append(dashes(f"rule-{g}", y)); y += 1.25 * ITEM + 10
        for r in group:
            parts, y = item(r, y, ITEM)
            body += parts; y += 14
    body.append({"type": "line", "id": "total-rule", "x": LEFT, "y": y + 6, "points": [[0, 0], [RIGHT - LEFT, 0]],
                 "strokeColor": INK, "strokeWidth": 3, "roughness": 0})
    body.append({"type": "line", "id": "total-rule-2", "x": LEFT, "y": y + 12, "points": [[0, 0], [RIGHT - LEFT, 0]],
                 "strokeColor": INK, "strokeWidth": 3, "roughness": 0})
    y += 26
    parts, y = item(10, y, BIG)
    body += parts; y += 20
    body.append(dashes("rule-end", y)); y += 1.25 * ITEM + 16
    bars, x = [], LEFT + 120                                         # a barcode of plain strokes
    for i, w in enumerate([3, 1, 2, 1, 4, 1, 1, 3, 2, 1, 3, 1, 2, 4, 1, 2, 1, 3, 1, 1, 2, 3, 1, 4, 2, 1, 1, 3, 2, 1, 3, 2, 1, 1, 4, 1, 2, 3]):
        if i % 2 == 0:
            bars.append({"type": "rectangle", "id": f"bar-{i}", "x": x, "y": y, "width": w * 3, "height": 60,
                         "strokeColor": "transparent", "backgroundColor": INK, "fillStyle": "solid", "roughness": 0})
        x += w * 3 + 2
    shift = (WIDTH - (x - LEFT - 120)) / 2 - (LEFT + 120)
    for bar in bars: bar["x"] += shift
    body += bars; y += 60 + 40
    return [zigzag(0, y), *body]


if __name__ == "__main__":
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE
    name = "grid-control-g4-receipt"
    for path in render_scene(skeleton(), folder, name): print(path)
    missing = missing_from_scene(HEADER, ROWS, folder / f"{name}.excalidraw")
    if missing: print("missing:", missing)
    assert not missing
