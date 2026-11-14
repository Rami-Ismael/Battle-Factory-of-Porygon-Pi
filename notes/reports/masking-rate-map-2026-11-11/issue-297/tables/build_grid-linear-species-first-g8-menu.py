"""Draw the guidance-8 grid table of Linear, species-first as a restaurant menu card.

Run:  python build_grid-linear-species-first-g8-menu.py [output folder]

Each measure is a dish: its name on the left, a dotted leader, then two prices, teal for
Linear, species-first and blue for the control. A first render measures every text, so the
leaders end where each price begins; the second render draws the card.
"""
import json
import re
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[4] / "code" / "src"))
from excalidraw_table import missing_from_scene, render_scene  # noqa: E402

NAME = "grid-linear-species-first-g8-menu"
HEADER = ["Measure", "Linear, species-first, guidance 8", "The control at guidance 8"]
ROWS = [
    ["Ask-versus-got error, asks 1/24 to 17/24, with its 95% interval", "0.0038 (0.0017 to 0.0074)", "0.0033 (0.0014 to 0.0068)"],
    ["The same error for sampling seeds 1, 2 and 3", "0.0035, 0.0040, 0.0039", "0.0034, 0.0036, 0.0032"],
    ["Error on asks 18/24 to 23/24", "0.1087", "0.0909"],
    ["Win rate at asks 15/24 to 17/24", "0.539 (0.497 to 0.582)", "0.556 (0.518 to 0.594)"],
    ["Win rate over all 23 asks", "0.385 (0.358 to 0.413)", "0.390 (0.364 to 0.418)"],
    ["First-try legality", "100.0%", "100.0%"],
    ["Distinct line-ups", "2,060", "2,126"],
    ["Teams with a line-up absent from the training teams", "35.6%", "31.9%"],
    ["Teams with a line-up absent from earlier search runs and tournament records", "85.7%", "86.2%"],
    ["Effective species and top-species share, in 5-team subsets", "13.1 and 0.145", "13.2 and 0.147"],
    ["Teams and battles", "4,416 and 216,384", "4,416 and 216,384"],
]
GROUPS = [[0, 1, 2], [3, 4], [5, 6, 7, 8], [9, 10]]          # ornaments sit between these groups of rows; the groups carry no names

CREAM, INK, BROWN = "#fff9db", "#3b2a14", "#8a6a3b"
TEAL, TEAL_STRONG, BLUE, BLUE_STRONG = "#96f2d7", "#099268", "#a5d8ff", "#1971c2"
TEAL_INK = "#087f5b"                                         # darker teal for the prices: 4.5 to 1 or more on the cream card
HAND, CLEAN = 5, 6
WIDTH = 1260
NAME_LEFT, NAME_CHARS, NAME_SIZE, LINE = 64, 44, 24, 30
COLUMNS = [(690, 940, TEAL, TEAL_INK), (975, 1220, BLUE, BLUE_STRONG)]   # left, right, fill, strong for each price column
VALUE_SIZE, NOTE_SIZE = 22, 18


def wrapped(text, limit):
    lines, line = [], ""
    for word in text.split():
        if line and len(line) + 1 + len(word) > limit:
            lines.append(line); line = word
        else:
            line = f"{line} {word}".strip()
    return [*lines, line]


def price(cell):
    """The price and, when the cell has one, its interval on a second line."""
    match = re.fullmatch(r"(.+?) (\(.+\))", cell)
    return [match[1], match[2]] if match else [cell]


def text(id_, x, y, words, size, font, colour, align="left"):
    return {"type": "text", "id": id_, "x": x, "y": y, "text": words, "fontSize": size, "fontFamily": font, "strokeColor": colour, "textAlign": align}


def pieces():
    """Every text of the card, unplaced: (id, words, size, font)."""
    found = []
    for r, row in enumerate(ROWS):
        for l, line in enumerate(wrapped(row[0], NAME_CHARS)):
            found.append((f"name-{r}-{l}", line, NAME_SIZE, HAND))
        for c in (1, 2):
            for p, part in enumerate(price(row[c])):
                found.append((f"price-{r}-{c}-{p}", part, VALUE_SIZE if p == 0 else NOTE_SIZE, CLEAN))
    return found


def measure():
    with tempfile.TemporaryDirectory() as work:
        render_scene([text(id_, 0, 60 * i, words, size, font, INK) for i, (id_, words, size, font) in enumerate(pieces())], work, "measure")
        elements = json.loads((Path(work) / "measure.excalidraw").read_text(encoding="utf-8"))["elements"]
    return {element["id"]: element["width"] for element in elements}


def holder(id_, x, y, width, height, words, size, font, colour, align="center", **shape):
    base = {"type": "rectangle", "id": id_, "x": x, "y": y, "width": width, "height": height,
            "strokeColor": "transparent", "backgroundColor": "transparent", "roughness": 0}
    base.update(shape)
    base["label"] = {"text": words, "fontSize": size, "fontFamily": font, "textAlign": align, "verticalAlign": "middle", "strokeColor": colour}
    return base


def ornament(id_, y):
    middle = WIDTH / 2
    return [
        {"type": "line", "id": f"{id_}-l", "x": middle - 230, "y": y, "points": [[0, 0], [90, -6], [190, 0]], "strokeColor": BROWN, "strokeWidth": 2, "roughness": 1},
        {"type": "line", "id": f"{id_}-r", "x": middle + 40, "y": y, "points": [[0, 0], [100, -6], [190, 0]], "strokeColor": BROWN, "strokeWidth": 2, "roughness": 1},
        {"type": "diamond", "id": f"{id_}-d", "x": middle - 16, "y": y - 13, "width": 32, "height": 26, "strokeColor": BROWN, "backgroundColor": "#ffd8a8", "fillStyle": "solid", "strokeWidth": 2, "roughness": 1},
        {"type": "ellipse", "id": f"{id_}-a", "x": middle - 250, "y": y - 5, "width": 10, "height": 10, "strokeColor": BROWN, "backgroundColor": BROWN, "fillStyle": "solid", "roughness": 1},
        {"type": "ellipse", "id": f"{id_}-b", "x": middle + 240, "y": y - 5, "width": 10, "height": 10, "strokeColor": BROWN, "backgroundColor": BROWN, "fillStyle": "solid", "roughness": 1},
    ]


def card(widths):
    front, back = [], []
    y = 50
    front.append(holder("title", 40, y, WIDTH - 80, 56, "Grid finished: Linear, species-first, guidance 8", 40, HAND, INK))
    y += 70
    flourish = [[0, 0], [70, -14], [140, 0], [210, 14], [280, 0]]
    for side, x in (("fl", WIDTH / 2 - 320), ("fr", WIDTH / 2 + 40)):
        front.append({"type": "line", "id": side, "x": x, "y": y, "points": flourish, "strokeColor": BROWN, "strokeWidth": 2, "roughness": 1})
    front.append({"type": "ellipse", "id": "fc", "x": WIDTH / 2 - 14, "y": y - 14, "width": 28, "height": 28, "strokeColor": BROWN, "backgroundColor": "#ffc9c9", "fillStyle": "solid", "strokeWidth": 2, "roughness": 1})
    y += 34
    front.append(holder("lead", 80, y, WIDTH - 160, 40, "Linear, species-first at guidance 8 matches the control: error 0.0038 against 0.0033.", 22, CLEAN, BROWN))
    y += 70
    front.append(text("h-measure", NAME_LEFT, y + 22, HEADER[0], 26, HAND, BROWN))
    for c, (left, right, fill, strong) in enumerate(COLUMNS, start=1):
        front.append(holder(f"h-{c}", left, y, right - left, 70, HEADER[c].replace(" guidance", "\nguidance"), 20, CLEAN, INK,
                            backgroundColor=fill, fillStyle="solid", strokeColor=strong, strokeWidth=2, roughness=1, roundness={"type": 3}))
    y += 96
    for g, group in enumerate(GROUPS):
        if g:
            front.extend(ornament(f"orn-{g}", y + 12)); y += 40
        for r in group:
            row = ROWS[r]
            lines = wrapped(row[0], NAME_CHARS)
            for l, line in enumerate(lines):
                front.append(text(f"name-{r}-{l}", NAME_LEFT, y + l * LINE, line, NAME_SIZE, HAND, INK))
            last = y + (len(lines) - 1) * LINE
            baseline = last + NAME_SIZE * 1.0
            starts = []
            for c, (left, right, fill, strong) in enumerate(COLUMNS, start=1):
                parts = price(row[c])
                first = widths[f"price-{r}-{c}-0"]
                starts.append(right - first)
                front.append(text(f"price-{r}-{c}-0", right - first, last + 2, parts[0], VALUE_SIZE, CLEAN, strong))
                if len(parts) > 1:
                    note = widths[f"price-{r}-{c}-1"]
                    front.append(text(f"price-{r}-{c}-1", right - note, last + 32, parts[1], NOTE_SIZE, CLEAN, strong))
            leader_from = NAME_LEFT + widths[f"name-{r}-{len(lines) - 1}"] + 12
            back.append({"type": "line", "id": f"lead-{r}-1", "x": leader_from, "y": baseline, "points": [[0, 0], [starts[0] - 14 - leader_from, 0]],
                         "strokeColor": BROWN, "strokeWidth": 2, "strokeStyle": "dotted", "roughness": 0})
            if starts[1] - 26 - COLUMNS[0][1] >= 24:                  # the second leader shows only where the two prices leave room for it
                back.append({"type": "line", "id": f"lead-{r}-2", "x": COLUMNS[0][1] + 12, "y": baseline, "points": [[0, 0], [starts[1] - 14 - COLUMNS[0][1] - 12, 0]],
                             "strokeColor": BROWN, "strokeWidth": 2, "strokeStyle": "dotted", "roughness": 0})
            tall = 58 if any(len(price(row[c])) > 1 for c in (1, 2)) else LINE
            y = last + tall + 22
    front.extend(ornament("orn-end", y + 14)); y += 46
    front.append(holder("verdict", 80, y, WIDTH - 160, 40, "The verdict at this guidance is matches.", 28, HAND, TEAL_STRONG))
    y += 70
    frame = [
        {"type": "rectangle", "id": "card", "x": 0, "y": 0, "width": WIDTH, "height": y, "strokeColor": BROWN, "backgroundColor": CREAM, "fillStyle": "solid", "strokeWidth": 4, "roughness": 1},
        {"type": "rectangle", "id": "card-inner", "x": 16, "y": 16, "width": WIDTH - 32, "height": y - 32, "strokeColor": BROWN, "backgroundColor": "transparent", "strokeWidth": 1.5, "roughness": 2},
    ]
    return frame + back + front


def main(folder):
    folder = Path(folder)
    render_scene(card(measure()), folder, NAME)
    missing = missing_from_scene(HEADER, ROWS, folder / f"{NAME}.excalidraw")
    if missing: print("missing:", missing)
    assert not missing


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else HERE)
