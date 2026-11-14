"""Draw the guidance-4 grid table for Linear, species-first as a railway departure board.

Each measure is one row of the board. The measure name sits where a train's destination goes, and
the two values sit on dark split-flap tiles under the teal and the blue column headers.

Run:  python build_grid-linear-species-first-g4-departure-board.py [output folder]
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[4] / "code" / "src"))
from excalidraw_table import missing_from_scene, plain, render_scene  # noqa: E402

NAME = "grid-linear-species-first-g4-departure-board"
HEADER = ["Measure", "Linear, species-first, guidance 4", "The control at guidance 4"]
ROWS = [
    ["Ask-versus-got error, asks 1/24 to 17/24, with its 95% interval", "0.0049 (0.0022 to 0.0092)", "0.0049 (0.0022 to 0.0091)"],
    ["The same error for sampling seeds 1, 2 and 3", "0.0062, 0.0041, 0.0048", "0.0057, 0.0047, 0.0045"],
    ["Error on asks 18/24 to 23/24", "0.1210", "0.1079"],
    ["Win rate at asks 15/24 to 17/24", "0.527 (0.486 to 0.569)", "0.532 (0.492 to 0.574)"],
    ["Win rate over all 23 asks", "0.373 (0.345 to 0.401)", "0.375 (0.349 to 0.403)"],
    ["First-try legality", "100.0%", "100.0%"],
    ["Distinct line-ups", "1,971", "1,918"],
    ["Teams with a line-up absent from the training teams", "26.4%", "24.4%"],
    ["Teams with a line-up absent from earlier search runs and tournament records", "83.7%", "84.4%"],
    ["Effective species and top-species share, in 5-team subsets", "14.1 and 0.143", "14.2 and 0.143"],
    ["Teams and battles", "4,416 and 216,384", "4,416 and 216,384"],
]

TEAL, BLUE = ("#96f2d7", "#099268"), ("#a5d8ff", "#1971c2")
BOARD, FLAP, SPLIT, STEEL, STEEL_EDGE = "#1e1e1e", "#343a40", "#000000", "#adb5bd", "#495057"
WHITE, PALE_YELLOW, YELLOW = "#ffffff", "#fff3bf", "#fcc419"
NUNITO, LILITA, COMIC = 6, 7, 8

LEFT, RIGHT = 20, 1260                     # outer edge of the metal frame
INNER = 20                                 # frame thickness
MEASURE_X, MEASURE_W = 70, 530
COLUMNS = [(616, 290, TEAL), (922, 290, BLUE)]
TOP_ROWS, ROW_PITCH, TILE_H = 318, 100, 82
HINGE = 22                                 # length of each hinge stub at a tile's left and right edges


def value_lines(cell):
    """Break a value before its interval, so each flap holds a short line."""
    text = plain(cell)
    if " (" in text: return text.replace(" (", "\n(")
    if text.count(", ") == 2:
        first, rest = text.split(", ", 1)
        return f"{first},\n{rest}"
    return text


def holder(id_, x, y, width, height, text, size, font, colour, align="left", vertical="middle"):
    return {"type": "rectangle", "id": id_, "x": x, "y": y, "width": width, "height": height,
            "strokeColor": "transparent", "backgroundColor": "transparent", "roughness": 0,
            "label": {"text": text, "fontSize": size, "fontFamily": font, "textAlign": align, "verticalAlign": vertical, "strokeColor": colour}}


def skeleton():
    bottom = TOP_ROWS + ROW_PITCH * len(ROWS) + 10
    s = []
    # the ceiling beam and the two posts the board hangs from
    s.append({"type": "rectangle", "id": "beam", "x": 120, "y": 0, "width": 1040, "height": 22, "backgroundColor": STEEL, "fillStyle": "cross-hatch",
              "strokeColor": STEEL_EDGE, "strokeWidth": 2, "roughness": 0})
    for i, px in enumerate((220, 1036)):
        s.append({"type": "rectangle", "id": f"post-{i}", "x": px, "y": 22, "width": 24, "height": 70, "backgroundColor": STEEL, "fillStyle": "solid",
                  "strokeColor": STEEL_EDGE, "strokeWidth": 2, "roughness": 0})
    # the metal frame and the dark board inside it
    s.append({"type": "rectangle", "id": "frame", "x": LEFT, "y": 90, "width": RIGHT - LEFT, "height": bottom - 90 + INNER,
              "backgroundColor": STEEL, "fillStyle": "solid", "strokeColor": STEEL_EDGE, "strokeWidth": 4, "roughness": 0, "roundness": {"type": 3}})
    s.append({"type": "rectangle", "id": "board", "x": LEFT + INNER, "y": 90 + INNER, "width": RIGHT - LEFT - 2 * INNER, "height": bottom - 90 - INNER,
              "backgroundColor": BOARD, "fillStyle": "solid", "strokeColor": STEEL_EDGE, "strokeWidth": 2, "roughness": 0})
    for i, (bx, by) in enumerate([(LEFT + 6, 96), (RIGHT - 18, 96), (LEFT + 6, bottom + INNER - 18), (RIGHT - 18, bottom + INNER - 18)]):
        s.append({"type": "ellipse", "id": f"bolt-{i}", "x": bx, "y": by, "width": 12, "height": 12, "backgroundColor": STEEL_EDGE, "fillStyle": "solid",
                  "strokeColor": STEEL_EDGE, "strokeWidth": 1, "roughness": 0})
    # the board's title
    s.append({"type": "text", "id": "title", "x": MEASURE_X, "y": 128, "text": "DEPARTURES", "fontSize": 48, "fontFamily": LILITA, "strokeColor": YELLOW})
    s.append(holder("subtitle", 400, 132, 812, 50, "Grid finished: Linear, species-first, guidance 4", 26, NUNITO, WHITE, "right"))
    s.append({"type": "line", "id": "title-rule", "x": MEASURE_X - 20, "y": 200, "points": [[0, 0], [1162, 0]], "strokeColor": "#868e96", "strokeWidth": 2})
    # the column headers: the measure where the destination goes, then the two values
    s.append(holder("head-0", MEASURE_X, 214, MEASURE_W, 86, HEADER[0], 26, LILITA, PALE_YELLOW, "left", "bottom"))
    for c, (x, w, (pale, strong)) in enumerate(COLUMNS, start=1):
        s.append({"type": "rectangle", "id": f"head-{c}", "x": x, "y": 214, "width": w, "height": 86, "backgroundColor": strong, "fillStyle": "solid",
                  "strokeColor": pale, "strokeWidth": 2, "roughness": 0, "roundness": {"type": 3},
                  "label": {"text": HEADER[c], "fontSize": 22, "fontFamily": NUNITO, "textAlign": "center", "verticalAlign": "middle", "strokeColor": WHITE}})
    # one row of the board per measure
    for r, row in enumerate(ROWS):
        y = TOP_ROWS + ROW_PITCH * r
        if r: s.append({"type": "line", "id": f"rule-{r}", "x": MEASURE_X - 20, "y": y - 9, "points": [[0, 0], [1162, 0]],
                        "strokeColor": "#495057", "strokeWidth": 1, "strokeStyle": "dashed"})
        s.append({"type": "ellipse", "id": f"lamp-{r}", "x": MEASURE_X - 22, "y": y + TILE_H / 2 - 6, "width": 12, "height": 12,
                  "backgroundColor": YELLOW, "fillStyle": "solid", "strokeColor": YELLOW, "strokeWidth": 1, "roughness": 0})
        s.append(holder(f"measure-{r}", MEASURE_X, y, MEASURE_W, TILE_H, plain(row[0]), 22, NUNITO, WHITE))
        for c, (x, w, (pale, strong)) in enumerate(COLUMNS, start=1):
            s.append({"type": "rectangle", "id": f"flap-{r}-{c}", "x": x, "y": y, "width": w, "height": TILE_H, "backgroundColor": FLAP, "fillStyle": "solid",
                      "strokeColor": "#000000", "strokeWidth": 2, "roughness": 0, "roundness": {"type": 3}})
            s.append({"type": "rectangle", "id": f"edge-{r}-{c}", "x": x + 6, "y": y + 8, "width": 6, "height": TILE_H - 16, "backgroundColor": strong,
                      "fillStyle": "solid", "strokeColor": strong, "strokeWidth": 1, "roughness": 0})
            # the hinge shows only as two stubs at the tile's edges, so it never runs through the value
            for side, sx in (("l", x + 2), ("r", x + w - HINGE - 2)):
                s.append({"type": "line", "id": f"split-{r}-{c}-{side}", "x": sx, "y": y + TILE_H / 2, "points": [[0, 0], [HINGE, 0]],
                          "strokeColor": SPLIT, "strokeWidth": 2, "roughness": 0})
            s.append(holder(f"value-{r}-{c}", x + 16, y, w - 22, TILE_H, value_lines(row[c]), 22, COMIC, PALE_YELLOW, "center"))
    return s


if __name__ == "__main__":
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE
    written = render_scene(skeleton(), folder, NAME)
    for path in written: print(path)
    missing = missing_from_scene(HEADER, ROWS, written[0])
    if missing: print("missing from the scene:", missing)
    assert not missing
