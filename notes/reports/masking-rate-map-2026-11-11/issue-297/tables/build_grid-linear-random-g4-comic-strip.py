"""Redraw the guidance 4 grid table for Linear, random order as a comic page: one panel per measure, two speakers per panel."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[4] / "code" / "src"))
from excalidraw_table import missing_from_scene, plain, render_scene  # noqa: E402

NAME = "grid-linear-random-g4-comic-strip"
HEADER = ["Measure", "Linear, random order, guidance 4", "The control at guidance 4"]
ROWS = [
    ["Ask-versus-got error, asks 1/24 to 17/24, with its 95% interval", "0.0293 (0.0220 to 0.0377)", "0.0049 (0.0022 to 0.0091)"],
    ["The same error for sampling seeds 1, 2 and 3", "0.0270, 0.0303, 0.0311", "0.0057, 0.0047, 0.0045"],
    ["Error on asks 18/24 to 23/24", "0.2529", "0.1079"],
    ["Win rate at asks 15/24 to 17/24", "0.364 (0.328 to 0.401)", "0.532 (0.492 to 0.574)"],
    ["Win rate over all 23 asks", "0.266 (0.242 to 0.290)", "0.375 (0.349 to 0.403)"],
    ["First-try legality", "90.5%", "100.0%"],
    ["Distinct line-ups", "3,336", "1,918"],
    ["Teams with a line-up absent from the training teams", "88.3%", "24.4%"],
    ["Teams with a line-up absent from earlier search runs and tournament records", "97.6%", "84.4%"],
    ["Effective species and top-species share, in 5-team subsets", "19.5 and 0.114", "14.2 and 0.143"],
    ["Teams and battles", "4,416 and 216,384", "4,416 and 216,384"],
]

INK, PAGE, CAPTION = "#1e1e1e", "#fff9db", "#ffec99"
ORANGE = ("#ffd8a8", "#e8590c")
BLUE = ("#a5d8ff", "#1971c2")
W, H, GAP, TOP = 410, 490, 22, 90
HAND, MONO, POSTER = 5, 8, 7


def character(key, cx, feet, colours):
    """A speaker drawn from ellipses and lines: head, two eyes, a body, two arms and two legs."""
    pale, strong = colours
    head_top = feet - 112
    return [
        {"type": "line", "id": f"{key}-leg-l", "x": cx - 8, "y": feet - 28, "points": [[0, 0], [-10, 28]], "strokeColor": strong, "strokeWidth": 3},
        {"type": "line", "id": f"{key}-leg-r", "x": cx + 8, "y": feet - 28, "points": [[0, 0], [10, 28]], "strokeColor": strong, "strokeWidth": 3},
        {"type": "line", "id": f"{key}-arm-l", "x": cx - 20, "y": feet - 64, "points": [[0, 0], [-18, 14]], "strokeColor": strong, "strokeWidth": 3},
        {"type": "line", "id": f"{key}-arm-r", "x": cx + 20, "y": feet - 64, "points": [[0, 0], [18, -16]], "strokeColor": strong, "strokeWidth": 3},
        {"type": "ellipse", "id": f"{key}-body", "x": cx - 24, "y": feet - 78, "width": 48, "height": 54,
         "backgroundColor": pale, "fillStyle": "hachure", "strokeColor": strong, "strokeWidth": 2},
        {"type": "ellipse", "id": f"{key}-head", "x": cx - 21, "y": head_top, "width": 42, "height": 40,
         "backgroundColor": strong, "fillStyle": "solid", "strokeColor": strong, "strokeWidth": 2},
        {"type": "ellipse", "id": f"{key}-eye-l", "x": cx - 11, "y": head_top + 13, "width": 8, "height": 10,
         "backgroundColor": "#ffffff", "fillStyle": "solid", "strokeColor": INK, "strokeWidth": 1, "roughness": 0},
        {"type": "ellipse", "id": f"{key}-eye-r", "x": cx + 4, "y": head_top + 13, "width": 8, "height": 10,
         "backgroundColor": "#ffffff", "fillStyle": "solid", "strokeColor": INK, "strokeWidth": 1, "roughness": 0},
    ]


def bubble(key, x, y, text, colours, tail_from, tail_to):
    """A speech bubble that holds one value, with a tail that points at its speaker."""
    pale, strong = colours
    return [
        {"type": "arrow", "id": f"{key}-tail", "x": tail_from[0], "y": tail_from[1],
         "points": [[0, 0], [tail_to[0] - tail_from[0], tail_to[1] - tail_from[1]]],
         "strokeColor": strong, "strokeWidth": 3, "endArrowhead": "triangle", "startArrowhead": None},
        {"type": "ellipse", "id": f"{key}-bubble", "x": x, "y": y, "width": 270, "height": 104,
         "backgroundColor": "#ffffff", "fillStyle": "solid", "strokeColor": strong, "strokeWidth": 3,
         "label": {"text": text, "fontSize": 20, "fontFamily": MONO, "textAlign": "center", "verticalAlign": "middle", "strokeColor": INK}},
    ]


def panel(key, x0, y0, caption, left, right):
    """One comic panel: the caption box, two speech bubbles and the two speakers."""
    feet = y0 + H - 14
    parts = [
        {"type": "rectangle", "id": f"{key}-frame", "x": x0, "y": y0, "width": W, "height": H,
         "backgroundColor": PAGE, "fillStyle": "solid", "strokeColor": INK, "strokeWidth": 4},
        {"type": "rectangle", "id": f"{key}-caption", "x": x0 + 10, "y": y0 + 10, "width": W - 20, "height": 84,
         "backgroundColor": CAPTION, "fillStyle": "solid", "strokeColor": INK, "strokeWidth": 2, "roughness": 1,
         "label": {"text": caption, "fontSize": 20, "fontFamily": HAND, "textAlign": "left", "verticalAlign": "middle", "strokeColor": INK}},
    ]
    if left is not None:
        parts += bubble(f"{key}-o", x0 + 12, y0 + 108, left, ORANGE, (x0 + 80, y0 + 206), (x0 + 60, feet - 116))
        parts += bubble(f"{key}-b", x0 + 128, y0 + 222, right, BLUE, (x0 + 300, y0 + 322), (x0 + 344, feet - 116))
    parts += character(f"{key}-o-who", x0 + 58, feet, ORANGE)
    parts += character(f"{key}-b-who", x0 + 350, feet, BLUE)
    return parts


def legend(x0, y0):
    """The cast panel: it names the two speakers with the column headers."""
    feet = y0 + H - 14
    parts = panel("legend", x0, y0, plain(HEADER[0]), None, None)
    parts[1]["label"]["text"] = "The cast. Each caption box names the " + plain(HEADER[0]).lower() + "."
    for key, (cx, colours, text, ty) in {"legend-o": (58, ORANGE, plain(HEADER[1]), 118), "legend-b": (350, BLUE, plain(HEADER[2]), 228)}.items():
        pale, strong = colours
        tx = x0 + 12 if key == "legend-o" else x0 + 128
        parts += [
            {"type": "arrow", "id": f"{key}-tail", "x": tx + (68 if key == "legend-o" else 202), "y": y0 + ty + 84,
             "points": [[0, 0], [x0 + cx - tx - (68 if key == "legend-o" else 202), feet - 116 - y0 - ty - 84]],
             "strokeColor": strong, "strokeWidth": 3, "endArrowhead": "triangle", "startArrowhead": None},
            {"type": "rectangle", "id": f"{key}-tag", "x": tx, "y": y0 + ty, "width": 270, "height": 84,
             "backgroundColor": pale, "fillStyle": "solid", "strokeColor": strong, "strokeWidth": 3, "roundness": {"type": 3},
             "label": {"text": text, "fontSize": 20, "fontFamily": HAND, "textAlign": "center", "verticalAlign": "middle", "strokeColor": INK}},
        ]
    return parts


def skeleton():
    parts = [{"type": "text", "id": "title", "x": 0, "y": 0, "fontSize": 36, "fontFamily": POSTER, "strokeColor": INK,
              "text": "Linear, random order at guidance 4 loses to the control"},
             {"type": "text", "id": "subtitle", "x": 2, "y": 48, "fontSize": 22, "fontFamily": HAND, "strokeColor": INK,
              "text": "Eleven measures, one panel each. Each speaker says its value."}]
    slots = [("legend", None), *[(f"p{i}", row) for i, row in enumerate(ROWS)]]
    for index, (key, row) in enumerate(slots):
        x0, y0 = (index % 3) * (W + GAP), TOP + (index // 3) * (H + GAP)
        parts += legend(x0, y0) if row is None else panel(key, x0, y0, *(plain(cell) for cell in row))
    return parts


if __name__ == "__main__":
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE
    render_scene(skeleton(), folder, NAME)
    missing = missing_from_scene(HEADER, ROWS, folder / f"{NAME}.excalidraw")
    if missing: print("missing:", missing)
    assert not missing
    print("written to", folder)
