"""Draw the t check table of issue #297 as a fairground height sign: a striped post with a 0.01 sign, and one visitor per value.

Each visitor stands as tall as its value on the post's scale. Five groups, one per field type, of three visitors, one per ask.

Run:  /tmp/vgc-pilot/.venv/bin/python build_t-check-height-sign.py [output folder]
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[4] / "code" / "src"))
from excalidraw_table import missing_from_scene, plain, render_scene  # noqa: E402

NAME = "t-check-height-sign"
HEADER = ["Ask", "Species", "Ability and item", "Moves", "Stat Alignment", "Stat Points"]
ROWS = [
    ["6/24", "0.0009", "0.0021", "0.0050", "**0.0078**", "0.0035"],
    ["12/24", "0.0010", "0.0012", "0.0031", "0.0056", "0.0033"],
    ["17/24", "0.0009", "0.0015", "0.0026", "0.0055", "0.0034"],
]

INK = "#1e1e1e"
ASK_COLOURS = [("#d0bfff", "#6741d9"), ("#fab005", "#e67700"), ("#fcc2d7", "#c2255c")]   # pale fill, strong stroke, one pair per ask
POST_RED, POST_WHITE, SIGN_FILL, BAR_RED = "#fa5252", "#fff5f5", "#ffd43b", "#e03131"
SCALE = 52000                       # canvas units per unit of total variation, so the bar of 0.01 sits 520 units above the ground
BAR = 0.01
GROUND = 860
POST_X, POST_W = 20, 56
GROUPS_X, GROUP_W, STEP = 110, 236, 74
BODY_W, HEAD = 36, 26
FONT_CLEAN, FONT_POSTER, FONT_HAND = 6, 7, 5


def holder(id_, x, y, w, h, text, size, family=FONT_CLEAN, colour=INK, align="center", vertical="middle"):
    return {"type": "rectangle", "id": id_, "x": x, "y": y, "width": w, "height": h, "strokeColor": "transparent",
            "backgroundColor": "transparent", "roughness": 0,
            "label": {"text": text, "fontSize": size, "fontFamily": family, "textAlign": align, "verticalAlign": vertical, "strokeColor": colour}}


def post():
    elements, bar_y = [], GROUND - BAR * SCALE
    top = GROUND - 0.0125 * SCALE
    stripe = 0.001 * SCALE
    y, index = GROUND - stripe, 0
    while y > top - 1:
        elements.append({"type": "rectangle", "id": f"stripe-{index}", "x": POST_X, "y": y, "width": POST_W, "height": stripe,
                         "backgroundColor": POST_RED if index % 2 == 0 else POST_WHITE, "fillStyle": "solid", "strokeColor": INK,
                         "strokeWidth": 1, "roughness": 1})
        y, index = y - stripe, index + 1
    elements.append({"type": "ellipse", "id": "post-ball", "x": POST_X + 6, "y": y + stripe - 40, "width": POST_W - 12, "height": 40,
                     "backgroundColor": SIGN_FILL, "fillStyle": "solid", "strokeColor": INK, "strokeWidth": 2, "roughness": 1})
    elements.append({"type": "rectangle", "id": "post-foot", "x": POST_X - 14, "y": GROUND - 6, "width": POST_W + 28, "height": 16,
                     "backgroundColor": "#868e96", "fillStyle": "solid", "strokeColor": INK, "strokeWidth": 1, "roughness": 1})
    elements.append({"type": "line", "id": "bar-line", "x": POST_X + POST_W, "y": bar_y, "points": [[0, 0], [1290 - POST_X - POST_W, 0]],
                     "strokeColor": BAR_RED, "strokeWidth": 3, "strokeStyle": "dashed", "roughness": 0})
    elements.append({"type": "rectangle", "id": "sign", "x": POST_X + POST_W + 14, "y": bar_y - 92, "width": 560, "height": 84,
                     "backgroundColor": SIGN_FILL, "fillStyle": "solid", "strokeColor": INK, "strokeWidth": 3, "roughness": 1,
                     "roundness": {"type": 3}, "angle": -0.02,
                     "label": {"text": "You must be this tall to battle: 0.01", "fontSize": 30, "fontFamily": FONT_POSTER,
                               "textAlign": "center", "verticalAlign": "middle", "strokeColor": INK}})
    elements.append({"type": "arrow", "id": "sign-arrow", "x": POST_X + POST_W + 4, "y": bar_y - 30, "points": [[0, 0], [0, 26]],
                     "strokeColor": INK, "strokeWidth": 2, "endArrowhead": "triangle", "roughness": 0})
    return elements


def visitor(id_, cx, text, fill, stroke, tallest):
    total = float(text) * SCALE
    body_h = total - HEAD
    weight = 4 if tallest else 2
    head_top = GROUND - total
    parts = [
        {"type": "ellipse", "id": f"{id_}-head", "x": cx - HEAD / 2, "y": head_top, "width": HEAD, "height": HEAD,
         "backgroundColor": fill, "fillStyle": "solid", "strokeColor": stroke, "strokeWidth": weight, "roughness": 1},
        {"type": "rectangle", "id": f"{id_}-body", "x": cx - BODY_W / 2, "y": head_top + HEAD, "width": BODY_W, "height": body_h,
         "backgroundColor": fill, "fillStyle": "hachure" if not tallest else "cross-hatch", "strokeColor": stroke,
         "strokeWidth": weight, "roughness": 1, "roundness": {"type": 3}},
    ]
    if body_h > 30:
        arm_y = head_top + HEAD + 8
        for side, sign in (("l", -1), ("r", 1)):
            parts.append({"type": "line", "id": f"{id_}-arm-{side}", "x": cx + sign * BODY_W / 2, "y": arm_y,
                          "points": [[0, 0], [sign * 9, 16]], "strokeColor": stroke, "strokeWidth": weight, "roughness": 1})
    label_y = head_top - 34
    if tallest:
        parts.append({"type": "rectangle", "id": f"{id_}-value", "x": cx - 52, "y": label_y - 12, "width": 104, "height": 38,
                      "backgroundColor": stroke, "fillStyle": "solid", "strokeColor": INK, "strokeWidth": 2, "roughness": 1,
                      "label": {"text": text, "fontSize": 22, "fontFamily": FONT_CLEAN, "textAlign": "center",
                                "verticalAlign": "middle", "strokeColor": "#ffffff"}})
    else:
        parts.append(holder(f"{id_}-value", cx - 37, label_y, 74, 28, text, 18))
    return parts


def skeleton():
    elements = [
        {"type": "text", "id": "title", "x": 0, "y": 0, "text": "Every visitor stands under the 0.01 bar", "fontSize": 40,
         "fontFamily": FONT_POSTER, "strokeColor": INK},
        holder("subtitle", 0, 56, 1280, 30, "Each visitor is as tall as how far the model's choices move when it is told another t.",
               22, align="left"),
        holder("key-title", 760, 120, 60, 36, "Ask", 26, FONT_POSTER, align="left"),
    ]
    for a, row in enumerate(ROWS):
        fill, stroke = ASK_COLOURS[a]
        elements.append({"type": "ellipse", "id": f"key-dot-{a}", "x": 830 + a * 150, "y": 124, "width": 28, "height": 28,
                         "backgroundColor": fill, "fillStyle": "solid", "strokeColor": stroke, "strokeWidth": 2, "roughness": 1})
        elements.append(holder(f"key-ask-{a}", 864 + a * 150, 120, 90, 36, row[0], 22, align="left"))
    elements += post()
    elements.append({"type": "line", "id": "ground", "x": POST_X - 20, "y": GROUND, "points": [[0, 0], [1290, 0]],
                     "strokeColor": INK, "strokeWidth": 2, "roughness": 1})
    for f in range(5):
        gx = GROUPS_X + f * GROUP_W
        elements.append({"type": "rectangle", "id": f"mat-{f}", "x": gx + 4, "y": GROUND + 4, "width": GROUP_W - 8, "height": 92,
                         "backgroundColor": "#e9ecef", "fillStyle": "solid", "strokeColor": "#868e96", "strokeWidth": 1,
                         "roughness": 1, "roundness": {"type": 3}})
        for a, row in enumerate(ROWS):
            cell = row[f + 1]
            text = plain(cell)
            cx = gx + 44 + a * STEP
            fill, stroke = ASK_COLOURS[a]
            elements += visitor(f"v-{f}-{a}", cx, text, fill, stroke, tallest=cell.startswith("**"))
            elements.append(holder(f"ask-{f}-{a}", cx - 37, GROUND + 10, 74, 26, row[0], 18))
        elements.append(holder(f"field-{f}", gx, GROUND + 44, GROUP_W, 44, HEADER[f + 1], 22, FONT_POSTER))
    tall = next(e for e in elements if e["id"] == "v-3-0-value")
    elements.append(holder("tallest-note", tall["x"] - 200, tall["y"] + 4, 170, 30, "bold in the table", 18, FONT_HAND, align="right"))
    elements.append({"type": "arrow", "id": "tallest-arrow", "x": tall["x"] - 24, "y": tall["y"] + 19, "points": [[0, 0], [20, 0]],
                     "strokeColor": INK, "strokeWidth": 2, "endArrowhead": "triangle", "roughness": 0})
    return elements


if __name__ == "__main__":
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE
    written = render_scene(skeleton(), folder, NAME)
    missing = missing_from_scene(HEADER, ROWS, written[0])
    if missing:
        print("missing:", missing)
    assert not missing
    for path in written:
        print(path)
