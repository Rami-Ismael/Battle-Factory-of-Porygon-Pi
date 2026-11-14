"""Draw the error-gap table of issue #297 as six tightrope walkers, one rope per row.

Each rope is a number line for the row's error minus the control's. A blue post stands at zero for the control.
The walker stands at the row's gap and its balance pole spans the 95% interval of the difference.
Each group of three ropes has its own scale, and the scale's ends carry numbers taken from the table.

Run:  /tmp/vgc-pilot/.venv/bin/python build_error-gaps-tightrope.py [output folder]
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[4] / "code" / "src"))
from excalidraw_table import missing_from_scene, plain, render_scene  # noqa: E402

NAME = "error-gaps-tightrope"
HEADER = ["Row", "Guidance", "Error minus the control's", "95% interval of the difference", "Sampling seeds with a lower error", "Verdict"]
ROWS = [
    ["Linear, species-first", "2", "-0.0008", "-0.0022 to +0.0005", "2 of 3", "matches"],
    ["Linear, species-first", "4", "+0.0000", "-0.0009 to +0.0010", "1 of 3", "matches"],
    ["Linear, species-first", "8", "+0.0005", "-0.0002 to +0.0013", "0 of 3", "matches"],
    ["Linear, random order", "2", "+0.0274", "+0.0228 to +0.0319", "0 of 3", "loses"],
    ["Linear, random order", "4", "+0.0245", "+0.0196 to +0.0290", "0 of 3", "loses"],
    ["Linear, random order", "8", "+0.0236", "+0.0182 to +0.0287", "0 of 3", "loses"],
]

INK, GREY, ROPE = "#1e1e1e", "#495057", "#846358"
BLUE_FILL, BLUE = "#a5d8ff", "#1971c2"
COLOURS = {"Linear, species-first": ("#96f2d7", "#e6fcf5", "#099268"), "Linear, random order": ("#ffd8a8", "#fff4e6", "#e8590c")}
HEADS = {name: strong for name, (_, _, strong) in COLOURS.items()}   # strong fills, which the dark export turns lighter
STRIP_X0, STRIP_X1, STRIP_Y = 80, 1190, 290                           # the shared-scale strip under the note
NUNITO, LILITA = 6, 7

ROPE_X0, ROPE_X1, MARGIN = 40, 750, 40                                # rope ends, and the space kept inside them for the scale
CARD_X, CARD_W, CARD_H, BAND_H = 790, 470, 190, 215
TOP = 610                                                            # where the first group banner starts
BANNER_H = 60


def text(id_, x, y, words, size=20, family=NUNITO, colour=INK):
    return dict(type="text", id=id_, x=x, y=y, text=words, fontSize=size, fontFamily=family, strokeColor=colour)


def holder(id_, x, y, w, h, words, size=20, family=NUNITO, colour=INK, align="left"):
    return dict(type="rectangle", id=id_, x=x, y=y, width=w, height=h, strokeColor="transparent", backgroundColor="transparent",
                roughness=0, label=dict(text=words, fontSize=size, fontFamily=family, textAlign=align, verticalAlign="top", strokeColor=colour))


def line(id_, x, y, points, colour=INK, width=2, style="solid", roughness=1):
    return dict(type="line", id=id_, x=x, y=y, points=points, strokeColor=colour, strokeWidth=width, strokeStyle=style, roughness=roughness)


def dot(id_, cx, cy, d, fill, stroke):
    return dict(type="ellipse", id=id_, x=cx - d / 2, y=cy - d / 2, width=d, height=d, backgroundColor=fill, fillStyle="solid",
                strokeColor=stroke, strokeWidth=1, roughness=1)


def icons(p, x, y, strong, fill):
    """The six slot marks, one per column, at the card whose top-left corner is (x, y). The key card uses the same marks."""
    return [
        dict(type="rectangle", id=f"{p}-i-row", x=x + 16, y=y + 22, width=20, height=14, backgroundColor=fill, fillStyle="solid", strokeColor=strong, strokeWidth=2, roughness=1),
        dot(f"{p}-i-guidance", x + 338, y + 30, 18, fill, strong),
        dict(type="diamond", id=f"{p}-i-gap", x=x + 14, y=y + 70, width=24, height=24, backgroundColor=strong, fillStyle="solid", strokeColor=strong, strokeWidth=1, roughness=1),
        line(f"{p}-i-interval", x + 14, y + 127, [[0, 0], [24, 0]], INK, 3),
        dot(f"{p}-i-interval-l", x + 14, y + 127, 7, INK, INK),
        dot(f"{p}-i-interval-r", x + 38, y + 127, 7, INK, INK),
        *[dot(f"{p}-i-seed-{k}", x + 18 + 8 * k, y + 163, 6, strong, strong) for k in range(3)],
    ]


def card(p, y, cells, strong, fill, tint, key=False):
    """A circus programme card: every cell of one row, each in a fixed slot."""
    row, guidance, gap, interval, seeds, verdict = cells
    x = CARD_X
    big = (20, NUNITO) if key else (30, LILITA)
    parts = [dict(type="rectangle", id=f"{p}-card", x=x, y=y, width=CARD_W, height=CARD_H, backgroundColor=tint, fillStyle="solid",
                  strokeColor=strong, strokeWidth=2, roughness=1, roundness={"type": 3}),
             *icons(p, x, y, strong, fill),
             text(f"{p}-row", x + 48, y + 16, row, 22 if not key else 20, NUNITO),
             text(f"{p}-guidance", x + 356, y + (12 if not key else 16), guidance, 26 if not key else 20, LILITA if not key else NUNITO),
             text(f"{p}-gap", x + 48, y + (62 if not key else 70), gap, *big),
             text(f"{p}-interval", x + 48, y + 114, interval),
             text(f"{p}-seeds", x + 48, y + 150, seeds),
             dict(type="rectangle", id=f"{p}-stamp", x=x + 300, y=y + 56, width=145, height=44, angle=-0.08, backgroundColor="transparent",
                  strokeColor=strong, strokeWidth=3, roughness=1,
                  label=dict(text=verdict.upper() if not key else verdict, fontSize=26 if not key else 20, fontFamily=LILITA if not key else NUNITO,
                             textAlign="center", verticalAlign="middle", strokeColor=strong))]
    return parts


def walker(p, rope_y, at, lo, hi, fill, strong):
    """A stick figure standing on the rope at x = at, holding a pole from x = lo to x = hi."""
    hip, neck, pole_y = rope_y - 30, rope_y - 62, rope_y - 50
    return [
        line(f"{p}-leg-l", at, hip, [[0, 0], [-11, 30]], strong, 3),
        line(f"{p}-leg-r", at, hip, [[0, 0], [11, 30]], strong, 3),
        line(f"{p}-body", at, hip, [[0, 0], [0, neck - hip]], strong, 3),
        line(f"{p}-arm-l", at, neck + 6, [[0, 0], [-24, pole_y - neck - 6]], strong, 3),
        line(f"{p}-arm-r", at, neck + 6, [[0, 0], [24, pole_y - neck - 6]], strong, 3),
        line(f"{p}-pole", lo, pole_y, [[0, 0], [hi - lo, 0]], INK, 4, roughness=0),
        dot(f"{p}-weight-l", lo, pole_y, 14, INK, INK),
        dot(f"{p}-weight-r", hi, pole_y, 14, INK, INK),
        dot(f"{p}-head", at, neck - 14, 28, fill, INK),
    ]


def strip():
    """All six rows on one shared scale, from the smallest interval end in the table to the largest, with the control's zero."""
    rows = [[plain(c) for c in row] for row in ROWS]
    ends = [v for row in rows for v in row[3].split(" to ")]
    lo_label, hi_label = min(ends, key=float), max(ends, key=float)
    lo_v, hi_v = float(lo_label), float(hi_label)
    to_x = lambda v: STRIP_X0 + (v - lo_v) / (hi_v - lo_v) * (STRIP_X1 - STRIP_X0)
    y0, step, axis_y = STRIP_Y + 92, 22, STRIP_Y + 92 + 5 * 22 + 24
    zero = to_x(0.0)
    parts = [
        dict(type="rectangle", id="strip-frame", x=ROPE_X0, y=STRIP_Y - 10, width=1220, height=axis_y + 60 - STRIP_Y, backgroundColor="transparent",
             strokeColor=GREY, strokeWidth=1, strokeStyle="dashed", roughness=1, roundness={"type": 3}),
        text("strip-title", ROPE_X0 + 20, STRIP_Y, "All six on one scale", 26, LILITA, GREY),
        line("strip-zero", zero, y0 - 16, [[0, 0], [0, axis_y - y0 + 16]], BLUE, 2, "dashed", roughness=0),
        text("strip-zero-label", zero + 8, y0 - 46, "+0.0000 control", 18, NUNITO, BLUE),
        line("strip-axis", STRIP_X0, axis_y, [[0, 0], [STRIP_X1 - STRIP_X0, 0]], GREY, 2, roughness=0),
        line("strip-tick-l", STRIP_X0, axis_y, [[0, 0], [0, 10]], GREY, 2, roughness=0),
        line("strip-tick-r", STRIP_X1, axis_y, [[0, 0], [0, 10]], GREY, 2, roughness=0),
        text("strip-end-l", STRIP_X0 - 30, axis_y + 14, lo_label, 18, NUNITO, GREY),
        text("strip-end-r", STRIP_X1 - 32, axis_y + 14, hi_label, 18, NUNITO, GREY),
    ]
    for i, cells in enumerate(rows):
        strong = COLOURS[cells[0]][2]
        a, b = (to_x(float(v)) for v in cells[3].split(" to "))
        y = y0 + i * step
        parts += [line(f"strip-bar-{i}", a, y, [[0, 0], [b - a, 0]], strong, 4, roughness=0),
                  dict(type="ellipse", id=f"strip-dot-{i}", x=to_x(float(cells[2])) - 7, y=y - 7, width=14, height=14,
                       backgroundColor=HEADS[cells[0]], fillStyle="solid", strokeColor=INK, strokeWidth=2, roughness=0)]
    return parts


def skeleton():
    parts = [
        holder("title", ROPE_X0, 0, 720, 90, "Species-first walkers hold their poles across the control's post. Random order walks far to its right.", 30, LILITA),
        holder("note", ROPE_X0, 110, 720, 160,
               "Each walker stands at its row's error minus the control's. Its pole spans the 95% interval of the difference. "
               "The blue post marks the control. Each group of three ropes has its own scale, and the numbers under a rope's ends are that scale's ends.",
               20, NUNITO, GREY),
        text("key-title", CARD_X, 0, "Key to every card", 26, LILITA, GREY),
        *card("key", 50, HEADER, GREY, "#dee2e6", "#f8f9fa", key=True),
        *strip(),
    ]
    top = TOP
    for g, name in enumerate(COLOURS):
        fill, tint, strong = COLOURS[name]
        rows = [[plain(c) for c in row] for row in ROWS if row[0] == name]
        numbers = [float(row[2]) for row in rows] + [float(v) for row in rows for v in row[3].split(" to ")] + [0.0]
        lo_v, hi_v = min(numbers), max(numbers)
        lo_label = next(v for row in rows for v in [row[2], *row[3].split(" to ")] if float(v) == lo_v) if lo_v < 0 else "+0.0000"
        hi_label = next(v for row in rows for v in [row[2], *row[3].split(" to ")] if float(v) == hi_v)
        to_x = lambda v, lo_v=lo_v, hi_v=hi_v: ROPE_X0 + MARGIN + (v - lo_v) / (hi_v - lo_v) * (ROPE_X1 - ROPE_X0 - 2 * MARGIN)
        parts += [dict(type="rectangle", id=f"banner-{g}", x=ROPE_X0, y=top, width=ROPE_X1 - ROPE_X0, height=44, backgroundColor=fill,
                       fillStyle="solid", strokeColor=strong, strokeWidth=2, roughness=1,
                       label=dict(text=name, fontSize=26, fontFamily=LILITA, textAlign="center", verticalAlign="middle", strokeColor=strong))]
        top += BANNER_H
        for r, cells in enumerate(rows):
            p, rope_y = f"g{g}r{r}", top + 150
            zero = to_x(0.0)
            lo, hi = (to_x(float(v)) for v in cells[3].split(" to "))
            parts += [
                dict(type="rectangle", id=f"{p}-anchor-l", x=ROPE_X0 - 5, y=rope_y - 26, width=10, height=40, backgroundColor=GREY, fillStyle="solid", strokeColor=GREY, roughness=1),
                dict(type="rectangle", id=f"{p}-anchor-r", x=ROPE_X1 - 5, y=rope_y - 26, width=10, height=40, backgroundColor=GREY, fillStyle="solid", strokeColor=GREY, roughness=1),
                line(f"{p}-rope", ROPE_X0, rope_y, [[0, 0], [(ROPE_X1 - ROPE_X0) / 2, 6], [ROPE_X1 - ROPE_X0, 0]], ROPE, 3),
                dict(type="rectangle", id=f"{p}-post", x=zero - 7, y=rope_y - 110, width=14, height=116, backgroundColor=BLUE_FILL, fillStyle="cross-hatch",
                     strokeColor=BLUE, strokeWidth=2, roughness=1),
                text(f"{p}-post-label", zero + 12, rope_y - 112, "control", 18, NUNITO, BLUE),
                line(f"{p}-tick-l", to_x(lo_v), rope_y + 8, [[0, 0], [0, 12]], GREY, 2, roughness=0),
                line(f"{p}-tick-r", to_x(hi_v), rope_y + 8, [[0, 0], [0, 12]], GREY, 2, roughness=0),
                text(f"{p}-end-l", to_x(lo_v) - 30, rope_y + 30, lo_label, 18, NUNITO, GREY),
                text(f"{p}-end-r", to_x(hi_v) - 30, rope_y + 30, hi_label, 18, NUNITO, GREY),
                *walker(p, rope_y, to_x(float(cells[2])), lo, hi, HEADS[name], strong),
                *card(p, top + 10, cells, strong, fill, tint),
            ]
            top += BAND_H
        top += 30
    return parts


if __name__ == "__main__":
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE
    render_scene(skeleton(), folder, NAME)
    missing = missing_from_scene(HEADER, ROWS, folder / f"{NAME}.excalidraw")
    if missing: print("missing:", missing)
    assert not missing
    print("drawn", folder / NAME)
