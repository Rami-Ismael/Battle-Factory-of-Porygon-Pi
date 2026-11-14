"""Draw the line-ups table of issue #297 as a shopping-centre atrium: one floor per row, an up escalator for the
grid total and a down escalator for the mean inside one ask, with an information sign at each guidance landing.

Run:  /tmp/vgc-pilot/.venv/bin/python build_line-ups-escalators.py [output folder]
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[4] / "code" / "src"))
from excalidraw_table import missing_from_scene, plain, render_scene  # noqa: E402

NAME = "line-ups-escalators"
HEADER = ["Row", "Guidance", "Distinct line-ups over the grid's 4,416 teams", "Mean distinct line-ups among one ask's 192 teams",
          "Mean at asks 15/24 to 17/24", "Asks that share one line-up, mean"]
ROWS = [
    ["Per-field, uncorrected t, the control", "2", "1,513", "167.0", "172", "2.54"],
    ["Per-field, uncorrected t, the control", "4", "1,918", "156.3", "159", "1.87"],
    ["Per-field, uncorrected t, the control", "8", "2,126", "134.9", "112", "1.46"],
    ["Linear, species-first", "2", "1,578", "162.9", "161", "2.37"],
    ["Linear, species-first", "4", "1,971", "155.3", "145", "1.81"],
    ["Linear, species-first", "8", "2,060", "132.4", "96", "1.48"],
    ["Linear, random order", "2", "2,646", "191.7", "192", "1.67"],
    ["Linear, random order", "4", "3,336", "191.6", "192", "1.32"],
    ["Linear, random order", "8", "3,820", "191.2", "192", "1.15"],
]

INK, GREY, BELT, SIGN, SIGN_EDGE = "#1e1e1e", "#495057", "#adb5bd", "#fff9db", "#f08c00"
COLOURS = {"control": ("#a5d8ff", "#1971c2"), "species-first": ("#96f2d7", "#099268"), "random": ("#ffd8a8", "#e8590c")}
WIDTH = 1300
LANDINGS = [420, 780, 1128]                                         # x centre of the guidance 2, 4 and 8 landings
STEP_W, STEP_H = 150, 50
TOP = 150                                                            # where the first floor starts, below the title
FLOOR_H = 640
UP_HIGH, UP_LOW = 150, 285                                           # step centre range of the up escalator, from the floor top
DOWN_HIGH, DOWN_LOW = 370, 470                                       # step centre range of the down escalator
SLAB = 515
UP_SCALE, DOWN_SCALE = (1400, 3900), (130, 195)                      # one value scale per escalator, shared by every floor


def number(cell):
    return float(cell.replace(",", ""))


def colours_of(row_name):
    return next(pair for word, pair in COLOURS.items() if word in row_name)


def holder(id_, x, y, w, h, text, size, family, colour=INK, align="center", vertical="middle"):
    return dict(type="rectangle", id=id_, x=x, y=y, width=w, height=h, strokeColor="transparent", backgroundColor="transparent", roughness=0,
                label=dict(text=text, fontSize=size, fontFamily=family, textAlign=align, verticalAlign=vertical, strokeColor=colour))


def centre_y(value, scale, high, low):
    lo, hi = scale
    return high + (low - high) * (hi - value) / (hi - lo)


def escalator(parts, id_, ys, strong, fill, texts, white_text):
    """A belt through the three step centres with a flat run at each end, a handrail above it, and the three value steps."""
    xs = LANDINGS
    belt = [(xs[0] - 140, ys[0]), *zip(xs, ys), (xs[-1] + 140, ys[-1])]
    x0, y0 = belt[0]
    points = [[x - x0, y - y0] for x, y in belt]
    parts.append(dict(type="line", id=f"{id_}-belt", x=x0, y=y0, points=points, strokeColor=BELT, strokeWidth=20, roughness=0))
    parts.append(dict(type="line", id=f"{id_}-teeth", x=x0, y=y0 + 6, points=points, strokeColor=GREY, strokeWidth=3, strokeStyle="dashed", roughness=0))
    parts.append(dict(type="arrow", id=f"{id_}-rail", x=x0, y=y0 - 44, points=points, strokeColor=strong, strokeWidth=4, roughness=1,
                      endArrowhead="triangle"))
    for g, (x, y, text) in enumerate(zip(xs, ys, texts)):
        parts.append(dict(type="rectangle", id=f"{id_}-step-{g}", x=x - STEP_W / 2, y=y - STEP_H / 2, width=STEP_W, height=STEP_H,
                          backgroundColor=strong if white_text else fill, fillStyle="solid", strokeColor=strong if white_text else strong,
                          strokeWidth=2, roughness=1, roundness=dict(type=3),
                          label=dict(text=text, fontSize=28, fontFamily=7, strokeColor="#ffffff" if white_text else INK)))


def floor(parts, f, name, rows):
    pale, strong = colours_of(name)
    y0 = TOP + f * FLOOR_H
    parts.append(dict(type="rectangle", id=f"floor-{f}", x=0, y=y0, width=WIDTH, height=SLAB + 20, backgroundColor=pale, fillStyle="solid",
                      strokeColor=strong, strokeWidth=2, roughness=1, opacity=45))
    parts.append(dict(type="rectangle", id=f"fascia-{f}", x=0, y=y0, width=WIDTH, height=56, backgroundColor=strong, fillStyle="solid",
                      strokeColor=strong, strokeWidth=2, roughness=1,
                      label=dict(text=f"Floor {f + 1}:  {name}", fontSize=30, fontFamily=7, strokeColor="#ffffff", textAlign="left")))
    for g, x in enumerate(LANDINGS):
        parts.append(dict(type="line", id=f"pillar-{f}-{g}", x=x, y=y0 + 100, points=[[0, 0], [0, SLAB - 100]], strokeColor=strong,
                          strokeWidth=1, strokeStyle="dotted", roughness=0))
        parts.append(holder(f"landing-{f}-{g}", x - 110, y0 + 64, 220, 34, f"Guidance {rows[g][1]}", 24, 5, strong))
    # the directory boards at the left say which escalator carries which column
    parts.append(dict(type="rectangle", id=f"up-board-{f}", x=16, y=y0 + 120, width=230, height=150, backgroundColor="#ffffff",
                      fillStyle="solid", strokeColor=strong, strokeWidth=2, roughness=1, roundness=dict(type=3)))
    parts.append(holder(f"up-word-{f}", 30, y0 + 128, 202, 36, "UP", 30, 7, strong, "left"))
    parts.append(holder(f"up-text-{f}", 30, y0 + 166, 202, 96, HEADER[2], 18, 6, INK, "left", "top"))
    parts.append(dict(type="rectangle", id=f"down-board-{f}", x=16, y=y0 + 330, width=230, height=160, backgroundColor="#ffffff",
                      fillStyle="solid", strokeColor=strong, strokeWidth=2, roughness=1, roundness=dict(type=3)))
    parts.append(holder(f"down-word-{f}", 30, y0 + 338, 202, 36, "DOWN", 30, 7, strong, "left"))
    parts.append(holder(f"down-text-{f}", 30, y0 + 376, 202, 104, HEADER[3], 18, 6, INK, "left", "top"))
    up = [y0 + centre_y(number(r[2]), UP_SCALE, UP_HIGH, UP_LOW) for r in rows]
    down = [y0 + centre_y(number(r[3]), DOWN_SCALE, DOWN_HIGH, DOWN_LOW) for r in rows]
    escalator(parts, f"up-{f}", up, strong, pale, [r[2] for r in rows], True)
    escalator(parts, f"down-{f}", down, strong, pale, [r[3] for r in rows], False)
    parts.append(dict(type="rectangle", id=f"slab-{f}", x=0, y=y0 + SLAB, width=WIDTH, height=20, backgroundColor=GREY, fillStyle="cross-hatch",
                      strokeColor=GREY, strokeWidth=2, roughness=1))
    for g, (x, r) in enumerate(zip(LANDINGS, rows)):
        top = y0 + SLAB + 34
        for side in (-150, 150):
            parts.append(dict(type="line", id=f"hang-{f}-{g}-{side}", x=x + side, y=y0 + SLAB + 20, points=[[0, 0], [0, 14]],
                              strokeColor=GREY, strokeWidth=2, roughness=0))
        parts.append(dict(type="rectangle", id=f"sign-{f}-{g}", x=x - 170, y=top, width=340, height=76, backgroundColor=SIGN, fillStyle="solid",
                          strokeColor=SIGN_EDGE, strokeWidth=2, roughness=1,
                          label=dict(text=f"{HEADER[4]}: {r[4]}\n{HEADER[5]}: {r[5]}", fontSize=18, fontFamily=6, textAlign="left",
                                     strokeColor=INK)))


def legend(parts, y0):
    parts.append(dict(type="rectangle", id="legend", x=0, y=y0, width=WIDTH, height=300, backgroundColor="#f1f3f5", fillStyle="solid",
                      strokeColor=GREY, strokeWidth=2, roughness=1, roundness=dict(type=3)))
    parts.append(dict(type="text", id="legend-title", x=24, y=y0 + 16, text="Atrium directory: the six columns of the table", fontSize=28,
                      fontFamily=7, strokeColor=INK))
    keys = [
        ("Floor banner", "#1971c2", True),
        ("Landing name", None, False),
        ("UP escalator step", GREY, True),
        ("DOWN escalator step", "#ffd8a8", False),
        ("Sign, first line", SIGN, False),
        ("Sign, second line", SIGN, False),
    ]
    for i, ((key, fill, white), header) in enumerate(zip(keys, HEADER)):
        col, row = i % 2, i // 2
        x, y = 24 + col * 640, y0 + 70 + row * 74
        if fill:
            parts.append(dict(type="rectangle", id=f"key-{i}", x=x, y=y, width=210, height=50, backgroundColor=fill, fillStyle="solid",
                              strokeColor=SIGN_EDGE if fill == SIGN else GREY, strokeWidth=2, roughness=1, roundness=dict(type=3),
                              label=dict(text=key, fontSize=20, fontFamily=6, strokeColor="#ffffff" if white else INK)))
        else:
            parts.append(holder(f"key-{i}", x, y, 210, 50, key, 24, 5, GREY))
        parts.append(holder(f"key-head-{i}", x + 226, y, 380, 50, plain(header), 20, 6, INK, "left"))


def skeleton():
    parts = [
        dict(type="text", id="title", x=0, y=0, text="Guidance sends the grid total up and the count inside one ask down", fontSize=34,
             fontFamily=7, strokeColor=INK),
        dict(type="text", id="subtitle", x=0, y=52,
             text="One floor for each row. Landings left to right are guidance 2, 4 and 8.\n"
                  "Step heights follow the values, on one scale for every UP escalator and one for every DOWN escalator.",
             fontSize=20, fontFamily=6, strokeColor=GREY),
    ]
    for f in range(3):
        floor(parts, f, ROWS[3 * f][0], ROWS[3 * f:3 * f + 3])
    legend(parts, TOP + 3 * FLOOR_H + 10)
    return parts


if __name__ == "__main__":
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE
    written = render_scene(skeleton(), folder, NAME)
    for path in written: print(path)
    missing = missing_from_scene(HEADER, ROWS, written[0])
    if missing: print("missing:", missing)
    assert not missing
