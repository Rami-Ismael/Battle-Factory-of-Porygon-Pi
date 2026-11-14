"""Draw the guidance 8 grid table for Linear, random order as a stadium scoreboard.

Run:  python build_grid-linear-random-g8-scoreboard.py [output folder]
"""
import math
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[4] / "code" / "src"))
from excalidraw_table import missing_from_scene, plain, render_scene  # noqa: E402

NAME = "grid-linear-random-g8-scoreboard"
TITLE = "Grid finished: Linear, random order, guidance 8"
HEADER = ["Measure", "Linear, random order, guidance 8", "The control at guidance 8"]
ROWS = [
    ["Ask-versus-got error, asks 1/24 to 17/24, with its 95% interval", "0.0269 (0.0198 to 0.0352)", "0.0033 (0.0014 to 0.0068)"],
    ["The same error for sampling seeds 1, 2 and 3", "0.0261, 0.0270, 0.0279", "0.0034, 0.0036, 0.0032"],
    ["Error on asks 18/24 to 23/24", "0.2492", "0.0909"],
    ["Win rate at asks 15/24 to 17/24", "0.368 (0.333 to 0.405)", "0.556 (0.518 to 0.594)"],
    ["Win rate over all 23 asks", "0.272 (0.248 to 0.296)", "0.390 (0.364 to 0.418)"],
    ["First-try legality", "89.2%", "100.0%"],
    ["Distinct line-ups", "3,820", "2,126"],
    ["Teams with a line-up absent from the training teams", "90.2%", "31.9%"],
    ["Teams with a line-up absent from earlier search runs and tournament records", "97.6%", "86.2%"],
    ["Effective species and top-species share, in 5-team subsets", "19.0 and 0.118", "13.2 and 0.147"],
    ["Teams and battles", "4,416 and 216,384", "4,416 and 216,384"],
]

NUNITO, POSTER, MONO = 6, 7, 8
BOARD, BOARD_EDGE, DIGIT_BOX = "#212529", "#000000", "#000000"
ORANGE_FILL, ORANGE_STRONG, ORANGE_DIGIT = "#ffd8a8", "#e8590c", "#ffc078"
BLUE_FILL, BLUE_STRONG, BLUE_DIGIT = "#a5d8ff", "#1971c2", "#74c0fc"
WHITE, BULB, STEEL = "#f8f9fa", "#ffe066", "#495057"

W = 1260                                   # board width
M = 26                                     # board inner margin
BOX = 370                                  # digit box width
GAP = 18
LEFT_X, MID_X = M, M + BOX + GAP
MID_W = W - 2 * (M + BOX + GAP)
RIGHT_X = W - M - BOX
MEASURE_SIZE = 20


def holder(id_, x, y, width, height, text, size, family, colour, align="center", vertical="middle"):
    return dict(type="rectangle", id=id_, x=x, y=y, width=width, height=height, strokeColor="transparent",
                backgroundColor="transparent", roughness=0,
                label=dict(text=text, fontSize=size, fontFamily=family, textAlign=align, verticalAlign=vertical, strokeColor=colour))


def split_value(cell):
    """A value and its interval as two scoreboard lines, or the whole cell as one."""
    found = re.fullmatch(r"(\S+) (\(.+\))", cell)
    return (found.group(1), found.group(2)) if found else (cell, None)


def digits(prefix, x, y, height, cell, colour):
    value, interval = split_value(plain(cell))
    big = 34 if len(value) <= 7 else 26 if len(value) <= 17 else 23
    items = [dict(type="rectangle", id=f"{prefix}-box", x=x, y=y, width=BOX, height=height, backgroundColor=DIGIT_BOX,
                  fillStyle="solid", strokeColor=colour, strokeWidth=2, roughness=0, roundness=dict(type=3))]
    if interval:
        items.append(holder(f"{prefix}-value", x + 10, y + 6, BOX - 20, height * 0.58 - 6, value, big, MONO, colour, vertical="bottom"))
        items.append(holder(f"{prefix}-interval", x + 10, y + height * 0.58, BOX - 20, height * 0.42 - 6, interval, 20, MONO, colour, vertical="top"))
    else:
        items.append(holder(f"{prefix}-value", x + 10, y + 4, BOX - 20, height - 8, value, big, MONO, colour))
    return items


def build():
    scene = []
    # floodlights: a gantry with two banks of lamps, on a mast above each leg
    top_board = 190
    for bank, cx in enumerate((210, W - 210)):
        scene.append(dict(type="line", id=f"mast-{bank}", x=cx, y=110, points=[[0, 0], [0, top_board - 110]], strokeColor=STEEL, strokeWidth=6, roughness=0))
        scene.append(dict(type="rectangle", id=f"bank-{bank}", x=cx - 170, y=10, width=340, height=104, backgroundColor=STEEL,
                          fillStyle="solid", strokeColor=BOARD_EDGE, strokeWidth=2, roughness=1, roundness=dict(type=3)))
        for row in range(2):
            for col in range(5):
                scene.append(dict(type="ellipse", id=f"lamp-{bank}-{row}-{col}", x=cx - 156 + col * 64, y=18 + row * 46, width=56, height=40,
                                  backgroundColor=BULB, fillStyle="solid", strokeColor="#f08c00", strokeWidth=2, roughness=1))
    for side, dx in (("l", -1), ("r", 1)):
        for bank, cx in enumerate((210, W - 210)):
            scene.append(dict(type="line", id=f"beam-{bank}-{side}", x=cx + dx * 120, y=114, points=[[0, 0], [dx * 60, 60]],
                              strokeColor="#fab005", strokeWidth=2, strokeStyle="dashed", roughness=1))

    # rows: height follows the wrapped measure name
    heights = []
    for row in ROWS:
        lines = math.ceil(len(plain(row[0])) * 0.52 * MEASURE_SIZE / (MID_W - 20))
        has_interval = split_value(plain(row[1]))[1] is not None
        heights.append(max(lines * MEASURE_SIZE * 1.3 + 26, 96 if has_interval else 70))
    title_h, plate_h, row_gap = 70, 74, 12
    first_row = top_board + 24 + title_h + 14 + plate_h + 22
    board_h = first_row - top_board + sum(heights) + row_gap * (len(ROWS) - 1) + 36

    # legs, then the board in front of them
    leg_top, leg_h = top_board + board_h - 20, 200
    for side, lx in (("l", 170), ("r", W - 250)):
        scene.append(dict(type="rectangle", id=f"leg-{side}", x=lx, y=leg_top, width=80, height=leg_h, backgroundColor=STEEL,
                          fillStyle="cross-hatch", strokeColor=BOARD_EDGE, strokeWidth=2, roughness=1))
    scene.append(dict(type="line", id="ground", x=60, y=leg_top + leg_h, points=[[0, 0], [W - 120, 0]], strokeColor="#2f9e44", strokeWidth=4, roughness=2))
    scene.append(dict(type="rectangle", id="board", x=0, y=top_board, width=W, height=board_h, backgroundColor=BOARD, fillStyle="solid",
                      strokeColor=BOARD_EDGE, strokeWidth=4, roughness=1, roundness=dict(type=3)))
    for i in range(int(W / 60)):                     # marquee bulbs along the board's top and bottom edges
        for edge, ey in (("t", top_board + 6), ("b", top_board + board_h - 18)):
            scene.append(dict(type="ellipse", id=f"bulb-{edge}-{i}", x=24 + i * 60, y=ey, width=12, height=12, backgroundColor=BULB,
                              fillStyle="solid", strokeColor=BULB, strokeWidth=1, roughness=0))

    y = top_board + 24
    scene.append(holder("title", M, y, W - 2 * M, title_h, plain(TITLE), 34, POSTER, BULB))
    y += title_h + 14
    scene.append(dict(type="rectangle", id="plate-left", x=LEFT_X, y=y, width=BOX, height=plate_h, backgroundColor=ORANGE_STRONG, fillStyle="solid",
                      strokeColor=ORANGE_FILL, strokeWidth=3, roughness=1, angle=-0.02,
                      label=dict(text=plain(HEADER[1]), fontSize=28, fontFamily=POSTER, strokeColor=WHITE)))
    scene.append(dict(type="rectangle", id="plate-mid", x=MID_X + 60, y=y + 10, width=MID_W - 120, height=plate_h - 20, backgroundColor=STEEL,
                      fillStyle="solid", strokeColor=WHITE, strokeWidth=2, roughness=1,
                      label=dict(text=plain(HEADER[0]), fontSize=28, fontFamily=POSTER, strokeColor=WHITE)))
    scene.append(dict(type="rectangle", id="plate-right", x=RIGHT_X, y=y, width=BOX, height=plate_h, backgroundColor=BLUE_STRONG, fillStyle="solid",
                      strokeColor=BLUE_FILL, strokeWidth=3, roughness=1, angle=0.02,
                      label=dict(text=plain(HEADER[2]), fontSize=28, fontFamily=POSTER, strokeColor=WHITE)))

    y = first_row
    for r, (row, height) in enumerate(zip(ROWS, heights)):
        scene += digits(f"left-{r}", LEFT_X, y, height, row[1], ORANGE_DIGIT)
        if r < len(ROWS) - 1: scene.append(dict(type="line", id=f"rule-{r}", x=MID_X, y=y + height + row_gap / 2, points=[[0, 0], [MID_W, 0]],
                          strokeColor=STEEL, strokeWidth=1, strokeStyle="dotted", roughness=0))
        measure = plain(row[0]).replace("for sampling", "for\nsampling")     # keeps the seed list on one line
        scene.append(holder(f"measure-{r}", MID_X + 10, y, MID_W - 20, height, measure, MEASURE_SIZE, NUNITO, WHITE))
        scene += digits(f"right-{r}", RIGHT_X, y, height, row[2], BLUE_DIGIT)
        y += height + row_gap
    return scene


if __name__ == "__main__":
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE
    render_scene(build(), folder, NAME)
    missing = missing_from_scene(HEADER, ROWS, folder / f"{NAME}.excalidraw")
    if missing: print("missing:", missing)
    assert not missing
    print(folder / f"{NAME}-light.png")
