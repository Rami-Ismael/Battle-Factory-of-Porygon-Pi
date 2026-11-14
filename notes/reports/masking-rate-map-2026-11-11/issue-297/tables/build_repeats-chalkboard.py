"""Draw the repeats-between-asks table as a school chalkboard of subtraction sums, one sum per table row."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[4] / "code" / "src"))
from excalidraw_table import missing_from_scene, plain, render_scene  # noqa: E402

NAME = "repeats-chalkboard"
HEADER = ["Row", "Guidance", "Counted ask by ask", "Repeats between asks", "Distinct over the grid"]
ROWS = [
    ["Per-field, uncorrected t, the control", "2", "3,842", "2,329", "1,513"],
    ["Per-field, uncorrected t, the control", "4", "3,594", "1,676", "1,918"],
    ["Per-field, uncorrected t, the control", "8", "3,102", "976", "2,126"],
    ["Linear, species-first", "2", "3,746", "2,168", "1,578"],
    ["Linear, species-first", "4", "3,571", "1,600", "1,971"],
    ["Linear, species-first", "8", "3,046", "986", "2,060"],
    ["Linear, random order", "2", "4,408", "1,762", "2,646"],
    ["Linear, random order", "4", "4,406", "1,070", "3,336"],
    ["Linear, random order", "8", "4,398", "578", "3,820"],
]

CHALK, BOARD, WOOD, WOOD_DARK = "#f8f9fa", "#24513b", "#a9713a", "#5c3a1a"
CHALK_TINTS = {"control": "#a5d8ff", "species-first": "#96f2d7", "random": "#ffd8a8"}   # the rows' pale fills, used as chalk
HAND = 5
FRAME_W, BORDER = 1300, 34
BOARD_L, BOARD_R = BORDER, FRAME_W - BORDER
COL_W = (BOARD_R - BOARD_L - 60) / 3
ROW_TOP, ROW_H = 425, 335


def tint_of(row_name):
    return next(colour for word, colour in CHALK_TINTS.items() if word in row_name)


def text(id, x, y, words, size, colour=CHALK):
    return dict(type="text", id=id, x=x, y=y, text=words, fontSize=size, fontFamily=HAND, strokeColor=colour)


def holder(id, x, y, w, h, words, size, colour, align="right"):
    return dict(type="rectangle", id=id, x=x, y=y, width=w, height=h, strokeColor="transparent", backgroundColor="transparent", roughness=0,
                label=dict(text=words, fontSize=size, fontFamily=HAND, textAlign=align, verticalAlign="top", strokeColor=colour))


def chalk_line(id, x, y, length, colour, width=3):
    return dict(type="line", id=id, x=x, y=y, points=[[0, 0], [length, 0]], strokeColor=colour, strokeWidth=width, roughness=2)


def sum_at(prefix, centre, top, guidance, words, size, colour, number_w):
    """A column subtraction: the guidance above, the minuend, minus the subtrahend, a chalk line, then the answer."""
    counted, repeats, distinct = words
    right = centre + number_w / 2
    line_h = size * 1.25 + 8
    return [
        holder(f"{prefix}-guidance", centre - 150, top, 300, 36, guidance, 26, colour, "center"),
        holder(f"{prefix}-counted", right - number_w, top + 46, number_w, line_h, counted, size, colour),
        text(f"{prefix}-minus", right - number_w - 44, top + 46 + line_h, "-", size + 4, colour),
        holder(f"{prefix}-repeats", right - number_w, top + 46 + line_h, number_w, line_h, repeats, size, colour),
        chalk_line(f"{prefix}-rule", right - number_w - 50, top + 54 + 2 * line_h, number_w + 60, colour),
        holder(f"{prefix}-distinct", right - number_w, top + 64 + 2 * line_h, number_w, line_h, distinct, size, colour),
    ]


def scene():
    header, rows = [plain(cell) for cell in HEADER], [[plain(cell) for cell in row] for row in ROWS]
    for row in rows:                                                     # check each sum before drawing it
        counted, repeats, distinct = (int(cell.replace(",", "")) for cell in row[2:])
        if counted - repeats != distinct: print(f"does not subtract: {row}")
    board_bottom = ROW_TOP + 3 * ROW_H + 10
    s = [
        dict(type="rectangle", id="frame", x=0, y=0, width=FRAME_W, height=board_bottom + BORDER + 40, backgroundColor=WOOD, fillStyle="solid",
             strokeColor=WOOD_DARK, strokeWidth=3, roughness=1),
        dict(type="rectangle", id="board", x=BOARD_L, y=BORDER, width=BOARD_R - BOARD_L, height=board_bottom - BORDER, backgroundColor=BOARD,
             fillStyle="solid", strokeColor=WOOD_DARK, strokeWidth=2, roughness=1),
        dict(type="rectangle", id="ledge", x=0, y=board_bottom, width=FRAME_W, height=34, backgroundColor=WOOD_DARK, fillStyle="solid",
             strokeColor=WOOD_DARK, strokeWidth=2, roughness=1),
        dict(type="rectangle", id="chalk-stick", x=900, y=board_bottom - 4, width=90, height=18, backgroundColor=CHALK, fillStyle="solid",
             strokeColor="#ced4da", strokeWidth=1, roughness=1, angle=-0.08),
        dict(type="rectangle", id="chalk-stub", x=1010, y=board_bottom + 2, width=34, height=16, backgroundColor="#ffd8a8", fillStyle="solid",
             strokeColor="#ced4da", strokeWidth=1, roughness=1, angle=0.2),
        dict(type="rectangle", id="eraser", x=230, y=board_bottom - 14, width=150, height=30, backgroundColor="#495057", fillStyle="solid",
             strokeColor="#212529", strokeWidth=2, roughness=1),
        dict(type="rectangle", id="eraser-felt", x=230, y=board_bottom + 8, width=150, height=8, backgroundColor="#ced4da", fillStyle="solid",
             strokeColor="#212529", strokeWidth=1, roughness=1),
        text("title", 90, 58, "Take away the repeats", 48),
    ]
    # the worked example: the five column names, written where the real sums put their values
    key_top = 140
    s += [
        dict(type="rectangle", id="key", x=70, y=key_top, width=FRAME_W - 140, height=250, backgroundColor="transparent",
             strokeColor=CHALK, strokeWidth=2, strokeStyle="dashed", roughness=2),
        text("key-tag", 96, key_top + 18, "Worked example", 30, "#ffec99"),
        holder("key-row", 96, key_top + 70, 330, 40, header[0], 30, CHALK, "left"),
        chalk_line("key-row-rule", 96, key_top + 112, 90, CHALK, 2),
        text("key-row-note", 96, key_top + 130, "written above each\nline of three sums", 24, "#ced4da"),
    ]
    s += sum_at("key", 800, key_top + 18, header[1], header[2:], 32, CHALK, 420)
    for r in range(3):
        group = rows[3 * r: 3 * r + 3]
        top, tint = ROW_TOP + r * ROW_H, tint_of(group[0][0])
        s += [text(f"row-{r}", 90, top, group[0][0], 30, tint), chalk_line(f"row-{r}-rule", 90, top + 42, 24 * len(group[0][0]) * 0.55 + 40, tint, 2)]
        for k, row in enumerate(group):
            centre = BOARD_L + 30 + COL_W * (k + 0.5)
            s += sum_at(f"sum-{r}-{k}", centre, top + 62, f"{header[1].lower()} {row[1]}", row[2:], 42, tint, 150)
    return s


if __name__ == "__main__":
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE
    render_scene(scene(), folder, NAME)
    scene_path = folder / f"{NAME}.excalidraw"
    missing = missing_from_scene(HEADER, ROWS, scene_path)
    if missing: print("missing:", missing)
    assert not missing
    print(scene_path)
