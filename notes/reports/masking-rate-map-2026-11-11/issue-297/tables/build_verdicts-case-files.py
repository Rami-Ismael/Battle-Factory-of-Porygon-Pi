"""Draw the verdict table of issue #297 as three case files on a judge's desk: one manila folder per row,
the row name on its tab, the Why text on a sheet inside, and the verdict as a tilted rubber stamp.

Run:  /tmp/vgc-pilot/.venv/bin/python build_verdicts-case-files.py [output folder]
"""
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[4] / "code" / "src"))
from excalidraw_table import missing_from_scene, plain, render_scene  # noqa: E402

NAME = "verdicts-case-files"
HEADER = ["Row", "Verdict against the control", "Why"]
ROWS = [
    ["Linear, species-first", "**Matches** at guidance 2, 4 and 8", "Error gaps of -0.0008, 0.0000 and +0.0005. Every interval spans zero, and no gap reaches the 0.002 of training luck"],
    ["Linear, random order", "**Loses** at guidance 2, 4 and 8", "Error higher by 0.024 to 0.027 on every sampling seed"],
    ["Per-field, corrected t", "Not battled", "Its probe read 0.0078 against the bar of 0.01"],
]

INK, DESK, DESK_DARK, MANILA, MANILA_EDGE, SHEET = "#1e1e1e", "#c08a52", "#6b4520", "#f5d9a0", "#a8803e", "#fffdf6"
TABS = [("#96f2d7", "#099268"), ("#ffd8a8", "#e8590c"), ("#d0bfff", "#6741d9")]   # teal species-first, orange random order, violet per-field corrected t
STAMPS = [("#2f9e44", "#ebfbee"), ("#e03131", "#fff5f5"), ("#868e96", "#f8f9fa")]  # match, loss, not battled
STAMP_ANGLES = [-0.09, 0.07, -0.06]
SHEET_ANGLES = [0.008, -0.01, 0.012]

DESK_X, DESK_Y, DESK_W = 0, 0, 1260
FOLDER_X, FOLDER_W, FOLDER_H, TAB_W, TAB_H = 30, 1200, 220, 440, 58
FIRST_FOLDER_Y, FOLDER_STEP = 250, 300
STAMP_W, STAMP_H, SHEET_X, SHEET_W, SHEET_H = 420, 150, 520, 680, 160


def holder(id_, x, y, w, h, text, size, family, colour=INK, align="left", valign="top", angle=0.0):
    return dict(type="rectangle", id=id_, x=x, y=y, width=w, height=h, angle=angle, strokeColor="transparent",
                backgroundColor="transparent", roughness=0,
                label=dict(text=text, fontSize=size, fontFamily=family, textAlign=align, verticalAlign=valign, strokeColor=colour))


def placed(centre, offset, size, angle):
    """Top-left corner of an element of `size` whose centre sits at `offset` from `centre` in a frame rotated by `angle`."""
    (cx, cy), (dx, dy), (w, h) = centre, offset, size
    rx, ry = dx * math.cos(angle) - dy * math.sin(angle), dx * math.sin(angle) + dy * math.cos(angle)
    return cx + rx - w / 2, cy + ry - h / 2


def stamp(i, x0, y0, verdict):
    ink, tint = STAMPS[i]
    a, centre = STAMP_ANGLES[i], (x0 + STAMP_W / 2, y0 + STAMP_H / 2)
    parts = [dict(type="rectangle", id=f"stamp-{i}", x=x0, y=y0, width=STAMP_W, height=STAMP_H, angle=a, backgroundColor=tint,
                  fillStyle="solid", strokeColor=ink, strokeWidth=4, strokeStyle="dashed", roughness=1, opacity=92),
             dict(type="rectangle", id=f"stamp-inner-{i}", x=x0 + 12, y=y0 + 12, width=STAMP_W - 24, height=STAMP_H - 24, angle=a,
                  backgroundColor="transparent", strokeColor=ink, strokeWidth=2, roughness=1)]
    if "**" in verdict:
        word, rest = verdict.split("**")[1], verdict.split("**")[2].strip()
        wx, wy = placed(centre, (0, -22), (STAMP_W - 40, 74), a)
        parts.append(holder(f"stamp-word-{i}", wx, wy, STAMP_W - 40, 74, word, 64, 7, ink, "center", "middle", a))
        rx, ry = placed(centre, (0, 42), (STAMP_W - 40, 36), a)
        parts.append(holder(f"stamp-rest-{i}", rx, ry, STAMP_W - 40, 36, rest, 28, 7, ink, "center", "middle", a))
    else:
        wx, wy = placed(centre, (0, 0), (STAMP_W - 40, 74), a)
        parts.append(holder(f"stamp-word-{i}", wx, wy, STAMP_W - 40, 74, verdict, 60, 7, ink, "center", "middle", a))
    return parts


def skeleton():
    desk_h = FIRST_FOLDER_Y + 2 * FOLDER_STEP + FOLDER_H + 30 - DESK_Y
    parts = [
        dict(type="rectangle", id="desk", x=DESK_X, y=DESK_Y, width=DESK_W, height=desk_h, backgroundColor=DESK, fillStyle="solid",
             strokeColor=DESK_DARK, strokeWidth=3, roughness=1, roundness=dict(type=3)),
        dict(type="rectangle", id="desk-grain", x=DESK_X + 10, y=DESK_Y + 10, width=DESK_W - 20, height=desk_h - 20, backgroundColor=DESK_DARK,
             fillStyle="hachure", strokeColor="transparent", roughness=2, opacity=25),
        dict(type="text", id="title", x=DESK_X + 40, y=DESK_Y + 40, text="Three case files,\neach judged against the control",
             fontSize=40, fontFamily=7, strokeColor=INK),
    ]
    # the desk plaque names what each part of a folder carries: the table's three column headers
    px, py, pw, ph = 720, DESK_Y + 28, 500, 136
    parts += [dict(type="rectangle", id="plaque-base", x=px - 14, y=py + ph - 6, width=pw + 28, height=22, backgroundColor=DESK_DARK,
                   fillStyle="solid", strokeColor=DESK_DARK, roughness=0),
              dict(type="rectangle", id="plaque", x=px, y=py, width=pw, height=ph, backgroundColor="#343a40", fillStyle="solid",
                   strokeColor="#f08c00", strokeWidth=4, roughness=0, roundness=dict(type=3)),
              holder("plaque-text", px + 20, py + 16, pw - 40, ph - 32,
                     f"Tab: {HEADER[0]}\nStamp: {HEADER[1]}\nSheet: {HEADER[2]}", 28, 6, "#ffffff", "center", "middle")]
    for i, (row, verdict, why) in enumerate(ROWS):
        y0 = FIRST_FOLDER_Y + i * FOLDER_STEP
        fill, strong = TABS[i]
        parts += [
            dict(type="rectangle", id=f"tab-{i}", x=FOLDER_X + 30, y=y0 - TAB_H + 14, width=TAB_W, height=TAB_H, backgroundColor=fill,
                 fillStyle="solid", strokeColor=strong, strokeWidth=3, roughness=1, roundness=dict(type=3)),
            holder(f"tab-text-{i}", FOLDER_X + 50, y0 - TAB_H + 14, TAB_W - 40, 40, row, 30, 7, INK, "left", "middle"),
            dict(type="rectangle", id=f"folder-{i}", x=FOLDER_X, y=y0, width=FOLDER_W, height=FOLDER_H, backgroundColor=MANILA,
                 fillStyle="solid", strokeColor=MANILA_EDGE, strokeWidth=2, roughness=1),
            dict(type="rectangle", id=f"folder-strip-{i}", x=FOLDER_X, y=y0, width=FOLDER_W, height=10, backgroundColor=strong,
                 fillStyle="solid", strokeColor="transparent", roughness=0),
        ]
        sx, sy = SHEET_X, y0 + 34
        parts += [
            dict(type="rectangle", id=f"sheet-shadow-{i}", x=sx + 6, y=sy + 7, width=SHEET_W, height=SHEET_H, angle=SHEET_ANGLES[i],
                 backgroundColor=MANILA_EDGE, fillStyle="solid", strokeColor="transparent", roughness=0, opacity=40),
            dict(type="rectangle", id=f"sheet-{i}", x=sx, y=sy, width=SHEET_W, height=SHEET_H, angle=SHEET_ANGLES[i], backgroundColor=SHEET,
                 fillStyle="solid", strokeColor="#adb5bd", strokeWidth=1, roughness=1),
            dict(type="line", id=f"sheet-margin-{i}", x=sx + 30, y=sy + 4, points=[[0, 0], [0, SHEET_H - 8]], strokeColor="#ffa8a8",
                 strokeWidth=1, roughness=0, angle=SHEET_ANGLES[i]),
        ]
        centre = (sx + SHEET_W / 2, sy + SHEET_H / 2)
        tx, ty = placed(centre, (15, 0), (SHEET_W - 80, SHEET_H - 40), SHEET_ANGLES[i])
        parts.append(holder(f"why-{i}", tx, ty, SHEET_W - 80, SHEET_H - 40, plain(why), 24, 6, INK, "left", "middle", SHEET_ANGLES[i]))
        parts += stamp(i, FOLDER_X + 40, y0 + 40, verdict)
    return parts


if __name__ == "__main__":
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE
    written = render_scene(skeleton(), folder, NAME)
    for path in written: print(path)
    missing = missing_from_scene(HEADER, ROWS, written[0])
    if missing: print("missing:", missing)
    assert not missing
