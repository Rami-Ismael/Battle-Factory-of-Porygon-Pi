"""Draw the 'How to read this' glossary of issue #297 as a cork noticeboard: one pinned sticky note per term.

Run:  /tmp/vgc-pilot/.venv/bin/python build_terms-corkboard.py [output folder]
"""
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[4] / "code" / "src"))
from excalidraw_table import missing_from_scene, render_scene  # noqa: E402

NAME = "terms-corkboard"
HEADER = ["Term", "What it means here"]
ROWS = [
    ["Row", "One way to train and sample the model. Four are compared. The control is Per-field, uncorrected t: our own settings"],
    ["Species-first and random order", "The order fields are filled in: all six species first, or any order"],
    ["Ask and got", "The ask is the win rate the model is told to aim for. Got is the win rate its teams then score"],
    ["Line-up", "A team's six species. More distinct line-ups means more variety"],
    ["Guidance", "How hard the sampler pushes toward the win rate asked for. 1 is no push, and higher pushes harder"],
    ["Ask-versus-got error", "How far the win rate the teams get is from the win rate asked for. Lower is better. A row must be 0.001 lower to count"],
]

INK, WOOD, WOOD_DARK, CORK, CORK_LINES, TAPE = "#1e1e1e", "#a0682f", "#5c3a17", "#e9c08a", "#b5793a", "#fff3bf"
NOTE_FILLS = ["#ffec99", "#ffc9c9", "#b2f2bb", "#d0bfff", "#99e9f2", "#eebefa"]
ANGLES = [-0.045, 0.03, -0.02, 0.05, -0.035, 0.025]                  # radians, a different small tilt per note
PINS = ["#e03131", "#1971c2", "#2f9e44", "#e03131", "#f08c00", "#6741d9"]

NOTE_W, NOTE_H, GAP_X, GAP_Y = 350, 300, 40, 40
PAD, HEAD_TOP, BODY_H = 20, 48, 160
HEAD_LINES = [1, 2, 1, 1, 1, 1]                                     # how many lines each term wraps to, read from a trial render
LINE = 38                                                            # one line of a size-30 Lilita One heading
BOARD_X, BOARD_Y, FRAME = 0, 60, 34
CORK_X, CORK_Y = BOARD_X + FRAME, BOARD_Y + FRAME
CORK_W = 3 * NOTE_W + 4 * GAP_X
TAPE_H, NOTES_TOP = 62, CORK_Y + 110
CORK_H = NOTES_TOP - CORK_Y + 2 * NOTE_H + GAP_Y + 50


def holder(id_, x, y, w, h, text, size, family, align="left", angle=0.0):
    return dict(type="rectangle", id=id_, x=x, y=y, width=w, height=h, angle=angle, strokeColor="transparent",
                backgroundColor="transparent", roughness=0,
                label=dict(text=text, fontSize=size, fontFamily=family, textAlign=align, verticalAlign="top", strokeColor=INK))


def placed(centre, offset, size, angle):
    """Top-left corner of an element of `size` whose centre sits at `offset` from `centre` in the note's rotated frame."""
    (cx, cy), (dx, dy), (w, h) = centre, offset, size
    rx, ry = dx * math.cos(angle) - dy * math.sin(angle), dx * math.sin(angle) + dy * math.cos(angle)
    return cx + rx - w / 2, cy + ry - h / 2


def skeleton():
    parts = [
        dict(type="text", id="title", x=CORK_X, y=0, text="How to read this", fontSize=36, fontFamily=7, strokeColor=INK),
        dict(type="rectangle", id="frame", x=BOARD_X, y=BOARD_Y, width=CORK_W + 2 * FRAME, height=CORK_H + 2 * FRAME,
             backgroundColor=WOOD, fillStyle="solid", strokeColor=WOOD_DARK, strokeWidth=3, roughness=1),
        dict(type="rectangle", id="frame-grain", x=BOARD_X + 8, y=BOARD_Y + 8, width=CORK_W + 2 * FRAME - 16, height=CORK_H + 2 * FRAME - 16,
             backgroundColor=WOOD_DARK, fillStyle="hachure", strokeColor="transparent", strokeWidth=1, roughness=2, opacity=35),
        dict(type="rectangle", id="cork", x=CORK_X, y=CORK_Y, width=CORK_W, height=CORK_H,
             backgroundColor=CORK, fillStyle="solid", strokeColor=WOOD_DARK, strokeWidth=2, roughness=1),
        dict(type="rectangle", id="cork-grain", x=CORK_X, y=CORK_Y, width=CORK_W, height=CORK_H,
             backgroundColor=CORK_LINES, fillStyle="cross-hatch", strokeColor="transparent", strokeWidth=1, roughness=2, opacity=45),
    ]
    # the masking tape strip across the top carries the two column headers
    tape_x, tape_w, tape_angle = CORK_X + 30, CORK_W - 60, -0.012
    parts.append(dict(type="rectangle", id="tape", x=tape_x, y=CORK_Y + 22, width=tape_w, height=TAPE_H, angle=tape_angle,
                      backgroundColor=TAPE, fillStyle="solid", strokeColor="#c9a227", strokeStyle="dashed", strokeWidth=1, roughness=2, opacity=92))
    tape_centre = (tape_x + tape_w / 2, CORK_Y + 22 + TAPE_H / 2)
    for id_, text, dx, w in (("head-term", HEADER[0], -tape_w / 2 + 40 + 130, 260), ("head-meaning", HEADER[1], 140, 520)):
        x, y = placed(tape_centre, (dx, 0), (w, 44), tape_angle)
        parts.append(holder(id_, x, y, w, 44, text, 30, 5, "center", tape_angle))
    for i, (term, meaning) in enumerate(ROWS):
        col, row = i % 3, i // 3
        x0, y0 = CORK_X + GAP_X + col * (NOTE_W + GAP_X), NOTES_TOP + row * (NOTE_H + GAP_Y)
        a, centre = ANGLES[i], (x0 + NOTE_W / 2, y0 + NOTE_H / 2)
        parts.append(dict(type="rectangle", id=f"shadow-{i}", x=x0 + 7, y=y0 + 9, width=NOTE_W, height=NOTE_H, angle=a,
                          backgroundColor=WOOD_DARK, fillStyle="solid", strokeColor="transparent", roughness=0, opacity=30))
        parts.append(dict(type="rectangle", id=f"note-{i}", x=x0, y=y0, width=NOTE_W, height=NOTE_H, angle=a,
                          backgroundColor=NOTE_FILLS[i], fillStyle="solid", strokeColor="#868e96", strokeWidth=1, roughness=1))
        inner, head_h = NOTE_W - 2 * PAD, HEAD_LINES[i] * LINE + 10
        hx, hy = placed(centre, (0, -NOTE_H / 2 + HEAD_TOP + head_h / 2), (inner, head_h), a)
        parts.append(holder(f"term-{i}", hx, hy, inner, head_h, term, 30, 7, "left", a))
        bx, by = placed(centre, (0, -NOTE_H / 2 + HEAD_TOP + head_h + 6 + BODY_H / 2), (inner, BODY_H), a)
        parts.append(holder(f"meaning-{i}", bx, by, inner, BODY_H, meaning, 22, 6, "left", a))
        px, py = placed(centre, (0, -NOTE_H / 2 + 20), (28, 28), a)
        parts.append(dict(type="ellipse", id=f"pin-{i}", x=px, y=py, width=28, height=28, backgroundColor=PINS[i], fillStyle="solid",
                          strokeColor=INK, strokeWidth=2, roughness=0))
        parts.append(dict(type="ellipse", id=f"pin-shine-{i}", x=px + 6, y=py + 5, width=8, height=8, backgroundColor="#ffffff",
                          fillStyle="solid", strokeColor="transparent", roughness=0))
    return parts


if __name__ == "__main__":
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE
    written = render_scene(skeleton(), folder, NAME)
    for path in written: print(path)
    missing = missing_from_scene(HEADER, ROWS, written[0])
    if missing: print("missing:", missing)
    assert not missing
