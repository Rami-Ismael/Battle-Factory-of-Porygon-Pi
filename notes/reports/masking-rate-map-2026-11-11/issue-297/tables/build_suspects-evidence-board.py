"""Draw the suspects table of issue #297 as a detective's evidence board: a pinned photo card per suspect,
a red string from each to the note card that holds the test that would settle it.

Run:  /tmp/vgc-pilot/.venv/bin/python build_suspects-evidence-board.py [output folder]
"""
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[4] / "code" / "src"))
from excalidraw_table import missing_from_scene, plain, render_scene  # noqa: E402

NAME = "suspects-evidence-board"
HEADER = ["Suspect", "Test that would settle it"]
ROWS = [
    ["The 2.8 came from a weaker model, the all-teams model after 21 epochs, whose species-first error was large", "Random order on the control's checkpoint, the same three grids"],
    ["Random order draws weaker teams whatever the ask: got stays between 0.32 and 0.39 from ask 13/24 upward", "Decode the six species first and the other 78 fields in random order"],
    ["The lookahead overrides the model's draw often", "Compare the species the model wanted with the species the lookahead left"],
]

INK, PAPER, WOOD, CORK = "#1e1e1e", "#ffffff", "#a0682f", "#e9c08a"
PHOTO, SHADOW, STRING, NOTE, NOTE_LINE = "#dee2e6", "#495057", "#e03131", "#fff3bf", "#74c0fc"
ORANGE_FILL, ORANGE = "#ffd8a8", "#e8590c"                            # the random-order row's colours: every suspect is about it
SUSPECT_ANGLES, NOTE_ANGLES = [-0.035, 0.028, -0.018], [0.03, -0.024, 0.042]

BOARD_W, FRAME = 1260, 28
CARD_X, CARD_W, NOTE_X, NOTE_W = 70, 470, 730, 460
CARD_H = [296, 296, 268]
NOTE_H = [146, 146, 172]
ROW_GAP, ROWS_TOP = 60, 260
TEXT_SIZE = 22


def holder(id_, x, y, w, h, text, size, family, align="left", colour=INK, vertical="top"):
    return dict(type="rectangle", id=id_, x=x, y=y, width=w, height=h, strokeColor="transparent", backgroundColor="transparent",
                roughness=0, label=dict(text=text, fontSize=size, fontFamily=family, textAlign=align, verticalAlign=vertical, strokeColor=colour))


def tilted(parts, ox, oy, w, h, angle):
    """Move parts drawn relative to a card's top-left corner onto the board, turning them with the card about its centre."""
    cx, cy, cos, sin = ox + w / 2, oy + h / 2, math.cos(angle), math.sin(angle)
    turn = lambda px, py: (cx + (px - w / 2) * cos - (py - h / 2) * sin, cy + (px - w / 2) * sin + (py - h / 2) * cos)
    placed = []
    for part in parts:
        part = dict(part)
        if part["type"] in ("line", "arrow"):
            x0, y0 = turn(part["x"], part["y"])
            part["points"] = [[dx * cos - dy * sin, dx * sin + dy * cos] for dx, dy in part["points"]]
        else:
            mx, my = turn(part["x"] + part["width"] / 2, part["y"] + part["height"] / 2)
            x0, y0 = mx - part["width"] / 2, my - part["height"] / 2
        part.update(x=x0, y=y0, angle=angle)
        placed.append(part)
    return placed, turn


def suspect_card(i, row, top):
    w, h = CARD_W, CARD_H[i]
    parts = [
        dict(type="rectangle", id=f"card-{i}", x=0, y=0, width=w, height=h, backgroundColor=PAPER, fillStyle="solid", strokeColor=INK, strokeWidth=1, roughness=1),
        dict(type="rectangle", id=f"photo-{i}", x=20, y=20, width=170, height=150, backgroundColor=PHOTO, fillStyle="solid", strokeColor=SHADOW, strokeWidth=1, roughness=1),
        dict(type="rectangle", id=f"shoulders-{i}", x=40, y=122, width=130, height=48, backgroundColor=SHADOW, fillStyle="solid", strokeColor=SHADOW, roughness=1, roundness={"type": 3}),
        dict(type="ellipse", id=f"head-{i}", x=75, y=46, width=60, height=72, backgroundColor=SHADOW, fillStyle="solid", strokeColor=SHADOW, roughness=1),
        dict(type="rectangle", id=f"tag-{i}", x=215, y=60, width=200, height=52, backgroundColor=ORANGE_FILL, fillStyle="solid", strokeColor=ORANGE, strokeWidth=2, roughness=1,
             label=dict(text=f"SUSPECT {i + 1}", fontSize=24, fontFamily=8, textAlign="center", verticalAlign="middle", strokeColor=INK)),
        holder(f"suspect-{i}", 20, 188, w - 40, h - 208, plain(row[0]), TEXT_SIZE, 5),
    ]
    placed, turn = tilted(parts, CARD_X, top, w, h, SUSPECT_ANGLES[i])
    return placed, turn(w - 26, 16)


def note_card(i, row, top):
    w, h = NOTE_W, NOTE_H[i]
    parts = [
        dict(type="rectangle", id=f"note-{i}", x=0, y=0, width=w, height=h, backgroundColor=NOTE, fillStyle="solid", strokeColor=INK, strokeWidth=1, roughness=1),
        dict(type="line", id=f"note-rule-{i}", x=14, y=52, points=[[0, 0], [w - 28, 0]], strokeColor=STRING, strokeWidth=1, roughness=1),
        holder(f"note-head-{i}", 46, 12, w - 70, 36, f"TEST FOR SUSPECT {i + 1}", 20, 8, colour=SHADOW),
        holder(f"test-{i}", 24, 66, w - 48, h - 84, plain(row[1]), TEXT_SIZE, 5),
    ]
    placed, turn = tilted(parts, NOTE_X, top, w, h, NOTE_ANGLES[i])
    return placed, turn(24, 26)


def pin(id_, x, y):
    return dict(type="ellipse", id=id_, x=x - 10, y=y - 10, width=20, height=20, backgroundColor=STRING, fillStyle="solid", strokeColor="#a61e4d", strokeWidth=1, roughness=0)


def skeleton():
    tops, y = [], ROWS_TOP
    for i in range(len(ROWS)):
        tops.append(y)
        y += max(CARD_H[i], NOTE_H[i]) + ROW_GAP
    board_h = y + 60
    elements = [
        dict(type="rectangle", id="frame", x=0, y=0, width=BOARD_W, height=board_h, backgroundColor=WOOD, fillStyle="solid", strokeColor="#5c3a17", strokeWidth=2, roughness=1),
        dict(type="rectangle", id="cork", x=FRAME, y=FRAME, width=BOARD_W - 2 * FRAME, height=board_h - 2 * FRAME, backgroundColor=CORK, fillStyle="solid", strokeColor="#5c3a17", strokeWidth=1, roughness=1),
        dict(type="rectangle", id="title", x=270, y=56, width=720, height=72, angle=-0.012, backgroundColor=PAPER, fillStyle="solid", strokeColor=INK, strokeWidth=1, roughness=1,
             label=dict(text="OPEN CASE: THE EXPECTATION THAT MISSED", fontSize=32, fontFamily=7, textAlign="center", verticalAlign="middle", strokeColor=INK)),
        dict(type="rectangle", id="head-suspect", x=CARD_X + 105, y=168, width=260, height=50, angle=0.015, backgroundColor=PAPER, fillStyle="solid", strokeColor=INK, strokeWidth=1, roughness=0,
             label=dict(text=HEADER[0], fontSize=26, fontFamily=8, textAlign="center", verticalAlign="middle", strokeColor=INK)),
        dict(type="rectangle", id="head-test", x=NOTE_X + 20, y=168, width=420, height=50, angle=-0.015, backgroundColor=PAPER, fillStyle="solid", strokeColor=INK, strokeWidth=1, roughness=0,
             label=dict(text=HEADER[1], fontSize=26, fontFamily=8, textAlign="center", verticalAlign="middle", strokeColor=INK)),
    ]
    strings, pins = [], []
    for i, row in enumerate(ROWS):
        card, (ax, ay) = suspect_card(i, row, tops[i])
        note, (bx, by) = note_card(i, row, tops[i] + (CARD_H[i] - NOTE_H[i]) / 2 - 30)
        elements += card + note
        dx, dy = bx - ax, by - ay
        strings.append(dict(type="line", id=f"string-{i}", x=ax, y=ay, points=[[0, 0], [dx / 2, dy / 2 + 28], [dx, dy]], strokeColor=STRING, strokeWidth=3, roughness=1))
        pins += [pin(f"pin-suspect-{i}", ax, ay), pin(f"pin-note-{i}", bx, by)]
    elements += strings + pins
    elements.append(dict(type="rectangle", id="footer", x=NOTE_X + 40, y=board_h - 112, width=380, height=50, angle=0.02, backgroundColor=PAPER, fillStyle="solid",
                         strokeColor=STRING, strokeWidth=2, roughness=1,
                         label=dict(text="None of these was run.", fontSize=24, fontFamily=8, textAlign="center", verticalAlign="middle", strokeColor=STRING)))
    return elements


if __name__ == "__main__":
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE
    render_scene(skeleton(), folder, NAME)
    missing = missing_from_scene(HEADER, ROWS, folder / f"{NAME}.excalidraw")
    if missing: print("missing:", missing)
    assert not missing
