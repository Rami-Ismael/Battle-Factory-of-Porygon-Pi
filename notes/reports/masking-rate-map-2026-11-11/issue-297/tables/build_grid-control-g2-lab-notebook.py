"""Draw the control's guidance 2 grid table as a page torn from a lab notebook, in light and dark PNGs.

Run:  python build_grid-control-g2-lab-notebook.py [output folder]
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[4] / "code" / "src"))
from excalidraw_table import missing_from_scene, plain, render_scene  # noqa: E402

NAME = "grid-control-g2-lab-notebook"
HEADER = ["Measure", "Per-field, uncorrected t, guidance 2"]
ROWS = [
    ["Ask-versus-got error, asks 1/24 to 17/24, with its 95% interval", "0.0084 (0.0045 to 0.0134)"],
    ["The same error for sampling seeds 1, 2 and 3", "0.0109, 0.0067, 0.0079"],
    ["Error on asks 18/24 to 23/24", "0.1352"],
    ["Win rate at asks 15/24 to 17/24", "0.497 (0.458 to 0.538)"],
    ["Win rate over all 23 asks", "0.354 (0.328 to 0.381)"],
    ["First-try legality", "100.0%"],
    ["Distinct line-ups", "1,513"],
    ["Teams with a line-up absent from the training teams", "17.5%"],
    ["Teams with a line-up absent from earlier search runs and tournament records", "82.7%"],
    ["Effective species and top-species share, in 5-team subsets", "15.0 and 0.136"],
    ["Teams and battles", "4,416 and 216,384"],
]

HAND = 5
PAPER, RULE, MARGIN, INK, PENCIL = "#fff9db", "#a5d8ff", "#e03131", "#1971c2", "#343a40"
TAPE = "#e9ecef"
PAGE_W = 1240
LEFT, LABEL_W = 132, 560                  # the measure is written right of the red margin
VALUE_X, VALUE_W = 712, 492               # the value is written large at the right
PITCH = 44                                # space between two ruled lines
ENTRY = 2 * PITCH                         # a one-line entry takes two ruled spaces: one left blank, one written on
TOP = 208                                 # first ruled line under the tape
# measured widths of each value at size 30, from a trial render, so the hand-drawn underline fits the ink
VALUE_WIDTHS = {0: 396, 1: 343, 2: 98, 3: 339, 4: 338, 5: 107, 6: 70, 7: 84, 8: 93, 9: 212, 10: 266}


def holder(id_, x, y, width, height, text, size, colour, align="left", vertical="middle"):
    return dict(type="rectangle", id=id_, x=x, y=y, width=width, height=height, strokeColor="transparent",
                backgroundColor="transparent", roughness=0,
                label=dict(text=text, fontSize=size, fontFamily=HAND, textAlign=align, verticalAlign=vertical, strokeColor=colour))


def skeleton():
    written = [measure_lines(plain(measure)) for measure, _ in ROWS]
    page_h = TOP + PITCH * sum(len(lines) + 1 for lines in written) + 48
    items = [dict(type="rectangle", id="page", x=0, y=0, width=PAGE_W, height=page_h, backgroundColor=PAPER, fillStyle="solid",
                  strokeColor="#ced4da", strokeWidth=1, roughness=1, angle=-0.004)]
    # ruled lines, every PITCH units, from under the tape to the foot of the page
    y = TOP
    while y < page_h - 20:
        items.append(dict(type="line", id=f"rule-{y}", x=12, y=y, points=[[0, 0], [PAGE_W - 24, 0]], strokeColor=RULE, strokeWidth=1, roughness=0))
        y += PITCH
    items.append(dict(type="line", id="margin", x=108, y=8, points=[[0, 0], [0, page_h - 16]], strokeColor=MARGIN, strokeWidth=2, roughness=0))
    for index, hole_y in enumerate(range(120, page_h - 60, (page_h - 180) // 4)):
        items.append(dict(type="ellipse", id=f"hole-{index}", x=34, y=hole_y, width=38, height=38, backgroundColor="#f1f3f5",
                          fillStyle="solid", strokeColor="#adb5bd", strokeWidth=2, roughness=1))
    items.append(holder("title", LEFT, 26, 1060, 56, "Grid finished: the control at guidance 2", 34, PENCIL, vertical="middle"))
    # the label taped across the top carries the column headers
    items.append(dict(type="rectangle", id="tape", x=118, y=104, width=1100, height=84, backgroundColor=TAPE, fillStyle="solid",
                      strokeColor="#adb5bd", strokeWidth=1, roughness=1, opacity=85, angle=-0.012))
    items.append(dict(type="rectangle", id="tape-hatch", x=118, y=104, width=1100, height=84, backgroundColor="#ced4da", fillStyle="hachure",
                      strokeColor="transparent", strokeWidth=1, roughness=1, opacity=35, angle=-0.012))
    items.append(holder("head-0", LEFT + 6, 110, LABEL_W, 72, plain(HEADER[0]), 30, PENCIL))
    items.append(holder("head-1", VALUE_X - 40, 110, VALUE_W + 40, 72, plain(HEADER[1]), 28, INK, align="right"))
    top = TOP
    for r, ((measure, value), lines) in enumerate(zip(ROWS, written)):
        # one blank ruled space between entries, then each line of handwriting in its own space between two ruled lines
        top += PITCH * (len(lines) + 1)         # the entry's last line, and its value, are written on this ruled line
        for k, line in enumerate(lines):
            band = top - PITCH * (len(lines) - k)
            items.append(holder(f"measure-{r}-{k}", LEFT, band, LABEL_W, PITCH, line, 24, PENCIL))
        items.append(holder(f"value-{r}", VALUE_X, top - PITCH, VALUE_W, PITCH, plain(value), 30, INK, align="right"))
        # the underline is the ruled line itself, gone over in ink under the value
        width = VALUE_WIDTHS.get(r, 12 * len(plain(value)) + 30)
        right = VALUE_X + VALUE_W
        items.append(dict(type="line", id=f"under-{r}", x=right - width - 6, y=top,
                          points=[[0, 0], [width + 8, 0]], strokeColor=INK, strokeWidth=3, roughness=0))
    return items


def measure_lines(text, limit=45):
    """Break a measure into lines short enough for one ruled space each, keeping the break after 'share,'."""
    lines = []
    for part in text.replace("share, in ", "share,\nin ").split("\n"):
        line = ""
        for word in part.split(" "):
            if line and len(line) + 1 + len(word) > limit:
                lines.append(line)
                line = word
            else:
                line = f"{line} {word}" if line else word
        lines.append(line)
    return lines


def main():
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE
    render_scene(skeleton(), folder, NAME)
    scene_path = folder / f"{NAME}.excalidraw"
    missing = missing_from_scene(HEADER, ROWS, scene_path)
    if missing: print("missing:", missing)
    assert not missing


if __name__ == "__main__":
    main()
