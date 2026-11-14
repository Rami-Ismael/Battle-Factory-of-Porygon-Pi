"""Draw the guidance-2 table of Linear, species-first against the control as a boxing tale of the tape."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[4] / "code" / "src"))
from excalidraw_table import missing_from_scene, plain, render_scene  # noqa: E402

NAME = "grid-linear-species-first-g2-tale-of-the-tape"
HEADER = ["Measure", "Linear, species-first, guidance 2", "The control at guidance 2"]
ROWS = [
    ["Ask-versus-got error, asks 1/24 to 17/24, with its 95% interval", "0.0075 (0.0039 to 0.0126)", "0.0084 (0.0045 to 0.0134)"],
    ["The same error for sampling seeds 1, 2 and 3", "0.0086, 0.0072, 0.0071", "0.0109, 0.0067, 0.0079"],
    ["Error on asks 18/24 to 23/24", "0.1521", "0.1352"],
    ["Win rate at asks 15/24 to 17/24", "0.503 (0.461 to 0.545)", "0.497 (0.458 to 0.538)"],
    ["Win rate over all 23 asks", "0.353 (0.326 to 0.380)", "0.354 (0.328 to 0.381)"],
    ["First-try legality", "100.0%", "100.0%"],
    ["Distinct line-ups", "1,578", "1,513"],
    ["Teams with a line-up absent from the training teams", "18.3%", "17.5%"],
    ["Teams with a line-up absent from earlier search runs and tournament records", "81.1%", "82.7%"],
    ["Effective species and top-species share, in 5-team subsets", "14.9 and 0.138", "15.0 and 0.136"],
    ["Teams and battles", "4,416 and 216,384", "4,416 and 216,384"],
]

TEAL, TEAL_STRONG, BLUE, BLUE_STRONG = "#96f2d7", "#099268", "#a5d8ff", "#1971c2"
INK, PLATE, WHITE, ROPE = "#1e1e1e", "#343a40", "#ffffff", "#e03131"
POSTER, CLEAN = 7, 6
LEFT, MID, RIGHT, SIDE, PLATE_W = 0, 430, 850, 410, 400
WIDTH = RIGHT + SIDE
ROW_H, ROW_GAP = 86, 14


def label(text, size, family, colour, align="center"):
    return dict(text=text, fontSize=size, fontFamily=family, textAlign=align, verticalAlign="middle", strokeColor=colour)


def box(id, x, y, w, h, fill, stroke, text=None, angle=0, **extra):
    shape = dict(type="rectangle", id=id, x=x, y=y, width=w, height=h, backgroundColor=fill, fillStyle="solid",
                 strokeColor=stroke, strokeWidth=2, roughness=1, roundness={"type": 3}, angle=angle, **extra)
    if text: shape["label"] = text
    return shape


def holder(id, x, y, w, h, text):
    return dict(type="rectangle", id=id, x=x, y=y, width=w, height=h, strokeColor="transparent", backgroundColor="transparent", roughness=0, label=text)


def scene():
    header, rows = [plain(cell) for cell in HEADER], [[plain(cell) for cell in row] for row in ROWS]
    s = [dict(type="text", id="title", x=WIDTH / 2 - 261, y=0, text="TALE OF THE TAPE", fontSize=64, fontFamily=POSTER, strokeColor=ROPE)]
    top = 110
    # ring ropes behind the corner banners
    for k, colour in enumerate((ROPE, INK, ROPE)):
        s.append(dict(type="line", id=f"rope-{k}", x=-6, y=top + 40 + 40 * k, points=[[0, 0], [WIDTH + 12, 0]], strokeColor=colour, strokeWidth=3, roughness=1))
    s += [
        dict(type="text", id="corner-left", x=LEFT + 20, y=top, text="TEAL CORNER", fontSize=26, fontFamily=POSTER, strokeColor=TEAL_STRONG),
        dict(type="text", id="corner-right", x=RIGHT + SIDE - 190, y=top, text="BLUE CORNER", fontSize=26, fontFamily=POSTER, strokeColor=BLUE_STRONG),
        box("banner-left", LEFT, top + 40, SIDE, 120, TEAL_STRONG, INK, label(header[1], 36, POSTER, WHITE), angle=-0.03),
        box("banner-right", RIGHT, top + 40, SIDE, 120, BLUE_STRONG, INK, label(header[2], 36, POSTER, WHITE), angle=0.03),
        dict(type="ellipse", id="vs", x=WIDTH / 2 - 55, y=top + 45, width=110, height=110, backgroundColor="#ffec99", fillStyle="solid",
             strokeColor=ROPE, strokeWidth=3, roughness=1, label=label("VS", 44, POSTER, ROPE)),
    ]
    y = top + 200
    s.append(box("plate-head", MID, y, PLATE_W, 56, "#ffec99", INK, label(header[0], 30, POSTER, INK)))
    y += 56 + ROW_GAP + 6
    for r, (measure, left, right) in enumerate(rows):
        s += [
            box(f"left-{r}", LEFT, y + 6, SIDE, ROW_H - 12, TEAL, TEAL_STRONG),
            box(f"right-{r}", RIGHT, y + 6, SIDE, ROW_H - 12, BLUE, BLUE_STRONG),
            holder(f"left-text-{r}", LEFT + 22, y + 6, SIDE - 44, ROW_H - 12, label(left, 28, POSTER, INK, "right")),
            holder(f"right-text-{r}", RIGHT + 22, y + 6, SIDE - 44, ROW_H - 12, label(right, 28, POSTER, INK, "left")),
            dict(type="line", id=f"tie-left-{r}", x=LEFT + SIDE, y=y + ROW_H / 2, points=[[0, 0], [MID - SIDE, 0]], strokeColor=TEAL_STRONG, strokeWidth=3),
            dict(type="line", id=f"tie-right-{r}", x=MID + PLATE_W, y=y + ROW_H / 2, points=[[0, 0], [RIGHT - MID - PLATE_W, 0]], strokeColor=BLUE_STRONG, strokeWidth=3),
            box(f"plate-{r}", MID, y, PLATE_W, ROW_H, PLATE, INK, label(measure, 20, CLEAN, WHITE)),
        ]
        y += ROW_H + ROW_GAP
    return s


if __name__ == "__main__":
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE
    render_scene(scene(), folder, NAME)
    missing = missing_from_scene(HEADER, ROWS, folder / f"{NAME}.excalidraw")
    if missing: print("missing:", missing)
    assert not missing
    print("wrote", folder)
