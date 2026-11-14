"""Draw the comment's Every measure table as a binder page of collectable stat cards.

Nine cards sit in a three by three grid: one grid row for each experiment row, one grid column for
each guidance value. Each card lists all ten measures. Today's default model is a grey reference card
below the grid.

Run:  python build_every-measure-stat-cards.py [output folder]
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[4] / "code" / "src"))
from excalidraw_table import missing_from_scene, plain, render_scene  # noqa: E402

NAME = "every-measure-stat-cards"
HEADER = ["Row", "Guidance", "Error on asks 1/24 to 17/24, with 95% interval", "Error on asks 18/24 to 23/24", "Win rate at asks 15/24 to 17/24", "Pooled win rate", "First-try legality", "Line-ups", "Novel line-ups against training teams", "Effective species", "Top-species share", "Elite line-ups, then those that held"]
ROWS = [
    ["Per-field, uncorrected t, the control", "2", "0.0084, 0.0045 to 0.0134", "0.1352", "0.497", "0.354", "100.0%", "1,513", "17.5%", "15.0", "0.136", "186, then 81"],
    ["Per-field, uncorrected t, the control", "4", "0.0049, 0.0022 to 0.0091", "0.1079", "0.532", "0.375", "100.0%", "1,918", "24.4%", "14.2", "0.143", "254, then 132"],
    ["Per-field, uncorrected t, the control", "8", "0.0033, 0.0014 to 0.0068", "0.0909", "0.556", "0.390", "100.0%", "2,126", "31.9%", "13.2", "0.147", "314, then 162"],
    ["Linear, species-first", "2", "0.0075, 0.0039 to 0.0126", "0.1521", "0.503", "0.353", "100.0%", "1,578", "18.3%", "14.9", "0.138", "172, then 64"],
    ["Linear, species-first", "4", "0.0049, 0.0022 to 0.0092", "0.1210", "0.527", "0.373", "100.0%", "1,971", "26.4%", "14.1", "0.143", "231, then 95"],
    ["Linear, species-first", "8", "0.0038, 0.0017 to 0.0074", "0.1087", "0.539", "0.385", "100.0%", "2,060", "35.6%", "13.1", "0.145", "230, then 127"],
    ["Linear, random order", "2", "0.0358, 0.0276 to 0.0447", "0.2879", "0.340", "0.247", "90.4%", "2,646", "87.8%", "20.0", "0.110", "31, then 7"],
    ["Linear, random order", "4", "0.0293, 0.0220 to 0.0377", "0.2529", "0.364", "0.266", "90.5%", "3,336", "88.3%", "19.5", "0.114", "79, then 17"],
    ["Linear, random order", "8", "0.0269, 0.0198 to 0.0352", "0.2492", "0.368", "0.272", "89.2%", "3,820", "90.2%", "19.0", "0.118", "91, then 28"],
    ["Today's default model, for reference", "2", "0.0053", "not read here", "0.515", "0.371", "100.0%", "1,437", "not comparable", "14.9", "0.135", "176, then 81"],
]

INK, MUTED = "#1e1e1e", "#495057"
# pale card fill, lightest tint for the stat block, strong shade for the banner and values
COLOURS = {"control": ("#a5d8ff", "#e7f5ff", "#1971c2"), "species-first": ("#96f2d7", "#e6fcf5", "#099268"),
           "random": ("#ffd8a8", "#fff4e6", "#e8590c"), "reference": ("#dee2e6", "#f8f9fa", "#495057")}
NUNITO, LILITA, COMIC = 6, 7, 8

LEFT, CARD_W, GAP = 104, 380, 22                  # cards start right of the ring holes
PAD, INNER = 16, 380 - 2 * 16
# measured heights of the stat name and value holders, label padding included, from the trial render
NAME_H = [59] + [35] * 9
VALUE_H, STAT_GAP, BANNER_H = 38, 6, 72


def colours_of(row):
    for word, colours in COLOURS.items():
        if word in row.lower(): return colours
    return COLOURS["reference"]


def holder(id, x, y, width, height, text, size, font, colour, align="left", vertical="top"):
    return dict(type="rectangle", id=id, x=x, y=y, width=width, height=height, strokeColor="transparent", backgroundColor="transparent",
                roughness=0, label=dict(text=text, fontSize=size, fontFamily=font, textAlign=align, verticalAlign=vertical, strokeColor=colour))


def card_height():
    return 30 + BANNER_H + 20 - 8 + sum(NAME_H) + 10 * (VALUE_H + STAT_GAP) + PAD + 14


def card(index, row, x, y, reference=False):
    pale, tint, strong = colours_of(row[0])
    cells = [plain(cell) for cell in row]
    height, k = card_height(), f"c{index}"
    out = [
        dict(type="rectangle", id=f"{k}-pocket", x=x - 8, y=y - 8, width=CARD_W + 16, height=height + 16, strokeColor="#adb5bd",
             backgroundColor="transparent", strokeStyle="dashed", strokeWidth=1, roughness=0, roundness={"type": 3}),
        dict(type="rectangle", id=f"{k}-card", x=x, y=y, width=CARD_W, height=height, strokeColor=strong, backgroundColor=pale,
             fillStyle="solid", strokeWidth=4, roughness=1, roundness={"type": 3}),
        dict(type="rectangle", id=f"{k}-tab", x=x + 14, y=y + 5, width=64, height=33, strokeColor=strong, backgroundColor=tint,
             fillStyle="solid", strokeWidth=1, roughness=0, roundness={"type": 3},
             label=dict(text=plain(HEADER[0]), fontSize=18, fontFamily=COMIC, strokeColor=INK)),
        dict(type="rectangle", id=f"{k}-banner", x=x + 10, y=y + 40, width=CARD_W - 120, height=BANNER_H, strokeColor=strong,
             backgroundColor=strong, fillStyle="solid", strokeWidth=2, roughness=1,
             label=dict(text=cells[0], fontSize=22, fontFamily=LILITA, textAlign="left", verticalAlign="middle", strokeColor="#ffffff")),
        dict(type="ellipse", id=f"{k}-badge", x=x + CARD_W - 104, y=y + 14, width=96, height=96, strokeColor=strong,
             backgroundColor="#ffffff", fillStyle="solid", strokeWidth=4, roughness=1),
        holder(f"{k}-badge-word", x + CARD_W - 104, y + 30, 96, 24, plain(HEADER[1]), 18, NUNITO, MUTED, "center"),
        holder(f"{k}-badge-value", x + CARD_W - 104, y + 52, 96, 44, cells[1], 40, LILITA, strong, "center"),
    ]
    top = y + 30 + BANNER_H + 20
    block_h = sum(NAME_H) + 10 * (VALUE_H + STAT_GAP) + PAD
    out.append(dict(type="rectangle", id=f"{k}-stats", x=x + 10, y=top - 8, width=CARD_W - 20, height=block_h, strokeColor=strong,
                    backgroundColor=tint, fillStyle="solid", strokeWidth=1, roughness=0, roundness={"type": 3}))
    for s in range(10):
        column = s + 2
        out.append(holder(f"{k}-name-{s}", x + PAD, top, INNER, NAME_H[s], plain(HEADER[column]), 18, NUNITO, MUTED))
        out.append(holder(f"{k}-value-{s}", x + PAD, top + NAME_H[s], INNER, VALUE_H, cells[column], 22, COMIC, INK, "right"))
        top += NAME_H[s] + VALUE_H + STAT_GAP
        if s < 9:
            out.append(dict(type="line", id=f"{k}-rule-{s}", x=x + PAD, y=top - STAT_GAP / 2, points=[[0, 0], [INNER, 0]],
                            strokeColor=strong, strokeWidth=1, strokeStyle="dotted", roughness=0))
    return out


def skeleton():
    height = card_height()
    grid_top = 150
    page_h = grid_top + 3 * (height + GAP) + 120 + height + 60
    out = [
        dict(type="rectangle", id="sleeve", x=0, y=0, width=1300, height=page_h, strokeColor="#868e96", backgroundColor="#f1f3f5",
             fillStyle="solid", strokeWidth=2, roughness=0, roundness={"type": 3}),
        dict(type="rectangle", id="spine", x=0, y=0, width=70, height=page_h, strokeColor="#343a40", backgroundColor="#495057",
             fillStyle="solid", strokeWidth=2, roughness=0),
    ]
    for hole in range(6):
        cy = page_h * (hole + 0.5) / 6
        out.append(dict(type="ellipse", id=f"hole-{hole}", x=17, y=cy - 18, width=36, height=36, strokeColor="#212529",
                        backgroundColor="#ffffff", fillStyle="solid", strokeWidth=3, roughness=0))
    out.append(dict(type="text", id="title", x=LEFT, y=28, text="Every measure", fontSize=44, fontFamily=LILITA, strokeColor=INK))
    out.append(holder("subtitle", LEFT, 86, 1170, 30, "One card for each row at each guidance. Rows run down the page, guidance runs across.",
                      20, NUNITO, MUTED))
    for i, row in enumerate(ROWS[:9]):
        r, c = divmod(i, 3)
        out += card(i, row, LEFT + c * (CARD_W + GAP), grid_top + r * (height + GAP))
    ref_top = grid_top + 3 * (height + GAP) + 80
    out.append(dict(type="line", id="divider", x=LEFT, y=ref_top - 50, points=[[0, 0], [3 * CARD_W + 2 * GAP, 0]],
                    strokeColor="#868e96", strokeWidth=2, strokeStyle="dashed", roughness=1))
    out += card(9, ROWS[9], LEFT, ref_top, reference=True)
    out.append(dict(type="rectangle", id="ref-stamp", x=LEFT + CARD_W + 60, y=ref_top + 60, width=300, height=70, strokeColor="#495057",
                    backgroundColor="transparent", strokeWidth=3, roughness=1, angle=-0.08,
                    label=dict(text="REFERENCE CARD", fontSize=30, fontFamily=LILITA, strokeColor="#495057")))
    return out


def main():
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE
    render_scene(skeleton(), folder, NAME)
    scene_path = folder / f"{NAME}.excalidraw"
    missing = missing_from_scene(HEADER, ROWS, scene_path)
    if missing: print("missing from the scene:", missing)
    assert not missing
    print(scene_path)


if __name__ == "__main__":
    main()
