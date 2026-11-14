"""Draw the 'why the line-up totals differ' table of issue #297 as a newspaper front page: one story per row.

The Row cell is each story's headline, the 'Why its total differs' cell its standfirst, and the Evidence cell its
body, one sentence to a paragraph. The first story carries the three column names as small section labels.

Run:  /tmp/vgc-pilot/.venv/bin/python build_why-the-totals-differ-newspaper.py [output folder]
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[4] / "code" / "src"))
from excalidraw_table import missing_from_scene, plain, render_scene, sentences  # noqa: E402

NAME = "why-the-totals-differ-newspaper"
HEADER = ["Row", "Why its total differs", "Evidence"]
ROWS = [
    ["Per-field, uncorrected t, the control", "At guidance 2 the asks blur together: neighbouring asks draw the same training line-ups, so one line-up is counted at many asks. Stronger guidance pins each line-up to a few asks. The low asks then fill with novel line-ups that are drawn at one ask only, and the high asks narrow onto a few training line-ups", "2,216 of the 2,329 repeats at guidance 2 are a line-up drawn at the next ask too, and 45 of the 50 most widespread line-ups are training line-ups. The 20 most widespread take 239 ask slots at guidance 2, 96 at guidance 4 and 34 at guidance 8. Novel line-ups go from 502 to 834 to 1,236, and 1,130 of the 1,236 are drawn at one ask. Training line-ups go from 1,011 to 890. At ask 21 the count falls from 151 to 58 and the most common line-up goes from 6 teams to 35"],
    ["Linear, species-first", "The same thing happens, on many of the same line-ups. Its gap to the control is small and mostly inside the spread between sampling seeds", "The two rows share 776 line-ups at guidance 2. The gap of +65 at guidance 2 is +48, +6 and +25 by sampling seed. The gap of −66 at guidance 8 is −2, +2 and +20 by seed, so it is noise. One difference stands out: at asks 15/24 to 23/24 under guidance 8 the control holds 226 line-ups, 93% of its teams on training line-ups, and this row holds 345, with 77%"],
    ["Linear, random order", "It is the same model as Linear, species-first, but it rarely puts a training team's six species together, so almost every team in an ask has its own line-up. Its total rises with guidance only because the few repeats between asks disappear", "12% of its teams have a training line-up at guidance 2, against 82% for species-first. Its 10 most used species take 43% of species slots, against 64%. It draws 58 species that appear nowhere in the training table, in 24% of its teams, and species-first draws none. Repeats fall from 1,762 to 578, and most of the lost repeats are novel line-ups: 1,486 to 479. Teams with a novel line-up win 0.236, against 0.323 for teams with a training line-up"],
]
PAPER_NAME = "The Line-up Gazette"

INK, GREY, PAPER, PAPER_EDGE = "#1e1e1e", "#495057", "#fbf5e6", "#868e96"
COLOURS = [("#a5d8ff", "#1971c2"), ("#96f2d7", "#099268"), ("#ffd8a8", "#e8590c")]   # control, species-first, random order
POSTER, CLEAN, MONO = 7, 6, 8

SHEET_W, MARGIN, GUTTER = 1260, 34, 42
COL_W = (SHEET_W - 2 * MARGIN - 2 * GUTTER) // 3
HEAD_SIZE, STAND_SIZE, BODY_SIZE, KICKER_SIZE = 34, 23, 19, 18
KICKER_H, PARA_GAP = 34, 9
# heights Excalidraw gave each wrapped text holder in a trial render, by id
MEASURED = {
    "head-0": 89, "head-1": 50, "head-2": 50,
    "stand-0": 321, "stand-1": 166, "stand-2": 259,
    "body-0-0": 113, "body-0-1": 87, "body-0-2": 87, "body-0-3": 36, "body-0-4": 87,
    "body-1-0": 62, "body-1-1": 62, "body-1-2": 62, "body-1-3": 139,
    "body-2-0": 87, "body-2-1": 62, "body-2-2": 87, "body-2-3": 87, "body-2-4": 87,
}


def holder(id_, x, y, w, text, size, family, colour=INK, align="left", h=None):
    return dict(type="rectangle", id=id_, x=x, y=y, width=w, height=h or MEASURED.get(id_, size * 1.25), strokeColor="transparent",
                backgroundColor="transparent", roughness=0,
                label=dict(text=text, fontSize=size, fontFamily=family, textAlign=align, verticalAlign="top", strokeColor=colour))


def line(id_, x, y, dx, dy, width=1, colour=INK, style="solid"):
    return dict(type="line", id=id_, x=x, y=y, points=[[0, 0], [dx, dy]], strokeColor=colour, strokeWidth=width, strokeStyle=style, roughness=0)


def skeleton():
    parts = []
    top = 28
    parts.append(holder("masthead", MARGIN, top, SHEET_W - 2 * MARGIN, PAPER_NAME, 76, POSTER, align="center", h=96))
    rule_y = top + 108
    parts.append(line("rule-thick", MARGIN, rule_y, SHEET_W - 2 * MARGIN, 0, 5))
    parts.append(line("rule-thin", MARGIN, rule_y + 10, SHEET_W - 2 * MARGIN, 0, 1))

    # every story shares the same bands, so the first story's section labels name the band for all three
    story_top = rule_y + 36
    head_y = story_top + 22 + KICKER_H
    head_h = max(MEASURED[f"head-{i}"] for i in range(3))
    stand_y = head_y + head_h + 10 + KICKER_H
    stand_h = max(MEASURED[f"stand-{i}"] for i in range(3))
    rule2_y = stand_y + stand_h + 14
    body_y = rule2_y + 18 + KICKER_H
    bottom = body_y
    stories = []
    for i, row in enumerate(ROWS):
        x = MARGIN + i * (COL_W + GUTTER)
        fill, strong = COLOURS[i]
        stories.append(dict(type="rectangle", id=f"bar-{i}", x=x, y=story_top, width=COL_W, height=14, backgroundColor=strong,
                            fillStyle="solid", strokeColor=strong, strokeWidth=1, roughness=0))
        stories.append(dict(type="rectangle", id=f"bar-tint-{i}", x=x, y=story_top + 14, width=COL_W, height=6, backgroundColor=fill,
                            fillStyle="solid", strokeColor="transparent", roughness=0))
        if i == 0:
            for id_, text, y in (("label-row", HEADER[0], head_y), ("label-why", HEADER[1], stand_y), ("label-evidence", HEADER[2], body_y)):
                stories.append(holder(id_, x, y - KICKER_H, COL_W, text.upper(), KICKER_SIZE, MONO, colour=GREY, h=KICKER_H))
        stories.append(holder(f"head-{i}", x, head_y, COL_W, plain(row[0]), HEAD_SIZE, POSTER))
        stories.append(holder(f"stand-{i}", x, stand_y, COL_W, plain(row[1]), STAND_SIZE, CLEAN, colour=strong))
        stories.append(line(f"rule-stand-{i}", x, rule2_y, COL_W, 0, 1, GREY, "dashed"))
        y = body_y
        for p, sentence in enumerate(sentences(plain(row[2]))):
            pid = f"body-{i}-{p}"
            stories.append(holder(pid, x, y, COL_W, sentence, BODY_SIZE, CLEAN))
            y += MEASURED.get(pid, 10) + PARA_GAP
        bottom = max(bottom, y)
    for g in range(2):
        gx = MARGIN + (g + 1) * COL_W + g * GUTTER + GUTTER / 2
        stories.append(line(f"column-rule-{g}", gx, story_top, 0, bottom - story_top - PARA_GAP, 1, PAPER_EDGE))
    sheet_h = bottom + 24
    parts[:0] = [
        dict(type="rectangle", id="sheet-shadow", x=8, y=10, width=SHEET_W, height=sheet_h, backgroundColor=PAPER_EDGE,
             fillStyle="solid", strokeColor="transparent", roughness=0, opacity=35),
        dict(type="rectangle", id="sheet", x=0, y=0, width=SHEET_W, height=sheet_h, backgroundColor=PAPER, fillStyle="solid",
             strokeColor=PAPER_EDGE, strokeWidth=1, roughness=1),
    ]
    return parts + stories


if __name__ == "__main__":
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE
    written = render_scene(skeleton(), folder, NAME)
    for path in written: print(path)
    missing = missing_from_scene(HEADER, ROWS, written[0])
    if missing: print("missing:", missing)
    assert not missing
