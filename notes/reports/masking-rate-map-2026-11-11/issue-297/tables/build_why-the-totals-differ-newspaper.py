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
    ["Per-field, uncorrected t, the control", "At guidance 2, neighbouring asks draw the same training line-ups. Each line-up appears at 2.54 asks on average. At guidance 8 it appears at 1.46. The low asks fill with novel line-ups that appear at one ask only. The high asks narrow onto a few training line-ups", "At guidance 2, 2,216 of the 2,329 repeats also appear at the neighbouring ask. 45 of the 50 most widespread line-ups at guidance 2 are training line-ups. The 20 most widespread at guidance 2 take 239 ask slots there, 96 at guidance 4 and 34 at guidance 8. Novel line-ups number 502, 834 and 1,236 at guidance 2, 4 and 8. Of the 1,236, 1,130 appear at one ask. Training line-ups number 1,011, 1,084 and 890. At ask 21/24, the 192 teams have 151 distinct line-ups at guidance 2 and 58 at guidance 8. The most common line-up at that ask grows from 6 teams to 35"],
    ["Linear, species-first", "Its asks share line-ups the same way, and many of them are the control's line-ups. Its line-up total differs from the control's by +65, +53 and −66 at guidance 2, 4 and 8. At guidance 8 the gap is inside the control's seed-to-seed spread. At guidance 2 every seed of this row is above every seed of the control", "The two rows share 776 line-ups at guidance 2. At guidance 2 the control's seeds have 525, 543 and 534 line-ups, a spread of 18. This row's seeds have 573, 549 and 559, gaps of +48, +6 and +25. At guidance 8 the control's seeds have 783, 787 and 742, a spread of 45, and the gaps are −2, +2 and +20. Seed gaps do not add up to the pooled gap, because the line-up total counts a line-up once, however many seeds draw it. At guidance 8, line-ups repeat across this row's seeds 272 times, against 186 for the control. The extra repeats put its pooled total 66 below the control's, while its seed gaps run from −2 to +20. One difference is unexplained. At asks 15/24 to 23/24 under guidance 8, the control has 226 distinct line-ups among 1,728 teams, with 93% of those teams on training line-ups. This row has 345, with 77%. At guidance 2 the two have 379 and 459. The control's teams at those asks win 0.560, against 0.538 for this row. With one training run per masking schedule, this table cannot separate the masking schedule from training luck"],
    ["Linear, random order", "It uses the Linear, species-first checkpoint and rarely puts a training team's six species together. At guidance 2, the 192 teams of one ask have 191.7 distinct line-ups on average. Repeats between asks are 40% of its ask-by-ask count at guidance 2, against 61% for the control. Its total rises with guidance only because the repeat share falls, to 13% at guidance 8. Its ask-by-ask count stays between 4,398 and 4,408", "Every number here is at guidance 2, except the last two sentences. 12% of its teams have a training line-up, against 82% for Linear, species-first. Its 10 most used species make up 43% of the species it draws, against 64%. It draws 58 species that no training team has, in 24% of its teams. Linear, species-first draws no such species at any guidance. Its teams on a novel line-up win 0.236, against 0.323 for its teams on a training line-up. From guidance 2 to 8 its repeats fall from 1,762 to 578. Most of the lost repeats are novel line-ups, which fall from 1,486 to 479"],
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
MEASURED = {"head-0": 89, "stand-0": 290, "body-0-0": 62, "body-0-1": 62, "body-0-2": 87, "body-0-3": 62, "body-0-4": 36, "body-0-5": 62, "body-0-6": 87, "body-0-7": 62, "head-1": 50, "stand-1": 321, "body-1-0": 62, "body-1-1": 87, "body-1-2": 62, "body-1-3": 87, "body-1-4": 113, "body-1-5": 87, "body-1-6": 87, "body-1-7": 36, "body-1-8": 113, "body-1-9": 36, "body-1-10": 36, "body-1-11": 62, "body-1-12": 87, "head-2": 50, "stand-2": 445, "body-2-0": 62, "body-2-1": 62, "body-2-2": 62, "body-2-3": 62, "body-2-4": 62, "body-2-5": 87, "body-2-6": 62, "body-2-7": 62}


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
