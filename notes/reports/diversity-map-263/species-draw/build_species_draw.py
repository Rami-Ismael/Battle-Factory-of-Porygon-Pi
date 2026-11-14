"""The ticket picture for the species-draw diversity run: what Pokémon 2's species draw sees under each decoding order,
and how temperature on species flattens the odds of the draw.

Run:  /tmp/vgc-pilot/.venv/bin/python notes/reports/diversity-map-263/species-draw/build_species_draw.py [folder]
"""
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[3] / "code" / "src"))
import excalidraw_table as drawn

INK, GREY, MASK = "#1e1e1e", "#495057", "#dee2e6"
FILLED, FILLED_EDGE = "#96f2d7", "#099268"          # a filled field
NOW, NOW_EDGE = "#ffd8a8", "#e8590c"                # the species being drawn
FONT, FONT_BOLD = 6, 7
COL, BOX_W, LEFT = 196, 176, 40
skeleton = []


def box(id_, x, y, w, h, text, fill, edge, size=20, colour=INK, dashed=False):
    skeleton.append(dict(type="rectangle", id=id_, x=x, y=y, width=w, height=h, backgroundColor=fill, fillStyle="solid",
                         strokeColor=edge, strokeWidth=2, strokeStyle="dashed" if dashed else "solid", roughness=0,
                         roundness=dict(type=3), label=dict(text=text, fontSize=size, fontFamily=FONT, strokeColor=colour)))


def text(id_, x, y, words, size=22, colour=INK, font=FONT):
    skeleton.append(dict(type="text", id=id_, x=x, y=y, text=words, fontSize=size, fontFamily=font, strokeColor=colour))


def team(prefix, top, set_one_filled):
    """Six Pokémon, each a species box over a set box, at the moment Pokémon 2's species is drawn."""
    for p in range(6):
        x = LEFT + p * COL
        text(f"{prefix}_p{p}", x + 8, top, f"Pokémon {p + 1}", size=20, colour=GREY)
        if p == 0: box(f"{prefix}_s{p}", x, top + 30, BOX_W, 56, "species filled", FILLED, FILLED_EDGE)
        elif p == 1: box(f"{prefix}_s{p}", x, top + 30, BOX_W, 56, "species drawn now", NOW, NOW_EDGE)
        else: box(f"{prefix}_s{p}", x, top + 30, BOX_W, 56, "masked", MASK, GREY, colour=GREY, dashed=True)
        if p == 0 and set_one_filled: box(f"{prefix}_t{p}", x, top + 96, BOX_W, 96, "ability, item, moves, Stat Alignment, Stat Points filled", FILLED, FILLED_EDGE, size=18)
        else: box(f"{prefix}_t{p}", x, top + 96, BOX_W, 96, "set masked", MASK, GREY, colour=GREY, dashed=True)


text("title", LEFT, 20, "What Pokémon 2's species draw sees, and how it draws", size=32, font=FONT_BOLD)
text("sub", LEFT, 66, "Default model, guidance 2. Each arm changes one of the two choices a pass makes: which field, or which value.", size=20, colour=GREY)

text("a_head", LEFT, 120, "Control: species-first order. Only Pokémon 1's species is visible", size=24, font=FONT_BOLD)
team("a", 160, set_one_filled=False)

text("b_head", LEFT, 400, "Arm 1, Pokémon-by-Pokémon order: Pokémon 1's whole set is visible too", size=24, font=FONT_BOLD)
team("b", 440, set_one_filled=True)

text("c_head", LEFT, 680, "Arms 2 and 3, temperature on species only: the same view, flatter odds", size=24, font=FONT_BOLD)
LOGITS = [3.0, 2.2, 1.6, 1.2, 0.9, 0.6, 0.4, 0.2]   # illustrative species logits, not the model's


def odds(temperature):
    weights = [math.exp(l / temperature) for l in LOGITS]
    return [w / sum(weights) for w in weights]


BASE, SCALE, BAR = 960, 520, 44
for g, (temperature, label) in enumerate([(1.0, "Temperature 1"), (1.5, "Temperature 1.5 on species")]):
    left = LEFT + 40 + g * 600
    for i, p in enumerate(odds(temperature)):
        h = p * SCALE
        skeleton.append(dict(type="rectangle", id=f"c_bar{g}{i}", x=left + i * (BAR + 14), y=BASE - h, width=BAR, height=h,
                             backgroundColor=NOW if g else FILLED, fillStyle="solid", strokeColor=NOW_EDGE if g else FILLED_EDGE,
                             strokeWidth=2, roughness=0))
    skeleton.append(dict(type="line", id=f"c_axis{g}", x=left - 10, y=BASE, points=[[0, 0], [8 * (BAR + 14) + 6, 0]], strokeColor=GREY, strokeWidth=2, roughness=0))
    text(f"c_label{g}", left, BASE + 12, label, size=22)
text("c_note", LEFT, BASE + 56, "Odds of eight species, made up to show the shape. Set fields stay at temperature 1.", size=20, colour=GREY)

if __name__ == "__main__":
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE
    drawn.render_scene(skeleton, folder, "species-draw")
    print("wrote", folder / "species-draw-light.png")
