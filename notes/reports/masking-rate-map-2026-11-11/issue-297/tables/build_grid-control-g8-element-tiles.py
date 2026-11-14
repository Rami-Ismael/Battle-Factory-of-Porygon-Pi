"""Redraw the control grid at guidance 8 as tiles from a periodic table: one element tile per measure.

Run: /tmp/vgc-pilot/.venv/bin/python build_grid-control-g8-element-tiles.py [output folder]
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[4] / "code" / "src"))
from excalidraw_table import missing_from_scene, plain, render_scene  # noqa: E402

NAME = "grid-control-g8-element-tiles"
HEADER = ["Measure", "Per-field, uncorrected t, guidance 8"]
ROWS = [
	["Ask-versus-got error, asks 1/24 to 17/24, with its 95% interval", "0.0033 (0.0014 to 0.0068)"],
	["The same error for sampling seeds 1, 2 and 3", "0.0034, 0.0036, 0.0032"],
	["Error on asks 18/24 to 23/24", "0.0909"],
	["Win rate at asks 15/24 to 17/24", "0.556 (0.518 to 0.594)"],
	["Win rate over all 23 asks", "0.390 (0.364 to 0.418)"],
	["First-try legality", "100.0%"],
	["Distinct line-ups", "2,126"],
	["Teams with a line-up absent from the training teams", "31.9%"],
	["Teams with a line-up absent from earlier search runs and tournament records", "86.2%"],
	["Effective species and top-species share, in 5-team subsets", "13.2 and 0.147"],
	["Teams and battles", "4,416 and 216,384"],
]
# the kind of each measure, in table order, and the shade of blue each kind wears: fill, text colour
KINDS = ["error", "error", "error", "win rate", "win rate", "legality", "line-ups", "line-ups", "line-ups", "species", "counts"]
SHADES = {"error": ("#1971c2", "#ffffff"), "win rate": ("#80c4fc", "#03203d"), "legality": ("#a5d8ff", "#0b3d6e"),
	"line-ups": ("#d0ebff", "#0b3d6e"), "species": ("#d0ebff", "#0b3d6e"), "counts": ("#e7f5ff", "#0b3d6e")}
EDGE, INK = "#1864ab", "#1e1e1e"
W, H, GAP, TOP = 290, 330, 14, 90
# (column, row) of each tile: four to a row in table order, and the key in the last free place
PLACES = [(column, row) for row in range(3) for column in range(4)][:11]
KEY_PLACE = (3, 2)


def holder(id_, x, y, width, height, text, size, font, colour, align="center", vertical="middle"):
	return {"type": "rectangle", "id": id_, "x": x, "y": y, "width": width, "height": height, "strokeColor": "transparent",
		"backgroundColor": "transparent", "roughness": 0,
		"label": {"text": text, "fontSize": size, "fontFamily": font, "textAlign": align, "verticalAlign": vertical, "strokeColor": colour}}


def split_value(value):
	"""The big part and the interval under it; a value without an interval stays whole."""
	if " (" in value:
		big, rest = value.split(" (", 1)
		return big, "(" + rest
	return value, None


def tile(n, column, row, measure, value, kind):
	x, y = column * (W + GAP), TOP + row * (H + GAP)
	fill, text = SHADES[kind]
	big, small = split_value(plain(value))
	size = 50 if len(big) <= 6 else 38 if len(big) <= 14 else 32
	parts = [{"type": "rectangle", "id": f"tile-{n}", "x": x, "y": y, "width": W, "height": H, "backgroundColor": fill, "fillStyle": "solid",
			"strokeColor": EDGE, "strokeWidth": 2, "roughness": 1},
		{"type": "text", "id": f"number-{n}", "x": x + 14, "y": y + 10, "text": str(n), "fontSize": 26, "fontFamily": 7, "strokeColor": text},
		holder(f"kind-{n}", x + W - 174, y + 12, 160, 26, kind, 18, 6, text, align="right", vertical="top"),
		holder(f"value-{n}", x + 14, y + 52, W - 28, 96 if small else 128, big, size, 7, text)]
	if small:
		parts.append(holder(f"interval-{n}", x + 14, y + 148, W - 28, 30, small, 22, 6, text))
	parts.append({"type": "line", "id": f"rule-{n}", "x": x + 24, "y": y + 192, "points": [[0, 0], [W - 48, 0]], "strokeColor": text, "strokeWidth": 1, "roughness": 0, "opacity": 60})
	parts.append(holder(f"measure-{n}", x + 16, y + 202, W - 32, 110, plain(measure), 20, 6, text, vertical="top"))
	return parts


def skeleton():
	elements = [{"type": "text", "id": "title", "x": 0, "y": 0, "text": "The control grid as a periodic table", "fontSize": 40, "fontFamily": 7, "strokeColor": EDGE},
		{"type": "text", "id": "subtitle", "x": 2, "y": 50, "text": "Each tile is one row of the table. Its number is the row's place.", "fontSize": 22, "fontFamily": 6, "strokeColor": INK}]
	x, width, top = KEY_PLACE[0] * (W + GAP), W, TOP + KEY_PLACE[1] * (H + GAP)
	elements += [{"type": "rectangle", "id": "key", "x": x, "y": top, "width": width, "height": H, "backgroundColor": "transparent",
			"strokeColor": EDGE, "strokeWidth": 3, "strokeStyle": "dashed", "roughness": 1},
		holder("key-number", x + 14, top + 12, 120, 26, "row", 18, 6, EDGE, align="left", vertical="top"),
		holder("key-kind", x + width - 174, top + 12, 160, 26, "kind", 18, 6, EDGE, align="right", vertical="top"),
		holder("key-value", x + 14, top + 46, width - 28, 140, plain(HEADER[1]), 30, 7, EDGE),
		{"type": "line", "id": "key-rule", "x": x + 24, "y": top + 192, "points": [[0, 0], [width - 48, 0]], "strokeColor": EDGE, "strokeWidth": 1, "roughness": 0},
		holder("key-measure", x + 14, top + 202, width - 28, 60, plain(HEADER[0]), 30, 7, EDGE, vertical="top")]
	for n, ((column, row), (measure, value), kind) in enumerate(zip(PLACES, ROWS, KINDS, strict=True), start=1):
		elements += tile(n, column, row, measure, value, kind)
	return elements


if __name__ == "__main__":
	folder = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE
	render_scene(skeleton(), folder, NAME)
	scene_path = folder / f"{NAME}.excalidraw"
	missing = missing_from_scene(HEADER, ROWS, scene_path)
	if missing: print("missing:", missing)
	assert not missing
	print("wrote", scene_path)
