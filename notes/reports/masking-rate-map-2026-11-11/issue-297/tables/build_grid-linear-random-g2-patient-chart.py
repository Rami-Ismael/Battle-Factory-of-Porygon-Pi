"""Redraw the guidance 2 grid table for Linear, random order as a hospital chart on a clipboard.

Run:  /tmp/vgc-pilot/.venv/bin/python build_grid-linear-random-g2-patient-chart.py [output folder]
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[4] / "code" / "src"))
from excalidraw_table import missing_from_scene, plain, render_scene  # noqa: E402

NAME = "grid-linear-random-g2-patient-chart"
HEADER = ["Measure", "Linear, random order, guidance 2", "The control at guidance 2"]
ROWS = [
    ["Ask-versus-got error, asks 1/24 to 17/24, with its 95% interval", "0.0358 (0.0276 to 0.0447)", "0.0084 (0.0045 to 0.0134)"],
    ["The same error for sampling seeds 1, 2 and 3", "0.0355, 0.0369, 0.0352", "0.0109, 0.0067, 0.0079"],
    ["Error on asks 18/24 to 23/24", "0.2879", "0.1352"],
    ["Win rate at asks 15/24 to 17/24", "0.340 (0.304 to 0.378)", "0.497 (0.458 to 0.538)"],
    ["Win rate over all 23 asks", "0.247 (0.224 to 0.271)", "0.354 (0.328 to 0.381)"],
    ["First-try legality", "90.4%", "100.0%"],
    ["Distinct line-ups", "2,646", "1,513"],
    ["Teams with a line-up absent from the training teams", "87.8%", "17.5%"],
    ["Teams with a line-up absent from earlier search runs and tournament records", "97.5%", "82.7%"],
    ["Effective species and top-species share, in 5-team subsets", "20.0 and 0.110", "15.0 and 0.136"],
    ["Teams and battles", "4,416 and 216,384", "4,416 and 216,384"],
]

INK, BROWN, BROWN_DARK, SHEET, RULE = "#1e1e1e", "#a9713b", "#6b4220", "#fff9db", "#a5d8ff"
METAL, METAL_DARK = "#ced4da", "#495057"
ORANGE, ORANGE_STRONG, BLUE, BLUE_STRONG = "#ffd8a8", "#e8590c", "#a5d8ff", "#1971c2"
NUNITO, HAND, POSTER = 6, 5, 7

BOARD_X, BOARD_W = 20, 1200
SHEET_X, SHEET_W = 60, 1120
MEASURE_X, MEASURE_W = 84, 462
PATIENT_X, REFERENCE_X, VALUE_W = 560, 868, 292
HEAD_Y, HEAD_H = 330, 96
ROW_Y, ROW_H, PILL_H = HEAD_Y + HEAD_H + 16, 78, 56


def holder(id_, x, y, width, height, text, size, font=NUNITO, align="left", colour=INK, vertical="middle"):
    return {"type": "rectangle", "id": id_, "x": x, "y": y, "width": width, "height": height, "strokeColor": "transparent",
            "backgroundColor": "transparent", "roughness": 0,
            "label": {"text": text, "fontSize": size, "fontFamily": font, "textAlign": align, "verticalAlign": vertical, "strokeColor": colour}}


def pill(id_, x, y, text, fill, stroke, size=20, height=PILL_H, font=NUNITO, width=VALUE_W):
    return {"type": "rectangle", "id": id_, "x": x, "y": y, "width": width, "height": height, "backgroundColor": fill, "fillStyle": "solid",
            "strokeColor": stroke, "strokeWidth": 2, "roughness": 1, "roundness": {"type": 3},
            "label": {"text": text, "fontSize": size, "fontFamily": font, "textAlign": "center", "verticalAlign": "middle", "strokeColor": INK}}


def heartbeat(x, y, width):
    """A flat trace with a beat every 200 units."""
    points, at = [[0, 0]], 0
    while at + 200 <= width:
        points += [[at + 90, 0], [at + 105, -14], [at + 118, 10], [at + 130, -40], [at + 144, 26], [at + 156, 0], [at + 170, -10], [at + 184, 0]]
        at += 200
    points.append([width, 0])
    return {"type": "line", "id": "heartbeat", "x": x, "y": y, "points": points, "strokeColor": ORANGE_STRONG, "strokeWidth": 3, "roughness": 1}


def skeleton():
    rows = [[plain(cell) for cell in row] for row in ROWS]
    header = [plain(cell) for cell in HEADER]
    bottom = ROW_Y + ROW_H * len(rows) + 40
    parts = [
        {"type": "rectangle", "id": "board", "x": BOARD_X, "y": 40, "width": BOARD_W, "height": bottom + 50 - 40, "backgroundColor": BROWN,
         "fillStyle": "solid", "strokeColor": BROWN_DARK, "strokeWidth": 3, "roughness": 1, "roundness": {"type": 3}},
        {"type": "rectangle", "id": "sheet", "x": SHEET_X, "y": 96, "width": SHEET_W, "height": bottom - 96, "backgroundColor": SHEET,
         "fillStyle": "solid", "strokeColor": "#868e96", "strokeWidth": 1, "roughness": 1},
        {"type": "rectangle", "id": "clip", "x": 470, "y": 0, "width": 300, "height": 120, "backgroundColor": METAL, "fillStyle": "solid",
         "strokeColor": METAL_DARK, "strokeWidth": 3, "roughness": 1, "roundness": {"type": 3}},
        {"type": "ellipse", "id": "clip-hole", "x": 590, "y": 16, "width": 60, "height": 30, "backgroundColor": BROWN_DARK, "fillStyle": "solid",
         "strokeColor": METAL_DARK, "strokeWidth": 2, "roughness": 1},
        {"type": "rectangle", "id": "clip-bar", "x": 500, "y": 70, "width": 240, "height": 26, "backgroundColor": "#adb5bd", "fillStyle": "solid",
         "strokeColor": METAL_DARK, "strokeWidth": 2, "roughness": 1, "roundness": {"type": 3}},
        holder("title", SHEET_X + 30, 140, SHEET_W - 60, 56, "PATIENT CHART", 46, POSTER, "center"),
        holder("subtitle", SHEET_X + 30, 200, SHEET_W - 60, 34, "Vital signs, one per line, beside the reference", 22, HAND, "center"),
        heartbeat(SHEET_X + 30, 290, SHEET_W - 60),
        holder("tag-measure", MEASURE_X, HEAD_Y - 2, MEASURE_W, 30, "VITAL SIGN", 18, POSTER, "left", "#495057"),
        holder("head-measure", MEASURE_X, HEAD_Y + 30, MEASURE_W, 56, header[0], 30, HAND, "left"),
        holder("tag-patient", PATIENT_X, HEAD_Y - 2, VALUE_W, 30, "PATIENT", 18, POSTER, "center", ORANGE_STRONG),
        pill("head-patient", PATIENT_X, HEAD_Y + 30, header[1], ORANGE_STRONG, ORANGE_STRONG, 22, 62, HAND),
        holder("tag-reference", REFERENCE_X, HEAD_Y - 2, VALUE_W, 30, "REFERENCE", 18, POSTER, "center", BLUE_STRONG),
        pill("head-reference", REFERENCE_X, HEAD_Y + 30, header[2], BLUE_STRONG, BLUE_STRONG, 22, 62, HAND),
        {"type": "line", "id": "rule-head", "x": SHEET_X + 10, "y": ROW_Y - 6, "points": [[0, 0], [SHEET_W - 20, 0]],
         "strokeColor": "#e03131", "strokeWidth": 2, "roughness": 0},
    ]
    parts[11]["label"]["strokeColor"] = "#ffffff"
    parts[13]["label"]["strokeColor"] = "#ffffff"
    for index, (measure, patient, reference) in enumerate(rows):
        top = ROW_Y + ROW_H * index
        parts.append(holder(f"measure-{index}", MEASURE_X, top + 4, MEASURE_W, ROW_H - 8, measure, 20))
        parts.append(pill(f"patient-{index}", PATIENT_X, top + (ROW_H - PILL_H) / 2, patient, ORANGE, ORANGE_STRONG))
        parts.append(pill(f"reference-{index}", REFERENCE_X, top + (ROW_H - PILL_H) / 2, reference, BLUE, BLUE_STRONG))
        if index < len(rows) - 1:
            parts.append({"type": "line", "id": f"rule-{index}", "x": SHEET_X + 10, "y": top + ROW_H, "points": [[0, 0], [SHEET_W - 20, 0]],
                          "strokeColor": RULE, "strokeWidth": 1, "roughness": 0})
    return parts


if __name__ == "__main__":
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE
    written = render_scene(skeleton(), folder, NAME)
    for path in written: print(path)
    missing = missing_from_scene(HEADER, ROWS, written[0])
    if missing: print("missing:", missing)
    assert not missing
