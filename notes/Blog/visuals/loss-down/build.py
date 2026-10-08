"""Build loss-down.html from template.html + the repo's results (2026-10-08).

    python3 build.py      -> loss-down.html (publish that file)
Inputs: results/lossdown.json, results/lossdown_figdata.json, results/lossdown_battle.json (optional),
sprites from ../search-loop/sprites and ../item-sprites (embedded as data URIs).
"""
import base64, json, math
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = Path.home() / "Documents/code/vgc-team-generator-pilot/results"
R = json.load(open(RES / "lossdown.json"))
FIG = json.load(open(RES / "lossdown_figdata.json"))
OLD = "OLD asked_vs_got.pt (2026-10-04, best so far)"


def uri(p):
    return "data:image/png;base64," + base64.b64encode(Path(p).read_bytes()).decode()


def exact(k):
    return R[k]["test_exact_order"]["nll"]


def paired(a, b):
    x = [p - q for p, q in zip(R[a]["per_team_test_exact"], R[b]["per_team_test_exact"])]
    m = sum(x) / len(x); sd = math.sqrt(sum((v - m) ** 2 for v in x) / (len(x) - 1))
    return round(m, 4), round(sd / math.sqrt(len(x)), 4)


STEPS = [  # (results key, label, what changed)
    (OLD, "Old model", "asked_vs_got.pt, trained 2026-10-04 on 14,736 teams"),
    (OLD + " + structured output at eval only", "Rule out illegal values", "same old weights; impossible values masked when scoring"),
    ("A1_all_legal", "Retrain with the masks", "all 145k teams, masks in training"),
    ("B2_family", "Train on the test family", "only the 16k Aug–Sep loop teams + corpus"),
    ("B3_family_base", "Fix a training bias", "no forced mask when a draw hides nothing"),
    ("S1_family_species_first", "Reveal species first", "unmasking schedule 2,1,1,1,1"),
    ("S7_family_block_sharp", "Full block order", "species → ability/item → moves → nature"),
    ("F1_final_medium", "Medium model", "d256 × 6 layers, validation teams folded in"),
    ("ENS_F1_G2", "Mix in the matrix model", "F1 + G2 (trained on all 9,272 matrix teams)"),
]
stairs = []
for i, (k, label, note) in enumerate(STEPS):
    d = paired(k, STEPS[i - 1][0]) if i else (0.0, 0.0)
    stairs.append(dict(label=label, note=note, nll=round(exact(k), 4), delta=d[0], se=d[1]))

tried = [
    dict(label="Prime partial masking (2 sub-tokens per field)", ours=2.026, ref=1.782, unit="ELBO, vs the same model without it"),
    dict(label="Source tags as metadata", ours=1.838, ref=1.823, unit="ELBO, all-data model with vs without"),
    dict(label="Pretrain on all 146k, then fine-tune", ours=round(exact("G1_final_all_matrix"), 3), ref=round(exact("G2_family_matrix_noprtrain"), 3), unit="exact loss, vs no pretraining"),
    dict(label="Moves stored in popularity order", ours=round(exact("MO1_block_popular_moves"), 3), ref=round(exact("S5_family_block_order"), 3), unit="exact loss, vs alphabetical"),
    dict(label="No time input (RADD)", ours=round(exact("N1_family_notime"), 3), ref=round(exact("B3_family_base"), 3), unit="exact loss, vs with time input"),
    dict(label="Ensemble with small seeds", ours=1.484, ref=1.481, unit="own-order ELBO, vs F1 alone"),
]

m1 = R["M1_matrix_rows"]
matrix = dict(scalar=round(-m1["scalar_vs_uncond"]["diff"] * 48, 3), scalar_se=round(m1["scalar_vs_uncond"]["se"] * 48, 3),
              row=round(-m1["row_vs_scalar"]["diff"] * 48, 3), row_se=round(m1["row_vs_scalar"]["se"] * 48, 3))

team = FIG["team"]
import numpy as np
team["test_median"] = float(np.median(np.load(RES / "lossdata.npz")["wte"]))
sprites = {s: uri(HERE.parent / "search-loop/sprites" / f"{s}.png") for s in team["species_id"]}
items = {}
for st in team["walks"]["new"]:
    if st["field"] == "item":
        key = "".join(ch for ch in st["truth"].lower() if ch.isalnum())
        p = HERE.parent / "item-sprites" / f"item-{key}.png"
        if p.exists(): items[st["truth"]] = uri(p)

battle = None
bp = RES / "lossdown_battle.json"
if bp.exists():
    b = json.load(open(bp))
    if "summary" in b and all(s.get("got") is not None for s in b["summary"].values()):
        battle = [dict(cell=t, **{k: s[k] for k in ("asked", "guidance", "got", "se", "old_got", "old_se", "valid",
                                                     "old_valid", "species_sets", "old_sets", "copies", "battles")})
                  for t, s in b["summary"].items()]

READ = HERE / "battle_read.html"                       # written after the battles: the result in prose
data = dict(battleRead=READ.read_text() if READ.exists() else None, team=team, sprites=sprites, items=items, options=FIG["options"], stairs=stairs, tried=tried,
            matrix=matrix, battle=battle,
            head=dict(old=round(exact(OLD), 3), new=round(exact("ENS_F1_G2"), 3),
                      diff=paired("ENS_F1_G2", OLD)))
blob = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
html = (HERE / "template.html").read_text().replace("/*DATA*/null", blob)
(HERE / "loss-down.html").write_text(html)
print(f"loss-down.html {len(html)/1024:.0f} KB · battle data: {'yes' if battle else 'not yet'}")
