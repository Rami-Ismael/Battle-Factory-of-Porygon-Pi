"""Build the label-precision page: results JSON + sprites -> index.html (2026-10-16).

    /tmp/vgc-pilot/.venv/bin/python build.py
Reads results/label_precision.json, results/label_precision_sweep.json and the two reference sweeps from the code
repo; embeds only the sprites the page shows. Template: template.html (placeholders /*DATA*/ and /*SPRITES*/).
"""
import base64, json, math, sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, "/tmp/vgc-pilot/src")
import diversity_loops as DL
from corpus import parse_team_text, norm

HERE = Path(__file__).resolve().parent
RES = Path("/Users/ramiismael/Documents/code/vgc-team-generator-pilot/results")
SPRITES = HERE.parent / "search-loop" / "sprites"
MODELS = {  # key: (sweep file, model name in it, label)
    "v3_original": (RES / "ask_sweep_dbbo.json", "v3_original", "No fine-tune (v3)"),
    "v3_precise": (RES / "ask_sweep_precise.json", "v3_precise", "1,105 re-battled teams (earlier run)"),
    "lp_precise": (RES / "label_precision_sweep.json", "lp_precise", "Precise teams, precise labels"),
    "lp_noisy": (RES / "label_precision_sweep.json", "lp_noisy", "Same teams, 49-battle labels"),
    "lp_all": (RES / "label_precision_sweep.json", "lp_all", "Every team, labels as recorded"),
}


def species(p): return [norm(s["species"]) for s in parse_team_text(p)]


def main():
    lp = json.load(open(RES / "label_precision.json"))
    pre = [r for r in lp["teams"] if r["battles"] >= 192]
    need = set()
    scatter = []
    for r in pre:
        sp = species(r["paste"]); need |= set(sp)
        scatter.append([round(r["y"], 4), round(r["y_one"], 4), r["battles"], sp])
    hist = lambda ys: [sum(1 for y in ys if i / 40 <= y < (i + 1) / 40 or (i == 39 and y == 1)) for i in range(40)]
    curves, teams = {}, {}
    for key, (f, m, label) in MODELS.items():
        if not f.exists(): continue
        o = json.load(open(f)); pts = []
        for a in o["config"]["asks"]:
            c = o["cells"].get(f"{m}_y{a}")
            if not c or "y" not in c: continue
            ys = [y for y in c["y"] if y is not None]
            se = (sum((y - sum(ys) / len(ys)) ** 2 for y in ys) / (len(ys) - 1)) ** .5 / math.sqrt(len(ys))
            comps = Counter(DL.comp_paste(p) for p in c["pastes"])
            pts.append(dict(ask=a, mean=sum(ys) / len(ys), se=se, n=len(ys), valid=c["valid"], sets=len(comps)))
            if abs(a - 0.6) < 1e-9:
                by = {}
                for p, y in zip(c["pastes"], c["y"]):
                    if y is not None: by.setdefault(DL.comp_paste(p), []).append((y, species(p)))
                top = sorted(by.items(), key=lambda kv: (-len(kv[1]), -sum(y for y, _ in kv[1])))[:3]
                teams[key] = [dict(n=len(v), mean=sum(y for y, _ in v) / len(v), sp=v[0][1]) for _, v in top]
                for t in teams[key]: need |= set(t["sp"])
        if pts: curves[key] = dict(label=label, points=pts)
    summ = lp.get("summary", {})
    data = dict(noise=lp["noise"], n_precise=len(pre), n_all=len(lp["teams"]), train=lp.get("train"),
                hist_all=hist([r["y"] for r in lp["teams"]]), hist_precise=hist([r["y"] for r in pre]),
                scatter=scatter, curves=curves, teams=teams, contrasts=summ.get("contrasts", {}),
                summary={k: dict(top=v["top_mean"], slope=v["dial_slope"]) for k, v in summ.get("curves", {}).items()},
                partial={k: len(v["points"]) for k, v in curves.items() if len(v["points"]) < 9})
    sprites = {s: "data:image/png;base64," + base64.b64encode((SPRITES / f"{s}.png").read_bytes()).decode()
               for s in sorted(need) if (SPRITES / f"{s}.png").exists()}
    html = (HERE / "template.html").read_text()
    html = html.replace("/*DATA*/null", json.dumps(data, separators=(",", ":"))).replace("/*SPRITES*/{}", json.dumps(sprites))
    (HERE / "index.html").write_text(html)
    print(f"index.html: {len(html) / 1e6:.2f} MB · curves {list(curves)} · sprites {len(sprites)}/{len(need)} "
          f"(missing {sorted(need - set(sprites))[:12]})")


if __name__ == "__main__":
    main()
