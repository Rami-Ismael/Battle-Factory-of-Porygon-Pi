"""Build index.html of the 2026-11-11 diversity page from ~/vgc-data/results/diversity_vs_training_2026-11-11.json.

    python3 build.py          (any Python 3; reads the JSON, writes index.html beside this file)

Charts are ECharts 6.1.0 drawn in the page from the embedded numbers; template.html holds the words.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULT = Path.home() / "vgc-data/results/diversity_vs_training_2026-11-11.json"
SEEDS = (1, 2, 3)
ROW_NAMES = {
    **{f"model_s{s}": f"default model, sampling seed {s}" for s in SEEDS},
    **{f"train_s{s}": f"training teams matched to the asks, draw {s}" for s in SEEDS},
    **{f"train_all_s{s}": f"training teams drawn over all win rates, draw {s}" for s in SEEDS},
    "val": "validation teams", "copied": "one team repeated", "random": "random legal teams",
}
ROW_ORDER = [f"model_s{s}" for s in SEEDS] + [f"train_s{s}" for s in SEEDS] + [f"train_all_s{s}" for s in SEEDS] + [
    "val", "copied", "random"]
# Figure 1: (level, key, reader name)
RATIO_ROWS = [
    ("Line-up", "distinct_lineups", "Distinct line-ups"), ("Line-up", "effective_lineups", "Effective line-ups"),
    ("Line-up", "distinct_species", "Distinct species"), ("Line-up", "effective_species", "Effective species"),
    ("Line-up", "distinct_pairs", "Distinct species pairs"), ("Line-up", "vendi_lineup", "Vendi Score, line-ups"),
    ("Builds", "distinct_sets", "Distinct builds"), ("Builds", "effective_sets", "Effective builds"),
    ("Builds", "sets_per_species", "Builds per species, all uses"),
    ("Builds", "builds_per_species", "Builds per species at 30 uses"),
    ("Builds", "vendi_sets", "Vendi Score, builds"), ("Builds", "distance_sets", "Mean distance, builds"),
    ("Stat Points", "distinct_statpoints", "Distinct Stat Points"),
    ("Stat Points", "effective_statpoints", "Effective Stat Points"),
    ("Stat Points", "vendi_statpoints", "Vendi Score, Stat Points"),
    ("Stat Points", "distance_statpoints", "Mean distance, Stat Points"),
    ("Whole team", "distinct_teams84", "Distinct teams, 84 fields"), ("Whole team", "vendi_team", "Vendi Score, whole team"),
    ("Whole team", "dcscore_team", "DCScore at temperature 0.1"), ("Whole team", "distance_team", "Mean distance, whole team"),
    ("Whole team", "nearest_mean", "Mean nearest training distance"),
]
ARCHETYPES = ["sun", "rain", "sand", "snow", "electric terrain", "grassy terrain", "misty terrain", "psychic terrain",
              "Trick Room", "Tailwind", "none"]


def mean(xs): return sum(xs) / len(xs)


def build():
    r = json.loads(RESULT.read_text())
    M, F, V, VA = r["measures"], r["floor"], r["verdict"], r["verdict_all_win_rates"]
    ratio = []
    for level, key, name in RATIO_ROWS:
        model = [M[f"model_s{s}"][key] for s in SEEDS]; train = [M[f"train_s{s}"][key] for s in SEEDS]
        tall = [M[f"train_all_s{s}"][key] for s in SEEDS]; tm = mean(train)
        ratio.append(dict(level=level, key=key, name=name, model=model, train=train, train_all=tall, tmean=tm,
                          lo=1 + F[key]["lo"] / tm, hi=1 + F[key]["hi"] / tm, val=M["val"][key], copied=M["copied"][key],
                          random=M["random"][key], differs=V[key]["differs"], differs_all=VA[key]["differs"],
                          ordered=V[key].get("anchors_ordered")))
    rows = [dict(key=k, name=ROW_NAMES[k], **{m: M[k][m] for m in M[k] if not isinstance(M[k][m], list)}) for k in ROW_ORDER]
    data = dict(n=r["n"], equal_uses=r["equal_uses"], ratio=ratio, rows=rows, archetypes=ARCHETYPES,
                floor={k: F[k] for k in F}, verdict=V, verdict_all=VA, shared=r["builds_shared_species"],
                vocab=r["setter_vocabulary"], sizes=r["sizes"])
    page = (HERE / "template.html").read_text().replace("/*DATA*/null", json.dumps(data, separators=(",", ":")))
    (HERE / "index.html").write_text(page)
    print(f"wrote {HERE / 'index.html'}")


if __name__ == "__main__":
    build()
