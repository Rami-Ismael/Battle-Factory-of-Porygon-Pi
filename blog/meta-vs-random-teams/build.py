"""Build index.html for "Why meta teams beat random teams".

Reads the random-fill baseline results on the laptop and the EDA outputs in code/src/meta_vs_random_eda/out
(run code/src/meta_vs_random_eda/run.sh first), embeds sprites and item icons as data URIs, and fills template.html.
Run from the repository root:  python3 blog/meta-vs-random-teams/build.py
"""
import base64, json, re, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SPR = ROOT / "notes/Blog/visuals/search-loop/sprites"
ITM = ROOT / "notes/Blog/visuals/item-sprites"
EDA = ROOT / "code/src/meta_vs_random_eda"
PILOT = Path("/Users/ramiismael/Documents/code/vgc-team-generator-pilot/results")
sys.argv = [sys.argv[0], str(EDA / "out")]
sys.path.insert(0, str(EDA))
from mcat import parse, cat, CATS  # noqa: E402  (mcat reads the fill results on import)

ident = lambda s: re.sub(r"[^a-z0-9]", "", s.lower())
uri = lambda p: "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode()

fill = json.load(open(PILOT / "random_fill_baseline.json"))
tasks = {t["start_team"]: t["team_paste"] for t in json.load(open(PILOT / "random_fill_tasks.json"))["tasks"]}
orig = {int(k): v for k, v in fill["original"].items()}
SHOW = ["ability6", "stats6", "alignment6", "item6", "move24"] + [f"pokemon{k}" for k in range(1, 7)]


def sprite(species):
    s = ident(species)
    for cand in (s, s.replace("mega", "").rstrip("xy") if "mega" in s else s, ident(species.split("-")[0])):
        if (SPR / f"{cand}.png").exists(): return cand
    return None


def team(paste):
    out = []
    for m in parse(paste):
        out.append(dict(species=m["species"], item=m["item"], ability=m["ability"], nature=m["nature"],
                        points=" / ".join(f"{v} {k}" for k, v in m["evs"].items() if v),
                        sprite=sprite(m["species"]), moves=[dict(name=mv, kind=cat(mv, m)) for mv in m["moves"]]))
    return out


# the starting team: every shown fill has all sprites, original near the 43-team mean, steady drop with k
cands = []
for t in orig:
    fs = {f"{f['task']}{f['k']}": f for f in fill["fills"] if f["start_team"] == t and f.get("score") is not None}
    if not all(k in fs for k in SHOW): continue
    teams = [team(tasks[t])] + [team(fs[k]["paste"]) for k in SHOW]
    if any(m["sprite"] is None for tm in teams for m in tm): continue
    ks = [fs[f"pokemon{k}"]["score"] for k in range(1, 7)]
    cands.append((abs(orig[t] - 0.446) + 0.5 * sum(max(0, b - a) for a, b in zip(ks, ks[1:])), t, fs))
_, START, fs = min(cands)

summ = fill["summary"]
out = json.load(open(EDA / "out/c_moves.json")); sp = json.load(open(EDA / "out/d_species.json")); al = json.load(open(EDA / "out/e_all.json"))
data = dict(
    start=START, real=dict(team=team(tasks[START]), score=orig[START]),
    fills={k: dict(team=team(fs[k]["paste"]), score=fs[k]["score"], mean=summ[k]["filled"], ci=summ[k]["ci"]) for k in SHOW},
    real_mean=sum(orig.values()) / len(orig), n_start=len(orig),
    move_value={c: out["cost"][c]["value"] for c in CATS[:6]}, mix=out["mix"],
    by_k=sp["by_k"], spread=al["spread_curve"], protect=al["protect_curve"], n_all=al["n"],
    sources=dict(fills="random_fill_baseline.json (results/, vgc-team-generator-pilot)",
                 values="code/src/meta_vs_random_eda/out/c_moves.json", by_k="code/src/meta_vs_random_eda/out/d_species.json",
                 curves="code/src/meta_vs_random_eda/out/e_all.json"))
names = {m["sprite"] for tm in [data["real"]["team"]] + [f["team"] for f in data["fills"].values()] for m in tm}
items = {m["item"] for tm in [data["real"]["team"]] + [f["team"] for f in data["fills"].values()] for m in tm if m["item"]}
data["sprites"] = {n: uri(SPR / f"{n}.png") for n in names}
data["items"] = {i: uri(ITM / f"item-{ident(i)}.png") for i in items if (ITM / f"item-{ident(i)}.png").exists()}
(HERE / "index.html").write_text((HERE / "template.html").read_text().replace("/*DATA*/null", json.dumps(data)))
print(f"start team {START} ({orig[START]:.3f}); {len(names)} sprites, {len(data['items'])}/{len(items)} item icons")
