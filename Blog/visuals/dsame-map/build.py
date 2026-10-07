"""Build elites-on-a-map.html for the deep-surrogate MAP-Elites experiment
(repo: docs/dsame-experiment.md, scripts/dsame_experiment.py, results/dsame_experiment.json).

The inner-loop replay is REAL: the experiment's own surrogate (trained on the 328 starting labels, plus
pilot turns when the pilot has run), its legality-checked mutation operators and its insert rule. Until the
pilot has run, battle length has no data, so the turn axis is an illustrative proxy and the page says so.
Results sections fill in from results/dsame_experiment.json as stages finish.

Run with the experiment's environment:  /tmp/vgc-pilot/.venv/bin/python build.py
"""
import base64, copy, importlib.util, json, math, random, re, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = Path("/Users/ramiismael/Documents/code/vgc-team-generator-pilot")
SPR = HERE.parent / "search-loop/sprites"
spec = importlib.util.spec_from_file_location("dsx", REPO / "scripts/dsame_experiment.py")
X = importlib.util.module_from_spec(spec); spec.loader.exec_module(X)
from corpus import parse_team_text, norm

OPS = {"A_move_swap": "move swap", "B_ability_swap": "ability swap", "C_item_swap": "item swap",
       "D_alignment_swap": "alignment swap", "E_spread_copy": "Stat Point spread copy", "F_candidate_copy": "whole-Pokémon copy"}
res = json.load(open(X.OUT)) if X.OUT.exists() else {}
pilot = res.get("pilot")
ctx = X.Ctx()
start_t, start_y = ctx.start, ctx.start_y
ratio = [X.ob_ratio(t) for t in start_t]
if pilot:
    turns_lab = [t if t is not None else np.nan for t in pilot["turns"]]; edges = pilot["edges"]; turns_real = True
else:                                    # no battle-length data yet: illustrative proxy, labelled on the page
    turns_lab = [np.nan] * len(start_t)
    turns_real = False
from corpus import dex_entry
def speed_proxy(team):                   # stand-in for battle length before the pilot: faster teams -> shorter battles
    return 14.0 - float(np.mean([dex_entry(s["species"])["baseStats"]["spe"] for s in team])) / 20.0
if not turns_real:
    edges = {"ratio": np.linspace(*np.percentile(ratio, [1, 99]), X.GRID + 1)[1:-1].tolist(),
             "turns": np.linspace(*np.percentile([speed_proxy(t) for t in start_t], [1, 99]), X.GRID + 1)[1:-1].tolist()}

F = X.A.Feats(start_t + ctx.corpus)
deep = X.Deep(F.dim, 5)
train_s = deep.fit(F.mat(start_t), np.asarray(start_y, float), np.asarray(turns_lab, float))
def predict(teams):
    w, t = deep.predict(F.mat(teams))
    if not turns_real:
        t = np.array([speed_proxy(tm) for tm in teams])
    return w, t
cell = lambda r, t: (int(np.clip(np.digitize(r, edges["ratio"]), 0, X.GRID - 1)),
                     int(np.clip(np.digitize(t, edges["turns"]), 0, X.GRID - 1)))
key = lambda t: [norm(s["species"]) for s in t]

pw, pt = predict(start_t)
arch = {}
for t, w_, tu in zip(start_t, pw, pt):
    c = cell(X.ob_ratio(t), tu)
    if c not in arch or w_ > arch[c][0]: arch[c] = (float(w_), t)
init = {f"{c[0]},{c[1]}": dict(w=w_, mons=key(t)) for c, (w_, t) in arch.items()}

mut = X.mutation_module(5); mut["rng"] = np.random.default_rng(5); rng = random.Random(5)
from hps_generate import Validator
val = Validator(); events = []
while len(events) < 60:
    pc = rng.choice(list(arch)); parent = arch[pc][1]; child = copy.deepcopy(parent)
    op = rng.choice([o for o in mut["COND"] if o[0] in "ABCDEF"])
    if mut["COND"][op](child): continue
    txt = mut["team_to_text"](child)
    if val(txt) is not None: continue
    ct = parse_team_text(txt); w_, tu = predict([ct]); w_, tu = float(w_[0]), float(tu[0])
    c = cell(X.ob_ratio(ct), tu); inc = arch.get(c)
    outcome = "new" if inc is None else ("replace" if w_ > inc[0] else "reject")
    sig = lambda s: (norm(s["species"]), norm(s.get("item") or ""), norm(s["ability"]), norm(s.get("nature") or ""),
                     tuple(sorted(norm(m) for m in s["moves"])), tuple(int((s.get("evs") or {}).get(k, 0)) for k in X.SPK.values()))
    events.append(dict(parent=f"{pc[0]},{pc[1]}", pmons=key(parent), cmons=key(ct), op=OPS.get(op, op),
                       changed=[i for i in range(min(len(parent), len(ct))) if sig(parent[i]) != sig(ct[i])],
                       cell=f"{c[0]},{c[1]}", w=w_, inc=None if inc is None else inc[0], outcome=outcome,
                       ratio=X.ob_ratio(ct), turns=tu))
    if outcome != "reject": arch[c] = (w_, ct)
val.close()

species = {m for v in init.values() for m in v["mons"]} | {m for e in events for m in e["pmons"] + e["cmons"]}
results = {"validate": res.get("validate"), "pilot": None if not pilot else {k: v for k, v in pilot.items() if k != "turns"},
           "gens": {}, "verify": res.get("verify") or {}, "summary": res.get("summary"), "archives": {}}
for rk, run in (res.get("runs") or {}).items():
    results["gens"][rk] = {g: dict(mean=float(np.mean(r["y"])), max=float(max(r["y"])), info=r["info"]) for g, r in run["gens"].items()}
    if pilot:
        a = {}
        for r in run["gens"].values():
            for p, y, tu in zip(r["pastes"], r["y"], r["turns"]):
                if tu is None: continue
                t = parse_team_text(p); c = cell(X.ob_ratio(t), tu)
                if f"{c[0]},{c[1]}" not in a or y > a[f"{c[0]},{c[1]}"]["w"]:
                    a[f"{c[0]},{c[1]}"] = dict(w=y, mons=key(t)); species.update(key(t))
        results["archives"][rk] = a
sprites = {}
for s in species:
    p = SPR / f"{s}.png"
    if not p.exists() and "mega" in s: p = SPR / f"{s.split('mega')[0]}.png"
    if p.exists(): sprites[s] = "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode()
data = dict(grid=X.GRID, edges=edges, turns_real=turns_real, init=init, events=events, train_seconds=train_s,
            params=deep.params, n_start=len(start_t), config=dict(gens=X.GENS, seeds=X.SEEDS, battle=X.BATTLE, battles=X.BATTLES,
            children=X.CHILDREN, ensemble=X.ENSEMBLE, epochs=X.EPOCHS, topv=X.TOPV, verify_cell=X.VERIFY_CELL),
            results=results, sprites=sprites)
html = (HERE / "template.html").read_text().replace("/*__DATA__*/null", json.dumps(data))
(HERE / "elites-on-a-map.html").write_text(html)
print("wrote elites-on-a-map.html", f"{len(html)/1024:.0f} KB · events {len(events)} "
      f"({sum(e['outcome']=='new' for e in events)} new, {sum(e['outcome']=='replace' for e in events)} replace, "
      f"{sum(e['outcome']=='reject' for e in events)} reject) · cells {len(init)} → {len(arch)} · turns real: {turns_real}")
