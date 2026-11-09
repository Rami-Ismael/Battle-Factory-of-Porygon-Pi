"""Why does remasking lift win rate at every ask instead of following it? (2026-10-30)
Follow-up to remask_ask_gap.py (docs/remask-ask-gap.md, "Why remasking ignores the ask").

Two controls, both scored like the remasking grid (64 Showdown-legal, non-held-out teams per cell, seeds 1-3 at base
7300, guidance 2, temperature 1, one battle against each of the 49 distinct top-50 meta teams, slots shuffled):

- Random remasking: remasking 10% at 48 passes, but the fields to mask are drawn at random among the filled ones
  instead of the least likely. Asks 0.3, 0.5, 0.531. If it keeps plain masking's ask dependence and diversity, the
  least-likely selection rule is what drives the sampler to the most common team; if it doesn't, the network's own
  redraws do.
- On-grid asks: plain masking at asks 7/24, 8/24 and 13/24 (0.2917, 0.3333, 0.5417). Most training labels are
  24-battle scores, so they sit on steps of 1/24; asks 0.3 and 0.531 sit between steps. If plain masking's win rate at
  the off-grid asks breaks the trend of their on-grid neighbours, the ask encoding is to blame.

- Alphabetical order (ran 2026-10-30; the command was removed 2026-11-07 with listed order, and its results stay
  under the key "listed"): the plain masking and remasking 10% teams of the grid, re-battled in listed order.
  The matchup database stores every team in canonical form (matchup_db.canonical_slots sorts the six Pokemon by
  species), so "listed" battles them in alphabetical order, not the order the model drew them. Matrix-labelled
  training teams were battled the same way before 2026-10-28; every grid here is shuffled. The paired difference per
  team is how far apart the two orders score for these teams.

`probe` needs no battles: the first species pick's distribution across a fine grid of asks, and how much the ask moves
each field type once the other 47 fields are visible.

    python remask_why.py probe | gen | battle | report        (run with /tmp/vgc-pilot/.venv/bin/python)
"""
import json, math, os, sys
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, "/tmp/vgc-pilot/src")
sys.path.insert(0, str(HERE))
import discrete_diffusion_model
sys.path.insert(0, str(HERE))
import remask_ask_gap as RA
import lossdown as L
import matchup_db as MDB
from team_diversity import slots_from_paste

if Path(discrete_diffusion_model.__file__).resolve().parent != HERE:
    raise SystemExit(f"loaded {discrete_diffusion_model.__file__}, not the discrete diffusion model beside this script")

OUT = RA.RES / "remask_why.json"
CELLS = [("random10", 0.10, a) for a in (0.3, 0.5, 0.531)] + [("plain", 0.0, a) for a in (7 / 24, 8 / 24, 13 / 24)]
CORE = {"garchomp", "basculegion", "kingambit", "whimsicott", "charizard", "floetteeternal"}   # training set's top 6
LOWEST = discrete_diffusion_model._lowest_confidence
FINE_ASKS = [0.25, 7 / 24, 0.3, 8 / 24, 0.4, 0.45, 0.48, 0.5, 0.52, 0.531, 13 / 24, 0.56, 0.6]


def random_pick(conf, fraction):
    """_lowest_confidence's count of fields, chosen at random among the filled ones."""
    filled = conf.isfinite()
    k = torch.floor(fraction * filled.sum(1) + torch.rand(len(conf), device=conf.device)).long()
    score = torch.rand(conf.shape, device=conf.device).masked_fill(~filled, float("inf"))
    columns = torch.arange(conf.shape[1], device=conf.device).expand_as(conf)
    rank = torch.empty_like(columns).scatter_(1, score.argsort(1), columns)
    return (rank < k[:, None]) & filled


def save(out):
    tmp = OUT.with_suffix(".tmp"); json.dump(out, open(tmp, "w")); os.replace(tmp, OUT)


def guided(w, x, t, ask, g=RA.GUIDANCE):
    """{field: guided log-probs} at one ask (one value per row), as the sampler mixes them."""
    a = w.logprobs(x, t, torch.full((len(x),), ask, device=L.DEV))
    n = w.logprobs(x, t, torch.full((len(x),), float("nan"), device=L.DEV))
    return {k: g * a[k] + (1 - g) * n[k] for k in a}


def probs(lp_col, legal):
    return torch.softmax(lp_col.masked_fill(~legal, -float("inf")), -1)


def probe():
    out = json.load(open(OUT)) if OUT.exists() else {}
    w = discrete_diffusion_model.DiscreteDiffusionModel.load(); V = w.V
    core_ids = [V.stoi["species"][s] for s in V.itos["species"] if s in CORE]
    # 1. The first species pick (column 0, nothing visible) across a fine grid of asks.
    x0 = torch.zeros(1, L.COLS, dtype=torch.long, device=L.DEV); t0 = torch.ones(1, device=L.DEV)
    first = {}
    with torch.no_grad():
        for a in FINE_ASKS:
            p = probs(guided(w, x0, t0, a)["species"][:, 0], w.legal_values(x0, 0))[0]
            first[a] = p
    tv = {f"{a:.4f}": {f"{b:.4f}": round(0.5 * float((first[a] - first[b]).abs().sum()), 3) for b in FINE_ASKS} for a in FINE_ASKS}
    core = {f"{a:.4f}": round(float(first[a][core_ids].sum()), 3) for a in FINE_ASKS}
    # 2. How much the ask moves each field once the other 47 fields are visible: 192 plain teams at ask 0.5.
    torch.manual_seed(RA.SEED0 + 1); x = w.sample(192, ask=0.5, guidance=RA.GUIDANCE)
    t = torch.full((len(x),), 1 / L.COLS, device=L.DEV)
    by_field = {}
    with torch.no_grad():
        for c in range(L.COLS):
            xm = x.clone(); xm[:, c] = 0; legal = w.legal_values(xm, c)
            key, i = L.KEY_OF_COL[c], L.COLS_BY_KEY[L.KEY_OF_COL[c]].index(c)
            lo, hi = (probs(guided(w, xm, t, a)[key][:, i], legal) for a in (0.3, 0.531))
            d = by_field.setdefault(key, dict(tv=[], entropy=[], p_current=[]))
            d["tv"].append(float(0.5 * (lo - hi).abs().sum(1).mean()))
            d["entropy"].append(float(-(hi * hi.clamp_min(1e-12).log()).sum(1).mean()))
            d["p_current"].append(float(hi.gather(1, x[:, c:c + 1]).mean()))
    fields = {k: {m: float(np.mean(v)) for m, v in d.items()} for k, d in by_field.items()}
    out["probe"] = dict(first_species_tv=tv, first_species_core_share=core, full_context_by_field=fields)
    save(out)
    print("first species pick, share on the six core species by ask:", core)
    print("total variation of the first species pick from ask 0.5:", {a: tv["0.5000"][a] for a in tv})
    for k, f in fields.items():
        print(f"  {k:8s} ask 0.3 vs 0.531, total variation {f['tv']:.3f} · entropy at 0.531 {f['entropy']:.2f} nats · "
              f"probability of the current value {f['p_current']:.2f}")


def gen():
    out = json.load(open(OUT)) if OUT.exists() else {}
    cells = out.setdefault("cells", {}); w = discrete_diffusion_model.DiscreteDiffusionModel.load()
    for seed in RA.SEEDS:
        for arm, remask, ask in CELLS:
            tag = f"{arm}_a{ask:.4f}_s{seed}"
            if tag in cells: continue
            discrete_diffusion_model._lowest_confidence = random_pick if arm == "random10" else LOWEST
            r = RA.draw(w, ask, remask, 48, seed)
            cells[tag] = dict(arm=arm, remask=remask, ask=ask, seed=seed, tried=r["tried"], rejected=r["rejected"],
                              legal=(r["tried"] - r["rejected"]) / max(r["tried"], 1), redraws=r["redraws"],
                              div=RA.diversity(r["kept"]), pastes=r["kept"])
            save(out)
            print(f"  {tag:22s} legal {cells[tag]['legal']:.1%} · line-ups {cells[tag]['div']['distinct_lineups']} · "
                  f"entropy {cells[tag]['div']['species_entropy']:.2f}", flush=True)


def battle():
    RA.require_shuffled(); con = MDB.connect(); out = json.load(open(OUT))
    for tag, c in out["cells"].items():
        if "y" in c: continue
        sc = MDB.score_pastes(con, c["pastes"], per_cell=1, seed=RA.SEED0 + c["seed"], origin=f"rmwhy:{tag}")
        if not RA.complete(sc, 1): print(f"  {tag} INCOMPLETE: left for the next run", flush=True); continue
        y = [s["score"] for s in sc]
        c["y"] = [s["score"] for s in sc]; c["win"] = dict(mean=float(np.mean(y)), se=float(np.std(y, ddof=1) / math.sqrt(len(y))))
        save(out); print(f"  {tag:22s} win {c['win']['mean']:.3f}", flush=True)


def report():
    out = json.load(open(OUT)); grid = json.load(open(RA.OUT))["cells"]
    rows = [("plain masking", "plain", a, grid) for a in (0.3, 0.5, 0.531)] + \
           [("remasking 10%, least likely", "remask10", a, grid) for a in (0.3, 0.5, 0.531)] + \
           [("remasking 10%, random", "random10", a, out["cells"]) for a in (0.3, 0.5, 0.531)] + \
           [("plain masking, on-grid ask", "plain", a, out["cells"]) for a in (7 / 24, 8 / 24, 13 / 24)]
    summ = []
    for name, arm, ask, src in rows:
        cs = [c for c in src.values() if c["arm"] == arm and abs(c["ask"] - ask) < 1e-9 and "win" in c]
        if len(cs) != len(RA.SEEDS): print(f"{name} at {ask:.4f}: incomplete"); continue
        teams = [slots_from_paste(p) for c in cs for p in c["pastes"]]
        core = float(np.mean([CORE <= {s["species"] for s in t} for t in teams if t]))
        win = [c["win"]["mean"] for c in cs]
        s = dict(name=name, ask=ask, win=float(np.mean(win)), win_sd=float(np.std(win, ddof=1)),
                 lineups=float(np.mean([c["div"]["distinct_lineups"] for c in cs])),
                 entropy=float(np.mean([c["div"]["species_entropy"] for c in cs])), all_core=core)
        summ.append(s)
        print(f"{name:30s} ask {ask:.4f} · win {s['win']:.3f} ± {s['win_sd']:.3f} · line-ups {s['lineups']:4.1f} · "
              f"entropy {s['entropy']:.2f} · all six core {core:.0%}")
    out["summary"] = summ; save(out)


if __name__ == "__main__":
    {"probe": probe, "gen": gen, "battle": battle, "report": report}[sys.argv[1]]()
