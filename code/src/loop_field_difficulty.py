"""What the 84-field model's own predictions say about looping. No looped model is trained or run here.

Three read-only measurements on the all-teams 84-field checkpoint (`stat84_all_teams_ep21.pt`), no ask, no guidance:
  by_field:  effective options (exp of entropy, legal values only) of each field type at the step it commits.
  clash:     if the six items, or the six species, committed together in one step, how many teams repeat one.
  lock_grid: on real team MB607 with all species visible, effective options of each hidden field, with nothing
             else decided and with each one Pokémon's whole set decided.

    /tmp/vgc-pilot/.venv/bin/python code/src/loop_field_difficulty.py OUT.json
"""
import json
import sys
import time
from pathlib import Path

import torch

torch.backends.mps.is_available = lambda: False        # other sessions are training on the GPU; this is small
sys.path.insert(0, str(Path(__file__).parent))

from corpus import parse_team
import diffusion84 as M
import lossdown as L
import team84 as T

CK = Path.home() / "vgc-data/stat_points_84/hub/vgc-regmb-discrete-diffusion-84-fields-all-teams-ep21/stat84_all_teams_ep21.pt"
TEAM = Path.home() / "vgc-data/teams/reg_mb/MB607.txt"
N, NCLASH, SEED = 256, 2000, 0
NF, NSLOT, STAT0 = M.NF, M.NSLOT, M.STAT0
NSTAT = M.SP.NSTAT
NAMES = ["species", "ability", "item", "move 1", "move 2", "move 3", "move 4", "Stat Alignment"]
STATS = ["HP", "Atk", "Def", "SpA", "SpD", "Spe"]


def field_of(c):
    """(Pokémon index, field name) of column c."""
    if c >= STAT0:
        return (c - STAT0) // NSTAT, "Stat Points " + STATS[(c - STAT0) % NSTAT]
    return c // NF, NAMES[c % NF]


def group(name):
    return "move" if name.startswith("move") else ("Stat Points" if name.startswith("Stat Points") else name)


def entropy(lc):
    p = torch.softmax(lc, -1)
    return -(p * torch.log(p.clamp_min(1e-12))).sum(-1)


def column(m, x, c, t, w):
    lc = m._column(m._logprobs(x, t, w), c)
    return lc.masked_fill(~m._legal(x, c), -float("inf"))


@torch.no_grad()
def by_field(m, n):
    """Run the deployed sampler, one field per step, and record each field's entropy as it commits."""
    w = torch.full((n,), M.NAN); x = torch.zeros(n, M.COLS, dtype=torch.long)
    rows = []
    for step, c in enumerate(M.ORDER):
        t = torch.full((n,), 1.0 - step / M.COLS)
        lc = column(m, x, c, t, w)
        rows.append(dict(step=step, column=c, field=group(field_of(c)[1]), pokemon=field_of(c)[0],
                         entropy=float(entropy(lc).mean()), top1=float(torch.softmax(lc, -1).max(-1).values.mean())))
        v = torch.multinomial(torch.softmax(lc, -1), 1).squeeze(1)
        x[:, c] = v + 1 if c >= STAT0 else v
    return rows, x


@torch.no_grad()
def clash(m, n, offset):
    """offset 2 = item, 0 = species. Walk the deployed order up to the first such field, then read all six at once."""
    w = torch.full((n,), M.NAN); x = torch.zeros(n, M.COLS, dtype=torch.long)
    cols = [s * NF + offset for s in range(NSLOT)]
    step = 0
    for c in M.ORDER:
        if c in cols: break
        t = torch.full((n,), 1.0 - step / M.COLS)
        x[:, c] = torch.multinomial(torch.softmax(column(m, x, c, t, w), -1), 1).squeeze(1)
        step += 1
    t = torch.full((n,), 1.0 - step / M.COLS)
    lp = m._logprobs(x, t, w)
    sampled, greedy = [], []
    for c in cols:
        lc = m._column(lp, c).masked_fill(~m._legal(x, c), -float("inf"))
        sampled.append(torch.multinomial(torch.softmax(lc, -1), 1).squeeze(1))
        greedy.append(lc.argmax(-1))
    def rate(vs):
        a = torch.stack(vs, 1)
        if offset == 0:                                    # Species Clause compares base species
            base = torch.tensor(m.tables.base_of if hasattr(m.tables, "base_of") else list(range(int(a.max()) + 1)))
            a = base[a]
        return float(sum(len(set(r.tolist())) < NSLOT for r in a) / len(a))
    return dict(commit_at_step=step, sampled_together=rate(sampled), most_likely_together=rate(greedy), n=n)


@torch.no_grad()
def lock_grid(m):
    team = parse_team(TEAM)
    row = torch.as_tensor(T.encode(m.V, team), dtype=torch.long)
    species = [m.V.decode_field(s * NF, int(row[s * NF])) for s in range(NSLOT)]
    conditions = [None, *range(NSLOT)]
    out = []
    for lock in conditions:
        x = torch.zeros(1, M.COLS, dtype=torch.long)
        seen = [s * NF for s in range(NSLOT)]
        if lock is not None:
            seen += [c for c in range(M.COLS) if field_of(c)[0] == lock]
        x[0, seen] = row[seen]
        t = torch.tensor([1.0 - len(set(seen)) / M.COLS]); w = torch.tensor([M.NAN])
        cells = []
        for c in range(M.COLS):
            if c in seen: continue
            lc = column(m, x, c, t, w)
            p = torch.softmax(lc, -1)[0]
            tok = int(row[c]); tok = tok - 1 if c >= STAT0 else tok
            cells.append(dict(column=c, pokemon=field_of(c)[0], field=field_of(c)[1], entropy=float(entropy(lc)[0]),
                              p_true=float(p[tok])))
        out.append(dict(lock=lock, cells=cells))
    return dict(team="MB607", species=species, row=[int(v) for v in row], conditions=out)


@torch.no_grad()
def item_choice(m, top=4, show=5):
    """On MB607 with species visible: each Pokémon's likeliest items, then the same after giving one Pokémon an item."""
    row = torch.as_tensor(T.encode(m.V, parse_team(TEAM)), dtype=torch.long)
    base = torch.zeros(1, M.COLS, dtype=torch.long)
    base[0, [s * NF for s in range(NSLOT)]] = row[[s * NF for s in range(NSLOT)]]
    w = torch.tensor([M.NAN])

    def read(x):
        t = torch.tensor([1.0 - int((x[0] != 0).sum()) / M.COLS])
        out = {}
        for s in range(NSLOT):
            c = s * NF + 2
            if int(x[0, c]) != 0: continue
            p = torch.softmax(column(m, x, c, t, w), -1)[0]
            v, i = p.topk(show)
            out[s] = [[m.V.decode_field(c, int(k)), round(float(q), 4)] for q, k in zip(v, i) if q > 0]
        return out

    free = read(base)
    given = {}
    for s in range(NSLOT):
        c = s * NF + 2
        p = torch.softmax(column(m, base, c, torch.tensor([1.0 - NSLOT / M.COLS]), w), -1)[0]
        for k in p.topk(top).indices.tolist():
            x = base.clone(); x[0, c] = k
            given[f"{s}|{m.V.decode_field(c, k)}"] = read(x)
    return dict(free=free, given=given, true_items=[m.V.decode_field(s * NF + 2, int(row[s * NF + 2])) for s in range(NSLOT)])


@torch.no_grad()
def binding(m, teams=40):
    """Do item predictions depend on the species beside them? Compare, before the legal mask, the six item predictions
    within a real team (species visible) with how much real item choices differ between species in the corpus."""
    import glob
    import random
    from collections import Counter, defaultdict
    files = sorted(glob.glob(str(TEAM.parent / "MB*.txt"))); random.Random(SEED).shuffle(files)
    rows, cond = [], defaultdict(Counter)
    for f in files:
        try: r = torch.as_tensor(T.encode(m.V, parse_team(f)), dtype=torch.long)
        except ValueError: continue
        rows.append(r)
        for s in range(NSLOT): cond[int(r[s * NF])][int(r[s * NF + 2])] += 1
    w, t, tv = torch.tensor([M.NAN]), torch.tensor([1 - NSLOT / M.COLS]), []
    for r in rows[:teams]:
        x = torch.zeros(1, M.COLS, dtype=torch.long); x[0, [s * NF for s in range(NSLOT)]] = r[[s * NF for s in range(NSLOT)]]
        lp = m._logprobs(x, t, w); ps = [torch.softmax(m._column(lp, s * NF + 2)[0], -1) for s in range(NSLOT)]
        tv += [float(.5 * (ps[a] - ps[b]).abs().sum()) for a in range(NSLOT) for b in range(a + 1, NSLOT)]
    sp = [k for k, v in cond.items() if sum(v.values()) >= 20]; ctv = []
    for i, a in enumerate(sp):
        for b in sp[i + 1:]:
            na, nb = sum(cond[a].values()), sum(cond[b].values())
            ctv.append(.5 * sum(abs(cond[a][k] / na - cond[b][k] / nb) for k in set(cond[a]) | set(cond[b])))
    return dict(model_within_team_total_variation=sum(tv) / len(tv), teams=teams, corpus_teams=len(rows),
                corpus_between_species_total_variation=sum(ctv) / len(ctv), species_with_20_appearances=len(sp),
                n_train=m.info.get("n_train"))


def main(path):
    torch.manual_seed(SEED); t0 = time.time()
    m = M.Model84.load(CK)
    for net, _ in m.nets: net.eval()
    if len(sys.argv) > 2 and sys.argv[2] == "--items-only":     # add item_choice to an existing file
        res = json.loads(Path(path).read_text()); res["item_choice"] = item_choice(m); res["binding"] = binding(m)
        res["lock_grid"]["decoded"] = T.decode(m.V, res["lock_grid"]["row"])
        res["n_parameters"] = sum(p.numel() for p in m.nets[0][0].parameters())
        Path(path).write_text(json.dumps(res, indent=1))
        print(res["n_parameters"], json.dumps(res["item_choice"]["free"])[:900]); return
    rows, x = by_field(m, N)
    items = torch.stack([x[:, s * NF + 2] for s in range(NSLOT)], 1)
    res = dict(
        checkpoint=CK.name, layers=m.nets[0][1].get("nlayer", 6), n_teams=N, seed=SEED,
        order=[field_of(c)[1] for c in M.ORDER][:24],
        deployed_item_repeats=float(sum(len(set(r.tolist())) < NSLOT for r in items) / N),
        by_field=rows, clash_item=clash(m, NCLASH, 2), clash_species=clash(m, NCLASH, 0), lock_grid=lock_grid(m),
        seconds=time.time() - t0, script="code/src/loop_field_difficulty.py", no_looped_model_ran=True,
        ck_dir=str(L.CK))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("deployed_item_repeats", "clash_item", "clash_species", "seconds", "order")}))
    by = {}
    for r in rows: by.setdefault(r["field"], []).append(r["entropy"])
    import math
    print({k: round(math.exp(sum(v) / len(v)), 2) for k, v in by.items()})


if __name__ == "__main__":
    main(sys.argv[1])
