"""Pass@k re-read of stored ask grids (GitHub issue #331): does a sampler's breadth pay off when k teams are drawn?

The ask grids judged each row by its mean over all teams, which measures one sample. Pass@k is the chance that at least
one of k teams succeeds (Fang et al., COLM 2026, https://arxiv.org/abs/2604.00375). A team succeeds when its line-up is
an elite line-up that held: the team's win rate against the top-50 meta teams is at least 0.531, and its line-up's best
team still reaches 0.531 in the re-battle of 4 games per opponent (line_up_yield.py, all 23 asks). A grid with no
re-battle falls back to the one-score elite, the first condition alone, and its results say so.

    python pass_at_k.py read NAME [NAME ...]   any ask grid, by the name after "stat_points_84_", into results/pass_at_k/NAME.json
    python pass_at_k.py issue331               the ten grids of #331 and the verdict, into results/pass_at_k_issue331.json
    python pass_at_k.py figures DIR            the ECharts pages of #331 into DIR
    python pass_at_k.py pngs DIR               each page as light and dark PNGs, for the issue comment
    python pass_at_k.py comment URL            the verdict comment; URL holds {name} and {theme}

A grid needs its per-opponent snapshot (masking_rate_rows.py opponents ROW GUIDANCE) and, for the held definition, its
re-battle (line_up_yield.py rebattle). PAK_FRESH_DB names the re-battle database (default: the masking-schedule
battles' fresh.sqlite); MATCHUP_DB the grid's battle database. Both are opened read-only.

Pass@k for each ask uses the unbiased estimator 1 - C(n-c, k) / C(n, k) over its n = 192 teams, k = 1 to 48, so n is at
least 4k (Fang et al. App. E). An ask band's Pass@k is the mean over its asks. Potential@k is the same mean weighted by
each ask's 1 - Pass@1 (Jeon et al. 2026, https://arxiv.org/abs/2609.34509, App. C.4). The cost axis puts network calls
per legal team under k (Pass@NFE, Olausson et al., COLM 2026, https://arxiv.org/abs/2604.09921, §4.1).

The interval is a bootstrap that resamples the sampling seeds, then the teams within each ask and drawn seed, and the 49
opponents once for every grid, so two grids at one draw face the same opponents. The opponents move the one score and
the re-battle score together. A line-up that was not re-battled cannot hold in any draw.
"""
import json
import math
import os
import re
import sqlite3
import sys
import zlib
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import masking_rate_rows as MRR

RES = MRR.RES
OUT_DIR = RES / "pass_at_k"
ISSUE = RES / "pass_at_k_issue331.json"
FRESH_DB = Path(os.environ.get("PAK_FRESH_DB", Path.home() / "vgc-data/masking_rate_battles/fresh.sqlite"))
CUT, K_MAX, YIELD_VIEW = 0.531, 48, "all_23_asks"
BANDS = {"asks_13_to_17": range(13, 18), "asks_18_to_23": range(18, 24)}
BAND_READER = {"asks_13_to_17": "Asks 13/24 to 17/24", "asks_18_to_23": "Asks 18/24 to 23/24"}
PASSES_PER_TRY = 84                                     # one network pass per field, in both decoding orders (diffusion84.sample, remask84.sample_random)
DRAWS, DRAW_SEED = MRR.DRAWS, MRR.DRAW_SEED
HELD, ONE_SCORE = "held", "one_score"
SPECIES_FIRST, RANDOM, CONTROL, SPECIES_THEN_RANDOM = "sched_linear_species_first", "sched_linear_random", "sched_species_first", "sched_linear_species_then_random"
ISSUE_ROWS = (CONTROL, SPECIES_FIRST, RANDOM)
READER = {CONTROL: "Per-field, uncorrected t, the control", SPECIES_FIRST: "Linear, species-first", RANDOM: "Linear, random order",
          SPECIES_THEN_RANDOM: "Linear, species then random order"}


# ---------------------------------------------------------------- the estimators
def pass_at_k(n, c, k):
    """Unbiased Pass@k from n samples with c successes: 1 - C(n-c, k) / C(n, k) (Chen et al. 2021; Fang et al. Eq. 1)."""
    if not 0 <= c <= n or not 1 <= k <= n: raise ValueError(f"need 0 <= c <= n and 1 <= k <= n, got n={n}, c={c}, k={k}")
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def pass_table(n, k_max=K_MAX):
    """[k_max, n + 1]: row k - 1, column c is Pass@k for c successes in n samples."""
    return np.array([[pass_at_k(n, c, k) for c in range(n + 1)] for k in range(1, k_max + 1)])


def brute_force(outcomes, k):
    """Pass@k by listing every k-subset of the outcomes: the share with at least one success."""
    from itertools import combinations
    subsets = list(combinations(outcomes, k))
    return sum(any(s) for s in subsets) / len(subsets)


def potential_at_k(pass_k, pass_1):
    """Σ Pass@k·(1 - Pass@1) / Σ (1 - Pass@1) over asks (Jeon et al. App. C.4). pass_k [asks, ...], pass_1 [asks, ...]:
    the asks the first sample already solves count for less."""
    pass_k, w = np.asarray(pass_k, float), 1.0 - np.asarray(pass_1, float)
    if pass_k.ndim == w.ndim + 1: w = np.expand_dims(w, 1)               # pass_k [asks, K, ...] and pass_1 [asks, ...]
    total = w.sum(0)
    return np.divide((pass_k * w).sum(0), total, out=np.full(np.broadcast(pass_k.sum(0), total).shape, np.nan), where=total > 0)


# ---------------------------------------------------------------- one grid's teams
def weighted(rate, weights, counts=None):
    """Each team's weighted win rate over the opponents it battled: [teams], or [teams, DRAWS] with opponents counted `counts` [DRAWS, opponents] times."""
    seen = ~np.isnan(rate); won = np.where(seen, rate, 0.0)
    if counts is None: return (won @ weights) / (seen @ weights)
    w = (counts * weights).T
    return (won @ w) / np.maximum(seen @ w, 1e-12)


def read_only(path):
    if not Path(path).exists(): raise SystemExit(f"{path}: no such database")
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


def fresh_rates(pastes, columns, weights, battle_db, fresh_db):
    """[pastes, opponents] re-battle win rates against the grid's own opponent columns, matched by canonical paste."""
    sys.path.insert(0, "/tmp/vgc-pilot/src")
    import matchup_db as MDB
    battle, fresh = read_only(battle_db), read_only(fresh_db)
    hash_of = dict(battle.execute(f"SELECT team_id, canon_hash FROM team WHERE team_id IN ({','.join('?' * len(columns))})", [int(c) for c in columns]).fetchall())
    fresh_col = {h: (t, w) for t, h, w in fresh.execute("SELECT s.team_id, t.canon_hash, s.weight FROM team_set s JOIN team t USING (team_id) WHERE s.set_name='top50'")}
    if len(hash_of) != len(columns) or set(hash_of.values()) - set(fresh_col): raise SystemExit("an opponent of the grid is not in the re-battle database's top-50 set")
    cols = [fresh_col[hash_of[int(c)]][0] for c in columns]
    if not np.allclose([fresh_col[hash_of[int(c)]][1] for c in columns], weights): raise SystemExit("opponent weights differ between the grid and the re-battle")
    (policy,) = fresh.execute("SELECT policy_id FROM policy WHERE row_policy LIKE '%+slots-shuffled' AND row_policy NOT LIKE 'rust%'").fetchone()
    out = np.full((len(pastes), len(cols)), np.nan)
    for i, paste in enumerate(pastes):
        row = fresh.execute("SELECT team_id FROM team WHERE canon_hash=?", (MDB.canon_hash(MDB.canonical_paste(paste)),)).fetchone()
        if row is None: raise SystemExit("a re-battled team is not in the re-battle database")
        cells = MDB.row_cells(fresh, policy, row[0], cols)
        for j, c in enumerate(cols):
            if c in cells: out[i, j] = cells[c][0] / cells[c][1]
    return out


def load(name, fresh_db=FRESH_DB, battle_db=None):
    """One grid's teams: {ask: {seed: dict(rate [teams, opponents], line_ups [teams])}}, opponent weights, tries, and the
    re-battle of its elite line-ups when there is one."""
    sys.path.insert(0, "/tmp/vgc-pilot/src")
    import line_up_yield as LY
    grid = json.load(open(RES / f"stat_points_84_{name}.json"))
    snapshot = RES / f"masking_rate_rows_opponents_{name}.npz"
    if not snapshot.exists(): raise SystemExit(f"{name}: no per-opponent snapshot; run masking_rate_rows.py opponents first")
    data = np.load(snapshot); weights = data["weights"]
    teams = LY.grid_teams(name); cells = {}
    for c in grid["cells"].values():
        rate = data[f"a{c['numerator']:02d}_s{c['seed']}"]
        mine = [t for t in teams if (t["ask"], t["seed"]) == (c["numerator"], c["seed"])]
        if len(rate) != len(mine): raise SystemExit(f"{name} a{c['numerator']:02d}_s{c['seed']}: opponent rows and teams differ")
        if np.abs(weighted(rate, weights) - np.array([t["y"] for t in mine])).max(initial=0) > 1e-9: raise SystemExit(f"{name}: snapshot and stored scores differ")
        cells.setdefault(c["numerator"], {})[c["seed"]] = dict(rate=rate, line_ups=[t["line_up"] for t in mine], tried=c.get("tried"), kept=len(c["pastes"]))
    out = dict(name=name, reader=READER.get(re.sub(r"_g[0-9.]+$", "", name), grid["reader"]), guidance=float(grid["guidance"]), guided=not grid.get("unguided") and float(grid["guidance"]) != 1.0,
               passes=grid.get("passes", PASSES_PER_TRY), weights=weights, cells=cells,
               seeds=sorted({s for by in cells.values() for s in by}), definition=ONE_SCORE, candidates={})
    yield_path = RES / f"line_up_yield_{name}.json"
    if yield_path.exists():
        stored = json.load(open(yield_path)); fresh = {f["paste"]: f["fresh"] for f in stored["fresh"]}
        cand = LY.candidates(teams)                                  # {line-up: its best team over all 23 asks}, as the re-battle chose them
        pastes = [cand[lu]["paste"] for lu in sorted(cand)]
        rates = fresh_rates(pastes, data["columns"], weights, battle_db or default_battle_db(), fresh_db) if pastes else np.empty((0, len(weights)))
        if not np.allclose(weighted(rates, weights), [fresh[p] for p in pastes], atol=1e-9, rtol=0): raise SystemExit(f"{name}: re-battle rates no longer give the stored fresh scores")
        held = {lu for lu, p in zip(sorted(cand), pastes) if fresh[p] >= CUT}; out_games = stored["per_cell"]
        if held != {tuple(lu) for lu in stored["views"][YIELD_VIEW]["held_line_ups"]}: raise SystemExit(f"{name}: held line-ups differ from line_up_yield's")
        out.update(definition=HELD, candidates={lu: rates[i] for i, lu in enumerate(sorted(cand))}, rebattle_games_per_opponent=out_games)
    return out


def default_battle_db():
    sys.path.insert(0, "/tmp/vgc-pilot/src")
    import matchup_db as MDB
    return MDB.DEFAULT_DB


# ---------------------------------------------------------------- successes and their bootstrap
def successes(grid, definition, counts=None):
    """{(ask, seed): bool [teams]} or [teams, DRAWS] with the opponents counted `counts` times."""
    w = grid["weights"]
    if definition == HELD:
        held = {lu: weighted(r[None], w, counts)[0] >= CUT for lu, r in grid["candidates"].items()}
        never = np.zeros(() if counts is None else len(counts), bool)
    out = {}
    for k, by in grid["cells"].items():
        for s, cell in by.items():
            ok = weighted(cell["rate"], w, counts) >= CUT
            if definition == HELD and cell["line_ups"]: ok = ok & np.stack([held.get(lu, never) for lu in cell["line_ups"]])
            out[(k, s)] = ok
    return out


def counts_at(grid, asks, definition, counts, rng):
    """Point (c [asks], n [asks]) and draws (c [asks, DRAWS], n [asks, DRAWS]): seeds resampled, then teams within each
    ask and drawn seed."""
    point = successes(grid, definition); draws = successes(grid, definition, counts); seeds = grid["seeds"]
    c = np.array([sum(point[(k, s)].sum() for s in seeds) for k in asks]); n = np.array([sum(len(point[(k, s)]) for s in seeds) for k in asks])
    picked = rng.integers(0, len(seeds), (DRAWS, len(seeds)))          # [DRAWS, slots]: the seed each slot draws, shared by every ask
    cd, nd = np.zeros((len(asks), DRAWS), int), np.zeros((len(asks), DRAWS), int)
    for a, k in enumerate(asks):
        for slot in range(len(seeds)):
            for i, s in enumerate(seeds):
                ok = draws[(k, s)]; m = len(ok)
                pick = rng.integers(0, m, (DRAWS, m))
                got = ok[pick, np.arange(DRAWS)[:, None]].sum(1)
                here = picked[:, slot] == i
                cd[a, here] += got[here]; nd[a, here] += m
    return (c, n), (cd, nd)


def pass_curves(c, n, k_max=K_MAX):
    """[k_max, ...] Pass@k for arrays of successes c and team counts n of the same shape."""
    c, n = np.asarray(c), np.asarray(n); out = np.empty((k_max, *c.shape))
    for size in np.unique(n):
        if size < 4 * k_max: raise SystemExit(f"{size} teams at an ask: fewer than 4 x {k_max}, as Fang et al. App. E advise")
        table = pass_table(int(size), k_max); here = n == size
        out[:, here] = table[:, c[here]]
    return out


def calls_per_legal_team(grid, asks):
    """Network calls per legal team over these asks: every try costs PASSES_PER_TRY passes, a guided pass two calls,
    and every try the validator rejected counts. None when the grid stored no try counts."""
    cells = [cell for k in asks for cell in grid["cells"][k].values()]
    if any(cell["tried"] is None for cell in cells): return None
    return grid.get("passes", PASSES_PER_TRY) * (2 if grid["guided"] else 1) * sum(c["tried"] for c in cells) / sum(c["kept"] for c in cells)


def band_result(grid, definition, asks, counts, rng):
    (c, n), (cd, nd) = counts_at(grid, list(asks), definition, counts, rng)
    p, pd = pass_curves(c, n), pass_curves(cd, nd)                       # [K, asks] and [K, asks, DRAWS]
    pk, pkd = p.mean(1), pd.mean(1)
    pot, potd = potential_at_k(p.transpose(1, 0), p[0]), potential_at_k(pd.transpose(1, 0, 2), pd[0])
    iv = lambda d: [[float(np.quantile(x, .025)), float(np.quantile(x, .975))] for x in d]
    return dict(successes_by_ask={str(k): int(x) for k, x in zip(asks, c)}, teams_by_ask={str(k): int(x) for k, x in zip(asks, n)},
                pass_at_k=pk.tolist(), pass_at_k_interval=iv(pkd), potential_at_k=pot.tolist(), potential_at_k_interval=iv(potd),
                calls_per_legal_team=calls_per_legal_team(grid, asks)), pkd


def read_grid(grid, counts, definitions=None, *, allow_short=False):
    """Every band and definition of one loaded grid. Returns (result, {(definition, band): [K, DRAWS] Pass@k draws})."""
    definitions = definitions or ([HELD, ONE_SCORE] if grid["definition"] == HELD else [ONE_SCORE])
    result = dict(name=grid["name"], reader=grid["reader"], guidance=grid["guidance"], definition=grid["definition"],
                  held_line_ups=int(sum(weighted(r[None], grid["weights"])[0] >= CUT for r in grid["candidates"].values())) if grid["definition"] == HELD else None,
                  rebattled_line_ups=len(grid["candidates"]) if grid["definition"] == HELD else None,
                  rebattle_games_per_opponent=grid.get("rebattle_games_per_opponent"), bands={}); draws = {}
    for band, asks in BANDS.items():
        if allow_short:
            minimum = min(len(grid["seeds"]) * min(c["kept"] for c in grid["cells"][k].values()) for k in asks)
            if minimum < 4 * K_MAX:
                result.setdefault("skipped_bands", {})[band] = f"Bootstrap can draw {minimum} teams at an ask. Pass@{K_MAX} requires {4 * K_MAX}."
                continue
        for d in definitions:
            rng = np.random.default_rng([DRAW_SEED, 331, zlib.crc32(grid['name'].encode()), list(BANDS).index(band), definitions.index(d)])   # each grid's own stream, whatever else is read
            result["bands"].setdefault(d, {})[band], draws[(d, band)] = band_result(grid, d, asks, counts, rng)
    return result, draws


def read(names, *, allow_short=False):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name in names:
        grid = load(name); counts = MRR.opponent_draws(len(grid["weights"]))
        result, _ = read_grid(grid, counts, allow_short=allow_short)
        result.update(k=list(range(1, K_MAX + 1)), draws=DRAWS, draw_seed=DRAW_SEED, cut=CUT)
        json.dump(result, open(OUT_DIR / f"{name}.json", "w"), indent=1)
        b = result["bands"].get(result["definition"], {})
        print(f"{name}: {result['definition']}, " + ", ".join(f"{band} Pass@1 {v['pass_at_k'][0]:.3f} Pass@{K_MAX} {v['pass_at_k'][-1]:.3f}" for band, v in b.items()))
        for band, reason in result.get("skipped_bands", {}).items():
            print(f"{name}: {band} skipped. {reason}")
    print("PASS_AT_K_READ_DONE")


# ---------------------------------------------------------------- issue #331
def rule(random_point, first_point, random_draws, first_draws, random_iv, first_iv):
    """The ticket's rule at one guidance and band: random order pays off where its Pass@k interval lies above
    Linear, species-first's. Also the paired difference, random order minus species-first, with the opponents shared."""
    above = [k for k in range(1, K_MAX + 1) if random_iv[k - 1][0] > first_iv[k - 1][1]]
    diff = random_draws - first_draws
    lo, hi = np.quantile(diff, .025, axis=1), np.quantile(diff, .975, axis=1)
    return dict(pays_off=bool(above), k_where_random_interval_lies_above=above,
                difference=[float(a - b) for a, b in zip(random_point, first_point)], difference_interval=[[float(a), float(b)] for a, b in zip(lo, hi)],
                k_where_random_is_lower_paired=[k for k in range(1, K_MAX + 1) if hi[k - 1] < 0],
                k_where_random_is_higher_paired=[k for k in range(1, K_MAX + 1) if lo[k - 1] > 0])


def issue331():
    names = [MRR.grid_name(row, g) for row in ISSUE_ROWS for g in MRR.GUIDANCES] + [SPECIES_THEN_RANDOM]
    grids = {n: load(n) for n in names}
    counts = MRR.opponent_draws(len(grids[names[0]]["weights"]))
    if any(len(g["weights"]) != counts.shape[1] or not np.allclose(g["weights"], grids[names[0]]["weights"]) for g in grids.values()):
        raise SystemExit("the grids do not share one opponent set")
    results, draws = {}, {}
    for n in names:
        results[n], draws[n] = read_grid(grids[n], counts)
        print(f"{n}: read", flush=True)
    verdict = {}
    for g in MRR.GUIDANCES:
        r, f = MRR.grid_name(RANDOM, g), MRR.grid_name(SPECIES_FIRST, g)
        for band in BANDS:
            rb, fb = results[r]["bands"][HELD][band], results[f]["bands"][HELD][band]
            verdict[f"{g}/{band}"] = rule(rb["pass_at_k"], fb["pass_at_k"], draws[r][(HELD, band)], draws[f][(HELD, band)], rb["pass_at_k_interval"], fb["pass_at_k_interval"])
    pays = [key for key, v in verdict.items() if v["pays_off"]]
    out = dict(question="Does Linear, random order's extra breadth pay off at large k?", answer="yes" if pays else "no", pays_off_at=pays,
               rule="random order's breadth pays off if, at some k up to 48, its Pass@k interval lies above Linear, species-first's at the same guidance in either band",
               success=f"the team's win rate is at least {CUT} and its line-up held in the re-battle ({YIELD_VIEW} view of line_up_yield.py); "
                       f"{SPECIES_THEN_RANDOM} has no re-battle, so its success is the one-score elite alone",
               resampled="sampling seeds, then teams within each ask and drawn seed; the 49 opponents once for every grid, moving the one score and the re-battle together",
               k=list(range(1, K_MAX + 1)), draws=DRAWS, draw_seed=DRAW_SEED, cut=CUT, passes_per_try=PASSES_PER_TRY, opponents=int(counts.shape[1]),
               sampling_seeds=len(grids[names[0]]["seeds"]),
               bands={b: list(a) for b, a in BANDS.items()}, verdict=verdict, grids=results)
    json.dump(out, open(ISSUE, "w"), indent=1)
    for key, v in verdict.items():
        print(f"guidance {key}: pays off {v['pays_off']}; random lower (paired) at k {v['k_where_random_is_lower_paired'][:1]}..{v['k_where_random_is_lower_paired'][-1:]}")
    print("ANSWER", out["answer"]); print("PASS_AT_K_ISSUE331_DONE")

# ---------------------------------------------------------------- figures and the comment of #331
BLUE, AQUA, ORANGE, PURPLE, PLAIN = "#2a78d6", "#1baf7a", "#e0702a", "#8a5cd6", "#6f6e69"   # masking_rate_rows_figures.py colors, plus one
COLOR = {CONTROL: BLUE, SPECIES_FIRST: AQUA, RANDOM: ORANGE, SPECIES_THEN_RANDOM: PURPLE}
METRIC = {"pass_at_k": "Pass@k", "potential_at_k": "Potential@k"}
FIGURES = {"pass-at-k": ("pass_at_k", "k"), "potential-at-k": ("potential_at_k", "k"), "pass-at-network-calls": ("pass_at_k", "calls"), "one-score": ("pass_at_k", "k")}


def panels(result, metric, x_kind, rows, guidances, definition=HELD):
    """Small multiples: one row of panels per ask band, one column per guidance, one line and 95% band per experiment row."""
    from echarts_chart import band
    grids, xs, ys, titles, series = [], [], [], [], []
    cols, nrows = len(guidances), len(BANDS); seen = set()
    for r, (b, asks) in enumerate(BANDS.items()):
        for c, g in enumerate(guidances):
            i = r * cols + c; left, width = 8 + c * (88 / cols), 88 / cols - 5
            top = 12 + r * 44
            grids.append(dict(left=f"{left:.1f}%", width=f"{width:.1f}%", top=f"{top}%", height="30%"))
            titles.append(dict(text=f"Guidance {g:g} · {BAND_READER[b].lower()}" if cols > 1 else BAND_READER[b], left=f"{left + width / 2:.1f}%", top=f"{top - 5}%",
                               textAlign="center", textStyle=dict(fontSize=13, fontWeight="normal")))
            last_row, first_col = r == nrows - 1, c == 0
            xs.append(dict(type="value", gridIndex=i, min=1 if x_kind == "k" else 0, max=K_MAX if x_kind == "k" else None,
                           name=("k, teams drawn" if x_kind == "k" else "Network calls per legal team, k teams") if last_row else "",
                           nameLocation="middle", nameGap=28, splitLine=dict(show=False)))
            ys.append(dict(type="value", gridIndex=i, min=0, max=1, interval=0.25, axisLabel=dict(show=first_col, formatter="js:v => Math.round(v * 100) + '%'"),
                           name=METRIC[metric] if first_col else "", nameLocation="middle", nameGap=44, splitLine=dict(lineStyle=dict(opacity=0.35))))
            for row in rows:
                name = MRR.grid_name(row, g) if row != SPECIES_THEN_RANDOM else SPECIES_THEN_RANDOM
                if name not in result["grids"] or (row == SPECIES_THEN_RANDOM and g != 2): continue
                v = result["grids"][name]["bands"][definition][b]
                scale = 1.0 if x_kind == "k" else v["calls_per_legal_team"]
                x = [k * scale for k in range(1, K_MAX + 1)]
                lo, hi = [a for a, _ in v[f"{metric}_interval"]], [h for _, h in v[f"{metric}_interval"]]
                series += band(READER[row], COLOR[row], x, lo, hi, stack=f"band{i}{row}", xAxisIndex=i, yAxisIndex=i)
                series.append(dict(type="line", name=READER[row], color=COLOR[row], symbol="none", lineStyle=dict(width=2.2), emphasis=dict(focus="series"),
                                   xAxisIndex=i, yAxisIndex=i, data=[[a, y] for a, y in zip(x, v[metric])]))
                seen.add(READER[row])
    return dict(title=titles, grid=grids, xAxis=xs, yAxis=ys, series=series, animation=False,
                legend=dict(top=0, data=[READER[r] for r in rows if READER[r] in seen]),
                tooltip=dict(trigger="axis", valueFormatter="js:v => v == null ? '' : (v * 100).toFixed(1) + '%'"))


def figures(folder):
    from echarts_chart import write_chart
    r = json.load(open(ISSUE)); folder = Path(folder); out = []
    v = r["verdict"]; worst48 = max(x["difference"][-1] for x in v.values()); best48 = min(x["difference"][-1] for x in v.values())
    extra = [G["bands"][HELD][b]["calls_per_legal_team"] / r["grids"][MRR.grid_name(SPECIES_FIRST, G["guidance"])]["bands"][HELD][b]["calls_per_legal_team"] - 1
             for n, G in r["grids"].items() if n.startswith(RANDOM) for b in BANDS]
    cost = (min(extra), max(extra))
    words = facts(r)
    success = (f"Success: the team's win rate against the top-50 meta teams is at least {CUT} and its line-up held at {CUT} in the re-battle of {words['games']} games per opponent. "
               f"Band: 95% bootstrap over sampling seeds, teams within each ask and seed, and the {words['opponents']} opponents. {words['n']} teams per ask; Pass@k is the mean over the band's asks.")
    out.append(write_chart(panels(r, "pass_at_k", "k", ISSUE_ROWS, MRR.GUIDANCES), folder / "pass-at-k.html", height=640,
        title="Random order's breadth never pays off: at every k up to 48 its Pass@k stays below species-first's",
        subtitle=f"At k = 48, Linear, random order trails Linear, species-first by {-worst48:.2f} to {-best48:.2f}, at every guidance and in both ask bands.",
        note=success))
    out.append(write_chart(panels(r, "potential_at_k", "k", ISSUE_ROWS, MRR.GUIDANCES), folder / "potential-at-k.html", height=640,
        title="Potential@k tells the same story: the asks the first team misses are not rescued by more random-order teams",
        subtitle="Potential@k weights each ask by 1 - Pass@1 (Jeon et al., App. C.4).", note=success))
    out.append(write_chart(panels(r, "pass_at_k", "calls", ISSUE_ROWS, MRR.GUIDANCES), folder / "pass-at-network-calls.html", height=640,
        title=f"Counted in network calls, random order falls further behind: each of its legal teams costs {cost[0]:.0%} to {cost[1]:.0%} more",
        subtitle="x: k times the network calls per legal team. 84 passes per try, two calls per guided pass, and every try the validator rejected counts.",
        note=success + " Tries are stored per cell, so each team is charged its cell's mean."))
    out.append(write_chart(panels(r, "pass_at_k", "k", (SPECIES_FIRST, RANDOM, SPECIES_THEN_RANDOM), [2], ONE_SCORE), folder / "one-score.html", height=640,
        title="Without the re-battle, drawing the species first puts random order level with species-first",
        subtitle=f"Guidance 2, one-score elite: the team's win rate is at least {CUT}. Linear, species then random order has no re-battle, so it is judged on this alone.",
        note=f"Band: 95% bootstrap over sampling seeds, teams within each ask and seed, and the {words['opponents']} opponents. {words['n']} teams per ask."))
    for o in out: print(o)


def facts(r):
    """Numbers the captions and the comment state, read from the results so the words follow the data."""
    G = r["grids"]; calls = lambda row: [G[MRR.grid_name(row, g)]["bands"][HELD][b]["calls_per_legal_team"] for g in MRR.GUIDANCES for b in BANDS]
    first, rand = calls(SPECIES_FIRST), calls(RANDOM)
    games = {G[n]["rebattle_games_per_opponent"] for n in G if G[n]["rebattle_games_per_opponent"]}
    teams = {x for n in G for d in G[n]["bands"].values() for b in d.values() for x in b["teams_by_ask"].values()}
    if len(games) != 1 or len(teams) != 1: raise SystemExit(f"the grids differ in re-battle games {games} or teams per ask {teams}; the words assume one of each")
    return dict(games=games.pop(), n=teams.pop(), opponents=r["opponents"], seeds=r["sampling_seeds"], first_calls=(min(first), max(first)), random_calls=(min(rand), max(rand)),
                extra=(min(rand) / max(first) - 1, max(rand) / min(first) - 1))


def comment(url):
    """The verdict comment, picture first. url holds {name} (a FIGURES key) and {theme} (light or dark)."""
    import datetime
    r = json.load(open(ISSUE)); v = r["verdict"]; G = r["grids"]
    pic = lambda name, alt: (f'<picture><source media="(prefers-color-scheme: dark)" srcset="{url.format(name=name, theme="dark")}">'
                             f'<img width="860" alt="{alt}" src="{url.format(name=name, theme="light")}"></picture>')
    at = lambda row, g, b, k, d=HELD: G[MRR.grid_name(row, g)]["bands"][d][b]["pass_at_k"][k - 1]
    ivs = lambda row, g, b, k, d=HELD: G[MRR.grid_name(row, g)]["bands"][d][b]["pass_at_k_interval"][k - 1]
    first_full = lambda row, g, b: next((k for k in range(1, K_MAX + 1) if at(row, g, b, k) >= 0.99), None)
    d48 = [x["difference"][-1] for x in v.values()]; words = facts(r)
    pct = lambda pair: f"{pair[0]:.0%} to {pair[1]:.0%}"
    cost = lambda pair: f"{pair[0]:.0f}" if round(pair[0]) == round(pair[1]) else f"{pair[0]:.0f} to {pair[1]:.0f}"
    succ = lambda row: [x for g in MRR.GUIDANCES for b in BANDS for x in G[MRR.grid_name(row, g)]["bands"][HELD][b]["successes_by_ask"].values()]
    span = lambda row: f"{min(succ(row))} to {max(succ(row))}"
    full = [first_full(row, g, b) for row in (CONTROL, SPECIES_FIRST) for g in MRR.GUIDANCES for b in BANDS]
    r48 = [at(RANDOM, g, b, 48) for g in MRR.GUIDANCES for b in BANDS]
    lines = [f"**Pass@k re-read, {datetime.date.today()}: no. Linear, random order's extra breadth never pays off.** At every k from 1 to 48, at guidance 2, 4 and 8, "
             f"in both ask bands, its Pass@k stays below Linear, species-first's. At k = 48 it trails by {-max(d48):.2f} to {-min(d48):.2f}.", "",
             pic("pass-at-k", f"Six panels, ask bands 13/24 to 17/24 and 18/24 to 23/24 by guidance 2, 4 and 8. The control and Linear, species-first rise to 99% by k = {min(full)} to {max(full)}; "
                 f"Linear, random order rises slowly and ends between {min(r48):.0%} and {max(r48):.0%} at k = 48."), "",
             f"Pass@k is the chance that at least one of k teams drawn at an ask succeeds; a team succeeds when its win rate against the top-50 meta teams is at least {CUT} "
             f"and its line-up held at {CUT} in the re-battle of {words['games']} games per opponent. Each band's curve is the mean over its asks, from the {words['n']} teams of each ask.", "",
             "- **Rule, written before reading:** random order's breadth pays off if, at some k up to 48, its Pass@k interval lies above Linear, species-first's at the same guidance in either band. "
             f"It does at {sum(x['pays_off'] for x in v.values())} of the {len(v)} guidance-and-band pairs. The paired difference, random order minus species-first with the same opponents drawn, "
             f"has its interval below zero at {sum(len(x['k_where_random_is_lower_paired']) for x in v.values())} of the {len(v) * K_MAX} points ({len(v)} pairs × {K_MAX} values of k) and above zero at {sum(len(x['k_where_random_is_higher_paired']) for x in v.values())}.",
             f"- **Why the breadth does not help:** random order draws {span(RANDOM)} held successes among an ask's {words['n']} teams, where species-first draws {span(SPECIES_FIRST)}. "
             "More distinct line-ups do not make up for so few successes: its Pass@48 reaches "
             f"{min(r48):.0%} to {max(r48):.0%}, "
             "while species-first reaches 99% by k = " + "; ".join(" and ".join(str(first_full(SPECIES_FIRST, g, b)) for b in BANDS) + f" at guidance {g}" for g in MRR.GUIDANCES) + " (asks 13/24 to 17/24, then 18/24 to 23/24).",
             "- **Baseline:** Linear, species-first on the same linear checkpoint, at the same guidance, from the masking-schedule grids of "
             "[Sample and battle the four rows and post the results table](https://github.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/issues/297).", "",
             "### Potential@k", "",
             pic("potential-at-k", "Potential@k in the same six panels; the curves match the Pass@k curves closely."), "",
             "Potential@k weights each ask by 1 − Pass@1, so asks that one team already solves count for less (Jeon et al., App. C.4). Random order solves few asks with its first team, so its weights are nearly even and its curves barely move.", "",
             "### Against network calls", "",
             pic("pass-at-network-calls", "Pass@k against network calls per legal team; random order's curves stretch " + pct(words["extra"]) + " to the right."), "",
             "The x axis is k times the network calls per legal team (Pass@NFE, Olausson et al. §4.1): 84 passes per try, two calls per guided pass, and every rejected try counted. "
             f"Species-first costs {cost(words['first_calls'])} calls per legal team; random order costs {cost(words['random_calls'])}, because it needs {pct(words['extra'])} more tries per legal team: the validator rejects some of its teams, and none of species-first's. "
             f"All {len(G)} grids stored their try counts.", "",
             "### Linear, species then random order", "",
             pic("one-score", "Guidance 2, one-score elite: Linear, species then random order and Linear, species-first overlap; Linear, random order sits far below."), "",
             f"This grid, from [Does Linear, random order draw weaker teams because it draws species late?](https://github.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/issues/322), has no re-battle, so it is judged on the one-score elite alone (win rate at least {CUT}), "
             "and so are the two rows beside it here. Drawing the species first brings random order level with species-first, "
             f"Pass@1 {at(SPECIES_THEN_RANDOM, 2, 'asks_13_to_17', 1, ONE_SCORE):.3f} against "
             f"{at(SPECIES_FIRST, 2, 'asks_13_to_17', 1, ONE_SCORE):.3f} at asks 13/24 to 17/24, which agrees with that ticket's verdict.", "",
             "<details><summary>Every number at k = 1, 8 and 48</summary>", "",
             "| Row | Guidance | Asks | Pass@1 | Pass@8 | Pass@48 | Random order minus species-first at k = 48 |", "|---|---:|---|---|---|---|---|"]
    f = lambda x, i: f"{x:.3f} [{i[0]:.3f}, {i[1]:.3f}]"
    for g in MRR.GUIDANCES:
        for b in BANDS:
            for row in ISSUE_ROWS:
                d = v[f"{g}/{b}"]
                tail = f"{d['difference'][-1]:+.3f} [{d['difference_interval'][-1][0]:+.3f}, {d['difference_interval'][-1][1]:+.3f}]" if row == RANDOM else ""
                lines.append(f"| {READER[row]} | {g} | {BAND_READER[b][5:]} | {f(at(row, g, b, 1), ivs(row, g, b, 1))} | {f(at(row, g, b, 8), ivs(row, g, b, 8))} | {f(at(row, g, b, 48), ivs(row, g, b, 48))} | {tail} |")
    lines += ["", f"Held successes per ask, out of {words['n']} teams:", ""]
    for g in MRR.GUIDANCES:
        lines.append(f"- Guidance {g}: " + "; ".join(f"{READER[row]} " + ", ".join(str(x) for b in BANDS for x in G[MRR.grid_name(row, g)]['bands'][HELD][b]['successes_by_ask'].values()) for row in ISSUE_ROWS))
    lines += ["", "Asks run 13/24 to 23/24 in each list.", "",
              f"Interval: 95% bootstrap, {r['draws']:,} draws. Each draw resamples the {words['seeds']} sampling seeds, then the teams within each ask and drawn seed, and the {words['opponents']} opponents once for every grid; "
              "the drawn opponents move the one score and the re-battle score together. A line-up never re-battled cannot hold. Estimator: 1 − C(n−c, k) / C(n, k) with n = {words['n']}, so n ≥ 4k (Fang et al. App. E); "
              "tested against brute-force enumeration. Script: `code/src/pass_at_k.py issue331`; any other ask grid: `pass_at_k.py read NAME`. Results: `~/vgc-data/results/pass_at_k_issue331.json`.",
              "</details>"]
    return "\n".join(lines) + "\n"


def pngs(folder, width=960, scale=2):
    """<name>-light.png and <name>-dark.png beside each page of `figures`, drawn by headless Chrome told the colour scheme."""
    import base64, tempfile, time
    from excalidraw_table import Browser
    folder = Path(folder)
    with tempfile.TemporaryDirectory() as work:
        browser = Browser(Path(work) / "profile")
        try:
            for name in FIGURES:
                for theme in ("light", "dark"):
                    browser.call("Emulation.setEmulatedMedia", features=[dict(name="prefers-color-scheme", value=theme)])
                    browser.call("Emulation.setDeviceMetricsOverride", width=width, height=900, deviceScaleFactor=scale, mobile=False)
                    browser.call("Page.navigate", url=(folder / f"{name}.html").resolve().as_uri())
                    deadline = time.monotonic() + 60
                    while not browser.value("!!document.querySelector('#chart svg') && document.readyState === 'complete'"):
                        if time.monotonic() > deadline: raise RuntimeError(f"{name}: the chart did not draw")
                        time.sleep(0.25)
                    time.sleep(0.5)
                    if browser.value("matchMedia('(prefers-color-scheme: dark)').matches") != (theme == "dark"): raise RuntimeError("the colour scheme did not take")
                    height = browser.value("Math.ceil(document.querySelector('main').getBoundingClientRect().bottom)")
                    shot = browser.call("Page.captureScreenshot", format="png", clip=dict(x=0, y=0, width=width, height=height, scale=1))
                    (folder / f"{name}-{theme}.png").write_bytes(base64.b64decode(shot["data"])); print(folder / f"{name}-{theme}.png")
        finally:
            browser.close()


if __name__ == "__main__":
    {"read": lambda: read([name for name in sys.argv[2:] if name != "--allow-short"], allow_short="--allow-short" in sys.argv[2:]),
     "issue331": issue331, "figures": lambda: figures(sys.argv[2]), "pngs": lambda: pngs(sys.argv[2]),
     "comment": lambda: print(comment(sys.argv[2]), end="")}[sys.argv[1]]()
