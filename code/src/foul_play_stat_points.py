"""Does the Foul Play deterministic argmax win rate respond to Stat Points? (issue #448, map #416).

Four Stat-Point arms on 20 human-constructed top-50 meta teams, holding all 48 non-Stat-Point fields
(including Stat Alignment) fixed, battled against the 48 other distinct top-50 meta teams at 4 battles
per opponent under both Foul Play deterministic argmax (`rust_foulplay_det`, 30,000 iterations, 1 sample)
and behaviour cloning (`bc_100.zip`), both sides with Pokémon order shuffled every battle.

Protocol and pre-registered storyboard: code/docs/foul-play-stat-points.md.

    PY=/tmp/vgc-pilot/.venv/bin/python
    $PY src/foul_play_stat_points.py sample
    $PY src/foul_play_stat_points.py smoke
    $PY src/foul_play_stat_points.py battle-bc
    $PY src/foul_play_stat_points.py battle-fp
    $PY src/foul_play_stat_points.py report
"""
import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
CODE = HERE.parent
sys.path.insert(0, str(HERE))

OUT = Path(os.environ.get("FP_SP_OUT", Path.home() / "vgc-data" / "foul_play_stat_points_448"))
SPREADS_PATH = CODE / "data" / "foul-play-doubles-spreads.json"
SEED = 448
N_BASE = 20
PER_CELL = 4
BOOTSTRAP = 2000

STATS = ("HP", "Atk", "Def", "SpA", "SpD", "Spe")
MAX_STAT = 32
MAX_TOTAL = 66

ARMS = ("stored", "redrawn", "random_legal", "zeroed")
EDITED_ARMS = ("redrawn", "random_legal", "zeroed")
ARM_NAMES = {
    "stored": "Stored Stat Points",
    "redrawn": "Redrawn same-species Stat Points",
    "random_legal": "Random legal Stat Points",
    "zeroed": "Zeroed Stat Points",
}
ARM_SHORT = {
    "stored": "Stored",
    "redrawn": "Redrawn same-species",
    "random_legal": "Random legal",
    "zeroed": "Zeroed",
}
LABELER_NAMES = {
    "foul_play": "Foul Play deterministic argmax",
    "behaviour_cloning": "Behaviour cloning",
}


# ---------------------------------------------------------------- Stat-Point transforms
def random_legal_spread(rng):
    """Uniform random legal 66-point allocation across the six stats, each stat in 0..32."""
    n_stats = len(STATS)
    while True:
        cuts = sorted(rng.sample(range(MAX_TOTAL + n_stats - 1), n_stats - 1))
        prev, parts = -1, []
        for c in cuts:
            parts.append(c - prev - 1)
            prev = c
        parts.append(MAX_TOTAL + n_stats - 2 - prev)
        if max(parts) <= MAX_STAT:
            return dict(zip(STATS, parts))


def zeroed_spread():
    return {s: 0 for s in STATS}


def _norm_spread(evs):
    return {s: int(evs.get(s, evs.get(s.lower(), 0))) for s in STATS}


def redraw_spread(species_key, stored_evs, spreads, rng):
    """Draw a real VGCPastes Stat-Point spread for `species_key`, weighted by count, preferring one different from stored."""
    rows = spreads[species_key]
    stored = _norm_spread(stored_evs)
    candidates = [(_norm_spread(r["evs"]), max(1, int(r.get("count", 1)))) for r in rows]
    diff = [(e, w) for e, w in candidates if e != stored]
    pool = diff if diff else candidates
    evs_list, weights = zip(*pool)
    return dict(rng.choices(evs_list, weights=weights, k=1)[0])


def l1_stat_distance(slots_a, slots_b):
    """Sum of |SP_a - SP_b| across all 36 Stat-Point fields, slots paired in order."""
    return sum(abs(int(a["evs"].get(s, 0)) - int(b["evs"].get(s, 0)))
               for a, b in zip(slots_a, slots_b) for s in STATS)


def apply_arm(slots, arm, spreads, rng):
    """Return six slot dicts with identical 48 non-Stat-Point fields (including Stat Alignment) and edited Stat Points."""
    import matchup_db as M
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; expected one of {ARMS}")
    out = []
    for slot in slots:
        evs = _norm_spread(slot["evs"])
        if arm == "stored":
            new_evs = dict(evs)
        elif arm == "zeroed":
            new_evs = zeroed_spread()
        elif arm == "random_legal":
            new_evs = random_legal_spread(rng)
        elif arm == "redrawn":
            new_evs = redraw_spread(M._norm(slot["species"]), evs, spreads, rng)
        out.append({
            "species": slot["species"],
            "item": slot["item"],
            "ability": slot["ability"],
            "nature": slot["nature"],
            "evs": new_evs,
            "moves": list(slot["moves"]),
        })
    return out


def slots_to_paste(slots):
    """Canonical Showdown paste from six slot dicts (matches matchup_db.slot_text without needing /tmp/vgc-pilot)."""
    blocks = []
    for s in slots:
        head = s["species"] + (f" @ {s['item']}" if s["item"] else "")
        lines = [head, f"Ability: {s['ability']}", "Level: 50"]
        ev = " / ".join(f"{s['evs'][k]} {k}" for k in STATS if s["evs"].get(k))
        if ev:
            lines.append(f"EVs: {ev}")
        lines.append(f"{s['nature']} Nature")
        lines += [f"- {m}" for m in s["moves"]]
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks) + "\n"


# ---------------------------------------------------------------- base teams & opponents
def eligible_meta_entries(meta_entries, spreads, held_grids=frozenset()):
    """Distinct canonical top-50 meta teams whose six species all appear in `spreads` and that are not held out."""
    import matchup_db as M
    seen_ids, out = set(), []
    for entry in sorted(meta_entries, key=lambda e: (e["team_id"], e["id"])):
        if entry["team_id"] in seen_ids:
            continue
        seen_ids.add(entry["team_id"])
        if entry.get("grid_hash") in held_grids:
            continue
        if all(M._norm(s["species"]) in spreads for s in entry["slots"]):
            out.append(entry)
    return out


def draw_base_teams(eligible, n=N_BASE, seed=SEED):
    """Draw `n` base teams without replacement from sorted `eligible`, returned sorted by team_id."""
    ordered = sorted(eligible, key=lambda e: (e["team_id"], e["id"]))
    picked = random.Random(seed).sample(ordered, n)
    return sorted(picked, key=lambda e: (e["team_id"], e["id"]))


def opponents_for_base(base_id, meta_weights):
    """All distinct top-50 meta opponents except `base_id` itself, preserving `team_set` weights."""
    return {tid: w for tid, w in meta_weights.items() if tid != base_id}


def seed_top50(con):
    """Insert all top-50 meta teams into `con` (private DB) with exact `team_set` weights; returns meta entries & weights."""
    import matchup_db as M
    import top50 as T50
    from showdown_validator import Validator

    raw = json.loads(Path(T50.TOP50).read_text())
    entries_by_tid = {}
    val = Validator()
    try:
        with con:
            con.execute("DELETE FROM team_set WHERE set_name='top50'")
            for item in raw:
                paste = Path(T50.resolve(item["id"])).read_text()
                tid, _ = M.add_team(con, paste, "vgcpastes", val)
                con.execute(
                    "INSERT INTO team_set VALUES ('top50', ?, 1) "
                    "ON CONFLICT (set_name, team_id) DO UPDATE SET weight = weight + 1",
                    (tid,),
                )
                if tid not in entries_by_tid:
                    canon, ghash, legal, err = con.execute(
                        "SELECT paste, grid_hash, legal, legal_error FROM team WHERE team_id=?", (tid,)
                    ).fetchone()
                    if not legal:
                        raise RuntimeError(f"top-50 team {item['id']} failed validation: {err}")
                    entries_by_tid[tid] = {
                        "id": item["id"],
                        "team_id": tid,
                        "slots": M.canonical_slots(canon),
                        "grid_hash": ghash,
                    }
    finally:
        val.close()
    weights = dict(con.execute("SELECT team_id, weight FROM team_set WHERE set_name='top50' ORDER BY team_id"))
    return list(entries_by_tid.values()), weights


def sample(con, out=OUT, n=N_BASE, seed=SEED, per_cell=PER_CELL, spreads_path=SPREADS_PATH):
    """Draw the 20 human top-50 base teams and register all 80 Stat-Point variants once; reuse if already saved."""
    import heldout as H
    import matchup_db as M
    from showdown_validator import Validator

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "spec.json"
    if path.exists():
        saved = json.loads(path.read_text())
        if (saved["seed"], saved["n_base"], saved["per_cell"]) != (seed, n, per_cell):
            raise ValueError(f"{path} holds seed={saved['seed']} n={saved['n_base']} per_cell={saved['per_cell']}")
        return saved

    meta_entries, meta_weights = seed_top50(con)
    held_grids = H.identities()
    spreads = json.loads(Path(spreads_path).read_text())
    elig = eligible_meta_entries(meta_entries, spreads, held_grids=held_grids)
    picked = draw_base_teams(elig, n=n, seed=seed)

    val = Validator()
    bases = []
    try:
        for i, entry in enumerate(picked):
            base_tid = entry["team_id"]
            base_slots = entry["slots"]
            base_grid = entry["grid_hash"]
            arms_info = {}
            for a_idx, arm in enumerate(ARMS):
                rng = random.Random(seed * 10000 + i * 10 + a_idx)
                edited = apply_arm(base_slots, arm, spreads, rng)
                paste = slots_to_paste(edited)
                vid, _ = M.add_team(con, paste, f"sp448:{arm}", val)
                vgrid, legal, err = con.execute(
                    "SELECT grid_hash, legal, legal_error FROM team WHERE team_id=?", (vid,)
                ).fetchone()
                if not legal:
                    raise RuntimeError(f"{entry['id']} arm {arm} failed validation: {err}")
                if vgrid != base_grid:
                    raise RuntimeError(f"{entry['id']} arm {arm} changed non-Stat-Point fields (grid_hash mismatch)")
                if arm != "stored" and vid == base_tid:
                    raise RuntimeError(f"{entry['id']} arm {arm} produced the same paste as stored")
                arms_info[arm] = {
                    "team_id": vid,
                    "l1_distance": l1_stat_distance(base_slots, edited),
                    "total_sp": sum(sum(s["evs"].values()) for s in edited),
                }
            bases.append({
                "mb_id": entry["id"],
                "base_team_id": base_tid,
                "species": [s["species"] for s in base_slots],
                "arms": arms_info,
            })
    finally:
        val.close()

    saved = {
        "issue": 448,
        "seed": seed,
        "n_base": n,
        "per_cell": per_cell,
        "eligible_meta_count": len(elig),
        "total_distinct_meta": len(meta_weights),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "meta_weights": {str(k): float(v) for k, v in sorted(meta_weights.items())},
        "bases": bases,
    }
    path.write_text(json.dumps(saved, indent=1) + "\n")
    return saved


# ---------------------------------------------------------------- battles
def rust_runner(out=OUT, workers=8):
    import rust_matchup_runner as R
    return R.RustMatchupRunner("rust_foulplay_det", samples=1, iterations=30000, workers=workers,
                               evidence=Path(out) / "evidence")


def bc_policy_id(con):
    import matchup_db as M
    pid = M.get_policy(con, note="issue 448 behaviour cloning, slots shuffled")
    row, col = con.execute("SELECT row_policy, col_policy FROM policy WHERE policy_id=?", (pid,)).fetchone()
    if not row.startswith("bc_100.zip:") or "+slots-shuffled" not in row or row != col:
        raise ValueError(f"policy {pid} ({row}, {col}) is not shuffled-slot behaviour cloning")
    return pid


def battle_spec(con, spec, policy_id, runner=None, label="", log=print):
    """Run all missing cells for every base team's four arms against its 48 non-self meta opponents."""
    import matchup_db as M
    meta_weights = {int(k): float(v) for k, v in spec["meta_weights"].items()}
    per_cell = spec["per_cell"]
    total = 0
    for idx, base in enumerate(spec["bases"], 1):
        opps = sorted(opponents_for_base(base["base_team_id"], meta_weights))
        rows = [base["arms"][arm]["team_id"] for arm in ARMS]
        t0 = time.perf_counter()
        ran = M.ensure(
            con, policy_id, rows, opps, per_cell,
            seed=spec["seed"] + idx, runner=runner,
            note=f"issue 448 {label} {base['mb_id']}",
        )
        total += ran
        log(f"[{label}] base {idx:2d}/{len(spec['bases'])} ({base['mb_id']}): {ran} battles in {time.perf_counter() - t0:.1f}s")
    return total


def smoke(out=OUT, workers=2):
    """Live smoke check in a separate SQLite DB: 1 meta base team (stored + zeroed) x 2 opponents x 2 battles on both labelers."""
    import heldout as H
    import matchup_db as M

    folder = Path(out) / "smoke"
    folder.mkdir(parents=True, exist_ok=True)
    db_path = folder / "smoke.sqlite"
    if db_path.exists():
        db_path.unlink()
    con = M.connect(db_path)
    try:
        meta_entries, _ = seed_top50(con)
        spreads = json.loads(SPREADS_PATH.read_text())
        elig = eligible_meta_entries(meta_entries, spreads, held_grids=H.identities())
        b0 = elig[0]
        opps = [e["team_id"] for e in elig[1:3]]
        z_slots = apply_arm(b0["slots"], "zeroed", spreads, random.Random(SEED))
        from showdown_validator import Validator
        val = Validator()
        try:
            z_tid, _ = M.add_team(con, slots_to_paste(z_slots), "smoke:zeroed", val)
        finally:
            val.close()
        rows = [b0["team_id"], z_tid]

        fp_runner = rust_runner(folder, workers=workers)
        fp_pid = fp_runner.register(con)
        n_fp = M.ensure(con, fp_pid, rows, opps, 2, seed=SEED, runner=fp_runner, note="issue 448 smoke fp")
        if n_fp != 8 or fp_runner.last["errors"] or fp_runner.last["fallbacks"]:
            raise RuntimeError(f"Foul Play smoke failed: {n_fp} battles, {fp_runner.last}")

        bc_pid = bc_policy_id(con)
        n_bc = M.ensure(con, bc_pid, rows, opps, 2, seed=SEED, note="issue 448 smoke bc")
        if n_bc != 8:
            raise RuntimeError(f"Behaviour cloning smoke failed: {n_bc} battles")
        return {"foul_play_battles": n_fp, "foul_play_last": fp_runner.last, "behaviour_cloning_battles": n_bc}
    finally:
        con.close()


# ---------------------------------------------------------------- analysis & reporting
def directed_cells(con, policy_id, rows, cols):
    out = {}
    for r in rows:
        for c in cols:
            if r == c:
                continue
            lo, hi = (r, c) if r < c else (c, r)
            got = con.execute(
                "SELECT lo_wins, hi_wins, ties FROM cell WHERE policy_id=? AND lo=? AND hi=?",
                (policy_id, lo, hi),
            ).fetchone()
            if got and sum(got):
                out[(r, c)] = (got[0], got[1], got[2]) if r == lo else (got[1], got[0], got[2])
    return out


def weighted_win_rate(cells, team_id, opp_weights):
    """Meta-weighted win rate and unbiased within-cell battle-noise SE over `opp_weights`."""
    total_w = sum(opp_weights.values())
    mean = 0.0
    var_sum = 0.0
    for opp_id, w in opp_weights.items():
        rec = cells.get((team_id, opp_id))
        if rec is None:
            raise KeyError(f"missing cell ({team_id}, {opp_id})")
        wins, losses, ties = rec
        n = wins + losses + ties
        p = (wins + 0.5 * ties) / n
        mean += w * p
        if n > 1:
            s2 = (wins * (1.0 - p) ** 2 + losses * (0.0 - p) ** 2 + ties * (0.5 - p) ** 2) / (n - 1)
            var_sum += (w ** 2) * (s2 / n)
        else:
            var_sum += (w ** 2) * (p * (1.0 - p))
    return mean / total_w, math.sqrt(var_sum) / total_w


def paired_bootstrap_ci(values, seed=SEED, boot=BOOTSTRAP):
    xs = list(values)
    n = len(xs)
    if n == 0:
        return None
    rng = random.Random(seed)
    means = []
    for _ in range(boot):
        means.append(sum(xs[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    return [means[int(0.025 * boot)], means[int(0.975 * boot) - 1]]


def pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    return sxy / math.sqrt(sxx * syy) if sxx > 0 and syy > 0 else None


def held_out_battled(con, policy_id):
    import heldout as H
    held = sorted(H.team_ids(con))
    if not held:
        return 0
    ph = ",".join("?" * len(held))
    return con.execute(
        f"SELECT COUNT(*) FROM cell WHERE policy_id=? AND (lo IN ({ph}) OR hi IN ({ph}))",
        [policy_id, *held, *held],
    ).fetchone()[0]


def _mean_sd(xs):
    m = sum(xs) / len(xs)
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) if len(xs) > 1 else 0.0
    return m, sd


def _incomplete_cells(cells_by_lab, spec, meta_weights, per_cell):
    incomplete = {}
    for lab, cells in cells_by_lab.items():
        miss = 0
        for b in spec["bases"]:
            opps = opponents_for_base(b["base_team_id"], meta_weights)
            for arm in ARMS:
                tid = b["arms"][arm]["team_id"]
                for opp in opps:
                    if sum(cells.get((tid, opp), (0, 0, 0))) < per_cell:
                        miss += 1
        incomplete[lab] = miss
    return incomplete


def _team_arm_scores(cells, base, opps):
    arm_scores = {}
    for arm in ARMS:
        tid = base["arms"][arm]["team_id"]
        wr, se = weighted_win_rate(cells, tid, opps)
        arm_scores[arm] = {"win_rate": wr, "battle_noise_se": se}
    stored_wr = arm_scores["stored"]["win_rate"]
    stored_se = arm_scores["stored"]["battle_noise_se"]
    for arm in ARMS:
        d = arm_scores[arm]["win_rate"] - stored_wr
        cse = 0.0 if arm == "stored" else math.hypot(arm_scores[arm]["battle_noise_se"], stored_se)
        arm_scores[arm]["delta_vs_stored"] = d
        arm_scores[arm]["contrast_se"] = cse
    return arm_scores


def _arm_summary(per_team, lab, arm, boot):
    wrs = [t[lab][arm]["win_rate"] for t in per_team]
    deltas = [t[lab][arm]["delta_vs_stored"] for t in per_team]
    cses = [t[lab][arm]["contrast_se"] for t in per_team]
    l1s = [t["l1_distance"][arm] for t in per_team]
    wm, wsd = _mean_sd(wrs)
    dm, dsd = _mean_sd(deltas)
    entry = {
        "reader_name": ARM_NAMES[arm],
        "win_rate": {"mean": wm, "sd": wsd, "ci95": paired_bootstrap_ci(wrs, SEED + 1, boot)},
        "mean_l1_distance": sum(l1s) / len(l1s),
        "mean_battle_noise_se": sum(t[lab][arm]["battle_noise_se"] for t in per_team) / len(per_team),
    }
    if arm != "stored":
        entry["delta_vs_stored"] = {
            "mean": dm,
            "sd": dsd,
            "ci95": paired_bootstrap_ci(deltas, SEED + 2, boot),
            "mean_contrast_se": sum(cses) / len(cses),
            "teams_dropped": sum(1 for d in deltas if d < 0),
            "teams_dropped_gt_2se": sum(1 for d, s in zip(deltas, cses) if s > 0 and d < -2.0 * s),
            "teams_gained": sum(1 for d in deltas if d > 0),
        }
    return entry


def _labeler_gap(per_team, arm, boot):
    fp_d = [t["foul_play"][arm]["delta_vs_stored"] for t in per_team]
    bc_d = [t["behaviour_cloning"][arm]["delta_vs_stored"] for t in per_team]
    gaps = [f - b for f, b in zip(fp_d, bc_d)]
    gm, gsd = _mean_sd(gaps)
    bc_m = sum(bc_d) / len(bc_d)
    fp_m = sum(fp_d) / len(fp_d)
    gap = {
        "reader_name": ARM_NAMES[arm],
        "mean": gm,
        "sd": gsd,
        "ci95": paired_bootstrap_ci(gaps, SEED + 3, boot),
        "ratio_fp_over_bc": None if abs(bc_m) < 1e-6 else fp_m / bc_m,
        "teams_fp_dropped_more": sum(1 for g in gaps if g < 0),
    }
    return gap, pearson(fp_d, bc_d)


def build_report(con, spec, fp_pid, bc_pid, boot=BOOTSTRAP):
    meta_weights = {int(k): float(v) for k, v in spec["meta_weights"].items()}
    per_cell = spec["per_cell"]
    all_meta = sorted(meta_weights)
    all_variants = [b["arms"][arm]["team_id"] for b in spec["bases"] for arm in ARMS]

    policies = {"foul_play": fp_pid, "behaviour_cloning": bc_pid}
    cells_by_lab = {lab: directed_cells(con, pid, all_variants, all_meta) for lab, pid in policies.items()}
    incomplete = _incomplete_cells(cells_by_lab, spec, meta_weights, per_cell)

    per_team = []
    for b in spec["bases"]:
        opps = opponents_for_base(b["base_team_id"], meta_weights)
        row = {
            "mb_id": b["mb_id"],
            "base_team_id": b["base_team_id"],
            "species": b.get("species", []),
            "l1_distance": {arm: b["arms"][arm]["l1_distance"] for arm in ARMS},
            "total_sp": {arm: b["arms"][arm]["total_sp"] for arm in ARMS},
        }
        for lab, cells in cells_by_lab.items():
            row[lab] = _team_arm_scores(cells, b, opps)
        per_team.append(row)

    arms_summary = {
        lab: {arm: _arm_summary(per_team, lab, arm, boot) for arm in ARMS}
        for lab in ("foul_play", "behaviour_cloning")
    }

    labeler_gap = {}
    team_drop_corr = {}
    for arm in EDITED_ARMS:
        labeler_gap[arm], team_drop_corr[arm] = _labeler_gap(per_team, arm, boot)

    held = held_out_battled(con, fp_pid) + held_out_battled(con, bc_pid)
    battles_total = {
        lab: con.execute("SELECT COALESCE(SUM(battles), 0) FROM cell WHERE policy_id=?", (pid,)).fetchone()[0]
        for lab, pid in policies.items()
    }
    return {
        "issue": 448,
        "seed": spec["seed"],
        "n_base": len(per_team),
        "per_cell": per_cell,
        "policies": policies,
        "battles_in_db": battles_total,
        "incomplete_cells": incomplete,
        "held_out_cells": held,
        "arms": arms_summary,
        "labeler_gap": labeler_gap,
        "team_drop_correlation": team_drop_corr,
        "per_team": per_team,
    }


def _fmt(x, d=3, signed=False):
    if x is None:
        return "n/a"
    return f"{x:+.{d}f}" if signed else f"{x:.{d}f}"


def _ci(c, d=3, signed=False):
    if not c:
        return ""
    return f" [{_fmt(c[0], d, signed)}, {_fmt(c[1], d, signed)}]"


def report_markdown(rep):
    fp = rep["arms"]["foul_play"]
    bc = rep["arms"]["behaviour_cloning"]
    gap = rep["labeler_gap"]
    lines = [
        f"### Stat-Point sensitivity on {rep['n_base']} human top-50 meta teams (issue #448)",
        "",
        "All four Stat-Point arms keep the 48 non-Stat-Point fields (including Stat Alignment) identical. "
        f"Each arm battled the 48 other distinct top-50 meta teams at {rep['per_cell']} battles per opponent "
        "with Pokémon order shuffled every battle.",
        "",
        "| Stat-Point arm | Mean L1 Stat Points moved | Foul Play deterministic argmax win rate (Δ vs Stored [95% CI]) | "
        "Behaviour-cloning win rate (Δ vs Stored [95% CI]) | Labeler gap (Foul Play − behaviour cloning) [95% CI] |",
        "|---|---:|---|---|---|",
    ]
    for arm in ARMS:
        l1 = fp[arm]["mean_l1_distance"]
        fp_wr = _fmt(fp[arm]["win_rate"]["mean"])
        bc_wr = _fmt(bc[arm]["win_rate"]["mean"])
        if arm == "stored":
            fp_col, bc_col, gap_col = f"**{fp_wr}** (control)", f"{bc_wr} (control)", "—"
        else:
            fd = fp[arm]["delta_vs_stored"]
            bd = bc[arm]["delta_vs_stored"]
            g = gap[arm]
            fp_col = f"**{fp_wr}** ({_fmt(fd['mean'], signed=True)}{_ci(fd['ci95'], signed=True)})"
            bc_col = f"{bc_wr} ({_fmt(bd['mean'], signed=True)}{_ci(bd['ci95'], signed=True)})"
            gap_col = f"**{_fmt(g['mean'], signed=True)}**{_ci(g['ci95'], signed=True)}"
        lines.append(f"| {ARM_SHORT[arm]} | {l1:.0f} | {fp_col} | {bc_col} | {gap_col} |")

    lines += [
        "",
        "Team-level consistency (out of "
        f"{rep['n_base']} human meta teams): "
        + "; ".join(
            f"{ARM_SHORT[arm]}: Foul Play deterministic argmax dropped {fp[arm]['delta_vs_stored']['teams_dropped']}/{rep['n_base']} "
            f"({fp[arm]['delta_vs_stored']['teams_dropped_gt_2se']} by >2× battle-noise SE), "
            f"behaviour cloning dropped {bc[arm]['delta_vs_stored']['teams_dropped']}/{rep['n_base']} "
            f"({bc[arm]['delta_vs_stored']['teams_dropped_gt_2se']} by >2× SE)"
            for arm in EDITED_ARMS
        )
        + ".",
        "",
        "Mean per-team battle-noise contrast SE (arm vs Stored): "
        + ", ".join(
            f"{ARM_SHORT[arm]} Foul Play {_fmt(fp[arm]['delta_vs_stored']['mean_contrast_se'])} / "
            f"behaviour cloning {_fmt(bc[arm]['delta_vs_stored']['mean_contrast_se'])}"
            for arm in EDITED_ARMS
        )
        + ".",
        "",
        f"Unique battles recorded: Foul Play deterministic argmax {rep['battles_in_db']['foul_play']:,}, "
        f"behaviour cloning {rep['battles_in_db']['behaviour_cloning']:,}. "
        f"Incomplete cells: {rep['incomplete_cells']}. Held-out (ranks 51–100) cells: {rep['held_out_cells']}.",
    ]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- storyboard ECharts figures
FP_COLOR, BC_COLOR, GAP_COLOR, INK_COLOR = "#2a6fdb", "#8a8a84", "#c2410c", "#55554f"


def _vertical_ci_series(name, xs, los, his, color):
    """Vertical 95% confidence interval whiskers overlaid on a category x-axis."""
    return [
        {
            "type": "line", "name": name, "data": [[x, round(lo, 4)], [x, round(hi, 4)]],
            "symbol": "none", "lineStyle": {"color": color, "width": 2.2},
            "tooltip": {"show": False}, "silent": True,
        }
        for x, lo, hi in zip(xs, los, his)
    ]


def build_storyboard_figures(rep, out_dir):
    """Build the four storyboard ECharts pages cut to one idea each."""
    from echarts_chart import write_chart

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fp = rep["arms"]["foul_play"]
    bc = rep["arms"]["behaviour_cloning"]
    gap = rep["labeler_gap"]
    n = rep["n_base"]

    arm_axis_labels = ["Stored", "Redrawn\nsame-species", "Random\nlegal", "Zeroed"]
    edited_axis_labels = ["Redrawn\nsame-species", "Random\nlegal", "Zeroed"]
    fp_means = [round(fp[a]["win_rate"]["mean"], 4) for a in ARMS]
    bc_means = [round(bc[a]["win_rate"]["mean"], 4) for a in ARMS]
    fp_z = fp["zeroed"]["delta_vs_stored"]["mean"]
    bc_z = bc["zeroed"]["delta_vs_stored"]["mean"]
    fp_r = fp["redrawn"]["delta_vs_stored"]["mean"]
    bc_r = bc["redrawn"]["delta_vs_stored"]["mean"]
    fp_rand = fp["random_legal"]["delta_vs_stored"]["mean"]
    bc_rand = bc["random_legal"]["delta_vs_stored"]["mean"]

    def signed_bar_points(vals, color):
        return [
            {
                "value": round(v, 4),
                "itemStyle": {"color": color},
                "label": {
                    "show": True,
                    "position": "top" if v >= 0 else "bottom",
                    "formatter": f"{v:+.3f}",
                    "fontSize": 11,
                },
            }
            for v in vals
        ]

    # Teaser (0_teaser.html): two-panel summary — left: four-arm ladder; right: paired change vs Stored
    teaser_opt = {
        "title": [
            {"text": "Mean win rate vs 48 meta opponents", "left": "6%", "top": 4,
             "textStyle": {"fontSize": 13, "fontWeight": "normal"}},
            {"text": "Paired change vs Stored Stat Points", "left": "57%", "top": 4,
             "textStyle": {"fontSize": 13, "fontWeight": "normal"}},
        ],
        "legend": {"bottom": 0, "data": [LABELER_NAMES["foul_play"], LABELER_NAMES["behaviour_cloning"]]},
        "tooltip": {"trigger": "axis"},
        "grid": [
            {"left": 48, "width": "42%", "top": 38, "bottom": 66},
            {"left": "58%", "right": 18, "top": 38, "bottom": 66},
        ],
        "xAxis": [
            {"gridIndex": 0, "type": "category", "data": arm_axis_labels, "axisLabel": {"interval": 0, "fontSize": 11}},
            {"gridIndex": 1, "type": "category", "data": edited_axis_labels, "axisLabel": {"interval": 0, "fontSize": 11}},
        ],
        "yAxis": [
            {"gridIndex": 0, "type": "value", "min": 0, "max": 0.65},
            {"gridIndex": 1, "type": "value", "min": -0.32, "max": 0.08},
        ],
        "series": [
            {"name": LABELER_NAMES["foul_play"], "type": "line", "xAxisIndex": 0, "yAxisIndex": 0,
             "data": fp_means, "symbolSize": 8, "lineStyle": {"width": 3, "color": FP_COLOR},
             "itemStyle": {"color": FP_COLOR}, "emphasis": {"focus": "series"}},
            {"name": LABELER_NAMES["behaviour_cloning"], "type": "line", "xAxisIndex": 0, "yAxisIndex": 0,
             "data": bc_means, "symbolSize": 8, "lineStyle": {"width": 3, "color": BC_COLOR},
             "itemStyle": {"color": BC_COLOR}, "emphasis": {"focus": "series"}},
            {"name": LABELER_NAMES["foul_play"], "type": "bar", "xAxisIndex": 1, "yAxisIndex": 1,
             "data": signed_bar_points([fp[a]["delta_vs_stored"]["mean"] for a in EDITED_ARMS], FP_COLOR),
             "emphasis": {"focus": "series"}},
            {"name": LABELER_NAMES["behaviour_cloning"], "type": "bar", "xAxisIndex": 1, "yAxisIndex": 1,
             "data": signed_bar_points([bc[a]["delta_vs_stored"]["mean"] for a in EDITED_ARMS], BC_COLOR),
             "emphasis": {"focus": "series"}},
        ],
    }
    write_chart(
        teaser_opt, out_dir / "0_teaser.html",
        "Both labelers punish broken Stat Points; neither separates real spreads",
        f"{n} human top-50 meta teams, all 48 non-Stat-Point fields fixed, {rep['per_cell']} battles per opponent.",
        f"Paired change vs Stored Stat Points — Redrawn same-species: Foul Play deterministic argmax {fp_r:+.3f}, "
        f"behaviour cloning {bc_r:+.3f}; Random legal: {fp_rand:+.3f} vs {bc_rand:+.3f}; Zeroed: {fp_z:+.3f} vs {bc_z:+.3f} "
        f"(gap {gap['zeroed']['mean']:+.3f} [{gap['zeroed']['ci95'][0]:+.3f}, {gap['zeroed']['ci95'][1]:+.3f}]).",
        height=380,
    )

    # Frame 1 (1_arm_ladder.html): Mean win rate across the four Stat-Point rungs for both labelers
    fp_pts, bc_pts = [], []
    for idx, _ in enumerate(ARMS):
        fv, bv = fp_means[idx], bc_means[idx]
        fp_pos = "top" if fv >= bv else "bottom"
        bc_pos = "bottom" if fv >= bv else "top"
        fp_pts.append({"value": fv, "label": {"show": True, "position": fp_pos, "distance": 10,
                                              "formatter": f"{fv:.3f}", "fontSize": 11, "color": FP_COLOR, "fontWeight": "bold"}})
        bc_pts.append({"value": bv, "label": {"show": True, "position": bc_pos, "distance": 10,
                                              "formatter": f"{bv:.3f}", "fontSize": 11, "color": BC_COLOR}})
    f1_series = [
        {"name": LABELER_NAMES["foul_play"], "type": "line", "data": fp_pts, "symbolSize": 9,
         "lineStyle": {"width": 3, "color": FP_COLOR}, "itemStyle": {"color": FP_COLOR}, "emphasis": {"focus": "series"}},
        {"name": LABELER_NAMES["behaviour_cloning"], "type": "line", "data": bc_pts, "symbolSize": 9,
         "lineStyle": {"width": 3, "color": BC_COLOR}, "itemStyle": {"color": BC_COLOR}, "emphasis": {"focus": "series"}},
    ]
    f1_series += _vertical_ci_series(
        "Foul Play 95% CI", arm_axis_labels,
        [fp[a]["win_rate"]["ci95"][0] for a in ARMS],
        [fp[a]["win_rate"]["ci95"][1] for a in ARMS], FP_COLOR,
    )
    f1_series += _vertical_ci_series(
        "Behaviour cloning 95% CI", arm_axis_labels,
        [bc[a]["win_rate"]["ci95"][0] for a in ARMS],
        [bc[a]["win_rate"]["ci95"][1] for a in ARMS], BC_COLOR,
    )
    write_chart(
        {
            "legend": {"top": 0, "data": [LABELER_NAMES["foul_play"], LABELER_NAMES["behaviour_cloning"]]},
            "tooltip": {"trigger": "axis"},
            "grid": {"left": 64, "right": 32, "top": 42, "bottom": 68},
            "xAxis": {"type": "category", "data": arm_axis_labels, "axisLabel": {"interval": 0, "fontSize": 11},
                      "name": "Stat-Point arm (48 other fields fixed)", "nameLocation": "middle", "nameGap": 46},
            "yAxis": {"type": "value", "name": "mean win rate", "nameLocation": "middle", "nameGap": 44, "min": 0, "max": 0.65},
            "series": f1_series,
        },
        out_dir / "1_arm_ladder.html",
        "Both labelers drop steeply when Stat Points are scrambled or zeroed",
        f"{n} human top-50 meta teams, {rep['per_cell']} battles per opponent. Whiskers show bootstrap 95% intervals across base teams.",
        f"Mean L1 Stat Points moved from Stored: Redrawn same-species {fp['redrawn']['mean_l1_distance']:.0f}, "
        f"Random legal {fp['random_legal']['mean_l1_distance']:.0f}, Zeroed {fp['zeroed']['mean_l1_distance']:.0f}.",
        height=410,
    )

    # Frame 2 (2_labeler_gap.html): Paired deltas vs Stored with 95% CIs, both labelers + difference-in-differences
    rows_f2 = []
    for arm in EDITED_ARMS:
        rows_f2.append((f"{ARM_SHORT[arm]} — Foul Play", fp[arm]["delta_vs_stored"]["mean"], fp[arm]["delta_vs_stored"]["ci95"], FP_COLOR))
        rows_f2.append((f"{ARM_SHORT[arm]} — Behaviour cloning", bc[arm]["delta_vs_stored"]["mean"], bc[arm]["delta_vs_stored"]["ci95"], BC_COLOR))
        rows_f2.append((f"{ARM_SHORT[arm]} — Gap (Foul Play − BC)", gap[arm]["mean"], gap[arm]["ci95"], GAP_COLOR))
    y_names = [r[0] for r in rows_f2][::-1]
    f2_series = [
        {"type": "line", "data": [[0, y_names[0]], [0, y_names[-1]]], "symbol": "none",
         "lineStyle": {"color": INK_COLOR, "type": "dashed", "width": 1.2}, "silent": True, "tooltip": {"show": False}},
    ]
    for label, val, (lo, hi), color in rows_f2:
        f2_series.append({
            "type": "line", "data": [[round(lo, 4), label], [round(hi, 4), label]],
            "symbol": "none", "lineStyle": {"color": color, "width": 3}, "tooltip": {"show": False}, "silent": True,
        })
        f2_series.append({
            "name": label, "type": "scatter", "data": [[round(val, 4), label]],
            "symbolSize": 13, "itemStyle": {"color": color}, "emphasis": {"focus": "series"},
            "label": {"show": True, "position": "top", "distance": 7,
                      "formatter": f"{val:+.3f}", "fontSize": 11, "color": color, "fontWeight": "bold"},
        })
    zero_gap = gap["zeroed"]["mean"]
    write_chart(
        {
            "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"}},
            "grid": {"left": 270, "right": 36, "top": 24, "bottom": 54},
            "xAxis": {"type": "value", "min": -0.32, "max": 0.10,
                      "name": "paired win-rate change vs Stored Stat Points (negative = worse)",
                      "nameLocation": "middle", "nameGap": 34},
            "yAxis": {"type": "category", "data": y_names, "axisTick": {"show": False}},
            "series": f2_series,
        },
        out_dir / "2_labeler_gap.html",
        f"Foul Play penalizes Zeroed Stat Points more ({zero_gap:+.3f}), not Random legal",
        f"Paired across the same {n} human meta teams (95% bootstrap intervals). Negative Gap means Foul Play deterministic argmax drops more than behaviour cloning (BC).",
        " | ".join(f"{ARM_SHORT[a]} gap: {gap[a]['mean']:+.3f} [{gap[a]['ci95'][0]:+.3f}, {gap[a]['ci95'][1]:+.3f}]" for a in EDITED_ARMS),
        height=410,
    )

    # Frame 3 (3_team_drops.html): Team-by-team drops sorted by Foul Play zeroed drop, with ±2x battle-noise SE guides
    ordered = sorted(rep["per_team"], key=lambda t: t["foul_play"]["zeroed"]["delta_vs_stored"])
    team_labels = [t["mb_id"] for t in ordered]
    two_se = round(2.0 * fp["redrawn"]["delta_vs_stored"]["mean_contrast_se"], 4)
    f3_series = [
        {"name": "Foul Play — Zeroed", "type": "bar", "data": [round(t["foul_play"]["zeroed"]["delta_vs_stored"], 4) for t in ordered],
         "itemStyle": {"color": FP_COLOR}, "emphasis": {"focus": "series"}},
        {"name": "Foul Play — Random legal", "type": "bar", "data": [round(t["foul_play"]["random_legal"]["delta_vs_stored"], 4) for t in ordered],
         "itemStyle": {"color": "#6aa0f0"}, "emphasis": {"focus": "series"}},
        {"name": "Foul Play — Redrawn same-species", "type": "bar", "data": [round(t["foul_play"]["redrawn"]["delta_vs_stored"], 4) for t in ordered],
         "itemStyle": {"color": "#b7d1f7"}, "emphasis": {"focus": "series"}},
        {"name": "Behaviour cloning — Zeroed", "type": "line", "data": [round(t["behaviour_cloning"]["zeroed"]["delta_vs_stored"], 4) for t in ordered],
         "symbolSize": 7, "lineStyle": {"width": 2, "color": BC_COLOR, "type": "dashed"}, "itemStyle": {"color": BC_COLOR},
         "emphasis": {"focus": "series"}},
        {"name": "±2× noise SE", "type": "line", "data": [-two_se] * len(team_labels), "symbol": "none",
         "lineStyle": {"width": 1.2, "color": GAP_COLOR, "type": "dotted"}, "silent": True, "tooltip": {"show": False}},
        {"name": "±2× noise SE", "type": "line", "data": [two_se] * len(team_labels), "symbol": "none",
         "lineStyle": {"width": 1.2, "color": GAP_COLOR, "type": "dotted"}, "silent": True, "tooltip": {"show": False}},
    ]
    write_chart(
        {
            "legend": {"top": 0, "data": ["Foul Play — Zeroed", "Foul Play — Random legal",
                                          "Foul Play — Redrawn same-species", "Behaviour cloning — Zeroed"]},
            "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"}},
            "dataZoom": [{"type": "inside"}],
            "grid": {"left": 64, "right": 20, "top": 58, "bottom": 70},
            "xAxis": {"type": "category", "data": team_labels, "axisLabel": {"interval": 0, "rotate": 35, "fontSize": 11},
                      "name": "human top-50 meta team (VGCPastes ID, sorted by Foul Play Zeroed drop)",
                      "nameLocation": "middle", "nameGap": 48},
            "yAxis": {"type": "value", "name": "Δ win rate vs Stored", "nameLocation": "middle", "nameGap": 46},
            "series": f3_series,
        },
        out_dir / "3_team_drops.html",
        f"{fp['zeroed']['delta_vs_stored']['teams_dropped_gt_2se']}/{n} human teams drop below noise on Zeroed; "
        f"{fp['redrawn']['delta_vs_stored']['teams_dropped_gt_2se']}/{n} on Redrawn same-species",
        f"Dotted orange lines mark ±2× mean Redrawn contrast SE (±{two_se:.3f}); per-team counts use each team's own contrast SE "
        f"(Zeroed {fp['zeroed']['delta_vs_stored']['teams_dropped_gt_2se']}/{n}, "
        f"Random legal {fp['random_legal']['delta_vs_stored']['teams_dropped_gt_2se']}/{n}, "
        f"Redrawn same-species {fp['redrawn']['delta_vs_stored']['teams_dropped_gt_2se']}/{n}).",
        f"Observed across-team SD of Redrawn same-species Δ: Foul Play {fp['redrawn']['delta_vs_stored']['sd']:.3f} vs battle-noise SE "
        f"{fp['redrawn']['delta_vs_stored']['mean_contrast_se']:.3f}; behaviour cloning {bc['redrawn']['delta_vs_stored']['sd']:.3f} "
        f"vs {bc['redrawn']['delta_vs_stored']['mean_contrast_se']:.3f}.",
        height=440,
    )


# ---------------------------------------------------------------- CLI
def main(argv=None):
    import matchup_db as M

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--n-base", type=int, default=N_BASE)
    ap.add_argument("--per-cell", type=int, default=PER_CELL)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("sample", "smoke", "battle-bc", "battle-fp", "report"):
        sub.add_parser(name)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    say = lambda m: print(f"{time.strftime('%m-%d %H:%M:%S')} {m}", flush=True)

    if a.cmd == "smoke":
        res = smoke(out, workers=min(a.workers, 2))
        say(f"smoke passed: {json.dumps(res)}")
        return

    db_path = out / "battles.sqlite"
    con = M.connect(db_path)
    try:
        spec = sample(con, out=out, n=a.n_base, seed=SEED, per_cell=a.per_cell)
        if a.cmd == "sample":
            say(f"sampled {len(spec['bases'])} human meta teams from {spec['eligible_meta_count']} eligible: {out / 'spec.json'}")
            return

        if a.cmd == "battle-bc":
            pid = bc_policy_id(con)
            ran = battle_spec(con, spec, pid, runner=None, label="bc", log=say)
            say(f"behaviour cloning complete: {ran} battles run (policy {pid})")
            return

        if a.cmd == "battle-fp":
            runner = rust_runner(out=out, workers=a.workers)
            pid = runner.register(con)
            pinned = out / "fp_policy_id"
            if pinned.exists() and int(pinned.read_text()) != pid:
                raise SystemExit(f"Rust inputs changed: policy {pid} now, {pinned.read_text().strip()} when run began")
            pinned.write_text(f"{pid}\n")
            ran = battle_spec(con, spec, pid, runner=runner, label="fp", log=say)
            say(f"Foul Play deterministic argmax complete: {ran} battles run (policy {pid})")
            return

        if a.cmd == "report":
            pinned = out / "fp_policy_id"
            fp_pid = int(pinned.read_text()) if pinned.exists() else rust_runner(out=out, workers=a.workers).register(con)
            bc_pid = bc_policy_id(con)
            rep = build_report(con, spec, fp_pid, bc_pid)
            (out / "report.json").write_text(json.dumps(rep, indent=1) + "\n")
            md = report_markdown(rep)
            (out / "report.md").write_text(md)
            build_storyboard_figures(rep, out / "storyboard")
            print(md)
    finally:
        con.close()


if __name__ == "__main__":
    main()
