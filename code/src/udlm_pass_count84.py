"""Pass-count selection and tabulation for Uniform, no filter at guidance 1 (issue #277, map #270).

Evaluates Uniform, no filter (udlm_loglinear_nofilter) across K in {84, 128, 256} reverse passes
at guidance 1 (asks 1/24 to 17/24, 3 sampling seeds, 64 validator-accepted teams per cell, no battles):
  - First-try legality (Showdown validator) and rejection-reason breakdown;
  - Simultaneous-update diagnostics (mean_fields_changed_per_pass, duplicate_species_pass_share,
    final_duplicate_species_share) and per-field-type revision numbers (value_changes_beyond_first,
    theoretical noise floor, changes_of_mind, confident_changes_of_mind, phase split);
  - Every metric in diversity_metrics.py (categorical_48, with_spreads, composition) both within-ask
    (primary diversity axis) and pooled across asks 1/24..17/24, plus mean species replacement distance,
    training-line-up swap counts, and 5-team-subset effective species;
  - Pre-registered selection rule (docs/uniform-discrete-diffusion.md, Plan §1): smallest K* in
    {84, 128, 256} within seed noise of K = 256 on first-try legality and within-ask diversity, plus
    the < 5.0% legality floor short-circuit check;
  - Two ECharts storyboard figures via echarts_chart.write_chart.

Pure analysis, selection-rule, and figure-building helpers load no /tmp/vgc-pilot or ~/vgc-data
modules so CI tests run cleanly on CPU.

Usage (from code/):
    /tmp/vgc-pilot/.venv/bin/python src/udlm_pass_count84.py
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import diversity_metrics as DM
from echarts_chart import write_chart
import uniform_math84 as UMath
import uniform_preflight84 as UP

MODEL_ID = "udlm_loglinear_nofilter"
ARM_READER = "Uniform, no filter"
PASS_COUNTS = (84, 128, 256)
SCORED_ASKS = tuple(range(1, 18))
SEEDS = (1, 2, 3)
KEEP_PER_CELL = 64
MAX_TRIES = 1280
LEGALITY_FLOOR = 0.05
LEGALITY_TOL = 0.02
MATCHED_SAMPLE_SEED = 438

RES_DIR = Path.home() / "vgc-data/results"
SUMMARY_PATH = RES_DIR / "udlm_pass_count_g1.json"
FIG_DIR = HERE.parent / "docs/figures/udlm_pass_count_g1"
TABLE_NAME = UP.TABLE_NAME

FIELD_LABELS = {
    "species": "Species",
    "ability": "Ability",
    "item": "Item",
    "move": "Move",
    "nature": "Stat Alignment",
    "stat": "Stat Points",
}
PASS_COLORS = {84: "#2f6f9f", 128: "#0f766e", 256: "#c2410c"}
FIELD_METRIC_KEYS = ("unique_fraction", "repeat_fraction", "novel_draw_fraction", "novel_unique_fraction")
REP_KEYS = ("categorical_48", "with_spreads", "composition")


def grid_path(passes, res_dir=RES_DIR, model=MODEL_ID):
    """Default ask_grid84 results path for Uniform, no filter at guidance 1 and explicit pass count K."""
    return Path(res_dir) / f"stat_points_84_{model}_g1_k{int(passes)}.json"


def _seed_stats(values):
    arr = np.asarray(values, dtype=float)
    mean = float(np.mean(arr))
    sd = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    se = sd / math.sqrt(len(arr)) if len(arr) > 1 else 0.0
    return dict(
        by_seed=[float(v) for v in arr],
        mean=mean,
        sd=sd,
        se=se,
        min=float(np.min(arr)),
        max=float(np.max(arr)),
    )


def _weighted_mean(pairs):
    valid = [(val, weight) for val, weight in pairs if weight > 0 and math.isfinite(val)]
    total_w = sum(weight for _, weight in valid)
    if total_w <= 0:
        return float("nan")
    return float(sum(val * weight for val, weight in valid) / total_w)


def mean_species_replacement_distance(compositions):
    """Mean 6 - shared_species across all unordered distinct pairs of six-species compositions."""
    return UP._mean_species_replacement_distance(compositions)


def _normalize_slot(slot):
    evs_in = slot.get("evs") or {}
    evs = {stat: int(evs_in.get(stat, 0)) for stat in DM.STATS}
    return dict(
        species=str(slot["species"]),
        ability=str(slot.get("ability") or ""),
        item=str(slot.get("item") or ""),
        nature=str(slot.get("nature") or ""),
        moves=[str(m) for m in slot.get("moves", []) if m],
        evs=evs,
    )


def _extract_diversity_scalars(measured):
    out = {}
    for rep in REP_KEYS:
        out[rep] = {k: measured[rep][k] for k in FIELD_METRIC_KEYS}
    out["composition"]["largest_share"] = measured["composition"]["largest_share"]
    return out


def _aggregate_within_ask_diversity(ask_records):
    out = {}
    for rep in REP_KEYS:
        keys = FIELD_METRIC_KEYS + (("largest_share",) if rep == "composition" else ())
        out[rep] = {}
        for key in keys:
            vals = [rec[rep][key] for rec in ask_records if rec[rep][key] is not None]
            out[rep][key] = float(np.mean(vals)) if vals else None
    return out


def _summarize_nested_diversity(seed_dicts):
    out = {}
    for rep in REP_KEYS:
        keys = FIELD_METRIC_KEYS + (("largest_share",) if rep == "composition" else ())
        out[rep] = {}
        for key in keys:
            vals = [d[rep][key] for d in seed_dicts if d is not None and d[rep][key] is not None]
            out[rep][key] = _seed_stats(vals) if vals else None
    return out


def _aggregate_revisions_for_cells(cells, bucket):
    revs = [c["revisions"][bucket] for c in cells if isinstance(c.get("revisions"), dict) and bucket in c["revisions"]]
    if not revs:
        return None
    weights = [int(r.get("teams", 0)) for r in revs]
    by_field = {}
    for field_key in UMath.FIELD_KEYS:
        by_phase = {
            phase: _weighted_mean(
                [(r["by_field"][field_key]["changes_of_mind_by_phase"][phase], w) for r, w in zip(revs, weights)]
            )
            for phase in UMath.PHASE_NAMES
        }
        by_field[field_key] = dict(
            vocab_size=int(revs[0]["by_field"][field_key]["vocab_size"]),
            noise_floor=float(revs[0]["by_field"][field_key]["noise_floor"]),
            value_changes_beyond_first=_weighted_mean(
                [(r["by_field"][field_key]["value_changes_beyond_first"], w) for r, w in zip(revs, weights)]
            ),
            changes_of_mind=_weighted_mean(
                [(r["by_field"][field_key]["changes_of_mind"], w) for r, w in zip(revs, weights)]
            ),
            confident_changes_of_mind=_weighted_mean(
                [(r["by_field"][field_key]["confident_changes_of_mind"], w) for r, w in zip(revs, weights)]
            ),
            changes_of_mind_by_phase=by_phase,
        )
    return dict(
        teams=int(sum(weights)),
        mean_fields_changed_per_pass=_weighted_mean(
            [(r["mean_fields_changed_per_pass"], w) for r, w in zip(revs, weights)]
        ),
        duplicate_species_pass_share=_weighted_mean(
            [(r["duplicate_species_pass_share"], w) for r, w in zip(revs, weights)]
        ),
        final_duplicate_species_share=_weighted_mean(
            [(r["final_duplicate_species_share"], w) for r, w in zip(revs, weights)]
        ),
        by_field=by_field,
    )


def _summarize_revisions_by_seed(seed_revs):
    if not seed_revs or any(r is None for r in seed_revs):
        return None
    by_field = {}
    for field_key in UMath.FIELD_KEYS:
        by_field[field_key] = dict(
            vocab_size=seed_revs[0]["by_field"][field_key]["vocab_size"],
            noise_floor=seed_revs[0]["by_field"][field_key]["noise_floor"],
            value_changes_beyond_first=_seed_stats([r["by_field"][field_key]["value_changes_beyond_first"] for r in seed_revs]),
            changes_of_mind=_seed_stats([r["by_field"][field_key]["changes_of_mind"] for r in seed_revs]),
            confident_changes_of_mind=_seed_stats([r["by_field"][field_key]["confident_changes_of_mind"] for r in seed_revs]),
            changes_of_mind_by_phase={
                phase: _seed_stats([r["by_field"][field_key]["changes_of_mind_by_phase"][phase] for r in seed_revs])
                for phase in UMath.PHASE_NAMES
            },
        )
    return dict(
        teams_total=int(sum(r["teams"] for r in seed_revs)),
        mean_fields_changed_per_pass=_seed_stats([r["mean_fields_changed_per_pass"] for r in seed_revs]),
        duplicate_species_pass_share=_seed_stats([r["duplicate_species_pass_share"] for r in seed_revs]),
        final_duplicate_species_share=_seed_stats([r["final_duplicate_species_share"] for r in seed_revs]),
        by_field=by_field,
    )


def _swaps_from_compositions(compositions, train_lineups):
    """Species swaps from closest training line-up (0 = copied, 1, 2, 3+); empty dict if no reference."""
    if not compositions or not train_lineups:
        return None
    names = sorted({sp for lu in train_lineups for sp in lu} | {sp for lu in compositions for sp in lu})
    idx = {sp: i for i, sp in enumerate(names)}
    train_sorted = sorted(train_lineups)
    t_mat = np.zeros((len(train_sorted), len(names)), dtype=np.float32)
    for row_i, lu in enumerate(train_sorted):
        t_mat[row_i, [idx[sp] for sp in lu]] = 1.0
    v_mat = np.zeros((len(compositions), len(names)), dtype=np.float32)
    for row_i, lu in enumerate(compositions):
        v_mat[row_i, [idx[sp] for sp in lu]] = 1.0
    swap_vals = np.concatenate(
        [6 - (v_mat[i : i + 512] @ t_mat.T).max(axis=1) for i in range(0, len(v_mat), 512)]
    ).astype(int)
    return dict(
        copied=float(np.mean(swap_vals == 0)),
        one=float(np.mean(swap_vals == 1)),
        two=float(np.mean(swap_vals == 2)),
        three_or_more=float(np.mean(swap_vals >= 3)),
    )


def _scored_cells_by_seed(grid_data, asks=SCORED_ASKS, seeds=SEEDS):
    by_seed = {s: {} for s in seeds}
    ask_set = set(asks)
    for cell in grid_data.get("cells", {}).values():
        k = int(cell["numerator"])
        s = int(cell["seed"])
        if k in ask_set and s in by_seed:
            by_seed[s][k] = cell
    return by_seed


def _seed_diversity_metrics(seed_cells, parse_paste, reference, train_lineups):
    ask_div_records, ask_srds, ask_eff = [], [], []
    pooled_teams, pooled_comps = [], []
    for cell in seed_cells:
        teams = [_normalize_slot_list(parse_paste(p)) for p in cell.get("pastes", [])]
        comps = [DM.team_keys(t)[2] for t in teams]
        pooled_teams.extend(teams)
        pooled_comps.extend(comps)
        if teams:
            ask_div_records.append(_extract_diversity_scalars(DM.measure(teams, reference)))
        if len(comps) >= 2:
            ask_srds.append(mean_species_replacement_distance(comps))
        if cell.get("div") and cell["div"].get("effective_species") is not None:
            ask_eff.append(float(cell["div"]["effective_species"]))
    within_div = _aggregate_within_ask_diversity(ask_div_records) if ask_div_records else None
    pooled_div = _extract_diversity_scalars(DM.measure(pooled_teams, reference)) if pooled_teams else None
    pooled_srd = mean_species_replacement_distance(pooled_comps) if len(pooled_comps) >= 2 else float("nan")
    swaps = _swaps_from_compositions(pooled_comps, train_lineups)
    return dict(
        within_ask_diversity=within_div,
        pooled_diversity=pooled_div,
        within_ask_srd=float(np.mean(ask_srds)) if ask_srds else float("nan"),
        pooled_srd=float(pooled_srd),
        subset_effective_species=float(np.mean(ask_eff)) if ask_eff else float("nan"),
        swaps=swaps,
        pooled_compositions=pooled_comps,
    )


def _normalize_slot_list(slots):
    return [_normalize_slot(slot) for slot in slots]


def matched_lineup_uniqueness(grids, parse_paste):
    """Compare the same number of teams per ask and seed across all pass counts."""
    cells = {k: _scored_cells_by_seed(grids[k], SCORED_ASKS, SEEDS) for k in PASS_COUNTS}
    by_passes = {k: [] for k in PASS_COUNTS}
    kept_counts = {}
    for seed in SEEDS:
        by_seed = {k: [] for k in PASS_COUNTS}
        for ask in SCORED_ASKS:
            pastes = {k: cells[k][seed][ask]["pastes"] for k in PASS_COUNTS}
            size = min(map(len, pastes.values()))
            kept_counts[f"a{ask:02d}_s{seed}"] = size
            if not size:
                continue
            for k, pool in pastes.items():
                rng = np.random.default_rng([MATCHED_SAMPLE_SEED, seed, ask, k])
                indices = rng.choice(len(pool), size, replace=False)
                keys = {DM.team_keys(_normalize_slot_list(parse_paste(pool[i])))[2] for i in indices}
                by_seed[k].append(len(keys) / size)
        for k, values in by_seed.items():
            if values:
                by_passes[k].append(float(np.mean(values)))
    return {k: _seed_stats(values) if values else None for k, values in by_passes.items()}, kept_counts


def summarize_grid_data(grid_data, *, expected_passes, parse_paste, reference=None, train_lineups=None):
    """Compute all #277 legality, diagnostic, revision, and diversity metrics for one pass-count grid."""
    if int(grid_data.get("passes", -1)) != int(expected_passes):
        raise ValueError(f"expected passes={expected_passes}, got {grid_data.get('passes')!r}")
    if float(grid_data.get("guidance", -1.0)) != 1.0 or bool(grid_data.get("unguided")):
        raise ValueError(f"expected conditional guidance=1.0, got guidance={grid_data.get('guidance')!r}")
    by_seed = _scored_cells_by_seed(grid_data)
    missing = [(k, s) for s in SEEDS for k in SCORED_ASKS if k not in by_seed[s]]
    if missing:
        raise ValueError(f"K={expected_passes}: missing {len(missing)} scored cells (first 5: {missing[:5]})")

    keep_target = int(grid_data.get("keep", KEEP_PER_CELL))
    all_cells = [by_seed[s][k] for s in SEEDS for k in SCORED_ASKS]
    seed_legals, seed_tries, seed_kept, seed_div_list = [], [], [], []
    seed_rev_all, seed_rev_kept = [], []
    reasons = Counter()
    for s in SEEDS:
        cells_s = [by_seed[s][k] for k in SCORED_ASKS]
        tried_s = sum(int(c["tried"]) for c in cells_s)
        rej_s = sum(int(c["rejected"]) for c in cells_s)
        seed_legals.append((tried_s - rej_s) / max(tried_s, 1))
        seed_tries.append(tried_s)
        seed_kept.append(sum(len(c.get("pastes", [])) for c in cells_s))
        seed_rev_all.append(_aggregate_revisions_for_cells(cells_s, "all_tries"))
        seed_rev_kept.append(_aggregate_revisions_for_cells(cells_s, "kept"))
        seed_div_list.append(_seed_diversity_metrics(cells_s, parse_paste, reference, train_lineups))
        for cell in cells_s:
            reasons.update(dict(cell.get("reasons", [])))

    total_tries = int(sum(seed_tries))
    total_rejected = int(sum(int(c["rejected"]) for c in all_cells))
    total_heldout = int(sum(int(c.get("heldout_dropped", 0)) for c in all_cells))
    complete_cells = sum(len(c.get("pastes", [])) >= keep_target for c in all_cells)
    legality_by_ask = {
        str(k): float(
            sum(int(by_seed[s][k]["tried"]) - int(by_seed[s][k]["rejected"]) for s in SEEDS)
            / max(sum(int(by_seed[s][k]["tried"]) for s in SEEDS), 1)
        )
        for k in SCORED_ASKS
    }
    swaps_summary = (
        {
            bucket: _seed_stats([sd["swaps"][bucket] for sd in seed_div_list])
            for bucket in ("copied", "one", "two", "three_or_more")
        }
        if all(sd["swaps"] is not None for sd in seed_div_list)
        else None
    )
    all_comps = [cp for sd in seed_div_list for cp in sd["pooled_compositions"]]
    return dict(
        passes=int(expected_passes),
        reader=f"{int(expected_passes)} passes",
        model=grid_data.get("model"),
        checkpoint=grid_data.get("checkpoint"),
        checkpoint_sha256=grid_data.get("checkpoint_sha256"),
        seed0=int(grid_data["seed0"]),
        keep_target=keep_target,
        max_tries=int(grid_data.get("max_tries", MAX_TRIES)),
        cells_expected=len(SCORED_ASKS) * len(SEEDS),
        cells_complete=int(complete_cells),
        all_cells_complete=bool(complete_cells == len(SCORED_ASKS) * len(SEEDS)),
        teams_kept=int(sum(seed_kept)),
        teams_kept_by_seed=seed_kept,
        total_tries=total_tries,
        total_rejected=total_rejected,
        total_heldout_dropped=total_heldout,
        mean_tries_per_cell=float(total_tries / len(all_cells)),
        max_tries_per_cell=int(max(int(c["tried"]) for c in all_cells)),
        legality=dict(**_seed_stats(seed_legals), pooled=float((total_tries - total_rejected) / max(total_tries, 1))),
        legality_by_ask=legality_by_ask,
        rejection_reasons=reasons.most_common(12),
        revisions_all_tries=_summarize_revisions_by_seed(seed_rev_all),
        revisions_kept=_summarize_revisions_by_seed(seed_rev_kept),
        within_ask_diversity=_summarize_nested_diversity([sd["within_ask_diversity"] for sd in seed_div_list]),
        pooled_diversity=_summarize_nested_diversity([sd["pooled_diversity"] for sd in seed_div_list]),
        within_ask_species_replacement_distance=_seed_stats([sd["within_ask_srd"] for sd in seed_div_list]),
        pooled_species_replacement_distance=_seed_stats([sd["pooled_srd"] for sd in seed_div_list]),
        subset_effective_species=_seed_stats([sd["subset_effective_species"] for sd in seed_div_list]),
        swaps=swaps_summary,
        all_seeds_distinct_lineups=len(set(all_comps)),
    )


def _compare_metric_against_256(vals_k, vals_ref, *, paired, abs_tol=None):
    """Compare 3-seed metric values for K against K=256 using the pre-registered seed-noise rule."""
    arr_k = np.asarray(vals_k, dtype=float)
    arr_ref = np.asarray(vals_ref, dtype=float)
    mean_k = _comparison_mean(arr_k)
    mean_ref = _comparison_mean(arr_ref)
    if mean_k is None or mean_ref is None:
        return dict(
            mean_k=mean_k, mean_256=mean_ref, diff=None, se_diff=None, threshold=abs_tol,
            signs_agree=False, within_threshold=False, within_seed_noise=False, direction=None,
        )
    diff = mean_k - mean_ref
    if paired:
        diffs = arr_k - arr_ref
        se = float(np.std(diffs, ddof=1) / math.sqrt(len(diffs))) if len(diffs) > 1 else 0.0
        signs_agree = bool(np.all(diffs > 0) or np.all(diffs < 0))
    else:
        var_k = float(np.var(arr_k, ddof=1)) if len(arr_k) > 1 else 0.0
        var_ref = float(np.var(arr_ref, ddof=1)) if len(arr_ref) > 1 else 0.0
        se = math.sqrt(var_k / len(arr_k) + var_ref / len(arr_ref))
        # Unpaired seeds: strict range separation (every seed of K lies on the same side of every seed of 256).
        signs_agree = bool(float(np.max(arr_k)) < float(np.min(arr_ref)) or float(np.min(arr_k)) > float(np.max(arr_ref)))
    threshold = float(abs_tol) if abs_tol is not None else float(se)
    within_tol = abs(diff) <= threshold if abs_tol is not None else abs(diff) < threshold
    within_noise = bool(within_tol or not signs_agree)
    return dict(
        mean_k=mean_k,
        mean_256=mean_ref,
        diff=diff,
        se_diff=float(se),
        threshold=threshold,
        signs_agree=signs_agree,
        within_threshold=within_tol,
        within_seed_noise=within_noise,
        direction="above_256" if diff > 0 else "below_256" if diff < 0 else "equal",
    )


def _comparison_mean(values):
    return float(np.mean(values)) if len(values) == len(SEEDS) and np.isfinite(values).all() else None


def select_pass_count(pass_summaries):
    """Apply the pre-registered #277 pass-count selection and legality-floor short-circuit rules.

    Rule (docs/uniform-discrete-diffusion.md, Plan §1), with matched sample sizes for uniqueness:
      - Short-circuit if first-try legality < 5.0% across all three pass counts {84, 128, 256}.
      - Choose the smallest K* in {84, 128, 256} whose 3-seed mean is within seed noise of K = 256 on:
          1. first-try legality (within 2 percentage points, OR 3 seeds do not agree in sign vs 256);
          2. within-ask composition.unique_fraction (< 1 seed SE, OR 3 seeds do not agree in sign vs 256);
          3. within-ask mean species replacement distance (< 1 seed SE, OR 3 seeds do not agree in sign vs 256).
      - Both pass counts need a finite measurement for every seed on each metric.
    """
    ref = pass_summaries[256]
    short_circuit = all(pass_summaries[k]["legality"]["mean"] < LEGALITY_FLOOR for k in PASS_COUNTS)
    comparisons = {}
    chosen = 256
    for k in PASS_COUNTS:
        if k == 256:
            comparisons[str(k)] = dict(passes=256, is_reference=True, within_seed_noise=True)
            continue
        cand = pass_summaries[k]
        paired = cand.get("seed0") == ref.get("seed0")
        leg = _compare_metric_against_256(cand["legality"]["by_seed"], ref["legality"]["by_seed"], paired=paired, abs_tol=LEGALITY_TOL)
        uniq = None
        if cand["matched_lineup_uniqueness"] is not None:
            uniq = _compare_metric_against_256(
                cand["matched_lineup_uniqueness"]["by_seed"], ref["matched_lineup_uniqueness"]["by_seed"], paired=paired,
            )
        srd = _compare_metric_against_256(
            cand["within_ask_species_replacement_distance"]["by_seed"],
            ref["within_ask_species_replacement_distance"]["by_seed"],
            paired=paired,
        )
        all_ok = bool(leg["within_seed_noise"] and uniq and uniq["within_seed_noise"] and srd["within_seed_noise"])
        comparisons[str(k)] = dict(
            passes=k,
            paired_seeds=paired,
            legality=leg,
            within_ask_unique_fraction=uniq,
            within_ask_species_replacement_distance=srd,
            within_seed_noise=all_ok,
        )
        if all_ok and chosen == 256:
            chosen = k
    return dict(
        chosen_passes=int(chosen),
        legality_floor=LEGALITY_FLOOR,
        short_circuit=bool(short_circuit),
        g_max_override="none" if short_circuit else None,
        comparisons_vs_256=comparisons,
    )


def _git_commit_sha():
    frozen_commit = HERE.parent / "COMMIT"
    if frozen_commit.is_file():
        return frozen_commit.read_text().strip()
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=HERE.parent, check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _build_training_reference():
    sys.path.insert(0, "/tmp/vgc-pilot/src")
    sys.path.insert(0, str(HERE))
    import asked_vs_got as X
    import team84 as T

    table_path = X.RES / TABLE_NAME
    table_sha = hashlib.sha256(table_path.read_bytes()).hexdigest()
    data = np.load(table_path)
    mask = (~data["all_val"]) & (~np.isnan(data["all_W"]))
    rows = data["all_X"][mask].astype(np.int64)
    vocab = T.regulation_vocab()
    teams = [T.decode(vocab, row) for row in rows]
    reference = DM.Reference(
        teams,
        scope="generation_training",
        format_id="gen9vgc2026regmb",
        provenance=dict(table=TABLE_NAME, sha256=table_sha, labelled_train_rows=len(rows)),
    )
    train_lineups = set(reference.keys[2])
    return reference, train_lineups, table_sha


def _parse_paste_fn():
    sys.path.insert(0, "/tmp/vgc-pilot/src")
    from corpus import parse_team_text

    return lambda paste: parse_team_text(paste, blank_ends_slot=True)


def evaluate_pass_counts(res_dir=RES_DIR, save_path=SUMMARY_PATH, *, model=MODEL_ID, issue=277):
    """Load the three K in {84, 128, 256} guidance-1 grids, tabulate all metrics, and apply the selection rule."""
    reference, train_lineups, table_sha = _build_training_reference()
    parse_paste = _parse_paste_fn()
    grids = {k: json.loads(grid_path(k, res_dir=res_dir, model=model).read_text()) for k in PASS_COUNTS}
    summaries = {}
    ckpts = set()
    for k in PASS_COUNTS:
        summaries[k] = summarize_grid_data(
            grids[k], expected_passes=k, parse_paste=parse_paste, reference=reference, train_lineups=train_lineups
        )
        ckpts.add(summaries[k]["checkpoint_sha256"])
    if len(ckpts) != 1:
        raise ValueError(f"pass-count grids came from different checkpoints: {sorted(ckpts)}")
    matched, kept_counts = matched_lineup_uniqueness(grids, parse_paste)
    for k, stat in matched.items():
        summaries[k]["matched_lineup_uniqueness"] = stat
    decision = select_pass_count(summaries)
    report = dict(
        issue=issue,
        map_issue=270,
        arm_id=model,
        arm_reader=ARM_READER,
        guidance=1.0,
        scored_asks=list(SCORED_ASKS),
        seeds=list(SEEDS),
        battles=0,
        commit=_git_commit_sha(),
        checkpoint=summaries[84]["checkpoint"],
        checkpoint_sha256=summaries[84]["checkpoint_sha256"],
        table=TABLE_NAME,
        table_sha256=table_sha,
        reference=reference.metadata,
        decision=decision,
        uniqueness_sample_sizes=kept_counts,
        uniqueness_sample_seed=MATCHED_SAMPLE_SEED,
        by_passes={str(k): summaries[k] for k in PASS_COUNTS},
    )
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(json.dumps(report, indent=2) + "\n")
    return report


def _panel_bar_and_seeds(series_name, pass_keys, mean_vals, seed_lists, color_map, axis_idx):
    bars = {
        "type": "bar",
        "name": series_name,
        "xAxisIndex": axis_idx,
        "yAxisIndex": axis_idx,
        "barWidth": "44%",
        "emphasis": {"focus": "series"},
        "label": {"show": True, "position": "top", "distance": 6, "fontSize": 12, "fontWeight": "bold"},
        "data": [
            {"value": round(val, 4) if val is not None else None,
             "itemStyle": {"color": color_map[k], "opacity": 0.85, "borderRadius": [4, 4, 0, 0]}}
            for k, val in zip(pass_keys, mean_vals)
        ],
    }
    dots = {
        "type": "scatter",
        "name": f"{series_name} (3 seeds)",
        "xAxisIndex": axis_idx,
        "yAxisIndex": axis_idx,
        "symbolSize": 7,
        "z": 5,
        "itemStyle": {"color": "#0b0b0b", "borderColor": "#ffffff", "borderWidth": 1.2},
        "tooltip": {"show": False},
        "data": [[idx, round(sv, 4)] for idx, slist in enumerate(seed_lists) for sv in slist],
    }
    return bars, dots


def build_selection_figure(report):
    """Storyboard Figure 1: 3-panel pass-count comparison across the three pre-registered selection criteria."""
    by_p = report["by_passes"]
    chosen = report["decision"]["chosen_passes"]
    labels = [f"{k} passes" for k in PASS_COUNTS]

    leg_means = [100.0 * by_p[str(k)]["legality"]["mean"] for k in PASS_COUNTS]
    leg_seeds = [[100.0 * v for v in by_p[str(k)]["legality"]["by_seed"]] for k in PASS_COUNTS]
    uniq = [by_p[str(k)]["matched_lineup_uniqueness"] for k in PASS_COUNTS]
    uniq_means = [stat["mean"] if stat else None for stat in uniq]
    uniq_seeds = [stat["by_seed"] if stat else [] for stat in uniq]
    srd_means = [by_p[str(k)]["within_ask_species_replacement_distance"]["mean"] for k in PASS_COUNTS]
    srd_seeds = [by_p[str(k)]["within_ask_species_replacement_distance"]["by_seed"] for k in PASS_COUNTS]

    ref_leg = leg_means[-1]
    leg_bar, leg_dots = _panel_bar_and_seeds("First-try legality", PASS_COUNTS, leg_means, leg_seeds, PASS_COLORS, 0)
    leg_bar["label"]["formatter"] = "js:p => p.value.toFixed(1) + '%'"
    leg_bar["markLine"] = {
        "silent": True,
        "symbol": ["none", "none"],
        "lineStyle": {"type": "dashed", "color": "#b45309", "width": 1.4},
        "label": {"formatter": "5% usability floor", "position": "insideEndTop", "fontSize": 10},
        "data": [{"yAxis": 5.0}],
    }
    leg_bar["markArea"] = {
        "silent": True,
        "itemStyle": {"color": "#c2410c", "opacity": 0.10},
        "data": [[{"yAxis": max(0.0, ref_leg - 2.0)}, {"yAxis": min(100.0, ref_leg + 2.0)}]],
    }

    uniq_bar, uniq_dots = _panel_bar_and_seeds("Within-ask line-up uniqueness", PASS_COUNTS, uniq_means, uniq_seeds, PASS_COLORS, 1)
    uniq_bar["label"]["formatter"] = "js:p => p.value.toFixed(3)"
    srd_bar, srd_dots = _panel_bar_and_seeds("Within-ask species replacement", PASS_COUNTS, srd_means, srd_seeds, PASS_COLORS, 2)
    srd_bar["label"]["formatter"] = "js:p => p.value.toFixed(2)"

    lefts, width = ("7%", "39%", "71%"), "24%"
    panel_titles = [
        dict(text="First-try legality (%)", left="19%", top=12, textAlign="center", textStyle=dict(fontSize=13, fontWeight="bold")),
        dict(text="Within-ask line-up uniqueness", left="51%", top=12, textAlign="center", textStyle=dict(fontSize=13, fontWeight="bold")),
        dict(text="Within-ask species replacement", left="83%", top=12, textAlign="center", textStyle=dict(fontSize=13, fontWeight="bold")),
    ]
    grids = [dict(left=lft, width=width, top=50, bottom=44) for lft in lefts]
    x_axes = [dict(type="category", gridIndex=i, data=labels, axisLabel=dict(interval=0, fontSize=11.5)) for i in range(3)]
    y_max_leg = max(30.0, math.ceil((max(leg_means) + 8.0) / 5.0) * 5.0)
    y_axes = [
        dict(type="value", gridIndex=0, min=0, max=y_max_leg, axisLabel=dict(formatter="{value}%")),
        dict(type="value", gridIndex=1, min=0, max=1.05, interval=0.25),
        dict(type="value", gridIndex=2, min=0, max=6.0, interval=1.5),
    ]
    option = dict(
        animation=False,
        title=panel_titles,
        grid=grids,
        tooltip=dict(trigger="axis", axisPointer=dict(type="shadow")),
        xAxis=x_axes,
        yAxis=y_axes,
        series=[leg_bar, leg_dots, uniq_bar, uniq_dots, srd_bar, srd_dots],
    )
    if report["decision"]["short_circuit"]:
        title = f"Uniform, no filter stays below the 5% legality floor at all three pass counts ({leg_means[0]:.1f}%–{leg_means[-1]:.1f}%)"
    else:
        title = (
            f"Pass-count selection locks {chosen} passes for Uniform, no filter at guidance 1 "
            f"(legality {leg_means[0]:.1f}%, {leg_means[1]:.1f}%, {leg_means[2]:.1f}%)"
        )
    subtitle = (
        "Black dots show sampling seeds across asks 1/24 to 17/24. The shaded band marks ±2 percentage points around "
        "256 passes. The dashed line marks the 5% legality floor."
    )
    note = (
        "Selection rule (docs/uniform-discrete-diffusion.md): pick the smallest K in {84, 128, 256} within seed noise of "
        "256 passes on first-try legality (±2 pp or mixed seed signs) and within-ask line-up uniqueness and mean species "
        "replacement distance (< 1 seed standard error or mixed seed signs). Uniqueness uses the same number of "
        "teams per cell at each pass count and drops cells empty at any pass count."
    )
    return option, title, subtitle, note


def build_mechanism_figure(report):
    """Storyboard Figure 2: simultaneous-update diagnostics and per-field changes of mind across K=84, 128, 256."""
    by_p = report["by_passes"]
    labels = [f"{k} passes" for k in PASS_COUNTS]
    changed = [round(by_p[str(k)]["revisions_all_tries"]["mean_fields_changed_per_pass"]["mean"], 2) for k in PASS_COUNTS]
    dup_pass = [round(100.0 * by_p[str(k)]["revisions_all_tries"]["duplicate_species_pass_share"]["mean"], 1) for k in PASS_COUNTS]
    final_dup = [round(100.0 * by_p[str(k)]["revisions_all_tries"]["final_duplicate_species_share"]["mean"], 1) for k in PASS_COUNTS]

    field_names = [FIELD_LABELS[f] for f in UMath.FIELD_KEYS]
    mind_series = [
        {
            "type": "bar",
            "name": f"{k} passes",
            "xAxisIndex": 1,
            "yAxisIndex": 1,
            "itemStyle": {"color": PASS_COLORS[k]},
            "emphasis": {"focus": "series"},
            "data": [
                round(by_p[str(k)]["revisions_all_tries"]["by_field"][f]["changes_of_mind"]["mean"], 2)
                for f in UMath.FIELD_KEYS
            ],
        }
        for k in PASS_COUNTS
    ]
    left_series = [
        {
            "type": "bar",
            "name": "Duplicate species at final pass (%)",
            "xAxisIndex": 0,
            "yAxisIndex": 0,
            "itemStyle": {"color": "#b45309"},
            "emphasis": {"focus": "series"},
            "label": {"show": True, "position": "top", "formatter": "{c}%", "fontSize": 11},
            "data": final_dup,
        },
        {
            "type": "bar",
            "name": "Duplicate species share of passes (%)",
            "xAxisIndex": 0,
            "yAxisIndex": 0,
            "itemStyle": {"color": "#d97706", "opacity": 0.55},
            "emphasis": {"focus": "series"},
            "label": {"show": True, "position": "top", "formatter": "{c}%", "fontSize": 11},
            "data": dup_pass,
        },
        {
            "type": "line",
            "name": "Mean fields changed / pass (right axis)",
            "xAxisIndex": 0,
            "yAxisIndex": 2,
            "color": "#2f6f9f",
            "symbolSize": 8,
            "lineStyle": {"width": 2.5},
            "emphasis": {"focus": "series"},
            "data": changed,
        },
    ]
    option = dict(
        animation=False,
        title=[
            dict(text="Simultaneous updates & duplicate-species collisions", left="25%", top=40, textAlign="center", textStyle=dict(fontSize=13, fontWeight="bold")),
            dict(text="Changes of mind per field (all tries)", left="76%", top=40, textAlign="center", textStyle=dict(fontSize=13, fontWeight="bold")),
        ],
        legend=dict(top=0, itemWidth=16, textStyle=dict(fontSize=11)),
        tooltip=dict(trigger="axis", axisPointer=dict(type="shadow")),
        grid=[dict(left="7%", width="36%", top=80, bottom=52), dict(left="55%", width="41%", top=80, bottom=52)],
        xAxis=[
            dict(type="category", gridIndex=0, data=labels, axisLabel=dict(interval=0, fontSize=11)),
            dict(type="category", gridIndex=1, data=field_names, axisLabel=dict(interval=0, fontSize=10.5, rotate=18)),
        ],
        yAxis=[
            dict(type="value", gridIndex=0, min=0, max=60, name="Duplicate-species share (%)", nameLocation="middle", nameGap=38, axisLabel=dict(formatter="{value}%")),
            dict(type="value", gridIndex=1, min=0, name="Mean top-prediction flips", nameLocation="middle", nameGap=38),
            dict(type="value", gridIndex=0, position="right", min=0, name="Fields / pass", nameLocation="middle", nameGap=34, splitLine=dict(show=False)),
        ],
        series=left_series + mind_series,
    )
    title = (
        f"More passes halve simultaneous edits ({changed[0]:.2f} → {changed[-1]:.2f} fields/pass) "
        f"while final duplicate-species rate moves from {final_dup[0]:.1f}% to {final_dup[-1]:.1f}%"
    )
    subtitle = (
        "Left: simultaneous-update diagnostics averaged across all tries before validator redraws (guidance 1, asks 1/24 to 17/24). "
        "Right: mean top-prediction flips (changes of mind) per column across the six field types."
    )
    note = (
        "Logged inside UniformSampler84 across all reverse passes z^(0)..z^(K) before Showdown validation and before Stat Alignment repair. "
        "Duplicate species share counts teams where at least two of the six slots hold the same species token."
    )
    return option, title, subtitle, note


def build_figures(report, out_dir=FIG_DIR):
    """Write both #277 ECharts storyboard figures to out_dir and return their paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    opt1, title1, sub1, note1 = build_selection_figure(report)
    p1 = write_chart(opt1, out_dir / "01_pass_count_selection.html", title=title1, subtitle=sub1, note=note1, height=420)
    opt2, title2, sub2, note2 = build_mechanism_figure(report)
    p2 = write_chart(opt2, out_dir / "02_simultaneous_updates_and_revisions.html", title=title2, subtitle=sub2, note=note2, height=440)
    return [p1, p2]


def _fmt_mean_se(stat, *, pct=False, digits=3):
    if stat is None:
        return "—"
    if pct:
        return f"{100.0 * stat['mean']:.1f}% ± {100.0 * stat['se']:.1f}%"
    return f"{stat['mean']:.{digits}f} ± {stat['se']:.{digits}f}"


def _fmt_mean(stat, fmt):
    return f"{stat['mean']:{fmt}}" if stat is not None else "Unavailable"


def format_markdown_tables(report):
    """Return reader-facing Markdown tables covering legality, simultaneous-update diagnostics, diversity, and revisions."""
    by_p = report["by_passes"]
    lines = [
        "### Pass-count comparison at guidance 1 (asks 1/24 to 17/24, 3 seeds, no battles)",
        "",
        f"Uniform, no filter (`{report['arm_id']}`) uses checkpoint `{report['checkpoint']}` at guidance 1. Each cell targets 64 accepted teams.",
        "Uniqueness for selection uses the smallest kept count per ask and seed across the three pass counts. "
        "A fixed random seed picks teams without replacement. Cells empty at any pass count are dropped from uniqueness.",
        "",
        "| Pass count | First-try legality | Complete cells | Mean tries / cell | Fields changed / pass | Dup. species pass share | Final dup. species share | Within-ask line-up uniq. | Within-ask species repl. | Pooled novel draw share |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for k in PASS_COUNTS:
        r = by_p[str(k)]
        rev = r["revisions_all_tries"]
        pool_comp = r["pooled_diversity"]["composition"]
        lines.append(
            f"| **{k} passes** | {_fmt_mean_se(r['legality'], pct=True)} | {r['cells_complete']}/{r['cells_expected']} | "
            f"{r['mean_tries_per_cell']:.1f} | {_fmt_mean_se(rev['mean_fields_changed_per_pass'], digits=2)} | "
            f"{_fmt_mean_se(rev['duplicate_species_pass_share'], pct=True)} | "
            f"{_fmt_mean_se(rev['final_duplicate_species_share'], pct=True)} | "
            f"{_fmt_mean_se(r['matched_lineup_uniqueness'], digits=3)} | "
            f"{_fmt_mean_se(r['within_ask_species_replacement_distance'], digits=3)} | "
            f"{_fmt_mean_se(pool_comp['novel_draw_fraction'], pct=True)} |"
        )

    lines += [
        "",
        "### Diversity across all kept teams (within-ask / pooled)",
        "",
        "Measured against the `214,055` labelled training teams of `lossdata84_shuffled_only.npz` (3-seed mean ± SE):",
        "",
        "| Pass count | Cat-48 uniq. (WA / pool) | Spreads uniq. (WA / pool) | Line-up uniq. (WA / pool) | Line-up largest share (WA / pool) | Line-up novel draws (WA / pool) | Line-up novel uniq. (WA / pool) | Species repl. (WA / pool) | 5-team eff. species |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for k in PASS_COUNTS:
        r = by_p[str(k)]
        wa, po = r["within_ask_diversity"], r["pooled_diversity"]
        lines.append(
            f"| **{k} passes** | "
            f"{_fmt_mean(wa['categorical_48']['unique_fraction'], '.3f')} / {_fmt_mean(po['categorical_48']['unique_fraction'], '.3f')} | "
            f"{_fmt_mean(wa['with_spreads']['unique_fraction'], '.3f')} / {_fmt_mean(po['with_spreads']['unique_fraction'], '.3f')} | "
            f"{_fmt_mean(wa['composition']['unique_fraction'], '.3f')} / {_fmt_mean(po['composition']['unique_fraction'], '.3f')} | "
            f"{_fmt_mean(wa['composition']['largest_share'], '.3f')} / {_fmt_mean(po['composition']['largest_share'], '.3f')} | "
            f"{_fmt_mean(wa['composition']['novel_draw_fraction'], '.1%')} / {_fmt_mean(po['composition']['novel_draw_fraction'], '.1%')} | "
            f"{_fmt_mean(wa['composition']['novel_unique_fraction'], '.1%')} / {_fmt_mean(po['composition']['novel_unique_fraction'], '.1%')} | "
            f"{r['within_ask_species_replacement_distance']['mean']:.3f} / {r['pooled_species_replacement_distance']['mean']:.3f} | "
            f"{r['subset_effective_species']['mean']:.2f} |"
        )

    lines += [
        "",
        "### Revision numbers by field type (all tries, before validator redraws)",
        "",
        "Value changes beyond the first (`R_c`), deterministic fixed-predictor noise floor (App. A Eq. 25, `γ ≤ 1`), and changes of mind (`M_c`):",
        "",
        "| Field type | 84 passes (`R_c` / floor / `M_c`) | 128 passes (`R_c` / floor / `M_c`) | 256 passes (`R_c` / floor / `M_c`) |",
        "| --- | ---: | ---: | ---: |",
    ]
    for field_key in UMath.FIELD_KEYS:
        cells = []
        for k in PASS_COUNTS:
            bf = by_p[str(k)]["revisions_all_tries"]["by_field"][field_key]
            cells.append(
                f"{bf['value_changes_beyond_first']['mean']:.2f} / {bf['noise_floor']:.2f} / {bf['changes_of_mind']['mean']:.2f}"
            )
        lines.append(f"| {FIELD_LABELS[field_key]} | {cells[0]} | {cells[1]} | {cells[2]} |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--issue", type=int, default=277)
    parser.add_argument("--results-dir", type=Path, default=RES_DIR)
    parser.add_argument("--summary", type=Path, default=SUMMARY_PATH)
    parser.add_argument("--figure-dir", type=Path, default=FIG_DIR)
    args = parser.parse_args()
    report = evaluate_pass_counts(args.results_dir, args.summary, model=args.model, issue=args.issue)
    pages = build_figures(report, args.figure_dir)
    print(format_markdown_tables(report))
    print()
    print(f"Chosen K*: {report['decision']['chosen_passes']} passes (short_circuit={report['decision']['short_circuit']})")
    for page in pages:
        print(f"Figure: {page}")
    print("UDLM_PASS_COUNT_DONE")


if __name__ == "__main__":
    main()
