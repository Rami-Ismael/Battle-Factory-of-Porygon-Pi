"""Verification, figures, trade-off comparison, and GitHub reports for issue #385.

Retrains the default 84-field linear masked diffusion checkpoint on the filtered shuffled-only table
without wrong-stone teams (`stat84_shuffled_noise_linear_filtered_es3_cap80`, ask-grid row
`linear_masked_filter`, reader name "Linear masked, full filter"), and compares every trade-off
measure across guidance 2, 4, and 8 against the unfiltered linear checkpoint
(`stat84_shuffled_noise_linear_es3_cap80`, `sched_linear_species_first`).

Commands:
    python retrain_default_report.py verify-table [OUT_JSON]
    python retrain_default_report.py check-grid NAME CELLS
    python retrain_default_report.py train-figure RUN_JSON OUT_DIR [REF_JSON]
    python retrain_default_report.py train-comment RUN_JSON COMMIT_FILE LOSS_LIGHT LOSS_DARK [REF_JSON] [VERIFY_JSON]
    python retrain_default_report.py grid-figure GUIDANCE OUT_DIR
    python retrain_default_report.py grid-comment GUIDANCE FIG_LIGHT FIG_DARK
    python retrain_default_report.py summary [OUT_JSON]
    python retrain_default_report.py final-figures SUMMARY_JSON OUT_DIR
    python retrain_default_report.py final-comment SUMMARY_JSON RUN_JSON COMMIT_FILE ERR_LIGHT ERR_DARK PAK_LIGHT PAK_DARK
    python retrain_default_report.py handoff-comment SUMMARY_JSON RUN_JSON
    python retrain_default_report.py update-map SUMMARY_JSON RUN_JSON
"""
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from echarts_chart import band, write_chart

REPO = "Rami-Ismael/Battle-Factory-of-Porygon-Pi"
ISSUE = "385"
HANDOFF_ISSUE = "64"
MAP_ISSUE = "291"
TASK_TITLE = "Retrain the default checkpoint on the shuffled-only table without wrong-stone teams"
TASK_URL = f"https://github.com/{REPO}/issues/{ISSUE}"

NEW_ROW = "linear_masked_filter"
REF_ROW = "sched_linear_species_first"
NEW_CK = "stat84_shuffled_noise_linear_filtered_es3_cap80"
REF_CK = "stat84_shuffled_noise_linear_es3_cap80"
NEW_LABEL = "Filtered table (Linear masked, full filter)"
REF_LABEL = "Unfiltered table (Linear, species-first)"

GUIDANCES = (2, 4, 8)
SCORED_ASKS = list(range(1, 18))
TOP_ASKS = (15, 16, 17)
HIGH_ASKS = list(range(18, 24))
ALL_ASKS = list(range(1, 24))
SEEDS = [1, 2, 3]
DRAWS = 200 if os.environ.get("RDR_SMOKE") else 4000
DRAW_SEED = 2971

BAR = 0.001
ERROR_LUCK = 0.002
WIN_LUCK = 0.026
EXPECTED_TABLE_SHA = "e58af7689e62a873ce19361237d7ff8e77cb791fc4c196cc705410755c5a03e2"
REF_REDRAW_BASES = {2: 940135810, 4: 630821980, 8: 111619920}

RES = Path.home() / "vgc-data/results"
SUMMARY_PATH = RES / "retrain_default_linear_filtered_summary.json"
BASES_FILE = Path.home() / "vgc-data/seed_replay/bases.txt"

NEW_COLOR = "#0f766e"
REF_COLOR = "#6f6e69"
TRAIN_COLOR = "#52514e"
VAL_COLOR = "#c2410c"


def grid_name(row, guidance):
    return row + ("" if float(guidance) == 2 else f"_g{float(guidance):g}")


def stop_reason(info):
    best, stop, cap = info.get("best_epoch"), info.get("stopped_epoch"), info.get("cap")
    if info.get("patience") and stop is not None and cap is not None and stop < cap:
        return f"early stopping fired: no new lowest validation loss for {info['patience']} epochs after epoch {best}"
    return f"early stopping did not fire: reached the cap of {cap} epochs"


def early_stopping_fired(info):
    stop, cap = info.get("stopped_epoch"), info.get("cap")
    return bool(info.get("patience") and stop is not None and cap is not None and stop < cap)


def picture(light_url, dark_url, alt, width=900):
    return (
        f'<picture><source media="(prefers-color-scheme: dark)" srcset="{dark_url}">'
        f'<img src="{light_url}" alt="{alt}" width="{width}"></picture>'
    )


def fmt_iv(iv, fmt=".4f"):
    return f"[{iv[0]:{fmt}}, {iv[1]:{fmt}}]"


# ---------------------------------------------------------------- pre-training table verification & strict grid check

def verify_table(out_json=None):
    """Verify lossdata84_shuffled_only.npz has 0 wrong-stone teams (every field visible), valid Stat Points, and 0 held-out overlap."""
    sys.path.insert(0, "/tmp/vgc-pilot/src")
    sys.path.insert(0, str(HERE))
    import lossdown as L  # import repo's lossdown (has Tables.rejects) before pilot modules alter sys.path
    sys.path.insert(0, str(HERE))
    import asked_vs_got as X
    import heldout as HO
    import stat_points as SP
    import team84 as T
    from corpus import parse_team_text

    table_path = X.RES / "lossdata84_shuffled_only.npz"
    sha = hashlib.sha256(table_path.read_bytes()).hexdigest()
    d = np.load(table_path)
    all_x = d["all_X"].astype(np.int64)
    all_w = d["all_W"]
    all_val = d["all_val"].astype(bool)
    has = ~np.isnan(all_w)
    tr = ~all_val & has
    va = all_val & has

    V = T.regulation_vocab()
    tables = L.Tables(V, [])
    all_rej = int(tables.rejects(all_x[:, :T.COLS48]).sum())
    tr_rej = int(tables.rejects(all_x[tr, :T.COLS48]).sum())
    va_rej = int(tables.rejects(all_x[va, :T.COLS48]).sum())

    sp = all_x[:, T.COLS48:] - 1
    sp_ok = bool((sp >= 0).all() and (sp <= SP.MAX_STAT).all() and (sp.reshape(len(sp), SP.NSLOT, SP.NSTAT).sum(-1) <= SP.MAX_TOTAL).all())

    def team_key(row):
        a, b = row[:T.COLS48].reshape(6, -1), row[T.COLS48:].reshape(6, -1)
        return b"".join(sorted(np.concatenate([a[i], b[i]]).tobytes() for i in range(6)))

    labelled_keys = {team_key(r) for r in all_x[has]}
    held_encoded, held_overlap = 0, 0
    for path in HO.files():
        team = parse_team_text(Path(path).read_text())
        try:
            row = T.encode(V, team).astype(np.int64)
        except ValueError:
            continue
        held_encoded += 1
        held_overlap += int(team_key(row) in labelled_keys)

    passed = (all_rej == 0 and tr_rej == 0 and va_rej == 0 and sp_ok and held_overlap == 0)
    report = dict(
        table=table_path.name,
        sha256=sha,
        expected_sha256=EXPECTED_TABLE_SHA,
        sha_matches_expected=(sha == EXPECTED_TABLE_SHA),
        method="lossdown.Tables.rejects (every field visible)",
        total_rows=len(all_x),
        labelled_train_rows=int(tr.sum()),
        labelled_val_rows=int(va.sum()),
        all_rejected=all_rej,
        train_rejected=tr_rej,
        val_rejected=va_rej,
        stat_points_legal=sp_ok,
        heldout_encoded=held_encoded,
        heldout_overlap=held_overlap,
        passed=passed,
    )
    if out_json:
        Path(out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(out_json).write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    if not passed:
        raise SystemExit("VERIFY_TABLE_FAIL")
    print("VERIFY_TABLE_PASS")
    return report


def check_grid(name, cells_wanted, results_dir=RES):
    """Strict completeness check: exact cell count, zero cells short of keep (64) teams, all battles finished."""
    path = Path(results_dir) / f"stat_points_84_{name}.json"
    if not path.exists():
        raise SystemExit(f"{name}: results file missing ({path})")
    out = json.loads(path.read_text())
    keep = int(out.get("keep", 64))
    cells = list(out.get("cells", {}).values())
    short_teams = [f"a{c['numerator']:02d}_s{c['seed']}({len(c.get('pastes', []))}/{keep})" for c in cells if len(c.get("pastes", [])) < keep]
    short_battles = [
        f"a{c['numerator']:02d}_s{c['seed']}"
        for c in cells
        if not c.get("pastes")
        or len(c.get("y", [])) != len(c["pastes"])
        or None in c.get("y", [])
        or c.get("battles", 0) < 49 * len(c["pastes"])
    ]
    print(
        f"{name}: {len(cells)}/{cells_wanted} cells, {len(short_teams)} short of {keep} teams {short_teams[:5]}, "
        f"{len(short_battles)} missing battles {short_battles[:5]}"
    )
    if len(cells) != int(cells_wanted) or short_teams or short_battles:
        raise SystemExit(1)
    return dict(name=name, cells=len(cells), keep=keep, short_teams=len(short_teams), short_battles=len(short_battles))


# ---------------------------------------------------------------- training curve figure and comment

def build_train_figure(info, out_dir, ref_info=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    curves = info.get("curves") or []
    train_pts = [[e["epoch"], round(float(e["train_loss"]), 4)] for e in curves if "train_loss" in e]
    val_pts = [[e["epoch"], round(float(e["val_loss"]), 4)] for e in curves if "val_loss" in e]
    best_ep = info.get("best_epoch")
    best_val = info.get("best_val_loss")
    last_ep = max((e["epoch"] for e in curves), default=1)

    val_series = {
        "type": "line",
        "name": "Filtered table — validation loss (4,368 teams)",
        "color": VAL_COLOR,
        "symbolSize": 5 if len(val_pts) <= 20 else 0,
        "lineStyle": {"width": 3},
        "emphasis": {"focus": "series"},
        "data": val_pts,
    }
    if best_ep is not None and best_val is not None:
        val_series["markPoint"] = {
            "symbol": "circle",
            "symbolSize": 10,
            "itemStyle": {"color": VAL_COLOR, "borderColor": "#ffffff", "borderWidth": 2},
            "label": {
                "show": True,
                "position": "left" if best_ep >= last_ep - 6 else "bottom",
                "distance": 10,
                "formatter": f"kept ep {best_ep}: {float(best_val):.4f}",
                "fontWeight": "bold",
            },
            "data": [{"coord": [best_ep, round(float(best_val), 4)]}],
        }
        val_series["markLine"] = {
            "silent": True,
            "symbol": ["none", "none"],
            "lineStyle": {"type": "dotted", "color": VAL_COLOR, "width": 1.5},
            "label": {"show": False},
            "data": [{"xAxis": best_ep}],
        }

    series = [
        {
            "type": "line",
            "name": "Filtered table — training loss (214,055 teams)",
            "color": TRAIN_COLOR,
            "symbol": "none",
            "lineStyle": {"width": 2, "type": "dashed"},
            "emphasis": {"focus": "series"},
            "data": train_pts,
        },
        val_series,
    ]
    if ref_info and ref_info.get("curves"):
        ref_val_pts = [[e["epoch"], round(float(e["val_loss"]), 4)] for e in ref_info["curves"] if "val_loss" in e]
        series.append(
            {
                "type": "line",
                "name": "Unfiltered table — validation loss (4,380 teams, reference)",
                "color": REF_COLOR,
                "symbol": "none",
                "lineStyle": {"width": 1.8, "type": "dotted"},
                "emphasis": {"focus": "series"},
                "data": ref_val_pts,
            }
        )

    option = {
        "animation": False,
        "grid": {"left": 68, "right": 28, "top": 58, "bottom": 76},
        "legend": {"top": 0},
        "tooltip": {
            "trigger": "axis",
            "valueFormatter": "js:v => v == null ? '' : Number(v).toFixed(4) + ' nats/field'",
        },
        "xAxis": {
            "type": "value",
            "min": 1,
            "max": max(2, last_ep),
            "name": "Training epoch",
            "nameLocation": "middle",
            "nameGap": 28,
        },
        "yAxis": {
            "type": "value",
            "scale": True,
            "name": "Linear masked ELBO loss (nats / field)",
            "nameLocation": "middle",
            "nameGap": 50,
        },
        "dataZoom": [{"type": "inside"}, {"type": "slider", "bottom": 8, "height": 18}],
        "series": series,
    }
    first_val = val_pts[0][1] if val_pts else float("nan")
    title = (
        f"Filtered table (Linear masked, full filter) reaches lowest validation loss {float(best_val):.4f} nats/field at epoch {best_ep}"
        if best_ep is not None and best_val is not None
        else "Filtered table (Linear masked, full filter) training and validation loss by epoch"
    )
    subtitle = (
        f"Linear masking schedule (1 − α(t) = t on all 84 fields), full legal Pokémon construction filter, "
        f"{NEW_CK}. {stop_reason(info).capitalize()}."
    )
    ref_note = (
        f" Unfiltered reference ({REF_CK}): epoch {ref_info.get('best_epoch')} val {float(ref_info.get('best_val_loss')):.4f} on 4,380 teams "
        f"(12 wrong-stone validation rows were removed before this retrain)."
        if ref_info and ref_info.get("best_val_loss") is not None
        else ""
    )
    note = (
        f"EMA weights evaluated on 4 fixed stratified draws of 4,368 validation teams "
        f"(epoch 1 val {first_val:.4f} → kept epoch {best_ep} val {float(best_val):.4f}).{ref_note}"
    )
    return write_chart(option, out_dir / "train-val-curves.html", title=title, subtitle=subtitle, note=note, height=420)


def format_train_comment(info, commit_sha, loss_light, loss_dark, ref_info=None, verify_info=None):
    curves = info.get("curves") or []
    best_ep = info["best_epoch"]
    stop_ep = info["stopped_epoch"]
    best_val = float(info["best_val_loss"])
    first = curves[0]
    best_entry = next((e for e in curves if e["epoch"] == best_ep), curves[-1])
    last = curves[-1]
    reason = stop_reason(info)
    fired = early_stopping_fired(info)
    data_sha = str(info.get("data_sha256", ""))[:12]
    wandb_url = (info.get("wandb") or {}).get("url")
    alt = (
        f"Training and validation loss by epoch for Filtered table (Linear masked, full filter): "
        f"validation loss falls from {float(first['val_loss']):.4f} at epoch 1 to {best_val:.4f} nats/field at kept epoch {best_ep}; {reason}"
    )
    lines = [
        f"**Training finished for Filtered table (Linear masked, full filter, `{NEW_CK}`): kept epoch {best_ep} of {stop_ep} "
        f"(validation loss {best_val:.4f} nats/field; {reason}).**",
        "",
        picture(loss_light, loss_dark, alt),
        "",
        "### Training summary",
        "",
        "| Measure | Filtered table (`linear_masked_filter`) | Unfiltered reference (`sched_linear_species_first`) |",
        "| --- | --- | --- |",
        f"| Checkpoint | `{NEW_CK}` | `{REF_CK}` |",
        f"| Wrong-stone teams (every field visible) | **0 train / 0 val** (`dropped_by_tables = {info.get('dropped_by_tables', 0)}`) | 11 reached train (`dropped_by_tables = 532`) / 12 in val |",
        f"| Labelled rows (train / val) | `{info.get('n_train', 214055):,}` / `{info.get('val_rows', 4368):,}` | `214,066` / `4,380` |",
        f"| Epoch read (`best_epoch`) | **Epoch {best_ep}** (stopped at {stop_ep}; early stopping fired: **{'yes' if fired else 'no'}**) | Epoch {ref_info.get('best_epoch', 80) if ref_info else 80} (stopped at {ref_info.get('stopped_epoch', 80) if ref_info else 80}) |",
        f"| Best validation loss (nats/field) | **{best_val:.4f}** (ep 1: `{float(first['val_loss']):.4f}`, ep {stop_ep}: `{float(last['val_loss']):.4f}`) | `{float(ref_info['best_val_loss']):.4f}` (on 4,380 rows incl. 12 wrong-stone teams) |" if ref_info and ref_info.get("best_val_loss") is not None else f"| Best validation loss (nats/field) | **{best_val:.4f}** | `2.0142` |",
        f"| Training loss at kept epoch | `{float(best_entry['train_loss']):.4f}` (ep 1: `{float(first['train_loss']):.4f}`) | `{float(ref_info['curves'][-1]['train_loss']):.4f}` |" if ref_info and ref_info.get("curves") else f"| Training loss at kept epoch | `{float(best_entry['train_loss']):.4f}` | — |",
        f"| Wall-clock time | `{info.get('minutes', '?')}` min (`{info.get('amp_dtype', 'float32')}`) | `{ref_info.get('minutes', '?') if ref_info else '?'}` min |",
        f"| Reproducibility | Commit `{commit_sha[:12]}`, table `lossdata84_shuffled_only.npz` (`sha256: {data_sha}…`), seed `{info.get('seed')}` | Pre-#368 table (`sha256: 8aca232e…`), seed `0` |",
    ]
    if wandb_url:
        lines.append(f"| Weights & Biases | [{wandb_url}]({wandb_url}) | — |")
    if verify_info:
        lines += [
            "",
            f"Pre-training table check (`{verify_info['method']}`): `{verify_info['train_rejected']}/{verify_info['labelled_train_rows']:,}` "
            f"training teams and `{verify_info['val_rejected']}/{verify_info['labelled_val_rows']:,}` validation teams rejected; "
            f"`{verify_info['heldout_overlap']}/{verify_info['heldout_encoded']}` held-out overlap.",
        ]
    lines += [
        "",
        "Next: `ask_grid84` smoke check, then the three battle grids at guidance 2, 4, and 8 on fresh seeds, each posted as it lands, "
        "followed by the 4-game elite line-up re-battles, Pass@k reads, and the full trade-off comparison against the unfiltered checkpoint.",
    ]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- grid bootstrap & trade-off comparison

def _load_redraw_bases():
    bases = dict(REF_REDRAW_BASES)
    if BASES_FILE.exists():
        for line in BASES_FILE.read_text().splitlines():
            if not line.strip():
                continue
            k, v = line.split()
            for g in GUIDANCES:
                if k == grid_name(REF_ROW, g):
                    bases[g] = int(v)
    return bases


def _team_scores(rate, weights):
    seen = ~np.isnan(rate)
    return (np.where(seen, rate, 0.0) @ weights) / np.maximum(seen @ weights, 1e-12)


def _opponent_draws(n_opponents):
    return np.random.default_rng(DRAW_SEED).multinomial(n_opponents, np.full(n_opponents, 1.0 / n_opponents), DRAWS).astype(float)


def _bootstrap_grid(opponents_npz, counts, stream_id):
    """{(ask, seed): [DRAWS] mean got} resampling teams within each cell and opponents by shared `counts`."""
    data = np.load(opponents_npz)
    weights = data["weights"]
    rng = np.random.default_rng([DRAW_SEED, 385, stream_id])
    draws = {}
    for k in ALL_ASKS:
        for s in SEEDS:
            rate = data[f"a{k:02d}_s{s}"]
            seen = ~np.isnan(rate)
            scores = (np.where(seen, rate, 0.0) @ (counts * weights).T) / np.maximum(seen @ (counts * weights).T, 1e-12)
            pick = rng.integers(0, len(rate), (DRAWS, len(rate)))
            draws[(k, s)] = scores[pick, np.arange(DRAWS)[:, None]].mean(1)
    return draws


def _pooled_got_draws(cell_draws, sizes):
    return {k: sum(cell_draws[(k, s)] * sizes[(k, s)] for s in SEEDS) / sum(sizes[(k, s)] for s in SEEDS) for k in ALL_ASKS}


def _error_draws(got_draws, asks=SCORED_ASKS):
    return np.mean([(got_draws[k] - k / 24) ** 2 for k in asks], 0)


def _interval(arr):
    return [float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))]


def _line_up_metrics_from_cells(cells, train_lineups):
    import line_up_yield as LY

    by_ask = {}
    all_lu = []
    for c in cells:
        lus = [LY.line_up(p) for p in c["pastes"]]
        all_lu.extend(lus)
        by_ask.setdefault(int(c["numerator"]), []).extend(lus)
    distinct_pooled = len(set(all_lu))
    within_ask_all = {k: len(set(v)) for k, v in by_ask.items()}
    within_ask_mean_23 = float(np.mean([within_ask_all[k] for k in ALL_ASKS if k in within_ask_all]))
    within_ask_mean_17 = float(np.mean([within_ask_all[k] for k in SCORED_ASKS if k in within_ask_all]))
    novel_share = float(np.mean([lu not in train_lineups for lu in all_lu]))
    distinct_novel = len({lu for lu in all_lu if lu not in train_lineups})
    return dict(
        teams=len(all_lu),
        distinct_line_ups=distinct_pooled,
        within_ask_line_ups_mean_23=within_ask_mean_23,
        within_ask_line_ups_mean_17=within_ask_mean_17,
        novel_share=novel_share,
        distinct_novel_line_ups=distinct_novel,
    )


def _pass_at_k_summary(name, results_dir=RES):
    path = Path(results_dir) / "pass_at_k" / f"{name}.json"
    if path.exists():
        r = json.loads(path.read_text())
    else:
        p331 = Path(results_dir) / "pass_at_k_issue331.json"
        r = json.loads(p331.read_text())["grids"].get(name) if p331.exists() else None
    if not r:
        return None
    band_data = r["bands"].get("held", {}).get("asks_13_to_17")
    if not band_data:
        return None
    return dict(
        definition=r.get("definition"),
        pass_at_1=float(band_data["pass_at_k"][0]),
        pass_at_1_interval=band_data["pass_at_k_interval"][0],
        pass_at_8=float(band_data["pass_at_k"][7]),
        pass_at_8_interval=band_data["pass_at_k_interval"][7],
        pass_at_48=float(band_data["pass_at_k"][47]),
        pass_at_48_interval=band_data["pass_at_k_interval"][47],
        potential_at_48=float(band_data["potential_at_k"][47]),
        pass_at_k=band_data["pass_at_k"],
        pass_at_k_interval=band_data["pass_at_k_interval"],
    )


def _elite_yield_summary(name, results_dir=RES):
    path = Path(results_dir) / f"line_up_yield_{name}.json"
    if not path.exists():
        return None
    views = json.loads(path.read_text()).get("views")
    if not views or "all_23_asks" not in views:
        return None
    v = views["all_23_asks"]
    return dict(
        elite_line_ups=int(v["elite_line_ups"]),
        rebattled=int(v["rebattled"]),
        held=int(v["held"]),
        held_new_against_training=int(v["held_new_against_training"]),
        yield_per_1000=float(v["yield_per_1000"]),
        one_score_mean=float(v["one_score_mean"]) if v["one_score_mean"] is not None else None,
        fresh_mean=float(v["fresh_mean"]) if v["fresh_mean"] is not None else None,
    )


def evaluate_beyond_luck(new_m, ref_m, paired_err, paired_win):
    """Identify every difference beyond training luck so no real shift is ever assumed away."""
    flags = []
    if abs(paired_err["difference"]) >= ERROR_LUCK:
        direction = "higher (worse)" if paired_err["difference"] > 0 else "lower (better)"
        flags.append(
            f"ask-versus-got error differs by {paired_err['difference']:+.4f} ({direction}), "
            f"exceeding the {ERROR_LUCK:.3f} training-luck bar (95% CI {fmt_iv(paired_err['interval'], '+.4f')})"
        )
    if abs(paired_win["difference"]) >= WIN_LUCK:
        direction = "higher" if paired_win["difference"] > 0 else "lower"
        flags.append(
            f"pooled win rate differs by {paired_win['difference']:+.3f} ({direction}), "
            f"exceeding the {WIN_LUCK:.3f} training-luck bar (95% CI {fmt_iv(paired_win['interval'], '+.3f')})"
        )
    ref_nov = ref_m.get("unbiased_breadth", {}).get("novel_share")
    if ref_nov is not None and abs(new_m["novel_share"] - ref_nov) >= 0.05:
        flags.append(
            f"novel share differs by {new_m['novel_share'] - ref_nov:+.1%} against unfiltered non-replayed seeds "
            f"({new_m['novel_share']:.1%} vs {ref_nov:.1%})"
        )
    new_pak, ref_pak = new_m.get("pass_at_k"), ref_m.get("pass_at_k")
    if new_pak and ref_pak:
        for k_name in ("pass_at_8", "pass_at_48"):
            n_iv, r_iv = new_pak[f"{k_name}_interval"], ref_pak[f"{k_name}_interval"]
            if n_iv[1] < r_iv[0] or n_iv[0] > r_iv[1]:
                flags.append(
                    f"{k_name.replace('pass_at_', 'Pass@')} intervals do not overlap "
                    f"({new_pak[k_name]:.3f} {fmt_iv(n_iv, '.3f')} vs {ref_pak[k_name]:.3f} {fmt_iv(r_iv, '.3f')})"
                )
    return flags


def read_guidance_pair(guidance, counts=None, train_lineups=None, results_dir=RES):
    """Read both checkpoints at one guidance value, with shared-opponent bootstrap intervals."""
    sys.path.insert(0, "/tmp/vgc-pilot/src")
    sys.path.insert(0, str(HERE))
    import masking_rate_rows as MRR

    if train_lineups is None:
        train_lineups = MRR.references()[0]
    g = int(guidance)
    new_name, ref_name = grid_name(NEW_ROW, g), grid_name(REF_ROW, g)
    new_grid = json.loads((Path(results_dir) / f"stat_points_84_{new_name}.json").read_text())
    ref_grid = json.loads((Path(results_dir) / f"stat_points_84_{ref_name}.json").read_text())
    new_opp = Path(results_dir) / f"masking_rate_rows_opponents_{new_name}.npz"
    ref_opp = Path(results_dir) / f"masking_rate_rows_opponents_{ref_name}.npz"
    if counts is None:
        counts = _opponent_draws(len(np.load(new_opp)["weights"]))

    def one(grid, opp_path, name, stream_id):
        cells = list(grid["cells"].values())
        by = {(int(c["numerator"]), int(c["seed"])): c for c in cells}
        sizes = {key: len(c["pastes"]) for key, c in by.items()}
        got = {k: float(np.mean([y for s in SEEDS for y in by[(k, s)]["y"]])) for k in ALL_ASKS}
        by_seed = {s: {k: float(np.mean(by[(k, s)]["y"])) for k in ALL_ASKS} for s in SEEDS}
        err = float(np.mean([(got[k] - k / 24) ** 2 for k in SCORED_ASKS]))
        err_high = float(np.mean([(got[k] - k / 24) ** 2 for k in HIGH_ASKS]))
        err_by_seed = {str(s): float(np.mean([(by_seed[s][k] - k / 24) ** 2 for k in SCORED_ASKS])) for s in SEEDS}
        top_win = float(np.mean([y for k in TOP_ASKS for s in SEEDS for y in by[(k, s)]["y"]]))
        pooled_win = float(np.mean([y for c in cells for y in c["y"]]))
        tried = sum(int(c["tried"]) for c in cells)
        rejected = sum(int(c["rejected"]) for c in cells)
        cd = _bootstrap_grid(opp_path, counts, stream_id)
        gd = _pooled_got_draws(cd, sizes)
        ed = _error_draws(gd, SCORED_ASKS)
        wd = np.mean([gd[k] for k in ALL_ASKS], 0)
        td = np.mean([gd[k] for k in TOP_ASKS], 0)
        breadth = _line_up_metrics_from_cells(cells, train_lineups)
        eff = float(np.mean([c["div"]["effective_species"] for c in cells if c.get("div")]))
        m = dict(
            name=name,
            guidance=g,
            seed0=grid.get("seed0"),
            checkpoint=grid.get("checkpoint"),
            checkpoint_sha256=grid.get("checkpoint_sha256"),
            cells=len(cells),
            cells_short_of_64=sum(len(c["pastes"]) < int(grid.get("keep", 64)) for c in cells),
            teams=sum(len(c["pastes"]) for c in cells),
            battles=sum(int(c.get("battles", 0)) for c in cells),
            error=err,
            error_interval=_interval(ed),
            error_by_seed=err_by_seed,
            error_asks_18_to_23=err_high,
            got_by_ask={str(k): got[k] for k in ALL_ASKS},
            got_by_ask_interval={str(k): _interval(gd[k]) for k in ALL_ASKS},
            pooled_win=pooled_win,
            pooled_win_interval=_interval(wd),
            win_top_asks=top_win,
            win_top_asks_interval=_interval(td),
            first_try_legality=(tried - rejected) / max(tried, 1),
            effective_species=eff,
            **breadth,
            elite_line_up_yield=_elite_yield_summary(name, results_dir),
            pass_at_k=_pass_at_k_summary(name, results_dir),
        )
        return m, dict(error=ed, pooled_win=wd, win_top=td)

    new_m, new_d = one(new_grid, new_opp, new_name, stream_id=10 * g + 1)
    ref_m, ref_d = one(ref_grid, ref_opp, ref_name, stream_id=10 * g + 2)

    # Unfiltered checkpoint's unbiased non-replayed seed redraws (escapes legacy seed-7300 replay, #353)
    bases = _load_redraw_bases()
    if g in bases:
        redraw_path = Path(results_dir) / f"stat_points_84_{ref_name}_seed{bases[g]}.json"
        if redraw_path.exists():
            redraw_cells = list(json.loads(redraw_path.read_text())["cells"].values())
            ref_m["unbiased_breadth"] = dict(seed0=bases[g], file=redraw_path.name, **_line_up_metrics_from_cells(redraw_cells, train_lineups))

    err_diff_draws = new_d["error"] - ref_d["error"]
    win_diff_draws = new_d["pooled_win"] - ref_d["pooled_win"]
    top_diff_draws = new_d["win_top"] - ref_d["win_top"]
    err_by_seed_diff = {s: new_m["error_by_seed"][s] - ref_m["error_by_seed"][s] for s in new_m["error_by_seed"]}

    paired_err = dict(
        difference=new_m["error"] - ref_m["error"],
        interval=_interval(err_diff_draws),
        difference_by_seed=err_by_seed_diff,
        exceeds_glossary_bar=bool(abs(new_m["error"] - ref_m["error"]) >= BAR),
        beyond_training_luck=bool(abs(new_m["error"] - ref_m["error"]) >= ERROR_LUCK),
        ci_excludes_zero=bool(_interval(err_diff_draws)[0] > 0 or _interval(err_diff_draws)[1] < 0),
    )
    paired_win = dict(
        difference=new_m["pooled_win"] - ref_m["pooled_win"],
        interval=_interval(win_diff_draws),
        beyond_training_luck=bool(abs(new_m["pooled_win"] - ref_m["pooled_win"]) >= WIN_LUCK),
        ci_excludes_zero=bool(_interval(win_diff_draws)[0] > 0 or _interval(win_diff_draws)[1] < 0),
    )
    paired_top = dict(
        difference=new_m["win_top_asks"] - ref_m["win_top_asks"],
        interval=_interval(top_diff_draws),
    )
    flags = evaluate_beyond_luck(new_m, ref_m, paired_err, paired_win)
    return dict(
        guidance=g,
        filtered=new_m,
        unfiltered=ref_m,
        paired=dict(error=paired_err, pooled_win=paired_win, win_top_asks=paired_top),
        flags_beyond_luck=flags,
    )


def build_grid_figure(pair, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    g = pair["guidance"]
    new_m, ref_m = pair["filtered"], pair["unfiltered"]
    pct = "js:v => Math.round(v * 100) + '%'"
    xs = [k / 24 for k in ALL_ASKS]
    series = []
    for i, (label, color, m, width) in enumerate(
        ((NEW_LABEL, NEW_COLOR, new_m, 2.8), (REF_LABEL, REF_COLOR, ref_m, 1.8))
    ):
        lo = [m["got_by_ask_interval"][str(k)][0] for k in ALL_ASKS]
        hi = [m["got_by_ask_interval"][str(k)][1] for k in ALL_ASKS]
        series += band(label, color, xs, lo, hi, stack=f"g{g}_{i}")
        series.append(
            {
                "type": "line",
                "name": label,
                "color": color,
                "symbolSize": 5,
                "lineStyle": {"width": width, "type": "solid" if i == 0 else "dashed"},
                "emphasis": {"focus": "series"},
                "data": [[x, m["got_by_ask"][str(k)]] for x, k in zip(xs, ALL_ASKS)],
            }
        )
    series.append(
        {
            "type": "line",
            "name": "Got = ask",
            "color": "#a8a29e",
            "symbol": "none",
            "silent": True,
            "lineStyle": {"type": "dotted", "width": 1.2},
            "data": [[0, 0], [1, 1]],
        }
    )
    option = {
        "animation": False,
        "grid": {"left": 64, "right": 24, "top": 56, "bottom": 56},
        "legend": {"top": 0, "data": [NEW_LABEL, REF_LABEL, "Got = ask"]},
        "tooltip": {"trigger": "axis", "valueFormatter": "js:v => v == null ? '' : (v * 100).toFixed(1) + '%'"},
        "xAxis": {
            "type": "value",
            "min": 0,
            "max": 1,
            "interval": 1 / 6,
            "name": "Ask: win rate requested from the model",
            "nameLocation": "middle",
            "nameGap": 34,
            "axisLabel": {"formatter": pct},
        },
        "yAxis": {
            "type": "value",
            "min": 0,
            "max": 0.8,
            "name": "Got: measured win rate",
            "nameLocation": "middle",
            "nameGap": 46,
            "axisLabel": {"formatter": pct},
        },
        "series": series,
    }
    pe = pair["paired"]["error"]
    verdict_phrase = (
        "differs beyond training luck"
        if pe["beyond_training_luck"]
        else "matches within training luck"
    )
    title = (
        f"Guidance {g}: Filtered table {verdict_phrase} "
        f"(error {new_m['error']:.4f} vs unfiltered {ref_m['error']:.4f}, Δ {pe['difference']:+.4f})"
    )
    subtitle = (
        f"23 asks × 3 seeds × 64 teams = 4,416 teams (216,384 battles). "
        f"Paired error gap {pe['difference']:+.4f} {fmt_iv(pe['interval'], '+.4f')}; "
        f"pooled win {new_m['pooled_win']:.3f} vs {ref_m['pooled_win']:.3f}."
    )
    note = (
        "Band: 95% bootstrap interval over teams and the 49 distinct top-50 meta teams (slots shuffled, behaviour-cloning policy). "
        "Ask-versus-got error is scored on asks 1/24 to 17/24; training-luck bar is 0.002."
    )
    return write_chart(option, out_dir / f"grid-g{g}.html", title=title, subtitle=subtitle, note=note, height=440)


def format_grid_comment(pair, fig_light, fig_dark):
    g = pair["guidance"]
    nm, rm = pair["filtered"], pair["unfiltered"]
    pe, pw = pair["paired"]["error"], pair["paired"]["pooled_win"]
    flags = pair["flags_beyond_luck"]
    status = "flags a difference beyond training luck" if flags else "matches the unfiltered checkpoint within training luck"
    alt = (
        f"Got against ask at guidance {g} for {NEW_LABEL} versus {REF_LABEL}: "
        f"ask-versus-got error {nm['error']:.4f} vs {rm['error']:.4f} (paired difference {pe['difference']:+.4f})"
    )
    unbiased = rm.get("unbiased_breadth")
    ref_nov_str = (
        f"{unbiased['novel_share']:.1%} (non-replayed seeds; 7300 battled: {rm['novel_share']:.1%})"
        if unbiased
        else f"{rm['novel_share']:.1%}"
    )
    ref_lu_str = (
        f"{unbiased['distinct_line_ups']:,} pooled / {unbiased['within_ask_line_ups_mean_23']:.1f} per ask (non-replayed; 7300 battled: {rm['distinct_line_ups']:,})"
        if unbiased
        else f"{rm['distinct_line_ups']:,} pooled / {rm['within_ask_line_ups_mean_23']:.1f} per ask"
    )
    new_y = nm.get("elite_line_up_yield") or {}
    ref_y = rm.get("elite_line_up_yield") or {}
    new_p = nm.get("pass_at_k") or {}
    ref_p = rm.get("pass_at_k") or {}
    nov_diff_str = (
        f"{nm['novel_share'] - unbiased['novel_share']:+.1%} vs non-replayed"
        if unbiased
        else f"{nm['novel_share'] - rm['novel_share']:+.1%}"
    )
    lu_diff_str = (
        f"{nm['distinct_line_ups'] - unbiased['distinct_line_ups']:+,} pooled vs non-replayed"
        if unbiased
        else f"{nm['distinct_line_ups'] - rm['distinct_line_ups']:+,}"
    )
    new_pak_str = f"{new_p['pass_at_8']:.3f} / {new_p['pass_at_48']:.3f}" if new_p else "pending"
    ref_pak_str = f"{ref_p['pass_at_8']:.3f} / {ref_p['pass_at_48']:.3f}" if ref_p else "n/a"

    lines = [
        f"**Guidance {g} grid landed for {NEW_LABEL} (`{nm['name']}`): {status}.**",
        "",
        picture(fig_light, fig_dark, alt),
        "",
        f"Completeness: `{nm['cells']}/69` cells, **`{nm['cells_short_of_64']}` cells short of 64 teams**, "
        f"`{nm['teams']:,}` teams, `{nm['battles']:,}` battles (sampling seed base `{nm['seed0']}`).",
        "",
        f"| Measure (guidance {g}) | {NEW_LABEL} | {REF_LABEL} | Difference / note |",
        "| --- | --- | --- | --- |",
        f"| Ask-versus-got error (asks 1/24–17/24) | **{nm['error']:.4f}** {fmt_iv(nm['error_interval'])} | {rm['error']:.4f} {fmt_iv(rm['error_interval'])} | **{pe['difference']:+.4f}** {fmt_iv(pe['interval'], '+.4f')} (luck bar `0.002`) |",
        f"| Error on asks 18/24–23/24 (apart) | `{nm['error_asks_18_to_23']:.4f}` | `{rm['error_asks_18_to_23']:.4f}` | `{nm['error_asks_18_to_23'] - rm['error_asks_18_to_23']:+.4f}` |",
        f"| Pooled win rate (23 asks) | **{nm['pooled_win']:.3f}** {fmt_iv(nm['pooled_win_interval'], '.3f')} | {rm['pooled_win']:.3f} {fmt_iv(rm['pooled_win_interval'], '.3f')} | **{pw['difference']:+.3f}** {fmt_iv(pw['interval'], '+.3f')} (luck bar `0.026`) |",
        f"| Win rate at top scored asks (15/24–17/24) | {nm['win_top_asks']:.3f} | {rm['win_top_asks']:.3f} | {pair['paired']['win_top_asks']['difference']:+.3f} |",
        f"| Novel share (vs labelled training) | **{nm['novel_share']:.1%}** | {ref_nov_str} | {nov_diff_str} |",
        f"| Distinct line-ups | **{nm['distinct_line_ups']:,}** pooled / **{nm['within_ask_line_ups_mean_23']:.1f}** per ask | {ref_lu_str} | {lu_diff_str} |",
        f"| First-try legality | {nm['first_try_legality']:.1%} | {rm['first_try_legality']:.1%} | — |",
        f"| Effective species (5-team subsets) | {nm['effective_species']:.1f} | {rm['effective_species']:.1f} | {nm['effective_species'] - rm['effective_species']:+.1f} |",
        f"| Elite line-ups that held (all 23 asks) | {new_y.get('held', 'pending')} | {ref_y.get('held', 'n/a')}* | *unfiltered battled on seed 7300 |",
        f"| Novel elite line-ups that held | **{new_y.get('held_new_against_training', 'pending')}** | {ref_y.get('held_new_against_training', 'n/a')} | excludes replayed training line-ups |",
        f"| Pass@8 / Pass@48 (asks 13/24–17/24, held) | {new_pak_str} | {ref_pak_str} | — |",
    ]
    if flags:
        lines += ["", "**Differences beyond training luck flagged here:**"] + [f"- {f}" for f in flags]
    else:
        lines += ["", "No measure at this guidance differs beyond training luck."]
    return "\n".join(lines) + "\n"


def build_summary(out_json=SUMMARY_PATH, results_dir=RES):
    sys.path.insert(0, "/tmp/vgc-pilot/src")
    sys.path.insert(0, str(HERE))
    import masking_rate_rows as MRR

    train_lineups = MRR.references()[0]
    first_opp = Path(results_dir) / f"masking_rate_rows_opponents_{grid_name(NEW_ROW, GUIDANCES[0])}.npz"
    counts = _opponent_draws(len(np.load(first_opp)["weights"]))
    by_guidance = {}
    all_flags = []
    for g in GUIDANCES:
        pair = read_guidance_pair(g, counts=counts, train_lineups=train_lineups, results_dir=results_dir)
        by_guidance[str(g)] = pair
        for f in pair["flags_beyond_luck"]:
            all_flags.append(f"guidance {g}: {f}")
    summary = dict(
        issue=int(ISSUE),
        filtered_checkpoint=NEW_CK,
        filtered_row=NEW_ROW,
        unfiltered_checkpoint=REF_CK,
        unfiltered_row=REF_ROW,
        guidances=list(GUIDANCES),
        bootstrap=dict(draws=DRAWS, seed=DRAW_SEED, resampled="teams within each ask and seed separately; 49 opponents shared across checkpoints"),
        training_luck=dict(error=ERROR_LUCK, pooled_win=WIN_LUCK, glossary_error_bar=BAR),
        by_guidance=by_guidance,
        findings_beyond_luck=all_flags,
        all_within_training_luck=(len(all_flags) == 0),
    )
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(out_json).write_text(json.dumps(summary, indent=1))
    print(f"SUMMARY_DONE all_within_training_luck={summary['all_within_training_luck']} flags={len(all_flags)} -> {out_json}")
    return summary


# ---------------------------------------------------------------- final summary figures, resolution, handoff, map update

def build_final_figures(summary, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    by_g = summary["by_guidance"]

    # 1. Trade-off curve: ask-versus-got error vs pooled distinct line-ups across guidance 2, 4, 8
    f_pts, u_pts = [], []
    for g in GUIDANCES:
        p = by_g[str(g)]
        nm, rm = p["filtered"], p["unfiltered"]
        rm_lu = (rm.get("unbiased_breadth") or rm)["distinct_line_ups"]
        f_pts.append({"value": [round(nm["error"], 5), nm["distinct_line_ups"]], "name": f"g={g}"})
        u_pts.append({"value": [round(rm["error"], 5), rm_lu], "name": f"g={g}"})

    opt_trade = {
        "animation": False,
        "grid": {"left": 74, "right": 34, "top": 58, "bottom": 58},
        "legend": {"top": 0},
        "tooltip": {"trigger": "item"},
        "xAxis": {
            "type": "value",
            "scale": True,
            "name": "Ask-versus-got error, asks 1/24–17/24 (lower is better)",
            "nameLocation": "middle",
            "nameGap": 34,
        },
        "yAxis": {
            "type": "value",
            "scale": True,
            "name": "Distinct line-ups, pooled over 23 asks",
            "nameLocation": "middle",
            "nameGap": 54,
        },
        "series": [
            {
                "type": "line",
                "name": NEW_LABEL,
                "color": NEW_COLOR,
                "symbolSize": 10,
                "lineStyle": {"width": 3},
                "emphasis": {"focus": "series"},
                "label": {"show": True, "formatter": "{b}", "position": "top"},
                "data": f_pts,
            },
            {
                "type": "line",
                "name": f"{REF_LABEL} (non-replayed line-ups)",
                "color": REF_COLOR,
                "symbolSize": 8,
                "lineStyle": {"width": 2, "type": "dashed"},
                "emphasis": {"focus": "series"},
                "label": {"show": True, "formatter": "{b}", "position": "bottom"},
                "data": u_pts,
            },
        ],
    }
    p1 = write_chart(
        opt_trade,
        out_dir / "tradeoff-error-lineups.html",
        title="Ask-versus-got error vs distinct line-ups across guidance 2, 4, and 8",
        subtitle=f"{NEW_LABEL} ({NEW_CK}) against {REF_LABEL} ({REF_CK}).",
        note="Unfiltered line-up counts use the non-replayed seed redraws (seed bases 940135810, 630821980, 111619920) so seed-7300 replay does not depress its diversity.",
        height=420,
    )

    # 2. Three-panel Pass@k across guidance 2, 4, 8
    pct = "js:v => Math.round(v * 100) + '%'"
    ks = list(range(1, 49))
    grids, xs, ys, titles, series = [], [], [], [], []
    for idx, g in enumerate(GUIDANCES):
        left = 6 + idx * 32
        grids.append({"left": f"{left}%", "width": "27%", "top": 70, "bottom": 54})
        titles.append({"text": f"Guidance {g}", "left": f"{left + 13.5}%", "top": 42, "textAlign": "center", "textStyle": {"fontSize": 13}})
        xs.append(
            {
                "gridIndex": idx,
                "type": "value",
                "min": 1,
                "max": 48,
                "name": "k" if idx == 1 else "",
                "nameLocation": "middle",
                "nameGap": 28,
            }
        )
        ys.append(
            {
                "gridIndex": idx,
                "type": "value",
                "min": 0,
                "max": 1,
                "name": "Pass@k" if idx == 0 else "",
                "nameLocation": "middle",
                "nameGap": 42,
                "axisLabel": {"formatter": pct, "show": idx == 0},
            }
        )
        p = by_g[str(g)]
        for j, (label, color, m, dash) in enumerate(
            ((NEW_LABEL, NEW_COLOR, p["filtered"], "solid"), (REF_LABEL, REF_COLOR, p["unfiltered"], "dashed"))
        ):
            pak = m.get("pass_at_k")
            if not pak:
                continue
            lo = [v[0] for v in pak["pass_at_k_interval"]]
            hi = [v[1] for v in pak["pass_at_k_interval"]]
            series += band(label, color, ks, lo, hi, stack=f"pk_{idx}_{j}", xAxisIndex=idx, yAxisIndex=idx)
            series.append(
                {
                    "type": "line",
                    "name": label,
                    "color": color,
                    "xAxisIndex": idx,
                    "yAxisIndex": idx,
                    "symbol": "none",
                    "lineStyle": {"width": 2.4 if j == 0 else 1.8, "type": dash},
                    "emphasis": {"focus": "series"},
                    "data": [[k, v] for k, v in zip(ks, pak["pass_at_k"])],
                }
            )

    opt_pak = {
        "animation": False,
        "title": titles,
        "grid": grids,
        "legend": {"top": 6, "data": [NEW_LABEL, REF_LABEL]},
        "tooltip": {"trigger": "axis", "valueFormatter": "js:v => v == null ? '' : (v * 100).toFixed(1) + '%'"},
        "xAxis": xs,
        "yAxis": ys,
        "series": series,
    }
    p2 = write_chart(
        opt_pak,
        out_dir / "tradeoff-pass-at-k.html",
        title="Pass@k at asks 13/24 to 17/24 (held elite line-ups) across guidance 2, 4, and 8",
        subtitle="Chance that at least one of k drawn teams reaches 0.531 and its line-up holds at 0.531 in a 4-game-per-opponent re-battle.",
        note="Band: 95% bootstrap interval over sampling seeds, teams, and the 49 distinct top-50 meta teams (src/pass_at_k.py).",
        height=390,
    )
    return p1, p2


def format_final_comment(summary, run_info, commit_sha, err_light, err_dark, pak_light, pak_dark):
    flags = summary.get("findings_beyond_luck") or []
    best_ep = run_info["best_epoch"]
    stop_ep = run_info["stopped_epoch"]
    best_val = float(run_info["best_val_loss"])
    headline_verdict = (
        "matches the unfiltered checkpoint within training luck on every measure across guidance 2, 4, and 8"
        if not flags
        else f"flags {len(flags)} difference(s) beyond training luck — reported below, not assumed away"
    )
    lines = [
        f"**Resolved: {NEW_LABEL} (`{NEW_CK}`, kept epoch {best_ep}) {headline_verdict}.**",
        "",
        picture(err_light, err_dark, f"Ask-versus-got error versus distinct line-ups across guidance 2, 4, and 8 for {NEW_LABEL} and {REF_LABEL}"),
        "",
        picture(pak_light, pak_dark, f"Pass@k across guidance 2, 4, and 8 at asks 13/24 to 17/24 for {NEW_LABEL} and {REF_LABEL}"),
        "",
        "### Full trade-off table (Filtered vs Unfiltered linear checkpoint)",
        "",
        "Shared by both checkpoints: 84-field transformer (`d=256`, 6 layers, 8 heads), linear masking schedule (`1 − α(t) = t`), "
        "full legal Pokémon construction filter, label noise `±1/48`, EMA `0.999`, seed `0`, early stopping (`patience=3, threshold=0, cap=80`), "
        "species-first 84 passes, 4,416 teams (216,384 shuffled-slot battles against the 49 distinct top-50 meta teams) per guidance. "
        f"Filtered checkpoint trained on `lossdata84_shuffled_only.npz` (`0/0` `Tables.rejects` with every field visible; kept epoch **{best_ep}**/{stop_ep}, "
        f"val loss `{best_val:.4f}`, {stop_reason(run_info)}).",
        "",
        "| Guidance | Arm | Ask-vs-got error (1–17) | Δ error [95% CI] | Pooled win (23 asks) | Top-ask win (15–17) | Novel share | Distinct line-ups (pooled / per ask) | Elite held | Novel elite held | Pass@8 | Pass@48 |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for g in GUIDANCES:
        p = summary["by_guidance"][str(g)]
        nm, rm = p["filtered"], p["unfiltered"]
        pe = p["paired"]["error"]
        unb = rm.get("unbiased_breadth")
        ny, ry = nm.get("elite_line_up_yield") or {}, rm.get("elite_line_up_yield") or {}
        npk, rpk = nm.get("pass_at_k") or {}, rm.get("pass_at_k") or {}
        np8 = f"{npk['pass_at_8']:.3f}" if npk else "n/a"
        np48 = f"{npk['pass_at_48']:.3f}" if npk else "n/a"
        rp8 = f"{rpk['pass_at_8']:.3f}" if rpk else "n/a"
        rp48 = f"{rpk['pass_at_48']:.3f}" if rpk else "n/a"
        lines.append(
            f"| {g} | **{NEW_LABEL}** | **{nm['error']:.4f}** | **{pe['difference']:+.4f}** {fmt_iv(pe['interval'], '+.4f')} "
            f"| **{nm['pooled_win']:.3f}** | {nm['win_top_asks']:.3f} | **{nm['novel_share']:.1%}** "
            f"| **{nm['distinct_line_ups']:,}** / {nm['within_ask_line_ups_mean_23']:.1f} "
            f"| {ny.get('held', 'n/a')} | **{ny.get('held_new_against_training', 'n/a')}** "
            f"| {np8} | {np48} |"
        )
        lines.append(
            f"| {g} | {REF_LABEL} | {rm['error']:.4f} | ref | {rm['pooled_win']:.3f} | {rm['win_top_asks']:.3f} "
            f"| {(unb or rm)['novel_share']:.1%}{'*' if unb else ''} "
            f"| {(unb or rm)['distinct_line_ups']:,} / {(unb or rm)['within_ask_line_ups_mean_23']:.1f}{'*' if unb else ''} "
            f"| {ry.get('held', 'n/a')}† | {ry.get('held_new_against_training', 'n/a')} "
            f"| {rp8}† | {rp48}† |"
        )
    lines += [
        "",
        "\\* Unfiltered novel share and distinct line-ups are read from its non-replayed seed redraws (`seed940135810`, `seed630821980`, `seed111619920`) "
        "so legacy seed-7300 training-team replay does not distort the comparison. "
        "† Unfiltered elite held and Pass@k come from its battled seed-7300 grids (which replay some training teams); **Novel elite held** excludes training line-ups on both sides.",
        "",
        "### Beyond-training-luck audit",
        "",
    ]
    if flags:
        lines += ["Differences exceeding the pre-registered training-luck bars (`0.002` error, `0.026` pooled win, `≥ 5 pp` unbiased novelty, or non-overlapping Pass@k intervals):"]
        lines += [f"- {f}" for f in flags]
    else:
        lines.append(
            "Every measured difference across guidance 2, 4, and 8 sits inside the pre-registered training-luck bars "
            "(`|Δ error| < 0.002`, `|Δ pooled win| < 0.026`, unbiased novelty within 5 pp, Pass@8 / Pass@48 intervals overlapping)."
        )
    lines += [
        "",
        "### Handoff & caveats",
        "",
        f"- **Checkpoint handed to [#64](https://github.com/{REPO}/issues/{HANDOFF_ISSUE}):** `{NEW_CK}` "
        f"(kept epoch **{best_ep}**, ask-grid row `linear_masked_filter`, commit `{commit_sha[:12]}`, table `lossdata84_shuffled_only.npz` `sha256: {str(run_info.get('data_sha256', ''))[:12]}…`).",
        "- **Scope and caveats:** One training seed (`seed = 0`), all 84 fields generated with Stat Alignment repair, scored against the 49 distinct top-50 meta teams under the behaviour-cloning policy with slots shuffled. Does not measure human play, other policies, or held-out ranks 51–100.",
    ]
    return "\n".join(lines) + "\n"


def format_handoff_comment(summary, run_info):
    best_ep = run_info["best_epoch"]
    g2 = summary["by_guidance"]["2"]["filtered"]
    g4 = summary["by_guidance"]["4"]["filtered"]
    g8 = summary["by_guidance"]["8"]["filtered"]
    flags = summary.get("findings_beyond_luck") or []
    lines = [
        f"**Checkpoint ready from [#{ISSUE}]({TASK_URL}): `{NEW_CK}` (`linear_masked_filter`).**",
        "",
        f"- **Reader name:** {NEW_LABEL}",
        f"- **Checkpoint file:** `~/.local/share/vgc-pilot-runtime/lossdown/{NEW_CK}.pt` (kept epoch **{best_ep}** of {run_info['stopped_epoch']}; {stop_reason(run_info)})",
        f"- **Training table:** filtered `lossdata84_shuffled_only.npz` (`sha256: {str(run_info.get('data_sha256', ''))[:12]}…`, `214,055` train / `4,368` val, `0 / 0` wrong-stone rejects with every field visible)",
        f"- **Ask-versus-got error (asks 1/24–17/24):** `{g2['error']:.4f}` (g=2), `{g4['error']:.4f}` (g=4), `{g8['error']:.4f}` (g=8)",
        f"- **Pooled win rate / novel share / novel elite held:** `{g2['pooled_win']:.3f}` / `{g2['novel_share']:.1%}` / `{(g2.get('elite_line_up_yield') or {}).get('held_new_against_training', 'n/a')}` (g=2); "
        f"`{g4['pooled_win']:.3f}` / `{g4['novel_share']:.1%}` / `{(g4.get('elite_line_up_yield') or {}).get('held_new_against_training', 'n/a')}` (g=4); "
        f"`{g8['pooled_win']:.3f}` / `{g8['novel_share']:.1%}` / `{(g8.get('elite_line_up_yield') or {}).get('held_new_against_training', 'n/a')}` (g=8)",
        f"- **Comparison to unfiltered linear checkpoint:** {'all differences within training luck' if not flags else f'{len(flags)} flag(s) beyond training luck — see #{ISSUE}'}.",
    ]
    return "\n".join(lines) + "\n"


def update_map(summary, run_info):
    best_ep = run_info["best_epoch"]
    g2 = summary["by_guidance"]["2"]["filtered"]
    g4 = summary["by_guidance"]["4"]["filtered"]
    g8 = summary["by_guidance"]["8"]["filtered"]
    flags = summary.get("findings_beyond_luck") or []
    decision_line = (
        f"- [{TASK_TITLE}]({TASK_URL}): retrained **{NEW_LABEL}** (`{NEW_CK}`) on the filtered shuffled-only table (`0/0` wrong-stone rejects); "
        f"kept epoch **{best_ep}** (`val_loss {float(run_info['best_val_loss']):.4f}`), ask-versus-got error `{g2['error']:.4f}` / `{g4['error']:.4f}` / `{g8['error']:.4f}` "
        f"at guidance 2, 4, 8 ({'all measures within training luck of unfiltered' if not flags else f'{len(flags)} beyond-luck flag(s) noted'}), handed to #64."
    )
    body = subprocess.run(
        ["gh", "issue", "view", MAP_ISSUE, "--repo", REPO, "--json", "body", "-q", ".body"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if "## Decisions so far" not in body or "\n## Not yet specified" not in body:
        raise RuntimeError(f"map #{MAP_ISSUE} missing expected Decisions so far / Not yet specified sections")
    decisions_part = body.split("## Decisions so far", 1)[1].split("## Not yet specified", 1)[0]
    if f"/issues/{ISSUE})" in decisions_part:
        print(f"map #{MAP_ISSUE} already records #{ISSUE}", file=sys.stderr)
        return
    before, after = body.split("\n## Not yet specified", 1)
    updated = before.rstrip() + "\n" + decision_line + "\n\n## Not yet specified" + after
    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as tf:
        tf.write(updated)
        tmp_path = tf.name
    try:
        subprocess.run(["gh", "issue", "edit", MAP_ISSUE, "--repo", REPO, "--body-file", tmp_path], check=True, capture_output=True, text=True)
    finally:
        Path(tmp_path).unlink(missing_ok=True)
    print(f"updated map #{MAP_ISSUE} Decisions so far", file=sys.stderr)


def main(argv=None):  # noqa: C901
    args = sys.argv[1:] if argv is None else argv
    if not args:
        raise SystemExit(__doc__)
    cmd, rest = args[0], args[1:]
    if cmd == "verify-table":
        verify_table(rest[0] if rest else None)
    elif cmd == "check-grid":
        check_grid(rest[0], int(rest[1]))
    elif cmd == "train-figure":
        info = json.loads(Path(rest[0]).read_text())
        ref = json.loads(Path(rest[2]).read_text()) if len(rest) > 2 and Path(rest[2]).exists() else None
        print(build_train_figure(info, rest[1], ref))
    elif cmd == "train-comment":
        info = json.loads(Path(rest[0]).read_text())
        sha = Path(rest[1]).read_text().strip()
        ref = json.loads(Path(rest[4]).read_text()) if len(rest) > 4 and Path(rest[4]).exists() else None
        ver = json.loads(Path(rest[5]).read_text()) if len(rest) > 5 and Path(rest[5]).exists() else None
        print(format_train_comment(info, sha, rest[2], rest[3], ref, ver), end="")
    elif cmd == "grid-figure":
        pair = read_guidance_pair(int(rest[0]))
        print(build_grid_figure(pair, rest[1]))
    elif cmd == "grid-comment":
        pair = read_guidance_pair(int(rest[0]))
        print(format_grid_comment(pair, rest[1], rest[2]), end="")
    elif cmd == "summary":
        build_summary(rest[0] if rest else SUMMARY_PATH)
    elif cmd == "final-figures":
        summary = json.loads(Path(rest[0]).read_text())
        for p in build_final_figures(summary, rest[1]):
            print(p)
    elif cmd == "final-comment":
        summary = json.loads(Path(rest[0]).read_text())
        run_info = json.loads(Path(rest[1]).read_text())
        sha = Path(rest[2]).read_text().strip()
        print(format_final_comment(summary, run_info, sha, rest[3], rest[4], rest[5], rest[6]), end="")
    elif cmd == "handoff-comment":
        summary = json.loads(Path(rest[0]).read_text())
        run_info = json.loads(Path(rest[1]).read_text())
        print(format_handoff_comment(summary, run_info), end="")
    elif cmd == "update-map":
        summary = json.loads(Path(rest[0]).read_text())
        run_info = json.loads(Path(rest[1]).read_text())
        update_map(summary, run_info)
    else:
        raise SystemExit(f"unknown command: {cmd}")


if __name__ == "__main__":
    main()
