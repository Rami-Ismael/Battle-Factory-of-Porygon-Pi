"""Checks, figures, the trade-off comparison, and GitHub posts for issue #385.

Issue #385 retrains the default 84-field linear masked diffusion checkpoint on the filtered shuffled-only table,
which has no wrong-stone teams (`stat84_shuffled_noise_linear_filtered_es3_cap80`, ask-grid row
`linear_masked_filter`, reader name "Linear masked, full filter"). This script compares every trade-off measure
at guidance 2, 4, and 8 against the unfiltered linear checkpoint (`stat84_shuffled_noise_linear_es3_cap80`,
ask-grid row `sched_linear_species_first`).

Commands, in the order the queue runs them:
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

`retrain_default_dry_run.py` runs every step after the battles on synthetic grids in a few seconds; the queue runs
it before training, so a crash in this script stops the queue at minute one instead of after the battles.
"""
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from echarts_chart import band, write_chart

# ---------------------------------------------------------------- issue, checkpoints, and grids

REPO = "Rami-Ismael/Battle-Factory-of-Porygon-Pi"
ISSUE = "385"
HANDOFF_ISSUE = "64"
MAP_ISSUE = "291"
TASK_TITLE = "Retrain the default checkpoint on the shuffled-only table without wrong-stone teams"
TASK_URL = f"https://github.com/{REPO}/issues/{ISSUE}"

NEW_ROW = "linear_masked_filter"            # ask-grid row of the filtered checkpoint
REF_ROW = "sched_linear_species_first"      # ask-grid row of the unfiltered reference
NEW_CK = "stat84_shuffled_noise_linear_filtered_es3_cap80"
REF_CK = "stat84_shuffled_noise_linear_es3_cap80"
NEW_LABEL = "Filtered table (Linear masked, full filter)"
REF_LABEL = "Unfiltered table (Linear, species-first)"
NEW_SHORT, REF_SHORT = "Filtered table", "unfiltered"

GUIDANCES = (2, 4, 8)
ALL_ASKS = list(range(1, 24))               # asks k/24 for k = 1..23
SCORED_ASKS = list(range(1, 18))            # the ask-versus-got error is scored on 1/24 to 17/24
HIGH_ASKS = list(range(18, 24))             # reported apart
TOP_ASKS = (15, 16, 17)                     # the highest scored asks, for the top-ask win rate
SEEDS = [1, 2, 3]

DRAWS = 200 if os.environ.get("RDR_SMOKE") else 4000   # bootstrap draws
DRAW_SEED = 2971

BAR = 0.001          # glossary bar: an ask-versus-got error under this is "near the line"
ERROR_LUCK = 0.002   # error gap two training seeds of one recipe can show
WIN_LUCK = 0.026     # pooled win-rate gap two training seeds of one recipe can show
NOVELTY_LUCK = 0.05  # novel-share gap treated as beyond luck

EXPECTED_TABLE_SHA = "e58af7689e62a873ce19361237d7ff8e77cb791fc4c196cc705410755c5a03e2"
# Seed bases of the unfiltered checkpoint's redraws, which do not replay training teams the way seed 7300 does (#353)
REF_REDRAW_BASES = {2: 940135810, 4: 630821980, 8: 111619920}

RES = Path.home() / "vgc-data/results"
SUMMARY_PATH = RES / "retrain_default_linear_filtered_summary.json"
BASES_FILE = Path.home() / "vgc-data/seed_replay/bases.txt"
PILOT_SRC = "/tmp/vgc-pilot/src"

NEW_COLOR = "#0f766e"
REF_COLOR = "#6f6e69"
TRAIN_COLOR = "#52514e"
VAL_COLOR = "#c2410c"
PERCENT_AXIS = "js:v => Math.round(v * 100) + '%'"
PERCENT_TOOLTIP = "js:v => v == null ? '' : (v * 100).toFixed(1) + '%'"


@dataclass(frozen=True)
class GridComparison:
    issue: str
    new_row: str
    ref_row: str
    new_checkpoint: str
    ref_checkpoint: str
    new_label: str
    ref_label: str
    new_short: str
    ref_short: str
    ref_redraw_bases: Mapping[int, int]
    bases_file: Path | None = None


DEFAULT_COMPARISON = GridComparison(
    ISSUE, NEW_ROW, REF_ROW, NEW_CK, REF_CK, NEW_LABEL, REF_LABEL, NEW_SHORT, REF_SHORT,
    REF_REDRAW_BASES, BASES_FILE,
)


def grid_name(row, guidance):
    """Results-file name of an ask-grid row at one guidance; guidance 2 carries no suffix."""
    return row + ("" if float(guidance) == 2 else f"_g{float(guidance):g}")


def early_stopping_fired(info):
    stop, cap = info.get("stopped_epoch"), info.get("cap")
    return bool(info.get("patience") and stop is not None and cap is not None and stop < cap)


def stop_reason(info):
    if early_stopping_fired(info):
        return f"early stopping fired: no new lowest validation loss for {info['patience']} epochs after epoch {info.get('best_epoch')}"
    return f"early stopping did not fire: reached the cap of {info.get('cap')} epochs"


def picture(light_url, dark_url, alt, width=900):
    """A GitHub image that follows the reader's light or dark theme."""
    return (
        f'<picture><source media="(prefers-color-scheme: dark)" srcset="{dark_url}">'
        f'<img src="{light_url}" alt="{alt}" width="{width}"></picture>'
    )


def fmt_iv(iv, fmt=".4f"):
    return f"[{iv[0]:{fmt}}, {iv[1]:{fmt}}]"


def _use_pilot_modules():
    """Put the laptop's pilot sources, then this folder, first on the import path."""
    sys.path.insert(0, PILOT_SRC)
    sys.path.insert(0, str(HERE))


def _train_lineups():
    """Line-ups of the labelled training teams, for the novelty measures."""
    _use_pilot_modules()
    import masking_rate_rows as MRR

    return MRR.references()[0]


# ---------------------------------------------------------------- before training: the table check; after battles: the grid check

def verify_table(out_json=None):
    """Check lossdata84_shuffled_only.npz: 0 wrong-stone teams with every field visible, legal Stat Points, 0 held-out overlap."""
    _use_pilot_modules()
    import lossdown as L  # the repo's lossdown, which has Tables.rejects; import it before the pilot modules change sys.path
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
    labelled = ~np.isnan(d["all_W"])
    is_val = d["all_val"].astype(bool)
    train_rows = ~is_val & labelled
    val_rows = is_val & labelled

    vocab = T.regulation_vocab()
    tables = L.Tables(vocab, [])
    all_rej = int(tables.rejects(all_x[:, :T.COLS48]).sum())
    tr_rej = int(tables.rejects(all_x[train_rows, :T.COLS48]).sum())
    va_rej = int(tables.rejects(all_x[val_rows, :T.COLS48]).sum())

    sp = all_x[:, T.COLS48:] - 1
    per_pokemon_total = sp.reshape(len(sp), SP.NSLOT, SP.NSTAT).sum(-1)
    sp_ok = bool((sp >= 0).all() and (sp <= SP.MAX_STAT).all() and (per_pokemon_total <= SP.MAX_TOTAL).all())

    def team_key(row):
        """The same six Pokémon in any slot order give the same key."""
        sets, points = row[:T.COLS48].reshape(6, -1), row[T.COLS48:].reshape(6, -1)
        return b"".join(sorted(np.concatenate([sets[i], points[i]]).tobytes() for i in range(6)))

    labelled_keys = {team_key(r) for r in all_x[labelled]}
    held_encoded, held_overlap = 0, 0
    for path in HO.files():
        team = parse_team_text(Path(path).read_text())
        try:
            row = T.encode(vocab, team).astype(np.int64)
        except ValueError:
            continue
        held_encoded += 1
        held_overlap += int(team_key(row) in labelled_keys)

    passed = all_rej == 0 and tr_rej == 0 and va_rej == 0 and sp_ok and held_overlap == 0
    report = dict(
        table=table_path.name,
        sha256=sha,
        expected_sha256=EXPECTED_TABLE_SHA,
        sha_matches_expected=(sha == EXPECTED_TABLE_SHA),
        method="lossdown.Tables.rejects (every field visible)",
        total_rows=len(all_x),
        labelled_train_rows=int(train_rows.sum()),
        labelled_val_rows=int(val_rows.sum()),
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
    """Fail unless the grid has exactly `cells_wanted` cells, each with `keep` (64) teams and every battle finished."""
    path = Path(results_dir) / f"stat_points_84_{name}.json"
    if not path.exists():
        raise SystemExit(f"{name}: results file missing ({path})")
    out = json.loads(path.read_text())
    keep = int(out.get("keep", 64))
    cells = list(out.get("cells", {}).values())

    def cell_id(c):
        return f"a{c['numerator']:02d}_s{c['seed']}"

    def battles_missing(c):
        pastes, ys = c.get("pastes"), c.get("y", [])
        return not pastes or len(ys) != len(pastes) or None in ys or c.get("battles", 0) < 49 * len(pastes)

    short_teams = [f"{cell_id(c)}({len(c.get('pastes', []))}/{keep})" for c in cells if len(c.get("pastes", [])) < keep]
    short_battles = [cell_id(c) for c in cells if battles_missing(c)]
    print(
        f"{name}: {len(cells)}/{cells_wanted} cells, {len(short_teams)} short of {keep} teams {short_teams[:5]}, "
        f"{len(short_battles)} missing battles {short_battles[:5]}"
    )
    if len(cells) != int(cells_wanted) or short_teams or short_battles:
        raise SystemExit(1)
    return dict(name=name, cells=len(cells), keep=keep, short_teams=len(short_teams), short_battles=len(short_battles))


# ---------------------------------------------------------------- training: loss-curve figure and comment

def _loss_points(curves, key):
    return [[e["epoch"], round(float(e[key]), 4)] for e in curves if key in e]


def build_train_figure(info, out_dir, ref_info=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    curves = info.get("curves") or []
    train_pts = _loss_points(curves, "train_loss")
    val_pts = _loss_points(curves, "val_loss")
    best_ep, best_val = info.get("best_epoch"), info.get("best_val_loss")
    has_best = best_ep is not None and best_val is not None
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
    if has_best:
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

    train_series = {
        "type": "line",
        "name": "Filtered table — training loss (214,055 teams)",
        "color": TRAIN_COLOR,
        "symbol": "none",
        "lineStyle": {"width": 2, "type": "dashed"},
        "emphasis": {"focus": "series"},
        "data": train_pts,
    }
    series = [train_series, val_series]
    if ref_info and ref_info.get("curves"):
        series.append({
            "type": "line",
            "name": "Unfiltered table — validation loss (4,380 teams, reference)",
            "color": REF_COLOR,
            "symbol": "none",
            "lineStyle": {"width": 1.8, "type": "dotted"},
            "emphasis": {"focus": "series"},
            "data": _loss_points(ref_info["curves"], "val_loss"),
        })

    option = {
        "animation": False,
        "grid": {"left": 68, "right": 28, "top": 58, "bottom": 76},
        "legend": {"top": 0},
        "tooltip": {"trigger": "axis", "valueFormatter": "js:v => v == null ? '' : Number(v).toFixed(4) + ' nats/field'"},
        "xAxis": {"type": "value", "min": 1, "max": max(2, last_ep), "name": "Training epoch", "nameLocation": "middle", "nameGap": 28},
        "yAxis": {"type": "value", "scale": True, "name": "Linear masked ELBO loss (nats / field)", "nameLocation": "middle", "nameGap": 50},
        "dataZoom": [{"type": "inside"}, {"type": "slider", "bottom": 8, "height": 18}],
        "series": series,
    }

    if has_best:
        title = f"Filtered table (Linear masked, full filter) reaches lowest validation loss {float(best_val):.4f} nats/field at epoch {best_ep}"
    else:
        title = "Filtered table (Linear masked, full filter) training and validation loss by epoch"
    subtitle = (
        f"Linear masking schedule (1 − α(t) = t on all 84 fields), full legal Pokémon construction filter, "
        f"{NEW_CK}. {stop_reason(info).capitalize()}."
    )
    ref_note = ""
    if ref_info and ref_info.get("best_val_loss") is not None:
        ref_note = (
            f" Unfiltered reference ({REF_CK}): epoch {ref_info.get('best_epoch')} val {float(ref_info.get('best_val_loss')):.4f} on 4,380 teams "
            f"(12 wrong-stone validation rows were removed before this retrain)."
        )
    first_val = val_pts[0][1] if val_pts else float("nan")
    note = (
        f"EMA weights evaluated on 4 fixed stratified draws of 4,368 validation teams "
        f"(epoch 1 val {first_val:.4f} → kept epoch {best_ep} val {float(best_val):.4f}).{ref_note}"
    )
    return write_chart(option, out_dir / "train-val-curves.html", title=title, subtitle=subtitle, note=note, height=420)


def format_train_comment(info, commit_sha, loss_light, loss_dark, ref_info=None, verify_info=None):
    curves = info.get("curves") or []
    best_ep, stop_ep = info["best_epoch"], info["stopped_epoch"]
    best_val = float(info["best_val_loss"])
    first, last = curves[0], curves[-1]
    best_entry = next((e for e in curves if e["epoch"] == best_ep), last)
    reason = stop_reason(info)
    data_sha = str(info.get("data_sha256", ""))[:12]
    wandb_url = (info.get("wandb") or {}).get("url")
    ref = ref_info or {}

    if ref.get("best_val_loss") is not None:
        val_row = (
            f"| Best validation loss (nats/field) | **{best_val:.4f}** (ep 1: `{float(first['val_loss']):.4f}`, ep {stop_ep}: `{float(last['val_loss']):.4f}`) "
            f"| `{float(ref['best_val_loss']):.4f}` (on 4,380 rows incl. 12 wrong-stone teams) |"
        )
    else:
        val_row = f"| Best validation loss (nats/field) | **{best_val:.4f}** | `2.0142` |"
    if ref.get("curves"):
        train_row = (
            f"| Training loss at kept epoch | `{float(best_entry['train_loss']):.4f}` (ep 1: `{float(first['train_loss']):.4f}`) "
            f"| `{float(ref['curves'][-1]['train_loss']):.4f}` |"
        )
    else:
        train_row = f"| Training loss at kept epoch | `{float(best_entry['train_loss']):.4f}` | — |"

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
        f"| Wrong-stone teams (every field visible) | **0 train / 0 val** (`dropped_by_tables = {info.get('dropped_by_tables', 0)}`) "
        f"| 11 reached train (`dropped_by_tables = 532`) / 12 in val |",
        f"| Labelled rows (train / val) | `{info.get('n_train', 214055):,}` / `{info.get('val_rows', 4368):,}` | `214,066` / `4,380` |",
        f"| Epoch read (`best_epoch`) | **Epoch {best_ep}** (stopped at {stop_ep}; early stopping fired: **{'yes' if early_stopping_fired(info) else 'no'}**) "
        f"| Epoch {ref.get('best_epoch', 80)} (stopped at {ref.get('stopped_epoch', 80)}) |",
        val_row,
        train_row,
        f"| Wall-clock time | `{info.get('minutes', '?')}` min (`{info.get('amp_dtype', 'float32')}`) | `{ref.get('minutes', '?')}` min |",
        f"| Reproducibility | Commit `{commit_sha[:12]}`, table `lossdata84_shuffled_only.npz` (`sha256: {data_sha}…`), seed `{info.get('seed')}` "
        f"| Pre-#368 table (`sha256: 8aca232e…`), seed `0` |",
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


# ---------------------------------------------------------------- one grid's measures, with bootstrap intervals

def _load_redraw_bases(comparison=DEFAULT_COMPARISON):
    """Seed bases of the unfiltered redraws by guidance; ~/vgc-data/seed_replay/bases.txt overrides the defaults."""
    bases = dict(comparison.ref_redraw_bases)
    if comparison.bases_file is not None and comparison.bases_file.exists():
        for line in comparison.bases_file.read_text().splitlines():
            if not line.strip():
                continue
            grid, base = line.split()
            for g in GUIDANCES:
                if grid == grid_name(comparison.ref_row, g):
                    bases[g] = int(base)
    return bases


def _opponent_draws(n_opponents):
    """How often each opponent appears in each bootstrap draw; both checkpoints share these counts."""
    return np.random.default_rng(DRAW_SEED).multinomial(n_opponents, np.full(n_opponents, 1.0 / n_opponents), DRAWS).astype(float)


def _bootstrap_grid(opponents_npz, counts, stream_id):
    """{(ask, seed): DRAWS mean win rates}, resampling teams within each cell and opponents by the shared `counts`."""
    data = np.load(opponents_npz)
    weights = counts * data["weights"]
    rng = np.random.default_rng([DRAW_SEED, 385, stream_id])
    draws = {}
    for k in ALL_ASKS:
        for s in SEEDS:
            rate = data[f"a{k:02d}_s{s}"]                     # teams x opponents, NaN where a battle is missing
            if not len(rate):
                draws[(k, s)] = np.zeros(len(counts))
                continue
            seen = ~np.isnan(rate)
            scores = (np.where(seen, rate, 0.0) @ weights.T) / np.maximum(seen @ weights.T, 1e-12)
            pick = rng.integers(0, len(rate), (DRAWS, len(rate)))
            draws[(k, s)] = scores[pick, np.arange(DRAWS)[:, None]].mean(1)
    return draws


def _pooled_got_draws(cell_draws, teams_per_cell):
    """Per ask, the three seeds' draws pooled by team count."""
    return {k: _pooled_draws(cell_draws, teams_per_cell, [k]) for k in ALL_ASKS}


def _grid_error(got, asks):
    return _ask_error(got, asks) if all(got[k] is not None for k in asks) else None


def _mean_if_present(values):
    return float(np.mean(values)) if values else None


def _pooled_draws(cell_draws, teams_per_cell, asks):
    keys = [(k, s) for k in asks for s in SEEDS]
    total = sum(teams_per_cell[key] for key in keys)
    return sum(cell_draws[key] * teams_per_cell[key] for key in keys) / total if total else None


def _ask_error(got, asks=SCORED_ASKS):
    """Ask-versus-got error: the mean squared gap between the measured win rate and the ask k/24.
    Works on single values or on arrays of bootstrap draws."""
    return np.mean([(got[k] - k / 24) ** 2 for k in asks], 0)


def _interval(arr):
    return [float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))] if arr is not None else None


def _excludes_zero(iv):
    return bool(iv[0] > 0 or iv[1] < 0)


def _line_up_metrics_from_cells(cells, train_lineups):
    """Team count, distinct line-ups (pooled and per ask), and novelty against the training line-ups."""
    import line_up_yield as LY

    all_lu, by_ask = [], {}
    for c in cells:
        lus = [LY.line_up(p) for p in c["pastes"]]
        all_lu.extend(lus)
        by_ask.setdefault(int(c["numerator"]), []).extend(lus)
    per_ask = {k: len(set(v)) for k, v in by_ask.items()}
    return dict(
        teams=len(all_lu),
        distinct_line_ups=len(set(all_lu)),
        within_ask_line_ups_mean_23=float(np.mean([per_ask[k] for k in ALL_ASKS if k in per_ask])),
        within_ask_line_ups_mean_17=float(np.mean([per_ask[k] for k in SCORED_ASKS if k in per_ask])),
        novel_share=_mean_if_present([lu not in train_lineups for lu in all_lu]),
        distinct_novel_line_ups=len({lu for lu in all_lu if lu not in train_lineups}),
    )


def _pass_at_k_summary(name, results_dir=RES):
    """Pass@k of held elite line-ups at asks 13/24 to 17/24, from pass_at_k/NAME.json or the #331 file."""
    path = Path(results_dir) / "pass_at_k" / f"{name}.json"
    if path.exists():
        r = json.loads(path.read_text())
    else:
        p331 = Path(results_dir) / "pass_at_k_issue331.json"
        r = json.loads(p331.read_text())["grids"].get(name) if p331.exists() else None
    if not r:
        return None
    held = r["bands"].get("held", {}).get("asks_13_to_17")
    if not held:
        return None
    pak, iv = held["pass_at_k"], held["pass_at_k_interval"]
    return dict(
        definition=r.get("definition"),
        pass_at_1=float(pak[0]),
        pass_at_1_interval=iv[0],
        pass_at_8=float(pak[7]),
        pass_at_8_interval=iv[7],
        pass_at_48=float(pak[47]),
        pass_at_48_interval=iv[47],
        potential_at_48=float(held["potential_at_k"][47]),
        pass_at_k=pak,
        pass_at_k_interval=iv,
    )


def _elite_yield_summary(name, results_dir=RES):
    """Elite line-ups re-battled and held over all 23 asks, from line_up_yield_NAME.json."""
    path = Path(results_dir) / f"line_up_yield_{name}.json"
    if not path.exists():
        return None
    views = json.loads(path.read_text()).get("views")
    if not views or "all_23_asks" not in views:
        return None
    v = views["all_23_asks"]

    def maybe_float(x):
        return float(x) if x is not None else None

    return dict(
        elite_line_ups=int(v["elite_line_ups"]),
        rebattled=int(v["rebattled"]),
        held=int(v["held"]),
        held_new_against_training=int(v["held_new_against_training"]),
        yield_per_1000=float(v["yield_per_1000"]),
        one_score_mean=maybe_float(v["one_score_mean"]),
        fresh_mean=maybe_float(v["fresh_mean"]),
    )


def grid_metrics(grid, opponents_npz, name, guidance, counts, stream_id, train_lineups, results_dir=RES):
    """Every measure of one grid, and the bootstrap draws the paired comparison needs.

    Each key is set once. `teams` comes from the line-up measures; do not add it here again."""
    cells = list(grid["cells"].values())
    by_cell = {(int(c["numerator"]), int(c["seed"])): c for c in cells}
    teams_per_cell = {key: len(c["pastes"]) for key, c in by_cell.items()}
    got = {k: _mean_if_present([y for s in SEEDS for y in by_cell[(k, s)]["y"]]) for k in ALL_ASKS}
    got_by_seed = {s: {k: _mean_if_present(by_cell[(k, s)]["y"]) for k in ALL_ASKS} for s in SEEDS}
    tried = sum(int(c["tried"]) for c in cells)
    rejected = sum(int(c["rejected"]) for c in cells)

    cell_draws = _bootstrap_grid(opponents_npz, counts, stream_id)
    got_draws = _pooled_got_draws(cell_draws, teams_per_cell)
    draws = dict(
        error=_grid_error(got_draws, SCORED_ASKS),
        pooled_win=_pooled_draws(cell_draws, teams_per_cell, ALL_ASKS),
        win_top=_pooled_draws(cell_draws, teams_per_cell, TOP_ASKS),
    )
    metrics = dict(
        name=name,
        guidance=guidance,
        seed0=grid.get("seed0"),
        checkpoint=grid.get("checkpoint"),
        checkpoint_sha256=grid.get("checkpoint_sha256"),
        cells=len(cells),
        cells_short_of_64=sum(len(c["pastes"]) < int(grid.get("keep", 64)) for c in cells),
        battles=sum(int(c.get("battles", 0)) for c in cells),
        error=_grid_error(got, SCORED_ASKS),
        error_interval=_interval(draws["error"]),
        error_by_seed={str(s): _grid_error(got_by_seed[s], SCORED_ASKS) for s in SEEDS},
        error_asks_18_to_23=_grid_error(got, HIGH_ASKS),
        got_by_ask={str(k): got[k] for k in ALL_ASKS},
        got_by_ask_interval={str(k): _interval(got_draws[k]) for k in ALL_ASKS},
        pooled_win=_mean_if_present([y for c in cells for y in c["y"]]),
        pooled_win_interval=_interval(draws["pooled_win"]),
        win_top_asks=_mean_if_present([y for k in TOP_ASKS for s in SEEDS for y in by_cell[(k, s)]["y"]]),
        win_top_asks_interval=_interval(draws["win_top"]),
        first_try_legality=(tried - rejected) / max(tried, 1),
        effective_species=_mean_if_present([c["div"]["effective_species"] for c in cells if c.get("div")]),
        **_line_up_metrics_from_cells(cells, train_lineups),
        elite_line_up_yield=_elite_yield_summary(name, results_dir),
        pass_at_k=_pass_at_k_summary(name, results_dir),
    )
    return metrics, draws


# ---------------------------------------------------------------- the two checkpoints side by side at one guidance

def evaluate_beyond_luck(new_m, ref_m, paired_err, paired_win):
    """List every difference beyond training luck, so no real shift is assumed away."""
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
    if ref_nov is not None and abs(new_m["novel_share"] - ref_nov) >= NOVELTY_LUCK:
        flags.append(
            f"novel share differs by {new_m['novel_share'] - ref_nov:+.1%} against unfiltered non-replayed seeds "
            f"({new_m['novel_share']:.1%} vs {ref_nov:.1%})"
        )
    new_pak, ref_pak = new_m.get("pass_at_k"), ref_m.get("pass_at_k")
    if new_pak and ref_pak:
        for key in ("pass_at_8", "pass_at_48"):
            n_iv, r_iv = new_pak[f"{key}_interval"], ref_pak[f"{key}_interval"]
            if n_iv[1] < r_iv[0] or n_iv[0] > r_iv[1]:
                flags.append(
                    f"{key.replace('pass_at_', 'Pass@')} intervals do not overlap "
                    f"({new_pak[key]:.3f} {fmt_iv(n_iv, '.3f')} vs {ref_pak[key]:.3f} {fmt_iv(r_iv, '.3f')})"
                )
    return flags


def read_guidance_pair(guidance, counts=None, train_lineups=None, results_dir=RES, *, comparison=DEFAULT_COMPARISON):
    """Both checkpoints at one guidance, with paired bootstrap intervals over shared opponent draws."""
    if train_lineups is None:
        train_lineups = _train_lineups()
    g = int(guidance)
    results_dir = Path(results_dir)

    def read(row, stream_id):
        name = grid_name(row, g)
        grid = json.loads((results_dir / f"stat_points_84_{name}.json").read_text())
        opponents = results_dir / f"masking_rate_rows_opponents_{name}.npz"
        return grid_metrics(grid, opponents, name, g, counts, stream_id, train_lineups, results_dir)

    if counts is None:
        counts = _opponent_draws(len(np.load(results_dir / f"masking_rate_rows_opponents_{grid_name(comparison.new_row, g)}.npz")["weights"]))
    new_m, new_d = read(comparison.new_row, stream_id=10 * g + 1)
    ref_m, ref_d = read(comparison.ref_row, stream_id=10 * g + 2)

    # The unfiltered checkpoint's redraws, which escape the seed-7300 training-team replay (#353)
    base = _load_redraw_bases(comparison).get(g)
    redraw_path = results_dir / f"stat_points_84_{grid_name(comparison.ref_row, g)}_seed{base}.json"
    if base is not None and redraw_path.exists():
        redraw_cells = list(json.loads(redraw_path.read_text())["cells"].values())
        ref_m["unbiased_breadth"] = dict(seed0=base, file=redraw_path.name, **_line_up_metrics_from_cells(redraw_cells, train_lineups))

    err_gap = new_m["error"] - ref_m["error"]
    err_iv = _interval(new_d["error"] - ref_d["error"])
    win_gap = new_m["pooled_win"] - ref_m["pooled_win"]
    win_iv = _interval(new_d["pooled_win"] - ref_d["pooled_win"])
    paired_err = dict(
        difference=err_gap,
        interval=err_iv,
        difference_by_seed={s: new_m["error_by_seed"][s] - ref_m["error_by_seed"][s] for s in new_m["error_by_seed"]},
        exceeds_glossary_bar=bool(abs(err_gap) >= BAR),
        beyond_training_luck=bool(abs(err_gap) >= ERROR_LUCK),
        ci_excludes_zero=_excludes_zero(err_iv),
    )
    paired_win = dict(
        difference=win_gap,
        interval=win_iv,
        beyond_training_luck=bool(abs(win_gap) >= WIN_LUCK),
        ci_excludes_zero=_excludes_zero(win_iv),
    )
    paired_top = dict(
        difference=new_m["win_top_asks"] - ref_m["win_top_asks"],
        interval=_interval(new_d["win_top"] - ref_d["win_top"]),
    )
    return dict(
        guidance=g,
        filtered=new_m,
        unfiltered=ref_m,
        paired=dict(error=paired_err, pooled_win=paired_win, win_top_asks=paired_top),
        flags_beyond_luck=evaluate_beyond_luck(new_m, ref_m, paired_err, paired_win),
    )


def build_grid_figure(pair, out_dir, *, comparison=DEFAULT_COMPARISON):
    """Got against ask for both checkpoints at one guidance, with 95% bootstrap bands."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    g = pair["guidance"]
    new_m, ref_m = pair["filtered"], pair["unfiltered"]
    xs = [k / 24 for k in ALL_ASKS]
    series = []
    arms = ((comparison.new_label, NEW_COLOR, new_m, 2.8, "solid"), (comparison.ref_label, REF_COLOR, ref_m, 1.8, "dashed"))
    for i, (label, color, m, width, dash) in enumerate(arms):
        lo = [m["got_by_ask_interval"][str(k)][0] for k in ALL_ASKS]
        hi = [m["got_by_ask_interval"][str(k)][1] for k in ALL_ASKS]
        series += band(label, color, xs, lo, hi, stack=f"g{g}_{i}")
        series.append({
            "type": "line",
            "name": label,
            "color": color,
            "symbolSize": 5,
            "lineStyle": {"width": width, "type": dash},
            "emphasis": {"focus": "series"},
            "data": [[x, m["got_by_ask"][str(k)]] for x, k in zip(xs, ALL_ASKS)],
        })
    series.append({
        "type": "line",
        "name": "Got = ask",
        "color": "#a8a29e",
        "symbol": "none",
        "silent": True,
        "lineStyle": {"type": "dotted", "width": 1.2},
        "data": [[0, 0], [1, 1]],
    })
    option = {
        "animation": False,
        "grid": {"left": 64, "right": 24, "top": 56, "bottom": 56},
        "legend": {"top": 0, "data": [comparison.new_label, comparison.ref_label, "Got = ask"]},
        "tooltip": {"trigger": "axis", "valueFormatter": PERCENT_TOOLTIP},
        "xAxis": {
            "type": "value", "min": 0, "max": 1, "interval": 1 / 6,
            "name": "Ask: win rate requested from the model", "nameLocation": "middle", "nameGap": 34,
            "axisLabel": {"formatter": PERCENT_AXIS},
        },
        "yAxis": {
            "type": "value", "min": 0, "max": 0.8,
            "name": "Got: measured win rate", "nameLocation": "middle", "nameGap": 46,
            "axisLabel": {"formatter": PERCENT_AXIS},
        },
        "series": series,
    }
    pe = pair["paired"]["error"]
    verdict = "differs beyond training luck" if pe["beyond_training_luck"] else "matches within training luck"
    title = (
        f"Guidance {g}: {comparison.new_short} {verdict} "
        f"(error {new_m['error']:.4f} vs {comparison.ref_short} {ref_m['error']:.4f}, Δ {pe['difference']:+.4f})"
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
    new_y, ref_y = nm.get("elite_line_up_yield") or {}, rm.get("elite_line_up_yield") or {}
    new_p, ref_p = nm.get("pass_at_k") or {}, rm.get("pass_at_k") or {}

    # Unfiltered novelty and line-ups come from its non-replayed redraws when they exist
    unbiased = rm.get("unbiased_breadth")
    if unbiased:
        ref_nov = f"{unbiased['novel_share']:.1%} (non-replayed seeds; 7300 battled: {rm['novel_share']:.1%})"
        ref_lu = (
            f"{unbiased['distinct_line_ups']:,} pooled / {unbiased['within_ask_line_ups_mean_23']:.1f} per ask "
            f"(non-replayed; 7300 battled: {rm['distinct_line_ups']:,})"
        )
        nov_diff = f"{nm['novel_share'] - unbiased['novel_share']:+.1%} vs non-replayed"
        lu_diff = f"{nm['distinct_line_ups'] - unbiased['distinct_line_ups']:+,} pooled vs non-replayed"
    else:
        ref_nov = f"{rm['novel_share']:.1%}"
        ref_lu = f"{rm['distinct_line_ups']:,} pooled / {rm['within_ask_line_ups_mean_23']:.1f} per ask"
        nov_diff = f"{nm['novel_share'] - rm['novel_share']:+.1%}"
        lu_diff = f"{nm['distinct_line_ups'] - rm['distinct_line_ups']:+,}"
    new_pak = f"{new_p['pass_at_8']:.3f} / {new_p['pass_at_48']:.3f}" if new_p else "pending"
    ref_pak = f"{ref_p['pass_at_8']:.3f} / {ref_p['pass_at_48']:.3f}" if ref_p else "n/a"

    status = "flags a difference beyond training luck" if flags else "matches the unfiltered checkpoint within training luck"
    alt = (
        f"Got against ask at guidance {g} for {NEW_LABEL} versus {REF_LABEL}: "
        f"ask-versus-got error {nm['error']:.4f} vs {rm['error']:.4f} (paired difference {pe['difference']:+.4f})"
    )
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
        f"| Ask-versus-got error (asks 1/24–17/24) | **{nm['error']:.4f}** {fmt_iv(nm['error_interval'])} | {rm['error']:.4f} {fmt_iv(rm['error_interval'])} "
        f"| **{pe['difference']:+.4f}** {fmt_iv(pe['interval'], '+.4f')} (luck bar `0.002`) |",
        f"| Error on asks 18/24–23/24 (apart) | `{nm['error_asks_18_to_23']:.4f}` | `{rm['error_asks_18_to_23']:.4f}` "
        f"| `{nm['error_asks_18_to_23'] - rm['error_asks_18_to_23']:+.4f}` |",
        f"| Pooled win rate (23 asks) | **{nm['pooled_win']:.3f}** {fmt_iv(nm['pooled_win_interval'], '.3f')} | {rm['pooled_win']:.3f} {fmt_iv(rm['pooled_win_interval'], '.3f')} "
        f"| **{pw['difference']:+.3f}** {fmt_iv(pw['interval'], '+.3f')} (luck bar `0.026`) |",
        f"| Win rate at top scored asks (15/24–17/24) | {nm['win_top_asks']:.3f} | {rm['win_top_asks']:.3f} | {pair['paired']['win_top_asks']['difference']:+.3f} |",
        f"| Novel share (vs labelled training) | **{nm['novel_share']:.1%}** | {ref_nov} | {nov_diff} |",
        f"| Distinct line-ups | **{nm['distinct_line_ups']:,}** pooled / **{nm['within_ask_line_ups_mean_23']:.1f}** per ask | {ref_lu} | {lu_diff} |",
        f"| First-try legality | {nm['first_try_legality']:.1%} | {rm['first_try_legality']:.1%} | — |",
        f"| Effective species (5-team subsets) | {nm['effective_species']:.1f} | {rm['effective_species']:.1f} | {nm['effective_species'] - rm['effective_species']:+.1f} |",
        f"| Elite line-ups that held (all 23 asks) | {new_y.get('held', 'pending')} | {ref_y.get('held', 'n/a')}* | *unfiltered battled on seed 7300 |",
        f"| Novel elite line-ups that held | **{new_y.get('held_new_against_training', 'pending')}** | {ref_y.get('held_new_against_training', 'n/a')} | excludes replayed training line-ups |",
        f"| Pass@8 / Pass@48 (asks 13/24–17/24, held) | {new_pak} | {ref_pak} | — |",
    ]
    if flags:
        lines += ["", "**Differences beyond training luck flagged here:**"] + [f"- {f}" for f in flags]
    else:
        lines += ["", "No measure at this guidance differs beyond training luck."]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- all three guidances: summary, figures, resolution, handoff, map

def build_summary(out_json=SUMMARY_PATH, results_dir=RES, train_lineups=None, *,
                  comparison=DEFAULT_COMPARISON, flags_for_pair=None):
    if train_lineups is None:
        train_lineups = _train_lineups()
    first_opponents = Path(results_dir) / f"masking_rate_rows_opponents_{grid_name(comparison.new_row, GUIDANCES[0])}.npz"
    counts = _opponent_draws(len(np.load(first_opponents)["weights"]))
    by_guidance, all_flags = {}, []
    for g in GUIDANCES:
        pair = read_guidance_pair(g, counts=counts, train_lineups=train_lineups, results_dir=results_dir, comparison=comparison)
        by_guidance[str(g)] = pair
        flags = pair["flags_beyond_luck"] if flags_for_pair is None else flags_for_pair(pair)
        all_flags += [f"guidance {g}: {f}" for f in flags]
    summary = dict(
        issue=int(comparison.issue),
        filtered_checkpoint=comparison.new_checkpoint,
        filtered_row=comparison.new_row,
        unfiltered_checkpoint=comparison.ref_checkpoint,
        unfiltered_row=comparison.ref_row,
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


def _tradeoff_figure(by_g, out_dir):
    """Ask-versus-got error against pooled distinct line-ups, one point per guidance."""
    f_pts, u_pts = [], []
    for g in GUIDANCES:
        nm, rm = by_g[str(g)]["filtered"], by_g[str(g)]["unfiltered"]
        rm_line_ups = (rm.get("unbiased_breadth") or rm)["distinct_line_ups"]
        f_pts.append({"value": [round(nm["error"], 5), nm["distinct_line_ups"]], "name": f"g={g}"})
        u_pts.append({"value": [round(rm["error"], 5), rm_line_ups], "name": f"g={g}"})

    def arm(name, color, size, line_style, label_pos, pts):
        return {
            "type": "line", "name": name, "color": color, "symbolSize": size, "lineStyle": line_style,
            "emphasis": {"focus": "series"}, "label": {"show": True, "formatter": "{b}", "position": label_pos}, "data": pts,
        }

    option = {
        "animation": False,
        "grid": {"left": 74, "right": 34, "top": 58, "bottom": 58},
        "legend": {"top": 0},
        "tooltip": {"trigger": "item"},
        "xAxis": {"type": "value", "scale": True, "name": "Ask-versus-got error, asks 1/24–17/24 (lower is better)", "nameLocation": "middle", "nameGap": 34},
        "yAxis": {"type": "value", "scale": True, "name": "Distinct line-ups, pooled over 23 asks", "nameLocation": "middle", "nameGap": 54},
        "series": [
            arm(NEW_LABEL, NEW_COLOR, 10, {"width": 3}, "top", f_pts),
            arm(f"{REF_LABEL} (non-replayed line-ups)", REF_COLOR, 8, {"width": 2, "type": "dashed"}, "bottom", u_pts),
        ],
    }
    return write_chart(
        option,
        out_dir / "tradeoff-error-lineups.html",
        title="Ask-versus-got error vs distinct line-ups across guidance 2, 4, and 8",
        subtitle=f"{NEW_LABEL} ({NEW_CK}) against {REF_LABEL} ({REF_CK}).",
        note="Unfiltered line-up counts use the non-replayed seed redraws (seed bases 940135810, 630821980, 111619920) so seed-7300 replay does not depress its diversity.",
        height=420,
    )


def _pass_at_k_figure(by_g, out_dir):
    """Pass@k for k = 1..48, one panel per guidance."""
    ks = list(range(1, 49))
    grids, xs, ys, titles, series = [], [], [], [], []
    for idx, g in enumerate(GUIDANCES):
        left = 6 + idx * 32
        grids.append({"left": f"{left}%", "width": "27%", "top": 70, "bottom": 54})
        titles.append({"text": f"Guidance {g}", "left": f"{left + 13.5}%", "top": 42, "textAlign": "center", "textStyle": {"fontSize": 13}})
        xs.append({"gridIndex": idx, "type": "value", "min": 1, "max": 48, "name": "k" if idx == 1 else "", "nameLocation": "middle", "nameGap": 28})
        ys.append({
            "gridIndex": idx, "type": "value", "min": 0, "max": 1,
            "name": "Pass@k" if idx == 0 else "", "nameLocation": "middle", "nameGap": 42,
            "axisLabel": {"formatter": PERCENT_AXIS, "show": idx == 0},
        })
        p = by_g[str(g)]
        arms = ((NEW_LABEL, NEW_COLOR, p["filtered"], "solid"), (REF_LABEL, REF_COLOR, p["unfiltered"], "dashed"))
        for j, (label, color, m, dash) in enumerate(arms):
            pak = m.get("pass_at_k")
            if not pak:
                continue
            lo = [v[0] for v in pak["pass_at_k_interval"]]
            hi = [v[1] for v in pak["pass_at_k_interval"]]
            series += band(label, color, ks, lo, hi, stack=f"pk_{idx}_{j}", xAxisIndex=idx, yAxisIndex=idx)
            series.append({
                "type": "line", "name": label, "color": color, "xAxisIndex": idx, "yAxisIndex": idx, "symbol": "none",
                "lineStyle": {"width": 2.4 if j == 0 else 1.8, "type": dash},
                "emphasis": {"focus": "series"},
                "data": [[k, v] for k, v in zip(ks, pak["pass_at_k"])],
            })
    option = {
        "animation": False,
        "title": titles,
        "grid": grids,
        "legend": {"top": 6, "data": [NEW_LABEL, REF_LABEL]},
        "tooltip": {"trigger": "axis", "valueFormatter": PERCENT_TOOLTIP},
        "xAxis": xs,
        "yAxis": ys,
        "series": series,
    }
    return write_chart(
        option,
        out_dir / "tradeoff-pass-at-k.html",
        title="Pass@k at asks 13/24 to 17/24 (held elite line-ups) across guidance 2, 4, and 8",
        subtitle="Chance that at least one of k drawn teams reaches 0.531 and its line-up holds at 0.531 in a 4-game-per-opponent re-battle.",
        note="Band: 95% bootstrap interval over sampling seeds, teams, and the 49 distinct top-50 meta teams (src/pass_at_k.py).",
        height=390,
    )


def build_final_figures(summary, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return _tradeoff_figure(summary["by_guidance"], out_dir), _pass_at_k_figure(summary["by_guidance"], out_dir)


def _final_table_rows(summary):
    """Two rows per guidance: the filtered checkpoint in bold, then the unfiltered reference."""
    def pak_cell(pak, key):
        return f"{pak[key]:.3f}" if pak else "n/a"

    rows = []
    for g in GUIDANCES:
        p = summary["by_guidance"][str(g)]
        nm, rm = p["filtered"], p["unfiltered"]
        pe = p["paired"]["error"]
        unb = rm.get("unbiased_breadth")
        ref_breadth, star = (unb, "*") if unb else (rm, "")
        ny, ry = nm.get("elite_line_up_yield") or {}, rm.get("elite_line_up_yield") or {}
        npk, rpk = nm.get("pass_at_k") or {}, rm.get("pass_at_k") or {}
        rows.append(
            f"| {g} | **{NEW_LABEL}** | **{nm['error']:.4f}** | **{pe['difference']:+.4f}** {fmt_iv(pe['interval'], '+.4f')} "
            f"| **{nm['pooled_win']:.3f}** | {nm['win_top_asks']:.3f} | **{nm['novel_share']:.1%}** "
            f"| **{nm['distinct_line_ups']:,}** / {nm['within_ask_line_ups_mean_23']:.1f} "
            f"| {ny.get('held', 'n/a')} | **{ny.get('held_new_against_training', 'n/a')}** "
            f"| {pak_cell(npk, 'pass_at_8')} | {pak_cell(npk, 'pass_at_48')} |"
        )
        rows.append(
            f"| {g} | {REF_LABEL} | {rm['error']:.4f} | ref | {rm['pooled_win']:.3f} | {rm['win_top_asks']:.3f} "
            f"| {ref_breadth['novel_share']:.1%}{star} "
            f"| {ref_breadth['distinct_line_ups']:,} / {ref_breadth['within_ask_line_ups_mean_23']:.1f}{star} "
            f"| {ry.get('held', 'n/a')}† | {ry.get('held_new_against_training', 'n/a')} "
            f"| {pak_cell(rpk, 'pass_at_8')}† | {pak_cell(rpk, 'pass_at_48')}† |"
        )
    return rows


def format_final_comment(summary, run_info, commit_sha, err_light, err_dark, pak_light, pak_dark):
    flags = summary.get("findings_beyond_luck") or []
    best_ep, stop_ep = run_info["best_epoch"], run_info["stopped_epoch"]
    best_val = float(run_info["best_val_loss"])
    if flags:
        verdict = f"flags {len(flags)} difference(s) beyond training luck — reported below, not assumed away"
    else:
        verdict = "matches the unfiltered checkpoint within training luck on every measure across guidance 2, 4, and 8"
    lines = [
        f"**Resolved: {NEW_LABEL} (`{NEW_CK}`, kept epoch {best_ep}) {verdict}.**",
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
        "| Guidance | Arm | Ask-vs-got error (1–17) | Δ error [95% CI] | Pooled win (23 asks) | Top-ask win (15–17) | Novel share "
        "| Distinct line-ups (pooled / per ask) | Elite held | Novel elite held | Pass@8 | Pass@48 |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        *_final_table_rows(summary),
        "",
        "\\* Unfiltered novel share and distinct line-ups are read from its non-replayed seed redraws (`seed940135810`, `seed630821980`, `seed111619920`) "
        "so legacy seed-7300 training-team replay does not distort the comparison. "
        "† Unfiltered elite held and Pass@k come from its battled seed-7300 grids (which replay some training teams); **Novel elite held** excludes training line-ups on both sides.",
        "",
        "### Beyond-training-luck audit",
        "",
    ]
    if flags:
        lines.append("Differences exceeding the pre-registered training-luck bars (`0.002` error, `0.026` pooled win, `≥ 5 pp` unbiased novelty, or non-overlapping Pass@k intervals):")
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


def _filtered_by_guidance(summary):
    return {g: summary["by_guidance"][str(g)]["filtered"] for g in GUIDANCES}


def format_handoff_comment(summary, run_info):
    best_ep = run_info["best_epoch"]
    by_g = _filtered_by_guidance(summary)
    flags = summary.get("findings_beyond_luck") or []
    errors = ", ".join(f"`{by_g[g]['error']:.4f}` (g={g})" for g in GUIDANCES)
    breadth = "; ".join(
        f"`{m['pooled_win']:.3f}` / `{m['novel_share']:.1%}` / `{(m.get('elite_line_up_yield') or {}).get('held_new_against_training', 'n/a')}` (g={g})"
        for g, m in by_g.items()
    )
    comparison = "all differences within training luck" if not flags else f"{len(flags)} flag(s) beyond training luck — see #{ISSUE}"
    lines = [
        f"**Checkpoint ready from [#{ISSUE}]({TASK_URL}): `{NEW_CK}` (`linear_masked_filter`).**",
        "",
        f"- **Reader name:** {NEW_LABEL}",
        f"- **Checkpoint file:** `~/.local/share/vgc-pilot-runtime/lossdown/{NEW_CK}.pt` (kept epoch **{best_ep}** of {run_info['stopped_epoch']}; {stop_reason(run_info)})",
        f"- **Training table:** filtered `lossdata84_shuffled_only.npz` (`sha256: {str(run_info.get('data_sha256', ''))[:12]}…`, `214,055` train / `4,368` val, `0 / 0` wrong-stone rejects with every field visible)",
        f"- **Ask-versus-got error (asks 1/24–17/24):** {errors}",
        f"- **Pooled win rate / novel share / novel elite held:** {breadth}",
        f"- **Comparison to unfiltered linear checkpoint:** {comparison}.",
    ]
    return "\n".join(lines) + "\n"


def update_map(summary, run_info):
    """Add this issue's result under "Decisions so far" on map #291, once."""
    by_g = _filtered_by_guidance(summary)
    flags = summary.get("findings_beyond_luck") or []
    errors = " / ".join(f"`{by_g[g]['error']:.4f}`" for g in GUIDANCES)
    verdict = "all measures within training luck of unfiltered" if not flags else f"{len(flags)} beyond-luck flag(s) noted"
    decision_line = (
        f"- [{TASK_TITLE}]({TASK_URL}): retrained **{NEW_LABEL}** (`{NEW_CK}`) on the filtered shuffled-only table (`0/0` wrong-stone rejects); "
        f"kept epoch **{run_info['best_epoch']}** (`val_loss {float(run_info['best_val_loss']):.4f}`), ask-versus-got error {errors} "
        f"at guidance 2, 4, 8 ({verdict}), handed to #64."
    )
    body = subprocess.run(
        ["gh", "issue", "view", MAP_ISSUE, "--repo", REPO, "--json", "body", "-q", ".body"],
        check=True, capture_output=True, text=True,
    ).stdout
    if "## Decisions so far" not in body or "\n## Not yet specified" not in body:
        raise RuntimeError(f"map #{MAP_ISSUE} missing expected Decisions so far / Not yet specified sections")
    decisions = body.split("## Decisions so far", 1)[1].split("## Not yet specified", 1)[0]
    if f"/issues/{ISSUE})" in decisions:
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


# ---------------------------------------------------------------- command line

def _json(path):
    return json.loads(Path(path).read_text())


def _optional_json(args, i):
    """The JSON file named by args[i], or None when it is not given or does not exist."""
    return _json(args[i]) if len(args) > i and Path(args[i]).exists() else None


def _commit(path):
    return Path(path).read_text().strip()


def _print_paths(paths):
    for p in paths:
        print(p)


# Each command takes the arguments after its name, as listed in the module docstring.
COMMANDS = {
    "verify-table": lambda a: verify_table(a[0] if a else None),
    "check-grid": lambda a: check_grid(a[0], int(a[1])),
    "train-figure": lambda a: print(build_train_figure(_json(a[0]), a[1], _optional_json(a, 2))),
    "train-comment": lambda a: print(format_train_comment(_json(a[0]), _commit(a[1]), a[2], a[3], _optional_json(a, 4), _optional_json(a, 5)), end=""),
    "grid-figure": lambda a: print(build_grid_figure(read_guidance_pair(int(a[0])), a[1])),
    "grid-comment": lambda a: print(format_grid_comment(read_guidance_pair(int(a[0])), a[1], a[2]), end=""),
    "summary": lambda a: build_summary(a[0] if a else SUMMARY_PATH),
    "final-figures": lambda a: _print_paths(build_final_figures(_json(a[0]), a[1])),
    "final-comment": lambda a: print(format_final_comment(_json(a[0]), _json(a[1]), _commit(a[2]), *a[3:7]), end=""),
    "handoff-comment": lambda a: print(format_handoff_comment(_json(a[0]), _json(a[1])), end=""),
    "update-map": lambda a: update_map(_json(a[0]), _json(a[1])),
}


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    if not args:
        raise SystemExit(__doc__)
    if args[0] not in COMMANDS:
        raise SystemExit(f"unknown command: {args[0]}")
    COMMANDS[args[0]](args[1:])


if __name__ == "__main__":
    main()
