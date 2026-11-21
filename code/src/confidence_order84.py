"""Confidence order on the 84-field model: pick the field, then draw (GitHub issue #332).

Each pass scores every masked field of each team by the top probability of its drawing distribution: the guided
distribution after the legal Pokémon construction tables and the lookahead, at temperature 1. Each team fills its
highest-scoring field, with the value drawn from that field's distribution, not its most likely value. One field per
pass, 84 passes. This is top-probability position selection (Kim et al., ICML 2025, §4.1,
https://arxiv.org/abs/2502.06768; named by Jeon et al. 2026, §3, https://arxiv.org/abs/2609.34509). It is not
surest-draw order (LLaDA's low-confidence remasking), which draws every field and keeps the most probable draw.

    python confidence_order84.py speed                   seconds per 96 teams, confidence order against species-first
    python confidence_order84.py non_top_species_first   redraw Linear, species-first's stored guidance-2 grid with every draw
                                                         recorded: its non-top draw share, into results/confidence_order_non_top.json
    python confidence_order84.py read                    the verdict against Linear, species-first and Linear, random order,
                                                         into results/confidence_order.json
    python confidence_order84.py figure DIR              its charts as ECharts pages in DIR
    python confidence_order84.py comment URL             the verdict comment; URL holds {name} and {theme}

The ask grid row is sched_linear_confidence (ask_grid84.py), decoder "confidence".
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import diffusion84 as M
import sampler84 as S

sample_confidence = S.sample_confidence


def speed():
    """Seconds per 96 teams at ask 12/24, guidance 2: species-first (Model84.sample) and confidence order."""
    import ask_grid84 as A
    w = M.Model84.load(A.L.CK / f"{A.MODELS['sched_linear_confidence']['checkpoint']}.pt")
    for name, f in (("species-first", lambda: w.sample(96, ask=0.5, guidance=2.0)),
                    ("confidence order", lambda: S.Sampler84(w, "confidence").sample(96, ask=0.5, guidance=2.0))):
        torch.manual_seed(1); t0 = time.time(); f(); print(f"{name:18s} {time.time() - t0:6.1f} s per 96 teams", flush=True)


# ---------------------------------------------------------------- the verdict, read after the grid
ROW, SPECIES_FIRST, RANDOM = "sched_linear_confidence", "sched_linear_species_first", "sched_linear_random"
READER = {ROW: "Linear, confidence order", SPECIES_FIRST: "Linear, species-first", RANDOM: "Linear, random order"}
STREAM = {ROW: 0, SPECIES_FIRST: 1, RANDOM: 2}             # each row's own bootstrap stream, fixed so a new row moves no other
COLOR = {ROW: "#8a5cd6", SPECIES_FIRST: "#2a78d6", RANDOM: "#e0702a"}
PASS_BAND, PASS_DEFINITION, K_PREDICTED = "asks_13_to_17", "held", 48
ISSUE_URL = "https://github.com/Rami-Ismael/Battle-Factory-of-Porygon-Pi/issues/332"


def results_dir():
    import masking_rate_rows as MRR
    return MRR.RES


def non_top_path(): return results_dir() / "confidence_order_non_top.json"


def out_path(): return results_dir() / "confidence_order.json"


def team_non_top(cells):
    """{(ask numerator, seed): [non-top share of each kept team]} from cells whose refusal records carry non_top_fields."""
    return {(c["numerator"], c["seed"]): [r["non_top_fields"] / M.COLS for r in c["refusals"]] for c in cells}


def non_top_species_first():
    """Redraw Linear, species-first's stored guidance-2 grid through the fixed-order path, which draws the same teams as
    Model84.sample, and record each kept team's non-top draws. Stops if a redrawn cell's teams differ from the stored ones."""
    import os
    os.environ.update(AG84_MODEL=SPECIES_FIRST, AG84_GUIDANCE="2")
    import ask_grid84 as A
    stored = json.load(open(A.OUT))
    if A.SEED0 != stored.get("seed0", 7300): raise SystemExit(f"seed base {A.SEED0} is not the stored grid's {stored.get('seed0')}")
    arm = S.Sampler84(A.load_model(), "confidence", lookahead=False, order=M.ORDER)
    path = non_top_path(); out = json.load(open(path)) if path.exists() else {"cells": {}}
    out.update(grid=SPECIES_FIRST, seed0=A.SEED0, checkpoint_sha256=A.require_same_checkpoint(stored))
    for tag, cell in sorted(stored["cells"].items()):
        if tag in out["cells"]: continue
        t0 = time.time(); kept, _, counts = A.draw(arm, cell["numerator"] / 24, cell["seed"])
        if kept != cell["pastes"]: raise SystemExit(f"{tag}: the redrawn teams differ from the stored grid's")
        out["cells"][tag] = dict(numerator=cell["numerator"], seed=cell["seed"], non_top_fields=[r["non_top_fields"] for r in counts["refusals"]])
        tmp = path.with_suffix(".tmp"); json.dump(out, open(tmp, "w")); os.replace(tmp, path)
        print(f"  {tag:40s} {len(kept)} teams, identical; non-top share {np.mean(out['cells'][tag]['non_top_fields']) / M.COLS:.3f} "
              f"{time.time() - t0:4.0f}s", flush=True)
    print("CONFIDENCE_NON_TOP_DONE", flush=True)


def fill_order_measures(cells):
    """When confidence order fills the species: the mean pass (1 to 84) at which a team's six species are filled, the
    share of teams that fill all six species in the first six passes, and the mean pass of each field type."""
    import lossdown as L
    passes = {}                                            # field type -> passes at which it was filled
    species_last, first_six = [], []
    for c in cells:
        for r in c["refusals"]:
            at = {col: k + 1 for k, col in enumerate(r["fill_order"])}
            species = [at[col] for col in range(0, M.STAT0, 8)]
            species_last.append(max(species)); first_six.append(max(species) <= 6)
            for col, k in at.items(): passes.setdefault("stat_points" if col >= M.STAT0 else L.KEY_OF_COL[col], []).append(k)
    return dict(mean_pass_of_last_species=float(np.mean(species_last)), share_with_six_species_first=float(np.mean(first_six)),
                mean_pass_by_field_type={k: float(np.mean(v)) for k, v in sorted(passes.items(), key=lambda kv: np.mean(kv[1]))})


def pass_at_k_of(name, stored):
    """Pass@k (k = 1..48) at asks 13/24 to 17/24, held definition: the new grid's own file, or the #331 re-read."""
    path = results_dir() / "pass_at_k" / f"{name}.json"
    r = json.load(open(path)) if path.exists() else stored.get(name)
    band = r and r["bands"].get(PASS_DEFINITION, {}).get(PASS_BAND)
    if not band: return None
    return dict(pass_at_k=band["pass_at_k"], pass_at_k_interval=band["pass_at_k_interval"], potential_at_k=band["potential_at_k"],
                potential_at_k_interval=band["potential_at_k_interval"], calls_per_legal_team=band.get("calls_per_legal_team"))


def prediction(win_gap, error_gap, pass_new, pass_first):
    """The ticket's prediction, part by part. Worse: the paired interval excludes zero on the worse side."""
    k = K_PREDICTED - 1
    within = pass_first["pass_at_k_interval"][k][0] <= pass_new["pass_at_k"][k] <= pass_first["pass_at_k_interval"][k][1]
    parts = dict(win_rate_worse=win_gap["interval"][1] < 0, error_worse=error_gap["interval"][0] > 0, pass_at_48_within_species_first_interval=bool(within))
    held = sum(parts.values())
    return dict(parts=parts, verdict="confirmed" if held == 3 else "not confirmed" if held == 0 else "partly confirmed")


def read():
    import masking_rate_rows as MRR
    import species_late as SL
    MRR.use_pilot_src()
    counts = MRR.opponent_draws(len(np.load(MRR.opponents_path(SPECIES_FIRST, 2))["weights"]))
    rows = {row: SL.read_row(row, counts, 2, stream=[MRR.DRAW_SEED, 332, STREAM[row]]) for row in READER}
    summary = json.load(open(MRR.SUMMARY))["grids"] if MRR.SUMMARY.exists() else {}
    refs = MRR.references()
    stored_pass = json.load(open(MRR.RES / "pass_at_k_issue331.json"))["grids"]
    win = {row: dict(point=float(np.mean(list(m["got_by_ask"].values()))), draws=np.mean(list(m["got_by_ask_draws"].values()), 0)) for row, m in rows.items()}
    result = dict(question="Does confidence order (pick the field, then draw) on the linear checkpoint beat species-first order?",
                  prediction_written="mean win rate and ask-versus-got error worse than Linear, species-first's; Pass@k at k = 48 within its interval",
                  guidance=2, resampled="teams within each ask and sampling seed, separately in each row; the 49 opponents once for all three rows",
                  draws=MRR.DRAWS, draw_seed=MRR.DRAW_SEED, rows={}, against={})
    for row, m in rows.items():
        s = MRR.measures(MRR.load_grid(row, 2), refs) if row == ROW or row not in summary else summary[row]
        result["rows"][row] = dict(reader=READER[row], error=m["error"], error_interval=MRR.interval(m["error_draws"]),
                                   error_by_seed={str(k): v for k, v in m["error_by_seed"].items()},
                                   pooled_win=win[row]["point"], pooled_win_interval=MRR.interval(win[row]["draws"]),
                                   got_by_ask={str(k): v for k, v in m["got_by_ask"].items()},
                                   got_by_ask_interval={str(k): MRR.interval(v) for k, v in m["got_by_ask_draws"].items()},
                                   first_try_legality=s["first_try_legality"], line_ups=s["line_ups"], line_ups_by_seed=s["line_ups_by_seed"],
                                   new_against_training=s["new_against_training"], distinct_novel_line_ups_against_training=s["distinct_novel_line_ups_against_training"],
                                   effective_species=s["effective_species"], elite_line_up_yield=MRR.elite_yield(row),
                                   pass_at_k=pass_at_k_of(row, stored_pass))
    for other in (SPECIES_FIRST, RANDOM):
        result["against"][other] = dict(error=SL.paired(rows[ROW], rows[other]),
                                        pooled_win=dict(difference=win[ROW]["point"] - win[other]["point"], interval=MRR.interval(win[ROW]["draws"] - win[other]["draws"])))
    new_cells = list(json.load(open(MRR.grid_path(ROW, 2)))["cells"].values())
    first_non_top = json.load(open(non_top_path())) if non_top_path().exists() else None
    result["non_top_draw_share"] = {ROW: float(np.mean([v for vs in team_non_top(new_cells).values() for v in vs])),
                                    SPECIES_FIRST: float(np.mean([v / M.COLS for c in first_non_top["cells"].values() for v in c["non_top_fields"]])) if first_non_top else None}
    result["fill_order"] = fill_order_measures(new_cells)
    pf, pn = result["rows"][SPECIES_FIRST]["pass_at_k"], result["rows"][ROW]["pass_at_k"]
    result["prediction"] = prediction(result["against"][SPECIES_FIRST]["pooled_win"], result["against"][SPECIES_FIRST]["error"], pn, pf) if pf and pn else None
    json.dump(result, open(out_path(), "w"), indent=1)
    for row, m in result["rows"].items():
        print(f"{m['reader']:28s} error {m['error']:.4f} win {m['pooled_win']:.3f} legal {m['first_try_legality']:.1%} line-ups {m['line_ups']} "
              f"eff {m['effective_species']:.1f} Pass@48 {m['pass_at_k']['pass_at_k'][-1] if m['pass_at_k'] else None}")
    print("against species-first:", result["against"][SPECIES_FIRST]); print("non-top share:", result["non_top_draw_share"])
    print("fill order:", result["fill_order"]["mean_pass_of_last_species"], result["fill_order"]["share_with_six_species_first"])
    print("VERDICT", (result["prediction"] or {}).get("verdict")); print("CONFIDENCE_ORDER_READ_DONE")


def figure(folder):
    """Two pages: got against ask for the three rows, and Pass@k at asks 13/24 to 17/24, each with 95% bands."""
    from echarts_chart import band, write_chart
    r = json.load(open(out_path())); rows = r["rows"]; pct = "js:v => Math.round(v * 100) + '%'"
    asks = [int(k) for k in rows[ROW]["got_by_ask"]]; xs = [k / 24 for k in asks]; series = []
    for i, (row, m) in enumerate(rows.items()):
        series += band(m["reader"], COLOR[row], xs, [m["got_by_ask_interval"][str(k)][0] for k in asks], [m["got_by_ask_interval"][str(k)][1] for k in asks], stack=f"b{i}")
        series.append(dict(type="line", name=m["reader"], color=COLOR[row], symbolSize=6, lineStyle=dict(width=2.6 if row == ROW else 1.6),
                           emphasis=dict(focus="series"), data=[[x, m["got_by_ask"][str(k)]] for x, k in zip(xs, asks)]))
    series.append(dict(type="line", name="Got = ask", color="#6f6e69", symbol="none", silent=True, lineStyle=dict(type="dashed", width=1), data=[[0, 0], [1, 1]]))
    option = dict(grid=dict(left=64, right=24, top=56, bottom=56), legend=dict(top=0, data=[m["reader"] for m in rows.values()] + ["Got = ask"]),
                  tooltip=dict(trigger="axis", valueFormatter="js:v => v == null ? '' : (v * 100).toFixed(1) + '%'"),
                  xAxis=dict(type="value", min=0, max=1, interval=1 / 6, name="Ask: the win rate the model is asked for", nameLocation="middle", nameGap=34, axisLabel=dict(formatter=pct)),
                  yAxis=dict(type="value", min=0, max=0.8, name="Got: measured win rate", nameLocation="middle", nameGap=46, axisLabel=dict(formatter=pct)),
                  series=series, animation=False)
    gap = r["against"][SPECIES_FIRST]
    title = headline(r)
    write_chart(option, Path(folder) / "confidence-got.html", height=460, title=title,
                subtitle=f"Guidance 2, linear checkpoint. Ask-versus-got error {rows[ROW]['error']:.4f} with confidence order, {rows[SPECIES_FIRST]['error']:.4f} species-first "
                         f"(paired gap {gap['error']['difference']:+.4f}), {rows[RANDOM]['error']:.4f} random order.",
                note="Band: 95% bootstrap interval over teams and the 49 opponents. Got: one battle against each of the 49 distinct top-50 meta teams, "
                     "behaviour-cloning policy on both sides, slots shuffled; 192 teams per ask over 3 sampling seeds.")
    ks = list(range(1, K_PREDICTED + 1)); series = []
    for i, (row, m) in enumerate(rows.items()):
        p = m["pass_at_k"]
        if not p: continue
        series += band(m["reader"], COLOR[row], ks, [v[0] for v in p["pass_at_k_interval"]], [v[1] for v in p["pass_at_k_interval"]], stack=f"p{i}")
        series.append(dict(type="line", name=m["reader"], color=COLOR[row], symbol="none", lineStyle=dict(width=2.6 if row == ROW else 1.6),
                           emphasis=dict(focus="series"), data=[[k, v] for k, v in zip(ks, p["pass_at_k"])]))
    option = dict(grid=dict(left=64, right=24, top=56, bottom=56), legend=dict(top=0, data=[m["reader"] for m in rows.values()]),
                  tooltip=dict(trigger="axis", valueFormatter="js:v => v == null ? '' : (v * 100).toFixed(1) + '%'"),
                  xAxis=dict(type="value", min=1, max=K_PREDICTED, name="k: teams drawn at one ask", nameLocation="middle", nameGap=34),
                  yAxis=dict(type="value", min=0, max=1, name="Pass@k", nameLocation="middle", nameGap=46, axisLabel=dict(formatter=pct)),
                  series=series, animation=False)
    write_chart(option, Path(folder) / "confidence-pass-at-k.html", height=420,
                title="Pass@k: the chance that at least one of k teams has an elite line-up that held",
                subtitle="Asks 13/24 to 17/24, guidance 2. Elite: win rate at least 0.531 against the top-50 meta teams, and the line-up's best team "
                         "still reaches 0.531 in a re-battle of 4 games per opponent.",
                note="Band: 95% bootstrap interval over sampling seeds, teams and the 49 opponents (pass_at_k.py).")


def headline(r):
    gap = r["against"][SPECIES_FIRST]
    worse = gap["error"]["interval"][0] > 0 or gap["pooled_win"]["interval"][1] < 0
    better = gap["error"]["interval"][1] < 0 and gap["pooled_win"]["interval"][0] >= 0
    return ("Confidence order follows the ask worse than species-first order" if worse else
            "Confidence order follows the ask better than species-first order" if better else
            "Confidence order matches species-first order within the intervals")


def comment(figure_url):
    """The verdict comment, picture first. figure_url holds {name} (got or pass-at-k) and {theme} (light or dark)."""
    import datetime
    r = json.load(open(out_path())); rows = r["rows"]; new = rows[ROW]
    iv = lambda x, f="+.4f": f"[{x[0]:{f}}, {x[1]:{f}}]"
    pic = lambda name, alt: (f'<picture><source media="(prefers-color-scheme: dark)" srcset="{figure_url.format(name=name, theme="dark")}">'
                             f'<img alt="{alt}" src="{figure_url.format(name=name, theme="light")}"></picture>')
    pred = r["prediction"] or {"verdict": "not read", "parts": {}}
    word = lambda ok: "held" if ok else "did not hold"
    nt = r["non_top_draw_share"]; fo = r["fill_order"]
    lines = [f"**Result, {datetime.date.today()}: {headline(r).lower().replace('confidence order', 'Linear, confidence order', 1)}. Prediction {pred['verdict']}.**", "",
             pic("got", "Got against ask for Linear, confidence order, Linear, species-first and Linear, random order at guidance 2"), "",
             f"- Ask-versus-got error on asks 1/24 to 17/24: {new['error']:.4f}, against {rows[SPECIES_FIRST]['error']:.4f} for Linear, species-first "
             f"(paired by sampling seed: {r['against'][SPECIES_FIRST]['error']['difference']:+.4f} {iv(r['against'][SPECIES_FIRST]['error']['interval'])}) "
             f"and {rows[RANDOM]['error']:.4f} for Linear, random order.",
             f"- Mean win rate over the 23 asks: {new['pooled_win']:.3f}, against {rows[SPECIES_FIRST]['pooled_win']:.3f} "
             f"(gap {r['against'][SPECIES_FIRST]['pooled_win']['difference']:+.3f} {iv(r['against'][SPECIES_FIRST]['pooled_win']['interval'], '+.3f')}) "
             f"and {rows[RANDOM]['pooled_win']:.3f}.",
             f"- Non-top draw share (drawn value not the field's most likely value): {nt[ROW]:.3f} in confidence order, "
             + (f"{nt[SPECIES_FIRST]:.3f} in species-first order." if nt[SPECIES_FIRST] is not None else "species-first not measured."),
             f"- Where confidence order puts the species: the last of a team's six species is filled at pass {fo['mean_pass_of_last_species']:.1f} of 84 on average; "
             f"{fo['share_with_six_species_first']:.0%} of teams fill all six in the first six passes.", "",
             pic("pass-at-k", "Pass@k at asks 13/24 to 17/24 for the three rows"), "",
             "Prediction, written before the run, part by part: "
             + "; ".join(f"{k.replace('_', ' ')}: {word(v)}" for k, v in pred["parts"].items()) + ".", "",
             "| Measure | " + " | ".join(m["reader"] for m in rows.values()) + " |", "|---|" + "---|" * len(rows)]
    def cell(m, key):
        v = m.get(key)
        if key == "pass_48": return f"{m['pass_at_k']['pass_at_k'][-1]:.3f} {iv(m['pass_at_k']['pass_at_k_interval'][-1], '.3f')}" if m["pass_at_k"] else "n/a"
        if key == "potential_48": return f"{m['pass_at_k']['potential_at_k'][-1]:.3f}" if m["pass_at_k"] else "n/a"
        if key == "held": return str((m["elite_line_up_yield"] or {}).get("all_23_asks", {}).get("held", "n/a"))
        if key == "error": return f"{v:.4f} {iv(m['error_interval'], '.4f')}"
        if key in ("first_try_legality", "new_against_training"): return f"{v:.1%}"
        if key == "effective_species": return f"{v:.1f}"
        if key == "pooled_win": return f"{v:.3f}"
        return str(v)
    for key, label in (("error", "Ask-versus-got error, asks 1/24 to 17/24"), ("pooled_win", "Mean win rate, 23 asks"), ("first_try_legality", "First-try legality"),
                       ("line_ups", "Distinct line-ups"), ("new_against_training", "Teams with a novel line-up"),
                       ("distinct_novel_line_ups_against_training", "Distinct novel line-ups"), ("effective_species", "Effective species, 5-team subsets"),
                       ("held", "Elite line-ups that held, all asks"), ("pass_48", "Pass@48, asks 13/24 to 17/24"), ("potential_48", "Potential@48, asks 13/24 to 17/24")):
        lines.append(f"| {label} | " + " | ".join(cell(m, key) for m in rows.values()) + " |")
    lines += ["", "<details><summary>Method</summary>", "",
              "Confidence order: each pass scores every masked field by the top probability of its drawing distribution (guidance 2, legal Pokémon construction "
              "filter, lookahead), fills the highest-scoring field per team and draws its value at temperature 1; 84 passes. Checkpoint "
              "`stat84_shuffled_noise_linear_es3_cap80`, 23 asks × 3 sampling seeds × 64 teams, one battle per team against each of the 49 distinct top-50 meta "
              "teams, behaviour-cloning policy on both sides, slots shuffled. Interval: 95% bootstrap, teams resampled within each ask and sampling seed, the 49 "
              "opponents once for all three rows. Species-first's non-top share comes from redrawing its stored grid through the same code with a fixed order; "
              "every redrawn team matched the stored one.", "",
              "Scripts: `code/src/confidence_order84.py read`, `figure`, `comment`. Results: `~/vgc-data/results/confidence_order.json`.", "</details>"]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "comment": print(comment(sys.argv[2]), end="")
    else: {"speed": speed, "non_top_species_first": non_top_species_first, "read": read, "figure": figure}[cmd](*sys.argv[2:])
