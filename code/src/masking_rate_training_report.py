"""Measurements and figures after each masking-schedule training (docs/masking-rate-rows.md, Plan and Figures frame 4; issue #296).

The two trainings are train84.py's shuffled_per_field and shuffled_linear: 84-field models on the labelled teams of the
shuffled-only table, which differ only in the masking schedule. Reader names: "Per-field masking schedule" and "Linear masking schedule".

  gap <checkpoint>         training-versus-validation gap, the same loss function on both sides: as many training teams as
                           validation teams (seeded), every team scored with 128 mask draws at seed 12345, EMA network in
                           eval mode, exact labels, the checkpoint's own masking schedule. -> masking_rate_gap_<checkpoint>.json
  cross_rate <checkpoint>  validation loss on every labelled validation team under the per-field and under the linear masking
                           rate (diffusion84.val_loss). The loss under the checkpoint's own rate must reproduce its logged
                           best_val_loss within 1e-4; a mismatch is written into the JSON and the command exits non-zero.
                           -> masking_rate_cross_rate_<checkpoint>.json
  figures                  ECharts pages in docs/figures/masking_rate_rows/ from whatever results exist
  summary                  a Markdown table for a GitHub issue comment, printed

    /tmp/vgc-pilot/.venv/bin/python src/masking_rate_training_report.py gap stat84_shuffled_noise_per_field_es3_cap80     (from code/)
    options: --results-dir DIR (default ~/vgc-data/results), --run-info-dir DIR (default asked_vs_got.RES, where train84 writes
             run info), --out-dir DIR (figures), --draws N (gap), --limit N (score only the first N teams of each side; tests)

The gap is ported from ~/vgc-data/validation_loss_accuracy/run/validation_accuracy.py, sha256
0b9388243ad31fb843706c706f6a1b3fbeec764b62ec600faa1f051712674ab0: terms (its "random" estimator, the only one the gap
uses), summary, sizes and gap, with the same numerics. Only the import of diffusion84 differs: this repo's, not a frozen
copy. The 95% interval is the source's gap ± 1.96 standard errors. Its headline reading, as in the source's printout, drops
from both sides the teams the legal Pokémon construction tables exclude (training never sees them); all teams are kept too.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

import train84 as TR                                    # puts src/ and /tmp/vgc-pilot/src on sys.path, as when training
import asked_vs_got as X
import diffusion84 as M
import lossdown as L
from echarts_chart import write_chart

SOURCE = Path.home() / "vgc-data/validation_loss_accuracy/run/validation_accuracy.py"
SOURCE_SHA256 = "0b9388243ad31fb843706c706f6a1b3fbeec764b62ec600faa1f051712674ab0"
RESULTS = Path.home() / "vgc-data/results"
FIGURES = Path(__file__).resolve().parent.parent / "docs/figures/masking_rate_rows"
DRAWS, DRAW_SEED, SIZES_SEED = 128, 12345, 0
OWN_RATE_TOLERANCE = 1e-4
Z95 = 1.96
RATES = dict(per_field=M.SCHED, linear=TR.LINEAR_MASKING_RATE)          # masking-schedule key -> exponents
RATE_WORDS = dict(per_field="per-field masking schedule", linear="linear masking schedule")
# The two trainings: checkpoint -> reader name, own masking schedule, color
MODELS = {TR.NAMES["shuffled_per_field"]: dict(label="Per-field masking schedule", rate="per_field", color="#2a78d6"),
          TR.NAMES["shuffled_linear"]: dict(label="Linear masking schedule", rate="linear", color="#1baf7a")}
PLAIN, GREY = "#6f6e69", "#a9a8a2"


# ---------------------------------------------------------------- data and checkpoints

def load_table():
    return TR.load(TR.SHUFFLED_ONLY)


def labelled_split(table):
    """Training and validation masks of train84.train_masking_rate_arms: labelled teams, split by all_val."""
    labelled = ~np.isnan(table["all_W"])
    return ~table["all_val"] & labelled, table["all_val"] & labelled


def checkpoint_path(checkpoint):
    path = Path(checkpoint)
    if path.suffix == ".pt" and path.exists(): return path
    direct = L.CK / f"{checkpoint}.pt"
    if direct.exists(): return direct
    per_epoch = sorted(L.CK.glob(f"*/{checkpoint}.pt"))
    if per_epoch: return per_epoch[0]
    raise FileNotFoundError(f"no checkpoint {checkpoint}.pt in {L.CK} or its folders")


def load_model(checkpoint):
    return M.Model84.load(checkpoint_path(checkpoint))


def own_rate(info):
    """The masking-schedule key the checkpoint trained with, from its run info."""
    exponents = info.get("sched", M.SCHED)
    for key, value in RATES.items():
        if exponents == value: return key
    raise ValueError(f"masking-schedule exponents {exponents!r} are neither the per-field nor the linear rate")


def masking_rate_tensor(model, exponents=None):
    """The exponents as the loss uses them, on the device: the checkpoint's own (as the source's load()) unless given."""
    return M.sched84(model.info.get("sched", M.SCHED) if exponents is None else exponents).to(M.DEV)


# ---------------------------------------------------------------- the gap, ported from validation_accuracy.py

@torch.no_grad()
def terms(net, tables, rows, labels, exponents, draws, seed, batch=512):
    """Per-draw, per-team ELBO terms [draws, teams] and their t; t stratified at random within each of `draws` slices."""
    generator = torch.Generator().manual_seed(seed)
    team_rows = torch.as_tensor(np.asarray(rows, np.int64))
    team_labels = torch.as_tensor(np.asarray(labels, np.float32))
    exponents_cpu = exponents.cpu()
    teams = len(team_rows)
    out, times = torch.zeros(draws, teams), torch.zeros(draws, teams)
    for draw in range(draws):
        slot_orders = torch.rand(teams, M.NSLOT, generator=generator).argsort(1)
        t = ((draw + torch.rand(teams, generator=generator)) / draws).clamp(min=1e-3)
        uniform = torch.rand(teams, M.COLS, generator=generator)
        for start in range(0, teams, batch):
            part = slice(start, start + batch)
            hidden = (uniform[part] < t[part].view(-1, 1) ** exponents_cpu.view(1, -1)).to(M.DEV)
            shuffled = M.shuffle_slots(team_rows[part], slot_orders[part]).to(M.DEV)
            out[draw, part] = M.elbo_terms(net, tables, shuffled, hidden, t[part].to(M.DEV), team_labels[part].to(M.DEV),
                                           exponents).float().cpu()
        times[draw] = t
    return out, times


def summary_of(out):
    per_team = out.mean(0)
    return dict(mean=float(per_team.mean()), se=float(per_team.std() / len(per_team) ** 0.5))


def sizes(training, validation):
    """Every validation team, and as many training teams drawn without replacement at seed 0, sorted."""
    validation_index = np.flatnonzero(validation)
    training_index = np.sort(np.random.default_rng(SIZES_SEED).choice(np.flatnonzero(training), len(validation_index), replace=False))
    return training_index, validation_index


def gap(model, net, exponents, rows, labels, training_index, validation_index, draws):
    """Gap on all teams, and with the teams the legal Pokémon construction tables exclude removed (training drops them)."""
    training_terms, _ = terms(net, model.tables, rows[training_index], labels[training_index], exponents, draws, DRAW_SEED)
    validation_terms, _ = terms(net, model.tables, rows[validation_index], labels[validation_index], exponents, draws, DRAW_SEED)

    def accepted(index):
        return torch.as_tensor(~model.tables.check(np.asarray(rows[index], np.int64)[:, :M.STAT0]))

    out = {}
    for key, keep_training, keep_validation in (("all_teams", slice(None), slice(None)),
                                                ("rule_table_teams_removed", accepted(training_index), accepted(validation_index))):
        training_side, validation_side = summary_of(training_terms[:, keep_training]), summary_of(validation_terms[:, keep_validation])
        out[key] = dict(training=training_side, validation=validation_side, gap=validation_side["mean"] - training_side["mean"],
                        gap_se=(training_side["se"] ** 2 + validation_side["se"] ** 2) ** 0.5,
                        validation_teams=int(validation_terms[:, keep_validation].shape[1]))
    return out


def run_gap(checkpoint, results_dir=RESULTS, draws=DRAWS, limit=None):
    model, table = load_model(checkpoint), load_table()
    training_index, validation_index = sizes(*labelled_split(table))
    if limit: training_index, validation_index = training_index[:limit], validation_index[:limit]
    started = time.time()
    measured = gap(model, model.net.eval(), masking_rate_tensor(model), table["all_X"], table["all_W"], training_index, validation_index, draws)
    accepted_training = int((~model.tables.check(table["all_X"][training_index][:, :M.STAT0])).sum())
    for key, reading in measured.items():
        reading["interval_95"] = gap_interval(reading)
        reading["training_teams"] = len(training_index) if key == "all_teams" else accepted_training
    result = dict(checkpoint=checkpoint, checkpoint_path=str(checkpoint_path(checkpoint)), masking_rate=own_rate(model.info),
                  exponents=model.info.get("sched", M.SCHED), estimator=f"random, {draws} draws, seed {DRAW_SEED}",
                  teams_each_side=len(validation_index), data_sha256=table["sha256"], table=table["table"],
                  source=str(SOURCE), source_sha256=SOURCE_SHA256, device=M.DEV, limit=limit, **measured,
                  minutes=(time.time() - started) / 60)
    path = write_json(result, Path(results_dir) / f"masking_rate_gap_{checkpoint}.json")
    headline = measured["rule_table_teams_removed"]
    print(f"{checkpoint}: gap {headline['gap']:+.4f} ± {headline['gap_se']:.4f} (all teams {measured['all_teams']['gap']:+.4f}); "
          f"{result['minutes']:.1f} min -> {path}", flush=True)
    return result


# ---------------------------------------------------------------- validation loss under both masking schedules

def cross_rate(model, table, validation, limit=None):
    rows, labels = table["all_X"][validation], table["all_W"][validation]
    if limit: rows, labels = rows[:limit], labels[:limit]
    net = model.net.eval()
    accepted = ~model.tables.check(np.asarray(rows, np.int64)[:, :M.STAT0])   # training drops the teams the tables exclude
    losses, losses_accepted, seconds = {}, {}, {}
    for key, exponents in RATES.items():
        started = time.time()
        losses[key] = M.val_loss(net, model.tables, rows, labels, masking_rate_tensor(model, exponents))
        seconds[key] = time.time() - started
        losses_accepted[key] = M.val_loss(net, model.tables, rows[accepted], labels[accepted], masking_rate_tensor(model, exponents))
    return dict(validation_loss=losses, seconds=seconds, validation_teams=len(rows), own_masking_rate=own_rate(model.info),
                validation_loss_tables_accepted=losses_accepted, validation_teams_tables_accepted=int(accepted.sum()))


def comparable_losses(cross):
    """The losses to compare between models: without the validation teams the legal Pokémon construction tables exclude,
    whose forbidden values cost about 10,000 each when hidden. Results saved before that reading existed give all teams."""
    if "validation_loss_tables_accepted" in cross:
        return cross["validation_loss_tables_accepted"], cross["validation_teams_tables_accepted"]
    return cross["validation_loss"], cross["validation_teams"]


def own_rate_check(result, info):
    """Does the loss under the checkpoint's own masking schedule reproduce the best_val_loss its training logged?"""
    logged = info.get("best_val_loss")
    measured = result["validation_loss"][result["own_masking_rate"]]
    if logged is None:
        return dict(passed=False, logged=None, measured=measured, difference=None, tolerance=OWN_RATE_TOLERANCE,
                    reason="the run info holds no best_val_loss")
    difference = measured - float(logged)
    return dict(passed=abs(difference) <= OWN_RATE_TOLERANCE, logged=float(logged), measured=measured, difference=difference,
                tolerance=OWN_RATE_TOLERANCE)


def run_cross_rate(checkpoint, results_dir=RESULTS, limit=None):
    model, table = load_model(checkpoint), load_table()
    _training, validation = labelled_split(table)
    result = cross_rate(model, table, validation, limit)
    result["own_rate_check"] = check = own_rate_check(result, model.info)
    result.update(checkpoint=checkpoint, checkpoint_path=str(checkpoint_path(checkpoint)), exponents=model.info.get("sched", M.SCHED),
                  kept_epoch=model.info.get("best_epoch", model.info.get("best_pass")), patience=model.info.get("patience"),
                  data_sha256=table["sha256"], table=table["table"], device=M.DEV, limit=limit)
    path = write_json(result, Path(results_dir) / f"masking_rate_cross_rate_{checkpoint}.json")
    losses = result["validation_loss"]
    print(f"{checkpoint}: validation loss {losses['per_field']:.4f} under the per-field masking schedule, {losses['linear']:.4f} under "
          f"the linear masking schedule, on {result['validation_teams']} teams -> {path}", flush=True)
    if not check["passed"]:
        print(f"OWN-RATE CHECK FAILED: measured {check['measured']} against logged {check['logged']}"
              + ("" if check["difference"] is None else f" (difference {check['difference']:+.6f}, tolerance {OWN_RATE_TOLERANCE})"), flush=True)
    return result


# ---------------------------------------------------------------- reading results

def write_json(result, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=1, default=float))
    return path


def read_results(run_info_dir, results_dir):
    """Per checkpoint: run info, cross-rate and gap results, each None when its file is missing; and the missing files."""
    found, missing = {}, dict(run=[], cross=[], gap=[])
    for checkpoint in MODELS:
        paths = dict(run=Path(run_info_dir) / f"{checkpoint}.json",
                     cross=Path(results_dir) / f"masking_rate_cross_rate_{checkpoint}.json",
                     gap=Path(results_dir) / f"masking_rate_gap_{checkpoint}.json")
        found[checkpoint] = {}
        for key, path in paths.items():
            found[checkpoint][key] = json.loads(path.read_text()) if path.exists() else None
            if not path.exists(): missing[key].append(path)
    return found, missing


def kept_epoch(info):
    return info.get("best_epoch", info.get("best_pass"))


def stopped_epoch(info):
    return info.get("stopped_epoch", info.get("stopped_pass"))


def curve(info):
    return [(entry.get("epoch", entry.get("pass_")), entry["val_loss"]) for entry in info.get("curves", []) if entry.get("val_loss") is not None
            and entry.get("epoch", entry.get("pass_"))]


def headline_gap(gap_result):
    return gap_result["rule_table_teams_removed"]


def gap_interval(reading):
    return reading.get("interval_95") or [reading["gap"] - Z95 * reading["gap_se"], reading["gap"] + Z95 * reading["gap_se"]]


# ---------------------------------------------------------------- figures

def report_missing(chart, paths):
    for path in paths: print(f"{chart}: missing {path}", flush=True)


def epochs_chart(found, out_dir):
    have = {checkpoint: entry["run"] for checkpoint, entry in found.items() if entry["run"]}
    if not have: return None
    series = []
    for checkpoint, info in have.items():
        model = MODELS[checkpoint]
        points = [[epoch, round(loss, 4)] for epoch, loss in curve(info)]
        kept = kept_epoch(info)
        kept_loss = dict(points).get(kept)
        line = dict(name=model["label"], type="line", symbolSize=5, lineStyle=dict(color=model["color"], width=2.5),
                    itemStyle=dict(color=model["color"]), emphasis=dict(focus="series"), data=points)
        if kept_loss is not None:
            line["markPoint"] = dict(symbol="pin", symbolSize=38, itemStyle=dict(color=model["color"]),
                                     label=dict(formatter=f"kept\n{kept}", fontSize=10),
                                     data=[dict(name=f"kept epoch {kept}", coord=[kept, kept_loss], value=kept_loss)])
        series.append(line)
    longest = max(len(series_entry["data"]) for series_entry in series)
    option = dict(legend=dict(top=0), grid=dict(left=56, right=24, top=48, bottom=72), tooltip=dict(trigger="axis"),
                  xAxis=dict(type="value", name="Epoch", nameLocation="middle", nameGap=28, minInterval=1, min=1),
                  yAxis=dict(type="value", name="Validation loss", nameLocation="middle", nameGap=44, scale=True), series=series)
    if longest > 20: option["dataZoom"] = [dict(type="inside"), dict(type="slider", height=18, bottom=8)]
    missing = [MODELS[checkpoint]["label"] for checkpoint in MODELS if checkpoint not in have]
    return write_chart(option, Path(out_dir) / "04a_validation_loss_by_epoch.html",
                       title="Validation loss by epoch, each model under its own masking schedule",
                       subtitle="The two curves are measured under different masking schedules, so they are not comparable with each other. "
                                "Read each curve for its own lowest epoch; compare the models in the 2-by-2 chart.",
                       note="Pins mark the epoch early stopping kept: the lowest validation loss, patience 3, cap 80. Validation loss: the "
                            "84-field model's loss on the labelled validation teams of the shuffled-only table, 4 mask draws, seed 12345."
                            + (f" Not trained yet: {', '.join(missing)}." if missing else ""))


def two_by_two_chart(found, out_dir):
    have = {checkpoint: entry["cross"] for checkpoint, entry in found.items() if entry["cross"]}
    if not have: return None
    labels = [MODELS[checkpoint]["label"] for checkpoint in have]
    colors = dict(per_field="#2a78d6", linear="#1baf7a")
    series = [dict(name=f"Scored under the {RATE_WORDS[key]}", type="bar", barGap="12%", itemStyle=dict(color=colors[key]),
                   emphasis=dict(focus="series"), label=dict(show=True, position="top", color=PLAIN),
                   data=[round(comparable_losses(result)[0][key], 4) for result in have.values()]) for key in RATES]
    option = dict(legend=dict(top=0), grid=dict(left=56, right=24, top=56, bottom=48),
                  tooltip=dict(trigger="axis", axisPointer=dict(type="shadow")),
                  xAxis=dict(type="category", data=labels, name="Model, by the masking schedule it trained with", nameLocation="middle", nameGap=30),
                  yAxis=dict(type="value", name="Validation loss", nameLocation="middle", nameGap=44, scale=True), series=series)
    teams = sorted({comparable_losses(result)[1] for result in have.values()})
    return write_chart(option, Path(out_dir) / "04b_validation_loss_two_by_two.html",
                       title="Validation loss of each model under each masking schedule",
                       subtitle="Expected: each model is lowest under the masking schedule it trained on. Only bars of one color are comparable "
                                "with each other: a loss is comparable only with another measured under the same masking schedule.",
                       note=f"Each model at its kept epoch, scored on {', '.join(map(str, teams))} labelled validation teams of the shuffled-only "
                            "table, without the teams the legal Pokémon construction tables exclude, 4 mask draws, seed 12345. Source: masking_rate_cross_rate_<checkpoint>.json.")


def gap_chart(found, out_dir):
    have = {checkpoint: headline_gap(entry["gap"]) for checkpoint, entry in found.items() if entry["gap"]}
    if not have: return None
    labels = [MODELS[checkpoint]["label"] for checkpoint in have]
    series = [dict(name="Gap", type="scatter", symbolSize=12, itemStyle=dict(color=PLAIN), emphasis=dict(focus="series"),
                   data=[round(reading["gap"], 4) for reading in have.values()],
                   markLine=dict(symbol="none", silent=True, lineStyle=dict(color=GREY, type="dashed"),
                                 label=dict(formatter="no gap", position="insideEndTop", color=PLAIN), data=[dict(yAxis=0)]))]
    for checkpoint, reading in have.items():
        low, high = gap_interval(reading)
        label, color = MODELS[checkpoint]["label"], MODELS[checkpoint]["color"]
        series.append(dict(name=f"95% interval, {label}", type="line", symbol="none", silent=True, tooltip=dict(show=False),
                           lineStyle=dict(color=color, width=3), data=[[label, round(low, 4)], [label, round(high, 4)]]))
    teams = sorted({reading["validation_teams"] for reading in have.values()})
    option = dict(legend=dict(top=0, data=["Gap"]), grid=dict(left=64, right=24, top=48, bottom=40),
                  tooltip=dict(trigger="axis"), xAxis=dict(type="category", data=labels),
                  yAxis=dict(type="value", name="Validation minus training loss", nameLocation="middle", nameGap=50), series=series)
    return write_chart(option, Path(out_dir) / "04c_training_versus_validation_gap.html",
                       title="Training-versus-validation gap of each model",
                       subtitle="The same loss function on as many training teams as validation teams, each model under its own masking schedule. "
                                "Above zero, the model fits its training teams better than teams it never saw.",
                       note=f"Point: validation loss minus training loss; bar: 95% interval, ± 1.96 standard errors. 128 mask draws per team, "
                            f"seed 12345, {', '.join(map(str, teams))} validation teams; the teams the legal Pokémon construction tables "
                            "exclude are removed from both sides. Ported from validation_accuracy.py. Source: masking_rate_gap_<checkpoint>.json.")


def figures(run_info_dir=X.RES, results_dir=RESULTS, out_dir=FIGURES):
    """Each chart draws the models whose inputs exist; it names every missing input file, and is skipped when it has none."""
    found, missing = read_results(run_info_dir, results_dir)
    pages = []
    for chart, maker, key in (("validation loss by epoch", epochs_chart, "run"), ("2-by-2 of validation loss", two_by_two_chart, "cross"),
                              ("training-versus-validation gap", gap_chart, "gap")):
        page = maker(found, out_dir)
        report_missing(chart, missing[key])
        if page is None:
            print(f"{chart}: skipped, no input yet", flush=True)
        else:
            print(page, flush=True); pages.append(page)
    return pages


# ---------------------------------------------------------------- summary

def summary(run_info_dir=X.RES, results_dir=RESULTS):
    found, missing = read_results(run_info_dir, results_dir)
    header = ("| Model | Kept epoch | Stopped epoch | Training minutes | Validation loss under the per-field masking schedule "
              "| Validation loss under the linear masking schedule | Training-versus-validation gap, 95% interval |")
    lines = ["**Masking-schedule trainings** (issue #296)", "", header, "| --- | ---: | ---: | ---: | ---: | ---: | --- |"]
    for checkpoint, entry in found.items():
        info = entry["run"]
        if info is None: continue
        cross, gap_result = entry["cross"], entry["gap"]
        losses = [f"{comparable_losses(cross)[0][key]:.4f}" if cross else "not measured" for key in RATES]
        if gap_result:
            reading = headline_gap(gap_result); low, high = gap_interval(reading)
            gap_text = f"{reading['gap']:+.4f} ({low:+.4f} to {high:+.4f})"
        else:
            gap_text = "not measured"
        minutes = info.get("minutes")
        lines.append(f"| {MODELS[checkpoint]['label']} | {kept_epoch(info)} | {stopped_epoch(info)} | "
                     f"{'' if minutes is None else f'{minutes:.1f}'} | {' | '.join(losses)} | {gap_text} |")
    lines += ["", "Each model is read at its kept epoch, the lowest validation loss. Validation loss on the labelled validation teams "
              "of the shuffled-only table, without the teams the legal Pokémon construction tables exclude; a loss is comparable only with another measured under the same masking schedule. "
              "Gap: validation minus training loss, the same function on as many training teams as validation teams, each model "
              "under its own masking schedule, 128 mask draws; the interval is ± 1.96 standard errors."]
    absent = [path for paths in missing.values() for path in paths]
    if absent:
        lines += ["", "Missing inputs: " + ", ".join(f"`{path.name}`" for path in absent)]
    return "\n".join(lines)


# ---------------------------------------------------------------- command line

def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("gap", "cross_rate", "figures", "summary"))
    parser.add_argument("checkpoint", nargs="?")
    parser.add_argument("--results-dir", type=Path, default=RESULTS)
    parser.add_argument("--run-info-dir", type=Path, default=X.RES)
    parser.add_argument("--out-dir", type=Path, default=FIGURES)
    parser.add_argument("--draws", type=int, default=DRAWS)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args(argv)
    if args.command in ("gap", "cross_rate") and not args.checkpoint: parser.error(f"{args.command} needs a checkpoint name")
    return args


def main(argv):
    args = parse_args(argv)
    if args.command == "gap":
        run_gap(args.checkpoint, args.results_dir, args.draws, args.limit)
    elif args.command == "cross_rate":
        result = run_cross_rate(args.checkpoint, args.results_dir, args.limit)
        if not result["own_rate_check"]["passed"]: sys.exit(1)
    elif args.command == "figures":
        figures(args.run_info_dir, args.results_dir, args.out_dir)
    else:
        print(summary(args.run_info_dir, args.results_dir))


if __name__ == "__main__":
    main(sys.argv[1:])
