"""Issue #431 report: one shared GIDD-style learning-rate recipe on Uniform, no filter and Masked, full filter.

Reads the four run JSONs (each arm under the old 4e-4 cosine-to-0 recipe and the new 5e-4 cosine-to-5e-5 recipe) and
writes two ECharts pages plus a summary JSON:
  lr-recipe-val-loss.html   validation loss by epoch, both arms, both recipes, and the learning rate each recipe gave
  lr-recipe-slot-binding.html   Uniform, no filter: share of 64 unconstrained teams per epoch that break each
                            slot-binding rule (ability, move), old recipe against new
The slot-binding shift epoch is the first epoch whose share of teams with an ability the species can't have falls
below one half (issue #276 read the old run's collapse, 100% -> 3%, at epochs 58-61).

    /tmp/vgc-pilot/.venv/bin/python src/gidd_lr_report.py --out ../artifacts/gidd-lr-recipe-431
"""
import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from echarts_chart import write_chart
import lr_schedule84 as LS

RESULTS = Path.home() / "Documents/code/vgc-team-generator-pilot/results"
# Reader-facing name -> (old recipe run, new recipe run)
ARMS = {
    "Uniform, no filter": ("stat84_udlm_loglinear_es3_cap80", "stat84_udlm_loglinear_lr5e4_min5e5_es3_cap80"),
    "Masked, full filter": ("stat84_shuffled_noise_linear_filtered_es3_cap80",
                            "stat84_shuffled_noise_linear_filtered_lr5e4_min5e5_es3_cap80"),
}
OLD_RECIPE, NEW_RECIPE = "4e-4 cosine to 0", "5e-4 cosine to 5e-5 (GIDD)"
COLORS = {"Uniform, no filter": "#0f766e", "Masked, full filter": "#c2410c"}
ABILITY_RULE, MOVE_RULE = "ability the species can't have", "move the species can't learn"
STEPS_PER_EPOCH = math.ceil(214_055 / 256)


def load(name, results=RESULTS):
    return json.loads((Path(results) / f"{name}.json").read_text())


def shift_epoch(info, rule=ABILITY_RULE, below=0.5):
    """First epoch whose share of drawn teams breaking rule falls below `below`; None if it never does."""
    for entry in info["curves"]:
        if entry.get("rule_shares", {}).get(rule, 1.0) < below:
            return entry["epoch"]
    return None


def lr_by_epoch(info, epochs=80):
    """The learning rate at the last step of each epoch, rebuilt from the run's recorded recipe."""
    total = STEPS_PER_EPOCH * info.get("cap", epochs)
    warm = min(info.get("warmup", 500), max(1, total // 10))
    lr_at = LS.make_lr_at(info.get("lr_schedule", "cosine"), peak_lr=info["lr"], total_steps=total, warmup_steps=warm,
                          min_lr=info.get("min_lr", 0.0))
    return [lr_at(e * STEPS_PER_EPOCH - 1) for e in range(1, epochs + 1)]


def summary(runs):
    out = {}
    for arm, (old, new) in runs.items():
        row = {}
        for label, info in ((OLD_RECIPE, old), (NEW_RECIPE, new)):
            last = info["curves"][-1]
            row[label] = dict(best_epoch=info["best_epoch"], stopped_epoch=info["stopped_epoch"],
                              best_val_loss=info["best_val_loss"], final_illegal_share=last.get("illegal_share"),
                              final_errors_per_team=last.get("errors_per_team"),
                              final_rule_shares=last.get("rule_shares"), minutes=info.get("minutes"))
            if arm.startswith("Uniform"):
                row[label]["ability_shift_epoch"] = shift_epoch(info, ABILITY_RULE)
                row[label]["move_shift_epoch"] = shift_epoch(info, MOVE_RULE)
        row["best_val_loss_change"] = row[NEW_RECIPE]["best_val_loss"] - row[OLD_RECIPE]["best_val_loss"]
        out[arm] = row
    return out


def val_loss_figure(runs, out_dir):
    series = []
    for arm, (old, new) in runs.items():
        for label, info, dash in ((OLD_RECIPE, old, "dashed"), (NEW_RECIPE, new, "solid")):
            series.append({"type": "line", "name": f"{arm} · {label}", "color": COLORS[arm], "symbol": "none",
                           "lineStyle": {"type": dash, "width": 2}, "emphasis": {"focus": "series"},
                           "data": [[e["epoch"], round(e["val_loss"], 5)] for e in info["curves"]]})
    old_info, new_info = runs["Uniform, no filter"]
    for label, info, dash in ((OLD_RECIPE, old_info, "dashed"), (NEW_RECIPE, new_info, "solid")):
        series.append({"type": "line", "name": f"learning rate · {label}", "color": "#6f6e69", "symbol": "none",
                       "xAxisIndex": 1, "yAxisIndex": 1, "lineStyle": {"type": dash, "width": 1.5},
                       "emphasis": {"focus": "series"},
                       "data": [[i + 1, v] for i, v in enumerate(lr_by_epoch(info))]})
    option = {
        "tooltip": {"trigger": "axis"}, "legend": {"top": 0, "type": "scroll"},
        "grid": [{"left": 64, "right": 24, "top": 48, "height": "52%"}, {"left": 64, "right": 24, "top": "74%", "bottom": 48}],
        "xAxis": [{"type": "value", "min": 1, "max": 80, "gridIndex": 0, "axisLabel": {"show": False}},
                  {"type": "value", "min": 1, "max": 80, "gridIndex": 1, "name": "epoch", "nameLocation": "middle", "nameGap": 28}],
        "yAxis": [{"type": "value", "scale": True, "gridIndex": 0, "name": "validation loss (nats per field)"},
                  {"type": "value", "gridIndex": 1, "name": "learning rate", "axisLabel": {"formatter": "{value}"}}],
        "dataZoom": [{"type": "inside", "xAxisIndex": [0, 1]}, {"type": "slider", "xAxisIndex": [0, 1], "height": 16, "bottom": 8}],
        "series": series,
    }
    return write_chart(option, Path(out_dir) / "lr-recipe-val-loss.html",
                       title="Validation loss under the old and the shared GIDD-style learning-rate recipe",
                       subtitle="Dashed: 4e-4 cosine to 0 (laptop GPU). Solid: 5e-4 cosine to 5e-5 (Modal GPU). Lower is better.",
                       note="One training seed per run. Validation loss: 4 fixed draws on 4,368 held-out teams.", height=560)


def slot_binding_figure(runs, out_dir):
    old, new = runs["Uniform, no filter"]
    series = []
    for rule, color in ((ABILITY_RULE, "#0f766e"), (MOVE_RULE, "#c2410c")):
        for label, info, dash in ((OLD_RECIPE, old, "dashed"), (NEW_RECIPE, new, "solid")):
            series.append({"type": "line", "name": f"{rule} · {label}", "color": color, "symbol": "none",
                           "lineStyle": {"type": dash, "width": 2}, "emphasis": {"focus": "series"},
                           "data": [[e["epoch"], e["rule_shares"][rule]] for e in info["curves"] if "rule_shares" in e]})
    option = {"tooltip": {"trigger": "axis"}, "legend": {"top": 0, "type": "scroll"},
              "grid": {"left": 64, "right": 24, "top": 56, "bottom": 64},
              "xAxis": {"type": "value", "min": 1, "max": 80, "name": "epoch", "nameLocation": "middle", "nameGap": 28},
              "yAxis": {"type": "value", "min": 0, "max": 1, "name": "share of 64 drawn teams"},
              "dataZoom": [{"type": "inside"}, {"type": "slider", "height": 16, "bottom": 8}], "series": series}
    return write_chart(option, Path(out_dir) / "lr-recipe-slot-binding.html",
                       title="Uniform, no filter: when does slot binding click under each recipe?",
                       subtitle="Share of 64 teams drawn without the legal Pokémon construction filter that break each rule, every epoch.",
                       note="Draw: guidance 1, 128 passes, the same seed every epoch. Dashed: old recipe. Solid: shared GIDD-style recipe.",
                       height=460)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--results", type=Path, default=RESULTS)
    args = parser.parse_args(argv)
    runs = {arm: (load(old, args.results), load(new, args.results)) for arm, (old, new) in ARMS.items()}
    args.out.mkdir(parents=True, exist_ok=True)
    pages = [val_loss_figure(runs, args.out), slot_binding_figure(runs, args.out)]
    result = summary(runs)
    (args.out / "summary.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))
    for page in pages:
        print(page)


if __name__ == "__main__":
    main()
