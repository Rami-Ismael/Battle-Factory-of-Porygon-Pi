"""Issue #431 report: one shared GIDD-style learning-rate recipe on Uniform and Masked, full filter.

Reads the four run JSONs (each arm under the old 4e-4 cosine-to-0 recipe and the new 5e-4 cosine-to-5e-5 recipe) and
writes two ECharts pages plus a summary JSON:
  lr-recipe-val-loss.html   validation loss by epoch, both arms, both recipes, and the learning rate each recipe gave
  lr-recipe-field-matching.html   Uniform: share of 64 unconstrained teams per epoch that break each
                            Pokémon-to-field matching rule (ability, move), old recipe against new
The Pokémon-to-field matching shift epoch is the first epoch whose share of teams with an ability the species can't have falls
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
    "Uniform": ("stat84_udlm_loglinear_es3_cap80", "stat84_udlm_loglinear_lr5e4_min5e5_es3_cap80"),
    "Masked, full filter": ("stat84_shuffled_noise_linear_filtered_es3_cap80",
                            "stat84_shuffled_noise_linear_filtered_lr5e4_min5e5_es3_cap80"),
}
OLD_RECIPE, NEW_RECIPE = "old recipe", "shared GIDD-style recipe"
COLORS = {"Uniform": "#0f766e", "Masked, full filter": "#c2410c"}
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


LIGHT = {"Uniform": "#8fc9c1", "Masked, full filter": "#f2b08f"}     # each arm's old-recipe tint


def cliff(info, after=30):
    """(epoch, drop): the epoch, from `after` on, whose validation loss fell most from the epoch before."""
    curve = info["curves"]
    return max(((curve[i]["epoch"], curve[i - 1]["val_loss"] - curve[i]["val_loss"]) for i in range(1, len(curve))
                if curve[i]["epoch"] >= after), key=lambda pair: pair[1])


def val_loss_figure(runs, out_dir):
    """Figure 2: validation loss per epoch, both arms, both recipes, each run's late cliff marked."""
    series = []
    for arm, (old, new) in runs.items():
        for label, info, dash in ((OLD_RECIPE, old, "dashed"), (NEW_RECIPE, new, "solid")):
            epoch = cliff(info)[0]
            loss = next(e["val_loss"] for e in info["curves"] if e["epoch"] == epoch)
            series.append({"type": "line", "name": f"{arm} · {label}", "color": COLORS[arm], "symbol": "none",
                           "lineStyle": {"type": dash, "width": 2}, "emphasis": {"focus": "series"},
                           "data": [[e["epoch"], round(e["val_loss"], 5)] for e in info["curves"]],
                           "markPoint": {"symbol": "circle", "symbolSize": 9, "data": [{"coord": [epoch, round(loss, 5)]}],
                                         "itemStyle": {"color": COLORS[arm] if dash == "solid" else "#fff",
                                                       "borderColor": COLORS[arm], "borderWidth": 2},
                                         "label": {"show": False}}})
    option = {"tooltip": {"trigger": "axis"}, "legend": {"top": 0, "left": 0, "itemGap": 14},
              "grid": {"left": 64, "right": 24, "top": 100, "bottom": 72},
              "xAxis": {"type": "value", "min": 1, "max": 80, "name": "epoch", "nameLocation": "middle", "nameGap": 24},
              "yAxis": {"type": "value", "scale": True, "name": "validation loss (nats per field)"},
              "dataZoom": [{"type": "inside"}, {"type": "slider", "height": 14, "bottom": 4}], "series": series}
    return write_chart(option, Path(out_dir) / "lr-recipe-val-loss.html",
                       title="The shared recipe ends lower on both arms, and the gain arrives at the late drop",
                       subtitle="Dashed: old recipe, 4e-4 cosine to 0. Solid: shared GIDD-style recipe, 5e-4 cosine to 5e-5. "
                                "A dot marks each run's steepest drop after epoch 30. Lower is better.",
                       note="One training seed per run. Validation loss: 4 fixed draws on 4,368 held-out teams. Old recipe trained on "
                            "the laptop GPU, shared recipe on a Modal GPU; the curves agree within 0.007 through epoch 40.", height=520)


def cliff_lr_figure(runs, out_dir, teaser=False):
    """Figure 3 (and, with teaser=True, the teaser): the learning rate each recipe gives per epoch, with each run's steepest
    late drop placed on its own recipe's curve. The teaser adds the measured results under the plot."""
    old_info, new_info = runs["Uniform"]
    series = []
    for label, info, dash in ((OLD_RECIPE, old_info, "dashed"), (NEW_RECIPE, new_info, "solid")):
        series.append({"type": "line", "name": f"learning rate · {label}", "color": "#6f6e69", "symbol": "none",
                       "lineStyle": {"type": dash, "width": 2}, "emphasis": {"focus": "series"},
                       "data": [[i + 1, round(v * 1e4, 4)] for i, v in enumerate(lr_by_epoch(info))]})
    for arm, (old, new) in runs.items():
        short = arm.split(",")[0]
        for label, info, filled in ((OLD_RECIPE, old, False), (NEW_RECIPE, new, True)):
            epoch = cliff(info)[0]
            lr = lr_by_epoch(info)[epoch - 1] * 1e4
            series.append({"type": "scatter", "name": f"{short} steepest drop · {label}", "symbolSize": 12,
                           "itemStyle": {"color": COLORS[arm] if filled else "#fff", "borderColor": COLORS[arm], "borderWidth": 2.5},
                           "label": {"show": True, "position": "right" if short == "Uniform" else "left", "fontSize": 11,
                                     "formatter": f"{short}, epoch {epoch}"},
                           "data": [[epoch, round(lr, 3)]]})
    bottom = 150 if teaser else 56
    option = {"tooltip": {"trigger": "item"},
              "grid": {"left": 64, "right": 24, "top": 36, "bottom": bottom},
              "xAxis": {"type": "value", "min": 1, "max": 80, "name": "epoch", "nameLocation": "middle", "nameGap": 24},
              "yAxis": {"type": "value", "min": 0, "max": 5.5, "interval": 1, "name": "learning rate (×10⁻⁴)"}, "series": series}
    if teaser:
        tiles = [("−0.12 / −0.10", "best validation loss, Uniform / Masked"),
                 ("16% → 44%", "Uniform drawn teams legal on the first try (without legality tables)"),
                 ("0% → 0%", "Masked drawn teams legal without the filter, every epoch")]
        option["graphic"] = [{"type": "group", "left": f"{4 + 33 * i}%", "bottom": 8, "children": [
            {"type": "text", "style": {"text": big, "fontSize": 26, "fontWeight": 700, "fill": "#0b0b0b"}},
            {"type": "text", "top": 36, "style": {"text": small, "fontSize": 11.5, "fill": "#52514e", "width": 200, "overflow": "break"}}]}
            for i, (big, small) in enumerate(tiles)]
        return write_chart(option, Path(out_dir) / "lr-recipe-teaser.html",
                           title="One shared learning-rate recipe: the late drop lands at a higher learning rate, both arms end lower",
                           subtitle="Grey: the learning rate each recipe gives (dashed old 4e-4 to 0, solid shared 5e-4 to 5e-5). "
                                    "Dots: each run's steepest late drop in validation loss, hollow for the old recipe.",
                           note="One training seed per run, 80 epochs each, all still improving at the cap. Legality: 64 teams per "
                                "epoch at guidance 1, drawn without the legal Pokémon construction filter.", height=560)
    return write_chart(option, Path(out_dir) / "lr-recipe-cliff-lr.html",
                       title="Under the shared recipe, the late drop lands while the learning rate is still higher",
                       subtitle="Grey: the learning rate each recipe gives. Dots: each run's steepest late drop in validation loss, "
                                "hollow for the old recipe. Uniform's drop moved 10 epochs earlier; Masked's moved 2 epochs later.",
                       note="Steepest drop: the epoch from 30 on whose validation loss fell most from the epoch before. One seed per run.",
                       height=460)


def field_matching_figure(runs, out_dir):
    old, new = runs["Uniform"]
    series = []
    for rule, color in ((ABILITY_RULE, "#0f766e"), (MOVE_RULE, "#c2410c")):
        for label, info, dash in ((OLD_RECIPE, old, "dashed"), (NEW_RECIPE, new, "solid")):
            series.append({"type": "line", "name": f"{rule} · {label}", "color": color, "symbol": "none",
                           "lineStyle": {"type": dash, "width": 2}, "emphasis": {"focus": "series"},
                           "data": [[e["epoch"], e["rule_shares"][rule]] for e in info["curves"] if "rule_shares" in e]})
    option = {"tooltip": {"trigger": "axis"}, "legend": {"top": 0, "left": 0, "itemGap": 14},
              "grid": {"left": 64, "right": 24, "top": 96, "bottom": 72},
              "xAxis": {"type": "value", "min": 1, "max": 80, "name": "epoch", "nameLocation": "middle", "nameGap": 24},
              "yAxis": {"type": "value", "min": 0, "max": 1, "name": "share of 64 drawn teams"},
              "dataZoom": [{"type": "inside"}, {"type": "slider", "height": 14, "bottom": 4}], "series": series}
    return write_chart(option, Path(out_dir) / "lr-recipe-field-matching.html",
                       title="Uniform: when does Pokémon-to-field matching click under each recipe?",
                       subtitle="Share of 64 teams drawn without the legal Pokémon construction filter that break each rule, every epoch.",
                       note="Draw: guidance 1, 128 passes, the same seed every epoch. Dashed, old recipe: 4e-4 cosine to 0. Solid, shared GIDD-style recipe: 5e-4 cosine to 5e-5.",
                       height=500)


def unfiltered_curves(runs, results=RESULTS):
    """{arm: (old entries, new entries)}: per-epoch legality drawn without the legal Pokémon construction filter.
    Uniform's training check already drew without it. Masked, full filter's drew through it, so its entries
    come from the checkpoint_legality.py redraw of its epoch checkpoints, filter off; an arm without both redraws is left out."""
    out = {}
    for arm, (old, new) in runs.items():
        if arm.startswith("Uniform"):
            out[arm] = (old["curves"], new["curves"])
            continue
        paths = [Path(results) / f"{name}_legality_no_tables_g1.json" for name in ARMS[arm]]     # guidance 1, as Uniform's
        if all(path.exists() for path in paths):
            out[arm] = tuple(json.loads(path.read_text())["epochs"] for path in paths)
    return out


DRAWS = {"Uniform": "128 passes over all fields", "Masked, full filter": "one field per pass, species first"}


def draw_note(curves):
    parts = [f"{arm}: {DRAWS[arm]}" for arm in curves]
    return ("64 teams per epoch at the elite cut and guidance 1, drawn without the legal Pokémon construction filter, the same seed every epoch ("
            + "; ".join(parts) + "). "
            + ("Masked, full filter's draws are a redraw of its saved epoch checkpoints with the filter off. "
               if "Masked, full filter" in curves else "")
            + "Old recipe: 4e-4 cosine to 0. Shared GIDD-style recipe: 5e-4 cosine to 5e-5.")


def legality_figure(curves, out_dir):
    """First-try legal share per epoch, every arm and recipe, drawn without the filter."""
    series = []
    for arm, (old, new) in curves.items():
        for label, entries, dash in ((OLD_RECIPE, old, "dashed"), (NEW_RECIPE, new, "solid")):
            series.append({"type": "line", "name": f"{arm} · {label}", "color": COLORS[arm], "symbol": "none",
                           "lineStyle": {"type": dash, "width": 2}, "emphasis": {"focus": "series"},
                           "data": [[e["epoch"], round(1 - e["illegal_share"], 4)] for e in entries if "illegal_share" in e]})
    option = {"tooltip": {"trigger": "axis"}, "legend": {"top": 0, "left": 0, "itemGap": 14},
              "grid": {"left": 64, "right": 24, "top": 96, "bottom": 72},
              "xAxis": {"type": "value", "min": 1, "max": 80, "name": "epoch", "nameLocation": "middle", "nameGap": 24},
              "yAxis": {"type": "value", "min": 0, "max": 1, "name": "legal on the first try"},
              "dataZoom": [{"type": "inside"}, {"type": "slider", "height": 14, "bottom": 4}], "series": series}
    return write_chart(option, Path(out_dir) / "lr-recipe-legality.html",
                       title="Share of 64 drawn teams legal on the first try, every epoch (without legality tables)",
                       subtitle="Dashed: old recipe. Solid: shared GIDD-style recipe. Every draw skips the legal Pokémon construction filter, so the line shows when the network itself learns the rules. "
                                "All 214,055 training teams pass that filter, so an illegal team is the model's own.",
                       note=draw_note(curves), height=500)


RULES = ("ability the species can't have", "move the species can't learn", "Stat Points over 66",
         "species used more than once", "item used more than once", "same move twice on one Pokémon",
         "wrong Mega Stone", "other item the species can't hold")


# Rows of the rule heatmap: the five rules that need each field matched to its own Pokémon, then the three team-wide ones
HEATMAP_RULES = ("ability the species can't have", "move the species can't learn", "same move twice on one Pokémon",
                 "wrong Mega Stone", "other item the species can't hold", "species used more than once",
                 "item used more than once", "Stat Points over 66")
# Categorical slots 1-8 of the dataviz reference palette, light and dark steps, one per rule in HEATMAP_RULES order
RULE_LIGHT = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
RULE_DARK = ("#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#0ca30c", "#9085e9", "#e66767")


def all_rules_figure(curves, out_dir):
    """Line chart, one colour per legality rule: share of 64 teams drawn without the filter that break it, per epoch.
    One run at a time; the switch under the chart picks it."""
    options, runs = [], []
    for arm, (old, new) in curves.items():
        for label, entries in ((OLD_RECIPE, old), (NEW_RECIPE, new)):
            runs.append(f"{arm.split(',')[0]} · {'old' if label == OLD_RECIPE else 'shared'} recipe")
            options.append({"series": [{"type": "line", "name": rule, "color": color, "symbol": "none", "lineStyle": {"width": 2},
                                        "emphasis": {"focus": "series"},
                                        "data": [[e["epoch"], e["rule_shares"].get(rule, 0)] for e in entries if "rule_shares" in e]}
                                       for rule, color in zip(HEATMAP_RULES, RULE_LIGHT)]})
    pct = "js:v => Math.round(v * 100) + '%'"
    base = {"backgroundColor": "transparent", "tooltip": {"trigger": "axis", "valueFormatter": pct, "order": "valueDesc"},
            "legend": {"top": 0, "left": 0, "data": list(HEATMAP_RULES), "itemGap": 14},
            "timeline": {"axisType": "category", "data": runs, "autoPlay": False, "bottom": 0, "left": 70, "right": 70,
                         "controlStyle": {"show": False}, "label": {"fontSize": 11}},
            "grid": {"left": 56, "right": 24, "top": 96, "bottom": 104},
            "xAxis": {"type": "value", "min": 1, "max": 80, "name": "epoch", "nameLocation": "middle", "nameGap": 26},
            "yAxis": {"type": "value", "min": 0, "max": 1, "axisLabel": {"formatter": pct}}}
    return write_chart({"baseOption": base, "options": options}, Path(out_dir) / "lr-recipe-all-rules.html",
                       title="Share of drawn teams breaking each rule, every epoch (without legality tables)",
                       subtitle="One line per rule; pick the run under the chart. A team can break several rules. Without legality tables means the "
                                "draws skip the legal Pokémon construction filter; all 214,055 training teams pass it.",
                       note=draw_note(curves), height=560, dark=dict(zip(RULE_LIGHT, RULE_DARK)))


def rules_at_end_figure(curves, out_dir, epoch=80, arm="Uniform"):
    """Figure 5: the share of Uniform's drawn teams breaking each rule at the last epoch, old recipe against shared."""
    series = []
    for label, entries, color in ((OLD_RECIPE, curves[arm][0], LIGHT[arm]), (NEW_RECIPE, curves[arm][1], COLORS[arm])):
        last = next(e for e in entries if e["epoch"] == epoch)
        series.append({"type": "bar", "name": label, "color": color, "barGap": "10%", "barCategoryGap": "30%",
                       "itemStyle": {"borderRadius": [0, 3, 3, 0]}, "emphasis": {"focus": "series"},
                       "label": {"show": True, "position": "right", "fontSize": 11,
                                 "formatter": "js:p => Math.round(p.value * 100) + '%'"},
                       "data": [round(last["rule_shares"].get(rule, 0), 4) for rule in HEATMAP_RULES]})
    option = {"legend": {"top": 0, "left": 0},
              "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"}, "valueFormatter": "js:v => Math.round(v * 100) + '%'"},
              "grid": {"left": 210, "right": 48, "top": 40, "bottom": 32},
              "xAxis": {"type": "value", "min": 0, "max": 0.6, "interval": 0.2, "axisLabel": {"formatter": "js:v => Math.round(v * 100) + '%'"}},
              "yAxis": {"type": "category", "data": list(HEATMAP_RULES), "inverse": True, "axisTick": {"show": False}},
              "series": series}
    return write_chart(option, Path(out_dir) / "lr-recipe-rules-epoch80.html",
                       title=f"Uniform at epoch {epoch}: Stat Points and repeated species are what still break",
                       subtitle="Share of 64 drawn teams breaking each rule. Light: old recipe. Dark: shared GIDD-style recipe. Shorter is better.",
                       note=draw_note({arm: curves[arm]}), height=440, dark={LIGHT[arm]: "#4a5654", COLORS[arm]: "#2bb3a3"})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--results", type=Path, default=RESULTS)
    args = parser.parse_args(argv)
    runs = {arm: (load(old, args.results), load(new, args.results)) for arm, (old, new) in ARMS.items()}
    args.out.mkdir(parents=True, exist_ok=True)
    curves = unfiltered_curves(runs, args.results)
    pages = [cliff_lr_figure(runs, args.out, teaser=True), val_loss_figure(runs, args.out), cliff_lr_figure(runs, args.out),
             legality_figure(curves, args.out), rules_at_end_figure(curves, args.out),
             all_rules_figure(curves, args.out), field_matching_figure(runs, args.out)]
    result = summary(runs)
    (args.out / "summary.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))
    for page in pages:
        print(page)


if __name__ == "__main__":
    main()
