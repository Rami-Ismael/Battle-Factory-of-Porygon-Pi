"""Two figures for the masking-rate rows' pre-registration (docs/masking-rate-rows.md), from stored results only.

  01_t_sensitivity   the default model's drawing distributions barely move when a pass is told another t
  02_error_by_ask    the default model's got against its ask: where the ask-versus-got error sits
  11_t_sensitivity_new_per_field_model   the same check for the new per-field model, the control of #297, beside the default model

    /tmp/vgc-pilot/.venv/bin/python src/masking_rate_preflight_figures.py      (from code/)
"""
import json
from pathlib import Path

import numpy as np

from echarts_chart import write_chart

RES = Path.home() / "vgc-data/results"
OUT = Path(__file__).resolve().parent.parent / "docs/figures/masking_rate_rows"
BLUE, AQUA, GREY, PLAIN = "#2a78d6", "#1baf7a", "#a9a8a2", "#6f6e69"
FIELD_TYPES = dict(species="Species", ability_and_item="Ability and item", moves="Moves", stat_alignment="Stat Alignment",
                   stat_points="Stat Points")
RULES = dict(corrected=("Corrected t", BLUE), always_1=("Every pass told t = 1", GREY), always_1_over_84=("Every pass told t = 1/84", AQUA))


def t_sensitivity():
    probe = json.load(open(RES / "masking_rate_rows_preflight.json"))["t_sensitivity"]
    asks = list(probe["by_ask"].values())
    mean = lambda rule, field: float(np.mean([a["total_variation_by_rule_and_field_type"][rule][field] for a in asks]))
    series = [dict(name=name, type="bar", itemStyle=dict(color=color), emphasis=dict(focus="series"), barGap="10%",
                   data=[round(mean(rule, field), 4) for field in FIELD_TYPES]) for rule, (name, color) in RULES.items()]
    series[0]["markLine"] = dict(symbol="none", silent=True, lineStyle=dict(color=PLAIN, type="dashed"),
                                 label=dict(formatter="bar set before the run: 0.01", position="insideStartTop", color=PLAIN),
                                 data=[dict(yAxis=probe["bar"])])
    option = dict(legend=dict(top=0), grid=dict(left=56, right=24, top=48, bottom=40),
                  tooltip=dict(trigger="axis", axisPointer=dict(type="shadow")),
                  xAxis=dict(type="category", data=list(FIELD_TYPES.values())),
                  yAxis=dict(type="value", name="Mean total variation", nameLocation="middle", nameGap=40, max=0.05), series=series)
    return write_chart(option, OUT / "01_t_sensitivity.html", title="The default model barely reads the t it is told",
                       subtitle="How far its drawing distribution moves when a pass is told another t, on the same partly filled teams. "
                                "0 means the same distribution, 1 means no overlap.",
                       note="Default model: epoch 80 of the labelled-teams training with label noise, species-first order, guidance 2. Mean over asks "
                            "6/24, 12/24 and 17/24, 256 teams each. Uncorrected t is 1 − filled/84. Source: masking_rate_rows_preflight.json.")


NEW_PER_FIELD_MODEL = "stat84_shuffled_noise_per_field_es3_cap80"     # the control's model of #297, trained on the shuffled-only table
NEW_BLUES = ("#174f94", "#2a78d6", "#86b4ec")
DEFAULT_GREYS = ("#5c5b57", "#94938d", "#cfcec8")  # outside echarts_chart.DARK, so the three stay apart in dark mode


def t_sensitivity_new_per_field_model(results=None, out=OUT):
    """Frame 5 for the new per-field model: the corrected t's total variation by field type, one bar per ask, beside the
    default model's at the same asks. results is masking_rate_rows_preflight.json; the new model's key is saved_as in
    masking_rate_preflight.py, t_sensitivity_<checkpoint>."""
    results = results if results is not None else json.load(open(RES / "masking_rate_rows_preflight.json"))
    new, default = results[f"t_sensitivity_{NEW_PER_FIELD_MODEL}"], results["t_sensitivity"]
    assert new["bar"] == default["bar"]
    corrected = lambda probe, ask: probe["by_ask"][ask]["total_variation_by_rule_and_field_type"]["corrected"]
    series = [dict(name=f"{who}, ask {ask}", type="bar", itemStyle=dict(color=color), emphasis=dict(focus="series"),
                   barGap="8%", data=[round(corrected(probe, ask)[field], 4) for field in FIELD_TYPES])
              for who, probe, colors in (("New per-field model", new, NEW_BLUES), ("Default model", default, DEFAULT_GREYS))
              for ask, color in zip(probe["by_ask"], colors)]
    series[0]["markLine"] = dict(symbol="none", silent=True, lineStyle=dict(color=PLAIN, type="dashed"),
                                 label=dict(formatter="bar set before the run: 0.01", position="insideStartTop", color=PLAIN),
                                 data=[dict(yAxis=new["bar"])])
    largest = lambda probe: max((v, f, a) for a in probe["by_ask"] for f, v in corrected(probe, a).items())
    (top, field, ask), (default_top, *_) = largest(new), largest(default)
    option = dict(legend=dict(top=0), grid=dict(left=64, right=24, top=64, bottom=40),
                  tooltip=dict(trigger="axis", axisPointer=dict(type="shadow")),
                  xAxis=dict(type="category", data=list(FIELD_TYPES.values())),
                  yAxis=dict(type="value", name="Total variation, corrected t", nameLocation="middle", nameGap=48, max=0.012),
                  series=series)
    return write_chart(option, out / "11_t_sensitivity_new_per_field_model.html",
                       title=f"The new per-field model moves {top:.4f} at most against a bar of {new['bar']:g}, "
                             f"so it barely reads the t it is told",
                       subtitle="How far each model's drawing distribution moves when a pass is told the corrected t and not 1 − filled/84, "
                                "on the same partly filled teams. 0 means the same distribution, 1 means no overlap.",
                       note=f"New per-field model: the control of #297, per-field masking rate, trained on the shuffled-only table, checkpoint "
                            f"{NEW_PER_FIELD_MODEL}. Its largest reading is {FIELD_TYPES[field]} at ask {ask}; the default model's largest is "
                            f"{default_top:.4f}. Each bar is one ask, 256 teams, guidance 2, sampling seed {new['sampling_seed']}. "
                            "Source: masking_rate_rows_preflight.json.", height=460)


def error_by_ask():
    rows = json.load(open(RES / "stat_points_84_gen84_noise_byepoch_ep80.json"))["summary"]["asks"]
    labels = [f"{r['numerator']}/24" for r in rows]
    scored = [r for r in rows if r["numerator"] <= 17]
    miss = [(r["win"] - r["ask"]) ** 2 for r in scored]
    share = sum(m for r, m in zip(scored, miss) if r["numerator"] >= 13) / sum(miss)
    area = dict(silent=True, itemStyle=dict(color=BLUE, opacity=0.08),
                label=dict(color=PLAIN, formatter=f"{share:.0%} of the error", position="insideTop"), data=[[dict(xAxis="13/24"), dict(xAxis="17/24")]])
    series = [dict(name="Got", type="line", symbolSize=6, lineStyle=dict(color=BLUE, width=2.5), itemStyle=dict(color=BLUE),
                   emphasis=dict(focus="series"), data=[round(r["win"], 4) for r in rows], markArea=area),
              dict(name="Got = ask", type="line", symbol="none", lineStyle=dict(color=GREY, type="dashed"), itemStyle=dict(color=GREY),
                   emphasis=dict(focus="series"), data=[round(r["ask"], 4) for r in rows])]
    option = dict(legend=dict(top=0), grid=dict(left=56, right=24, top=48, bottom=48), tooltip=dict(trigger="axis"),
                  xAxis=dict(type="category", data=labels, name="Ask", nameLocation="middle", nameGap=30),
                  yAxis=dict(type="value", name="Win rate got", nameLocation="middle", nameGap=40, min=0, max=1), series=series)
    return write_chart(option, OUT / "02_error_by_ask.html", title="The default model follows its ask up to 12/24, then stops near 0.53",
                       subtitle="Mean win rate got at each ask, against the line got = ask. The ask-versus-got error counts asks 1/24 to 17/24.",
                       note="Win rate: one battle against each of the 49 distinct top-50 meta teams, behaviour-cloning policy on both sides, slots "
                            "shuffled. 192 teams for each ask, 3 sampling seeds. Error 0.0053 on asks 1/24 to 17/24, and 0.0007 on asks 1/24 to 12/24 alone. "
                            "Source: stat_points_84_gen84_noise_byepoch_ep80.json.")


if __name__ == "__main__":
    for page in (t_sensitivity(), error_by_ask(), t_sensitivity_new_per_field_model()): print(page)
