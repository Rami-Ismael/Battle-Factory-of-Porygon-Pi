"""Figures of "No tables in the loss" (issue #361; docs/no-tables-in-the-loss.md): illegal share by epoch, and errors per
team by epoch, one line per arm, both drawn without the legal Pokémon construction tables. Either arm may still be
partial; each line shows the epochs it has.

    python src/no_tables_in_the_loss_figure.py [folder]      # default ~/vgc-data/no_tables_in_the_loss
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, "/tmp/vgc-pilot/src")
sys.path.insert(0, str(HERE))
import asked_vs_got as X
from echarts_chart import write_chart

CONTROL = X.RES / "stat84_shuffled_noise_linear_es3_cap80_legality_no_tables.json"
ARM = X.RES / "stat84_shuffled_noise_linear_no_table_loss_es3_cap80.json"
FOLDER = Path.home() / "vgc-data" / "no_tables_in_the_loss"
LINES = {"No tables in the loss": "#c2410c", "Tables in the loss": "#52514e"}


def lines():
    """{reader name: [epoch entries]} for the arms that have any."""
    out = {}
    if ARM.exists(): out["No tables in the loss"] = [e for e in json.loads(ARM.read_text())["curves"] if "illegal_share" in e]
    if CONTROL.exists(): out["Tables in the loss"] = json.loads(CONTROL.read_text())["epochs"]
    return {k: v for k, v in out.items() if v}


def chart(data, key, name, axis, path, title, subtitle, percent):
    epochs = max(e["epoch"] for v in data.values() for e in v)
    fmt = "js:v => Math.round(v * 100) + '%'" if percent else None
    series = [dict(type="line", name=arm, color=LINES[arm], symbolSize=5, emphasis=dict(focus="series"),
                   data=[[e["epoch"], e[key]] for e in v]) for arm, v in data.items()]
    option = dict(grid=dict(left=64, right=24, top=56, bottom=80), legend=dict(top=0), animation=False,
                  tooltip=dict(trigger="axis", valueFormatter="js:v => v == null ? '' : " +
                               ("(v * 100).toFixed(1) + '%'" if percent else "v.toFixed(2)")),
                  xAxis=dict(type="value", min=1, max=epochs, name="Training epoch", nameLocation="middle", nameGap=30),
                  yAxis=dict(type="value", min=0, max=1 if percent else None, name=axis, nameLocation="middle", nameGap=46,
                             axisLabel=dict(formatter=fmt) if fmt else {}),
                  dataZoom=[dict(type="inside"), dict(type="slider", bottom=8, height=18)], series=series)
    return write_chart(option, path / name, title=title, subtitle=subtitle, height=440)


def reasons(entries):
    last = entries[-1]
    return f"epoch {last['epoch']}: " + "; ".join(f"{n} of {last['legality_teams']} {r}" for r, n in last["rejection_reasons"][:3])


def main(folder=FOLDER):
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    data = lines()
    if not data: raise SystemExit("no legality results yet")
    last = {arm: v[-1] for arm, v in data.items()}
    head = " · ".join(f"{arm} {e['illegal_share']:.0%} illegal at epoch {e['epoch']}" for arm, e in last.items())
    common = "64 teams per epoch at the elite cut 0.531, guidance 2, species-first, drawn without the tables and the Stat Point budget."
    a = chart(data, "illegal_share", "illegal-share-by-epoch.html", "Illegal share of first draws", folder,
              title=f"Illegal share by epoch: {head}", subtitle=common + " Top reasons, " +
              " | ".join(f"{arm}, {reasons(v)}" for arm, v in data.items()), percent=True)
    head = " · ".join(f"{arm} {e['errors_per_team']:.1f}" for arm, e in last.items())
    b = chart(data, "errors_per_team", "errors-per-team-by-epoch.html", "Validator errors per team", folder,
              title=f"Errors per team by epoch: {head}", subtitle=common + " A legal team has 0.", percent=False)
    print(a); print(b)


if __name__ == "__main__":
    main(*sys.argv[1:])
