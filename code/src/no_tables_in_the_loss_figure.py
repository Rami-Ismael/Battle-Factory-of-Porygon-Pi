"""Figure and result comment of "No tables in the loss" (issue #361; docs/no-tables-in-the-loss.md): illegal share by
epoch, one line per arm, both drawn without the legal Pokémon construction tables. Either arm may still be partial;
each line shows the epochs it has.

    python src/no_tables_in_the_loss_figure.py figure [folder]     # default ~/vgc-data/no_tables_in_the_loss/figure
    python src/no_tables_in_the_loss_figure.py comment <image url>  # the #361 result comment, Markdown on stdout
"""
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, "/tmp/vgc-pilot/src")
sys.path.insert(0, str(HERE))
import asked_vs_got as X
from echarts_chart import write_chart

CONTROL = X.RES / "stat84_shuffled_noise_linear_es3_cap80_legality_no_tables.json"
ARM_NAME = "stat84_shuffled_noise_linear_no_table_loss_es3_cap80"
ARM = X.RES / f"{ARM_NAME}.json"
FOLDER = Path.home() / "vgc-data" / "no_tables_in_the_loss" / "figure"
LINES = {"No tables in the loss": "#c2410c", "Tables in the loss": "#52514e"}
REFUTED_ABOVE = 0.875                                    # the plan: refuted if the new arm stays above this at every epoch
POOL = 10                                                # last epochs pooled for the comparison


def lines():
    """{reader name: [epoch entries]} for the arms that have any."""
    out = {}
    if ARM.exists(): out["No tables in the loss"] = [e for e in json.loads(ARM.read_text())["curves"] if "illegal_share" in e]
    if CONTROL.exists(): out["Tables in the loss"] = json.loads(CONTROL.read_text())["epochs"]
    return {k: v for k, v in out.items() if v}


def reasons(entries):
    last = entries[-1]
    return f"epoch {last['epoch']}: " + "; ".join(f"{n} of {last['legality_teams']} {r}" for r, n in last["rejection_reasons"][:3])


def figure(folder=FOLDER):
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    data = lines()
    if not data: raise SystemExit("no legality results yet")
    head = " · ".join(f"{arm} {v[-1]['illegal_share']:.0%} illegal at epoch {v[-1]['epoch']}" for arm, v in data.items())
    epochs = max(e["epoch"] for v in data.values() for e in v)
    series = [dict(type="line", name=arm, color=LINES[arm], symbolSize=5, emphasis=dict(focus="series"),
                   data=[[e["epoch"], e["illegal_share"]] for e in v]) for arm, v in data.items()]
    option = dict(grid=dict(left=64, right=24, top=56, bottom=80), legend=dict(top=0), animation=False,
                  tooltip=dict(trigger="axis", valueFormatter="js:v => v == null ? '' : (v * 100).toFixed(1) + '%'"),
                  xAxis=dict(type="value", min=1, max=epochs, name="Training epoch", nameLocation="middle", nameGap=30),
                  yAxis=dict(type="value", min=0, max=1, name="Illegal share of first draws", nameLocation="middle", nameGap=46,
                             axisLabel=dict(formatter="js:v => Math.round(v * 100) + '%'")),
                  dataZoom=[dict(type="inside"), dict(type="slider", bottom=8, height=18)], series=series)
    subtitle = ("64 teams per epoch at the elite cut 0.531, guidance 2, species-first, drawn without the tables and the Stat Point "
                "budget. Top reasons, " + " | ".join(f"{arm}, {reasons(v)}" for arm, v in data.items()))
    print(write_chart(option, folder / "illegal-share-by-epoch.html", title=f"Illegal share by epoch: {head}",
                      subtitle=subtitle, height=440))


def pooled(entries):
    """(share, teams) over the last POOL epochs."""
    tail = entries[-POOL:]; n = sum(e["legality_teams"] for e in tail)
    return sum(e["illegal_share"] * e["legality_teams"] for e in tail) / n, n


def round_trip():
    """The Round trip probe rerun on the new arm's saved model (run-experiment step 3.1)."""
    import diffusion84 as M
    import lossdown as L
    from no_tables_in_the_loss_preflight import round_trip as probe, training_rows
    return probe(M.Model84.load(L.CK / f"{ARM_NAME}.pt", X.setup()[0]), training_rows()[0])


def comment(image):
    data = lines()
    arm, control = data["No tables in the loss"], data["Tables in the loss"]
    info = json.loads(ARM.read_text())
    low = min(arm, key=lambda e: e["illegal_share"])
    below = [e["epoch"] for e in arm if e["illegal_share"] <= REFUTED_ABOVE]
    same = {e["epoch"]: e for e in control}
    shared = [e for e in arm if e["epoch"] in same]
    (pa, na), (pc, nc) = pooled(shared), pooled([same[e["epoch"]] for e in shared])
    gap = pa - pc; half = 1.96 * math.sqrt(pa * (1 - pa) / na + pc * (1 - pc) / nc)
    trip = round_trip()
    verdict = (f"**The claim holds:** the illegal share falls to {REFUTED_ABOVE:.3f} or below at epoch {below[0]}." if below else
               f"**Refuted:** the illegal share stays above {REFUTED_ABOVE:.3f} at every epoch, as the plan defined.")
    copies = max(e.get("copied_share", 0) for e in arm)
    print("\n".join([
        f"![Illegal share by training epoch, drawn without the tables: No tables in the loss {arm[-1]['illegal_share']:.0%} at "
        f"epoch {arm[-1]['epoch']}, Tables in the loss {control[-1]['illegal_share']:.0%} at epoch {control[-1]['epoch']}]({image})",
        "",
        verdict,
        "",
        "| | No tables in the loss (`shuffled_linear_no_table_loss`) | Tables in the loss (`shuffled_linear`) |",
        "| --- | --- | --- |",
        f"| Illegal share at epoch 1 | {arm[0]['illegal_share']:.0%} | {control[0]['illegal_share']:.0%} |",
        f"| Lowest illegal share | {low['illegal_share']:.0%} at epoch {low['epoch']} | {min(e['illegal_share'] for e in control):.0%} |",
        f"| Last {POOL} shared epochs pooled | {pa:.3f} ({na} teams) | {pc:.3f} ({nc} teams) |",
        f"| Epochs trained | {info['stopped_epoch']} (best {info['best_epoch']}, patience 3) | 80 |",
        f"| Top reasons, last epoch | {reasons(arm)} | {reasons(control)} |",
        "",
        f"Difference over the last {POOL} shared epochs: {gap:+.3f}, 95% interval {gap - half:+.3f} to {gap + half:+.3f} "
        f"(smallest effect worth acting on 0.125). Both drawn without the tables: 64 teams per epoch at the elite cut 0.531, "
        f"guidance 2, species-first, seed 12345. All 84 fields generated, none copied. No battles. Confirmatory. "
        f"One training seed per arm. Round trip on the new arm's saved model: {trip['identical']} of {trip['teams']} identical."
        + (f" Up to {copies:.0%} of an epoch's teams copy a training team." if copies else ""),
    ]))


if __name__ == "__main__":
    mode, *rest = sys.argv[1:] or ["figure"]
    {"figure": figure, "comment": comment}[mode](*rest)
