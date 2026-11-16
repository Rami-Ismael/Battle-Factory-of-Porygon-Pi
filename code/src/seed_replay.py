"""How novel are the species-first grids' teams once redrawn on seeds of their own? (GitHub issue #353)

Every grid before 2026-11-15 drew with torch.manual_seed(7300 + s). Earlier grids' teams entered the training table, so
species-first grids on those seeds redrew them and read as copies. The redraws (ask_grid84.py gen with AG84_SEED0, no
battles) sit beside each stored grid as stat_points_84_<grid>_seed<N>.json; ~/vgc-data/seed_replay/bases.txt names N.

    python seed_replay.py read              stored against redrawn, into results/seed_replay.json
    python seed_replay.py figure DIR        novel share, stored against redrawn, as an ECharts page in DIR
    python seed_replay.py comment URL       the result comment; URL holds {theme}
"""
import datetime
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import grid_novelty
import masking_rate_rows as MRR

BASES = Path.home() / "vgc-data/seed_replay/bases.txt"
OUT = MRR.RES / "seed_replay.json"
READER = {"sched_species_first": "Per-field, species-first (control)", "sched_linear_species_first": "Linear, species-first"}


swaps = grid_novelty.swaps


def measures(path, train):
    """Novel share, swap shares, distinct line-ups and effective species of one grid file's teams."""
    import line_up_yield as LY
    cells = json.load(open(path))["cells"].values()
    lus = [LY.line_up(p) for c in cells for p in c["pastes"]]
    s = swaps(lus, train)
    return dict(teams=len(lus), novel_share=float(np.mean([lu not in train for lu in lus])),
                swaps=dict(copied=float(np.mean(s == 0)), one=float(np.mean(s == 1)), two=float(np.mean(s == 2)), three_or_more=float(np.mean(s >= 3))),
                line_ups=len(set(lus)), novel_line_ups=len({lu for lu in lus if lu not in train}),
                effective_species=float(np.mean([c["div"]["effective_species"] for c in cells if c.get("div")])),
                seed0=json.load(open(path)).get("seed0", 7300))


def read():
    train = MRR.references()[0]; out = dict(reference=f"line-ups of the labelled training teams of {MRR.TABLE} ({len(train):,})", grids={})
    for line in BASES.read_text().split("\n"):
        if not line.strip(): continue
        name, base = line.split()
        stored, redrawn = MRR.RES / f"stat_points_84_{name}.json", MRR.RES / f"stat_points_84_{name}_seed{base}.json"
        out["grids"][name] = dict(stored=measures(stored, train), redrawn=measures(redrawn, train))
        a, b = out["grids"][name]["stored"], out["grids"][name]["redrawn"]
        print(f"{name:32s} novel {a['novel_share']:.3f} -> {b['novel_share']:.3f}   line-ups {a['line_ups']} -> {b['line_ups']}   "
              f"effective species {a['effective_species']:.2f} -> {b['effective_species']:.2f}")
    json.dump(out, open(OUT, "w"), indent=1); print("SEED_REPLAY_DONE")


def split(name):
    row, _, g = name.partition("_g"); return row, int(g) if g else 2


def figure(folder):
    """One dumbbell per grid: novel share on the replayed seeds 7301 to 7303 against on seeds of its own."""
    from echarts_chart import write_chart
    r = json.load(open(OUT))["grids"]; names = list(r)
    labels = [f"{READER[split(n)[0]]}, guidance {split(n)[1]}" for n in names]
    GREY, BLUE = "#9a9893", "#2a78d6"
    series = [dict(type="line", silent=True, symbol="none", color=GREY, lineStyle=dict(width=2, opacity=0.5), tooltip=dict(show=False),
                   data=[[r[n]["stored"]["novel_share"], i], [r[n]["redrawn"]["novel_share"], i]]) for i, n in enumerate(names)]
    for key, name, color in (("stored", "Stored grid, seeds 7301 to 7303", GREY), ("redrawn", "Redrawn on seeds of its own", BLUE)):
        series.append(dict(type="scatter", name=name, color=color, symbolSize=13, emphasis=dict(focus="series"),
                           label=dict(show=True, position="right" if key == "redrawn" else "left", fontSize=12, formatter="js:p => Math.round(p.value[0] * 100) + '%'"),
                           data=[[r[n][key]["novel_share"], i] for i, n in enumerate(names)]))
    option = dict(grid=dict(left=290, right=50, top=48, bottom=50), legend=dict(top=0),
                  tooltip=dict(trigger="item", formatter="js:p => p.seriesName + ': ' + (p.value[0] * 100).toFixed(1) + '% novel'"),
                  xAxis=dict(type="value", min=0, max=0.6, name="Share of teams with a novel line-up", nameLocation="middle", nameGap=32,
                             axisLabel=dict(formatter="js:v => Math.round(v * 100) + '%'")),
                  yAxis=dict(type="category", data=labels, inverse=True, axisTick=dict(show=False), axisLabel=dict(fontSize=13)),
                  series=series, animation=False)
    return write_chart(option, Path(folder) / "seed-replay.html", height=420,
                       title="On seeds of their own, species-first grids draw far more novel line-ups",
                       subtitle="The stored grids drew on seeds 7301 to 7303, where they redrew earlier grids' teams that are now training teams.",
                       note="Novel line-up: six species that appear together in no labelled training team of the shuffled-only table. "
                            "4,416 teams per grid: 23 asks, 3 sampling seeds, 64 teams per cell. Redraws are not battled.")


def comment(figure_url):
    r = json.load(open(OUT)); G = r["grids"]
    lo = min(g["redrawn"]["novel_share"] / g["stored"]["novel_share"] for g in G.values()); hi = max(g["redrawn"]["novel_share"] / g["stored"]["novel_share"] for g in G.values())
    lines = [f"**Result, {datetime.date.today()}: on seeds of their own the species-first grids' novel share rises {lo:.1f} to {hi:.1f} times.** "
             "The stored grids' low novelty came from the seeds they shared with earlier grids, not from species-first decoding.", "",
             f'<picture><source media="(prefers-color-scheme: dark)" srcset="{figure_url.format(theme="dark")}"><img alt="Novel share of six species-first grids, stored on seeds 7301 to 7303 against redrawn on seeds of their own" src="{figure_url.format(theme="light")}"></picture>', "",
             "| Grid | Novel share, stored → redrawn | Copied / 1 swap / 2 / 3+ swaps, redrawn | Distinct line-ups, stored → redrawn | Effective species, stored → redrawn |",
             "|---|---|---|---|---|"]
    for n, g in G.items():
        a, b = g["stored"], g["redrawn"]; s = b["swaps"]
        lines.append(f"| {READER[split(n)[0]]}, guidance {split(n)[1]} | {a['novel_share']:.3f} → {b['novel_share']:.3f} | "
                     f"{s['copied']:.2f} / {s['one']:.2f} / {s['two']:.2f} / {s['three_or_more']:.2f} | {a['line_ups']:,} → {b['line_ups']:,} | "
                     f"{a['effective_species']:.2f} → {b['effective_species']:.2f} |")
    lines += ["", "- **Novel share**: teams whose six species appear together in no labelled training team of the shuffled-only table.",
              "- **Swaps**: species swaps from the closest training line-up; 0 is a copied line-up.",
              "- **Effective species**: exp of the entropy of species counts within 5-team subsets of one cell, averaged over cells.",
              "- **Redrawn**: the same checkpoint, decoder and guidance, `ask_grid84.py gen` with `AG84_SEED0` set to the grid's own base; 4,416 teams, no battles. "
              "Got and the ask-versus-got error stay as stored.", "",
              "New grids no longer share seeds: `grid_seeds.py` gives each a base from its own name, and stored grids keep 7300 so they still reproduce. "
              "Script: `code/src/seed_replay.py`. Results: `~/vgc-data/results/seed_replay.json`."]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    {"read": read, "figure": lambda: print(figure(sys.argv[2])), "comment": lambda: print(comment(sys.argv[2]), end="")}[sys.argv[1]]()
