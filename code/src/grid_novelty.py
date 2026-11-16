"""Novelty and diversity of one ask grid's teams, printed and saved by every `ask_grid84.py report`.

Owner, 2026-11-16: every new model or decoding row is tested on whether it writes novel line-ups and diverse teams,
beside its win rate. Novelty compares each team's line-up (its six species) with the line-ups of the labelled
training teams; diversity compares the grid's teams with each other. Moved here from seed_replay.py, which found on
2026-11-15 that grids reusing seeds 7301-7303 redrew training teams (#353).
"""
import numpy as np

import line_up_yield as LY


def swaps(line_ups, train):
    """For each line-up, how many species swaps from the closest training line-up: 0 (copied) to 6."""
    names = sorted({s for lu in train for s in lu} | {s for lu in line_ups for s in lu}); idx = {s: i for i, s in enumerate(names)}
    def onehot(lus):
        m = np.zeros((len(lus), len(names)), np.float32)
        for i, lu in enumerate(lus): m[i, [idx[s] for s in lu]] = 1
        return m
    T, V = onehot(sorted(train)), onehot(line_ups)
    return np.concatenate([6 - (V[i:i + 512] @ T.T).max(1) for i in range(0, len(V), 512)]).astype(int)


def measures(line_ups, train):
    """Novel share, swap shares, and distinct and distinct novel line-ups of a list of line-ups."""
    s = swaps(line_ups, train)
    return dict(teams=len(line_ups), novel_share=float(np.mean([lu not in train for lu in line_ups])),
                swaps=dict(copied=float(np.mean(s == 0)), one=float(np.mean(s == 1)), two=float(np.mean(s == 2)), three_or_more=float(np.mean(s >= 3))),
                line_ups=len(set(line_ups)), novel_line_ups=len({lu for lu in line_ups if lu not in train}))


def from_cells(cells, train):
    """measures() over a grid's cells, plus line-ups within one ask and the cells' 5-team-subset effective species."""
    by_ask = {}
    for c in cells:
        by_ask.setdefault(c["numerator"], []).extend(LY.line_up(p) for p in c["pastes"])
    lus = [lu for v in by_ask.values() for lu in v]
    if not lus: return dict(teams=0)
    divs = [c["div"]["effective_species"] for c in cells if c.get("div")]
    return dict(measures(lus, train), within_ask_line_ups=float(np.mean([len(set(v)) for v in by_ask.values() if v])),
                effective_species=float(np.mean(divs)) if divs else None)


def summary_line(m):
    if not m.get("teams"): return "novelty and diversity: no teams"
    s, eff = m["swaps"], m.get("effective_species")
    return (f"novelty: {m['novel_share']:.1%} of {m['teams']:,} teams novel (copied {s['copied']:.1%}, 1 swap {s['one']:.1%}, "
            f"2 swaps {s['two']:.1%}, 3+ {s['three_or_more']:.1%}); diversity: {m['line_ups']:,} line-ups pooled "
            f"({m['novel_line_ups']:,} novel), {m['within_ask_line_ups']:.1f} per ask, effective species "
            f"{'n/a' if eff is None else f'{eff:.2f}'} (5-team subsets)")


def training_line_ups(table="lossdata84_shuffled_only.npz"):
    """Line-ups of the labelled training teams of the table the current models train on."""
    LY.TABLE, LY.LABELLED = table, True
    return LY.references("")[0]
