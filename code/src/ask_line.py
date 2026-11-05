"""How close an ask grid's mean got sits to the line got = ask (docs/asked-vs-got.md, GitHub issue #70).

The ask-versus-got error judges a model, replacing the slope on 2026-11-05: mean squared error between each ask and the mean got at that ask, scored on asks 1/24 to 17/24,
with asks 18/24 to 23/24 reported apart because the training data has almost nothing there. Near the line means under
0.001. Slope is reported last, because a steeper line can sit further from got = ask.
"""
import numpy as np

SCORED_ASKS = range(1, 18)
NEAR_LINE_MSE = 0.001


def line_measures(rows):
    """rows: one dict per ask, with "numerator" (out of 24) and "win", the mean got at that ask."""
    if not rows:
        return {}
    ask = np.array([r["numerator"] / 24 for r in rows])
    got = np.array([r["win"] for r in rows])
    mse = lambda rs: float(np.mean([(r["win"] - r["numerator"] / 24) ** 2 for r in rs])) if rs else None
    scored = mse([r for r in rows if r["numerator"] in SCORED_ASKS])
    return dict(mse_scored_asks=scored, near_line=None if scored is None else scored < NEAR_LINE_MSE,
                mse_asks_18_to_23=mse([r for r in rows if r["numerator"] not in SCORED_ASKS]),
                mean_abs_gap=float(np.mean(np.abs(got - ask))),
                slope=float(np.polyfit(ask, got, 1)[0]) if len(rows) > 1 else None)


def headline(line):
    """One report line: the error first, and a grid with no asks in a band says so instead of failing."""
    show = lambda v: "no asks" if v is None else f"{v:.4f}"
    return (f"ask-versus-got error: {show(line.get('mse_scored_asks'))} on asks 1/24 to 17/24 (near the line is under {NEAR_LINE_MSE}), "
            f"{show(line.get('mse_asks_18_to_23'))} on 18/24 to 23/24")
