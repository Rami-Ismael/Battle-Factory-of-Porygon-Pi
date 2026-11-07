"""The ask-versus-got error of one sampling arm, and its paired difference from a reference arm (src/ask_line.py, GitHub issue #93).

An arm is judged on a few asks, not the full 1/24 to 17/24 grid, so the error here is the mean squared gap between each named ask
and the mean got there. Lower is closer to the line got = ask.
"""
import numpy as np

from ask_line import line_measures


def ask_error(win_by_ask, asks):
    """win_by_ask: mean got per ask numerator (out of 24); asks: the numerators that count."""
    return line_measures([dict(numerator=k, win=win_by_ask[k]) for k in asks])["mse_scored_asks"]


def paired_difference(boot_arm, boot_ref, asks):
    """Arm error minus reference error over shared bootstrap draws, with a 95% interval.

    boot_arm, boot_ref: ask numerator -> array of bootstrap mean gots, draw i of both arms from the same resample.
    share_closer is the share of draws where the arm sits closer to the line than the reference.
    """
    def err(boot):
        return np.mean([(boot[k] - k / 24) ** 2 for k in asks], 0)

    delta = err(boot_arm) - err(boot_ref)
    return dict(interval=[float(np.quantile(delta, .025)), float(np.quantile(delta, .975))], share_closer=float(np.mean(delta < 0)))
