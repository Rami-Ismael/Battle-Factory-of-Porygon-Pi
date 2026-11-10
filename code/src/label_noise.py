"""Random noise on win-rate labels, so a model cannot read a team's source from the label's exact value. Needs only torch."""
import re
from fractions import Fraction

import torch

FLAGGED = re.compile(r"(.+_labelnoise(\d+(?:over\d+)?))(?:_ep(\d+))?")
TEAM_SETS = {"_labelled_": "labelled_teams", "_same_loop_": "loop_teams", "_same_records_": "battle_records"}


def noisy_labels(w, width, generator):
    """w plus fresh uniform noise in [-width, +width], clipped to [0, 1]. NaN (no label) stays NaN.
    The noise comes from its own CPU generator, so it leaves the training run's other random draws unchanged."""
    if not width: return w
    noise = (torch.rand(w.shape, generator=generator) * 2 - 1) * width
    return torch.where(torch.isnan(w), w, (w + noise.to(w.device, w.dtype)).clamp(0, 1))


def noise_width(text):
    """A width typed on the command line, as 1/48 or 0.02, kept exact. ValueError unless it is in [0, 1]."""
    try:
        width = Fraction(text)
    except ZeroDivisionError:
        raise ValueError(f"{text} divides by zero") from None
    if not 0 <= width <= 1: raise ValueError(f"{text} is outside [0, 1]")
    return width


def noise_tag(width):
    """Checkpoint-name suffix for a width chosen by flag. Exact, so two widths never share a file."""
    return "_labelnoise" + str(width).replace("/", "over")


def flagged_checkpoint(name):
    """Where a checkpoint name printed by train84 --label-noise lives under the checkpoint folder; None for any other name."""
    m = FLAGGED.fullmatch(name)
    return None if not m else f"{m[1]}/{name}" if m[3] else name


def flagged_row(name, info):
    """The ask grid's row for a checkpoint trained with --label-noise, read from the run info the checkpoint holds.
    Epochs count the weights the file holds: with patience the training restores its best epoch."""
    _, tag, epoch = FLAGGED.fullmatch(name).groups()
    width = Fraction(tag.replace("over", "/"))
    noise = f"Label noise ±{width}" if width else "No label noise"
    members = info.get("members", [info])
    if any(m.get("label_noise") != float(width) for m in members):
        raise ValueError(f"{name}: the run info does not hold the width in this name")
    if not epoch and any("stopped_epoch" not in m for m in members):
        raise ValueError(f"{name}: this training has not finished")
    if "members" in info:
        return dict(checkpoint=name, reader=f"{noise}, same teams", stopped="fixed epoch counts, as the arm sets",
                    epochs=dict(zip(("loop_teams", "battle_records"), (m["stopped_epoch"] for m in members))))
    teams = next((v for k, v in TEAM_SETS.items() if k in name), "all_teams")
    tau = f", tau {info['ddom_tau']} weights" if info.get("ddom_tau") else ""
    if epoch:
        count, stopped = int(epoch), "one training, a checkpoint per epoch"
    else:
        count = info["best_epoch"] if info.get("patience") else info["stopped_epoch"]
        stopped = f"training stopped at epoch {info['stopped_epoch']}; the file holds the weights from epoch {count}"
    return dict(checkpoint=flagged_checkpoint(name), reader=f"{noise}, {teams.replace('_', ' ')}{tau}, {count} epochs",
                epochs={teams: count}, stopped=stopped, labelled=teams == "labelled_teams")
