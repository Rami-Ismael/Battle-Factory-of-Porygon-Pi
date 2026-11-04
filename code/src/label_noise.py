"""Random noise on win-rate labels, so a model cannot read a team's source from the label's exact value. Needs only torch."""
from fractions import Fraction

import torch


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
