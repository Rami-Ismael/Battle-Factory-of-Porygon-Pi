"""Random noise on win-rate labels, so a model cannot read a team's source from the label's exact value. Needs only torch."""
import torch


def noisy_labels(w, width, generator):
    """w plus fresh uniform noise in [-width, +width], clipped to [0, 1]. NaN (no label) stays NaN.
    The noise comes from its own CPU generator, so it leaves the training run's other random draws unchanged."""
    if not width: return w
    noise = (torch.rand(w.shape, generator=generator) * 2 - 1) * width
    return torch.where(torch.isnan(w), w, (w + noise.to(w.device, w.dtype)).clamp(0, 1))
