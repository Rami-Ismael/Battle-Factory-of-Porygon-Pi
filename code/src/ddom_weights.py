"""Training weights from equation 5 of "Diffusion Models for Black-Box Optimization" (arXiv 2306.07180). Needs only numpy."""
import numpy as np


def bin_weights(labels, tau, k_share=0.01, bins=20):
    """One weight per team, averaging 1: |B| / (|B| + K) * exp(-(best - bin middle) / tau), with equal-width bins over
    the labels and K = k_share * N. Equation 5 sets no bin count; 20 comes from the code the paper links."""
    labels = np.asarray(labels, np.float64)
    if np.isnan(labels).any(): raise ValueError("every team needs a label: the rule is labelled teams only")
    edges = np.linspace(labels.min(), labels.max(), bins + 1)
    index = np.clip(np.digitize(labels, edges) - 1, 0, bins - 1)
    count = np.bincount(index, minlength=bins)
    middle = (edges[:-1] + edges[1:]) / 2
    per_bin = count / (count + k_share * len(labels)) * np.exp(-(labels.max() - middle) / tau)
    weights = per_bin[index]
    return weights / weights.mean()
