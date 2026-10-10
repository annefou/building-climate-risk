"""Response functions: turn a hazard measure into a 0-1 hazard score."""

import numpy as np
from scipy.stats import lognorm


def lognormal_cdf(x, shape: float = 0.25, scale: float = 4.0) -> np.ndarray:
    """Log-normal CDF of ``scale * x`` (the paper's transformation of flooded fractions)."""
    return lognorm.cdf(scale * np.asarray(x, float), shape)


def exceedance(x, threshold: float) -> np.ndarray:
    """1 where ``x`` reaches ``threshold``, else 0."""
    return (np.asarray(x, float) >= threshold).astype(float)
