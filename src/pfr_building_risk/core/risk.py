"""Risk (Vogelbacher et al. 2026, Eqs. 14-15) and the authors' iterated-mean classes."""

import numpy as np

CLASS_NAMES = ("no risk", "low", "medium", "high", "very high")


def risk(svpf, exposure, hazard) -> np.ndarray:
    """PFR = SV_PF x E x H (unit exponents)."""
    return np.asarray(svpf, float) * np.asarray(exposure, float) * np.asarray(hazard, float)


def iterated_means(values, n: int = 3) -> list[float]:
    """Class breaks: the mean, then the mean of the values above it, ``n`` times.

    As the authors' step-6 script; n = 3 is inferred from the paper's four non-zero classes.
    """
    v, breaks = np.asarray(values, float), []
    for _ in range(n):
        m = v.mean()
        breaks.append(float(m))
        v = v[v > m]
        if len(v) == 0:
            break
    return breaks


def risk_class(values, breaks) -> np.ndarray:
    """0 = no risk (value <= 0), then 1..len(breaks)+1 by the breaks (right-closed)."""
    v = np.asarray(values, float)
    return np.where(v <= 0, 0, np.digitize(v, breaks, right=True) + 1)
