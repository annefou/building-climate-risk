"""Combine vulnerability, exposure and hazard into risk indices and classes, from a spec.

The spec is a plain dict (a study can keep it in YAML), e.g. for the pluvial-flood
framework of Vogelbacher et al. (2026)::

    vulnerability:
      sensitivity: {columns: [C, ES], weights: [0.3, 0.7]}
      coping:      {columns: [WR, EDQ], weights: [0.5, 0.5]}
      svi_weights: [0.5, 0.5]
      threshold: 0.25
      exponent: 2.0
    risks:
      PFRWB: {exposure: R_G, hazard: HWB}
      PFRMA: {exposure: R,   hazard: HMA}
    classes: {iterations: 3}

``hazards`` maps each hazard name in ``risks`` to one value per building.
"""

from collections.abc import Mapping

import numpy as np
import pandas as pd

from .risk import iterated_means, risk, risk_class
from .vulnerability import social_vulnerability


def assess(buildings: pd.DataFrame, hazards: Mapping[str, np.ndarray], spec: Mapping) -> pd.DataFrame:
    """Per building: Sensitivity, CopingCapacity, SVI, SVPF, each hazard, each risk and its class."""
    v = spec["vulnerability"]
    out = social_vulnerability(
        buildings,
        sensitivity_cols=v["sensitivity"]["columns"], sensitivity_weights=v["sensitivity"]["weights"],
        coping_cols=v["coping"]["columns"], coping_weights=v["coping"]["weights"],
        svi_weights=v.get("svi_weights", (0.5, 0.5)),
        threshold=v.get("threshold", 0.25), exponent=v.get("exponent", 2.0))
    for name, values in hazards.items():
        out[name] = np.asarray(values, float)
    iterations = spec.get("classes", {}).get("iterations", 3)
    for name, r in spec["risks"].items():
        out[name] = risk(out["SVPF"], buildings[r["exposure"]].to_numpy(float), out[r["hazard"]])
        out[f"{name}_class"] = risk_class(out[name], iterated_means(out[name], iterations))
    return out
