"""Exposure: people per building and on given floors — hazard-agnostic.

Which floor matters depends on the hazard: the ground floor for flooding (water entering
the building), the top floor for heat, all floors for access and evacuation.
"""

import numpy as np
import pandas as pd


def allocate(units: pd.Series, unit_of: pd.Series, weight: pd.Series) -> pd.Series:
    """Share each unit's total (e.g. a census section's population) among its buildings
    in proportion to ``weight`` (e.g. dwellings, living area).

    ``units`` is indexed by unit id; ``unit_of`` and ``weight`` are per building. Buildings
    with weight 0, or outside every unit, get 0.
    """
    weight = weight.fillna(0).astype(float)
    total = weight.groupby(unit_of).transform("sum")
    share = np.divide(weight, total, out=np.zeros(len(weight)), where=total > 0)
    return pd.Series(unit_of.map(units).fillna(0).to_numpy() * share, index=weight.index)


def residents_on_floor(residents, floors, eligible=None, floor: str = "ground") -> np.ndarray:
    """Residents on one floor, assuming residents are spread evenly over the floors.

    ``floor`` is "ground" or "top" (both = residents / floors) or "all" (= residents).
    ``eligible`` (bool per building) marks buildings where that floor is residential;
    elsewhere 0 — e.g. the paper's mixed-use rule (ground floor commercial).
    """
    r = np.asarray(residents, float)
    if floor == "all":
        out = r
    elif floor in ("ground", "top"):
        out = r / np.clip(np.asarray(floors, float), 1, None)
    else:
        raise ValueError(f"floor must be 'ground', 'top' or 'all', not {floor!r}")
    return out if eligible is None else np.where(np.asarray(eligible, bool), out, 0.0)
