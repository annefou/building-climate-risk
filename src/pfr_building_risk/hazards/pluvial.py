"""Pluvial flood hazard (Vogelbacher et al. 2026, NHESS, Sect. 3.1.3).

- **Well-being (HWB)**: for each depth threshold (paper: >= 30, 40, ... 100 cm, nested
  flood-extent polygons), the flooded fraction of the ring around the building, through
  the log-normal response, summed over thresholds.
- **Mobility and accessibility (HMA)**: at the 30 cm threshold, the largest of the flooded
  fractions of the ring at ``buffers[0]`` and of the street area in the 0-``buffers[1]``
  ring and the ``buffers[1]``-``buffers[2]`` band, through the same response.
"""

from collections.abc import Mapping, Sequence

import geopandas as gpd
import numpy as np

from ..core.response import lognormal_cdf
from ..core.zonal import annulus, clip_to, covered_fraction, rings_without_buildings

flooded_fraction = covered_fraction


def hazard_wellbeing(footprints: gpd.GeoSeries, floods: Mapping[int, gpd.GeoSeries],
                     buffer: float = 2.0) -> np.ndarray:
    """HWB: sum over depth thresholds of the response to the flooded fraction of the ring."""
    ring = rings_without_buildings(footprints, buffer)
    return sum(lognormal_cdf(covered_fraction(ring, floods[d])) for d in sorted(floods))


def mobility_fractions(footprints: gpd.GeoSeries, flood: gpd.GeoSeries, streets: gpd.GeoSeries,
                       buffers: Sequence[float] = (5.0, 15.0, 30.0)) -> dict[str, np.ndarray]:
    """The three flooded fractions behind HMA, plus the street areas they are measured on."""
    near, mid, far = buffers
    f_near = covered_fraction(rings_without_buildings(footprints, near), flood)
    inner = clip_to(rings_without_buildings(footprints, mid), streets)
    outer = clip_to(annulus(footprints, mid, far), streets)
    return {"f_near": f_near, "f_inner": covered_fraction(inner, flood),
            "f_outer": covered_fraction(outer, flood),
            "street_inner_m2": inner.area.values, "street_outer_m2": outer.area.values}


def hazard_mobility(fractions: Mapping[str, np.ndarray]) -> np.ndarray:
    """HMA: response to the largest of the three flooded fractions (0 when all are 0)."""
    fmax = np.maximum.reduce([fractions["f_near"], fractions["f_inner"], fractions["f_outer"]])
    return np.where(fmax > 0, lognormal_cdf(fmax), 0.0)
