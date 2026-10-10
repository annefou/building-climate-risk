"""Building-level hazard indices from flood-extent polygons (Vogelbacher et al. 2026, Sect. 3.1.3).

Rings around a building exclude **every** building footprint, not only the building's
own, as the authors' stored intermediate fractions require. Distances are in the units
of the input CRS: use a projected CRS with ground metres (the authors' worked example
is stored in Web Mercator, where a "2 m" buffer is ~1.2 m on the ground at 53.5 N).

Spatial operations are local: each ring is differenced/intersected only with the
footprints / flood polygons that touch it (STRtree query). This is equal to using the
union of all of them, and scales to whole cities.
"""

from collections.abc import Mapping, Sequence

import geopandas as gpd
import numpy as np
import shapely
from scipy.stats import lognorm

SHAPE = 0.25
SCALE = 4.0


def _local_overlay(geoms: np.ndarray, others: gpd.GeoSeries, op: str) -> np.ndarray:
    """Per geometry: difference from / intersection with the union of the ``others`` it touches."""
    others = others[~others.is_empty & others.notna()]
    tree = shapely.STRtree(others.values)
    src, hit = tree.query(geoms, predicate="intersects")
    out = geoms.copy() if op == "difference" else np.full(len(geoms), shapely.Polygon(), dtype=object)
    if len(src) == 0:
        return out
    order = np.argsort(src, kind="stable")
    src, hit = src[order], hit[order]
    starts = np.flatnonzero(np.r_[True, src[1:] != src[:-1]])
    ends = np.r_[starts[1:], len(src)]
    other_vals = others.values
    for s, e in zip(starts, ends):
        i = src[s]
        local = shapely.union_all(other_vals[hit[s:e]])
        out[i] = (shapely.difference(geoms[i], local) if op == "difference"
                  else shapely.intersection(geoms[i], local))
    return out


def rings_without_buildings(footprints: gpd.GeoSeries, distance: float) -> gpd.GeoSeries:
    """Buffer each footprint by ``distance`` and remove every building footprint from it."""
    rings = _local_overlay(footprints.buffer(distance).values, footprints, "difference")
    return gpd.GeoSeries(rings, index=footprints.index, crs=footprints.crs)


def clip_to(areas: gpd.GeoSeries, mask: gpd.GeoSeries) -> gpd.GeoSeries:
    """Intersection of each area with the union of the ``mask`` polygons it touches."""
    return gpd.GeoSeries(_local_overlay(areas.values, mask, "intersection"),
                         index=areas.index, crs=areas.crs)


def flooded_fraction(areas: gpd.GeoSeries, flood: gpd.GeoSeries) -> np.ndarray:
    """Share of each area covered by the flood polygons (0 for empty areas)."""
    total = areas.area.values
    wet = shapely.area(_local_overlay(areas.values, flood, "intersection"))
    return np.divide(wet, total, out=np.zeros_like(total), where=total > 0)


def response(fraction, shape: float = SHAPE, scale: float = SCALE) -> np.ndarray:
    """Log-normal CDF of ``scale`` x flooded fraction (the paper's transformation)."""
    return lognorm.cdf(scale * np.asarray(fraction, float), shape)


def hazard_wellbeing(footprints: gpd.GeoSeries, floods: Mapping[int, gpd.GeoSeries],
                     buffer: float = 2.0) -> np.ndarray:
    """HWB: sum over depth thresholds of the response to the flooded fraction of the ring."""
    ring = rings_without_buildings(footprints, buffer)
    return sum(response(flooded_fraction(ring, floods[d])) for d in sorted(floods))


def mobility_fractions(footprints: gpd.GeoSeries, flood: gpd.GeoSeries, streets: gpd.GeoSeries,
                       buffers: Sequence[float] = (5.0, 15.0, 30.0)) -> dict[str, np.ndarray]:
    """Flooded fractions for HMA: the ring at buffers[0], and the street area in the
    0-buffers[1] ring (footprints removed) and the buffers[1]-buffers[2] annulus."""
    near, mid, far = buffers
    f_near = flooded_fraction(rings_without_buildings(footprints, near), flood)
    inner = clip_to(rings_without_buildings(footprints, mid), streets)
    outer = clip_to(footprints.buffer(far).difference(footprints.buffer(mid)), streets)
    return {"f_near": f_near, "f_inner": flooded_fraction(inner, flood),
            "f_outer": flooded_fraction(outer, flood),
            "street_inner_m2": inner.area.values, "street_outer_m2": outer.area.values}


def hazard_mobility(fractions: Mapping[str, np.ndarray]) -> np.ndarray:
    """HMA: response to the largest of the three flooded fractions (0 when all are 0)."""
    fmax = np.maximum.reduce([fractions["f_near"], fractions["f_inner"], fractions["f_outer"]])
    return np.where(fmax > 0, response(fmax), 0.0)
