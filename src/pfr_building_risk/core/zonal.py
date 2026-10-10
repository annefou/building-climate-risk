"""Zones around building footprints and what covers them — hazard-agnostic geometry.

Rings around a building exclude **every** building footprint, not only the building's
own (as the authors' stored intermediate fractions require for the flood hazard).
Distances are in the units of the input CRS: use a projected CRS in ground metres.

Overlays are local: each zone is differenced/intersected only with the polygons that
touch it (STRtree query). This equals using the union of all of them and scales to
whole cities.
"""

import geopandas as gpd
import numpy as np
import shapely


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


def annulus(footprints: gpd.GeoSeries, inner: float, outer: float) -> gpd.GeoSeries:
    """The band between ``inner`` and ``outer`` around each footprint (own buffers only)."""
    return footprints.buffer(outer).difference(footprints.buffer(inner))


def clip_to(areas: gpd.GeoSeries, mask: gpd.GeoSeries) -> gpd.GeoSeries:
    """Intersection of each area with the union of the ``mask`` polygons it touches."""
    return gpd.GeoSeries(_local_overlay(areas.values, mask, "intersection"),
                         index=areas.index, crs=areas.crs)


def covered_fraction(areas: gpd.GeoSeries, cover: gpd.GeoSeries) -> np.ndarray:
    """Share of each area covered by the ``cover`` polygons (0 for empty areas)."""
    total = areas.area.values
    covered = shapely.area(_local_overlay(areas.values, cover, "intersection"))
    return np.divide(covered, total, out=np.zeros_like(total), where=total > 0)
