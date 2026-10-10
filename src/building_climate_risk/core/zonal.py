"""Zones around building footprints and what covers them — hazard-agnostic geometry.

Rings around a building exclude **every** building footprint, not only the building's
own (as the authors' stored intermediate fractions require for the flood hazard).
Distances are in the units of the input CRS: use a projected CRS in ground metres.

Overlays are local: each zone is differenced/intersected only with the polygons that
touch it (STRtree query). This equals using the union of all of them and scales to
whole cities. City-wide polygons (one street network, one flood extent, one water body
with many thousands of vertices) are first split into tiles (``split_large``), so each
zone meets only small pieces; the pieces cover the original exactly, so results are
unchanged (to floating-point noise) and overlays become orders of magnitude faster.
"""

import geopandas as gpd
import numpy as np
import shapely

MAX_VERTICES = 2000      # polygons with more vertices than this are tiled before overlays
TILE_SIZE = 100.0        # tile edge, in CRS units (metres in a projected CRS)


def tile(geom: shapely.Geometry, size: float = TILE_SIZE) -> np.ndarray:
    """Cut one geometry into the non-empty pieces of a ``size`` grid over its bounds."""
    x0, y0, x1, y1 = geom.bounds
    xs, ys = np.meshgrid(np.arange(x0, x1, size), np.arange(y0, y1, size))
    pieces = shapely.intersection(geom, shapely.box(xs, ys, xs + size, ys + size).ravel())
    return pieces[~shapely.is_empty(pieces)]


def quarter(geom: shapely.Geometry, max_vertices: int = MAX_VERTICES,
            min_size: float = TILE_SIZE) -> list[shapely.Geometry]:
    """Split ``geom`` into quadrants, recursively, until every piece has at most
    ``max_vertices`` vertices or is smaller than ``min_size`` across.

    Each cut works on an ever smaller piece, so this is far faster than cutting the whole
    geometry with a fixed grid; the pieces cover ``geom`` exactly.
    """
    x0, y0, x1, y1 = geom.bounds
    if shapely.get_num_coordinates(geom) <= max_vertices or max(x1 - x0, y1 - y0) <= min_size:
        return [geom]
    xm, ym = (x0 + x1) / 2, (y0 + y1) / 2
    quads = shapely.intersection(geom, shapely.box([x0, xm, x0, xm], [y0, y0, ym, ym],
                                                   [xm, x1, xm, x1], [ym, ym, y1, y1]))
    return [p for q in quads[~shapely.is_empty(quads)] for p in quarter(q, max_vertices, min_size)]


def split_large(geoms: gpd.GeoSeries, max_vertices: int = MAX_VERTICES,
                size: float = TILE_SIZE) -> gpd.GeoSeries:
    """Split the geometries with more than ``max_vertices`` vertices (``quarter``); keep the
    others as they are.

    The pieces of a split geometry cover it exactly, so any union of the pieces a zone
    touches equals the part of the original it touches.
    """
    geoms = geoms[~geoms.is_empty & geoms.notna()]
    heavy = shapely.get_num_coordinates(geoms.values) > max_vertices
    if not heavy.any():
        return geoms
    pieces = [np.array(quarter(g, max_vertices, size), dtype=object) for g in geoms.values[heavy]]
    values = np.concatenate([geoms.values[~heavy], *pieces])
    return gpd.GeoSeries(values, crs=geoms.crs)


def _local_overlay(geoms: np.ndarray, others: gpd.GeoSeries, op: str) -> np.ndarray:
    """Per geometry: difference from / intersection with the union of the ``others`` it touches."""
    others = split_large(others, MAX_VERTICES, TILE_SIZE)     # module settings, read at call time
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
