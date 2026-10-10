import geopandas as gpd
import numpy as np
import shapely
from shapely.geometry import box

import building_climate_risk as bcr


def _grid(n: int = 4, size: float = 10.0, gap: float = 3.0) -> gpd.GeoSeries:
    step = size + gap
    return gpd.GeoSeries([box(i * step, j * step, i * step + size, j * step + size)
                          for i in range(n) for j in range(n)], crs="EPSG:25830")


def test_local_rings_equal_global_union():
    fp = _grid()
    local = bcr.rings_without_buildings(fp, 5)
    global_ = fp.buffer(5).difference(shapely.union_all(fp.values))
    assert np.allclose(local.area, global_.area)
    assert all(a.equals(b) for a, b in zip(local, global_))


def test_flooded_fraction_equals_global_union():
    fp = _grid()
    flood = gpd.GeoSeries([box(-5, -5, 20, 60), box(30, 0, 45, 15)], crs=fp.crs)
    ring = bcr.rings_without_buildings(fp, 2)
    got = bcr.flooded_fraction(ring, flood)
    want = ring.intersection(shapely.union_all(flood.values)).area / ring.area
    assert np.allclose(got, want)


def test_no_flood_gives_zero_hazard():
    fp = _grid(2)
    empty = gpd.GeoSeries([box(1000, 1000, 1001, 1001)], crs=fp.crs)
    assert np.all(bcr.hazard_wellbeing(fp, {30: empty, 40: empty}) == 0)
    fr = bcr.mobility_fractions(fp, empty, gpd.GeoSeries([box(-50, -50, 80, 80)], crs=fp.crs))
    assert np.all(bcr.hazard_mobility(fr) == 0)


def test_topsis_entropy_is_two_with_a_zero():
    x = np.array([[1.0, 0.0], [2.0, 1.0], [0.0, 3.0]])
    plain = bcr.topsis(x, [0.5, 0.5], entropy=False)
    with_u = bcr.topsis(x, [0.5, 0.5])
    assert np.isclose(with_u[0], 2 * plain[0]) and np.isclose(with_u[2], 2 * plain[2])


def test_iterated_means_and_classes():
    v = np.array([0, 1, 2, 3, 4, 10])
    breaks = bcr.iterated_means(v)  # 10/3, then mean(4, 10) = 7, then mean(10) = 10
    assert np.allclose(breaks, [10 / 3, 7, 10])
    # Right-closed breaks: a value equal to a break stays in the lower class.
    assert bcr.risk_class(v, breaks).tolist() == [0, 1, 1, 1, 2, 3]
    assert bcr.risk_class([11], breaks).tolist() == [4]


def _city_wide_polygon() -> gpd.GeoSeries:
    """One jagged polygon with many thousands of vertices, like a city's street network."""
    t = np.linspace(0, 2 * np.pi, 6000, endpoint=False)
    r = 400 + 40 * np.sin(37 * t) + 15 * np.cos(91 * t)
    ring = np.column_stack([500 + r * np.cos(t), 500 + r * np.sin(t)])
    return gpd.GeoSeries([shapely.Polygon(ring)], crs="EPSG:25830")


def test_tile_preserves_area():
    g = _city_wide_polygon().iloc[0]
    pieces = bcr.tile(g, 100.0)
    assert len(pieces) > 1
    assert np.isclose(shapely.area(pieces).sum(), g.area, rtol=1e-12)


def test_split_large_leaves_small_geometries_alone():
    small = _grid(2)
    assert bcr.split_large(small).equals(small)


def test_tiled_overlays_equal_untiled(monkeypatch):
    from building_climate_risk.core import zonal
    cover = _city_wide_polygon()
    assert shapely.get_num_coordinates(cover.values)[0] > zonal.MAX_VERTICES   # tiling applies
    zones = gpd.GeoSeries([shapely.Point(x, y).buffer(30) for x in range(100, 900, 70)
                           for y in range(100, 900, 70)], crs=cover.crs)
    tiled_frac = bcr.covered_fraction(zones, cover)
    tiled_clip = bcr.clip_to(zones, cover).area.values
    monkeypatch.setattr(zonal, "MAX_VERTICES", 10 ** 9)          # disable tiling
    plain_frac = bcr.covered_fraction(zones, cover)
    plain_clip = bcr.clip_to(zones, cover).area.values
    assert np.allclose(tiled_frac, plain_frac, rtol=0, atol=1e-9)
    assert np.allclose(tiled_clip, plain_clip, rtol=0, atol=1e-6)
