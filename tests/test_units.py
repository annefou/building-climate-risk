import geopandas as gpd
import numpy as np
import shapely
from shapely.geometry import box

import pfr_building_risk as pfr


def _grid(n: int = 4, size: float = 10.0, gap: float = 3.0) -> gpd.GeoSeries:
    step = size + gap
    return gpd.GeoSeries([box(i * step, j * step, i * step + size, j * step + size)
                          for i in range(n) for j in range(n)], crs="EPSG:25830")


def test_local_rings_equal_global_union():
    fp = _grid()
    local = pfr.rings_without_buildings(fp, 5)
    global_ = fp.buffer(5).difference(shapely.union_all(fp.values))
    assert np.allclose(local.area, global_.area)
    assert all(a.equals(b) for a, b in zip(local, global_))


def test_flooded_fraction_equals_global_union():
    fp = _grid()
    flood = gpd.GeoSeries([box(-5, -5, 20, 60), box(30, 0, 45, 15)], crs=fp.crs)
    ring = pfr.rings_without_buildings(fp, 2)
    got = pfr.flooded_fraction(ring, flood)
    want = ring.intersection(shapely.union_all(flood.values)).area / ring.area
    assert np.allclose(got, want)


def test_no_flood_gives_zero_hazard():
    fp = _grid(2)
    empty = gpd.GeoSeries([box(1000, 1000, 1001, 1001)], crs=fp.crs)
    assert np.all(pfr.hazard_wellbeing(fp, {30: empty, 40: empty}) == 0)
    fr = pfr.mobility_fractions(fp, empty, gpd.GeoSeries([box(-50, -50, 80, 80)], crs=fp.crs))
    assert np.all(pfr.hazard_mobility(fr) == 0)


def test_topsis_entropy_is_two_with_a_zero():
    x = np.array([[1.0, 0.0], [2.0, 1.0], [0.0, 3.0]])
    plain = pfr.topsis(x, [0.5, 0.5], entropy=False)
    with_u = pfr.topsis(x, [0.5, 0.5])
    assert np.isclose(with_u[0], 2 * plain[0]) and np.isclose(with_u[2], 2 * plain[2])


def test_iterated_means_and_classes():
    v = np.array([0, 1, 2, 3, 4, 10])
    breaks = pfr.iterated_means(v)  # 10/3, then mean(4, 10) = 7, then mean(10) = 10
    assert np.allclose(breaks, [10 / 3, 7, 10])
    # Right-closed breaks: a value equal to a break stays in the lower class.
    assert pfr.risk_class(v, breaks).tolist() == [0, 1, 1, 1, 2, 3]
    assert pfr.risk_class([11], breaks).tolist() == [4]
