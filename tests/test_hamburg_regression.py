"""Regression: reproduce the Hamburg reproduction's scenario C on the authors' example.

Fixtures (tests/data/):
- hamburg_example_v2.gpkg — the authors' worked example (37 buildings, flood depths
  30-100 cm, streets, and their ArcGIS output layer `results`), from Vogelbacher et al.,
  Urban Pluvial Flood Risk Toolbox v2, Zenodo 10.5281/zenodo.19860733, CC BY 4.0.
- hamburg_expected_scenario_c.csv / hamburg_expected_classes.csv — outputs of
  annefou/hamburg-pluvial-flood-replication v0.1.1 (10.5281/zenodo.23270558),
  notebooks/03_analysis.py scenario C.
"""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest

import pfr_building_risk as pfr

DATA = Path(__file__).parent / "data"
GPKG = DATA / "hamburg_example_v2.gpkg"
CRS = "EPSG:3857"  # the CRS the authors stored their data in
DEPTHS = range(30, 101, 10)


@pytest.fixture(scope="module")
def computed() -> pd.DataFrame:
    b = gpd.read_file(GPKG, layer="buildings").to_crs(CRS)
    b["ID"] = b["ID"].astype(int)
    floods = {d: gpd.read_file(GPKG, layer=f"flood_{d}").to_crs(CRS).geometry for d in DEPTHS}
    streets = gpd.read_file(GPKG, layer="streets").to_crs(CRS).geometry

    out = pfr.social_vulnerability(b)
    out["ID"] = b["ID"].values
    out["R"], out["R_G"] = b["RB"].astype(float).values, b["RBG"].astype(float).values
    out["HWB"] = pfr.hazard_wellbeing(b.geometry, floods, buffer=2)
    out["HMA"] = pfr.hazard_mobility(pfr.mobility_fractions(b.geometry, floods[30], streets))
    out["PFRMA"] = pfr.risk(out["SVPF"], out["R"], out["HMA"])
    out["PFRWB"] = pfr.risk(out["SVPF"], out["R_G"], out["HWB"])
    return out


def test_matches_hamburg_scenario_c(computed):
    expected = pd.read_csv(DATA / "hamburg_expected_scenario_c.csv")
    m = expected.merge(computed, on="ID", suffixes=("_exp", "_got"))
    assert len(m) == 37
    for col in ["Sensitivity", "CopingCapacity", "SVI", "SVPF", "HMA", "HWB", "PFRMA", "PFRWB"]:
        np.testing.assert_allclose(m[f"{col}_got"], m[f"{col}_exp"], rtol=0, atol=1e-9, err_msg=col)


def test_classes_match_authors_37_of_37(computed):
    ref = gpd.read_file(GPKG, layer="results").drop(columns="geometry")
    ref["ID"] = ref["ID"].astype(int)
    m = ref.merge(computed, on="ID")
    for ours, theirs in [("PFRWB", "PFR_WB"), ("PFRMA", "PFR_MA")]:
        got = pfr.risk_class(m[ours], pfr.iterated_means(m[ours]))
        authors = pfr.risk_class(m[theirs], pfr.iterated_means(m[theirs]))
        assert (got == authors).sum() == 37, ours
