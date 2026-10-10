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

import building_climate_risk as bcr

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

    out = bcr.social_vulnerability(b)
    out["ID"] = b["ID"].values
    out["R"], out["R_G"] = b["RB"].astype(float).values, b["RBG"].astype(float).values
    out["HWB"] = bcr.hazard_wellbeing(b.geometry, floods, buffer=2)
    out["HMA"] = bcr.hazard_mobility(bcr.mobility_fractions(b.geometry, floods[30], streets))
    out["PFRMA"] = bcr.risk(out["SVPF"], out["R"], out["HMA"])
    out["PFRWB"] = bcr.risk(out["SVPF"], out["R_G"], out["HWB"])
    return out


def test_matches_hamburg_scenario_c(computed):
    expected = pd.read_csv(DATA / "hamburg_expected_scenario_c.csv")
    m = expected.merge(computed, on="ID", suffixes=("_exp", "_got"))
    assert len(m) == 37
    for col in ["Sensitivity", "CopingCapacity", "SVI", "SVPF", "HMA", "HWB", "PFRMA", "PFRWB"]:
        np.testing.assert_allclose(m[f"{col}_got"], m[f"{col}_exp"], rtol=0, atol=1e-9, err_msg=col)


def test_assess_spec_reproduces_hamburg(computed):
    """The hazard-agnostic assess() with the paper's spec gives the same risks."""
    b = gpd.read_file(GPKG, layer="buildings")
    b["R"], b["R_G"] = b["RB"].astype(float), b["RBG"].astype(float)
    spec = {"vulnerability": {"sensitivity": {"columns": ["C", "ES"], "weights": [0.3, 0.7]},
                              "coping": {"columns": ["WR", "EDQ"], "weights": [0.5, 0.5]},
                              "svi_weights": [0.5, 0.5], "threshold": 0.25, "exponent": 2.0},
            "risks": {"PFRWB": {"exposure": "R_G", "hazard": "HWB"},
                      "PFRMA": {"exposure": "R", "hazard": "HMA"}},
            "classes": {"iterations": 3}}
    out = bcr.assess(b, {"HWB": computed["HWB"], "HMA": computed["HMA"]}, spec)
    for col in ["SVPF", "PFRWB", "PFRMA"]:
        np.testing.assert_allclose(out[col], computed[col], rtol=0, atol=1e-12, err_msg=col)


def test_classes_match_authors_37_of_37(computed):
    ref = gpd.read_file(GPKG, layer="results").drop(columns="geometry")
    ref["ID"] = ref["ID"].astype(int)
    m = ref.merge(computed, on="ID")
    for ours, theirs in [("PFRWB", "PFR_WB"), ("PFRMA", "PFR_MA")]:
        got = bcr.risk_class(m[ours], bcr.iterated_means(m[ours]))
        authors = bcr.risk_class(m[theirs], bcr.iterated_means(m[theirs]))
        assert (got == authors).sum() == 37, ours


def test_values_match_authors_arcgis_output(computed):
    """Against the authors' own published output layer (not only our earlier port):
    social vulnerability is exact; hazard and risk agree to the third/fourth decimal, the
    residual being ArcGIS geometry processing (snapping tolerance, true curves)."""
    ref = gpd.read_file(GPKG, layer="results").drop(columns="geometry")
    ref["ID"] = ref["ID"].astype(int)
    m = ref.merge(computed.add_prefix("got_").rename(columns={"got_ID": "ID"}), on="ID")
    assert len(m) == 37
    tolerances = {("Sen", "Sensitivity"): 1e-12, ("CCap", "CopingCapacity"): 1e-12,
                  ("SV", "SVI"): 1e-12, ("SV_PF", "SVPF"): 1e-12,
                  ("HWB", "HWB"): 2e-3, ("HMA", "HMA"): 1e-3,
                  ("PFR_WB", "PFRWB"): 5e-3, ("PFR_MA", "PFRMA"): 2e-2}
    for (theirs, ours), tol in tolerances.items():
        np.testing.assert_allclose(m[f"got_{ours}"], m[theirs].astype(float), rtol=0, atol=tol,
                                   err_msg=f"{ours} vs authors' {theirs}")
