import numpy as np
import pandas as pd

import pfr_building_risk as pfr
from pfr_building_risk.core.assess import assess


def test_allocate_by_weight():
    units = pd.Series({"A": 100.0, "B": 30.0})
    unit_of = pd.Series(["A", "A", "B", "B", None])
    weight = pd.Series([1, 3, 0, 2, 5])
    got = pfr.allocate(units, unit_of, weight)
    assert got.tolist() == [25.0, 75.0, 0.0, 30.0, 0.0]


def test_residents_on_floor_rules():
    r, f = [10.0, 10.0, 10.0], [2, 5, 0]
    assert pfr.residents_on_floor(r, f).tolist() == [5.0, 2.0, 10.0]          # floors < 1 -> 1
    assert pfr.residents_on_floor(r, f, eligible=[True, False, True]).tolist() == [5.0, 0.0, 10.0]
    assert pfr.residents_on_floor(r, f, floor="all").tolist() == r


def test_assess_matches_the_step_by_step_api():
    rng = np.random.default_rng(1)
    b = pd.DataFrame(rng.integers(0, 10, size=(20, 4)), columns=["C", "ES", "WR", "EDQ"])
    b["R"], b["R_G"] = rng.integers(1, 30, 20), rng.integers(0, 5, 20)
    h = {"HWB": rng.random(20) * 8, "HMA": rng.random(20)}
    spec = {"vulnerability": {"sensitivity": {"columns": ["C", "ES"], "weights": [0.3, 0.7]},
                              "coping": {"columns": ["WR", "EDQ"], "weights": [0.5, 0.5]}},
            "risks": {"PFRWB": {"exposure": "R_G", "hazard": "HWB"},
                      "PFRMA": {"exposure": "R", "hazard": "HMA"}}}
    out = assess(b, h, spec)
    sv = pfr.social_vulnerability(b)
    assert np.allclose(out["SVPF"], sv["SVPF"])
    assert np.allclose(out["PFRWB"], sv["SVPF"] * b["R_G"] * h["HWB"])
    assert out["PFRMA_class"].tolist() == pfr.risk_class(out["PFRMA"], pfr.iterated_means(out["PFRMA"])).tolist()
