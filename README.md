# pfr-building-risk

An open, tested Python implementation of the building-level urban pluvial flood risk
framework of Vogelbacher et al. (2026), *A high-resolution framework for urban pluvial
flood risk mapping*, Nat. Hazards Earth Syst. Sci. 26, 2765–2783,
[doi:10.5194/nhess-26-2765-2026](https://doi.org/10.5194/nhess-26-2765-2026).

Risk = social vulnerability × exposure × hazard, per building. The core is
**hazard-agnostic**; each hazard is a module. Pluvial flooding is the first; the paper
notes the framework "can be adapted to other hazards, such as heat" — a heat module
plugs into the same core.

| Module | What it computes |
|---|---|
| `core.vulnerability` | TOPSIS with Shannon entropy for sensitivity, coping capacity and their combination, for any indicators and weights; hazard-sensitivity transformation (the paper's SV_PF) |
| `core.exposure` | people per building from a coarser unit (e.g. census section), by dwellings or living area; people on the ground / top / all floors, with an eligibility rule (e.g. mixed use) |
| `core.zonal` | rings around footprints that exclude every building footprint; bands; clipping; covered fractions |
| `core.response` | response functions: the paper's log-normal CDF; threshold exceedance |
| `core.risk` | risk = SV × E × H; the authors' iterated-mean risk classes |
| `core.assess` | everything above from one spec (a dict, e.g. kept as YAML), for any number of risks |
| `hazards.pluvial` | hazard to well-being (flooded fraction of a ring around each building, 30–100 cm) and to mobility (ring + street area within 15 and 30 m, at 30 cm) |

The code was written from the paper and the authors' exported scripts, in the Hamburg
reproduction [annefou/hamburg-pluvial-flood-replication](https://github.com/annefou/hamburg-pluvial-flood-replication)
(v0.1.1, [10.5281/zenodo.23270558](https://doi.org/10.5281/zenodo.23270558)), where it
put every building of the authors' example in the same risk class as their ArcGIS
results. Where the paper's text and the authors' results differ, it follows the
results: sensitivity weights children 0.3 / elderly singles 0.7, TOPSIS vector
normalisation, entropy in all three TOPSIS steps, rings excluding all footprints,
three iterations for the classes.

Distances are in the units of the input CRS — use a projected CRS in ground metres.
Ring and flood overlays are local (each ring meets only the footprints and flood
polygons that touch it), so the code scales from a city quarter to a whole city.

## Install and test

```bash
pip install git+https://github.com/annefou/pfr-building-risk@<tag>
# or, for development:
pixi run test
```

The regression test (`tests/test_hamburg_regression.py`) re-runs the authors'
37-building example and requires every index to match the Hamburg reproduction to
1e-9 and all 37 risk classes to match the authors' own — both step by step and through
`assess()` with the paper's spec.

## Adding a hazard

A hazard module turns hazard data into one 0–1 (or summed) index per building, using
`core.zonal` for the zones around buildings and `core.response` for the response. Then
`assess()` combines it with vulnerability and exposure: pass the hazard's own
vulnerability indicators and weights, and the exposure column that hazard affects (e.g.
ground-floor residents for flooding, top-floor residents for heat).

## Test data

`tests/data/hamburg_example_v2.gpkg` holds the authors' worked example and output
layer, converted from Vogelbacher, A., von Szombathely, M., Lennartz, M., Poschlod, B.
and Sillmann, J.: Urban Pluvial Flood Risk Toolbox, v2, Zenodo,
[10.5281/zenodo.19860733](https://doi.org/10.5281/zenodo.19860733), CC BY 4.0.
The two CSV files are outputs of the Hamburg reproduction (MIT).

## License

MIT (code). Test data: CC BY 4.0 (authors' example) and MIT (reproduction outputs).
