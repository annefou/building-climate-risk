# pfr-building-risk

An open, tested Python implementation of the building-level urban pluvial flood risk
framework of Vogelbacher et al. (2026), *A high-resolution framework for urban pluvial
flood risk mapping*, Nat. Hazards Earth Syst. Sci. 26, 2765–2783,
[doi:10.5194/nhess-26-2765-2026](https://doi.org/10.5194/nhess-26-2765-2026).

Risk = social vulnerability × exposure × hazard, per building:

| Module | What it computes |
|---|---|
| `vulnerability` | TOPSIS with Shannon entropy for sensitivity, coping capacity and their combination; flood-sensitivity transformation SV_PF |
| `hazard` | hazard to well-being (flooded fraction of a ring around each building, 30–100 cm) and to mobility (ring + street area within 15 and 30 m, at 30 cm); rings exclude every building footprint |
| `risk` | PFR = SV_PF × E × H; the authors' iterated-mean risk classes |

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
1e-9 and all 37 risk classes to match the authors' own.

## Test data

`tests/data/hamburg_example_v2.gpkg` holds the authors' worked example and output
layer, converted from Vogelbacher, A., von Szombathely, M., Lennartz, M., Poschlod, B.
and Sillmann, J.: Urban Pluvial Flood Risk Toolbox, v2, Zenodo,
[10.5281/zenodo.19860733](https://doi.org/10.5281/zenodo.19860733), CC BY 4.0.
The two CSV files are outputs of the Hamburg reproduction (MIT).

## License

MIT (code). Test data: CC BY 4.0 (authors' example) and MIT (reproduction outputs).
