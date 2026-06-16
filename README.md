# Bayesian Wind Turbine Technology Projections

Reproducible analysis pipeline for a journal revision on probabilistic wind
turbine technology projections. The code fits Bayesian logistic technology
models for hub height, rotor diameter, and specific power; evaluates hindcast
benchmarks and sensitivity checks; and propagates technology and wind-climate
uncertainty into capacity factor and annual energy production.

## Environment

The project is managed with `uv` and is constrained to Python `>=3.11,<3.15`.
The committed `uv.lock` file is the pinned environment specification. In the
current Windows test environment (`Python 3.14.5`), the key resolved versions
are:

- `pymc==6.0.0`
- `arviz==1.1.0`
- `numpy==2.4.4`
- `scipy==1.17.1`
- `pandas==3.0.3`
- `matplotlib==3.10.9`

The same `uv.lock` also contains the Python 3.11/`<3.12` resolution branch with
`pymc==5.28.5` and `arviz==0.23.4`, which is the PyMC v5 environment used for
Python 3.11 reproduction.

Install:

```powershell
cd C:\Users\ischicker\Documents\ARBEIT\WINDPROJ_TURBINENKENNZAHLEN_PUBLICATION\turbine_projections
uv sync --all-extras
```

## Data Availability

Raw data are not downloaded by the pipeline. Place files manually under
`data/raw/`:

```text
data/raw/
|-- AT/
|   `-- Windparks_Gegenwart_und_Nahe_Zukunft_FINAL.xlsx
|-- US/
|   `-- uswtdb_*.csv
|-- DE/
|   `-- bnetza_mastr_wind.csv
`-- GOWIRES/
    `-- GOWIRES_V1.csv
```

The US data are open USWTDB data. The German data are open MaStR data. The
Austrian raw dataset is access restricted and is not released publicly; public
outputs use aggregated or anonymized Austrian summaries only. GOWIRES should be
cited via Zenodo DOI `10.5281/zenodo.18768952`, matching the manuscript's Data
Availability statement.

The `PUBLISH_RAW_DATA = False` flag in
`src/turbine_projections/config.py` prevents Austrian cleaned turbine-level data
from being treated as public-release material.

## Determinism

Production seed: `RANDOM_SEED = 42`.

Independent stratified subsample seeds:
`SUBSAMPLE_SEEDS = [42, 123, 456, 789, 1011]`.

Production subsample sizes:

- AT: full dataset
- DE: `N=5000`
- US: `N=5000`

Hindcast splits: train through 2015 for the main text and train through 2018 for
the supplement.

## Reproduce

Run the complete pipeline, including MCMC-heavy fits:

```powershell
uv run python scripts/00_run_all.py
```

Equivalent Make target:

```powershell
make reproduce
```

For a deterministic, lightweight check that skips MCMC-heavy fits:

```powershell
uv run python scripts/00_run_all.py --skip-heavy
```

Expected runtime depends on hardware. On the current workstation, the full
RD/SP multi-subsample revision check took about 73 minutes; full production fits
and sensitivity fits can take several additional hours. All MCMC scripts use
fixed seeds and resume/caching behavior where implemented.

## Main Outputs

- Processed data: `data/processed/`
- Posterior draws: `results/posteriors/`
- Tables: `results/tables/`
- Figures: `results/figures/`
- Revision summary: `REVISION_RESULTS_INTERPRET.md`
- Reproducibility checklist: `REPRODUCIBILITY.md`

## Tests

Run:

```powershell
uv run pytest
```

The current test suite contains 11 tests across data preparation, Bayesian model
helpers, smoke checks, and energy-yield utilities.
