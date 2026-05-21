# Bayesian Wind Turbine Technology Projections

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

Reproducible analysis pipeline for:

> **A Bayesian Framework for Probabilistic Wind Turbine Technology Projections:
> Multi-Region Validation and Application to Climate-Aware Energy Yield Estimation**
>
> Irene Schicker, Stefan Janisch, and Annemarie Lexer
>
> *Energies* (2026), submitted.

## Overview

This framework models the temporal evolution of onshore wind turbine hub height,
rotor diameter, and specific power as physically constrained logistic
growth/decay processes, producing full posterior predictive distributions via
MCMC sampling. It is validated across three markets:

| Region | Source | N (QC) | Period |
|--------|--------|--------|--------|
| Austria | IGW / UVP | 534 | 2000–2025 |
| Germany | MaStR | 31,202 | 1988–2026 |
| United States | USWTDB v8.3 | 71,457 | 1986–2025 |

The pipeline includes systematic benchmarking against linear, polynomial, and
MLE logistic alternatives, prior sensitivity analysis, and an end-to-end
climate-aware energy yield estimation coupling turbine posteriors with
GOWIRES wind resource projections under SSP2-4.5 and SSP5-8.5.

## Repository Structure

```
├── scripts/                 # Numbered pipeline scripts (run in order)
│   ├── 01_download_data.py  # Placeholder (raw data supplied manually)
│   ├── 02_prepare_data.py   # QC and cleaning
│   ├── 03_run_fits.py       # Bayesian MCMC fits (PyMC)
│   ├── 04_run_benchmarks.py # Hindcast validation
│   ├── 05_run_sensitivity.py# Prior sensitivity analysis
│   ├── 06_run_energy_yield.py# Climate-aware AEP estimation
│   └── 07_generate_figures.py# Publication figures
├── src/turbine_projections/ # Core library
│   ├── bayesian_model.py    # Logistic growth/decay models
│   ├── benchmarks.py        # Linear, quadratic, MLE logistic
│   ├── climate_scenarios.py # GOWIRES interface
│   ├── config.py            # Region/metric configurations & priors
│   ├── data_preparation.py  # QC pipeline
│   ├── energy_yield.py      # Synthetic power curves & AEP
│   ├── hindcast.py          # Train/test split validation
│   └── sensitivity.py       # Prior sensitivity framework
├── notebooks/               # Demonstration notebooks
├── tests/                   # Unit and smoke tests
├── data/
│   ├── raw/                 # Raw data (not tracked, see below)
│   └── processed/           # Cleaned datasets & GOWIRES extracts
├── pyproject.toml           # Dependencies (managed with uv)
└── uv.lock                  # Locked dependency versions
```

## Installation

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/ischicker/bayesian-turbine-projections.git
cd bayesian-turbine-projections
uv sync --all-extras
```

## Running the Pipeline

Execute the numbered scripts in order:

```bash
# 1. Prepare data (requires raw data in data/raw/, see Data section)
uv run python scripts/02_prepare_data.py --region all

# 2. Run Bayesian fits (10 chains × 4000 draws; ~30 min per region)
uv run python scripts/03_run_fits.py --region all

# 3. Benchmark comparison (hindcast validation)
uv run python scripts/04_run_benchmarks.py

# 4. Prior sensitivity analysis (3 prior sets × 9 fits)
uv run python scripts/05_run_sensitivity.py

# 5. Climate-aware energy yield estimation
uv run python scripts/06_run_energy_yield.py

# 6. Generate publication figures
uv run python scripts/07_generate_figures.py
```

Results are written to `results/tables/` and `results/figures/`.

## Data

### Public datasets (download yourself)

- **USWTDB** (US): https://eerscmap.usgs.gov/uswtdb/ — download CSV, place in `data/raw/US/`
- **MaStR** (Germany): https://www.marktstammdatenregister.de/ — export onshore wind, place in `data/raw/DE/`
- **GOWIRES**: https://doi.org/10.5281/zenodo.18768952 — download, place in `data/raw/GOWIRES/`

### Restricted dataset

- **Austrian turbine data** (IGW/UVP): Not publicly available. The cleaned
  dataset (`at_clean.csv`) is available from the authors upon reasonable request
  and with permission of the data providers. Aggregated annual statistics
  (`at_annual.csv`) are included in this repository.

### Included in repository

- `data/processed/at_annual.csv` — Austrian annual aggregates (non-confidential)
- `data/processed/de_annual.csv` — German annual aggregates
- `data/processed/us_annual.csv` — US annual aggregates
- `data/processed/gowires_wind_climate.csv` — GOWIRES Weibull parameters for reference sites
- `data/processed/gowires_reference_neighbors.csv` — GOWIRES neighbor metadata

## Tests

```bash
uv run pytest
```

## Citation

If you use this framework, please cite:

```bibtex
@article{Schicker2026bayesian,
  title   = {A {Bayesian} Framework for Probabilistic Wind Turbine Technology
             Projections: Multi-Region Validation and Application to
             Climate-Aware Energy Yield Estimation},
  author  = {Schicker, Irene and Janisch, Stefan and Lexer, Annemarie},
  journal = {Energies},
  year    = {2026},
  note    = {Submitted}
}
```

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE).

## Acknowledgments

This work was partially supported by the AI4WIND project (Austrian Research
Promotion Agency, FFG) and the Wind4Future project (Austrian Climate Research
Programme, ACRP, grant no. KR21KB0K00001).