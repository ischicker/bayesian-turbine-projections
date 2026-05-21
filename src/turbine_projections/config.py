"""Central configuration for turbine technology projections.

All paths are resolved relative to the repository workspace. Raw data are kept in
the workspace-level ``data/raw`` folder when it exists, while generated project
outputs live inside ``turbine_projections``.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = PACKAGE_ROOT.parent

DATA_DIR = PACKAGE_ROOT / "data"
WORKSPACE_DATA_DIR = WORKSPACE_ROOT / "data"
RAW_DATA_DIR = WORKSPACE_DATA_DIR / "raw" if (WORKSPACE_DATA_DIR / "raw").exists() else DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RESULTS_DIR = PACKAGE_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"
POSTERIORS_DIR = RESULTS_DIR / "posteriors"
SUPPLEMENT_DIR = RESULTS_DIR / "supplement"

REGIONS = ["AT", "DE", "US"]
TARGET_YEARS = [2030, 2055]
METRICS = ["hub_height", "rotor_diameter", "specific_power"]

METRIC_COLUMNS = {
    "hub_height": "hub_height_m",
    "rotor_diameter": "rotor_diameter_m",
    "specific_power": "specific_power_wm2",
}

REGION_COLORS = {
    "AT": "#E63946",
    "DE": "#457B9D",
    "US": "#2A9D8F",
}

METRIC_LABELS = {
    "hub_height_m": "Hub Height [m]",
    "rotor_diameter_m": "Rotor Diameter [m]",
    "specific_power_wm2": "Specific Power [W/m2]",
}

PLAUSIBILITY_LIMITS = {
    "hub_height_m": (20.0, 250.0),
    "rotor_diameter_m": (10.0, 250.0),
    "specific_power_wm2": (100.0, 800.0),
    "capacity_kw": (100.0, float("inf")),
}

EXPECTED_YEAR_RANGES = {
    "AT": (1998, 2028),
    "DE": (1985, 2026),
    "US": (1980, 2026),
}

EXPECTED_TURBINE_COUNTS = {
    "AT": (150, 600),
    "DE": (20_000, 40_000),
    "US": (60_000, 90_000),
}

RANDOM_SEED = 42

PUBLISH_RAW_DATA = False

MCMC_FULL = {"chains": 10, "draws": 4000, "tune": 1000}
MCMC_QUICK = {"chains": 4, "draws": 2000, "tune": 500}
MCMC_SMOKE = {"chains": 2, "draws": 100, "tune": 100}
TARGET_ACCEPT = 0.95
MAX_FIT_SAMPLES = 2000
REGION_MAX_FIT_SAMPLES = {
    "AT": None,
    "DE": 5000,
    "US": 2000,
}

HINDCAST_SPLITS = [2015, 2018]
PRIMARY_HINDCAST_TRAIN_END = 2015

REFERENCE_SITES = {
    "AT": {"lat": 47.5, "lon": 16.5, "name": "Weinviertel"},
    "DE": {"lat": 53.5, "lon": 9.0, "name": "Schleswig-Holstein"},
    "US": {"lat": 41.5, "lon": -99.5, "name": "Nebraska"},
}

WEIBULL_REFERENCE = {
    "AT": {"k": 2.0, "A": 7.5, "height": 100},
    "DE": {"k": 2.1, "A": 8.0, "height": 100},
    "US": {"k": 2.0, "A": 8.5, "height": 100},
}

PRIOR_CONFIG = {
    "AT": {
        "hub_height": {
            "model_type": "growth",
            "L": {"dist": "Normal", "mu": 200.0, "sigma": 40.0},
            "t0": {"dist": "Normal", "mu": 2015.0, "sigma": 5.0},
            "y0": {"dist": "Normal", "mu": 60.0, "sigma": 15.0},
            "k": {"dist": "HalfNormal", "sigma": 0.3},
            "sigma": {"dist": "HalfNormal", "sigma": 30.0},
            "nu": {"dist": "Exponential", "lam": 1 / 30, "offset": 2.0},
        },
        "rotor_diameter": {
            "model_type": "growth",
            "L": {"dist": "Normal", "mu": 200.0, "sigma": 50.0},
            "t0": {"dist": "Normal", "mu": 2015.0, "sigma": 5.0},
            "y0": {"dist": "Normal", "mu": 40.0, "sigma": 15.0},
            "k": {"dist": "HalfNormal", "sigma": 0.3},
            "sigma": {"dist": "HalfNormal", "sigma": 30.0},
            "nu": {"dist": "Exponential", "lam": 1 / 30, "offset": 2.0},
        },
        "specific_power": {
            "model_type": "decay",
            "L": {"dist": "Normal", "mu": 150.0, "sigma": 40.0},
            "t0": {"dist": "Normal", "mu": 2012.0, "sigma": 5.0},
            "y_min": {"dist": "Normal", "mu": 250.0, "sigma": 30.0},
            "k": {"dist": "HalfNormal", "sigma": 0.3},
            "sigma": {"dist": "HalfNormal", "sigma": 50.0},
            "nu": {"dist": "Exponential", "lam": 1 / 30, "offset": 2.0},
        },
    },
    "DE": {
        "hub_height": {
            "model_type": "growth",
            "L": {"dist": "Normal", "mu": 180.0, "sigma": 30.0},
            "t0": {"dist": "Normal", "mu": 2008.0, "sigma": 5.0},
            "y0": {"dist": "Normal", "mu": 30.0, "sigma": 10.0},
            "k": {"dist": "HalfNormal", "sigma": 0.3},
            "sigma": {"dist": "HalfNormal", "sigma": 30.0},
            "nu": {"dist": "Exponential", "lam": 1 / 30, "offset": 2.0},
        },
        "rotor_diameter": {
            "model_type": "growth",
            "L": {"dist": "Normal", "mu": 180.0, "sigma": 40.0},
            "t0": {"dist": "Normal", "mu": 2008.0, "sigma": 5.0},
            "y0": {"dist": "Normal", "mu": 20.0, "sigma": 10.0},
            "k": {"dist": "HalfNormal", "sigma": 0.3},
            "sigma": {"dist": "HalfNormal", "sigma": 30.0},
            "nu": {"dist": "Exponential", "lam": 1 / 30, "offset": 2.0},
        },
        "specific_power": {
            "model_type": "decay",
            "L": {"dist": "Normal", "mu": 150.0, "sigma": 30.0},
            "t0": {"dist": "Normal", "mu": 2012.0, "sigma": 5.0},
            "y_min": {"dist": "Normal", "mu": 250.0, "sigma": 30.0},
            "k": {"dist": "HalfNormal", "sigma": 0.3},
            "sigma": {"dist": "HalfNormal", "sigma": 50.0},
            "nu": {"dist": "Exponential", "lam": 1 / 30, "offset": 2.0},
        },
    },
    "US": {
        "hub_height": {
            "model_type": "growth",
            "L": {"dist": "Normal", "mu": 120.0, "sigma": 30.0},
            "t0": {"dist": "Normal", "mu": 2010.0, "sigma": 8.0},
            "y0": {"dist": "Normal", "mu": 25.0, "sigma": 10.0},
            "k": {"dist": "HalfNormal", "sigma": 0.3},
            "sigma": {"dist": "HalfNormal", "sigma": 30.0},
            "nu": {"dist": "Exponential", "lam": 1 / 30, "offset": 2.0},
        },
        "rotor_diameter": {
            "model_type": "growth",
            "L": {"dist": "Normal", "mu": 170.0, "sigma": 40.0},
            "t0": {"dist": "Normal", "mu": 2008.0, "sigma": 8.0},
            "y0": {"dist": "Normal", "mu": 15.0, "sigma": 10.0},
            "k": {"dist": "HalfNormal", "sigma": 0.3},
            "sigma": {"dist": "HalfNormal", "sigma": 30.0},
            "nu": {"dist": "Exponential", "lam": 1 / 30, "offset": 2.0},
        },
        "specific_power": {
            "model_type": "decay",
            "L": {"dist": "Normal", "mu": 200.0, "sigma": 40.0},
            "t0": {"dist": "Normal", "mu": 2005.0, "sigma": 5.0},
            "y_min": {"dist": "Normal", "mu": 200.0, "sigma": 30.0},
            "k": {"dist": "HalfNormal", "sigma": 0.3},
            "sigma": {"dist": "HalfNormal", "sigma": 50.0},
            "nu": {"dist": "Exponential", "lam": 1 / 30, "offset": 2.0},
        },
    },
}

def _scale_prior_sigmas(priors: dict[str, dict[str, object]], factor: float) -> dict[str, dict[str, object]]:
    scaled = deepcopy(priors)
    for region_config in scaled.values():
        for metric_config in region_config.values():
            for parameter, prior in metric_config.items():
                if parameter == "model_type" or not isinstance(prior, dict):
                    continue
                if "sigma" in prior:
                    prior["sigma"] = float(prior["sigma"]) * factor
    return scaled


def _make_diffuse_uniform_priors(priors: dict[str, dict[str, object]]) -> dict[str, dict[str, object]]:
    diffuse = _scale_prior_sigmas(priors, 4.0)
    for region_config in diffuse.values():
        for metric_name, metric_config in region_config.items():
            if metric_name in {"hub_height", "rotor_diameter"}:
                metric_config["L"] = {"dist": "Uniform", "lower": 1.0, "upper": 350.0}
                metric_config["y0"] = {"dist": "Uniform", "lower": -100.0, "upper": 150.0}
            elif metric_name == "specific_power":
                metric_config["L"] = {"dist": "Uniform", "lower": 1.0, "upper": 500.0}
                metric_config["y_min"] = {"dist": "Uniform", "lower": 100.0, "upper": 450.0}
    return diffuse


PRIOR_SETS = {
    "A_informative": deepcopy(PRIOR_CONFIG),
    "B_weakly_informative": _scale_prior_sigmas(PRIOR_CONFIG, 2.0),
    "C_diffuse_uniform": _make_diffuse_uniform_priors(PRIOR_CONFIG),
}
