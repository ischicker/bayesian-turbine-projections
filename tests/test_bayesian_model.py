import arviz as az
import numpy as np
import pandas as pd

from turbine_projections import config
from turbine_projections.bayesian_model import (
    build_logistic_decay_model,
    build_logistic_growth_model,
    compute_derived_capacity,
    fit_model,
    predict_future,
)


def _at_subset() -> pd.DataFrame:
    path = config.PROCESSED_DATA_DIR / "at_clean.csv"
    if path.exists():
        data = pd.read_csv(path).sort_values("year")
        return data.sample(n=min(100, len(data)), random_state=config.RANDOM_SEED)

    rng = np.random.default_rng(config.RANDOM_SEED)
    years = rng.integers(2000, 2026, size=100)
    return pd.DataFrame(
        {
            "year": years,
            "hub_height_m": 60 + 120 / (1 + np.exp(-0.15 * (years - 2014))) + rng.normal(0, 5, 100),
            "rotor_diameter_m": 40 + 110 / (1 + np.exp(-0.15 * (years - 2014))) + rng.normal(0, 5, 100),
            "specific_power_wm2": 250 + 140 / (1 + np.exp(0.2 * (years - 2012))) + rng.normal(0, 10, 100),
            "capacity_kw": 3000,
        }
    )


def test_growth_model_smoke_fit_predicts_future() -> None:
    data = _at_subset()
    priors = config.PRIOR_CONFIG["AT"]["hub_height"]
    model = build_logistic_growth_model(data, priors, "hub_height")
    trace = fit_model(model, {**config.MCMC_SMOKE, "target_accept": 0.9})

    assert isinstance(trace, az.InferenceData)
    median_2030 = float(trace.posterior["L"].median() + trace.posterior["y0"].median())
    assert 50.0 < median_2030 < 300.0

    predictions = predict_future(trace, "growth", config.TARGET_YEARS)
    assert list(predictions.columns) == ["year", "mean", "median", "q2_5", "q97_5", "std"]
    assert predictions["year"].tolist() == config.TARGET_YEARS


def test_compute_derived_capacity_is_positive() -> None:
    data = _at_subset()
    rd_model = build_logistic_growth_model(data, config.PRIOR_CONFIG["AT"]["rotor_diameter"], "rotor_diameter")
    sp_model = build_logistic_decay_model(data, config.PRIOR_CONFIG["AT"]["specific_power"], "specific_power")
    rd_trace = fit_model(rd_model, {**config.MCMC_SMOKE, "target_accept": 0.9})
    sp_trace = fit_model(sp_model, {**config.MCMC_SMOKE, "target_accept": 0.9})

    capacity = compute_derived_capacity(
        {"rotor_diameter": rd_trace, "specific_power": sp_trace},
        target_years=[2030],
    )

    assert (capacity["median"] > 0).all()
