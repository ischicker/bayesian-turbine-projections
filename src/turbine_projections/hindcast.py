"""Hindcast validation for configured train/test splits."""

from __future__ import annotations

import logging

import arviz as az
import numpy as np
import pandas as pd

from turbine_projections import config
from turbine_projections.benchmarks import (
    bayesian_predictive_samples,
    crps_ensemble,
    fit_linear,
    fit_mle_logistic,
    fit_quadratic,
    information_criteria,
    mae,
    rmse,
)
from turbine_projections.bayesian_model import posterior_predictive_samples

LOGGER = logging.getLogger(__name__)

MODEL_ORDER = ["linear", "quadratic", "mle_logistic", "bayesian_logistic"]


def load_region_data(region: str) -> pd.DataFrame:
    path = config.PROCESSED_DATA_DIR / f"{region.lower()}_clean.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing processed data: {path}")
    return pd.read_csv(path)


def split_train_test(data: pd.DataFrame, split_year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = data.loc[data["year"] <= split_year].copy()
    test = data.loc[(data["year"] > split_year) & (data["year"] <= 2025)].copy()
    return train, test


def _annual_medians(data: pd.DataFrame, metric: str) -> pd.DataFrame:
    column = config.METRIC_COLUMNS[metric]
    return (
        data.groupby("year", as_index=False)[column]
        .median()
        .rename(columns={column: "observed_median"})
        .sort_values("year")
    )


def evaluate_classical_models(region: str, metric: str, data: pd.DataFrame, split_year: int) -> tuple[list[dict], pd.DataFrame]:
    train, test = split_train_test(data, split_year)
    column = config.METRIC_COLUMNS[metric]
    models = [
        fit_linear(train, metric),
        fit_quadratic(train, metric),
        fit_mle_logistic(train, metric, region),
    ]
    rows: list[dict] = []
    prediction_frames: list[pd.DataFrame] = []
    test_years = test["year"].to_numpy(dtype=float)
    for fitted in models:
        pred = fitted.predict(test_years)
        rows.append(
            {
                "region": region,
                "metric": metric,
                "split_year": split_year,
                "model": fitted.model_name,
                "n_train": len(train),
                "n_test": len(test),
                "rmse": rmse(test[column].to_numpy(dtype=float), pred),
                "mae": mae(test[column].to_numpy(dtype=float), pred),
                "crps": np.nan,
                "coverage_95": np.nan,
                "interval_width_95": np.nan,
                "waic": np.nan,
                "loo": np.nan,
                "ic_n_obs": np.nan,
            }
        )
        annual = _annual_medians(test, metric)
        annual["model"] = fitted.model_name
        annual["prediction"] = fitted.predict(annual["year"].to_numpy(dtype=float))
        annual["region"] = region
        annual["metric"] = metric
        annual["split_year"] = split_year
        prediction_frames.append(annual)
    return rows, pd.concat(prediction_frames, ignore_index=True)


def evaluate_bayesian_model(
    region: str,
    metric: str,
    data: pd.DataFrame,
    split_year: int,
    information_criteria_cache: dict[tuple[str, str], dict[str, float]],
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Evaluate saved Bayesian posterior against a hindcast test window."""

    _, test = split_train_test(data, split_year)
    column = config.METRIC_COLUMNS[metric]
    posterior_path = config.POSTERIORS_DIR / f"{region.lower()}_{metric}.nc"
    if not posterior_path.exists():
        raise FileNotFoundError(f"Missing posterior: {posterior_path}")
    trace = az.from_netcdf(posterior_path)
    test_years = test["year"].to_numpy(dtype=float)
    predictive = bayesian_predictive_samples(trace, region, metric, test_years, max_samples=1000)
    median = np.median(predictive, axis=0)
    lower = np.quantile(predictive, 0.025, axis=0)
    upper = np.quantile(predictive, 0.975, axis=0)
    y_test = test[column].to_numpy(dtype=float)
    ic_key = (region, metric)
    if ic_key not in information_criteria_cache:
        information_criteria_cache[ic_key] = information_criteria(trace, region, metric, data)
    ic = information_criteria_cache[ic_key]
    row = {
        "region": region,
        "metric": metric,
        "split_year": split_year,
        "model": "bayesian_logistic",
        "n_train": int((data["year"] <= split_year).sum()),
        "n_test": len(test),
        "rmse": rmse(y_test, median),
        "mae": mae(y_test, median),
        "crps": crps_ensemble(y_test, predictive),
        "coverage_95": float(np.mean((y_test >= lower) & (y_test <= upper))),
        "interval_width_95": float(np.mean(upper - lower)),
        "waic": ic.get("waic", np.nan),
        "loo": ic.get("loo", np.nan),
        "ic_n_obs": ic.get("ic_n_obs", np.nan),
    }

    annual = _annual_medians(test, metric)
    latent = posterior_predictive_samples(
        trace,
        config.PRIOR_CONFIG[region][metric]["model_type"],
        annual["year"].to_numpy(dtype=float),
        max_samples=1000,
    )
    annual["model"] = "bayesian_logistic"
    annual["prediction"] = np.median(latent, axis=0)
    annual["q2_5"] = np.quantile(latent, 0.025, axis=0)
    annual["q97_5"] = np.quantile(latent, 0.975, axis=0)
    annual["region"] = region
    annual["metric"] = metric
    annual["split_year"] = split_year

    calibration = []
    for nominal in [0.5, 0.8, 0.9, 0.95]:
        alpha = (1.0 - nominal) / 2.0
        lo = np.quantile(predictive, alpha, axis=0)
        hi = np.quantile(predictive, 1.0 - alpha, axis=0)
        calibration.append(
            {
                "region": region,
                "metric": metric,
                "split_year": split_year,
                "nominal_coverage": nominal,
                "empirical_coverage": float(np.mean((y_test >= lo) & (y_test <= hi))),
            }
        )
    return row, annual, pd.DataFrame(calibration)


def run_hindcast_benchmarks(splits: list[int] | tuple[int, ...] = config.HINDCAST_SPLITS) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run all configured benchmark models for all regions, metrics, and splits."""

    rows: list[dict] = []
    predictions: list[pd.DataFrame] = []
    calibration: list[pd.DataFrame] = []
    ic_cache: dict[tuple[str, str], dict[str, float]] = {}
    for region in config.REGIONS:
        data = load_region_data(region)
        for metric in config.METRICS:
            for split_year in splits:
                LOGGER.info("Benchmarking %s %s split=%d", region, metric, split_year)
                classical_rows, classical_predictions = evaluate_classical_models(region, metric, data, split_year)
                rows.extend(classical_rows)
                predictions.append(classical_predictions)
                bayes_row, bayes_predictions, bayes_calibration = evaluate_bayesian_model(
                    region, metric, data, split_year, ic_cache
                )
                rows.append(bayes_row)
                predictions.append(bayes_predictions)
                calibration.append(bayes_calibration)
    return (
        pd.DataFrame(rows),
        pd.concat(predictions, ignore_index=True),
        pd.concat(calibration, ignore_index=True),
    )
