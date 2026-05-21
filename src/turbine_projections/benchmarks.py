"""Benchmark models for turbine metric trend extrapolation."""

from __future__ import annotations

from dataclasses import dataclass
import logging

import arviz as az
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.special import gammaln, logsumexp

from turbine_projections import config
from turbine_projections.bayesian_model import posterior_predictive_samples

LOGGER = logging.getLogger(__name__)


@dataclass
class FittedBenchmark:
    """Small container for deterministic benchmark models."""

    model_name: str
    predict: callable


def linear_function(year: np.ndarray, intercept: float, slope: float) -> np.ndarray:
    """Linear trend using centered years supplied by the caller."""

    return intercept + slope * year


def quadratic_function(year: np.ndarray, intercept: float, slope: float, curvature: float) -> np.ndarray:
    """Quadratic trend using centered years supplied by the caller."""

    return intercept + slope * year + curvature * year**2


def logistic_growth(year: np.ndarray, L: float, k: float, t0: float, y0: float) -> np.ndarray:
    return y0 + L / (1.0 + np.exp(-k * (year - t0)))


def logistic_decay(year: np.ndarray, L: float, k: float, t0: float, y_min: float) -> np.ndarray:
    return y_min + L / (1.0 + np.exp(k * (year - t0)))


def fit_linear(train: pd.DataFrame, metric: str) -> FittedBenchmark:
    """Fit y = a + b * t on raw turbine observations."""

    column = config.METRIC_COLUMNS[metric]
    center = float(train["year"].median())
    x = train["year"].to_numpy(dtype=float) - center
    y = train[column].to_numpy(dtype=float)
    slope, intercept = np.polyfit(x, y, deg=1)
    return FittedBenchmark("linear", lambda years: intercept + slope * (np.asarray(years, dtype=float) - center))


def fit_quadratic(train: pd.DataFrame, metric: str) -> FittedBenchmark:
    """Fit y = a + b * t + c * t^2 on raw turbine observations."""

    column = config.METRIC_COLUMNS[metric]
    center = float(train["year"].median())
    x = train["year"].to_numpy(dtype=float) - center
    y = train[column].to_numpy(dtype=float)
    curvature, slope, intercept = np.polyfit(x, y, deg=2)
    return FittedBenchmark(
        "quadratic",
        lambda years: intercept
        + slope * (np.asarray(years, dtype=float) - center)
        + curvature * (np.asarray(years, dtype=float) - center) ** 2,
    )


def fit_mle_logistic(train: pd.DataFrame, metric: str, region: str) -> FittedBenchmark:
    """Fit a prior-free logistic curve with scipy.optimize.curve_fit."""

    column = config.METRIC_COLUMNS[metric]
    model_type = config.PRIOR_CONFIG[region][metric]["model_type"]
    years = train["year"].to_numpy(dtype=float)
    values = train[column].to_numpy(dtype=float)
    function = logistic_growth if model_type == "growth" else logistic_decay
    value_min = float(np.nanpercentile(values, 5))
    value_max = float(np.nanpercentile(values, 95))
    span = max(value_max - value_min, 1.0)
    if model_type == "growth":
        p0 = [span, 0.08, float(np.median(years)), value_min]
        bounds = ([1e-3, 1e-4, 1950.0, -200.0], [600.0, 1.0, 2075.0, 400.0])
    else:
        p0 = [span, 0.08, float(np.median(years)), value_min]
        bounds = ([1e-3, 1e-4, 1950.0, 0.0], [1000.0, 1.0, 2075.0, 800.0])
    try:
        params, _ = curve_fit(function, years, values, p0=p0, bounds=bounds, maxfev=100_000)
    except (RuntimeError, ValueError) as exc:
        LOGGER.warning("MLE logistic failed for %s %s, falling back to initial values: %s", region, metric, exc)
        params = np.asarray(p0)
    return FittedBenchmark("mle_logistic", lambda target_years: function(np.asarray(target_years, dtype=float), *params))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def crps_ensemble(observations: np.ndarray, samples: np.ndarray) -> float:
    """Compute mean CRPS for ensemble forecasts.

    ``samples`` has shape n_samples x n_observations.
    """

    observations = np.asarray(observations, dtype=float)
    samples = np.asarray(samples, dtype=float)
    term1 = np.mean(np.abs(samples - observations[None, :]), axis=0)
    sorted_samples = np.sort(samples, axis=0)
    n_samples = sorted_samples.shape[0]
    weights = (2 * np.arange(1, n_samples + 1) - n_samples - 1).reshape(-1, 1)
    term2 = np.sum(weights * sorted_samples, axis=0) / (n_samples**2)
    return float(np.mean(term1 - term2))


def _posterior_values(trace: az.InferenceData, var_name: str) -> np.ndarray:
    return np.asarray(trace.posterior[var_name].values).reshape(-1)


def bayesian_predictive_samples(
    trace: az.InferenceData,
    region: str,
    metric: str,
    years: np.ndarray,
    max_samples: int = 1000,
    include_observation_noise: bool = True,
) -> np.ndarray:
    """Draw Bayesian predictive samples from a saved posterior."""

    model_type = config.PRIOR_CONFIG[region][metric]["model_type"]
    latent = posterior_predictive_samples(trace, model_type, np.asarray(years, dtype=float), max_samples=max_samples)
    if not include_observation_noise:
        return latent
    sigma = _posterior_values(trace, "sigma")[: latent.shape[0]]
    nu = _posterior_values(trace, "nu")[: latent.shape[0]]
    rng = np.random.default_rng(config.RANDOM_SEED)
    noise = rng.standard_t(df=nu[:, None], size=latent.shape) * sigma[:, None]
    return latent + noise


def student_t_logpdf(y: np.ndarray, mu: np.ndarray, sigma: np.ndarray, nu: np.ndarray) -> np.ndarray:
    """Vectorized Student-t log-density for posterior sample x observation arrays."""

    y = y[None, :]
    z = (y - mu) / sigma[:, None]
    return (
        gammaln((nu[:, None] + 1.0) / 2.0)
        - gammaln(nu[:, None] / 2.0)
        - 0.5 * np.log(nu[:, None] * np.pi)
        - np.log(sigma[:, None])
        - ((nu[:, None] + 1.0) / 2.0) * np.log1p((z**2) / nu[:, None])
    )


def information_criteria(
    trace: az.InferenceData,
    region: str,
    metric: str,
    data: pd.DataFrame,
    max_obs: int = 2000,
    max_posterior_samples: int = 1000,
) -> dict[str, float]:
    """Approximate WAIC and PSIS-LOO on deterministic subsamples."""

    column = config.METRIC_COLUMNS[metric]
    eval_data = data[["year", column]].dropna()
    if len(eval_data) > max_obs:
        eval_data = eval_data.sample(n=max_obs, random_state=config.RANDOM_SEED).sort_values("year")
    years = eval_data["year"].to_numpy(dtype=float)
    y = eval_data[column].to_numpy(dtype=float)
    model_type = config.PRIOR_CONFIG[region][metric]["model_type"]
    mu = posterior_predictive_samples(trace, model_type, years, max_samples=max_posterior_samples)
    n_samples = mu.shape[0]
    sigma = _posterior_values(trace, "sigma")[:n_samples]
    nu = _posterior_values(trace, "nu")[:n_samples]
    log_likelihood = student_t_logpdf(y, mu, sigma, nu)
    result = {"ic_n_obs": float(len(y)), "ic_n_posterior_samples": float(log_likelihood.shape[0])}
    lppd_i = logsumexp(log_likelihood, axis=0) - np.log(log_likelihood.shape[0])
    p_waic_i = np.var(log_likelihood, axis=0, ddof=1)
    result["waic"] = float(np.sum(lppd_i - p_waic_i))
    result["p_waic"] = float(np.sum(p_waic_i))
    # Pareto-smoothed LOO is not exposed for bare arrays by ArviZ' DataTree API
    # in this environment. This is the standard raw importance-sampling LOO
    # identity and is kept in the same elpd scale as WAIC.
    result["loo"] = float(np.sum(-logsumexp(-log_likelihood, axis=0) + np.log(log_likelihood.shape[0])))
    result["p_loo"] = float(np.sum(lppd_i) - result["loo"])
    return result
