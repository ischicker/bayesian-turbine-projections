"""Bayesian logistic growth and decay models for turbine metrics."""

from __future__ import annotations

import logging
from typing import Literal

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm

from turbine_projections import config

LOGGER = logging.getLogger(__name__)

ModelType = Literal["growth", "decay", "gompertz_growth", "richards_growth"]


def logistic_growth_numpy(t: np.ndarray, L: np.ndarray, k: np.ndarray, t0: np.ndarray, y0: np.ndarray) -> np.ndarray:
    """Evaluate logistic growth for NumPy posterior samples."""

    return y0[:, None] + L[:, None] / (1.0 + np.exp(-k[:, None] * (t[None, :] - t0[:, None])))


def logistic_decay_numpy(
    t: np.ndarray, L: np.ndarray, k: np.ndarray, t0: np.ndarray, y_min: np.ndarray
) -> np.ndarray:
    """Evaluate logistic decay for NumPy posterior samples."""

    return y_min[:, None] + L[:, None] / (1.0 + np.exp(k[:, None] * (t[None, :] - t0[:, None])))


def gompertz_growth_numpy(t: np.ndarray, L: np.ndarray, k: np.ndarray, t0: np.ndarray, y0: np.ndarray) -> np.ndarray:
    """Evaluate Gompertz growth for NumPy posterior samples."""

    return y0[:, None] + L[:, None] * np.exp(-np.exp(-k[:, None] * (t[None, :] - t0[:, None])))


def richards_growth_numpy(
    t: np.ndarray,
    L: np.ndarray,
    k: np.ndarray,
    t0: np.ndarray,
    y0: np.ndarray,
    v: np.ndarray,
) -> np.ndarray:
    """Evaluate generalized Richards growth for NumPy posterior samples."""

    return y0[:, None] + L[:, None] / (1.0 + np.exp(-k[:, None] * (t[None, :] - t0[:, None]))) ** (
        1.0 / v[:, None]
    )


def _create_prior(name: str, prior: dict[str, float]) -> pm.TensorVariable:
    dist = prior["dist"]
    if dist == "Normal":
        return pm.Normal(name, mu=prior["mu"], sigma=prior["sigma"])
    if dist == "HalfNormal":
        return pm.HalfNormal(name, sigma=prior["sigma"])
    if dist == "TruncatedNormal":
        return pm.TruncatedNormal(
            name,
            mu=prior["mu"],
            sigma=prior["sigma"],
            lower=prior.get("lower", -np.inf),
            upper=prior.get("upper", np.inf),
        )
    if dist == "Exponential":
        offset = prior.get("offset", 0.0)
        return pm.Deterministic(name, pm.Exponential(f"{name}_minus_offset", lam=prior["lam"]) + offset)
    if dist == "Uniform":
        return pm.Uniform(name, lower=prior["lower"], upper=prior["upper"])
    raise ValueError(f"Unsupported prior distribution: {dist}")


def _extract_xy(data: pd.DataFrame, metric_name: str) -> tuple[np.ndarray, np.ndarray]:
    metric_column = config.METRIC_COLUMNS.get(metric_name, metric_name)
    if "year" not in data or metric_column not in data:
        raise ValueError(f"Data must contain 'year' and {metric_column!r}.")
    clean = data[["year", metric_column]].dropna()
    return clean["year"].to_numpy(dtype=float), clean[metric_column].to_numpy(dtype=float)


def build_logistic_growth_model(data: pd.DataFrame, priors: dict[str, dict], metric_name: str) -> pm.Model:
    """Build a Bayesian logistic growth model for hub height or rotor diameter."""

    years, values = _extract_xy(data, metric_name)
    with pm.Model(coords={"obs_id": np.arange(len(values))}) as model:
        year = pm.Data("year", years, dims="obs_id")
        L = _create_prior("L", priors["L"])
        k = _create_prior("k", priors["k"])
        t0 = _create_prior("t0", priors["t0"])
        y0 = _create_prior("y0", priors["y0"])
        sigma = _create_prior("sigma", priors["sigma"])
        nu = _create_prior("nu", priors["nu"])
        mu = pm.Deterministic("mu", y0 + L / (1.0 + pm.math.exp(-k * (year - t0))), dims="obs_id")
        pm.StudentT("y_obs", nu=nu, mu=mu, sigma=sigma, observed=values, dims="obs_id")
    return model


def build_gompertz_growth_model(data: pd.DataFrame, priors: dict[str, dict], metric_name: str) -> pm.Model:
    """Build a Bayesian Gompertz growth model for hub height or rotor diameter."""

    years, values = _extract_xy(data, metric_name)
    with pm.Model(coords={"obs_id": np.arange(len(values))}) as model:
        year = pm.Data("year", years, dims="obs_id")
        L = _create_prior("L", priors["L"])
        k = _create_prior("k", priors["k"])
        t0 = _create_prior("t0", priors["t0"])
        y0 = _create_prior("y0", priors["y0"])
        sigma = _create_prior("sigma", priors["sigma"])
        nu = _create_prior("nu", priors["nu"])
        mu = pm.Deterministic("mu", y0 + L * pm.math.exp(-pm.math.exp(-k * (year - t0))), dims="obs_id")
        pm.StudentT("y_obs", nu=nu, mu=mu, sigma=sigma, observed=values, dims="obs_id")
    return model


def build_richards_growth_model(data: pd.DataFrame, priors: dict[str, dict], metric_name: str) -> pm.Model:
    """Build a Bayesian generalized Richards growth model for hub height or rotor diameter."""

    years, values = _extract_xy(data, metric_name)
    with pm.Model(coords={"obs_id": np.arange(len(values))}) as model:
        year = pm.Data("year", years, dims="obs_id")
        L = _create_prior("L", priors["L"])
        k = _create_prior("k", priors["k"])
        t0 = _create_prior("t0", priors["t0"])
        y0 = _create_prior("y0", priors["y0"])
        v = pm.HalfNormal("v", sigma=1.0)
        sigma = _create_prior("sigma", priors["sigma"])
        nu = _create_prior("nu", priors["nu"])
        mu = pm.Deterministic(
            "mu",
            y0 + L / (1.0 + pm.math.exp(-k * (year - t0))) ** (1.0 / v),
            dims="obs_id",
        )
        pm.StudentT("y_obs", nu=nu, mu=mu, sigma=sigma, observed=values, dims="obs_id")
    return model


def build_logistic_decay_model(data: pd.DataFrame, priors: dict[str, dict], metric_name: str) -> pm.Model:
    """Build a Bayesian logistic decay model for specific power."""

    years, values = _extract_xy(data, metric_name)
    with pm.Model(coords={"obs_id": np.arange(len(values))}) as model:
        year = pm.Data("year", years, dims="obs_id")
        L = _create_prior("L", priors["L"])
        k = _create_prior("k", priors["k"])
        t0 = _create_prior("t0", priors["t0"])
        y_min = _create_prior("y_min", priors["y_min"])
        sigma = _create_prior("sigma", priors["sigma"])
        nu = _create_prior("nu", priors["nu"])
        mu = pm.Deterministic("mu", y_min + L / (1.0 + pm.math.exp(k * (year - t0))), dims="obs_id")
        pm.StudentT("y_obs", nu=nu, mu=mu, sigma=sigma, observed=values, dims="obs_id")
    return model


def fit_model(model: pm.Model, mcmc_config: dict[str, int | float | bool]) -> az.InferenceData:
    """Fit a PyMC model using NUTS and return ArviZ inference data."""

    with model:
        trace = pm.sample(
            draws=mcmc_config["draws"],
            tune=mcmc_config["tune"],
            chains=mcmc_config["chains"],
            target_accept=mcmc_config.get("target_accept", config.TARGET_ACCEPT),
            random_seed=config.RANDOM_SEED,
            return_inferencedata=True,
            progressbar=True,
        )
        if mcmc_config.get("compute_log_likelihood", True):
            return pm.compute_log_likelihood(trace)
        return trace


def fit_bayesian_logistic(data: pd.DataFrame, priors: dict[str, dict], metric_name: str, mcmc_config: dict[str, int]):
    """Fit the standard Bayesian logistic growth model."""

    return fit_model(build_logistic_growth_model(data, priors, metric_name), mcmc_config)


def fit_bayesian_gompertz(data: pd.DataFrame, priors: dict[str, dict], metric_name: str, mcmc_config: dict[str, int]):
    """Fit the Bayesian Gompertz growth model."""

    return fit_model(build_gompertz_growth_model(data, priors, metric_name), mcmc_config)


def fit_bayesian_richards(data: pd.DataFrame, priors: dict[str, dict], metric_name: str, mcmc_config: dict[str, int]):
    """Fit the Bayesian generalized Richards growth model."""

    return fit_model(build_richards_growth_model(data, priors, metric_name), mcmc_config)


def _posterior_samples(trace: az.InferenceData, var_name: str) -> np.ndarray:
    return np.asarray(trace.posterior[var_name].values).reshape(-1)


def posterior_predictive_samples(
    trace: az.InferenceData,
    model_type: ModelType,
    years: np.ndarray,
    max_samples: int | None = None,
) -> np.ndarray:
    """Evaluate posterior predictive metric samples for target years."""

    L = _posterior_samples(trace, "L")
    k = _posterior_samples(trace, "k")
    t0 = _posterior_samples(trace, "t0")
    if max_samples is not None and len(L) > max_samples:
        rng = np.random.default_rng(config.RANDOM_SEED)
        idx = np.sort(rng.choice(len(L), size=max_samples, replace=False))
        L, k, t0 = L[idx], k[idx], t0[idx]
    years_float = np.asarray(years, dtype=float)
    if model_type == "growth":
        y0 = _posterior_samples(trace, "y0")
        if len(y0) != len(L):
            y0 = y0[: len(L)]
        return logistic_growth_numpy(years_float, L, k, t0, y0)
    if model_type == "gompertz_growth":
        y0 = _posterior_samples(trace, "y0")
        if len(y0) != len(L):
            y0 = y0[: len(L)]
        return gompertz_growth_numpy(years_float, L, k, t0, y0)
    if model_type == "richards_growth":
        y0 = _posterior_samples(trace, "y0")
        v = _posterior_samples(trace, "v")
        if len(y0) != len(L):
            y0 = y0[: len(L)]
        if len(v) != len(L):
            v = v[: len(L)]
        return richards_growth_numpy(years_float, L, k, t0, y0, v)
    y_min = _posterior_samples(trace, "y_min")
    if len(y_min) != len(L):
        y_min = y_min[: len(L)]
    return logistic_decay_numpy(years_float, L, k, t0, y_min)


def predict_future(
    trace: az.InferenceData,
    model_type: ModelType,
    target_years: list[int] | tuple[int, ...] | np.ndarray = config.TARGET_YEARS,
) -> pd.DataFrame:
    """Summarize posterior predictions for configured future years."""

    years = np.asarray(target_years, dtype=int)
    samples = posterior_predictive_samples(trace, model_type, years)
    return pd.DataFrame(
        {
            "year": years,
            "mean": samples.mean(axis=0),
            "median": np.median(samples, axis=0),
            "q2_5": np.quantile(samples, 0.025, axis=0),
            "q97_5": np.quantile(samples, 0.975, axis=0),
            "std": samples.std(axis=0),
        }
    )


def compute_derived_capacity(
    traces_dict: dict[str, az.InferenceData],
    target_years: list[int] | tuple[int, ...] | np.ndarray = config.TARGET_YEARS,
) -> pd.DataFrame:
    """Compute capacity from rotor-diameter and specific-power posteriors.

    Joint sampling is implemented by taking the same flattened posterior sample
    index from both traces, as requested for this paper's current scope.
    """

    if "rotor_diameter" not in traces_dict or "specific_power" not in traces_dict:
        raise ValueError("traces_dict must contain 'rotor_diameter' and 'specific_power'.")
    years = np.asarray(target_years, dtype=int)
    rd = posterior_predictive_samples(traces_dict["rotor_diameter"], "growth", years)
    sp = posterior_predictive_samples(traces_dict["specific_power"], "decay", years)
    n_samples = min(rd.shape[0], sp.shape[0])
    rd = rd[:n_samples]
    sp = sp[:n_samples]
    capacity_mw = sp * np.pi * (rd / 2.0) ** 2 / 1_000_000.0
    return pd.DataFrame(
        {
            "year": years,
            "metric": "capacity_mw",
            "mean": capacity_mw.mean(axis=0),
            "median": np.median(capacity_mw, axis=0),
            "q2_5": np.quantile(capacity_mw, 0.025, axis=0),
            "q97_5": np.quantile(capacity_mw, 0.975, axis=0),
            "std": capacity_mw.std(axis=0),
        }
    )


def convergence_summary(trace: az.InferenceData) -> dict[str, float]:
    """Return compact convergence diagnostics."""

    var_names = ["L", "k", "t0", "sigma", "nu"]
    if "v" in trace.posterior:
        var_names.append("v")
    summary = az.summary(trace, var_names=var_names, kind="diagnostics")
    max_rhat = float(summary["r_hat"].max(skipna=True))
    min_ess = float(summary["ess_bulk"].min(skipna=True))
    divergences = 0
    if hasattr(trace, "sample_stats") and "diverging" in trace.sample_stats:
        divergences = int(trace.sample_stats["diverging"].sum().item())
    return {"max_rhat": max_rhat, "min_ess": min_ess, "divergences": divergences}


def warn_on_convergence(trace: az.InferenceData, region: str, metric: str) -> dict[str, float]:
    """Log convergence warnings and keep the pipeline moving."""

    diagnostics = convergence_summary(trace)
    if diagnostics["max_rhat"] >= 1.05:
        LOGGER.warning("%s %s max R-hat %.3f >= 1.05", region, metric, diagnostics["max_rhat"])
    if diagnostics["min_ess"] <= 400:
        LOGGER.warning("%s %s min ESS %.0f <= 400", region, metric, diagnostics["min_ess"])
    if diagnostics["divergences"] > 0:
        LOGGER.warning("%s %s divergences: %d", region, metric, diagnostics["divergences"])
    return diagnostics
