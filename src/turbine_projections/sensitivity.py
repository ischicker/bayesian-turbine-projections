"""Prior sensitivity analysis across configured prior sets."""

from __future__ import annotations

import logging
import time

import arviz as az
import numpy as np
import pandas as pd

from turbine_projections import config
from turbine_projections.bayesian_model import (
    build_logistic_decay_model,
    build_logistic_growth_model,
    convergence_summary,
    fit_model,
    predict_future,
    warn_on_convergence,
)

LOGGER = logging.getLogger(__name__)

SENSITIVITY_YEARS = [2030, 2055]
PRIOR_SET_LABELS = {
    "A_informative": "A: informative",
    "B_weakly_informative": "B: weakly informative",
    "C_diffuse_uniform": "C: diffuse + uniform limits",
}


def load_region_data(region: str) -> pd.DataFrame:
    path = config.PROCESSED_DATA_DIR / f"{region.lower()}_clean.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing processed data: {path}")
    return pd.read_csv(path)


def stratified_year_subsample(data: pd.DataFrame, max_samples: int = config.MAX_FIT_SAMPLES) -> pd.DataFrame:
    if len(data) <= max_samples:
        return data.copy()
    rng = np.random.default_rng(config.RANDOM_SEED)
    years = np.array(sorted(data["year"].dropna().unique()))
    base = max(1, max_samples // len(years))
    remainder = max_samples - base * len(years)
    parts = []
    for index, year in enumerate(years):
        group = data.loc[data["year"] == year]
        n_year = base + (1 if index < remainder else 0)
        replace = len(group) < n_year
        chosen = rng.choice(group.index.to_numpy(), size=n_year, replace=replace)
        parts.append(data.loc[chosen])
    return pd.concat(parts, ignore_index=True).sample(frac=1.0, random_state=config.RANDOM_SEED).reset_index(drop=True)


def build_model(region: str, metric: str, prior_set: str, data: pd.DataFrame):
    priors = config.PRIOR_SETS[prior_set][region][metric]
    if priors["model_type"] == "growth":
        return build_logistic_growth_model(data, priors, metric)
    return build_logistic_decay_model(data, priors, metric)


def sensitivity_posterior_path(prior_set: str, region: str, metric: str):
    return config.POSTERIORS_DIR / f"sensitivity_{prior_set}_{region.lower()}_{metric}.nc"


def summarize_fit(trace: az.InferenceData, prior_set: str, region: str, metric: str, diagnostics: dict[str, float]) -> pd.DataFrame:
    model_type = config.PRIOR_SETS[prior_set][region][metric]["model_type"]
    pred = predict_future(trace, model_type, SENSITIVITY_YEARS)
    pred.insert(0, "prior_label", PRIOR_SET_LABELS[prior_set])
    pred.insert(0, "prior_set", prior_set)
    pred.insert(0, "metric", metric)
    pred.insert(0, "region", region)
    pred["max_rhat"] = diagnostics["max_rhat"]
    pred["min_ess"] = diagnostics["min_ess"]
    pred["divergences"] = diagnostics["divergences"]
    return pred


def run_prior_sensitivity(force: bool = False) -> pd.DataFrame:
    """Fit all prior sets across configured regions and metrics."""

    mcmc_config = dict(config.MCMC_QUICK)
    mcmc_config["target_accept"] = config.TARGET_ACCEPT
    outputs = []
    for region in config.REGIONS:
        data = load_region_data(region)
        fit_data = stratified_year_subsample(data)
        if len(fit_data) < len(data):
            LOGGER.info("Subsampled %s from %d to %d for prior sensitivity", region, len(data), len(fit_data))
        for metric in config.METRICS:
            for prior_set in config.PRIOR_SETS:
                posterior_path = sensitivity_posterior_path(prior_set, region, metric)
                start = time.perf_counter()
                if posterior_path.exists() and not force:
                    LOGGER.info("Skipping %s %s %s - posterior already exists", prior_set, region, metric)
                    trace = az.from_netcdf(posterior_path)
                    diagnostics = convergence_summary(trace)
                    elapsed = 0.0
                else:
                    LOGGER.info("Starting prior sensitivity fit: %s %s %s (%d rows)", prior_set, region, metric, len(fit_data))
                    model = build_model(region, metric, prior_set, fit_data)
                    trace = fit_model(model, mcmc_config)
                    elapsed = time.perf_counter() - start
                    diagnostics = warn_on_convergence(trace, region, f"{metric}_{prior_set}")
                    trace.to_netcdf(posterior_path)
                LOGGER.info(
                    "Finished sensitivity fit %s %s %s in %.1fs | R-hat %.3f | ESS %.0f | divergences %d",
                    prior_set,
                    region,
                    metric,
                    elapsed,
                    diagnostics["max_rhat"],
                    diagnostics["min_ess"],
                    diagnostics["divergences"],
                )
                outputs.append(summarize_fit(trace, prior_set, region, metric, diagnostics))
                pd.concat(outputs, ignore_index=True).to_csv(config.TABLES_DIR / "prior_sensitivity_predictions.csv", index=False)
    result = pd.concat(outputs, ignore_index=True)
    result.to_csv(config.TABLES_DIR / "prior_sensitivity_predictions.csv", index=False)
    return result


def convergence_table(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (region, metric, year), group in predictions.groupby(["region", "metric", "year"]):
        medians = group.set_index("prior_set")["median"]
        max_value = float(medians.max())
        min_value = float(medians.min())
        reference = max(abs(float(medians.loc["A_informative"])), 1e-9)
        rows.append(
            {
                "region": region,
                "metric": metric,
                "year": int(year),
                "median_informative": float(medians.loc["A_informative"]),
                "min_median": min_value,
                "max_median": max_value,
                "max_abs_deviation": max_value - min_value,
                "max_relative_deviation_pct": 100.0 * (max_value - min_value) / reference,
            }
        )
    table = pd.DataFrame(rows).sort_values(["region", "metric", "year"])
    table["prior_sensitive"] = (table["max_relative_deviation_pct"] > 10.0) | (table["max_abs_deviation"] > table["median_informative"].abs() * 0.1)
    table.to_csv(config.TABLES_DIR / "prior_sensitivity_convergence.csv", index=False)
    return table


def sensitivity_summary(convergence: pd.DataFrame) -> pd.DataFrame:
    summary = (
        convergence.groupby(["region", "metric"], as_index=False)
        .agg(
            max_abs_deviation=("max_abs_deviation", "max"),
            max_relative_deviation_pct=("max_relative_deviation_pct", "max"),
            prior_sensitive=("prior_sensitive", "max"),
        )
        .sort_values(["prior_sensitive", "max_relative_deviation_pct"], ascending=[False, False])
    )
    summary.to_csv(config.TABLES_DIR / "prior_sensitivity_summary.csv", index=False)
    return summary
