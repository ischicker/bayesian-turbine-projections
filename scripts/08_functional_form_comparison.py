"""Compare logistic, Gompertz, and Richards Bayesian growth models for DE hub height."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import logsumexp

from turbine_projections import config
from turbine_projections.bayesian_model import (
    convergence_summary,
    fit_bayesian_gompertz,
    fit_bayesian_logistic,
    fit_bayesian_richards,
    posterior_predictive_samples,
    predict_future,
    warn_on_convergence,
)
from turbine_projections.utils import configure_logging, ensure_directories

LOGGER = logging.getLogger(__name__)

REGION = "DE"
METRIC = "hub_height"
MODEL_SPECS = {
    "logistic": {
        "label": "Logistic",
        "model_type": "growth",
        "fit": fit_bayesian_logistic,
        "color": "#111111",
    },
    "gompertz": {
        "label": "Gompertz",
        "model_type": "gompertz_growth",
        "fit": fit_bayesian_gompertz,
        "color": "#E76F51",
    },
    "richards": {
        "label": "Richards",
        "model_type": "richards_growth",
        "fit": fit_bayesian_richards,
        "color": "#457B9D",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Overwrite cached functional-form posteriors.")
    return parser.parse_args()


def load_region_data() -> pd.DataFrame:
    path = config.PROCESSED_DATA_DIR / f"{REGION.lower()}_clean.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing processed data: {path}")
    return pd.read_csv(path)


def stratified_year_subsample(data: pd.DataFrame, n_samples: int = 5000) -> pd.DataFrame:
    """Draw a deterministic stratified sample by year."""

    if len(data) <= n_samples:
        return data.copy()
    rng = np.random.default_rng(config.RANDOM_SEED)
    years = np.array(sorted(data["year"].dropna().unique()))
    base = max(1, n_samples // len(years))
    remainder = n_samples - base * len(years)
    sampled_parts = []
    for index, year in enumerate(years):
        group = data.loc[data["year"] == year]
        n_year = base + (1 if index < remainder else 0)
        replace = len(group) < n_year
        chosen = rng.choice(group.index.to_numpy(), size=n_year, replace=replace)
        sampled_parts.append(data.loc[chosen])
    return pd.concat(sampled_parts, ignore_index=True).sample(frac=1.0, random_state=config.RANDOM_SEED)


def posterior_path(model_name: str) -> Path:
    return config.POSTERIORS_DIR / f"functional_form_{REGION.lower()}_{METRIC}_{model_name}.nc"


def _safe_elpd_value(criterion, name: str) -> float:
    return float(getattr(criterion, name, np.nan))


def _log_likelihood_matrix(trace: az.InferenceData) -> np.ndarray:
    if not hasattr(trace, "log_likelihood") or "y_obs" not in trace.log_likelihood:
        raise ValueError("InferenceData must include pointwise log_likelihood for WAIC/LOO.")
    values = np.asarray(trace.log_likelihood["y_obs"].values)
    return values.reshape((-1, values.shape[-1]))


def compute_waic(trace: az.InferenceData) -> dict[str, float]:
    """Compute WAIC from pointwise log-likelihood when ArviZ does not export az.waic."""

    if hasattr(az, "waic"):
        waic = az.waic(trace)
        return {
            "elpd_waic": _safe_elpd_value(waic, "elpd_waic"),
            "p_waic": _safe_elpd_value(waic, "p_waic"),
            "waic_se": _safe_elpd_value(waic, "se"),
        }

    log_likelihood = _log_likelihood_matrix(trace)
    lppd_i = logsumexp(log_likelihood, axis=0) - np.log(log_likelihood.shape[0])
    p_waic_i = np.var(log_likelihood, axis=0, ddof=1)
    elpd_waic_i = lppd_i - p_waic_i
    return {
        "elpd_waic": float(np.sum(elpd_waic_i)),
        "p_waic": float(np.sum(p_waic_i)),
        "waic_se": float(np.sqrt(len(elpd_waic_i) * np.var(elpd_waic_i, ddof=1))),
    }


def compute_loo(trace: az.InferenceData) -> dict[str, float]:
    loo = az.loo(trace)
    return {
        "elpd_loo": _safe_elpd_value(loo, "elpd_loo")
        if np.isfinite(_safe_elpd_value(loo, "elpd_loo"))
        else _safe_elpd_value(loo, "elpd"),
        "p_loo": _safe_elpd_value(loo, "p_loo")
        if np.isfinite(_safe_elpd_value(loo, "p_loo"))
        else _safe_elpd_value(loo, "p"),
        "loo_se": _safe_elpd_value(loo, "se"),
    }


def summarize_model(model_name: str, trace: az.InferenceData, elapsed_seconds: float, n_fit_samples: int) -> list[dict]:
    spec = MODEL_SPECS[model_name]
    diagnostics = convergence_summary(trace)
    predictions = predict_future(trace, spec["model_type"], config.TARGET_YEARS)
    waic = compute_waic(trace)
    loo = compute_loo(trace)
    rows = []
    for _, row in predictions.iterrows():
        rows.append(
            {
                "region": REGION,
                "metric": METRIC,
                "functional_form": model_name,
                "year": int(row["year"]),
                "projection_median": float(row["median"]),
                "projection_q2_5": float(row["q2_5"]),
                "projection_q97_5": float(row["q97_5"]),
                "elpd_waic": waic["elpd_waic"],
                "p_waic": waic["p_waic"],
                "waic_se": waic["waic_se"],
                "elpd_loo": loo["elpd_loo"],
                "p_loo": loo["p_loo"],
                "loo_se": loo["loo_se"],
                "max_rhat": diagnostics["max_rhat"],
                "min_ess": diagnostics["min_ess"],
                "divergences": diagnostics["divergences"],
                "n_fit_samples": n_fit_samples,
                "elapsed_seconds": elapsed_seconds,
            }
        )
    return rows


def plot_comparison(full_data: pd.DataFrame, traces: dict[str, az.InferenceData]) -> None:
    years_plot = np.arange(int(full_data["year"].min()), 2056)
    column = config.METRIC_COLUMNS[METRIC]
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    ax.scatter(
        full_data["year"],
        full_data[column],
        s=4,
        color=config.REGION_COLORS[REGION],
        alpha=0.08,
        edgecolors="none",
        label="Turbines",
    )
    ax.axvline(2015, ls="--", color="gray", lw=1.3, alpha=0.7, zorder=1)
    for model_name, trace in traces.items():
        spec = MODEL_SPECS[model_name]
        samples = posterior_predictive_samples(trace, spec["model_type"], years_plot, max_samples=1000)
        median = np.median(samples, axis=0)
        lower = np.quantile(samples, 0.025, axis=0)
        upper = np.quantile(samples, 0.975, axis=0)
        ax.plot(years_plot, median, color=spec["color"], linewidth=2.0, label=spec["label"])
        ax.fill_between(years_plot, lower, upper, color=spec["color"], alpha=0.16, linewidth=0)
    ax.set_xlabel("Year")
    ax.set_ylabel(config.METRIC_LABELS[column])
    ax.grid(alpha=0.25)
    ax.legend(facecolor="white", edgecolor="gray", framealpha=0.9)
    fig.tight_layout()
    for suffix in ["png", "pdf"]:
        fig.savefig(config.FIGURES_DIR / f"functional_form_comparison.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    configure_logging()
    ensure_directories([config.POSTERIORS_DIR, config.TABLES_DIR, config.FIGURES_DIR])

    full_data = load_region_data()
    fit_data = stratified_year_subsample(full_data, 5000)
    LOGGER.info("Subsampled %s %s from %d to %d rows", REGION, METRIC, len(full_data), len(fit_data))

    priors = config.PRIOR_CONFIG[REGION][METRIC]
    traces: dict[str, az.InferenceData] = {}
    rows: list[dict] = []
    for model_name, spec in MODEL_SPECS.items():
        path = posterior_path(model_name)
        start = time.perf_counter()
        if path.exists() and not args.force:
            LOGGER.info("Loading cached %s posterior from %s", model_name, path)
            trace = az.from_netcdf(path)
            elapsed = 0.0
            diagnostics = convergence_summary(trace)
        else:
            LOGGER.info("Fitting %s model for %s %s", model_name, REGION, METRIC)
            trace = spec["fit"](fit_data, priors, METRIC, config.MCMC_FULL)
            elapsed = time.perf_counter() - start
            diagnostics = warn_on_convergence(trace, REGION, f"{METRIC}_{model_name}")
            trace.to_netcdf(path)
        LOGGER.info(
            "%s finished in %.1fs | R-hat %.3f | ESS %.0f | divergences %d",
            model_name,
            elapsed,
            diagnostics["max_rhat"],
            diagnostics["min_ess"],
            diagnostics["divergences"],
        )
        traces[model_name] = trace
        rows.extend(summarize_model(model_name, trace, elapsed, len(fit_data)))

    result = pd.DataFrame(rows)
    result.to_csv(config.TABLES_DIR / "functional_form_comparison.csv", index=False)
    plot_comparison(full_data, traces)

    deviations = (
        result.groupby("year")["projection_median"]
        .agg(["min", "max"])
        .assign(max_deviation=lambda frame: frame["max"] - frame["min"])
    )
    LOGGER.info("=== Functional Form Comparison ===\n%s", result.to_string(index=False))
    LOGGER.info("Maximum median projection deviations by year:\n%s", deviations.to_string())


if __name__ == "__main__":
    main()
