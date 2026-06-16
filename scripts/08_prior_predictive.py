"""Generate prior predictive checks for all region-metric combinations."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from turbine_projections import config
from turbine_projections.bayesian_model import posterior_predictive_samples
from turbine_projections.utils import configure_logging, ensure_directories

LOGGER = logging.getLogger(__name__)

YEARS = np.arange(1985, 2061)
N_PRIOR_SAMPLES = 500
PHYSICAL_MIN = 0.0
PHYSICAL_MAX = 500.0
METRIC_TITLES = {
    "hub_height": "Hub height [m]",
    "rotor_diameter": "Rotor diameter [m]",
    "specific_power": "Specific power [W/m2]",
}


def load_region_data(region: str) -> pd.DataFrame:
    path = config.PROCESSED_DATA_DIR / f"{region.lower()}_clean.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing processed data: {path}")
    return pd.read_csv(path)


def sample_prior(prior: dict[str, float], rng: np.random.Generator, size: int) -> np.ndarray:
    dist = prior["dist"]
    if dist == "Normal":
        return rng.normal(prior["mu"], prior["sigma"], size=size)
    if dist == "HalfNormal":
        return np.abs(rng.normal(0.0, prior["sigma"], size=size))
    if dist == "TruncatedNormal":
        values = rng.normal(prior["mu"], prior["sigma"], size=size)
        return np.clip(values, prior.get("lower", -np.inf), prior.get("upper", np.inf))
    if dist == "Uniform":
        return rng.uniform(prior["lower"], prior["upper"], size=size)
    raise ValueError(f"Unsupported prior distribution for prior predictive trajectory: {dist}")


def prior_trajectories(region: str, metric: str, rng: np.random.Generator) -> np.ndarray:
    priors = config.PRIOR_CONFIG[region][metric]
    L = sample_prior(priors["L"], rng, N_PRIOR_SAMPLES)
    k = sample_prior(priors["k"], rng, N_PRIOR_SAMPLES)
    t0 = sample_prior(priors["t0"], rng, N_PRIOR_SAMPLES)
    year_grid = YEARS[None, :]

    if priors["model_type"] == "growth":
        y0 = sample_prior(priors["y0"], rng, N_PRIOR_SAMPLES)
        trajectories = y0[:, None] + L[:, None] / (1.0 + np.exp(-k[:, None] * (year_grid - t0[:, None])))
    elif priors["model_type"] == "decay":
        y_min = sample_prior(priors["y_min"], rng, N_PRIOR_SAMPLES)
        trajectories = y_min[:, None] + L[:, None] / (1.0 + np.exp(k[:, None] * (year_grid - t0[:, None])))
    else:
        raise ValueError(f"Unsupported model type for prior predictive checks: {priors['model_type']}")

    valid = np.isfinite(trajectories).all(axis=1)
    valid &= (trajectories >= PHYSICAL_MIN).all(axis=1)
    valid &= (trajectories <= PHYSICAL_MAX).all(axis=1)
    filtered = trajectories[valid]
    LOGGER.info(
        "%s %s prior predictive: kept %d/%d physical trajectories",
        region,
        metric,
        len(filtered),
        N_PRIOR_SAMPLES,
    )
    return filtered


def plot_observations(ax: plt.Axes, region: str, metric: str, data: pd.DataFrame) -> None:
    column = config.METRIC_COLUMNS[metric]
    if region == "AT":
        annual = data.groupby("year", as_index=False)[column].median()
        ax.scatter(
            annual["year"],
            annual[column],
            s=16,
            color="black",
            alpha=0.85,
            edgecolors="none",
            label="Observed data" if metric == "hub_height" else None,
        )
        return

    ax.scatter(
        data["year"],
        data[column],
        s=3,
        color="black",
        alpha=0.07,
        edgecolors="none",
        label="Observations" if region == "DE" and metric == "hub_height" else None,
    )


def plot_posterior(ax: plt.Axes, region: str, metric: str) -> None:
    posterior_path = config.POSTERIORS_DIR / f"{region.lower()}_{metric}.nc"
    if not posterior_path.exists():
        LOGGER.warning("Skipping posterior overlay; missing %s", posterior_path)
        return
    trace = az.from_netcdf(posterior_path)
    model_type = config.PRIOR_CONFIG[region][metric]["model_type"]
    samples = posterior_predictive_samples(trace, model_type, YEARS, max_samples=1000)
    median = np.median(samples, axis=0)
    lower = np.quantile(samples, 0.025, axis=0)
    upper = np.quantile(samples, 0.975, axis=0)
    ax.plot(YEARS, median, color="#1D4ED8", linewidth=2.0, label="Posterior median")
    ax.fill_between(YEARS, lower, upper, color="#60A5FA", alpha=0.25, linewidth=0, label="Posterior 95% CI")


def make_figure() -> None:
    rng = np.random.default_rng(config.RANDOM_SEED)
    data = {region: load_region_data(region) for region in config.REGIONS}
    fig, axes = plt.subplots(3, 3, figsize=(13.0, 9.2), sharex=True, sharey=False)

    for row, region in enumerate(config.REGIONS):
        for col, metric in enumerate(config.METRICS):
            ax = axes[row, col]
            trajectories = prior_trajectories(region, metric, rng)
            for trajectory in trajectories:
                ax.plot(YEARS, trajectory, color="#777777", alpha=0.05, linewidth=0.7, zorder=1)
            plot_observations(ax, region, metric, data[region])
            plot_posterior(ax, region, metric)

            column = config.METRIC_COLUMNS[metric]
            ax.set_xlim(YEARS.min(), YEARS.max())
            ax.set_ylim(0, 500)
            if row == 0:
                ax.set_title(METRIC_TITLES[metric])
            if col == 0:
                ax.set_ylabel(region)
            if row == len(config.REGIONS) - 1:
                ax.set_xlabel("Year")
            ax.grid(alpha=0.22)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    for suffix in ["png", "pdf"]:
        fig.savefig(config.FIGURES_DIR / f"prior_predictive_checks.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    configure_logging()
    ensure_directories([config.FIGURES_DIR])
    make_figure()
    LOGGER.info("Wrote prior predictive checks to %s", config.FIGURES_DIR / "prior_predictive_checks.pdf")


if __name__ == "__main__":
    main()
