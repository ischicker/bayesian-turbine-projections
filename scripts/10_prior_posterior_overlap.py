"""Plot prior-posterior overlap for rotor-diameter carrying capacity L."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde, norm

from turbine_projections import config
from turbine_projections.utils import configure_logging, ensure_directories

LOGGER = logging.getLogger(__name__)

METRIC = "rotor_diameter"
PARAMETER = "L"


def load_posterior_l(region: str) -> np.ndarray:
    path = config.POSTERIORS_DIR / f"{region.lower()}_{METRIC}.nc"
    if not path.exists():
        raise FileNotFoundError(f"Missing posterior: {path}")
    trace = az.from_netcdf(path)
    return np.asarray(trace.posterior[PARAMETER].values, dtype=float).reshape(-1)


def x_grid(prior_params: dict[str, dict[str, float]], posterior_samples: dict[str, np.ndarray]) -> np.ndarray:
    lower_candidates = []
    upper_candidates = []
    for region in config.REGIONS:
        prior = prior_params[region]
        posterior = posterior_samples[region]
        lower_candidates.extend([prior["mu"] - 4.0 * prior["sigma"], float(np.quantile(posterior, 0.001))])
        upper_candidates.extend([prior["mu"] + 4.0 * prior["sigma"], float(np.quantile(posterior, 0.999))])
    lower = max(0.0, min(lower_candidates))
    upper = max(upper_candidates)
    pad = 0.05 * (upper - lower)
    return np.linspace(lower - pad, upper + pad, 1200)


def make_plot() -> None:
    posterior_samples = {region: load_posterior_l(region) for region in config.REGIONS}
    prior_params = {region: config.PRIOR_CONFIG[region][METRIC][PARAMETER] for region in config.REGIONS}
    grid = x_grid(prior_params, posterior_samples)

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.7), sharex=True, sharey=False)
    overlap_rows = []
    for ax, region in zip(axes, config.REGIONS):
        prior = prior_params[region]
        posterior = posterior_samples[region]
        posterior_kde = gaussian_kde(posterior)
        prior_density = norm.pdf(grid, loc=prior["mu"], scale=prior["sigma"])
        posterior_density = posterior_kde(grid)
        overlap_density = np.minimum(prior_density, posterior_density)
        overlap_area = float(np.trapezoid(overlap_density, grid))
        posterior_median = float(np.median(posterior))
        posterior_q025 = float(np.quantile(posterior, 0.025))
        posterior_q975 = float(np.quantile(posterior, 0.975))

        ax.plot(grid, prior_density, color="#6C757D", linestyle="--", linewidth=2.0, label="Prior")
        ax.plot(grid, posterior_density, color=config.REGION_COLORS[region], linewidth=2.2, label="Posterior")
        ax.fill_between(grid, 0.0, overlap_density, color="#9CA3AF", alpha=0.35, label="Overlap")
        ax.axvline(posterior_median, color="black", linewidth=1.5, alpha=0.8)
        ax.set_title(region)
        ax.set_xlabel("Rotor-diameter L [m]")
        ax.grid(alpha=0.25)
        overlap_rows.append(
            {
                "region": region,
                "prior_mu": prior["mu"],
                "prior_sigma": prior["sigma"],
                "posterior_median": posterior_median,
                "posterior_q2_5": posterior_q025,
                "posterior_q97_5": posterior_q975,
                "overlap_area": overlap_area,
            }
        )
        LOGGER.info(
            "%s rotor-diameter L: prior N(%.1f, %.1f), posterior median %.1f [%.1f, %.1f], overlap %.3f",
            region,
            prior["mu"],
            prior["sigma"],
            posterior_median,
            posterior_q025,
            posterior_q975,
            overlap_area,
        )

    axes[0].set_ylabel("Density")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    for suffix in ["png", "pdf"]:
        fig.savefig(config.FIGURES_DIR / f"prior_posterior_overlap_rd.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Keep a tiny companion CSV for traceability; the requested artifact is the figure.
    import pandas as pd

    pd.DataFrame(overlap_rows).to_csv(config.TABLES_DIR / "prior_posterior_overlap_rd.csv", index=False)


def main() -> None:
    configure_logging()
    ensure_directories([config.FIGURES_DIR, config.TABLES_DIR])
    make_plot()
    LOGGER.info("Wrote %s", config.FIGURES_DIR / "prior_posterior_overlap_rd.pdf")


if __name__ == "__main__":
    main()
